"""Run the ScanNet20 OneFormer3D checkpoint on a ReVeMap reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from native_gravity import axis_frame
from panoptic_observations import CLASSES, reconcile


CHECKPOINT = "9ba46231ff62d831c24d1e42e99e0e8c61978857d7f0475e39e002cce34cb7b6"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def geometric_superpoints(mesh):
    import segmentator
    import torch

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not mesh.has_vertex_colors() or not mesh.has_triangles():
        raise ValueError("The native TSDF mesh must contain measured RGB and triangle faces")
    xyz = vertices.astype(np.float32)
    faces = triangles.astype(np.int64)
    groups = segmentator.segment_mesh(torch.from_numpy(xyz), torch.from_numpy(faces)).numpy()
    keep = np.bincount(groups)[groups] >= 20
    if keep.any() and not keep.all():
        nearest = cKDTree(xyz[keep]).query(xyz[~keep], workers=2)[1]
        groups[~keep] = groups[keep][nearest]
    _, groups = np.unique(groups, return_inverse=True)
    return xyz, groups.reshape(-1).astype(np.int64)


def infer_mesh(mesh, rotation, model):
    import spconv.pytorch as spconv
    import torch

    xyz, groups = geometric_superpoints(mesh)
    groups = torch.from_numpy(groups).cuda()
    colors = np.asarray(mesh.vertex_colors).astype(np.float32) * 255
    values = np.column_stack((xyz @ rotation.T, (colors - 127.5) / 127.5)).astype(np.float32)
    with torch.inference_mode():
        coords, features, inverse, shape = model.collate([torch.from_numpy(values).cuda()])
        sparse = spconv.SparseConvTensor(features, coords, shape, 1)
        features = model.extract_feat(sparse, groups, inverse, [0, int(groups.max()) + 1])
        output = model.decoder(features, features)
        semantic, instance = model.predict_by_feat_panoptic(output, groups)
        logits = output["sem_preds"][0]
        head = logits.argmax(1)[groups].cpu().numpy().astype(np.uint8)
        confidence = logits.softmax(1).amax(1)[groups].cpu().numpy().astype(np.float16)
        semantic = semantic.cpu().numpy().astype(np.uint8)
        instance = np.where(semantic >= 2, instance.cpu().numpy(), 0).astype(np.int32)
    if np.any(semantic >= len(CLASSES)) or np.any(instance[semantic >= 2] <= 0):
        raise ValueError("Unexpected model output for ScanNet20")
    return xyz, semantic, instance, head, confidence


def infer(mesh, poses, model):
    rotation, orientation = axis_frame(poses, np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    return (*infer_mesh(mesh, rotation, model), rotation, orientation)


def voted_labels(mesh, map_xyz, rotation, model, semantic, instance, *, vertex=None, classes=None,
                 view_semantics=None):
    from panoptic_vote import majority_vote

    results = [dict(semantic=semantic, instance=instance)]
    angles = (90, 45, 0) if classes is None else (45, 90, 0)
    for degrees in angles:
        angle = np.deg2rad(degrees)
        yaw = np.array([[np.cos(angle), -np.sin(angle), 0.],
                        [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        candidate = mesh
        if degrees == 0:
            candidate = mesh.simplify_vertex_clustering(voxel_size=0.035)
            candidate.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
        xyz, sem, ids, head, confidence = infer_mesh(candidate, yaw @ rotation, model)
        distance, nearest = cKDTree(xyz).query(map_xyz, workers=2)
        sem, ids = sem[nearest], ids[nearest]
        if classes is not None:
            sem, ids = reconcile(vertex, classes, sem, ids, head[nearest], confidence[nearest],
                                 distance.astype(np.float32))
        results.append(dict(semantic=sem, instance=ids))
    if view_semantics is not None:
        view_semantics.extend(result["semantic"] for result in results)
    return majority_vote(results)


def run(args):
    save_views = getattr(args, "save_view_evidence", False)
    if save_views and not getattr(args, "vote_views", False):
        raise ValueError("--save-view-evidence requires --vote-views")
    import open3d as o3d
    import torch
    from mmdet3d.registry import MODELS
    from mmengine import init_default_scope
    from mmengine.config import Config

    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {k:digest(getattr(args, k)) for k in ("mesh", "map", "trajectory", "config", "checkpoint")}
    if args.native_classes:
        inputs["native_classes"] = digest(args.native_classes)
    if inputs["checkpoint"] != CHECKPOINT:
        raise ValueError("Use the released OneFormer3D ScanNet20 v1.0 checkpoint")
    trajectory = json.loads(args.trajectory.read_text())
    if trajectory.get("gt_at_inference") is True or trajectory.get("matrix_convention") != "T_world_camera_m":
        raise ValueError("Expected a native metric world-from-camera trajectory")
    poses = np.stack([np.asarray(p["T_world_camera_m"]).reshape(4,4) for p in trajectory["poses"] if p.get("valid", True)])
    if not np.isfinite(poses).all():
        raise ValueError("Valid camera poses must be finite")
    torch.manual_seed(942)
    init_default_scope("mmdet3d")
    cfg = Config.fromfile(str(args.config))
    network = MODELS.build(cfg.model)
    state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    network.load_state_dict(state["state_dict"], strict=True)
    del state
    network.eval().cuda()
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    xyz, semantic, instance, head, confidence, rotation, orientation = infer(mesh, poses, network)
    cloud = PlyData.read(args.map, mmap=False)
    vertex = cloud["vertex"].data
    map_xyz = np.column_stack([vertex[k] for k in "xyz"])
    distance, nearest = cKDTree(xyz).query(map_xyz, workers=2)
    semantic, instance = semantic[nearest], instance[nearest]
    vote_views = getattr(args, "vote_views", False)
    classes = None
    if args.native_classes:
        classes = json.loads(args.native_classes.read_text())
        semantic, instance = reconcile(vertex, classes, semantic, instance,
                                       head[nearest], confidence[nearest], distance.astype(np.float32))
    view_semantics = [] if save_views else None
    if vote_views:
        semantic, instance = voted_labels(mesh, map_xyz, rotation, network, semantic, instance,
                                          vertex=vertex, classes=classes, view_semantics=view_semantics)
    vertex["semantic_id"] = semantic.astype(np.int32) + 1
    vertex["instance_id"] = instance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        result = Path(temp) / "result"
        result.mkdir()
        cloud.write(result / "map_labeled.ply")
        classes = {0:"unknown", **{i+1:c for i,c in enumerate(CLASSES)}}
        (result / "classes.json").write_text(json.dumps(classes, indent=2))
        np.savez_compressed(result / "panoptic.npz", semantic=semantic, instance=instance)
        if save_views:
            np.savez_compressed(result / "views.npz", semantic=np.stack(view_semantics))
        receipt = dict(inputs=inputs, vertex_count=len(vertex), rotation=rotation.tolist(), orientation=orientation,
                       label_space="ScanNet20", same_native_vertex_order=True, vote_views=vote_views,
                       outputs={f.name:digest(f) for f in result.iterdir()})
        (result / "result.json").write_text(json.dumps(receipt, indent=2))
        result.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-classes", type=Path, help="Use the input map's tracked objects to refine each expert mask")
    parser.add_argument("--vote-views", action="store_true", help="Vote across four mesh predictions")
    parser.add_argument("--save-view-evidence", action="store_true",
                        help="Keep the four semantic predictions on native map points in views.npz")
    for field in ("mesh", "map", "trajectory", "config", "checkpoint", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    run(parser.parse_args())
