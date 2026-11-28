"""Extend native instance masks along ScanNet20 surfaces classified by Point Transformer V3."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from refine_native_panoptic import digest

CHECKPOINT = "40206376ff2f83f48e4d1bc27d5c5d96be7c87c5d11eb45fe7be501959040e7f"


def xyz(vertices):
    return np.column_stack([vertices[axis] for axis in "xyz"]).astype(np.float32)


def classify_surface(mesh, rotation, weights, device):
    import point_transformer_v3 as ptv3
    import torch

    vertices = mesh["vertex"].data
    points = xyz(vertices)
    color = np.column_stack([vertices[name] for name in ("red", "green", "blue")]).astype(np.float32)
    faces = mesh["face"].data["vertex_indices"]
    if not len(points) or not len(faces):
        raise ValueError("The native mesh must have vertices and triangular faces")
    if faces.ndim == 1:
        faces = np.stack(faces)
    if faces.shape[1] != 3:
        raise ValueError("The native mesh must use triangular faces")
    faces = np.asarray(faces, dtype=np.int32)
    corners = points[faces]
    areas = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], areas)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    torch.manual_seed(942)
    model = ptv3.load("ptv3_scannet", device="cpu", precision="fp16" if device=="cuda" else "fp32",
                      weights=weights, warmup=False).to(device)
    logits = model.predict(points @ rotation.T, color, normals @ rotation.T)
    probability = (logits.float().softmax(1) * 255).round().byte().cpu().numpy()
    return points, faces, probability


def surface_components(labels, faces):
    starts = faces.ravel()
    ends = faces[:, [1, 2, 0]].ravel()
    keep = (labels[starts] == labels[ends]) & (labels[starts] >= 2)
    graph = coo_matrix((np.ones(keep.sum(), np.bool_), (starts[keep], ends[keep])),
                       shape=(len(labels), len(labels))).tocsr()
    return connected_components(graph, directed=False)[1]


def extend(semantic, instance, counts, components, distance, map_points):
    semantic = semantic.copy()
    instance = instance.copy()
    seen = distance < .10
    for obj in np.unique(instance[instance > 0]):
        mask = instance == obj
        measured = mask & seen
        if measured.sum() < 30:
            continue
        mean = counts[measured].mean(0) / 255
        previous = np.bincount(semantic[mask]).argmax()
        winner = int(mean.argmax())
        if winner >= 2 and mean[winner] >= .55 and mean[previous] < .23:
            semantic[mask] = winner

    labels = counts.argmax(1)
    indices = np.flatnonzero(seen & (labels >= 2))
    indices = indices[np.argsort(components[indices], kind="stable")]
    boundaries = np.r_[0, np.flatnonzero(np.diff(components[indices])) + 1, len(indices)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        points = indices[start:stop]
        if len(points) < 50:
            continue
        category = int(labels[points[0]])
        confidence = counts[points, category]
        if confidence.mean() < .58 * 255:
            continue
        seeds = points[(semantic[points] == category) & (instance[points] > 0)]
        available = points[(instance[points] == 0) & (confidence > 145)]
        if len(available) < 30 or len(seeds) < 35 or len(seeds) < .12 * len(points):
            continue
        radius = .13
        if len(seeds) >= .4 * len(points) and np.unique(instance[seeds]).size == 1:
            radius = .25
        separation, nearest = cKDTree(map_points[seeds]).query(map_points[available], workers=2)
        accept = separation < radius
        target = available[accept]
        semantic[target] = category
        instance[target] = instance[seeds[nearest[accept]]]
    return semantic, instance


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {name:digest(getattr(args, name)) for name in ("checkpoint", "map", "mesh")}
    if inputs["checkpoint"] != CHECKPOINT:
        raise ValueError("Expected the Pointcept PTv3 ScanNet20 base model_best.pth release")
    anchor_file = args.anchor / "result.json"
    mask_file = args.anchor / "panoptic.npz"
    classes_file = args.anchor / "classes.json"
    receipt = json.loads(anchor_file.read_text())
    if not receipt.get("same_native_vertex_order") or receipt.get("label_space") != "ScanNet20":
        raise ValueError("Expected the native ScanNet20 panoptic readout")
    for name in ("map", "mesh"):
        if receipt["inputs"][name] != inputs[name]:
            raise ValueError(f"The anchor was made from a different {name}")
    inputs["anchor"] = digest(anchor_file)
    for path in (mask_file, classes_file):
        inputs[path.name] = digest(path)
        if inputs[path.name] != receipt["outputs"][path.name]:
            raise ValueError(f"The anchor output {path.name} differs from its receipt")
    rotation = np.asarray(receipt["rotation"], dtype=np.float32)
    if rotation.shape != (3,3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise ValueError("Expected an orthonormal native gravity rotation in the anchor receipt")
    mesh = PlyData.read(args.mesh, known_list_len={"face":{"vertex_indices":3}})
    points, faces, probability = classify_surface(mesh, rotation, args.checkpoint, args.device)
    cloud = PlyData.read(args.map, mmap=False)
    vertex = cloud["vertex"].data
    map_points = xyz(vertex)
    nearest_tree = cKDTree(points)
    distance, nearest = nearest_tree.query(map_points, workers=2)
    components = surface_components(probability.argmax(1), faces)
    with np.load(mask_file, allow_pickle=False) as masks:
        semantic, instance = masks["semantic"], masks["instance"]
    if semantic.shape != (len(vertex),) or instance.shape != semantic.shape:
        raise ValueError("The anchor masks must cover every native map vertex")
    semantic, instance = extend(semantic, instance, probability[nearest], components[nearest],
                                distance.astype(np.float32), map_points)
    vertex["semantic_id"] = semantic.astype(np.int32) + 1
    vertex["instance_id"] = instance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        result = Path(temp) / "result"
        result.mkdir()
        np.savez_compressed(result / "panoptic.npz", semantic=semantic, instance=instance)
        if getattr(args, "save_point_evidence", False):
            np.savez_compressed(result / "mesh.npz", proba=probability, nearest=nearest.astype("i4"),
                                distance=distance.astype("f4"))
        cloud.write(result / "map_labeled.ply")
        (result / "classes.json").write_bytes(classes_file.read_bytes())
        audit = dict(inputs=inputs, vertex_count=len(vertex), mesh_vertices=len(points),
                     rotation=rotation.tolist(), label_space="ScanNet20", same_native_vertex_order=True,
                     device=args.device, outputs={p.name:digest(p) for p in result.iterdir()})
        (result / "result.json").write_text(json.dumps(audit, indent=2))
        result.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mesh","map","anchor","checkpoint","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda","cpu"), default="cuda")
    parser.add_argument("--save-point-evidence", action="store_true",
                        help="Save per-mesh-vertex PTv3 probabilities and native point correspondences")
    run(parser.parse_args())
