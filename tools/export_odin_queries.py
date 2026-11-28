"""Export ODIN's ScanNet20 RGB-D queries on a reconstructed ReVeMap surface."""

import argparse
from contextlib import contextmanager
import io
import json
import struct
import sys
import tempfile
from pathlib import Path
import zlib

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from odin_runtime import load_model, publish, sha256
from prepare_odin_geometry import groups_from_mesh, mesh_arrays, rotation_from, trajectory, vertices

WIDTH, HEIGHT = 320, 240
CANONICAL = np.array([*range(2, 20), 0, 1], np.uint8)


def views(poses, valid, mode):
    available = np.flatnonzero(valid)
    if mode == "cap64":
        chosen = np.flatnonzero(valid & (np.arange(len(poses)) % 20 == 0))
        if not len(chosen):
            chosen = available[::20]
        if len(chosen) > 64:
            chosen = chosen[np.linspace(0, len(chosen) - 1, 64).round().astype(int)]
        return chosen.tolist()
    pool = np.unique(np.r_[available[::max(1, len(poses) // 350)], available[-1]])
    position, forward = poses[pool, :3, 3], poses[pool, :3, 2]
    selected = [0]
    distance = np.full(len(pool), np.inf)
    for _ in range(min(19, len(pool)) - 1):
        i = selected[-1]
        delta = np.linalg.norm(position - position[i], axis=1) + .8 * (1 - np.clip(forward @ forward[i], -1, 1))
        distance = np.minimum(distance, delta)
        distance[selected] = -1
        selected.append(int(distance.argmax()))
    return sorted(pool[selected].tolist())


def sensor_frames(path, selected, expected_count, inpaint):
    images, depths = [], []
    selected = set(selected)
    with path.open("rb") as stream:
        version, = struct.unpack("<I", stream.read(4))
        if version != 4:
            raise ValueError("Expected a ScanNet v4 .sens recording")
        length, = struct.unpack("<Q", stream.read(8))
        stream.seek(length, 1)
        _, color_to_sensor, depth_k, depth_to_sensor = np.frombuffer(stream.read(256), "<f4").reshape(4, 4, 4)
        color_codec, depth_codec, _, _, w, h = struct.unpack("<iiIIII", stream.read(24))
        scale, count = struct.unpack("<fQ", stream.read(12))
        if (color_codec, depth_codec) != (2, 1) or not np.allclose(color_to_sensor, depth_to_sensor, atol=1e-6):
            raise ValueError("Expected aligned ScanNet JPEG color and zlib-ushort depth")
        if count != expected_count or scale <= 0:
            raise ValueError("Raw frame count or depth scale does not match the estimated trajectory")
        for frame in range(count):
            # Skip the reference pose matrix and timestamps. Only stored RGB and depth enter the model.
            stream.seek(80, 1)
            color_size, depth_size = struct.unpack("<QQ", stream.read(16))
            if frame not in selected:
                stream.seek(color_size + depth_size, 1)
                continue
            with Image.open(io.BytesIO(stream.read(color_size))) as rgb:
                image = np.array(rgb.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR))
            raw = np.frombuffer(zlib.decompress(stream.read(depth_size)), "<u2").reshape(h, w)
            holes = raw == 0
            if inpaint and holes.any() and not holes.all():
                fill = cv2.inpaint(raw.astype("f4"), holes.astype("u1"), 5, cv2.INPAINT_TELEA)
                raw = np.where(holes, np.maximum(fill, 0), raw).astype("u2")
            depth = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST).astype("f4") / scale
            images.append(torch.from_numpy(image.transpose(2, 0, 1).copy()))
            depths.append(torch.from_numpy(depth))
    k = np.eye(4, dtype="f4")
    k[0, 0], k[1, 1] = depth_k[0, 0] * WIDTH / w, depth_k[1, 1] * HEIGHT / h
    k[0, 2] = (depth_k[0, 2] + .5) * WIDTH / w - .5
    k[1, 2] = (depth_k[1, 2] + .5) * HEIGHT / h - .5
    return images, torch.stack(depths), torch.from_numpy(k)


def model_input(args, cfg, categories, selected, poses, xyz, rgb, groups, rotation):
    from odin.modeling.backproject.backproject import backprojector_dataloader

    center = xyz.mean(0).astype(float)
    camera = poses[selected].copy()
    camera[:, :3, :3] = rotation @ camera[:, :3, :3]
    camera[:, :3, 3] = (camera[:, :3, 3] - center) @ rotation.T
    coords = torch.from_numpy(((xyz - center) @ rotation.T).astype("f4"))
    images, depths, k = sensor_frames(args.sens, selected, len(poses), cfg.INPUT.INPAINT_DEPTH)
    if cfg.IGNORE_DEPTH_MAX != -1:
        depths[depths > cfg.IGNORE_DEPTH_MAX] = 0.
    cameras = torch.from_numpy(camera.astype("f4"))
    intrinsics = k[None].repeat(len(selected), 1, 1)
    ph, pw = (-HEIGHT) % 32, (-WIDTH) % 32
    shapes = [torch.zeros(len(selected), 1, (HEIGHT + ph) // k, (WIDTH + pw) // k) for k in (4, 8, 16, 32)]
    levels, coords, _ = backprojector_dataloader(shapes, depths, cameras, intrinsics, augment=False,
        method=cfg.MODEL.INTERPOLATION_METHOD, scannet_pc=coords, padding=(ph, pw), do_rot_scale=cfg.DO_ROT_SCALE)
    return dict(images=images, depths=list(depths), poses=list(cameras), intrinsics=list(intrinsics),
                valids=[d > 0 for d in depths], multi_scale_xyz=levels[::-1], scannet_coords=coords,
                scannet_segments=torch.from_numpy(groups.astype("i8")), scannet_color=torch.from_numpy(rgb),
                scannet_labels=torch.zeros(len(xyz), 2, dtype=torch.long), num_classes=20, all_classes=categories,
                dataset_name="scannet_test_single_revemap", file_name=args.sens.stem + "/rgb/frame.jpg", decoder_3d=True)


@contextmanager
def gpu_lock(path):
    if path is None:
        yield
    else:
        import fcntl
        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield


def export(args):
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    xyz, rgb, faces = mesh_arrays(args.mesh)
    map_xyz = vertices(args.map)
    _, nearest = cKDTree(xyz).query(map_xyz, workers=args.cpu_threads)
    poses, valid = trajectory(args.trajectory)
    rotation = rotation_from(args.upright, vertices(args.mesh).astype(float) if args.upright is None else xyz, faces, poses[valid])
    if args.superpoints:
        with np.load(args.superpoints, allow_pickle=False) as data:
            groups = data["groups"]
        receipt = args.superpoints.parent / "geometry.json"
        if receipt.is_file():
            receipt = json.loads(receipt.read_text(encoding="utf-8"))
            if receipt.get("mesh") != sha256(args.mesh) or receipt.get("files", {}).get(args.superpoints.name) != sha256(args.superpoints):
                raise ValueError("Superpoint receipt does not match the input mesh or groups")
    else:
        sys.path[:0] = [str(p.resolve()) for p in args.python_path]
        groups = groups_from_mesh(xyz, faces)
    if groups.shape != (len(xyz),) or not np.issubdtype(groups.dtype, np.integer) or groups.min() < 0:
        raise ValueError("Superpoints must be one nonnegative integer per target mesh vertex")
    if not np.array_equal(np.unique(groups), np.arange(groups.max() + 1)):
        raise ValueError("Superpoint IDs must be compact and zero-based")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {n: sha256(getattr(args, n)) for n in ("sens", "trajectory", "mesh", "map", "weights", "superpoints", "upright") if getattr(args, n)}
    with tempfile.TemporaryDirectory(prefix="revemap-odin-") as temporary:
        cfg, model, cats, runtime = load_model(args.odin_root, args.detectron2_root, args.weights, Path(temporary), args.python_path)
        for name in args.views:
            dest = args.output / name
            dest.mkdir(exist_ok=True)
            ids = views(poses, valid, name)
            item = model_input(args, cfg, cats, ids, poses, xyz, rgb, groups, rotation)
            with gpu_lock(args.lock), torch.inference_mode():
                model.cuda()
                try:
                    result = model([item])[0]
                    pred = result["instances_3d"]
                    masks = pred["pred_masks"].detach().cpu().numpy().T
                    semantic = CANONICAL[result["semantic_3d"].detach().cpu().numpy()]
                    classes = CANONICAL[pred["pred_classes"].detach().cpu().numpy().astype(int) - 1]
                    scores = pred["pred_scores"].detach().cpu().numpy().astype("f4")
                finally:
                    model.cpu()
                del result, pred
                torch.cuda.empty_cache()
            if masks.shape != (100, len(xyz)) or not np.isfinite(scores).all():
                raise ValueError("Unexpected ODIN instance query dimensions or non-finite scores")
            arrays = dict(semantic=semantic, classes=classes, scores=scores, masks=np.packbits(masks, axis=1),
                          map_to_mesh=nearest.astype("i4"), mesh_count=len(xyz))
            signature = dict(inputs=sources, view_policy=name, frames=ids, driver=sha256(Path(__file__)),
                             model_source=runtime["source_python_sha256"], model_config=runtime["config"],
                             rotation=rotation.tolist())
            publish(dest, "queries.npz", arrays, signature, dict(frames=ids, inputs=sources, runtime=runtime,
                    view_policy=name, mesh_vertices=len(xyz), map_points=len(nearest), rotation=rotation.tolist()))
            print(f"{name}: {len(ids)} RGB-D views, {len(xyz)} mesh vertices, {dest / 'queries.npz'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sens", "trajectory", "mesh", "map", "weights", "odin-root", "detectron2-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--superpoints", type=Path, help="Optional NPZ containing compact groups in target mesh order")
    parser.add_argument("--upright", type=Path, help="Optional 3×3/4×4 NPY or anchor result JSON with a room rotation")
    parser.add_argument("--python-path", type=Path, action="append", default=[], help="Extra inference dependency directory, repeatable")
    parser.add_argument("--views", nargs="+", choices=("view19", "cap64"), default=("view19", "cap64"))
    parser.add_argument("--lock", type=Path, help="Optional shared Linux GPU lock file")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    export(args)


if __name__ == "__main__":
    main()
