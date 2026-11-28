"""Prepare ODIN geometric superpoints and upright axes from a ReVeMap reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from native_gravity import axis_frame
from odin_runtime import publish, sha256


def vertices(path):
    data = PlyData.read(path)["vertex"].data
    return np.column_stack([data[k] for k in "xyz"])


def mesh_arrays(path):
    mesh = PlyData.read(path)
    data = mesh["vertex"].data
    xyz = np.column_stack([data[k] for k in "xyz"]).astype(np.float32)
    rgb = np.column_stack([data[k] for k in ("red", "green", "blue")]).astype(np.float32)
    face = mesh["face"].data
    key = "vertex_indices" if "vertex_indices" in face.dtype.names else "vertex_index"
    triangles = np.asarray(np.stack(face[key]), dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("Expected a colored triangle mesh")
    return xyz, rgb, triangles


def trajectory(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("gt_at_inference") or doc.get("matrix_convention") != "T_world_camera_m":
        raise ValueError("Expected an estimated ReVeMap T_world_camera_m trajectory")
    records = doc["poses"]
    ids = np.array([int(r["frame_id"]) for r in records])
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("Expected one ordered estimated pose row per raw sensor frame")
    poses = np.array([r["T_world_camera_m"] for r in records], dtype=float).reshape(-1, 4, 4)
    valid = np.array([bool(r.get("valid", True)) for r in records])
    if not valid.any() or not np.isfinite(poses[valid]).all():
        raise ValueError("No valid finite estimated camera poses")
    return poses, valid


def groups_from_mesh(xyz, triangles):
    import segmentator
    import torch

    labels = segmentator.segment_mesh(torch.from_numpy(xyz), torch.from_numpy(triangles)).cpu().numpy()
    keep = np.bincount(labels)[labels] >= 20
    if keep.any() and not keep.all():
        nearest = cKDTree(xyz[keep]).query(xyz[~keep])[1]
        labels[~keep] = labels[keep][nearest]
    return np.unique(labels, return_inverse=True)[1].astype(np.int32)


def rotation_from(path, xyz, triangles, poses):
    if path is None:
        rotation, _ = axis_frame(poses, xyz, triangles)
    elif path.suffix == ".npy":
        rotation = np.load(path, allow_pickle=False)
    else:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        for key in ("rotation", "rotation_xyz_to_upright"):
            if key in receipt:
                rotation = np.asarray(receipt[key], dtype=float)
                break
        else:
            raise ValueError("The anchor result must contain rotation or rotation_xyz_to_upright")
    rotation = np.asarray(rotation)[:3, :3]
    if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .999:
        raise ValueError("Invalid upright rotation")
    return rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Colored target triangle mesh in native coordinates")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--anchor-mesh", type=Path, help="Optional original reconstruction for its room axes")
    parser.add_argument("--anchor-trajectory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.anchor_mesh) != bool(args.anchor_trajectory):
        parser.error("Supply both --anchor-mesh and --anchor-trajectory")
    xyz, _, faces = mesh_arrays(args.mesh)
    groups = groups_from_mesh(xyz, faces)
    ref = args.anchor_mesh or args.mesh
    _, _, af = mesh_arrays(ref) if args.anchor_mesh else (xyz, None, faces)
    ax = vertices(ref).astype(float)
    poses, valid = trajectory(args.anchor_trajectory or args.trajectory)
    rotation = rotation_from(None, ax, af, poses[valid])
    sources = {"mesh": sha256(args.mesh), "anchor_mesh": sha256(ref),
               "anchor_trajectory": sha256(args.anchor_trajectory or args.trajectory)}
    publish(args.output, "groups.npz", {"groups": groups}, {**sources, "driver": sha256(Path(__file__))},
            {**sources, "rotation": rotation.tolist()}, seal_name="geometry.json")


if __name__ == "__main__":
    main()
