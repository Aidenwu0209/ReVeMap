"""Reintegrate a ScanNet RGB-D stream with a corrected native camera trajectory."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial.transform import Rotation, Slerp

from native_geometry.measurements import read_scan
from native_geometry.projective import projective_solution
from refine_native_panoptic import digest


def interpolate_corrections(frame_ids, trajectory, key_ids, adjusted):
    """Apply half the measured keyframe correction in the original world frame."""
    native_keys = np.stack([trajectory[int(k)] for k in key_ids])
    delta = adjusted @ np.linalg.inv(native_keys)
    rotations = Rotation.from_rotvec(
        0.5 * Rotation.from_matrix(delta[:, :3, :3]).as_rotvec()
    )
    query = np.clip(frame_ids, key_ids[0], key_ids[-1])
    correction = np.tile(np.eye(4), (len(frame_ids), 1, 1))
    correction[:, :3, :3] = Slerp(key_ids, rotations)(query).as_matrix()
    for axis in range(3):
        correction[:, axis, 3] = np.interp(query, key_ids, 0.5 * delta[:, axis, 3])
    original = np.stack([trajectory[int(k)] for k in frame_ids])
    return dict(zip(map(int, frame_ids), correction @ original))


def integrate(frames, data, poses):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,
        sdf_trunc=0.08,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    used = 0
    for k in data["raw_indices"]:
        if k not in poses:
            continue
        depth = cv2.imread(str(frames / f"{k}.png"), cv2.IMREAD_UNCHANGED)
        color = cv2.cvtColor(cv2.imread(str(frames / f"{k}.jpg")), cv2.COLOR_BGR2RGB)
        h, w = depth.shape
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, *data["intrinsics"])
        observation = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1000,
            depth_trunc=4.5,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(observation, intrinsics, np.linalg.inv(poses[k]))
        used += 1
    return volume.extract_triangle_mesh(), used


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    trajectory = json.loads(args.trajectory.read_text())
    if (
        trajectory.get("gt_at_inference") is True
        or trajectory.get("matrix_convention") != "T_world_camera_m"
    ):
        raise ValueError("Expected native metric world-from-camera poses")
    valid = {
        p["frame_id"]: np.asarray(p["T_world_camera_m"]).reshape(4, 4)
        for p in trajectory["poses"]
        if p.get("valid", True)
    }
    if 0 not in valid or not all(np.isfinite(p).all() for p in valid.values()):
        raise ValueError("Expected finite native poses including frame zero")
    cloud = PlyData.read(args.map, mmap=False)
    vertex = cloud["vertex"].data
    xyz = np.column_stack([vertex[axis] for axis in "xyz"])
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Expected a nonempty point cloud with finite coordinates")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        tmp = Path(temp)
        data = read_scan(args.sensor, tmp)
        if len(set(data["raw_indices"][::2]) & valid.keys()) < 4:
            raise ValueError("At least four matched native keyframes are required")
        warped, info, key_ids, adjusted = projective_solution(tmp, data, valid, xyz)
        poses = interpolate_corrections(sorted(valid), valid, key_ids, adjusted)
        np.testing.assert_allclose(poses[0], valid[0], atol=1e-6)
        mesh, count = integrate(tmp, data, poses)
        out = tmp / "result"
        out.mkdir()
        if (
            not mesh.has_vertex_colors()
            or not mesh.has_triangles()
            or not o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
        ):
            raise RuntimeError("Failed to extract the colored TSDF mesh")
        coords = xyz + 0.5 * (warped - xyz)
        for axis, values in zip("xyz", coords.T):
            vertex[axis] = values
        cloud.write(out / "map.ply")
        for node in trajectory["poses"]:
            if node["frame_id"] in poses and node.get("valid", True):
                node["T_world_camera_m"] = poses[node["frame_id"]].ravel().tolist()
        trajectory.pop("payload_sha256", None)
        trajectory.setdefault("metadata", {}).update(
            gt_consumed=False, depth_reprojection=True
        )
        (out / "trajectory.json").write_text(json.dumps(trajectory))
        receipt = dict(
            inputs={
                name: digest(getattr(args, name))
                for name in ("sensor", "trajectory", "map")
            },
            outputs={p.name: digest(p) for p in sorted(out.iterdir())},
            projective=info,
            integrated_frames=count,
            points=len(xyz),
            mesh_vertices=len(mesh.vertices),
            mesh_triangles=len(mesh.triangles),
        )
        (out / "result.json").write_text(json.dumps(receipt, indent=2))
        out.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("sensor", "trajectory", "map", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    run(parser.parse_args())
