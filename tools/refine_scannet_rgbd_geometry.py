"""Refine an existing labeled map using the original ScanNet RGB-D measurements."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from native_geometry.features import feature_adjustment
from native_geometry.measurements import combine, read_scan, smooth_motion
from native_geometry.projective import projective
from plyfile import PlyData


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    trajectory = json.loads(args.trajectory.read_text())
    if (
        trajectory.get("gt_at_inference") is True
        or trajectory.get("matrix_convention") != "T_world_camera_m"
    ):
        raise ValueError("Expected native metric world-from-camera poses")
    poses = {
        p["frame_id"]: np.asarray(p["T_world_camera_m"]).reshape(4, 4)
        for p in trajectory["poses"]
        if p.get("valid", True)
    }
    if 0 not in poses or not all(np.isfinite(p).all() for p in poses.values()):
        raise ValueError("The first frame and all selected native poses must be finite")
    cloud = PlyData.read(args.map, mmap=False)
    vertex = cloud["vertex"].data
    xyz = np.column_stack([vertex[k] for k in "xyz"])
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Expected a nonempty point cloud with finite coordinates")
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        tmp = Path(temp)
        meta = read_scan(args.sensor, tmp)
        if (
            min(len(set(meta["raw_indices"][::step]) & poses.keys()) for step in (2, 3))
            < 4
        ):
            raise ValueError(
                "At least four matched native keyframes per estimator are required"
            )
        depth_xyz, depth_info = projective(tmp, meta, poses, xyz)
        feature_xyz, feature_info = feature_adjustment(tmp, meta, poses, xyz)
        warped = combine(xyz, depth_xyz, feature_xyz)
        if args.smooth:
            warped = smooth_motion(xyz, warped)
        if not np.isfinite(warped).all():
            raise ValueError("Nonfinite refined coordinates")
        for j, k in enumerate("xyz"):
            vertex[k] = warped[:, j]
        result = tmp / "result"
        result.mkdir()
        cloud.write(result / "map_labeled.ply")
        receipt = dict(
            sensor_sha256=sha(args.sensor),
            trajectory_sha256=sha(args.trajectory),
            original_map_sha256=sha(args.map),
            refined_map_sha256=sha(result / "map_labeled.ply"),
            points=len(xyz),
            native_first_camera_fixed=True,
            smooth=args.smooth,
            projective=depth_info,
            feature_bundle=feature_info,
        )
        (result / "result.json").write_text(json.dumps(receipt, indent=2))
        result.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth", action="store_true", help="Smooth the measured local corrections")
    for item in ("sensor", "trajectory", "map", "output"):
        parser.add_argument("--" + item, type=Path, required=True)
    run(parser.parse_args())
