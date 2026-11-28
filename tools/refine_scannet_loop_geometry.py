"""Measure RGB-D revisits and apply their 0.6-strength pose correction to a native map."""

import argparse
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def sensor_intrinsics(path):
    with path.open("rb") as stream:
        version, length = struct.unpack("<IQ", stream.read(12))
        if version != 4:
            raise ValueError("Expected a ScanNet v4 sensor file")
        stream.seek(length, 1)
        cameras = np.frombuffer(stream.read(256), "<f4").reshape(4, 4, 4)
        k = cameras[2]
        return [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])]


def digest(path):
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.trajectory.read_text())
    if (
        data.get("gt_at_inference") is not False
        or data.get("matrix_convention") != "T_world_camera_m"
    ):
        raise ValueError(
            "Expected a metric ReVeMap trajectory estimated from the sensor"
        )
    valid = [n for n in data["poses"] if n.get("valid", True)]
    poses = {
        n["frame_id"]: np.asarray(n["T_world_camera_m"]).reshape(4, 4) for n in valid
    }
    if (
        len(poses) != len(valid)
        or 0 not in poses
        or not all(np.isfinite(v).all() for v in poses.values())
    ):
        raise ValueError(
            "Expected unique native frame IDs, finite poses and a first camera"
        )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    common = dict(
        sensor_sha256=digest(args.sensor),
        trajectory_sha256=digest(args.trajectory),
        GT_used=False,
    )
    with TemporaryDirectory(dir=args.output.parent) as directory:
        tmp = Path(directory)
        result = tmp / "result"
        result.mkdir()
        if args.command == "measure":
            from native_geometry.loop_measurements import measure

            edges = measure(
                args.sensor, data, sensor_intrinsics(args.sensor), args.checkpoint
            )
            receipt = dict(
                **common, checkpoint_sha256=digest(args.checkpoint), edges=edges
            )
            (result / "edges.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False)
            )
        else:
            import cv2
            from plyfile import PlyData
            from native_geometry.joint import joint_geometry
            from native_geometry.measurements import read_scan

            cv2.setNumThreads(1)
            external = json.loads(args.edges.read_text())
            if any(external[k] != value for k, value in common.items()):
                raise ValueError(
                    "Loop measurements do not match the supplied sensor and trajectory"
                )
            frames = read_scan(args.sensor, tmp)
            if len(set(frames["raw_indices"]) & poses.keys()) < 4:
                raise ValueError("Expected at least four measured native keyframes")
            raw = PlyData.read(args.map, mmap=False)
            vertex = raw["vertex"].data
            xyz = np.column_stack([vertex[c] for c in "xyz"])
            if not len(xyz) or not np.isfinite(xyz).all():
                raise ValueError(
                    "Native map must have finite coordinates and at least one vertex"
                )
            cloud = (
                raw if args.labeled is None else PlyData.read(args.labeled, mmap=False)
            )
            labeled = cloud["vertex"].data
            if len(labeled) != len(vertex) or any(
                not np.array_equal(labeled[c], vertex[c])
                for c in ("red", "green", "blue")
            ):
                raise ValueError(
                    "The labeled input must retain the native vertex order and colors"
                )
            warped, counts = joint_geometry(tmp, frames, poses, xyz, external["edges"])
            changed = xyz + 0.6 * (warped - xyz)
            if not np.isfinite(changed).all():
                raise ValueError("Nonfinite coordinates after graph optimization")
            for j, axis in enumerate("xyz"):
                labeled[axis] = changed[:, j]
            output = result / "map_labeled.ply"
            cloud.write(output)
            info = dict(
                **common,
                **counts,
                map_sha256=digest(args.map),
                edges_sha256=digest(args.edges),
                labeled_sha256=digest(args.labeled or args.map),
                fraction=0.6,
                points=len(xyz),
                prediction_sha256=digest(output),
            )
            (result / "result.json").write_text(json.dumps(info, indent=2))
        result.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    measure = subs.add_parser(
        "measure", help="Measure independent revisits with MASt3R on CPU"
    )
    apply = subs.add_parser(
        "apply", help="Combine verified revisits with bidirectional RGB-D odometry"
    )
    for cmd in (measure, apply):
        for name in ("sensor", "trajectory", "output"):
            cmd.add_argument("--" + name, type=Path, required=True)
    measure.add_argument("--checkpoint", type=Path, required=True)
    apply.add_argument(
        "--map",
        type=Path,
        required=True,
        help="Original native map with its unwarped coordinates",
    )
    apply.add_argument(
        "--labeled",
        type=Path,
        help="Optional voted map with the same vertex order and colors",
    )
    apply.add_argument("--edges", type=Path, required=True)
    run(parser.parse_args())
