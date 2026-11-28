"""Filter added RGB-D surface points contradicted by both estimated trajectories."""

import argparse
import json
import struct
import zlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def _header(stream):
    version, length = struct.unpack("<IQ", stream.read(12))
    if version != 4:
        raise ValueError("Expected a ScanNet v4 .sens stream")
    stream.seek(length, 1)
    calibration = np.frombuffer(stream.read(256), "<f4").reshape(4, 4, 4)
    color, depth, _, _, w, h, scale, count = struct.unpack("<iiIIIIfQ", stream.read(36))
    if (color, depth, scale) != (2, 1, 1000) or min(w, h) < 5 or not count:
        raise ValueError("Expected JPEG RGB and zlib millimeter depth")
    if any(not np.allclose(calibration[i], np.eye(4), atol=1e-6) for i in (1, 3)):
        raise ValueError("Expected registered sensor extrinsics")
    k = calibration[2]
    return (
        int(count),
        int(h),
        int(w),
        (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])),
    )


def _votes(points, pose, depth, k):
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    indices = np.flatnonzero((z > 0.3) & (z < 4.45))
    p = camera[indices]
    u = np.rint(k[0] * p[:, 0] / p[:, 2] + k[2]).astype(int)
    v = np.rint(k[1] * p[:, 1] / p[:, 2] + k[3]).astype(int)
    h, w = depth.shape
    inside = (u > 1) & (v > 1) & (u < w - 2) & (v < h - 2)
    u, v, indices = u[inside], v[inside], indices[inside]
    samples = np.stack(
        [
            depth[v, u],
            depth[v - 1, u],
            depth[v + 1, u],
            depth[v, u - 1],
            depth[v, u + 1],
        ],
        axis=1,
    )
    low, high = samples.min(axis=1), samples.max(axis=1)
    stable = (low > 250) & (high < 4500) & (high - low < 80)
    indices = indices[stable]
    predicted = z[indices]
    residual = samples[stable, 0].astype("f4") * 0.001 - predicted
    supports = indices[np.abs(residual) <= 0.035 + 0.004 * predicted]
    free = indices[residual > 0.105 + 0.008 * predicted]
    return supports, free


def free_space_keep(sensor, primary_poses, secondary_poses, additions):
    """Return a boolean mask in the original additions order. Poses are T_world_camera in meters."""
    x = np.asarray(additions, dtype="f4")
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite N x 3 addition coordinates")
    grid = np.floor(x / 0.04).astype("i8") + 65536
    if grid.size and (grid.min() < 0 or grid.max() >= 131072):
        raise ValueError("Point outside the 4 cm voxel key range")
    keys = (grid[:, 0] << 34) | (grid[:, 1] << 17) | grid[:, 2]
    _, first, indices = np.unique(keys, return_index=True, return_inverse=True)
    cells = x[first]
    votes = np.zeros((2, len(cells), 2), dtype=np.int32)
    poses = [np.asarray(p, dtype=float) for p in (primary_poses, secondary_poses)]
    with Path(sensor).open("rb") as stream:
        count, h, w, k = _header(stream)
        if any(p.shape != (count, 4, 4) or not np.isfinite(p).all() for p in poses):
            raise ValueError(
                "Expected one finite 4 x 4 pose per raw sensor frame in each trajectory"
            )
        chosen = set(range(0, count, 30)) | {count - 1}
        for i in range(count):
            # Seek past the reference camera matrix and sensor timestamps.
            stream.seek(80, 1)
            nc, nd = struct.unpack("<QQ", stream.read(16))
            stream.seek(nc, 1)
            if i not in chosen:
                stream.seek(nd, 1)
                continue
            depth = np.frombuffer(zlib.decompress(stream.read(nd)), "<u2").reshape(h, w)
            for j, trajectory in enumerate(poses):
                supported, free = _votes(cells, trajectory[i], depth, k)
                votes[j, supported, 0] += 1
                votes[j, free, 1] += 1
    reject = (votes[:, :, 1] >= 2) & (votes[:, :, 1] > votes[:, :, 0])
    return ~reject.all(axis=0)[indices.reshape(-1)]


def _poses(path):
    data = json.loads(path.read_bytes())
    if (
        data.get("matrix_convention") != "T_world_camera_m"
        or data.get("gt_at_inference") is True
    ):
        raise ValueError("Expected estimated ReVeMap world-from-camera trajectories")
    rows = data["poses"]
    if not rows or any(
        row["frame_id"] != i or not row.get("valid", True) for i, row in enumerate(rows)
    ):
        raise ValueError(
            "Estimated trajectories must contain complete contiguous raw frame IDs"
        )
    return np.stack([np.asarray(row["T_world_camera_m"]).reshape(4, 4) for row in rows])


def _points(path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as f:
            return f["xyz"]
    if path.suffix.lower() == ".ply":
        from plyfile import PlyData

        points = PlyData.read(path)["vertex"].data
        return np.column_stack([points[k] for k in "xyz"])
    raise ValueError("Supply an additions NPZ with xyz or a PLY file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "sensor",
        "primary-trajectory",
        "secondary-trajectory",
        "additions",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    keep = free_space_keep(
        args.sensor,
        _poses(args.primary_trajectory),
        _poses(args.secondary_trajectory),
        _points(args.additions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=args.output.parent, suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, keep=keep)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Kept {int(keep.sum())} of {len(keep)} additions")


if __name__ == "__main__":
    main()
