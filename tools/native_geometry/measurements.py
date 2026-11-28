"""Read RGB-D measurements and transfer their estimated motions to native vertices."""

import struct
import zlib

import cv2
import numpy as np
from scipy.spatial import cKDTree


def read_scan(sensor, directory):
    """Decode every sixth measurement from a ScanNet v4 JPEG/zlib sensor stream."""
    frames = []
    with sensor.open("rb") as stream:
        version, name_length = struct.unpack("<IQ", stream.read(12))
        if version != 4:
            raise ValueError(f"Unsupported ScanNet sensor version {version}")
        stream.seek(name_length, 1)
        calibration = np.frombuffer(stream.read(256), "<f4").reshape(4, 4, 4)
        color_format, depth_format, _, _, w, h, scale, count = struct.unpack(
            "<iiIIIIfQ", stream.read(36)
        )
        if (color_format, depth_format, scale) != (2, 1, 1000) or min(h, w) < 8:
            raise ValueError(
                "Expected registered JPEG RGB and zlib-compressed millimeter depth"
            )
        for frame in range(count):
            stream.seek(64, 1)  # Skip the reference camera matrix stored in .sens.
            _, _, color_bytes, depth_bytes = struct.unpack("<QQQQ", stream.read(32))
            if frame % 6:
                stream.seek(color_bytes + depth_bytes, 1)
                continue
            color = cv2.imdecode(
                np.frombuffer(stream.read(color_bytes), np.uint8), cv2.IMREAD_COLOR
            )
            depth = np.frombuffer(
                zlib.decompress(stream.read(depth_bytes)), "<u2"
            ).reshape(h, w)
            color = cv2.resize(color, (w, h), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(
                str(directory / f"{frame}.jpg"), color
            ) or not cv2.imwrite(str(directory / f"{frame}.png"), depth):
                raise OSError(f"Failed to write the decoded RGB-D sample {frame}")
            frames.append(frame)
    k = calibration[2]
    return {"raw_indices": frames, "intrinsics": [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]}


def transfer_motion(xyz, before, after):
    if len(before) < 4:
        raise ValueError(
            "At least four valid depth samples are required to transfer motion"
        )
    distances, neighbors = cKDTree(before).query(xyz, k=4, workers=2)
    weights = 1 / np.maximum(distances, 0.005) ** 2
    weights[distances > 0.12] = 0
    delta = ((after - before)[neighbors] * weights[:, :, None]).sum(axis=1)
    delta /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    return (xyz + delta).astype(np.float32)


def combine(xyz, projective_xyz, feature_xyz):
    projective_delta, feature_delta = projective_xyz - xyz, feature_xyz - xyz
    a = np.linalg.norm(projective_delta, axis=1)
    b = np.linalg.norm(feature_delta, axis=1)
    direction = (projective_delta * feature_delta).sum(axis=1) / np.maximum(
        0.0001, a * b
    )
    magnitude = np.minimum(a, b) / np.maximum(0.01, np.maximum(a, b))
    agreement = np.clip(direction, 0, 1) * magnitude
    return (
        xyz + (0.5 * projective_delta + 0.2 * agreement[:, None] * feature_delta)
    ).astype(np.float32)


def smooth_motion(xyz, warped):
    """Average measured displacements across neighboring 12 cm control cells."""
    cells = np.floor(xyz / 0.12).astype(np.int32)
    _, inverse, count = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    delta = warped - xyz
    centers = np.column_stack([np.bincount(inverse, weights=x) / count for x in xyz.T])
    changes = np.column_stack(
        [np.bincount(inverse, weights=x) / count for x in delta.T]
    )
    # A one-element list keeps the neighbor dimension for single-cell maps.
    neighbors = list(range(1, min(12, len(centers)) + 1))
    distance, index = cKDTree(centers).query(centers, k=neighbors, workers=2)
    weight = np.exp(-(distance**2) / (2 * 0.18**2)) * np.sqrt(count[index])
    mean = (changes[index] * weight[:, :, None]).sum(axis=1) / weight.sum(axis=1)[
        :, None
    ]
    return (xyz + (0.6 * delta + 0.4 * mean[inverse])).astype(np.float32)
