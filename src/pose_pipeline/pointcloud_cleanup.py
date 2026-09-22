"""Explicit, single-pass radius cleanup of a point-cloud PLY.

The 6 cm / four-other-vertices defaults were evaluated on a 2 cm TSDF map.
Sparse real structures can be removed. Neighbour counts are not independent
view support, and removed vertices are candidates, not ground-truth noise.
This command does not change capture, pose estimation, or fusion defaults.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def radius_keep_mask(points, *, radius_m=0.06, min_other_neighbors=4):
    """Count other vertices once, including distinct vertices at equal positions.

    Queries use the complete input cloud; survivors are not filtered again.
    Input positions and their order are never changed.
    """
    from scipy.spatial import cKDTree

    if isinstance(radius_m, (bool, np.bool_)) or not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and positive")
    if (isinstance(min_other_neighbors, (bool, np.bool_))
            or not isinstance(min_other_neighbors, (int, np.integer))
            or min_other_neighbors < 1):
        raise ValueError("min_other_neighbors must be a positive integer")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("points must be a nonempty, finite N x 3 array")
    counts = cKDTree(points).query_ball_point(
        points, float(radius_m), return_length=True, workers=2,
    ) - 1
    return counts >= min_other_neighbors


def cleanup_pointcloud(input_path, output_dir, *, radius_m=0.06, min_other_neighbors=4):
    """Write kept/removed vertices, source indices, and a completion receipt.

    Coordinates must be in metres. Supports a single vertex element with scalar
    properties. Meshes, edges and list-valued properties are rejected rather than silently dropping metadata
    or leaving invalid topology. Every scalar vertex field is copied unchanged.
    An empty removed set is still written as a valid zero-vertex PLY. Removing
    all vertices fails. Existing output directories are never overwritten.
    """
    from plyfile import PlyData, PlyListProperty

    source = Path(input_path).resolve(strict=True)
    requested_output = Path(output_dir)
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f"Output already exists: {requested_output}")
    output = requested_output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    payload = source.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    cloud = PlyData.read(io.BytesIO(payload))
    if len(cloud.elements) != 1 or cloud.elements[0].name != "vertex":
        raise ValueError("Expected a point-cloud PLY with only a vertex element; topology is unsupported")
    if any(isinstance(prop, PlyListProperty) for prop in cloud["vertex"].properties):
        raise ValueError("Only scalar vertex properties are supported")
    vertices = cloud["vertex"].data
    if not {"x", "y", "z"}.issubset(vertices.dtype.names):
        raise ValueError("Point cloud must contain x, y and z vertex properties")
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    keep = radius_keep_mask(points, radius_m=radius_m, min_other_neighbors=min_other_neighbors)
    if not keep.any():
        raise ValueError("No vertices survive; reduce the filter strength")
    if _digest(source) != source_hash:
        raise RuntimeError("Input changed while computing cleanup")

    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, mask in (("kept", keep), ("removed", ~keep)):
        indices = np.flatnonzero(mask)
        selected = vertices[indices].copy()
        result = copy.deepcopy(cloud)
        result["vertex"].data = selected
        result.text = False  # Preserve floating-point values without ASCII rounding.
        cloud_path = output / f"{name}.ply"
        index_path = output / f"{name}_indices.npy"
        result.write(cloud_path)
        np.save(index_path, indices, allow_pickle=False)
        check = PlyData.read(cloud_path)["vertex"].data
        if check.dtype != selected.dtype or len(check) != len(selected):
            raise RuntimeError("PLY round trip changed the vertex layout")
        for field in selected.dtype.names:
            equal = np.array_equal(check[field], selected[field], equal_nan=selected[field].dtype.kind == "f")
            if not equal:
                raise RuntimeError(f"PLY round trip changed vertex field {field}")
        artifacts[cloud_path.name] = _digest(cloud_path)
        artifacts[index_path.name] = _digest(index_path)
    if _digest(source) != source_hash:
        raise RuntimeError("Input changed during cleanup; no completion receipt was written")
    receipt = {
        "schema": "conservative_pointcloud_cleanup.v2",
        "source": str(source), "source_sha256": source_hash,
        "radius_m": float(radius_m), "min_other_neighbors": int(min_other_neighbors),
        "iterations": 1, "input_points": len(vertices),
        "kept_points": int(keep.sum()), "removed_points": int((~keep).sum()),
        "input_unchanged": True, "point_positions_unchanged": True,
        "vertex_fields_unchanged": True, "artifact_sha256": artifacts,
        "meaning": "Sparse-point cleanup; removed vertices are candidates, not ground-truth noise.",
        "default_parameter_scope": "Evaluated on a 2 cm TSDF map; validate sparse structures and other sampling densities.",
    }
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input scalar-property point-cloud PLY, coordinates in metres")
    parser.add_argument("--output", type=Path, required=True, help="New output directory (must not exist)")
    parser.add_argument("--radius-m", type=float, default=0.06)
    parser.add_argument("--min-other-neighbors", type=int, default=4)
    args = parser.parse_args(argv)
    receipt = cleanup_pointcloud(args.input, args.output, radius_m=args.radius_m,
                                 min_other_neighbors=args.min_other_neighbors)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
