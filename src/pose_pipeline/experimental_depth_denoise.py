"""Experimental depth rejection for offline RGB-D geometry comparisons.

Depth images are H x W uint16 millimetres, with zero invalid. Poses always
map camera coordinates (metres) to world coordinates. No operation fills holes.
This is an offline candidate, not a claim that estimated poses are accurate.
Temporal rejection deliberately needs visible free-space evidence on both sides
of the centre frame; unsupported or occluded observations are not contradictions.

These candidates are not connected to capture or GUI defaults. Spatial rejection
can remove real small features and thin-line endpoints. Recorded Orbbec tests
also exposed continuous white-wall deletion with temporal rejection and estimated
poses: a depth contradiction can reflect pose or alignment error rather than a
bad measurement. These filters therefore require scene-specific validation and
must not be treated as a demonstrated improvement to reconstruction accuracy.
"""
from __future__ import annotations

from time import perf_counter
from typing import Callable, Iterable, Mapping

import numpy as np


ALGORITHM = "conservative_spatial_temporal_v2"
DEPTH_SCALE = 1000.0


def _depth(depth: np.ndarray) -> np.ndarray:
    if not isinstance(depth, np.ndarray) or depth.dtype != np.dtype(np.uint16):
        raise TypeError("depth must be a NumPy uint16 array in millimetres")
    if depth.ndim != 2 or not all(depth.shape):
        raise ValueError("depth must have a nonempty H x W shape")
    return depth


def _thresholds(abs_threshold_mm: float, rel_threshold: float) -> None:
    if not np.isfinite(abs_threshold_mm) or abs_threshold_mm < 0:
        raise ValueError("abs_threshold_mm must be finite and nonnegative")
    if not np.isfinite(rel_threshold) or rel_threshold < 0:
        raise ValueError("rel_threshold must be finite and nonnegative")


def _intrinsics(value) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        values = [value[name] for name in ("fx", "fy", "cx", "cy")]
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (3, 3):
            if not np.allclose(array[2], [0, 0, 1]) or array[0, 1] != 0 or array[1, 0] != 0:
                raise ValueError("intrinsics must be an unskewed pinhole matrix")
            values = [array[0, 0], array[1, 1], array[0, 2], array[1, 2]]
        elif array.shape == (4,):
            values = array
        else:
            raise ValueError("intrinsics must be (fx, fy, cx, cy), a dict, or a 3 x 3 matrix")
    fx, fy, cx, cy = map(float, values)
    if not all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        raise ValueError("intrinsics must be finite, with positive focal lengths")
    return fx, fy, cx, cy


def _pose(value) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("t_world_camera must be a finite 4 x 4 rigid transform")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("t_world_camera must have homogeneous last row [0,0,0,1]")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError("t_world_camera must contain a proper rigid rotation")
    return pose


def spatial_support_filter(
    depth: np.ndarray,
    *,
    min_support: int = 2,
    abs_threshold_mm: float = 30.0,
    rel_threshold: float = 0.01,
) -> tuple[np.ndarray, dict]:
    """Reject valid pixels with fewer than min_support matching 3 x 3 neighbours.

    The centre is excluded. Matching means |neighbour-centre| <=
    max(abs_threshold_mm, rel_threshold * centre). No padding or wrapped
    neighbours are counted. At least two actual neighbour positions must exist.
    One-pixel-wide lines have two along-line supports in their interior, but
    unsupported endpoints and tiny real features can still be rejected.
    """
    started = perf_counter()
    _depth(depth)
    _thresholds(abs_threshold_mm, rel_threshold)
    if not isinstance(min_support, (int, np.integer)) or not 0 <= min_support <= 8:
        raise ValueError("min_support must be an integer in [0,8]")
    height, width = depth.shape
    valid = depth > 0
    values = depth.astype(np.float32)
    threshold = np.maximum(abs_threshold_mm, rel_threshold * values)
    support = np.zeros(depth.shape, dtype=np.uint8)
    actual_positions = np.zeros(depth.shape, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            if y0 >= y1 or x0 >= x1:
                continue
            centre = (slice(y0, y1), slice(x0, x1))
            neighbour = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
            actual_positions[centre] += 1
            support[centre] += (
                valid[centre] & valid[neighbour]
                & (np.abs(values[centre] - values[neighbour]) <= threshold[centre])
            )
    removed = valid & (actual_positions >= 2) & (support < min_support)
    filtered = depth.copy()
    filtered[removed] = 0
    return filtered, {
        "algorithm": "spatial_depth_support_v2",
        "parameters": {"min_support": int(min_support), "kernel_size": 3,
                       "abs_threshold_mm": float(abs_threshold_mm), "rel_threshold": float(rel_threshold)},
        "valid_pixels": int(valid.sum()), "removed_pixels": int(removed.sum()),
        "kept_pixels": int(np.count_nonzero(filtered)),
        "elapsed_ms": (perf_counter() - started) * 1000,
    }


def temporal_consistency_maps(
    depth: np.ndarray,
    intrinsics,
    t_world_camera,
    references: Iterable[Mapping],
    *,
    abs_threshold_mm: float = 30.0,
    rel_threshold: float = 0.01,
    max_references: int = 8,
) -> dict:
    """Count independent reference support, free-space conflict, and unknowns.

    A reference is {'depth': uint16 image, 'intrinsics': K,
    't_world_camera': 4x4 pose, 'side': -1 or +1}; an optional 'frame_id' or 'id'
    allows explicit deduplication. Otherwise repeated depth array objects are
    deduplicated. Distinct reference images without IDs are the caller's
    responsibility. Missing depth/K/pose references are skipped (not conflicts).

    Reprojection uses camera->world poses and metres; comparisons use reference
    optical-axis depth in millimetres. The nearest projected pixel must be valid
    and not foreground-occluded before any 3 x 3 neighbour can supply support.
    There is no depth interpolation. A conflict requires a complete, valid 3 x 3
    footprint whose depths are ALL behind the projected point beyond tolerance.
    Holes, crop boundaries, foreground occlusion, and behind-camera projections
    are unknown. Each reference contributes at most one vote per centre pixel.
    """
    started = perf_counter()
    _depth(depth)
    _thresholds(abs_threshold_mm, rel_threshold)
    if not isinstance(max_references, (int, np.integer)) or not 1 <= max_references <= 255:
        raise ValueError("max_references must be an integer in [1,255]")
    references = list(references)
    if len(references) > max_references:
        raise ValueError(f"at most {max_references} references are allowed")
    fx, fy, cx, cy = _intrinsics(intrinsics)
    centre_pose = _pose(t_world_camera)
    valid = depth > 0
    y, x = np.nonzero(valid)
    z = depth[y, x].astype(np.float32) / DEPTH_SCALE
    px = (x.astype(np.float32) - cx) * z / fx
    py = (y.astype(np.float32) - cy) * z / fy
    names = ("support_count", "conflict_count", "unknown_count", "conflict_before", "conflict_after")
    counts = {name: np.zeros(z.shape, dtype=np.uint8) for name in names}
    seen = set()
    available = skipped = duplicates = 0
    for reference in references:
        if not isinstance(reference, Mapping):
            raise TypeError("each reference must be a mapping")
        if any(reference.get(name) is None for name in ("depth", "intrinsics", "t_world_camera")):
            skipped += 1
            continue
        side = reference.get("side")
        if side not in (-1, 1):
            raise ValueError("each reference side must be -1 (before) or +1 (after)")
        identity = reference.get("frame_id", reference.get("id"))
        key = ("id", str(identity)) if identity is not None else ("array", id(reference["depth"]))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        reference_depth = _depth(reference["depth"])
        rfx, rfy, rcx, rcy = _intrinsics(reference["intrinsics"])
        reference_pose = _pose(reference["t_world_camera"])
        # inv(T_world_ref) @ T_world_centre, exploiting rigid rotations.
        rotation = reference_pose[:3, :3].T @ centre_pose[:3, :3]
        translation = reference_pose[:3, :3].T @ (centre_pose[:3, 3] - reference_pose[:3, 3])
        rotation = rotation.astype(np.float32)
        translation = translation.astype(np.float32)
        rx = rotation[0, 0] * px + rotation[0, 1] * py + rotation[0, 2] * z + translation[0]
        ry = rotation[1, 0] * px + rotation[1, 1] * py + rotation[1, 2] * z + translation[1]
        rz = rotation[2, 0] * px + rotation[2, 1] * py + rotation[2, 2] * z + translation[2]
        projected_u = np.full(z.shape, -1, dtype=np.float32)
        projected_v = np.full(z.shape, -1, dtype=np.float32)
        positive = rz > 0
        np.divide(rx * rfx, rz, out=projected_u, where=positive)
        np.divide(ry * rfy, rz, out=projected_v, where=positive)
        projected_u += rcx
        projected_v += rcy
        height, width = reference_depth.shape
        in_view = (positive & np.isfinite(projected_u) & np.isfinite(projected_v)
                   & (projected_u >= 0) & (projected_u <= width - 1)
                   & (projected_v >= 0) & (projected_v <= height - 1))
        projected_x = np.zeros(z.shape, dtype=np.int32)
        projected_y = np.zeros(z.shape, dtype=np.int32)
        projected_x[in_view] = np.floor(projected_u[in_view] + 0.5).astype(np.int32)
        projected_y[in_view] = np.floor(projected_v[in_view] + 0.5).astype(np.int32)
        predicted_mm = rz * DEPTH_SCALE
        tolerance = np.maximum(abs_threshold_mm, rel_threshold * predicted_mm)
        nearest = reference_depth[projected_y, projected_x].astype(np.float32)
        eligible = in_view & (nearest > 0) & (nearest >= predicted_mm - tolerance)
        support = np.zeros(z.shape, dtype=bool)
        conflict = eligible & (projected_x >= 1) & (projected_x < width - 1) & (projected_y >= 1) & (projected_y < height - 1)
        # Process only eligible projections; no support is borrowed across an
        # invalid nearest pixel, and no centre can get more than one ref vote.
        indices = np.flatnonzero(eligible)
        ex, ey = projected_x[indices], projected_y[indices]
        predicted = predicted_mm[indices]
        tolerance_sub = tolerance[indices]
        support_sub = np.zeros(indices.shape, dtype=bool)
        conflict_sub = conflict[indices].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = ex + dx, ey + dy
                actual = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
                sample = reference_depth[np.clip(ny, 0, height - 1), np.clip(nx, 0, width - 1)].astype(np.float32)
                observed = actual & (sample > 0)
                support_sub |= observed & (np.abs(sample - predicted) <= tolerance_sub)
                conflict_sub &= observed & (sample > predicted + tolerance_sub)
        support[indices] = support_sub
        conflict[indices] = conflict_sub
        counts["support_count"] += support
        counts["conflict_count"] += conflict
        counts["unknown_count"] += ~(support | conflict)
        counts["conflict_before" if side == -1 else "conflict_after"] += conflict
        available += 1
    result = {}
    for name, array in counts.items():
        result[name] = np.zeros(depth.shape, dtype=np.uint8)
        result[name][y, x] = array
    result.update({"available_refs": available, "skipped_refs": skipped,
                   "duplicate_refs": duplicates, "elapsed_ms": (perf_counter() - started) * 1000})
    return result


def temporal_consistency_filter(
    depth: np.ndarray,
    intrinsics,
    t_world_camera,
    references: Iterable[Mapping],
    *,
    min_conflicts: int = 2,
    abs_threshold_mm: float = 30.0,
    rel_threshold: float = 0.01,
    max_references: int = 8,
) -> tuple[np.ndarray, dict]:
    """Reject only unsupported points contradicted before AND after the frame."""
    started = perf_counter()
    if not isinstance(min_conflicts, (int, np.integer)) or min_conflicts < 2:
        raise ValueError("min_conflicts must be an integer of at least 2")
    maps = temporal_consistency_maps(depth, intrinsics, t_world_camera, references,
                                     abs_threshold_mm=abs_threshold_mm, rel_threshold=rel_threshold,
                                     max_references=max_references)
    valid = depth > 0
    removed = (valid & (maps["support_count"] == 0)
               & (maps["conflict_count"] >= min_conflicts)
               & (maps["conflict_before"] >= 1) & (maps["conflict_after"] >= 1))
    filtered = depth.copy()
    filtered[removed] = 0
    return filtered, {
        "algorithm": "bidirectional_visible_free_space_v2",
        "parameters": {"min_conflicts": int(min_conflicts), "abs_threshold_mm": float(abs_threshold_mm),
                       "rel_threshold": float(rel_threshold), "max_references": int(max_references)},
        "valid_pixels": int(valid.sum()), "removed_pixels": int(removed.sum()),
        "kept_pixels": int(np.count_nonzero(filtered)),
        "pixels_with_support": int(np.count_nonzero(maps["support_count"])),
        "pixels_with_any_conflict": int(np.count_nonzero(maps["conflict_count"])),
        "available_refs": maps["available_refs"], "skipped_refs": maps["skipped_refs"],
        "duplicate_refs": maps["duplicate_refs"],
        "elapsed_ms": (perf_counter() - started) * 1000,
    }


def robust_denoise_depth(
    depth: np.ndarray,
    *,
    intrinsics=None,
    t_world_camera=None,
    references=None,
    spatial: bool = True,
    min_support: int = 2,
    min_conflicts: int = 2,
    abs_threshold_mm: float = 30.0,
    rel_threshold: float = 0.01,
    max_references: int = 8,
    bilateral: Callable | None = None,
) -> tuple[np.ndarray, dict]:
    """Compose spatial rejection, optional temporal rejection, optional smoothing.

    Pass an existing masked bilateral function as ``bilateral`` if wanted. It may
    return either an array or (array, stats). The callback cannot fill original
    or newly rejected holes; attempts to do so are rejected as contract errors.
    References should contain raw observations with reliable, fixed poses.
    """
    started = perf_counter()
    _depth(depth)
    filtered = depth.copy()
    stages = {}
    if spatial:
        filtered, stages["spatial"] = spatial_support_filter(filtered, min_support=min_support,
            abs_threshold_mm=abs_threshold_mm, rel_threshold=rel_threshold)
    if references is not None:
        if intrinsics is None or t_world_camera is None:
            raise ValueError("temporal filtering requires centre intrinsics and t_world_camera")
        filtered, stages["temporal"] = temporal_consistency_filter(filtered, intrinsics, t_world_camera,
            references, min_conflicts=min_conflicts, abs_threshold_mm=abs_threshold_mm,
            rel_threshold=rel_threshold, max_references=max_references)
    if bilateral is not None:
        invalid_before = filtered == 0
        result = bilateral(filtered.copy())
        if isinstance(result, tuple):
            smoothed, callback_stats = result
        else:
            smoothed, callback_stats = result, {}
        _depth(smoothed)
        if smoothed.shape != depth.shape or not np.array_equal(smoothed == 0, invalid_before):
            raise ValueError("bilateral callback must preserve shape and all invalid pixels")
        filtered = smoothed.copy()
        stages["bilateral"] = callback_stats
    valid = depth > 0
    removed = valid & (filtered == 0)
    kept = valid & (filtered > 0)
    delta = np.abs(depth.astype(np.int32) - filtered.astype(np.int32))
    return filtered, {
        "algorithm": ALGORITHM, "depth_scale": DEPTH_SCALE,
        "valid_pixels": int(valid.sum()), "removed_pixels": int(removed.sum()),
        "kept_pixels": int(kept.sum()), "changed_kept_pixels": int(np.count_nonzero(delta[kept])),
        "original_holes_preserved": bool(np.all(filtered[~valid] == 0)),
        "stages": stages, "elapsed_ms": (perf_counter() - started) * 1000,
    }
