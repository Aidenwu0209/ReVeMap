"""Conservative, single-frame filtering for uint16 millimetre depth images.

Only existing valid pixels are smoothed. A zero depth is invalid and stays zero;
it contributes no weight to nearby pixels. There is no temporal state, hole
filling, point rejection, or change to the image geometry. Large discontinuities
are excluded with a hard range gate in addition to bilateral range weighting.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np


ALGORITHM = "masked_bilateral_v1"
DEFAULT_CONFIG = {
    "kernel_size": 3,
    "sigma_space_px": 1.0,
    "sigma_depth_mm": 10.0,
    "max_neighbor_delta_mm": 30.0,
    "max_adjustment_mm": 10.0,
}


def denoise_depth(depth: np.ndarray, mode: str = "bilateral") -> tuple[np.ndarray, dict]:
    """Return an independent depth image and JSON-compatible per-frame stats.

    ``depth`` must be an H x W uint16 array in millimetres, with 0 invalid.
    ``mode`` is ``bilateral`` or ``off``. The input is never modified, including
    when disabled. Reported changes are measured after rounding to uint16, and
    ``mean_abs_change_mm`` averages over *all valid input pixels*, not just those
    changed. ``elapsed_ms`` measures filtering and the associated statistics.

    These fixed defaults suppress small spatial fluctuations while bounding
    each change. They do not remove gross outliers. Real surfaces separated by
    at most 30 mm can still blend within the 3 x 3 neighbourhood.
    """
    started = perf_counter()
    if not isinstance(depth, np.ndarray) or depth.dtype != np.dtype(np.uint16):
        raise TypeError("depth must be a NumPy uint16 array in millimetres")
    if depth.ndim != 2:
        raise ValueError("depth must have shape H x W")
    if mode not in ("off", "bilateral"):
        raise ValueError("depth denoise mode must be 'off' or 'bilateral'")

    parameters = dict(DEFAULT_CONFIG)
    filtered = depth.copy()
    valid = depth != 0
    valid_pixels = int(np.count_nonzero(valid))
    enabled = mode == "bilateral"

    if enabled and valid_pixels:
        height, width = depth.shape
        center_depth = depth.astype(np.float32)
        # The centre has unit weight; accumulate relative changes to preserve
        # constant surfaces exactly and avoid summing large absolute depths.
        weighted_delta = np.zeros(depth.shape, dtype=np.float32)
        weight_sum = np.ones(depth.shape, dtype=np.float32)
        space_denom = 2.0 * parameters["sigma_space_px"] ** 2
        range_denom = 2.0 * parameters["sigma_depth_mm"] ** 2

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                # Crop to actual overlap: no wraparound, border duplication, or
                # padding with invented depth observations.
                y_start, y_stop = max(0, -dy), min(height, height - dy)
                x_start, x_stop = max(0, -dx), min(width, width - dx)
                if y_start >= y_stop or x_start >= x_stop:
                    continue
                center = (slice(y_start, y_stop), slice(x_start, x_stop))
                neighbor = (
                    slice(y_start + dy, y_stop + dy),
                    slice(x_start + dx, x_stop + dx),
                )
                delta = center_depth[neighbor] - center_depth[center]
                accepted = valid[center] & valid[neighbor]
                accepted &= np.abs(delta) <= parameters["max_neighbor_delta_mm"]
                spatial_weight = np.exp(-(dx * dx + dy * dy) / space_denom)
                weight = np.exp(-(delta * delta) / range_denom)
                weight *= spatial_weight
                weight *= accepted
                weighted_delta[center] += weight * delta
                weight_sum[center] += weight

        adjustment = weighted_delta / weight_sum
        np.clip(
            adjustment,
            -parameters["max_adjustment_mm"],
            parameters["max_adjustment_mm"],
            out=adjustment,
        )
        rounded = np.rint(center_depth[valid] + adjustment[valid])
        # Valid pixels must remain valid even at the uint16 range boundaries.
        filtered[valid] = np.clip(rounded, 1, np.iinfo(np.uint16).max).astype(np.uint16)

    change = np.abs(filtered.astype(np.int32) - depth.astype(np.int32))
    changed_pixels = int(np.count_nonzero(change))
    stats = {
        "algorithm": ALGORITHM,
        "enabled": enabled,
        "parameters": parameters,
        "valid_pixels": valid_pixels,
        "changed_pixels": changed_pixels,
        "mean_abs_change_mm": float(change.sum(dtype=np.int64) / valid_pixels)
        if valid_pixels else 0.0,
        "max_abs_change_mm": int(change.max()) if change.size else 0,
        "elapsed_ms": (perf_counter() - started) * 1000.0,
    }
    return filtered, stats
