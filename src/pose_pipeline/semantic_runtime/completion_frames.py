"""Select additional depth views using only estimated poses and the coarse input map."""

from bisect import bisect_left, bisect_right

import numpy as np


def _visible_cells(points, pose):
    # A central camera cone estimates surface overlap across different image formats.
    # Retain the nearest surface sample per angular bin to limit visibility through walls.
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    ids = np.flatnonzero((camera[:, 2] >= 0.3) & (camera[:, 2] <= 5.0))
    xy = camera[ids, :2] / camera[ids, 2, None]
    inside = np.max(np.abs(xy), axis=1) < 0.35
    ids = ids[inside]
    bins = np.floor((xy[inside] + 0.35) * 32).astype(int)
    keys = bins[:, 0] * 32 + bins[:, 1]
    order = np.lexsort((camera[ids, 2], keys))
    _, first = np.unique(keys[order], return_index=True)
    return ids[order[first]]


def select_depth_frames(semantic, poses, available, original, policy):
    """Keep every eligible mask and the reference number of depth-only frame slots."""
    valid = set(poses).intersection(available)
    baseline = sorted((semantic | {s + 2 for s in semantic}) & valid)
    if policy == "offset2":
        return baseline
    if policy != "complementary":
        raise ValueError(f"Unknown depth frame policy: {policy}")
    if not baseline or not len(original):
        return baseline

    # One surface representative per 15 cm cell prevents dense walls dominating the score.
    finite = np.asarray(original)[np.isfinite(original).all(axis=1)]
    _, kept = np.unique(np.floor(finite / 0.15).astype(np.int64), axis=0, return_index=True)
    points = finite[kept]
    if not len(points):
        return baseline
    if len(points) > 8192:
        points = points[np.linspace(0, len(points) - 1, 8192, dtype=int)]
    observed = np.zeros(len(points), dtype=np.int32)
    for fid in sorted(semantic & valid):
        observed[_visible_cells(points, poses[fid])] += 1

    valid_ids = sorted(valid)
    selected = list(baseline)
    for i, fid in enumerate(baseline):
        if fid in semantic or i == 0 or i + 1 == len(baseline):
            continue
        left, right = baseline[i - 1], baseline[i + 1]
        # One independent slot between adjacent masks keeps the sampling phase of
        # every retained semantic frame fixed. Irregular spacing uses the reference.
        if left not in semantic or right not in semantic:
            continue
        candidates = valid_ids[bisect_right(valid_ids, left):bisect_left(valid_ids, right)]
        best_key = None
        best_cells = None
        for candidate in candidates:
            pose = poses[candidate]
            cells = _visible_cells(points, pose)
            # Prefer a view of under-observed surfaces with parallax from both masks.
            # Forward direction is a unit vector, so length 1 scales its angular
            # change to an equivalent displacement at a one metre working distance.
            displacement = min(
                np.linalg.norm(pose[:3, 3] - poses[anchor][:3, 3])
                + np.linalg.norm(pose[:3, 2] - poses[anchor][:3, 2])
                for anchor in (left, right)
            )
            score = np.sum(1.0 / (1 + observed[cells])) * displacement
            key = (score, -abs(candidate - fid), -candidate)
            if best_key is None or key > best_key:
                best_key, best_cells = key, cells
                selected[i] = candidate
        if best_cells is not None:
            observed[best_cells] += 1
    return selected
