"""Check large visual-refill corrections against depth from independent neighbors."""

from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from .rgbd_measured import _read_depth, depth_check


def verify_refill(manifest, graph, refill, anchors):
    """Return a complete trajectory; replace only nonanchors verified on both sides."""
    # Keep graph restorations and the projection comparisons at graph precision.
    graph = np.asarray(graph, dtype=np.float64)
    refill = np.asarray(refill, dtype=np.float64)
    n = len(manifest.frames)
    if graph.shape != (n, 4, 4) or refill.shape != graph.shape:
        raise ValueError("Graph and refill must cover the complete manifest")
    fixed = np.zeros(n, dtype=bool)
    fixed[anchors] = True
    if not np.allclose(graph[fixed], refill[fixed], atol=1e-5, rtol=1e-5):
        raise ValueError("Graph and refill disagree on fixed anchors")
    distance = np.linalg.norm(graph[:, :3, 3] - refill[:, :3, 3], axis=1)
    relative = graph[:, :3, :3].transpose(0, 2, 1) @ refill[:, :3, :3]
    angle = Rotation.from_matrix(relative).magnitude()
    candidates = ~fixed & ((distance > 0.05) | (angle > np.deg2rad(3.0)))
    corrected = refill.copy()
    times = np.array([f.timestamp_us for f in manifest.frames])

    @lru_cache(maxsize=10)
    def depth(i):
        return _read_depth(manifest.frames[i], manifest.depth_scale)

    rows = []
    changed = []
    for i in np.flatnonzero(candidates).tolist():
        neighbors = []
        for direction in (-1, 1):
            for offset in range(1, 4):
                j = i + offset * direction
                if j < 0 or j >= n or abs(times[j] - times[i]) > 2_000_000:
                    break
                if not candidates[j]:
                    neighbors.append(j)
                    break
        checks = []
        if len(neighbors) == 2:
            a, ka = depth(i)
            for j in neighbors:
                b, kb = depth(j)
                trials = {}
                for label, pose in (("graph", graph[i]), ("refill", refill[i])):
                    transform = np.linalg.solve(refill[j], pose)
                    forward = depth_check(a, b, ka, transform, target_intrinsics=kb)
                    reverse = depth_check(b, a, kb, np.linalg.inv(transform), target_intrinsics=ka)
                    trials[label] = {
                        "support": min(forward["projected"], reverse["projected"]),
                        "agreement": min(forward["inlier_ratio"], reverse["inlier_ratio"]),
                    }
                checks.append({"neighbor": j, **trials})
        accept = len(checks) == 2 and all(
            min(c["graph"]["support"], c["refill"]["support"]) >= 200
            and c["graph"]["agreement"] >= 0.85
            and c["refill"]["agreement"] < 0.50
            for c in checks
        )
        if accept:
            corrected[i] = graph[i]
            changed.append(i)
        rows.append({"ordinal": i, "frame_id": manifest.frames[i].frame_id,
                     "translation_m": float(distance[i]), "angle_deg": float(np.rad2deg(angle[i])),
                     "checks": checks, "restored": accept})
    return corrected, {"policy": "bidirectional-depth-v1", "frames": n, "fixed_anchors": int(fixed.sum()),
                       "restored_ordinals": changed, "candidates": rows}
