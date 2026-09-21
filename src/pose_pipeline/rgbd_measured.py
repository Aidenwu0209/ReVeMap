"""Refine dense RGB-D keyframes with the original measured SIFT/SVD graph.

The graph's non-keyframe poses only transport the existing dense estimates.
They must pass through :func:`rgbd_refill.run_visual_refill` before fusion.
This module preserves the original SVD measurements and information matrices;
it does not enable the separate information-coordinate or calibration pilots.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import (
    PoseRecord,
    load_manifest,
    load_trajectory,
    sha256_file,
    validate_se3,
    write_manifest,
    write_trajectory,
)


@lru_cache(maxsize=1)
def _sift():
    return cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)


@lru_cache(maxsize=1)
def _matcher():
    return cv2.BFMatcher(cv2.NORM_L2)


def features(frame, scale):
    """Use the original grayscale, resize, depth-sample and SIFT conventions."""
    color = cv2.imread(str(frame.color_path), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None or depth.ndim != 2:
        raise ValueError(f"Cannot decode RGB-D frame {frame.frame_id}")
    h, w = depth.shape
    ch, cw = color.shape
    width = min(cw, 640)
    height = round(ch * width / cw)
    color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    fx, fy, cx, cy = frame.intrinsics
    sx, sy = width / w, height / h
    fx, fy, cx, cy = fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
    if frame.rotate_ccw:
        color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fx, fy, cx, cy = fy, fx, cy, width - 1 - cx
    kp, desc = _sift().detectAndCompute(color, None)
    uv = np.array([p.pt for p in kp], dtype=np.float64).reshape(-1, 2)
    xy = np.rint(uv).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, depth.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, depth.shape[0] - 1)
    z = depth[xy[:, 1], xy[:, 0]].astype(float) / scale
    pts = np.c_[(uv[:, 0] - cx) * z / fx, (uv[:, 1] - cy) * z / fy, z]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    return uv, desc, pts, k


def fit(a, b):
    """Fit the original proper rigid transform from source to target points."""
    ca, cb = a.mean(0), b.mean(0)
    u, s, vt = np.linalg.svd((a - ca).T @ (b - cb))
    rot = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = cb - rot @ ca
    return transform, s


def measured_pair(a, b):
    """Measure source→target with PnP initialization and three rigid SVD fits.

    The returned H keeps the original target-point, rotation/translation ordering
    and fixed .03 m scale. It is not the separate Ad(T) congruence experiment or a
    calibrated independent-observation covariance.
    """
    if a[1] is None or b[1] is None or min(len(a[1]), len(b[1])) < 2:
        return None
    matcher = _matcher()
    forward = matcher.knnMatch(a[1], b[1], k=2)
    reverse = matcher.knnMatch(b[1], a[1], k=2)
    back = {
        x.queryIdx: x.trainIdx for x, y in reverse if x.distance < 0.75 * y.distance
    }
    pairs = [
        (x.queryIdx, x.trainIdx)
        for x, y in forward
        if x.distance < 0.75 * y.distance and back.get(x.trainIdx) == x.queryIdx
    ]
    if len(pairs) < 30:
        return None
    si, ti = np.array(pairs).T
    x, y, uv = a[2][si], b[2][ti], b[0][ti]
    good = (x[:, 2] > 0.2) & (x[:, 2] < 4.5) & (y[:, 2] > 0.2) & (y[:, 2] < 4.5)
    x, y, uv = x[good], y[good], uv[good]
    if len(x) < 30:
        return None
    cv2.setRNGSeed(43)
    ok, rv, tv, ind = cv2.solvePnPRansac(
        x,
        uv,
        b[3],
        None,
        iterationsCount=1024,
        reprojectionError=2.5,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or ind is None or len(ind) < 25:
        return None
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rv)[0]
    transform[:3, 3] = tv.ravel()
    for _ in range(3):
        err = np.linalg.norm(x @ transform[:3, :3].T + transform[:3, 3] - y, axis=1)
        valid = err < 0.07
        if valid.sum() < 25 or valid.mean() < 0.5:
            return None
        transform, s = fit(x[valid], y[valid])
    err = np.linalg.norm(x @ transform[:3, :3].T + transform[:3, 3] - y, axis=1)
    valid = err < 0.07
    if valid.sum() < 25 or valid.mean() < 0.5 or s[1] < 0.01:
        return None
    z = x[valid] @ transform[:3, :3].T + transform[:3, 3]
    if len(z) > 400:
        z = z[np.linspace(0, len(z) - 1, 400, dtype=int)]
    jacobian = np.zeros((len(z), 3, 6))
    jacobian[:, :, 3:] = np.eye(3)
    jacobian[:, 0, 1], jacobian[:, 0, 2] = z[:, 2], -z[:, 1]
    jacobian[:, 1, 0], jacobian[:, 1, 2] = -z[:, 2], z[:, 0]
    jacobian[:, 2, 0], jacobian[:, 2, 1] = z[:, 1], -z[:, 0]
    information = np.einsum("nai,naj->ij", jacobian, jacobian) / (0.03**2)
    return (
        transform,
        information,
        {
            "matches": len(pairs),
            "both_depth_matches": len(x),
            "inliers": int(valid.sum()),
            "inlier_ratio": float(valid.mean()),
            "inlier_rmse_m": float(np.sqrt(np.mean(err[valid] ** 2))),
            "scatter_singular_values": s.tolist(),
        },
    )


def depth_check(a, b, k, transform):
    """Original bidirectional depth support check; this is not a GT metric."""
    fx, fy, cx, cy = k
    y, x = np.mgrid[1 : a.shape[0] : 4, 1 : a.shape[1] : 4]
    z = a[y, x]
    valid = (z > 0.2) & (z < 4.5)
    xyz = np.c_[
        (x[valid] - cx) * z[valid] / fx, (y[valid] - cy) * z[valid] / fy, z[valid]
    ]
    pts = xyz @ transform[:3, :3].T + transform[:3, 3]
    zz = pts[:, 2]
    u = np.rint(fx * pts[:, 0] / np.maximum(zz, 1e-6) + cx).astype(int)
    v = np.rint(fy * pts[:, 1] / np.maximum(zz, 1e-6) + cy).astype(int)
    keep = (zz > 0.2) & (u >= 0) & (u < b.shape[1]) & (v >= 0) & (v < b.shape[0])
    d, zz = b[v[keep], u[keep]], zz[keep]
    good = (d > 0.2) & (d < 4.5)
    residual = np.abs(d[good] - zz[good])
    inlier = residual < 0.08
    return {
        "projected": int(good.sum()),
        "inlier_ratio": float(inlier.mean()) if len(residual) else 0.0,
        "median_m": float(np.median(residual)) if len(residual) else None,
        "rmse_inliers_m": (
            float(np.sqrt(np.mean(residual[inlier] ** 2))) if inlier.any() else None
        ),
    }


def _read_depth(frame, scale):
    from reconstruction.rgbd_refusion import _read_rgbd

    # Keep the existing RGB/depth resize, CCW and intrinsic handling. The
    # original helper's extra Open3D RGBDImage was unused by depth_check.
    _, depth, intrinsics = _read_rgbd(frame)
    return depth.astype(np.float32) / scale, intrinsics


def _quality(previous, current, transform):
    return (
        depth_check(previous[0], current[0], current[1], transform),
        depth_check(current[0], previous[0], current[1], np.linalg.inv(transform)),
    )


def _verified_result(directory: Path) -> dict:
    """Read a completed stage and check its immutable output hashes."""
    directory = Path(directory).resolve()
    receipt = json.loads((directory / "result.json").read_text())
    if receipt.get("gt_consumed") is not False:
        raise ValueError(f"Stage must declare gt_consumed=false: {directory}")
    seal = receipt.get("seal")
    if not isinstance(seal, dict) or not seal:
        raise ValueError(f"Stage has no output seal: {directory}")
    for name, expected in seal.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha256_file(path) != expected:
            raise ValueError(f"Stage output SHA mismatch: {path}")
    return receipt


def run_measured_graph(dense_dir: Path, output_dir: Path) -> dict:
    """Create a measured graph from a complete dense run, without using GT.

    Output must not exist. Failure leaves the partial directory for diagnosis and
    never substitutes an identity or a partial trajectory as a completed result.
    """
    import open3d as o3d

    source, out = Path(dense_dir).resolve(), Path(output_dir).resolve()
    receipt = _verified_result(source)
    source_result_sha = sha256_file(source / "result.json")
    manifest = load_manifest(source / "raw_manifest.json")
    original = load_trajectory(source / "trajectory.json")[0]
    if len(original) != len(manifest.frames) or any(
        p.frame_id != f.frame_id or p.timestamp_us != f.timestamp_us or not p.valid
        for p, f in zip(original, manifest.frames)
    ):
        raise ValueError("Measured graph requires a complete ordered raw trajectory")
    old = np.stack([p.t_world_camera for p in original])
    with np.load(source / "final_keyframes.npz", allow_pickle=False) as archive:
        timestamps = archive["timestamps"]
    if (
        timestamps.ndim != 1
        or not len(timestamps)
        or not np.isfinite(timestamps).all()
        or not np.equal(timestamps, np.rint(timestamps)).all()
    ):
        raise ValueError("Keyframe timestamps must be nonempty integer raw ordinals")
    keyframes = np.unique(np.r_[timestamps.astype(int), len(manifest.frames) - 1])
    if keyframes[0] != 0 or keyframes[-1] >= len(manifest.frames):
        raise ValueError("Keyframes must begin at zero and remain inside the raw input")
    initial = old[keyframes]
    n = len(keyframes)
    out.mkdir(parents=True, exist_ok=False)
    write_manifest(out / "raw_manifest.json", manifest)
    cv2.setNumThreads(1)
    started = time.time()

    pairs = set((i, i + 1) for i in range(n - 1))
    pairs.update((i, i + 2) for i in range(n - 2))
    centers = initial[:, :3, 3]
    for i in range(n):
        distance = np.linalg.norm(centers - centers[i], axis=1)
        candidates = np.flatnonzero(
            (distance < 1.0) & (np.abs(keyframes - keyframes[i]) > 60)
        )
        keep = []
        for j in candidates:
            angle = Rotation.from_matrix(
                initial[j, :3, :3].T @ initial[i, :3, :3]
            ).magnitude()
            if angle < np.deg2rad(55):
                keep.append(int(j))
        quarters = np.linspace(0, n, 5, dtype=int)
        for low, high in zip(quarters[:-1], quarters[1:]):
            choices = [j for j in keep if low <= j < high]
            if choices:
                pairs.add(tuple(sorted((i, min(choices, key=lambda j: distance[j])))))
    pairs = sorted(pairs, key=lambda ij: (ij[1] - ij[0] > 2, ij))
    (out / "pair_selection.json").write_text(
        json.dumps(
            {
                "keyframe_ordinals": keyframes.tolist(),
                "pairs": pairs,
                "source": "estimated pose proximity followed by measured RGB-D verification",
                "gt_consumed": False,
            },
            indent=2,
        )
    )

    @lru_cache(maxsize=1024)
    def feature(i):
        return features(manifest.frames[int(keyframes[i])], manifest.depth_scale)

    @lru_cache(maxsize=12)
    def rgbd(i):
        return _read_depth(manifest.frames[int(keyframes[i])], manifest.depth_scale)

    reg = o3d.pipelines.registration
    graph = reg.PoseGraph()
    for transform in initial:
        graph.nodes.append(reg.PoseGraphNode(transform))
    for i in range(n - 1):
        graph.edges.append(
            reg.PoseGraphEdge(
                i,
                i + 1,
                np.linalg.inv(initial[i + 1]) @ initial[i],
                np.eye(6),
                False,
            )
        )
    accepted = 0
    with (out / "measured_edges.jsonl").open("x") as log:
        for step, (i, j) in enumerate(pairs):
            measured = measured_pair(feature(i), feature(j))
            row = {
                "source_node": i,
                "target_node": j,
                "source_ordinal": int(keyframes[i]),
                "target_ordinal": int(keyframes[j]),
                "accepted_measured_constraint": False,
            }
            if measured is not None:
                transform, information, fit_info = measured
                forward, reverse = _quality(rgbd(i), rgbd(j), transform)
                good = (
                    min(forward["projected"], reverse["projected"]) >= 200
                    and min(forward["inlier_ratio"], reverse["inlier_ratio"]) >= 0.85
                )
                row.update(
                    fit=fit_info,
                    forward=forward,
                    reverse=reverse,
                    T_target_source=transform.tolist(),
                    accepted_measured_constraint=bool(good),
                )
                if good:
                    graph.edges.append(
                        reg.PoseGraphEdge(i, j, transform, information, j - i > 2)
                    )
                    accepted += 1
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % 50 == 0:
                print(
                    "MEASURED_EDGES",
                    manifest.sequence_id,
                    step,
                    len(pairs),
                    accepted,
                    round(time.time() - started, 1),
                    flush=True,
                )
    criteria = reg.GlobalOptimizationConvergenceCriteria()
    criteria.max_iteration = 100
    options = reg.GlobalOptimizationOption(
        max_correspondence_distance=0.07,
        edge_prune_threshold=0.25,
        preference_loop_closure=1.0,
        reference_node=0,
    )
    reg.global_optimization(
        graph, reg.GlobalOptimizationLevenbergMarquardt(), criteria, options
    )
    optimized = np.stack([node.pose for node in graph.nodes])
    for i, transform in enumerate(optimized):
        validate_se3(transform, f"optimized anchor {i}")
    np.savez_compressed(
        out / "optimized_keyframes.npz",
        timestamps=keyframes,
        poses_T_world_camera=optimized,
    )
    nearest = np.abs(
        np.arange(len(manifest.frames))[:, None] - keyframes[None, :]
    ).argmin(1)
    correction = optimized @ np.linalg.inv(initial)
    full = correction[nearest] @ old
    full[keyframes] = optimized
    if not np.isfinite(full).all():
        raise RuntimeError("Nonfinite graph poses")
    records = [
        PoseRecord(
            f.frame_id,
            f.timestamp_us,
            transform,
            True,
            "measured_keyframe_graph_pending_visual_refill",
        )
        for f, transform in zip(manifest.frames, full)
    ]
    write_trajectory(
        out / "trajectory.json",
        records,
        sequence_id=manifest.sequence_id,
        arm="diagnostic",
        metadata={
            "gt_consumed": False,
            "nonkeyframes_reoptimized_after_graph": False,
            "nonkeyframe_relative_source": "original dense visually estimated anchor-relative transform",
            "promotion_eligible": False,
        },
    )
    np.save(out / "final_raw_poses.npy", full)
    _verified_result(source)
    if sha256_file(source / "result.json") != source_result_sha:
        raise RuntimeError("Dense source receipt changed during graph construction")
    report = {
        "gt_consumed": False,
        "promotion_eligible": False,
        "requires_visual_refill_before_promotion": True,
        "identity_fallback_used": False,
        "algorithm": "original_measured_sift_svd_graph",
        "information_model": "original_target_point_JtJ_rotation_translation",
        "raw_frame_count": len(manifest.frames),
        "final_pose_count": len(records),
        "keyframe_count": n,
        "measured_constraint_count": accepted,
        "weak_original_dense_prior_count": n - 1,
        "source_receipt_sha256": source_result_sha,
        "wrapper_sha256": sha256_file(Path(__file__)),
        "runtime_versions": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "open3d": o3d.__version__,
        },
        "runtime_s": time.time() - started,
        "seal": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
    }
    (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("MEASURED_GRAPH_SEALED", manifest.sequence_id, n, accepted, flush=True)
    return report
