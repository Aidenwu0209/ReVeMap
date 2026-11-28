"""Jointly refine a native map with short RGB-D and independent long-baseline factors."""

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

from .measurements import transfer_motion

ODO = o3d.pipelines.odometry
REG = o3d.pipelines.registration


def relative(poses, left, right):
    return np.linalg.solve(poses[right], poses[left])


def size(transform):
    return np.linalg.norm(transform[:3, 3]), Rotation.from_matrix(
        transform[:3, :3]
    ).magnitude()


def joint_solution(samples, meta, pose, xyz, measured):
    fx, fy, cx, cy = meta["intrinsics"]
    long_edges = [
        e
        for e in measured
        if e["accepted"]
        and e["translation_diff"] <= 1.0
        and e["angle_diff_deg"] <= 25
        and e["frame_i"] in pose
        and e["frame_j"] in pose
    ]
    image_ids = [k for k in meta["raw_indices"] if k in pose]
    node_ids = sorted(
        set(image_ids) | {e[k] for e in long_edges for k in ("frame_i", "frame_j")}
    )
    node = {i: t for t, i in enumerate(node_ids)}
    T = np.stack([pose[k] for k in node_ids])
    graph = REG.PoseGraph()
    for camera in T:
        graph.nodes.append(REG.PoseGraphNode(camera))
    for i in range(len(T) - 1):
        graph.edges.append(
            REG.PoseGraphEdge(i, i + 1, relative(T, i, i + 1), np.eye(6) * 450.0, False)
        )
    color_depth, local = {}, {}
    for k in image_ids:
        rgb = cv2.imread(str(samples / f"{k}.jpg"))
        depth = cv2.imread(str(samples / f"{k}.png"), cv2.IMREAD_UNCHANGED)
        h, w = depth.shape
        rgb = cv2.cvtColor(
            cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        color_depth[k] = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(np.ascontiguousarray(depth[::2, ::2])),
            depth_scale=1000,
            depth_trunc=4.5,
        )
        zmap = depth[::8, ::8] * 0.001
        y, x = np.where((zmap > 0.2) & (zmap < 4.5))
        z = zmap[y, x]
        local[k] = np.column_stack(((8 * x - cx) * z / fx, (8 * y - cy) * z / fy, z))
    K = o3d.camera.PinholeCameraIntrinsic(
        w // 2, h // 2, fx / 2, fy / 2, cx / 2, cy / 2
    )
    opt = ODO.OdometryOption()
    opt.iteration_number_per_pyramid_level = o3d.utility.IntVector([12, 7, 4])
    opt.depth_diff_max, opt.depth_min, opt.depth_max = 0.1, 0.2, 4.5
    term = ODO.RGBDOdometryJacobianFromHybridTerm()
    candidates = set()
    image_T = np.stack([pose[k] for k in image_ids])
    for i, fid in enumerate(image_ids):
        for gap in (1, 2, 4):
            if i + gap < len(image_ids):
                candidates.add((i, i + gap))
        if i % 2:
            continue
        ds = np.linalg.norm(image_T[:, :3, 3] - pose[fid][:3, 3], axis=1)
        allowed = (
            (ds < 1.45)
            & (image_T[:, :3, 2] @ pose[fid][:3, 2] > 0.48)
            & (abs(np.arange(len(ds)) - i) > 8)
        )
        for group in np.array_split(np.flatnonzero(allowed), 4):
            if len(group):
                candidates.add(tuple(sorted((i, int(group[ds[group].argmin()])))))
    accepted, information = 0, []
    for a, b in sorted(candidates):
        ka, kb = image_ids[a], image_ids[b]
        i, j = node[ka], node[kb]
        prior = relative(T, i, j)
        ok, M, H = ODO.compute_rgbd_odometry(
            color_depth[ka], color_depth[kb], K, prior, term, opt
        )
        if not ok or H[5, 5] < 1500:
            continue
        shift, angle = size(np.linalg.solve(prior, M))
        if shift > 0.22 or angle > np.deg2rad(12):
            continue
        ok, reverse, _ = ODO.compute_rgbd_odometry(
            color_depth[kb], color_depth[ka], K, np.linalg.inv(prior), term, opt
        )
        if not ok:
            continue
        drift, angle = size(reverse @ M)
        if drift > 0.035 or angle > np.deg2rad(2):
            continue
        graph.edges.append(REG.PoseGraphEdge(i, j, M, H, b - a > 4))
        accepted += 1
        information.append(H[5, 5])
    # Each independent learned long-baseline constraint represents dozens of held-out RGB-D matches.
    # Use two ordinary short-baseline edges' median information and a bounded empirical precision.
    ref = np.median(information) if information else 5000.0
    for e in long_edges:
        reliability = min(2.0, max(0.5, 2.5 / max(0.5, e["hold_median_cm"])))
        H = np.diag([3.3] * 3 + [1.0] * 3) * (2 * ref * reliability)
        graph.edges.append(
            REG.PoseGraphEdge(
                node[e["frame_i"]],
                node[e["frame_j"]],
                np.array(e["T_target_source"]),
                H,
                False,
            )
        )
    REG.global_optimization(
        graph,
        REG.GlobalOptimizationLevenbergMarquardt(),
        REG.GlobalOptimizationConvergenceCriteria(),
        REG.GlobalOptimizationOption(
            max_correspondence_distance=0.06, edge_prune_threshold=0.2, reference_node=0
        ),
    )
    solved = np.stack([np.asarray(n.pose) for n in graph.nodes])
    np.testing.assert_allclose(solved[0], T[0], atol=1e-6)
    original, corrected = [], []
    for k in image_ids:
        src_points = local[k]
        original.append(src_points @ pose[k][:3, :3].T + pose[k][:3, 3])
        sol = solved[node[k]]
        corrected.append(src_points @ sol[:3, :3].T + sol[:3, 3])
    coordinates = transfer_motion(xyz, np.vstack(original), np.vstack(corrected))
    counts = {
        "nodes": len(node_ids),
        "dense_edges": accepted,
        "long_edges": len(long_edges),
    }
    return coordinates, counts, np.asarray(node_ids), solved


def joint_geometry(samples, meta, pose, xyz, measured):
    coordinates, counts, _, _ = joint_solution(samples, meta, pose, xyz, measured)
    return coordinates, counts


def full_frame_poses(native, keys, solved, fraction):
    """Interpolate joint corrections in the native world basis, then blend each camera."""
    frame_ids = np.array(sorted(native))
    base = np.stack([native[i] for i in frame_ids])
    before = np.stack([native[k] for k in keys])
    rotations = Rotation.from_matrix(
        solved[:, :3, :3] @ before[:, :3, :3].transpose(0, 2, 1)
    )
    query = np.clip(frame_ids, keys[0], keys[-1])
    full = base.copy()
    full[:, :3, :3] = Slerp(keys, rotations)(query).as_matrix() @ base[:, :3, :3]
    for axis in range(3):
        delta = solved[:, axis, 3] - before[:, axis, 3]
        full[:, axis, 3] += np.interp(query, keys, delta)
    correction = Rotation.from_matrix(
        full[:, :3, :3] @ base[:, :3, :3].transpose(0, 2, 1)
    )
    output = base.copy()
    output[:, :3, :3] = (
        Rotation.from_rotvec(fraction * correction.as_rotvec()).as_matrix()
        @ base[:, :3, :3]
    )
    output[:, :3, 3] += fraction * (full[:, :3, 3] - base[:, :3, 3])
    return dict(zip(frame_ids.tolist(), output))
