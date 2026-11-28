"""Refine native keyframe poses with bidirectional projective RGB-D odometry."""

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .measurements import transfer_motion

REG = o3d.pipelines.registration
ODO = o3d.pipelines.odometry


def relative(T, i, j):
    return np.linalg.solve(T[j], T[i])


def size(T):
    return np.linalg.norm(T[:3, 3]), Rotation.from_matrix(T[:3, :3]).magnitude()


def projective(frames, data, pose, xyz):
    cv2.setNumThreads(1)
    fids = data["raw_indices"][::2]
    fids = [i for i in fids if i in pose]
    T = np.stack([pose[i] for i in fids])
    fx, fy, cx, cy = data["intrinsics"]
    cams, targets = [], []
    for fid in fids:
        color = cv2.imread(str(frames / f"{fid}.jpg"))
        dep = cv2.imread(str(frames / f"{fid}.png"), cv2.IMREAD_UNCHANGED)
        # Block mean color and nearest native depth sample with explicit intrinsics.
        h, w = dep.shape
        color = cv2.cvtColor(
            cv2.resize(color, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        depth = np.ascontiguousarray(dep[::2, ::2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1000,
            depth_trunc=4.5,
            convert_rgb_to_intensity=True,
        )
        cams.append(rgbd)
        sampler = dep[::8, ::8] * 0.001
        ys, xs = np.where((sampler > 0.2) & (sampler < 4.5))
        z = sampler[ys, xs]
        targets.append(
            np.column_stack(((8 * xs - cx) * z / fx, (8 * ys - cy) * z / fy, z))
        )
    K = o3d.camera.PinholeCameraIntrinsic(
        w // 2, h // 2, fx / 2, fy / 2, cx / 2, cy / 2
    )
    centers = T[:, :3, 3]
    forward = T[:, :3, 2]
    graph = REG.PoseGraph()
    for value in T:
        graph.nodes.append(REG.PoseGraphNode(value))
    pairs = set()
    for i in range(len(T)):
        for j in (i + 1, i + 2):
            if j < len(T):
                pairs.add((i, j))
        distance = np.linalg.norm(centers - centers[i], axis=1)
        valid = (
            (distance < 1.45)
            & ((forward @ forward[i]) > 0.48)
            & (np.abs(np.arange(len(T)) - i) > 4)
        )
        # Revisit candidates from separated times, so redundant neighbors do not dominate.
        for chunk in np.array_split(np.flatnonzero(valid), 4):
            if len(chunk):
                j = int(chunk[np.argmin(distance[chunk])])
                pairs.add(tuple(sorted((i, j))))
    opt = ODO.OdometryOption()
    opt.iteration_number_per_pyramid_level = o3d.utility.IntVector([12, 7, 4])
    opt.depth_diff_max = 0.1
    opt.depth_min = 0.2
    opt.depth_max = 4.5
    term = ODO.RGBDOdometryJacobianFromHybridTerm()
    edges = []
    for i in range(len(T) - 1):
        graph.edges.append(
            REG.PoseGraphEdge(i, i + 1, relative(T, i, i + 1), np.eye(6) * 300, False)
        )
    for i, j in sorted(pairs):
        initial = relative(T, i, j)
        good, M, H = ODO.compute_rgbd_odometry(cams[i], cams[j], K, initial, term, opt)
        if not good or H[5, 5] < 1500:
            continue
        shift, angle = size(np.linalg.solve(initial, M))
        if shift > 0.22 or angle > np.deg2rad(12):
            continue
        back, B, _ = ODO.compute_rgbd_odometry(
            cams[j], cams[i], K, np.linalg.inv(initial), term, opt
        )
        if not back:
            continue
        drift, rotation = size(B @ M)
        if drift > 0.035 or rotation > np.deg2rad(2):
            continue
        graph.edges.append(REG.PoseGraphEdge(i, j, M, H, j > i + 2))
        edges.append([i, j, float(shift), float(angle), float(drift)])
    params = REG.GlobalOptimizationOption(
        max_correspondence_distance=0.06, edge_prune_threshold=0.20, reference_node=0
    )
    REG.global_optimization(
        graph,
        REG.GlobalOptimizationLevenbergMarquardt(),
        REG.GlobalOptimizationConvergenceCriteria(),
        params,
    )
    poses = np.stack([np.asarray(n.pose) for n in graph.nodes])
    assert np.allclose(poses[0], T[0], atol=1e-6)
    raw = np.vstack([z @ p[:3, :3].T + p[:3, 3] for z, p in zip(targets, T)])
    updated = np.vstack([z @ p[:3, :3].T + p[:3, 3] for z, p in zip(targets, poses)])
    return transfer_motion(xyz, raw, updated), {
        "keyframes": len(T),
        "candidates": len(pairs),
        "accepted": len(edges),
    }
