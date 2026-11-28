"""Joint metric camera adjustment with symmetric raw image and depth residuals."""

import cv2
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from .measurements import transfer_motion


def frame_features(d, f, K, sift):
    im = cv2.imread(str(d / f"{f}.jpg"), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(d / f"{f}.png"), cv2.IMREAD_UNCHANGED).astype(float) * 0.001
    key, desc = sift.detectAndCompute(im, None)
    if desc is None:
        return np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 128), np.float32)
    uv = np.array([k.pt for k in key])
    x, y = np.rint(uv).astype(int).T
    h, w = depth.shape
    x = x.clip(2, w - 3)
    y = y.clip(2, h - 3)
    z = depth[y, x]
    keep = (z > 0.25) & (z < 4.5)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        keep &= abs(depth[y + dy, x + dx] - z) < 0.06
    uv, z, desc = uv[keep], z[keep], desc[keep]
    xyz = np.column_stack(
        ((uv[:, 0] - K[0, 2]) * z / K[0, 0], (uv[:, 1] - K[1, 2]) * z / K[1, 1], z)
    )
    return uv, xyz, desc


def matches(a, b, K):
    if min(len(a[2]), len(b[2])) < 10:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    bound = lambda ds: {
        x.queryIdx: x.trainIdx for x, y in ds if x.distance < 0.78 * y.distance
    }
    ab = bound(matcher.knnMatch(a[2], b[2], k=2))
    ba = bound(matcher.knnMatch(b[2], a[2], k=2))
    pairs = [(i, j) for i, j in ab.items() if ba.get(j) == i]
    if len(pairs) < 12:
        return None
    i, j = np.asarray(pairs).T
    ok, r, t, inliers = cv2.solvePnPRansac(
        a[1][i],
        b[0][j],
        K,
        None,
        iterationsCount=120,
        reprojectionError=3.5,
        confidence=0.995,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 12:
        return None
    selected = inliers.ravel()
    i, j = i[selected], j[selected]
    delta = a[1][i] @ cv2.Rodrigues(r)[0].T + t.ravel() - b[1][j]
    good = np.linalg.norm(delta, axis=1) < 0.085
    if good.sum() < 12 or good.mean() < 0.65:
        return None
    i, j = i[good], j[good]
    sample = np.linspace(0, len(i) - 1, min(70, len(i)), dtype=int)
    i, j = i[sample], j[sample]
    return a[1][i], b[1][j], a[0][i], b[0][j]


def linearize(P, Q, U, K, a, b):
    world = P @ a[:3, :3].T + a[:3, 3]
    target = (world - b[:3, 3]) @ b[:3, :3]
    x, y, z = target.T
    z = z.clip(0.12, None)
    uv = np.column_stack((K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]))
    # Pixels and raw depth have different noise scales. Use a symmetric robust loss.
    sz = 0.015 + 0.007 * Q[:, 2] ** 2
    residual = np.column_stack(((uv - U) / 1.5, (target[:, 2] - Q[:, 2]) / sz))
    j = np.zeros((len(P), 3, 3))
    j[:, 0, 0] = K[0, 0] / (z * 1.5)
    j[:, 0, 2] = -K[0, 0] * x / (z * z * 1.5)
    j[:, 1, 1] = K[1, 1] / (z * 1.5)
    j[:, 1, 2] = -K[1, 1] * y / (z * z * 1.5)
    j[:, 2, 2] = 1 / sz
    grad = j @ b[:3, :3].T
    jac = np.concatenate((np.cross(world[:, None, :], grad), grad), axis=2)
    wt = np.minimum(1.0, 2.5 / np.maximum(np.linalg.norm(residual, axis=1), 1e-6))
    A = jac.reshape(-1, 6)
    w = np.repeat(wt, 3)
    return (A.T * w) @ A, (A.T * w) @ residual.ravel()


def optimize(T, edges, K):
    if not edges:
        return T.copy()
    n = len(T)
    poses = T.copy()
    for it in range(9):
        blocks = {}
        rhs = np.zeros((n, 6))

        def add(i, j, H):
            if (i, j) in blocks:
                blocks[i, j] += H
            else:
                blocks[i, j] = H.copy()

        for i, j, (a, b, ua, ub) in edges:
            h, g = linearize(a, b, ub, K, poses[i], poses[j])
            hr, gr = linearize(b, a, ua, K, poses[j], poses[i])
            h += hr
            g -= gr
            add(i, i, h)
            add(j, j, h)
            add(i, j, -h)
            add(j, i, -h)
            rhs[i] -= g
            rhs[j] += g
        row, col, val = [], [], []
        for (i, j), h in blocks.items():
            row.extend(np.repeat(6 * i + np.arange(6), 6))
            col.extend(np.tile(6 * j + np.arange(6), 6))
            val.extend(h.ravel())
        H = coo_matrix((val, (row, col)), shape=(6 * n, 6 * n)).tocsr()[6:, 6:]
        step = spsolve(
            H + eye(H.shape[0], format="csr") * 800.0, rhs.ravel()[6:]
        ).reshape(-1, 6)
        scale = np.maximum(1.0, np.linalg.norm(step[:, 3:], axis=1) / 0.12)
        step /= scale[:, None]
        update = np.tile(np.eye(4), (n - 1, 1, 1))
        update[:, :3, :3] = Rotation.from_rotvec(step[:, :3]).as_matrix()
        update[:, :3, 3] = step[:, 3:]
        poses[1:] = update @ poses[1:]
        if np.max(abs(step)) < 1e-4:
            break
    return poses


def feature_adjustment(src, meta, pose, xyz):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(391)
    fids = [k for k in meta["raw_indices"][::3] if k in pose]
    fx, fy, cx, cy = meta["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    T = np.stack([pose[k] for k in fids])
    sift = cv2.SIFT_create(nfeatures=850)
    features = [frame_features(src, f, K, sift) for f in fids]
    z = T[:, :3, 2]
    centers = T[:, :3, 3]
    pairs = set()
    for i in range(len(T)):
        for j in (i + 1, i + 2, i + 3):
            if j < len(T):
                pairs.add((i, j))
        d = np.linalg.norm(centers - centers[i], axis=1)
        valid = (d < 1.8) & (z @ z[i] > 0.42) & (np.abs(np.arange(len(T)) - i) > 4)
        for chunk in np.array_split(np.flatnonzero(valid), 6):
            if len(chunk):
                pairs.add(tuple(sorted((i, int(chunk[d[chunk].argmin()])))))
    edges = []
    for i, j in sorted(pairs):
        match = matches(features[i], features[j], K)
        if match is not None:
            edges.append((i, j, match))
    new = optimize(T, edges, K)
    # Carry original surface labels with the nearby measured keyframes.
    before, after = [], []
    for fid, a, b in zip(fids, T, new):
        dep = (
            cv2.imread(str(src / f"{fid}.png"), cv2.IMREAD_UNCHANGED)[::8, ::8] * 0.001
        )
        ys, xs = np.where((dep > 0.2) & (dep < 4.5))
        vd = dep[ys, xs]
        local = np.column_stack(((xs * 8 - cx) * vd / fx, (ys * 8 - cy) * vd / fy, vd))
        before.append(local @ a[:3, :3].T + a[:3, 3])
        after.append(local @ b[:3, :3].T + b[:3, 3])
    before, after = np.vstack(before), np.vstack(after)
    return transfer_motion(xyz, before, after), {
        "keyframes": len(T),
        "candidates": len(pairs),
        "accepted": len(edges),
    }
