"""Measure long-baseline camera motion with MASt3R correspondences and raw sensor depths."""

import io
import struct
import time
import zlib

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

W, H = 320, 240


def read_rgbd(path, choose):
    ret = {}
    with path.open("rb") as f:
        if int.from_bytes(f.read(4), "little") != 4:
            raise ValueError("Expected ScanNet sensor format v4")
        size = int.from_bytes(f.read(8), "little")
        f.seek(size + 256, 1)
        cc, dc, wc, hc, wd, hd = struct.unpack("<2i4I", f.read(24))
        if (cc, dc) != (2, 1):
            raise ValueError("Expected JPEG color and zlib depth")
        scale, n = struct.unpack("<fQ", f.read(12))
        if scale != 1000:
            raise ValueError("Expected millimeter sensor depths")
        for i in range(n):
            f.seek(80, 1)  # skip stored reference camera-to-world plus two timestamps
            nc, nd = struct.unpack("<2Q", f.read(16))
            if i in choose:
                rgb = np.asarray(
                    Image.open(io.BytesIO(f.read(nc)))
                    .convert("RGB")
                    .resize((W, H), Image.Resampling.BILINEAR)
                )
                dep = np.frombuffer(zlib.decompress(f.read(nd)), dtype="<u2").reshape(
                    hd, wd
                )
                ret[i] = (rgb, dep)
            else:
                f.seek(nc + nd, 1)
    return ret


def choose_pairs(ids, frames, pose):
    # Retrieval features do not use RGB-D benchmark poses. Rank candidate
    # revisits with coarse appearance and native estimated frustum similarity.
    feats = {
        f: np.asarray(
            Image.fromarray(frames[f][0]).convert("L").resize((24, 18)), float
        ).ravel()
        for f in ids
    }
    feat = np.array([feats[f] for f in ids])
    feat -= feat.mean(axis=1, keepdims=True)
    feat /= np.maximum(1, np.linalg.norm(feat, axis=1, keepdims=True))
    pairs = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            gap = j - i
            if gap < max(160, len(pose) * 0.1):
                continue
            angle = Rotation.from_matrix(
                np.linalg.inv(pose[i, :3, :3]) @ pose[j, :3, :3]
            ).magnitude()
            dist = np.linalg.norm(pose[i, :3, 3] - pose[j, :3, 3])
            if angle > np.deg2rad(80) or dist > 3:
                continue
            score = (
                feat[a] @ feat[b]
                - 0.1 * dist
                - 0.08 * angle
                + 0.03 * np.log1p(gap / 100.0)
            )
            pairs.append((score, i, j))
    chosen = []
    for score, i, j in sorted(pairs, reverse=True):
        if any(abs(i - k) < 96 and abs(j - l) < 96 for k, l in chosen):
            continue
        chosen.append((i, j))
        if len(chosen) == 8:
            break
    return chosen


def image_view(rgb, frame):
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
    return dict(img=x.unsqueeze(0), true_shape=torch.tensor([[H, W]]), instance=[frame])


def pixels(desc, step=4):
    y, x = np.mgrid[step // 2 : H : step, step // 2 : W : step]
    xy = np.column_stack((x.ravel(), y.ravel()))
    return desc[xy[:, 1], xy[:, 0]].contiguous(), xy


def depth_xyz(xy, dep, K):
    h, w = dep.shape
    u = np.rint((xy[:, 0] + 0.5) * w / W - 0.5).astype(int)
    v = np.rint((xy[:, 1] + 0.5) * h / H - 0.5).astype(int)
    z = dep[v, u].astype(float) * 0.001
    fx, fy, cx, cy = K
    return np.column_stack(((u - cx) / fx * z, (v - cy) / fy * z, z))


def fit(a, b):
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    r = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = cb - r @ ca
    return t


def distances(x, y, t):
    return np.linalg.norm(x @ t[:3, :3].T + t[:3, 3] - y, axis=1)


def process(model, view_a, view_b, K1, K2, prior, seed):
    t = time.monotonic()
    with torch.inference_mode():
        p1, p2 = model(image_view(view_a[0], "source"), image_view(view_b[0], "target"))
    d1, xy1 = pixels(p1["desc"][0])
    d2, xy2 = pixels(p2["desc"][0])
    sim = d1 @ d2.T
    b = sim.argmax(1)
    a = sim.argmax(0)
    idx = torch.arange(len(b))
    good = (a[b] == idx) & (sim[idx, b] > 0.75)
    u = xy1[idx[good].numpy()]
    v = xy2[b[good].numpy()]
    xyz1 = depth_xyz(u, view_a[1], K1)
    xyz2 = depth_xyz(v, view_b[1], K2)
    ok = (xyz1[:, 2] > 0.3) & (xyz2[:, 2] > 0.3) & (xyz1[:, 2] < 4) & (xyz2[:, 2] < 4)
    x, y = xyz1[ok], xyz2[ok]
    u, v = u[ok], v[ok]
    # Separate spatial tiles for proposal fitting and residual checks.
    dev = ((u[:, 0] // 28 + u[:, 1] // 24) % 3) != 0
    aa, bb = x[dev], y[dev]
    xx, yy = x[~dev], y[~dev]
    z = dict(
        matches=len(x),
        fitting=len(aa),
        heldout=len(xx),
        model_seconds=time.monotonic() - t,
        accepted=False,
    )
    if len(xx) < 20 or len(aa) < 30:
        return z
    rng = np.random.default_rng(seed)
    best = np.zeros(len(aa), bool)
    for _ in range(450):
        j = rng.choice(len(aa), 3, False)
        T = fit(aa[j], bb[j])
        mask = distances(aa, bb, T) < 0.07
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 20:
        return z
    T = fit(aa[best], bb[best])
    for gate in (0.08, 0.06, 0.05):
        mask = distances(aa, bb, T) < gate
        if mask.sum() >= 20:
            T = fit(aa[mask], bb[mask])
    ref = distances(xx, yy, prior)
    new = distances(xx, yy, T)
    inlier = new < 0.07
    bmask = ref < 0.07
    coverage = len(
        np.unique(
            np.column_stack((v[~dev][inlier, 0] // 40, v[~dev][inlier, 1] // 40)),
            axis=0,
        )
    )
    delta = np.linalg.inv(prior) @ T
    z.update(
        fit_inliers=int(mask.sum()),
        hold_inliers=int(inlier.sum()),
        hold_ratio=float(inlier.mean()),
        prior_hold_ratio=float(bmask.mean()),
        hold_median_cm=float(100 * np.median(new)),
        prior_hold_median_cm=float(100 * np.median(ref)),
        covered_image_bins=coverage,
        translation_diff=float(np.linalg.norm(delta[:3, 3])),
        angle_diff_deg=float(
            np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude())
        ),
        T_target_source=T.tolist(),
        accepted=bool(
            inlier.sum() >= 20
            and inlier.mean() > 0.55
            and coverage >= 5
            and np.median(new) < 0.05
        ),
    )
    return z


def measure(sensor, trajectory, intrinsics, checkpoint):
    from mast3r.model import AsymmetricMASt3R

    torch.set_num_threads(2)
    model = AsymmetricMASt3R.from_pretrained(str(checkpoint)).to("cpu").eval()
    # Sampling and normalization match the official-split paired validation.
    pose = {
        p["frame_id"]: np.asarray(p["T_world_camera_m"]).reshape(4, 4)
        for p in trajectory["poses"]
        if p.get("valid", True)
    }
    camera = np.tile(
        np.eye(4), (max(p["frame_id"] for p in trajectory["poses"]) + 1, 1, 1)
    )
    for frame, matrix in pose.items():
        camera[frame] = matrix
    ids = sorted(f for f in pose if f % 40 == 0)
    if len(ids) < 2:
        return []
    frames = read_rgbd(sensor, set(ids))
    if set(frames) != set(ids):
        raise ValueError("An estimated keyframe falls outside the RGB-D stream")
    edges = []
    for i, j in choose_pairs(ids, frames, camera):
        record = process(
            model,
            frames[i],
            frames[j],
            intrinsics,
            intrinsics,
            np.linalg.inv(pose[j]) @ pose[i],
            i + j,
        )
        record.update(frame_i=i, frame_j=j)
        edges.append(record)
    return edges
