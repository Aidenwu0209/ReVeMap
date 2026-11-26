"""Complete measured surfaces using multi-view votes and existing instance IDs."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from .common import read, sha, write


def diverse(group, frames, pose_features, size):
    # Each span is a lower bound on actual camera translation or forward-axis change.
    passed = np.zeros(size, bool)
    for axis, threshold in enumerate((0.10, 0.10, 0.10, 0.16, 0.16, 0.16)):
        values = pose_features[frames, axis]
        lo = np.full(size, np.inf, np.float32)
        hi = np.full(size, -np.inf, np.float32)
        np.minimum.at(lo, group, values)
        np.maximum.at(hi, group, values)
        passed |= (hi - lo) >= threshold
    return passed


def votes(cells, categories, n, base, frames=None, poses=None):
    selected = np.zeros(n, np.int32)
    count = np.zeros(n, np.int32)
    total = np.bincount(cells, minlength=n)
    pairs, inv, counts = np.unique(
        cells.astype(np.int64) * base + categories,
        return_inverse=True,
        return_counts=True,
    )
    valid = counts >= 2
    if frames is not None:
        valid &= diverse(inv, frames, poses, len(pairs))
    cells_ = pairs[valid] // base
    values = pairs[valid] % base
    counts = counts[valid]
    order = np.lexsort((values, -counts, cells_))
    where = np.unique(cells_[order], return_index=True)[1]
    take = order[where]
    selected[cells_[take]] = values[take]
    count[cells_[take]] = counts[take]
    return selected, count, total


def observations(front, original, frames, trajectory, checks, depth_scale):
    by_id = {x["frame_id"]: x for x in frames}
    poses = {
        x["frame_id"]: np.asarray(x["T_world_camera_m"]).reshape(4, 4)
        for x in trajectory["poses"]
        if x["valid"]
    }
    semantic_ids = {int(k) for k in checks}
    used = sorted(
        (semantic_ids | {i + 2 for i in semantic_ids}) & poses.keys() & by_id.keys()
    )
    features = np.array(
        [np.r_[poses[t][:3, 3], poses[t][:3, 2]] for t in used], dtype=np.float32
    )
    chunks = []
    node_base = 1
    meta = [None]
    witness = []
    for j, fid in enumerate(used):
        p = poses[fid]
        frame = by_id[fid]
        depth_path = Path(frame["depth_path"])
        semantic_frame = fid in semantic_ids
        if semantic_frame and sha(depth_path) != checks[str(fid)]["depth_sha256"]:
            raise ValueError("Depth hash mismatch")
        with Image.open(depth_path) as image:
            dep = np.asarray(image, dtype=np.float32) * np.float32(1 / depth_scale)
        fx, fy, cx, cy = frame["intrinsics"]
        if frame.get("rotate_ccw", False):
            width = dep.shape[1]
            dep = np.rot90(dep)
            fx, fy, cx, cy = fy, fx, cy, width - 1 - cx
        height, width = dep.shape
        rr, cc = np.mgrid[2 + j % 5 : height - 2 : 5, 2 + (j // 5) % 5 : width - 2 : 5]
        z = dep[rr, cc]
        ok = (z >= 0.3) & (z <= 5.0)
        for y, x in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            ok &= np.abs(dep[rr + y, cc + x] - z) <= np.maximum(0.08, z * 0.03)
        y, x, z = rr[ok], cc[ok], z[ok]
        cam = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
        world = (cam @ p[:3, :3].T + p[:3, 3]).astype(np.float32)
        grid = np.floor(world / 0.03).astype(np.int64) + 65536
        if grid.size and (grid.min() < 0 or grid.max() >= 131072):
            raise ValueError("Voxel range")
        keys = (grid[:, 0] << 34) | (grid[:, 1] << 17) | grid[:, 2]
        keys, idx = np.unique(keys, return_index=True)
        s = np.zeros(len(keys), np.int16)
        nodes = np.zeros(len(keys), np.int32)
        if semantic_frame:
            zp = front / "semantic/frames" / f"{fid:06}.npz"
            if sha(zp) != checks[str(fid)]["mask_sha256"]:
                raise ValueError(f"SAM3 mask changed for frame {fid}")
            with np.load(zp) as f:
                sem = f["semantic"]
                local = f["local_instance"]
                if sem.shape != dep.shape or local.shape != dep.shape:
                    raise ValueError(f"Unregistered semantic masks for frame {fid}")
                s = sem[y[idx], x[idx]].astype(np.int16)
                mid = local[y[idx], x[idx]]
            for m in np.unique(mid[mid > 0]):
                mask = mid == m
                labels, counts = np.unique(s[mask & (s > 0)], return_counts=True)
                if not len(labels):
                    continue
                cl = int(labels[np.argmax(counts)])
                if cl in (10, 19):
                    continue
                pick = mask & (s == cl)
                if pick.sum() < 12:
                    continue
                nodes[pick] = node_base
                meta.append((j, cl))
                node_base += 1
            if len(witness) < 3 and (
                j == 0 or j in (10, 20) or (j > 20 and len(witness) < 3)
            ):
                proj = front / "fused" / f"projection_{fid:06}.npz"
                with np.load(proj) as pr:
                    v, u, ids = pr["row"], pr["col"], pr["point_ids"]
                zz = dep[v, u]
                c = np.column_stack(((u - cx) * zz / fx, (v - cy) * zz / fy, zz))
                err = np.linalg.norm(c @ p[:3, :3].T + p[:3, 3] - original[ids], axis=1)
                if len(err):
                    median = float(np.median(err))
                    witness.append([fid, median])
                    if median > 0.05:
                        raise ValueError(f"Projection witness {fid}: {median}")
        chunks.append((keys, world[idx], s, nodes, np.full(len(keys), j, np.int32)))

    return [np.concatenate(a) for a in zip(*chunks)], features, meta, witness


def tracks(n, vid, node, meta, xyz, tree, base_sem, base_ins, features):
    """Join overlapping masks across distinct views, respecting conflicting anchors."""
    positive = node > 0
    ix = np.argsort(node[positive], kind="stable")
    nid = node[positive][ix]
    cells = vid[positive][ix]
    ids, starts = np.unique(nid, return_index=True)
    groups = (
        {int(k): np.sort(a) for k, a in zip(ids, np.split(cells, starts[1:]))}
        if len(ids)
        else {}
    )
    dd, nn = tree.query(xyz, workers=1)
    anchor = np.where(dd <= 0.05, base_ins[nn], 0)
    last = {}
    parent = list(range(len(meta)))
    known = np.zeros(len(meta), np.int32)
    seen = [set() for _ in meta]
    edges = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k, v in groups.items():
        frame, cl = meta[k]
        seen[k].add(frame)
        pick = anchor[v][base_sem[nn[v]] == cl]
        labels, counts = np.unique(pick[pick > 0], return_counts=True)
        if len(labels):
            q = counts.argmax()
            if counts[q] >= max(15, 0.6 * counts.sum(), 0.15 * len(v)):
                known[k] = labels[q]
        if cl not in last:
            last[cl] = np.zeros(n, np.int32)
        older, hits = np.unique(last[cl][v], return_counts=True)
        for b, h in zip(older, hits):
            if b and h >= 12 and meta[b][0] != frame:
                ratio = h / min(len(v), len(groups[b]))
                if ratio >= 0.3:
                    edges.append((float(ratio), int(h), k, int(b)))
        last[cl][v] = k
    for _, _, a, b in sorted(edges, reverse=True):
        a, b = find(a), find(b)
        if (
            a == b
            or (known[a] and known[b] and known[a] != known[b])
            or not seen[a].isdisjoint(seen[b])
        ):
            continue
        if len(seen[a]) < len(seen[b]):
            a, b = b, a
        parent[b] = a
        seen[a].update(seen[b])
        known[a] = max(known[a], known[b])
    components = defaultdict(list)
    for k in groups:
        components[find(k)].append(k)
    lookup = np.zeros(len(meta), np.int32)
    new = int(base_ins.max())
    for root, members in sorted(components.items()):
        owner = known[root]
        if not owner:
            f = features[sorted(seen[root])]
            if len(f) < 3 or not np.any(
                np.ptp(f, axis=0) >= np.array([0.1, 0.1, 0.1, 0.16, 0.16, 0.16])
            ):
                continue
            if len(np.unique(np.concatenate([groups[k] for k in members]))) < 80:
                continue
            new += 1
            owner = new
        lookup[members] = owner
    return lookup


def complete_labels(original_xyz, base_semantic, base_instance, obs, features, meta):
    """Return additional XYZ, nearest source indices and final point labels."""
    if not obs or not len(original_xyz) or not len(obs[0]):
        return (
            np.empty((0, 3)),
            np.empty(0, dtype=int),
            base_semantic.copy(),
            base_instance.copy(),
        )
    keys, world, observed_classes, nodes, view_ids = obs
    _, ix, counts = np.unique(keys, return_inverse=True, return_counts=True)
    keep = (counts >= 2) & diverse(ix, view_ids, features, len(counts))
    xyz = np.stack(
        [np.bincount(ix, weights=world[:, j]) / counts for j in range(3)], axis=1
    )[keep]
    if not np.any(keep):
        return (
            np.empty((0, 3)),
            np.empty(0, dtype=int),
            base_semantic.copy(),
            base_instance.copy(),
        )
    m = keep[ix]
    voxel_ids = np.cumsum(keep)[ix[m]] - 1
    observed_classes, nodes, view_ids = observed_classes[m], nodes[m], view_ids[m]
    del obs, keys, world, ix, m, counts, keep
    n = len(xyz)
    tree = cKDTree(original_xyz)
    dist, near = tree.query(xyz, workers=1)
    append = dist > 0.022
    add = xyz[append]
    geometry_semantic = np.where(dist[append] <= 0.08, base_semantic[near[append]], 0)
    geometry_instance = np.where(dist[append] <= 0.08, base_instance[near[append]], 0)
    valid = observed_classes > 0
    class_votes, support, total_votes = votes(
        voxel_ids[valid], observed_classes[valid], n, 256, view_ids[valid], features
    )
    class_votes[support < total_votes * 0.60] = 0
    track_owner = tracks(
        n, voxel_ids, nodes, meta, xyz, tree, base_semantic, base_instance, features
    )
    owner = track_owner[nodes]
    ok = (
        (observed_classes == class_votes[voxel_ids])
        & (observed_classes > 0)
        & (owner > 0)
    )
    instance_votes, support, total_votes = votes(
        voxel_ids[ok],
        owner[ok],
        n,
        int(max(owner.max(), 1)) + 1,
        view_ids[ok],
        features,
    )
    instance_votes[support <= 0.5 * total_votes] = 0
    # Require at least 80 actually assigned cells for every newly created ID.
    ids, ct = np.unique(
        instance_votes[instance_votes > base_instance.max()], return_counts=True
    )
    if len(ids):
        instance_votes[np.isin(instance_votes, ids[ct < 80])] = 0
    d, idx = cKDTree(xyz).query(original_xyz, workers=1)
    original_semantic = base_semantic.copy()
    original_instance = base_instance.copy()
    replace = (d <= 0.04) & (original_semantic == 0) & (class_votes[idx] > 0)
    original_semantic[replace] = class_votes[idx[replace]]
    replace = (
        (d <= 0.04)
        & (original_instance == 0)
        & (original_semantic == class_votes[idx])
        & (instance_votes[idx] > 0)
    )
    original_instance[replace] = instance_votes[idx[replace]]
    added_class_votes = class_votes[append]
    added_owner_votes = instance_votes[append]
    added_semantic = np.where(
        added_class_votes > 0, added_class_votes, geometry_semantic
    )
    added_instance = np.where(
        (added_owner_votes > 0) & (added_class_votes > 0),
        added_owner_votes,
        np.where(added_semantic == geometry_semantic, geometry_instance, 0),
    )
    semantic = np.r_[original_semantic, added_semantic]
    instance = np.r_[original_instance, added_instance]
    # Suppress provisional births only after voting. Removing them earlier
    # would alter the majority among competing existing tracks.
    unanchored = instance > base_instance.max()
    geometric_semantic = np.r_[base_semantic, geometry_semantic]
    geometric_instance = np.r_[base_instance, geometry_instance]
    instance[unanchored] = np.where(
        semantic[unanchored] == geometric_semantic[unanchored],
        geometric_instance[unanchored],
        0,
    )
    return add, near[append], semantic, instance


def publish(output, manifest, base_map, destination):
    """Publish a separate completed PLY, keeping the fusion/refinement sources intact."""
    output, base_map = Path(output), Path(base_map)
    config = read(manifest)
    trajectory = Path(read(output / "mapping/mapping_result.json")["trajectory"])
    frames = read(output / "semantic/FRAMES.json")
    checks = {str(row["frame_id"]): row for row in frames}
    source = PlyData.read(base_map)
    vertices = source["vertex"].data
    original = np.column_stack([vertices[key] for key in "xyz"])
    obs, features, meta, witness = observations(
        output,
        original,
        config["frames"],
        read(trajectory),
        checks,
        config["depth_scale"],
    )
    added, nearest, semantic, instance = complete_labels(
        original, vertices["semantic_id"], vertices["instance_id"], obs, features, meta
    )
    del obs
    size = len(vertices)
    result = np.zeros(size + len(added), dtype=vertices.dtype)
    result[:size] = vertices
    for axis, key in enumerate("xyz"):
        result[key][size:] = added[:, axis]
    # Refined PLYs may be semantic-colored. The viewer expects RGB aligned
    # with the completed geometry, including the unchanged prefix.
    rgb_source = PlyData.read(output / "fused/export/map_labeled.ply")["vertex"].data
    if len(rgb_source) != size or any(
        not np.array_equal(rgb_source[key], vertices[key]) for key in "xyz"
    ):
        raise ValueError("Completion source and fused RGB differ in point order")
    for key in ("red", "green", "blue"):
        result[key][:size] = rgb_source[key]
        result[key][size:] = rgb_source[key][nearest]
    for key, values in (("semantic_id", semantic), ("instance_id", instance)):
        result[key] = values
    if not np.isin(instance[instance > 0], vertices["instance_id"]).all():
        raise ValueError("Completion produced an unanchored ID")
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    path = destination / "map_labeled.ply"
    elements = [
        PlyElement.describe(result, "vertex") if elem.name == "vertex" else elem
        for elem in source.elements
    ]
    PlyData(elements, text=False).write(path)
    receipt = destination / "result.json"
    write(
        receipt,
        dict(
            status="completed",
            mode="anchored",
            GT_used=False,
            base_map_sha256=sha(base_map),
            original_points=size,
            added_points=len(added),
            semantic_views=len(checks),
            depth_views=len(features),
            projection_witness=witness,
            instance_births=0,
            added_rgb="nearest original vertex",
        ),
    )
    return path, receipt
