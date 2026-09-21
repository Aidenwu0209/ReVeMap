"""Offline instance fusion from cached, depth-checked SAM3 mask observations.

Inspired by MaskClustering's view consensus, adapted to sparse prompt masks.
Missing labels abstain; no RGB model, GT, XYZ modification, or label inference
is performed. Existing semantic labels remain an external, immutable input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ConsensusConfig:
    min_mask_points: int = 50
    min_output_points: int = 50
    min_group_frames: int = 2
    visibility_fraction: float = .3
    containment_fraction: float = .8
    split_piece_fraction: float = .2
    split_frame_fraction: float = .2
    min_split_frames: int = 2
    min_support_frames: int = 3
    consensus_fraction: float = .9
    cannot_link_frames: int = 2
    min_point_views: int = 2
    ownership_share: float = .65
    ownership_margin: float = .15
    min_object_score: float = .8
    object_score_mode: str = 'mean_observations'


def _config(config):
    cfg = ConsensusConfig(**config) if isinstance(config, dict) else (config or ConsensusConfig())
    integer_keys = ('min_mask_points', 'min_output_points', 'min_group_frames',
                    'min_split_frames', 'min_support_frames', 'cannot_link_frames', 'min_point_views')
    for key in integer_keys:
        if not isinstance(getattr(cfg, key), int) or isinstance(getattr(cfg, key), bool) or getattr(cfg, key) < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key, value in asdict(cfg).items():
        if key not in (*integer_keys, 'object_score_mode') and not 0 < value <= 1:
            raise ValueError(f'{key} must be in (0,1]')
    if cfg.object_score_mode not in ('mean_observations', 'max_point'):
        raise ValueError('unsupported object_score_mode')
    if cfg.containment_fraction <= .5:
        raise ValueError('containment must identify at most one mask per frame')
    if cfg.min_support_frames < 3:
        raise ValueError('at least three support frames required for an independent observer')
    return cfg


def _observations(n, frames, cfg):
    points, scores, origins, local_ids, visible = [], [], [], [], []
    seen = set()
    if any(not isinstance(f['frame_id'], (int, np.integer)) or isinstance(f['frame_id'], (bool, np.bool_)) for f in frames):
        raise ValueError('frame_id must be an integer')
    for ordinal, frame in enumerate(sorted(frames, key=lambda f: int(f['frame_id']))):
        fid = int(frame['frame_id'])
        if fid in seen:
            raise ValueError('duplicate frame would double count evidence')
        seen.add(fid)
        ids = np.asarray(frame['point_ids'])
        masks = np.asarray(frame['mask_ids'])
        sem = np.asarray(frame['semantic'])
        confidence = np.asarray(frame['confidence'])
        interior = np.asarray(frame['interior'])
        if (ids.ndim != 1 or not all(a.shape == ids.shape for a in (masks, sem, confidence, interior))
                or not np.issubdtype(ids.dtype, np.integer)
                or not np.issubdtype(masks.dtype, np.integer)
                or not np.issubdtype(sem.dtype, np.integer)
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= n)
                or np.any(masks < 0) or np.any(sem < 0)
                or not np.isfinite(confidence).all() or np.any(confidence < 0)
                or np.any(confidence > 1) or not np.isfinite(interior).all()
                or not np.isin(interior, (0, 1)).all()):
            raise ValueError('invalid projected frame; require unique in-range point IDs')
        visible.append(ids.astype(np.int32))
        safe = interior.astype(bool) & (sem > 0) & (confidence >= .5)
        for mask_id in np.unique(masks[safe]):
            if mask_id <= 0:
                continue
            selected = safe & (masks == mask_id)
            if selected.sum() < cfg.min_mask_points:
                continue
            points.append(ids[selected].astype(np.int32))
            scores.append(confidence[selected].astype(np.float32))
            origins.append(ordinal)
            local_ids.append((fid, int(mask_id)))
    def stack(rows, dtype):
        indptr = np.r_[0, np.cumsum([len(r) for r in rows])]
        indices = np.concatenate(rows) if rows else np.empty(0, np.int32)
        return sparse.csr_matrix((np.ones(len(indices), dtype=dtype), indices, indptr),
                                 shape=(len(rows), n))
    membership = stack(points, np.int32)
    visibility = stack(visible, np.int32)
    weights = membership.astype(np.float32)
    if scores:
        weights.data = np.concatenate(scores)
    return membership, visibility, weights, np.asarray(origins, np.int32), local_ids


def _view_statistics(membership, visibility, origins, cfg):
    """Count original map identities, never nearest neighbours or mask pixels."""
    overlaps = (membership @ membership.T).tocsr()
    visible = (membership @ visibility.T).toarray()
    sizes = np.asarray(membership.sum(axis=1)).ravel()
    eligible = (visible >= cfg.visibility_fraction * sizes[:, None]) & (visible > 0)
    split_count = np.zeros(len(sizes), np.int32)
    evaluable_count = np.zeros(len(sizes), np.int32)
    for i in range(len(sizes)):
        row = overlaps.getrow(i)
        for frame in np.flatnonzero(eligible[i]):
            if frame == origins[i]:
                continue
            counts = row.data[origins[row.indices] == frame]
            denom = visible[i, frame]
            # Unknown mass is kept in the denominator: a small labelled island
            # cannot turn into a confident whole-object observation.
            if counts.sum() < cfg.containment_fraction * denom:
                continue
            evaluable_count[i] += 1
            split_count[i] += int(np.sum(counts >= cfg.split_piece_fraction * denom) >= 2
                                  and counts.max(initial=0) < cfg.containment_fraction * denom)
    filtered = ((split_count >= cfg.min_split_frames)
                & (split_count > cfg.split_frame_fraction * evaluable_count))
    return overlaps, visible, eligible, filtered, split_count, evaluable_count


def _cluster(positive, negative, origins, valid_nodes, cfg):
    """Globally rank edges and veto contradictions across whole components.

    Unlike connected components, a positive A-B-C path cannot override measured
    A-C separation. This is constrained graph agglomeration, not a reproduction
    of MaskClustering's iterative re-projection of cluster unions.
    """
    n = len(origins)
    parent = np.arange(n)
    members = {int(i): {int(i)} for i in np.flatnonzero(valid_nodes)}
    cannot = [set(np.flatnonzero(negative[i] >= cfg.cannot_link_frames)) for i in range(n)]
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    edges = []
    for a, b in zip(*np.nonzero(np.triu(positive >= cfg.min_support_frames, 1))):
        if not (valid_nodes[a] and valid_nodes[b]):
            continue
        p, neg = int(positive[a, b]), int(negative[a, b])
        if neg >= cfg.cannot_link_frames or p / (p + neg) < cfg.consensus_fraction:
            continue
        # At least three distinct supporters imply one outside both source
        # frames. The minimum cannot be lowered below 3 in production config.
        edges.append((-p, neg, int(a), int(b)))
    merged, vetoed = 0, 0
    for _, _, a, b in sorted(edges):
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        if any(cannot[x] & members[rb] for x in members[ra]):
            vetoed += 1
            continue
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra].update(members.pop(rb))
        merged += 1
    groups = sorted((sorted(g) for g in members.values()), key=lambda g: g[0])
    return groups, {'candidate_edges': len(edges), 'accepted_merges': merged,
                    'component_contradiction_vetoes': vetoed}


def fuse_instances(n_points, frames, baseline_semantic, config=None):
    """Return unique point ownership and an audit; never mutate caller inputs."""
    cfg = _config(config)
    semantic = np.asarray(baseline_semantic)
    if (not isinstance(n_points, (int, np.integer)) or isinstance(n_points, (bool, np.bool_))
            or n_points < 0 or semantic.shape != (n_points,)
            or not np.issubdtype(semantic.dtype, np.integer) or np.any(semantic < 0)):
        raise ValueError('one nonnegative baseline semantic ID per map point required')
    membership, visibility, weights, origins, identifiers = _observations(n_points, frames, cfg)
    n = membership.shape[0]
    audit = {'method': 'sparse-view consensus with component cannot-link veto',
             'config': asdict(cfg), 'frame_count': len(frames), 'mask_nodes': n,
             'semantic_labels_modified': False, 'gt_consumed': False,
             'unknown_observations': 'abstain; not counted as agreement or separation',
             'point_support': 'only cached depth-consistent mask interiors; no spatial expansion'}
    if n == 0:
        return np.zeros(n_points, np.int32), {**audit, 'retained_instances': 0, 'filtered_masks': []}
    overlaps, visible, eligible, filtered, splits, evaluable = _view_statistics(membership, visibility, origins, cfg)
    # A filtered mask is neither an object node nor a supporting observer.
    dominant = np.full((n, visibility.shape[0]), -1, np.int32)
    for i in np.flatnonzero(~filtered):
        row = overlaps.getrow(i)
        for j, count in zip(row.indices, row.data):
            f = origins[j]
            if not filtered[j] and eligible[i, f] and count >= cfg.containment_fraction * visible[i, f]:
                dominant[i, f] = j
    rows, cols = np.nonzero(dominant >= 0)
    observer = sparse.csr_matrix((np.ones(len(rows), np.int32), (rows, dominant[rows, cols])), shape=(n, n))
    positive = (observer @ observer.T).toarray()
    available = (dominant >= 0).astype(np.int32)
    negative = available @ available.T - positive
    assert np.all(negative >= 0) and np.all(positive <= visibility.shape[0])
    groups, graph_audit = _cluster(positive, negative, origins, ~filtered, cfg)
    group_ids, node_ids = [], []
    usable_groups = []
    group_rejections = {'insufficient_frames': 0, 'low_score': 0}
    for group in groups:
        if len(np.unique(origins[group])) < cfg.min_group_frames:
            group_rejections['insufficient_frames'] += 1
            continue
        score = (float(weights[group].max(axis=0).tocsr().data.mean())
                 if cfg.object_score_mode == 'max_point'
                 else float(weights[group].sum() / membership[group].sum()))
        if score < cfg.min_object_score:
            group_rejections['low_score'] += 1
            continue
        gid = len(usable_groups)
        usable_groups.append(group)
        group_ids.extend([gid] * len(group))
        node_ids.extend(group)
    audit.update(graph_audit)
    audit.update(filtered_masks=[{'frame_id': identifiers[i][0], 'mask_id': identifiers[i][1],
                                 'split_frames': int(splits[i]), 'evaluable_frames': int(evaluable[i])}
                                for i in np.flatnonzero(filtered)],
                 groups_before_output=len(groups), eligible_groups=len(usable_groups),
                 group_rejections=group_rejections)
    if not usable_groups:
        return np.zeros(n_points, np.int32), {**audit, 'retained_instances': 0}
    grouping = sparse.csr_matrix((np.ones(len(node_ids), np.int32), (group_ids, node_ids)),
                                 shape=(len(usable_groups), n))
    counts = (grouping @ membership).tocsc()
    scores = (grouping @ weights).tocsc()
    scores.sort_indices(); counts.sort_indices()
    # Every map point belongs to at most one local mask per input frame, so
    # these sums count distinct frames even if a cluster has same-frame parts.
    assert np.array_equal(scores.indices, counts.indices) and np.array_equal(scores.indptr, counts.indptr)
    owner = np.full(n_points, -1, np.int32)
    ambiguous = 0
    low_support = 0
    observed = np.flatnonzero(np.diff(scores.indptr))
    for point in observed:
        lo, hi = scores.indptr[point:point+2]
        values = scores.data[lo:hi]
        k = int(values.argmax())
        total, best = float(values.sum()), float(values[k])
        second = float(np.partition(values, -2)[-2]) if len(values) > 1 else 0.
        low_support += int(counts.data[lo+k] < cfg.min_point_views)
        if (counts.data[lo+k] >= cfg.min_point_views and best >= cfg.ownership_share * total
                and best - second >= cfg.ownership_margin * total):
            owner[point] = scores.indices[lo+k]
        elif len(values) > 1:
            ambiguous += 1
    output = np.zeros(n_points, np.int32)
    inventory = []
    # Preserve the existing semantic map exactly. Different fixed semantic IDs
    # within a geometric cluster receive separate output instance IDs.
    for group_id, group in enumerate(usable_groups):
        ids = np.flatnonzero(owner == group_id)
        for category in np.unique(semantic[ids]):
            if category <= 0:
                continue
            selected = ids[semantic[ids] == category]
            if len(selected) < cfg.min_output_points:
                continue
            instance_id = len(inventory) + 1
            output[selected] = instance_id
            inventory.append({'instance_id': instance_id, 'semantic_id': int(category),
                              'point_count': len(selected), 'mask_nodes': len(group),
                              'supporting_frames': sorted({identifiers[i][0] for i in group})})
    audit.update(retained_instances=len(inventory), objects=inventory,
                 ambiguous_ownership_points=ambiguous,
                 points_in_eligible_groups=len(observed),
                 points_below_min_views=low_support,
                 owned_points_without_semantic=int(np.sum((owner >= 0) & (semantic == 0))),
                 owned_known_points_in_small_objects=int(np.sum((owner >= 0) & (semantic > 0) & (output == 0))),
                 instance_coverage=float(np.mean(output > 0)) if n_points else 0.)
    return output, audit
