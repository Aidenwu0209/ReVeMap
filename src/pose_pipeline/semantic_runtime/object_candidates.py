"""Keep unresolved geometric objects separate from confirmed map ownership."""
from pathlib import Path
from collections import Counter

import numpy as np

from .enhance import validate_labels


def build_candidates(base, geometry, *, min_points=50, min_views=2):
    n = validate_labels(base)
    groups, support = geometry['object_id'], geometry['support_views']
    for array in (groups, support):
        if array.shape != (n,) or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
            raise ValueError('invalid geometric candidate evidence')
    candidates = np.zeros(n, np.int32)
    next_id = int(base['instance'].max(initial=0)) + 1
    objects, rejected = [], []
    unknown = (base['semantic'] == 0) & (base['instance'] == 0)
    for gid in np.unique(groups[unknown & (support >= min_views)]):
        if gid == 0:
            continue
        group = groups == gid
        selected = group & unknown & (support >= min_views)
        owners = np.unique(base['instance'][group & (base['instance'] > 0)])
        categories = np.unique(base['semantic'][group & (base['semantic'] > 0)])
        reason = ('conflicting_owners_or_categories' if len(owners) > 1 or len(categories) > 1
                  else 'insufficient_points' if selected.sum() < min_points else None)
        if reason:
            rejected.append({'object_id': int(gid), 'reason': reason, 'points': int(selected.sum())})
            continue
        oid = int(owners[0]) if len(owners) else next_id
        if not len(owners):
            next_id += 1
        candidates[selected] = oid
        objects.append({'object_id': int(gid), 'candidate_instance_id': oid,
                        'unknown_points': int(selected.sum()), 'existing_owner': bool(len(owners))})
    return candidates, {'objects': objects, 'rejected': rejected,
                        'candidate_points': int(np.count_nonzero(candidates)),
                        'min_support_views': min_views, 'GT_used': False}


def extend_candidates_with_pure_masks(base, geometry, evidence, existing, classes):
    """Add unresolved candidate hints from pure masks in the original group.

    Only the verified policy calls this. No label or established owner changes;
    candidates must still pass the subsequent naming/grounding refinement.
    """
    from scipy.sparse import isspmatrix_csr
    n = validate_labels(base)
    existing = np.asarray(existing)
    arrays = [existing, np.asarray(geometry['object_id']), np.asarray(geometry['support_views'])]
    if any(a.shape != (n,) or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) for a in arrays):
        raise ValueError('invalid candidate geometry or ownership')
    groups, support = arrays[1:]
    unknown = (base['semantic'] == 0) & (base['instance'] == 0)
    if np.any((existing > 0) & ~unknown):
        raise ValueError('candidates must remain unknown and unowned')
    if classes.get('0') != 'unknown' or not {str(int(s)) for s in np.unique(base['semantic'])}.issubset(classes):
        raise ValueError('incomplete candidate class dictionary')
    membership = evidence['membership']
    if not isspmatrix_csr(membership) or membership.shape[1] != n:
        raise ValueError('candidate evidence requires a matching CSR membership')
    rows = membership.shape[0]
    origins, node_groups, eligible = [np.asarray(evidence[k]) for k in ('origins', 'group_ids', 'eligible_nodes')]
    valid = np.asarray(evidence['valid_nodes'])
    if (origins.shape != (rows,) or node_groups.shape != (rows,) or eligible.ndim != 1
            or any(not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) for a in (origins, node_groups, eligible))
            or valid.shape != (rows,) or valid.dtype != np.bool_
            or len(np.unique(eligible)) != len(eligible) or np.any(eligible >= rows)):
        raise ValueError('invalid candidate evidence dimensions or node IDs')
    if (np.any(~valid[eligible]) or np.any(node_groups[eligible] == 0)
            or not np.array_equal(np.sort(eligible), np.flatnonzero(node_groups > 0))
            or not np.isin(groups[groups > 0], node_groups[eligible]).all()):
        raise ValueError('candidate nodes must be eligible, unfiltered and grouped')
    indptr, indices, data = membership.indptr, membership.indices, membership.data
    if (indptr.shape != (rows + 1,) or not np.issubdtype(indptr.dtype, np.integer)
            or indptr[0] != 0 or indptr[-1] != len(indices)
            or len(indices) != len(data) or np.any(np.diff(indptr) < 0)
            or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= n)
            or not np.issubdtype(data.dtype, np.integer) or np.any(data != 1)
            or any(len(np.unique(indices[lo:hi])) != hi-lo for lo, hi in zip(indptr[:-1], indptr[1:]))):
        raise ValueError('candidate membership must contain unique binary point IDs')
    unresolved = unknown & (existing == 0) & (support >= 2) & (groups > 0)
    mixed = []
    for gid in np.unique(groups[unresolved]):
        categories = np.unique(base['semantic'][(groups == gid) & (base['semantic'] > 0)])
        if len(categories) > 1:
            mixed.append(gid)
    unresolved &= np.isin(groups, mixed)
    votes, owner_categories = {}, {}
    mask_reasons = Counter()
    source_masks = []
    for node in eligible:
        points = indices[indptr[node]:indptr[node + 1]]
        target = points[unresolved[points]]
        if not len(target):
            continue
        categories = np.unique(base['semantic'][points][base['semantic'][points] > 0])
        owners = np.unique(base['instance'][points][base['instance'][points] > 0])
        if len(categories) != 1 or len(owners) != 1:
            mask_reasons['mixed_or_missing_category_or_owner'] += 1
            continue
        owner, category = int(owners[0]), int(categories[0])
        if owner not in owner_categories:
            owner_categories[owner] = np.unique(base['semantic'][base['instance'] == owner])
        if not np.array_equal(owner_categories[owner], [category]):
            mask_reasons['owner_not_unique_same_class'] += 1
            continue
        known_anchors = np.sum((base['instance'][points] == owner) & (base['semantic'][points] == category))
        if known_anchors < 30:
            mask_reasons['fewer_than_30_known_owner_anchors'] += 1
            continue
        frame, group = int(origins[node]), int(node_groups[node])
        for point in target:
            votes.setdefault(int(point), set()).add((frame, owner, group))
        source_masks.append(dict(node=int(node), owner=owner, frame=frame, group_id=group,
                                 known_anchors=int(known_anchors), target_points=len(target),
                                 winning_group_target_points=int(np.sum(groups[target] == group))))
    proposed, point_reasons = {}, Counter()
    for point, witnesses in votes.items():
        owners = {owner for _, owner, _ in witnesses}
        if len(owners) != 1:
            point_reasons['contradictory_pure_owners'] += 1
            continue
        if len({frame for frame, _, group in witnesses if group == groups[point]}) < 2:
            point_reasons['fewer_than_two_winning_group_pure_owner_frames'] += 1
            continue
        owner = next(iter(owners))
        category = int(owner_categories[owner][0])
        if classes[str(category)].strip().lower() in ('floor', 'wall'):
            point_reasons['structural_surface_owner'] += 1
            continue
        proposed.setdefault(owner, []).append(point)
    output, objects = existing.copy(), []
    for owner, points in sorted(proposed.items()):
        if len(points) < 50:
            point_reasons['fewer_than_50_candidate_points'] += len(points)
            continue
        output[points] = owner
        counts = Counter(len({f for f, _, g in votes[p] if g == groups[p]}) for p in points)
        objects.append(dict(instance_id=owner, new_candidate_points=len(points),
                            semantic_id=int(owner_categories[owner][0]),
                            distinct_view_histogram={str(k): v for k, v in sorted(counts.items())}))
    return output, dict(method='pure_mask_owner_votes_v1', objects=objects,
                        minimum_known_owner_anchors=30, minimum_distinct_frames=2,
                        minimum_candidate_points_per_owner=50, label_assignment=False,
                        instance_assignment=False, GT_used=False,
                        usable_mask_count=len(source_masks), source_masks=source_masks,
                        mask_rejections=dict(mask_reasons), point_rejections=dict(point_reasons),
                        target_unknown_points=int(unresolved.sum()), points_with_pure_mask_vote=len(votes),
                        points_without_pure_mask_vote=int(unresolved.sum())-len(votes),
                        new_candidate_points=int(np.sum(output != existing)),
                        candidate_points=int(np.count_nonzero(output)))


def candidate_anchors(base, candidates=None):
    """Temporary anchors for naming/grounding; never an exported prediction."""
    n = validate_labels(base)
    anchors = {k: v.copy() for k, v in base.items()}
    if candidates is None:
        return anchors
    candidates = np.asarray(candidates)
    if (candidates.shape != (n,) or not np.issubdtype(candidates.dtype, np.integer)
            or np.any(candidates < 0)):
        raise ValueError('invalid candidate ownership')
    selected = candidates > 0
    if np.any(selected & ((base['semantic'] > 0) | (base['instance'] > 0))):
        raise ValueError('candidates may only reference unknown unowned points')
    for oid in np.unique(candidates[selected]):
        known = np.unique(base['semantic'][base['instance'] == oid])
        if len(known) > 1:
            raise ValueError('candidate owner has conflicting semantics')
    anchors['instance'][selected] = candidates[selected]
    return anchors


def load_candidates(directory):
    path = Path(directory) / 'candidates.npz'
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        return archive['candidate_instance'].copy()
