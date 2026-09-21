"""Read-only point-loss attribution of the unmodified multiview backend.

The profiler captures the real function's return locals, avoiding an almost
identical diagnostic implementation drifting from the algorithm being audited.
Reasons are mutually exclusive and name the deepest stage reached by any
observation of a point. They are not counterfactual causal-effect estimates.
"""
from __future__ import annotations
import sys
import numpy as np
from .sam3_multiview import fuse_instances

REASONS = (
    'no_depth_consistent_projection', 'no_projected_mask',
    'projected_semantic_abstention', 'point_confidence_below_threshold',
    'outside_semantic_interior', 'mask_below_min_points',
    'all_mask_nodes_filtered', 'all_groups_insufficient_frames',
    'frame_eligible_groups_low_score', 'point_insufficient_views',
    'ambiguous_ownership', 'owned_point_semantic_unknown',
    'owned_semantic_part_below_output_size', 'retained_instance',
)


def audit_losses(n_points, frames, semantic, old_instances, config=None,
                 expected_instances=None):
    """Return actual output, uint8 reason per point, and exact loss accounting."""
    captured = {}
    previous = sys.getprofile()
    if previous is not None:
        raise RuntimeError('diagnostic must not replace an existing profiler')
    def capture(frame, event, argument):
        if event == 'return' and frame.f_code is fuse_instances.__code__:
            captured.update(frame.f_locals)
    sys.setprofile(capture)
    try:
        output, original_audit = fuse_instances(n_points, frames, semantic, config)
    finally:
        sys.setprofile(previous)
    old = np.asarray(old_instances)
    if old.shape != (n_points,) or not np.issubdtype(old.dtype, np.integer) or np.any(old < 0):
        raise ValueError('invalid old instance ownership')
    if expected_instances is not None and not np.array_equal(output, expected_instances):
        raise AssertionError('audited algorithm does not exactly reproduce the sealed output')
    reason = np.zeros(n_points, np.uint8)
    def mark(ids, stage):
        reason[ids] = np.maximum(reason[ids], stage)
    for frame in frames:
        ids = np.asarray(frame['point_ids'])
        mask = np.asarray(frame['mask_ids']) > 0
        known = mask & (np.asarray(frame['semantic']) > 0)
        confident = known & (np.asarray(frame['confidence']) >= .5)
        safe = confident & np.asarray(frame['interior']).astype(bool)
        for selected, stage in ((np.ones(len(ids), bool), 1), (mask, 2),
                                (known, 3), (confident, 4), (safe, 5)):
            mark(ids[selected], stage)
    membership = captured['membership']
    cfg = captured['cfg']
    if membership.shape[0]:
        mark(np.unique(membership.indices), 6)
        filtered = captured['filtered']
        mark(np.unique(membership[~filtered].indices), 7)
        group_rows = []
        for group in captured['groups']:
            ids = np.unique(membership[group].indices)
            frame_count = len(np.unique(captured['origins'][group]))
            if frame_count < cfg.min_group_frames:
                state, score = 'insufficient_frames', None
            else:
                mark(ids, 8)
                weights = captured['weights']
                score = (float(weights[group].max(axis=0).tocsr().data.mean())
                         if cfg.object_score_mode == 'max_point'
                         else float(weights[group].sum() / membership[group].sum()))
                state = 'low_score' if score < cfg.min_object_score else 'eligible'
            group_rows.append({'nodes': [int(i) for i in group], 'distinct_frames': frame_count,
                               'score': score, 'state': state, 'point_count': len(ids),
                               'old_owned_points': int(np.sum(old[ids] > 0)),
                               'predicted_class_counts': {str(int(c)): int(k) for c, k in
                                   zip(*np.unique(np.asarray(semantic)[ids], return_counts=True))}})
        if captured['usable_groups']:
            counts, scores = captured['counts'], captured['scores']
            observed = np.flatnonzero(np.diff(scores.indptr))
            mark(observed, 9)
            for point in observed:
                lo, hi = scores.indptr[point:point + 2]
                best = int(scores.data[lo:hi].argmax())
                if counts.data[lo + best] >= cfg.min_point_views:
                    mark(np.array([point]), 10)
            owner = captured['owner']
            mark(np.flatnonzero(owner >= 0), 11)
            mark(np.flatnonzero((owner >= 0) & (np.asarray(semantic) > 0)), 12)
    else:
        group_rows = []
    mark(np.flatnonzero(output > 0), 13)
    lost = (old > 0) & (output == 0)
    def tally(mask):
        return {name: int(np.sum(mask & (reason == i))) for i, name in enumerate(REASONS)}
    loss_reasons = tally(lost)
    assert sum(loss_reasons.values()) == int(lost.sum())
    classes = []
    for category in np.unique(semantic):
        selected = np.asarray(semantic) == category
        classes.append({'semantic_id': int(category), 'semantic_points': int(selected.sum()),
                        'old_instance_points': int(np.sum(selected & (old > 0))),
                        'new_instance_points': int(np.sum(selected & (output > 0))),
                        'lost_instance_points': int(np.sum(selected & lost)),
                        'newly_owned_points': int(np.sum(selected & (old == 0) & (output > 0))),
                        'loss_reasons': tally(selected & lost)})
    audit = {'reason_policy': 'deepest stage reached by any observation; ordered exclusive accounting',
             'counterfactual_causation_measured': False, 'baseline_output_equal': expected_instances is not None,
             'lost_points': int(lost.sum()), 'newly_owned_points': int(np.sum((old == 0) & (output > 0))),
             'net_lost_points': int(np.count_nonzero(old) - np.count_nonzero(output)),
             'loss_reasons': loss_reasons, 'all_point_reasons': tally(np.ones(n_points, bool)),
             'predicted_classes': classes, 'group_membership_details': group_rows,
             'original_fusion_audit': original_audit}
    return output, reason, audit
