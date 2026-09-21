"""Conservative measured-support recovery using tentative 3D object guides.

This is a bounded MV3DIS-inspired adaptation, not its full mask selector or
continuous depth weighting. Old objects are hypotheses, not ground truth. All
existing consensus owners are immutable; unknown/unobserved points abstain.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class GuidedConfig:
    min_mask_points: int = 30
    min_visible_fraction: float = .3
    containment_fraction: float = .8
    mask_guide_purity: float = .8
    split_piece_fraction: float = .2
    min_split_frames: int = 2
    min_support_frames: int = 2
    min_point_views: int = 2
    min_point_confidence: float = .8
    min_output_points: int = 50
    anchor_min_fraction: float = .2
    anchor_dominance: float = .8


def recover_instances(n_points, frames, semantic, guide_instances, current_instances, config=None, blocked_masks=()):
    """Return instance labels, JSON audit, and exact newly owned point support.

    Only guide-owned/current-unassigned points can gain owners. Every recovered
    point has direct interior-mask support in distinct depth-checked frames.
    No neighbourhood fill, semantic inference, existing owner merge or relabel.
    """
    cfg = GuidedConfig(**config) if isinstance(config, dict) else (config or GuidedConfig())
    if not isinstance(cfg, GuidedConfig):
        raise ValueError('invalid config')
    integer_keys = ('min_mask_points', 'min_split_frames', 'min_support_frames',
                    'min_point_views', 'min_output_points')
    for k, v in asdict(cfg).items():
        if k in integer_keys:
            if type(v) is not int or v < 1:
                raise ValueError(k + ' must be a positive integer')
        elif not np.isfinite(v) or not 0 < v <= 1:
            raise ValueError(k + ' must be in (0, 1]')
    if min(cfg.containment_fraction, cfg.mask_guide_purity, cfg.anchor_dominance) <= .5:
        raise ValueError('dominance must exceed one half')
    if not isinstance(n_points, (int, np.integer)) or isinstance(n_points, (bool, np.bool_)) or n_points < 0:
        raise ValueError('invalid point count')
    semantic, guides, current = map(np.asarray, (semantic, guide_instances, current_instances))
    for a in (semantic, guides, current):
        if a.shape != (n_points,) or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
            raise ValueError('invalid point label array')
    if np.any((current > 0) & (semantic == 0)):
        raise ValueError('current owner with unknown semantic is outside this arm contract')
    blocked = set()
    for pair in blocked_masks:
        if (len(pair) != 2 or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in pair)):
            raise ValueError('blocked masks require frame-ID/mask-ID integer pairs')
        blocked.add(tuple(map(int, pair)))
    output = current.astype(np.int32, copy=True)
    if current.max(initial=0) >= np.iinfo(np.int32).max:
        raise ValueError('instance ID overflow')
    guide_points = {int(g): np.flatnonzero(guides == g) for g in np.unique(guides) if g > 0}
    guide_evidence = {g: {'frames': [], 'splits': [], 'ambiguous_mask_frames': [], 'blocked_mask_frames': []} for g in guide_points}
    seen = set()
    for frame in sorted(frames, key=lambda f: int(f['frame_id'])):
        fid = frame['frame_id']
        if not isinstance(fid, (int, np.integer)) or isinstance(fid, (bool, np.bool_)) or fid in seen:
            raise ValueError('frame IDs must be unique integers')
        seen.add(int(fid))
        ids, masks, sem, conf, interior = (np.asarray(frame[k]) for k in
            ('point_ids', 'mask_ids', 'semantic', 'confidence', 'interior'))
        if (ids.ndim != 1 or any(a.shape != ids.shape for a in (masks, sem, conf, interior))
                or any(not np.issubdtype(a.dtype, np.integer) for a in (ids, masks, sem))
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= n_points)
                or np.any(masks < 0) or np.any(sem < 0) or not np.isfinite(conf).all()
                or np.any((conf < 0) | (conf > 1)) or not np.isin(interior, (0, 1)).all()):
            raise ValueError('invalid projected frame')
        safe = (interior.astype(bool) & (sem > 0) & (conf >= .5) & (masks > 0))
        visible_guides = guides[ids]
        # Mask uniqueness is measured among existing nonzero guide hypotheses.
        # Unknown outside-guide mass never receives any ownership from this arm.
        purity = {}
        for m in np.unique(masks[safe]):
            mg = visible_guides[safe & (masks == m)]
            mg = mg[mg > 0]
            if not len(mg):
                continue
            gs, counts = np.unique(mg, return_counts=True)
            best = int(counts.argmax())
            purity[int(m)] = (int(gs[best]), float(counts[best] / counts.sum()))
        for g in np.unique(visible_guides):
            if g <= 0:
                continue
            g = int(g)
            visible = visible_guides == g
            nv = int(visible.sum())
            if nv < cfg.min_visible_fraction * len(guide_points[g]) or nv < cfg.min_mask_points:
                continue
            chosen = visible & safe
            ms, counts = np.unique(masks[chosen], return_counts=True)
            if not len(ms):
                continue
            best = int(counts.argmax())
            if (counts.sum() >= cfg.containment_fraction * nv
                    and np.sum(counts >= cfg.split_piece_fraction * nv) >= 2
                    and counts[best] < cfg.containment_fraction * nv):
                guide_evidence[g]['splits'].append(int(fid))
            if counts[best] < max(cfg.min_mask_points, cfg.containment_fraction * nv):
                continue
            mask = int(ms[best])
            # Previously measured undersegmentation cannot become positive
            # support merely because the coarse guide hides its other pieces.
            if (int(fid), mask) in blocked:
                guide_evidence[g]['blocked_mask_frames'].append(int(fid))
                continue
            if purity.get(mask, (0, 0))[0] != g or purity[mask][1] < cfg.mask_guide_purity:
                guide_evidence[g]['ambiguous_mask_frames'].append(int(fid))
                continue
            observed = chosen & (masks == mask)
            guide_evidence[g]['frames'].append((int(fid), mask, ids[observed].copy(), conf[observed].copy()))
    rows, point_support = [], []
    next_id = int(output.max(initial=0)) + 1
    rejections = {}
    for g, ids in guide_points.items():
        evidence = guide_evidence[g]
        row = {'guide_instance_id': g, 'guide_points': len(ids),
               'supporting_frames': [v[0] for v in evidence['frames']],
               'split_frames': evidence['splits'],
               'ambiguous_mask_frames': evidence['ambiguous_mask_frames'],
               'blocked_mask_frames': evidence['blocked_mask_frames'], 'recovered_points': 0}
        def reject(reason):
            row['status'] = reason
            rejections[reason] = rejections.get(reason, 0) + 1
            rows.append(row)
        if len(evidence['splits']) >= cfg.min_split_frames:
            reject('repeated_observed_split'); continue
        classes = np.unique(semantic[ids])
        if len(classes) != 1 or classes[0] == 0:
            reject('mixed_or_unknown_guide_semantics'); continue
        if len(evidence['frames']) < cfg.min_support_frames:
            reject('insufficient_compatible_frames'); continue
        existing, counts = np.unique(current[ids][current[ids] > 0], return_counts=True)
        anchor = 0
        if len(existing):
            best = int(counts.argmax())
            significant = counts >= max(cfg.min_mask_points, cfg.split_piece_fraction * counts.sum())
            if significant.sum() > 1:
                reject('multiple_consensus_anchors'); continue
            if (counts[best] < max(cfg.min_mask_points, cfg.anchor_min_fraction * len(ids))
                    or counts[best] < cfg.anchor_dominance * counts.sum()):
                reject('weak_or_ambiguous_consensus_anchor'); continue
            anchor = int(existing[best])
            if np.any(semantic[current == anchor] != classes[0]):
                reject('anchor_semantic_conflict'); continue
        support = np.zeros(len(ids), np.int32)
        confidence = np.zeros(len(ids), np.float32)
        for fid, mid, points, scores in evidence['frames']:
            local = np.searchsorted(ids, points)
            support[local] += 1
            np.maximum.at(confidence, local, scores)
        eligible = ((current[ids] == 0) & (support >= cfg.min_point_views)
                    & (confidence >= cfg.min_point_confidence))
        selected = ids[eligible]
        row['supported_lost_points'] = len(selected)
        # Minimum output size applies to newly created instances. Existing
        # anchors may receive smaller directly measured corrections.
        if not len(selected) or (not anchor and len(selected) < cfg.min_output_points):
            reject('no_sufficient_measured_recovery'); continue
        if not anchor:
            anchor = next_id
            next_id += 1
        output[selected] = anchor
        row.update(status='recovered', destination_instance_id=anchor,
                   recovered_points=len(selected), new_instance=anchor > current.max(initial=0))
        rows.append(row)
        # One row per directly supporting frame-mask-point observation.
        recovered_lookup = np.zeros(len(ids), bool)
        recovered_lookup[eligible] = True
        for fid, mid, points, scores in evidence['frames']:
            keep = recovered_lookup[np.searchsorted(ids, points)]
            if np.any(keep):
                point_support.append((points[keep], np.full(keep.sum(), fid, np.int32),
                                      np.full(keep.sum(), mid, np.int32), scores[keep]))
    recovered = np.flatnonzero((current == 0) & (output > 0)).astype(np.int32)
    def concat(index, dtype):
        return np.concatenate([r[index] for r in point_support]).astype(dtype) if point_support else np.empty(0, dtype)
    provenance = {'point_id': recovered, 'guide_instance_id': guides[recovered].astype(np.int32),
                  'destination_instance_id': output[recovered],
                  'support_point_id': concat(0, np.int32), 'support_frame_id': concat(1, np.int32),
                  'support_mask_id': concat(2, np.int32), 'support_confidence': concat(3, np.float32)}
    if len(recovered):
        _, support_counts = np.unique(provenance['support_point_id'], return_counts=True)
        provenance['distinct_support_frames'] = support_counts.astype(np.int32)
    else:
        provenance['distinct_support_frames'] = np.empty(0, np.int32)
    assert np.array_equal(output[current > 0], current[current > 0])
    assert np.all(guides[recovered] > 0) and np.all(semantic[recovered] > 0)
    audit = {'method': 'conservative original-point recovery using tentative 3D object guides',
             'paper_scope': 'MV3DIS-inspired common 3D guide only; not full reproduction or depth reweighting',
             'config': asdict(cfg), 'guides': rows, 'guide_rejections': rejections,
             'recovered_points': len(recovered), 'recovered_guides': sum(r['status'] == 'recovered' for r in rows),
             'frame_count': len(frames), 'semantic_labels_modified': False,
             'prior_filtered_mask_nodes': [list(pair) for pair in sorted(blocked)],
             'prior_filtered_mask_nodes_may_support_recovery': False,
             'current_owners_preserved': True, 'ground_truth_consumed': False,
             'old_guides_are_correctness_evidence': False,
             'scope': 'Only existing-guide lost ownership with repeated measured mask support; no spatial expansion',
             'instance_coverage': float(np.mean(output > 0)) if n_points else 0.}
    return output, audit, provenance
