"""Explicit object naming from fixed, depth-checked multiview semantic evidence.

This inexpensive naming ablation does not implement ConceptGraphs' CLIP or
language-model stages. It never changes point labels, object ownership or map
geometry. One eligible original frame contributes exactly one object-class vote.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class NamingConfig:
    min_point_confidence: float = .8
    min_frame_points: int = 30
    min_frame_class_fraction: float = .8
    min_eligible_frames: int = 3
    min_frame_vote_fraction: float = .8


def name_objects(instances, frames, config=None):
    """Return object IDs and explicit named/unknown results with frame evidence."""
    cfg = config or NamingConfig()
    for key in ('min_frame_points', 'min_eligible_frames'):
        value = getattr(cfg, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('min_point_confidence', 'min_frame_class_fraction', 'min_frame_vote_fraction'):
        if not .5 < getattr(cfg, key) <= 1:
            raise ValueError(f'{key} must exceed 0.5 and not exceed 1')
    instance = np.asarray(instances)
    if instance.ndim != 1 or not np.issubdtype(instance.dtype, np.integer) or np.any(instance < 0):
        raise ValueError('one nonnegative integer instance ID per original map point required')
    ids, counts = np.unique(instance[instance > 0], return_counts=True)
    inventory = {int(i): {'instance_id': int(i), 'point_count': int(n), 'frame_evidence': []}
                 for i, n in zip(ids, counts)}
    seen = set()
    for frame in sorted(frames, key=lambda f: f['frame_id']):
        fid = frame['frame_id']
        if not isinstance(fid, (int, np.integer)) or isinstance(fid, (bool, np.bool_)):
            raise ValueError('frame ID must be an integer')
        fid = int(fid)
        if fid in seen:
            raise ValueError('duplicate original frame would double-count evidence')
        seen.add(fid)
        points, semantic, confidence, interior = (np.asarray(frame[k]) for k in
            ('point_ids', 'semantic', 'confidence', 'interior'))
        if (points.ndim != 1 or not np.issubdtype(points.dtype, np.integer)
            or len(np.unique(points)) != len(points) or np.any(points < 0) or np.any(points >= len(instance))
            or any(a.shape != points.shape for a in (semantic, confidence, interior))
            or not np.issubdtype(semantic.dtype, np.integer) or np.any(semantic < 0)
            or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1))
            or not np.isin(interior, (0, 1)).all()):
            raise ValueError('invalid unique depth-checked frame observations')
        object_id = instance[points]
        qualifying = (object_id > 0) & interior.astype(bool) & (semantic > 0) & (confidence >= cfg.min_point_confidence)
        for iid in np.unique(object_id[object_id > 0]):
            visible = object_id == iid
            selected = visible & qualifying
            count = int(selected.sum())
            evidence = {'frame_id': fid, 'visible_object_points': int(visible.sum()),
                'qualifying_object_points': count, 'eligible': False, 'class_id': 0,
                'weighted_class_fraction': None, 'reason': 'insufficient_qualifying_points'}
            if count >= cfg.min_frame_points:
                labels, inverse = np.unique(semantic[selected], return_inverse=True)
                weights = np.bincount(inverse, weights=confidence[selected])
                best = int(weights.argmax())
                fraction = float(weights[best] / weights.sum())
                evidence.update(weighted_class_fraction=fraction,
                    class_weights={str(int(k)): float(v) for k, v in zip(labels, weights)},
                    dominant_class_id=int(labels[best]), reason='mixed_frame_classes')
                if fraction >= cfg.min_frame_class_fraction:
                    evidence.update(eligible=True, class_id=int(labels[best]), reason='eligible')
            inventory[int(iid)]['frame_evidence'].append(evidence)
    named = []
    for iid, obj in sorted(inventory.items()):
        evidence = obj['frame_evidence']
        votes = [row['class_id'] for row in evidence if row['eligible']]
        labels, counts = np.unique(votes, return_counts=True)
        fraction, winner, agreed = 0., 0, 0
        if len(votes):
            best = int(counts.argmax()); agreed = int(counts[best])
            fraction = float(agreed / len(votes)); winner = int(labels[best])
        reason = 'insufficient_eligible_frames'
        if len(votes) >= cfg.min_eligible_frames:
            reason = 'named' if fraction >= cfg.min_frame_vote_fraction else 'inconsistent_frame_votes'
        semantic_id = winner if reason == 'named' else 0
        named.append({**obj, 'semantic_id': semantic_id,
            'naming_status': reason, 'eligible_frames': len(votes),
            'supporting_frames': [row['frame_id'] for row in evidence if row['eligible'] and row['class_id'] == winner],
            'class_frame_votes': {str(int(k)): int(v) for k, v in zip(labels, counts)},
            'winning_class_id_before_abstention': winner, 'winning_class_frames': agreed,
            'winning_frame_vote_fraction': fraction,
            'vote_fraction_is_calibrated_probability': False,
            'coarse_class_preserved': semantic_id in (33, 34)})
    audit = {'method': 'confidence-gated frame-majority object naming from cached semantic observations',
        'config': asdict(cfg), 'distinct_input_frames': len(seen), 'objects': len(named),
        'named_objects': sum(row['semantic_id'] > 0 for row in named),
        'unknown_objects': sum(row['semantic_id'] == 0 for row in named),
        'semantic_map_modified': False, 'instance_map_modified': False,
        'gt_consumed': False, 'new_model_inference': False,
        'one_vote_per_original_frame': True, 'coarse_ids_remapped_to_fine': False}
    return named, audit
