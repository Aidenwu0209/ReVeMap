"""Fixed cached-mask ablation separating object identity from semantic naming.

No model inference, ground truth, geometry growth or semantic overwrite. Raw
proposals remain conditioned on the original text prompts; this is not generic
discovery of every object in a scene.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from .sam3_multiview import fuse_instances


@dataclass(frozen=True)
class UnknownConfig:
    mask_score: float = .5
    duplicate_iou: float = .8
    ownership_margin: float = .1
    min_mask_points: int = 30
    min_output_points: int = 50


def raw_mask_partition(packed_masks, shape, scores, config=None):
    """Deduplicate across prompt classes before assigning one local mask/pixel.

    A same-frame duplicate never becomes an extra independent observation.
    Nonduplicate masks with similar scores abstain where they overlap.
    """
    cfg = config or UnknownConfig()
    if not (0 < cfg.mask_score <= 1 and 0 < cfg.duplicate_iou <= 1
            and 0 <= cfg.ownership_margin <= 1):
        raise ValueError('invalid raw-mask configuration')
    h, w = (int(s) for s in shape)
    packed = np.asarray(packed_masks)
    scores = np.asarray(scores, np.float32)
    if h < 1 or w < 1 or packed.dtype != np.uint8 or packed.shape != (len(scores), (h*w+7)//8):
        raise ValueError('invalid packed masks')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('invalid mask scores')
    masks = np.unpackbits(packed, axis=1, count=h*w).astype(bool)
    areas = masks.sum(axis=1)
    kept, duplicates = [], []
    for i in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i)):
        if scores[i] < cfg.mask_score or areas[i] == 0:
            continue
        duplicate = None
        for j in kept:
            intersection = int(np.count_nonzero(masks[i] & masks[j]))
            union = int(areas[i] + areas[j] - intersection)
            if intersection / union >= cfg.duplicate_iou:
                duplicate = j
                break
        if duplicate is None:
            kept.append(i)
        else:
            duplicates.append({'raw_mask_id': i+1, 'kept_raw_mask_id': duplicate+1})
    best, second = np.zeros(h*w, np.float32), np.zeros(h*w, np.float32)
    owner = np.zeros(h*w, np.int32)
    for i in kept:
        selected = masks[i]
        win = selected & (scores[i] > best)
        second[win] = best[win]
        best[win] = scores[i]
        owner[win] = i+1
        runner_up = selected & ~win & (scores[i] > second)
        second[runner_up] = scores[i]
    accepted = (owner > 0) & (best-second >= cfg.ownership_margin)
    owner[~accepted] = 0
    confidence = np.where(accepted, best, 0).reshape(h, w)
    owner = owner.reshape(h, w)
    # Instance boundaries, including gaps left by competing masks, erode once.
    interior = np.zeros((h, w), bool)
    mid = owner[1:-1, 1:-1]
    interior[1:-1, 1:-1] = ((mid > 0) & (mid == owner[:-2, 1:-1])
        & (mid == owner[2:, 1:-1]) & (mid == owner[1:-1, :-2]) & (mid == owner[1:-1, 2:]))
    audit = {'raw_masks': len(scores), 'kept_masks': len(kept), 'duplicates': duplicates,
             'raw_union_pixels': int(np.any(masks, axis=0).sum()),
             'partition_pixels': int(np.count_nonzero(owner)),
             'interior_pixels': int(interior.sum()),
             'ambiguous_overlap_pixels': int(np.sum((best >= cfg.mask_score) & ~accepted)),
             'retained_raw_mask_ids': [i+1 for i in kept]}
    return owner, confidence, interior, audit


def fuse_unknown(n_points, frames, baseline_semantic, config=None):
    """Fuse measured object masks without requiring or rewriting their names."""
    cfg = config or UnknownConfig()
    semantic = np.asarray(baseline_semantic)
    if semantic.shape != (n_points,) or not np.issubdtype(semantic.dtype, np.integer) or np.any(semantic < 0):
        raise ValueError('invalid frozen semantic map')
    # Internal foreground sentinel 1 is never exported as a semantic class.
    # Existing per-frame safety/score/interior gates remain active.
    classless = [{**f, 'semantic': (np.asarray(f['mask_ids']) > 0).astype(np.int32)} for f in frames]
    instances, audit = fuse_instances(n_points, classless, np.ones(n_points, np.int32),
        {'min_mask_points': cfg.min_mask_points, 'min_output_points': cfg.min_output_points,
         'min_point_views': 1, 'min_group_frames': 2, 'object_score_mode': 'max_point'})
    for obj in audit.get('objects', []):
        selected = instances == obj['instance_id']
        labels, counts = np.unique(semantic[selected], return_counts=True)
        obj.pop('semantic_id')
        obj['semantic_histogram'] = {str(int(k)): int(v) for k, v in zip(labels, counts)}
        obj['semantic_naming_executed'] = False
    audit.update(unknown_config=asdict(cfg), semantic_labels_modified=False,
        semantic_naming_executed=False, instance_requires_semantic_label=False,
        owned_unknown_semantic_points=int(np.sum((instances > 0) & (semantic == 0))),
        unknown_is_not_a_new_recognized_class=True)
    return instances, audit
