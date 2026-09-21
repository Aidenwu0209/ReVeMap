"""Diagnostics for matched SAM3 image/video masks, without ground-truth input.

Same-map-point agreement measures temporal consistency, not segmentation accuracy.
The caller must preserve masks and imagery for visual review of stable mistakes.
"""
from __future__ import annotations

import numpy as np


def depth_masks(masks, shape):
    """Nearest-neighbour sampling onto the existing calibrated depth image."""
    from PIL import Image
    masks = np.asarray(masks, dtype=bool)
    h, w = shape
    if len(masks) == 0:
        return np.zeros((0, h, w), bool)
    masks = masks.reshape(len(masks), *masks.shape[-2:])
    return np.stack([np.asarray(Image.fromarray(m).resize((w, h), Image.Resampling.NEAREST))
                     if m.shape != (h, w) else m for m in masks])


def matched_temporal_metrics(frames):
    """Compare adjacent observations only where both pass frozen depth visibility.

    Each frame contains visible_map_ids and per-mask boolean projected_masks.
    Empty/empty pairs have undefined foreground IoU and are excluded explicitly.
    IDs are ignored: a per-image predictor has no temporal IDs to compare fairly.
    """
    pairs = []
    union_visible, union_positive = set(), set()
    for f in frames:
        ids = np.asarray(f['visible_map_ids'])
        masks = np.asarray(f['projected_masks'], bool)
        positive = masks.any(0) if len(masks) else np.zeros(len(ids), bool)
        union_visible.update(ids.tolist()); union_positive.update(ids[positive].tolist())
    for a, b in zip(frames, frames[1:]):
        common, ia, ib = np.intersect1d(a['visible_map_ids'], b['visible_map_ids'], return_indices=True)
        ma, mb = np.asarray(a['projected_masks'], bool), np.asarray(b['projected_masks'], bool)
        pa = ma[:, ia].any(0) if len(ma) else np.zeros(len(common), bool)
        pb = mb[:, ib].any(0) if len(mb) else np.zeros(len(common), bool)
        union = int((pa | pb).sum()); intersection = int((pa & pb).sum())
        pairs.append({'frame_a': int(a['frame_id']), 'frame_b': int(b['frame_id']),
                      'common_visible_points': len(common), 'positive_union_points': union,
                      'positive_intersection_points': intersection,
                      'foreground_iou': intersection / union if union else None,
                      'foreground_disagreement': int((pa ^ pb).sum()) / union if union else None})
    valid = [r for r in pairs if r['positive_union_points']]
    total_union = sum(r['positive_union_points'] for r in valid)
    return {'pairs': pairs, 'eligible_pairs': len(valid), 'all_pairs': len(pairs),
            'empty_foreground_pairs_excluded': len(pairs) - len(valid),
            'mean_foreground_iou': float(np.mean([r['foreground_iou'] for r in valid])) if valid else None,
            'weighted_foreground_iou': sum(r['positive_intersection_points'] for r in valid) / total_union if total_union else None,
            'observed_map_points': len(union_visible), 'segmented_map_points_union': len(union_positive),
            'segmented_fraction_of_observed_map': len(union_positive) / len(union_visible) if union_visible else None,
            'accuracy_measured': False}


def per_mask_temporal_metrics(frames, tracked_ids=False, minimum_points=20, match_iou=.25):
    """Posthoc geometry-based mask matching; IDs are evaluated, never ground truth.

    Retain masks with >=20 points jointly visible in each adjacent pair. Hungarian
    matching maximizes IoU, then accepts pairs at IoU >=.25. Merge/split-like events
    require >=20 intersection points and >=.5 intersection/min(area). These are
    transitions in predictions, not verified mistakes or official tracking scores.
    """
    from scipy.optimize import linear_sum_assignment
    rows=[];total_matches=total_unmatched=stable_ids=0;ious=[]
    for a,b in zip(frames,frames[1:]):
        common,ia,ib=np.intersect1d(a['visible_map_ids'],b['visible_map_ids'],return_indices=True)
        ma=np.asarray(a['projected_masks'],bool)[:,ia];mb=np.asarray(b['projected_masks'],bool)[:,ib]
        keep_a=np.flatnonzero(ma.sum(1)>=minimum_points);keep_b=np.flatnonzero(mb.sum(1)>=minimum_points)
        ma=ma[keep_a];mb=mb[keep_b]
        inter=ma.astype(np.int32)@mb.astype(np.int32).T
        aa=ma.sum(1);ab=mb.sum(1);union=aa[:,None]+ab[None,:]-inter
        iou=inter/np.maximum(union,1)
        rr,cc=linear_sum_assignment(-iou)
        accepted=[(int(r),int(c)) for r,c in zip(rr,cc) if iou[r,c]>=match_iou]
        overlap=(inter>=minimum_points)&(inter/np.maximum(np.minimum(aa[:,None],ab[None,:]),1)>=.5)
        stable=sum(int(a['object_ids'][keep_a[r]]==b['object_ids'][keep_b[c]]) for r,c in accepted) if tracked_ids else None
        vals=[float(iou[r,c]) for r,c in accepted];ious.extend(vals)
        unmatched=len(ma)+len(mb)-2*len(accepted);total_matches+=len(accepted);total_unmatched+=unmatched
        if stable is not None:stable_ids+=stable
        rows.append({'frame_a':int(a['frame_id']),'frame_b':int(b['frame_id']),
                     'common_visible_points':len(common),'eligible_masks_a':len(ma),'eligible_masks_b':len(mb),
                     'matched_pairs':len(accepted),'matched_ious':vals,'unmatched_mask_observations':unmatched,
                     'merge_like_targets':int((overlap.sum(0)>=2).sum()),
                     'split_like_sources':int((overlap.sum(1)>=2).sum()),'matched_same_id':stable})
    return {'settings':{'minimum_common_visible_mask_points':minimum_points,'hungarian_match_iou':match_iou,
                        'transition_overlap_coefficient':.5},
            'matched_pairs':total_matches,'mean_matched_iou':float(np.mean(ious)) if ious else None,
            'unmatched_mask_observations':total_unmatched,
            'merge_like_targets':sum(r['merge_like_targets'] for r in rows),
            'split_like_sources':sum(r['split_like_sources'] for r in rows),
            'same_id_fraction_of_geometrically_matched_pairs':stable_ids/total_matches if tracked_ids and total_matches else None,
            'ground_truth_tracking_accuracy':False,'pairs':rows}
