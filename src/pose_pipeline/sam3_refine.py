"""Resolve compatible SAM concepts with measured SGF evidence, without GT.

Unresolved subtypes are explicit coarse classes, never silently remapped to a
benchmark label. Existing known pixel labels are retained.
"""
from __future__ import annotations
import numpy as np
from .sam3_fusion import PixelClaims

FAMILIES = ((8, 17, 33), (7, 14, 34))
FAMILY_NAMES = {33: 'table-like object (desk/table unresolved)',
                34: 'curtain-like object (subtype unresolved)'}


def interior(semantic):
    safe = np.zeros_like(semantic, dtype=bool)
    s = semantic[1:-1, 1:-1]
    safe[1:-1, 1:-1] = ((s > 0) & (s == semantic[:-2, 1:-1])
        & (s == semantic[2:, 1:-1]) & (s == semantic[1:-1, :-2])
        & (s == semantic[1:-1, 2:]))
    return safe


def resolve_families(claims, anchor_rows, anchor_cols, sgf_labels,
                     sgf_confidence, min_anchors=50, agreement=.8):
    raw = claims.finalize()
    sem, inst, conf = (x.copy() for x in raw)
    records = []
    # Each anchor must be one distinct original map point already depth-tested.
    for ca, cb, coarse in FAMILIES:
        family_score = np.maximum(claims.scores[ca], claims.scores[cb])
        rest = claims.scores.copy(); rest[[ca, cb]] = 0
        allowed = ((sem == 0) & (claims.scores[ca] >= .5)
            & (claims.scores[cb] >= .5)
            & (family_score - rest.max(axis=0) >= .1))
        pairs = np.unique(np.stack([claims.masks[ca][allowed],
                                    claims.masks[cb][allowed]], axis=1), axis=0)
        for ma, mb in pairs:
            region = allowed & (claims.masks[ca] == ma) & (claims.masks[cb] == mb)
            anchored = region[anchor_rows, anchor_cols] & (sgf_confidence >= .5)
            compatible = anchored & np.isin(sgf_labels, [ca, cb])
            counts = [int(np.sum(compatible & (sgf_labels == c))) for c in [ca, cb]]
            total = sum(counts); winner = int(np.argmax(counts)); output_class = coarse
            if total >= min_anchors and counts[winner]/total >= agreement:
                output_class = [ca, cb][winner]
            # One canonical mask identity per compatible overlap, no ID aliases.
            mask_id = len(claims.records) + 1
            score = float(family_score[region].max())
            claims.records.append({'mask_id':mask_id, 'class_id':output_class,
                'score':score, 'pixel_count':int(region.sum()),
                'origin':'compatible SAM masks with SGF subtype anchors',
                'parent_mask_ids':[int(ma),int(mb)]})
            sem[region] = output_class; inst[region] = mask_id
            conf[region] = family_score[region]
            records.append({'family':[ca,cb], 'mask_ids':[int(ma),int(mb)],
                'anchor_counts':counts, 'class_id':output_class,
                'pixels':int(region.sum()), 'used_sgf':output_class != coarse})
    known = raw[0] > 0
    assert np.array_equal(sem[known], raw[0][known])
    return sem, inst, conf, records, raw


def infer_claims(processor, image, depth_shape, taxonomy):
    import torch
    from PIL import Image
    claims = PixelClaims(depth_shape, 35)
    packed = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = processor.set_image(image)
        for category in taxonomy:
            if category['prompt'] is None: continue
            processor.reset_all_prompts(state)
            result = processor.set_text_prompt(prompt=category['prompt'], state=state)
            scores = result['scores'].float().cpu().numpy()
            masks = result['masks'].cpu().numpy()
            for mask, score in zip(masks, scores):
                mask = mask.reshape(image.height, image.width)
                if mask.shape != depth_shape:
                    mask = np.array(Image.fromarray(mask).resize(
                        (depth_shape[1], depth_shape[0]), Image.Resampling.NEAREST))
                claims.add(category['id'], mask, float(score))
                packed.append(np.packbits(mask.reshape(-1)))
    packed = np.asarray(packed, np.uint8).reshape(len(packed), -1) if packed else np.zeros((0, (np.prod(depth_shape)+7)//8), np.uint8)
    return claims, packed
