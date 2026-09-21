"""Conservative RGB-D label projection onto an immutable estimated map.

No learned model, GT, or nearest-surface label extrapolation lives here. Each
map point receives at most one observation per distinct camera frame.
"""
from __future__ import annotations

import numpy as np
from .contracts import validate_se3


def visible_map_pixels(xyz, t_world_camera, intrinsics, depth_m, tolerance=.05):
    """Return original map indices and image pixels agreeing with measured depth."""
    xyz = np.asarray(xyz, dtype=np.float64)
    t = validate_se3(t_world_camera)
    depth = np.asarray(depth_m)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('finite Nx3 original map required')
    if depth.ndim != 2 or tolerance <= 0:
        raise ValueError('HxW depth and positive depth tolerance required')
    fx, fy, cx, cy = intrinsics
    if not np.isfinite(intrinsics).all() or min(fx, fy) <= 0:
        raise ValueError('invalid camera intrinsics')
    camera = (xyz - t[:3, 3]) @ t[:3, :3]
    h, w = depth.shape
    ids = np.flatnonzero(camera[:, 2] > 0)
    q = camera[ids]
    u = np.rint(q[:, 0] * fx / q[:, 2] + cx).astype(np.int64)
    v = np.rint(q[:, 1] * fy / q[:, 2] + cy).astype(np.int64)
    inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    ids, u, v = ids[inside], u[inside], v[inside]
    measured = depth[v, u]
    valid = (np.isfinite(measured) & (measured > 0)
             & (np.abs(camera[ids, 2] - measured) <= tolerance))
    return ids[valid], v[valid], u[valid]


class PixelClaims:
    """Keep the best mask per class and abstain at conflicting class overlaps.

    SAM concept scores are not calibrated across text prompts. The fixed margin
    is a conservative heuristic, not a posterior probability guarantee.
    """
    def __init__(self, shape, class_count):
        self.scores = np.zeros((class_count, *shape), np.float32)
        self.masks = np.zeros((class_count, *shape), np.int32)
        self.records = []

    def add(self, class_id, mask, score):
        if not 0 < class_id < len(self.scores):
            raise ValueError('class ID out of taxonomy')
        mask = np.asarray(mask, bool)
        if mask.shape != self.scores.shape[1:] or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid model mask/score')
        mask_id = len(self.records) + 1
        self.records.append({'mask_id': mask_id, 'class_id': int(class_id), 'score': float(score),
                             'pixel_count': int(mask.sum())})
        win = mask & (score > self.scores[class_id])
        self.scores[class_id, win] = score
        self.masks[class_id, win] = mask_id

    def finalize(self, threshold=.5, margin=.1):
        semantic = self.scores.argmax(axis=0).astype(np.int16)
        confidence = np.take_along_axis(self.scores, semantic[None], axis=0)[0]
        top = np.partition(self.scores, -2, axis=0)[-2:]
        known = (confidence >= threshold) & ((top[-1] - top[-2]) >= margin)
        instance = np.take_along_axis(self.masks, semantic[None], axis=0)[0]
        return (np.where(known, semantic, 0).astype(np.int16),
                np.where(known, instance, 0).astype(np.int32),
                np.where(known, confidence, 0).astype(np.float32))


class MapVotes:
    def __init__(self, n, class_count):
        self.scores = np.zeros((n, class_count), np.float32)
        self.counts = np.zeros((n, class_count), np.uint32)
        self.visible_count = np.zeros(n, np.uint32)
        self.frames = set()

    def add(self, frame_id, point_ids, semantic, confidence):
        if frame_id in self.frames:
            raise ValueError('duplicate frame would double count semantic evidence')
        ids = np.asarray(point_ids, np.int64)
        sem = np.asarray(semantic, np.int64)
        conf = np.asarray(confidence, np.float32)
        if ids.shape != sem.shape or ids.shape != conf.shape or len(np.unique(ids)) != len(ids):
            raise ValueError('one label per unique visible map point required')
        if (np.any(ids < 0) or np.any(ids >= len(self.scores)) or np.any(sem < 0)
                or np.any(sem >= self.scores.shape[1]) or not np.isfinite(conf).all()
                or np.any(conf < 0) or np.any(conf > 1)):
            raise ValueError('invalid vote')
        self.frames.add(frame_id)
        self.visible_count[ids] += 1
        known = sem > 0
        self.scores[ids[known], sem[known]] += conf[known]
        self.counts[ids[known], sem[known]] += 1

    def finalize(self, min_views=2, vote_share=.65, margin=.15):
        sem = self.scores.argmax(axis=1).astype(np.int32)
        idx = np.arange(len(sem))
        total = self.scores.sum(axis=1)
        best = self.scores[idx, sem]
        second = np.partition(self.scores, -2, axis=1)[:, -2]
        confidence = np.divide(best, total, out=np.zeros_like(best), where=total > 0)
        support = self.counts[idx, sem]
        known = ((support >= min_views) & (confidence >= vote_share)
                 & ((best - second) >= margin * total) & (sem > 0))
        return np.where(known, sem, 0), np.where(known, confidence, 0), support


class GeometricInstances:
    """Associate measured masks by shared map support, never by category alone.

    This is an explicit geometric baseline, not SGAligner inference. Ambiguous
    point ownership stays unknown. SGA descriptors may be evaluated separately.
    """
    def __init__(self, min_points=30, min_overlap=.4):
        self.tracks = []
        self.min_points = min_points
        self.min_overlap = min_overlap

    def add(self, frame_id, point_ids, mask_ids, classes):
        used = set()
        assignments = {}
        for mask_id in np.unique(mask_ids):
            if mask_id <= 0:
                continue
            chosen = mask_ids == mask_id
            ids = np.unique(point_ids[chosen])
            labels = np.unique(classes[chosen])
            if len(ids) < self.min_points or len(labels) != 1 or labels[0] <= 0:
                continue
            category = int(labels[0]); candidates = []
            points = set(ids.tolist())
            for i, track in enumerate(self.tracks):
                if i in used or category != track['category'] or frame_id in track['frames']:
                    continue
                shared = len(points & track['points'])
                overlap = shared / min(len(points), len(track['points']))
                if shared >= self.min_points and overlap >= self.min_overlap:
                    candidates.append((overlap, i))
            candidates.sort(reverse=True)
            # Two plausible owners indicate a merged mask: do not merge tracks.
            if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .15:
                continue
            if candidates:
                i = candidates[0][1]
                self.tracks[i]['points'].update(points)
                self.tracks[i]['frames'].add(frame_id)
            else:
                i = len(self.tracks)
                self.tracks.append({'category': category, 'points': points, 'frames': {frame_id}})
            used.add(i); assignments[int(mask_id)] = i + 1
        return assignments

    def finalize(self, semantic, min_views=2):
        instances = np.zeros(len(semantic), np.int32)
        ambiguous = np.zeros(len(semantic), bool)
        for i, track in enumerate(self.tracks):
            if len(track['frames']) < min_views:
                continue
            ids = np.array(sorted(track['points']), np.int64)
            ids = ids[semantic[ids] == track['category']]
            ambiguous[ids] |= instances[ids] > 0
            instances[ids] = i + 1
        instances[ambiguous] = 0
        return instances
