"""Deterministic, equal-budget semantic views from the final refill trajectory.

This selects SAM3 input only; reconstruction still consumes every raw frame.
Pose separation reduces redundant views but does not prove independence.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..contracts import bind_manifest_trajectory, load_manifest, load_trajectory, validate_se3
from .common import read, selected_frames, sha

POLICIES = ('stride', 'quality', 'quality-diverse')
MIN_TRANSLATION_M = .12
MIN_ROTATION_DEG = 8.


def selection_budget(frames, stride=5, budget=None, policy='stride'):
    if policy not in POLICIES:
        raise ValueError(f'unknown view policy: {policy}')
    default = len(selected_frames(frames, stride))
    if budget is None:
        return default
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= len(frames):
        raise ValueError('view budget must be a positive integer no larger than the raw frame count')
    return budget


def distinct_pose(left, right):
    """Match the refinement selector's translation / viewing-axis thresholds."""
    translation = np.linalg.norm(left[:3, 3] - right[:3, 3])
    angle = np.degrees(np.arccos(np.clip(left[:3, 2] @ right[:3, 2], -1., 1.)))
    return bool(translation >= MIN_TRANSLATION_M or angle >= MIN_ROTATION_DEG)


def select_candidates(candidates, budget, *, diverse=False):
    """Rank finite quality scores, then return exactly budget IDs in time order.

    Insufficient geometric diversity falls back to remaining quality-ranked
    views. Those additions are explicit in the audit and are not counted as
    geometrically separated views.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= len(candidates):
        raise ValueError('invalid candidate budget')
    seen = set()
    for row in candidates:
        if (not isinstance(row['frame_id'], int) or isinstance(row['frame_id'], bool)
                or row['frame_id'] in seen or not np.isfinite(row['quality'])
                or row['quality'] < 0):
            raise ValueError('unique frame IDs and finite nonnegative quality required')
        seen.add(row['frame_id'])
        if diverse:
            validate_se3(row['pose'])
    ordered = sorted(enumerate(candidates), key=lambda pair: (-pair[1]['quality'], pair[0]))
    selected = []
    reasons = {}
    if diverse:
        for index, row in ordered:
            if all(distinct_pose(np.asarray(row['pose']), np.asarray(candidates[i]['pose'])) for i in selected):
                selected.append(index)
                reasons[index] = 'quality_and_pose_diversity'
                if len(selected) == budget:
                    break
    else:
        selected = [index for index, _ in ordered[:budget]]
        reasons = {index: 'quality' for index in selected}
    fallback = []
    for index, _ in ordered:
        if len(selected) == budget:
            break
        if index not in reasons:
            selected.append(index)
            fallback.append(index)
            reasons[index] = 'quality_budget_fallback'
    selected.sort()
    return [candidates[index]['frame_id'] for index in selected], {
        'selection_reasons': {str(candidates[i]['frame_id']): reasons[i] for i in selected},
        'diversity_fallback_frame_ids': [candidates[i]['frame_id'] for i in sorted(fallback)],
        'diverse_frame_count': len(selected) - len(fallback) if diverse else None,
        'pose_diversity_is_statistical_independence': False,
    }


def frame_quality(frame):
    """CPU-only frame proxy; no learned masks, reconstructed map, or GT input."""
    from PIL import Image
    with Image.open(frame.color_path) as image:
        gray = np.asarray(image.convert('L').resize((320, 240)), dtype=np.float32)
    with Image.open(frame.depth_path) as image:
        depth = np.asarray(image)
    if depth.ndim != 2:
        raise ValueError('quality selection requires a single-channel depth image')
    laplacian = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
                 + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    sharpness = float(laplacian.var())
    valid_depth_fraction = float(np.mean(np.isfinite(depth) & (depth > 0)))
    quality = valid_depth_fraction * (.5 + .5 * sharpness / (sharpness + 50.))
    return {'quality': quality, 'sharpness': sharpness, 'valid_depth_fraction': valid_depth_fraction}


def build_view_plan(manifest_path, trajectory_path, *, policy='stride', stride=5, budget=None):
    manifest_path, trajectory_path = Path(manifest_path).resolve(), Path(trajectory_path).resolve()
    initial = {'manifest_sha256': sha(manifest_path), 'trajectory_sha256': sha(trajectory_path)}
    manifest = load_manifest(manifest_path)
    poses, trajectory = load_trajectory(trajectory_path)
    bound = bind_manifest_trajectory(manifest, poses)
    if (trajectory['sequence_id'] != manifest.sequence_id
            or [frame.frame_id for frame, _ in bound] != [frame.frame_id for frame in manifest.frames]):
        raise ValueError('semantic selection requires final poses in the manifest sequence order')
    count = selection_budget(manifest.frames, stride, budget, policy)
    audit, candidates = {}, []
    if policy == 'stride':
        if budget is None:
            chosen = selected_frames(manifest.frames, stride)
        else:
            # An explicit budget controls temporal sampling for a fair baseline.
            indices = np.linspace(0, len(bound) - 1, count, dtype=int)
            chosen = [manifest.frames[int(i)] for i in indices]
        frame_ids = [frame.frame_id for frame in chosen]
    else:
        candidates = [{'frame_id': frame.frame_id, 'pose': pose.t_world_camera.tolist(),
                       **frame_quality(frame)} for frame, pose in bound]
        frame_ids, audit = select_candidates(candidates, count, diverse=policy == 'quality-diverse')
    if initial != {'manifest_sha256': sha(manifest_path), 'trajectory_sha256': sha(trajectory_path)}:
        raise ValueError('selection manifest or final trajectory changed while selecting')
    return {'schema': 'semantic_view_plan.v1', 'policy': policy, 'stride': stride,
            'requested_budget': budget, 'effective_budget': count, 'selected_frame_ids': frame_ids,
            'raw_frames': len(bound), 'reconstruction_frame_count': len(bound),
            'manifest': str(manifest_path), 'trajectory': str(trajectory_path), **initial,
            'min_translation_m': MIN_TRANSLATION_M, 'min_rotation_deg': MIN_ROTATION_DEG,
            'quality_scope': 'full-frame RGB sharpness and valid depth fraction; no object visibility',
            'GT_used': False, 'candidates': candidates, **audit}


def frames_from_plan(manifest_path, plan_path, expected_sha256):
    """Validate a pipeline-owned plan before loading any SAM3 weights."""
    if not expected_sha256 or sha(plan_path) != expected_sha256:
        raise ValueError('view plan digest mismatch')
    plan = read(plan_path)
    if plan.get('schema') != 'semantic_view_plan.v1' or sha(manifest_path) != plan['manifest_sha256']:
        raise ValueError('view plan manifest mismatch')
    if sha(plan['trajectory']) != plan['trajectory_sha256']:
        raise ValueError('final trajectory changed after view selection')
    manifest = load_manifest(manifest_path)
    ids = plan['selected_frame_ids']
    if (len(ids) != plan['effective_budget'] or len(set(ids)) != len(ids)
            or not all(isinstance(fid, int) and not isinstance(fid, bool) for fid in ids)):
        raise ValueError('invalid selected frame IDs')
    wanted = set(ids)
    frames = [frame for frame in manifest.frames if frame.frame_id in wanted]
    if [frame.frame_id for frame in frames] != ids or not frames:
        raise ValueError('selected frame IDs must be an ordered manifest subset')
    return frames, plan
