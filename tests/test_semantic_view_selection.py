"""Same inference budget, final-pose provenance, and explicit diversity fallback."""
from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np
import pytest
from PIL import Image

from pose_pipeline.contracts import FrameRecord, PoseRecord, SequenceManifest, write_manifest, write_trajectory
from pose_pipeline.semantic_runtime.common import read, sha, write
from pose_pipeline.semantic_runtime.view_selection import (
    build_view_plan, distinct_pose, frame_quality, frames_from_plan, select_candidates, selection_budget,
)


@pytest.fixture
def sequence(tmp_path):
    frames, poses = [], []
    for i in range(12):
        color, depth = tmp_path / f'rgb_{i}.png', tmp_path / f'depth_{i}.png'
        Image.fromarray(np.full((32, 32, 3), 100 + i, np.uint8)).save(color)
        Image.fromarray(np.full((32, 32), 1000, np.uint16)).save(depth)
        frames.append(FrameRecord(i, i * 1000, color, depth, (50., 50., 16., 16.)))
        pose = np.eye(4)
        pose[0, 3] = i * .2
        poses.append(PoseRecord(i, i * 1000, pose, source='final_refill_test'))
    manifest = tmp_path / 'manifest.json'
    trajectory = tmp_path / 'trajectory.json'
    write_manifest(manifest, SequenceManifest('orbbec', 'fixture', tmp_path, 1000., tuple(frames), 'test'))
    write_trajectory(trajectory, poses, sequence_id='fixture', arm='refill')
    return manifest, trajectory, frames


def candidate(fid, quality, x):
    pose = np.eye(4)
    pose[0, 3] = x
    return {'frame_id': fid, 'quality': quality, 'pose': pose.tolist()}


def test_diversity_avoids_near_duplicate_high_quality_views():
    rows = [candidate(0, 1., 0.), candidate(1, .99, .01), candidate(2, .9, .3)]
    quality, _ = select_candidates(rows, 2)
    diverse, audit = select_candidates(rows, 2, diverse=True)
    assert quality == [0, 1]
    assert diverse == [0, 2]
    assert audit['diverse_frame_count'] == 2
    assert audit['diversity_fallback_frame_ids'] == []
    assert audit['pose_diversity_is_statistical_independence'] is False


def test_static_camera_fills_equal_budget_and_discloses_fallback():
    rows = [candidate(i, .9, 0.) for i in range(5)]
    selected, audit = select_candidates(rows, 3, diverse=True)
    assert selected == [0, 1, 2]
    assert audit['diverse_frame_count'] == 1
    assert audit['diversity_fallback_frame_ids'] == [1, 2]
    assert select_candidates(rows, 3, diverse=True) == (selected, audit)


def test_diversity_accepts_rotation_without_translation():
    left = np.eye(4)
    right = np.eye(4)
    theta = np.deg2rad(10)
    right[:3, :3] = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    assert distinct_pose(left, right)


def test_default_stride_unchanged_and_all_policies_match_explicit_budget(sequence, monkeypatch):
    manifest, trajectory, _ = sequence
    default = build_view_plan(manifest, trajectory)
    assert default['selected_frame_ids'] == [0, 5, 10, 11]
    from pose_pipeline.semantic_runtime import view_selection
    monkeypatch.setattr(view_selection, 'frame_quality', lambda f: {'quality': f.frame_id / 12.})
    for policy in ('stride', 'quality', 'quality-diverse'):
        plan = build_view_plan(manifest, trajectory, policy=policy, budget=3)
        assert len(plan['selected_frame_ids']) == plan['effective_budget'] == 3
        assert plan['raw_frames'] == plan['reconstruction_frame_count'] == 12
        assert plan == build_view_plan(manifest, trajectory, policy=policy, budget=3)
        assert plan['trajectory_sha256'] == sha(trajectory)
    assert build_view_plan(manifest, trajectory, policy='quality', budget=3)['selected_frame_ids'] == [9, 10, 11]
    assert build_view_plan(manifest, trajectory, policy='quality')['effective_budget'] == 4


def test_quality_measures_valid_depth_and_sharpness(sequence):
    _, _, frames = sequence
    flat = frame_quality(frames[0])
    assert flat == {'quality': .5, 'sharpness': 0., 'valid_depth_fraction': 1.}
    checker = ((np.indices((240, 320)).sum(axis=0) % 2) * 255).astype(np.uint8)
    Image.fromarray(checker).save(frames[1].color_path)
    assert frame_quality(frames[1])['quality'] > flat['quality']
    Image.fromarray(np.zeros((32, 32), np.uint16)).save(frames[1].depth_path)
    assert frame_quality(frames[1])['quality'] == 0


def test_verified_plan_roundtrip_and_changed_pose_rejection(sequence, tmp_path):
    manifest, trajectory, _ = sequence
    plan = build_view_plan(manifest, trajectory, budget=3)
    path = tmp_path / 'VIEW_PLAN.json'
    write(path, plan)
    frames, loaded = frames_from_plan(manifest, path, sha(path))
    assert [f.frame_id for f in frames] == plan['selected_frame_ids']
    assert loaded == plan
    with pytest.raises(ValueError, match='digest'):
        frames_from_plan(manifest, path, '0' * 64)
    trajectory.write_text(trajectory.read_text() + '\n')
    with pytest.raises(ValueError, match='trajectory changed'):
        frames_from_plan(manifest, path, sha(path))


@pytest.mark.parametrize('budget', [0, -1, True, 1.5, 13])
def test_invalid_budgets_fail_before_inference(sequence, budget):
    _, _, frames = sequence
    with pytest.raises(ValueError, match='view budget'):
        selection_budget(frames, budget=budget)


@pytest.mark.parametrize('schedule', ['serial', 'parallel'])
@pytest.mark.parametrize('input_changed', [False, True])
def test_pipeline_selects_after_refill_with_full_mapping_input(sequence, tmp_path, monkeypatch, schedule, input_changed):
    from pose_pipeline.semantic_runtime import pipeline, view_selection
    manifest, trajectory, frames = sequence
    output = tmp_path / 'run'
    runtime = tmp_path / 'runtime.json'
    config = {'cpu_python': 'cpu-python', 'sam3_python': 'gpu-python'}
    write(runtime, config)
    monkeypatch.setattr(pipeline, 'validate_runtime', lambda config, model: config)
    monkeypatch.setattr(view_selection, 'frame_quality', lambda frame: {'quality': .9})
    events = []
    class FakeProcesses:
        def __init__(self, root, timeout):
            self.mapping_done = False
        def launch(self, name, command):
            args = command[command.index('pose_pipeline.semantic_runtime.worker') + 1:]
            opts = {args[i]: args[i + 1] for i in range(1, len(args), 2)}
            events.append(name)
            if name == 'mapping':
                assert opts['--manifest'] == str(manifest)
                target = Path(opts['--output'])
                (target / 'refill').mkdir(parents=True)
                (target / 'refill/trajectory.json').write_bytes(trajectory.read_bytes())
                write(target / 'run_status.json', {'stages': [{'stage': 'refill', 'returncode': 0}]})
            elif name == 'view_selection':
                assert 'mapping' in events
                assert (output / 'mapping/refill/trajectory.json').exists()
                assert self.mapping_done == (schedule == 'serial')
                plan = build_view_plan(opts['--manifest'], opts['--trajectory'], policy=opts['--view-policy'],
                                       stride=int(opts['--stride']), budget=int(opts['--view-budget']))
                write(Path(opts['--output']) / 'VIEW_PLAN.json', plan)
            elif name == 'sam3':
                selected, _ = frames_from_plan(opts['--manifest'], opts['--view-plan'], opts['--view-plan-sha256'])
                assert len(selected) == 3
            elif name == 'fusion':
                fused = output / 'fused'
                (fused / 'export').mkdir(parents=True)
                (fused / 'export/map_labeled.ply').write_bytes(b'mock exported geometry')
                write(fused / 'classes.json', {'0': 'unknown'})
                write(fused / 'result.json', {'status': 'completed'})
            elif name == 'backfill':
                write(output / 'fused/instance_names.json', [])
                if input_changed:
                    frames[0].color_path.write_bytes(b'input changed after reconstruction')
            return (name, SimpleNamespace(poll=lambda: None), 0)
        def wait(self, job):
            if job[0] == 'mapping':
                self.mapping_done = True
        def check(self, job):
            return None
        def close(self):
            pass
    monkeypatch.setattr(pipeline, 'Processes', FakeProcesses)
    args = argparse.Namespace(manifest=manifest, runtime=runtime, output=output,
        schedule=schedule, vlm='none', stride=5, view_policy='quality-diverse', view_budget=3)
    if input_changed:
        with pytest.raises(RuntimeError, match='raw input changed'):
            pipeline.run(args)
        assert not (output / 'fused/ARTIFACTS.json').exists()
        assert not (output / 'COMPLETE.json').exists()
        assert read(output / 'FAILURE.json')['error_type'] == 'RuntimeError'
        return
    result = pipeline.run(args)
    assert events == ['mapping', 'view_selection', 'sam3', 'fusion', 'backfill']
    assert result['raw_frames'] == len(frames) == 12
    assert result['selected_frames'] == result['view_budget'] == 3
    assert read(output / 'CONFIG.json')['view_policy'] == 'quality-diverse'
    inventory = read(result['artifacts'])
    assert inventory['files']['names']['sha256'] == sha(output / 'fused/instance_names.json')
