"""Reuse only successful, immutable stages and preserve the failed source attempt."""
from pathlib import Path

import pytest

from pose_pipeline.semantic_runtime.checkpoints import CheckpointStore
from pose_pipeline.semantic_runtime.common import read, sha, write


def mapping(root):
    directory = root / 'mapping'
    directory.mkdir(parents=True)
    receipt = {'status': 'completed'}
    for key, name in [('manifest', 'manifest.json'), ('trajectory', 'refill/trajectory.json'),
                      ('final_cloud', 'fusion/map.ply')]:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((key + ' verified bytes').encode())
        receipt[key], receipt[key + '_sha256'] = str(path), sha(path)
    write(directory / 'mapping_result.json', receipt)


def semantic(root):
    directory = root / 'semantic'
    path = directory / 'frames/000003.npz'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'immutable mask fixture')
    crop = directory / 'crops/3.png'
    crop.parent.mkdir()
    crop.write_bytes(b'immutable crop fixture')
    write(directory / 'COMPLETE.json', {'status': 'completed', 'selected_frames': 1})
    write(directory / 'FRAMES.json', [{'frame_id': 3, 'mask_sha256': sha(path)}])
    write(directory / 'CROP_TASKS.json', [{'crops': [{'file': str(crop), 'sha256': sha(crop)}]}])


def test_rebinds_copies_and_never_reuses_later_stage_files(tmp_path):
    old, new = tmp_path / 'failed', tmp_path / 'retry'
    mapping(old)
    semantic(old)
    store = CheckpointStore(old, {'inputs': 'fixed', 'source': 'fixed'})
    for stage in ('mapping', 'sam3'):
        store.seal(stage)
    write(old / 'semantic/vlm/FAILURE.json', {'status': 'failed'})
    original_receipt = (old / 'mapping/mapping_result.json').read_bytes()
    retry = CheckpointStore(new, store.expected_context, old)
    assert retry.restore('mapping') and retry.restore('sam3')
    receipt = read(new / 'mapping/mapping_result.json')
    for key in ('manifest', 'trajectory', 'final_cloud'):
        assert Path(receipt[key]).is_relative_to(new)
        assert sha(receipt[key]) == receipt[key + '_sha256']
    crop = read(new / 'semantic/CROP_TASKS.json')[0]['crops'][0]
    assert Path(crop['file']).is_relative_to(new) and sha(crop['file']) == crop['sha256']
    assert not (new / 'semantic/vlm').exists()
    assert (old / 'mapping/mapping_result.json').read_bytes() == original_receipt
    assert read(new / 'RESUME.json')['stages'][1]['reused'] is True


@pytest.mark.parametrize('damage', ['input', 'source', 'missing-seal', 'bad-seal', 'file', 'receipt', 'incomplete'])
def test_changed_or_partial_stages_fall_back_without_copying(tmp_path, damage):
    old, new = tmp_path / 'old', tmp_path / 'new'
    mapping(old)
    context = {'inputs': 'fixed', 'source': 'fixed'}
    CheckpointStore(old, context).seal('mapping')
    expected = dict(context)
    if damage in ('input', 'source'):
        expected['inputs' if damage == 'input' else 'source'] = 'changed'
    elif damage == 'missing-seal':
        (old / 'checkpoints/mapping.json').unlink()
    elif damage == 'bad-seal':
        write(old / 'checkpoints/mapping.json', {})
    elif damage == 'file':
        (old / 'mapping/fusion/map.ply').write_bytes(b'tamper')
    elif damage == 'receipt':
        write(old / 'mapping/mapping_result.json', {'status': 'failed'})
    else:
        (old / 'mapping/refill/trajectory.json').unlink()
    retry = CheckpointStore(new, expected, old)
    assert retry.restore('mapping') is False
    assert not (new / 'mapping').exists()
    assert not (new / '.reuse-mapping').exists()
    assert read(new / 'RESUME.json')['stages'][0]['reused'] is False


def test_rejects_overlapping_attempts_and_unfinished_stage(tmp_path):
    with pytest.raises(ValueError, match='separate'):
        CheckpointStore(tmp_path / 'old/new', {}, tmp_path / 'old')
    semantic(tmp_path)
    write(tmp_path / 'semantic/COMPLETE.json', {'status': 'failed', 'selected_frames': 1})
    with pytest.raises(ValueError, match='complete frame inventory'):
        CheckpointStore(tmp_path, {}).seal('sam3')
    assert not (tmp_path / 'checkpoints/sam3.json').exists()


def test_symlink_replacement_cannot_escape_stage(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    mapping(old)
    CheckpointStore(old, {}).seal('mapping')
    parent = old / 'mapping/fusion'
    moved = old / 'elsewhere'
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    assert not CheckpointStore(new, {}, old).restore('mapping')


def test_sam3_plan_receipt_rebound_and_resealed(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    mapping(old)
    semantic(old)
    old_plan = {'trajectory': str(old / 'mapping/refill/trajectory.json'),
                'trajectory_sha256': sha(old / 'mapping/refill/trajectory.json'),
                'selected_frame_ids': [3], 'policy': 'stride'}
    write(old / 'semantic/VIEW_PLAN.json', old_plan)
    receipt = read(old / 'semantic/COMPLETE.json')
    receipt['view_plan_sha256'] = sha(old / 'semantic/VIEW_PLAN.json')
    write(old / 'semantic/COMPLETE.json', receipt)
    store = CheckpointStore(old, {})
    store.seal('mapping')
    store.seal('sam3')
    retry = CheckpointStore(new, {}, old)
    assert retry.restore('mapping')
    new_plan = {**old_plan, 'trajectory': str(new / 'mapping/refill/trajectory.json')}
    write(new / 'view_selection/VIEW_PLAN.json', new_plan)
    assert retry.restore('sam3', view_plan=new / 'view_selection/VIEW_PLAN.json')
    assert read(new / 'semantic/VIEW_PLAN.json') == new_plan
    current = read(new / 'semantic/COMPLETE.json')
    assert current['view_plan_sha256'] == sha(new / 'view_selection/VIEW_PLAN.json')
    assert current['reused_original_plan_sha256'] == receipt['view_plan_sha256']


def test_run_path_aliases_are_rebound_to_canonical_new_attempt(tmp_path):
    alias = tmp_path / 'alias'
    actual = tmp_path / 'actual'
    actual.mkdir()
    alias.symlink_to(actual, target_is_directory=True)
    old = alias / 'old'
    new = alias / 'new'
    mapping(old)
    semantic(old)
    store = CheckpointStore(old, {})
    store.seal('mapping')
    store.seal('sam3')
    retry = CheckpointStore(new, {}, old)
    assert retry.restore('mapping') and retry.restore('sam3')
    receipt = read(new / 'mapping/mapping_result.json')
    assert Path(receipt['final_cloud']).is_relative_to(new.resolve())
    crop = read(new / 'semantic/CROP_TASKS.json')[0]['crops'][0]
    assert Path(crop['file']).is_relative_to(new.resolve())


def test_changed_raw_input_cannot_seal_a_failed_attempt_for_later_reuse(tmp_path):
    raw = tmp_path / 'depth.png'
    raw.write_bytes(b'original raw input')
    expected = sha(raw)
    old = tmp_path / 'interrupted'
    mapping(old)
    store = CheckpointStore(old, {'inputs': {str(raw): expected}})
    raw.write_bytes(b'changed during mapping')
    with pytest.raises(ValueError, match='input changed before stage seal'):
        store.seal('mapping')
    assert not (old / 'checkpoints/mapping.json').exists()
    raw.write_bytes(b'original raw input')
    assert not CheckpointStore(tmp_path / 'retry', store.expected_context, old).restore('mapping')


@pytest.mark.parametrize('record,key', [('runtime.json', 'runtime'), ('CONFIG.json', 'options'), ('INPUTS.json', 'inputs')])
def test_actual_worker_runtime_and_execution_records_are_checked_before_seal(tmp_path, monkeypatch, record, key):
    from pose_pipeline.semantic_runtime import checkpoints
    expected = {'inputs': {}, 'runtime': {'model': 'original'}, 'options': {}, 'external_runtime': {}}
    for name, field in [('runtime.json', 'runtime'), ('CONFIG.json', 'options'), ('INPUTS.json', 'inputs')]:
        write(tmp_path / name, expected[field])
    mapping(tmp_path)
    monkeypatch.setattr(checkpoints, 'context', lambda *_: expected)
    store = CheckpointStore(tmp_path, expected)
    write(tmp_path / record, {'changed': 'after worker launch'})
    with pytest.raises(ValueError, match='execution record changed before stage seal'):
        store.seal('mapping')
    assert not (tmp_path / 'checkpoints/mapping.json').exists()
