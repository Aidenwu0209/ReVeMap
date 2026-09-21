"""Safety properties for measured point assignment and independent view selection."""
import numpy as np
import pytest
from pose_pipeline.semantic_runtime.refinement.assignment import assign_unknown
from pose_pipeline.semantic_runtime.refinement.observations import choose
from pose_pipeline.semantic_runtime.refinement.registered_input import RegisteredInput


def evidence(extra=0, frames=(0, 20, 40)):
    n = 200
    base = {'semantic': np.zeros(n, np.int32), 'instance': np.zeros(n, np.int32),
            'confidence': np.zeros(n, np.float32)}
    base['instance'][:80] = 1
    base['instance'][150:] = 2
    base['semantic'][150:] = 3
    ob = {'instance_id': 1, 'semantic_id': 0, 'votes': {'quality_canonical': {
        'frames': list(frames), 'name': 'box', 'labels': ['box'] * len(frames)}}}
    queries = {(fid, 'box'): {'visible': np.arange(n), 'candidates': [
        {'score': .9, 'points': np.arange(80 + extra)}]} for fid in frames}
    return base, {'0': 'unknown', '3': 'chair'}, [ob], queries


def test_unknown_fill_preserves_known_labels_instances_and_unseen_points():
    base, classes, obs, queries = evidence()
    new, _, _, audit = assign_unknown(base, classes, obs, queries)
    assert np.all(new['semantic'][:80] > 0)
    assert np.array_equal(new['semantic'][80:], base['semantic'][80:])
    assert np.array_equal(new['instance'], base['instance'])
    assert np.array_equal(new['confidence'], base['confidence'])
    assert audit[0]['strict_points'] == 80


def test_duplicates_and_out_of_view_evidence_rejected():
    base, classes, obs, queries = evidence(frames=(0, 0, 0))
    with pytest.raises(ValueError, match='duplicate naming frame'):
        assign_unknown(base, classes, obs, queries)
    base, classes, obs, queries = evidence()
    queries[0, 'box']['visible'] = np.arange(10)
    with pytest.raises(ValueError, match='invisible'):
        assign_unknown(base, classes, obs, queries)


def test_fragment_needs_three_frames_and_never_expands_ownership():
    base, classes, obs, queries = evidence(extra=40)
    strict, _, _, _ = assign_unknown(base, classes, obs, queries)
    assert np.array_equal(strict['semantic'], base['semantic'])
    new, _, _, audit = assign_unknown(base, classes, obs, queries, fragments=True)
    assert audit[0]['fragment_points'] == 80
    assert np.all(new['semantic'][80:150] == 0)
    del queries[40, 'box']
    new, _, _, _ = assign_unknown(base, classes, obs, queries, fragments=True)
    assert np.array_equal(new['semantic'], base['semantic'])


def test_direct_point_support_not_whole_instance_painting():
    base, classes, obs, queries = evidence()
    for q in queries.values():q['candidates'][0]['points'] = np.arange(60)
    new, _, _, _ = assign_unknown(base, classes, obs, queries)
    assert np.all(new['semantic'][:60] > 0)
    assert np.all(new['semantic'][60:80] == 0)


def test_heldout_cannot_create_support_and_consensus_is_rechecked():
    base, classes, obs, queries = evidence()
    queries[100, 'box'] = queries.pop((40, 'box'))
    queries[120, 'box'] = queries.pop((20, 'box'))
    new, _, _, _ = assign_unknown(base, classes, obs, queries)
    assert np.array_equal(new['semantic'], base['semantic'])
    obs[0]['votes']['quality_canonical']['name'] = 'piano'
    with pytest.raises(ValueError, match='consensus'):
        assign_unknown(base, classes, obs, queries)


def test_select_views_rejects_repeated_camera_and_prefers_quality():
    poses = []
    for i in [0, 0, 1, 2]:
        p = np.eye(4);p[0, 3] = i * .2;poses.append(p)
    views = [{'frame_id': i*20, 'quality': q, 'pose': p.tolist()}
             for i, (q, p) in enumerate(zip([1, 5, 3, 4], poses))]
    chosen = choose(views, True)
    assert [v['frame_id'] for v in chosen] == [20, 60, 40]


def test_input_requires_explicit_registration_contract():
    with pytest.raises(ValueError, match='rgb_registration'):
        RegisteredInput('scannet/scene0030_00', {})
    reader = RegisteredInput('scannet/scene0030_00', {'rgb_registration': 'already_registered'})
    assert reader.offsets is None


def test_all_stages_get_absolute_import_path_and_correct_interpreters(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from pose_pipeline.semantic_runtime.refinement import __main__ as runner
    config = {'cpu_python': '/cpu/python', 'vlm_python': '/vlm/python', 'sam3_python': '/sam/python'}
    (tmp_path / 'runtime.json').write_text(json.dumps(config))
    monkeypatch.setattr(runner, 'validate_workspace', lambda root: (['scene'], {}))
    monkeypatch.setattr(runner, 'validate_runtime', lambda *a, **kw: None)
    calls = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda cmd, **kw: calls.append((cmd, kw)))
    runner.run(tmp_path, 'all', True)
    assert [x[0][0] for x in calls] == ['/cpu/python', '/vlm/python', '/cpu/python', '/sam/python', '/cpu/python']
    source = str(Path(runner.__file__).resolve().parents[3])
    assert (Path(source) / 'pose_pipeline').is_dir()
    assert all(x[1]['env']['PYTHONPATH'].split(runner.os.pathsep)[0] == source for x in calls)
    assert all('--fragments' in x[0] and x[1]['check'] for x in calls)
