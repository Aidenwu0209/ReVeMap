"""Counterexamples for the production fusion -> candidate -> refinement path."""
from copy import deepcopy
import json

import numpy as np
import pytest

from pose_pipeline.semantic_runtime.multiview import fuse_instances
from pose_pipeline.semantic_runtime.object_candidates import build_candidates, candidate_anchors
from pose_pipeline.semantic_runtime.refinement.assignment import assign_unknown


def observations(n=100):
    return [dict(frame_id=fid, point_ids=np.arange(n), mask_ids=np.ones(n, np.int32),
                 semantic=np.ones(n, np.int32), confidence=np.full(n, .95), interior=np.ones(n, bool))
            for fid in (0, 20, 40)]


def test_unknown_geometric_object_reaches_refinement_without_publishing_unverified_owners():
    n = 100
    base = dict(semantic=np.zeros(n, np.int32), instance=np.zeros(n, np.int32), confidence=np.zeros(n))
    original = deepcopy(base)
    geometry = {}
    base['instance'], _ = fuse_instances(n, observations(n), base['semantic'], geometry_out=geometry)
    assert not base['instance'].any()  # Current output contract remains valid.
    candidate, report = build_candidates(base, geometry)
    assert report['candidate_points'] == n
    assert np.all(candidate == 1)
    ob = dict(instance_id=1, semantic_id=0, votes={'quality_canonical': {
        'frames': [0, 20, 40], 'labels': ['chair'] * 3, 'name': 'chair'}})
    queries = {(fid, 'chair'): {'visible': np.arange(n), 'candidates': [
        {'points': np.arange(70), 'score': .9}]} for fid in (0, 20, 40)}
    new, _, _, _ = assign_unknown(base, {'0':'unknown'}, [ob], queries, candidate_instance=candidate)
    assert np.all(new['instance'][:70] == 1) and np.all(new['semantic'][:70] > 0)
    assert not new['instance'][70:].any() and not new['semantic'][70:].any()
    for key in original:
        np.testing.assert_array_equal(base[key], original[key])


def test_candidate_cannot_overwrite_known_points_or_claim_another_class_owner():
    base = dict(semantic=np.r_[np.ones(60, int), np.zeros(60, int)],
                instance=np.r_[np.ones(60, int), np.zeros(60, int)], confidence=np.zeros(120))
    with pytest.raises(ValueError, match='unknown unowned'):
        candidate_anchors(base, np.ones(120, int))
    candidate = np.r_[np.zeros(60, int), np.ones(60, int)]
    ob = dict(instance_id=1, semantic_id=1, votes={'quality_canonical': {
        'frames':[0,20,40], 'labels':['table']*3, 'name':'table'}})
    queries = {(fid,'table'):{'visible':np.arange(120),'candidates':[{'points':np.arange(120),'score':.95}]}
               for fid in (0,20,40)}
    new, _, _, _ = assign_unknown(base, {'0':'unknown','1':'chair'}, [ob], queries, candidate_instance=candidate)
    for key in base:
        np.testing.assert_array_equal(new[key], base[key])
    ob['votes']['quality_canonical'].update(labels=['chair']*3, name='chair')
    queries = {(fid,'chair'):value for (fid,_),value in queries.items()}
    new, _, _, _ = assign_unknown(base, {'0':'unknown','1':'chair'}, [ob], queries, candidate_instance=candidate)
    assert np.all(new['semantic'] == 1) and np.all(new['instance'] == 1)


def test_ambiguous_objects_and_single_view_unknown_points_are_not_candidates():
    base = dict(semantic=np.r_[np.ones(50, int), np.ones(50, int), np.zeros(60, int)],
                instance=np.r_[np.ones(50, int), np.full(50,2), np.zeros(60, int)], confidence=np.zeros(160))
    geometry = dict(object_id=np.ones(160, int), support_views=np.full(160,3))
    candidate, report = build_candidates(base, geometry)
    assert not candidate.any() and report['rejected'][0]['reason'] == 'conflicting_owners_or_categories'
    base['instance'][:] = 0; base['semantic'][:] = 0; geometry['support_views'][:] = 1
    candidate, _ = build_candidates(base, geometry)
    assert not candidate.any()


def test_fragment_merging_requires_same_class_and_independent_shared_masks():
    from pose_pipeline.semantic_runtime.fragment_merge import merge_fragments
    instance=np.r_[np.ones(100,int),np.full(100,2)]
    semantic=np.ones(200,int)
    frames=observations(200)
    result,audit=merge_fragments(instance,semantic,frames)
    assert np.all(result==1) and len(audit['merges'])==1
    unchanged,_=merge_fragments(instance,semantic,frames[:2])
    np.testing.assert_array_equal(unchanged,instance)
    semantic[100:]=2
    unchanged,_=merge_fragments(instance,semantic,frames)
    np.testing.assert_array_equal(unchanged,instance)
    semantic[:]=1
    for f in frames:f['mask_ids'][100:]=2
    unchanged,_=merge_fragments(instance,semantic,frames)
    np.testing.assert_array_equal(unchanged,instance)


def test_fragment_component_needs_joint_evidence_not_just_a_positive_chain():
    from pose_pipeline.semantic_runtime.fragment_merge import merge_fragments
    instance=np.repeat([1,2,3],40);semantic=np.ones(120,int)
    frames=observations(120)
    for f in frames:f['mask_ids'][80:]=0
    extra=observations(120)
    for f in extra:f['frame_id']+=60;f['mask_ids'][:40]=0
    result,audit=merge_fragments(instance,semantic,frames+extra)
    assert result[0]==result[40]!=result[80]
    assert audit['component_vetoes']==1
    assert np.array_equal(instance,np.repeat([1,2,3],40))
    with pytest.raises(ValueError,match='duplicate'):
        merge_fragments(instance,semantic,frames+frames)


def test_no_work_refinement_does_not_load_models(tmp_path, monkeypatch):
    from pose_pipeline.semantic_runtime.refinement import observations as module, grounding
    monkeypatch.setattr(module, 'R', tmp_path)
    monkeypatch.setattr(grounding, 'R', tmp_path)
    (tmp_path/'objects'/'fixture').mkdir(parents=True)
    (tmp_path/'objects'/'fixture'/'CROP_INDEX.json').write_text('[]')
    (tmp_path/'runtime.json').write_text('{}')
    (tmp_path/'DECISIONS.json').write_text('[]')
    module.name_all(['fixture'])
    grounding.ground()
    assert json.loads((tmp_path/'naming'/'COMPLETE.json').read_text())['actual_requests'] == 0
    assert json.loads((tmp_path/'grounding'/'COMPLETE.json').read_text())['queries'] == 0


def test_legacy_production_fusion_still_matches_reference_algorithm():
    from pose_pipeline.sam3_multiview import fuse_instances as reference
    frames = observations(200)
    frames[0]['mask_ids'][100:] = 2
    semantic = np.r_[np.ones(80,int),np.full(80,2),np.zeros(40,int)]
    expected, _ = reference(200, frames, semantic)
    actual, _ = fuse_instances(200, frames, semantic, geometry_out={})
    np.testing.assert_array_equal(actual, expected)


def test_bridge_binds_candidates_to_the_exact_base_and_input_lock(tmp_path):
    from pose_pipeline.live_semantic import prepare_refinement
    from pose_pipeline.semantic_runtime.common import sha
    from pose_pipeline.semantic_runtime.refinement.__main__ import validate_workspace
    output=tmp_path/'run';(output/'mapping').mkdir(parents=True);(output/'fused').mkdir()
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    runtime=tmp_path/'runtime.json';runtime.write_text('{}')
    trajectory=tmp_path/'trajectory.json';trajectory.write_text('{}')
    (output/'mapping/mapping_result.json').write_text(json.dumps({'trajectory':str(trajectory)}))
    for name in ('target.npz','map_labels.npz','candidates.npz'):
        (output/'fused'/name).write_bytes(name.encode())
    (output/'fused/classes.json').write_text('{"0":"unknown"}')
    (output/'fused/CANDIDATES.json').write_text(json.dumps({
        'labels_sha256':sha(output/'fused/map_labels.npz'),'candidates_sha256':sha(output/'fused/candidates.npz')}))
    root=prepare_refinement(output,manifest,runtime)
    _,lock=validate_workspace(root)
    assert lock['inputs/capture/candidates.npz']==sha(output/'fused/candidates.npz')
    assert (root/'inputs/capture/base.npz').read_bytes()==b'map_labels.npz'


def test_refinement_same_class_unknown_points_are_still_grounded(tmp_path,monkeypatch):
    from pose_pipeline.semantic_runtime.refinement import grounding
    monkeypatch.setattr(grounding,'R',tmp_path)
    ob=dict(scene='fixture',instance_id=1,semantic_id=1,unknown_points=60,original_name='chair',
            votes={'quality_canonical':{'name':'chair','frames':[0,20]}},validation=[])
    (tmp_path/'DECISIONS.json').write_text(json.dumps([ob]))
    tasks=grounding.query_plan()
    assert set(tasks)=={('fixture',0),('fixture',20)}
    assert all('chair' in names for names in tasks.values())
