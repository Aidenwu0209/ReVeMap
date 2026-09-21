import numpy as np
import pytest
from pose_pipeline.sam3_fusion import visible_map_pixels, PixelClaims, MapVotes, GeometricInstances


def test_projection_uses_inverse_pose_and_rejects_occluded_or_missing_depth():
    t = np.eye(4); t[:3, 3] = [2, 0, 0]
    xyz = np.array([[2, 0, 1], [2, 0, 2], [3, 0, 1], [2, 0, -1]])
    depth = np.array([[1., 0.], [0., 0.]])
    ids, v, u = visible_map_pixels(xyz, t, (1, 1, 0, 0), depth)
    assert ids.tolist() == [0]
    assert u.tolist() == v.tolist() == [0]


def test_competing_concepts_abstain_and_same_class_masks_do_not_conflict():
    p = PixelClaims((1, 3), 3)
    p.add(1, [[1, 1, 0]], .8)
    p.add(1, [[1, 0, 0]], .85)
    p.add(2, [[0, 1, 1]], .83)
    sem, inst, conf = p.finalize()
    assert sem.tolist() == [[1, 0, 2]]
    assert inst[0, 0] == 2 and inst[0, 1] == 0
    assert conf[0, 1] == 0


def test_votes_require_distinct_views_and_keep_conflicts_unknown():
    v = MapVotes(4, 3)
    v.add(1, [0, 1, 2], [1, 1, 0], [.8, .9, 0])
    with pytest.raises(ValueError, match='duplicate frame'):
        v.add(1, [0], [1], [.8])
    assert not v.finalize()[0].any()
    v.add(2, [0, 1, 2], [1, 2, 0], [.8, .9, 0])
    assert v.finalize()[0].tolist() == [1, 0, 0, 0]
    assert v.visible_count.tolist() == [2, 2, 2, 0]


def test_instances_do_not_merge_disjoint_objects_of_same_class():
    tracker = GeometricInstances(min_points=2)
    ids = np.arange(6); masks = np.array([1, 1, 1, 2, 2, 2]); cats = np.ones(6, int)
    tracker.add(0, ids, masks, cats)
    assert not tracker.finalize(cats).any()
    tracker.add(1, ids, masks, cats)
    inst = tracker.finalize(cats)
    assert len(np.unique(inst)) == 2 and np.all(inst > 0)
    assert inst[0] != inst[-1]


def test_competing_instance_owners_abstain():
    tracker = GeometricInstances(min_points=2)
    cats = np.ones(5, int)
    tracker.add(0, np.array([0, 1, 2, 3]), np.array([1, 1, 2, 2]), np.ones(4, int))
    tracker.add(1, np.array([0, 1, 2, 3]), np.array([1, 1, 2, 2]), np.ones(4, int))
    # One merged observation cannot collapse both established tracks.
    assert tracker.add(2, np.arange(4), np.ones(4, int), np.ones(4, int)) == {}
    inst = tracker.finalize(cats)
    assert inst[0] != inst[2] and inst[4] == 0


def test_checkpoint_absent_or_wrong_digest_fails_before_model_import(tmp_path):
    from pose_pipeline.sam3_mapping import load_model
    with pytest.raises(ValueError, match='existing local checkpoint'):
        load_model(tmp_path/'missing.pt', '0'*64)
    path = tmp_path/'wrong.pt'; path.write_bytes(b'not model weights')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        load_model(path, '0'*64)


def test_export_preserves_geometry_and_rejects_reordered_map(tmp_path):
    import json, hashlib
    from plyfile import PlyData, PlyElement
    from pose_pipeline.sam3_export import export
    v = np.zeros(3, dtype=[(k, 'f8') for k in ['x','y','z']] + [(k,'u1') for k in ['red','green','blue']])
    v['x'] = [1.123456789, 2, 3]; v['red'] = [20, 30, 40]
    baseline = tmp_path/'map.ply'; PlyData([PlyElement.describe(v, 'vertex')]).write(str(baseline))
    xyz = np.stack([v[k] for k in ['x','y','z']], axis=1)
    labels = tmp_path/'labels.npz'; np.savez(labels, semantic=[1,0,2], instance=[1,0,2], confidence=[1,0,.8])
    result = tmp_path/'result.json'
    d = {'status':'completed','geometry_xyz_sha256':hashlib.sha256(xyz.tobytes()).hexdigest(),
         'sga_inference_executed':False,'complete_full_sequence':False}
    result.write_text(json.dumps(d))
    receipt = export(baseline, labels, result, tmp_path/'out')
    assert all(receipt['original_vertex_properties_preserved'].values())
    v = v[::-1].copy(); PlyData([PlyElement.describe(v,'vertex')]).write(str(baseline))
    with pytest.raises(ValueError, match='exact baseline point order'):
        export(baseline, labels, result, tmp_path/'out2')
