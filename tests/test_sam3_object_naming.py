import numpy as np
import pytest
from pose_pipeline.sam3_object_naming import name_objects, NamingConfig


def frame(fid, labels, score=.95, points=None):
    labels = np.asarray(labels, np.int32)
    return {'frame_id': fid, 'point_ids': np.arange(len(labels)) if points is None else np.asarray(points),
            'semantic': labels, 'confidence': np.full(len(labels), score),
            'interior': np.ones(len(labels), bool)}


def test_names_without_changing_unknown_point_semantics_or_ownership():
    instances = np.ones(40, np.int32); before = instances.copy()
    fs = [frame(i, [7]*40) for i in range(3)]
    objects, audit = name_objects(instances, fs)
    assert objects[0]['semantic_id'] == 7 and objects[0]['eligible_frames'] == 3
    assert objects[0]['class_frame_votes'] == {'7': 3}
    np.testing.assert_array_equal(instances, before)
    assert not audit['semantic_map_modified'] and not audit['instance_map_modified']


def test_two_eligible_frames_do_not_name_object():
    objects, _ = name_objects(np.ones(40, int), [frame(i, [7]*40) for i in range(2)])
    assert objects[0]['semantic_id'] == 0
    assert objects[0]['naming_status'] == 'insufficient_eligible_frames'


def test_single_dense_frame_cannot_outvote_multiple_independent_frames():
    fs = [frame(0, [7]*1000)] + [frame(i, [34]*30) for i in range(1,5)]
    objects, _ = name_objects(np.ones(1000, int), fs)
    assert objects[0]['semantic_id'] == 34
    assert objects[0]['coarse_class_preserved']
    assert objects[0]['class_frame_votes'] == {'7': 1, '34': 4}


def test_coarse_and_fine_votes_are_not_remapped_or_pooled():
    fs = [frame(i, [c]*40) for i,c in enumerate([7,7,34,34])]
    objects, _ = name_objects(np.ones(40, int), fs)
    assert objects[0]['semantic_id'] == 0
    assert objects[0]['naming_status'] == 'inconsistent_frame_votes'


def test_mixed_pixel_class_frame_abstains():
    fs = [frame(i, [7]*20+[14]*20) for i in range(4)]
    objects, _ = name_objects(np.ones(40, int), fs)
    assert objects[0]['semantic_id'] == 0
    assert objects[0]['eligible_frames'] == 0
    assert all(f['reason'] == 'mixed_frame_classes' for f in objects[0]['frame_evidence'])


def test_qualifying_points_need_interior_and_confidence_and_known_semantic():
    fs = [frame(i, [7]*40, .79) for i in range(3)]
    assert name_objects(np.ones(40, int), fs)[0][0]['semantic_id'] == 0
    fs = [frame(i, [7]*40) for i in range(3)]
    for f in fs: f['interior'][:11] = False
    assert name_objects(np.ones(40, int), fs)[0][0]['eligible_frames'] == 0


def test_duplicate_original_frame_and_points_are_rejected():
    with pytest.raises(ValueError, match='duplicate original frame'):
        name_objects(np.ones(40, int), [frame(0, [7]*40)]*3)
    with pytest.raises(ValueError, match='invalid unique'):
        name_objects(np.ones(40, int), [frame(0, [7]*40, points=[0]*40)])


def test_no_objects_and_no_frames_stay_unknown_without_error():
    objects, audit = name_objects(np.zeros(40, int), [])
    assert objects == [] and audit['objects'] == 0
    objects, _ = name_objects(np.ones(40, int), [])
    assert objects[0]['semantic_id'] == 0
