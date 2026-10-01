"""Counterexamples for filtering before ranking and retaining established classes."""
from copy import deepcopy

import numpy as np
import pytest

from pose_pipeline.semantic_runtime.refinement.assignment import assign_unknown
from pose_pipeline.semantic_runtime.refinement.grounding import query_plan
from pose_pipeline.semantic_runtime.refinement.observations import object_identity


def fixture(masks, frames=(0, 20, 40)):
    n = 180
    base = dict(semantic=np.zeros(n, np.int32), instance=np.zeros(n, np.int32),
                confidence=np.zeros(n, np.float32))
    base['instance'][:100] = 1
    observation = dict(instance_id=1, semantic_id=0, votes={'quality_canonical': {
        'frames': list(frames), 'labels': ['cabinet'] * len(frames), 'name': 'cabinet'}})
    queries = {(fid, 'cabinet'): dict(visible=np.arange(n), candidates=deepcopy(masks))
               for fid in frames}
    return base, {'0': 'unknown'}, [observation], queries


@pytest.mark.parametrize('rejected', [
    dict(points=np.arange(100), score=.49),  # Best overlap, failed score.
    dict(points=np.arange(140), score=.95),  # Best product, failed purity.
])
def test_rejected_best_mask_cannot_hide_a_strict_eligible_mask(rejected):
    args = fixture([rejected, dict(points=np.arange(60), score=.8)])
    new, _, _, audit = assign_unknown(*args)
    assert np.count_nonzero(new['semantic']) == 60
    assert np.all(new['semantic'][60:] == 0)
    assert all(v['strict'] for v in audit[0]['views'])
    assert all(v['evidence']['purity'] == 1 for v in audit[0]['views'])


def test_fragment_selection_has_its_own_gates_and_remains_opt_in():
    masks = [dict(points=np.arange(150), score=.55),
             dict(points=np.r_[np.arange(90), np.arange(100, 150)], score=.8)]
    args = fixture(masks)
    strict, _, _, _ = assign_unknown(*args)
    assert not strict['semantic'].any()
    new, _, _, audit = assign_unknown(*args, fragments=True)
    assert np.count_nonzero(new['semantic']) == 90
    assert audit[0]['fragment_points'] == 90
    assert all(v['fragment_evidence']['score'] == .8 for v in audit[0]['views'])


def test_multiple_masks_in_one_frame_still_count_once():
    args = fixture([dict(points=np.arange(100), score=.9)] * 5)
    args[3].pop((20, 'cabinet'))
    args[3].pop((40, 'cabinet'))
    new, _, _, _ = assign_unknown(*args)
    assert not new['semantic'].any()


def test_known_category_survives_a_majority_of_unknown_candidates():
    classes = {'0': 'unknown', '10': 'cabinet', '4': 'chair'}
    assert object_identity(np.r_[np.full(30, 10), np.zeros(100)], classes) == (10, 'cabinet')
    assert object_identity(np.zeros(100), classes) == (0, 'unknown')
    assert object_identity(np.r_[np.full(30, 10), np.full(20, 4)], classes) is None


def test_impossible_category_change_does_not_schedule_grounding():
    ob = dict(scene='capture', instance_id=1, semantic_id=10, unknown_points=100,
              original_name='cabinet', validation=[], votes={
                  'quality_canonical': dict(name='door', frames=[0, 20, 40])})
    assert query_plan([ob]) == {}
    ob['votes']['quality_canonical']['name'] = 'cabinet'
    assert set(query_plan([ob])) == {('capture', 0), ('capture', 20), ('capture', 40)}
