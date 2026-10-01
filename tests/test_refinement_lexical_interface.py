"""Class 13's historical typo must not block unchanged grounding/assignment gates."""
import numpy as np
import pytest

from pose_pipeline.semantic_runtime.refinement.assignment import assign_unknown
from pose_pipeline.semantic_runtime.refinement.grounding import query_plan
from pose_pipeline.semantic_runtime.refinement.normalization import (
    RULESET_VERSION, canonicalize_name, export_rules, normalize_name,
)


def fixture(name='refrigerator', score=.9):
    # 68 candidate points have direct multi-view support; 32 stay unseen.
    base = dict(semantic=np.r_[np.full(60, 13, int), np.zeros(100, int)],
                instance=np.r_[np.full(60, 48, int), np.zeros(100, int)],
                confidence=np.r_[np.full(60, .9), np.zeros(100)])
    candidate = np.r_[np.zeros(60, int), np.full(100, 48, int)]
    classes = {'0': 'unknown', '13': 'refridgerator', '4': 'chair', '9': 'door', '10': 'cabinet'}
    observation = dict(scene='capture', instance_id=48, semantic_id=13,
                       original_name='refridgerator', unknown_points=100, validation=[],
                       votes={'quality_canonical': dict(frames=[0,20,40], labels=[name]*3, name=name)})
    queries = {(fid,name): dict(visible=np.arange(160),
                               candidates=[dict(points=np.arange(128), score=score)])
               for fid in (0,20,40)}
    return base, classes, observation, queries, candidate


def test_spelling_repair_preserves_raw_fine_and_explicit_lexical_provenance():
    result = normalize_name('refridgerator')
    assert result['raw_name'] == result['fine_class'] == 'refridgerator'
    assert result['canonical_name'] == result['standard_class'] == 'refrigerator'
    assert result['transformations'] == [dict(kind='lexical', **{'from':'refridgerator', 'to':'refrigerator'})]
    assert result['ruleset_version'] == RULESET_VERSION == 'english_object_names_v2_20261001'
    assert export_rules()['spelling_corrections'] == {'refridgerator':'refrigerator'}
    assert len({canonicalize_name(n) for n in ('refrigerator','chair','door','cabinet')}) == 4


def test_class_typo_reaches_grounding_and_direct_assignment_without_renumbering():
    base, classes, ob, queries, candidate = fixture()
    before = {key:value.copy() for key,value in base.items()}
    tasks = query_plan([ob])
    assert set(tasks) == {('capture',fid) for fid in (0,20,40)}
    assert all(set(names) == {'refrigerator'} for names in tasks.values())
    # Use exactly the canonical queries scheduled across the module boundary.
    scheduled = {(fid,name):queries[fid,name] for (_,fid),names in tasks.items() for name in names}
    result, dictionary, _, audit = assign_unknown(base, classes, [ob], scheduled,
                                                  candidate_instance=candidate)
    assert np.all(result['semantic'][60:128] == 13)
    assert np.all(result['instance'][60:128] == 48)
    assert not result['semantic'][128:].any() and not result['instance'][128:].any()
    assert dictionary == classes
    assert audit[0]['strict_points'] == 68
    for key in base:
        np.testing.assert_array_equal(base[key], before[key])
        np.testing.assert_array_equal(result[key][:60], before[key][:60])
    np.testing.assert_array_equal(result['confidence'], base['confidence'])


@pytest.mark.parametrize('incompatible', ['chair','door','cabinet'])
def test_exact_spelling_repair_does_not_allow_category_replacement(incompatible):
    base, classes, ob, queries, candidate = fixture(incompatible)
    assert query_plan([ob]) == {}
    # Assignment must also reject even if an incompatible query is supplied.
    result, dictionary, _, _ = assign_unknown(base, classes, [ob], queries,
                                               candidate_instance=candidate)
    for key in base:
        np.testing.assert_array_equal(result[key], base[key])
    assert dictionary == classes


def test_spelling_repair_does_not_relax_mask_score_requirement():
    base, classes, ob, queries, candidate = fixture(score=.49)
    assert query_plan([ob])
    result, _, _, _ = assign_unknown(base, classes, [ob], queries, candidate_instance=candidate)
    for key in base:
        np.testing.assert_array_equal(result[key], base[key])
