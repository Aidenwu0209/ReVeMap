import numpy as np

from pose_pipeline.semantic_runtime import multiview
from pose_pipeline.semantic_runtime.object_candidates import build_candidates
from pose_pipeline.semantic_runtime.object_candidates import extend_candidates_with_pure_masks
from scipy.sparse import csr_matrix
import pytest


def candidate_fixture():
    base = dict(semantic=np.r_[np.ones(40, int), np.full(40, 2), np.zeros(60, int)],
                instance=np.r_[np.ones(40, int), np.full(40, 2), np.zeros(60, int)],
                confidence=np.zeros(140))
    geometry = dict(object_id=np.ones(140, int), support_views=np.full(140, 2))
    mask = np.r_[np.ones(40), np.zeros(40), np.ones(60)]
    membership = csr_matrix(np.stack([mask, mask, mask]).astype(np.int32))
    metadata = dict(eligible_nodes=np.arange(3), valid_nodes=np.ones(3, bool),
                    origins=np.arange(3), group_ids=np.ones(3, int))
    return base, geometry, membership, metadata, np.zeros(140, int), [1], {'0': 'unknown', '1': 'cabinet', '2': 'wall'}


def production_candidate(*args):
    base, geometry, membership, metadata, existing, _, classes = args
    return extend_candidates_with_pure_masks(base, geometry, dict(membership=membership, **metadata), existing, classes)


@pytest.mark.parametrize('implementation', [production_candidate])
def test_candidate_probe_produces_hints_only_and_keeps_all_input_arrays(implementation):
    args = candidate_fixture()
    original = {k: v.copy() for k, v in args[0].items()}
    result, audit = implementation(*args)
    assert np.all(result[80:] == 1) and not result[:80].any()
    assert audit['new_candidate_points'] == 60
    for k in original:
        np.testing.assert_array_equal(original[k], args[0][k])
    assert not args[4].any()


@pytest.mark.parametrize('failure', ['duplicate_frame', 'different_group', 'mixed_class', 'few_anchors',
                                     'small_target', 'no_geometry_support', 'opposing_wall_owner', 'opposing_other_group_owner'])
@pytest.mark.parametrize('implementation', [production_candidate])
def test_candidate_probe_rejects_unreliable_direct_evidence(failure, implementation):
    args = list(candidate_fixture())
    if failure == 'duplicate_frame':
        args[3]['origins'][:] = 0
    elif failure == 'different_group':
        args[3]['group_ids'][:2] = 2
    elif failure == 'mixed_class':
        args[2] = csr_matrix(np.ones((3, 140), np.int32))
    elif failure == 'few_anchors':
        x = args[2].toarray(); x[:, :15] = 0; args[2] = csr_matrix(x)
    elif failure == 'small_target':
        args[1]['object_id'][129:] = 0
        args[1]['support_views'][129:] = 0
    elif failure == 'no_geometry_support':
        args[1]['support_views'][:] = 1
    else:
        x = args[2].toarray(); x[2, :40] = 0; x[2, 40:80] = 1; args[2] = csr_matrix(x)
        if failure == 'opposing_other_group_owner':
            args[3]['group_ids'][2] = 2
    result, _ = implementation(*args)
    assert not result.any()


@pytest.mark.parametrize('implementation', [production_candidate])
def test_filtered_mask_cannot_supply_a_candidate_vote(implementation):
    args = list(candidate_fixture())
    args[3]['valid_nodes'][0] = False
    with pytest.raises(ValueError, match='unfiltered'):
        implementation(*args)


@pytest.mark.parametrize('frames', [[], [dict(frame_id=0, point_ids=np.arange(80), mask_ids=np.ones(80, int),
                                            semantic=np.ones(80, int), confidence=np.full(80, .95), interior=np.ones(80, bool))]])
def test_candidate_evidence_has_complete_empty_schema_without_eligible_groups(frames):
    evidence, geometry = {}, {}
    output, _ = multiview.fuse_instances(80, frames, np.zeros(80, int),
                                       geometry_out=geometry, candidate_evidence_out=evidence)
    assert set(evidence) == {'membership', 'origins', 'valid_nodes', 'eligible_nodes', 'group_ids'}
    assert evidence['membership'].shape[1] == 80
    assert not len(evidence['eligible_nodes']) and not evidence['group_ids'].any()
    assert not output.any()
    base = dict(semantic=np.zeros(80, int), instance=np.zeros(80, int), confidence=np.zeros(80))
    result, audit = extend_candidates_with_pure_masks(base, geometry, evidence, np.zeros(80, int), {'0': 'unknown'})
    assert not result.any() and audit['new_candidate_points'] == 0


@pytest.mark.parametrize('invalid', ['duplicate_eligible', 'negative_origin', 'zero_group', 'invalid_valid_dtype',
                                    'wrong_width', 'fractional_membership'])
def test_production_candidate_evidence_rejects_malformed_schema(invalid):
    args = list(candidate_fixture())
    if invalid == 'duplicate_eligible':args[3]['eligible_nodes'][:] = 0
    elif invalid == 'negative_origin':args[3]['origins'][0] = -1
    elif invalid == 'zero_group':args[3]['group_ids'][0] = 0
    elif invalid == 'invalid_valid_dtype':args[3]['valid_nodes'] = np.ones(3, int)
    elif invalid == 'wrong_width':args[2] = csr_matrix(np.ones((3, 141), np.int32))
    else:args[2] = args[2].astype(float) * .5
    with pytest.raises(ValueError):production_candidate(*args)


def test_unsorted_unique_csr_membership_matches_sorted_input_without_mutation():
    args = list(candidate_fixture())
    original = args[2]
    indices = np.concatenate([original.indices[lo:hi][::-1] for lo, hi in zip(original.indptr[:-1], original.indptr[1:])])
    args[2] = csr_matrix((original.data.copy(), indices.copy(), original.indptr.copy()), shape=original.shape)
    assert not args[2].has_sorted_indices
    canonical_args = list(args)
    canonical_args[2] = original
    expected, _ = production_candidate(*canonical_args)
    actual, _ = production_candidate(*args)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(args[2].indices, indices)
    assert not args[2].has_sorted_indices
