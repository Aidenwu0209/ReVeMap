"""Sparse singleton signatures must preserve the component merge contract."""
import numpy as np

from pose_pipeline.semantic_runtime.fragment_merge import merge_fragments


def test_sparse_singletons_keep_empty_rows_and_local_mask_ids_separate():
    instance = np.repeat([3, 17, 400, 901], 40)
    semantic = np.repeat([1, 2, 1, 1], 40)
    frames = []
    for fid in range(3):
        # The first and third IDs are same-class fragments; an empty CSR row
        # between them must neither borrow counts nor shift a winning mask.
        points = np.r_[0:40, 80:160]
        masks = np.r_[np.full(40, 900), np.full(40, 900), np.full(40, 8)]
        frames.append(dict(frame_id=fid, point_ids=points, mask_ids=masks))
    result, audit = merge_fragments(instance, semantic, frames)
    np.testing.assert_array_equal(result, np.repeat([3, 17, 3, 901], 40))
    assert audit['merges'] == [dict(left_instance=3, right_instance=400,
                                   support_frames=3, separation_frames=0)]


def test_mixed_semantics_abstain_even_when_shared_masks_are_confident():
    instance = np.repeat([10, 20, 30], 40)
    semantic = np.ones(120, int)
    semantic[41] = 0
    frames = [dict(frame_id=fid, point_ids=np.arange(120), mask_ids=np.ones(120, int))
              for fid in range(3)]
    result, _ = merge_fragments(instance, semantic, frames)
    np.testing.assert_array_equal(result, np.repeat([10, 20, 10], 40))


def test_abstaining_unlabelled_points_remain_in_containment_denominator():
    instance = np.repeat([1, 2], 100)
    semantic = np.ones(200, int)
    # Removing unsafe boundary observations must not reduce the denominator.
    frames = [dict(frame_id=fid, point_ids=np.arange(200),
                   mask_ids=np.tile(np.r_[np.ones(79, int), np.zeros(21, int)], 2))
              for fid in range(3)]
    result, audit = merge_fragments(instance, semantic, frames)
    np.testing.assert_array_equal(result, instance)
    assert not audit['merges']


def test_input_order_does_not_change_deterministic_id_representative():
    instance = np.repeat([100, 3, 40], 40)
    semantic = np.ones(120, int)
    frames = [dict(frame_id=fid, point_ids=np.arange(120)[::-1], mask_ids=np.ones(120, int))
              for fid in (9, 1, 5)]
    result, audit = merge_fragments(instance, semantic, frames)
    assert np.all(result == 3)
    assert len(audit['merges']) == 2


def test_empty_owner_set_and_unknown_categories_do_not_create_instances():
    for instance, semantic in ((np.zeros(120, int), np.ones(120, int)),
                               (np.ones(120, int), np.zeros(120, int)),
                               (np.empty(0, int), np.empty(0, int))):
        frames = [dict(frame_id=fid, point_ids=np.arange(len(instance)),
                       mask_ids=np.ones(len(instance), int)) for fid in range(3)]
        result, audit = merge_fragments(instance, semantic, frames)
        np.testing.assert_array_equal(result, instance)
        assert not audit['merges']


def test_unsafe_shared_mask_boundary_can_be_removed_without_inventing_a_vote():
    instance = np.repeat([1, 2], 40)
    semantic = np.ones(80, int)
    frames = [dict(frame_id=fid, point_ids=np.arange(80), mask_ids=np.ones(80, int),
                   interior=np.r_[np.ones(40, bool), np.zeros(40, bool)],
                   confidence=np.full(80, .95), semantic=semantic.copy()) for fid in range(3)]
    merged, _ = merge_fragments(instance, semantic, frames)
    assert np.all(merged == 1)
    safe = [{**f, 'mask_ids': np.where(f['interior'] & (f['confidence'] >= .5)
                                      & (f['semantic'] > 0), f['mask_ids'], 0)} for f in frames]
    result, audit = merge_fragments(instance, semantic, safe)
    np.testing.assert_array_equal(result, instance)
    assert not audit['merges']
