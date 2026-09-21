import numpy as np
import pytest
from pose_pipeline.sam3_loss_audit import audit_losses, REASONS
from pose_pipeline.sam3_multiview import ConsensusConfig, fuse_instances
from pose_pipeline.sam3_unknown import raw_mask_partition, fuse_unknown, UnknownConfig


def frame(fid, masks, score=.95):
    masks = np.asarray(masks, np.int32)
    return dict(frame_id=fid, point_ids=np.arange(len(masks)), mask_ids=masks,
                semantic=(masks > 0).astype(np.int32), confidence=np.full(len(masks), score),
                interior=np.ones(len(masks), bool))


def test_loss_audit_reproduces_output_and_accounts_all_points():
    frames = [frame(i, [1, 1, 1, 1, 0, 2]) for i in range(3)]
    semantic = np.array([1, 1, 1, 0, 1, 2])
    cfg = ConsensusConfig(min_mask_points=2, min_output_points=2)
    expected, _ = fuse_instances(6, frames, semantic, cfg)
    output, reason, audit = audit_losses(6, frames, semantic, np.ones(6, int), cfg, expected)
    assert np.array_equal(output, expected)
    assert [REASONS[i] for i in reason] == ['retained_instance']*3 + [
        'owned_point_semantic_unknown', 'no_projected_mask', 'mask_below_min_points']
    assert sum(audit['loss_reasons'].values()) == audit['lost_points'] == 3


@pytest.mark.parametrize('score,frames,expected', [(.79, 3, 'frame_eligible_groups_low_score'),
                                                (.95, 2, 'all_groups_insufficient_frames')])
def test_group_rejections_are_point_not_group_counts(score, frames, expected):
    _, reason, audit = audit_losses(4, [frame(i, [1]*4, score) for i in range(frames)],
        np.ones(4, int), np.ones(4, int), ConsensusConfig(min_mask_points=1, min_output_points=1))
    assert all(REASONS[i] == expected for i in reason)
    assert audit['loss_reasons'][expected] == 4


def test_empty_observations_count_no_mask():
    _, reasons, audit = audit_losses(2, [frame(0, [0, 0])], np.ones(2, int), np.ones(2, int))
    assert list(reasons) == [1, 1]
    assert audit['lost_points'] == 2


def test_raw_duplicate_prompts_are_one_mask_and_do_not_abstain():
    masks = np.zeros((2, 7, 7), bool); masks[:, 1:6, 1:6] = True
    owner, _, _, audit = raw_mask_partition(np.packbits(masks.reshape(2, -1), axis=1), (7, 7), [.9, .9])
    assert audit['kept_masks'] == 1 and len(audit['duplicates']) == 1
    assert np.count_nonzero(owner) == 25


def test_nested_different_objects_are_not_suppressed_as_duplicates():
    masks = np.zeros((2, 9, 9), bool); masks[0, 1:8, 1:8] = True; masks[1, 3:6, 3:6] = True
    owner, _, _, audit = raw_mask_partition(np.packbits(masks.reshape(2, -1), axis=1), (9, 9), [.9, .92])
    assert audit['kept_masks'] == 2
    assert audit['ambiguous_overlap_pixels'] == 9
    assert not owner[3:6, 3:6].any()


def test_unknown_contract_preserves_semantics_and_object_identity():
    semantic = np.array([0, 0, 7, 14]); original = semantic.copy()
    output, audit = fuse_unknown(4, [frame(i, [1]*4) for i in range(3)], semantic,
                                UnknownConfig(min_mask_points=1, min_output_points=1))
    assert np.all(output == 1)
    assert np.array_equal(semantic, original)
    assert audit['objects'][0]['semantic_histogram'] == {'0': 2, '7': 1, '14': 1}
    assert audit['owned_unknown_semantic_points'] == 2


def test_duplicate_frame_cannot_create_unknown_object_evidence():
    with pytest.raises(ValueError, match='duplicate frame'):
        fuse_unknown(4, [frame(0, [1]*4)]*3, np.zeros(4, int),
                     UnknownConfig(min_mask_points=1, min_output_points=1))
