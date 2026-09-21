import numpy as np
from pose_pipeline.sam3_tracking_probe import matched_temporal_metrics, per_mask_temporal_metrics, depth_masks


def frame(fid, visible, masks):
    return {'frame_id': fid, 'visible_map_ids': np.asarray(visible),
            'projected_masks': np.asarray(masks, bool).reshape(-1, len(visible))}


def test_temporal_agreement_uses_only_common_depth_visible_points():
    # Point 1 disappears behind the camera; this must not count as flicker.
    result = matched_temporal_metrics([frame(0,[1,2,3],[[1,1,0]]),frame(1,[2,3,4],[[1,0,1]])])
    assert result['mean_foreground_iou'] == 1
    assert result['observed_map_points'] == 4
    assert result['segmented_map_points_union'] == 3


def test_empty_predictions_do_not_receive_perfect_consistency_score():
    result = matched_temporal_metrics([frame(0,[1,2],[]),frame(1,[1,2],[])])
    assert result['mean_foreground_iou'] is None
    assert result['empty_foreground_pairs_excluded'] == 1


def test_foreground_disagreement_is_symmetric_and_ids_are_not_needed():
    result = matched_temporal_metrics([frame(0,[1,2,3],[[1,1,0]]),frame(1,[1,2,3],[[0,1,1]])])
    assert result['mean_foreground_iou'] == 1/3
    assert result['pairs'][0]['foreground_disagreement'] == 2/3


def test_depth_mask_resampling_preserves_empty_shape():
    assert depth_masks(np.zeros((0,1,12,16), bool), (3,4)).shape == (0,3,4)


def test_mask_matching_detects_merge_hidden_by_perfect_union_agreement():
    a=frame(0,[1,2,3,4],[[1,1,0,0],[0,0,1,1]])
    b=frame(1,[1,2,3,4],[[1,1,1,1]])
    assert matched_temporal_metrics([a,b])['mean_foreground_iou']==1
    result=per_mask_temporal_metrics([a,b],minimum_points=1)
    assert result['merge_like_targets']==1
    assert result['unmatched_mask_observations']==1


def test_mask_matching_does_not_treat_assigned_ids_as_ground_truth():
    a=frame(0,[1,2,3,4],[[1,1,0,0],[0,0,1,1]]);a['object_ids']=[0,1]
    b=frame(1,[1,2,3,4],[[1,1,0,0],[0,0,1,1]]);b['object_ids']=[1,0]
    result=per_mask_temporal_metrics([a,b],tracked_ids=True,minimum_points=1)
    assert result['same_id_fraction_of_geometrically_matched_pairs']==0
