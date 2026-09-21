"""Safety and causal fixtures for guide-assisted lost-owner recovery."""
import numpy as np
import pytest
from pose_pipeline.sam3_guided import recover_instances

CFG = dict(min_mask_points=2, min_output_points=4)

def frame(fid, masks, semantic=None, confidence=None, interior=None):
    masks = np.asarray(masks, np.int32)
    return dict(frame_id=fid, point_ids=np.arange(len(masks), dtype=np.int32), mask_ids=masks,
                semantic=np.ones(len(masks), np.int32) if semantic is None else np.asarray(semantic, np.int32),
                confidence=np.full(len(masks), .9, np.float32) if confidence is None else np.asarray(confidence, np.float32),
                interior=np.ones(len(masks), bool) if interior is None else np.asarray(interior, bool))

def test_recovery_has_exact_distinct_frame_provenance_and_preserves_anchor():
    sem=np.ones(10, np.int32); guides=sem.copy(); current=np.array([3]*4+[0]*6, np.int32)
    result,audit,p=recover_instances(10,[frame(0,[1]*10),frame(20,[7]*10)],sem,guides,current,CFG)
    assert np.all(result==3) and audit['recovered_points']==6
    assert np.array_equal(p['point_id'],np.arange(4,10))
    assert np.all(p['distinct_support_frames']==2)
    for point in p['point_id']:
        where=p['support_point_id']==point
        assert set(p['support_frame_id'][where])=={0,20}
        assert set(p['support_mask_id'][where])=={1,7}
    assert np.array_equal(current,np.array([3]*4+[0]*6))

def test_adjacent_objects_never_merge_and_ambiguous_joint_masks_abstain():
    sem=np.ones(12,np.int32); guides=np.array([1]*6+[2]*6,np.int32); current=np.zeros(12,np.int32)
    result,audit,_=recover_instances(12,[frame(0,[1]*12),frame(20,[1]*12)],sem,guides,current,CFG)
    assert not result.any() and audit['recovered_points']==0
    result,_,_=recover_instances(12,[frame(0,[1]*6+[2]*6),frame(20,[5]*6+[7]*6)],sem,guides,current,CFG)
    assert np.all(result[:6]==result[0]) and np.all(result[6:]==result[6])
    assert result[0]>0 and result[6]>0 and result[0]!=result[6]

def test_mixed_baseline_guide_vetoed_by_repeated_visible_split():
    sem=np.ones(10,np.int32); current=np.zeros(10,np.int32)
    frames=[frame(0,[1]*10),frame(20,[2]*10),frame(40,[1]*5+[2]*5),frame(60,[8]*5+[9]*5)]
    result,audit,_=recover_instances(10,frames,sem,sem,current,CFG)
    assert not result.any()
    assert audit['guides'][0]['status']=='repeated_observed_split'
    assert audit['guides'][0]['split_frames']==[40,60]

def test_unknown_mask_or_unobserved_points_abstain():
    sem=np.ones(10,np.int32); current=np.zeros(10,np.int32)
    # Two views see a consistent mask on 8 points, 2 remain unknown.
    frames=[frame(0,[1]*8+[0]*2),frame(20,[2]*8+[0]*2)]
    result,_,p=recover_instances(10,frames,sem,sem,current,CFG)
    assert np.all(result[:8]>0) and np.all(result[8:]==0)
    assert np.array_equal(p['point_id'],np.arange(8))
    result,_,_=recover_instances(10,[frame(0,[1]*10,semantic=[0]*10),frame(20,[2]*10,semantic=[0]*10)],sem,sem,current,CFG)
    assert not result.any()

def test_no_baseline_guide_does_not_invent_an_object():
    sem=np.ones(10,np.int32); empty=np.zeros(10,np.int32)
    result,_,_=recover_instances(10,[frame(0,[1]*10),frame(20,[2]*10)],sem,empty,empty,CFG)
    assert not result.any()

def test_multiple_existing_instances_never_merge_or_fill_between_them():
    sem=np.ones(10,np.int32); current=np.array([3]*4+[4]*4+[0]*2,np.int32)
    result,audit,_=recover_instances(10,[frame(0,[1]*10),frame(20,[2]*10)],sem,sem,current,CFG)
    assert np.array_equal(result,current)
    assert audit['guides'][0]['status']=='multiple_consensus_anchors'

def test_single_frame_not_repeated_and_one_view_point_does_not_gain_owner():
    sem=np.ones(10,np.int32); current=np.zeros(10,np.int32)
    result,_,_=recover_instances(10,[frame(0,[1]*10)],sem,sem,current,CFG)
    assert not result.any()
    with pytest.raises(ValueError,match='unique'):
        recover_instances(10,[frame(0,[1]*10),frame(0,[2]*10)],sem,sem,current,CFG)
    result,_,p=recover_instances(10,[frame(0,[1]*10),frame(20,[2]*8+[0]*2)],sem,sem,current,CFG)
    assert np.all(result[:8]>0) and np.all(result[8:]==0)
    assert np.all(p['distinct_support_frames']==2)

def test_low_confidence_never_recovered_and_invalid_input_fails():
    sem=np.ones(10,np.int32); current=np.zeros(10,np.int32)
    result,_,_=recover_instances(10,[frame(0,[1]*10,confidence=[.7]*10),frame(20,[2]*10,confidence=[.7]*10)],sem,sem,current,CFG)
    assert not result.any()
    with pytest.raises(ValueError,match='dominance'):
        recover_instances(10,[],sem,sem,current,dict(mask_guide_purity=.5))
    with pytest.raises(ValueError,match='invalid projected'):
        recover_instances(10,[frame(0,[1]*10,confidence=[float('nan')]*10)],sem,sem,current,CFG)


def test_previous_undersegmentation_mask_cannot_supply_recovery_support():
    sem=np.ones(10,np.int32); current=np.zeros(10,np.int32)
    frames=[frame(0,[1]*10),frame(20,[2]*10)]
    result,audit,p=recover_instances(10,frames,sem,sem,current,CFG,blocked_masks=[(0,1)])
    assert not result.any() and not len(p['point_id'])
    assert audit['guides'][0]['blocked_mask_frames']==[0]
    assert audit['prior_filtered_mask_nodes']==[[0,1]]
    assert audit['guides'][0]['status']=='insufficient_compatible_frames'
