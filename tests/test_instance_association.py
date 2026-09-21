import numpy as np
from pose_pipeline.instance_association import compatible,object_consensus,greedy_pairs


def test_family_compatibility_does_not_merge_distinct_fine_classes():
    assert compatible(8,33) and compatible(17,33)
    assert not compatible(8,17)
    assert not compatible(5,33)


def test_object_fusion_preserves_fine_labels_and_needs_multiple_frames():
    xyz=np.arange(300,dtype=float).reshape(100,3)/100
    base={'semantic':np.zeros(100,dtype=np.int32),'instance':np.zeros(100,dtype=np.int32),'confidence':np.zeros(100)}
    base['semantic'][0]=19
    a={'track_id':1,'category':5,'points':list(range(80)),'frames':[0],'mean_point_score':.9}
    b={'track_id':1,'category':5,'points':list(range(20,100)),'frames':[20],'mean_point_score':.9}
    alone,_,_=object_consensus([[a],[b]],[],base,xyz)
    assert np.count_nonzero(alone['semantic'])==1
    merged,objects,report=object_consensus([[a],[b]],[(1,1)],base,xyz)
    assert merged['semantic'][0]==19 and np.all(merged['semantic'][1:]==5)
    assert len(objects)==1 and report['semantic_added_points']==99


def test_shared_pixels_do_not_get_two_unmerged_object_owners():
    xyz=np.zeros((100,3));base={'semantic':np.full(100,5),'instance':np.ones(100,dtype=int),'confidence':np.ones(100)}
    a={'track_id':1,'category':5,'points':list(range(100)),'frames':[0,40],'mean_point_score':.9}
    b={'track_id':1,'category':5,'points':list(range(100)),'frames':[20,60],'mean_point_score':.9}
    labels,_,report=object_consensus([[a],[b]],[],base,xyz)
    assert not labels['instance'].any() and report['ambiguous_ownership_points']==100


def test_sga_embedding_never_overrides_failed_geometry():
    rows=[{'source_track':1,'target_track':1,'rank_score':0.,'accepted_geometry':False},
          {'source_track':1,'target_track':2,'rank_score':.2,'accepted_geometry':True,'coverage_5cm':.8,'rmse_m':.01}]
    assert greedy_pairs(rows,'sga')==[(1,2)]
