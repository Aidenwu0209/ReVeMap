from pose_pipeline.semantic_runtime.object_memory import association, representatives, voted_name, cache_identity


def test_geometry_support_requires_enough_points_and_share():
    assert not association([1]*29+[0]*2)['eligible']
    assert not association([1]*64+[2]*36)['eligible']
    assert association([1]*65+[2]*35)['eligible']


def test_same_frame_and_cached_aliases_are_not_independent_evidence():
    x = {'frame_id':1,'label':'chair','executed':True}
    assert voted_name([x,x,dict(x,frame_id=2,executed=False)]) == 'unknown'
    assert voted_name([x,dict(x,frame_id=2)]) == 'chair'
    assert voted_name([x,dict(x,frame_id=2),dict(x,label='sofa',frame_id=3),dict(x,label='sofa',frame_id=4)]) == 'unknown'


def test_views_keep_distinct_frames_and_determinism():
    crops=[{'frame_id':f,'mask_id':m,'mask_pixels':a,'association':{'share':.9},'view_direction':v}
           for f,m,a,v in [(1,1,100,[1,0,0]),(1,2,90,[1,0,0]),(2,1,80,[0,1,0]),(3,1,60,[-1,0,0]),(4,1,1,[0,0,1])]]
    selected=representatives(crops)
    assert {x['frame_id'] for x in selected} == {1,2,3}
    assert selected == representatives(list(reversed(crops)))


def test_cache_invalidates_for_changed_membership_or_evidence():
    args=['s','g','i','m','p',['a','b']]
    assert cache_identity(*args) == cache_identity(*args[:-1],['b','a'])
    for pos in range(5):
        changed=args.copy();changed[pos]+='x'
        assert cache_identity(*args) != cache_identity(*changed)
    assert cache_identity(*args) != cache_identity(*args[:-1],['a','c'])
