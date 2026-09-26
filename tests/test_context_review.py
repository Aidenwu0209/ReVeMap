from PIL import Image
from pose_pipeline.semantic_runtime.context_review import review_reasons,review_views,decisions,context_image

def rec(frame,label):return {'frame_id':frame,'label':label,'executed':True}

def test_triggers_use_predictions_without_ground_truth():
    assert review_reasons('unknown','chair',[])==['unknown_name']
    assert review_reasons('fridge','refridgerator',[rec(1,'fridge')])==[]
    assert set(review_reasons('table','chair',[rec(1,'table'),rec(2,'chair')]))=={'sam_vlm_category_conflict','cross_view_disagreement'}

def test_context_repeated_same_frame_is_not_independent_support():
    assert decisions('unknown','chair',[rec(1,'chair'),rec(1,'chair')])[:2]==('unknown','unknown')
    assert decisions('unknown','chair',[rec(1,'chair'),rec(2,'chair')])[:2]==('chair','chair')
    assert decisions('chair','chair',[rec(1,'table'),rec(2,'table')])[:2]==('table','chair')

def test_unseen_frame_priority_and_fallback_are_explicit():
    crops=[{'frame_id':f,'mask_id':1,'mask_pixels':100,'association':{'share':1.},'view_direction':[1.,0.,0.]} for f in range(5)]
    chosen,mode=review_views(crops,[rec(0,'chair'),rec(1,'chair'),rec(2,'table')])
    assert {x['frame_id'] for x in chosen}=={3,4} and mode=='previously_uninferred_frames'
    chosen,mode=review_views(crops,[rec(f,'chair') for f in range(4)])
    assert len({x['frame_id'] for x in chosen})==3 and mode=='existing_frames_wider_context'

def test_context_crop_uses_actual_bounded_pixels(tmp_path):
    original=Image.new('RGB',(100,80),(20,40,60));before=original.tobytes()
    out=tmp_path/'pair.png';audit=context_image(original,original.crop((0,0,20,30)),[0,0,20,30],out,64)
    assert audit['context_box']==[0,0,40,60]
    assert Image.open(out).size==(136,64) and original.tobytes()==before
