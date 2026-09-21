"""Replay old fixed-prompt PixelClaims with one saved descriptive blue mask.

No new model invocation, no class-taxonomy or full-map change. Class 35 exists
only in this counterfactual diagnostic and denotes a prompt description.
"""
from pathlib import Path
import json
import numpy as np
import run_tracking_probe as probe
from pose_pipeline.sam3_fusion import PixelClaims


def counts(labels,region):
    ids,n=np.unique(labels[region],return_counts=True)
    return {str(int(i)):int(c) for i,c in zip(ids,n)}


def main():
    O=probe.R/'tracking/blue_equipment';blue=np.load(O/'text_blue_machine/004811.npz')
    blue_shape=tuple(blue['mask_shape']);assert blue_shape[0]==1
    region=np.unpackbits(blue['masks_packed'],axis=-1)[...,:blue_shape[-1]].astype(bool)[0]
    family=probe.B/'family_v3/orbbec/scan_20260909_142829_5ef1fa'
    cache=family/'frames/004811.npz';f=np.load(cache);shape=tuple(f['depth_shape']);assert shape==region.shape
    row=next(json.loads(l) for l in (family/'frames.jsonl').read_text().splitlines() if json.loads(l)['frame_id']==4811)
    source=json.loads((O/'INPUTS.json').read_text())['frames'][0]
    assert row['color_sha256']==source['color_sha256'] and row['depth_sha256']==source['depth_sha256']
    assert probe.sha256_file(Path(source['color_path']))==source['color_sha256']
    assert probe.sha256_file(Path(source['depth_path']))==source['depth_sha256']
    taxonomy_path=probe.B/'code/configs/sam3_indoor_v1.json';tax=json.loads(taxonomy_path.read_text())['classes']
    names={0:'unknown',**{c['id']:c['name'] for c in tax},33:'table-like',34:'curtain-like',35:'blue equipment (prompt description)'}
    active=[c['prompt'] for c in tax if c['prompt'] is not None]
    assert len(active)==31 and 'machine' not in active and 'blue machine' not in active
    count=int(f['raw_mask_count']);packed=f['raw_masks_packed'];assert len(packed)==count
    masks=np.unpackbits(packed,axis=-1)[...,:int(np.prod(shape))].reshape(count,*shape).astype(bool)
    claims=PixelClaims(shape,36)
    for mask,rec in zip(masks,row['masks'][:count]):
        assert int(mask.sum())==rec['pixel_count']
        claims.add(rec['class_id'],mask,rec['score'])
    old,_,_=claims.finalize();raw_reference=probe.A/'batch_v1/orbbec/scan_20260909_142829_5ef1fa/frames/004811.npz'
    assert np.array_equal(old,np.load(raw_reference)['semantic']),'cached pre-family raw replay mismatch'
    best_before=claims.scores.argmax(0);overlaps={str(c['id']):int(((claims.scores[c['id']]>=.5)&region).sum()) for c in tax}
    claims.add(35,region,float(blue['scores'][0]));new,_,_=claims.finalize()
    audit={'version':'blue-competition-replay-v1','new_model_calls':0,'frame_id':4811,
        'temporary_class':{'id':35,'name':names[35],'prompt':'blue machine','semantic_business_name_known':False},
        'old_active_prompt_count':len(active),'old_prompts':active,'old_has_machine_prompt':False,
        'settings':{'PixelClaims_threshold':.5,'PixelClaims_margin':.1,'class_count_for_counterfactual':36},
        'same_rgb_depth_hashes':True,'same_depth_mask_shape':[int(n) for n in shape],'raw_cached_prediction_reproduced':True,
        'blue_mask_pixels':int(region.sum()),'new_blue_mask_score':float(blue['scores'][0]),
        'old_pre_family_classes_inside_blue':counts(old,region),
        'old_family_classes_inside_blue':counts(f['semantic'],region),
        'old_best_class_before_margin_inside_blue':counts(best_before,region),
        'old_class_overlap_pixels_inside_blue':{k:v for k,v in overlaps.items() if v},
        'new_pre_family_classes_inside_blue':counts(new,region),
        'retained_blue_pixels':int(((new==35)&region).sum()),
        'new_unknown_pixels_inside_blue':int(((new==0)&region).sum()),
        'changed_pixels_outside_blue_candidate':int(((old!=new)&~region).sum()),
        'classes':names,'full_3d_fusion_executed':False,'accuracy_measured':False,'production_taxonomy_changed':False,
        'sources_sha256':{str(p):probe.sha256_file(p) for p in [Path(__file__),cache,family/'frames.jsonl',taxonomy_path,raw_reference,O/'text_blue_machine/004811.npz']}}
    probe.write_json(O/'BLUE_COMPETITION.json',audit)
    image=probe.Image.open(O/'rgb.png').convert('RGB')
    # Color only the newly retained provisional class, not all old class masks.
    probe.overlay(image,(new==35)[None],np.array([1]),'cached old31 + blue machine: retained blue candidate').save(O/'blue_competition_overlay.png')
    np.savez_compressed(O/'blue_competition_labels.npz',old_pre_family=old,new_pre_family=new,blue_candidate=region)
    print(json.dumps(audit))


if __name__=='__main__':main()
