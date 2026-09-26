"""Sealed B/D/E diagnostics with heldout scenes reported separately."""
from collections import Counter
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_context_review_trial import base,config,read,write,sha
from evaluate_object_memory_trial import ground_truth
from pose_pipeline.scene_graph import _label
from pose_pipeline.contracts import load_manifest,load_trajectory,bind_manifest_trajectory,validate_se3


def names(scene,repeat,arm):
    path=ROOT/'runs'/f'repeat-{repeat}'/scene/('D' if arm=='E' else arm)
    file='instance_names_E.json' if arm=='E' else 'instance_names.json'
    return {x['instance_id']:x['vlm_name'] for x in read(path/'fused'/file)}


def crop_gt(scene,reference,gtlock):
    """Use GT camera pose for original mask-depth pixels, avoiding map drift."""
    from reconstruction.rgbd_refusion import _read_rgbd
    path=ROOT/'runs/repeat-1'/scene/'D';records=read(path/'semantic/vlm/CONTEXT_RECORDS.json')
    if not records:return []
    mesh=PlyData.read(reference/(scene+'_vh_clean_2.ply'))['vertex'].data
    xyz=np.column_stack([mesh[k] for k in ('x','y','z')]);tree=cKDTree(xyz)
    segs=np.asarray(read(reference/(scene+'_vh_clean_2.0.010000.segs.json'))['segIndices'])
    gt_ids=np.zeros(len(segs),np.int64);gt_names={}
    for g in read(reference/(scene+'.aggregation.json'))['segGroups']:
        oid=int(g['objectId'])+1;gt_ids[np.isin(segs,g['segments'])]=oid;gt_names[oid]=g['label']
    geom=read(path/'mapping/mapping_result.json');m=load_manifest(Path(geom['manifest']),require_files=False)
    poses,_=load_trajectory(Path(geom['trajectory']));frames={f.frame_id:f for f,p in bind_manifest_trajectory(m,poses)}
    bnames=names(scene,1,'B');rows=[];sources=read(ROOT/'inputs'/scene/'CONTEXT_SOURCES.json')
    for rec in records:
        fid=rec['frame_id'];frame=frames[fid];gtpath=m.root/'pose'/f'{fid}.txt'
        row={'frame_id':fid,'mask_id':rec['mask_id'],'object_id':rec['review_object_id'],
             'B_object_name':bnames[rec['review_object_id']],'context_crop_name':rec['label'],
             'sam_class_id':rec['sam_class_id'],'image_file':rec['file']}
        if not gtpath.exists():rows.append({**row,'eligible':False,'reason':'missing_GT_pose'});continue
        mat=np.loadtxt(gtpath);gtlock[str(gtpath)]=sha(gtpath)
        if not np.isfinite(mat).all():rows.append({**row,'eligible':False,'reason':'nonfinite_GT_pose'});continue
        mat=validate_se3(mat,'crop GT pose');assert not frame.rotate_ccw
        _,depth,intrinsics=_read_rgbd(frame);depth=depth.astype(float)/m.depth_scale
        with np.load(sources[str(fid)]['mask']) as z:mask=z['local_instance']==rec['mask_id']
        assert mask.shape==depth.shape
        v,u=np.nonzero(mask&np.isfinite(depth)&(depth>0));z=depth[v,u];fx,fy,cx,cy=intrinsics
        camera=np.column_stack(((u-cx)*z/fx,(v-cy)*z/fy,z));world=camera@mat[:3,:3].T+mat[:3,3]
        distances,nearest=tree.query(world,workers=1);supported=(distances<=.05)&(gt_ids[nearest]>0)
        counts=Counter(gt_ids[nearest[supported]].tolist());gt,count=counts.most_common(1)[0] if counts else (0,0)
        target=_label(gt_names.get(gt,'unknown'));eligible=int(supported.sum())>=30 and count>=.8*supported.sum()
        rows.append({**row,'eligible':bool(eligible),'target_GT_name':gt_names.get(gt,'unknown'),
                     'canonical_GT_name':target,'GT_object_id':gt,'valid_depth_pixels':len(z),
                     'supported_pixels':int(supported.sum()),'dominant_GT_pixels':count,
                     'B_matches_crop_GT':_label(row['B_object_name'])==target,
                     'context_matches_crop_GT':_label(rec['label'])==target})
    return rows


def main():
    assert read(ROOT/'PREDICTIONS_COMPLETE.json')['status']=='completed'
    base.verify(read(ROOT/'INPUT_LOCK.json'));base.verify(read(ROOT/'PREDICTIONS_LOCK.json'))
    out=ROOT/'evaluation';out.mkdir(exist_ok=False);results=[];gtlock={};changes=[]
    for scene in config()['scenes']:
        path=ROOT/'runs/repeat-1'/scene/'B'
        reference,projected,near,gt_names,lock=ground_truth(scene,path);gtlock.update(lock)
        with np.load(path/'fused/map_labels.npz') as z:instance=z['instance']
        classes={_label(x) for x in read(path/'fused/classes.json').values()}-{'unknown'}
        eligible_frames={}
        for c in read(path/'OBJECT_STORE.json')['crops']:
            if c['association']['eligible']:eligible_frames.setdefault(c['association']['object_id'],set()).add(c['frame_id'])
        truth={};diagnostics=[]
        for oid in sorted(int(i) for i in np.unique(instance) if i>0):
            owned=instance==oid;supported=owned&near&(projected>0);counts=Counter(projected[supported].tolist())
            gt,count=counts.most_common(1)[0] if counts else (0,0);target=_label(gt_names.get(gt,'unknown'))
            qualify=(int(supported.sum())>=30 and supported.sum()>=.5*owned.sum() and count>=.8*supported.sum()
                     and target in classes and len(eligible_frames.get(oid,[]))>=2 and target not in ('wall','floor'))
            diagnostics.append({'object_id':oid,'GT_object_id':gt,'GT_name':gt_names.get(gt,'unknown'),
                                'canonical_GT_name':target,'owned_points':int(owned.sum()),'supported_points':int(supported.sum()),
                                'dominant_GT_points':count,'eligible':bool(qualify)})
            if qualify:truth[oid]=target
        write(out/(scene+'-GT-targets.json'),diagnostics)
        for repeat in (1,2):
            bnames=names(scene,repeat,'B')
            for arm in ('B','D','E'):
                predicted=names(scene,repeat,arm);actual_arm='D' if arm=='E' else arm
                p=ROOT/'runs'/f'repeat-{repeat}'/scene/actual_arm
                timing=read(p/'TIMING.json');complete=read(p/'semantic/vlm/COMPLETE.json')
                details=[{'object_id':oid,'target':target,'B':bnames[oid],'prediction':predicted[oid],
                          'correct':_label(predicted[oid])==target,'B_correct':_label(bnames[oid])==target} for oid,target in truth.items()]
                changed=[{'object_id':oid,'B':bnames[oid],'candidate':name,'GT_target':truth.get(oid),
                          'GT_eligible':oid in truth} for oid,name in predicted.items() if name!=bnames[oid]]
                for x in changed:changes.append({'scene':scene,'repeat':repeat,'arm':arm,**x})
                results.append({**timing,'arm':arm,'timing_reused_from_D':arm=='E',
                    'split':'heldout' if scene in config()['heldout_scenes'] else 'development',
                    'named_objects':sum(n!='unknown' for n in predicted.values()),'GT_eligible':len(details),
                    'GT_correct':sum(x['correct'] for x in details),'GT_unknown':sum(x['prediction']=='unknown' for x in details),
                    'GT_named_unmatched':sum(x['prediction']!='unknown' and not x['correct'] for x in details),
                    'GT_new_correct':sum(x['correct'] and not x['B_correct'] for x in details),
                    'GT_regressions':sum(not x['correct'] and x['B_correct'] for x in details),
                    'GT_out_of_vocab':sum(_label(x['prediction']) not in classes and x['prediction']!='unknown' for x in details),
                    'changed_names':len(changed),'GT_details':details,'changed_details':changed,
                    'peak_reserved_mib':complete['peak_reserved_mib'],'peak_rss_mib':complete['peak_rss_mib'],
                    'context_stage_seconds':complete['context_stage_seconds']})
        write(out/(scene+'-crop-GT-diagnostic.json'),crop_gt(scene,reference,gtlock))
    repeats=[]
    for scene in config()['scenes']:
        for arm in ('B','D','E'):
            row={'scene':scene,'arm':arm,'same_final_names':names(scene,1,arm)==names(scene,2,arm)}
            if arm!='E':
                ps=[ROOT/'runs'/f'repeat-{i}'/scene/arm/'semantic/vlm' for i in (1,2)]
                key=lambda p,f:{(x['frame_id'],x['mask_id']):x['label'] for x in read(p/f)}
                row['same_base_crop_labels']=key(ps[0],'RECORDS.json')==key(ps[1],'RECORDS.json')
                row['same_context_crop_labels']=key(ps[0],'CONTEXT_RECORDS.json')==key(ps[1],'CONTEXT_RECORDS.json')
                row['same_review_decisions']=read(ps[0]/'REVIEWS.json')==read(ps[1]/'REVIEWS.json')
            repeats.append(row)
    totals=[]
    keys=('tail_wall_seconds','base_calls','context_calls','total_calls','reviewed_objects','objects','named_objects',
          'GT_eligible','GT_correct','GT_unknown','GT_named_unmatched','GT_new_correct','GT_regressions','GT_out_of_vocab','changed_names')
    for repeat in (1,2):
        for split in ('development','heldout','all'):
            for arm in ('B','D','E'):
                rows=[x for x in results if x['repeat']==repeat and x['arm']==arm and (split=='all' or x['split']==split)]
                totals.append({'repeat':repeat,'split':split,'arm':arm,**{k:sum(x[k] for x in rows) for k in keys}})
    write(out/'DETAILS.json',results);write(out/'TOTALS.json',totals);write(out/'CHANGES.json',changes)
    write(out/'REPEATABILITY.json',repeats);write(out/'GT_LOCK.json',gtlock)
    base.verify(read(ROOT/'INPUT_LOCK.json'));base.verify(read(ROOT/'PREDICTIONS_LOCK.json'));base.verify(gtlock)
    write(out/'COMPLETE.json',{'status':'completed','GT_only_after_prediction_seal':True,'all_locks_passed':True,
                              'primary_protocol_unchanged':True,'heldout_reported_separately':True,
                              'crop_GT_scope':'correlated reviewed mask views using true per-frame GT pose; diagnostic only, not independent object accuracy'})
    print(json.dumps([x for x in totals if x['repeat']==2],indent=2))


if __name__=='__main__':main()
