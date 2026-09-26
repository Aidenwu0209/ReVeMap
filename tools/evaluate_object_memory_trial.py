"""Post-seal evaluation only. Does not alter predictions or select thresholds."""
from collections import Counter
import json
from pathlib import Path
import sys
import statistics
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from run_object_memory_trial import read,write,sha,verify,plan,source,store
from pose_pipeline.scene_graph import _label
from pose_pipeline.contracts import load_manifest,load_trajectory,bind_manifest_trajectory,validate_se3
from pose_pipeline.evaluation import evaluate_result


def ground_truth(scene,path):
    reference=(Path('/home/aidenwu/Documents/Scannet/Original_files_from_scannet_dataset')/scene
               if scene in plan()['development_scenes'] else Path('/home/aidenwu/Desktop/Jojo_intern_10_scannet_scenes')/scene)
    files=[reference/(scene+s) for s in ('_vh_clean_2.ply','_vh_clean_2.0.010000.segs.json','.aggregation.json')]
    mesh=PlyData.read(files[0])['vertex'].data
    segments=np.asarray(read(files[1])['segIndices']);assert len(segments)==len(mesh)
    ids=np.zeros(len(segments),np.int64);names={}
    for x in read(files[2])['segGroups']:
        oid=int(x['objectId'])+1;mask=np.isin(segments,x['segments']);assert not (ids[mask]>0).any()
        ids[mask]=oid;names[oid]=x['label']
    geometry=read(path/'mapping/mapping_result.json')
    manifest=load_manifest(Path(geometry['manifest']),require_files=False)
    poses,_=load_trajectory(Path(geometry['trajectory']))
    for frame,pose in bind_manifest_trajectory(manifest,poses):
        p=manifest.root/'pose'/f'{pose.frame_id}.txt'
        if not p.exists():continue
        gt_pose=np.loadtxt(p)
        if not np.isfinite(gt_pose).all():continue
        gt_pose=validate_se3(gt_pose,'GT alignment')
        alignment=gt_pose@np.linalg.inv(pose.t_world_camera);files.append(p);break
    else:raise ValueError('missing alignment pose')
    with np.load(path/'fused/target.npz') as z:xyz=z['xyz']
    world=xyz@alignment[:3,:3].T+alignment[:3,3]
    gt_xyz=np.column_stack([mesh[k] for k in ('x','y','z')])
    distance,nearest=cKDTree(gt_xyz).query(world,workers=1)
    return reference,ids[nearest],distance<=.05,names,{str(p):sha(p) for p in files}


def main():
    assert read(ROOT/'PREDICTIONS_COMPLETE.json')['status']=='completed'
    verify(read(ROOT/'INPUT_LOCK.json'));verify(read(ROOT/'CONFIG_LOCK.json'));verify(read(ROOT/'PREDICTIONS_LOCK.json'))
    evaluation=ROOT/'evaluation';evaluation.mkdir(exist_ok=False)
    output=[];gtlock={}
    for scene in plan()['scenes']:
        baseline=ROOT/'runs/repeat-1'/scene/'A'
        reference,projected,near,gt_names,lock=ground_truth(scene,baseline);gtlock.update(lock)
        metrics=evaluate_result(baseline/'fused',reference)
        write(evaluation/(scene+'-geometry.json'),metrics)
        with np.load(baseline/'fused/map_labels.npz') as z:instance=z['instance']
        fixed=read(ROOT/'runs/repeat-1'/scene/'B/OBJECT_STORE.json')['crops']
        eligible_frames={}
        for c in fixed:
            if c['association']['eligible']:
                eligible_frames.setdefault(c['association']['object_id'],set()).add(c['frame_id'])
        classes={_label(x) for x in read(baseline/'fused/classes.json').values()}-{'unknown'}
        truth={};diagnostics=[]
        # Independent GT target: an object must have >=30 near annotated
        # vertices, >=50% geometric support, and >=80% one-GT-object purity.
        # Names outside the pre-existing shared class vocabulary are reported
        # as unrecognized, not silently converted using GT-specific aliases.
        for oid in sorted(int(i) for i in np.unique(instance) if i>0):
            owned=instance==oid; supported=owned&near&(projected>0)
            counts=Counter(projected[supported].tolist())
            gt,count=counts.most_common(1)[0] if counts else (0,0)
            target=_label(gt_names.get(gt,'unknown'))
            qualify=(int(supported.sum())>=30 and supported.sum()>=.5*owned.sum()
                     and count>=.8*supported.sum() and target in classes
                     and len(eligible_frames.get(oid,[]))>=2 and target not in ('wall','floor'))
            diagnostics.append({'object_id':oid,'gt_object_id':gt,'gt_name':gt_names.get(gt,'unknown'),
                'canonical_gt_name':target,'owned_points':int(owned.sum()),'supported_points':int(supported.sum()),
                'dominant_gt_points':count,'eligible_for_naming_metric':bool(qualify)})
            if qualify:truth[oid]=target
        write(evaluation/(scene+'-GT-targets.json'),diagnostics)
        for repeat in (1,2):
            a_names={x['instance_id']:x['vlm_name'] for x in read(ROOT/'runs'/f'repeat-{repeat}'/scene/'A/fused/instance_names.json')}
            for arm in ('A','B','C'):
                path=ROOT/'runs'/f'repeat-{repeat}'/scene/arm
                objects=read(path/'fused/instance_names.json')
                names={x['instance_id']:x['vlm_name'] for x in objects}
                details=[]
                for oid,target in truth.items():
                    prediction=_label(names[oid]);known=prediction in classes
                    details.append({'object_id':oid,'target':target,'name':names[oid],'canonical':prediction,
                                    'shared_vocabulary':known,'correct':prediction==target})
                record={**read(path/'TIMING.json'), 'naming_gt_eligible_objects':len(details),
                    'gt_exact_correct':sum(d['correct'] for d in details),
                    'gt_shared_vocab_named':sum(d['shared_vocabulary'] for d in details),
                    'gt_shared_vocab_wrong':sum(d['shared_vocabulary'] and not d['correct'] for d in details),
                    'gt_unknown':sum(d['name']=='unknown' for d in details),
                    'gt_out_of_vocab_name':sum(d['name']!='unknown' and not d['shared_vocabulary'] for d in details),
                    'gt_details':details,
                    'name_equal_A':sum(names[i]==a_names[i] for i in names),'name_changed_A':[
                        {'object_id':i,'A':a_names[i],'candidate':names[i]} for i in names if names[i]!=a_names[i]]}
                if arm!='A':
                    crops=read(path/'OBJECT_STORE.json')['crops'];groups=read(path/'semantic/vlm/COMPLETE.json')['groups']
                    record.update(geometrically_eligible_crops=sum(x['association']['eligible'] for x in crops),
                                  fallback_objects=sum(x.get('fallback',False) for x in groups),
                                  selection_seconds=read(path/'OBJECT_STORE.json')['seconds'])
                if arm=='C':record['clip']=read(path/'CLIP.json')
                output.append(record)
    # Compare actual predictions across the two independent inference runs.
    repeats=[]
    for scene in plan()['scenes']:
        for arm in ('A','B','C'):
            a=read(ROOT/'runs/repeat-1'/scene/arm/'semantic/vlm/RECORDS.json')
            b=read(ROOT/'runs/repeat-2'/scene/arm/'semantic/vlm/RECORDS.json')
            keyed=lambda rows:{(c['frame_id'],c['mask_id']):c['label'] for c in rows}
            left,right=keyed(a),keyed(b)
            names=lambda n:{x['instance_id']:x['vlm_name'] for x in read(ROOT/'runs'/f'repeat-{n}'/scene/arm/'fused/instance_names.json')}
            repeats.append({'scene':scene,'arm':arm,'same_crop_keys':left.keys()==right.keys(),
                            'same_crop_labels':left==right,'same_final_names':names(1)==names(2)})
    write(evaluation/'REPEATABILITY.json',repeats)
    write(evaluation/'DETAILS.json',output);write(evaluation/'GT_LOCK.json',gtlock)
    totals=[]
    for repeat in (1,2):
        for arm in ('A','B','C'):
            selected=[x for x in output if x['repeat']==repeat and x['arm']==arm]
            row={'repeat':repeat,'arm':arm}
            for key in ('tail_wall_seconds','vlm_calls','vlm_inference_seconds','named_objects','objects','naming_gt_eligible_objects',
                        'gt_exact_correct','gt_shared_vocab_named','gt_shared_vocab_wrong','gt_unknown','gt_out_of_vocab_name'):
                row[key]=sum(x[key] for x in selected)
            totals.append(row)
    write(evaluation/'TOTALS.json',totals)
    verify(read(ROOT/'INPUT_LOCK.json'));verify(read(ROOT/'CONFIG_LOCK.json'));verify(read(ROOT/'PREDICTIONS_LOCK.json'));verify(gtlock)
    write(evaluation/'COMPLETE.json',{'status':'completed','GT_used_only_after_prediction_seal':True,
        'naming_metric':'strict existing-alias canonical-name comparison on fixed GT-supported, pure, nameable foreground objects; unrecognized subtypes count as unmatched, so not a complete semantic correctness score',
        'geometry_evaluation':'unchanged project diagnostic; not ScanNet benchmark AP',
        'gt_target_gates':{'min_points':30,'support_share':.5,'instance_purity':.8,'min_eligible_crop_frames':2,'distance_m':.05},
        'all_locks_passed':True})
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
