"""Fresh B/D inference and E decision replay; isolated from production."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import run_object_memory_trial as base
from pose_pipeline.semantic_runtime.common import read,write,sha,PROMPT
from pose_pipeline.semantic_runtime.object_memory import representatives,crop_key,voted_name
from pose_pipeline.semantic_runtime.context_review import review_reasons,review_views,decisions,context_image,CONTEXT_PROMPT


def config():return read(ROOT/'docs/context-review-plan.json')

def fusion_plan():
    p=read(ROOT/'docs/object-memory-plan.json');c=config()
    return {**p,'scenes':c['scenes'],'development_scenes':c['legacy_fixed_input_scenes'],
            'registered_root':'/home/aidenwu/Documents/SGF-SGA-experiments/semantic_pipeline_round2_20260915_v1/adaptive/scannet'}

base.plan=fusion_plan


def prepare():
    from pose_pipeline.contracts import load_manifest,load_trajectory,bind_manifest_trajectory
    base.prepare()
    lock=read(ROOT/'INPUT_LOCK.json')
    for p in (ROOT/'src/revemap/resources').glob('*.json'):lock[str(p)]=sha(p)
    for p in [ROOT/'docs/context-review-plan.json',Path(fusion_plan()['runtime_file'])]:lock[str(p)]=sha(p)
    for scene in config()['scenes']:
        data=ROOT/'inputs'/scene;geom=read(data/'GEOMETRY.json')
        m=load_manifest(Path(geom['manifest']));poses,_=load_trajectory(Path(geom['trajectory']))
        frames={f.frame_id:f for f,p in bind_manifest_trajectory(m,poses)};sources={}
        for row in read(data/'FRAMES.json'):
            fid=row['frame_id'];frame=frames[fid]
            if scene in config()['legacy_fixed_input_scenes']:
                image=Path(fusion_plan()['registered_root'])/scene/'frames'/f'{fid:06}.png';rotate=False
                digest=row['registered_image_sha256']
            else:image=frame.color_path;rotate=frame.rotate_ccw;digest=row['color_sha256']
            assert sha(image)==digest
            mask=base.source(scene)/'semantic/frames'/f'{fid:06}.npz'
            sources[str(fid)]={'image':str(image),'image_sha256':digest,'rotate_ccw':rotate,
                               'mask':str(mask),'mask_sha256':row['mask_sha256']}
            lock[str(image)]=digest
        write(data/'CONTEXT_SOURCES.json',sources);lock[str(data/'CONTEXT_SOURCES.json')]=sha(data/'CONTEXT_SOURCES.json')
    write(ROOT/'INPUT_LOCK.json',lock)
    print('CONTEXT_PREPARED',len(lock),flush=True)


def make_context(path,crop):
    import numpy as np
    from PIL import Image
    scene=read(path/'CONTEXT.json')['scene'];fid=crop['frame_id']
    src=read(ROOT/'inputs'/scene/'CONTEXT_SOURCES.json')[str(fid)]
    assert sha(src['image'])==src['image_sha256'] and sha(src['mask'])==src['mask_sha256']
    with Image.open(src['image']) as im:full=im.convert('RGB')
    if src['rotate_ccw']:full=full.transpose(Image.Transpose.ROTATE_90)
    with np.load(src['mask']) as z:mask=z['local_instance']==crop['mask_id']
    yy,xx=np.nonzero(mask);assert len(xx)>0
    h,w=mask.shape
    box=[int(round(xx.min()*full.width/w)),int(round(yy.min()*full.height/h)),
         min(full.width,int(round((xx.max()+1)*full.width/w))),min(full.height,int(round((yy.max()+1)*full.height/h)))]
    folder=path/'semantic/context_images';folder.mkdir(exist_ok=True)
    output=folder/(crop_key(crop)+'.png')
    with Image.open(crop['file']) as tight:audit=context_image(full,tight,box,output)
    return {**crop,'original_crop_file':crop['file'],'original_crop_sha256':crop['sha256'],
            'file':str(output),'sha256':sha(output),'context_image_audit':{**audit,**src}}


def name(path):
    import torch
    from pose_pipeline.semantic_runtime import vlm
    arm=read(path/'CONTEXT.json')['arm'];out=path/'semantic/vlm';out.mkdir(exist_ok=False)
    started=time.monotonic();p=fusion_plan();runtime=read(p['runtime_file']);model_id=p['vlm']
    crops=read(path/'OBJECT_STORE.json')['crops'];classes=read(path/'fused/classes.json')
    import numpy as np
    with np.load(path/'fused/map_labels.npz') as z:
        inst=z['instance'];sem=z['semantic']
    object_class={int(i):int(np.unique(sem[inst==i])[0]) for i in np.unique(inst) if i>0}
    for c in crops:assert sha(c['file'])==c['sha256']
    vlm.PROMPT=PROMPT
    model=vlm.create_namer(model_id,{**runtime['models'][model_id],'threads':2,'log_dir':str(out)})
    write(out/'MODEL.json',model.audit)
    write(out/'INPUTS.json',{'base_prompt':PROMPT,'context_prompt':CONTEXT_PROMPT,'GT_in_requests':False,'same_model_reused_for_review':True})
    grouped=defaultdict(list)
    for c in crops:
        if c['association']['eligible']:grouped[c['association']['object_id']].append(c)
    records=[];groups=[];byobject={};seen=set()
    def infer(c):
        key=crop_key(c);assert key not in seen;seen.add(key)
        x={**c,**model.infer(c['file'])};records.append(x);return x
    # Exactly the previously tested B selection/fallback, before any context.
    for oid,cs in sorted(grouped.items()):
        if len({c['frame_id'] for c in cs})<2:
            groups.append({'object_id':oid,'skip':'fewer_than_2_distinct_frames'});continue
        initial=representatives(cs,3);evidence=[infer(c) for c in initial]
        fallback=voted_name(evidence)=='unknown'
        if fallback:
            for c in cs:
                if crop_key(c) not in seen:evidence.append(infer(c))
        byobject[oid]=evidence
        groups.append({'object_id':oid,'eligible_crops':len(cs),'initial':[crop_key(c) for c in initial],
                       'fallback':fallback,'calls':len(evidence),'name':voted_name(evidence)})
    base_done=time.monotonic();reviews=[];context_records=[]
    if arm=='D':
        vlm.PROMPT=CONTEXT_PROMPT
        for oid,evidence in sorted(byobject.items()):
            before=voted_name(evidence);sam_name=classes[str(object_class[oid])]
            reasons=review_reasons(before,sam_name,evidence)
            if not reasons:continue
            selected,mode=review_views(grouped[oid],evidence)
            context=[]
            for c in selected:
                prepared=make_context(path,c)
                result={**prepared,**model.infer(prepared['file']),'review_object_id':oid,'view_choice':mode}
                context.append(result);context_records.append(result)
            d,e,reason=decisions(before,sam_name,context)
            reviews.append({'object_id':oid,'base_name':before,'sam_class':sam_name,'triggers':reasons,
                            'view_choice':mode,'context_keys':[crop_key(c) for c in context],
                            'context_name':voted_name(context),'D_name':d,'E_name':e,'decision_reason':reason})
    completed=time.monotonic();vlm.PROMPT=PROMPT
    # Original backfill sees ONLY the base records. Context evidence is applied
    # separately per object, so a repeated frame can never vote twice.
    write(out/'response_000000.json',{'task_id':0,'completed_at':base_done,'crops':records})
    write(out/'RECORDS.json',records);write(out/'CONTEXT_RECORDS.json',context_records);write(out/'REVIEWS.json',reviews)
    write(out/'COMPLETE.json',{'status':'completed','model':model_id,'base_calls':len(records),'context_calls':len(context_records),
        'total_calls':len(records)+len(context_records),'base_inference_seconds':sum(x['request_seconds'] for x in records),
        'context_inference_seconds':sum(x['request_seconds'] for x in context_records),
        'context_stage_seconds':completed-base_done,'seconds_including_load':completed-started,'groups':groups,
        'reviewed_objects':len(reviews),'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
    model.close()


def apply_review(path):
    original=read(path/'fused/instance_names.json');reviews=read(path/'semantic/vlm/REVIEWS.json')
    records=read(path/'semantic/vlm/CONTEXT_RECORDS.json');byid={x['object_id']:x for x in reviews}
    write(path/'fused/instance_names_B_shadow.json',original)
    for arm in ('D','E'):
        output=[]
        for obj in original:
            oid=obj['instance_id'];row=dict(obj)
            if oid in byid:
                review=byid[oid];assert obj['vlm_name']==review['base_name']
                target=review[arm+'_name'];row['base_vlm_name']=obj['vlm_name'];row['context_review']=review
                if target!=obj['vlm_name']:
                    ev=[x for x in records if x['review_object_id']==oid and x['label']==target]
                    assert len({x['frame_id'] for x in ev})>=2
                    row.update(vlm_name=target,support_frames=sorted({x['frame_id'] for x in ev}),evidence=ev,
                               assignment_scope='context-review naming metadata only; original geometry and semantic/instance IDs retained')
            output.append(row)
        write(path/'fused'/('instance_names.json' if arm=='D' else 'instance_names_E.json'),output)


def worker(python,mode,path):
    stream=(path/f'{mode}.log').open('w')
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONPATH':str(ROOT/'src'),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    p=subprocess.Popen([str(python),'-u',str(Path(__file__).resolve()),mode,'--path',str(path)],stdout=stream,stderr=subprocess.STDOUT,env=env)
    return p,stream,time.monotonic()


def run():
    import numpy as np
    from pose_pipeline.semantic_runtime import backfill
    base.verify(read(ROOT/'INPUT_LOCK.json'));runtime=read(fusion_plan()['runtime_file']);model_id=fusion_plan()['vlm']
    python=runtime['models'][model_id].get('python',runtime['vlm_python']);times=[]
    for repeat,order in enumerate(config()['repeats'],1):
        for scene in config()['scenes']:
            for arm in order:
                path=base.setup(scene,arm,repeat);tick=time.monotonic()
                fusion_seconds=base.wait(worker(base.CPU,'fuse',path));base.store(path)
                naming_seconds=base.wait(worker(python,'name',path));backfill.main(argparse.Namespace(arm_root=path))
                if arm=='D':apply_review(path)
                elapsed=time.monotonic()-tick;complete=read(path/'semantic/vlm/COMPLETE.json')
                names=read(path/'fused/instance_names.json')
                row={'scene':scene,'arm':arm,'repeat':repeat,'tail_wall_seconds':elapsed,
                     'fusion_process_seconds':fusion_seconds,'naming_process_seconds':naming_seconds,
                     'base_calls':complete['base_calls'],'context_calls':complete['context_calls'],'total_calls':complete['total_calls'],
                     'reviewed_objects':complete['reviewed_objects'],'objects':len(names),'named_objects':sum(x['vlm_name']!='unknown' for x in names)}
                write(path/'TIMING.json',row);times.append(row);write(ROOT/'TIMINGS.json',times);print('DONE',json.dumps(row),flush=True)
    parity=[]
    for scene in config()['scenes']:
        first=ROOT/'runs/repeat-1'/scene/'B'
        with np.load(first/'fused/map_labels.npz') as z:baseline={k:z[k] for k in z.files}
        xyz=np.load(first/'fused/target.npz')['xyz']
        for repeat in (1,2):
            b=ROOT/'runs'/f'repeat-{repeat}'/scene/'B';d=b.parent/'D'
            getnames=lambda p,name='instance_names.json':{x['instance_id']:x['vlm_name'] for x in read(p/'fused'/name)}
            assert getnames(b)==getnames(d,'instance_names_B_shadow.json')
            for arm in ('B','D'):
                path=b.parent/arm
                with np.load(path/'fused/map_labels.npz') as z:assert all(np.array_equal(z[k],v) for k,v in baseline.items())
                assert np.array_equal(np.load(path/'fused/target.npz')['xyz'],xyz)
                parity.append({'scene':scene,'repeat':repeat,'arm':arm,'labels_xyz_equal':True,'B_shadow_equal':True})
        if scene in config()['development_scenes']:
            old=Path('/home/aidenwu/Documents/ReVeMap-object-memory-20260927/runs/repeat-2')/scene/'B'
            assert getnames(first)==getnames(old)
    write(ROOT/'PARITY.json',parity);base.verify(read(ROOT/'INPUT_LOCK.json'))
    lock={str(p):sha(p) for p in (ROOT/'runs').rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.ply','.png')}
    write(ROOT/'PREDICTIONS_LOCK.json',lock)
    write(ROOT/'PREDICTIONS_COMPLETE.json',{'status':'completed','GT_used':False,'files':len(lock),'raw_model_runs':len(times),
           'E_scope':'decision replay from D; same inference cost, no separate GPU run'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','name','fuse']);parser.add_argument('--path',type=Path)
    a=parser.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='run':run()
    elif a.mode=='fuse':base.fuse(a.path)
    else:name(a.path)
