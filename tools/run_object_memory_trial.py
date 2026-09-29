"""Object memory B with the original A reference for fixed-input validation."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pose_pipeline.semantic_runtime.common import read, write, sha, PROMPT, model_spec
from pose_pipeline.semantic_runtime.object_memory import association, representatives, crop_key, voted_name, cache_identity

OLD = Path('/home/aidenwu/Documents/ReVeMap-comparison-20260922')
NEW = Path('/home/aidenwu/Documents/ReVeMap-ScanNet10-20260926')


def plan():
    config = read(ROOT/'docs/object-memory-plan.json')
    if set(config['arms']) != {'A', 'B'} or any(
            len(order) != 2 or set(order) != {'A', 'B'} for order in config['repeat_orders']):
        raise ValueError('Only object memory B and its original A reference are supported')
    return config


def verify(lock):
    for p, digest in lock.items():
        if sha(p) != digest:
            raise ValueError('artifact changed: '+p)


def source(scene):
    return OLD/'matched'/scene/'baseline/run-01' if scene in plan()['development_scenes'] else NEW/'runs'/scene/'pipeline'


def prepare():
    import numpy as np
    from PIL import Image
    from pose_pipeline.contracts import load_manifest, load_trajectory, bind_manifest_trajectory
    lock = {}
    for folder in ('src', 'tools'):
        for p in (ROOT/folder).rglob('*.py'):
            lock[str(p)] = sha(p)
    lock[str(ROOT/'docs/object-memory-plan.json')] = sha(ROOT/'docs/object-memory-plan.json')
    summary = []
    for scene in plan()['scenes']:
        src = source(scene); out = ROOT/'inputs'/scene
        out.mkdir(parents=True, exist_ok=False)
        geom = read(src/'mapping/mapping_result.json')
        rows = read(src/'semantic/FRAMES.json')
        if scene not in plan()['development_scenes']:
            start = max(0, (len(rows)-25)//2)
            rows = rows[start:start+25]
        pool = OLD/'views'/scene/'pool' if scene in plan()['development_scenes'] else src/'semantic'
        pool_rows = {x['frame_id']:x for x in read(pool/'FRAMES.json')}
        tasks = {x['frame_id']:x for x in read(pool/'CROP_TASKS.json')}
        selected = []
        manifest = load_manifest(Path(geom['manifest']))
        poses, _ = load_trajectory(Path(geom['trajectory']))
        bound = {f.frame_id:(f,p) for f,p in bind_manifest_trajectory(manifest,poses)}
        for row in rows:
            fid = row['frame_id']
            if scene in plan()['development_scenes']:
                # The older pool is another SAM3 run, so its mask IDs/crops
                # cannot be borrowed. Reconstruct from the EXACT frozen masks
                # and their hash-verified registered RGB instead.
                png=Path('/home/aidenwu/Documents/SGF-SGA-experiments/semantic_pipeline_round2_20260915_v1/adaptive/scannet')/scene/'frames'/f'{fid:06}.png'
                assert sha(png)==row['registered_image_sha256']
                lock[str(png)]=row['registered_image_sha256']
                image=Image.open(png).convert('RGB');candidates=[];crops=[]
                with np.load(src/'semantic/frames'/f'{fid:06}.npz') as z:
                    local=z['local_instance'];semantic=z['semantic']
                assert image.size==(local.shape[1],local.shape[0])
                for mid in np.unique(local[local>0]):
                    mask=local==mid;cats=np.unique(semantic[mask]);assert len(cats)==1
                    cid=int(cats[0]);y,x=np.nonzero(mask)
                    if cid in (0,10,19) or len(x)<300:continue
                    box=(int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1)
                    if box[2]-box[0]>=10 and box[3]-box[1]>=10:candidates.append((len(x),int(mid),cid,box))
                (out/'crops').mkdir(exist_ok=True)
                for area,mid,cid,(x0,y0,x1,y1) in sorted(candidates,key=lambda x:(-x[0],x[1]))[:8]:
                    dx,dy=max(2,round((x1-x0)*.15)),max(2,round((y1-y0)*.15))
                    box=[max(0,x0-dx),max(0,y0-dy),min(image.width,x1+dx),min(image.height,y1+dy)]
                    p=out/'crops'/f'{fid:06}_{mid:04}.png';image.crop(box).save(p)
                    crops.append({'frame_id':fid,'mask_id':mid,'sam_class_id':cid,'file':str(p),'sha256':sha(p),'bbox_rgb':box,'mask_pixels':area})
                task={'task_id':len(selected),'frame_id':fid,'crops':crops}
            else:
                if pool_rows[fid]['mask_sha256']!=row['mask_sha256']:raise ValueError('crop/mask mismatch')
                task=tasks[fid]
            selected.append(task)
            for crop in task['crops']:
                assert sha(crop['file']) == crop['sha256']
                lock[crop['file']] = crop['sha256']
            frame, pose = bound[fid]
            for p, digest in [(frame.color_path,row['color_sha256']),(frame.depth_path,row['depth_sha256']),
                              (src/'semantic/frames'/f'{fid:06}.npz',row['mask_sha256'])]:
                assert sha(p) == digest
                lock[str(p)] = digest
        for p in [src/'mapping/mapping_result.json',src/'semantic/FRAMES.json',pool/'CROP_TASKS.json',
                  Path(geom['manifest']),Path(geom['trajectory']),Path(geom['final_cloud'])]:
            lock[str(p)] = sha(p)
        write(out/'FRAMES.json', rows)
        write(out/'CROP_TASKS.json',selected)
        write(out/'GEOMETRY.json', geom)
        write(out/'POSES.json',{str(r['frame_id']):bound[r['frame_id']][1].t_world_camera for r in rows})
        write(out/'SOURCE.json',{'source':str(src), 'crop_pool':str(pool)})
        summary.append({'scene':scene,'frames':len(rows),'crops':sum(len(t['crops']) for t in selected),
                        'first_frame':rows[0]['frame_id'],'last_frame':rows[-1]['frame_id']})
    for p in (ROOT/'inputs').rglob('*.json'):
        lock[str(p)] = sha(p)
    write(ROOT/'INPUT_LOCK.json',lock)
    write(ROOT/'PREPARED.json',summary)
    print(json.dumps(summary),flush=True)


def setup(scene, arm, repeat):
    if arm not in ('A', 'B'):
        raise ValueError('Unsupported arm: ' + arm)
    path=ROOT/'runs'/f'repeat-{repeat}'/scene/arm
    path.mkdir(parents=True,exist_ok=False)
    (path/'mapping').mkdir(); (path/'semantic').mkdir()
    write(path/'mapping/mapping_result.json',read(ROOT/'inputs'/scene/'GEOMETRY.json'))
    write(path/'semantic/FRAMES.json',read(ROOT/'inputs'/scene/'FRAMES.json'))
    write(path/'semantic/CROP_TASKS.json',read(ROOT/'inputs'/scene/'CROP_TASKS.json'))
    (path/'semantic/frames').symlink_to(source(scene)/'semantic/frames',target_is_directory=True)
    write(path/'CONTEXT.json',{'scene':scene,'arm':arm,'repeat':repeat})
    return path


def fuse(path):
    from pose_pipeline.semantic_runtime import fusion
    from pose_pipeline.artifacts import write_artifact_manifest
    native_write=fusion.write
    def scoped(p,v):
        if Path(p)==path/'fused/result.json':
            v={**v,'complete_full_sequence':False,'raw_window_complete':False,
               'pipeline_scope':'25 cached SAM3 views on fixed full geometry; fresh fusion and separately measured naming; no new SLAM/SAM3',
               'new_SLAM_inference':False,'new_SAM3_inference':False}
        native_write(p,v)
    fusion.write=scoped
    fusion.main(argparse.Namespace(arm_root=path))
    geom=read(path/'mapping/mapping_result.json'); out=path/'fused'
    write_artifact_manifest(out,map_path=out/'export/map_labeled.ply',classes_path=out/'classes.json',
        result_path=out/'result.json',manifest_path=Path(geom['manifest']),trajectory_path=Path(geom['trajectory']))
    write(path/'FUSION_RESOURCE.json',{'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


def store(path):
    import numpy as np
    started=time.monotonic(); ctx=read(path/'CONTEXT.json'); scene=ctx['scene']
    with np.load(path/'fused/map_labels.npz') as z:
        inst=z['instance']; sem=z['semantic']
    with np.load(path/'fused/target.npz') as z: xyz=z['xyz']
    poses=read(ROOT/'inputs'/scene/'POSES.json')
    centres={int(i):xyz[inst==i].mean(0) for i in np.unique(inst) if i>0}
    all_crops=[]
    for task in read(path/'semantic/CROP_TASKS.json'):
        with np.load(path/'fused'/f'projection_{task["frame_id"]:06}.npz') as z:
            point_ids=z['point_ids']; mask_ids=z['mask_ids']
        for original in task['crops']:
            crop=dict(original)
            ids=point_ids[mask_ids==crop['mask_id']]
            a=association(inst[ids]); crop['association']=a
            if a['object_id']:
                direction=np.asarray(poses[str(task['frame_id'])])[:3,3]-centres[a['object_id']]
                direction=direction/max(float(np.linalg.norm(direction)),1e-12)
                crop['view_direction']=direction.tolist()
            all_crops.append(crop)
    spec=model_spec(plan()['vlm'])
    identity=cache_identity(scene,sha(path/'fused/target.npz'),sha(path/'fused/map_labels.npz'),
                            spec['revision'],hashlib.sha256(PROMPT.encode()).hexdigest(),[c['sha256'] for c in all_crops])
    write(path/'OBJECT_STORE.json',{'identity':identity,'crops':all_crops,
           'scope':'predicted final objects, no GT; metadata evidence only','seconds':time.monotonic()-started})
    return all_crops


def name(path):
    from pose_pipeline.semantic_runtime.vlm import create_namer
    import torch
    started=time.monotonic();arm=read(path/'CONTEXT.json')['arm'];out=path/'semantic/vlm'
    if arm not in ('A', 'B'):
        raise ValueError('Unsupported arm: ' + arm)
    out.mkdir(exist_ok=False)
    runtime=read(plan()['runtime_file']);model_id=plan()['vlm']
    all_crops=([c for t in read(path/'semantic/CROP_TASKS.json') for c in t['crops']] if arm=='A'
               else read(path/'OBJECT_STORE.json')['crops'])
    for c in all_crops:assert sha(c['file'])==c['sha256']
    model=create_namer(model_id,{**runtime['models'][model_id], 'threads':2,'log_dir':str(out)})
    write(out/'MODEL.json',model.audit);write(out/'INPUTS.json',{'prompt':PROMPT,'model':model_id,'GT_in_requests':False})
    records=[];groups_audit=[];seen=set()
    def infer(c):
        key=crop_key(c)
        assert key not in seen;seen.add(key)
        result={**c,**model.infer(c['file'])};records.append(result)
        return result
    if arm=='A':
        for c in all_crops:infer(c)
    else:
        grouped=defaultdict(list)
        for c in all_crops:
            if c['association']['eligible']:
                grouped[c['association']['object_id']].append(c)
        for oid,crops in sorted(grouped.items()):
            if len({c['frame_id'] for c in crops})<2:
                groups_audit.append({'object_id':oid,'skip':'fewer_than_2_distinct_frames'});continue
            initial=representatives(crops,3);evidence=[infer(c) for c in initial]
            fallback=voted_name(evidence)=='unknown'
            if fallback:
                for c in crops:
                    if crop_key(c) not in seen:evidence.append(infer(c))
            groups_audit.append({'object_id':oid,'eligible_crops':len(crops),'initial':[crop_key(c) for c in initial],
                'fallback':fallback,'calls':len(evidence),'name':voted_name(evidence)})
    completed=time.monotonic()
    write(out/'response_000000.json',{'task_id':0,'completed_at':completed,'crops':records})
    write(out/'RECORDS.json',records)
    write(out/'COMPLETE.json',{'status':'completed','model':model_id,'crops':len(records),'crops_executed':sum(bool(c['executed']) for c in records),
          'seconds_including_load':completed-started,'inference_seconds':sum(c['request_seconds'] for c in records),
          'groups':groups_audit,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
    model.close()


def worker(python,mode,path):
    stream=(path/f'{mode}.log').open('w')
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONPATH':str(ROOT/'src'),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    proc=subprocess.Popen([str(python),'-u',str(Path(__file__).resolve()),mode,'--path',str(path)],stdout=stream,stderr=subprocess.STDOUT,env=env)
    return proc,stream,time.monotonic()


def wait(job):
    proc,stream,started=job
    code=proc.wait();stream.close()
    if code:raise RuntimeError('worker failed '+str(proc.args)+' code '+str(code))
    return time.monotonic()-started


def run():
    from pose_pipeline.semantic_runtime import backfill
    import numpy as np
    runtime=read(plan()['runtime_file']);vlm_python=runtime['models'][plan()['vlm']].get('python',runtime['vlm_python'])
    verify(read(ROOT/'INPUT_LOCK.json'))
    cpu_python = Path(plan().get('cpu_python', sys.executable))
    rows=[]
    for repeat,order in enumerate(plan()['repeat_orders'],1):
        for scene in plan()['scenes']:
            for arm in order:
                path=setup(scene,arm,repeat);start=time.monotonic();times={}
                if arm=='A':
                    naming=worker(vlm_python,'name',path)
                    times['fusion_process_seconds']=wait(worker(cpu_python,'fuse',path))
                    times['naming_process_elapsed_until_join']=wait(naming)
                else:
                    times['fusion_process_seconds']=wait(worker(cpu_python,'fuse',path))
                    store(path)
                    times['naming_process_seconds']=wait(worker(vlm_python,'name',path))
                backfill.main(argparse.Namespace(arm_root=path))
                tail=time.monotonic()-start
                complete=read(path/'semantic/vlm/COMPLETE.json')
                names=read(path/'fused/instance_names.json')
                record={'scene':scene,'arm':arm,'repeat':repeat,'tail_wall_seconds':tail,
                        'vlm_calls':complete['crops'],'vlm_inference_seconds':complete['inference_seconds'],
                        'named_objects':sum(x['vlm_name']!='unknown' for x in names),'objects':len(names),
                        'vlm_peak_reserved_mib':complete['peak_reserved_mib'],**times}
                write(path/'TIMING.json',record);rows.append(record);write(ROOT/'TIMINGS.json',rows)
                print('DONE',json.dumps(record),flush=True)
    # All geometry, semantics, confidence and instance arrays must match per scene.
    parity=[]
    for scene in plan()['scenes']:
        with np.load(ROOT/'runs/repeat-1'/scene/'A/fused/map_labels.npz') as z:
            baseline={k:z[k] for k in z.files}
        for repeat in (1,2):
            for arm in ('A','B'):
                path=ROOT/'runs'/f'repeat-{repeat}'/scene/arm
                with np.load(path/'fused/map_labels.npz') as z:
                    passed={k:bool(np.array_equal(v,z[k])) for k,v in baseline.items()}
                assert all(passed.values())
                parity.append({'scene':scene,'repeat':repeat,'arm':arm,'parity':passed})
        if scene in plan()['development_scenes']:
            with np.load(source(scene)/'fused/map_labels.npz') as z:
                assert all(np.array_equal(v,z[k]) for k,v in baseline.items())
    write(ROOT/'MAP_PARITY.json',parity)
    verify(read(ROOT/'INPUT_LOCK.json'))
    outputs={str(p):sha(p) for p in (ROOT/'runs').rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.ply')}
    write(ROOT/'PREDICTIONS_LOCK.json',outputs)
    write(ROOT/'PREDICTIONS_COMPLETE.json',{'status':'completed','prediction_files':len(outputs),'inputs_unchanged':True,'GT_used':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','fuse','name']);parser.add_argument('--path',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run()
    else:globals()[args.mode](args.path)
