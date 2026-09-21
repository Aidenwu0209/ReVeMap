"""One fresh SAM3 prediction per frame; nested sampling arms share predictions."""
from pathlib import Path
import os,sys,json,time,traceback,hashlib,subprocess
R=Path(os.environ['EXPERIMENT_SCENE_DIR']);CODE=Path('/mnt/d/SGF-SGA-experiments/scannet0030_frame_ablation_20260912_v1/code')
A=Path('/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1')
B=Path('/mnt/d/SGF-SGA-experiments/sam3_sga_20260912_v1')
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path[:0]=[str(CODE/'src'),str(A/'sam3_official')]
import numpy as np
from pose_pipeline.contracts import load_manifest,load_trajectory,bind_manifest_trajectory,sha256_file
from pose_pipeline.sam3_mapping import load_model,read_frame,save_overlay
from pose_pipeline.sam3_refine import infer_claims,resolve_families,interior,FAMILY_NAMES
from pose_pipeline.sam3_fusion import visible_map_pixels

def write(name,value):
 p=R/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def main():
 started=time.perf_counter();plan=json.loads((R/'PLAN.json').read_text());job=plan['job'];N=plan['total_frames'];key=job['key']
 m=load_manifest(Path(job['manifest']));poses,_=load_trajectory(Path(job['trajectory']));bound=bind_manifest_trajectory(m,poses)
 assert len(bound)==len(m.frames)==N
 assert [f.frame_id for f,p in bound]==list(range(N))
 assert all(f.color_path.is_file() and f.depth_path.is_file() for f,p in bound)
 xyz=np.load(job['target'])['xyz']
 prior=np.load(B/'priors'/f"{key.replace('/','__')}.npz");assert str(prior['xyz_sha256'])==hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest()
 inputs={str(p):sha256_file(p) for p in [Path(job['manifest']),Path(job['trajectory']),Path(job['target']),B/'priors'/f"{key.replace('/','__')}.npz",CODE/'configs/sam3_indoor_v1.json',A/'env-spec.json']}
 assert inputs[str(A/'env-spec.json')]=='bf541ae39b5f33d05f5ec938944551a96bf0a6dc76c56846c0de8677c1930f96'
 write('INPUTS.json',inputs)
 loadstart=time.perf_counter();processor,audit=load_model(A/'sam3.pt',plan['checkpoint_sha256']);write('MODEL.json',{**audit,'cold_load_and_checkpoint_hash_seconds':time.perf_counter()-loadstart})
 tax=json.loads((CODE/'configs/sam3_indoor_v1.json').read_text())['classes']
 (R/'frames').mkdir(exist_ok=True);done=set();rows=[];fresh_infer=0
 for stride in [20,10,5]:
  selected=list(range(0,N,stride))
  if selected[-1]!=(N-1):selected.append((N-1))
  for fid in selected:
   if fid in done:continue
   frame,pose=bound[fid];tick=time.perf_counter();image,depth,K=read_frame(frame)
   ids,v,u=visible_map_pixels(xyz,pose.t_world_camera,K,depth.astype(float)/m.depth_scale)
   prep=time.perf_counter()-tick;t=time.perf_counter()
   claims,packed=infer_claims(processor,image,depth.shape,tax)
   inference=time.perf_counter()-t;fresh_infer+=inference
   sem,inst,conf,repairs,raw=resolve_families(claims,v,u,prior['semantic'][ids],prior['confidence'][ids])
   path=R/'frames'/f'{fid:06}.npz'
   np.savez_compressed(path,semantic=sem,local_instance=inst,confidence=conf.astype(np.float16),visible_map_ids=ids,projected_semantic=sem[v,u],projected_local_instance=inst[v,u],projected_confidence=conf[v,u],interior=interior(sem)[v,u],raw_semantic=raw[0][v,u],raw_confidence=raw[2][v,u],raw_interior=interior(raw[0])[v,u],raw_masks_packed=packed,depth_shape=depth.shape)
   equality=None
   old=B/'family_v3'/key/'frames'/path.name
   if old.exists():
    with np.load(old) as z:
     equality={k:bool(np.array_equal(z[k],value)) for k,value in {'semantic':sem,'local_instance':inst,'visible_map_ids':ids,'projected_semantic':sem[v,u],'projected_local_instance':inst[v,u],'projected_confidence':conf[v,u],'interior':interior(sem)[v,u]}.items()}
    if not all(equality.values()):raise RuntimeError(f'Shared-frame prediction/projection drift at {fid}: {equality}')
   row={'frame_id':fid,'color_sha256':sha256_file(frame.color_path),'depth_sha256':sha256_file(frame.depth_path),'cache_sha256':sha256_file(path),'masks':claims.records,'old_cache_equal':equality,'prepare_seconds':prep,'inference_seconds':inference,'total_frame_seconds':time.perf_counter()-tick,'depth_consistent_points':len(ids)}
   with (R/'frames.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   done.add(fid);rows.append(row)
   write('STATUS.json',{'status':'running','phase_stride':stride,'processed_unique_frames':len(done),'planned_unique_frames':len(set(range(0,N,5))|{N-1}),'total_raw_frames':N,'wall_seconds':time.perf_counter()-started,'inference_seconds':fresh_infer,'latest_frame':fid})
   if fid in {0,(N//4)//20*20,(N//2)//20*20,(3*N//4)//20*20,N-1}:image.save(R/'frames'/f'{fid:06}_rgb.jpg');save_overlay(image,sem,tax+[{'id':k,'name':v} for k,v in FAMILY_NAMES.items()],R/'frames'/f'{fid:06}_overlay.png')
   if len(done)%25==0:print(json.dumps({'done':len(done),'stride_phase':stride,'wall_seconds':time.perf_counter()-started}),flush=True)
  write(f'READY_stride{stride}.json',{'selected_frame_ids':selected,'frames':len(selected),'new_inference':True,'input_prediction_cache_shared':True,'sum_frame_seconds':sum(r['total_frame_seconds'] for r in rows if r['frame_id'] in set(selected)),'sum_inference_seconds':sum(r['inference_seconds'] for r in rows if r['frame_id'] in set(selected))})
  print('ARM_READY',stride,flush=True)
 assert len(done)==len(set(range(0,N,5))|{N-1})
 assert all(sha256_file(Path(p))==h for p,h in inputs.items())
 write('STATUS.json',{'status':'completed','processed_unique_frames':len(done),'planned_unique_frames':len(set(range(0,N,5))|{N-1}),'total_raw_frames':N,'wall_seconds':time.perf_counter()-started,'inference_seconds':fresh_infer,'input_hashes_verified':True})
if __name__=='__main__':
 try:main()
 except Exception:
  write('FAILURE.json',{'status':'failed','error':traceback.format_exc()});raise
