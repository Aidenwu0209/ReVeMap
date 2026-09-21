"""Recompute existing semantic and guided instance pipeline at a fixed stride.
No GT consumed. All three arms use identical source functions and thresholds.
"""
from pathlib import Path
import os,sys,json,time,argparse,hashlib,resource
R=Path(os.environ['EXPERIMENT_SCENE_DIR']);CODE=Path('/mnt/d/SGF-SGA-experiments/scannet0030_frame_ablation_20260912_v1/code')
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(CODE/'src'))
import numpy as np
from pose_pipeline.sam3_fusion import MapVotes,GeometricInstances
from pose_pipeline.sam3_sga import select_tracks,compatible,measure_candidates,greedy_pairs,object_consensus
from sam3_multiview_blas import fuse_instances
from pose_pipeline.sam3_guided import recover_instances
from pose_pipeline.contracts import sha256_file
B=Path('/mnt/d/SGF-SGA-experiments/sam3_sga_20260912_v1')
G=Path('/mnt/d/SGF-SGA-experiments/sam3_guided_20260912_v1')

def default(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
def write(path,data):path.write_text(json.dumps(data,indent=2,default=default)+'\n')
def inv(xyz,labels):
 objects=[]
 for k in np.unique(labels['instance']):
  if k<=0:continue
  ix=np.flatnonzero(labels['instance']==k);p=xyz[ix];cs,ns=np.unique(labels['semantic'][ix],return_counts=True)
  objects.append({'instance_id':int(k),'semantic_id':int(cs[ns.argmax()]),'point_count':len(ix),'semantic_histogram':dict(zip(map(str,cs),map(int,ns))),'center':p.mean(0).tolist(),'min':p.min(0).tolist(),'max':p.max(0).tolist()})
 return objects

def main(stride):
 started=time.perf_counter();ready=json.loads((R/f'READY_stride{stride}.json').read_text());selected=ready['selected_frame_ids']
 O=R/f'stride{stride}';O.mkdir(exist_ok=False)
 plan=json.loads((R/'PLAN.json').read_text());job=plan['job'];key=job['key'];xyz=np.load(job['target'])['xyz'];n=len(xyz)
 rawvotes=MapVotes(n,33);votes=MapVotes(n,35);rawhigh=np.zeros(n,np.float32);high=np.zeros(n,np.float32)
 streams=[GeometricInstances(),GeometricInstances()];frames=[];inputs={}
 rows={r['frame_id']:r for r in map(json.loads,(R/'frames.jsonl').read_text().splitlines())}
 for ordinal,fid in enumerate(selected):
  p=R/'frames'/f'{fid:06}.npz';inputs[str(p)]=sha256_file(p);assert inputs[str(p)]==rows[fid]['cache_sha256']
  with np.load(p) as z:
   f={k:z[v].copy() for k,v in {'point_ids':'visible_map_ids','mask_ids':'projected_local_instance','semantic':'projected_semantic','confidence':'projected_confidence','interior':'interior'}.items()};f['frame_id']=fid
   ids=f['point_ids'];rawvotes.add(fid,ids,z['raw_semantic'],z['raw_confidence']);votes.add(fid,ids,f['semantic'],f['confidence'])
   take=z['raw_interior'];rawhigh[ids[take]]=np.maximum(rawhigh[ids[take]],z['raw_confidence'][take])
  safe=f['interior'];high[ids[safe]]=np.maximum(high[ids[safe]],f['confidence'][safe])
  streams[ordinal%2].add(fid,ids[safe],f['mask_ids'][safe],f['semantic'][safe]);frames.append(f)
  if (ordinal+1)%100==0:print('loaded',stride,ordinal+1,flush=True)
 rawsem,rawconf,_=rawvotes.finalize();single=(rawsem==0)&(rawvotes.counts.sum(1)==1)&(rawhigh>=.9)
 rawsem[single]=rawvotes.scores.argmax(1)[single];rawconf[single]=rawhigh[single]
 sem,conf,_=votes.finalize();single=(sem==0)&(votes.counts.sum(1)==1)&(high>=.9)
 sem[single]=votes.scores.argmax(1)[single];conf[single]=high[single]
 known=rawsem>0;sem[known]=rawsem[known];conf[known]=rawconf[known]
 base={'semantic':sem,'confidence':conf};trackrecords=[]
 for tr in streams:
  trackrecords.append([{'track_id':i+1,'category':t['category'],'frames':sorted(t['frames']),'points':sorted(t['points']),'mean_point_score':float(high[np.array(sorted(t['points']),np.int64)].mean())} for i,t in enumerate(tr.tracks)])
 if stride==20:
  assert selected==json.loads((B/'family_v3'/key/'result.json').read_text())['selected_frame_ids']
  for i in range(2):assert trackrecords[i]==json.loads((B/'family_v3'/key/f'stream_{i}_tracks.json').read_text()),f'stream{i} differs'
  with np.load(B/'family_v3'/key/'map_labels.npz') as z:
   assert np.array_equal(sem,z['semantic']) and np.array_equal(conf,z['confidence']),'family baseline differs'
 for i,rows2 in enumerate(trackrecords):write(O/f'stream_{i}_tracks.json',rows2)
 chosen=[select_tracks(t) for t in trackrecords]
 if min(map(len,chosen))<2:pairs=[]
 else:
  candidates=[(a['track_id'],b['track_id'],0.) for a in chosen[0] for b in chosen[1] if compatible(a['category'],b['category'])]
  measured=measure_candidates(candidates,*chosen,xyz);pairs=greedy_pairs(measured,'geometry')
 c,objects,cm=object_consensus(trackrecords,pairs,base,xyz)
 write(O/'geometric_consensus.json',{'pairs':pairs,'metrics':cm})
 np.savez_compressed(O/'C_map_labels.npz',**c)
 if stride==20:
  with np.load(B/'objects_geometry'/key/'map_labels.npz') as z:
   assert all(np.array_equal(z[k],v) for k,v in c.items()),'C baseline differs'
 t=time.perf_counter();cfg={'min_mask_points':30,'min_output_points':50,'min_point_views':1,'min_group_frames':2,'object_score_mode':'max_point'}
 current,audit=fuse_instances(n,frames,c['semantic'].copy(),config=cfg);multi_seconds=time.perf_counter()-t
 write(O/'multiview_audit.json',audit);np.savez_compressed(O/'R3_map_labels.npz',**{**c,'instance':current})
 blocked=[(int(r['frame_id']),int(r['mask_id'])) for r in audit['filtered_masks']]
 t=time.perf_counter();instances,ga,provenance=recover_instances(n,frames,c['semantic'].copy(),c['instance'].copy(),current,blocked_masks=blocked);guide_seconds=time.perf_counter()-t
 assert np.array_equal(instances[current>0],current[current>0]);assert np.all(provenance['distinct_support_frames']>=2)
 labels={**c,'instance':instances};np.savez_compressed(O/'map_labels.npz',**labels)
 write(O/'objects.json',inv(xyz,labels));write(O/'guided_audit.json',ga);np.savez_compressed(O/'recovered_provenance.npz',**provenance)
 (O/'classes.json').write_bytes((B/'family_v3'/key/'classes.json').read_bytes())
 equality=None
 if stride==20:
  with np.load(G/'guided_recovery_maskveto'/key/'map_labels.npz') as z:equality={k:bool(np.array_equal(z[k],v)) for k,v in labels.items()}
  assert all(equality.values()),f'guided baseline differs {equality}'
 assert all(sha256_file(Path(p))==h for p,h in inputs.items())
 result={'status':'completed','key':job['key'],'stride':stride,'processed_frames':len(selected),'selected_frame_ids':selected,'complete_full_sequence':len(selected)==plan['total_frames'],'geometry_xyz_sha256':hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest(),'map_points':n,'semantic_coverage':float(np.mean(labels['semantic']>0)),'instance_coverage':float(np.mean(instances>0)),'wallfloor_excluded_instance_coverage':float(np.mean(instances[~np.isin(labels['semantic'],[10,19])]>0)),'old_stride20_exact_match':equality,'cached_frame_input_sha256':inputs,'exact_binary_count_blas':True,'environment':{'python':sys.executable,'numpy':np.__version__},'backend_seconds':time.perf_counter()-started,'multiview_seconds':multi_seconds,'guided_seconds':guide_seconds,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'time_definition':'CPU replay incl cache verification, voting, C association, R3, guided recovery, output; model/frame timing separately in READY receipt','ground_truth_consumed':False,'sga_inference_executed':False,'geometry_modified':False,'quality_accepted':False}
 write(O/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['cached_frame_input_sha256','selected_frame_ids']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stride',type=int,required=True);args=p.parse_args();main(args.stride)
