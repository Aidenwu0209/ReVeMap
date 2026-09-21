"""Controlled object observations and measured semantic revision; never reads GT."""
from pathlib import Path
import os, sys, json, time, argparse, hashlib
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
R = None  # Explicit workspace supplied by the CLI.
import numpy as np
from pose_pipeline.semantic_runtime.common import read, write, sha
from .normalization import canonicalize_name, normalize_name

SETTINGS = {'candidate_stride':20, 'min_object_points':50, 'min_visible_points':30,
 'min_visible_fraction':.15,'bbox_min_side':10,'context_fraction':.15,
 'views':3,'heldout_views':1,'distinct_translation_m':.12,'distinct_rotation_deg':8.,
 'ground_score':.5,'ground_coverage':.5,'ground_iou':.4,'ground_purity':.75,
 'known_score':.65,'known_coverage':.6,'known_iou':.45,'known_advantage':.15,
 'known_frames':3,'unknown_frames':2,'point_support_frames':2}

def distinct(a,b):
 d=np.linalg.norm(np.asarray(a)[:3,3]-np.asarray(b)[:3,3])
 angle=np.degrees(np.arccos(np.clip(np.asarray(a)[:3,2]@np.asarray(b)[:3,2],-1,1)))
 return d>=SETTINGS['distinct_translation_m'] or angle>=SETTINGS['distinct_rotation_deg']

def choose(candidates, quality):
 selected=[]
 order=sorted(candidates,key=(lambda x:(-x['quality'],x['frame_id'])) if quality else lambda x:x['frame_id'])
 for v in order:
  if any(abs(v['frame_id']-s['frame_id'])<20 or not distinct(v['pose'],s['pose']) for s in selected):continue
  selected.append(v)
  if len(selected)==SETTINGS['views']:break
 return selected

def prepare(scene):
 from pose_pipeline.contracts import load_manifest,load_trajectory,bind_manifest_trajectory
 from .registered_input import RegisteredInput
 from pose_pipeline.sam3_fusion import visible_map_pixels
 from PIL import Image
 import cv2
 tick=time.perf_counter();out=R/'objects'/scene;out.mkdir(parents=True,exist_ok=False)
 inp=R/'inputs'/scene;spec=read(inp/'INPUT.json');m=load_manifest(inp/'manifest.json');ps,_=load_trajectory(inp/'trajectory.json');bound=bind_manifest_trajectory(m,ps)
 reader=RegisteredInput(scene, spec);read_frame=reader.read
 xyz=np.load(inp/'target.npz')['xyz'];base=np.load(inp/'base.npz');inst=base['instance'];sem=base['semantic'];classes=read(inp/'classes.json')
 assert len(xyz)==len(inst)
 objects=[]
 for oid in np.unique(inst):
  if oid<=0:continue
  on=inst==oid;sid,c=np.unique(sem[on],return_counts=True);sid=int(sid[c.argmax()]);size=int(on.sum())
  if size<SETTINGS['min_object_points'] or sid in (10,19):continue
  objects.append({'instance_id':int(oid),'point_count':size,'semantic_id':sid,'original_name':classes[str(sid)],'candidates':[]})
 ids_objects={x['instance_id']:x for x in objects};byframe={f.frame_id:(f,p) for f,p in bound}
 frames=bound[::SETTINGS['candidate_stride']]
 if frames[-1][0].frame_id!=bound[-1][0].frame_id:frames.append(bound[-1])
 inputs={str(inp/n):sha(inp/n) for n in ['INPUT.json','manifest.json','trajectory.json','target.npz','base.npz','classes.json']}
 frame_rows=[]
 for f,p in frames:
  image,depth,K=read_frame(f);h,w=depth.shape;ids,v,u=visible_map_pixels(xyz,p.t_world_camera,K,depth.astype(float)/m.depth_scale)
  # The full-frame quality is used only to rank observations, never as a label.
  gray=cv2.cvtColor(np.array(image.resize((320,240))),cv2.COLOR_RGB2GRAY);sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
  unique,cnt=np.unique(inst[ids],return_counts=True)
  for oid,count in zip(unique,cnt):
   if int(oid) not in ids_objects:continue
   ob=ids_objects[int(oid)];fraction=count/ob['point_count']
   if count<SETTINGS['min_visible_points'] or fraction<SETTINGS['min_visible_fraction']:continue
   on=inst[ids]==oid;xx=u[on];yy=v[on];bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)];bw=bbox[2]-bbox[0];bh=bbox[3]-bbox[1]
   if min(bw,bh)<SETTINGS['bbox_min_side'] or bw*bh/(h*w)>.8:continue
   quality=float(fraction*np.sqrt(bw*bh/(h*w))*(.5+.5*sharp/(sharp+50)))
   ob['candidates'].append({'frame_id':f.frame_id,'visible_points':int(count),'visible_fraction':float(fraction),'bbox_depth':bbox,'depth_shape':[h,w], 'sharpness':sharp,'quality':quality,'pose':p.t_world_camera.tolist()})
  frame_rows.append({'frame_id':f.frame_id,'visible_points':len(ids),'sharpness':sharp})
  inputs[str(f.color_path)]=sha(f.color_path);inputs[str(f.depth_path)]=sha(f.depth_path)
  if len(frame_rows)%30==0:print('PREPARE',scene,len(frame_rows),len(frames),flush=True)
 plans=[];needed=set()
 for ob in objects:
  temporal=choose(ob['candidates'],False);quality=choose(ob['candidates'],True)
  # Validation is excluded from both policies and needs a genuinely different view.
  used=temporal+quality;hold=[]
  for v in sorted(ob['candidates'],key=lambda x:(-x['quality'],x['frame_id'])):
   if all(abs(v['frame_id']-q['frame_id'])>=20 and distinct(v['pose'],q['pose']) for q in used):hold=[v];break
  plan={**{k:v for k,v in ob.items() if k!='candidates'},'candidate_count':len(ob['candidates']),'temporal':temporal,'quality':quality,'validation':hold}
  plan['ready']=len(temporal)>=2 and len(quality)>=2;plans.append(plan)
  for v in temporal+quality+hold:needed.add(v['frame_id'])
 (out/'frames').mkdir();(out/'crops').mkdir()
 crop_index=[]
 for fid in sorted(needed):
  f,p=byframe[fid];image,depth,K=read_frame(f);ids,v,u=visible_map_pixels(xyz,p.t_world_camera,K,depth.astype(float)/m.depth_scale);h,w=depth.shape
  image_path=out/'frames'/f'{fid:06}.png';image.save(image_path)
  projection=out/'frames'/f'{fid:06}.npz';np.savez_compressed(projection,point_ids=ids,row=v,col=u,depth_shape=depth.shape)
  for ob in plans:
   views={x['frame_id']:x for mode in ['temporal','quality','validation'] for x in ob[mode]}
   if fid not in views:continue
   view=views[fid];x0,y0,x1,y1=view['bbox_depth'];dx=max(2,round((x1-x0)*.15));dy=max(2,round((y1-y0)*.15))
   box=[max(0,round((x0-dx)*image.width/w)),max(0,round((y0-dy)*image.height/h)),min(image.width,round((x1+dx)*image.width/w)),min(image.height,round((y1+dy)*image.height/h))]
   crop=out/'crops'/f'{ob["instance_id"]}_{fid:06}.png';image.crop(box).save(crop)
   crop_index.append({'scene':scene,'instance_id':ob['instance_id'],'frame_id':fid,'file':str(crop.relative_to(R)),'sha256':sha(crop),'image':str(image_path.relative_to(R)),'image_sha256':sha(image_path),'projection':str(projection.relative_to(R)),'projection_sha256':sha(projection),'bbox_rgb':box})
 write(out/'VIEW_PLAN.json',{'scene':scene,'settings':SETTINGS,'objects':plans,'source':spec,'candidate_frame_count':len(frames),'GT_used':False})
 write(out/'CROP_INDEX.json',crop_index);write(out/'INPUTS.json',inputs)
 write(out/'RGB_REGISTRATION.json',reader.audit)
 write(out/'PREPARED.json',{'objects':len(plans),'ready':sum(o['ready'] for o in plans),'unique_crops':len(crop_index),'unique_images':len(needed),'seconds':time.perf_counter()-tick,'GT_used':False})
 print('PREPARED',scene,read(out/'PREPARED.json'),flush=True)

def name_all(scenes, witness=False):
 from pose_pipeline.semantic_runtime.vlm import create_namer
 cfg=read(R/'runtime.json');output=R/('witness/naming' if witness else 'naming');output.mkdir(parents=True,exist_ok=False)
 rows=[x for s in scenes for x in read(R/'objects'/s/'CROP_INDEX.json')]
 if witness:rows=rows[:1]
 model=create_namer('qwen3vl_2b_nf4',cfg['models']['qwen3vl_2b_nf4']);write(output/'MODEL.json',model.audit)
 cache={};results=[];start=time.perf_counter()
 try:
  for i,row in enumerate(rows):
   path=R/row['file'];assert sha(path)==row['sha256']
   reused=row['sha256'] in cache
   if not reused:cache[row['sha256']]=model.infer(path)
   answer=cache[row['sha256']];rec={**row,**answer,'canonical':normalize_name(answer['label']),'request_reused':reused};results.append(rec)
   write(output/f'{i:05}.json',rec)
   if i%50==0:print('NAME',i,len(rows),rec['label'],flush=True)
 finally:model.close()
 write(output/'RECORDS.json',results);write(output/'COMPLETE.json',{'status':'completed','rows':len(rows),'actual_requests':len(cache),'seconds_excluding_load':time.perf_counter()-start,'GT_used':False})

def decisions(scenes):
 from collections import Counter
 rows=read(R/'naming/RECORDS.json');lookup={(x['scene'],x['instance_id'],x['frame_id']):x for x in rows};result=[]
 for scene in scenes:
  for ob in read(R/'objects'/scene/'VIEW_PLAN.json')['objects']:
   votes={}
   for mode in ['temporal','quality']:
    for norm in [False,True]:
     names=[lookup[scene,ob['instance_id'],v['frame_id']]['label'] for v in ob[mode]]
     labels=[canonicalize_name(n) if norm else n for n in names];cnt=Counter(n for n in labels if n!='unknown');r=cnt.most_common()
     name=r[0][0] if r and r[0][1]>=2 and (len(r)==1 or r[0][1]>r[1][1]) else 'unknown'
     votes[f'{mode}_{"canonical" if norm else "raw"}']={'name':name,'labels':labels,'frames':[v['frame_id'] for v in ob[mode]],'support_frames':[v['frame_id'] for v,n in zip(ob[mode],labels) if n==name] if name!='unknown' else []}
   result.append({**ob,'scene':scene,'votes':votes})
 write(R/'DECISIONS.json',result)
 return result
