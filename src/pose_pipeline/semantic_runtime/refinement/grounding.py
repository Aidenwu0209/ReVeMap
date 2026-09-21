"""Independent text-mask confirmation and point-evidence-only semantic revisions."""
from pathlib import Path
import os,sys,argparse,time,hashlib
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
R = None  # Explicit workspace supplied by the CLI.
import numpy as np
from pose_pipeline.semantic_runtime.common import read,write,sha
from .normalization import canonicalize_name,normalize_name
from .observations import SETTINGS

def erode(m):
 out=np.zeros_like(m);out[1:-1,1:-1]=m[1:-1,1:-1]&m[:-2,1:-1]&m[2:,1:-1]&m[1:-1,:-2]&m[1:-1,2:];return out

def query_plan():
 tasks={}
 for ob in read(R/'DECISIONS.json'):
  old=canonicalize_name(ob['original_name']);frames={}
  for mode,vote in ob['votes'].items():
   name=canonicalize_name(vote['name'])
   if name=='unknown' or normalize_name(vote['name'])['is_part'] or name==old:continue
   # Text uses the consensus category, with separate old-class counter-evidence.
   for fid in vote['frames']:
    frames.setdefault((fid,name,'fusion'),set()).add(mode)
   for view in ob['validation']:frames.setdefault((view['frame_id'],name,'validation'),set()).add(mode)
  for (fid,name,role),modes in frames.items():
   key=(ob['scene'],fid);items=tasks.setdefault(key,{})
   items.setdefault(name,[]).append({'instance_id':ob['instance_id'],'new_name':name,'old_name':old,'role':role,'modes':sorted(modes),'old_semantic_id':ob['semantic_id']})
   if old!='unknown':items.setdefault(old,[])
 return tasks

def ground(witness=False):
 from PIL import Image
 import torch
 cfg=read(R/'runtime.json');sys.path.insert(0,cfg['sam3_source'])
 from pose_pipeline.sam3_mapping import load_model
 tasks=query_plan();out=R/('witness/grounding' if witness else 'grounding');out.mkdir(parents=True,exist_ok=False)
 if witness:tasks=dict(list(sorted(tasks.items()))[:1])
 processor,audit=load_model(Path(cfg['sam3_checkpoint']),cfg['sam3_sha256']);write(out/'MODEL.json',audit)
 records=[];start=time.perf_counter();image_count=0
 for (scene,fid),names in sorted(tasks.items()):
  base=np.load(R/'inputs'/scene/'base.npz');inst=base['instance'];p=R/'objects'/scene/'frames'/f'{fid:06}.npz';z=np.load(p);ids,v,u,shape=z['point_ids'],z['row'],z['col'],tuple(z['depth_shape']);im_path=p.with_suffix('.png');image=Image.open(im_path).convert('RGB')
  target=out/scene;target.mkdir(parents=True,exist_ok=True)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=processor.set_image(image);image_count+=1
   for name,uses in sorted(names.items()):
    processor.reset_all_prompts(state);tick=time.perf_counter();res=processor.set_text_prompt(state=state,prompt=name);torch.cuda.synchronize();seconds=time.perf_counter()-tick
    masks=res['masks'].cpu().numpy();scores=res['scores'].float().cpu().numpy();data={'scores':scores,'visible':ids};packed=[]
    for i,(mask,score) in enumerate(zip(masks,scores)):
     mask=mask.reshape(image.height,image.width)>0
     if mask.shape!=shape:mask=np.asarray(Image.fromarray(mask).resize((shape[1],shape[0]),Image.Resampling.NEAREST),bool)
     data[f'points_{i}']=np.unique(ids[erode(mask)[v,u]]);packed.append(np.packbits(mask.reshape(-1)))
    data['packed_masks']=np.asarray(packed,np.uint8).reshape(len(packed),-1) if packed else np.empty((0,(np.prod(shape)+7)//8),np.uint8);data['depth_shape']=shape
    path=target/f'{fid:06}_{hashlib.sha256(name.encode()).hexdigest()[:12]}.npz';np.savez_compressed(path,**data)
    record={'scene':scene,'frame_id':fid,'name':name,'uses':uses,'file':str(path.relative_to(R)),'sha256':sha(path),'image_sha256':sha(im_path),'projection_sha256':sha(p),'seconds':seconds,'masks':len(scores)};records.append(record)
    write(out/'PROGRESS.json',{'queries':len(records),'images':image_count,'last':record});print('GROUND',scene,fid,name,len(scores),flush=True)
   del state,res;torch.cuda.empty_cache()
 write(out/'RECORDS.json',records);write(out/'COMPLETE.json',{'status':'completed','queries':len(records),'images':image_count,'seconds_excluding_load':time.perf_counter()-start,'GT_used':False,'peak_cuda_MiB':torch.cuda.max_memory_allocated()/1024**2})

def assess(points,visible,inst,oid,score):
 anchor=visible[inst[visible]==oid];shared=np.intersect1d(points,anchor);coverage=len(shared)/max(1,len(anchor));purity=len(shared)/max(1,len(points));iou=len(shared)/max(1,len(points)+len(anchor)-len(shared))
 return {'score':float(score),'coverage':coverage,'purity':purity,'iou':iou,'quality':coverage*purity,'own_points':shared,'mask_points':points}
