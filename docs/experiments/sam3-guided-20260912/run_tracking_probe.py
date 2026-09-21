"""Finite, frozen-input SAM3 image/video and assisted-visual probes on ssh44."""
from pathlib import Path
import argparse, ctypes, gc, hashlib, json, os, subprocess, sys, time, traceback

R = Path(__file__).resolve().parent
A = Path('/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1')
B = Path('/mnt/d/SGF-SGA-experiments/sam3_sga_20260912_v1')
sys.path[:0] = [str(R/'code/src'), str(B/'code/src'), str(A/'code/src'), str(A/'sam3_official')]
# R4 can be deployed as an additive module before the full parent source bundle.
# Resolve existing baseline modules normally, but prefer explicitly deployed R4 modules.
import pose_pipeline
pose_pipeline.__path__[:0] = [str(base/'code/src/pose_pipeline') for base in (R,B,A)]
import numpy as np
from PIL import Image, ImageDraw
from pose_pipeline.contracts import load_manifest, load_trajectory, bind_manifest_trajectory, sha256_file
from pose_pipeline.sam3_mapping import read_frame, load_model, write_json
from pose_pipeline.sam3_fusion import visible_map_pixels
from pose_pipeline.sam3_tracking_probe import depth_masks, matched_temporal_metrics, per_mask_temporal_metrics

CHECKPOINT_SHA = '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
SPEC_SHA = 'bf541ae39b5f33d05f5ec938944551a96bf0a6dc76c56846c0de8677c1930f96'
PLAN = {'version': 'tracking-probe-v1', 'seed': 42, 'inference_threshold': .5,
    'clips': [
        {'name': 'orbbec_curtain', 'key': 'orbbec/scan_20260909_142829_5ef1fa', 'frame_ids': list(range(1200,1440,5)), 'prompt': 'curtain'},
        {'name': 'orbbec_reel', 'key': 'orbbec/scan_20260909_142829_5ef1fa', 'frame_ids': list(range(2400,2480,5)), 'prompt': 'cable reel'},
        {'name': 'scannet_chair', 'key': 'scannet/scene0030_00', 'frame_ids': list(range(0,160,5)), 'prompt': 'chair'},
        {'name': '3rscan_bed', 'key': '3rscan/00d42bed-778d-2ac6-86a7-0e0e5f5f5660', 'frame_ids': [0,1,6,7,30,31], 'prompt': 'bed'}],
    'visual_prompt': {'clip': 'orbbec_reel', 'frame_id': 2400, 'rgb_size': [640,480],
                      'positive_box_xyxy': [35,0,355,309], 'negative_box_xyxy': [410,180,630,285],
                      'selection': 'raw RGB inspected and boxes fixed before new model inference',
                      'extra_supervision': True, 'no_business_class_name_inferred': True},
    'scope': '102 RGB-D observations, 4 single-concept clips; 64 Orbbec, 32 ScanNet, 6 archived 3RScan; not full-scene mapping',
    'comparison': 'same decoded RGB and frozen depth/poses/map for image vs video; image has no cross-frame memory',
    'depth_tolerance_m': .05, 'gt_consumed': False, 'quality_accepted': False,
    'interpretation': 'temporal agreement and coverage are diagnostic, not correctness; no full Any3DIS reproduction',
    'visual_arms': ['single_frame_text', 'single_frame_visual_positive', 'single_frame_visual_positive_negative', 'video_visual_positive'],
    'visual_video_prompt': 'one positive box on first frame; later frames have no additional human prompts',
    'timeout_seconds': 1800}


def code_provenance():
    official=A/'sam3_official'
    files={str(p):sha256_file(p) for p in official.rglob('*.py')}
    files[str(R/'run_tracking_probe.py')]=sha256_file(R/'run_tracking_probe.py')
    files[str(R/'code/src/pose_pipeline/sam3_tracking_probe.py')]=sha256_file(R/'code/src/pose_pipeline/sam3_tracking_probe.py')
    return {'source_files_sha256':files,
            'official_commit':subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip(),
            'official_tracked_changes':subprocess.check_output(['git','-C',str(official),'status','--porcelain','--untracked-files=no'],text=True).strip(),
            'env_spec_file_sha256':sha256_file(B/'env-spec.json'), 'checkpoint_sha256':sha256_file(A/'sam3.pt')}


def preserve_inputs(path,audit):
    if path.exists():assert json.loads(path.read_text())==audit, 'input drift between image and video arms'
    else:write_json(path,audit)


def verify_inputs(root):
    checked={}
    for path in root.glob('*/INPUTS.json'):
        audit=json.loads(path.read_text());expected=dict(audit['inputs'])
        for f in audit['frames']:
            expected[f['color_path']]=f['color_sha256'];expected[f['depth_path']]=f['depth_sha256']
        for p,h in expected.items():
            actual=checked.setdefault(p,sha256_file(Path(p)))
            assert actual==h,f'input changed: {p}'
    return {'verified_files':len(checked),'all_hashes_unchanged':True}


def setup():
    import torch
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(2)
    ctypes.CDLL(None).prctl(4,0,0,0,0)
    assert sha256_file(B/'env-spec.json') == SPEC_SHA, 'environment declaration changed'
    assert sha256_file(A/'sam3.pt') == CHECKPOINT_SHA, 'checkpoint mismatch'
    return torch


def load_clip(clip):
    jobs = json.loads((B/'jobs_full_available.json').read_text()) + json.loads((B/'jobs_3rscan_archived.json').read_text())
    job = next(j for j in jobs if j['key'] == clip['key'])
    manifest = load_manifest(Path(job['manifest'])); poses, _ = load_trajectory(Path(job['trajectory']))
    lookup = {f.frame_id:(f,p) for f,p in bind_manifest_trajectory(manifest,poses)}
    xyz = np.load(job['target'])['xyz']; frames=[]
    for fid in clip['frame_ids']:
        frame, pose = lookup[fid]; image, depth, K = read_frame(frame)
        ids,v,u = visible_map_pixels(xyz,pose.t_world_camera,K,depth.astype(float)/manifest.depth_scale)
        frames.append({'frame_id':fid, 'image':image, 'depth_shape':depth.shape,
                       'visible_map_ids':ids,'v':v,'u':u,
                       'color_path':str(frame.color_path),'depth_path':str(frame.depth_path),
                       'color_sha256':sha256_file(frame.color_path),'depth_sha256':sha256_file(frame.depth_path),
                       'rgb_decoded_sha256':hashlib.sha256(np.asarray(image).tobytes()).hexdigest()})
    audit={'job':job,'inputs':{str(p):sha256_file(Path(p)) for p in (job['manifest'],job['trajectory'],job['target'])},
           'map_points':len(xyz),'frames':[{k:v for k,v in f.items() if k in ('frame_id','color_path','depth_path','color_sha256','depth_sha256','rgb_decoded_sha256')} for f in frames]}
    return frames,audit


def normalize_output(output, kind):
    if kind=='image':
        masks=output['masks'].detach().cpu().numpy(); scores=output['scores'].float().detach().cpu().numpy()
        ids=np.arange(len(masks),dtype=np.int64)+1
    else:
        masks=np.asarray(output['out_binary_masks']); ids=np.asarray(output['out_obj_ids'])
        scores=np.asarray(output.get('out_probs',np.ones(len(masks))))
    assert len(masks)==len(ids) and np.isfinite(scores).all()
    if len(masks):masks=masks.reshape(len(masks),*masks.shape[-2:])
    return masks,ids,scores


def overlay(image,masks,ids,title):
    rgb=np.asarray(image).copy(); h,w=rgb.shape[:2]
    masks=depth_masks(masks,(h,w))
    for mask,i in zip(masks,ids):
        color=np.array([50+int(i)*v%206 for v in [73,151,199]])
        rgb[mask]=(.55*rgb[mask]+.45*color).astype(np.uint8)
    im=Image.fromarray(rgb);d=ImageDraw.Draw(im);d.rectangle((0,0,w,22),fill='black');d.text((4,4),title,fill='white')
    return im


def save_frame(path, frame, output, kind):
    masks,ids,scores=normalize_output(output,kind)
    dm=depth_masks(masks,frame['depth_shape']); projected=dm[:,frame['v'],frame['u']]
    path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/f"{frame['frame_id']:06}.npz",masks_packed=np.packbits(dm,axis=-1),
                        mask_shape=dm.shape,ids=ids,scores=scores,visible_map_ids=frame['visible_map_ids'],projected_masks=projected)
    return {'frame_id':frame['frame_id'],'mask_count':len(dm),'pixel_fraction':float(dm.any(0).mean()) if len(dm) else 0.,
            'mask_pixels':[int(m.sum()) for m in dm], 'object_ids':ids.tolist(),
            'visible_map_ids':frame['visible_map_ids'],'projected_masks':projected}


def compact(records):
    return [{k:v for k,v in f.items() if k not in ('visible_map_ids','projected_masks')} for f in records]


def image_pass(clips, torch, root):
    processor,model_audit=load_model(A/'sam3.pt',CHECKPOINT_SHA)
    write_json(root/'IMAGE_MODEL.json',model_audit)
    for clip in clips:
        frames, audit=load_clip(clip);O=root/clip['name'];O.mkdir(exist_ok=True)
        preserve_inputs(O/'INPUTS.json',audit); records=[];t=time.monotonic();visual_seconds=0.
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for f in frames:
                state=processor.set_image(f['image']);out=processor.set_text_prompt(clip['prompt'],state)
                records.append(save_frame(O/'image',f,out,'image'))
                if f['frame_id'] in (frames[0]['frame_id'],frames[len(frames)//2]['frame_id'],frames[-1]['frame_id']):
                    masks,ids,_=normalize_output(out,'image');overlay(f['image'],masks,ids,f"image {clip['prompt']} {f['frame_id']}").save(O/'image'/f"{f['frame_id']:06}_overlay.png")
                if clip['name']=='orbbec_reel' and f is frames[0]:
                    tv=time.monotonic()
                    visual_image(processor,state,f,O)
                    visual_seconds+=time.monotonic()-tv
                del state
        total_seconds=time.monotonic()-t
        result={'seconds':total_seconds-visual_seconds,'total_with_assisted_probe_seconds':total_seconds,
                'assisted_probe_seconds':visual_seconds,'frames':compact(records),'metrics':matched_temporal_metrics(records),
                'per_mask_metrics':per_mask_temporal_metrics(records)}
        write_json(O/'IMAGE_RESULT.json',result);print(json.dumps({'stage':'image','clip':clip['name'],'seconds':result['seconds']}),flush=True)
    del processor;gc.collect();torch.cuda.empty_cache()


def visual_image(processor,state,frame,O):
    vp=PLAN['visual_prompt']; w,h=vp['rgb_size']; processor.reset_all_prompts(state)
    for name,xyxy,label in [('positive',vp['positive_box_xyxy'],True),('positive_negative',vp['negative_box_xyxy'],False)]:
        x1,y1,x2,y2=xyxy; box=[(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h]
        out=processor.add_geometric_prompt(box,label,state)
        record=save_frame(O/f'visual_image_{name}',frame,out,'image')
        masks,ids,scores=normalize_output(out,'image');overlay(frame['image'],masks,ids,f"assisted visual {name}").save(O/f'visual_image_{name}'/'overlay.png')
        write_json(O/f'visual_image_{name}'/'result.json',compact([record])[0])


def video_pass(clips,torch,root,smoke=False):
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    t=time.monotonic();pred=Sam3VideoPredictor(checkpoint_path=str(A/'sam3.pt'),strict_state_dict_loading=True,compile=False)
    model_audit={'seconds':time.monotonic()-t,'checkpoint_sha256':CHECKPOINT_SHA,'strict_state_dict_loading':True,
                 'source_commit':subprocess.check_output(['git','-C',str(A/'sam3_official'),'rev-parse','HEAD'],text=True).strip(),
                 'torch':torch.__version__,'cuda':torch.version.cuda,
                 'device':torch.cuda.get_device_name(),'seed':42,'compiled':False,'offload_video_to_cpu':True}
    write_json(root/'VIDEO_MODEL.json',model_audit)
    for clip in clips:
        frames,audit=load_clip(clip)
        if smoke:frames=frames[:2]
        O=root/clip['name'];O.mkdir(exist_ok=True);preserve_inputs(O/'INPUTS.json',audit)
        for arm in (['video','video_visual'] if clip['name']=='orbbec_reel' and not smoke else ['video']):
            t=time.monotonic();session=pred.handle_request({'type':'start_session','resource_path':[f['image'] for f in frames],
                           'offload_video_to_cpu':True})['session_id']
            request={'type':'add_prompt','session_id':session,'frame_index':0,'output_prob_thresh':.5}
            if arm=='video_visual':
                x1,y1,x2,y2=PLAN['visual_prompt']['positive_box_xyxy'];w,h=PLAN['visual_prompt']['rgb_size']
                request.update(bounding_boxes=[[x1/w,y1/h,(x2-x1)/w,(y2-y1)/h]],bounding_box_labels=[1])
            else:request['text']=clip['prompt']
            anchor=pred.handle_request(request)
            records={}
            for response in pred.handle_stream_request({'type':'propagate_in_video','session_id':session,
                    'propagation_direction':'forward','start_frame_index':0,'max_frame_num_to_track':len(frames),'output_prob_thresh':.5}):
                ix=response['frame_index'];f=frames[ix];out=response['outputs']
                records[ix]=save_frame(O/arm,f,out,'video')
                if ix in (0,len(frames)//2,len(frames)-1):
                    masks,ids,_=normalize_output(out,'video');overlay(f['image'],masks,ids,f"{arm} {clip['prompt']} {f['frame_id']}").save(O/arm/f"{f['frame_id']:06}_overlay.png")
                print(json.dumps({'stage':arm,'clip':clip['name'],'frame':ix,'masks':records[ix]['mask_count']}),flush=True)
            assert len(records)==len(frames), 'incomplete propagation'
            records=[records[i] for i in range(len(frames))]
            result={'seconds':time.monotonic()-t,'frames':compact(records),'metrics':matched_temporal_metrics(records),
                    'per_mask_metrics':per_mask_temporal_metrics(records,tracked_ids=True),
                    'unique_track_ids':sorted({i for f in records for i in f['object_ids']}),
                    'assisted':arm=='video_visual','anchor_frame':frames[0]['frame_id'],
                    'max_gpu_memory_allocated_bytes':torch.cuda.max_memory_allocated()}
            write_json(O/f'{arm.upper()}_RESULT.json',result)
            pred.handle_request({'type':'close_session','session_id':session,'clear_cache_threshold':0})
            print(json.dumps({'stage':arm,'clip':clip['name'],'seconds':result['seconds'],'done':True}),flush=True)
    pred.shutdown();del pred;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],default='smoke');args=parser.parse_args()
    root=R/'tracking'/args.mode;root.mkdir(parents=True,exist_ok=False)
    write_json(root/'PLAN.json',PLAN)
    started=time.monotonic()
    try:
        before=code_provenance();write_json(root/'PROVENANCE_BEFORE.json',before)
        torch=setup();x=torch.randn(8,8,device='cuda');w=x@x;assert torch.isfinite(w).all()
        write_json(root/'KERNEL_WITNESS.json',{'seed':42,'shape':list(w.shape),'sum':float(w.sum()),'device':torch.cuda.get_device_name()})
        clips=PLAN['clips'][:1] if args.mode=='smoke' else PLAN['clips']
        if args.mode=='full':image_pass(clips,torch,root)
        video_pass(clips,torch,root,smoke=args.mode=='smoke')
        input_verification=verify_inputs(root)
        after=code_provenance();write_json(root/'PROVENANCE_AFTER.json',after)
        assert before==after,'model source, runner, weights or environment declaration changed during inference'
        status={'status':'completed','seconds':time.monotonic()-started,'runner_sha256':sha256_file(Path(__file__)),
                'module_sha256':sha256_file(R/'code/src/pose_pipeline/sam3_tracking_probe.py'),
                'env_spec_file_sha256':SPEC_SHA,'input_verification':input_verification,
                'provenance_unchanged':True,'gt_consumed':False,'quality_accepted':False}
    except BaseException:
        status={'status':'failed','seconds':time.monotonic()-started,'error':traceback.format_exc()};traceback.print_exc()
    write_json(root/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
