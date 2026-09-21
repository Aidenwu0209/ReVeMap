"""Run fixed loss attribution and two unknown-object cache ablations on SSH44."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code/src'))
import numpy as np
import scipy
from pose_pipeline.sam3_loss_audit import audit_losses, REASONS
from pose_pipeline.sam3_unknown import UnknownConfig, raw_mask_partition, fuse_unknown
from pose_pipeline.sam3_fusion import PixelClaims, visible_map_pixels
from pose_pipeline.contracts import load_manifest, load_trajectory, bind_manifest_trajectory


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def arrsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x:
        x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x))+'\n')


def event(**value):
    print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **value}), flush=True)


def inventory(xyz, instance, semantic):
    objects = []
    for i in np.unique(instance):
        if i == 0: continue
        ids = np.flatnonzero(instance == i)
        labels, counts = np.unique(semantic[ids], return_counts=True)
        objects.append({'instance_id': int(i), 'point_count': len(ids),
            'semantic_id': int(labels[0]) if len(labels) == 1 else 0,
            'semantic_id_policy': 'single fixed class only; mixed/unknown remains 0',
            'semantic_histogram': {str(int(k)): int(v) for k,v in zip(labels, counts)},
            'semantic_naming_executed': False, 'center': xyz[ids].mean(axis=0).tolist(),
            'min': xyz[ids].min(axis=0).tolist(), 'max': xyz[ids].max(axis=0).tolist()})
    assert sum(o['point_count'] for o in objects) == int(np.count_nonzero(instance))
    return objects


def run(job):
    import cv2
    key = job['key']; started = time.perf_counter()
    cfg = UnknownConfig()
    cache = Path(job['cache_root']); baseline = Path(job['baseline_root'])
    reference = ROOT.parent/'sam3_multiview_20260912_v1/consensus_matched'/key
    destination = ROOT/'loss_audit'/key
    destination.mkdir(parents=True, exist_ok=False)
    selected = job['selected_frame_ids']
    records = [json.loads(s) for s in (cache/'frames.jsonl').read_text().splitlines() if s]
    assert [r['frame_id'] for r in records] == selected
    assert [r['ordinal'] for r in records] == list(range(len(selected)))
    paths = [cache/'frames'/f'{fid:06}.npz' for fid in selected]
    assert set(paths) == set((cache/'frames').glob('*.npz'))
    source_paths = [Path(__file__), *[ROOT/'code/src/pose_pipeline'/name for name in
        ('sam3_loss_audit.py','sam3_unknown.py','sam3_multiview.py','sam3_fusion.py','contracts.py')]]
    source_hashes = {str(p): sha(p) for p in source_paths}
    inputs = [Path(job['manifest']), Path(job['trajectory']), Path(job['target']),
        baseline/'map_labels.npz', baseline/'result.json', baseline/'classes.json',
        cache/'result.json', cache/'frames.jsonl', reference/'map_labels.npz',
        ROOT/'inputs/JOBS.json', ROOT/'PLAN_UNKNOWN.json', ROOT/'ENV_REUSE.json', *paths]
    input_hashes = {str(p): sha(p) for p in inputs}
    environment = json.loads((ROOT/'ENV_REUSE.json').read_text())
    assert sha(environment['original_spec_path']) == environment['original_spec_file_sha256']
    input_hashes[environment['original_spec_path']] = sha(environment['original_spec_path'])
    with np.load(job['target'], allow_pickle=False) as z: xyz = z['xyz'].copy()
    assert len(xyz) == job['expected_points'] and arrsha(xyz) == job['geometry_xyz_sha256']
    with np.load(baseline/'map_labels.npz', allow_pickle=False) as z:
        original = {k:z[k].copy() for k in z.files}
    with np.load(reference/'map_labels.npz', allow_pickle=False) as z: expected = z['instance'].copy()
    for field in ('semantic','confidence'):
        assert arrsha(original[field]) == job[f'baseline_{field}_sha256']
    n = len(xyz); frames = []
    for fid,path in zip(selected,paths):
        with np.load(path, allow_pickle=False) as z:
            f = {out:z[src].copy() for out,src in {
                'point_ids':'visible_map_ids','mask_ids':'projected_local_instance',
                'semantic':'projected_semantic','confidence':'projected_confidence',
                'interior':'interior'}.items()}
        f['frame_id'] = fid; frames.append(f)
    _, reasons, loss = audit_losses(n, frames, original['semantic'], original['instance'],
        {'min_mask_points':30,'min_output_points':50,'min_point_views':1,
         'min_group_frames':2,'object_score_mode':'max_point'}, expected)
    write(destination/'backend_loss.json', loss)
    np.savez_compressed(destination/'point_loss_reasons.npz', reason=reasons,
        old_instance=original['instance'], matched_instance=expected)
    write(destination/'reason_dictionary.json', dict(enumerate(REASONS)))
    event(key=key, phase='loss_verified', lost=loss['lost_points'], reasons=loss['loss_reasons'])
    # Freeze semantic independence arm before accessing raw-mask outcomes.
    arms = [('unknown_contract', frames)]
    for stage, fs in arms:
        run_arm(job, stage, fs, original, xyz, input_hashes, source_hashes, cfg, started)
    manifest = load_manifest(Path(job['manifest']))
    poses, _ = load_trajectory(Path(job['trajectory']))
    bound = {f.frame_id:(f,p) for f,p in bind_manifest_trajectory(manifest, poses)}
    raw_frames = []; raw_audit = []; raw_seen = np.zeros(n,bool); preclass_seen = np.zeros(n,bool)
    family_seen = np.zeros(n,bool)
    raw_started = time.perf_counter()
    for ordinal,(fid,path,row,cached_frame) in enumerate(zip(selected, paths, records, frames)):
        frame, pose = bound[fid]
        assert sha(frame.depth_path) == row['depth_sha256']
        input_hashes[str(frame.depth_path)] = row['depth_sha256']
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError('invalid source depth')
        fx,fy,cx,cy = frame.intrinsics
        if frame.rotate_ccw:
            old_width = depth.shape[1]
            depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
            fx,fy,cx,cy = fy,fx,cy,old_width-1.-cx
        ids,v,u = visible_map_pixels(xyz,pose.t_world_camera,(fx,fy,cx,cy),
            depth.astype(float)/manifest.depth_scale, tolerance=.05)
        assert np.array_equal(ids,cached_frame['point_ids']), 'cached depth visibility mismatch'
        with np.load(path, allow_pickle=False) as z:
            shape = tuple(z['depth_shape']); packed = z['raw_masks_packed'].copy()
            raw_count = int(z['raw_mask_count']); semantic_image = z['semantic']
            assert shape == depth.shape and raw_count == len(packed)
            assert np.array_equal(semantic_image[v,u], cached_frame['semantic'])
        raw_records = row['masks'][:raw_count]
        assert [r['mask_id'] for r in raw_records] == list(range(1,raw_count+1))
        scores = [r['score'] for r in raw_records]
        local, confidence, interior, mask_audit = raw_mask_partition(packed, shape, scores, cfg)
        # Reconstruct pre-family PixelClaims to audit losses without speculation.
        claims = PixelClaims(shape, 35)
        masks = np.unpackbits(packed,axis=1,count=depth.size).reshape(raw_count,*shape).astype(bool)
        for record,mask in zip(raw_records,masks):
            assert int(mask.sum()) == record['pixel_count']
            claims.add(record['class_id'], mask, record['score'])
        preclass_sem,_,_ = claims.finalize()
        raw_seen[ids[np.any(masks[:,v,u],axis=0)]] = True
        preclass_seen[ids[preclass_sem[v,u]>0]] = True
        family_seen[ids[cached_frame['semantic']>0]] = True
        raw_frames.append({'frame_id':fid,'point_ids':ids,'mask_ids':local[v,u],
            'semantic':(local[v,u]>0).astype(np.int32),'confidence':confidence[v,u],
            'interior':interior[v,u]})
        raw_audit.append({'frame_id':fid,'depth_projection_equal_to_cache':True,**mask_audit})
        if ordinal % 30 == 0 or ordinal == len(selected)-1:
            event(key=key,phase='raw_reprojection',processed=ordinal+1,total=len(selected))
    removed = (original['instance']>0)&(expected==0)
    frontend = {'policy':'ever-observed sets, not exclusive causal attribution',
        'raw_mask_projected_points':int(raw_seen.sum()),'pre_family_known_projected_points':int(preclass_seen.sum()),
        'post_family_known_projected_points':int(family_seen.sum()),
        'lost_points_seen_in_raw_masks':int(np.sum(removed&raw_seen)),
        'lost_points_never_known_before_family':int(np.sum(removed&raw_seen&~preclass_seen)),
        'lost_points_never_known_after_family':int(np.sum(removed&raw_seen&~family_seen)),
        'raw_prompt_candidates_are_not_class_agnostic_discovery':True,
        'predicted_classes':[]}
    classes = json.loads((baseline/'classes.json').read_text())
    for label in np.unique(original['semantic']):
        sel = removed&(original['semantic']==label)
        frontend['predicted_classes'].append({'semantic_id':int(label),'predicted_name':classes.get(str(label)),
            'lost_points':int(sel.sum()),'seen_raw':int(np.sum(sel&raw_seen)),
            'seen_pre_family_known':int(np.sum(sel&preclass_seen)),
            'seen_post_family_known':int(np.sum(sel&family_seen))})
    write(destination/'frontend_loss.json',frontend)
    write(destination/'raw_mask_dedup.json',raw_audit)
    result = run_arm(job,'raw_unknown',raw_frames,original,xyz,input_hashes,source_hashes,cfg,raw_started)
    write(destination/'result.json',{'status':'completed','key':key,'source_files_sha256':source_hashes,
        'input_sha256':input_hashes,'backend_output_equal_to_r3':True,'raw_depth_projection_equal_to_cache':True,
        'selected_frame_ids':selected,'processed_frames':len(raw_frames),'gt_consumed':False,
        'seconds':time.perf_counter()-started,'raw_arm_seconds':result['seconds']})


def run_arm(job,stage,frames,original,xyz,input_hashes,source_hashes,cfg,started):
    output = ROOT/stage/job['key']; output.mkdir(parents=True,exist_ok=False)
    t = time.perf_counter(); instance,audit = fuse_unknown(len(xyz),frames,original['semantic'],cfg)
    fusion = time.perf_counter()-t
    labels = {**original,'instance':instance}
    assert all(np.array_equal(v,labels[k]) for k,v in original.items() if k!='instance')
    np.savez_compressed(output/'map_labels.npz',**labels)
    with np.load(output/'map_labels.npz',allow_pickle=False) as z:
        assert all(np.array_equal(z[k],v) for k,v in labels.items())
    objects = inventory(xyz,instance,original['semantic'])
    write(output/'objects.json',objects); write(output/'fusion_audit.json',audit)
    (output/'classes.json').write_bytes((Path(job['baseline_root'])/'classes.json').read_bytes())
    write(output/'scene_graph.json',{'node_file':'objects.json','relations':[],
        'new_relation_prediction_executed':False,'reason':'new instance IDs; no relation inference'})
    assert all(sha(p)==value for p,value in {**input_hashes,**source_hashes}.items()), 'input/source changed'
    previous = json.loads((Path(job['baseline_root'])/'result.json').read_text())
    result = {'status':'completed','key':job['key'],'variant':stage,'map_points':len(xyz),
        'geometry_xyz_sha256':arrsha(xyz),'geometry_modified':False,'pose_feedback':False,
        'semantic_unchanged':True,'confidence_unchanged':True,'instance_requires_semantic_label':False,
        'semantic_naming_executed':False,'unknown_instance_contract_opt_in':True,
        'semantic_coverage':float(np.mean(labels['semantic']>0)),
        'instance_coverage':float(np.mean(instance>0)),
        'unknown_semantic_instance_points':int(np.sum((instance>0)&(labels['semantic']==0))),
        'object_count':len(objects),'baseline_instance_coverage':float(np.mean(original['instance']>0)),
        'newly_assigned_instance_points':int(np.sum((instance>0)&(original['instance']==0))),
        'removed_instance_points':int(np.sum((instance==0)&(original['instance']>0))),
        'selected_frame_ids':job['selected_frame_ids'],'selected_frames':len(frames),'processed_frames':len(frames),
        'complete_selected_frames':True,'complete_full_sequence':False,
        'total_raw_frames':previous.get('total_raw_frames'),'scope':job.get('scope','frozen selected cached frames'),
        'new_model_inference':False,'sam3_inference_executed':False,'sga_inference_executed':False,
        'model_inference_reused':True,'raw_depth_reprocessed':stage=='raw_unknown',
        'raw_rgbd_reprocessed':False,'checkpoint_loaded_this_run':False,'gt_consumed':False,
        'ground_truth_consumed':False,'quality_accepted':False,'new_relation_prediction_executed':False,
        'baseline_run':job['baseline_root'],'cached_frames_run':job['cache_root'],
        'cached_baseline_provenance_receipt':str(Path(job['baseline_root'])/'result.json'),
        'cached_model_provenance_receipt':str(Path(job['cache_root'])/'result.json'),
        'config':asdict(cfg),'input_sha256':dict(input_hashes),'source_files_sha256':source_hashes,
        'input_hashes_verified_after_run':True,'source_sha256':source_hashes[str(Path(__file__))],
        'baseline_array_sha256':{k:arrsha(v) for k,v in original.items()},
        'output_array_sha256':{k:arrsha(v) for k,v in labels.items()},
        'fusion_wall_seconds':fusion,'seconds':time.perf_counter()-started,
        'time_definition':'includes preparation for this arm, fusion, label export and hash verification; no model inference',
        'environment':{'python':sys.executable,'python_version':platform.python_version(),
            'numpy':np.__version__,'scipy':scipy.__version__,'reuse_receipt':str(ROOT/'ENV_REUSE.json')}}
    write(output/'result.json',result)
    event(key=job['key'],stage=stage,phase='completed',coverage=result['instance_coverage'],
        unknown_points=result['unknown_semantic_instance_points'],objects=len(objects),fusion_seconds=fusion)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--key'); args=parser.parse_args()
    jobs=json.loads((ROOT/'inputs/JOBS.json').read_text())
    if args.key: jobs=[j for j in jobs if j['key']==args.key]
    if not jobs: raise ValueError('no selected jobs')
    for job in jobs:
        try: run(job)
        except BaseException:
            event(key=job['key'],phase='failed',error=traceback.format_exc()); raise


if __name__=='__main__': main()
