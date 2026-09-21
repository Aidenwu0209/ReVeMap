"""SAM 3 image concepts on frozen RGB-D trajectories (experimental sidecar).

Requires an explicitly hashed local checkpoint. Geometry and poses are never
modified. A completed sampled run is not complete-frame semantic inference or
evidence of semantic accuracy. Instance IDs use measured overlap, not SGA.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import time

import numpy as np
from .contracts import load_manifest, load_trajectory, bind_manifest_trajectory, sha256_file
from .sam3_fusion import PixelClaims, MapVotes, GeometricInstances, visible_map_pixels


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_model(checkpoint, expected_sha256):
    if not checkpoint.is_file() or len(expected_sha256) != 64:
        raise ValueError('an existing local checkpoint and explicit SHA-256 are required')
    actual = sha256_file(checkpoint)
    if actual != expected_sha256:
        raise ValueError('checkpoint SHA-256 mismatch; no fallback is permitted')
    import torch
    from sam3 import model_builder
    from sam3.model.sam3_image_processor import Sam3Processor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(2)
    # Avoid enormous native crash dumps on the host system drive.
    if os.name == 'posix':
        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)
    audit = {'checkpoint_sha256': actual, 'checkpoint_bytes': checkpoint.stat().st_size}
    original_loader = model_builder._load_checkpoint

    def checked_load(model, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        if 'model' in ckpt and isinstance(ckpt['model'], dict):
            ckpt = ckpt['model']
        image_ckpt = {k.replace('detector.', ''): v for k, v in ckpt.items() if 'detector' in k}
        missing, unexpected = model.load_state_dict(image_ckpt, strict=False)
        audit.update(missing_keys=list(missing), unexpected_keys=list(unexpected),
                     loaded_image_tensors=len(image_ckpt))
        if missing or not image_ckpt:
            raise RuntimeError(f'incomplete SAM 3 image checkpoint: {missing}')

    model_builder._load_checkpoint = checked_load
    try:
        model = model_builder.build_sam3_image_model(checkpoint_path=str(checkpoint),
            load_from_HF=False, device='cuda', eval_mode=True, compile=False)
    finally:
        model_builder._load_checkpoint = original_loader
    audit.update(torch=torch.__version__, cuda=torch.version.cuda,
                 device=torch.cuda.get_device_name(), random_weight_fallback=False,
                 load_from_HF=False, precision='bfloat16 autocast', seed=42)
    return Sam3Processor(model, device='cuda', confidence_threshold=.5), audit


def read_frame(frame):
    """Use original RGB resolution for SAM; masks are resampled onto depth pixels."""
    from PIL import Image
    from reconstruction.rgbd_refusion import _read_rgbd
    _, depth, intrinsics = _read_rgbd(frame)
    image = Image.open(frame.color_path).convert('RGB')
    if frame.rotate_ccw:
        image = image.transpose(Image.Transpose.ROTATE_90)
    return image, depth, intrinsics


def infer_frame(processor, image, depth_shape, taxonomy):
    import torch
    from PIL import Image
    claims = PixelClaims(depth_shape, max(c['id'] for c in taxonomy) + 1)
    h, w = depth_shape
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = processor.set_image(image)
        for category in taxonomy:
            if category['prompt'] is None:
                continue
            processor.reset_all_prompts(state)
            result = processor.set_text_prompt(prompt=category['prompt'], state=state)
            scores = result['scores'].float().cpu().numpy()
            masks = result['masks'].cpu().numpy()
            if not np.isfinite(scores).all() or masks.shape[0] != len(scores):
                raise RuntimeError('invalid model output')
            for mask, score in zip(masks, scores):
                mask = mask.reshape(image.height, image.width)
                if mask.shape != (h, w):
                    mask = np.array(Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST))
                claims.add(category['id'], mask, float(score))
        del state
    return (*claims.finalize(), claims.records)


def save_overlay(image, semantic, taxonomy, path):
    from PIL import Image, ImageDraw
    rgb = np.array(image.resize((semantic.shape[1], semantic.shape[0])))
    colors = np.stack([50 + semantic.astype(np.int64) * m % 206 for m in (73, 151, 199)], axis=-1)
    rgb = np.where((semantic > 0)[..., None], .5 * rgb + .5 * colors, rgb).astype(np.uint8)
    canvas = Image.fromarray(rgb); draw = ImageDraw.Draw(canvas)
    names = {c['id']: c['name'] for c in taxonomy}
    legend = ', '.join(f'{k}: {names[int(k)]}' for k in np.unique(semantic) if k > 0)
    # Legend is also saved as UTF-8 text; no tiny opaque label abbreviations.
    for i in range(0, len(legend), 85):
        draw.text((5, 5 + (i // 85) * 14), legend[i:i+85], fill='white', stroke_fill='black', stroke_width=1)
    canvas.save(path)
    path.with_suffix('.txt').write_text(legend + '\n')


def run_scene(job, output, processor, model_audit, taxonomy, args):
    output.mkdir(parents=True, exist_ok=False)
    (output/'frames').mkdir()
    manifest_path = Path(job['manifest']); trajectory_path = Path(job['trajectory'])
    manifest = load_manifest(manifest_path)
    poses, _ = load_trajectory(trajectory_path)
    bound = bind_manifest_trajectory(manifest, poses)
    source_paths = [manifest_path, trajectory_path]
    target = Path(job['target']) if job.get('target') else None
    if target:
        source_paths.append(target)
        with np.load(target) as f:
            xyz = f['xyz'].copy()
        votes = MapVotes(len(xyz), max(c['id'] for c in taxonomy)+1)
        instances = GeometricInstances()
    inputs = {str(p): sha256_file(p) for p in source_paths}
    selected = bound[::args.stride]
    if bound[-1] is not selected[-1]:
        selected.append(bound[-1])
    if args.frame_ids:
        wanted = {int(x) for x in args.frame_ids.split(',')}
        selected = [(f, p) for f, p in bound if f.frame_id in wanted]
        if {f.frame_id for f, _ in selected} != wanted:
            raise ValueError('requested frames unavailable')
    if not selected:
        raise ValueError('no selected RGB-D frames')
    total_raw_frames = int(job.get('full_raw_frame_count', len(manifest.frames)))
    if total_raw_frames < len(manifest.frames):
        raise ValueError('original frame count cannot be smaller than available input')
    status = {'status': 'running', 'key': job['key'], 'total_raw_frames': total_raw_frames,
              'available_raw_frames': len(manifest.frames),
              'available_poses': len(bound), 'selected_frame_ids': [f.frame_id for f, _ in selected],
              'selected_frames': len(selected), 'processed_frames': 0, 'input_sha256': inputs,
              'model': model_audit, 'source_sha256': sha256_file(Path(__file__)),
              'fusion_source_sha256': sha256_file(Path(__file__).with_name('sam3_fusion.py')),
              'taxonomy_sha256': sha256_file(args.taxonomy), 'ground_truth_consumed': False,
              'pose_feedback': False, 'sga_inference_executed': False,
              'instance_method': 'shared measured map support; geometric baseline',
              'settings': {'stride': args.stride, 'depth_tolerance_m': .05, 'model_threshold': .5,
                           'pixel_class_margin': .1, 'minimum_views': 2, 'map_vote_share': .65},
              'quality_accepted': False, 'geometry_modified': False}
    write_json(output/'status.json', status)
    write_json(output/'classes.json', {'0': 'unknown', **{str(c['id']): c['name'] for c in taxonomy}})
    started = time.monotonic()
    try:
        with (output/'frames.jsonl').open('x') as log:
            for frame, pose in selected:
                now = time.monotonic()
                image, depth, intrinsics = read_frame(frame)
                sem, local_instance, conf, records = infer_frame(processor, image, depth.shape, taxonomy)
                ids = np.array([], np.int64); v = u = np.array([], np.int64)
                if target:
                    ids, v, u = visible_map_pixels(xyz, pose.t_world_camera, intrinsics,
                                                  depth.astype(float)/manifest.depth_scale)
                    votes.add(frame.frame_id, ids, sem[v, u], conf[v, u])
                    assignments = instances.add(frame.frame_id, ids, local_instance[v, u], sem[v, u])
                else:
                    assignments = {}
                np.savez_compressed(output/'frames'/f'{frame.frame_id:06}.npz', semantic=sem,
                    local_instance=local_instance, confidence=conf.astype(np.float16),
                    visible_map_ids=ids, projected_semantic=sem[v, u],
                    projected_local_instance=local_instance[v, u])
                row = {'frame_id': frame.frame_id, 'color_sha256': sha256_file(frame.color_path),
                       'depth_sha256': sha256_file(frame.depth_path), 'rgb_size': list(image.size),
                       'depth_shape': list(depth.shape), 'seconds': time.monotonic()-now,
                       'known_pixels': int((sem>0).sum()), 'total_pixels': int(sem.size),
                       'visible_map_points': len(ids), 'masks': records,
                       'local_to_global_instance': assignments}
                log.write(json.dumps(row)+'\n'); log.flush()
                if len(selected) <= 12 or status['processed_frames'] % 30 == 0 or frame is selected[-1][0]:
                    save_overlay(image, sem, taxonomy, output/'frames'/f'{frame.frame_id:06}_overlay.png')
                status['processed_frames'] += 1
                status['seconds'] = time.monotonic()-started
                write_json(output/'status.json', status)
                print(json.dumps({k: status[k] for k in ('key','processed_frames','selected_frames','seconds')}), flush=True)
        if target:
            semantic, confidence, support = votes.finalize()
            instance = instances.finalize(semantic)
            np.savez_compressed(output/'map_labels.npz', semantic=semantic, instance=instance,
                confidence=confidence, support=support, visible_count=votes.visible_count)
            status.update(map_points=len(xyz), semantic_coverage=float(np.mean(semantic > 0)),
                instance_coverage=float(np.mean(instance > 0)), observed_map_coverage=float(np.mean(votes.visible_count>0)),
                geometry_xyz_sha256=__import__('hashlib').sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest())
            objects = []
            for track_id in np.unique(instance):
                if track_id <= 0: continue
                pts = xyz[instance == track_id]
                tr = instances.tracks[track_id-1]
                objects.append({'instance_id': int(track_id), 'semantic_id': tr['category'],
                    'point_count': len(pts), 'supporting_frames': sorted(tr['frames']),
                    'center': pts.mean(axis=0).tolist(), 'min': pts.min(axis=0).tolist(), 'max': pts.max(axis=0).tolist()})
            write_json(output/'objects.json', objects)
        if inputs != {str(p): sha256_file(p) for p in source_paths}:
            raise RuntimeError('frozen inputs changed during inference')
        status.update(status='completed', complete_selected_frames=True,
                      complete_full_sequence=len(selected)==total_raw_frames, seconds=time.monotonic()-started)
        write_json(output/'result.json', status)
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output/'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('jobs', 'output', 'checkpoint', 'taxonomy'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--stride', type=int, default=20)
    parser.add_argument('--frame-ids')
    args = parser.parse_args()
    if args.stride <= 0: parser.error('stride must be positive')
    taxonomy = json.loads(args.taxonomy.read_text())['classes']
    ids = [c['id'] for c in taxonomy]
    if len(ids) != len(set(ids)) or min(ids) < 1:
        parser.error('taxonomy IDs must be unique and positive')
    processor, audit = load_model(args.checkpoint, args.checkpoint_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'model_load.json', audit)
    for job in json.loads(args.jobs.read_text()):
        run_scene(job, args.output/job['key'], processor, audit, taxonomy, args)


if __name__ == '__main__':
    main()
