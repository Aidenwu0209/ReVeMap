"""Full raw RGB-D tracking using an explicitly configured DROID-W provider.

The RGB/depth resampling, network settings and solver schedule preserve the
SVD recovery experiment. This module runs in a dedicated GPU subprocess;
CPU graph construction and final fusion do not need this runtime.
"""
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
import importlib
import os
import sys
import json
import time
import numpy as np
import cv2
import torch
import yaml

from .contracts import (
    load_manifest, write_manifest, write_trajectory, PoseRecord, sha256_file,
)


def load_provider(provider_root: Path) -> SimpleNamespace:
    """Load the configured source checkout without installing or downloading it."""
    root = Path(provider_root).resolve(strict=True)
    required = (
        'src/modules/droid_net/__init__.py', 'src/depth_video.py', 'src/frontend.py',
        'src/backend.py', 'src/trajectory_filler.py', 'src/motion_filter.py',
        'configs/droid_w.yaml', 'pretrained/droid.pth',
    )
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    sys.path.insert(0, str(root))
    # The provider resolves its own optional resources from its checkout.
    os.chdir(root)
    names = {
        'DroidNet': ('src.modules.droid_net', 'DroidNet'),
        'DepthVideo': ('src.depth_video', 'DepthVideo'),
        'Frontend': ('src.frontend', 'Frontend'),
        'Backend': ('src.backend', 'Backend'),
        'PoseTrajectoryFiller': ('src.trajectory_filler', 'PoseTrajectoryFiller'),
        'motion_module': ('src.motion_filter', None),
    }
    values = {'PROVIDER': root}
    for name, (module_name, attribute) in names.items():
        module = importlib.import_module(module_name)
        try:
            Path(module.__file__).resolve().relative_to(root)
        except ValueError as error:
            raise RuntimeError(f'Conflicting provider module: {module_name}') from error
        values[name] = module if attribute is None else getattr(module, attribute)
    return SimpleNamespace(**values)


class Printer:
    def print(self, message, *args):
        print(message, flush=True)

class RawStream:
    def __init__(self, manifest):
        self.manifest = manifest
        self.frames = manifest.frames
        if not self.frames:
            raise ValueError('No RGB-D frames')
        if len({tuple(f.intrinsics) for f in self.frames}) != 1:
            raise ValueError('DROID trajectory filler requires constant intrinsics')
        _, _, self.k = self.read(0)
    def __len__(self):
        return len(self.frames)
    def read(self, i):
        f = self.frames[i]
        c = cv2.imread(str(f.color_path), cv2.IMREAD_COLOR)
        d = cv2.imread(str(f.depth_path), cv2.IMREAD_UNCHANGED)
        if c is None or d is None:
            raise RuntimeError(f'Missing RGB-D frame {f.frame_id}')
        if d.ndim != 2 or d.dtype != np.uint16:
            raise ValueError(f'Frame {f.frame_id} requires a uint16 depth image')
        h, w = d.shape
        # Preserve native RGB texture until the final tracking resize.
        fx, fy, cx, cy = f.intrinsics
        if f.rotate_ccw:
            c = cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE)
            d = cv2.rotate(d, cv2.ROTATE_90_COUNTERCLOCKWISE)
            fx, fy, cx, cy = fy, fx, cy, w-1-cx
        h, w = d.shape
        H, W = (320, 240) if h > w else (240, 320)
        sx, sy = W/w, H/h
        k = torch.tensor([fx*sx, fy*sy, (cx+.5)*sx-.5, (cy+.5)*sy-.5], dtype=torch.float32)
        c = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA)
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)/self.manifest.depth_scale
        d[(d < .2) | (d > 4.5)] = 0
        image = torch.from_numpy(cv2.cvtColor(c, cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float()[None]/255.
        return image, torch.from_numpy(d), k
    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError(i)
        image, depth, _ = self.read(i)
        return i, image, depth, None
    def get_intrinsic(self):
        return self.k.clone()

def run_dense(manifest_path: Path, output_dir: Path, provider_root: Path) -> dict:
    """Estimate all raw frames with measured depth and final visual refinement."""
    # Provider imports change cwd; retain the caller's manifest identity.
    manifest_path = Path(manifest_path).resolve(strict=True)
    m = load_manifest(manifest_path)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    api = load_provider(provider_root)
    PROVIDER = api.PROVIDER
    DroidNet, DepthVideo = api.DroidNet, api.DepthVideo
    Frontend, Backend = api.Frontend, api.Backend
    PoseTrajectoryFiller, motion_module = api.PoseTrajectoryFiller, api.motion_module
    key = m.dataset + '/' + m.sequence_id
    write_manifest(out/'raw_manifest.json', m)
    denied = []
    def prohibit_gt(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if p.endswith('.pose.txt') or '/pose/' in p or p.endswith('.ply') or p.endswith('_vh_clean_2.ply'):
                denied.append(p)
                raise PermissionError('GT/mesh access prohibited during inference: '+p)
    sys.addaudithook(prohibit_gt)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    np.random.seed(43); torch.manual_seed(43); torch.cuda.manual_seed_all(43)
    torch.backends.cudnn.benchmark = False
    stream = RawStream(m)
    image, depth, intrinsic = stream.read(0)
    cfg = yaml.safe_load((PROVIDER/'configs/droid_w.yaml').read_text())
    cfg.update(scene=key.replace('/','_'), debug=False, save_gt_poses=False)
    cfg['data'] = {'output': str(out), 'input_folder': str(out)}
    cfg['cam'].update(H_out=image.shape[-2], W_out=image.shape[-1])
    cfg['mapping']['enable'] = False
    cfg['mapping']['uncertainty_params']['activate'] = False
    t = cfg['tracking']
    t['buffer'] = min(len(stream)+32, 1600)
    t['force_keyframe_every_n_frames'] = 6
    t['frontend'].update(enable_opt_dyn_mask=False, enable_online_ba=False, enable_loop=False)
    t['backend'].update(metric_depth_reg=True, normalize=False)
    t['uncertainty_params'].update(activate=False, visualize=False, enable_affine_transform=False, gamma_depth=.05)
    (out/'effective_config.json').write_text(json.dumps(cfg, indent=2))
    net = DroidNet()
    checkpoint = PROVIDER/'pretrained/droid.pth'
    state = OrderedDict((k.replace('module.',''),v) for k,v in torch.load(str(checkpoint), weights_only=True).items())
    for name in ['update.weight.2.weight','update.weight.2.bias','update.delta.2.weight','update.delta.2.bias']:
        state[name] = state[name][:2]
    net.load_state_dict(state); net.cuda().eval()
    video = DepthVideo(cfg, Printer())
    current = {'depth': None}
    motion_module.get_metric_depth_estimator = lambda cfg: None
    def sensor_depth(estimator, timestamp, image, cfg, device, **kwargs):
        if current['depth'] is None:
            raise RuntimeError('No bound sensor frame')
        return current['depth'].to(device)
    motion_module.predict_metric_depth = sensor_depth
    motion = motion_module.MotionFilter(net, video, cfg, thresh=t['motion_filter']['thresh'])
    frontend = Frontend(net, video, cfg)
    backend = Backend(net, video, cfg)
    started = time.time(); progress = []; last_ba = 0
    with torch.no_grad():
        for i in range(len(stream)):
            if video.counter.value >= t['buffer']-18:
                raise RuntimeError('Keyframe buffer exhausted; no truncation or reset allowed')
            image, depth, intrinsic = stream.read(i)
            current['depth'] = depth
            before = video.counter.value
            forced = motion.track(i, image, intrinsic)
            if video.counter.value > before:
                prior = video.mono_disps[before]
                valid = prior > 0
                if valid.any():
                    video.disps[before] = torch.where(valid, prior, prior[valid].median())
            frontend(forced, None)
            if frontend.is_initialized and video.counter.value >= last_ba+64:
                print('ONLINE_BA', i, video.counter.value, flush=True)
                backend.dense_ba(steps=2, enable_update_uncer=False, enable_udba=False)
                last_ba = video.counter.value
            if not torch.isfinite(video.poses[:video.counter.value]).all():
                raise RuntimeError('Nonfinite dense trajectory')
            if i % 50 == 0:
                row = {'frame':i, 'keyframes':video.counter.value, 'runtime_s':time.time()-started}
                progress.append(row); print('PROGRESS', key, row, flush=True)
        if not frontend.is_initialized:
            raise RuntimeError('Frontend did not initialize')
        n = video.counter.value
        np.savez_compressed(out/'online_keyframes.npz', poses=video.poses[:n].cpu().numpy(), timestamps=video.timestamp[:n].cpu().numpy())
        # Release the local correlation volume before the low-memory global pass.
        frontend.graph.clear_edges()
        del frontend, motion
        torch.cuda.empty_cache()
        print('FINAL_BA', key, n, flush=True)
        backend.dense_ba(steps=6, enable_update_uncer=False, enable_udba=False)
        backend.dense_ba(steps=6, enable_update_uncer=False, enable_udba=False)
        keyframe_ids = video.timestamp[:n].cpu().numpy().astype(int).tolist()
        np.savez_compressed(out/'final_keyframes.npz', poses=video.poses[:n].cpu().numpy(), timestamps=video.timestamp[:n].cpu().numpy())
        filler = PoseTrajectoryFiller(cfg, net, video, Printer())
        full, _ = filler(stream)
        poses = full.inv().matrix().cpu().numpy()
    np.save(out/'final_raw_poses.npy', poses)
    if poses.shape != (len(stream),4,4) or not np.isfinite(poses).all():
        raise RuntimeError('Invalid full dense trajectory')
    records = [PoseRecord(f.frame_id,f.timestamp_us,T,True,'dense_sensor_rgbd') for f,T in zip(m.frames,poses)]
    write_trajectory(out/'trajectory.json', records, sequence_id=m.sequence_id, arm='candidate', metadata={'gt_consumed':False,'sensor_depth_at_inference':True,'nonkeyframes_visually_optimized':True})
    result = {'raw_frame_count':len(stream),'final_pose_count':len(poses),'keyframe_count':n,'keyframe_ordinals':keyframe_ids,'gt_consumed':False,'gt_access_denials':denied,'manifest_input_sha256':sha256_file(manifest_path),'identity_fallback_used':False,'runtime_s':time.time()-started,'wrapper_sha256':sha256_file(Path(__file__)),'checkpoint_sha256':sha256_file(checkpoint),'progress':progress}
    result['source_sha256'] = {str(p.relative_to(PROVIDER)):sha256_file(p) for p in PROVIDER.glob('src/**/*.py')}
    result['seal'] = {p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print('RESULT',key,len(poses),n,flush=True)
    return result
