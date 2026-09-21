"""Temporary DROID-W keyframe map; final mapping uses unchanged run-rgbd."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
import time
import signal
import cv2
import numpy as np
import torch
import yaml

from .rgbd_droid import load_provider, Printer, RawStream
from .live_io import atomic_json, journal_frames, frame_record, point_cloud, publish_cloud, preview_next_frame


def run(session, provider):
    session, provider = session.resolve(), provider.resolve()
    stop = False
    def requested(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    def stopping():
        return stop or (session / "stop_preview").exists()
    status = {"status": "loading", "processed_frames": 0, "keyframes": 0}
    atomic_json(session / "preview_status.json", status)
    try:
        api = load_provider(provider)
        while not stopping():
            rows = journal_frames(session / "capture" / "frames.jsonl")
            if rows:
                break
            time.sleep(.1)
        if stopping():
            return
        records = [frame_record(row) for row in rows]
        stream = RawStream(SimpleNamespace(frames=records, depth_scale=1000.))
        image, depth, intrinsic = stream.read(0)
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        np.random.seed(43); torch.manual_seed(43); torch.cuda.manual_seed_all(43)
        torch.backends.cudnn.benchmark = False
        cfg = yaml.safe_load((provider / "configs/droid_w.yaml").read_text())
        cfg.update(scene="live_preview", debug=False, save_gt_poses=False)
        cfg["data"] = {"output": str(session / "preview_cache"), "input_folder": str(session)}
        cfg["cam"].update(H_out=image.shape[-2], W_out=image.shape[-1])
        cfg["mapping"]["enable"] = False
        cfg["mapping"]["uncertainty_params"]["activate"] = False
        t = cfg["tracking"]
        # Preview memory is bounded; this does not limit the saved sequence.
        t["buffer"] = 512
        # Lighter preview cadence; final run-rgbd retains the baseline 6.
        # Visual motion may still insert an earlier keyframe.
        t["force_keyframe_every_n_frames"] = 18
        t["frontend"].update(enable_opt_dyn_mask=False, enable_online_ba=False, enable_loop=False)
        t["backend"].update(metric_depth_reg=True, normalize=False)
        t["uncertainty_params"].update(activate=False, visualize=False, enable_affine_transform=False, gamma_depth=.05)
        atomic_json(session / "preview_config.json", cfg)
        net = api.DroidNet()
        weights = OrderedDict((k.replace("module.", ""), v) for k, v in torch.load(str(provider / "pretrained/droid.pth"), weights_only=True).items())
        for name in ("update.weight.2.weight", "update.weight.2.bias", "update.delta.2.weight", "update.delta.2.bias"):
            weights[name] = weights[name][:2]
        net.load_state_dict(weights); net.cuda().eval()
        video = api.DepthVideo(cfg, Printer())
        current = {"depth": None}
        api.motion_module.get_metric_depth_estimator = lambda cfg: None
        def measured_depth(estimator, timestamp, image, cfg, device, **kwargs):
            return current["depth"].to(device)
        api.motion_module.predict_metric_depth = measured_depth
        motion = api.motion_module.MotionFilter(net, video, cfg, thresh=t["motion_filter"]["thresh"])
        frontend = api.Frontend(net, video, cfg)
        backend = api.Backend(net, video, cfg)
        i, last_ba, last_publish = 0, 0, 0.
        processed, skipped = 0, 0
        cache = {}
        started = time.monotonic()
        with torch.no_grad():
            while not stopping():
                rows = journal_frames(session / "capture" / "frames.jsonl")
                records.extend(frame_record(row) for row in rows[len(records):])
                if i >= len(records):
                    time.sleep(.05)
                    continue
                if video.counter.value >= t["buffer"]-18:
                    status.update(status="limited", message="实时关键帧缓存已满；原始采集继续，停止后仍会完整建图。")
                    break
                image, depth, intrinsic = stream.read(i)
                current["depth"] = depth
                before = video.counter.value
                forced = motion.track(i, image, intrinsic)
                if video.counter.value > before:
                    prior = video.mono_disps[before]
                    valid = prior > 0
                    if valid.any():
                        video.disps[before] = torch.where(valid, prior, prior[valid].median())
                frontend(forced, None)
                n = video.counter.value
                if frontend.is_initialized and n >= last_ba+64:
                    backend.dense_ba(steps=2, enable_update_uncer=False, enable_udba=False)
                    last_ba = n
                if not torch.isfinite(video.poses[:n]).all():
                    raise RuntimeError("Nonfinite live trajectory")
                processed += 1
                next_i = preview_next_frame(i, len(records))
                skipped += next_i-i-1
                i = next_i
                status.update(status="tracking" if frontend.is_initialized else "initializing",
                    processed_frames=processed, skipped_preview_frames=skipped,
                    keyframes=n, backlog=len(records)-i,
                    fps=processed/max(.01, time.monotonic()-started))
                atomic_json(session / "preview_status.json", status)
                if frontend.is_initialized and time.monotonic()-last_publish > 1.:
                    import lietorch
                    twcs = lietorch.SE3(video.poses[:n].clone()).inv().matrix().cpu().numpy()
                    ordinals = video.timestamp[:n].cpu().numpy().astype(int)
                    select = np.linspace(0, n-1, min(n, 100)).astype(int)
                    points = []
                    used = set()
                    for j in select:
                        ordinal = int(ordinals[j])
                        used.add(ordinal)
                        if ordinal not in cache:
                            im, dep, k = stream.read(ordinal)
                            color = (im[0].permute(1, 2, 0).numpy()[:, :, ::-1]*255).clip(0, 255).astype(np.uint8)
                            cache[ordinal] = (dep.numpy()*1000, color, k.numpy())
                        dep, color, k = cache[ordinal]
                        points.append(point_cloud(dep, color, k, twcs[j], stride=10))
                    cache = {k: v for k, v in cache.items() if k in used}
                    publish_cloud(session, np.concatenate(points), kind="preview", revision=time.time_ns())
                    last_publish = time.monotonic()
        if stopping():
            status["status"] = "stopped"
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        atomic_json(session / "preview_status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.session, args.provider_root)
