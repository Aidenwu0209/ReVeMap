"""Orbbec hardware-D2C capture or paced raw RGB-D replay, without pose filtering."""
import argparse
from pathlib import Path
import json
import signal
import time
import cv2
import numpy as np

from .contracts import load_manifest
from .live_io import atomic_json, seal_capture


def decode_color(frame, ob):
    data = np.asarray(frame.get_data(), dtype=np.uint8)
    h, w, fmt = frame.get_height(), frame.get_width(), frame.get_format()
    if fmt == ob.OBFormat.RGB:
        return data.reshape(h, w, 3)[:, :, ::-1].copy()
    if fmt == ob.OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == ob.OBFormat.MJPG:
        result = cv2.imdecode(data.reshape(-1), cv2.IMREAD_COLOR)
        if result is None:
            raise RuntimeError("相机 MJPEG 解码失败")
        return result
    if fmt in (ob.OBFormat.YUYV, ob.OBFormat.YUY2):
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    raise RuntimeError(f"Unsupported color format: {fmt}")


def camera_frames(pipeline=None):
    import pyorbbecsdk as ob
    ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
    if pipeline is None:
        context = ob.Context()
        if context.query_devices().get_count() == 0:
            raise RuntimeError("未检测到奥比中光相机。请接入 ssh33 的 USB 3 接口后重试。")
    # Injection permits SDK playback validation of the identical D2C path.
    pipe = ob.Pipeline() if pipeline is None else pipeline
    pipe.enable_frame_sync()
    profiles = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    supported = {ob.OBFormat.RGB, ob.OBFormat.BGR, ob.OBFormat.MJPG,
                 ob.OBFormat.YUYV, ob.OBFormat.YUY2}
    candidates = [profiles[i].as_video_stream_profile() for i in range(profiles.get_count())]
    candidates = [p for p in candidates if p.get_format() in supported and p.get_fps() <= 30]
    candidates.sort(key=lambda p: (abs(p.get_width()-640)+abs(p.get_height()-480), -p.get_fps()))
    cfg = ob.Config()
    chosen = None
    for color_profile in candidates:
        depth_profiles = pipe.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
        if depth_profiles.get_count():
            chosen = color_profile
            cfg.enable_stream(color_profile)
            cfg.enable_stream(depth_profiles[0])
            break
    if chosen is None:
        raise RuntimeError("相机没有兼容的硬件 D2C 配置；未使用未经验证的对齐替代。")
    cfg.set_align_mode(ob.OBAlignMode.HW_MODE)
    cfg.set_depth_scale_require(True)
    cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipe.start(cfg)
    last = time.monotonic()
    try:
        while True:
            fs = pipe.wait_for_frames(1000)
            if fs is None:
                if time.monotonic()-last > 8:
                    raise RuntimeError("相机连续 8 秒未返回 RGB-D，请检查 USB 连接。")
                yield None
                continue
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if c is None or d is None:
                yield None
                continue
            last = time.monotonic()
            h, w = c.get_height(), c.get_width()
            if (d.get_height(), d.get_width()) != (h, w):
                raise RuntimeError("D2C 深度与彩色图尺寸不匹配")
            tc, td = c.get_timestamp_us(), d.get_timestamp_us()
            if abs(tc-td) > 15000:
                yield None
                continue
            k = d.get_stream_profile().as_video_stream_profile().get_intrinsic()
            if (k.width, k.height) != (w, h):
                raise RuntimeError("D2C 内参与图像尺寸不匹配")
            scale = d.get_depth_scale()
            if not np.isfinite(scale) or scale <= 0:
                raise RuntimeError("Invalid SDK depth scale")
            depth = np.asarray(d.get_data()).view(np.uint16).reshape(h, w)
            depth = np.rint(depth.astype(float)*scale).clip(0, 65535).astype(np.uint16)
            color = decode_color(c, ob)
            # Same centre crop and calibration convention as the validated SDK exporter.
            cw, ch = min(w, round(h*640/480)), min(h, round(w*480/640))
            ox, oy = (w-cw)//2, (h-ch)//2
            color = cv2.resize(color[oy:oy+ch, ox:ox+cw], (640, 480))
            depth = cv2.resize(depth[oy:oy+ch, ox:ox+cw], (640, 480), interpolation=cv2.INTER_NEAREST)
            intrinsic = (k.fx*640/cw, k.fy*480/ch, (k.cx-ox)*640/cw, (k.cy-oy)*480/ch)
            yield color, depth, intrinsic, tc, {"depth_timestamp_us": td, "sdk_frame_id": c.get_index(), "sdk_depth_scale_mm": scale}
    finally:
        pipe.stop()


def replay_frames(manifest_path, fps, maximum):
    manifest = load_manifest(manifest_path)
    frames = manifest.frames[:maximum] if maximum else manifest.frames
    for frame in frames:
        color = cv2.imread(str(frame.color_path))
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None or depth.dtype != np.uint16:
            raise RuntimeError(f"Invalid replay RGB-D: {frame.frame_id}")
        intrinsic = frame.intrinsics
        if frame.rotate_ccw:
            w = depth.shape[1]
            color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)
            depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
            fx, fy, cx, cy = intrinsic
            intrinsic = (fy, fx, cy, w-1-cx)
        depth = np.rint(depth.astype(float)*1000./manifest.depth_scale).clip(0, 65535).astype(np.uint16)
        yield color, depth, intrinsic, frame.timestamp_us, {"source_frame_id": frame.frame_id}
        time.sleep(1./fps)


def capture(args):
    root = args.session / "capture"
    root.mkdir()
    for name in ("color", "depth"):
        (root / name).mkdir()
    cv2.setNumThreads(1)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    source = (replay_frames(args.replay, args.fps, args.max_frames) if args.replay else camera_frames())
    count, rejected, previous = 0, 0, -1
    started = time.monotonic()
    status = {"status": "starting", "frames": 0}
    atomic_json(args.session / "capture_status.json", status)
    try:
        with (root / "frames.jsonl").open("x") as journal:
            for sample in source:
                if stopped or (args.session / "stop_capture").exists():
                    break
                if sample is None:
                    rejected += 1
                    continue
                color, depth, intrinsic, timestamp, audit = sample
                if timestamp <= previous:
                    raise RuntimeError("相机时间戳不单调，已停止采集并保留数据。")
                previous = timestamp
                if color.shape[:2] != depth.shape:
                    raise RuntimeError("RGB-D image shape mismatch")
                cp, dp = root / "color" / f"{count:06d}.png", root / "depth" / f"{count:06d}.png"
                if not cv2.imwrite(str(cp), color, [cv2.IMWRITE_PNG_COMPRESSION, 1]) or not cv2.imwrite(str(dp), depth, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError("RGB-D 写盘失败，请检查剩余空间。")
                journal.write(json.dumps({"frame_id": count, "timestamp_us": timestamp,
                    "color_path": str(cp), "depth_path": str(dp), "intrinsics": list(intrinsic),
                    "rotate_ccw": False, **audit})+"\n")
                journal.flush()
                count += 1
                if count % 2 == 0 or count == 1:
                    for name, img in (("color", color), ("depth", cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255/4500), cv2.COLORMAP_TURBO))):
                        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            tmp = args.session / (name+".jpg.tmp")
                            tmp.write_bytes(encoded.tobytes())
                            tmp.replace(args.session / (name+".jpg"))
                status.update(status="recording", frames=count, rejected_pairs=rejected,
                    elapsed_s=time.monotonic()-started, fps=count/max(.01, time.monotonic()-started),
                    valid_depth_fraction=float((depth > 0).mean()))
                atomic_json(args.session / "capture_status.json", status)
        source.close()
        count = seal_capture(root, source="gui_replay_raw_rgbd" if args.replay else "live_hardware_d2c_center_crop_no_pose_filter")
        status.update(status="sealed", frames=count)
    except BaseException as error:
        status.update(status="failed", error=str(error), frames=count)
        # Seal recoverable frames, but do not automatically process a failed capture.
        if count and not (root / "manifest.json").exists():
            seal_capture(root, source="interrupted_gui_capture")
        raise
    finally:
        source.close()
        atomic_json(args.session / "capture_status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    capture(parser.parse_args())
