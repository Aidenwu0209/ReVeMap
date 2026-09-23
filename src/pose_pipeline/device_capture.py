"""Orbbec hardware-D2C capture or paced raw RGB-D replay, without pose filtering."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import signal
import time
import cv2
import numpy as np

from .contracts import load_manifest, write_manifest, sha256_file
from .live_io import atomic_json, seal_capture, guard_gui_parent
from .live_capture import camera_frames, decode_color
from .wireless_capture import wireless_frames
from .depth_denoise import ALGORITHM, DEFAULT_CONFIG, denoise_depth


def seal_depth_capture(root, source, processing):
    """Publish separate processed and raw manifests; retain acquisition evidence."""
    enabled = processing["enabled"]
    count = seal_capture(root, source=source + ("_" + ALGORITHM if enabled else ""))
    if enabled:
        manifest = load_manifest(root / "manifest.json")
        raw_frames = tuple(replace(frame, depth_path=root / "depth_raw" / frame.depth_path.name)
                           for frame in manifest.frames)
        write_manifest(root / "manifest_raw.json", replace(
            manifest, frames=raw_frames, source=source + "_raw_received"))
    atomic_json(root / "depth_processing.json", processing)
    return count






def ipad_frames(bind, port, minimum_confidence=1, maximum_depth_m=5.0, idle_timeout=15.0):
    """Live iPad LiDAR capture: accept one Scan-app protocol-v1 TCP stream.

    Colour is kept at native camera resolution; depth is resampled (nearest)
    onto the same grid so the capture-stage shape check passes. Stream close
    (app stopped) ends the capture naturally, like replay exhaustion.
    """
    import socket
    from . import ipad_protocol as receiver
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind, port))
    server.listen(1)
    conn = None
    locked_intrinsics = None
    try:
        conn, _ = server.accept()
        conn.settimeout(idle_timeout)
        stream = conn.makefile("rb")
        while True:
            try:
                header, color_payload, depth_payload, confidence_payload = receiver.receive_packet(stream)
            except EOFError:
                return
            except socket.timeout:
                raise RuntimeError(f"iPad 数据流 {idle_timeout:.0f} 秒无数据，已停止采集并保留已收帧。")
            except receiver.PacketError as error:
                raise RuntimeError(f"iPad 数据包校验失败: {error}")
            frame = receiver.decode_frame(header, color_payload, depth_payload,
                                          confidence_payload, minimum_confidence,
                                          maximum_depth_m)
            color = frame["color_native"]
            depth = cv2.resize(frame["depth_mm"], (color.shape[1], color.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            if locked_intrinsics is None:
                locked_intrinsics = tuple(float(v) for v in frame["camera_native"][:4])
            intrinsic = locked_intrinsics
            yield color, depth, intrinsic, int(frame["timestamp_ns"] // 1000), {
                "ipad_session_id": frame["session_id"],
                "source_frame_id": frame["frame_id"],
                "valid_depth_ratio": frame["valid_depth_ratio"],
            }
    finally:
        if conn is not None:
            conn.close()
        server.close()


def packet_frames(directory, minimum_confidence=1, maximum_depth_m=5.0):
    """Read sealed local recordings using the same v1 decoder and locked K."""
    from .live_upload import UploadStore
    from . import ipad_protocol as receiver
    store = UploadStore(directory.parent)
    manifest = store.manifest(directory.name)
    store.seal(directory.name)
    locked_intrinsics = None
    for i in range(len(manifest["frames"])):
        with (directory / f"f{i:06d}.bin").open("rb") as stream:
            header, color_payload, depth_payload, confidence_payload = receiver.receive_packet(stream)
            if stream.read(1):
                raise RuntimeError("Unexpected data after recorded frame")
        frame = receiver.decode_frame(header, color_payload, depth_payload, confidence_payload,
                                      minimum_confidence, maximum_depth_m)
        color = frame["color_native"]
        depth = cv2.resize(frame["depth_mm"], (color.shape[1], color.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        if locked_intrinsics is None:
            locked_intrinsics = tuple(float(v) for v in frame["camera_native"][:4])
        yield color, depth, locked_intrinsics, int(frame["timestamp_ns"] // 1000), {
            "ipad_session_id": frame["session_id"], "source_frame_id": frame["frame_id"],
            "valid_depth_ratio": frame["valid_depth_ratio"], "upload_id": directory.name,
        }


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
    # Only new wireless recordings are filtered; replay/import never filters twice.
    mode = getattr(args, "depth_denoise", "bilateral") if args.wireless_host else "off"
    if mode not in ("off", "bilateral"):
        raise ValueError("depth-denoise must be off or bilateral")
    enabled = mode == "bilateral"
    if enabled:
        (root / "depth_raw").mkdir()
    processing = {"schema": "depth_processing.v1", "enabled": enabled,
                  "algorithm": ALGORITHM, "mode": mode, "parameters": dict(DEFAULT_CONFIG),
                  "implementation_sha256": sha256_file(Path(__file__).with_name("depth_denoise.py")),
                  "raw_depth_directory": "depth_raw" if enabled else "depth",
                  "working_depth_directory": "depth", "frames": 0,
                  "total_filter_ms": 0., "changed_pixels": 0, "valid_pixels": 0,
                  "max_abs_change_mm": 0, "zero_mask_preserved": True}
    atomic_json(root / "depth_processing.json", processing)
    cv2.setNumThreads(1)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    guard_gui_parent()
    packet_dir = getattr(args, "packet_dir", None)
    source = (packet_frames(packet_dir, args.ipad_min_confidence, args.ipad_max_depth_m) if packet_dir
              else replay_frames(args.replay, args.fps, args.max_frames) if args.replay
              else ipad_frames(args.ipad_bind, args.ipad_port, args.ipad_min_confidence,
                               args.ipad_max_depth_m, args.ipad_idle_timeout) if args.ipad_port
              else wireless_frames(args.wireless_host, args.wireless_port) if args.wireless_host
              else camera_frames())
    count, rejected, previous = 0, 0, -1
    started = time.monotonic()
    status = {"status": "starting", "frames": 0, "depth_processing": processing}
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
                raw_depth = depth
                audit = dict(audit)
                if enabled:
                    rp = root / "depth_raw" / dp.name
                    if not cv2.imwrite(str(rp), raw_depth, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                        raise RuntimeError("原始深度写盘失败；停止采集，未提交本帧。")
                    depth, stats = denoise_depth(raw_depth, mode=mode)
                    if not np.array_equal(raw_depth == 0, depth == 0):
                        raise RuntimeError("Depth denoising changed the valid-depth mask")
                    audit.update(raw_depth_path=str(rp), raw_depth_sha256=sha256_file(rp),
                                 depth_processing=stats)
                if not cv2.imwrite(str(cp), color, [cv2.IMWRITE_PNG_COMPRESSION, 1]) or not cv2.imwrite(str(dp), depth, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError("RGB-D 写盘失败，请检查剩余空间。")
                journal.write(json.dumps({"frame_id": count, "timestamp_us": timestamp,
                    "color_path": str(cp), "depth_path": str(dp), "intrinsics": list(intrinsic),
                    "rotate_ccw": False, **audit})+"\n")
                journal.flush()
                count += 1
                if enabled:
                    processing["total_filter_ms"] += stats["elapsed_ms"]
                    processing["changed_pixels"] += stats["changed_pixels"]
                    processing["valid_pixels"] += stats["valid_pixels"]
                    processing["max_abs_change_mm"] = max(processing["max_abs_change_mm"], stats["max_abs_change_mm"])
                processing["frames"] = count
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
                if args.wireless_host and args.max_frames and count >= args.max_frames:
                    break
        source.close()
        if packet_dir and (stopped or (args.session / "stop_capture").exists()):
            raise RuntimeError("Local recording import interrupted; uploaded packets retained")
        count = seal_depth_capture(root, source=("ipad_lidar_verified_local_upload" if packet_dir else
                           "gui_replay_raw_rgbd" if args.replay else
                           "ipad_lidar_tcp_live_native_grid" if args.ipad_port else
                           "wireless_orbbec_d2c_sensor_timestamps" if args.wireless_host else
                           "live_hardware_d2c_center_crop_no_pose_filter"), processing=processing)
        status.update(status="sealed", frames=count)
    except BaseException as error:
        status.update(status="failed", error=str(error), frames=count)
        # Seal recoverable frames, but do not automatically process a failed capture.
        if count and not (root / "manifest.json").exists():
            seal_depth_capture(root, source="interrupted_gui_capture", processing=processing)
        raise
    finally:
        source.close()
        atomic_json(args.session / "capture_status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--packet-dir", type=Path)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--wireless-host", help="Orbbec host receiving EtherSensePing")
    parser.add_argument("--wireless-port", type=int, default=1024, help="UDP request and reverse TCP port")
    parser.add_argument("--depth-denoise", choices=("off", "bilateral"), default="bilateral",
                        help="Wireless depth filter; raw received depth is retained separately")
    parser.add_argument("--ipad-port", type=int, default=0)
    parser.add_argument("--ipad-bind", default="0.0.0.0")
    parser.add_argument("--ipad-min-confidence", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--ipad-max-depth-m", type=float, default=5.0)
    parser.add_argument("--ipad-idle-timeout", type=float, default=15.0)
    args = parser.parse_args()
    if not 1 <= args.wireless_port <= 65535 or args.max_frames < 0:
        parser.error("wireless-port must be 1..65535 and max-frames nonnegative")
    if sum(bool(v) for v in (args.replay, args.packet_dir, args.ipad_port, args.wireless_host)) > 1:
        parser.error("Choose only one capture source")
    capture(args)
