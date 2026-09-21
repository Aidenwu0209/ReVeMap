"""Visually refine every non-keyframe against fixed measured-graph anchors.

The original filler and motion-only BA are retained. Source projection uses
only measured sensor disparity. A zero-depth anchor may have a numerical
disparity placeholder only when the original filler never uses it as a source.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import (
    PoseRecord,
    load_manifest,
    sha256_file,
    validate_se3,
    write_manifest,
    write_trajectory,
)
from .rgbd_empty_anchor import Policy
from .rgbd_measured import _verified_result


def run_visual_refill(
    dense_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    provider_root: Path,
) -> dict:
    """Create a full raw trajectory with fixed graph anchors and actual BA support.

    Run in a dedicated provider/GPU process. All paths are resolved before the
    provider changes its working directory. Outputs are create-only, and an
    unsupported frame fails explicitly without writing a completed result.
    """
    import cv2
    import torch
    from .rgbd_droid import RawStream, Printer, load_provider

    dense, source = Path(dense_dir).resolve(), Path(graph_dir).resolve()
    out, provider_path = Path(output_dir).resolve(), Path(provider_root).resolve()
    receipt = _verified_result(source)
    dense_receipt = _verified_result(dense)
    graph_result_sha = sha256_file(source / "result.json")
    dense_result_sha = sha256_file(dense / "result.json")
    if dense_result_sha != receipt.get("source_receipt_sha256"):
        raise ValueError("Graph and dense source receipts do not match")
    provider_sources = dense_receipt.get("source_sha256")
    if not isinstance(provider_sources, dict) or not provider_sources:
        raise ValueError("Dense result has no provider source hashes")
    for name, expected in provider_sources.items():
        path = (provider_path / name).resolve()
        if not path.is_relative_to(provider_path) or sha256_file(path) != expected:
            raise ValueError(f"Provider source SHA mismatch: {path}")
    manifest = load_manifest(source / "raw_manifest.json")
    dense_manifest = load_manifest(dense / "raw_manifest.json")
    if manifest.as_dict() != dense_manifest.as_dict():
        raise ValueError("Graph and dense raw inputs differ")
    with np.load(source / "optimized_keyframes.npz", allow_pickle=False) as anchors:
        timestamps = anchors["timestamps"]
        world = anchors["poses_T_world_camera"]
    if (
        timestamps.ndim != 1
        or not len(timestamps)
        or not np.isfinite(timestamps).all()
        or not np.equal(timestamps, np.rint(timestamps)).all()
    ):
        raise ValueError("Anchor timestamps must be nonempty integer raw ordinals")
    ids = timestamps.astype(int)
    n = len(ids)
    if (
        ids[0] != 0
        or ids[-1] != len(manifest.frames) - 1
        or not np.all(np.diff(ids) > 0)
    ):
        raise ValueError("Full strictly ordered anchor endpoints are required")
    if world.shape != (n, 4, 4):
        raise ValueError("One world-from-camera transform is required per anchor")
    for k, transform in enumerate(world):
        validate_se3(transform, f"graph anchor {k}")
    checkpoint = provider_path / "pretrained/droid.pth"
    if sha256_file(checkpoint) != dense_receipt.get("checkpoint_sha256"):
        raise ValueError("DROID checkpoint differs from the dense source")

    provider = load_provider(provider_path)
    import lietorch

    out.mkdir(parents=True, exist_ok=False)
    write_manifest(out / "raw_manifest.json", manifest)
    policy = Policy(ids, len(manifest.frames), out)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    np.random.seed(43)
    torch.backends.cudnn.benchmark = False
    cfg = json.loads((dense / "effective_config.json").read_text())
    cfg["tracking"]["buffer"] = n + 32
    cfg["tracking"]["force_keyframe_every_n_frames"] = 1
    cfg["data"] = {"output": str(out), "input_folder": str(out)}
    key = f"{manifest.dataset}/{manifest.sequence_id}"
    cfg["scene"] = key.replace("/", "_")
    (out / "effective_config.json").write_text(json.dumps(cfg, indent=2))
    stream = RawStream(manifest)
    net = provider.DroidNet()
    state = OrderedDict(
        (k.replace("module.", ""), v)
        for k, v in torch.load(str(checkpoint), weights_only=True).items()
    )
    for name in [
        "update.weight.2.weight",
        "update.weight.2.bias",
        "update.delta.2.weight",
        "update.delta.2.bias",
    ]:
        state[name] = state[name][:2]
    net.load_state_dict(state)
    net.cuda().eval()
    video = provider.DepthVideo(cfg, Printer())
    current = {"depth": None}
    motion_module = provider.motion_module
    old_estimator, old_predict = (
        motion_module.get_metric_depth_estimator,
        motion_module.predict_metric_depth,
    )
    motion_module.get_metric_depth_estimator = lambda cfg: None

    def sensor_depth(estimator, timestamp, image, cfg, device, **kwargs):
        if current["depth"] is None:
            raise RuntimeError("No measured depth is bound to the current anchor")
        return current["depth"].to(device)

    motion_module.predict_metric_depth = sensor_depth
    try:
        motion = motion_module.MotionFilter(net, video, cfg)
        started = time.time()
        camera_world = np.linalg.inv(world)
        pose7 = np.c_[
            camera_world[:, :3, 3],
            Rotation.from_matrix(camera_world[:, :3, :3]).as_quat(),
        ]
        with torch.no_grad():
            for k, i in enumerate(ids):
                image, depth, intrinsic = stream.read(int(i))
                current["depth"] = depth
                motion.track(int(i), image, intrinsic)
                if video.counter.value != k + 1:
                    raise RuntimeError(
                        "Motion filter did not insert every fixed graph anchor"
                    )
                video.poses[k] = torch.as_tensor(
                    pose7[k],
                    dtype=video.poses.dtype,
                    device=video.poses.device,
                )
                prior = video.mono_disps[k]
                valid = prior > 0
                if not valid.any():
                    policy.allow_unused_empty(k)
                    # Only a numerical initialization. Measured mono remains
                    # zero and this node cannot supply any filler observation.
                    video.disps[k] = 1
                else:
                    video.disps[k] = torch.where(valid, prior, prior[valid].median())
                if k % 100 == 0:
                    print("REFILL_ANCHORS", key, k, n, flush=True)
            del motion
            torch.cuda.empty_cache()
            fixed = video.poses[:n].clone()
            original_ba = video.ba
            counts = {"motion_only_ba_calls": 0, "masked_source_pixels": 0}

            def ba(target, weight, eta, ii, jj, *args, **kwargs):
                valid = video.mono_disps[ii] > 0
                mask = valid.reshape(
                    *([1] * (weight.ndim - 4)), len(ii), video.ht // 8, video.wd // 8, 1
                )
                counts["masked_source_pixels"] += int((~valid).sum())
                counts["motion_only_ba_calls"] += 1
                kwargs["motion_only"] = True
                masked_weight = weight * mask
                policy.audit_ba(
                    video, target, masked_weight, valid, ii, jj, args, kwargs
                )
                value = original_ba(target, masked_weight, eta, ii, jj, *args, **kwargs)
                if not torch.equal(video.poses[:n], fixed):
                    raise RuntimeError("Visual refill changed a fixed graph anchor")
                return value

            video.ba = ba
            filler = provider.PoseTrajectoryFiller(cfg, net, video, Printer())
            all7 = torch.empty((len(manifest.frames), 7), device="cuda")
            all7[torch.as_tensor(ids, device="cuda")] = fixed
            known = set(ids.tolist())
            timestamps, images, intrinsics = [], [], []
            optimized_count = 0

            def flush():
                nonlocal optimized_count
                if not timestamps:
                    return
                policy.begin_batch(timestamps)
                result = filler._PoseTrajectoryFiller__fill(
                    timestamps, images, None, intrinsics, None
                )[0]
                all7[torch.as_tensor(timestamps, device="cuda")] = result.data
                optimized_count += len(timestamps)
                timestamps.clear()
                images.clear()
                intrinsics.clear()

            for i in range(len(manifest.frames)):
                if i in known:
                    continue
                image, _, intrinsic = stream.read(i)
                timestamps.append(i)
                images.append(image)
                intrinsics.append(intrinsic)
                if len(timestamps) == 16:
                    flush()
                if i % 200 == 0:
                    print(
                        "VISUAL_REFILL",
                        key,
                        i,
                        len(manifest.frames),
                        round(time.time() - started, 1),
                        flush=True,
                    )
            flush()
            if (
                optimized_count != len(manifest.frames) - n
                or not torch.isfinite(all7).all()
                or video.counter.value != n
            ):
                raise RuntimeError(
                    "Visual refill did not return every raw frame with finite poses"
                )
            policy.finalize(video)
            full = lietorch.SE3(all7).inv().matrix().cpu().numpy()
        if not np.allclose(full[ids], world, atol=1e-5, rtol=1e-5):
            raise RuntimeError(
                "Refilled anchor matrices differ from the measured graph"
            )
        np.save(out / "final_raw_poses.npy", full)
        np.savez_compressed(
            out / "final_keyframes.npz", timestamps=ids, poses=fixed.cpu().numpy()
        )
        records = [
            PoseRecord(
                f.frame_id,
                f.timestamp_us,
                transform,
                True,
                "measured_graph_with_sensor_visual_refill",
            )
            for f, transform in zip(manifest.frames, full)
        ]
        write_trajectory(
            out / "trajectory.json",
            records,
            sequence_id=manifest.sequence_id,
            arm="candidate",
            metadata={
                "gt_consumed": False,
                "nonkeyframes_visually_optimized": True,
                "fixed_graph_anchors_preserved": True,
                "empty_depth_policy": "only_anchors_unused_by_all_original_filler_sources",
            },
        )
        _verified_result(source)
        _verified_result(dense)
        if (
            sha256_file(source / "result.json") != graph_result_sha
            or sha256_file(dense / "result.json") != dense_result_sha
        ):
            raise RuntimeError("A source receipt changed during visual refill")
        report = {
            "gt_consumed": False,
            "quality_assessment": "requires_separate_evaluation",
            "identity_fallback_used": False,
            "raw_frame_count": len(manifest.frames),
            "final_pose_count": len(records),
            "keyframe_count": n,
            "nonkeyframes_visually_optimized_count": optimized_count,
            "fixed_anchor_tensor_byte_identical": True,
            "fixed_anchor_matrix_max_abs_difference": float(
                np.max(np.abs(full[ids] - world))
            ),
            "ba_audit": counts,
            "empty_anchor_count": len(policy.empty),
            "source_graph_receipt_sha256": graph_result_sha,
            "source_dense_receipt_sha256": dense_result_sha,
            "wrapper_sha256": sha256_file(Path(__file__)),
            "raw_stream_helper_sha256": sha256_file(
                Path(__file__).with_name("rgbd_droid.py")
            ),
            "empty_anchor_helper_sha256": sha256_file(
                Path(__file__).with_name("rgbd_empty_anchor.py")
            ),
            "runtime_s": time.time() - started,
            "seal": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
        }
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(
            "VISUAL_REFILL_SEALED",
            key,
            len(manifest.frames),
            n,
            optimized_count,
            flush=True,
        )
        return report
    finally:
        motion_module.get_metric_depth_estimator = old_estimator
        motion_module.predict_metric_depth = old_predict
        if policy.log is not None and not policy.log.closed:
            policy.log.close()
