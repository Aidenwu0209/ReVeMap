"""Isolated SAM3/VLM workers used by both the pipeline and standalone tests."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from .common import CONFIG_ROOT, PROMPT, event, read, selected_frames, sha, write


def name_tasks(tasks, model_id, config, output):
    from .vlm import create_namer
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_config = {**config.get("models", {}).get(model_id, {}), "log_dir": str(output)}
    model = None
    start = time.monotonic()
    try:
        for task in tasks:
            for crop in task["crops"]:
                if sha(crop["file"]) != crop["sha256"]:
                    raise ValueError("crop changed before inference")
        write(output / "INPUTS.json", {"model": model_id, "tasks": tasks,
                                      "prompt": PROMPT, "GT_in_requests": False})
        model = create_namer(model_id, model_config)
        write(output / "MODEL.json", model.audit)
        records = []
        for task in tasks:
            tick = time.monotonic()
            crops = [{**crop, **model.infer(crop["file"])} for crop in task["crops"]]
            result = {"task_id": task["task_id"], "started_at": tick,
                      "completed_at": time.monotonic(), "crops": crops}
            write(output / f'response_{task["task_id"]:06}.json', result)
            records.extend(crops)
        write(output / "RECORDS.json", records)
        write(output / "COMPLETE.json", {
            "status": "completed", "model": model_id, "requests": len(tasks),
            "crops": len(records), "crops_executed": sum(bool(x.get("executed")) for x in records),
            "seconds_including_load": time.monotonic() - start,
            "mean_request_seconds": sum(x["request_seconds"] for x in records) / max(1, len(records)),
            "timing_scope": "crop naming only, not raw RGB-D pipeline FPS"})
    except BaseException as error:
        # An exception from a provider may contain a URL: persist only its type.
        write(output / "FAILURE.json", {"error_type": type(error).__name__, "model": model_id})
        raise
    finally:
        if model is not None:
            model.close()


def infer_sam3(args, config):
    import numpy as np
    import torch
    from ..contracts import load_manifest
    from ..sam3_mapping import load_model, read_frame, save_overlay
    from ..sam3_refine import infer_claims
    sys.path.insert(0, config["sam3_source"])
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    (output / "crops").mkdir()
    manifest = load_manifest(args.manifest)
    plan_path = getattr(args, 'view_plan', None)
    if plan_path:
        from .view_selection import frames_from_plan
        frames, view_plan = frames_from_plan(args.manifest, plan_path, args.view_plan_sha256)
        write(output / 'VIEW_PLAN.json', view_plan)
    else:
        if getattr(args, 'view_policy', 'stride') != 'stride' or getattr(args, 'view_budget', None) is not None:
            raise ValueError('nondefault semantic selection requires a verified view plan')
        frames = selected_frames(manifest.frames, args.stride)
        view_plan = {'policy': 'stride', 'effective_budget': len(frames),
                     'selected_frame_ids': [frame.frame_id for frame in frames]}
    taxonomy = read(CONFIG_ROOT / "sam3_indoor_v1.json")["classes"]
    processor, audit = load_model(Path(config["sam3_checkpoint"]), config["sam3_sha256"])
    write(output / "MODEL.json", audit)
    tasks, records = [], []
    started = time.monotonic()
    from ..sam3_text_cache import cached_text_features
    with cached_text_features(processor) as text_cache:
        for ordinal, frame in enumerate(frames):
            event(output, "frame_start", frame_id=frame.frame_id)
            tick = time.monotonic()
            image, depth, _ = read_frame(frame)
            claims, packed = infer_claims(processor, image, depth.shape, taxonomy)
            semantic, instance, confidence = claims.finalize()
            path = output / "frames" / f"{frame.frame_id:06}.npz"
            np.savez_compressed(path, semantic=semantic, local_instance=instance, confidence=confidence,
                                raw_masks_packed=packed, depth_shape=depth.shape)
            candidates = []
            for row in claims.records:
                if row["class_id"] in (10, 19):
                    continue
                yy, xx = np.nonzero(instance == row["mask_id"])
                if len(xx) < 300:
                    continue
                bbox = (int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1))
                if bbox[2] - bbox[0] >= 10 and bbox[3] - bbox[1] >= 10:
                    candidates.append((len(xx), row, bbox))
            candidates.sort(key=lambda x: (-x[0], x[1]["mask_id"]))
            crops = []
            for area, row, bbox in candidates[:8]:
                x0, y0, x1, y1 = bbox
                dx, dy = max(2, round((x1-x0)*.15)), max(2, round((y1-y0)*.15))
                height, width = depth.shape
                box = (max(0, round((x0-dx)*image.width/width)), max(0, round((y0-dy)*image.height/height)),
                       min(image.width, round((x1+dx)*image.width/width)), min(image.height, round((y1+dy)*image.height/height)))
                crop = output / "crops" / f'{frame.frame_id:06}_{row["mask_id"]:04}.png'
                image.crop(box).save(crop)
                crops.append({"frame_id": frame.frame_id, "mask_id": row["mask_id"], "sam_class_id": row["class_id"],
                              "file": str(crop), "sha256": sha(crop), "bbox_rgb": box, "mask_pixels": area})
            tasks.append({"task_id": ordinal, "frame_id": frame.frame_id, "crops": crops})
            records.append({"frame_id": frame.frame_id, "mask_sha256": sha(path),
                            "color_sha256": sha(frame.color_path), "depth_sha256": sha(frame.depth_path),
                            "seconds": time.monotonic() - tick, "crops": len(crops)})
            write(output / "FRAMES.json", records)
            if ordinal in (0, len(frames)-1):
                image.save(output / "frames" / f'{frame.frame_id:06}_rgb.jpg')
                save_overlay(image, semantic, taxonomy, output / "frames" / f'{frame.frame_id:06}_overlay.png')
            torch.cuda.empty_cache()
            event(output, "frame_complete", frame_id=frame.frame_id)
    write(output / "CROP_TASKS.json", tasks)
    write(output / "COMPLETE.json", {"status": "completed", "selected_frames": len(frames),
          "raw_frames": len(manifest.frames), "stride": args.stride,
          "view_policy": view_plan['policy'], "view_budget": view_plan['effective_budget'],
          "view_plan_sha256": sha(plan_path) if plan_path else None,
          "seconds_after_load": time.monotonic()-started, "GT_used": False,
          "text_feature_cache": text_cache})
    # The process exits before a local VLM is loaded, releasing ALL SAM3 CUDA state.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "sam3", "naming", "mapping", "fusion", "backfill"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--model", default="qwen3vl_2b_bf16")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--view-policy", choices=('stride', 'quality', 'quality-diverse'), default='stride')
    parser.add_argument("--view-budget", type=int)
    parser.add_argument("--view-plan", type=Path)
    parser.add_argument("--view-plan-sha256")
    parser.add_argument('--semantic-confidence-policy', choices=('legacy', 'track'), default='legacy')
    parser.add_argument('--semantic-conflict-policy', choices=('consensus', 'abstain'), default='consensus')
    args = parser.parse_args()
    from ..live_io import guard_parent_process
    guard_parent_process()
    config = read(args.runtime)
    if args.stage == "select":
        from .view_selection import build_view_plan
        if args.manifest is None or args.trajectory is None:
            raise ValueError('view selection requires manifest and final refill trajectory')
        plan = build_view_plan(args.manifest, args.trajectory, policy=args.view_policy,
                               stride=args.stride, budget=args.view_budget)
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / 'VIEW_PLAN.json', plan)
    elif args.stage == "sam3":
        infer_sam3(args, config)
    elif args.stage == "naming":
        name_tasks(read(args.tasks), args.model, config, args.output)
    elif args.stage == "mapping":
        from ..rgbd_mapping import run_rgbd_mapping
        run_rgbd_mapping(manifest_path=args.manifest, output_dir=args.output,
            provider_root=Path(config["provider_root"]), gpu_python=Path(config["gpu_python"]),
            cpu_python=Path(config["cpu_python"]), threads=config.get("threads", 2),
            stage_timeout_s=config.get("stage_timeout", 7200))
    elif args.stage == "fusion":
        from .fusion import main as fuse
        fuse(argparse.Namespace(arm_root=args.output, semantic_confidence_policy=args.semantic_confidence_policy,
                                semantic_conflict_policy=args.semantic_conflict_policy))
    else:
        from .backfill import main as backfill
        backfill(argparse.Namespace(arm_root=args.output))


if __name__ == "__main__":
    main()
