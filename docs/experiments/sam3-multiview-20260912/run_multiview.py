"""CPU-only instance fusion from sealed SAM3 projected-mask caches.

This runner does not load models, RGB-D images, poses, or ground truth into the
fusion algorithm. It preserves every objects_geometry field except instance.
Run with the existing R1 environment, after deploying this file and code/:
  /mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1/env/bin/python \
      /mnt/d/SGF-SGA-experiments/sam3_multiview_20260912_v1/run_multiview.py \
      --key scannet/scene0030_00
Omit --key to process all five recorded scenes. Existing scene outputs are refused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
STAGE = "consensus_instances"
sys.path.insert(0, str(ROOT / "code/src"))

import numpy as np
import scipy


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=json_default) + "\n")
    temp.replace(path)


def emit(log, event, **fields):
    row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    text = json.dumps(row, default=json_default)
    log.write(text + "\n")
    log.flush()
    print(text, flush=True)


def verify_hashes(hashes):
    changed = [str(path) for path, expected in hashes.items()
               if not Path(path).is_file() or sha256_file(path) != expected]
    if changed:
        raise RuntimeError(f"Input/source changed during run: {changed}")


def frame_inventory(cache, selected):
    """Require the exact selected frame list, JSONL order, and NPZ inventory."""
    records = [json.loads(line) for line in (cache / "frames.jsonl").read_text().splitlines() if line]
    ids = [int(row["frame_id"]) for row in records]
    if ids != selected or len(set(ids)) != len(ids):
        raise ValueError("Cached frames.jsonl order/IDs differ from the frozen selection")
    if [int(row["ordinal"]) for row in records] != list(range(len(selected))):
        raise ValueError("Cached frame ordinals are incomplete")
    files = [cache / "frames" / f"{frame_id:06}.npz" for frame_id in selected]
    if set((cache / "frames").glob("*.npz")) != set(files):
        raise ValueError("Cached NPZ inventory differs from the complete selected frame set")
    return files, records


def load_frame(path, frame_id, n_points):
    mapping = {"point_ids": "visible_map_ids", "mask_ids": "projected_local_instance",
               "semantic": "projected_semantic", "confidence": "projected_confidence",
               "interior": "interior"}
    with np.load(path, allow_pickle=False) as cache:
        frame = {dest: cache[source].copy() for dest, source in mapping.items()}
    count = len(frame["point_ids"])
    if any(value.shape != (count,) for value in frame.values()):
        raise ValueError(f"Projected field length mismatch: {path}")
    for key in ("point_ids", "mask_ids", "semantic"):
        if not np.issubdtype(frame[key].dtype, np.integer):
            raise ValueError(f"Non-integer {key}: {path}")
    if count and (frame["point_ids"].min() < 0 or frame["point_ids"].max() >= n_points):
        raise ValueError(f"Out-of-range original map point ID: {path}")
    if np.unique(frame["point_ids"]).size != count:
        raise ValueError(f"Duplicate original map point ID in one frame: {path}")
    if np.any(frame["mask_ids"] < 0) or np.any(frame["semantic"] < 0):
        raise ValueError(f"Negative projected label: {path}")
    confidence = frame["confidence"]
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError(f"Invalid projected confidence: {path}")
    if frame["interior"].dtype != np.bool_:
        raise ValueError(f"Interior mask is not boolean: {path}")
    frame["frame_id"] = frame_id
    return frame


def object_inventory(xyz, instances, semantic, confidence):
    """Inventory exactly the written point ownership, including mixed categories."""
    objects = []
    for instance_id in np.unique(instances):
        if instance_id <= 0:
            continue
        point_ids = np.flatnonzero(instances == instance_id)
        labels, counts = np.unique(semantic[point_ids], return_counts=True)
        majority = int(counts.argmax())
        points = xyz[point_ids]
        objects.append({"instance_id": int(instance_id),
                        "semantic_id": int(labels[majority]),
                        "point_count": int(len(point_ids)),
                        "semantic_histogram": {str(int(k)): int(v) for k, v in zip(labels, counts)},
                        "majority_semantic_fraction": float(counts[majority] / len(point_ids)),
                        "mean_semantic_confidence": float(confidence[point_ids].mean()),
                        "center": points.mean(axis=0).tolist(),
                        "min": points.min(axis=0).tolist(),
                        "max": points.max(axis=0).tolist()})
    if sum(row["point_count"] for row in objects) != int(np.count_nonzero(instances)):
        raise RuntimeError("Object inventory does not match final instance ownership")
    return objects


def run_scene(job, environment, stage="consensus_instances"):
    global STAGE
    STAGE = stage
    from pose_pipeline.sam3_multiview import fuse_instances

    key = job["key"]
    output = ROOT / STAGE / key
    output.mkdir(parents=True, exist_ok=False)
    with (output / "events.jsonl").open("x") as log:
        try:
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            cache = Path(job["cache_root"])
            baseline = Path(job["baseline_root"])
            cache_result = json.loads((cache / "result.json").read_text())
            previous = json.loads((baseline / "result.json").read_text())
            selected = [int(value) for value in job["selected_frame_ids"]]
            if cache_result["status"] != "completed" or previous["status"] != "completed":
                raise ValueError("Both cache and baseline must be completed runs")
            if any(r["selected_frame_ids"] != selected or r["processed_frames"] != len(selected)
                   for r in (cache_result, previous)):
                raise ValueError("Cache/baseline selected frames are not identical and complete")
            frame_files, frame_records = frame_inventory(cache, selected)
            source_files = [Path(__file__).resolve(), *sorted((ROOT / "code/src/pose_pipeline").rglob("*.py"))]
            if not (ROOT / "code/src/pose_pipeline/sam3_multiview.py").is_file():
                raise FileNotFoundError("New multiview fusion source is missing")
            input_files = [ROOT / "inputs/JOBS.json", ROOT / "ENV_REUSE.json",
                           Path(job["target"]), Path(job["manifest"]), Path(job["trajectory"]),
                           Path(environment["original_spec_path"]), cache / "result.json",
                           cache / "frames.jsonl", baseline / "result.json",
                           baseline / "map_labels.npz", baseline / "classes.json", *frame_files]
            if (ROOT / "PLAN.json").is_file():
                input_files.append(ROOT / "PLAN.json")
            if stage == "consensus_matched":
                input_files.append(ROOT / "PLAN_MATCHED.json")
            input_hashes = {str(path): sha256_file(path) for path in input_files}
            source_hashes = {str(path): sha256_file(path) for path in source_files}
            if input_hashes[environment["original_spec_path"]] != environment["original_spec_file_sha256"]:
                raise ValueError("Warm reuse environment spec differs from the recorded existing spec")
            with np.load(job["target"], allow_pickle=False) as target:
                xyz = target["xyz"].copy()
            if xyz.shape != (int(job["expected_points"]), 3) or not np.isfinite(xyz).all():
                raise ValueError("Target geometry is not the expected finite XYZ map")
            xyz_hash = array_sha256(xyz)
            if any(r["geometry_xyz_sha256"] != xyz_hash for r in (cache_result, previous, job)):
                raise ValueError("Target point order/precision hash differs from cache/baseline")
            n_points = len(xyz)
            with np.load(baseline / "map_labels.npz", allow_pickle=False) as data:
                labels = {name: data[name].copy() for name in data.files}
            if any(labels[name].shape != (n_points,) for name in ("semantic", "confidence", "instance")):
                raise ValueError("Baseline label length differs from target map")
            baseline_array_hashes = {name: array_sha256(value) for name, value in labels.items()}
            for name in ("semantic", "confidence"):
                if baseline_array_hashes[name] != job[f"baseline_{name}_sha256"]:
                    raise ValueError(f"Frozen baseline {name} changed since local preparation")
            emit(log, "inputs_verified", key=key, selected_frames=len(selected), map_points=n_points,
                 geometry_xyz_sha256=xyz_hash, cached_model_inference=True, new_model_inference=False)
            frames = []
            for ordinal, (frame_id, path) in enumerate(zip(selected, frame_files)):
                frames.append(load_frame(path, frame_id, n_points))
                if (ordinal + 1) % 30 == 0 or ordinal == len(selected) - 1:
                    emit(log, "cache_loaded", key=key, loaded_frames=ordinal + 1,
                         selected_frames=len(selected))
            fusion_wall, fusion_cpu = time.perf_counter(), time.process_time()
            config = ({"min_mask_points": 30, "min_output_points": 50, "min_point_views": 1,
                       "min_group_frames": 2, "object_score_mode": "max_point"}
                      if stage == "consensus_matched" else None)
            instances, audit = fuse_instances(n_points, frames, labels["semantic"].copy(), config=config)
            fusion_wall = time.perf_counter() - fusion_wall
            fusion_cpu = time.process_time() - fusion_cpu
            if not isinstance(instances, np.ndarray) or instances.shape != (n_points,) or instances.dtype != np.int32:
                raise TypeError("fuse_instances must return an int32 ndarray of shape [n_points]")
            if np.any(instances < 0) or np.any((instances > 0) & (labels["semantic"] == 0)):
                raise ValueError("Invalid instance ID or instance assigned to unknown semantic point")
            if not isinstance(audit, dict):
                raise TypeError("fuse_instances audit must be a dictionary")
            json.dumps(audit, default=json_default)
            original_instances = labels["instance"]
            labels["instance"] = instances
            if any(array_sha256(labels[name]) != digest for name, digest in baseline_array_hashes.items()
                   if name != "instance"):
                raise RuntimeError("Fusion changed a frozen non-instance field")
            objects = object_inventory(xyz, instances, labels["semantic"], labels["confidence"])
            np.savez_compressed(output / "map_labels.npz", **labels)
            with np.load(output / "map_labels.npz", allow_pickle=False) as saved:
                if saved.files != list(labels) or any(not np.array_equal(saved[name], value, equal_nan=True)
                                                     for name, value in labels.items()):
                    raise RuntimeError("Written map labels did not round-trip exactly")
            write_json(output / "objects.json", objects)
            (output / "classes.json").write_bytes((baseline / "classes.json").read_bytes())
            write_json(output / "scene_graph.json", {
                "node_file": "objects.json", "relations": [], "new_relation_prediction_executed": False,
                "reason": "This experiment changes instance ownership only; old relation endpoints are not remapped or claimed valid."})
            write_json(output / "fusion_audit.json", audit)
            verify_hashes(input_hashes)
            verify_hashes(source_hashes)
            if frame_inventory(cache, selected)[0] != frame_files:
                raise RuntimeError("Frame inventory changed during run")
            observed = np.zeros(n_points, dtype=bool)
            for frame in frames:
                observed[frame["point_ids"]] = True
            result = {
                "status": "completed", "key": key, "variant": STAGE,
                "parent_run": str(baseline), "cached_frames_run": str(cache),
                "map_points": n_points, "geometry_xyz_sha256": xyz_hash,
                "geometry_modified": False, "pose_feedback": False,
                "selected_frame_ids": selected, "selected_frames": len(selected),
                "processed_frames": len(frames), "complete_selected_frames": True,
                "total_raw_frames": previous.get("total_raw_frames"),
                "available_raw_frames": previous.get("available_raw_frames"),
                "available_poses": previous.get("available_poses"),
                "complete_full_sequence": False,
                "scope": job.get("scope", "All frozen selected cached frames; no new raw RGB-D pass"),
                "full_raw_frame_count": job.get("full_raw_frame_count", previous.get("total_raw_frames")),
                "semantic_coverage": float(np.mean(labels["semantic"] > 0)),
                "instance_coverage": float(np.mean(instances > 0)),
                "observed_map_coverage": float(observed.mean()), "object_count": len(objects),
                "baseline_instance_coverage": float(np.mean(original_instances > 0)),
                "newly_assigned_instance_points": int(np.sum((instances > 0) & (original_instances == 0))),
                "removed_instance_points": int(np.sum((instances == 0) & (original_instances > 0))),
                "instance_numeric_ids_comparable_to_baseline": False,
                "semantic_unchanged": True, "confidence_unchanged": True,
                "non_instance_fields_preserved": [name for name in labels if name != "instance"],
                "baseline_array_sha256": baseline_array_hashes,
                "output_array_sha256": {name: array_sha256(value) for name, value in labels.items()},
                "input_sha256": input_hashes, "input_hashes_verified_after_run": True,
                "source_sha256": source_hashes[str(Path(__file__).resolve())],
                "source_files_sha256": source_hashes,
                "environment": {"python": sys.executable, "python_version": platform.python_version(),
                                "numpy": np.__version__, "scipy": scipy.__version__,
                                "platform": platform.platform(), "reuse_receipt": str(ROOT / "ENV_REUSE.json")},
                "computation": "new CPU multiview instance fusion from cached per-frame SAM3 projections",
                "model_inference_reused": True, "sam3_inference_executed": False,
                "sga_inference_executed": False, "sga_output_consumed": False,
                "cached_sgf_sga_prior_used": not key.startswith("3rscan/"),
                "cached_prior_scope": "Four full-input scenes reuse family_v3 SGF subtype anchors; 3RScan has no SGF prior. No new prior inference.",
                "cached_model_provenance_receipt": str(cache / "result.json"),
                "cached_baseline_provenance_receipt": str(baseline / "result.json"),
                "checkpoint_loaded_this_run": False, "raw_rgbd_reprocessed": False,
                "ground_truth_consumed": False, "gt_consumed": False, "quality_accepted": False,
                "new_relation_prediction_executed": False,
                "fusion_wall_seconds": fusion_wall, "fusion_cpu_seconds": fusion_cpu,
                "seconds": time.perf_counter() - started_wall,
                "total_cpu_seconds": time.process_time() - started_cpu,
                "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "time_definition": "seconds is this scene's CPU cache loading, new fusion, export and verification; excludes historical model inference and initial module import",
                "fusion_audit_file": "fusion_audit.json", "frame_provenance_records": len(frame_records),
            }
            write_json(output / "result.json", result)
            emit(log, "completed", key=key, objects=len(objects), semantic_coverage=result["semantic_coverage"],
                 instance_coverage=result["instance_coverage"], fusion_wall_seconds=fusion_wall,
                 seconds=result["seconds"], output=str(output))
            return result
        except Exception:
            failure = {"status": "failed", "key": key, "variant": STAGE,
                       "error": traceback.format_exc(), "quality_accepted": False}
            write_json(output / "failure.json", failure)
            emit(log, "failed", **failure)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", help="Exact dataset/scene key; default runs all five recorded scenes")
    parser.add_argument("--stage", choices=("consensus_instances", "consensus_matched"), default="consensus_instances")
    args = parser.parse_args()
    jobs = json.loads((ROOT / "inputs/JOBS.json").read_text())
    if args.key:
        jobs = [job for job in jobs if job["key"] == args.key]
        if not jobs:
            parser.error(f"Unknown key: {args.key}")
    environment = json.loads((ROOT / "ENV_REUSE.json").read_text())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_{os.getpid()}"
    failed = 0
    with (ROOT / f"RUNS_{run_id}.jsonl").open("x") as log:
        for job in jobs:
            emit(log, "scene_started", key=job["key"])
            try:
                result = run_scene(job, environment, args.stage)
                emit(log, "scene_completed", key=job["key"], seconds=result["seconds"])
            except Exception:
                failed += 1
                emit(log, "scene_failed", key=job["key"], error=traceback.format_exc())
        emit(log, "batch_finished", requested=len(jobs), completed=len(jobs) - failed, failed=failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
