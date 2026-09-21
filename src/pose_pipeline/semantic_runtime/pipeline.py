"""Run fresh RGB-D mapping with serial or bounded stage-level parallelism."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time

from .common import SOURCE_ROOT, event, model_spec, read, sha, validate_runtime, write


class Processes:
    """Own only children launched by this run, including their worker descendants."""
    def __init__(self, output, timeout):
        self.output, self.timeout, self.jobs = Path(output), timeout, []

    def launch(self, name, command):
        env = dict(os.environ, PYTHONPATH=str(SOURCE_ROOT), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        event(self.output, "process_start", stage=name)
        with (self.output / (name + ".log")).open("x") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        job = (name, child, time.monotonic())
        self.jobs.append(job)
        return job

    def check(self, job):
        name, child, start = job
        code = child.poll()
        if code is not None and code != 0:
            raise RuntimeError(f"{name} failed ({code}); see {self.output / (name + '.log')}")
        if code is None and time.monotonic() - start > self.timeout:
            raise TimeoutError(f"{name} exceeded {self.timeout}s")
        return code

    def wait(self, job):
        while self.check(job) is None:
            # A sibling failure must stop the run even when waiting on this job.
            for sibling in self.jobs:
                self.check(sibling)
            time.sleep(.05)
        event(self.output, "process_complete", stage=job[0])

    def close(self):
        for _, child, _ in self.jobs:
            # The leader may already have failed while a GPU descendant remains.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        end = time.monotonic() + 5
        for _, child, _ in self.jobs:
            try:
                child.wait(timeout=max(.01, end-time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for _, child, _ in self.jobs:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def run(args):
    from ..contracts import load_manifest
    config = validate_runtime(read(args.runtime), args.vlm)
    manifest = load_manifest(args.manifest)
    if args.stride < 1 or not manifest.frames:
        raise ValueError("nonempty manifest and positive stride required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Resolve all runtime paths before workers switch interpreters; no machine
    # paths or credentials are hardcoded in the implementation.
    runtime = output / "runtime.json"
    write(runtime, config)
    paths = [args.manifest] + [p for f in manifest.frames for p in (f.color_path, f.depth_path)]
    inputs = {str(Path(p).resolve()): sha(p) for p in paths}
    write(output / "INPUTS.json", inputs)
    write(output / "CONFIG.json", {"schedule": args.schedule, "vlm": args.vlm, "stride": args.stride,
                                  "raw_frames": len(manifest.frames), "profile": "sam3_stride5_stage_v1"})
    processes = Processes(output, config.get("stage_timeout", 7200))
    def command(stage, python, target, *extra):
        return [python, "-u", "-m", "pose_pipeline.semantic_runtime.worker", stage,
                "--runtime", str(runtime), "--output", str(target), *map(str, extra)]
    start = time.monotonic()
    try:
        event(output, "run_start", schedule=args.schedule)
        mapping = processes.launch("mapping", command("mapping", config["cpu_python"], output / "mapping",
                                   "--manifest", args.manifest.resolve()))
        if args.schedule == "serial":
            processes.wait(mapping)
        else:
            # run_rgbd_mapping marks refill complete only after the final GPU
            # process exits. SAM3 may now overlap the remaining CPU TSDF fusion.
            while True:
                processes.check(mapping)
                status = output / "mapping/run_status.json"
                if status.exists() and any(row["stage"] == "refill" and row.get("returncode") == 0
                                           for row in read(status).get("stages", [])):
                    break
                if mapping[1].poll() is not None:
                    raise RuntimeError("mapping ended without a completed GPU refill stage")
                time.sleep(.05)
        sam3 = processes.launch("sam3", command("sam3", config["sam3_python"], output / "semantic",
                                 "--manifest", args.manifest.resolve(), "--stride", args.stride))
        processes.wait(mapping)
        processes.wait(sam3)
        # SAM3 process exit releases weights and allocator state before any VLM.
        naming = None
        if args.vlm != "none":
            python = config.get("models", {}).get(args.vlm, {}).get("python", config["vlm_python"])
            naming = processes.launch("vlm", command("naming", python, output / "semantic/vlm",
                                      "--tasks", output / "semantic/CROP_TASKS.json", "--model", args.vlm))
        if args.schedule == "serial" and naming:
            processes.wait(naming)
        fusion = processes.launch("fusion", command("fusion", config["cpu_python"], output))
        processes.wait(fusion)
        if naming:
            processes.wait(naming)
        backfill = processes.launch("backfill", command("backfill", config["cpu_python"], output))
        processes.wait(backfill)
        end = time.monotonic()
        for path, digest in inputs.items():
            if sha(path) != digest:
                raise RuntimeError("raw input changed during execution")
        summary = {"status": "completed", "schedule": args.schedule, "vlm": args.vlm,
                   "raw_frames": len(manifest.frames), "seconds": end-start,
                   "raw_fps": len(manifest.frames)/(end-start), "stride": args.stride,
                   "scope": "fresh raw RGB-D map + SAM3 masks/fusion + optional VLM naming metadata",
                   "includes_offline_P2_enhancement": False, "VLM_changes_semantic_id": False,
                   "timing": "all worker startup/load/inference/fusion/export; excludes pre/post input hashing",
                   "GT_used": False, "local_models_resident_together": False,
                   "map": str(output / "fused/export/map_labeled.ply"),
                   "names": str(output / "fused/instance_names.json")}
        write(output / "COMPLETE.json", summary)
        return summary
    except BaseException as error:
        write(output / "FAILURE.json", {"status": "failed", "error_type": type(error).__name__,
                                        "seconds": time.monotonic()-start})
        raise
    finally:
        processes.close()
