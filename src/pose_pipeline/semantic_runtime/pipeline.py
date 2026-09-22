"""Run fresh RGB-D mapping with serial or bounded stage-level parallelism."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .common import SOURCE_ROOT, event, model_spec, read, sha, validate_runtime, write


class Processes:
    """Own only children launched by this run, including their worker descendants."""
    def __init__(self, output, timeout):
        self.output, self.timeout, self.jobs = Path(output), timeout, []

    def launch(self, name, command):
        env = dict(os.environ, PYTHONPATH=str(SOURCE_ROOT), REVEMAP_SUPERVISOR_PID=str(os.getpid()), OMP_NUM_THREADS="2",
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
    from .view_selection import selection_budget
    config = validate_runtime(read(args.runtime), args.vlm)
    manifest = load_manifest(args.manifest)
    if args.stride < 1 or not manifest.frames:
        raise ValueError("nonempty manifest and positive stride required")
    view_policy = getattr(args, 'view_policy', 'stride')
    view_budget = getattr(args, 'view_budget', None)
    effective_budget = selection_budget(manifest.frames, args.stride, view_budget, view_policy)
    confidence_policy = getattr(args, 'semantic_confidence_policy', 'legacy')
    conflict_policy = getattr(args, 'semantic_conflict_policy', 'consensus')
    if confidence_policy not in ('legacy', 'track') or conflict_policy not in ('consensus', 'abstain'):
        raise ValueError('invalid semantic evidence policy')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Resolve all runtime paths before workers switch interpreters; no machine
    # paths or credentials are hardcoded in the implementation.
    runtime = output / "runtime.json"
    write(runtime, config)
    paths = [args.manifest] + [p for f in manifest.frames for p in (f.color_path, f.depth_path)]
    inputs = {str(Path(p).resolve()): sha(p) for p in paths}
    write(output / "INPUTS.json", inputs)
    options = {"schedule": args.schedule, "vlm": args.vlm, "stride": args.stride,
                                  "raw_frames": len(manifest.frames), "profile": "sam3_stage_v2",
                                  "view_policy": view_policy, "view_budget": view_budget,
                                  "effective_view_budget": effective_budget,
                                  "semantic_confidence_policy": confidence_policy,
                                  "semantic_conflict_policy": conflict_policy}
    write(output / "CONFIG.json", options)
    processes = Processes(output, config.get("stage_timeout", 7200))
    def command(stage, python, target, *extra):
        return [python, "-u", "-m", "pose_pipeline.semantic_runtime.worker", stage,
                "--runtime", str(runtime), "--output", str(target), *map(str, extra)]
    start = time.monotonic()
    previous_sigterm = None
    if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
        def terminated(signum, _frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        previous_sigterm = signal.signal(signal.SIGTERM, terminated)
    from ..live_io import guard_parent_process
    guard_parent_process()
    try:
        event(output, "run_start", schedule=args.schedule)
        checkpoints = None
        if getattr(args, 'checkpoint_stages', False) or getattr(args, 'resume_from', None):
            from .checkpoints import CheckpointStore, context
            checkpoints = CheckpointStore(output, context(inputs, config, options),
                                          getattr(args, 'resume_from', None))
        mapping_reused = bool(checkpoints and checkpoints.restore('mapping'))
        sam3_reused = False
        mapping = None if mapping_reused else processes.launch("mapping", command("mapping", config["cpu_python"], output / "mapping",
                                   "--manifest", args.manifest.resolve()))
        if mapping and args.schedule == "serial":
            processes.wait(mapping)
            if checkpoints:
                checkpoints.seal('mapping')
        elif mapping:
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
        # Selection sees finalized refill poses, never dense or graph estimates.
        # It may overlap CPU TSDF fusion, and does not subset the mapping input.
        selection_command = command('select', config['cpu_python'], output / 'view_selection',
            '--manifest', args.manifest.resolve(), '--trajectory', output / 'mapping/refill/trajectory.json',
            '--stride', args.stride, '--view-policy', view_policy)
        if view_budget is not None:
            selection_command.extend(['--view-budget', str(view_budget)])
        selection = processes.launch('view_selection', selection_command)
        processes.wait(selection)
        view_plan = output / 'view_selection/VIEW_PLAN.json'
        selection_audit = read(view_plan)
        if checkpoints and mapping_reused:
            sam3_reused = checkpoints.restore('sam3', view_plan=view_plan)
            if sam3_reused and [row['frame_id'] for row in read(output / 'semantic/FRAMES.json')] != selection_audit['selected_frame_ids']:
                raise ValueError('restored SAM3 frames do not match the current selection')
        sam3 = None if sam3_reused else processes.launch("sam3", command("sam3", config["sam3_python"], output / "semantic",
                                 "--manifest", args.manifest.resolve(), "--stride", args.stride,
                                 '--view-plan', view_plan, '--view-plan-sha256', sha(view_plan)))
        if mapping and args.schedule != 'serial':
            processes.wait(mapping)
            if checkpoints:
                checkpoints.seal('mapping')
        if sam3:
            processes.wait(sam3)
            if checkpoints:
                checkpoints.seal('sam3')
        # SAM3 process exit releases weights and allocator state before any VLM.
        naming = None
        if args.vlm != "none":
            python = config.get("models", {}).get(args.vlm, {}).get("python", config["vlm_python"])
            naming = processes.launch("vlm", command("naming", python, output / "semantic/vlm",
                                      "--tasks", output / "semantic/CROP_TASKS.json", "--model", args.vlm))
        if args.schedule == "serial" and naming:
            processes.wait(naming)
        fusion = processes.launch("fusion", command("fusion", config["cpu_python"], output,
            '--semantic-confidence-policy', confidence_policy, '--semantic-conflict-policy', conflict_policy))
        processes.wait(fusion)
        if naming:
            processes.wait(naming)
        backfill = processes.launch("backfill", command("backfill", config["cpu_python"], output))
        processes.wait(backfill)
        end = time.monotonic()
        # Final inventory/COMPLETE are success seals, so input verification must
        # finish first. A failed run must not leave a newly sealed inventory.
        for path, digest in inputs.items():
            if sha(path) != digest:
                raise RuntimeError("raw input changed during execution")
        if read(runtime) != config or read(output / 'CONFIG.json') != options or read(output / 'INPUTS.json') != inputs:
            raise RuntimeError('execution records changed during execution')
        if checkpoints:
            checkpoints.verify_context()
        from ..artifacts import write_artifact_manifest
        fused = output / 'fused'
        write_artifact_manifest(fused, map_path=fused / 'export/map_labeled.ply',
            classes_path=fused / 'classes.json', result_path=fused / 'result.json',
            manifest_path=args.manifest, trajectory_path=output / 'mapping/refill/trajectory.json',
            extra_files={'names': fused / 'instance_names.json'})
        summary = {"status": "completed", "schedule": args.schedule, "vlm": args.vlm,
                   "raw_frames": len(manifest.frames), "seconds": end-start,
                   "raw_fps": None if mapping_reused or sam3_reused else len(manifest.frames)/(end-start),
                   "effective_replay_frames_per_second": len(manifest.frames)/(end-start) if mapping_reused or sam3_reused else None,
                   "stride": args.stride,
                   "view_policy": view_policy, "view_budget": effective_budget,
                   "selected_frames": len(selection_audit['selected_frame_ids']),
                   "view_plan": str(view_plan), "view_plan_sha256": sha(view_plan),
                   "diversity_fallback_frames": len(selection_audit.get('diversity_fallback_frame_ids', [])),
                   "scope": ("verified completed-stage reuse + fresh fusion/optional naming" if mapping_reused or sam3_reused
                             else "fresh raw RGB-D map + SAM3 masks/fusion + optional VLM naming metadata"),
                   "stage_reuse": {"mapping": mapping_reused, "sam3": sam3_reused},
                   "semantic_confidence_policy": confidence_policy, "semantic_conflict_policy": conflict_policy,
                   "includes_offline_P2_enhancement": False, "VLM_changes_semantic_id": False,
                   "timing": "all worker startup/load/inference/fusion/export; excludes pre/post input hashing and final inventory sealing; includes checkpoint verification/copy when enabled",
                   "GT_used": False, "local_models_resident_together": False,
                   "map": str(output / "fused/export/map_labeled.ply"),
                   "artifacts": str(output / "fused/ARTIFACTS.json"),
                   "names": str(output / "fused/instance_names.json")}
        write(output / "COMPLETE.json", summary)
        return summary
    except BaseException as error:
        write(output / "FAILURE.json", {"status": "failed", "error_type": type(error).__name__,
                                        "seconds": time.monotonic()-start})
        raise
    finally:
        processes.close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
