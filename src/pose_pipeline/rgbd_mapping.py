"""Run a complete raw RGB-D sequence through tracking, refinement and fusion.

Each stage has its own process and log so CUDA state is released before the
next scene. Failed stages retain their output and never produce a completed
mapping receipt. No precomputed trajectory or dataset ground truth is needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .contracts import load_manifest, load_trajectory, sha256_file


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _run_stage(command: list[str], log: Path, timeout_s: float, env: dict) -> dict:
    started = time.monotonic()
    with log.open("xb") as stream:
        child = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT,
            env=env, start_new_session=True,
        )
        try:
            code = child.wait(timeout=timeout_s)
        except BaseException:
            # Only this stage's process group belongs to this runner.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    return {
        "command": command, "returncode": code,
        "seconds": time.monotonic() - started, "log": str(log),
    }


def _completed_result(manifest_path: Path, output_dir: Path) -> dict:
    manifest = load_manifest(manifest_path)
    trajectory = output_dir / "refill" / "trajectory.json"
    poses, _ = load_trajectory(trajectory)
    expected = [(f.frame_id, f.timestamp_us) for f in manifest.frames]
    actual = [(p.frame_id, p.timestamp_us) for p in poses]
    if actual != expected or not all(p.valid for p in poses):
        raise RuntimeError("Final trajectory does not cover every raw input frame")
    receipt = json.loads((output_dir / "fusion" / "refusion_result.json").read_text())
    cloud = Path(receipt["cloud"])
    if not (
        receipt["status"] == "completed"
        and receipt["integrated_frame_count"] == len(expected)
        and receipt["trajectory_pose_count"] == len(expected)
        and receipt["requested_frame_count"] == len(expected)
        and receipt["manifest_sha256"] == sha256_file(manifest_path)
        and receipt["point_count"] > 0
        and receipt["trajectory_sha256"] == sha256_file(trajectory)
        and receipt["identity_fallback_used"] is False
        and receipt["gt_consumed"] is False
        and cloud.is_file()
        and receipt["cloud_sha256"] == sha256_file(cloud)
    ):
        raise RuntimeError("Full-frame fusion receipt does not match its outputs")
    return {
        "schema": "raw_rgbd_mapping_result.v1", "status": "completed",
        "dataset": manifest.dataset, "sequence_id": manifest.sequence_id,
        "raw_frame_count": len(expected), "final_pose_count": len(poses),
        "integrated_frame_count": receipt["integrated_frame_count"],
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "trajectory": str(trajectory), "trajectory_sha256": sha256_file(trajectory),
        "final_cloud": str(cloud), "final_cloud_sha256": receipt["cloud_sha256"],
        "identity_fallback_used": False, "gt_consumed": False,
        "quality_acceptance": "reported_by_separate_cross_dataset_evaluation",
    }


def run_rgbd_mapping(
    *, manifest_path: Path, output_dir: Path, provider_root: Path,
    gpu_python: Path, cpu_python: Path,
    stage_timeout_s: float = 7200, device: str = "0", threads: int = 2,
) -> dict:
    """Run one sequence, preserving the failed stage if any operation fails."""
    if stage_timeout_s <= 0 or threads < 1:
        raise ValueError("Stage timeout and thread count must be positive")
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = load_manifest(manifest_path)
    output_dir = Path(output_dir).resolve()
    provider_root = Path(provider_root).resolve(strict=True)
    gpu_python = Path(gpu_python).absolute()
    cpu_python = Path(cpu_python).absolute()
    if not all(p.is_file() for p in (gpu_python, cpu_python)):
        raise FileNotFoundError("Both GPU and CPU Python executables are required")
    output_dir.mkdir(parents=True, exist_ok=False)
    log_dir = output_dir / "logs"
    log_dir.mkdir()
    source_root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ, "PYTHONPATH": str(source_root),
        "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads), "NUMEXPR_NUM_THREADS": str(threads),
    }
    plan = {
        "schema": "raw_rgbd_mapping_run.v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(manifest_path),
        "raw_frame_count": len(manifest.frames), "stages": [],
        "provider_root": str(provider_root), "device": device,
        "stage_timeout_s": stage_timeout_s,
        "source_sha256": {
            str(p.relative_to(source_root)): sha256_file(p)
            for package in ("pose_pipeline", "reconstruction")
            for p in sorted((source_root / package).glob("*.py"))
        },
    }
    _write(output_dir / "run_status.json", plan)
    previous_sigterm = None
    if (threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL):
        # A standalone CLI has no matrix runner's signal handler. Raise through
        # _run_stage so its isolated child group is stopped and failure is saved.
        def terminated(signum, _frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        previous_sigterm = signal.signal(signal.SIGTERM, terminated)
    try:
        for stage in ("dense", "graph", "refill", "fusion"):
            gpu = stage in {"dense", "refill"}
            command = [
                str(gpu_python if gpu else cpu_python), "-u", "-m",
                "pose_pipeline.rgbd_mapping", "--stage", stage,
                "--manifest", str(manifest_path), "--output", str(output_dir),
                "--provider-root", str(provider_root),
            ]
            plan["current_stage"] = stage
            _write(output_dir / "run_status.json", plan)
            result = _run_stage(
                command, log_dir / (stage + ".log"), stage_timeout_s,
                {**env, "CUDA_VISIBLE_DEVICES": device if gpu else ""},
            )
            plan["stages"].append({"stage": stage, **result})
            _write(output_dir / "run_status.json", plan)
            if result["returncode"] != 0:
                raise RuntimeError(f"{stage} failed; details retained in {result['log']}")
        if sha256_file(manifest_path) != plan["manifest_sha256"]:
            raise RuntimeError("Input manifest changed during mapping")
        if any(sha256_file(source_root / name) != digest
               for name, digest in plan["source_sha256"].items()):
            raise RuntimeError("Mapping source changed during execution")
        result = _completed_result(manifest_path, output_dir)
        result["stages"] = plan["stages"]
        result["source_sha256"] = plan["source_sha256"]
        _write(output_dir / "mapping_result.json", result)
        plan.update(status="completed", current_stage=None)
        return result
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        plan["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write(output_dir / "run_status.json", plan)


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Isolated RGB-D mapping stage")
    parser.add_argument("--stage", choices=("dense", "graph", "refill", "fusion"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage == "dense":
        from .rgbd_droid import run_dense
        run_dense(args.manifest.resolve(), root / "dense", args.provider_root)
    elif args.stage == "graph":
        from .rgbd_measured import run_measured_graph
        run_measured_graph(root / "dense", root / "graph")
    elif args.stage == "refill":
        from .rgbd_refill import run_visual_refill
        run_visual_refill(root / "dense", root / "graph", root / "refill", args.provider_root)
    else:
        from reconstruction.rgbd_refusion import FullRefusionRequest, run_full_rgbd_refusion
        manifest = load_manifest(args.manifest)
        run_full_rgbd_refusion(FullRefusionRequest(
            manifest=args.manifest.resolve(), trajectory=root / "refill" / "trajectory.json",
            output_dir=root / "fusion", fused_frame_ids=tuple(f.frame_id for f in manifest.frames),
            voxel_length_m=0.02, sdf_trunc_m=0.08, depth_trunc_m=4.5,
        ))


if __name__ == "__main__":
    _main()
