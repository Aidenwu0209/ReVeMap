"""Completion must describe the actual raw sequence and its generated cloud."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pose_pipeline.contracts import (
    FrameRecord, PoseRecord, SequenceManifest, sha256_file,
    write_manifest, write_trajectory,
)
from pose_pipeline.rgbd_mapping import _completed_result, run_rgbd_mapping


def inputs(root):
    frames = []
    for i in range(2):
        color, depth = root / f"{i}.jpg", root / f"{i}.png"
        color.write_bytes(b"fixture color")
        depth.write_bytes(b"fixture depth")
        frames.append(FrameRecord(i + 5, i * 1000, color, depth, (500, 500, 10, 10)))
    manifest = root / "manifest.json"
    write_manifest(manifest, SequenceManifest("scannet", "fixture", root, 1000, tuple(frames), "test"))
    return manifest, frames


def completed_files(root, manifest, frames, *, wrong_timestamp=False):
    (root / "refill").mkdir(parents=True)
    (root / "fusion").mkdir()
    poses = [PoseRecord(f.frame_id, f.timestamp_us, np.eye(4)) for f in frames]
    if wrong_timestamp:
        poses[-1] = replace(poses[-1], timestamp_us=2000)
    trajectory = root / "refill" / "trajectory.json"
    write_trajectory(trajectory, poses, sequence_id="fixture", arm="candidate")
    cloud = root / "fusion" / "refused.ply"
    cloud.write_bytes(b"fixture cloud")
    receipt = dict(
        status="completed", cloud=str(cloud), cloud_sha256=sha256_file(cloud),
        point_count=1, integrated_frame_count=2, trajectory_pose_count=2,
        requested_frame_count=2, manifest_sha256=sha256_file(manifest),
        trajectory_sha256=sha256_file(trajectory), identity_fallback_used=False,
        gt_consumed=False,
    )
    (root / "fusion" / "refusion_result.json").write_text(json.dumps(receipt))
    return cloud


def test_completion_binds_raw_timestamps_and_cloud_bytes(tmp_path):
    manifest, frames = inputs(tmp_path)
    root = tmp_path / "out"
    cloud = completed_files(root, manifest, frames)
    assert _completed_result(manifest, root)["final_pose_count"] == 2
    cloud.write_bytes(b"changed after fusion")
    with pytest.raises(RuntimeError, match="fusion receipt"):
        _completed_result(manifest, root)


def test_matching_frame_count_cannot_hide_wrong_timestamps(tmp_path):
    manifest, frames = inputs(tmp_path)
    root = tmp_path / "out"
    completed_files(root, manifest, frames, wrong_timestamp=True)
    with pytest.raises(RuntimeError, match="every raw input frame"):
        _completed_result(manifest, root)


def test_failed_stage_keeps_log_and_never_starts_later_stages(tmp_path, monkeypatch):
    manifest, _ = inputs(tmp_path)
    calls = []

    def failed_stage(command, log, timeout_s, env):
        calls.append(command)
        log.write_text("CUDA initialization failed\n")
        return {"returncode": 1, "log": str(log), "seconds": 0.1}

    monkeypatch.setattr("pose_pipeline.rgbd_mapping._run_stage", failed_stage)
    root = tmp_path / "out"
    with pytest.raises(RuntimeError, match="dense failed"):
        run_rgbd_mapping(
            manifest_path=manifest, output_dir=root, provider_root=tmp_path,
            gpu_python=Path(sys.executable), cpu_python=Path(sys.executable),
        )
    assert len(calls) == 1
    assert not (root / "mapping_result.json").exists()
    status = json.loads((root / "run_status.json").read_text())
    assert status["status"] == "failed" and status["current_stage"] == "dense"
    assert (root / "logs" / "dense.log").read_text() == "CUDA initialization failed\n"
