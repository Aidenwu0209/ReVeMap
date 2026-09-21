"""A cancelled CLI must stop its isolated stage and retain failure evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from pose_pipeline.contracts import FrameRecord, SequenceManifest, write_manifest


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group runner")
def test_sigterm_stops_stage_and_records_failure(tmp_path):
    color, depth = tmp_path / "color.jpg", tmp_path / "depth.png"
    color.write_bytes(b"fixture")
    depth.write_bytes(b"fixture")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, SequenceManifest(
        "scannet", "fixture", tmp_path, 1000,
        (FrameRecord(7, 1234, color, depth, (500, 500, 10, 10)),), "test",
    ))
    stage_pid_path = tmp_path / "stage.pid"
    script = r'''
import os, sys
from pathlib import Path
from pose_pipeline import rgbd_mapping as m
base = Path(sys.argv[1])
original = m._run_stage
def sleeping_stage(command, log, timeout_s, env):
    code = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    return original([sys.executable, '-c', code, str(base/'stage.pid')], log, timeout_s, env)
m._run_stage = sleeping_stage
m.run_rgbd_mapping(manifest_path=base/'manifest.json', output_dir=base/'out',
    provider_root=base, gpu_python=Path(sys.executable), cpu_python=Path(sys.executable))
'''
    source = Path(__file__).resolve().parents[1] / "src"
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(source)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stage_pid = None
    try:
        deadline = time.monotonic() + 10
        while not stage_pid_path.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert stage_pid_path.exists(), child.communicate(timeout=5)
        stage_pid = int(stage_pid_path.read_text())
        child.send_signal(signal.SIGTERM)
        child.communicate(timeout=10)
        assert child.returncode != 0
        with pytest.raises(ProcessLookupError):
            os.kill(stage_pid, 0)
        status = json.loads((tmp_path / "out/run_status.json").read_text())
        assert status["status"] == "failed" and status["current_stage"] == "dense"
        assert "Signal" in status["error"]
        assert (tmp_path / "out/logs/dense.log").exists()
        assert not (tmp_path / "out/mapping_result.json").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
        if stage_pid is not None:
            try:
                os.killpg(stage_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
