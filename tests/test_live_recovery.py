"""CPU regression tests for committed capture data and recoverable GUI sessions."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pose_pipeline import live_capture
from pose_pipeline.live_gui import Controller, pipeline_progress
from pose_pipeline.live_io import FrameJournalReader, atomic_json, journal_frames, seal_capture


def wait_until(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError("real process did not reach the expected state")


def process_running(pid):
    state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


@pytest.fixture
def process_probe(tmp_path):
    """Real OS fault worker: emits stage files, never emits model predictions."""
    module = tmp_path / "recovery_process_probe.py"
    module.write_text('''import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
args,_=p.parse_known_args();args.output.mkdir(parents=True,exist_ok=True)
stage=os.environ.get('RECOVERY_TEST_STAGE','mapping')
(args.output/'events_probe.jsonl').write_text(json.dumps({'event':'process_start','stage':stage,'monotonic':time.monotonic()})+'\\n')
(args.output/'partial-stage.bin').write_bytes(b'incomplete fault-injection stage; not model output')
if os.environ.get('RECOVERY_TEST_DESCENDANT'):
 code="import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
 child=subprocess.Popen([sys.executable,'-c',code])
 (args.output/'descendant.pid').write_text(str(child.pid))
if os.environ.get('RECOVERY_TEST_EXIT_ZERO'):
 sys.exit(0)
(args.output/'ready').write_text(str(os.getpid()))
while True:time.sleep(.05)
''')
    return module


def probe_controller(tmp_path, module, *, stage="mapping"):
    class ProbeController(Controller):
        def launch(self, python, original_module, arguments, name, *, output=None):
            return super().launch(sys.executable, module.stem, arguments, name, output=output)
    controller = ProbeController(SimpleNamespace(replay=None, output=tmp_path,
        cpu_python=sys.executable, runtime=tmp_path / 'runtime.json'))
    controller.env['PYTHONPATH'] = str(module.parent) + os.pathsep + controller.env['PYTHONPATH']
    controller.env['RECOVERY_TEST_STAGE'] = stage
    return controller


@pytest.mark.parametrize("stage", ["mapping", "sam3", "vlm", "refine/ground", "export"])
def test_real_stage_process_kill_and_retry_preserves_previous_attempt(tmp_path, process_probe, stage):
    session = saved_session(tmp_path)
    controller = probe_controller(tmp_path, process_probe, stage=stage)
    options = {"vlm": "none", "schedule": "serial", "refine": False}
    with patch.object(controller, 'validate_options', return_value=options):
        controller.reprocess({'session': session.name, **options})
        first = Path(controller.state['attempt'])
        wait_until(lambda: (first / 'pipeline/ready').exists())
        os.kill(controller.children[0].pid, signal.SIGKILL)
        controller.thread.join(timeout=6)
        assert not controller.thread.is_alive()
        assert controller.state['status'] == 'failed'
        before = {str(p.relative_to(first)): p.read_bytes() for p in first.rglob('*') if p.is_file()}
        restarted = probe_controller(tmp_path, process_probe, stage=stage)
        restarted.open_session(session.name)
        assert restarted.state['status'] == 'failed' and restarted.snapshot()['can_reprocess']
        with patch.object(restarted, 'validate_options', return_value=options):
            restarted.reprocess({'session': session.name, **options})
            second = Path(restarted.state['attempt'])
            try:
                wait_until(lambda: (second / 'pipeline/ready').exists())
                assert second != first
                command = json.loads((second / 'mapping_launch.json').read_text())['command']
                assert '--checkpoint-stages' in command
                assert command[command.index('--resume-from') + 1] == str(first / 'pipeline')
            finally:
                restarted.cancel()
                restarted.thread.join(timeout=6)
        assert not restarted.thread.is_alive()
        assert restarted.state['status'] == 'cancelled'
        assert before == {str(p.relative_to(first)): p.read_bytes() for p in first.rglob('*') if p.is_file()}
        assert all(not process_running(p.pid) for p in controller.children + restarted.children)


def test_real_cancel_reaps_descendant_after_its_leader_exits(tmp_path, process_probe):
    session = saved_session(tmp_path)
    controller = probe_controller(tmp_path, process_probe)
    controller.env['RECOVERY_TEST_DESCENDANT'] = '1'
    options = {"vlm": "none", "schedule": "serial", "refine": False}
    with patch.object(controller, 'validate_options', return_value=options):
        controller.reprocess({'session': session.name, **options})
    output = controller.pipeline_path()
    try:
        wait_until(lambda: (output / 'ready').exists())
        descendant = int((output / 'descendant.pid').read_text())
        wait_until(lambda: process_running(descendant))
        controller.cancel()
        controller.thread.join(timeout=6)
        assert not controller.thread.is_alive()
        assert controller.state['status'] == 'cancelled'
        wait_until(lambda: not process_running(descendant))
        assert not process_running(controller.children[0].pid)
    finally:
        controller.cancel()
        for child in controller.children:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass


def test_real_zero_exit_without_final_seal_is_not_completed(tmp_path, process_probe):
    session = saved_session(tmp_path)
    controller = probe_controller(tmp_path, process_probe)
    controller.env['RECOVERY_TEST_EXIT_ZERO'] = '1'
    options = {"vlm": "none", "schedule": "serial", "refine": False}
    with patch.object(controller, 'validate_options', return_value=options):
        controller.reprocess({'session': session.name, **options})
    controller.thread.join(timeout=6)
    assert not controller.thread.is_alive()
    assert controller.state['status'] == 'failed'
    assert '封存' in controller.state['error']


def test_second_gui_cannot_reprocess_while_prior_worker_is_alive(tmp_path, process_probe):
    session = saved_session(tmp_path)
    first = probe_controller(tmp_path, process_probe)
    options = {"vlm": "none", "schedule": "serial", "refine": False}
    with patch.object(first, 'validate_options', return_value=options):
        first.reprocess({'session': session.name, **options})
    try:
        wait_until(lambda: (first.pipeline_path() / 'ready').exists())
        original = (session / 'session.json').read_bytes()
        second = probe_controller(tmp_path, process_probe)
        with patch.object(second, 'validate_options', return_value=options):
            with pytest.raises(ValueError, match='仍在退出'):
                second.reprocess({'session': session.name, **options})
        assert (session / 'session.json').read_bytes() == original
        assert len(list((session / 'attempts').iterdir())) == 1
    finally:
        first.cancel(); first.thread.join(timeout=6)
    assert not first.thread.is_alive()


@pytest.mark.parametrize('after_first_frame', [False, True])
def test_real_stalled_capture_has_supervisor_deadline(tmp_path, after_first_frame):
    controller = Controller(SimpleNamespace(replay=None, first_frame_timeout=.3, capture_frame_timeout=.3))
    controller.session = tmp_path
    atomic_json(tmp_path / 'capture_status.json', {'frames': int(after_first_frame), 'status': 'recording'})
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    controller.children = [child]
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            (controller.wait_capture if after_first_frame else controller.wait_first_frame)(child)
        assert time.monotonic() - start < 2
    finally:
        controller.cleanup_children()
    assert not process_running(child.pid)


def test_real_writer_killed_mid_journal_recovers_only_committed_frame(tmp_path):
    session = saved_session(tmp_path, status='recording', seal=False)
    journal = session / 'capture/frames.jsonl'
    committed = journal.read_bytes().split(b'\n')[0] + b'\n'
    journal.write_bytes(committed)
    ready = session / 'writer_ready'
    code = "from pathlib import Path;import sys,time;p=Path(sys.argv[1]);f=p.open('ab');f.write(b'{\"frame_id\":1,');f.flush();Path(sys.argv[2]).touch();time.sleep(60)"
    child = subprocess.Popen([sys.executable, '-c', code, str(journal), str(ready)], start_new_session=True)
    try:
        wait_until(ready.exists)
        child.kill(); child.wait(timeout=3)
        before = journal.read_bytes()
        controller = Controller(SimpleNamespace(replay=None, output=tmp_path))
        controller.open_session(session.name)
        assert controller.state['status'] == 'interrupted'
        assert seal_capture(session / 'capture', source='explicit_process_interrupt_test') == 1
        assert journal.read_bytes() == before
        assert len(json.loads((session / 'capture/manifest.json').read_text())['frames']) == 1
    finally:
        if child.poll() is None: child.kill(); child.wait()


def test_real_replay_capture_seals_after_supervisor_sigkill(tmp_path):
    """Actual capture module with synthetic RGB-D inputs; no SDK or models."""
    from PIL import Image
    from pose_pipeline.contracts import FrameRecord, SequenceManifest, write_manifest
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    color, depth = inputs / 'color.png', inputs / 'depth.png'
    Image.fromarray(np.zeros((6, 8, 3), np.uint8)).save(color)
    Image.fromarray(np.full((6, 8), 1000, np.uint16)).save(depth)
    manifest = inputs / 'manifest.json'
    write_manifest(manifest, SequenceManifest('orbbec', 'replay-fixture', inputs, 1000.,
        tuple(FrameRecord(i, (i+1)*1000, color, depth, (8., 8., 4., 3.)) for i in range(200)), 'synthetic process recovery fixture'))
    session = tmp_path / 'scan_parent_killed'; session.mkdir()
    atomic_json(session / 'session.json', {'status': 'recording', 'mode': 'replay'})
    code = """import os,subprocess,sys,time
from pathlib import Path
session,manifest=sys.argv[1:]
child=subprocess.Popen([sys.executable,'-m','pose_pipeline.live_capture','--session',session,'--replay',manifest,'--fps','20'],env={**os.environ,'REVEMAP_GUI_PARENT_PID':str(os.getpid())},start_new_session=True)
Path(session,'capture.pid').write_text(str(child.pid))
while True:time.sleep(.05)
"""
    env = {**os.environ, 'PYTHONPATH': str(Path(live_capture.__file__).resolve().parents[1]), 'PYTHONDONTWRITEBYTECODE': '1'}
    with (session / 'capture.log').open('w') as log:
        parent = subprocess.Popen([sys.executable, '-c', code, str(session), str(manifest)], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid = None
    try:
        wait_until(lambda: (session / 'capture.pid').exists())
        pid = int((session / 'capture.pid').read_text())
        wait_until(lambda: json.loads((session / 'capture_status.json').read_text()).get('frames', 0) >= 2
                   if (session / 'capture_status.json').exists() else False)
        parent.kill(); parent.wait(timeout=3)
        wait_until(lambda: (session / 'capture/manifest.json').exists())
        wait_until(lambda: not process_running(pid))
        status = json.loads((session / 'capture_status.json').read_text())
        assert status['status'] == 'sealed' and status['frames'] >= 2
        controller = Controller(SimpleNamespace(replay=None, output=tmp_path))
        controller.open_session(session.name)
        assert controller.state['status'] == 'interrupted'
        assert controller.snapshot()['can_reprocess']
        assert not (session / 'pipeline/GUI_RESULT.json').exists()
    finally:
        if parent.poll() is None: parent.kill(); parent.wait()
        if pid and process_running(pid):
            try: os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError: pass


def test_parent_watchdog_cascades_across_real_setsid_process_tree(tmp_path):
    """Nested supervisors each own a session, like CLI -> mapping -> GPU stage."""
    script = tmp_path / 'owned_tree.py'
    script.write_text('''import os,subprocess,sys,time
from pathlib import Path
from pose_pipeline.live_io import guard_parent_process
root=Path(sys.argv[1]);depth=int(sys.argv[2])
guard_parent_process(interval=.02,grace=.2)
if depth:
 subprocess.Popen([sys.executable,__file__,str(root),str(depth-1)],
  env={**os.environ,'REVEMAP_SUPERVISOR_PID':str(os.getpid())},start_new_session=True)
(root/(str(depth)+'.pid')).write_text(str(os.getpid()))
while True:time.sleep(.02)
''')
    env = {**os.environ, 'PYTHONPATH': str(Path(live_capture.__file__).resolve().parents[1]), 'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('REVEMAP_SUPERVISOR_PID', None); env.pop('REVEMAP_GUI_PARENT_PID', None)
    leader = subprocess.Popen([sys.executable, str(script), str(tmp_path), '2'], env=env, start_new_session=True)
    pids = []
    try:
        wait_until(lambda: all((tmp_path / f'{depth}.pid').exists() for depth in range(3)))
        pids = [int((tmp_path / f'{depth}.pid').read_text()) for depth in range(3)]
        assert len({os.getpgid(pid) for pid in pids}) == 3
        leader.kill(); leader.wait(timeout=3)
        wait_until(lambda: all(not process_running(pid) for pid in pids))
    finally:
        if leader.poll() is None: leader.kill(); leader.wait()
        for pid in pids:
            if process_running(pid):
                try: os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError: pass


def test_semantic_bridge_forwards_checkpoint_resume_options(tmp_path, monkeypatch):
    from pose_pipeline import live_semantic
    from pose_pipeline.semantic_runtime import pipeline
    received = {}
    def stop_at_pipeline(args):
        received.update(vars(args))
        raise RuntimeError('no model execution in CLI option test')
    monkeypatch.setattr(pipeline, 'run', stop_at_pipeline)
    monkeypatch.setattr(sys, 'argv', ['live_semantic', '--manifest', str(tmp_path / 'manifest.json'),
        '--runtime', str(tmp_path / 'runtime.json'), '--output', str(tmp_path / 'new'),
        '--checkpoint-stages', '--resume-from', str(tmp_path / 'previous')])
    with patch.object(live_semantic.signal, 'signal'):
        with pytest.raises(RuntimeError, match='no model execution'):
            live_semantic.main()
    assert received['checkpoint_stages'] is True
    assert received['resume_from'] == tmp_path / 'previous'


def test_incremental_journal_commits_only_complete_rows(tmp_path):
    path = tmp_path / "frames.jsonl"
    reader = FrameJournalReader(path)
    assert reader.poll() == []
    first = json.dumps({"frame_id": 0, "name": "彩色"}, ensure_ascii=False).encode() + b"\n"
    second = json.dumps({"frame_id": 1, "name": "深度"}, ensure_ascii=False).encode() + b"\n"
    path.write_bytes(first + second[:25])
    assert reader.poll() == [{"frame_id": 0, "name": "彩色"}]
    assert reader.offset == len(first)
    assert reader.poll() == []
    with path.open("ab") as stream:
        stream.write(second[25:])
    assert reader.poll() == [{"frame_id": 1, "name": "深度"}]
    assert reader.poll() == []
    assert reader.offset == path.stat().st_size
    assert len(journal_frames(path)) == 2


@pytest.mark.parametrize("change", ["truncate", "replace"])
def test_incremental_journal_rejects_rewritten_capture(tmp_path, change):
    path = tmp_path / "frames.jsonl"
    path.write_text('{"frame_id":0}\n')
    reader = FrameJournalReader(path)
    assert len(reader.poll()) == 1
    if change == "truncate":
        path.write_text("")
    else:
        replacement = tmp_path / "replacement"
        replacement.write_text('{"frame_id":1}\n')
        replacement.replace(path)
    with pytest.raises(ValueError, match="replaced or truncated"):
        reader.poll()


class Profiles(list):
    def get_count(self):
        return len(self)


def fake_camera(monkeypatch, samples):
    """Exercise the SDK iterator itself, including each rejected-pair branch."""
    clock = [0.]
    profile = Mock()
    profile.as_video_stream_profile.return_value = profile
    profile.get_format.return_value = "RGB"
    profile.get_fps.return_value = 30
    profile.get_width.return_value = 4
    profile.get_height.return_value = 3
    profile.get_intrinsic.return_value = SimpleNamespace(width=4, height=3, fx=4., fy=4., cx=2., cy=1.)
    color = Mock()
    color.get_width.return_value = 4
    color.get_height.return_value = 3
    color.get_timestamp_us.return_value = 1000
    color.get_format.return_value = "RGB"
    color.get_data.return_value = np.zeros((3, 4, 3), np.uint8)
    color.get_index.return_value = 0
    depth = Mock()
    depth.get_width.return_value = 4
    depth.get_height.return_value = 3
    depth.get_timestamp_us.return_value = 1000
    depth.get_stream_profile.return_value = profile
    depth.get_depth_scale.return_value = 1.
    depth.get_data.return_value = np.full((3, 4), 1000, np.uint16)

    def read(_timeout):
        clock[0] += 1.
        kind = next(samples)
        if kind == "no_frames":
            return None
        depth.get_timestamp_us.return_value = 30000 if kind == "sync_mismatch" else 1000
        return SimpleNamespace(get_color_frame=lambda: None if kind == "missing_color" else color,
                               get_depth_frame=lambda: None if kind == "missing_depth" else depth)

    pipe = Mock()
    pipe.get_stream_profile_list.return_value = Profiles([profile])
    pipe.get_d2c_depth_profile_list.return_value = Profiles([profile])
    pipe.wait_for_frames.side_effect = read
    sdk = SimpleNamespace(Context=SimpleNamespace(set_logger_level=lambda _: None),
                          OBLogLevel=SimpleNamespace(ERROR=0), Config=Mock,
                          OBSensorType=SimpleNamespace(COLOR_SENSOR=0),
                          OBAlignMode=SimpleNamespace(HW_MODE=0),
                          OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE=0),
                          OBFormat=SimpleNamespace(**{name: name for name in ("RGB", "BGR", "MJPG", "YUYV", "YUY2")}))
    monkeypatch.setitem(__import__("sys").modules, "pyorbbecsdk", sdk)
    monkeypatch.setattr(live_capture.time, "monotonic", lambda: clock[0])
    return pipe


@pytest.mark.parametrize("reason", ["no_frames", "missing_color", "missing_depth", "sync_mismatch"])
def test_camera_invalid_pairs_have_startup_deadline(monkeypatch, reason):
    pipe = fake_camera(monkeypatch, iter([reason] * 4))
    counts = {}
    frames = live_capture.camera_frames(pipe, startup_timeout=3, valid_timeout=2, rejected_counts=counts)
    assert next(frames) is None
    assert next(frames) is None
    with pytest.raises(RuntimeError, match="3 秒未收到有效同步"):
        next(frames)
    assert counts == {reason: 3}
    pipe.stop.assert_called_once()


def test_camera_invalid_traffic_does_not_reset_last_valid_deadline(monkeypatch):
    pipe = fake_camera(monkeypatch, iter(["valid", "sync_mismatch", "missing_depth"]))
    frames = live_capture.camera_frames(pipe, startup_timeout=20, valid_timeout=2)
    assert next(frames)[1].shape == (480, 640)
    assert next(frames) is None
    with pytest.raises(RuntimeError, match="2 秒未收到有效同步"):
        next(frames)
    pipe.stop.assert_called_once()


def test_capture_failure_seals_saved_rgbd(tmp_path):
    def source():
        yield np.zeros((3, 4, 3), np.uint8), np.full((3, 4), 1000, np.uint16), (4., 4., 2., 1.), 1000, {}
        raise RuntimeError("injected connection loss")

    args = SimpleNamespace(session=tmp_path, replay=None)
    with patch.object(live_capture, "camera_frames", return_value=source()), patch.object(live_capture.signal, "signal"):
        with pytest.raises(RuntimeError, match="injected connection loss"):
            live_capture.capture(args)
    state = json.loads((tmp_path / "capture_status.json").read_text())
    manifest = json.loads((tmp_path / "capture/manifest.json").read_text())
    assert state["status"] == "failed" and state["frames"] == 1
    assert len(manifest["frames"]) == 1
    assert Path(manifest["frames"][0]["color_path"]).is_file()
    assert Path(manifest["frames"][0]["depth_path"]).is_file()


def test_supervisor_first_frame_deadline(tmp_path):
    c = Controller(SimpleNamespace(replay=None, first_frame_timeout=.5))
    c.session = tmp_path
    camera = Mock()
    camera.poll.return_value = None
    with patch("pose_pipeline.live_gui.time.monotonic", side_effect=[0., .3, .6]), patch("pose_pipeline.live_gui.time.sleep"):
        with pytest.raises(TimeoutError, match="首个有效"):
            c.wait_first_frame(camera)


def saved_session(tmp_path, status="failed", seal=True):
    session = tmp_path / "scan_example"
    capture = session / "capture"
    capture.mkdir(parents=True)
    color, depth = capture / "color.png", capture / "depth.png"
    color.write_bytes(b"original color")
    depth.write_bytes(b"original depth")
    row = {"frame_id": 0, "timestamp_us": 1000, "color_path": str(color), "depth_path": str(depth),
           "intrinsics": [4., 4., 2., 1.]}
    (capture / "frames.jsonl").write_text(json.dumps(row) + "\n" + '{"uncommitted":')
    if seal:
        seal_capture(capture, source="test_capture")
    atomic_json(session / "session.json", {"status": status, "mode": "camera", "error": "original failure", "started": 1.})
    return session


@pytest.mark.parametrize("sealed", [False, True])
def test_reprocess_keeps_prior_attempts_and_raw_data(tmp_path, sealed):
    session = saved_session(tmp_path, seal=sealed)
    original = (session / "session.json").read_bytes()
    args = SimpleNamespace(replay=None, output=tmp_path)
    c = Controller(args)
    options = {"vlm": "none", "schedule": "serial", "refine": False}
    with patch.object(c, "validate_options", return_value=options), patch("pose_pipeline.live_gui.threading.Thread") as thread:
        thread.return_value.is_alive.return_value = False
        c.reprocess({"session": session.name, **options})
        first = Path(c.state["attempt"])
        assert Path(c.state["pipeline"]) == first / "pipeline"
        c.update(status="failed", error="first attempt failed")
        first_receipt = (first / "attempt.json").read_bytes()
        (first / "mapping.log").write_text("failure evidence")
        c.reprocess({"session": session.name, **options})
    assert Path(c.state["attempt"]) != first
    assert (first / "attempt.json").read_bytes() == first_receipt
    assert (first / "mapping.log").read_text() == "failure evidence"
    assert original in [p.read_bytes() for p in (session / "history").glob("*.json")]
    assert (session / "capture/color.png").read_bytes() == b"original color"
    manifest = json.loads((session / "capture/manifest.json").read_text())
    assert len(manifest["frames"]) == 1
    assert c.sessions()[0]["attempts"] == 2


def test_restart_lists_and_opens_interrupted_session_without_rewriting_receipt(tmp_path):
    session = saved_session(tmp_path, status="mapping")
    before = (session / "session.json").read_bytes()
    c = Controller(SimpleNamespace(replay=None, output=tmp_path))
    assert c.sessions()[0]["status"] == "interrupted"
    c.open_session(session.name)
    assert c.snapshot()["status"] == "interrupted"
    assert c.snapshot()["can_reprocess"]
    assert (session / "session.json").read_bytes() == before
    for name in ("../scan_example", str(session), "..", None):
        with pytest.raises(ValueError):
            c.open_session(name)


def test_failed_processing_can_retry_in_another_attempt(tmp_path):
    session = saved_session(tmp_path)
    c = Controller(SimpleNamespace(replay=None, output=tmp_path, cpu_python="unused", runtime=tmp_path / "runtime.json"))
    options = {"vlm": "none", "schedule": "serial", "refine": False}
    child = Mock()
    child.wait.return_value = 2
    with patch.object(c, "validate_options", return_value=options), patch.object(c, "launch", return_value=child):
        c.reprocess({"session": session.name, **options})
        c.thread.join(timeout=3)
        assert not c.thread.is_alive()
        first = Path(c.state["attempt"])
        assert c.snapshot()["status"] == "failed"
        assert c.snapshot()["can_reprocess"]
        assert json.loads((first / "attempt.json").read_text())["status"] == "failed"
        c.reprocess({"session": session.name, **options})
        c.thread.join(timeout=3)
        assert c.state["attempt"] != str(first)
        assert c.state["status"] == "failed"


def test_http_history_open_completed_result_and_reject_unauthorized_write(tmp_path):
    from http.server import ThreadingHTTPServer
    import threading
    import urllib.error
    import urllib.request
    from plyfile import PlyData, PlyElement
    from pose_pipeline.live_gui import handler

    session = saved_session(tmp_path, status="completed")
    vertices = np.zeros(1, dtype=[(key, "f4") for key in ("x", "y", "z")] +
                        [(key, "u1") for key in ("red", "green", "blue")] +
                        [(key, "i4") for key in ("semantic_id", "instance_id")])
    vertices["semantic_id"], vertices["instance_id"] = 1, 7
    cloud = session / "result.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(cloud)
    classes = session / "classes.json"
    trajectory = session / "trajectory.json"
    atomic_json(trajectory, {})
    atomic_json(classes, {"0": "unknown", "1": "chair"})
    atomic_json(session / "session.json", {"status": "completed", "result": {
        "final_cloud": str(cloud), "raw_map": str(cloud), "classes": str(classes), "trajectory": str(trajectory)}})
    c = Controller(SimpleNamespace(replay=None, output=tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(c, "test-token", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/sessions") as response:
            assert json.load(response)["sessions"][0]["id"] == session.name
        body = json.dumps({"session": session.name}).encode()
        request = urllib.request.Request(base + "/api/open", data=body,
                                         headers={"X-Scan-Token": "wrong", "Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        request.headers["X-scan-token"] = "test-token"
        with urllib.request.urlopen(request) as response:
            state = json.load(response)
        assert state["status"] == "completed"
        assert state["cloud"]["points"] == 1
        with urllib.request.urlopen(base + "/final.ply") as response:
            assert response.read() == cloud.read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_progress_tracks_semantics_after_geometry(tmp_path):
    rows = [{"event": "process_start", "stage": "mapping", "monotonic": 1},
            {"event": "process_complete", "stage": "mapping", "monotonic": 2},
            {"event": "process_start", "stage": "sam3", "monotonic": 3}]
    (tmp_path / "events_1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    state = {"status": "mapping", "options": {"refine": True}}
    p = pipeline_progress(tmp_path, state)
    assert p["label"] == "SAM3 分割"
    assert {row["id"]: row["status"] for row in p["stages"]}["mapping"] == "completed"
    rows.append({"event": "process_complete", "stage": "sam3", "monotonic": 4})
    (tmp_path / "events_1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    atomic_json(tmp_path / "GUI_STAGE.json", {"stage": "refine/ground"})
    assert pipeline_progress(tmp_path, state)["label"] == "未知点补全 / 候选定位"
    assert all(row["status"] == "completed" for row in pipeline_progress(tmp_path, {**state, "status": "completed"})["stages"])


def sealed_gui_fixture(controller):
    """Synthetic one-point artifact for publication checks; no model was run."""
    from plyfile import PlyData, PlyElement
    from pose_pipeline.artifacts import write_artifact_manifest
    from pose_pipeline.contracts import PoseRecord, write_trajectory
    output = controller.pipeline_path()
    output.mkdir()
    vertices = np.zeros(1, dtype=[(key, 'f4') for key in ('x', 'y', 'z')] +
        [(key, 'u1') for key in ('red', 'green', 'blue')] +
        [(key, 'i4') for key in ('semantic_id', 'instance_id')])
    cloud = output / 'map.ply'
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(cloud)
    classes, trajectory = output / 'classes.json', output / 'trajectory.json'
    atomic_json(classes, {'0': 'unknown'})
    write_trajectory(trajectory, [PoseRecord(0, 1000, np.eye(4))], sequence_id=controller.session.name, arm='synthetic-test')
    result = {'status': 'completed', 'final_cloud': str(cloud), 'raw_map': str(cloud),
        'classes': str(classes), 'trajectory': str(trajectory), 'artifacts': str(output / 'ARTIFACTS.json'),
        'model_inference_executed': False, 'scope': 'synthetic publication test only'}
    atomic_json(output / 'GUI_RESULT.json', result)
    write_artifact_manifest(output, map_path=cloud, classes_path=classes,
        result_path=output / 'GUI_RESULT.json', manifest_path=controller.session / 'capture/manifest.json', trajectory_path=trajectory)
    atomic_json(output / 'GUI_STAGE.json', {'stage': 'completed'})
    return result


def test_final_result_validates_before_committing_attempt_and_rejects_old_output(tmp_path):
    controller = Controller(SimpleNamespace(replay=None, output=tmp_path))
    controller.session = saved_session(tmp_path)
    controller.new_attempt()
    first_output = controller.pipeline_path()
    result = sealed_gui_fixture(controller)
    assert controller.validate_final_result(first_output) == result
    assert json.loads((Path(controller.state['attempt']) / 'attempt.json').read_text())['status'] == 'mapping'
    controller.update(status='failed', error='injected before GUI commit')
    first_receipt = (Path(controller.state['attempt']) / 'attempt.json').read_bytes()
    controller.new_attempt()
    second = controller.pipeline_path(); second.mkdir()
    # Copy only old pointers and its inventory: no new output is manufactured.
    for name in ('GUI_RESULT.json', 'ARTIFACTS.json', 'GUI_STAGE.json'):
        (second / name).write_bytes((first_output / name).read_bytes())
    with pytest.raises(ValueError):
        controller.validate_final_result(second)
    assert (first_output.parent / 'attempt.json').read_bytes() == first_receipt
    assert controller.state['status'] == 'mapping'


def test_history_does_not_publish_tampered_completed_attempt(tmp_path):
    controller = Controller(SimpleNamespace(replay=None, output=tmp_path))
    controller.session = saved_session(tmp_path)
    controller.new_attempt()
    result = sealed_gui_fixture(controller)
    controller.update(status='completed', result=result)
    session = controller.session
    before = (session / 'session.json').read_bytes()
    Path(result['classes']).write_text('{"0":"tampered"}')
    restarted = Controller(SimpleNamespace(replay=None, output=tmp_path))
    restarted.open_session(session.name)
    assert restarted.state['status'] == 'failed'
    assert '摘要' in restarted.state['error']
    assert (session / 'session.json').read_bytes() == before
    assert not (session / 'cloud.json').exists()


def test_history_cannot_bind_old_map_to_a_failed_new_attempt(tmp_path):
    controller = Controller(SimpleNamespace(replay=None, output=tmp_path))
    controller.session = saved_session(tmp_path)
    controller.new_attempt()
    old_output = controller.pipeline_path()
    old_result = sealed_gui_fixture(controller)
    controller.update(status='completed', result=old_result)
    controller.new_attempt()
    failed_attempt = Path(controller.state['attempt'])
    controller.update(status='failed', error='new attempt has no map')
    before = (failed_attempt / 'attempt.json').read_bytes()
    # An inconsistent history pointer must not publish the old map or turn the
    # failed attempt into a completed one via show_cloud -> update.
    atomic_json(controller.session / 'session.json', {
        **controller.state, 'status': 'completed', 'pipeline': str(old_output), 'result': old_result})
    restarted = Controller(SimpleNamespace(replay=None, output=tmp_path))
    restarted.open_session(controller.session.name)
    assert restarted.state['status'] == 'failed'
    assert '归属' in restarted.state['error']
    assert (failed_attempt / 'attempt.json').read_bytes() == before
    assert not (controller.session / 'cloud.json').exists()


def test_final_result_rejects_an_attempt_outside_its_session(tmp_path):
    controller = Controller(SimpleNamespace(replay=None, output=tmp_path))
    controller.session = saved_session(tmp_path)
    controller.new_attempt()
    original_output = controller.pipeline_path()
    sealed_gui_fixture(controller)
    foreign = tmp_path / 'foreign_attempt'
    Path(controller.state['attempt']).rename(foreign)
    controller.state.update(attempt=str(foreign), pipeline=str(foreign / 'pipeline'))
    with pytest.raises(ValueError, match='归属'):
        controller.validate_final_result(foreign / 'pipeline')
    assert not original_output.exists()


def test_retry_waits_for_real_group_members_after_leader_exit(tmp_path):
    attempt = tmp_path / 'attempts/old'
    attempt.mkdir(parents=True)
    script = tmp_path / 'early_exit.py'
    script.write_text('''import subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
Path(sys.argv[1]).write_text(str(p.pid))
''')
    leader = subprocess.Popen([sys.executable, str(script), str(tmp_path / 'descendant.pid')],
                              start_new_session=True)
    try:
        leader.wait(timeout=3)
        child = int((tmp_path / 'descendant.pid').read_text())
        assert process_running(child)
        atomic_json(attempt / 'mapping_launch.json', {'pid': leader.pid, 'process_start': 'exited owner'})
        controller = Controller(SimpleNamespace(replay=None, output=tmp_path))
        with pytest.raises(ValueError, match='仍有工作进程'):
            controller.ensure_session_workers_stopped(tmp_path)
        assert process_running(child)  # This read-only guard must not kill it.
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if leader.poll() is None:
            leader.kill(); leader.wait()


def test_retry_does_not_treat_a_reused_leader_pid_as_owned(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempts/old'
    attempt.mkdir(parents=True)
    atomic_json(attempt / 'mapping_launch.json', {'pid': 123456, 'process_start': 'original start'})
    controller = Controller(SimpleNamespace(replay=None, output=tmp_path))
    monkeypatch.setattr(controller, 'process_identity', lambda pid: {'started': 'different start', 'command': 'unrelated'})
    def do_not_probe_unrelated_group(pid):
        raise AssertionError('reused leader PID is not this attempt\'s owned group')
    monkeypatch.setattr(controller, 'process_group_members', do_not_probe_unrelated_group)
    controller.ensure_session_workers_stopped(tmp_path)
