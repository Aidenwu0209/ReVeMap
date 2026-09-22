"""CPU regression tests for committed capture data and recoverable GUI sessions."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pose_pipeline import live_capture
from pose_pipeline.live_gui import Controller, pipeline_progress
from pose_pipeline.live_io import FrameJournalReader, atomic_json, journal_frames, seal_capture


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
