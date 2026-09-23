"""Loopback-only scan GUI with isolated camera, preview, and final-mapping workers."""
import argparse
import hashlib
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import time
from urllib.parse import urlparse, parse_qs
import webbrowser

from .live_io import BASE_COMMIT, atomic_json, read_json, publish_cloud, journal_frames, seal_capture


ACTIVE = {"starting", "recording", "stopping", "mapping", "cancelling"}


def pipeline_progress(pipeline, state):
    """Read structured process events instead of inferring work from log names."""
    labels = [("capture", "采集 RGB-D"), ("mapping", "轨迹与几何"), ("view_selection", "选择视图"), ("sam3", "SAM3 分割"),
              ("vlm", "多视角命名"), ("fusion", "语义融合"), ("backfill", "名称写回")]
    if state.get("options", {}).get("vlm") == "none":
        labels = [(key, label) for key, label in labels if key != "vlm"]
    if state.get("options", {}).get("refine", False):
        labels.append(("refine", "未知点补全"))
    labels.append(("export", "导出结果"))
    stages = {key: {"id": key, "label": label, "status": "pending"} for key, label in labels}
    status = state.get("status")
    stages["capture"]["status"] = "running" if status in {"starting", "recording", "stopping"} else "completed"
    if status == "idle":
        stages["capture"]["status"] = "pending"
    elif status in {"failed", "interrupted", "cancelled"} and not state.get("attempt") and not pipeline.exists():
        stages["capture"]["status"] = status
    events = sorted((row for path in pipeline.glob("events_*.jsonl") for row in journal_frames(path)),
                    key=lambda row: row.get("monotonic", 0))
    for row in events:
        key = row.get("stage")
        if key in stages and row.get("event") in {"process_start", "process_complete"}:
            stages[key]["status"] = "running" if row["event"] == "process_start" else "completed"
    gui_stage = read_json(pipeline / "GUI_STAGE.json").get("stage", "")
    if gui_stage.startswith("refine/") and "refine" in stages:
        detail = gui_stage.split("/", 1)[1]
        names = {"prepare": "整理视图", "name": "多视角命名", "decide": "证据核对",
                 "ground": "候选定位", "apply": "补全写回"}
        stages["refine"].update(status="running", detail=names.get(detail, detail))
    elif gui_stage == "export":
        if "refine" in stages:
            stages["refine"]["status"] = "completed"
        stages["export"]["status"] = "running"
    if status == "completed":
        for row in stages.values():
            row["status"] = "completed"
    elif status in {"failed", "interrupted", "cancelled"}:
        for row in stages.values():
            if row["status"] == "running":
                row["status"] = status
    running = [row["label"] + (" / " + row["detail"] if row.get("detail") else "")
               for row in stages.values() if row["status"] == "running"]
    return {"stages": list(stages.values()), "label": " + ".join(running),
            "elapsed_s": max(0, state.get("finished", time.time()) - state.get("processing_started", state.get("started", time.time())))}


class Controller:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.state = {"status": "idle", "base_commit": BASE_COMMIT,
                      "mode": "replay" if args.replay else "camera"}
        self.session = None
        self.thread = None
        self.children = []
        self._scene_cache = None
        self.env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                    "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state["updated"] = time.time()
            if self.session:
                atomic_json(self.session / "session.json", self.state)
                if self.state.get("attempt"):
                    atomic_json(Path(self.state["attempt"]) / "attempt.json", self.state)

    def pipeline_path(self):
        return Path(self.state.get("pipeline", self.session / "pipeline"))

    def snapshot(self):
        with self.lock:
            value = dict(self.state)
            if self.session:
                value["session"] = str(self.session)
                for key, name in (("capture", "capture_status.json"), ("preview", "preview_status.json"),
                                  ("cloud", "cloud.json")):
                    value[key] = read_json(self.session / name)
                pipe = self.pipeline_path()
                value["mapping"] = read_json(pipe / "mapping/run_status.json")
                # Large provenance hashes stay in the on-disk receipt.
                value["mapping"].pop("source_sha256", None)
                value["progress"] = pipeline_progress(pipe, value)
                value["semantic_stage"] = value["progress"]["label"]
                value["can_reprocess"] = ((self.session / "capture/manifest.json").is_file() or
                                           (self.session / "capture/frames.jsonl").is_file())
                if value.get("status") == "completed":
                    value["scene_context"] = self.scene_context()
            return value

    def scene_context(self):
        """Bind a query/evidence interaction to one completed map and attempt."""
        result = self.state.get("result", {})
        files = []
        for field in ("final_cloud", "classes", "artifacts", "names"):
            if result.get(field):
                path = Path(result[field])
                try:
                    stat = path.stat()
                    files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
                except OSError:
                    files.append((str(path.resolve()), None, None))
        identity = [str(self.session), self.state.get("attempt"), files]
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def validate_options(self, options=None):
        options = options or {}
        model = options.get("vlm", "qwen3vl_2b_nf4")
        schedule = options.get("schedule", "serial")
        from .semantic_runtime.common import registry, validate_runtime
        if model not in registry() or registry()[model]["kind"] == "ocr" or schedule not in ("serial", "parallel"):
            raise ValueError("Invalid model or schedule")
        refine = options.get("refine", True)
        if type(refine) is not bool:
            raise ValueError("refine must be boolean")
        validate_runtime(read_json(self.args.runtime), model)
        if refine:
            validate_runtime(read_json(self.args.runtime), "qwen3vl_2b_nf4", raw_mapping=False)
        return {"vlm": model, "schedule": schedule, "refine": refine}

    def ensure_idle(self):
        if self.thread and self.thread.is_alive():
            raise ValueError("当前扫描仍在进行")

    def resolve_session(self, name):
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid session name")
        root = self.args.output.resolve()
        path = (root / name).resolve(strict=True)
        if path.parent != root or not (path / "session.json").is_file():
            raise ValueError("Unknown session")
        return path

    def sessions(self):
        rows = []
        for file in sorted(self.args.output.glob("*/session.json"), reverse=True):
            try:
                path = self.resolve_session(file.parent.name)
                state = read_json(file)
                active = self.session == path and self.thread and self.thread.is_alive()
                status = state.get("status", "unknown")
                if status in ACTIVE and not active:
                    status = "interrupted"
                rows.append({"id": path.name, "status": status, "started": state.get("started"),
                             "can_reprocess": ((path / "capture/manifest.json").is_file() or
                                               (path / "capture/frames.jsonl").is_file()),
                             "attempts": len(list((path / "attempts").glob("*/attempt.json")))})
            except (ValueError, OSError, TypeError):
                continue
        return rows

    def open_session(self, name):
        with self.lock:
            self.ensure_idle()
            path = self.resolve_session(name)
            state = read_json(path / "session.json")
            if state.get("status") in ACTIVE:
                state.update(status="interrupted", error="上次处理已中断，可使用已保存数据重新处理。")
            self.session, self.state = path, state
            self.children = []
            if state.get("status") == "completed":
                try:
                    if state.get("attempt"):
                        self.state["result"] = self.validate_final_result(self.pipeline_path())
                        from .artifacts import load_artifacts
                        load_artifacts(self.pipeline_path(), require_provenance=True)
                    self.show_cloud(state.get("view", "semantic_id"))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.state.update(status="failed", error=f"历史结果无法打开：{error}；可从原始数据重新处理。")

    def start(self, options=None):
        options = self.validate_options(options)
        with self.lock:
            self.ensure_idle()
            self.options = options
            self.cancelled.clear()
            stamp = datetime.now().strftime("scan_%Y%m%d_%H%M%S_")+secrets.token_hex(3)
            self.session = (self.args.output / stamp).resolve()
            self.session.mkdir(parents=True, exist_ok=False)
            self.children = []
            self.state = {"status": "starting", "base_commit": BASE_COMMIT,
                          "mode": "replay" if self.args.replay else "camera", "started": time.time(), "options": self.options}
            atomic_json(self.session / "session.json", self.state)
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def reprocess(self, options=None):
        requested = options or {}
        options = self.validate_options(requested)
        with self.lock:
            self.ensure_idle()
            name = requested.get("session") or (self.session.name if self.session else None)
            path = self.resolve_session(name)
            self.ensure_session_workers_stopped(path)
            manifest = path / "capture/manifest.json"
            if not manifest.is_file():
                # Abrupt termination can leave a valid committed journal without
                # a manifest. Seal those rows explicitly on a retry request.
                seal_capture(path / "capture", source="recovered_gui_capture")
            from .contracts import load_manifest
            if not load_manifest(manifest).frames:
                raise ValueError("该扫描没有可重新处理的有效 RGB-D 帧")
            old = read_json(path / "session.json")
            previous_pipeline = Path(old["pipeline"]).resolve() if old.get("pipeline") else None
            if previous_pipeline and (not previous_pipeline.is_relative_to(path / "attempts")
                                      or not previous_pipeline.is_dir()):
                previous_pipeline = None
            history = path / "history"
            history.mkdir(exist_ok=True)
            atomic_json(history / (f"session_{time.time_ns()}.json"), old)
            self.session = path
            self.options = options
            self.cancelled.clear()
            self.children = []
            self.state = {"status": "mapping", "base_commit": BASE_COMMIT,
                          "mode": old.get("mode", "replay"), "started": old.get("started", time.time()),
                          "options": options, "resume_from": str(previous_pipeline) if previous_pipeline else None}
            self.new_attempt()
            self.thread = threading.Thread(target=self.run_reprocess, daemon=True)
            self.thread.start()

    def new_attempt(self):
        stamp = datetime.now().strftime("attempt_%Y%m%d_%H%M%S_") + secrets.token_hex(3)
        attempt = self.session / "attempts" / stamp
        attempt.mkdir(parents=True, exist_ok=False)
        self.update(status="mapping", attempt=str(attempt), pipeline=str(attempt / "pipeline"),
                    processing_started=time.time())

    @staticmethod
    def process_identity(pid):
        """Read a local process identity without signaling an unrelated PID."""
        result = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "stat=", "-o", "lstart=", "-o", "args="],
                                capture_output=True, text=True, check=False)
        fields = result.stdout.strip().split(maxsplit=6)
        if len(fields) != 7 or fields[0].startswith("Z"):
            return None
        return {"started": " ".join(fields[1:6]), "command": fields[6]}

    def ensure_session_workers_stopped(self, session):
        receipts = [session / "capture_launch.json", session / "preview_launch.json",
                    *(session / "attempts").glob("*/mapping_launch.json")]
        for path in receipts:
            receipt = read_json(path)
            if type(receipt.get("pid")) is not int or receipt["pid"] <= 0:
                continue
            current = self.process_identity(receipt["pid"])
            if current is None:
                # A start_new_session leader may exit before its owned group.
                # Do not allow a retry to overlap those remaining workers.
                if self.process_group_members(receipt["pid"]):
                    raise ValueError("该扫描的上次处理仍有工作进程在退出，请稍后重新处理；未改动已有 attempt。")
                continue
            # Creation time protects against PID reuse. Older launch receipts
            # can still be recognized by their exact session/attempt argument.
            same_start = receipt.get("process_start") == current["started"]
            legacy_match = (not receipt.get("process_start") and str(session) in current["command"])
            if same_start or legacy_match:
                raise ValueError("该扫描的上次处理进程仍在退出，请稍后重新处理；未改动已有 attempt。")

    @staticmethod
    def process_group_members(group_id):
        """Read live members of an owned launch group; never signal a process.

        The caller first checks any extant leader's creation identity, so a
        reused leader PID does not make an unrelated group block this session.
        An orphaned live group retains its PGID after its leader is reaped.
        """
        result = subprocess.run(["ps", "-ax", "-o", "pid=", "-o", "pgid=", "-o", "stat="],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise ValueError("无法确认上次处理进程是否已退出，请稍后重新处理。")
        members = []
        for line in result.stdout.splitlines():
            fields = line.split()
            if (len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit()
                    and int(fields[1]) == group_id and not fields[2].startswith("Z")):
                members.append(int(fields[0]))
        return members

    def stop(self):
        with self.lock:
            if self.state["status"] not in ("starting", "recording"):
                raise ValueError("当前没有正在采集的扫描")
            (self.session / "stop_capture").touch()
            self.update(status="stopping")

    @staticmethod
    def reap_cancelled(children, timeout=20.):
        # Give owned workers time to run their own descendant cleanup first.
        deadline = time.monotonic() + timeout
        for child in children:
            try:
                child.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        # A leader can exit before its descendants. Every group belongs to a
        # start_new_session launch, so also reap groups whose leader exited.
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()

    def cancel(self):
        with self.lock:
            if not self.thread or not self.thread.is_alive():
                return
            self.cancelled.set()
            (self.session / "stop_capture").touch()
            (self.session / "stop_preview").touch()
            self.update(status="cancelling")
            for child in self.children:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            # The owning run thread performs bounded cleanup before a new
            # attempt is allowed; no detached reaper races a later attempt.

    def launch(self, python, module, arguments, name, *, output=None):
        with self.lock:
            if self.cancelled.is_set():
                raise InterruptedError("已取消，原始数据保留")
            command = [str(python), "-u", "-m", module, *map(str, arguments)]
            output = output or self.session
            with (output / (name+".log")).open("xb") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env={**self.env, "REVEMAP_SUPERVISOR_PID": str(os.getpid())},
                                         start_new_session=True)
            self.children.append(child)
            identity = self.process_identity(child.pid)
            atomic_json(output / (name+"_launch.json"), {"pid": child.pid, "command": command,
                        "process_start": identity["started"] if identity else None})
            return child

    def wait_first_frame(self, camera):
        deadline = time.monotonic() + getattr(self.args, "first_frame_timeout", 30.)
        while camera.poll() is None:
            if self.cancelled.is_set():
                raise InterruptedError("已取消，原始数据保留")
            if read_json(self.session / "capture_status.json").get("frames", 0):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("等待首个有效 RGB-D 帧超时，请检查相机与采集进程日志。")
            time.sleep(.1)

    def process_manifest(self):
        output = self.pipeline_path()
        child = self.launch(self.args.cpu_python, "pose_pipeline.live_semantic", [
            "--manifest", self.session / "capture/manifest.json", "--runtime", self.args.runtime,
            "--output", output, "--schedule", self.options["schedule"],
            "--vlm", self.options["vlm"], *(["--refine"] if self.options["refine"] else []),
            "--checkpoint-stages",
            *(["--resume-from", self.state["resume_from"]] if self.state.get("resume_from") else []),
        ], "mapping", output=Path(self.state["attempt"]))
        while child.poll() is None:
            if self.cancelled.is_set():
                raise InterruptedError("已取消，原始数据保留")
            time.sleep(.1)
        code = child.wait()
        if self.cancelled.is_set():
            raise InterruptedError("已取消，原始数据保留")
        if code:
            raise RuntimeError(f"语义建图失败，日志：{Path(self.state['attempt']) / 'mapping.log'}")
        result = self.validate_final_result(output)
        self.update(status="completed", result=result, finished=time.time())
        self.show_cloud("semantic_id")

    def validate_final_result(self, output):
        """Validate before committing completed; the public reader rejects active attempts."""
        from .contracts import sha256_file, load_manifest, load_trajectory, bind_manifest_trajectory
        from plyfile import PlyData
        import numpy as np
        output = Path(output).resolve()
        session = self.session.resolve()
        attempt = Path(self.state.get("attempt", "")).resolve()
        if (attempt.parent != session / "attempts" or not attempt.is_dir()
                or output != attempt / "pipeline"
                or Path(self.state.get("pipeline", "")).resolve() != output):
            raise ValueError("结果的 session、attempt 与 pipeline 归属不一致")
        attempt_state = read_json(attempt / "attempt.json")
        if (Path(attempt_state.get("attempt", "")).resolve() != attempt
                or Path(attempt_state.get("pipeline", "")).resolve() != output):
            raise ValueError("结果与本次 attempt 的记录不一致")
        result_path, inventory_path = output / "GUI_RESULT.json", output / "ARTIFACTS.json"
        result, inventory = read_json(result_path), read_json(inventory_path)
        if (result.get("status") != "completed" or inventory.get("schema") != "revemap.artifacts.v1"
                or inventory.get("provenance_bound") is not True
                or read_json(output / "GUI_STAGE.json").get("stage") != "completed"):
            raise ValueError("本次处理未生成完整、已封存的 GUI 结果。")
        paths = {}
        for name, entry in inventory.get("files", {}).items():
            path = (inventory_path.parent / entry["path"]).resolve()
            if name == "input_manifest":
                if path != (self.session / "capture/manifest.json").resolve():
                    raise ValueError("结果绑定了其他采集数据")
            elif not path.is_relative_to(output):
                raise ValueError("结果引用了本次 attempt 以外的文件")
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError("GUI 结果文件摘要不一致：" + name)
            paths[name] = path
        required = {"map", "classes", "result", "input_manifest", "trajectory"}
        if not required <= paths.keys() or paths["result"] != result_path:
            raise ValueError("GUI 结果清单缺少必要文件")
        for field, key in (("final_cloud", "map"), ("classes", "classes"), ("trajectory", "trajectory")):
            if Path(result.get(field, "")).resolve() != paths[key]:
                raise ValueError("GUI 结果指针与清单不一致：" + field)
        if Path(result.get("artifacts", "")).resolve() != inventory_path:
            raise ValueError("GUI 结果指向其他 attempt 的清单")
        manifest = load_manifest(paths["input_manifest"])
        poses, trajectory = load_trajectory(paths["trajectory"])
        if (manifest.sequence_id != inventory.get("scene_id") or manifest.dataset != inventory.get("dataset")
                or trajectory.get("sequence_id") != manifest.sequence_id):
            raise ValueError("GUI 结果场景绑定不一致")
        bind_manifest_trajectory(manifest, poses)
        for name in ("FAILURE.json", "FUSION_FAILURE.json"):
            if (output / name).exists() or (output / "fused" / name).exists():
                raise ValueError("本次处理存在失败记录")
        raw = Path(result.get("raw_map", "")).resolve()
        if not raw.is_relative_to(output) or not raw.is_file():
            raise ValueError("GUI 原始地图不属于本次 attempt")
        final_vertex, raw_vertex = (PlyData.read(p)["vertex"].data for p in (paths["map"], raw))
        if not {"x", "y", "z", "semantic_id", "instance_id"} <= set(final_vertex.dtype.names):
            raise ValueError("GUI 地图缺少坐标或标签")
        if not {"x", "y", "z", "red", "green", "blue"} <= set(raw_vertex.dtype.names):
            raise ValueError("GUI 原始地图缺少坐标或颜色")
        if len(final_vertex) == 0 or len(final_vertex) != len(raw_vertex):
            raise ValueError("GUI 地图点数无效或点序不匹配")
        for name in ("x", "y", "z"):
            if not np.isfinite(final_vertex[name]).all() or not np.array_equal(final_vertex[name], raw_vertex[name]):
                raise ValueError("GUI 最终地图与原始地图坐标不匹配")
        return result

    def cleanup_children(self):
        for child in self.children:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        self.reap_cancelled(self.children)

    def run_reprocess(self):
        try:
            self.process_manifest()
        except BaseException as error:
            self.update(status="cancelled" if self.cancelled.is_set() else "failed",
                        error=str(error), finished=time.time())
        finally:
            self.cleanup_children()

    def finish_preview(self, child):
        (self.session / "stop_preview").touch()
        try:
            child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.reap_cancelled([child], timeout=0)

    def wait_capture(self, child):
        """Bound stalled SDK calls after startup by saved-frame progress."""
        previous = read_json(self.session / "capture_status.json").get("frames", 0)
        last_valid = time.monotonic()
        limit = getattr(self.args, "capture_frame_timeout", 15.)
        while child.poll() is None:
            if self.cancelled.is_set():
                raise InterruptedError("已取消，原始数据保留")
            count = read_json(self.session / "capture_status.json").get("frames", 0)
            if count > previous:
                previous, last_valid = count, time.monotonic()
            if not self.args.replay and time.monotonic() - last_valid >= limit:
                raise TimeoutError("采集进程长时间未保存有效 RGB-D 帧，已停止并保留原始数据。")
            time.sleep(.1)
        return child.wait()

    def run(self):
        preview = None
        try:
            arguments = ["--session", self.session]
            if self.args.replay:
                arguments += ["--replay", self.args.replay, "--fps", self.args.fps,
                              "--max-frames", self.args.max_frames]
            camera = self.launch(self.args.capture_python, "pose_pipeline.live_capture", arguments, "capture")
            # Wait for one real RGB-D frame before allocating GPU memory.
            self.wait_first_frame(camera)
            if camera.poll() is None and not self.cancelled.is_set():
                preview = self.launch(self.args.gpu_python, "pose_pipeline.live_preview",
                    ["--session", self.session, "--provider-root", self.args.provider_root], "preview")
                with self.lock:
                    if self.state["status"] == "starting":
                        self.update(status="recording")
            code = self.wait_capture(camera)
            if preview:
                self.finish_preview(preview)
            if self.cancelled.is_set():
                raise InterruptedError("已取消，原始数据保留")
            cap = read_json(self.session / "capture_status.json")
            if code or cap.get("status") != "sealed":
                raise RuntimeError(cap.get("error", "相机采集失败，详见 capture.log"))
            self.new_attempt()
            self.process_manifest()
        except BaseException as error:
            self.update(status="cancelled" if self.cancelled.is_set() else "failed", error=str(error), finished=time.time())
        finally:
            if preview and preview.poll() is None:
                self.finish_preview(preview)
            self.cleanup_children()

    def scene_graph(self, world_up=None):
        from .scene_graph import build_from_artifacts, build_graph
        from plyfile import PlyData
        import numpy as np
        with self.lock:
            if self.state["status"] != "completed":
                raise ValueError("地图尚未完成")
            result = self.state["result"]
            inventory = Path(result.get("artifacts", self.pipeline_path() / "ARTIFACTS.json"))
            files = [Path(result["final_cloud"]), Path(result["classes"])]
            names_file = Path(result.get("names", self.pipeline_path() / "fused/instance_names.json"))
            if names_file.is_file():
                files.append(names_file)
            if inventory.is_file():
                files.append(inventory)
                files.extend((inventory.parent / row["path"]).resolve()
                             for row in read_json(inventory).get("files", {}).values())
                for directory in [inventory.parent, *list(inventory.parent.parents)[:8]]:
                    files.extend(path for name in ("FAILURE.json", "FUSION_FAILURE.json", "COMPLETE.json", "GUI_RESULT.json", "attempt.json")
                                 if (path := directory / name).is_file())
            key = (tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files),
                   tuple(world_up) if world_up is not None else None)
            if self._scene_cache is not None and self._scene_cache[0] == key:
                return self._scene_cache[1]
            if inventory.is_file():
                graph = build_from_artifacts(inventory, world_up=world_up)
            else:
                # Old sessions remain inspectable, with no invented input binding.
                vertices = PlyData.read(files[0])["vertex"].data
                graph = build_graph(np.column_stack([vertices[k] for k in ("x", "y", "z")]),
                    vertices["semantic_id"], vertices["instance_id"], read_json(files[1]), world_up=world_up,
                    names=read_json(Path(result.get("names", self.pipeline_path() / "fused/instance_names.json"))) or [])
                graph["provenance"] = {"bound": False, "source": "legacy_gui_result"}
            self._scene_cache = (key, graph)
            return graph

    def query_scene(self, options):
        from .scene_graph import query_graph
        if not isinstance(options, dict) or set(options) - {"label", "nearest_to", "relation", "reference_id", "world_up", "context", "question", "query"}:
            raise ValueError("Invalid scene query")
        if options.get("label") is not None and (not isinstance(options["label"], str) or len(options["label"]) > 80):
            raise ValueError("对象名称需要不超过 80 字符")
        with self.lock:
            context = self.scene_context()
            if options.get("context") is not None and options["context"] != context:
                raise ValueError("地图已切换，请刷新后重新查询")
            graph = self.scene_graph(options.get("world_up"))
            if "question" in options or "query" in options:
                if any(k in options for k in ("label", "nearest_to", "relation", "reference_id")):
                    raise ValueError("Do not combine a question with legacy filters")
                from .scene_query import execute_query
                result = execute_query(graph, question=options.get("question"), plan=options.get("query"))
            else:
                result = query_graph(graph, **{k: v for k, v in options.items() if k not in {"world_up", "context"}})
            self.show_cloud(self.state.get("view", "semantic_id"), result["instance_ids"], context=context)
            result["provenance_bound"] = graph["provenance"]["bound"]
            result["context"] = context
            return {"query_result": result, "state": self.snapshot()}

    def evidence_image(self, instance_id, index, context):
        from .contracts import sha256_file
        with self.lock:
            if context != self.scene_context():
                raise ValueError("地图已切换，不能使用此前查询的对象证据")
            graph = self.scene_graph()
            node = next((n for n in graph["nodes"] if n["instance_id"] == instance_id), None)
            if node is None or not 0 <= index < len(node["evidence"]):
                raise ValueError("没有对应的对象观察图")
            record = node["evidence"][index]
            path = Path(record["file"])
            if sha256_file(path) != record.get("sha256"):
                raise ValueError("对象观察图已改变")
            # Only a hash-checked image already referenced by object evidence can
            # be served; arbitrary client file paths are never accepted.
            from PIL import Image
            from io import BytesIO
            output = BytesIO()
            with Image.open(path) as source:
                source.convert("RGB").save(output, format="JPEG")
            return output.getvalue()

    def show_cloud(self, mode, highlight=None, context=None):
        if mode not in ("rgb", "semantic_id", "instance_id"):
            raise ValueError("Unknown view")
        with self.lock:
            if self.state["status"] != "completed":
                raise ValueError("地图尚未完成")
            from plyfile import PlyData
            import numpy as np
            v = PlyData.read(self.state["result"]["final_cloud"])["vertex"].data
            xyz = np.column_stack([v[k] for k in ("x", "y", "z")])
            if mode == "rgb":
                # Refinement preserves positions but exports semantic colors.
                raw = PlyData.read(self.state["result"]["raw_map"])["vertex"].data
                rgb = np.column_stack([raw[k] for k in ("red", "green", "blue")]) / 255.
            else:
                ids = v[mode].astype(np.int64)
                rgb = np.column_stack([np.where(ids > 0, 50 + ids*m % 206, 90) for m in (73,151,199)]) / 255.
            if highlight is not None:
                if context != self.scene_context():
                    raise ValueError("地图已切换，请重新查询后高亮对象")
                if (not isinstance(highlight, list) or any(type(oid) is not int or oid <= 0 for oid in highlight)
                        or not set(highlight).issubset(set(v["instance_id"].tolist()))):
                    raise ValueError("Invalid highlighted instance IDs")
                selected = np.isin(v["instance_id"], highlight)
                rgb *= .18
                rgb[selected] = [1., .85, .25]
                if len(xyz) > 100000 and selected.any():
                    # Reserve a representative for every matching object before
                    # decimating the preview; small objects must remain visible.
                    selected_ids = np.flatnonzero(selected)
                    _, first = np.unique(v["instance_id"][selected_ids], return_index=True)
                    representatives = selected_ids[first]
                    if len(representatives) > 100000:
                        raise ValueError("匹配对象过多，请缩小查询范围")
                    rest = np.setdiff1d(selected_ids, representatives, assume_unique=True)
                    take = min(len(selected_ids), max(len(representatives), 80000)) - len(representatives)
                    retained = np.concatenate((representatives, rest[np.linspace(0, len(rest)-1, take).astype(int)] if take else [])).astype(int)
                    background = np.flatnonzero(~selected)
                    take = min(len(background), 100000-len(retained))
                    if take:
                        retained = np.concatenate((retained, background[np.linspace(0, len(background)-1, take).astype(int)]))
                    xyz, rgb = xyz[retained], rgb[retained]
            publish_cloud(self.session, np.column_stack((xyz, rgb)), kind="final", revision=time.time_ns())
            self.update(view=mode, highlighted_instances=highlight)


def handler(controller, token, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, data, mime="application/json"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if urlparse("http://"+self.headers.get("Host", "")).hostname not in {"127.0.0.1", "localhost"}:
                return self.respond(403, {"error": "loopback host required"})
            path = urlparse(self.path).path
            if path == "/":
                html = Path(__file__).with_name("live_gui.html").read_text()
                return self.respond(200, html.replace("__TOKEN__", token).encode(), "text/html; charset=utf-8")
            if path == "/api/classes":
                state = controller.snapshot()
                if state["status"] != "completed":
                    return self.respond(200, {})
                return self.respond(200, read_json(state["result"]["classes"]))
            if path == "/api/models":
                from .semantic_runtime.common import registry
                cfg = read_json(controller.args.runtime)
                models = [k for k, v in registry().items() if v["kind"] != "ocr" and (k == "none" or k in cfg.get("models", {}))]
                return self.respond(200, {"models": models})
            if path == "/api/status":
                return self.respond(200, controller.snapshot())
            if path == "/api/sessions":
                return self.respond(200, {"sessions": controller.sessions()})
            if path == "/api/evidence":
                try:
                    args = parse_qs(urlparse(self.path).query)
                    return self.respond(200, controller.evidence_image(int(args["instance_id"][0]), int(args["index"][0]), args["context"][0]), "image/jpeg")
                except (ValueError, OSError, KeyError, IndexError):
                    return self.respond(404, {"error": "对象观察图不可用"})
            with controller.lock:
                root = controller.session
            if root:
                files = {"/color.jpg": (root / "color.jpg", "image/jpeg"),
                         "/depth.jpg": (root / "depth.jpg", "image/jpeg"),
                         "/cloud.bin": (root / read_json(root / "cloud.json").get("file", "absent"), "application/octet-stream")}
                state = controller.snapshot()
                if state["status"] == "completed":
                    files["/final.ply"] = (Path(state["result"]["final_cloud"]), "application/octet-stream")
                    files["/trajectory.json"] = (Path(state["result"]["trajectory"]), "application/json")
                if path in files:
                    file, mime = files[path]
                    try:
                        return self.respond(200, file.read_bytes(), mime)
                    except FileNotFoundError:
                        pass
            return self.respond(404, {"error": "文件尚未生成"})

        def do_POST(self):
            if urlparse("http://"+self.headers.get("Host", "")).hostname not in {"127.0.0.1", "localhost"}:
                return self.respond(403, {"error": "loopback host required"})
            if self.headers.get("X-Scan-Token") != token:
                return self.respond(403, {"error": "invalid session token"})
            path = urlparse(self.path).path
            try:
                if path in {"/api/view", "/api/query"}:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4096:
                        raise ValueError("Invalid request size")
                    options = json.loads(self.rfile.read(size))
                    if not isinstance(options, dict):
                        raise ValueError("Expected options object")
                    if path == "/api/query":
                        return self.respond(200, controller.query_scene(options))
                    controller.show_cloud(options["mode"], options.get("highlight"), options.get("context"))
                    return self.respond(200, controller.snapshot())
                actions = {"/api/start": controller.start, "/api/stop": controller.stop,
                           "/api/cancel": controller.cancel, "/api/open": controller.open_session,
                           "/api/reprocess": controller.reprocess}
                if path not in actions:
                    return self.respond(404, {"error": "unknown action"})
                if path in {"/api/start", "/api/open", "/api/reprocess"}:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= size <= 4096:
                        raise ValueError("Invalid request size")
                    options = json.loads(self.rfile.read(size)) if size else {}
                    if not isinstance(options, dict):
                        raise ValueError("Expected options object")
                    if path == "/api/open":
                        controller.open_session(options.get("session"))
                    else:
                        actions[path](options)
                else:
                    actions[path]()
                self.respond(200, controller.snapshot())
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.respond(409, {"error": str(error)})
    return Handler


def main():
    import sys
    if any(arg in sys.argv for arg in ("--host", "--ipad-port", "--wireless-host", "--devices")):
        from .device_gui import main as device_main
        return device_main()
    p = argparse.ArgumentParser(description="ReVeMap RGB-D capture and semantic mapping GUI")
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--provider-root", type=Path, required=True)
    p.add_argument("--gpu-python", type=Path, required=True)
    p.add_argument("--cpu-python", type=Path, required=True)
    p.add_argument("--capture-python", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--replay", type=Path)
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--max-frames", type=int, default=0, help="Explicit replay-only frame limit")
    p.add_argument("--first-frame-timeout", type=float, default=30.,
                   help="Maximum seconds to wait for the first saved RGB-D frame")
    p.add_argument("--capture-frame-timeout", type=float, default=15.,
                   help="Maximum seconds without a newly saved RGB-D frame during camera capture")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    if not 0 < args.first_frame_timeout < float("inf"):
        p.error("first-frame-timeout must be finite and positive")
    if not 0 < args.capture_frame_timeout < float("inf"):
        p.error("capture-frame-timeout must be finite and positive")
    if args.fps <= 0 or args.max_frames < 0 or (args.max_frames and not args.replay):
        p.error("fps must be positive; max-frames is a nonnegative replay-only option")
    for key in ("provider_root", "gpu_python", "cpu_python", "capture_python"):
        value = getattr(args, key).absolute()
        if not value.exists():
            p.error(f"Missing {key}: {value}")
        setattr(args, key, value)
    args.output = args.output.resolve()
    if args.replay:
        args.replay = args.replay.resolve(strict=True)
    args.runtime = args.runtime.resolve(strict=True)
    controller = Controller(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(controller, secrets.token_hex(24), args.port))
    url = f"http://127.0.0.1:{args.port}"
    print(f"ReVeMap GUI: {url}\nSource: SGF-SGAligner/developnew@{BASE_COMMIT[:7]}\nOutputs: {args.output}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.cancel()
        if controller.thread:
            controller.thread.join(timeout=45)
        server.server_close()


if __name__ == "__main__":
    main()
