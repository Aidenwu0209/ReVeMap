"""Loopback-only scan GUI with isolated camera, preview, and final-mapping workers."""
import argparse
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
from urllib.parse import urlparse
import webbrowser

from .live_io import BASE_COMMIT, atomic_json, read_json, publish_cloud


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
        self.env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                    "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            if self.session:
                atomic_json(self.session / "session.json", self.state)

    def snapshot(self):
        with self.lock:
            value = dict(self.state)
            if self.session:
                value["session"] = str(self.session)
                for key, name in (("capture", "capture_status.json"), ("preview", "preview_status.json"),
                                  ("mapping", "pipeline/mapping/run_status.json"), ("cloud", "cloud.json")):
                    value[key] = read_json(self.session / name)
                # Large provenance hashes stay in the on-disk receipt.
                value["mapping"].pop("source_sha256", None)
                value["semantic_stage"] = read_json(self.session / "pipeline/GUI_STAGE.json").get("stage", "")
                if not value["semantic_stage"]:
                    pipe = self.session / "pipeline"
                    stage = "轨迹与几何建图"
                    for file, label in [("sam3.log", "SAM3 分割"), ("vlm.log", "VLM 命名 / 3D 融合"),
                                        ("fusion.log", "3D 融合 / 等待命名"), ("backfill.log", "命名结果写回")]:
                        if (pipe / file).exists():
                            stage = label
                    value["semantic_stage"] = stage
            return value

    def start(self, options=None):
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
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("当前扫描仍在进行")
            self.options = {"vlm": model, "schedule": schedule, "refine": refine}
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

    def stop(self):
        with self.lock:
            if self.state["status"] not in ("starting", "recording"):
                raise ValueError("当前没有正在采集的扫描")
            (self.session / "stop_capture").touch()
            self.update(status="stopping")

    @staticmethod
    def reap_cancelled(children):
        # Give owned workers time to run their own descendant cleanup first.
        deadline = time.monotonic() + 20
        for child in children:
            try:
                child.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
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
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
            threading.Thread(target=self.reap_cancelled, args=(list(self.children),), daemon=True).start()

    def launch(self, python, module, arguments, name):
        with self.lock:
            if self.cancelled.is_set():
                raise InterruptedError("已取消，原始数据保留")
            command = [str(python), "-u", "-m", module, *map(str, arguments)]
            with (self.session / (name+".log")).open("xb") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=self.env, start_new_session=True)
            self.children.append(child)
            atomic_json(self.session / (name+"_launch.json"), {"pid": child.pid, "command": command})
            return child

    def finish_preview(self, child):
        (self.session / "stop_preview").touch()
        try:
            child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def run(self):
        preview = None
        try:
            arguments = ["--session", self.session]
            if self.args.replay:
                arguments += ["--replay", self.args.replay, "--fps", self.args.fps,
                              "--max-frames", self.args.max_frames]
            camera = self.launch(self.args.capture_python, "pose_pipeline.live_capture", arguments, "capture")
            # Wait for one real RGB-D frame before allocating GPU memory.
            while camera.poll() is None:
                cap = read_json(self.session / "capture_status.json")
                if cap.get("frames", 0):
                    break
                time.sleep(.1)
            if camera.poll() is None and not self.cancelled.is_set():
                preview = self.launch(self.args.gpu_python, "pose_pipeline.live_preview",
                    ["--session", self.session, "--provider-root", self.args.provider_root], "preview")
                with self.lock:
                    if self.state["status"] == "starting":
                        self.update(status="recording")
            code = camera.wait()
            if preview:
                self.finish_preview(preview)
            if self.cancelled.is_set():
                raise InterruptedError("已取消，原始数据保留")
            cap = read_json(self.session / "capture_status.json")
            if code or cap.get("status") != "sealed":
                raise RuntimeError(cap.get("error", "相机采集失败，详见 capture.log"))
            self.update(status="mapping")
            child = self.launch(self.args.cpu_python, "pose_pipeline.live_semantic", [
                "--manifest", self.session / "capture/manifest.json", "--runtime", self.args.runtime,
                "--output", self.session / "pipeline", "--schedule", self.options["schedule"],
                "--vlm", self.options["vlm"], *(["--refine"] if self.options["refine"] else []),
            ], "mapping")
            code = child.wait()
            if self.cancelled.is_set():
                raise InterruptedError("已取消，原始数据保留")
            if code:
                raise RuntimeError("语义建图失败，详见 mapping.log 和 pipeline 下各阶段日志")
            result = read_json(self.session / "pipeline/GUI_RESULT.json")
            self.update(status="completed", result=result, finished=time.time())
            self.show_cloud("semantic_id")
        except BaseException as error:
            self.update(status="cancelled" if self.cancelled.is_set() else "failed", error=str(error))
        finally:
            if preview and preview.poll() is None:
                self.finish_preview(preview)
            for child in self.children:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)

    def show_cloud(self, mode):
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
            publish_cloud(self.session, np.column_stack((xyz, rgb)), kind="final", revision=time.time_ns())
            self.update(view=mode)


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
                if path == "/api/view":
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4096:
                        raise ValueError("Invalid request size")
                    controller.show_cloud(json.loads(self.rfile.read(size))["mode"])
                    return self.respond(200, controller.snapshot())
                actions = {"/api/start": controller.start, "/api/stop": controller.stop, "/api/cancel": controller.cancel}
                if path not in actions:
                    return self.respond(404, {"error": "unknown action"})
                if path == "/api/start":
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= size <= 4096:
                        raise ValueError("Invalid request size")
                    options = json.loads(self.rfile.read(size)) if size else {}
                    if not isinstance(options, dict):
                        raise ValueError("Expected options object")
                    controller.start(options)
                else:
                    actions[path]()
                self.respond(200, controller.snapshot())
            except (ValueError, OSError, KeyError) as error:
                self.respond(409, {"error": str(error)})
    return Handler


def main():
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
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
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
