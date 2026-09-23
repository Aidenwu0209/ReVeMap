"""Device scan GUI with isolated capture, preview, and verified mapping workers."""
import argparse
import ipaddress
import shutil
import socket
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

from .live_io import BASE_COMMIT, atomic_json, read_json, publish_cloud
from .live_upload import UploadStore, MAX_MANIFEST, durable_json


from .live_gui import Controller as CoreController, ACTIVE

class Controller(CoreController):
    """Device transports and the Scan UI using the same verified processing core."""
    def __init__(self, args):
        super().__init__(args)
        self.shutdown = threading.Event()
        self.camera_lock = threading.RLock()
        self.live_camera = None
        if args.wireless_host:
            from .wireless_preview import WirelessPreview
            self.live_camera = WirelessPreview(args.wireless_host, args.wireless_port,
                depth_denoise=getattr(args, "depth_denoise", "bilateral"))
        self.uploads = UploadStore(args.output / "uploads")
        self.upload_dir = None
        self.state['mode'] = 'replay' if args.replay else 'ipad' if args.ipad_port else 'wireless' if args.wireless_host else 'camera'

    def snapshot(self):
        state = super().snapshot()
        state['wireless_depth_denoise'] = getattr(self.args, 'depth_denoise', 'off') if self.args.wireless_host else 'off'
        return state

    def library_roots(self):
        return list(dict.fromkeys([self.args.output.resolve(), *[Path(p).resolve() for p in getattr(self.args, 'library_root', [])]]))

    def resolve_session(self, name):
        if not isinstance(name, str) or not _safe_session_id(name):
            raise ValueError('Invalid session name')
        candidates = [(root / name).resolve() for root in self.library_roots()
                      if (root / name / 'session.json').is_file() and (root / name).resolve().parent == root]
        if len(candidates) != 1:
            raise ValueError('Unknown or ambiguous session')
        return candidates[0]

    def session_view(self, name):
        root = self.resolve_session(name)
        with self.lock:
            if root == self.session:
                return self
        view = CoreController(self.args)
        view.session = root
        view.state = read_json(root / 'session.json')
        if view.state.get('status') in ACTIVE:
            view.state = {**view.state, 'status': 'interrupted'}
        if view.state.get('status') == 'completed':
            view.state['result'] = completed_result(root)
        return view

    def process_manifest(self):
        super().process_manifest()
        # The graph has no vertical relations until world-up is explicitly known.
        graph = self.scene_graph()
        graph['context'] = self.scene_context()
        atomic_json(self.pipeline_path() / 'scene_graph.json', graph)

    def camera_preview(self):
        with self.camera_lock:
            if self.live_camera is None:
                return {"status": "unavailable"}
            if self.state["status"] in {"starting", "recording", "stopping", "cancelling"}:
                return {"status": "paused"}
            self.live_camera.touch()
            return self.live_camera.snapshot()


    def start(self, options=None, upload_dir=None):
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
        with self.camera_lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("当前扫描仍在进行")
            if self.live_camera:
                self.live_camera.stop()
            with self.lock:
                self.upload_dir = upload_dir
                self.options = {"vlm": model, "schedule": schedule, "refine": refine}
                self.cancelled.clear()
                stamp = datetime.now().strftime("scan_%Y%m%d_%H%M%S_")+secrets.token_hex(3)
                self.session = (self.args.output / stamp).resolve()
                self.session.mkdir(parents=True, exist_ok=False)
                self.children = []
                self.state = {"status": "starting", "base_commit": BASE_COMMIT,
                              "mode": "replay" if self.args.replay else ("ipad" if self.args.ipad_port else "wireless" if self.args.wireless_host else "camera"), "started": time.time(), "options": self.options}
                if upload_dir is not None:
                    self.state["upload_id"] = upload_dir.name
                durable_json(self.session / "session.json", self.state)
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()


    def complete_upload(self, upload_id):
        with self.uploads.lock, self.camera_lock, self.lock:
            receipt = self.uploads.seal(upload_id)
            root = self.uploads.directory(upload_id)
            for session in self.args.output.glob("scan_*"):
                if read_json(session / "session.json").get("upload_id") == root.name:
                    receipt["session"] = session.name
                    return receipt
            self.start(options=getattr(self, "options", None) or {}, upload_dir=root)
            receipt["session"] = self.session.name
            return receipt


    def _standby_listen(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.args.ipad_bind, self.args.ipad_port))
        listener.listen(1)
        listener.settimeout(1.0)
        return listener


    def ipad_standby_loop(self):
        """Keep the stream port armed: an iPad connection auto-starts a scan.

        Handshake protocol with the capture stage: on a connection probe the
        standby listener closes itself so live_capture can bind the port; the
        Scan app auto-reconnects (~1 s) straight into the capture stage. The
        port is re-armed after the whole session (capture+mapping) finishes.
        """
        try:
            listener = self._standby_listen()
        except OSError as error:
            self.update(standby_error=f"待命端口绑定失败: {error}")
            return
        self.update(standby="listening")
        while not self.shutdown.is_set():
            try:
                connection, address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            # Only a real stream start (protocol magic) triggers a session;
            # bare probes (the app's connection preflight) are ignored.
            try:
                connection.settimeout(2.0)
                head = b""
                deadline = time.monotonic() + 2.0
                while len(head) < 8:
                    connection.settimeout(max(0.01, deadline - time.monotonic()))
                    part = connection.recv(8 - len(head))
                    if not part:
                        break
                    head += part
            except OSError:
                head = b""
            finally:
                connection.close()
            if head != b"SGFIPD01":
                continue
            with self.lock:
                busy = self.thread is not None and self.thread.is_alive()
            if busy:
                continue
            listener.close()
            try:
                self.start(options=getattr(self, "options", None) or {})
                self.update(standby="session_started", standby_peer=str(address[0]))
            except ValueError:
                pass
            deadline = time.monotonic() + 120
            while not self.shutdown.is_set():
                with self.lock:
                    thread = self.thread
                    session_dir = self.session
                if thread is None or not thread.is_alive():
                    with self.lock:
                        if self.state.get("status") == "completed" and self.session is not None:
                            self.session = None
                            self.thread = None
                            self.state["status"] = "idle"
                    break
                cap = read_json(session_dir / "capture_status.json") if session_dir else {}
                if (cap.get("status") == "starting" and not cap.get("frames")
                        and time.monotonic() > deadline):
                    self.cancel()
                    time.sleep(20)  # cool down: ignore reconnect spam from a stray app
                    break
                time.sleep(1.0)
            listener = None
            for _ in range(30):
                if self.shutdown.is_set():
                    return
                try:
                    listener = self._standby_listen()
                    self.update(standby="listening")
                    break
                except OSError:
                    time.sleep(2)
            if listener is None:
                self.update(standby_error="待命端口重启失败: 端口被长期占用")
                return


    def run(self):
        preview = None
        try:
            arguments = ["--session", self.session]
            if self.args.replay:
                arguments += ["--replay", self.args.replay, "--fps", self.args.fps,
                              "--max-frames", self.args.max_frames]
            if self.upload_dir is not None:
                arguments += ["--packet-dir", self.upload_dir]
            elif self.args.ipad_port:
                arguments += ["--ipad-port", self.args.ipad_port, "--ipad-bind", self.args.ipad_bind]
            elif self.args.wireless_host:
                arguments += ["--wireless-host", self.args.wireless_host, "--wireless-port", self.args.wireless_port,
                              "--depth-denoise", self.args.depth_denoise, "--max-frames", self.args.max_frames]
            camera = self.launch(self.args.capture_python, "pose_pipeline.device_capture", arguments, "capture")
            # Wait for one real RGB-D frame before allocating GPU memory.
            self.wait_first_frame(camera)
            if camera.poll() is None and not self.cancelled.is_set() and self.upload_dir is None:
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



def pipeline_path(root):
    root = Path(root).resolve()
    path = Path(read_json(root / 'session.json').get('pipeline', root / 'pipeline')).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Pipeline escaped its session')
    return path


def completed_result(root):
    root = Path(root).resolve()
    state = read_json(root / 'session.json')
    if state.get('status') != 'completed':
        raise ValueError('Session is not completed')
    pipeline = pipeline_path(root)
    if state.get('attempt'):
        from .artifacts import load_artifacts
        load_artifacts(pipeline, require_provenance=True)
    result = read_json(pipeline / 'GUI_RESULT.json')
    for key in ('final_cloud', 'classes', 'trajectory'):
        path = Path(result[key]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Invalid completed artifact')
    return result


def instances_body(session_dir):
    result = read_json(pipeline_path(session_dir) / "GUI_RESULT.json")
    if not result:
        return []
    names = read_json(pipeline_path(session_dir) / "fused/instance_names.json")
    objects = {o["instance_id"]: o for o in read_json(
        pipeline_path(session_dir) / "refinement/refined/capture/objects.json")}
    rows = []
    for n in names:
        o = objects.get(n["instance_id"], {})
        rows.append({
            "instance_id": n["instance_id"],
            "semantic_name": o.get("semantic_name", "?"),
            "vlm_name": n.get("vlm_name"),
            "point_count": n.get("point_count", 0),
        })
    rows.sort(key=lambda r: -r["point_count"])
    return rows


def _present_classes(session_dir):
    """Legend entries for classes that exist in this map: {id: 'name · count'}."""
    result = read_json(pipeline_path(session_dir) / "GUI_RESULT.json")
    if not result:
        return {}
    classes = read_json(result.get("classes")) or {}
    try:
        from plyfile import PlyData
        import numpy as np
        v = PlyData.read(result["final_cloud"])["vertex"].data
        ids, counts = np.unique(v["semantic_id"], return_counts=True)
    except Exception:
        return {}
    out = {}
    for value, count in sorted(zip(ids.tolist(), counts.tolist()), key=lambda t: -t[1]):
        name = classes.get(str(value))
        if name and count >= 50:
            out[str(value)] = f"{name} · {count:,}"
    return out


def _pipeline_timeline(root, cap):
    """Full-stage durations (seconds) derived from artifact timestamps."""
    start = (root / "session.json").stat().st_mtime if (root / "session.json").exists() else None
    marks = [
        ("capture", root / "capture_status.json"),
        ("mapping", pipeline_path(root) / "mapping.log"),
        ("sam3", pipeline_path(root) / "sam3.log"),
        ("vlm", pipeline_path(root) / "vlm.log"),
        ("backfill", pipeline_path(root) / "backfill.log"),
        ("done", pipeline_path(root) / "GUI_RESULT.json"),
    ]
    events = [(name, p.stat().st_mtime) for name, p in marks if p.exists()]
    out = []
    for i, (name, t) in enumerate(events):
        if i == 0:
            duration = max(0.0, t - start) if start else (cap.get("elapsed_s") or 0)
        else:
            duration = max(0.0, t - events[i-1][1])
        out.append({"stage": name, "seconds": round(duration, 1)})
    return out


def _safe_session_id(sid):
    core = sid[len("scan_"):] if sid.startswith("scan_") else ""
    parts = core.split("_")
    return len(parts) == 3 and all(p.isalnum() for p in parts)


def _session_summary(root, current_id):
    cap = read_json(root / "capture_status.json")
    sess = read_json(root / "session.json")
    cloud = read_json(root / "cloud.json")
    if root.name == current_id and sess.get("status") in {
            "starting", "recording", "stopping", "mapping", "cancelling"}:
        status = "active"
    elif sess.get("status") == "completed" and (pipeline_path(root) / "GUI_RESULT.json").exists():
        status = "completed"
    elif sess.get("status") in {"failed", "interrupted", "cancelled"}:
        status = sess["status"]
    elif cap.get("status") == "failed":
        status = "failed"
    elif (pipeline_path(root) / "mapping_result.json").exists():
        status = "incomplete"
    elif cap.get("status") == "sealed":
        status = "captured"
    elif cap.get("status") in ("starting", "cancelling", "cancelled"):
        status = "cancelled"
    else:
        status = cap.get("status", "unknown")
    return {
        "id": root.name,
        "mode": sess.get("mode", "camera"),
        "display_name": sess.get("display_name"),
        "status": status,
        "frames": cap.get("frames", 0),
        "points": cloud.get("points", 0),
        "cloud_kind": cloud.get("kind", ""),
        "modified": root.stat().st_mtime,
    }


def _historical_status(root):
    cap = read_json(root / "capture_status.json")
    sess = read_json(root / "session.json")
    cloud = read_json(root / "cloud.json")
    stage = read_json(pipeline_path(root) / "GUI_STAGE.json")
    if sess.get("status") == "completed" and (pipeline_path(root) / "GUI_RESULT.json").exists():
        status, semantic_stage = "completed", "completed"
    elif cap.get("status") == "failed":
        status, semantic_stage = "failed", ""
    elif (pipeline_path(root) / "mapping_result.json").exists():
        status, semantic_stage = "mapping", stage.get("stage", "")
    elif cap.get("status") == "sealed":
        status, semantic_stage = "captured", ""
    else:
        status, semantic_stage = cap.get("status", "unknown"), ""
    return {
        "status": status,
        "mode": sess.get("mode", "camera"),
        "session": str(root),
        "options": sess.get("options", {}),
        "capture": cap,
        "preview": {},
        "mapping": read_json(pipeline_path(root) / "mapping_result.json"),
        "cloud": cloud,
        "semantic_stage": semantic_stage,
        "view": read_json(root / "view.json").get("view", "semantic_id"),
        "timeline": [] if sess.get("mode") == "demo" else _pipeline_timeline(root, cap),
        "historical": True,
    }


def host_allowed(host_header):
    """Loopback plus private/Tailscale (CGNAT) IP literals; reject public DNS names."""
    try:
        name = urlparse("http://" + host_header).hostname or ""
    except ValueError:
        return False
    if name in {"127.0.0.1", "localhost", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or (
        ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10")
    )


def handler(controller, token, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, data, mime="application/json", filename=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", 'attachment; filename="' + filename + '"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not host_allowed(self.headers.get("Host", "")):
                return self.respond(403, {"error": "loopback or private-network host required"})
            path = urlparse(self.path).path
            from . import scene_http
            if scene_http.get(self, controller, path):
                return
            if path == "/":
                html = Path(__file__).with_name("device_gui.html").read_text()
                return self.respond(200, html.replace("__TOKEN__", token).encode(), "text/html; charset=utf-8")
            if path == "/api/instances":
                with controller.lock:
                    active = controller.session
                    completed = controller.state.get("status") == "completed"
                if active and completed:
                    return self.respond(200, instances_body(active))
                return self.respond(200, [])
            if path == "/api/classes":
                state = controller.snapshot()
                if state["status"] != "completed":
                    return self.respond(200, {})
                with controller.lock:
                    active = controller.session
                return self.respond(200, _present_classes(active) if active else {})
            if path == "/api/models":
                from .semantic_runtime.common import registry
                cfg = read_json(controller.args.runtime)
                models = [k for k, v in registry().items() if v["kind"] != "ocr" and (k == "none" or k in cfg.get("models", {}))]
                return self.respond(200, {"models": models})
            if path == "/api/status":
                return self.respond(200, controller.snapshot())
            if path in ("/camera-preview/color.jpg", "/camera-preview/depth.jpg"):
                data = controller.live_camera.image(Path(path).stem) if controller.live_camera else None
                return self.respond(200, data, "image/jpeg") if data else self.respond(404, {"error": "preview not ready"})
            if path == "/api/sessions":
                with controller.lock:
                    current_id = controller.session.name if controller.session else ""
                items = [_session_summary(child, current_id)
                         for child in sorted((p for library in controller.library_roots() for p in library.glob("scan_*")), reverse=True)
                         if child.is_dir()]
                return self.respond(200, {"current": current_id, "sessions": items})
            if path.startswith("/s/"):
                parts = path.split("/")
                sid = parts[2] if len(parts) > 2 else ""
                rest = "/".join(parts[3:])
                if not _safe_session_id(sid) or ".." in path:
                    return self.respond(404, {"error": "bad session id"})
                try:
                    root = controller.resolve_session(sid)
                except (ValueError, OSError):
                    return self.respond(404, {"error": "unknown session"})
                if not root.is_dir():
                    return self.respond(404, {"error": "unknown session"})
                if rest == "api/status":
                    with controller.lock:
                        is_current = controller.session is not None and controller.session.name == sid
                    if is_current:
                        return self.respond(200, controller.snapshot())
                    return self.respond(200, _historical_status(root))
                if rest in ("api/objects", "objects.bin", "object.ply"):
                    from .live_objects import load_objects, export_object
                    try:
                        query = parse_qs(urlparse(self.path).query)
                        if rest == "object.ply":
                            blob = export_object(root, query.get("kind", [""])[0], query.get("id", [""])[0])
                            return self.respond(200, blob, "application/octet-stream")
                        bundle = load_objects(root)
                        if rest == "api/objects":
                            return self.respond(200, bundle["catalog"])
                        if query.get("revision", [bundle["catalog"]["revision"]])[0] != bundle["catalog"]["revision"]:
                            return self.respond(409, {"error": "Point cloud changed; reload object catalog"})
                        return self.respond(200, bundle["data"].tobytes(), "application/octet-stream")
                    except FileNotFoundError:
                        return self.respond(404, {"error": "Completed object data not found"})
                    except (ValueError, KeyError, OSError):
                        return self.respond(422, {"error": "Object data unavailable or invalid selection"})
                if rest == "api/instances":
                    return self.respond(200, instances_body(root))
                if rest == "api/classes":
                    return self.respond(200, _present_classes(root))
                if rest == "cloud.bin":
                    try:
                        blob = (root / read_json(root / "cloud.json")["file"]).read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(len(blob)))
                        self.send_header("Cache-Control", "private, max-age=86400")
                        self.end_headers()
                        try:
                            self.wfile.write(blob)
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        return
                    except (FileNotFoundError, KeyError):
                        return self.respond(404, {"error": "cloud not generated"})
                if rest in ("color.jpg", "depth.jpg"):
                    try:
                        return self.respond(200, (root / rest).read_bytes(), "image/jpeg")
                    except FileNotFoundError:
                        return self.respond(404, {"error": "no image"})
                if rest == "final.ply":
                    try:
                        result = completed_result(root)
                        final = Path(result["final_cloud"]).resolve()
                        final.relative_to(root.resolve())
                        blob = final.read_bytes()
                        return self.respond(200, blob, "application/octet-stream")
                    except (FileNotFoundError, KeyError, ValueError):
                        return self.respond(404, {"error": "not completed"})
                if rest == "trajectory.json":
                    try:
                        blob = Path(completed_result(root)['trajectory']).read_bytes()
                        return self.respond(200, blob, "application/json")
                    except (OSError, KeyError, ValueError):
                        return self.respond(404, {"error": "not completed"})
                return self.respond(404, {"error": "unknown session route"})
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
            if not host_allowed(self.headers.get("Host", "")):
                return self.respond(403, {"error": "loopback or private-network host required"})
            if self.headers.get("X-Scan-Token") != token:
                return self.respond(403, {"error": "invalid session token"})
            path = urlparse(self.path).path
            from . import scene_http
            if scene_http.post(self, controller, path):
                return
            if path == "/api/camera-preview":
                return self.respond(200, controller.camera_preview())
            if path == "/api/uploads" or path.startswith("/api/uploads/"):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    self.connection.settimeout(60)
                    if path == "/api/uploads":
                        if not 0 < length <= MAX_MANIFEST:
                            raise ValueError("Invalid manifest size")
                        return self.respond(200, controller.uploads.prepare(json.loads(self.rfile.read(length))))
                    parts = path.split("/")
                    if len(parts) == 6 and parts[4] == "frames":
                        result = controller.uploads.put(parts[3], int(parts[5]), self.rfile, length)
                        return self.respond(200, result)
                    if len(parts) == 5 and parts[4] == "complete" and length == 0:
                        return self.respond(200, controller.complete_upload(parts[3]))
                    raise ValueError("Invalid upload route")
                except (ValueError, OSError, KeyError, TypeError) as error:
                    return self.respond(409, {"error": str(error)})
            if path.startswith("/s/") and path.endswith("/api/view"):
                parts = path.split("/")
                sid = parts[2] if len(parts) > 2 else ""
                if not _safe_session_id(sid):
                    return self.respond(404, {"error": "bad session id"})
                try:
                    root = controller.resolve_session(sid)
                except (ValueError, OSError):
                    return self.respond(404, {"error": "unknown session"})
                try:
                    result = completed_result(root)
                except (ValueError, OSError, KeyError):
                    return self.respond(409, {"error": "Completed session required"})
                if not root.is_dir() or not result:
                    return self.respond(404, {"error": "session result unavailable"})
                length = int(self.headers.get("Content-Length", 0) or 0)
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    payload = {}
                mode = str(payload.get("mode", "semantic_id"))
                if mode not in ("semantic_id", "instance_id", "rgb"):
                    return self.respond(400, {"error": "invalid view mode"})
                from plyfile import PlyData
                import numpy as np
                v = PlyData.read(result["final_cloud"])["vertex"].data
                xyz = np.column_stack([v[k] for k in ("x", "y", "z")])
                if mode == "rgb":
                    raw = PlyData.read(result["raw_map"])["vertex"].data
                    if len(raw) != len(v) or any(not np.array_equal(raw[k], v[k]) for k in ("x", "y", "z")):
                        return self.respond(409, {"error": "RGB geometry does not match the completed map"})
                    rgb = np.column_stack([raw[k] for k in ("red", "green", "blue")]) / 255.
                else:
                    ids = v[mode].astype(np.int64)
                    rgb = np.column_stack([np.where(ids > 0, 50 + ids*m % 206, 90) for m in (73, 151, 199)]) / 255.
                publish_cloud(root, np.column_stack((xyz, rgb)), kind="final", revision=time.time_ns())
                atomic_json(root / "view.json", {"view": mode})
                return self.respond(200, {"view": mode})
            if path == "/api/options":
                length = int(self.headers.get("Content-Length", 0) or 0)
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    payload = {}
                model = str(payload.get("vlm", ""))
                if not model:
                    return self.respond(400, {"error": "vlm is required"})
                merged = dict(getattr(controller, "options", None) or {})
                merged["vlm"] = model
                merged.setdefault("schedule", "serial")
                merged.setdefault("refine", True)
                # Validate exactly like start() so bad models fail here, not mid-scan.
                from .semantic_runtime.common import registry, validate_runtime
                if model not in registry() or registry()[model]["kind"] == "ocr"                         or merged["schedule"] not in ("serial", "parallel"):
                    return self.respond(400, {"error": "Invalid model or schedule"})
                validate_runtime(read_json(controller.args.runtime), model)
                if merged["refine"]:
                    validate_runtime(read_json(controller.args.runtime), "qwen3vl_2b_nf4", raw_mapping=False)
                controller.options = merged
                return self.respond(200, {"options": merged})
            if path == "/api/session/delete":
                length = int(self.headers.get("Content-Length", 0) or 0)
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    payload = {}
                sid = str(payload.get("id", ""))
                if not _safe_session_id(sid):
                    return self.respond(404, {"error": "bad session id"})
                with controller.lock:
                    if controller.session is not None and controller.session.name == sid:
                        return self.respond(409, {"error": "session is active"})
                try:
                    root = controller.resolve_session(sid)
                except (ValueError, OSError):
                    return self.respond(404, {"error": "unknown session"})
                if not root.is_dir():
                    return self.respond(404, {"error": "unknown session"})
                if read_json(root / 'session.json').get('status') in ACTIVE:
                    return self.respond(409, {"error": "session is active or requires recovery"})
                shutil.rmtree(root)
                return self.respond(200, {"deleted": sid})
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
            except (ValueError, OSError, KeyError, RuntimeError) as error:
                self.respond(409, {"error": str(error)})
    return Handler


def main():
    p = argparse.ArgumentParser(description="ReVeMap unified device capture and scene query GUI")
    p.add_argument("--devices", action="store_true")
    p.add_argument("--library-root", type=Path, action="append", default=[])
    p.add_argument("--first-frame-timeout", type=float, default=30.)
    p.add_argument("--capture-frame-timeout", type=float, default=15.)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--provider-root", type=Path, required=True)
    p.add_argument("--gpu-python", type=Path, required=True)
    p.add_argument("--cpu-python", type=Path, required=True)
    p.add_argument("--capture-python", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; use 0.0.0.0 to serve the tailnet/LAN")
    p.add_argument("--replay", type=Path)
    p.add_argument("--wireless-host", help="Orbbec host receiving EtherSensePing")
    p.add_argument("--wireless-port", type=int, default=1024, help="UDP request and reverse TCP port")
    p.add_argument("--depth-denoise", choices=("off", "bilateral"), default="bilateral",
                   help="Wireless depth filter, including idle preview; raw recording retained")
    p.add_argument("--ipad-port", type=int, default=0,
                   help="live iPad LiDAR capture: TCP port for the Scan app stream")
    p.add_argument("--ipad-bind", default="0.0.0.0")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--max-frames", type=int, default=0, help="Optional replay or wireless frame limit")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    if not 0 < args.first_frame_timeout < float('inf') or not 0 < args.capture_frame_timeout < float('inf'):
        p.error('Frame timeouts must be finite and positive')
    if args.fps <= 0 or args.max_frames < 0 or (args.max_frames and not (args.replay or args.wireless_host)):
        p.error("fps must be positive; max-frames requires replay or wireless capture")
    if not 1 <= args.wireless_port <= 65535 or sum(bool(v) for v in (args.replay, args.ipad_port, args.wireless_host)) > 1:
        p.error("Choose one input source and a wireless-port in 1..65535")
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
    server = ThreadingHTTPServer((args.host, args.port), handler(controller, secrets.token_hex(24), args.port))
    if args.ipad_port and not args.replay:
        threading.Thread(target=controller.ipad_standby_loop, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Scan GUI: {url}\nBase: developnew@{BASE_COMMIT[:7]}\nOutputs: {args.output}", flush=True)
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
        controller.shutdown.set()
        if controller.live_camera:
            controller.live_camera.stop()
        controller.cancel()
        if controller.thread:
            controller.thread.join(timeout=45)
        server.server_close()



if __name__ == "__main__":
    main()
