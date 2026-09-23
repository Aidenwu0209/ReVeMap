"""Leased, CPU-only wireless camera preview; keep the latest JPEG pair in RAM."""
import threading
import time

import cv2

from .wireless_capture import wireless_frames
from .depth_denoise import denoise_depth


class WirelessPreview:
    LEASE_SECONDS = 8.0
    RETRY_SECONDS = 5.0
    ENCODE_INTERVAL = 1/12

    def __init__(self, host, port=1024, *, depth_denoise="bilateral"):
        self.host, self.port = host, port
        if depth_denoise not in ("off", "bilateral"):
            raise ValueError("depth_denoise must be off or bilateral")
        self.depth_denoise = depth_denoise
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._stopping = False
        self._last_touch = self._retry_after = 0.0
        self._images = {}
        self._state = {"status": "stopped", "sequence": 0, "updated_at": None,
                       "timestamp_us": None, "depth_timestamp_us": None}

    def touch(self):
        """Renew the viewing lease; one worker owns the reverse TCP connection."""
        with self._lock:
            now = self._last_touch = time.monotonic()
            if (not self._stopping and now >= self._retry_after and
                    (self._thread is None or not self._thread.is_alive())):
                self._stop.clear()
                self._state.update(status="starting")
                self._state.pop("error", None)
                self._thread = threading.Thread(target=self._run, name="wireless-preview", daemon=True)
                try:
                    self._thread.start()
                except Exception as error:
                    self._thread = None
                    self._state.update(status="failed", error=str(error))
                    self._retry_after = now+self.RETRY_SECONDS
            return dict(self._state)

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def image(self, name):
        with self._lock:
            return self._images.get(name)

    def stop(self):
        """Release the camera before recording; failure prevents a second owner."""
        with self._lock:
            self._stopping = True
            self._stop.set()
            worker = self._thread
        if worker is not None:
            worker.join(timeout=3.0)
        with self._lock:
            if worker is not None and worker.is_alive():
                self._state.update(status="failed", error="Wireless preview did not stop")
                raise RuntimeError("Wireless preview did not stop; camera handoff cancelled")
            if self._thread is worker:
                self._thread = None
            self._last_touch = 0.0
            self._state["status"] = "stopped"
            self._stopping = False

    def _leased(self):
        with self._lock:
            return not self._stop.is_set() and time.monotonic()-self._last_touch < self.LEASE_SECONDS

    def _run(self):
        source, error, encoded_at = None, None, 0.0
        try:
            source = wireless_frames(self.host, self.port, timeout=0.3)
            while self._leased():
                sample = next(source)
                if not self._leased():
                    break
                now = time.monotonic()
                if sample is None or now-encoded_at < self.ENCODE_INTERVAL:
                    continue
                color, depth, _, timestamp, audit = sample
                depth, processing = denoise_depth(depth, mode=self.depth_denoise)
                depth_view = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255/4500), cv2.COLORMAP_TURBO)
                images = {}
                for name, picture in (("color", color), ("depth", depth_view)):
                    ok, data = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError("Wireless preview JPEG encoding failed")
                    images[name] = data.tobytes()
                encoded_at = now
                with self._lock:
                    self._images = images
                    self._state.update(status="live", sequence=max(self._state["sequence"]+1, time.time_ns()),
                        updated_at=time.time(), timestamp_us=timestamp,
                        depth_timestamp_us=audit.get("depth_timestamp_us"),
                        sensor_frame_id=audit.get("sensor_frame_id"),
                        depth_processing=processing,
                        width=color.shape[1], height=color.shape[0])
        except Exception as caught:
            error = str(caught) or "Wireless preview stream ended"
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as caught:
                    error = error or str(caught)
            with self._lock:
                if error:
                    self._state.update(status="failed", error=error)
                    self._retry_after = time.monotonic()+self.RETRY_SECONDS
                else:
                    self._state["status"] = "stopped"
