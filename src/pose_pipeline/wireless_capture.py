"""Receive the existing EtherSense UDP-request/reverse-TCP RGB-D protocol."""
import io
import json
import pickle
import socket
import struct
import time

import numpy as np


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 writes _core module paths; NumPy 1 uses core for the same arrays.
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_packet(raw):
    return _ArrayUnpickler(io.BytesIO(raw)).load()


def decode_packet(payload):
    """Validate measured metadata before writing a pinhole RGB-D recording."""
    if not isinstance(payload, dict):
        raise ValueError("Wireless capture requires RGB-D with calibration metadata")
    color, depth = payload["color"], payload["depth"]
    if not isinstance(color, np.ndarray) or color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("Wireless color must be a uint8 BGR image")
    if not isinstance(depth, np.ndarray) or depth.dtype not in (np.uint16, np.float32) or depth.shape != color.shape[:2]:
        raise ValueError("Wireless depth must be aligned uint16/float32 millimetres")
    if (payload["height"], payload["width"]) != depth.shape:
        raise ValueError("Wireless image and metadata dimensions differ")
    if payload["depth_scale"] != 1000 or payload["alignment"] != "depth_to_color":
        raise ValueError("Wireless capture requires depth-to-color alignment and millimetres")
    intrinsic = tuple(float(v) for v in payload["intrinsics"])
    if len(intrinsic) != 4 or not np.isfinite(intrinsic).all() or min(intrinsic[:2]) <= 0:
        raise ValueError("Invalid wireless camera intrinsics")
    tc, td = payload["timestamp_us"], payload["depth_timestamp_us"]
    if not isinstance(tc, int) or not isinstance(td, int) or min(tc, td) <= 0 or abs(tc-td) > 15000:
        raise ValueError("Wireless RGB-D timestamps must be positive sensor microseconds within 15 ms")
    for key in ("sensor_frame_id", "depth_frame_id"):
        if not isinstance(payload[key], int) or payload[key] < 0:
            raise ValueError("Invalid wireless sensor frame index")
    if not np.isfinite(depth).all() or (depth < 0).any() or (depth > 65535).any():
        raise ValueError("Wireless depth cannot be represented as millimetre uint16")
    depth = np.rint(depth).astype(np.uint16) if depth.dtype != np.uint16 else depth
    reserved = {"color", "depth", "intrinsics", "timestamp_us", "color_path", "depth_path", "frame_id", "rotate_ccw"}
    audit = {key: value for key, value in payload.items() if key not in reserved}
    try:
        json.dumps(audit, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Wireless audit metadata must contain finite JSON values") from error
    return color, depth, intrinsic, tc, audit


def _receive(connection, size, idle_timeout):
    data = bytearray()
    last = time.monotonic()
    while len(data) < size:
        try:
            block = connection.recv(min(size-len(data), 1024*1024))
        except socket.timeout:
            if time.monotonic()-last > idle_timeout:
                raise TimeoutError("Wireless RGB-D stream stopped sending data")
            yield None
            continue
        if not block:
            raise ConnectionError("Wireless RGB-D connection closed before capture was stopped")
        data.extend(block)
        last = time.monotonic()
    return data


def wireless_frames(host, port=1024, timeout=0.5, idle_timeout=15.0):
    """Listen before requesting; timeout yields let capture respond to its stop flag."""
    address = socket.gethostbyname(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("", port))
        listener.listen(1)
        listener.settimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as request:
            request.sendto(b"EtherSensePing", (address, port))
        started = time.monotonic()
        last_request = started
        while True:
            try:
                connection, peer = listener.accept()
            except socket.timeout:
                if time.monotonic()-started > idle_timeout:
                    raise TimeoutError("Wireless camera did not connect after the UDP request")
                # The preceding preview connection may still be releasing the camera.
                if time.monotonic()-last_request >= 1:
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as request:
                        request.sendto(b"EtherSensePing", (address, port))
                    last_request = time.monotonic()
                yield None
                continue
            if peer[0] == address:
                break
            connection.close()
        with connection:
            connection.settimeout(timeout)
            while True:
                header = yield from _receive(connection, 12, idle_timeout)
                size, _ = struct.unpack("<Id", header)
                if not 0 < size <= 64*1024*1024:
                    raise ValueError("Wireless payload length is outside 1..64 MiB")
                raw = yield from _receive(connection, size, idle_timeout)
                yield decode_packet(load_packet(raw))
