"""Synthetic loopback protocol tests only; no camera or GPU is contacted."""
from contextlib import contextmanager
import json
from pathlib import Path
import pickle
import signal
import socket
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from pose_pipeline.contracts import load_manifest
from pose_pipeline.device_capture import capture
from pose_pipeline.wireless_capture import decode_packet, wireless_frames


def packet(timestamp=1000000):
    return dict(color=np.full((3, 4, 3), (2, 40, 200), np.uint8),
                depth=np.full((3, 4), 1250.4, np.float32), intrinsics=[3., 3., 1.5, 1.],
                timestamp_us=timestamp, depth_timestamp_us=timestamp+100,
                sensor_frame_id=7, depth_frame_id=7, depth_scale=1000,
                alignment="depth_to_color", width=4, height=3)


def wire(payload):
    data = pickle.dumps(payload, protocol=4)
    return struct.pack("<Id", len(data), payload["timestamp_us"]/1000.)+data


@contextmanager
def camera(chunks, *, stay_open=False, drop_requests=0):
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    port = udp.getsockname()[1]
    errors, closed = [], threading.Event()
    def serve():
        try:
            udp.settimeout(5)
            for _ in range(drop_requests+1):
                message, peer = udp.recvfrom(1024)
                assert message == b"EtherSensePing"
            with socket.create_connection((peer[0], port), timeout=5) as connection:
                for part in chunks:
                    connection.sendall(part)
                    time.sleep(.005)
                if stay_open:
                    while connection.recv(1024):
                        pass
        except Exception as error:
            errors.append(error)
        finally:
            closed.set()
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield port, closed
    finally:
        thread.join(6)
        udp.close()
        if thread.is_alive():
            raise AssertionError("Fake camera was not closed")
        if errors:
            raise errors[0]


class WirelessCaptureTest(unittest.TestCase):
    def test_metadata_and_millimetres(self):
        color, depth, intrinsic, timestamp, audit = decode_packet(packet())
        self.assertEqual(depth.dtype, np.uint16)
        self.assertTrue((depth == 1250).all())
        self.assertEqual(tuple(color[0, 0]), (2, 40, 200))
        self.assertEqual(intrinsic, (3., 3., 1.5, 1.))
        self.assertEqual(timestamp, 1000000)
        self.assertEqual(audit["sensor_frame_id"], 7)
        self.assertNotIn("color", audit)
        self.assertNotIn("depth", audit)
        json.dumps(audit, allow_nan=False)

    def test_reject_invalid_metadata(self):
        for key, value in [("timestamp_us", 0), ("depth_timestamp_us", 2000000),
                           ("depth_scale", 1), ("alignment", "none"),
                           ("width", 8), ("intrinsics", [0., 3., 1., 1.]),
                           ("sensor_frame_id", -1), ("depth", np.full((3,4), np.nan, np.float32)),
                           ("extra_array", np.zeros(1))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode_packet(dict(packet(), **{key:value}))

    def test_fragmented_header_and_body(self):
        data = wire(packet())
        with camera([data[:3], data[3:11], data[11:30], data[30:]], stay_open=True) as (port, _):
            source = wireless_frames("127.0.0.1", port, timeout=.01)
            try:
                sample = next(value for value in source if value is not None)
                self.assertEqual(sample[3], 1000000)
            finally:
                source.close()

    def test_numpy2_sender_packet_on_receiver(self):
        raw = Path(__file__).with_name("synthetic_numpy2_packet.pkl").read_bytes()
        self.assertIn(b"numpy._core", raw)
        data = struct.pack("<Id", len(raw), 1000.)+raw
        with camera([data], stay_open=True) as (port, _):
            source = wireless_frames("127.0.0.1", port, timeout=.03)
            try:
                sample = next(value for value in source if value is not None)
                self.assertEqual(sample[3], 1000000)
                self.assertEqual(tuple(sample[0][0, 0]), (2, 40, 200))
                self.assertEqual(sample[1].dtype, np.uint16)
            finally:
                source.close()

    def test_retries_udp_after_first_request_is_dropped(self):
        with camera([wire(packet())], stay_open=True, drop_requests=1) as (port, closed):
            source = wireless_frames("127.0.0.1", port, timeout=.03, idle_timeout=3)
            try:
                sample = next(value for value in source if value is not None)
                self.assertEqual(sample[3], 1000000)
                self.assertEqual(tuple(sample[0][0, 0]), (2, 40, 200))
            finally:
                source.close()
            self.assertTrue(closed.wait(1))

    def test_stop_during_incomplete_header(self):
        with camera([wire(packet())[:4]], stay_open=True) as (port, closed):
            source = wireless_frames("127.0.0.1", port, timeout=.03)
            self.assertIsNone(next(source))
            source.close()
            self.assertTrue(closed.wait(1))

    def test_disconnect_and_oversized_payload(self):
        for data, expected in [(b"abc", ConnectionError), (struct.pack("<Id", 64*1024*1024+1, 1.), ValueError)]:
            with camera([data]) as (port, _):
                with self.assertRaises(expected):
                    list(wireless_frames("127.0.0.1", port, timeout=.1))

    def test_invalid_metadata_closes_connection(self):
        with camera([wire(dict(packet(), depth_scale=1))], stay_open=True) as (port, closed):
            with self.assertRaises(ValueError):
                list(wireless_frames("127.0.0.1", port, timeout=.1))
            self.assertTrue(closed.wait(1))

    def test_capture_seals_and_stops_at_frame_limit(self):
        with tempfile.TemporaryDirectory(prefix="wireless_synthetic_") as directory:
            with camera([wire(packet())], stay_open=True) as (port, _):
                saved = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
                try:
                    capture(SimpleNamespace(session=Path(directory), replay=None, packet_dir=None,
                        ipad_port=0, wireless_host="127.0.0.1", wireless_port=port, max_frames=1))
                finally:
                    for sig, handler in saved.items():
                        signal.signal(sig, handler)
            manifest = load_manifest(Path(directory)/"capture/manifest.json")
            self.assertEqual(len(manifest.frames), 1)
            self.assertEqual(manifest.source, "wireless_orbbec_d2c_sensor_timestamps_masked_bilateral_v1")
            self.assertEqual(manifest.depth_scale, 1000)
            self.assertEqual(cv2.imread(str(manifest.frames[0].depth_path), -1).dtype, np.uint16)
            status = json.loads((Path(directory)/"capture_status.json").read_text())
            self.assertEqual(status["status"], "sealed")


if __name__ == "__main__":
    unittest.main()
