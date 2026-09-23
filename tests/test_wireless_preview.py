"""In-memory fake sources only: no camera, socket, dataset, or GPU work."""
from contextlib import ExitStack
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pose_pipeline import wireless_preview as module


def wait_for(condition, timeout=1.0):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for preview state")


class Source:
    def __init__(self, empty=False, fail=False):
        self.empty, self.fail = empty, fail
        self.starts = 0
        self.closed = threading.Event()

    def __call__(self, host, port, timeout):
        self.starts += 1
        try:
            if self.fail:
                raise RuntimeError("Synthetic camera failure")
            while True:
                time.sleep(0.003)
                if self.empty:
                    yield None
                else:
                    yield (np.full((12, 16, 3), (10, 60, 180), np.uint8),
                           np.full((12, 16), 1200, np.uint16), (10, 10, 8, 6),
                           123456, {"depth_timestamp_us": 123460, "sensor_frame_id": 9})
        finally:
            self.closed.set()


class WirelessPreviewTest(unittest.TestCase):
    def test_ram_images_metadata_and_stop_release(self):
        source = Source()
        preview = module.WirelessPreview("unused.invalid", 1234)
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "wireless_frames", source))
            for target in ("builtins.open", "os.open", "pathlib.Path.open", "pathlib.Path.mkdir", "cv2.imwrite"):
                stack.enter_context(patch(target, side_effect=AssertionError("Preview attempted filesystem access")))
            try:
                preview.touch()
                wait_for(lambda: preview.snapshot()["status"] == "live")
                state = preview.snapshot()
                self.assertEqual((state["width"], state["height"]), (16, 12))
                self.assertEqual(state["timestamp_us"], 123456)
                self.assertEqual(state["depth_timestamp_us"], 123460)
                self.assertEqual(state["sensor_frame_id"], 9)
                for name in ("color", "depth"):
                    image = cv2.imdecode(np.frombuffer(preview.image(name), np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(image.shape, (12, 16, 3))
                self.assertIsNone(preview.image("other"))
            finally:
                preview.stop()
                preview.stop()
        self.assertTrue(source.closed.is_set())
        self.assertEqual(preview.snapshot()["status"], "stopped")

    def test_concurrent_touch_uses_one_worker_and_restart_revision_advances(self):
        source = Source()
        preview = module.WirelessPreview("unused.invalid")
        with patch.object(module, "wireless_frames", source):
            try:
                callers = [threading.Thread(target=preview.touch) for _ in range(12)]
                for caller in callers:
                    caller.start()
                for caller in callers:
                    caller.join()
                wait_for(lambda: preview.snapshot()["status"] == "live")
                self.assertEqual(source.starts, 1)
                old = preview.snapshot()["sequence"]
                preview.stop()
                preview.touch()
                wait_for(lambda: preview.snapshot()["status"] == "live")
                self.assertEqual(source.starts, 2)
                self.assertGreater(preview.snapshot()["sequence"], old)
            finally:
                preview.stop()

    def test_none_samples_obey_lease_and_release(self):
        source = Source(empty=True)
        preview = module.WirelessPreview("unused.invalid")
        preview.LEASE_SECONDS = 0.05
        with patch.object(module, "wireless_frames", source):
            try:
                preview.touch()
                for _ in range(4):
                    time.sleep(0.02)
                    preview.touch()
                self.assertEqual(source.starts, 1)
                self.assertFalse(source.closed.is_set())
                wait_for(source.closed.is_set)
                self.assertEqual(preview.snapshot()["status"], "stopped")
                self.assertIsNone(preview.image("color"))
            finally:
                preview.stop()

    def test_failed_connection_has_retry_cooldown(self):
        source = Source(fail=True)
        preview = module.WirelessPreview("unused.invalid")
        preview.RETRY_SECONDS = 0.08
        with patch.object(module, "wireless_frames", source):
            try:
                preview.touch()
                wait_for(lambda: preview.snapshot()["status"] == "failed")
                for _ in range(10):
                    preview.touch()
                self.assertEqual(source.starts, 1)
                time.sleep(0.1)
                preview.touch()
                wait_for(lambda: source.starts == 2)
            finally:
                preview.stop()

    def test_encoding_rate_is_bounded(self):
        source, encoded_at = Source(), []
        preview = module.WirelessPreview("unused.invalid")
        encode = cv2.imencode
        def track_encode(*args):
            encoded_at.append(time.monotonic())
            return encode(*args)
        with patch.object(module, "wireless_frames", source), patch.object(module.cv2, "imencode", track_encode):
            try:
                preview.touch()
                wait_for(lambda: len(encoded_at) >= 6)
            finally:
                preview.stop()
        color_times = encoded_at[::2]
        self.assertTrue(all(b-a >= preview.ENCODE_INTERVAL*0.9
                            for a, b in zip(color_times, color_times[1:])))

    def test_stop_timeout_prevents_a_second_owner(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        starts = []
        def blocked_source(*args, **kwargs):
            starts.append(True)
            try:
                entered.set()
                release.wait()
                yield None
            finally:
                closed.set()
        preview = module.WirelessPreview("unused.invalid")
        with patch.object(module, "wireless_frames", blocked_source):
            try:
                preview.touch()
                self.assertTrue(entered.wait(1))
                with self.assertRaisesRegex(RuntimeError, "handoff cancelled"):
                    preview.stop()
                preview.touch()
                self.assertEqual(len(starts), 1)
            finally:
                release.set()
                preview.stop()
        self.assertTrue(closed.is_set())


if __name__ == "__main__":
    unittest.main()
