"""Verify raw retention and the depth actually consumed by mapping contracts."""
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pose_pipeline.contracts import load_manifest, sha256_file
from pose_pipeline import device_capture as live_capture
from test_wireless_capture import camera, packet, wire


class DepthCaptureTest(unittest.TestCase):
    def run_capture(self, directory, **kwargs):
        args = dict(session=Path(directory), replay=None, packet_dir=None, ipad_port=0,
                    wireless_host="127.0.0.1", wireless_port=1, max_frames=1)
        args.update(kwargs)
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            live_capture.capture(SimpleNamespace(**args))
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)

    @staticmethod
    def noisy_packet():
        data = packet()
        depth = np.full((3, 4), 1250, np.uint16)
        depth[1, 1] = 1260
        depth[0, 0] = 0
        data["depth"] = depth
        return data

    def test_filtered_manifest_raw_bytes_and_metadata(self):
        data = self.noisy_packet()
        expected = data["depth"].copy()
        with tempfile.TemporaryDirectory() as directory:
            with camera([wire(data)], stay_open=True) as (port, _):
                self.run_capture(directory, wireless_port=port)
            root = Path(directory) / "capture"
            processed = load_manifest(root / "manifest.json")
            raw = load_manifest(root / "manifest_raw.json")
            raw_depth = cv2.imread(str(raw.frames[0].depth_path), -1)
            filtered = cv2.imread(str(processed.frames[0].depth_path), -1)
            np.testing.assert_array_equal(raw_depth, expected)
            self.assertLess(int(filtered[1, 1]), 1260)
            np.testing.assert_array_equal(filtered == 0, expected == 0)
            self.assertEqual(raw.frames[0].timestamp_us, processed.frames[0].timestamp_us)
            self.assertEqual(raw.frames[0].intrinsics, processed.frames[0].intrinsics)
            self.assertEqual(raw.frames[0].color_path, processed.frames[0].color_path)
            np.testing.assert_array_equal(cv2.imread(str(processed.frames[0].color_path)), data["color"])
            row = json.loads((root / "frames.jsonl").read_text())
            self.assertEqual(row["raw_depth_sha256"], sha256_file(raw.frames[0].depth_path))
            self.assertEqual(Path(row["depth_path"]), processed.frames[0].depth_path)
            self.assertTrue(row["depth_processing"]["enabled"])
            audit = json.loads((root / "depth_processing.json").read_text())
            self.assertEqual(audit["frames"], 1)
            self.assertGreater(audit["changed_pixels"], 0)

    def test_off_preserves_original_path_and_values(self):
        data = self.noisy_packet()
        with tempfile.TemporaryDirectory() as directory:
            with camera([wire(data)], stay_open=True) as (port, _):
                self.run_capture(directory, wireless_port=port, depth_denoise="off")
            root = Path(directory) / "capture"
            manifest = load_manifest(root / "manifest.json")
            np.testing.assert_array_equal(cv2.imread(str(manifest.frames[0].depth_path), -1), data["depth"])
            self.assertEqual(manifest.source, "wireless_orbbec_d2c_sensor_timestamps")
            self.assertFalse((root / "depth_raw").exists())
            self.assertFalse(json.loads((root / "depth_processing.json").read_text())["enabled"])

    def test_replay_does_not_denoise_again(self):
        data = self.noisy_packet()
        def frames(*_args):
            yield data["color"], data["depth"], tuple(data["intrinsics"]), 1000000, {}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(live_capture, "replay_frames", frames), patch.object(
                    live_capture, "denoise_depth", side_effect=AssertionError("Replay filtered twice")):
                self.run_capture(directory, replay=Path("synthetic"), wireless_host=None, fps=10)
            manifest = load_manifest(Path(directory) / "capture/manifest.json")
            np.testing.assert_array_equal(cv2.imread(str(manifest.frames[0].depth_path), -1), data["depth"])

    def test_raw_write_failure_never_commits_filtered_frame(self):
        data = self.noisy_packet()
        with tempfile.TemporaryDirectory() as directory:
            with camera([wire(data)], stay_open=True) as (port, _):
                with patch.object(live_capture.cv2, "imwrite", return_value=False):
                    with self.assertRaisesRegex(RuntimeError, "原始深度写盘失败"):
                        self.run_capture(directory, wireless_port=port)
            root = Path(directory) / "capture"
            self.assertEqual((root / "frames.jsonl").read_text(), "")
            self.assertFalse((root / "manifest.json").exists())
            self.assertEqual(json.loads((Path(directory) / "capture_status.json").read_text())["status"], "failed")

    def test_second_working_write_failure_counts_only_committed_frames(self):
        data = self.noisy_packet()
        def frames(*_args):
            for timestamp in (1000000, 2000000):
                yield data["color"], data["depth"], tuple(data["intrinsics"]), timestamp, {}
        imwrite = cv2.imwrite
        def fail_second_depth(path, *args):
            if Path(path).parent.name == "depth" and Path(path).name == "000001.png":
                return False
            return imwrite(path, *args)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(live_capture, "wireless_frames", frames), patch.object(
                    live_capture.cv2, "imwrite", fail_second_depth):
                with self.assertRaisesRegex(RuntimeError, "RGB-D 写盘失败"):
                    self.run_capture(directory, max_frames=2)
            root = Path(directory) / "capture"
            working, raw = load_manifest(root / "manifest.json"), load_manifest(root / "manifest_raw.json")
            self.assertEqual((len(working.frames), len(raw.frames)), (1, 1))
            row = json.loads((root / "frames.jsonl").read_text())
            processing = json.loads((root / "depth_processing.json").read_text())
            self.assertEqual(processing["frames"], 1)
            for key in ("changed_pixels", "valid_pixels", "max_abs_change_mm"):
                self.assertEqual(processing[key], row["depth_processing"][key])
            self.assertEqual(processing["total_filter_ms"], row["depth_processing"]["elapsed_ms"])


if __name__ == "__main__":
    unittest.main()
