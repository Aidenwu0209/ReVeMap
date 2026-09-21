from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_pipeline.contracts import (
    FrameRecord, PoseRecord, SequenceManifest,
    write_manifest, write_trajectory,
)
from reconstruction.rgbd_refusion import FullRefusionRequest, run_full_rgbd_refusion


@unittest.skipUnless(importlib.util.find_spec("open3d"), "Open3D runtime required")
class RGBDRefusionContractTests(unittest.TestCase):
    def test_default_frame_list_rejects_missing_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "color").mkdir()
            (root / "depth").mkdir()
            frames = []
            for index in range(2):
                color, depth = root / "color" / f"{index}.jpg", root / "depth" / f"{index}.png"
                color.write_bytes(b"not-read-before-contract-check")
                depth.write_bytes(b"not-read-before-contract-check")
                frames.append(FrameRecord(
                    index, index, color, depth, (500.0, 500.0, 1.0, 1.0),
                ))
            manifest = root / "manifest.json"
            trajectory = root / "trajectory.json"
            write_manifest(manifest, SequenceManifest(
                "scannet", "scene", root, 1000.0, tuple(frames), "test",
            ))
            write_trajectory(
                trajectory, [PoseRecord(0, 0, np.eye(4))],
                sequence_id="scene", arm="candidate",
            )
            with self.assertRaisesRegex(ValueError, "misses 1 fused frames"):
                run_full_rgbd_refusion(FullRefusionRequest(
                    manifest=manifest, trajectory=trajectory,
                    output_dir=root / "refusion",
                ))


if __name__ == "__main__":
    unittest.main()
