from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_pipeline.contracts import (
    FrameRecord,
    PoseRecord,
    SequenceManifest,
    bind_manifest_trajectory,
    load_manifest,
    load_legacy_tcw_mm,
    load_trajectory,
    write_input_sha256_audit,
    write_manifest,
    write_trajectory,
)


class PosePipelineContractTests(unittest.TestCase):
    def test_manifest_and_trajectory_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "color").mkdir()
            (root / "depth").mkdir()
            frames = []
            for index in range(2):
                color = root / "color" / f"{index}.jpg"
                depth = root / "depth" / f"{index}.png"
                color.write_bytes(b"rgb")
                depth.write_bytes(b"depth")
                frames.append(FrameRecord(
                    index, 1000 + index, color, depth,
                    (500.0, 500.0, 320.0, 240.0),
                ))
            manifest = SequenceManifest(
                "scannet", "scene", root, 1000.0, tuple(frames), "test",
            )
            manifest_path = root / "manifest.json"
            write_manifest(manifest_path, manifest)
            loaded = load_manifest(manifest_path)
            self.assertEqual(len(loaded.frames), 2)
            audit = write_input_sha256_audit(root / "inputs.sha256.jsonl", loaded)
            self.assertEqual(audit["frame_count"], 2)
            self.assertEqual(len(audit["records_sha256"]), 64)
            trajectory_path = root / "trajectory.json"
            records = [PoseRecord(
                index, 1000 + index, np.eye(4), source="test",
            ) for index in range(2)]
            write_trajectory(
                trajectory_path, records, sequence_id="scene", arm="baseline",
            )
            trajectory, payload = load_trajectory(trajectory_path)
            self.assertFalse(payload["identity_fallback_used"])
            self.assertEqual(len(payload["stable_pose_sha256_q1e7"]), 64)
            self.assertEqual(len(bind_manifest_trajectory(loaded, trajectory)), 2)

    def test_manifest_rejects_gt_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pose").mkdir()
            (root / "depth").mkdir()
            color = root / "pose" / "0.jpg"
            depth = root / "depth" / "0.png"
            color.write_bytes(b"rgb")
            depth.write_bytes(b"depth")
            manifest = SequenceManifest(
                "scannet", "scene", root, 1000.0,
                (FrameRecord(0, 1, color, depth, (1.0, 1.0, 0.0, 0.0)),),
                "test",
            )
            with self.assertRaisesRegex(ValueError, "forbidden GT part"):
                manifest.validate()

    def test_missing_pose_fails_exact_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "color").mkdir()
            (root / "depth").mkdir()
            frames = []
            for index in range(2):
                color = root / "color" / f"{index}.jpg"
                depth = root / "depth" / f"{index}.png"
                color.write_bytes(b"rgb")
                depth.write_bytes(b"depth")
                frames.append(FrameRecord(
                    index, index, color, depth, (1.0, 1.0, 0.0, 0.0),
                ))
            manifest = SequenceManifest(
                "scannet", "scene", root, 1000.0, tuple(frames), "test",
            ).validate()
            poses = [PoseRecord(0, 0, np.eye(4))]
            with self.assertRaisesRegex(ValueError, "does not cover"):
                bind_manifest_trajectory(manifest, poses)
            self.assertEqual(len(bind_manifest_trajectory(
                manifest, poses, allow_manifest_superset=True,
            )), 1)

    def test_legacy_tcw_mm_imports_as_world_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.txt"
            tcw = np.eye(4)
            tcw[0, 3] = 1000.0
            fields = " ".join(str(value) for value in tcw.reshape(-1))
            path.write_text(f"# contract\n7 99 {fields}\n")
            rows = load_legacy_tcw_mm(path)
            self.assertEqual(rows[0].frame_id, 7)
            self.assertAlmostEqual(rows[0].t_world_camera[0, 3], -1.0)


if __name__ == "__main__":
    unittest.main()
