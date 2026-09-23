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
    def test_plane_exports_a_measured_triangle_surface(self):
        import open3d as o3d
        from PIL import Image
        from pose_pipeline.contracts import sha256_file
        from pose_pipeline.surface import publish_surface

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            color, depth = root / 'color.png', root / 'depth.png'
            Image.fromarray(np.full((48, 48, 3), [90, 160, 210], dtype=np.uint8)).save(color)
            Image.fromarray(np.full((48, 48), 1000, dtype=np.uint16)).save(depth)
            manifest, trajectory = root / 'manifest.json', root / 'trajectory.json'
            write_manifest(manifest, SequenceManifest(
                'scannet', 'plane', root, 1000.,
                (FrameRecord(0, 0., color, depth, (50., 50., 23.5, 23.5)),), 'test'))
            write_trajectory(trajectory, [PoseRecord(0, 0., np.eye(4))], sequence_id='plane', arm='candidate')
            output = root / 'fusion'
            receipt = run_full_rgbd_refusion(FullRefusionRequest(
                manifest=manifest, trajectory=trajectory, output_dir=output,
                voxel_length_m=.02, sdf_trunc_m=.08))
            mesh = o3d.io.read_triangle_mesh(receipt['mesh'])
            self.assertGreater(len(mesh.triangles), 0)
            self.assertEqual(len(mesh.triangles), receipt['mesh_triangles'])
            self.assertEqual(sha256_file(receipt['mesh']), receipt['mesh_sha256'])
            self.assertTrue(mesh.has_vertex_colors() and mesh.has_vertex_normals())
            np.testing.assert_allclose(np.asarray(mesh.vertices)[:, 2], 1., atol=.025)
            source_hashes = {p: sha256_file(p) for p in root.rglob('*') if p.is_file()}
            index = publish_surface(root, map_path=receipt['cloud'], manifest=manifest,
                                    trajectory=trajectory, refusion_receipt=output / 'refusion_result.json',
                                    viewer_triangles=100)
            import json
            value = json.loads(index.read_text())
            self.assertLessEqual(value['viewer_triangles'], 100)
            self.assertTrue(all(sha256_file(p) == digest for p, digest in source_hashes.items()))
            with self.assertRaises(FileExistsError):
                publish_surface(root, map_path=receipt['cloud'], manifest=manifest,
                                trajectory=trajectory, refusion_receipt=output / 'refusion_result.json')

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
