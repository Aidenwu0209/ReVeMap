"""Small geometry and measured-source checks for the full raw mapping stages."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pose_pipeline.rgbd_empty_anchor import (
    audit_observations,
    permit_unused_empty,
    source_topology,
)
from pose_pipeline.rgbd_measured import measured_pair


class MeasuredRefillTests(unittest.TestCase):
    def test_metric_pair_preserves_original_transform_and_information_energy(self):
        rng = np.random.default_rng(37)
        source = rng.uniform([-0.6, -0.4, 1.0], [0.6, 0.4, 2.5], (80, 3))
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec([0.025, -0.04, 0.012]).as_matrix()
        transform[:3, 3] = [0.04, -0.03, 0.06]
        target = source @ transform[:3, :3].T + transform[:3, 3]
        intrinsic = np.array(
            [[510.0, 0.0, 320.0], [0.0, 505.0, 240.0], [0.0, 0.0, 1.0]]
        )

        def project(points):
            homogeneous = points @ intrinsic.T
            return homogeneous[:, :2] / homogeneous[:, 2:]

        descriptors = rng.normal(size=(80, 128)).astype(np.float32)
        result = measured_pair(
            (project(source), descriptors, source, intrinsic),
            (project(target), descriptors.copy(), target, intrinsic),
        )
        self.assertIsNotNone(result)
        actual, information, fit = result
        np.testing.assert_allclose(actual, transform, atol=1e-12, rtol=0)
        self.assertEqual(fit["inliers"], 80)
        # Check H against an independently perturbed target-point cloud, in
        # the original left, rotation/translation coordinates (not H-only).
        direction = np.array([0.2, -0.15, 0.1, -0.03, 0.08, 0.02])
        epsilon = 1e-6
        plus = (
            target @ Rotation.from_rotvec(epsilon * direction[:3]).as_matrix().T
            + epsilon * direction[3:]
        )
        minus = (
            target @ Rotation.from_rotvec(-epsilon * direction[:3]).as_matrix().T
            - epsilon * direction[3:]
        )
        derivative = (plus - minus) / (2 * epsilon)
        expected_energy = np.sum(derivative**2) / (0.03**2)
        self.assertAlmostEqual(
            float(direction @ information @ direction) / expected_energy, 1.0, places=8
        )

    def test_empty_anchor_requires_unused_topology_and_actual_target_support(self):
        # Anchor 1 has no non-keyframes on either side; anchors 2/3 bracket
        # the only unknown frame and may never be accepted without depth.
        topology = source_topology([0, 1, 2, 4], 5)
        self.assertTrue(permit_unused_empty(1, topology["source_used_nodes"]))
        with self.assertRaisesRegex(RuntimeError, "used by an actual native"):
            permit_unused_empty(2, topology["source_used_nodes"])
        arguments = dict(
            ii=[2, 3],
            jj=[4, 4],
            target_start=4,
            target_end=5,
            source_used_nodes=[2, 3],
            empty_source_nodes=[1],
            source_valid_pixels=[2, 2],
            total_target_components=[4, 4],
            finite_target_components=[4, 4],
            total_weight_components=[4, 4],
            finite_weight_components=[4, 4],
            nonzero_weight_components=[1, 0],
            nonzero_weight_pixels=[1, 0],
        )
        self.assertTrue(audit_observations(**arguments)["ok"])
        arguments["nonzero_weight_components"] = [0, 0]
        arguments["nonzero_weight_pixels"] = [0, 0]
        failures = audit_observations(**arguments)
        self.assertFalse(failures["ok"])
        self.assertIn(
            "NF_target_has_zero_measured_visual_weight",
            [row["reason"] for row in failures["failures"]],
        )
        arguments["nonzero_weight_components"] = [1, 0]
        arguments["nonzero_weight_pixels"] = [1, 0]
        arguments["finite_target_components"] = [3, 4]
        self.assertFalse(audit_observations(**arguments)["ok"])


if __name__ == "__main__":
    unittest.main()
