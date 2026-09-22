"""Depth-filter behavior tests; synthetic inputs, no camera or GPU required."""
import json
import unittest

import numpy as np

from pose_pipeline.depth_denoise import ALGORITHM, DEFAULT_CONFIG, denoise_depth


class DepthDenoiseTest(unittest.TestCase):
    def assert_preserved_contract(self, before, after):
        self.assertEqual(after.shape, before.shape)
        self.assertEqual(after.dtype, np.dtype(np.uint16))
        np.testing.assert_array_equal(after == 0, before == 0)
        self.assertFalse(np.shares_memory(before, after))
        change = np.abs(after.astype(np.int32) - before.astype(np.int32))
        self.assertLessEqual(int(change.max()) if change.size else 0, 10)

    def test_reduces_synthetic_planar_noise(self):
        rng = np.random.default_rng(20260922)
        depth = np.rint(1500 + rng.normal(0, 6, size=(120, 160))).astype(np.uint16)
        original = depth.copy()
        filtered, stats = denoise_depth(depth)
        rmse_before = float(np.sqrt(np.mean((depth.astype(np.float64) - 1500) ** 2)))
        rmse_after = float(np.sqrt(np.mean((filtered.astype(np.float64) - 1500) ** 2)))
        self.assertLess(rmse_after, 0.75 * rmse_before)
        self.assertLess(abs(float(filtered.mean()) - 1500), 0.2)
        self.assertGreater(stats["changed_pixels"], 0)
        self.assert_preserved_contract(depth, filtered)
        np.testing.assert_array_equal(depth, original)

    def test_step_and_one_pixel_wide_pole_are_preserved(self):
        for depth in (
            np.tile(np.array([1000, 1000, 1000, 1600, 1600], np.uint16), (9, 1)),
            np.tile(np.array([2000, 2000, 1000, 2000, 2000], np.uint16), (9, 1)),
        ):
            with self.subTest(depth=depth[0].tolist()):
                filtered, stats = denoise_depth(depth)
                np.testing.assert_array_equal(filtered, depth)
                self.assertEqual(stats["changed_pixels"], 0)
                self.assert_preserved_contract(depth, filtered)

    def test_neighbor_beyond_hard_gate_is_excluded(self):
        depth = np.full((3, 3), 1031, dtype=np.uint16)
        depth[1, 1] = 1000
        filtered, _ = denoise_depth(depth)
        np.testing.assert_array_equal(filtered, depth)

    def test_invalid_holes_do_not_pull_valid_depth_towards_zero(self):
        # Small nonzero values ensure this also tests validity independently of
        # the 30 mm hard gate: a zero neighbour would otherwise be accepted.
        depth = np.array([[0, 10, 10], [10, 0, 10], [10, 10, 0]], dtype=np.uint16)
        filtered, stats = denoise_depth(depth)
        np.testing.assert_array_equal(filtered, depth)
        self.assertEqual(stats["valid_pixels"], 6)
        self.assert_preserved_contract(depth, filtered)

    def test_all_zero_empty_and_single_pixel(self):
        for depth in (
            np.zeros((5, 7), dtype=np.uint16),
            np.zeros((0, 3), dtype=np.uint16),
            np.zeros((3, 0), dtype=np.uint16),
            np.array([[0]], dtype=np.uint16),
            np.array([[1]], dtype=np.uint16),
            np.array([[65535]], dtype=np.uint16),
        ):
            with self.subTest(shape=depth.shape, content=depth.tolist()):
                filtered, stats = denoise_depth(depth)
                np.testing.assert_array_equal(filtered, depth)
                self.assertEqual(stats["changed_pixels"], 0)
                self.assertEqual(stats["mean_abs_change_mm"], 0)
                self.assert_preserved_contract(depth, filtered)
                json.dumps(stats, allow_nan=False)

    def test_border_uses_only_actual_neighbors_without_wrapping(self):
        depth = np.array([[1000, 1010, 0, 2000]], dtype=np.uint16)
        filtered, _ = denoise_depth(depth)
        # One real neighbour 10 mm away: exp(-0.5) * exp(-0.5)
        # produces a 2.69 mm adjustment. Border padding would bias this result.
        np.testing.assert_array_equal(filtered, [[1003, 1007, 0, 2000]])
        transposed, _ = denoise_depth(depth.T)
        np.testing.assert_array_equal(transposed, filtered.T)
        self.assert_preserved_contract(depth, filtered)

    def test_extreme_valid_values_and_readonly_noncontiguous_input(self):
        base = np.array([
            [1, 10, 20, 0, 65515, 65525, 65535],
            [5, 12, 22, 0, 65517, 65530, 65532],
            [3, 9, 19, 0, 65512, 65521, 65531],
        ], dtype=np.uint16)
        depth = base[:, ::-1]
        original = base.copy()
        depth.flags.writeable = False
        filtered, _ = denoise_depth(depth)
        self.assert_preserved_contract(depth, filtered)
        np.testing.assert_array_equal(base, original)

    def test_off_returns_independent_unchanged_image(self):
        depth = np.array([[0, 1000, 1007], [1005, 991, 1003]], dtype=np.uint16)
        filtered, stats = denoise_depth(depth, mode="off")
        np.testing.assert_array_equal(filtered, depth)
        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["changed_pixels"], 0)
        self.assertEqual(stats["valid_pixels"], 5)
        self.assert_preserved_contract(depth, filtered)

    def test_stats_match_actual_changes_and_are_json_compatible(self):
        depth = np.array([[0, 1000, 1007], [1005, 991, 1003]], dtype=np.uint16)
        filtered, stats = denoise_depth(depth)
        changes = np.abs(filtered.astype(np.int32) - depth.astype(np.int32))
        self.assertEqual(stats["algorithm"], ALGORITHM)
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["parameters"], DEFAULT_CONFIG)
        self.assertEqual(stats["valid_pixels"], 5)
        self.assertEqual(stats["changed_pixels"], int(np.count_nonzero(changes)))
        self.assertEqual(stats["mean_abs_change_mm"], float(changes.sum()) / 5)
        self.assertEqual(stats["max_abs_change_mm"], int(changes.max()))
        self.assertGreaterEqual(stats["elapsed_ms"], 0)
        json.dumps(stats, allow_nan=False)
        stats["parameters"]["sigma_depth_mm"] = 999
        self.assertEqual(DEFAULT_CONFIG["sigma_depth_mm"], 10)

    def test_rejects_wrong_dtype_shape_and_mode(self):
        for dtype in (np.float32, np.float64, np.uint8, np.int16, np.int32):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                denoise_depth(np.ones((2, 2), dtype=dtype))
        with self.assertRaises(TypeError):
            denoise_depth([[1000]])
        for shape in ((3,), (2, 2, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                denoise_depth(np.ones(shape, dtype=np.uint16))
        with self.assertRaises(ValueError):
            denoise_depth(np.ones((2, 2), dtype=np.uint16), mode="median")


if __name__ == "__main__":
    unittest.main(verbosity=2)
