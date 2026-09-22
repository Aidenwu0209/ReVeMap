"""Synthetic contract tests for experimental filters, not real-scene acceptance."""
import json
import unittest

import numpy as np

from pose_pipeline.experimental_depth_denoise import (
    robust_denoise_depth,
    spatial_support_filter,
    temporal_consistency_filter,
    temporal_consistency_maps,
)


K = (20., 20., 4., 4.)
IDENTITY = np.eye(4)


def plane(depth=2000, shape=(9, 9)):
    return np.full(shape, depth, dtype=np.uint16)


def ref(depth, side=-1, pose=None, frame_id=None):
    value = {"depth": depth, "intrinsics": K,
             "t_world_camera": IDENTITY if pose is None else pose, "side": side}
    if frame_id is not None:
        value["frame_id"] = frame_id
    return value


class SpatialTests(unittest.TestCase):
    def test_large_isolated_spike_removed_and_input_unchanged(self):
        raw = plane()
        raw[4, 4] = 3000
        original = raw.copy()
        filtered, stats = spatial_support_filter(raw)
        self.assertEqual(filtered[4, 4], 0)
        self.assertEqual(stats["removed_pixels"], 1)
        np.testing.assert_array_equal(raw, original)

    def test_plane_holes_and_step_edges_preserved(self):
        raw = plane()
        raw[:, 5:] = 3000
        raw[3, 3] = 0
        filtered, stats = spatial_support_filter(raw)
        np.testing.assert_array_equal(filtered, raw)
        self.assertEqual(stats["removed_pixels"], 0)

    def test_one_pixel_line_interior_preserved_endpoints_explicitly_not_guaranteed(self):
        raw = plane()
        raw[:, 4] = 1000
        filtered, _ = spatial_support_filter(raw)
        np.testing.assert_array_equal(filtered[1:-1, 4], raw[1:-1, 4])
        self.assertEqual(filtered[0, 4], 0)
        self.assertEqual(filtered[-1, 4], 0)

    def test_sparse_line_over_invalid_background(self):
        raw = np.zeros((9, 9), dtype=np.uint16)
        raw[4, :] = 2000
        filtered, _ = spatial_support_filter(raw)
        np.testing.assert_array_equal(filtered[4, 1:-1], raw[4, 1:-1])
        self.assertTrue(np.all(filtered[raw == 0] == 0))

    def test_no_invented_border_neighbours(self):
        raw = np.array([[2000, 3000]], dtype=np.uint16)
        filtered, _ = spatial_support_filter(raw)
        np.testing.assert_array_equal(filtered, raw)
        raw = np.array([[2000, 3000, 2000]], dtype=np.uint16)
        filtered, _ = spatial_support_filter(raw)
        self.assertEqual(filtered[0, 1], 0)

    def test_range_scaled_tolerance_and_uint16_limits(self):
        raw = plane(10000)
        raw[4, 4] = 10080
        filtered, _ = spatial_support_filter(raw)
        self.assertEqual(filtered[4, 4], 10080)
        for value in (1, 65535):
            raw = plane(value)
            filtered, _ = spatial_support_filter(raw)
            np.testing.assert_array_equal(filtered, raw)

    def test_dtype_validation(self):
        with self.assertRaises(TypeError):
            spatial_support_filter(plane().astype(np.float32))
        with self.assertRaises(ValueError):
            spatial_support_filter(plane(), min_support=9)


class TemporalTests(unittest.TestCase):
    def test_identity_planes_supported_and_not_removed(self):
        raw = plane()
        refs = [ref(raw.copy(), -1), ref(raw.copy(), 1)]
        maps = temporal_consistency_maps(raw, K, IDENTITY, refs)
        np.testing.assert_array_equal(maps["support_count"], 2)
        np.testing.assert_array_equal(maps["conflict_count"], 0)
        filtered, stats = temporal_consistency_filter(raw, K, IDENTITY, refs)
        np.testing.assert_array_equal(filtered, raw)
        self.assertEqual(stats["removed_pixels"], 0)

    def test_axial_camera_motion_uses_camera_to_world_and_optical_z(self):
        raw = plane()
        moving_pose = IDENTITY.copy()
        moving_pose[2, 3] = 0.1
        maps = temporal_consistency_maps(raw, K, IDENTITY, [ref(plane(1900), 1, moving_pose)])
        self.assertEqual(maps["support_count"][4, 4], 1)
        self.assertEqual(maps["conflict_count"][4, 4], 0)

    def test_lateral_camera_motion_reprojects_thin_line(self):
        raw = plane(4000)
        raw[:, 4] = 2000
        observed = plane(4000)
        observed[:, 3] = 2000
        moving_pose = IDENTITY.copy()
        moving_pose[0, 3] = 0.1
        maps = temporal_consistency_maps(raw, K, IDENTITY, [ref(observed, 1, moving_pose)])
        self.assertEqual(maps["support_count"][4, 4], 1)

    def test_both_sides_free_space_conflict_removes_front_spike(self):
        raw = plane(3000)
        raw[4, 4] = 2000
        refs = [ref(plane(3000), -1), ref(plane(3000), 1)]
        filtered, stats = temporal_consistency_filter(raw, K, IDENTITY, refs)
        self.assertEqual(filtered[4, 4], 0)
        self.assertEqual(stats["removed_pixels"], 1)
        self.assertEqual(raw[4, 4], 2000)

    def test_same_side_multiple_conflicts_cannot_remove(self):
        raw = plane(2000)
        refs = [ref(plane(3000), -1), ref(plane(3000), -1)]
        filtered, stats = temporal_consistency_filter(raw, K, IDENTITY, refs)
        np.testing.assert_array_equal(filtered, raw)
        self.assertEqual(stats["removed_pixels"], 0)

    def test_any_support_blocks_rejection(self):
        raw = plane(2000)
        refs = [ref(plane(3000), -1), ref(plane(3000), 1), ref(plane(2000), 1)]
        filtered, _ = temporal_consistency_filter(raw, K, IDENTITY, refs)
        np.testing.assert_array_equal(filtered, raw)

    def test_foreground_occlusion_is_unknown_not_conflict(self):
        raw = plane(3000)
        refs = [ref(plane(2000), -1), ref(plane(2000), 1)]
        maps = temporal_consistency_maps(raw, K, IDENTITY, refs)
        np.testing.assert_array_equal(maps["unknown_count"], 2)
        np.testing.assert_array_equal(maps["conflict_count"], 0)
        filtered, _ = temporal_consistency_filter(raw, K, IDENTITY, refs)
        np.testing.assert_array_equal(filtered, raw)

    def test_invalid_projected_centre_cannot_borrow_nearby_support(self):
        raw = plane(2000)
        observed = raw.copy()
        observed[4, 4] = 0
        maps = temporal_consistency_maps(raw, K, IDENTITY, [ref(observed)])
        self.assertEqual(maps["support_count"][4, 4], 0)
        self.assertEqual(maps["conflict_count"][4, 4], 0)
        self.assertEqual(maps["unknown_count"][4, 4], 1)

    def test_hole_or_foreground_within_footprint_prevents_conflict(self):
        for neighbour_value in (0, 1000):
            observed = plane(3000)
            observed[4, 5] = neighbour_value
            maps = temporal_consistency_maps(plane(2000), K, IDENTITY, [ref(observed)])
            self.assertEqual(maps["conflict_count"][4, 4], 0)
            self.assertEqual(maps["unknown_count"][4, 4], 1)

    def test_foreground_nearest_cannot_borrow_nearby_background_support(self):
        observed = plane(3000)
        observed[4, 4] = 2000
        maps = temporal_consistency_maps(plane(3000), K, IDENTITY, [ref(observed)])
        self.assertEqual(maps["support_count"][4, 4], 0)
        self.assertEqual(maps["unknown_count"][4, 4], 1)

    def test_border_crop_and_outside_view_are_unknown(self):
        raw = plane(2000)
        maps = temporal_consistency_maps(raw, K, IDENTITY, [ref(plane(3000))])
        self.assertEqual(maps["conflict_count"][0, 0], 0)
        self.assertEqual(maps["unknown_count"][0, 0], 1)
        away = IDENTITY.copy()
        away[0, 3] = 100
        maps = temporal_consistency_maps(raw, K, IDENTITY, [ref(plane(3000), pose=away)])
        np.testing.assert_array_equal(maps["unknown_count"], 1)

    def test_behind_camera_unknown(self):
        pose = IDENTITY.copy()
        pose[2, 3] = 3
        maps = temporal_consistency_maps(plane(2000), K, IDENTITY, [ref(plane(), pose=pose)])
        np.testing.assert_array_equal(maps["unknown_count"], 1)

    def test_duplicate_references_cannot_enable_rejection(self):
        raw = plane(2000)
        same = plane(3000)
        filtered, stats = temporal_consistency_filter(raw, K, IDENTITY, [ref(same, -1), ref(same, 1)])
        np.testing.assert_array_equal(filtered, raw)
        self.assertEqual(stats["available_refs"], 1)
        self.assertEqual(stats["duplicate_refs"], 1)
        filtered, stats = temporal_consistency_filter(raw, K, IDENTITY,
            [ref(same.copy(), -1, frame_id="a"), ref(same.copy(), 1, frame_id="a")])
        np.testing.assert_array_equal(filtered, raw)
        self.assertEqual(stats["available_refs"], 1)

    def test_missing_reference_no_deletion_and_invalid_pixels_never_vote(self):
        raw = plane()
        raw[4, 4] = 0
        maps = temporal_consistency_maps(raw, K, IDENTITY, [ref(plane()), {"depth": None}])
        self.assertEqual(maps["skipped_refs"], 1)
        self.assertEqual(maps["support_count"][4, 4], 0)
        filtered, _ = temporal_consistency_filter(raw, K, IDENTITY, [])
        np.testing.assert_array_equal(filtered, raw)

    def test_equivalent_intrinsic_formats(self):
        raw = plane()
        for intrinsic in ({"fx":20,"fy":20,"cx":4,"cy":4}, np.array([[20,0,4],[0,20,4],[0,0,1]])):
            maps = temporal_consistency_maps(raw, intrinsic, IDENTITY, [ref(raw.copy())])
            np.testing.assert_array_equal(maps["support_count"], 1)

    def test_guard_reference_limit_pose_and_min_conflicts(self):
        with self.assertRaises(ValueError):
            temporal_consistency_maps(plane(), K, IDENTITY, [ref(plane())] * 9)
        with self.assertRaises(ValueError):
            temporal_consistency_filter(plane(), K, IDENTITY, [], min_conflicts=1)
        invalid = IDENTITY.copy()
        invalid[0, 0] = 2
        with self.assertRaises(ValueError):
            temporal_consistency_maps(plane(), K, invalid, [])


class CompositionTests(unittest.TestCase):
    def test_callback_and_input_preservation(self):
        raw = plane()
        raw[4, 4] = 3000
        def smooth(copy):
            copy[copy > 0] += 1
            return copy, {"stub": True}
        filtered, stats = robust_denoise_depth(raw, bilateral=smooth)
        self.assertEqual(filtered[4, 4], 0)
        self.assertEqual(filtered[3, 3], 2001)
        self.assertEqual(raw[4, 4], 3000)
        self.assertEqual(raw[3, 3], 2000)
        json.dumps(stats)

    def test_callback_cannot_fill_holes(self):
        raw = plane()
        raw[4, 4] = 0
        with self.assertRaises(ValueError):
            robust_denoise_depth(raw, bilateral=lambda x: np.maximum(x, 1))

    def test_disable_spatial_retains_temporally_supported_small_feature(self):
        raw = plane()
        raw[4, 4] = 1000
        filtered, stats = robust_denoise_depth(raw, spatial=False, intrinsics=K, t_world_camera=IDENTITY,
            references=[ref(raw.copy(), -1), ref(raw.copy(), 1)])
        np.testing.assert_array_equal(filtered, raw)
        self.assertNotIn("spatial", stats["stages"])


if __name__ == "__main__":
    unittest.main()
