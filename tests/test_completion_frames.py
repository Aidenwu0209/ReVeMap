import numpy as np
import pytest

from pose_pipeline.semantic_runtime.completion_frames import select_depth_frames


def test_replaces_only_intermediate_frames_at_fixed_budget():
    points = np.array([[x, y, 2.0] for x in (-0.1, 0.1) for y in (-0.2, 0.2)])
    poses = {i: np.eye(4) for i in range(8)}
    for i, x in enumerate((0, 0.01, 0.02, 0.2, 0.39, 0.4, 0.39, 0.38)):
        poses[i][0, 3] = x
    semantic = {0, 5}
    reference = select_depth_frames(semantic, poses, poses, points, "offset2")
    selected = select_depth_frames(semantic, poses, poses, points, "complementary")
    assert reference == [0, 2, 5, 7]
    assert selected == [0, 3, 5, 7]


def test_missing_frames_keep_reference_slots_and_ties():
    poses = {i: np.eye(4) for i in range(16) if i != 7}
    semantic = {0, 1, 5, 10, 14}
    available = set(poses) - {3, 13}
    points = np.array([[0., 0., 1.]])
    reference = select_depth_frames(semantic, poses, available, points, "offset2")
    assert reference == [0, 1, 2, 5, 10, 12, 14]
    assert select_depth_frames(semantic, poses, available, points, "complementary") == reference
    with pytest.raises(ValueError, match="depth frame policy"):
        select_depth_frames(semantic, poses, available, points, "invalid")
