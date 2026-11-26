import numpy as np

from pose_pipeline.semantic_runtime.completion import complete_labels


def test_unanchored_track_cannot_create_an_instance():
    original = np.array([[0.0, 0.0, 0.0]])
    semantic = np.array([4])
    instance = np.array([37])
    # Three masks observe the same 100 cells. Only the first added cell is
    # near enough to inherit ID 37 through the geometry-only fallback.
    xyz = np.column_stack((0.04 + np.arange(100) * 0.1, np.zeros((100, 2))))
    frames = np.repeat(np.arange(3), 100)
    features = np.zeros((3, 6))
    features[:, 0] = [0.0, 0.12, 0.24]
    observations = (
        np.tile(np.arange(100), 3),
        np.tile(xyz, (3, 1)),
        np.full(300, 4),
        frames + 1,
        frames,
    )
    added, _, classes, ids = complete_labels(
        original,
        semantic,
        instance,
        observations,
        features,
        [None, (0, 4), (1, 4), (2, 4)],
    )
    np.testing.assert_allclose(added, xyz)
    np.testing.assert_array_equal(classes, np.full(101, 4))
    np.testing.assert_array_equal(ids, [37, 37] + [0] * 99)


def test_no_baseline_motion_preserves_inputs():
    original = np.array([[1.0, 2.0, 3.0]])
    rows = (
        np.array([0, 0]),
        np.repeat(original, 2, axis=0),
        np.zeros(2, int),
        np.zeros(2, int),
        np.arange(2),
    )
    added, _, classes, ids = complete_labels(
        original, np.array([4]), np.array([37]), rows, np.zeros((2, 6)), [None]
    )
    assert added.shape == (0, 3)
    assert classes.tolist() == [4] and ids.tolist() == [37]
