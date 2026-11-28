import numpy as np
import pytest

from pose_pipeline.semantic_runtime.ownership_history import review_ownership


@pytest.mark.parametrize("verify, expected_end", [(False, 40), (True, 30)])
def test_history_requires_distinct_anchored_view_and_only_edits_added_points(verify, expected_end):
    # The second mask has insufficient anchors; fifteen shared cells tie it to
    # the first mask. Only ten of its added cells appear in the held-out view.
    groups = [np.arange(30), np.arange(15, 40), np.arange(30)]
    nodes = np.repeat([1, 2, 3], [len(g) for g in groups])
    features = np.zeros((3, 6))
    features[:, 0] = [0.25, 0.50, 1.25]
    distance = np.where(np.arange(40) < 20, 0.02, 0.20)
    append = np.arange(40) >= 20
    base_instance = np.full(40, 7)
    initial = np.full(20, 43)
    revised, audit = review_ownership(
        np.concatenate(groups), nodes, [None, (0, 3), (1, 3), (2, 3)],
        features, distance, np.arange(40), np.full(40, 3), base_instance,
        append, np.full(20, 3), initial, verify,
    )
    np.testing.assert_array_equal(revised, np.where(np.arange(20, 40) < expected_end, 7, 43))
    np.testing.assert_array_equal(initial, np.full(20, 43))
    assert audit["historical_masks"] == 1 and audit["changed_cells"] == expected_end - 20
