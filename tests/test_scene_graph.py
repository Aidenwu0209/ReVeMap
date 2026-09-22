import numpy as np
import pytest

from pose_pipeline.scene_graph import build_graph, query_graph


def box(low, high):
    return np.array([[x, y, z] for x in (low[0], high[0])
                     for y in (low[1], high[1]) for z in (low[2], high[2])], dtype=float)


def scene(up=None):
    points = np.concatenate([box([0, 0, 0], [1, 1, 1]),
                             box([.25, .25, 1.02], [.75, .75, 1.4]),
                             box([2, 0, 0], [3, 1, 1])])
    ids = np.repeat([1, 2, 3], 8)
    return build_graph(points, ids, ids, {"0": "unknown", "1": "table", "2": "cup", "3": "chair"},
                       world_up=up)


def test_relations_require_explicit_up_and_support_has_measured_gap():
    graph = scene()
    assert all(e["relation"] == "near" for e in graph["edges"])
    assert query_graph(graph, relation="above", reference_id=1)["reason"] == "world_up_required"
    graph = scene([0, 0, 1])
    result = query_graph(graph, label="杯子", relation="supported_by", reference_id=1)
    assert result["instance_ids"] == [2]
    assert result["evidence"][0]["vertical_gap_m"] == pytest.approx(.02)
    assert query_graph(graph, relation="supported_by", reference_id=2)["instance_ids"] == []


def test_label_and_nearest_queries_never_fabricate_and_preserve_ties():
    graph = scene()
    assert query_graph(graph, label="椅子")["instance_ids"] == [3]
    assert query_graph(graph, nearest_to=1)["instance_ids"] == [2]
    assert query_graph(graph, label="sofa")["status"] == "unknown"
    graph["nodes"][2]["centroid_m"] = graph["nodes"][1]["centroid_m"]
    assert query_graph(graph, nearest_to=1)["instance_ids"] == [2, 3]
    with pytest.raises(ValueError, match="does not exist"):
        query_graph(graph, nearest_to=99)


def test_near_relation_is_symmetric_and_explanation_filters_labels():
    graph = scene()
    assert query_graph(graph, relation="near", reference_id=2)["instance_ids"] == [1]
    assert query_graph(graph, label="chair", relation="near", reference_id=2)["evidence"] == []


def test_names_require_consensus_and_unknown_geometry_is_preserved():
    xyz = np.arange(18).reshape(6, 3).astype(float)
    sem, ids = np.array([0, 0, 1, 1, 1, 1]), np.array([0, 0, 2, 2, 3, 3])
    originals = (xyz.copy(), sem.copy(), ids.copy())
    graph = build_graph(xyz, sem, ids, {"0": "unknown", "1": "chair"}, names=[
        {"instance_id": 2, "vlm_name": "stool", "support_frames": [1]},
        {"instance_id": 3, "vlm_name": "stool", "support_frames": [1, 2], "evidence": [{"frame_id": 1}]},
    ])
    assert graph["unknown_points"] == 2
    assert query_graph(graph, label="stool")["instance_ids"] == [3]
    for current, original in zip((xyz, sem, ids), originals):
        np.testing.assert_array_equal(current, original)


def test_vertical_relations_follow_up_not_coordinate_axis():
    graph = scene([0, 0, -1])
    assert query_graph(graph, relation="supported_by", reference_id=1)["instance_ids"] == []
    assert query_graph(graph, relation="supported_by", reference_id=2)["instance_ids"] == [1]


def test_invalid_geometry_and_thresholds_rejected():
    for kwargs in ({"world_up": [0, 0, 0]}, {"near_m": float("nan")}, {"min_points": 0}):
        with pytest.raises(ValueError):
            build_graph(np.zeros((1, 3)), np.array([0]), np.array([1]), {"0": "unknown"}, **kwargs)
    with pytest.raises(ValueError):
        build_graph([[float("nan"), 0, 0]], np.array([0]), np.array([0]), {"0": "unknown"})


def test_query_unknown_names_whitespace_alias_and_source_boundaries():
    up = np.array([0., 0., 2.])
    graph = build_graph(np.arange(9).reshape(3, 3), np.array([0, 1, 1]), np.array([1, 2, 3]),
                        {'0': 'unknown', '1': 'refridgerator'}, world_up=up,
                        names=[{'instance_id': 2, 'vlm_name': 'Unknown', 'support_frames': [1, 2]},
                               {'instance_id': 3, 'vlm_name': None, 'support_frames': [1, 2]}])
    np.testing.assert_array_equal(up, [0, 0, 2])
    assert graph['semantic_unknown_points'] == 1
    assert graph['unassigned_instance_points'] == 0
    assert query_graph(graph, label='unknown')['instance_ids'] == [1]
    assert query_graph(graph, label='refrigerator')['instance_ids'] == [2, 3]
    for label in ('', '   ', 12):
        with pytest.raises(ValueError, match='nonempty text'):
            query_graph(graph, label=label)
