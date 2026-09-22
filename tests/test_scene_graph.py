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


def oracle_graph(up=(0, 0, 1)):
    """Hand-authored boxes; expected answers below are not computed by the graph."""
    bounds = [([0, 0, 0], [2, 2, 1]), ([.4, .4, 1.02], [.8, .8, 1.4]),
              ([1.1, 1.1, 1.2], [1.5, 1.5, 1.3]), ([-1.5, .5, 0], [-1, 1.5, 1]),
              ([3, .5, 0], [3.5, 1.5, 1]), ([8, 0, 0], [8.2, .2, .2]),
              ([10, 0, 0], [11, 1, 1]), ([20, 0, 0], [21, 1, 1]),
              ([23, 0, 0], [24, 1, 1])]
    xyz = np.concatenate([box(low, high) for low, high in bounds])
    return build_graph(xyz, np.repeat([1, 2, 3, 4, 4, 0, 5, 1, 4], 8),
        np.repeat(np.arange(1, 10), 8),
        {"0": "unknown", "1": "table", "2": "cup", "3": "book", "4": "chair", "5": "cabinet"},
        names=[{"instance_id": 7, "vlm_name": "chair", "support_frames": [1, 2]},
               {"instance_id": 8, "vlm_name": "workbench", "support_frames": [3, 4]},
               {"instance_id": 9, "vlm_name": "workbench", "support_frames": [5, 6]},
               {"instance_id": 6, "vlm_name": "lamp", "support_frames": [7]}], world_up=up)


# This acceptance oracle is fixed, human-authored geometry/query logic. It is
# deliberately independent of SAM3 outputs, scores, GT matching and graph edges.
QUERY_ORACLES = [
    ("list_all", True, {}, list(range(1, 10)), None),
    ("category_table", True, {"label": "table"}, [1, 8], None),
    ("category_chinese", True, {"label": "杯子"}, [2], None),
    ("category_case_space", True, {"label": "  BOOK "}, [3], None),
    ("unknown_object", True, {"label": "unknown"}, [6], None),
    ("missing_object", True, {"label": "monitor"}, [], "no_matching_object"),
    ("single_view_name_abstains", True, {"label": "lamp"}, [], "no_matching_object"),
    ("same_name_two_objects", True, {"label": "workbench"}, [8, 9], None),
    ("name_category_conflict", True, {"label": "chair"}, [4, 5, 7, 9], None),
    ("conflict_preserves_category", True, {"label": "cabinet"}, [7], None),
    ("nearest_tie", True, {"label": "chair", "nearest_to": 1}, [4, 5], None),
    ("nearest_self_excluded", True, {"label": "table", "nearest_to": 1}, [8], None),
    ("nearest_single_self_unknown", True, {"label": "cup", "nearest_to": 2}, [], "no_matching_object"),
    ("nearest_unknown_reference", True, {"nearest_to": 6}, [7], None),
    ("nearest_missing_category", True, {"label": "sofa", "nearest_to": 1}, [], "no_matching_object"),
    ("near_table", True, {"relation": "near", "reference_id": 1}, [2, 3], None),
    ("near_symmetric", True, {"relation": "near", "reference_id": 2}, [1, 3], None),
    ("near_filtered", True, {"label": "book", "relation": "near", "reference_id": 1}, [3], None),
    ("near_wrong_label", True, {"label": "chair", "relation": "near", "reference_id": 1}, [], "no_matching_object"),
    ("near_isolated", True, {"relation": "near", "reference_id": 8}, [], "no_matching_object"),
    ("above_table", True, {"relation": "above", "reference_id": 1}, [2, 3], None),
    ("above_gap_allowed", True, {"label": "book", "relation": "above", "reference_id": 1}, [3], None),
    ("support_small_gap", True, {"relation": "supported_by", "reference_id": 1}, [2], None),
    ("support_large_gap_abstains", True, {"label": "book", "relation": "supported_by", "reference_id": 1}, [], "no_matching_object"),
    ("above_direction", True, {"relation": "above", "reference_id": 2}, [], "no_matching_object"),
    ("support_direction", True, {"relation": "supported_by", "reference_id": 2}, [], "no_matching_object"),
    ("no_gravity_above", False, {"relation": "above", "reference_id": 1}, [], "world_up_required"),
    ("no_gravity_support", False, {"relation": "supported_by", "reference_id": 1}, [], "world_up_required"),
    ("no_gravity_near", False, {"relation": "near", "reference_id": 1}, [2, 3], None),
    ("no_gravity_category", False, {"label": "book"}, [3], None),
]


@pytest.mark.parametrize("name,gravity,query,expected,reason", QUERY_ORACLES, ids=[row[0] for row in QUERY_ORACLES])
def test_fixed_geometry_query_oracle(name, gravity, query, expected, reason):
    answer = query_graph(oracle_graph() if gravity else oracle_graph(None), **query)
    assert answer["instance_ids"] == expected
    assert answer["status"] == ("matched" if expected else "unknown")
    assert answer["reason"] == reason
    assert "not physical contact verification" in answer["relation_scope"]


def test_name_conflict_is_explained_and_does_not_relabel_object():
    answer = query_graph(oracle_graph(), label="chair")
    conflict = next(row for row in answer["label_matches"] if row["instance_id"] == 7)
    assert conflict == {"instance_id": 7, "fields": ["consensus_name"], "name_category_conflict": True}
    assert next(row for row in answer["objects"] if row["instance_id"] == 7)["label"] == "cabinet"


def test_sparse_large_map_avoids_all_pairs_and_dense_map_fails_explicitly(monkeypatch):
    import pose_pipeline.scene_graph as module
    calls = []
    original = module._box_distance
    monkeypatch.setattr(module, "_box_distance", lambda a, b: (calls.append(1), original(a, b))[1])
    ids = np.arange(1, 5001)
    xyz = np.column_stack([ids * 10., np.zeros(5000), np.zeros(5000)])
    graph = build_graph(xyz, np.zeros(5000, int), ids, {"0": "unknown"}, world_up=[0, 0, 1])
    assert len(graph["nodes"]) == 5000 and graph["edges"] == [] and len(calls) == 0
    with pytest.raises(ValueError, match="candidate limit exceeded"):
        build_graph(np.zeros((50, 3)), np.zeros(50, int), np.arange(1, 51),
                    {"0": "unknown"}, max_relation_candidates=10)


def test_serialized_duplicate_self_loop_dangling_and_vertical_cycle_rejected():
    import copy
    graph = scene([0, 0, 1])
    duplicate = copy.deepcopy(graph)
    duplicate["nodes"].append(duplicate["nodes"][0])
    with pytest.raises(ValueError, match="unique positive"):
        query_graph(duplicate)
    for source, target in [(1, 1), (1, 99)]:
        broken = copy.deepcopy(graph)
        broken["edges"].append(dict(source=source, target=target, relation="near", symmetric=True))
        with pytest.raises(ValueError, match="distinct existing"):
            query_graph(broken)
    graph["edges"].append(dict(source=1, target=2, relation="above", symmetric=False))
    with pytest.raises(ValueError, match="cycle"):
        query_graph(graph)


def test_long_vertical_chain_uses_iterative_validation():
    graph = scene([0, 0, 1])
    graph["nodes"] = [{**graph["nodes"][0], "instance_id": i, "centroid_m": [0, 0, -float(i)]} for i in range(1, 5001)]
    graph["edges"] = [dict(source=i, target=i + 1, relation="above", symmetric=False) for i in range(1, 5000)]
    assert query_graph(graph, relation="above", reference_id=3000)["instance_ids"] == [2999]


def test_class_tie_and_omitted_small_instance_are_explicit():
    graph = build_graph(np.zeros((3, 3)), np.array([1, 2, 1]), np.array([1, 1, 2]),
                        {"0": "unknown", "1": "chair", "2": "table"}, min_points=2)
    assert graph["omitted_small_instances"] == [2]
    assert graph["nodes"][0]["semantic_class_tie"] is True
    assert graph["nodes"][0]["semantic_class_counts"] == {"1": 1, "2": 1}
    assert query_graph(graph, label="unknown")["instance_ids"] == [1]
