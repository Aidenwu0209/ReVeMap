"""Measured object graphs and deterministic queries over a completed map.

Relations are geometric hypotheses. Vertical relations require an explicit
world-up vector; camera coordinates are never silently treated as gravity.
"""
from __future__ import annotations

import json
import heapq
from pathlib import Path

import numpy as np

from .contracts import sha256_file


SCHEMA = "revemap.scene_graph.v1"
ALIASES = {
    "椅子": "chair", "桌子": "table", "沙发": "sofa", "床": "bed",
    "柜子": "cabinet", "灯": "lamp", "门": "door", "窗户": "window",
    "书": "book", "显示器": "monitor", "瓶子": "bottle", "杯子": "cup",
    "箱子": "box", "盒子": "box", "纸箱": "box", "窗帘": "curtain",
    "墙": "wall", "墙壁": "wall", "地板": "floor", "地面": "floor",
    "三脚架": "tripod", "相机": "camera", "容器": "container",
    "冰箱": "refrigerator", "fridge": "refrigerator", "refridgerator": "refrigerator",
}


def _label(value):
    value = str(value).strip().lower()
    return ALIASES.get(value, value)


def _basis(up):
    vector = np.array(up, dtype=float, copy=True)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
        raise ValueError("world_up must be a finite nonzero 3-vector")
    vector /= np.linalg.norm(vector)
    seed = np.eye(3)[np.argmin(np.abs(vector))]
    x = np.cross(seed, vector)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(vector, x), vector))


def _box_distance(a, b):
    low_a, high_a = np.asarray(a["aabb_min_m"]), np.asarray(a["aabb_max_m"])
    low_b, high_b = np.asarray(b["aabb_min_m"]), np.asarray(b["aabb_max_m"])
    return float(np.linalg.norm(np.maximum(0., np.maximum(low_a - high_b, low_b - high_a))))


def _candidate_pairs(lows, highs, margin, limit):
    """Sweep the widest axis; cap dense workloads without dropping relations."""
    if len(lows) < 2:
        return
    lows, highs = np.asarray(lows), np.asarray(highs)
    axis = int(np.argmax(np.ptp((lows + highs) / 2, axis=0)))
    active, expiry, checked = {}, [], 0
    for index in np.argsort(lows[:, axis], kind="stable"):
        start = lows[index, axis] - margin
        while expiry and expiry[0][0] < start:
            _, expired = heapq.heappop(expiry)
            active.pop(expired)
        for other in active:
            checked += 1
            if checked > limit:
                raise ValueError("scene graph relation candidate limit exceeded; reduce map scope or explicitly raise max_relation_candidates")
            if np.all(np.maximum(lows[index] - highs[other], lows[other] - highs[index]) <= margin):
                yield min(int(index), int(other)), max(int(index), int(other))
        active[int(index)] = True
        heapq.heappush(expiry, (highs[index, axis], int(index)))


def build_graph(xyz, semantic, instance, classes, *, confidence=None, names=None,
                world_up=None, near_m=.5, support_gap_m=.05, footprint_overlap=.25,
                min_points=1, max_relation_candidates=1_000_000):
    """Build one node per positive instance without changing point labels."""
    xyz = np.asarray(xyz, dtype=float)
    semantic, instance = np.asarray(semantic), np.asarray(instance)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("finite Nx3 map points required")
    n = len(xyz)
    for values in (semantic, instance):
        if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
            raise ValueError("nonnegative integer point labels required")
    if (isinstance(min_points, bool) or not isinstance(min_points, int) or min_points < 1
            or type(max_relation_candidates) is not int or max_relation_candidates < 1
            or not np.isfinite([near_m, support_gap_m, footprint_overlap]).all()
            or near_m <= 0 or support_gap_m <= 0 or not 0 < footprint_overlap <= 1):
        raise ValueError("invalid scene graph thresholds")
    classes = {str(k): str(v) for k, v in classes.items()}
    if classes.get("0") != "unknown" or not set(map(str, np.unique(semantic))).issubset(classes):
        raise ValueError("class dictionary must include unknown and every observed class")
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=float)
        if confidence.shape != (n,) or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence must be finite in [0,1]")
    basis = None if world_up is None else _basis(world_up)
    records = {}
    for row in names or []:
        oid = int(row["instance_id"])
        if oid in records:
            raise ValueError("duplicate instance naming record")
        records[oid] = row
    nodes, projected, omitted = [], {}, []
    order = np.argsort(instance, kind="stable")
    ids, starts, sizes = np.unique(instance[order], return_index=True, return_counts=True)
    for oid, start, size in zip(ids, starts, sizes):
        if oid <= 0:
            continue
        indices = order[start:start + size]
        points = xyz[indices]
        if len(points) < min_points:
            omitted.append(int(oid))
            continue
        labels, counts = np.unique(semantic[indices], return_counts=True)
        winners = labels[counts == counts.max()]
        sid = int(winners[0]) if len(winners) == 1 else 0
        record = records.get(int(oid), {})
        frames = sorted({int(v) for v in record.get("support_frames", [])})
        raw_name = record.get("vlm_name")
        proposed = raw_name.strip().lower() if isinstance(raw_name, str) else "unknown"
        # A name is searchable only when its stored consensus has two frames.
        name = proposed if len(frames) >= 2 and proposed not in ("", "unknown", "none") else None
        node = {
            "instance_id": int(oid), "semantic_id": sid, "label": classes[str(sid)],
            "name": name, "name_source": "multiview_naming_metadata" if name else None,
            "name_category_conflict": bool(name and sid > 0 and _label(name) in {_label(v) for v in classes.values()}
                                            and _label(name) != _label(classes[str(sid)])),
            "semantic_class_counts": {str(int(k)): int(v) for k, v in zip(labels, counts)},
            "semantic_class_tie": len(winners) > 1,
            "point_count": len(points), "centroid_m": points.mean(axis=0).tolist(),
            "aabb_min_m": points.min(axis=0).tolist(), "aabb_max_m": points.max(axis=0).tolist(),
            "class_share": float(counts.max() / len(points)),
            "mean_observation_score": float(confidence[indices].mean()) if confidence is not None else None,
            "evidence_frames": frames, "evidence": record.get("evidence", []),
        }
        nodes.append(node)
        if basis is not None:
            aligned = points @ basis
            projected[int(oid)] = (aligned.min(axis=0), aligned.max(axis=0), aligned.mean(axis=0))
    edges = []
    for i, j in _candidate_pairs([n["aabb_min_m"] for n in nodes], [n["aabb_max_m"] for n in nodes],
                                 near_m, max_relation_candidates):
        a, b = nodes[i], nodes[j]
        distance = _box_distance(a, b)
        if distance <= near_m:
            edges.append({"source": a["instance_id"], "target": b["instance_id"],
                          "relation": "near", "symmetric": True, "aabb_gap_m": distance})
        for outer, inner in ((a, b), (b, a)):
            ol, oh = np.asarray(outer["aabb_min_m"]), np.asarray(outer["aabb_max_m"])
            il, ih = np.asarray(inner["aabb_min_m"]), np.asarray(inner["aabb_max_m"])
            if np.all(ol <= il) and np.all(oh >= ih) and (np.any(ol < il) or np.any(oh > ih)):
                edges.append({"source": outer["instance_id"], "target": inner["instance_id"],
                              "relation": "contains", "symmetric": False,
                              "basis": "strict AABB containment, not physical enclosure"})
    if basis is not None:
        for i, j in _candidate_pairs([projected[n["instance_id"]][0][:2] for n in nodes],
                                     [projected[n["instance_id"]][1][:2] for n in nodes],
                                     0., max_relation_candidates):
            a, b = nodes[i], nodes[j]
            for upper, lower in ((a, b), (b, a)):
                lo_u, hi_u, center_u = projected[upper["instance_id"]]
                lo_l, hi_l, center_l = projected[lower["instance_id"]]
                overlap = np.maximum(0, np.minimum(hi_u[:2], hi_l[:2]) - np.maximum(lo_u[:2], lo_l[:2]))
                denominator = min(np.prod(hi_u[:2] - lo_u[:2]), np.prod(hi_l[:2] - lo_l[:2]))
                share = float(np.prod(overlap) / denominator) if denominator > 0 else 0.
                gap = float(lo_u[2] - hi_l[2])
                if share >= footprint_overlap and gap >= -support_gap_m and center_u[2] > center_l[2]:
                    edge = {"source": upper["instance_id"], "target": lower["instance_id"],
                            "relation": "above", "symmetric": False,
                            "vertical_gap_m": gap, "footprint_overlap": share}
                    edges.append(edge)
                    if abs(gap) <= support_gap_m:
                        edges.append({**edge, "relation": "supported_by"})
    return {
        "schema": SCHEMA, "coordinate_units": "metres",
        "world_up": basis[:, 2].tolist() if basis is not None else None,
        "relation_scope": "AABB/footprint geometric hypotheses, not physical contact verification",
        "score_scope": "observation scores and class shares are not calibrated probabilities",
        "parameters": {"near_m": near_m, "support_gap_m": support_gap_m,
                       "footprint_overlap": footprint_overlap, "min_points": min_points},
        "resource_limits": {"max_relation_candidates_per_sweep": max_relation_candidates,
                            "on_exceeded": "error; no partial graph returned"},
        "nodes": nodes, "edges": edges, "omitted_small_instances": omitted,
        "unknown_points": int(np.sum(instance == 0)),
        "unknown_points_scope": "unassigned instance points; see semantic_unknown_points separately",
        "unassigned_instance_points": int(np.sum(instance == 0)),
        "semantic_unknown_points": int(np.sum(semantic == 0)),
    }


def build_from_artifacts(source, **options):
    """Use the same verified file resolution as the evaluator, including legacy maps."""
    from plyfile import PlyData
    from .artifacts import load_artifacts, verify_artifact_snapshot
    bundle = load_artifacts(Path(source), require_provenance=False)
    paths = bundle["paths"]
    vertices = PlyData.read(paths["map"])["vertex"].data
    required = {"x", "y", "z", "semantic_id", "instance_id"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError("labeled PLY lacks coordinates or instance/semantic IDs")
    classes = json.loads(paths["classes"].read_text())
    names = json.loads(paths["names"].read_text()) if "names" in paths else None
    graph = build_graph(
        np.column_stack([vertices[k] for k in ("x", "y", "z")]),
        vertices["semantic_id"], vertices["instance_id"], classes,
        confidence=vertices["semantic_confidence"] if "semantic_confidence" in vertices.dtype.names else None,
        names=names, **options,
    )
    graph["provenance"] = {"bound": bundle["provenance_bound"],
                           "map_sha256": sha256_file(paths["map"]),
                           "classes_sha256": sha256_file(paths["classes"]),
                           "source": str(Path(source).resolve())}
    receipt = json.loads(paths["result"].read_text())
    graph["provenance"].update(data_source=receipt.get("data_source", "completed_map"),
                               source_scope=receipt.get("scope", receipt.get("pipeline_scope")))
    verify_artifact_snapshot(bundle)
    return graph


def _validate_graph(graph):
    """Validate serialized graphs before queries; never overwrite duplicate IDs."""
    if graph.get("schema") != SCHEMA:
        raise ValueError("unsupported scene graph schema")
    nodes = {}
    for node in graph.get("nodes", []):
        oid = node.get("instance_id")
        if type(oid) is not int or oid <= 0 or oid in nodes:
            raise ValueError("graph requires unique positive instance IDs")
        center = np.asarray(node.get("centroid_m"), dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("graph node requires a finite centroid")
        nodes[oid] = node
    if graph.get("world_up") is not None:
        _basis(graph["world_up"])
    successors = {oid: set() for oid in nodes}
    for edge in graph.get("edges", []):
        source, target, relation = edge.get("source"), edge.get("target"), edge.get("relation")
        if type(source) is not int or type(target) is not int or source not in nodes or target not in nodes or source == target:
            raise ValueError("graph edge requires two distinct existing instances")
        if relation not in ("near", "above", "supported_by", "contains") or edge.get("symmetric") is not (relation == "near"):
            raise ValueError("invalid graph relation or symmetry")
        if relation in ("above", "supported_by"):
            if graph.get("world_up") is None:
                raise ValueError("vertical graph relations require world_up")
            successors[source].add(target)
    # Iterative topological validation also handles long support chains without
    # recursion-depth failures. Symmetric near cycles are valid and excluded.
    incoming = dict.fromkeys(nodes, 0)
    for targets in successors.values():
        for target in targets:
            incoming[target] += 1
    ready = [oid for oid, count in incoming.items() if count == 0]
    visited = 0
    while ready:
        oid = ready.pop()
        visited += 1
        for target in successors[oid]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise ValueError("vertical graph relations contain a cycle")
    return nodes


def query_graph(graph, *, label=None, nearest_to=None, relation=None, reference_id=None):
    """Query labels/names or measured relations; never invent missing objects."""
    nodes = _validate_graph(graph)
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError("label must be nonempty text; omit it to list every object")
    if nearest_to is not None and relation is not None:
        raise ValueError("nearest and relation queries are mutually exclusive")
    if reference_id is not None and relation is None:
        raise ValueError("reference_id is only valid with a relation query")
    if relation not in (None, "near", "above", "below", "supported_by", "contains", "inside"):
        raise ValueError("unsupported relation")
    selected = [n for n in nodes.values() if label is None or _label(label) in
                {_label(n["label"]), _label(n["name"]) if n["name"] else None}]
    reason, evidence = None, []
    reference = nearest_to if nearest_to is not None else reference_id
    if reference is not None and (isinstance(reference, bool) or not isinstance(reference, int) or reference not in nodes):
        raise ValueError("reference instance does not exist")
    if relation is not None and reference is None:
        raise ValueError("relation queries require reference_id")
    if relation in ("above", "below", "supported_by") and graph["world_up"] is None:
        selected, reason = [], "world_up_required"
    elif relation:
        matched = set()
        stored_relation = {"below": "above", "inside": "contains"}.get(relation, relation)
        reverse = relation in ("below", "inside")
        for edge in graph["edges"]:
            if edge["relation"] != stored_relation:
                continue
            source, target = (edge["target"], edge["source"]) if reverse else (edge["source"], edge["target"])
            if target == reference:
                matched.add(source)
                evidence.append(edge)
            elif edge.get("symmetric") and source == reference:
                matched.add(target)
                evidence.append(edge)
        selected = [n for n in selected if n["instance_id"] in matched]
        ids = {n["instance_id"] for n in selected}
        evidence = [e for e in evidence if e["source"] in ids or e["target"] in ids]
    elif nearest_to is not None:
        selected = [n for n in selected if n["instance_id"] != nearest_to]
        if selected:
            distances = {n["instance_id"]: float(np.linalg.norm(
                np.asarray(n["centroid_m"]) - nodes[nearest_to]["centroid_m"])) for n in selected}
            best = min(distances.values())
            # Preserve ties; instance ID must not silently resolve geometric ambiguity.
            selected = [n for n in selected if np.isclose(distances[n["instance_id"]], best, rtol=0, atol=1e-6)]
            evidence = [{"instance_id": n["instance_id"], "reference_id": nearest_to,
                         "centroid_distance_m": distances[n["instance_id"]]} for n in selected]
    selected.sort(key=lambda n: n["instance_id"])
    return {"status": "matched" if selected else "unknown", "reason": reason or (None if selected else "no_matching_object"),
            "instance_ids": [n["instance_id"] for n in selected], "objects": selected,
            "label_matches": [{"instance_id": n["instance_id"],
                               "fields": [field for field, value in (("class_label", n["label"]), ("consensus_name", n["name"]))
                                          if value and _label(value) == _label(label)],
                               "name_category_conflict": n.get("name_category_conflict", False)}
                              for n in selected] if label is not None else [],
            "evidence": evidence, "query": {"label": label, "nearest_to": nearest_to,
                                            "relation": relation, "reference_id": reference_id},
            "provenance": graph.get("provenance", {"bound": False}),
            "relation_scope": graph["relation_scope"], "score_scope": graph["score_scope"]}


def add_commands(commands):
    build = commands.add_parser("build-scene-graph", help="build measured object/relationship graph from a completed result")
    build.add_argument("--result", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--world-up", nargs=3, type=float, metavar=("X", "Y", "Z"))
    build.add_argument("--near-m", type=float, default=.5)
    def run_build(args):
        graph = build_from_artifacts(args.result, world_up=args.world_up, near_m=args.near_m)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(graph, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"graph": str(args.output), "objects": len(graph["nodes"]), "relations": len(graph["edges"])}))
    build.set_defaults(handler=run_build)
    query = commands.add_parser("query-scene", help="find objects, nearest objects, or geometry-supported relations")
    query.add_argument("--graph", type=Path, required=True)
    query.add_argument("--label")
    group = query.add_mutually_exclusive_group()
    group.add_argument("--nearest-to", type=int)
    group.add_argument("--relation", choices=("near", "above", "below", "supported_by", "contains", "inside"))
    group.add_argument("--question")
    group.add_argument("--query-json", type=Path)
    query.add_argument("--reference-id", type=int)
    def run_query(a):
        graph = json.loads(a.graph.read_text())
        if a.question is not None or a.query_json is not None:
            if a.label is not None or a.reference_id is not None:
                raise ValueError("Do not combine a question/query plan with legacy filters")
            from .scene_query import execute_query
            result = execute_query(graph, question=a.question,
                                   plan=json.loads(a.query_json.read_text()) if a.query_json else None)
        else:
            result = query_graph(graph, label=a.label, nearest_to=a.nearest_to,
                                 relation=a.relation, reference_id=a.reference_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    query.set_defaults(handler=run_query)
