"""Verified unknown-instance surface fill and grounded open-vocabulary naming.

This is the promoted *frozen geometry* route. Evidence is per original RGB-D
frame, with projected point IDs and independent SAM3 masks. It is separate from
the raw-window VLM metadata route and never consumes GT in label assignment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import numpy as np

from .common import parse_label, read, sha, write


def point_ids(values, n):
    ids = np.asarray(values)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= n):
        raise ValueError("invalid projected map point IDs")
    return np.unique(ids)


def validate_labels(labels):
    n = len(labels["instance"])
    for key in ("semantic", "instance", "confidence"):
        if labels[key].shape != (n,):
            raise ValueError("label shape mismatch")
    for key in ("semantic", "instance"):
        if not np.issubdtype(labels[key].dtype, np.integer) or np.any(labels[key] < 0):
            raise ValueError("nonnegative integer labels required")
    if not np.isfinite(labels["confidence"]).all():
        raise ValueError("non-finite confidence")
    return n


def select_pair_mask(candidates, visible, labels, owner):
    anchor = visible[labels["instance"][visible] == owner]
    rows = []
    for i, candidate in enumerate(candidates):
        points = point_ids(candidate["points"], len(labels["instance"]))
        if not np.isin(points, visible).all():
            raise ValueError("mask support includes invisible points")
        quality = float(candidate["score"])
        if not np.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("invalid SAM3 mask quality")
        coverage = len(np.intersect1d(points, anchor)) / max(1, len(anchor))
        if quality >= .5 and coverage >= .5:
            rows.append((coverage, quality, i, points))
    rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return rows[0][3] if rows else np.empty(0, np.int64)


def verified_fill(base, original, views):
    """P2: independent prompt agreement in >=2 frames, unknown points only."""
    n = validate_labels(base)
    if validate_labels(original) != n:
        raise ValueError("original/base point count differs")
    if not np.array_equal(base["semantic"], original["semantic"]):
        raise ValueError("T1 input changed original semantic labels")
    old = original["instance"] > 0
    if not np.array_equal(base["instance"][old], original["instance"][old]):
        raise ValueError("T1 input changed original instance ownership")
    grouped, seen = defaultdict(list), set()
    for row in views:
        pair = tuple(row["pair"])
        if len(pair) != 2 or pair[0] >= pair[1] or any(x <= int(original["instance"].max(initial=0)) for x in pair):
            raise ValueError("only two ordered newly born instance IDs may be joined")
        if any(not np.any(base["instance"] == x) or np.any(base["semantic"][base["instance"] == x] > 0) for x in pair):
            raise ValueError("pair must refer to existing unknown instances")
        key = (pair, row["frame_id"])
        if key in seen:
            raise ValueError("duplicate original frame in pair evidence")
        seen.add(key)
        visible = point_ids(row["visible"], n)
        left = select_pair_mask(row["left"], visible, base, pair[0])
        right = select_pair_mask(row["right"], visible, base, pair[1])
        shared = np.intersect1d(left, right)
        anchors = [visible[base["instance"][visible] == k] for k in pair]
        covers = all(len(a) >= 30 and len(np.intersect1d(p, a))/len(a) >= .5
                     for a in anchors for p in (left, right))
        iou = len(shared)/max(1, len(left)+len(right)-len(shared))
        owned = float(np.mean(original["instance"][shared] > 0)) if len(shared) else 0.
        stuff = float(np.mean(np.isin(original["semantic"][shared], [10, 19]))) if len(shared) else 0.
        passed = covers and len(shared) >= 30 and iou >= .7 and owned <= .2 and stuff <= .5
        grouped[pair].append({"frame_id": row["frame_id"], "passed": bool(passed), "points": shared,
                              "point_iou": iou, "known_owned_fraction": owned, "stuff_fraction": stuff})
    accepted = {pair: rows for pair, rows in grouped.items() if sum(r["passed"] for r in rows) >= 2}
    members = [x for pair in accepted for x in pair]
    if len(set(members)) != len(members):
        raise ValueError("transitive joins require independent verification; accepted pairs must be disjoint")
    labels = {k: v.copy() for k, v in base.items()}
    claims, owners = np.zeros(n, np.uint32), np.zeros(n, np.int64)
    for (a, b), rows in accepted.items():
        labels["instance"][labels["instance"] == b] = a
        support = np.zeros(n, np.uint32)
        for row in rows:
            if row["passed"]:
                support[row["points"]] += 1
        eligible = (support >= 2) & (base["instance"] == 0) & (base["semantic"] == 0)
        claims[eligible] += 1
        owners[eligible] = a
    selected = claims == 1
    labels["instance"][selected] = owners[selected]
    audit = {"accepted_pairs": [list(p) for p in accepted], "added_points": int(selected.sum()),
             "ambiguous_points_rejected": int(np.sum(claims > 1)),
             "views": [{"pair": list(pair), **{k: v for k, v in row.items() if k != "points"}}
                       for pair, rows in grouped.items() for row in rows], "GT_used": False}
    return labels, audit


def name_consensus(records):
    """One context-crop vote per original fusion frame; validation never votes."""
    votes, seen = defaultdict(Counter), set()
    for row in records:
        if row.get("role") != "fusion" or row.get("crop_mode", "context") != "context":
            continue
        key = (row["instance_id"], row["frame_id"])
        if key in seen:
            raise ValueError("duplicate context naming vote from one original frame")
        seen.add(key)
        label, valid = parse_label(row.get("raw_response", row.get("label", "unknown")))
        if valid and label != "unknown":
            votes[row["instance_id"]][label] += 1
    result = {}
    for instance, counts in votes.items():
        ordered = sorted(counts, key=lambda label: (-counts[label], label))
        if counts[ordered[0]] >= 2 and (len(ordered) < 2 or counts[ordered[0]] > counts[ordered[1]]):
            result[instance] = ordered[0]
    return result


def grounded_names(base, classes, naming, grounding):
    """Only >=2 independent SAM3 text confirmations may fill an unknown object."""
    n = validate_labels(base)
    proposed = name_consensus(naming)
    allowed_frames = defaultdict(set)
    for row in naming:
        if row.get('role') == 'fusion' and row.get('crop_mode', 'context') == 'context':
            allowed_frames[row['instance_id']].add(row['frame_id'])
    support, seen = defaultdict(set), set()
    for row in grounding:
        oid, fid, label = row["instance_id"], row["frame_id"], row["label"]
        if row["role"] != "fusion" or proposed.get(oid) != label:
            continue
        if fid not in allowed_frames[oid]:
            raise ValueError('SAM3 confirmation frame is outside the naming view plan')
        key = (oid, fid, label)
        if key in seen:
            raise ValueError("duplicate SAM3 text confirmation frame")
        seen.add(key)
        visible = point_ids(row["visible"], n)
        reference = point_ids(row["reference"], n)
        if not np.isin(reference, visible).all():
            raise ValueError("reference includes invisible points")
        anchor = visible[base["instance"][visible] == oid]
        for candidate in row["candidates"]:
            points = point_ids(candidate["points"], n)
            if not np.isin(points, visible).all():
                raise ValueError("text mask includes invisible points")
            score = float(candidate["score"])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid SAM3 text score")
            coverage = len(np.intersect1d(points, anchor))/max(1, len(anchor))
            shared = len(np.intersect1d(points, reference))
            iou = shared/max(1, len(points)+len(reference)-shared)
            other = float(np.mean((base["instance"][points] > 0) & (base["instance"][points] != oid))) if len(points) else 0.
            if score >= .5 and coverage >= .5 and iou >= .5 and other <= .2 and len(points) >= 30:
                support[oid].add(fid)
                break
    accepted = {oid: proposed[oid] for oid, frames in support.items() if len(frames) >= 2}
    classes = {str(k): str(v) for k, v in classes.items()}
    if classes.get("0") != "unknown" or not set(map(int, np.unique(base["semantic"]))).issubset(set(map(int, classes))):
        raise ValueError("class dictionary must cover all existing semantic IDs")
    for name in sorted(set(accepted.values())):
        if name not in classes.values():
            classes[str(max(map(int, classes))+1)] = name
    lookup = {name: int(k) for k, name in classes.items()}
    labels = {k: v.copy() for k, v in base.items()}
    strength = np.zeros(n, np.float32)
    audit = []
    for oid, name in sorted(accepted.items()):
        points = base["instance"] == oid
        if not points.any() or np.any(base["semantic"][points] > 0):
            # A known or mixed object cannot be overwritten by a language model.
            continue
        total_frames = {x["frame_id"] for x in naming if x["instance_id"] == oid and x.get("role") == "fusion" and x.get("crop_mode", "context") == "context"}
        value = len(support[oid])/max(1, len(total_frames))
        if value > 1:
            raise ValueError("confirmation includes frames outside the naming view plan")
        labels["semantic"][points] = lookup[name]
        strength[points] = value
        audit.append({"instance_id": oid, "label": name, "points": int(points.sum()),
                      "support_frames": sorted(support[oid]), "support_fraction": value})
    return labels, classes, strength, audit


def export_map(xyz, labels, classes, output, support):
    from plyfile import PlyData, PlyElement
    output = Path(output)
    vertex = np.empty(len(xyz), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("semantic_id", "i4"),
                      ("instance_id", "i4"), ("confidence", "f4"), ("naming_support", "f4")])
    for axis, key in enumerate(("x", "y", "z")):
        vertex[key] = xyz[:, axis]
    for key, source in (("semantic_id", "semantic"), ("instance_id", "instance"), ("confidence", "confidence")):
        vertex[key] = labels[source]
    vertex["naming_support"] = support
    for key, multiplier in zip(("red", "green", "blue"), (73, 151, 199)):
        vertex[key] = np.where(labels["semantic"] > 0, 50 + labels["semantic"].astype(np.int64)*multiplier % 206, 90)
    path = output / "semantic_labeled.ply"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)
    check = PlyData.read(path)["vertex"].data
    if not all(np.array_equal(vertex[k], check[k]) for k in vertex.dtype.names):
        raise RuntimeError("PLY readback differs")
    objects = []
    for oid in sorted(int(x) for x in np.unique(labels["instance"]) if x > 0):
        on = labels["instance"] == oid
        values, counts = np.unique(labels["semantic"][on], return_counts=True)
        sid = int(values[counts.argmax()])
        objects.append({"instance_id": oid, "semantic_id": sid, "semantic_name": classes[str(sid)],
                        "point_count": int(on.sum()), "center": xyz[on].mean(0).tolist()})
    write(output / "objects.json", objects)


def run(args):
    """Read a hash-checked portable bundle; never import experiment directories."""
    bundle = read(args.bundle)
    root = args.bundle.resolve().parent
    hashes = bundle["files"]
    def load_npz(name):
        path = root / name
        if name not in hashes or sha(path) != hashes[name]:
            raise ValueError("bundle digest mismatch: " + name)
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}
    for name, digest in hashes.items():
        if sha(root/name) != digest:
            raise ValueError("bundle changed: " + name)
    base = load_npz(bundle["base"])
    original = load_npz(bundle["original"])
    xyz = load_npz(bundle["target"])["xyz"]
    if xyz.shape != (validate_labels(base), 3) or not np.isfinite(xyz).all():
        raise ValueError("invalid geometry")
    if hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest() != bundle["geometry_xyz_sha256"]:
        raise ValueError("geometry point order/precision changed")
    identity = None
    if 'identity_queries' in bundle:
        from .identity import anchored_proposals
        queries = []
        for row in bundle['identity_queries']:
            data = load_npz(row['evidence'])
            queries.append({**row, 'visible': data['visible'],
                'candidates': [{'points': data[f'points_{i}'], 'score': score}
                               for i, score in enumerate(data['scores'])]})
        base, identity = anchored_proposals(base, queries, bundle['seed_regions'])
    views = []
    for row in bundle.get("pair_views", []):
        data = load_npz(row["evidence"])
        views.append({"pair": row["pair"], "frame_id": row["frame_id"], "visible": data["visible"],
                      **{side: [{"points": data[f"{side}_{i}"], "score": score} for i, score in enumerate(data[f"{side}_scores"])]
                         for side in ("left", "right")}})
    labels, surface = verified_fill(base, original, views) if args.surface == "verified" else (
        {k: v.copy() for k, v in base.items()}, {"disabled": True})
    names, ground = [], []
    classes = bundle["classes"]
    if args.vlm != "none":
        entry = bundle.get("models", {}).get(args.vlm)
        if entry is None:
            raise ValueError("bundle has no cached evidence for the selected VLM; run it first")
        names = entry["naming"]
        for row in entry["grounding"]:
            data = load_npz(row["evidence"])
            ground.append({**row, "visible": data["visible"], "reference": data["reference"],
                           "candidates": [{"points": data[f"points_{i}"], "score": score} for i, score in enumerate(data["scores"])]})
    final, classes, strength, naming = grounded_names(labels, classes, names, ground)
    for key in base:
        if key not in ("semantic", "instance") and not np.array_equal(final[key], base[key]):
            raise RuntimeError("unchanged field modified: " + key)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "map_labels.npz", **final)
    write(output / "classes.json", classes)
    export_map(xyz, final, classes, output, strength)
    result = {"status": "completed", "identity": identity, "surface": surface, "naming": naming, "vlm": args.vlm,
              "points": len(xyz), "new_named_points": int(np.sum(final["semantic"] != base["semantic"])),
              "original_positive_semantics_preserved": bool(np.array_equal(final["semantic"][original["semantic"] > 0], original["semantic"][original["semantic"] > 0])),
              "GT_used": False, "new_model_inference": False, "geometry_modified": False,
              "bundle_sha256": sha(args.bundle), "naming_support_is_calibrated_probability": False}
    write(output / "result.json", result)
    return result
