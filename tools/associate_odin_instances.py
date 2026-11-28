"""Refine a panoptic map with matching instance evidence from two ODIN view samplings."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from odin_runtime import publish, sha256
from prepare_odin_geometry import vertices


def direct_queries(path, map_count, carrier_hash):
    receipt = path.parent / "seal.json"
    if receipt.is_file():
        info = json.loads(receipt.read_text(encoding="utf-8"))
        expected_file = info.get("files", {}).get(path.name) or info.get("query_sha")
        expected_map = info.get("inputs", {}).get("map") or info.get("input_files", {}).get("map.ply")
        if (expected_file and expected_file != sha256(path)) or (expected_map and expected_map != carrier_hash):
            raise ValueError(f"Query receipt does not match the current map and query file: {path}")
    with np.load(path, allow_pickle=False) as data:
        near = data["map_to_mesh"]
        count = int(data["mesh_count"])
        if near.shape != (map_count,) or np.any(near < 0) or np.any(near >= count):
            raise ValueError(f"Invalid map-to-mesh indices in {path}")
        classes, scores = data["classes"], data["scores"]
        packed = data["masks"]
        if packed.shape != (len(scores), (count + 7) // 8) or len(classes) != len(scores):
            raise ValueError(f"Inconsistent query arrays in {path}")
        if not np.isfinite(scores).all() or np.any(classes < 2) or np.any(classes >= 20):
            raise ValueError(f"Invalid ScanNet20 query class or confidence in {path}")
        sem = data["semantic"][near].copy()
        ids = np.zeros(map_count, dtype=np.int32)
        next_id = 1
        for query in np.argsort(-scores, kind="stable"):
            if scores[query] < .55:
                break
            mask = np.unpackbits(packed[query], count=count)[near].astype(bool)
            free = mask & (ids == 0)
            if free.sum() < max(20, .5 * mask.sum()):
                continue
            sem[free], ids[free] = classes[query], next_id
            next_id += 1
    return sem, ids


def compact_instances(sem, instance):
    positive = (sem >= 2) & (sem < 20) & (instance > 0)
    compact = np.zeros(len(instance), dtype=np.int64)
    pairs = np.column_stack((instance[positive], sem[positive]))
    keys, inverse = np.unique(pairs, axis=0, return_inverse=True)
    compact[positive] = inverse + 1
    return compact, np.r_[0, keys[:, 0]]


def overlaps(a, class_a, b, class_b):
    na = int(a.max()) + 1
    nb = int(b.max()) + 1
    matrix = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb)
    size_a, size_b = matrix.sum(1), matrix.sum(0)
    labels_a, labels_b = np.zeros(na, int), np.zeros(nb, int)
    labels_a[a], labels_b[b] = class_a, class_b
    same = (labels_a[:, None] == labels_b) & (matrix > 0)
    same[0], same[:, 0] = False, False
    iou = matrix / np.maximum(1, size_a[:, None] + size_b - matrix)
    iou[~same] = 0
    return matrix, size_a, size_b, same, iou


def associate(xyz, base_sem, base_ids, view19, cap64, probabilities, distance):
    sem = base_sem.copy()
    ids = base_ids.astype(np.int64, copy=True)
    original, original_ids = compact_instances(base_sem, base_ids)
    sem64, region64 = cap64
    sem19, region19 = view19
    region64, region19 = region64.astype(np.int64), region19.astype(np.int64)
    matrix, base_size, size64, same, iou = overlaps(original, base_sem, region64, sem64)
    *_, cross_iou = overlaps(region64, sem64, region19, sem19)
    witness = cross_iou.argmax(1)
    strength = cross_iou.max(1)
    query_fraction = matrix / np.maximum(size64[None, :], 1)
    base_fraction = matrix / np.maximum(base_size[:, None], 1)
    next_id = int(ids.max()) + 1
    altered = set()
    counts = {"splits": 0, "joins": 0, "extended_vertices": 0}

    for i in range(1, len(base_size)):
        candidates = np.where(same[i] & (query_fraction[i] >= .65) & (base_fraction[i] >= .10) & (strength >= .35))[0]
        if len(candidates) < 2 or base_fraction[i, candidates].sum() < .55:
            continue
        belongs = original == i
        seeds = belongs & np.isin(region64, candidates)
        target_id = np.zeros(len(size64), dtype=np.int64)
        target_id[candidates] = np.r_[original_ids[i], np.arange(next_id, next_id + len(candidates) - 1)]
        ids[seeds] = target_id[region64[seeds]]
        remaining = np.flatnonzero(belongs & ~seeds)
        if len(remaining):
            d, nn = cKDTree(xyz[seeds]).query(xyz[remaining])
            close = d <= .20
            ids[remaining[close]] = ids[seeds][nn[close]]
        next_id += len(candidates) - 1
        altered.add(i)
        counts["splits"] += 1

    for j in range(1, len(size64)):
        if strength[j] < .50:
            continue
        components = np.where(same[:, j] & (base_fraction[:, j] >= .70) & (query_fraction[:, j] >= .10))[0]
        if len(components) < 2 or query_fraction[components, j].sum() < .60 or not altered.isdisjoint(components):
            continue
        leader = components[np.argmax(matrix[components, j])]
        ids[np.isin(original, components)] = original_ids[leader]
        altered.update(components)
        counts["joins"] += 1

    best_base, best_query = iou.argmax(0), iou.argmax(1)
    for j in range(1, len(size64)):
        i = best_base[j]
        if i == 0 or best_query[i] != j or iou[i, j] < .30 or strength[j] < .45 or i in altered:
            continue
        c = int(base_sem[np.flatnonzero(original == i)[0]])
        confirmed = (region64 == j) & (region19 == witness[j])
        available = (base_ids <= 0) | (base_sem < 2) | (base_sem == 20)
        add = confirmed & available & (distance < .08) & (probabilities[:, c] >= 150)
        ids[add], sem[add] = original_ids[i], c
        counts["extended_vertices"] += int(add.sum())
    if ids.max() > np.iinfo(np.int32).max:
        raise ValueError("Resulting instance IDs exceed the int32 output format")
    return sem.astype("u1"), ids.astype("i4"), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("map", "base", "point-evidence", "view19", "cap64", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    xyz = vertices(args.map)
    with np.load(args.base, allow_pickle=False) as f:
        sem, ins = f["semantic"].copy(), f["instance"].copy()
    if sem.shape != (len(xyz),) or ins.shape != sem.shape or not np.issubdtype(ins.dtype, np.integer):
        raise ValueError("The base panoptic arrays must have one label per map point")
    with np.load(args.point_evidence, allow_pickle=False) as f:
        near = f["nearest"]
        distance = f["distance"]
        evidence = f["proba"]
        if near.shape != sem.shape or distance.shape != sem.shape or evidence.ndim != 2 or evidence.shape[1] != 20:
            raise ValueError("Expected PTv3 proba[Nmesh,20], nearest[Nmap] and distance[Nmap]")
        if np.any(near < 0) or np.any(near >= len(evidence)) or evidence.dtype != np.uint8:
            raise ValueError("Expected valid PTv3 mesh indices and uint8 probabilities scaled to 0–255")
        probability = evidence[near]
    map_hash = sha256(args.map)
    view19 = direct_queries(args.view19, len(xyz), map_hash)
    cap64 = direct_queries(args.cap64, len(xyz), map_hash)
    sem, ins, counts = associate(xyz, sem, ins, view19, cap64, probability, distance)
    inputs = {k: sha256(getattr(args, k)) for k in ("map", "base", "point_evidence", "view19", "cap64")}
    publish(args.output, "combined.npz", dict(semantic=sem, instance=ins), dict(inputs=inputs, driver=sha256(Path(__file__))),
            dict(map_points=len(sem), changes=counts, inputs=inputs))
    print(args.output / "combined.npz", counts)


if __name__ == "__main__":
    main()
