"""Score a sealed native map on every ScanNet reference mesh vertex at 5 cm.

Outputs PQ, semantic mIoU, geometry precision and coverage in percent. Average
PQ and mIoU equally across scans to reproduce the 142-scan summary.
"""
import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from panoptic_observations import ALIASES, CLASSES


NYU40_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
UNKNOWN = 20


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def xyz(vertices):
    return np.column_stack([vertices[key] for key in "xyz"]).astype(np.float64)


def ground_truth(scan, table):
    name = scan.name
    vertices = PlyData.read(scan / (name + "_vh_clean_2.ply"))["vertex"].data
    segment = np.asarray(json_file(scan / (name + "_vh_clean_2.0.010000.segs.json"))["segIndices"])
    if len(vertices) != len(segment):
        raise ValueError("ScanNet geometry and segment indices differ in length")
    groups = json_file(scan / (name + ".aggregation.json"))["segGroups"]
    with table.open(encoding="utf-8-sig", newline="") as inp:
        lookup = {r["raw_category"]: int(r["nyu40id"]) for r in csv.DictReader(inp, delimiter="\t")}
    nyu_to_class = dict(zip(NYU40_IDS, range(len(CLASSES))))
    by_label = {}
    segment_to_instance = np.zeros(int(segment.max()) + 1, dtype=np.int64)
    for group in groups:
        by_label.setdefault(group["label"], []).extend(group["segments"])
        segment_to_instance[group["segments"]] = int(group["objectId"]) + 1
    segment_to_semantic = np.full(len(segment_to_instance), UNKNOWN)
    for label, indices in by_label.items():
        segment_to_semantic[indices] = nyu_to_class.get(lookup[label], UNKNOWN)
    return xyz(vertices), segment_to_semantic[segment], segment_to_instance[segment]


def original_anchor(sens, trajectory):
    if trajectory.get("matrix_convention") != "T_world_camera_m":
        raise ValueError("Expected T_world_camera_m in the original trajectory")
    estimates = {int(p["frame_id"]): p for p in trajectory["poses"] if p.get("valid", True)}
    with sens.open("rb") as stream:
        if struct.unpack("<I", stream.read(4))[0] != 4:
            raise ValueError("Expected a ScanNet v4 .sens")
        sensor_name_size = struct.unpack("<Q", stream.read(8))[0]
        stream.seek(sensor_name_size + 4 * 64 + 24, 1)
        _, frame_count = struct.unpack("<fQ", stream.read(12))
        for frame in range(frame_count):
            reference = np.frombuffer(stream.read(64), dtype="<f4").reshape(4, 4)
            stream.seek(16, 1)
            color_size, depth_size = struct.unpack("<QQ", stream.read(16))
            if frame in estimates and np.isfinite(reference).all():
                estimate = np.asarray(estimates[frame]["T_world_camera_m"], dtype=float).reshape(4, 4)
                if np.isfinite(estimate).all():
                    return frame, reference @ np.linalg.inv(estimate)
            stream.seek(color_size + depth_size, 1)
    raise ValueError("No frame has both a valid estimate and a finite reference pose")


def prediction(path, classes):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as labels:
            points, semantic, instance = (labels[k] for k in ("xyz", "semantic", "instance"))
    else:
        if classes is None:
            raise ValueError("--classes is required for a labeled PLY map")
        vertex = PlyData.read(path)["vertex"].data
        points, instance = xyz(vertex), vertex["instance_id"]
        semantic = np.full(len(vertex), UNKNOWN)
        for label_id, name in json_file(classes).items():
            label = name.strip().lower()
            label = ALIASES.get(label, label)
            if label in CLASSES:
                semantic[vertex["semantic_id"] == int(label_id)] = CLASSES.index(label)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or semantic.shape != (len(points),) or instance.shape != semantic.shape
            or not np.isfinite(points).all() or semantic.dtype.kind not in "iu"
            or instance.dtype.kind not in "iu" or np.any((semantic < 0) | (semantic > UNKNOWN))):
        raise ValueError("Invalid point coordinates or panoptic label arrays")
    return points.astype(np.float64), semantic.astype(np.int64), instance.astype(np.int64)


def metrics(ps, pi, gs, gi):
    valid = gs != UNKNOWN
    ps, pi, gs, gi = (v[valid].copy() for v in (ps, pi, gs, gi))
    for sem, ids in ((ps, pi), (gs, gi)):
        ids[(sem >= 2) & (ids == 0)] = -1
        ids[sem < 2] = 0
    count = UNKNOWN + 1
    conf = np.bincount(ps * count + gs, minlength=count * count).reshape(count, count)
    per_class, hits = {}, []
    for label, name in enumerate(CLASSES):
        pred = (ps == label) & (pi >= 0)
        ref = (gs == label) & (gi >= 0)
        pred_ids, pred_size = np.unique(pi[pred], return_counts=True)
        ref_ids, ref_size = np.unique(gi[ref], return_counts=True)
        common = pred & ref
        pairs, overlap = np.unique(np.column_stack((pi[common], gi[common])), axis=0, return_counts=True)
        union = pred_size[np.searchsorted(pred_ids, pairs[:, 0])] + ref_size[np.searchsorted(ref_ids, pairs[:, 1])] - overlap
        intersection_iou = overlap / union
        matched = intersection_iou > 0.5
        tp = int(matched.sum())
        fp, fn = len(pred_ids) - tp, len(ref_ids) - tp
        denominator = tp + (fp + fn) / 2
        intersection = conf[label, label]
        sem_union = conf[label].sum() + conf[:, label].sum() - intersection
        per_class[name] = dict(tp=tp, fp=fp, fn=fn,
                               pq_pct=100 * float(intersection_iou[matched].sum()) / denominator if denominator else 0.0,
                               miou_pct=100 * float(intersection) / sem_union if sem_union else 0.0)
        if label >= 2:
            hits.extend(pairs[matched, 1].tolist())
    present = [per_class[CLASSES[label]] for label in np.unique(gs)]
    if not present:
        raise ValueError("The reference has no ScanNet20 class")
    return dict(pq_pct=float(np.mean([r["pq_pct"] for r in present])),
                miou_pct=float(np.mean([r["miou_pct"] for r in present])),
                thing_tp=len(hits), thing_fp=sum(per_class[c]["fp"] for c in CLASSES[2:]),
                thing_fn=sum(per_class[c]["fn"] for c in CLASSES[2:]),
                matched_gt_thing_ids=sorted(hits), per_class=per_class)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, type=Path, help="Official sceneXXXX_YY directory with .sens and annotations")
    parser.add_argument("--label-tsv", required=True, type=Path, help="Official scannetv2-labels.combined.tsv")
    parser.add_argument("--trajectory", required=True, type=Path, help="Original estimated trajectory.json for the evaluation anchor")
    parser.add_argument("--prediction", required=True, type=Path, help="NPZ with xyz, ScanNet20 semantic and instance, or a labeled PLY")
    parser.add_argument("--classes", type=Path, help="PLY semantic_id to class-name JSON")
    args = parser.parse_args()
    gx, gs, gi = ground_truth(args.scan, args.label_tsv)
    px, ps, pi = prediction(args.prediction, args.classes)
    frame, transform = original_anchor(args.scan / (args.scan.name + ".sens"), json_file(args.trajectory))
    px = px @ transform[:3, :3].T + transform[:3, 3]
    gt_dist, nearest = cKDTree(px).query(gx)
    covered = gt_dist <= 0.05
    result = metrics(np.where(covered, ps[nearest], UNKNOWN), np.where(covered, pi[nearest], -1), gs, gi)
    pred_dist = cKDTree(gx).query(px)[0]
    result.update(scene=args.scan.name, points=len(px), ref_vertices=len(gx), alignment_frame=frame,
                  coverage_pct=100 * covered[gs != UNKNOWN].mean(),
                  precision_pct=100 * (pred_dist <= 0.05).mean(),
                  mean_symmetric_nn_m=(pred_dist.mean() + gt_dist.mean()) / 2)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
