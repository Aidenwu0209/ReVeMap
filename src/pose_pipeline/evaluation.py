"""Reference-vertex diagnostics; deliberately not an official mIoU/AP protocol."""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _validate(xyz, semantic, instance, name):
    xyz = np.asarray(xyz)
    semantic, instance = np.asarray(semantic), np.asarray(instance)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError(name + " requires finite Nx3 coordinates")
    for array in (semantic, instance):
        if array.shape != (len(xyz),) or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
            raise ValueError(name + " requires nonnegative integer labels")
    return xyz, semantic, instance


def _instances(pred_semantic, pred_instance, reference_semantic, reference_instance,
               projected_semantic, projected_instance, threshold, overlap_fraction):
    predicted_ids = np.unique(pred_instance[pred_instance > 0])
    reference_ids = np.unique(reference_instance[reference_instance > 0])
    categories = {}
    partial_unknown = []
    for iid in predicted_ids:
        values = np.unique(pred_semantic[pred_instance == iid])
        positive = values[values > 0]
        categories[int(iid)] = int(positive[0]) if len(positive) == 1 else (0 if not len(positive) else -1)
        if 0 in values and len(positive) == 1:
            partial_unknown.append(int(iid))
    gt_categories = {}
    for iid in reference_ids:
        values = np.unique(reference_semantic[reference_instance == iid])
        if len(values) != 1:
            raise ValueError("reference instance must have one semantic category")
        gt_categories[int(iid)] = int(values[0])
    overlaps = np.zeros((len(predicted_ids), len(reference_ids)), dtype=np.int64)
    pred_index = {int(value): i for i, value in enumerate(predicted_ids)}
    gt_index = {int(value): i for i, value in enumerate(reference_ids)}
    for (p, g), count in Counter(zip(projected_instance.tolist(), reference_instance.tolist())).items():
        if p in pred_index and g in gt_index:
            overlaps[pred_index[p], gt_index[g]] = count
    pred_sizes = np.array([np.sum(projected_instance == iid) for iid in predicted_ids])
    gt_sizes = np.array([np.sum(reference_instance == iid) for iid in reference_ids])
    unions = pred_sizes[:, None] + gt_sizes[None, :] - overlaps
    iou = np.divide(overlaps, unions, out=np.zeros_like(overlaps, dtype=float), where=unions > 0)
    same_class = np.array([[categories[int(p)] > 0 and categories[int(p)] == gt_categories[int(g)] for g in reference_ids]
                           for p in predicted_ids], dtype=bool).reshape(iou.shape)
    def match(valid, reference_count):
        matches = []
        if valid.size:
            # Maximize cardinality first, then total IoU. Merely maximizing IoU
            # can sacrifice a valid match and incorrectly lower precision/recall.
            benefit = valid * (min(iou.shape) + 1 + iou)
            rows, columns = linear_sum_assignment(benefit, maximize=True)
            matches = [{"predicted_id": int(predicted_ids[p]), "reference_id": int(reference_ids[g]),
                        "iou": float(iou[p, g])} for p, g in zip(rows, columns) if valid[p, g]]
        return {"predicted_instances": len(predicted_ids), "reference_instances": reference_count,
                "matched_instances": len(matches), "precision": _fraction(len(matches), len(predicted_ids)),
                "recall": _fraction(len(matches), reference_count), "matches": matches}
    class_aware = match((iou >= threshold) & same_class, sum(cid > 0 for cid in gt_categories.values()))
    class_agnostic = match(iou >= threshold, len(reference_ids))
    substantial = (overlaps > 0) & (overlaps >= overlap_fraction * gt_sizes[None, :])
    splits = {str(int(reference_ids[g])): [int(predicted_ids[p]) for p in np.flatnonzero(substantial[:, g])]
              for g in range(len(reference_ids)) if substantial[:, g].sum() > 1}
    merges = {str(int(predicted_ids[p])): [int(reference_ids[g]) for g in np.flatnonzero(substantial[p])]
              for p in range(len(predicted_ids)) if substantial[p].sum() > 1}
    return {"iou_threshold": threshold, "matching": "class-aware maximum-cardinality one-to-one; IoU tie-break",
            "support": "all GT vertices, nearest predicted vertex within distance threshold; no confidence-ranked AP",
            "class_policy": "one positive label plus unknown uses that label; entirely unknown or conflicting positive labels cannot match class-aware; all predicted instances remain in the precision denominator",
            "unknown_predicted_instances": [iid for iid, cid in categories.items() if cid == 0],
            "ambiguous_semantic_instances": [iid for iid, cid in categories.items() if cid < 0],
            "partially_named_instances": partial_unknown,
            **class_aware, "class_agnostic": class_agnostic,
            "split_reference_objects": splits, "merged_predicted_objects": merges,
            "split_merge_min_fraction_of_reference_object": overlap_fraction}


def evaluate_points(pred_xyz, pred_semantic, pred_instance, reference_xyz,
                    reference_semantic, reference_instance, *, distance_threshold=.05,
                    instance_iou_threshold=.5, split_merge_fraction=.1, before=None):
    """Compute transparent diagnostics on exact shared class IDs.

GT class 0 means outside the declared evaluation vocabulary. Geometry coverage
still includes all GT vertices. ``before`` is same-order fixed geometry labels
with keys ``semantic`` and ``instance``; ID changes are reported separately.
"""
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("distance threshold must be positive")
    if not 0 < instance_iou_threshold <= 1 or not 0 < split_merge_fraction <= 1:
        raise ValueError("IoU and split/merge thresholds must be in (0, 1]")
    xyz, sem, inst = _validate(pred_xyz, pred_semantic, pred_instance, "prediction")
    gtxyz, gtsem, gtinst = _validate(reference_xyz, reference_semantic, reference_instance, "reference")
    if not len(gtxyz):
        raise ValueError("reference contains no vertices")
    distance, nearest = cKDTree(gtxyz).query(xyz, workers=1)
    near = distance <= distance_threshold
    supported = near & (gtsem[nearest] > 0)
    labeled = supported & (sem > 0)
    projected_sem = np.zeros(len(gtxyz), dtype=np.int64)
    projected_inst = np.zeros(len(gtxyz), dtype=np.int64)
    covered = np.zeros(len(gtxyz), dtype=bool)
    if len(xyz):
        back_distance, back_nearest = cKDTree(xyz).query(gtxyz, workers=1)
        covered = back_distance <= distance_threshold
        projected_sem[covered] = sem[back_nearest[covered]]
        projected_inst[covered] = inst[back_nearest[covered]]
    instance = _instances(sem, inst, gtsem, gtinst, projected_sem, projected_inst,
                          instance_iou_threshold, split_merge_fraction)
    per_class = {}
    for cid in np.unique(gtsem[gtsem > 0]):
        target = gtsem == cid
        predicted = projected_sem == cid
        intersection = int(np.sum(target & predicted))
        union = int(np.sum(target | predicted))
        per_class[str(int(cid))] = {"reference_vertices": int(target.sum()),
            "covered_reference_vertices": int(np.sum(target & covered)),
            "correct_reference_vertices": intersection, "iou": _fraction(intersection, union)}
    known_gt = gtsem > 0
    purity_sum, assigned, dominant = 0, 0, Counter()
    for iid in np.unique(inst[inst > 0]):
        mask = (inst == iid) & near & (gtinst[nearest] > 0)
        if mask.sum() < 20:
            continue
        oid, count = Counter(gtinst[nearest[mask]].tolist()).most_common(1)[0]
        purity_sum += count
        assigned += int(mask.sum())
        dominant[int(oid)] += 1
    report = {"protocol": "revemap.reference_vertex_diagnostic.v1", "official_benchmark": False,
        "metric_scope": "Exact shared class names; bidirectional nearest-surface diagnostic, not official mIoU/AP",
        "distance_threshold_m": distance_threshold, "prediction_points": len(xyz),
        "near_reference_points": int(near.sum()), "supported_class_points": int(supported.sum()),
        "labeled_supported_points": int(labeled.sum()),
        "semantic_accuracy_with_unknown_as_error": _fraction(int(np.sum(supported & (sem == gtsem[nearest]))), int(supported.sum())),
        "semantic_accuracy_labeled_only": _fraction(int(np.sum(labeled & (sem == gtsem[nearest]))), int(labeled.sum())),
        "reference_vertices": len(gtxyz), "reference_covered_vertices": int(covered.sum()),
        "reference_geometry_coverage": float(covered.mean()),
        "reference_shared_class_vertices": int(known_gt.sum()),
        "reference_semantic_accuracy_unknown_or_uncovered_as_error": _fraction(int(np.sum(known_gt & (projected_sem == gtsem))), int(known_gt.sum())),
        "weighted_instance_purity_min20points": _fraction(purity_sum, assigned),
        "predicted_fragments_per_dominant_gt_object": dict(dominant),
        "instances": instance, "per_class": per_class}
    if before is not None:
        _, old_sem, old_inst = _validate(xyz, before["semantic"], before["instance"], "before")
        old = evaluate_points(xyz, old_sem, old_inst, gtxyz, gtsem, gtinst,
                              distance_threshold=distance_threshold,
                              instance_iou_threshold=instance_iou_threshold,
                              split_merge_fraction=split_merge_fraction)
        def owner_correct(ids, matching):
            correspondence = {row["predicted_id"]: row["reference_id"] for row in matching["matches"]}
            owners = np.array([correspondence.get(int(iid), 0) for iid in ids])
            return near & (owners > 0) & (owners == gtinst[nearest])
        new_correct = owner_correct(inst, instance["class_agnostic"])
        old_correct = owner_correct(old_inst, old["instances"]["class_agnostic"])
        added = (old_inst == 0) & (inst > 0)
        named = (old_sem == 0) & (sem > 0)
        old_sem_correct = supported & (old_sem > 0) & (old_sem == gtsem[nearest])
        now_sem_correct = supported & (sem > 0) & (sem == gtsem[nearest])
        report["before_after"] = {
            "scope": "same point order and geometry; ownership correctness uses class-agnostic IoU matching, independently of naming",
            "newly_owned_points": int(added.sum()), "newly_owned_correct_points": int(np.sum(added & new_correct)),
            "newly_owned_correct_fraction": _fraction(int(np.sum(added & new_correct)), int(added.sum())),
            "newly_owned_outside_shared_reference_support": int(np.sum(added & ~supported)),
            "newly_owned_outside_reference_instance_support": int(np.sum(added & ~(near & (gtinst[nearest] > 0)))),
            "unknown_semantic_recovered_points": int(named.sum()),
            "unknown_semantic_recovered_correct_fraction": _fraction(int(np.sum(named & now_sem_correct)), int(named.sum())),
            "existing_correct_owner_points": int(old_correct.sum()),
            "existing_owner_degraded_points": int(np.sum(old_correct & ~new_correct)),
            "existing_correct_semantic_points": int(old_sem_correct.sum()),
            "existing_semantic_degraded_points": int(np.sum(old_sem_correct & ~now_sem_correct)),
            "existing_owner_id_changed_points": int(np.sum((old_inst > 0) & (old_inst != inst))),
            "before_instance_precision": old["instances"]["precision"],
            "before_instance_recall": old["instances"]["recall"],
            "before_class_agnostic_instance_precision": old["instances"]["class_agnostic"]["precision"],
            "before_class_agnostic_instance_recall": old["instances"]["class_agnostic"]["recall"]}
    return report


def _class_dictionary(path):
    import json
    from pathlib import Path
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("classes must be an ID-to-name dictionary")
    result = {}
    for key, name in raw.items():
        try:
            cid = int(key)
        except (ValueError, TypeError) as error:
            raise ValueError("invalid semantic class ID") from error
        if str(cid) != key or cid < 0 or not isinstance(name, str) or not name.strip():
            raise ValueError("invalid semantic class definition")
        result[cid] = name
    positive = [name for cid, name in result.items() if cid > 0]
    if len(positive) != len(set(positive)):
        raise ValueError("duplicate class names make evaluation ambiguous")
    return result


def _read_prediction(paths, classes):
    from plyfile import PlyData
    vertices = PlyData.read(paths["map"])["vertex"].data
    if not {"x", "y", "z", "semantic_id", "instance_id"} <= set(vertices.dtype.names):
        raise ValueError("labeled map lacks required vertex fields")
    xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")])
    sem, inst = vertices["semantic_id"], vertices["instance_id"]
    if any(int(cid) not in classes for cid in np.unique(sem) if cid > 0):
        raise ValueError("prediction uses semantic IDs absent from classes")
    return _validate(xyz, sem, inst, "prediction")


def evaluate_result(result, reference_dir, *, manifest_path=None, trajectory_path=None,
                    before_result=None, allow_unbound_legacy=False, distance_threshold=.05,
                    instance_iou_threshold=.5, split_merge_fraction=.1):
    """Load checked artifacts and GT strictly after prediction has completed."""
    import json
    from pathlib import Path
    from plyfile import PlyData
    from .artifacts import load_artifacts
    from .contracts import bind_manifest_trajectory, load_manifest, load_trajectory, sha256_file

    artifacts = load_artifacts(result, manifest_path=manifest_path, trajectory_path=trajectory_path,
                               require_provenance=not allow_unbound_legacy,
                               allow_legacy_completion=allow_unbound_legacy)
    paths = artifacts["paths"]
    if not {"input_manifest", "trajectory"} <= paths.keys():
        raise ValueError("evaluation needs an input manifest and trajectory")
    inputs = dict(paths)
    if artifacts["manifest_path"]:
        inputs["artifacts"] = artifacts["manifest_path"]
    input_hashes = {str(path): sha256_file(path) for path in inputs.values()}
    manifest = load_manifest(paths["input_manifest"], require_files=False)
    poses, trajectory = load_trajectory(paths["trajectory"])
    if trajectory.get("sequence_id") != manifest.sequence_id:
        raise ValueError("trajectory scene differs from input manifest")
    bound_frames = bind_manifest_trajectory(manifest, poses)
    classes = _class_dictionary(paths["classes"])
    xyz, sem, inst = _read_prediction(paths, classes)
    before = None
    if before_result is not None:
        old = load_artifacts(before_result, manifest_path=paths["input_manifest"],
                             trajectory_path=paths["trajectory"],
                             require_provenance=not allow_unbound_legacy,
                             allow_legacy_completion=allow_unbound_legacy)
        inputs.update({"before_" + name: path for name, path in old["paths"].items()})
        if old["manifest_path"]:
            inputs["before_artifacts"] = old["manifest_path"]
        for name, path in inputs.items():
            if name.startswith("before_"):
                digest = sha256_file(path)
                if str(path) in input_hashes and input_hashes[str(path)] != digest:
                    raise RuntimeError("evaluation input changed: " + str(path))
                input_hashes[str(path)] = digest
        old_classes = _class_dictionary(old["paths"]["classes"])
        old_xyz, old_sem, old_inst = _read_prediction(old["paths"], old_classes)
        if xyz.dtype != old_xyz.dtype or not np.array_equal(xyz, old_xyz):
            raise ValueError("before/after evaluation requires identical point order, precision and geometry")
        vocabulary = {name: cid for cid, name in classes.items() if cid > 0}
        if any(name not in vocabulary for cid, name in old_classes.items() if cid > 0):
            raise ValueError("before class names must exist in after evaluation vocabulary")
        remapped = np.array([0 if cid == 0 else vocabulary[old_classes[int(cid)]] for cid in old_sem], dtype=np.int64)
        before = {"semantic": remapped, "instance": old_inst}
    reference = Path(reference_dir).resolve()
    reference_hashes = {}
    def record(path):
        reference_hashes[str(path)] = sha256_file(path)
        return path
    if manifest.dataset == "3rscan":
        gt = PlyData.read(record(reference / "labels.instances.annotated.v2.ply"))["vertex"].data
        metadata = json.loads(record(reference / "semseg.v2.json").read_text())
        annotated_scene = metadata.get("scan_id", metadata.get("scanId"))
        if annotated_scene is not None and annotated_scene != manifest.sequence_id:
            raise ValueError("reference annotation scene differs from input manifest")
        object_ids = np.asarray(gt["objectId"], dtype=np.int64)
        names = {int(row["objectId"]): row["label"] for row in metadata["segGroups"]}
    elif manifest.dataset == "scannet":
        scene = manifest.sequence_id
        gt = PlyData.read(record(reference / (scene + "_vh_clean_2.ply")))["vertex"].data
        segments = np.asarray(json.loads(record(reference / (scene + "_vh_clean_2.0.010000.segs.json")).read_text())["segIndices"])
        if len(segments) != len(gt):
            raise ValueError("ScanNet mesh/segment vertex mismatch")
        object_ids = np.zeros(len(gt), dtype=np.int64)
        names = {}
        for row in json.loads(record(reference / (scene + ".aggregation.json")).read_text())["segGroups"]:
            oid = int(row["objectId"]) + 1
            names[oid] = row["label"]
            selected = np.isin(segments, row["segments"])
            if np.any(object_ids[selected] > 0):
                raise ValueError("reference segments assigned to multiple objects")
            object_ids[selected] = oid
    else:
        raise ValueError("No verified Orbbec GT supported")
    from .contracts import validate_se3
    for frame, pose in bound_frames:
        pose_path = (manifest.root / "pose" / f"{pose.frame_id}.txt" if manifest.dataset == "scannet"
                     else manifest.root / f"frame-{pose.frame_id:06d}.pose.txt")
        if not pose_path.is_file():
            continue
        matrix = np.loadtxt(record(pose_path))
        if not np.isfinite(matrix).all():
            continue
        matrix = validate_se3(matrix, "GT alignment pose")
        if manifest.dataset == "3rscan" and frame.rotate_ccw:
            rotation = np.eye(4)
            rotation[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            matrix = matrix @ rotation
        alignment = matrix @ np.linalg.inv(pose.t_world_camera)
        break
    else:
        raise ValueError("No finite GT alignment pose")
    vocab = {name: cid for cid, name in classes.items() if cid > 0}
    gtsem = np.array([vocab.get(names.get(int(oid), ""), 0) for oid in object_ids], dtype=np.int64)
    gtxyz = np.column_stack([gt[key] for key in ("x", "y", "z")])
    world = xyz @ alignment[:3, :3].T + alignment[:3, 3]
    report = evaluate_points(world, sem, inst, gtxyz, gtsem, object_ids,
                             distance_threshold=distance_threshold, instance_iou_threshold=instance_iou_threshold,
                             split_merge_fraction=split_merge_fraction, before=before)
    report.update(status="completed", gt_role="post_prediction_evaluation_only",
                  provenance_bound=artifacts["provenance_bound"], scene_id=manifest.sequence_id,
                  legacy_completion_accepted=artifacts["legacy_completion_accepted"],
                  dataset=manifest.dataset, alignment="first finite paired pose; no ICP or scale fitting",
                  alignment_frame=pose.frame_id, T_dataset_estimated_world=alignment.tolist(),
                  input_sha256=input_hashes, reference_sha256=reference_hashes,
                  classes={str(cid): name for cid, name in classes.items()},
                  out_of_vocabulary_reference_labels=sorted(set(names.values()) - set(vocab)))
    if before_result is not None:
        report["before_after"]["before_provenance_bound"] = old["provenance_bound"]
        report["before_after"]["before_legacy_completion_accepted"] = old["legacy_completion_accepted"]
    for path, digest in {**input_hashes, **reference_hashes}.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("evaluation input changed: " + path)
    report["prediction_unchanged"] = True
    return report


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="run, fused/export directory or ARTIFACTS.json")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="optional; must match bound inventory")
    parser.add_argument("--trajectory", type=Path, help="optional; must match bound inventory")
    parser.add_argument("--before", type=Path, help="same-geometry result before recovery")
    parser.add_argument("--allow-unbound-legacy", action="store_true")
    parser.add_argument("--distance-threshold", type=float, default=.05)
    parser.add_argument("--instance-iou-threshold", type=float, default=.5)
    parser.add_argument("--split-merge-fraction", type=float, default=.1)
    args = parser.parse_args(argv)
    report = evaluate_result(args.result, args.reference_dir, manifest_path=args.manifest,
                             trajectory_path=args.trajectory, before_result=args.before,
                             allow_unbound_legacy=args.allow_unbound_legacy,
                             distance_threshold=args.distance_threshold,
                             instance_iou_threshold=args.instance_iou_threshold,
                             split_merge_fraction=args.split_merge_fraction)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "diagnostic.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "diagnostic": str(destination), "official_benchmark": False}))
    return report
