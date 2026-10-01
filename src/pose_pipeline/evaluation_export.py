"""GT-free input adapter for the supplied ScanNet CD/PQ evaluators.

Native artifacts remain in estimated-world coordinates with their native IDs.
This explicit export produces frame-0 coordinates and separate evaluator labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .artifacts import load_artifacts, verify_artifact_snapshot
from .contracts import load_trajectory, sha256_file


# Match compute_pq_test.py, including its historical shower-curtain spelling.
CLASSES = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "showercurtrain", "toilet", "sink", "bathtub",
    "otherfurniture", "unlabeled",
)
ALIASES = {"refridgerator": "refrigerator", "shower curtain": "showercurtrain"}


def _labels(vertices, names):
    for key in ("semantic_id", "instance_id"):
        values = vertices[key]
        if values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError(f"native {key} must contain nonnegative integers")
    # EvalPanoptic increments predicted IDs before encoding pairs with 65536.
    if np.any(vertices["instance_id"] >= 65535):
        raise ValueError("native instance_id exceeds the supplied PQ evaluator's id_offset")
    semantics = np.full(len(vertices), 20, dtype=np.int64)
    mapping = {}
    for native_id in np.unique(vertices["semantic_id"]):
        key = str(int(native_id))
        if key not in names or not isinstance(names[key], str):
            raise ValueError("classes.json has no name for semantic_id " + key)
        name = ALIASES.get(names[key], names[key])
        target = CLASSES.index(name) if name in CLASSES else 20
        semantics[vertices["semantic_id"] == native_id] = target
        mapping[key] = {"native_name": names[key], "evaluation_id": target,
                        "evaluation_name": CLASSES[target]}
    instances = vertices["instance_id"].astype(np.int64)
    instances[instances == 0] = -1
    return semantics, instances, mapping


def export_evaluation(source, output):
    """Export a completed, provenance-bound ScanNet artifact to a new directory.

No GT, resampling, registration, instance merging or label recovery is used.
EXPORT.json is written last; without it an interrupted export is incomplete.
"""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("evaluation output must be a new directory: " + str(output))
    bundle = load_artifacts(source, require_provenance=True)
    inventory = bundle["manifest"]
    if inventory["dataset"] != "scannet":
        raise ValueError("this evaluator adapter requires a ScanNet artifact")
    paths = bundle["paths"]
    poses, _ = load_trajectory(paths["trajectory"])
    first = next((pose for pose in poses if pose.frame_id == 0), None)
    if first is None:
        raise ValueError("frame 0 is required; another first valid frame cannot replace it")
    t0 = first.t_world_camera
    inverse = np.linalg.inv(t0)
    cloud = PlyData.read(paths["map"])
    vertices = cloud["vertex"].data
    required = {"x", "y", "z", "semantic_id", "instance_id"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError("native map must contain XYZ, semantic_id and instance_id")
    xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")]).astype(np.float64)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("native map must contain nonempty finite XYZ")
    names = json.loads(paths["classes"].read_text())
    semantics, instances, mapping = _labels(vertices, names)
    normalized = xyz @ inverse[:3, :3].T + inverse[:3, 3]
    converted = np.empty(len(vertices), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")] + [
        (key, vertices.dtype[key]) for key in vertices.dtype.names if key not in ("x", "y", "z")])
    for key in vertices.dtype.names:
        converted[key] = normalized[:, ("x", "y", "z").index(key)] if key in ("x", "y", "z") else vertices[key]
    verify_artifact_snapshot(bundle)
    output.mkdir(parents=True, exist_ok=False)
    # Point-cloud export deliberately contains vertices only; native vertex
    # labels are preserved. The external PQ script reads the separate arrays.
    ply_path = output / "frame0_prediction.ply"
    PlyData([PlyElement.describe(converted, "vertex")], text=False).write(ply_path)
    np.save(output / "semantic.npy", semantics, allow_pickle=False)
    np.save(output / "instance.npy", instances, allow_pickle=False)
    verify_artifact_snapshot(bundle)
    report = {
        "schema": "revemap.scannet_evaluation_export.v1", "status": "completed",
        "scene_id": inventory["scene_id"], "dataset": inventory["dataset"],
        "GT_used": False, "provenance_bound": True, "vertex_count": len(vertices),
        "coordinate_frame": "estimated_camera_frame_0", "units": "metres",
        "T_estimated_world_camera0_m": t0.tolist(),
        "T_camera0_estimated_world_m": inverse.tolist(),
        "external_evaluator_transform": "T_GT_world_camera0 @ inv(T_estimated_world_camera0)",
        "vertex_order_preserved": True, "native_vertex_labels_preserved": True,
        "resampled": False, "instance_merge_or_recovery": False,
        "ply_label_contract": "native IDs; use semantic.npy and instance.npy for PQ",
        "semantic_classes": list(CLASSES), "semantic_mapping": mapping,
        "instance_mapping": "native 0 -> -1; positive IDs unchanged",
        "unassigned_points": int(np.count_nonzero(instances == -1)),
        "input_sha256": bundle["input_sha256"],
        "adapter_sha256": sha256_file(Path(__file__)),
        "output_sha256": {name: sha256_file(output / name) for name in
                          ("frame0_prediction.ply", "semantic.npy", "instance.npy")},
    }
    with (output / "EXPORT.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="completed, provenance-bound ScanNet ARTIFACTS.json or its directory")
    parser.add_argument("--output", type=Path, required=True, help="new evaluation export directory")
    args = parser.parse_args(argv)
    report = export_evaluation(args.result, args.output)
    print(json.dumps({"status": report["status"], "scene_id": report["scene_id"],
                      "export": str(args.output / "EXPORT.json")}, indent=2))


if __name__ == "__main__":
    main()
