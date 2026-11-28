"""Reassign disputed object boundaries using four view predictions and independent PTv3 evidence."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from refine_native_panoptic import digest


def exchange_boundaries(xyz, semantic, instance, views, probabilities, nearest, distance):
    """Return ScanNet20 labels in native point order. Probabilities use the 0..255 PTv3 encoding."""
    count = len(xyz)
    if (xyz.shape != (count, 3) or semantic.shape != (count,) or instance.shape != (count,)
            or nearest.shape != (count,) or distance.shape != (count,) or views.shape != (4, count)):
        raise ValueError("Expected four view labels and one native map point per output label")
    if (probabilities.ndim != 2 or probabilities.shape[1] != 20 or probabilities.dtype != np.uint8
            or not np.issubdtype(nearest.dtype, np.integer)
            or np.any(nearest < 0) or np.any(nearest >= len(probabilities))):
        raise ValueError("Expected quantized ScanNet20 mesh probabilities and valid nearest vertex indices")
    if not np.isfinite(xyz).all() or not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("Expected finite native coordinates and nonnegative distances")
    measured = probabilities[nearest]
    prediction = measured.argmax(1)
    confidence = measured.max(1)
    support = (views == semantic).sum(0)
    disputed = ((instance > 0) & (semantic != prediction) & (distance < .08)
                & (confidence > 215) & (support <= 2))
    labels = semantic.copy()
    ids = instance.copy()
    for obj in np.unique(instance[disputed]):
        original = instance == obj
        points = original & disputed
        if points.sum() <= .6 * original.sum():
            structure = points & (prediction < 2)
            labels[structure] = prediction[structure]
            ids[structure] = 0
    for category in np.unique(prediction[disputed & (prediction >= 2)]):
        targets = np.flatnonzero(disputed & (prediction == category))
        seeds = np.flatnonzero((ids > 0) & (labels == category) & (prediction == category) & (confidence > 175))
        if len(seeds) < 30:
            continue
        separation, index = cKDTree(xyz[seeds]).query(xyz[targets], workers=2)
        near = separation < .12
        targets = targets[near]
        labels[targets] = category
        ids[targets] = ids[seeds[index[near]]]
    return labels, ids


def checked_inputs(directory, names, map_hash):
    receipt_file = directory / "result.json"
    record = json.loads(receipt_file.read_text())
    if (record.get("label_space") != "ScanNet20" or not record.get("same_native_vertex_order")
            or record["inputs"].get("map") != map_hash):
        raise ValueError(f"{directory} did not use this native ScanNet20 map")
    for name in names:
        if name not in record["outputs"] or digest(directory / name) != record["outputs"][name]:
            raise ValueError(f"{directory / name} differs from its receipt or was not saved")
    return record, digest(receipt_file)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    map_hash = digest(args.map)
    views_record, views_hash = checked_inputs(args.views, ["views.npz"], map_hash)
    if not views_record.get("vote_views") or "native_classes" not in views_record["inputs"]:
        raise ValueError("Run the four-view stage with --vote-views --native-classes and --save-view-evidence")
    surface_record, surface_hash = checked_inputs(args.surface, ["panoptic.npz", "classes.json", "mesh.npz"], map_hash)
    if (views_record["inputs"]["mesh"] != surface_record["inputs"]["mesh"]
            or surface_record["inputs"].get("anchor") != views_hash):
        raise ValueError("The semantic evidence and four views must come from the same native run")
    anchor = args.labels or args.surface
    _, label_hash = checked_inputs(anchor, ["panoptic.npz"], map_hash)
    with np.load(args.surface / "mesh.npz", allow_pickle=False) as evidence:
        key = "proba" if "proba" in evidence else "mesh_proba"
        probabilities, nearest, distance = evidence[key], evidence["nearest"], evidence["distance"]
    with np.load(anchor / "panoptic.npz", allow_pickle=False) as result:
        initial, initial_ids = result["semantic"], result["instance"]
    with np.load(args.views / "views.npz", allow_pickle=False) as evidence:
        views = evidence["semantic"]
    cloud = PlyData.read(args.map, mmap=False)
    v = cloud["vertex"].data
    xyz = np.column_stack([v[key] for key in "xyz"])
    labels, ids = exchange_boundaries(xyz, initial, initial_ids, views, probabilities, nearest, distance)
    v["semantic_id"] = labels.astype(np.int32) + 1
    v["instance_id"] = ids
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        result = Path(temp) / "result"
        result.mkdir()
        np.savez_compressed(result / "panoptic.npz", semantic=labels, instance=ids)
        cloud.write(result / "map_labeled.ply")
        (result / "classes.json").write_bytes((args.surface / "classes.json").read_bytes())
        record = dict(label_space="ScanNet20", same_native_vertex_order=True, vertex_count=len(v),
                      inputs=dict(map=map_hash, mesh=views_record["inputs"]["mesh"], views=views_hash,
                                  surface=surface_hash, labels=label_hash),
                      changed_points=int(((labels != initial) | (ids != initial_ids)).sum()),
                      outputs={p.name:digest(p) for p in result.iterdir()})
        (result / "result.json").write_text(json.dumps(record, indent=2))
        result.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("map", "views", "surface", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Use another sealed output on the same map as the starting labels")
    run(parser.parse_args())
