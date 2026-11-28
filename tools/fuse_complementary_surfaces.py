"""Add measured, non-overlapping surfaces from an independently tracked RGB-D reconstruction."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from panoptic_observations import CLASSES
from refine_native_panoptic import digest


def coordinates(vertices):
    points = np.column_stack([vertices[axis] for axis in "xyz"])
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected a nonempty, finite map in meters")
    return points


def labels(vertices):
    classes = vertices["semantic_id"].astype(np.int64)
    if np.any((classes < 0) | (classes > 21)):
        raise ValueError("Use maps labeled in the native ScanNet20 export convention")
    return np.where((classes >= 1) & (classes <= 20), classes-1, 20).astype("u1"), vertices["instance_id"]


def associate_and_select(primary, secondary, correspondence, primary_sem, primary_ids, secondary_sem, secondary_ids):
    if primary.shape != correspondence.shape or len(primary) < 8:
        raise ValueError("Expected one corresponding half-trajectory position per primary map vertex (at least eight)")
    if not all(arr.shape == (len(points),) for points, arr in (
        (primary,primary_sem),(primary,primary_ids),(secondary,secondary_sem),(secondary,secondary_ids)
    )):
        raise ValueError("Labels must use the vertex order of their own reconstruction")

    # Correspondences establish instance identity across reconstructions. Keep the measured
    # secondary positions for deciding what adds coverage and for writing the final map.
    distance, nearest = cKDTree(correspondence).query(secondary, k=8, workers=2)
    weights = 1 / np.maximum(.018, distance)**2
    offset = ((primary[nearest]-correspondence[nearest])*weights[:,:,None]).sum(1) / weights.sum(1)[:,None]
    common = secondary + offset * np.exp(-np.maximum(distance[:,0]-.12, 0)[:,None]/.22)
    reference = cKDTree(primary)
    spacing, nearby = reference.query(common, workers=2)
    mapped = np.full(len(secondary), -1, dtype=np.int64)
    next_id = int(primary_ids.max()) + 1
    reused = 0
    for tag in np.unique(secondary_ids[secondary_ids > 0]):
        mask = secondary_ids == tag
        category = int(np.bincount(secondary_sem[mask]).argmax())
        overlap = mask & (spacing < .07)
        compatible = overlap & (primary_sem[nearby] == category) & (primary_ids[nearby] > 0)
        choices, counts = np.unique(primary_ids[nearby[compatible]], return_counts=True)
        if (len(counts) and counts.max() >= max(16., .22*overlap.sum()) and counts.max() > .52*counts.sum()):
            mapped[mask] = choices[counts.argmax()]
            reused += 1
        else:
            mapped[mask] = next_id
            next_id += 1
    mapped[secondary_sem < 2] = 0
    gap = reference.query(secondary, workers=2)[0]
    add = (gap > .045) & (secondary_sem < 20)
    return add, mapped, reused


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {k:digest(getattr(args,k)) for k in ("primary","secondary","primary_in_secondary")}
    first = PlyData.read(args.primary)["vertex"].data
    second = PlyData.read(args.secondary)["vertex"].data
    paired = PlyData.read(args.primary_in_secondary)["vertex"].data
    p, q, r = map(coordinates,(first,second,paired))
    for field in ("red","green","blue"):
        if field not in first.dtype.names or field not in second.dtype.names:
            raise ValueError("Both measured RGB-D surfaces must include vertex colors")
        if field in paired.dtype.names and not np.array_equal(first[field],paired[field]):
            raise ValueError("Primary correspondence colors differ from the original vertex order")
    sem, ids = labels(first)
    other, other_ids = labels(second)
    add, assigned, reused = associate_and_select(p,q,r,sem,ids,other,other_ids)
    appended = np.zeros(int(add.sum()), dtype=first.dtype)
    for name in set(first.dtype.names) & set(second.dtype.names):
        appended[name] = second[name][add]
    if len(appended):
        info = np.iinfo(appended["instance_id"].dtype)
        if np.any((assigned[add] < info.min) | (assigned[add] > info.max)):
            raise ValueError("Merged instance IDs exceed the PLY field's representable range")
    appended["instance_id"] = assigned[add]
    appended["semantic_id"] = other[add].astype("i4") + 1
    result = np.concatenate((first,appended))
    # The evaluated point cloud stores metric positions at float32 precision. Determine
    # distances above using the original PLY precision before quantizing the output.
    for name in "xyz":
        result[name] = result[name].astype("f4")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        dest=Path(temp)/"result"
        dest.mkdir()
        PlyData([PlyElement.describe(result,"vertex")],text=False).write(dest/"map_labeled.ply")
        np.savez_compressed(dest/"panoptic.npz",semantic=np.r_[sem,other[add]],instance=result["instance_id"])
        classes={0:"unknown", **{i+1:c for i,c in enumerate(CLASSES)},21:"unlabeled"}
        (dest/"classes.json").write_text(json.dumps(classes,indent=2))
        record=dict(inputs=inputs, GT_used=False, label_space="ScanNet20", primary_vertices=len(first),
                    measured_secondary_vertices=len(second), appended_measured_vertices=len(appended),
                    linked_secondary_instances=reused, minimum_extra_distance_m=.045,
                    outputs={f.name:digest(f) for f in dest.iterdir()})
        (dest/"result.json").write_text(json.dumps(record,indent=2))
        dest.rename(args.output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ("primary","secondary","primary-in-secondary","output"):
        parser.add_argument("--"+option,type=Path,required=True)
    run(parser.parse_args())
