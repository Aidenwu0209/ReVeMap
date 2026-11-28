"""Keep measured surface geometry while withholding labels for previously unseen positive instance IDs."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData

from refine_native_panoptic import digest


def guard_added_instances(primary_instances, added_semantic, added_instances):
    """Return guarded suffix labels, preserving IDs that already occur in the primary prefix."""
    if isinstance(primary_instances, (set, frozenset)):
        primary_instances = np.fromiter(primary_instances, dtype=np.int64)
    else:
        primary_instances = np.asarray(primary_instances)
    added_semantic = np.asarray(added_semantic)
    added_instances = np.asarray(added_instances)
    if (primary_instances.ndim != 1 or added_semantic.ndim != 1
            or added_instances.shape != added_semantic.shape
            or any(a.dtype.kind not in "iu" for a in (primary_instances, added_semantic, added_instances))
            or np.any((added_semantic < 0) | (added_semantic > 20))):
        raise ValueError("Expected integer ScanNet20 labels in primary and appended point order")
    primary_ids = np.unique(primary_instances[primary_instances > 0])
    novel = ((added_instances > 0) & (added_semantic >= 2) & (added_semantic < 20)
             & ~np.isin(added_instances, primary_ids))
    semantic = added_semantic.copy()
    instance = added_instances.copy()
    semantic[novel] = 20
    instance[novel] = 0
    return semantic, instance


def run(source, output, *, sensor=None, primary_trajectory=None, secondary_trajectory=None,
        surface_policy="tight"):
    geometry_inputs = {"sensor": sensor, "primary_trajectory": primary_trajectory,
                       "secondary_trajectory": secondary_trajectory}
    provided = [p is not None for p in geometry_inputs.values()]
    if any(provided) and not all(provided):
        raise ValueError("Provide --sensor, --primary-trajectory and --secondary-trajectory together")
    if surface_policy not in ("tight", "relaxed") or (surface_policy == "relaxed" and not all(provided)):
        raise ValueError("A relaxed surface policy requires the sensor and both estimated trajectories")
    if output.exists():
        raise FileExistsError(output)
    receipt = source / "result.json"
    previous = json.loads(receipt.read_text())
    if previous.get("GT_used") is not False or previous.get("label_space") != "ScanNet20":
        raise ValueError("Expected a native ScanNet20 complementary surface result")
    for name in ("map_labeled.ply", "panoptic.npz", "classes.json"):
        if digest(source / name) != previous["outputs"][name]:
            raise ValueError(f"{name} differs from the recorded input")
    classes = json.loads((source / "classes.json").read_text())
    if classes.get("21") != "unlabeled":
        raise ValueError("Expected PLY semantic_id 21 for ScanNet20 unknown points")
    with np.load(source / "panoptic.npz", allow_pickle=False) as masks:
        sem, ids = masks["semantic"], masks["instance"]
    cloud = PlyData.read(source / "map_labeled.ply", mmap=False)
    vertex = cloud["vertex"].data
    n = previous["primary_vertices"]
    if (not 0 < n <= len(vertex) or sem.shape != (len(vertex),) or ids.shape != sem.shape
            or len(vertex) != n + previous["appended_measured_vertices"]
            or not np.array_equal(vertex["instance_id"], ids)):
        raise ValueError("Point counts or instance labels differ from the recorded fused map")
    inputs = {"complementary_surface_receipt": digest(receipt)}
    keep = np.ones(len(vertex) - n, dtype=bool)
    if all(provided):
        from filter_free_space_points import _poses, free_space_keep
        from surface_continuation import select_additions

        inputs.update({name: digest(path) for name, path in geometry_inputs.items()})
        if len(keep):
            xyz = np.column_stack([vertex[axis] for axis in "xyz"]).astype(np.float32)
            depth_keep = free_space_keep(sensor, _poses(primary_trajectory), _poses(secondary_trajectory), xyz[n:])
            keep = select_additions(xyz[:n], xyz[n:], sem[:n], sem[n:], ids[:n], ids[n:],
                                    depth_keep, policy=surface_policy)
            if keep.dtype != np.bool_ or keep.shape != (len(vertex) - n,):
                raise ValueError("The geometry filter must return one boolean per appended point")
    result_sem, result_ids = guard_added_instances(ids[:n], sem[n:], ids[n:])
    changed = result_ids != ids[n:]
    sem[n:], ids[n:] = result_sem, result_ids
    # Update only changed suffix vertices, keeping source field encodings everywhere else.
    vertices = np.flatnonzero(changed) + n
    vertex["semantic_id"][vertices] = 21
    vertex["instance_id"][vertices] = 0
    if not keep.all():
        point_mask = np.r_[np.ones(n, dtype=bool), keep]
        cloud["vertex"].data = vertex[point_mask]
        sem, ids = sem[point_mask], ids[point_mask]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        dest = Path(temp) / "result"
        dest.mkdir()
        cloud.write(dest / "map_labeled.ply")
        np.savez_compressed(dest / "panoptic.npz", semantic=sem, instance=ids)
        (dest / "classes.json").write_bytes((source / "classes.json").read_bytes())
        record = dict(inputs=inputs,
                      GT_used=False, label_space="ScanNet20", primary_vertices=n,
                      input_appended_vertices=len(keep), appended_measured_vertices=int(keep.sum()),
                      removed_appended_vertices=int((~keep).sum()), newly_unknown_points=int((changed & keep).sum()),
                      surface_policy=surface_policy if all(provided) else None,
                      outputs={p.name:digest(p) for p in dest.iterdir()})
        (dest / "result.json").write_text(json.dumps(record, indent=2))
        dest.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Output of fuse_complementary_surfaces.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, help="Raw ScanNet RGB-D .sens recording")
    parser.add_argument("--primary-trajectory", type=Path, help="Primary estimated camera poses")
    parser.add_argument("--secondary-trajectory", type=Path, help="Secondary estimated camera poses")
    parser.add_argument("--surface-policy", choices=("tight", "relaxed"), default="tight")
    args = parser.parse_args()
    run(args.input, args.output, sensor=args.sensor, primary_trajectory=args.primary_trajectory,
        secondary_trajectory=args.secondary_trajectory, surface_policy=args.surface_policy)
