"""Read-only semantic object inspection; never changes a published scan."""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import threading

import numpy as np
from plyfile import PlyData, PlyElement

_CACHE = OrderedDict()
_LOCK = threading.RLock()


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _path(root, value):
    if not value:
        raise FileNotFoundError("No completed point cloud")
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Object artifact escaped its session")
    return resolved


def _stamp(path):
    try:
        s = path.stat()
        return str(path), s.st_mtime_ns, s.st_size
    except OSError:
        return str(path), 0, 0


def load_objects(root):
    root = Path(root)
    from .device_gui import completed_result, pipeline_path
    index = pipeline_path(root) / "GUI_RESULT.json"
    result = completed_result(root)
    final = _path(root, result.get("final_cloud"))
    raw = _path(root, result.get("raw_map") or result.get("final_cloud"))
    classes = _path(root, result.get("classes"))
    names = Path(result.get("names", pipeline_path(root) / "fused/instance_names.json"))
    key = tuple(_stamp(p) for p in (index, final, raw, classes, names))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
        vertices = PlyData.read(str(final))["vertex"].data
        required = {"x", "y", "z", "semantic_id", "instance_id"}
        if not required.issubset(vertices.dtype.names):
            raise ValueError("This cloud has no per-point semantic / instance labels")
        xyz_all = np.column_stack([vertices[f] for f in "xyz"])
        finite = np.isfinite(xyz_all).all(axis=1)
        vertices = vertices[finite].copy()
        xyz = xyz_all[finite].astype("<f4")
        rgb = np.full(xyz.shape, .5, dtype="<f4")
        rgb_available = False
        # Geometry must match before borrowing RGB; equal lengths are not enough.
        try:
            raw_v = PlyData.read(str(raw))["vertex"].data
            raw_xyz = np.column_stack([raw_v[f] for f in "xyz"])
            if (len(raw_xyz) == len(xyz_all)
                    and np.allclose(raw_xyz[finite], xyz_all[finite], rtol=0, atol=1e-5)
                    and {"red", "green", "blue"}.issubset(raw_v.dtype.names)):
                rgb = np.column_stack([raw_v[f][finite] for f in ("red", "green", "blue")]).astype("<f4") / 255
                rgb_available = True
        except (OSError, ValueError, KeyError):
            pass
        semantic = vertices["semantic_id"].astype(np.int32)
        instance = vertices["instance_id"].astype(np.int32)
        if ((np.abs(semantic.astype(np.int64)) > 16777216).any()
                or (np.abs(instance.astype(np.int64)) > 16777216).any()):
            raise ValueError("Label IDs exceed viewer format precision")
        class_names = _json(classes)
        name_rows = _json(names)
        name_map = {int(n["instance_id"]): n.get("vlm_name") for n in name_rows} if isinstance(name_rows, list) else {}
        def row(kind, label, mask):
            points = xyz[mask]
            sem = int(label) if kind == "class" else int(np.unique(semantic[mask], return_counts=True)[0][np.argmax(np.unique(semantic[mask], return_counts=True)[1])])
            return {"kind": kind, "label_id": int(label), "semantic_id": sem,
                    "semantic_name": str(class_names.get(str(sem), "unknown")),
                    "vlm_name": name_map.get(int(label)) if kind == "instance" else None,
                    "point_count": int(mask.sum()),
                    "size_m": (points.max(axis=0) - points.min(axis=0)).astype(float).tolist()}
        groups = {}
        for kind, labels in (("class", semantic), ("instance", instance)):
            groups[kind + "es" if kind == "class" else "instances"] = sorted(
                [row(kind, label, labels == label) for label in np.unique(labels) if label > 0],
                key=lambda r: (-r["point_count"], r["label_id"]))
        revision = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
        catalog = {"revision": revision, "points": len(vertices), "rgb_available": rgb_available,
                   "format": "xyz_rgb_semantic_instance_float32_le", "stride": 8, **groups}
        data = np.column_stack((xyz, rgb, semantic, instance)).astype("<f4")
        bundle = {"catalog": catalog, "data": data, "vertices": vertices}
        _CACHE[key] = bundle
        while len(_CACHE) > 2:
            _CACHE.popitem(last=False)
        return bundle


def export_object(root, kind, label):
    if kind not in ("class", "instance") or not label.isascii() or not label.isdecimal() or not 0 < int(label) <= 16777216:
        raise ValueError("Invalid object selection")
    bundle = load_objects(root)
    vertices = bundle["vertices"]
    field = "semantic_id" if kind == "class" else "instance_id"
    mask = vertices[field] == int(label)
    if not mask.any():
        raise FileNotFoundError("Object not found")
    stream = io.BytesIO()
    PlyData([PlyElement.describe(vertices[mask], "vertex")], text=False, byte_order="<",
            comments=["Extracted by per-point label; original coordinates and attributes retained."]).write(stream)
    return stream.getvalue()
