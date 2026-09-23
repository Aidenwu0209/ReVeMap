"""TSDF surface exports bound to a completed map, RGB-D input and trajectory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import uuid

import numpy as np

from .contracts import sha256_file

SCHEMA = "revemap.surface.v1"
MAGIC = b"RVMESH01"


def encode_mesh(vertices, colors, normals, triangles):
    vertices, colors, normals = [np.asarray(a, dtype="<f4") for a in (vertices, colors, normals)]
    faces = np.asarray(triangles)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or colors.shape != vertices.shape or normals.shape != vertices.shape
            or not all(np.isfinite(a).all() for a in (vertices, colors, normals))
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Invalid surface geometry")
    packed = np.column_stack((vertices, np.clip(colors, 0, 1), normals)).astype("<f4")
    return (MAGIC + struct.pack("<II", len(vertices), len(faces))
            + packed.tobytes() + faces.astype("<u4").tobytes())


def publish_surface(pipeline, *, map_path, manifest, trajectory, refusion_receipt,
                    viewer_triangles=150000):
    """Create an additional export; never rewrite existing map inventories."""
    import open3d as o3d

    pipeline = Path(pipeline).resolve()
    index = pipeline / "SURFACE.json"
    if index.exists():
        raise FileExistsError("A surface is already published for this pipeline")
    receipt = json.loads(Path(refusion_receipt).read_text())
    mesh_path = Path(receipt["mesh"]).resolve()
    if (receipt.get("status") != "completed" or receipt.get("gt_consumed") is not False
            or receipt.get("identity_fallback_used") is not False
            or receipt["manifest_sha256"] != sha256_file(manifest)
            or receipt["trajectory_sha256"] != sha256_file(trajectory)
            or receipt["mesh_sha256"] != sha256_file(mesh_path)):
        raise ValueError("Surface source binding mismatch")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    full_vertices, full_triangles = len(mesh.vertices), len(mesh.triangles)
    if (full_vertices != receipt["mesh_vertices"] or full_triangles != receipt["mesh_triangles"]
            or full_triangles <= 0 or viewer_triangles <= 0):
        raise ValueError("Surface counts do not match receipt")
    if full_triangles > viewer_triangles:
        mesh = mesh.simplify_quadric_decimation(viewer_triangles)
    mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([.65, .68, .72])
    vertices = np.asarray(mesh.vertices)
    blob = encode_mesh(vertices, np.asarray(mesh.vertex_colors),
                       np.asarray(mesh.vertex_normals), np.asarray(mesh.triangles))
    folder = pipeline / ("surface-view-" + uuid.uuid4().hex[:12])
    folder.mkdir()
    viewer = folder / "surface.bin"
    viewer.write_bytes(blob)
    paths = dict(map=map_path, input_manifest=manifest, trajectory=trajectory,
                 mesh=mesh_path, refusion=refusion_receipt, viewer=viewer)
    value = {"schema": SCHEMA, "color_mode": "rgb", "vertices": full_vertices,
             "triangles": full_triangles, "viewer_vertices": len(vertices),
             "viewer_triangles": len(mesh.triangles),
             "bounds": {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
             "voxel_length_m": receipt["voxel_length_m"],
             "integrated_frame_count": receipt["integrated_frame_count"],
             "files": {name: {"path": os.path.relpath(Path(path).resolve(), pipeline),
                               "sha256": sha256_file(path)} for name, path in paths.items()}}
    # Link a complete index atomically and create-only, including concurrent runs.
    temporary = folder / "SURFACE.json"
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.link(temporary, index)
    return index


def load_surface(root):
    from .device_gui import completed_result, pipeline_path

    root = Path(root).resolve()
    result = completed_result(root)
    pipeline = pipeline_path(root)
    index = pipeline / "SURFACE.json"
    if not index.is_file():
        return None
    value = json.loads(index.read_text())
    if value.get("schema") != SCHEMA:
        raise ValueError("Unsupported surface schema")
    paths = {}
    for name in ("map", "input_manifest", "trajectory", "mesh", "refusion", "viewer"):
        entry = value["files"][name]
        path = (pipeline / entry["path"]).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != entry["sha256"]:
            raise ValueError("Surface artifact changed or escaped session: " + name)
        paths[name] = path
    if (paths["map"] != Path(result["final_cloud"]).resolve()
            or paths["trajectory"] != Path(result["trajectory"]).resolve()):
        raise ValueError("Surface belongs to another map or trajectory")
    receipt = json.loads(paths["refusion"].read_text())
    if (receipt.get("status") != "completed"
            or receipt.get("mesh_sha256") != value["files"]["mesh"]["sha256"]
            or receipt.get("manifest_sha256") != value["files"]["input_manifest"]["sha256"]
            or receipt.get("trajectory_sha256") != value["files"]["trajectory"]["sha256"]):
        raise ValueError("Surface provenance mismatch")
    public = {k: v for k, v in value.items() if k != "files"}
    public.update(available=True, revision=sha256_file(index),
                  mesh_sha256=value["files"]["mesh"]["sha256"])
    return {"metadata": public, "paths": paths,
            "hashes": {name: entry['sha256'] for name, entry in value['files'].items()}}


def main():
    """Add a surface to a historical completed session using its exact inputs."""
    import argparse
    from .device_gui import completed_result, pipeline_path
    from reconstruction.rgbd_refusion import FullRefusionRequest, run_full_rgbd_refusion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    root = args.session.resolve()
    result = completed_result(root)
    pipeline = pipeline_path(root)
    if (pipeline / "SURFACE.json").exists():
        print(json.dumps(load_surface(root)["metadata"]))
        return
    geometry = json.loads((pipeline / "mapping/mapping_result.json").read_text())
    original = json.loads((pipeline / "mapping/fusion/refusion_result.json").read_text())
    manifest, trajectory = Path(geometry["manifest"]), Path(result["trajectory"])
    if (sha256_file(manifest) != original["manifest_sha256"]
            or sha256_file(trajectory) != original["trajectory_sha256"]):
        raise ValueError("Historical inputs have changed")
    snapshot = sha256_file(pipeline / "GUI_RESULT.json")
    output = pipeline / ("surface-fusion-" + uuid.uuid4().hex[:12])
    run_full_rgbd_refusion(FullRefusionRequest(
        manifest=manifest, trajectory=trajectory, output_dir=output,
        voxel_length_m=original["voxel_length_m"], sdf_trunc_m=original["sdf_trunc_m"],
        depth_trunc_m=original["depth_trunc_m"]))
    if sha256_file(pipeline / "GUI_RESULT.json") != snapshot or pipeline_path(root) != pipeline:
        raise ValueError("Session changed during surface generation")
    publish_surface(pipeline, map_path=result["final_cloud"], manifest=manifest,
                    trajectory=trajectory, refusion_receipt=output / "refusion_result.json")
    print(json.dumps(load_surface(root)["metadata"]))


if __name__ == "__main__":
    main()
