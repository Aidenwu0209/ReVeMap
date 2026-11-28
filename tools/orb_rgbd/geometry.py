"""Decode measured ScanNet RGB-D and reconstruct the ORB/DROID halfway surface."""

import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pose_pipeline.semantic_runtime.completion import diverse


def header(stream):
    version, length = struct.unpack("<IQ", stream.read(12))
    if version != 4:
        raise ValueError("Expected a ScanNet v4 sensor stream")
    stream.seek(length, 1)
    calibration = np.frombuffer(stream.read(256), "<f4").reshape(4, 4, 4)
    color, depth, _, _, w, h, scale, n = struct.unpack("<iiIIIIfQ", stream.read(36))
    if (color, depth, scale) != (2, 1, 1000) or not all(
        np.allclose(calibration[i], np.eye(4), atol=1e-6) for i in (1, 3)
    ):
        raise ValueError("Expected registered JPEG RGB and zlib millimeter depth")
    k = calibration[2]
    return (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])), w, h, n


def measurements(sensor):
    with sensor.open("rb") as stream:
        _, w, h, n = header(stream)
        for i in range(n):
            stream.seek(80, 1)  # skip per-frame reference pose and timestamps
            nc, nd = struct.unpack("<QQ", stream.read(16))
            color = cv2.imdecode(np.frombuffer(stream.read(nc), "u1"), cv2.IMREAD_COLOR)
            depth = np.frombuffer(zlib.decompress(stream.read(nd)), "<u2").reshape(h, w)
            if color is None:
                raise ValueError(f"Invalid JPEG at frame {i}")
            color = cv2.cvtColor(
                cv2.resize(color, (w, h), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            yield i, color, depth


def connected_poses(directory, native):
    def rows(name, width):
        p = directory / name
        if not p.is_file():
            raise ValueError(f"Missing ORB export {p}")
        t = [
            line
            for line in p.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return np.loadtxt(t, ndmin=2).reshape(-1, width) if t else np.empty((0, width))

    def table(name, time_scale):
        out = {}
        for r in rows(name, 8):
            if np.isfinite(r).all():
                i = round(r[0] * time_scale * 30)
                if not 0 <= i < len(native) or i in out:
                    raise ValueError(f"Ambiguous ORB timestamp {r[0]} in {name}")
                out[i] = r[1:]
        return out

    def matrices(values, ids):
        v = np.stack([values[int(i)] for i in ids])
        t = np.repeat(np.eye(4)[None], len(v), axis=0)
        t[:, :3, 3] = v[:, :3]
        t[:, :3, :3] = Rotation.from_quat(v[:, 3:]).as_matrix()
        return t

    tum = table("CameraTrajectory.txt", 1)
    largest = table("LargestMapTrajectory.txt", 1e-9)
    states = rows("FrameTrajectoryRaw.txt", 19)
    if len(states) != len(native) or not np.array_equal(
        states[:, 0], np.arange(len(native))
    ):
        raise ValueError("Expected one ORB tracking state per input frame")
    ok = states[:, 2] == 2
    ids = np.array(sorted(k for k in tum.keys() & largest.keys() if ok[k]), dtype=int)
    assisted = native.copy()
    accepted = np.zeros(len(native), dtype=bool)
    if len(ids):
        t = matrices(tum, ids)
        e = matrices(largest, ids)
        other = t[0] @ np.linalg.inv(e[0]) @ e
        translation = np.linalg.norm(t[:, :3, 3] - other[:, :3, 3], axis=1)
        rotation = Rotation.from_matrix(
            t[:, :3, :3].transpose(0, 2, 1) @ other[:, :3, :3]
        ).magnitude()
        keep = (translation <= 0.02) & (rotation <= np.deg2rad(0.5))
        ids, t = ids[keep], t[keep]
        if len(ids):
            assisted[ids] = native[ids[0]] @ np.linalg.inv(t[0]) @ t
            accepted[ids] = True
    half = native.copy()
    half[:, :3, 3] = 0.5 * (native[:, :3, 3] + assisted[:, :3, 3])
    delta = Rotation.from_matrix(
        native[:, :3, :3].transpose(0, 2, 1) @ assisted[:, :3, :3]
    )
    half[:, :3, :3] = (
        native[:, :3, :3] @ Rotation.from_rotvec(0.5 * delta.as_rotvec()).as_matrix()
    )
    half[~accepted] = native[~accepted]
    return assisted, half, accepted


def world(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def completion_samples(raw, k, pose, ordinal):
    d = raw.astype(np.float32) * np.float32(0.001)
    h, w = raw.shape
    y, x = np.mgrid[2 + ordinal % 5 : h - 2 : 5, 2 + (ordinal // 5) % 5 : w - 2 : 5]
    z = d[y, x]
    keep = (z >= 0.3) & (z <= 5.0)
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        keep &= np.abs(d[y + dy, x + dx] - z) <= np.maximum(0.08, 0.03 * z)
    x, y, z = x[keep], y[keep], z[keep]
    points = world(
        np.column_stack(((x - k[2]) * z / k[0], (y - k[3]) * z / k[1], z)), pose
    ).astype("f4")
    grid = np.floor(points / 0.03).astype("i8") + 65536
    if grid.size and (grid.min() < 0 or grid.max() >= 131072):
        raise ValueError("Completion point outside the 3 cm voxel key range")
    keys, first = np.unique(
        (grid[:, 0] << 34) | (grid[:, 1] << 17) | grid[:, 2], return_index=True
    )
    return keys, points[first], np.full(len(keys), ordinal, "i4")


def completed_cloud(cloud, chunks, features, output, coord_type):
    p = PlyData.read(cloud)["vertex"].data
    base = np.column_stack([p[k] for k in "xyz"])
    keys, points, views = (np.concatenate(v) for v in zip(*chunks))
    _, group, count = np.unique(keys, return_inverse=True, return_counts=True)
    take = (count >= 2) & diverse(group, views, features, len(count))
    new = np.stack(
        [np.bincount(group, weights=points[:, j]) / count for j in range(3)], axis=1
    )[take]
    distances, nearest = cKDTree(base).query(new)
    new, nearest = new[distances > 0.022], nearest[distances > 0.022]
    v = np.zeros(
        len(base) + len(new),
        dtype=[(k, coord_type) for k in "xyz"]
        + [(k, "u1") for k in ("red", "green", "blue")]
        + [("semantic_id", "i4"), ("instance_id", "i4")],
    )
    for j, k in enumerate("xyz"):
        v[k] = np.r_[base[:, j], new[:, j]]
    for k in ("red", "green", "blue"):
        v[k] = np.r_[p[k], p[k][nearest]]
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(output)
    return len(v)


def reconstruct(sensor, native_map, native, assisted, half, accepted, out):
    with sensor.open("rb") as f:
        k, w, h, n = header(f)
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, *k)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        0.02, 0.08, o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    samples = set(range(0, n, 5)) | {n - 1}
    frames = sorted(i for i in samples | {s + 2 for s in samples} if i < n)
    lookup = {i: j for j, i in enumerate(frames)}
    features = np.array(
        [np.r_[half[i, :3, 3], half[i, :3, 2]] for i in frames], dtype="f4"
    )
    anchor = np.flatnonzero(accepted)
    motion_frames = set(range(0, n, 12)) | {n - 1} | set(anchor[:1])
    chunks, original, deltas = [], [], []
    for i, color, depth in measurements(sensor):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1000,
            depth_trunc=4.5,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intr, np.linalg.inv(half[i]))
        if i in lookup:
            chunks.append(completion_samples(depth, k, half[i], lookup[i]))
        if i in motion_frames and len(anchor):
            y, x = np.mgrid[4:h:8, 4:w:8]
            z = depth[y, x].astype(float) * 0.001
            keep = (z > 0.3) & (z < 4.0)
            cam = np.column_stack(
                (
                    (x[keep] - k[2]) * z[keep] / k[0],
                    (y[keep] - k[3]) * z[keep] / k[1],
                    z[keep],
                )
            )
            before = world(cam, native[i])
            original.append(before.astype("f4"))
            deltas.append((world(cam, assisted[i]) - before).astype("f4"))
    primary = PlyData.read(native_map, mmap=False)
    v = primary["vertex"].data
    x = np.column_stack([v[k] for k in "xyz"])
    result = x.copy()
    if original:
        tree = cKDTree(np.vstack(original))
        shifts = np.vstack(deltas)
        for start in range(0, len(x), 100000):
            sel = slice(start, start + 100000)
            d, index = tree.query(x[sel], k=4, workers=2)
            weights = np.where(d <= 0.12, 1 / np.maximum(0.005, d) ** 2, 0)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            delta = (weights[:, :, None] * shifts[index]).sum(axis=1).astype(x.dtype)
            full = (x[sel] + delta).astype("f4")
            result[sel] = x[sel] + 0.5 * (full - x[sel])
    for j, key in enumerate("xyz"):
        v[key] = result[:, j]
    primary.write(out / "primary_in_secondary.ply")
    cloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not len(cloud.points) or not len(mesh.triangles):
        raise ValueError("The measured surface is empty")
    if not o3d.io.write_triangle_mesh(
        str(out / "mesh.ply"), mesh
    ) or not o3d.io.write_point_cloud(str(out / "tsdf_points.ply"), cloud):
        raise OSError("Cannot write the reconstructed RGB-D surface")
    count = completed_cloud(
        out / "tsdf_points.ply", chunks, features, out / "A.ply", x.dtype
    )
    return {
        "integrated_frames": n,
        "new_A_points": count,
        "primary_points": len(x),
        "mesh_vertices": len(mesh.vertices),
    }
