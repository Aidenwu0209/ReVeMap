"""Fail-closed full-frame RGB-D refusion.

The former environment probe has been replaced by a native Open3D TSDF path.
It consumes only a GT-free RGB-D manifest and a complete
``T_world_camera`` trajectory. Missing requested poses are fatal and are
never replaced by identity transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from pose_pipeline.contracts import (
    bind_manifest_trajectory,
    load_manifest,
    load_trajectory,
    sha256_file,
)


@dataclass
class RefusionRequest:
    """Legacy pair authorization contract retained for API compatibility."""

    reference_scan: str
    source_scan: str
    transform: np.ndarray


@dataclass(frozen=True)
class FullRefusionRequest:
    manifest: Path
    trajectory: Path
    output_dir: Path
    fused_frame_ids: tuple[int, ...] | None = None
    voxel_length_m: float = 0.02
    sdf_trunc_m: float = 0.08
    depth_trunc_m: float = 4.50


def check_refusion_authorization(
    decision: dict, transform: np.ndarray | None,
) -> bool:
    if not decision.get("usable_for_reconstruction"):
        return False
    if transform is None:
        return False
    value = np.asarray(transform, dtype=np.float64)
    return value.shape == (4, 4) and np.isfinite(value).all()


def _read_rgbd(frame):
    import cv2

    color = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise FileNotFoundError(f"missing RGB-D frame {frame.frame_id}")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"depth frame {frame.frame_id} is not uint16 HxW")
    fx, fy, cx, cy = frame.intrinsics
    if frame.rotate_ccw:
        old_width = depth.shape[1]
        color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fx, fy, cx, cy = fy, fx, cy, old_width - 1.0 - cx
    height, width = depth.shape
    if color.shape[:2] != depth.shape:
        color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return color, depth, (fx, fy, cx, cy)


def run_full_rgbd_refusion(request: FullRefusionRequest) -> dict:
    import open3d as o3d

    manifest = load_manifest(request.manifest)
    trajectory, trajectory_payload = load_trajectory(request.trajectory)
    bound = bind_manifest_trajectory(
        manifest, trajectory, allow_manifest_superset=True,
    )
    frame_by_id = {frame.frame_id: (frame, pose) for frame, pose in bound}
    selected_ids = (
        tuple(frame.frame_id for frame in manifest.frames)
        if request.fused_frame_ids is None else tuple(request.fused_frame_ids)
    )
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("fused frame ids must be non-empty and unique")
    missing = sorted(set(selected_ids) - set(frame_by_id))
    if missing:
        raise ValueError(
            f"trajectory/manifest misses {len(missing)} fused frames; "
            f"first={missing[:10]}"
        )
    output_dir = Path(request.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=request.voxel_length_m,
        sdf_trunc=request.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for frame_id in selected_ids:
        frame, pose = frame_by_id[frame_id]
        color, depth, intrinsics = _read_rgbd(frame)
        height, width = depth.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color, dtype=np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.uint16)),
            depth_scale=manifest.depth_scale,
            depth_trunc=request.depth_trunc_m,
            convert_rgb_to_intensity=False,
        )
        fx, fy, cx, cy = intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy,
        )
        volume.integrate(
            rgbd,
            intrinsic,
            np.linalg.inv(pose.t_world_camera),
        )
    cloud = volume.extract_point_cloud()
    points = np.asarray(cloud.points)
    if not len(points) or not np.isfinite(points).all():
        raise RuntimeError("refusion produced an empty or non-finite cloud")
    cloud_path = output_dir / "refused.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False):
        raise RuntimeError("Open3D failed to write refused.ply")
    report = {
        "schema": "rgbd_full_refusion.v1",
        "status": "completed",
        "sequence_id": manifest.sequence_id,
        "requested_frame_count": len(selected_ids),
        "integrated_frame_count": len(selected_ids),
        "trajectory_pose_count": len(trajectory),
        "point_count": int(len(points)),
        "voxel_length_m": request.voxel_length_m,
        "sdf_trunc_m": request.sdf_trunc_m,
        "depth_trunc_m": request.depth_trunc_m,
        "manifest_sha256": sha256_file(request.manifest),
        "trajectory_sha256": sha256_file(request.trajectory),
        "trajectory_payload_sha256": trajectory_payload["payload_sha256"],
        "cloud": str(cloud_path),
        "cloud_sha256": sha256_file(cloud_path),
        "identity_fallback_used": False,
        "gt_consumed": False,
    }
    with (output_dir / "refusion_result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def run_rgbd_refusion(
    request: RefusionRequest,
    *,
    output_dir: str | Path,
    manifest: str | Path | None = None,
    trajectory: str | Path | None = None,
    fused_frame_ids: Sequence[int] | None = None,
    **_ignored,
) -> dict:
    """Compatibility wrapper that now requires real frame-level inputs."""
    if manifest is None or trajectory is None:
        return {
            "status": "failed",
            "stage": "full_frame_contract",
            "reason": "manifest and trajectory are required; probe-only refusion was removed",
        }
    return run_full_rgbd_refusion(FullRefusionRequest(
        manifest=Path(manifest),
        trajectory=Path(trajectory),
        output_dir=Path(output_dir),
        fused_frame_ids=(
            None if fused_frame_ids is None
            else tuple(int(value) for value in fused_frame_ids)
        ),
    ))
