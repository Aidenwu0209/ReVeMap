"""Strict, GT-free RGB-D and pose trajectory contracts.

All public trajectory matrices are ``T_world_camera`` in metres.  Dataset
ground truth belongs to the separate evaluator and is rejected here even when
the referenced file exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA = "rgbd_sequence_manifest.v1"
TRAJECTORY_SCHEMA = "pose_trajectory.v1"
FORBIDDEN_PARTS = frozenset({
    "pose", "poses", "gt", "ground_truth", "ground-truth", "evaluation",
    "evaluations", "mesh", "meshes",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_se3(value: object, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise ValueError(f"{name} rotation is not proper")
    return np.ascontiguousarray(matrix)


def _audit_rgbd_path(path: Path, *, root: Path, kind: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{kind} path escapes sequence root: {resolved}") from exc
    parts = {part.lower() for part in resolved.parts}
    forbidden = sorted(parts & FORBIDDEN_PARTS)
    if forbidden:
        raise ValueError(f"{kind} path contains forbidden GT part: {forbidden}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{kind} file missing: {resolved}")
    return resolved


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_us: int
    color_path: Path
    depth_path: Path
    intrinsics: tuple[float, float, float, float]
    rotate_ccw: bool = False

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp_us": self.timestamp_us,
            "color_path": str(self.color_path),
            "depth_path": str(self.depth_path),
            "intrinsics": list(self.intrinsics),
            "rotate_ccw": self.rotate_ccw,
        }


@dataclass(frozen=True)
class SequenceManifest:
    dataset: str
    sequence_id: str
    root: Path
    depth_scale: float
    frames: tuple[FrameRecord, ...]
    source: str

    def validate(self, *, require_files: bool = True) -> "SequenceManifest":
        if self.dataset not in {"scannet", "3rscan", "orbbec"}:
            raise ValueError(f"unsupported dataset: {self.dataset}")
        if not self.sequence_id or not self.frames:
            raise ValueError("sequence id and frames are required")
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
        root = self.root.resolve()
        seen: set[int] = set()
        previous_timestamp = -1
        for frame in self.frames:
            if frame.frame_id in seen:
                raise ValueError(f"duplicate frame id: {frame.frame_id}")
            if frame.timestamp_us < previous_timestamp:
                raise ValueError("frame timestamps must be monotonic")
            if len(frame.intrinsics) != 4 or not np.isfinite(frame.intrinsics).all():
                raise ValueError(f"bad intrinsics for frame {frame.frame_id}")
            if frame.intrinsics[0] <= 0 or frame.intrinsics[1] <= 0:
                raise ValueError(f"non-positive focal length for frame {frame.frame_id}")
            if require_files:
                _audit_rgbd_path(frame.color_path, root=root, kind="color")
                _audit_rgbd_path(frame.depth_path, root=root, kind="depth")
            seen.add(frame.frame_id)
            previous_timestamp = frame.timestamp_us
        return self

    def as_dict(self) -> dict:
        rotations = {bool(frame.rotate_ccw) for frame in self.frames}
        if len(rotations) != 1:
            raise ValueError("a manifest cannot mix native and rotated camera bases")
        camera_basis = (
            "image_rotated_ccw_from_native"
            if next(iter(rotations)) else "native_sensor_camera"
        )
        unsigned = {
            "schema": MANIFEST_SCHEMA,
            "dataset": self.dataset,
            "sequence_id": self.sequence_id,
            "root": str(self.root.resolve()),
            "depth_scale": self.depth_scale,
            "source": self.source,
            "matrix_convention": "T_world_camera_m",
            "camera_basis": camera_basis,
            "gt_at_inference": False,
            "forbidden_inputs": sorted(FORBIDDEN_PARTS),
            "frames": [frame.as_dict() for frame in self.frames],
        }
        return {**unsigned, "payload_sha256": stable_json_sha256(unsigned)}


@dataclass(frozen=True)
class PoseRecord:
    frame_id: int
    timestamp_us: int
    t_world_camera: np.ndarray
    valid: bool = True
    source: str = "unknown"

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp_us": self.timestamp_us,
            "T_world_camera_m": validate_se3(
                self.t_world_camera, f"frame {self.frame_id}",
            ).reshape(-1).tolist(),
            "valid": bool(self.valid),
            "source": self.source,
        }


def _create_only_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_manifest(path: Path, manifest: SequenceManifest) -> None:
    manifest.validate()
    _create_only_json(path, manifest.as_dict())


def write_input_sha256_audit(path: Path, manifest: SequenceManifest) -> dict:
    """Hash every inference-visible RGB-D input without opening GT assets."""
    manifest.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in manifest.frames:
        rows.append({
            "frame_id": frame.frame_id,
            "timestamp_us": frame.timestamp_us,
            "color_path": str(frame.color_path),
            "color_sha256": sha256_file(frame.color_path),
            "depth_path": str(frame.depth_path),
            "depth_sha256": sha256_file(frame.depth_path),
        })
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ) + "\n")
    summary = {
        "schema": "rgbd_input_sha256_audit.v1",
        "sequence_id": manifest.sequence_id,
        "frame_count": len(rows),
        "manifest_payload_sha256": manifest.as_dict()["payload_sha256"],
        "records_sha256": stable_json_sha256(rows),
        "gt_consumed": False,
    }
    _create_only_json(path.with_suffix(path.suffix + ".summary.json"), summary)
    return summary


def load_manifest(path: Path, *, require_files: bool = True) -> SequenceManifest:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema mismatch")
    if payload.get("gt_at_inference") is not False:
        raise ValueError("manifest must declare gt_at_inference=false")
    expected = payload.get("payload_sha256")
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    if expected != stable_json_sha256(unsigned):
        raise ValueError("manifest payload SHA mismatch")
    frames = tuple(FrameRecord(
        frame_id=int(row["frame_id"]),
        timestamp_us=int(row["timestamp_us"]),
        color_path=Path(row["color_path"]),
        depth_path=Path(row["depth_path"]),
        intrinsics=tuple(float(value) for value in row["intrinsics"]),
        rotate_ccw=bool(row.get("rotate_ccw", False)),
    ) for row in payload["frames"])
    return SequenceManifest(
        dataset=str(payload["dataset"]),
        sequence_id=str(payload["sequence_id"]),
        root=Path(payload["root"]),
        depth_scale=float(payload["depth_scale"]),
        frames=frames,
        source=str(payload["source"]),
    ).validate(require_files=require_files)


def write_trajectory(
    path: Path,
    records: Iterable[PoseRecord],
    *,
    sequence_id: str,
    arm: str,
    metadata: Mapping[str, object] | None = None,
) -> None:
    rows = list(records)
    if not rows:
        raise ValueError("trajectory is empty")
    seen: set[int] = set()
    serializable = []
    for row in rows:
        if row.frame_id in seen:
            raise ValueError(f"duplicate pose frame: {row.frame_id}")
        if not row.valid:
            raise ValueError(f"invalid poses cannot be serialized: {row.frame_id}")
        serializable.append(row.as_dict())
        seen.add(row.frame_id)
    stable_pose_payload = [{
        "frame_id": row["frame_id"],
        "timestamp_us": row["timestamp_us"],
        "T_world_camera_m_q1e7": np.round(
            np.asarray(row["T_world_camera_m"], dtype=np.float64), 7,
        ).tolist(),
    } for row in serializable]
    unsigned = {
        "schema": TRAJECTORY_SCHEMA,
        "sequence_id": sequence_id,
        "arm": arm,
        "matrix_convention": "T_world_camera_m",
        "identity_fallback_used": False,
        "gt_at_inference": False,
        "metadata": dict(metadata or {}),
        "stable_pose_sha256_q1e7": stable_json_sha256(stable_pose_payload),
        "poses": serializable,
    }
    _create_only_json(path, {
        **unsigned, "payload_sha256": stable_json_sha256(unsigned),
    })


def load_trajectory(path: Path) -> tuple[list[PoseRecord], dict]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError("trajectory schema mismatch")
    if payload.get("matrix_convention") != "T_world_camera_m":
        raise ValueError("trajectory convention mismatch")
    if payload.get("identity_fallback_used") is not False:
        raise ValueError("identity fallback trajectory is forbidden")
    expected = payload.get("payload_sha256")
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    if expected != stable_json_sha256(unsigned):
        raise ValueError("trajectory payload SHA mismatch")
    records = []
    seen = set()
    for row in payload.get("poses", []):
        frame_id = int(row["frame_id"])
        if frame_id in seen or row.get("valid") is not True:
            raise ValueError("trajectory contains duplicate or invalid pose")
        records.append(PoseRecord(
            frame_id=frame_id,
            timestamp_us=int(row["timestamp_us"]),
            t_world_camera=validate_se3(
                np.asarray(row["T_world_camera_m"], dtype=np.float64).reshape(4, 4),
                f"frame {frame_id}",
            ),
            valid=True,
            source=str(row.get("source", "unknown")),
        ))
        seen.add(frame_id)
    if not records:
        raise ValueError("trajectory is empty")
    return records, payload


def load_legacy_tcw_mm(
    path: Path,
    *,
    allowed_frame_ids: set[int] | None = None,
    source: str = "DPV-SLAM",
) -> list[PoseRecord]:
    """Import the frozen SGF ``T_cw`` text contract without filling gaps."""
    records = []
    seen = set()
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 18:
            raise ValueError(
                f"legacy trajectory line {line_number} has {len(fields)} fields"
            )
        frame_id, timestamp_us = int(fields[0]), int(fields[1])
        if allowed_frame_ids is not None and frame_id not in allowed_frame_ids:
            continue
        if frame_id in seen:
            raise ValueError(f"duplicate legacy trajectory frame {frame_id}")
        t_camera_world = np.asarray(
            [float(value) for value in fields[2:]], dtype=np.float64,
        ).reshape(4, 4)
        t_camera_world[:3, 3] /= 1000.0
        t_camera_world = validate_se3(
            t_camera_world, f"legacy frame {frame_id}",
        )
        records.append(PoseRecord(
            frame_id=frame_id,
            timestamp_us=timestamp_us,
            t_world_camera=validate_se3(np.linalg.inv(t_camera_world)),
            valid=True,
            source=source,
        ))
        seen.add(frame_id)
    if not records:
        raise ValueError("legacy trajectory has no selected valid poses")
    return records


def bind_manifest_trajectory(
    manifest: SequenceManifest,
    trajectory: Sequence[PoseRecord],
    *,
    allow_manifest_superset: bool = False,
) -> list[tuple[FrameRecord, PoseRecord]]:
    pose_by_id = {row.frame_id: row for row in trajectory}
    if len(pose_by_id) != len(trajectory):
        raise ValueError("trajectory contains duplicate frame ids")
    frame_by_id = {frame.frame_id: frame for frame in manifest.frames}
    if len(frame_by_id) != len(manifest.frames):
        raise ValueError("manifest contains duplicate frame ids")
    bound = []
    for pose in trajectory:
        frame = frame_by_id.get(pose.frame_id)
        if frame is None:
            raise ValueError(f"manifest misses trajectory frame {pose.frame_id}")
        if pose.timestamp_us != frame.timestamp_us:
            raise ValueError(f"timestamp mismatch at frame {frame.frame_id}")
        bound.append((frame, pose))
    if len(bound) != len(trajectory):
        raise ValueError("manifest and trajectory frame sets differ")
    if not allow_manifest_superset and len(bound) != len(manifest.frames):
        raise ValueError("trajectory does not cover every manifest frame")
    return bound
