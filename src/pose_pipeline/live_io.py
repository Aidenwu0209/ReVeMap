"""Disk protocol shared by the camera, online preview and GUI processes."""
from pathlib import Path
import json
import numpy as np

from .contracts import FrameRecord, SequenceManifest, write_manifest

BASE_COMMIT = "18ddeca303bf9937e855d3ac3113a90dd852a462"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default


class FrameJournalReader:
    """Tail committed rows without reparsing the growing capture journal.

    The offset only advances past complete newline-terminated records. A partial
    UTF-8/JSON row remains on disk until the writer commits it. Replacing or
    truncating a live journal is an error, never a reason to replay old frames.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None
        self.size = 0

    def poll(self):
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            if self.identity is not None:
                raise ValueError("Active frame journal disappeared")
            return []
        with stream:
            import os
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if self.identity is not None and (identity != self.identity or stat.st_size < self.size):
                raise ValueError("Active frame journal was replaced or truncated")
            stream.seek(self.offset)
            data = stream.read()
            end = data.rfind(b"\n") + 1
            rows = [json.loads(line) for line in data[:end].split(b"\n") if line]
            self.identity, self.size = identity, stat.st_size
            self.offset += end
            return rows


def journal_frames(path):
    """Read a snapshot of all newline-committed rows."""
    return FrameJournalReader(path).poll()


def frame_record(row):
    return FrameRecord(
        row["frame_id"], row["timestamp_us"], Path(row["color_path"]),
        Path(row["depth_path"]), tuple(row["intrinsics"]), row.get("rotate_ccw", False),
    )


def preview_next_frame(index, available):
    """Bound catch-up gaps to three saved frames; final mapping never uses this."""
    remaining = max(0, available-index-1)
    return min(available, index+min(3, max(1, remaining//8)))


def seal_capture(root, *, source):
    root = Path(root)
    rows = journal_frames(root / "frames.jsonl")
    if not rows:
        raise RuntimeError("未收到有效 RGB-D 帧，请检查相机连接。")
    write_manifest(root / "manifest.json", SequenceManifest(
        "orbbec", root.parent.name, root, 1000.,
        tuple(frame_record(row) for row in rows), source,
    ))
    return len(rows)


def point_cloud(depth, color, intrinsic, twc, stride=8):
    """Backproject aligned millimetre RGB-D; never substitute a missing pose."""
    from .contracts import validate_se3
    twc = validate_se3(twc)
    yy, xx = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    zz = depth[::stride, ::stride].astype(np.float32) / 1000.
    valid = np.isfinite(zz) & (zz >= .2) & (zz <= 4.5)
    fx, fy, cx, cy = intrinsic
    xyz = np.stack(((xx-cx)*zz/fx, (yy-cy)*zz/fy, zz), -1)[valid]
    xyz = xyz @ twc[:3, :3].T + twc[:3, 3]
    rgb = color[::stride, ::stride][valid][:, ::-1].astype(np.float32)/255.
    return np.column_stack((xyz, rgb)).astype("<f4")


def publish_cloud(root, points, *, kind, revision):
    root = Path(root)
    points = np.asarray(points, dtype="<f4").reshape(-1, 6)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > 100000:
        points = points[np.linspace(0, len(points)-1, 100000).astype(int)]
    # Unique generation names prevent old metadata pairing with new bytes.
    name = f"cloud_{revision}.bin"
    tmp = root / (name + ".tmp")
    points.tofile(tmp)
    tmp.replace(root / name)
    atomic_json(root / "cloud.json", {
        "kind": kind, "revision": revision, "file": name, "points": len(points),
    })
    for old in root.glob("cloud_*.bin"):
        if old.name != name:
            try:
                old.unlink()
            except FileNotFoundError:
                pass
