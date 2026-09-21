"""RGB-D and semantic mapping implementation used by ReVeMap.

The historical package name is retained for input and worker compatibility.
"""

from .contracts import (
    MANIFEST_SCHEMA,
    TRAJECTORY_SCHEMA,
    FrameRecord,
    PoseRecord,
    SequenceManifest,
    load_manifest,
    load_legacy_tcw_mm,
    load_trajectory,
    write_manifest,
    write_trajectory,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "FrameRecord",
    "PoseRecord",
    "SequenceManifest",
    "load_manifest",
    "load_legacy_tcw_mm",
    "load_trajectory",
    "write_manifest",
    "write_trajectory",
]
