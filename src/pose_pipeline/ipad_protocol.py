#!/usr/bin/env python3
"""Record Scan app TCP v1 packets as an original-SGF ScanNet folder.

The produced ``frame-XXXXXX.pose.txt`` files are camera-to-world transforms in
metres.  They are converted from ARKit camera axes (+X right, +Y up, -Z
forward) to OpenCV/ScanNet camera axes (+X right, +Y down, +Z forward), then
rebased so the first accepted camera is the world origin.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import time
from typing import Any, BinaryIO
import zlib

import cv2
import numpy as np


MAGIC = b"SGFIPD01"
MAX_HEADER = 64 * 1024
MAX_COLOR = 32 * 1024 * 1024
MAX_DEPTH = 16 * 1024 * 1024
MAX_CONFIDENCE = 4 * 1024 * 1024
ARKIT_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


class PacketError(RuntimeError):
    pass


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("TCP connection closed mid-packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def integer(header: dict[str, Any], key: str, minimum: int, maximum: int) -> int:
    value = header.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PacketError(f"{key} must be an integer")
    if not math.isfinite(float(value)) or int(value) != value:
        raise PacketError(f"{key} must be a finite integer")
    output = int(value)
    if output < minimum or output > maximum:
        raise PacketError(f"{key} is outside [{minimum}, {maximum}]")
    return output


def string(header: dict[str, Any], key: str) -> str:
    value = header.get(key)
    if not isinstance(value, str) or not value:
        raise PacketError(f"{key} must be a non-empty string")
    return value


def checked_payload(
    stream: BinaryIO,
    header: dict[str, Any],
    name: str,
    maximum: int,
) -> bytes:
    size = integer(header, f"{name}_bytes", 0, maximum)
    payload = read_exact(stream, size)
    checksums = header.get("crc32")
    if not isinstance(checksums, dict) or name not in checksums:
        raise PacketError(f"crc32.{name} is required")
    expected = checksums[name]
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise PacketError(f"crc32.{name} must be an integer")
    expected_int = int(expected)
    if expected_int != expected or not 0 <= expected_int <= 0xFFFFFFFF:
        raise PacketError(f"crc32.{name} is out of range")
    actual = zlib.crc32(payload) & 0xFFFFFFFF
    if actual != expected_int:
        raise PacketError(
            f"crc32 mismatch for {name}: expected {expected_int}, got {actual}"
        )
    return payload


def receive_packet(stream: BinaryIO) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    prefix = read_exact(stream, 12)
    if prefix[:8] != MAGIC:
        raise PacketError("invalid packet magic")
    header_size = struct.unpack(">I", prefix[8:])[0]
    if not 2 <= header_size <= MAX_HEADER:
        raise PacketError(f"invalid JSON header length {header_size}")
    try:
        header = json.loads(read_exact(stream, header_size))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PacketError(f"invalid JSON header: {error}") from error
    if not isinstance(header, dict):
        raise PacketError("JSON header must be an object")
    color = checked_payload(stream, header, "color", MAX_COLOR)
    depth = checked_payload(stream, header, "depth", MAX_DEPTH)
    confidence = checked_payload(stream, header, "confidence", MAX_CONFIDENCE)
    return header, color, depth, confidence


def decode_frame(
    header: dict[str, Any],
    color_payload: bytes,
    depth_payload: bytes,
    confidence_payload: bytes,
    minimum_confidence: int,
    maximum_depth_m: float,
) -> dict[str, Any]:
    if integer(header, "schema_version", 1, 1) != 1:
        raise PacketError("only schema version 1 is supported")
    if string(header, "image_orientation") != "sensor_native":
        raise PacketError("only sensor_native image orientation is supported")
    if string(header, "color_encoding") != "jpeg":
        raise PacketError("only JPEG color is supported")
    if string(header, "depth_encoding") != "float32_le_meters":
        raise PacketError("only little-endian Float32 metre depth is supported")
    if string(header, "confidence_encoding") != "uint8":
        raise PacketError("only UInt8 confidence is supported")

    color_width = integer(header, "color_width", 1, 16384)
    color_height = integer(header, "color_height", 1, 16384)
    depth_width = integer(header, "depth_width", 16, 16384)
    depth_height = integer(header, "depth_height", 16, 16384)
    reference_width = integer(header, "intrinsics_reference_width", 1, 16384)
    reference_height = integer(header, "intrinsics_reference_height", 1, 16384)
    frame_id = integer(header, "frame_id", 0, 1 << 53)
    timestamp_ns = integer(header, "timestamp_ns", 0, 1 << 53)

    encoded = np.frombuffer(color_payload, dtype=np.uint8)
    color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if color is None or color.shape[:2] != (color_height, color_width):
        raise PacketError("decoded JPEG dimensions do not match the header")

    count = depth_width * depth_height
    if len(depth_payload) != count * 4:
        raise PacketError("depth payload length does not match its dimensions")
    depth_m = np.frombuffer(depth_payload, dtype="<f4").reshape(
        depth_height, depth_width
    ).copy()
    if confidence_payload:
        if len(confidence_payload) != count:
            raise PacketError("confidence payload length does not match depth")
        confidence = np.frombuffer(confidence_payload, dtype=np.uint8).reshape(
            depth_height, depth_width
        )
    else:
        confidence = np.full((depth_height, depth_width), 2, dtype=np.uint8)

    valid = (
        np.isfinite(depth_m)
        & (depth_m > 0.0)
        & (depth_m <= maximum_depth_m)
        & (confidence >= minimum_confidence)
    )
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(
        np.rint(depth_m[valid] * 1000.0), 1, maximum_depth_m * 1000.0
    ).astype(np.uint16)
    color_depth = cv2.resize(
        color, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR
    )

    intrinsics = np.asarray(header.get("camera_intrinsics"), dtype=np.float64)
    if intrinsics.shape != (9,) or not np.isfinite(intrinsics).all():
        raise PacketError("camera_intrinsics must contain 9 finite row-major values")
    intrinsics = intrinsics.reshape(3, 3)
    scale_x = depth_width / reference_width
    scale_y = depth_height / reference_height
    camera = np.array(
        [
            intrinsics[0, 0] * scale_x,
            intrinsics[1, 1] * scale_y,
            intrinsics[0, 2] * scale_x,
            intrinsics[1, 2] * scale_y,
            depth_width,
            depth_height,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(camera).all() or camera[0] <= 0 or camera[1] <= 0:
        raise PacketError("scaled depth intrinsics are invalid")
    camera_native = np.array(
        [
            intrinsics[0, 0] * color_width / reference_width,
            intrinsics[1, 1] * color_height / reference_height,
            intrinsics[0, 2] * color_width / reference_width,
            intrinsics[1, 2] * color_height / reference_height,
            color_width,
            color_height,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(camera_native).all() or camera_native[0] <= 0 or camera_native[1] <= 0:
        raise PacketError("scaled colour intrinsics are invalid")

    raw_pose = np.asarray(header.get("arkit_camera_to_world_m"), dtype=np.float64)
    if raw_pose.shape != (16,) or not np.isfinite(raw_pose).all():
        raise PacketError(
            "arkit_camera_to_world_m must contain 16 finite row-major values"
        )
    raw_pose = raw_pose.reshape(4, 4)
    rotation = raw_pose[:3, :3]
    if (
        not np.allclose(raw_pose[3], [0.0, 0.0, 0.0, 1.0], atol=2e-3)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-3)
        or not 0.98 <= np.linalg.det(rotation) <= 1.02
    ):
        raise PacketError("ARKit pose is not a valid rigid transform")

    return {
        "frame_id": frame_id,
        "timestamp_ns": timestamp_ns,
        "session_id": string(header, "session_id"),
        "tracking_state": string(header, "tracking_state"),
        "color": color_depth,
        "color_native": color,
        "camera_native": camera_native,
        "depth_mm": depth_mm,
        "camera": camera,
        "raw_pose": raw_pose,
        "cv_pose": raw_pose @ ARKIT_FROM_CV,
        "valid_depth_ratio": float(np.count_nonzero(valid) / valid.size),
        "depth_median_m": float(np.median(depth_m[valid])) if valid.any() else None,
        "color_crc32": zlib.crc32(color_payload) & 0xFFFFFFFF,
        "depth_crc32": zlib.crc32(depth_payload) & 0xFFFFFFFF,
        "confidence_crc32": zlib.crc32(confidence_payload) & 0xFFFFFFFF,
    }


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]
