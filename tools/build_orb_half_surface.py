"""Build a second ScanNet map with halfway ORB/DROID poses and one native 3D expert."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from orb_rgbd.geometry import connected_poses, header, reconstruct
from refine_native_panoptic import digest

from pose_pipeline.contracts import PoseRecord, validate_se3, write_trajectory

NAMES = ("CameraTrajectory.txt", "LargestMapTrajectory.txt", "FrameTrajectoryRaw.txt")


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    p = path.with_suffix(".tmp")
    p.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    p.replace(path)


def settings(sensor):
    with sensor.open("rb") as stream:
        k, width, height, count = header(stream)
    values = dict(
        zip(
            ("Camera1.fx", "Camera1.fy", "Camera1.cx", "Camera1.cy"), map(np.float32, k)
        )
    )
    values.update(
        {
            "Camera.width": width,
            "Camera.height": height,
            "Camera.fps": 30,
            "Camera.RGB": 0,
            **{"Camera1." + name: 0.0 for name in ("k1", "k2", "p1", "p2", "k3")},
            "Stereo.ThDepth": 40.0,
            "Stereo.b": 0.075,
            "RGBD.DepthMapFactor": 1000.0,
            "ORBextractor.nFeatures": 2000,
            "ORBextractor.scaleFactor": 1.2,
            "ORBextractor.nLevels": 8,
            "ORBextractor.iniThFAST": 20,
            "ORBextractor.minThFAST": 7,
            "Viewer.KeyFrameSize": 0.05,
            "Viewer.KeyFrameLineWidth": 1.0,
            "Viewer.GraphLineWidth": 0.9,
            "Viewer.PointSize": 2.0,
            "Viewer.CameraSize": 0.08,
            "Viewer.CameraLineWidth": 3.0,
            "Viewer.ViewpointX": 0.0,
            "Viewer.ViewpointY": -0.7,
            "Viewer.ViewpointZ": -1.8,
            "Viewer.ViewpointF": 500.0,
        }
    )
    text = ["%YAML:1.0", 'File.version: "1.0"', 'Camera.type: "PinHole"']
    return "\n".join(
        text + [f"{key}: {value}" for key, value in values.items()]
    ) + "\n", count


def raw_exports(args, sensor_hash):
    def verified(folder, current=False):
        record = read(folder / "ORB_RAW_SEALED.json")
        if record["source_archive_sha256"] != sensor_hash or record["returncode"] != 0:
            raise ValueError(
                "The ORB export must be successful and match this .sens file"
            )
        if current and (
            record["runner"] != digest(args.orb_runner)
            or record["vocabulary"] != digest(args.vocabulary)
        ):
            raise ValueError(
                "The existing ORB run used different executable or vocabulary bytes"
            )
        for name in NAMES:
            if digest(folder / name) != record["files"][name]:
                raise ValueError(f"Changed ORB export {name}")
        return folder

    if args.orb_replay:
        return verified(args.orb_replay)
    if not args.vocabulary:
        raise ValueError("--vocabulary is required with --orb-runner")
    config, count = settings(args.sensor)
    for attempt in (1, 2):
        folder = args.output / f"orb_attempt_{attempt}"
        receipt = folder / "ORB_RAW_SEALED.json"
        if receipt.exists():
            if read(receipt)["returncode"] == 0:
                return verified(folder, current=True)
            continue
        folder.mkdir()
        (folder / "settings.yaml").write_text(config, encoding="utf-8")
        with (folder / "runner.log").open("wb") as log:
            result = subprocess.run(
                [
                    str(args.orb_runner),
                    str(args.vocabulary),
                    str(folder / "settings.yaml"),
                    str(args.sensor),
                ],
                cwd=folder,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        write(
            receipt,
            {
                "returncode": result.returncode,
                "frames": count,
                "GT_used": False,
                "source_archive_sha256": sensor_hash,
                "runner": digest(args.orb_runner),
                "vocabulary": digest(args.vocabulary),
                "settings": digest(folder / "settings.yaml"),
                "files": {
                    name: digest(folder / name)
                    for name in NAMES
                    if (folder / name).exists()
                },
            },
        )
        if result.returncode == 0:
            return verified(folder, current=True)
    raise RuntimeError(
        f"ORB tracking failed twice. Both attempts are preserved in {args.output}"
    )


def run(args):
    if not args.geometry_only and (not args.config or not args.checkpoint):
        raise ValueError("The single expert requires --config and --checkpoint")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        name: digest(getattr(args, name))
        for name in ("sensor", "trajectory", "native_map")
    }
    source = raw_exports(args, input_hashes["sensor"])
    input_hashes["orb_exports"] = digest(source / "ORB_RAW_SEALED.json")
    receipt = out / "geometry.json"
    if not receipt.exists():
        payload = read(args.trajectory)
        if (
            payload.get("gt_at_inference") is True
            or payload.get("matrix_convention") != "T_world_camera_m"
        ):
            raise ValueError(
                "Expected estimated metric world-from-camera poses in native sensor coordinates"
            )
        _, count = settings(args.sensor)
        rows = payload["poses"]
        if len(rows) != count or any(
            r["frame_id"] != i or not r.get("valid", True) for i, r in enumerate(rows)
        ):
            raise ValueError(
                "Expected a complete native DROID trajectory, one valid pose per raw sensor frame"
            )
        native = np.stack(
            [
                validate_se3(np.asarray(r["T_world_camera_m"]).reshape(4, 4))
                for r in rows
            ]
        )
        assisted, half, accepted = connected_poses(source, native)
        scene = payload["sequence_id"]
        with TemporaryDirectory(dir=out) as temp:
            tmp = Path(temp)
            records = [
                PoseRecord(
                    r["frame_id"],
                    r["timestamp_us"],
                    half[i],
                    source="half_DROID_ORB" if accepted[i] else "native_DROID",
                )
                for i, r in enumerate(rows)
            ]
            write_trajectory(
                tmp / "trajectory.json",
                records,
                sequence_id=scene,
                arm="half_DROID_ORB",
                metadata={"gt_consumed": False},
            )
            info = reconstruct(
                args.sensor, args.native_map, native, assisted, half, accepted, tmp
            )
            used = np.flatnonzero(accepted)
            np.save(tmp / "orb_supported.npy", accepted)
            files = {p.name: digest(p) for p in tmp.iterdir()}
            for name in files:
                os.replace(tmp / name, out / name)
            write(
                receipt,
                dict(
                    inputs=input_hashes,
                    files=files,
                    GT_used=False,
                    supported_frames=len(used),
                    first_common_frame=int(used[0]) if len(used) else None,
                    **info,
                ),
            )
    geometry = read(receipt)
    if geometry["inputs"] != input_hashes or any(
        digest(out / k) != v for k, v in geometry["files"].items()
    ):
        raise ValueError("Existing geometry or input identity changed")
    if args.geometry_only:
        return
    expert = out / "single"
    if not (expert / "result.json").exists():
        subprocess.run(
            [
                str(args.expert_python),
                str(Path(__file__).with_name("refine_native_panoptic.py")),
                "--mesh",
                str(out / "mesh.ply"),
                "--map",
                str(out / "A.ply"),
                "--trajectory",
                str(out / "trajectory.json"),
                "--config",
                str(args.config),
                "--checkpoint",
                str(args.checkpoint),
                "--orientation",
                "camera-normal",
                "--output",
                str(expert),
            ],
            check=True,
        )
    seal = read(expert / "result.json")
    expected = {
        "mesh": geometry["files"]["mesh.ply"],
        "map": geometry["files"]["A.ply"],
        "trajectory": geometry["files"]["trajectory.json"],
        "config": digest(args.config),
        "checkpoint": digest(args.checkpoint),
    }
    if (
        seal["inputs"] != expected
        or seal["orientation"].get("method") != "camera-normal"
        or any(digest(expert / k) != v for k, v in seal["outputs"].items())
    ):
        raise ValueError(
            "Single-expert inputs or outputs differ from the frozen geometry"
        )
    write(
        out / "result.json",
        {
            "GT_used": False,
            "geometry": digest(receipt),
            "single": digest(expert / "result.json"),
            "secondary": "single/map_labeled.ply",
            "primary_in_secondary": "primary_in_secondary.ply",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("sensor", "trajectory", "native-map", "output"):
        parser.add_argument(
            "--" + field, type=lambda x: Path(x).resolve(), required=True
        )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--orb-runner", type=lambda x: Path(x).resolve())
    choice.add_argument(
        "--orb-replay",
        type=lambda x: Path(x).resolve(),
        help="Directory containing sealed raw backend exports",
    )
    for name in ("vocabulary", "config", "checkpoint"):
        parser.add_argument("--" + name, type=lambda x: Path(x).resolve())
    parser.add_argument("--expert-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--geometry-only", action="store_true")
    run(parser.parse_args())
