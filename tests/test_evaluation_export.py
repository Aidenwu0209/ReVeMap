"""External evaluator input contracts, independent of model inference or GT."""
import json

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from pose_pipeline.artifacts import write_artifact_manifest
from pose_pipeline.contracts import (
    FrameRecord, PoseRecord, SequenceManifest, sha256_file, write_manifest, write_trajectory,
)
from pose_pipeline.evaluation_export import export_evaluation


def make_artifact(root, *, frame=0, transform=None, bound=True, status="completed",
                  semantic=None, instances=None, names=None):
    root.mkdir()
    raw = root / "raw"
    raw.mkdir()
    color, depth = raw / "rgb.jpg", raw / "depth.png"
    color.write_bytes(b"rgb")
    depth.write_bytes(b"depth")
    manifest, trajectory = root / "input.json", root / "trajectory.json"
    seq = SequenceManifest("scannet", "scene0000_00", raw, 1000.,
                           (FrameRecord(frame, 1000, color, depth, (1., 1., 0., 0.)),), "synthetic")
    write_manifest(manifest, seq)
    if transform is None:
        transform = np.array([[0., -1., 0., 2.], [1., 0., 0., -3.],
                              [0., 0., 1., 4.], [0., 0., 0., 1.]])
    write_trajectory(trajectory, [PoseRecord(frame, 1000, transform)],
                     sequence_id=seq.sequence_id, arm="synthetic")
    vertices = np.zeros(5, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"),
                                  ("red", "u1"), ("semantic_id", "i4"), ("instance_id", "i4")])
    camera_xyz = np.array([[1., 2., 3.], [4., 1., 2.], [0., 0., 0.],
                           [2., -3., .25], [-1., .5, 2.]])
    world_xyz = camera_xyz @ transform[:3, :3].T + transform[:3, 3]
    for col, key in enumerate(("x", "y", "z")):
        vertices[key] = world_xyz[:, col]
    vertices["red"] = [10, 20, 30, 40, 50]
    vertices["semantic_id"] = semantic if semantic is not None else [0, 1, 2, 3, 4]
    vertices["instance_id"] = instances if instances is not None else [0, 7, 7, 9, 12]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(root / "map.ply")
    (root / "classes.json").write_text(json.dumps(names if names is not None else {
        "0": "unknown", "1": "chair", "2": "refridgerator", "3": "shower curtain", "4": "monitor"}))
    (root / "result.json").write_text(json.dumps({"status": status}))
    write_artifact_manifest(root, map_path=root / "map.ply", classes_path=root / "classes.json",
                            result_path=root / "result.json", manifest_path=manifest if bound else None,
                            trajectory_path=trajectory if bound else None)
    return camera_xyz, vertices


@pytest.mark.parametrize("transform", [None, np.eye(4)])
def test_inverse_pose_and_label_contract_preserve_sources(tmp_path, transform):
    root, output = tmp_path / "source", tmp_path / "export"
    expected_xyz, native = make_artifact(root, transform=transform)
    before = {path: sha256_file(path) for path in root.rglob("*") if path.is_file()}
    report = export_evaluation(root, output)
    converted = PlyData.read(output / "frame0_prediction.ply")["vertex"].data
    np.testing.assert_allclose(np.column_stack([converted[k] for k in ("x", "y", "z")]), expected_xyz)
    for key in ("semantic_id", "instance_id", "red"):
        np.testing.assert_array_equal(converted[key], native[key])
    np.testing.assert_array_equal(np.load(output / "semantic.npy"), [20, 4, 14, 15, 20])
    np.testing.assert_array_equal(np.load(output / "instance.npy"), [-1, 7, 7, 9, 12])
    assert {path: sha256_file(path) for path in before} == before
    assert report["GT_used"] is False
    assert report["vertex_count"] == 5
    assert report["unassigned_points"] == 1
    for name, digest in report["output_sha256"].items():
        assert sha256_file(output / name) == digest
    exported = {p: sha256_file(p) for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        export_evaluation(root, output)
    assert {p: sha256_file(p) for p in exported} == exported


@pytest.mark.parametrize("options,match", [
    ({"frame": 1}, "frame 0 is required"),
    ({"bound": False}, "no bound input provenance"),
    ({"status": "running"}, "must be completed"),
    ({"instances": [-1, 1, 2, 3, 4]}, "nonnegative integers"),
    ({"instances": [0, 65535, 2, 3, 4]}, "id_offset"),
    ({"semantic": [0, 1, 2, 3, 5]}, "no name for semantic_id 5"),
])
def test_bad_inputs_fail_without_publishing(tmp_path, options, match):
    root, output = tmp_path / "source", tmp_path / "export"
    make_artifact(root, **options)
    with pytest.raises(ValueError, match=match):
        export_evaluation(root, output)
    assert not output.exists()


@pytest.mark.parametrize("filename", ["map.ply", "classes.json", "trajectory.json"])
def test_hash_mismatch_is_rejected(tmp_path, filename):
    root, output = tmp_path / "source", tmp_path / "export"
    make_artifact(root)
    with (root / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        export_evaluation(root, output)
    assert not output.exists()


def test_changed_source_during_export_cannot_publish_completion(tmp_path, monkeypatch):
    root, output = tmp_path / "source", tmp_path / "export"
    make_artifact(root)
    original_save = np.save

    def changed(*args, **kwargs):
        original_save(*args, **kwargs)
        with (root / "classes.json").open("ab") as stream:
            stream.write(b" ")

    monkeypatch.setattr(np, "save", changed)
    with pytest.raises(ValueError, match="digest mismatch"):
        export_evaluation(root, output)
    assert not (output / "EXPORT.json").exists()
