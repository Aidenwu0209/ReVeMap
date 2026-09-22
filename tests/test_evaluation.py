"""Metric counterexamples and a real tiny export-to-GT evaluation pipeline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from pose_pipeline.artifacts import load_artifacts, write_artifact_manifest
from pose_pipeline.contracts import FrameRecord, PoseRecord, SequenceManifest, write_manifest, write_trajectory
from pose_pipeline.evaluation import evaluate_points, evaluate_result, main
from pose_pipeline.sam3_export import export


def points():
    xyz = np.column_stack([np.r_[np.arange(4) * .01, 1 + np.arange(4) * .01], np.zeros(8), np.ones(8)])
    return xyz, np.ones(8, np.int64), np.repeat([1, 2], 4)


def test_missing_whole_object_reduces_recall_and_gt_coverage():
    xyz, sem, inst = points()
    report = evaluate_points(xyz[:4], sem[:4], inst[:4], xyz, sem, inst)
    assert report["semantic_accuracy_labeled_only"] == 1
    assert report["reference_geometry_coverage"] == .5
    assert report["instances"]["precision"] == 1
    assert report["instances"]["recall"] == .5


def test_fragmentation_does_not_get_perfect_instance_precision():
    xyz, sem, inst = points()
    fragments = np.repeat([10, 11, 12, 13], 2)
    report = evaluate_points(xyz, sem, fragments, xyz, sem, inst)
    assert report["instances"]["precision"] == .5
    assert report["instances"]["recall"] == 1
    assert len(report["instances"]["split_reference_objects"]) == 2


def test_merged_objects_and_wrong_category_cannot_both_match():
    xyz, sem, inst = points()
    merged = evaluate_points(xyz, sem, sem, xyz, sem, inst)["instances"]
    assert merged["recall"] == .5
    assert merged["merged_predicted_objects"] == {"1": [1, 2]}
    wrong = evaluate_points(xyz, sem * 2, inst, xyz, sem, inst)
    assert wrong["instances"]["matched_instances"] == 0
    assert wrong["reference_semantic_accuracy_unknown_or_uncovered_as_error"] == 0


def test_unknown_recovery_and_existing_degradation_are_separate():
    xyz, sem, inst = points()
    old = {"semantic": np.r_[sem[:4], np.zeros(4, int)], "instance": np.r_[inst[:4], np.zeros(4, int)]}
    good = evaluate_points(xyz, sem, inst, xyz, sem, inst, before=old)["before_after"]
    assert good["newly_owned_points"] == 4
    assert good["newly_owned_correct_fraction"] == 1
    assert good["unknown_semantic_recovered_correct_fraction"] == 1
    assert good["existing_owner_degraded_points"] == 0
    wrong_sem = np.r_[np.full(4, 2), sem[4:]]
    bad = evaluate_points(xyz, wrong_sem, inst, xyz, sem, inst, before=old)["before_after"]
    assert bad["newly_owned_correct_fraction"] == 1
    assert bad["existing_owner_degraded_points"] == 0  # Naming cannot erase correct geometry ownership.
    assert bad["existing_semantic_degraded_points"] == 4


def test_unknown_and_partially_named_instances_remain_evaluable():
    xyz, sem, inst = points()
    unknown = np.zeros(8, int)
    before = {"semantic": unknown, "instance": np.r_[inst[:4], np.zeros(4, int)]}
    report = evaluate_points(xyz, unknown, inst, xyz, sem, inst, before=before)
    assert report["instances"]["matched_instances"] == 0
    assert report["instances"]["class_agnostic"]["recall"] == 1
    assert report["before_after"]["newly_owned_correct_fraction"] == 1
    partial = unknown.copy()
    partial[[0, 4]] = 1
    report = evaluate_points(xyz, partial, inst, xyz, sem, inst)
    assert report["instances"]["recall"] == 1
    assert report["instances"]["partially_named_instances"] == [1, 2]
    partial[1] = 2
    report = evaluate_points(xyz, partial, inst, xyz, sem, inst)
    assert report["instances"]["recall"] == .5
    assert report["instances"]["ambiguous_semantic_instances"] == [1]
    assert report["instances"]["class_agnostic"]["recall"] == 1


def test_empty_prediction_is_total_miss_without_nan():
    xyz, sem, inst = points()
    report = evaluate_points(np.empty((0, 3)), np.empty(0, int), np.empty(0, int), xyz, sem, inst)
    assert report["instances"]["recall"] == 0
    assert report["instances"]["precision"] is None
    assert report["reference_geometry_coverage"] == 0
    json.dumps(report, allow_nan=False)


@pytest.fixture
def fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    color, depth = raw / "color.jpg", raw / "depth.png"
    color.write_bytes(b"rgb")
    depth.write_bytes(b"depth")
    sequence = SequenceManifest("scannet", "scene0000_00", raw, 1000.,
        (FrameRecord(0, 1000, color, depth, (1., 1., 0., 0.)),), "synthetic-test")
    manifest = tmp_path / "input.json"
    write_manifest(manifest, sequence)
    trajectory = tmp_path / "trajectory.json"
    write_trajectory(trajectory, [PoseRecord(0, 1000, np.eye(4))], sequence_id=sequence.sequence_id, arm="synthetic")
    (raw / "pose").mkdir()
    np.savetxt(raw / "pose" / "0.txt", np.eye(4))
    xyz, sem, inst = points()
    vertices = np.zeros(8, dtype=[(key, "f8") for key in ("x", "y", "z")] + [(key, "u1") for key in ("red", "green", "blue")])
    for index, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, index]
    baseline = tmp_path / "map.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(baseline)
    reference = tmp_path / "reference"
    reference.mkdir()
    PlyData([PlyElement.describe(vertices, "vertex")]).write(reference / "scene0000_00_vh_clean_2.ply")
    (reference / "scene0000_00_vh_clean_2.0.010000.segs.json").write_text(json.dumps({"segIndices": inst.tolist()}))
    (reference / "scene0000_00.aggregation.json").write_text(json.dumps({"segGroups": [
        {"objectId": 0, "segments": [1], "label": "chair"},
        {"objectId": 1, "segments": [2], "label": "chair"}]}))
    def make_result(name="run", semantics=None, instances=None, bound=True):
        root = tmp_path / name
        fused = root / "fused"
        fused.mkdir(parents=True)
        np.savez(fused / "labels.npz", semantic=sem if semantics is None else semantics,
                 instance=inst if instances is None else instances, confidence=np.ones(8))
        (fused / "classes.json").write_text(json.dumps({"0": "unknown", "1": "chair", "2": "table"}))
        receipt = {"status": "completed", "geometry_xyz_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
                   "sga_inference_executed": False, "complete_full_sequence": True}
        if bound:
            receipt["provenance"] = {"manifest": str(manifest), "trajectory": str(trajectory),
                                      "scene_id": sequence.sequence_id, "dataset": "scannet"}
        (fused / "result.json").write_text(json.dumps(receipt))
        export(baseline, fused / "labels.npz", fused / "result.json", fused / "export")
        return root
    return dict(make_result=make_result, manifest=manifest, trajectory=trajectory,
                reference=reference, sequence=sequence, xyz=xyz, sem=sem, inst=inst)


def test_export_then_evaluate_without_manually_copying_metadata(fixture, tmp_path):
    root = fixture["make_result"]()
    for source in (root, root / "fused", root / "fused/export", root / "fused/export/ARTIFACTS.json"):
        report = evaluate_result(source, fixture["reference"])
        assert report["provenance_bound"] and report["prediction_unchanged"]
        assert report["instances"]["precision"] == report["instances"]["recall"] == 1
        assert str(root / "fused/classes.json") in report["input_sha256"]
        assert report["official_benchmark"] is False
    main(["--result", str(root), "--reference-dir", str(fixture["reference"]), "--output", str(tmp_path / "diagnostic")])
    assert (tmp_path / "diagnostic/diagnostic.json").is_file()


@pytest.mark.parametrize("change", ["classes", "map", "result"])
def test_changed_artifact_is_rejected(fixture, change):
    root = fixture["make_result"]()
    paths = load_artifacts(root)["paths"]
    with paths[change].open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        evaluate_result(root, fixture["reference"])


def test_wrong_trajectory_manifest_and_scene_are_rejected(fixture, tmp_path):
    root = fixture["make_result"]()
    wrong_trajectory = tmp_path / "other_trajectory.json"
    shifted = np.eye(4)
    shifted[0, 3] = 1
    write_trajectory(wrong_trajectory, [PoseRecord(0, 1000, shifted)], sequence_id="scene0000_00", arm="other")
    with pytest.raises(ValueError, match="differs from bound"):
        evaluate_result(root, fixture["reference"], trajectory_path=wrong_trajectory)
    wrong_manifest = tmp_path / "other_manifest.json"
    wrong_manifest.write_text(fixture["manifest"].read_text() + " ")
    with pytest.raises(ValueError, match="differs from bound"):
        evaluate_result(root, fixture["reference"], manifest_path=wrong_manifest)
    inventory = root / "fused/export/ARTIFACTS.json"
    payload = json.loads(inventory.read_text())
    payload["scene_id"] = "another_scene"
    inventory.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="scene differs"):
        evaluate_result(root, fixture["reference"])


def test_legacy_requires_explicit_unbound_opt_in(fixture):
    root = fixture["make_result"](bound=False)
    (root / "fused/export/ARTIFACTS.json").unlink()
    assert not load_artifacts(root)["provenance_bound"]
    with pytest.raises(ValueError, match="no bound input provenance"):
        evaluate_result(root, fixture["reference"], manifest_path=fixture["manifest"], trajectory_path=fixture["trajectory"])
    report = evaluate_result(root, fixture["reference"], manifest_path=fixture["manifest"],
                             trajectory_path=fixture["trajectory"], allow_unbound_legacy=True)
    assert report["provenance_bound"] is False


def test_before_after_uses_same_geometry_and_bound_inputs(fixture):
    root = fixture["make_result"]("after")
    old_inst = fixture["inst"].copy()
    old_inst[4:] = 0
    old = fixture["make_result"]("before", instances=old_inst)
    report = evaluate_result(root, fixture["reference"], before_result=old)
    assert report["before_after"]["newly_owned_correct_fraction"] == 1
    assert report["before_after"]["newly_owned_points"] == 4
    paths = load_artifacts(old)["paths"]
    vertices = PlyData.read(paths["map"])["vertex"].data[::-1].copy()
    PlyData([PlyElement.describe(vertices, "vertex")]).write(paths["map"])
    (old / "fused/export/ARTIFACTS.json").unlink()
    write_artifact_manifest(old / "fused/export", map_path=paths["map"], classes_path=paths["classes"],
                            result_path=paths["result"], manifest_path=paths["input_manifest"],
                            trajectory_path=paths["trajectory"])
    with pytest.raises(ValueError, match="identical point order"):
        evaluate_result(root, fixture["reference"], before_result=old)


def test_bound_inventory_requires_matching_trajectory_scene(fixture, tmp_path):
    root = fixture["make_result"]()
    paths = load_artifacts(root)["paths"]
    other = tmp_path / "wrong_scene.json"
    write_trajectory(other, [PoseRecord(0, 1000, np.eye(4))], sequence_id="other", arm="test")
    with pytest.raises(ValueError, match="trajectory scene differs"):
        write_artifact_manifest(tmp_path / "invalid", map_path=paths["map"], classes_path=paths["classes"],
                                result_path=paths["result"], manifest_path=paths["input_manifest"],
                                trajectory_path=other)


def test_final_inventory_takes_precedence_and_hashes_names(fixture):
    root = fixture["make_result"]()
    paths = load_artifacts(root)["paths"]
    names = root / "fused/instance_names.json"
    names.write_text("[]")
    write_artifact_manifest(root / "fused", map_path=paths["map"], classes_path=paths["classes"],
                            result_path=paths["result"], manifest_path=paths["input_manifest"],
                            trajectory_path=paths["trajectory"], extra_files={"names": names})
    assert load_artifacts(root)["paths"]["names"] == names
    names.write_text("[1]")
    with pytest.raises(ValueError, match="digest mismatch: names"):
        load_artifacts(root)


@pytest.mark.parametrize("failure", ["FAILURE.json", "FUSION_FAILURE.json"])
def test_failed_run_cannot_be_consumed_through_early_export(fixture, failure):
    root = fixture["make_result"]()
    (root / "runtime.json").write_text("{}")
    (root / "COMPLETE.json").write_text('{"status":"completed"}')
    (root / failure).write_text('{"status":"failed","error":"input changed"}')
    for source in (root, root / "fused", root / "fused/export", root / "fused/export/ARTIFACTS.json"):
        with pytest.raises(ValueError, match="failed run"):
            load_artifacts(source)


def test_running_attempt_requires_final_completion_receipt(fixture):
    root = fixture["make_result"]()
    (root / "INPUTS.json").write_text("{}")
    with pytest.raises(ValueError, match="not completed"):
        load_artifacts(root / "fused/export/ARTIFACTS.json")
    (root / "COMPLETE.json").write_text('{"status":"completed"}')
    assert load_artifacts(root)["provenance_bound"]


def test_exact_historical_refinement_receipt_needs_explicit_opt_in(fixture, tmp_path):
    import shutil
    raw = fixture["make_result"]()
    refined = tmp_path / "refined/scene0000_00"
    refined.mkdir(parents=True)
    shutil.copyfile(raw / "fused/export/map_labeled.ply", refined / "semantic_labeled.ply")
    shutil.copyfile(raw / "fused/classes.json", refined / "classes.json")
    shutil.copyfile(raw / "fused/labels.npz", refined / "map_labels.npz")
    receipt = {"scene": "scene0000_00", "changed_points": 0,
               "known_labels_preserved": True, "geometry_modified": False,
               "instance_ids_preserved": True, "GT_used": False, "fragment_policy": False,
               "scope": "fixed geometry semantic refinement; no SLAM or SGA inference", "objects": []}
    (refined / "RESULT.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="explicit legacy opt-in"):
        load_artifacts(refined)
    report = evaluate_result(refined, fixture["reference"], manifest_path=fixture["manifest"],
                             trajectory_path=fixture["trajectory"], allow_unbound_legacy=True)
    assert report["legacy_completion_accepted"] is True
    assert report["provenance_bound"] is False
    assert report["instances"]["recall"] == 1
    receipt["geometry_modified"] = True
    (refined / "RESULT.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="prediction must be completed"):
        load_artifacts(refined, allow_legacy_completion=True)


@pytest.mark.parametrize("status", ["failed", "mapping", "cancelling", "cancelled"])
def test_base_completed_does_not_hide_gui_attempt_failure(fixture, status):
    pipeline = fixture["make_result"]("session/attempts/attempt_test/pipeline")
    (pipeline / "runtime.json").write_text("{}")
    (pipeline / "COMPLETE.json").write_text('{"status":"completed"}')
    attempt = pipeline.parent
    (attempt / "attempt.json").write_text(json.dumps({"status": status}))
    paths = {"map": pipeline / "fused/export/map_labeled.ply", "classes": pipeline / "fused/classes.json",
             "result": pipeline / "fused/result.json"}
    refined = pipeline / "refinement/refined/capture"
    write_artifact_manifest(refined, map_path=paths["map"], classes_path=paths["classes"],
                            result_path=paths["result"], manifest_path=fixture["manifest"],
                            trajectory_path=fixture["trajectory"])
    for source in (pipeline, pipeline / "fused/export/ARTIFACTS.json", refined / "ARTIFACTS.json"):
        with pytest.raises(ValueError, match="GUI attempt is not completed"):
            load_artifacts(source)
    (attempt / "attempt.json").write_text('{"status":"completed"}')
    with pytest.raises(ValueError, match="no completed final result"):
        load_artifacts(refined)
    (pipeline / "GUI_RESULT.json").write_text('{"status":"completed"}')
    assert load_artifacts(refined)["provenance_bound"]
    (attempt.parent.parent / "session.json").write_text('{"status":"failed","attempt":"newer_attempt"}')
    assert load_artifacts(refined)["provenance_bound"]


def test_unrelated_enclosing_failure_does_not_poison_result(fixture, tmp_path):
    root = fixture["make_result"]()
    (tmp_path / "FAILURE.json").write_text('{"status":"failed"}')
    assert load_artifacts(root)["provenance_bound"]


def test_parent_gui_runtime_config_is_not_an_unfinished_run(fixture, tmp_path):
    root = fixture["make_result"]("synthetic-demo/pipeline", bound=False)
    (tmp_path / "runtime.json").write_text('{"cpu_python":"python"}')
    # Refinement workspaces have an input plan and runtime, not a raw run receipt.
    (tmp_path / "INPUT_PLAN.json").write_text("{}")
    assert load_artifacts(root)["provenance_bound"] is False
