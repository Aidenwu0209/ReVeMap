"""Versioned, hash-bound result files shared by export, evaluation and clients.

Paths are relative to the inventory whenever possible. A legacy directory can
be inspected, but cannot silently acquire verified input provenance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .contracts import sha256_file

ARTIFACT_SCHEMA = "revemap.artifacts.v1"
ARTIFACT_FILENAME = "ARTIFACTS.json"


def _read(path):
    return json.loads(Path(path).read_text())


def _identity(manifest_path, trajectory_path, scene_id=None, dataset=None):
    from .contracts import bind_manifest_trajectory, load_manifest, load_trajectory
    manifest = load_manifest(manifest_path, require_files=False)
    poses, trajectory = load_trajectory(trajectory_path)
    if trajectory.get("sequence_id") != manifest.sequence_id:
        raise ValueError("trajectory scene differs from input manifest")
    bind_manifest_trajectory(manifest, poses)
    if scene_id is not None and scene_id != manifest.sequence_id:
        raise ValueError("artifact scene differs from input manifest")
    if dataset is not None and dataset != manifest.dataset:
        raise ValueError("artifact dataset differs from input manifest")
    return manifest.sequence_id, manifest.dataset


def write_artifact_manifest(output_dir, *, map_path, classes_path, result_path,
                            manifest_path=None, trajectory_path=None,
                            scene_id=None, dataset=None, extra_files=None):
    """Write an inventory only after its immutable files have been finalized."""
    output = Path(output_dir).resolve()
    if (manifest_path is None) != (trajectory_path is None):
        raise ValueError("input manifest and trajectory must be supplied together")
    paths = {"map": map_path, "classes": classes_path, "result": result_path}
    if manifest_path is not None:
        scene_id, dataset = _identity(manifest_path, trajectory_path, scene_id, dataset)
        paths.update(input_manifest=manifest_path, trajectory=trajectory_path)
    for name, path in (extra_files or {}).items():
        if name in paths:
            raise ValueError("extra file overrides a required artifact: " + name)
        paths[name] = path
    entries = {}
    for name, path in paths.items():
        path = Path(path).resolve()
        entries[name] = {"path": os.path.relpath(path, output), "sha256": sha256_file(path)}
    value = {"schema": ARTIFACT_SCHEMA, "scene_id": scene_id, "dataset": dataset,
             "provenance_bound": manifest_path is not None, "files": entries}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ARTIFACT_FILENAME
    # Create-only avoids accidentally replacing the provenance of an old run.
    with destination.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return destination


def _legacy_paths(directory):
    """Resolve both raw fused/export and frozen enhancement directory layouts."""
    roots = [directory, directory / "export", directory / "fused" / "export", directory / "fused"]
    for root in roots:
        for name in ("map_labeled.ply", "semantic_labeled.ply"):
            map_path = root / name
            if not map_path.is_file():
                continue
            for metadata in (root, root.parent):
                for result_name in ("result.json", "RESULT.json"):
                    paths = {"map": map_path, "classes": metadata / "classes.json",
                             "result": metadata / result_name}
                    if all(path.is_file() for path in paths.values()):
                        return paths
    raise ValueError("no complete ReVeMap result found; expected map, classes and result files")


def _check_run_state(source, inventory_path=None):
    """Reject early exports from failed or still-running pipeline attempts.

The bounded ancestor walk covers run/fused/export and a GUI final inventory;
it does not scan unrelated runs or the entire filesystem hierarchy.
"""
    state_files = set()
    origins = {source if source.is_dir() else source.parent}
    if inventory_path is not None:
        origins.add(inventory_path.parent)
    candidates = set(origins)
    run_directories = set(origins)
    for origin in origins:
        ancestry = [origin, *list(origin.parents)[:8]]
        candidates.update(ancestry)
        # The nearest persisted launch owns this output. A different enclosing
        # repository's runtime or FAILURE file must not poison a nested result.
        for directory in ancestry:
            if (directory / "INPUTS.json").is_file():
                run_directories.add(directory)
                break
        if origin.name == "export" and origin.parent.name == "fused":
            run_directories.add(origin.parent.parent)
        elif origin.name == "fused":
            run_directories.add(origin.parent)
    for directory in candidates:
        attempt = directory / "attempt.json"
        belongs_to_attempt = any(origin.is_relative_to(directory / "pipeline") for origin in origins)
        if attempt.is_file() and belongs_to_attempt:
            state_files.add(attempt)
            if _read(attempt).get("status") != "completed":
                raise ValueError("GUI attempt is not completed: " + str(directory))
            gui_result = directory / "pipeline" / "GUI_RESULT.json"
            if not gui_result.is_file() or _read(gui_result).get("status") != "completed":
                raise ValueError("GUI attempt has no completed final result: " + str(directory))
            state_files.add(gui_result)
    for directory in run_directories:
        failures = [directory / name for name in ("FAILURE.json", "FUSION_FAILURE.json")]
        if any(path.is_file() for path in failures):
            raise ValueError("result belongs to a failed run: " + str(directory))
        # An export-only fixture or archived replay need not invent a pipeline
        # receipt. Actual raw attempts have these persisted launch inputs.
        if not (directory / "INPUTS.json").is_file():
            continue
        state_files.add(directory / "INPUTS.json")
        complete = directory / "COMPLETE.json"
        gui = directory / "GUI_RESULT.json"
        finished = any(path.is_file() and _read(path).get("status") == "completed"
                       for path in (complete, gui))
        if not finished:
            raise ValueError("pipeline run is not completed: " + str(directory))
        state_files.update(path for path in (complete, gui) if path.is_file())
    return {str(path): sha256_file(path) for path in state_files}


def _legacy_refinement_complete(receipt, result_path):
    """Recognize only the historical refinement writer's exact receipt shape."""
    return (result_path.name.casefold() == "result.json" and "status" not in receipt
            and receipt.get("scope") == "fixed geometry semantic refinement; no SLAM or SGA inference"
            and receipt.get("known_labels_preserved") is True
            and receipt.get("geometry_modified") is False
            and receipt.get("instance_ids_preserved") is True
            and receipt.get("GT_used") is False
            and isinstance(receipt.get("scene"), str) and bool(receipt["scene"])
            and type(receipt.get("changed_points")) is int and receipt["changed_points"] >= 0
            and isinstance(receipt.get("fragment_policy"), bool)
            and isinstance(receipt.get("objects"), list)
            and (result_path.parent / "map_labels.npz").is_file())


def load_artifacts(source, *, manifest_path=None, trajectory_path=None, require_provenance=False,
                   allow_legacy_completion=False):
    """Resolve files, verify hashes and optionally require bound RGB-D provenance.

Returns ``paths`` (Path values), the parsed ``manifest``, its ``manifest_path``
and ``provenance_bound``. Explicit input paths must match the bound digests.
"""
    source = Path(source).resolve()
    if source.is_file():
        candidates = [source]
    else:
        candidates = [source / ARTIFACT_FILENAME, source / "fused" / ARTIFACT_FILENAME,
                      source / "export" / ARTIFACT_FILENAME,
                      source / "fused" / "export" / ARTIFACT_FILENAME]
    inventory_path = next((path for path in candidates if path.is_file()), None)
    snapshot = _check_run_state(source, inventory_path)
    inventory = None
    if inventory_path is not None:
        snapshot[str(inventory_path)] = sha256_file(inventory_path)
        inventory = _read(inventory_path)
        if inventory.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("unsupported artifact schema")
        entries = inventory.get("files", {})
        if not {"map", "classes", "result"} <= entries.keys():
            raise ValueError("artifact inventory is missing required files")
        paths = {}
        for name, entry in entries.items():
            path = (inventory_path.parent / entry["path"]).resolve()
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError("artifact digest mismatch: " + name)
            snapshot[str(path)] = entry["sha256"]
            paths[name] = path
        bound = {"input_manifest", "trajectory"} <= paths.keys()
        if inventory.get("provenance_bound") is not bound:
            raise ValueError("inconsistent artifact provenance declaration")
        if bound:
            _identity(paths["input_manifest"], paths["trajectory"],
                      inventory.get("scene_id"), inventory.get("dataset"))
    else:
        if not source.is_dir():
            raise ValueError("artifact inventory does not exist")
        paths = _legacy_paths(source)
        bound = False
        snapshot.update({str(path): sha256_file(path) for path in paths.values()})
    # A copied inventory must not detach an early map from its failed owner.
    for path, digest in _check_run_state(paths["map"], paths["result"]).items():
        if path in snapshot and snapshot[path] != digest:
            raise RuntimeError("artifact completion changed while loading: " + path)
        snapshot[path] = digest
    if require_provenance and not bound:
        raise ValueError("result has no bound input provenance; explicitly allow unbound legacy evaluation")
    for name, supplied in (("input_manifest", manifest_path), ("trajectory", trajectory_path)):
        if supplied is not None:
            supplied = Path(supplied).resolve()
            if name in paths and sha256_file(supplied) != sha256_file(paths[name]):
                raise ValueError("supplied " + name + " differs from bound artifact")
            paths[name] = supplied
            digest = sha256_file(supplied)
            if str(supplied) in snapshot and snapshot[str(supplied)] != digest:
                raise RuntimeError("artifact changed while loading: " + str(supplied))
            snapshot[str(supplied)] = digest
    receipt = _read(paths["result"])
    legacy_completed = False
    if receipt.get("status") != "completed":
        legacy_completed = (inventory_path is None and allow_legacy_completion
                            and _legacy_refinement_complete(receipt, paths["result"]))
        if not legacy_completed:
            raise ValueError("prediction must be completed; recognized old refinement receipts require explicit legacy opt-in")
    for path, digest in snapshot.items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("artifact changed while loading: " + path)
    _check_run_state(source, inventory_path)
    return {"paths": paths, "manifest": inventory, "manifest_path": inventory_path,
            "provenance_bound": bound, "legacy_completion_accepted": bool(legacy_completed),
            "input_sha256": snapshot,
            "load_options": dict(source=source, manifest_path=manifest_path, trajectory_path=trajectory_path,
                                 require_provenance=require_provenance, allow_legacy_completion=allow_legacy_completion)}


def verify_artifact_snapshot(bundle):
    """Recheck completion and resolution, including a newly published final inventory."""
    current = load_artifacts(**bundle["load_options"])
    if (current["manifest_path"] != bundle["manifest_path"] or current["paths"] != bundle["paths"]
            or current["input_sha256"] != bundle["input_sha256"]):
        raise RuntimeError("artifact selection or completion changed during consumption")
