"""Geometry decisions, PLY field preservation, and explicit output contracts."""
import hashlib
import json

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from pose_pipeline.pointcloud_cleanup import cleanup_pointcloud, radius_keep_mask


def write_cloud(path, *, text=False, byte_order="<"):
    vertices = np.zeros(6, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"),
                                   ("red", "u1"), ("semantic_id", "i4"),
                                   ("instance_id", "i4"), ("nx", "f4"), ("confidence", "f4")])
    vertices["x"] = [0, 0.01, 0.02, 0.03, 0.04, 5]
    vertices["red"] = [0, 45, 90, 135, 200, 255]
    vertices["semantic_id"] = [0, -1, 22, 5, 7, 3]
    vertices["instance_id"] = [17, 4, 99, 0, -1, 100]
    vertices["nx"] = np.linspace(-1, 1, 6)
    vertices["confidence"] = [0.1, 0.5, np.nan, 1, 0, 0.9]
    PlyData([PlyElement.describe(vertices, "vertex", comments=["field metadata"])],
            text=text, byte_order=byte_order, comments=["source metadata"],
            obj_info=["original observations"]).write(path)
    return PlyData.read(path)["vertex"].data.copy()


def test_excludes_self_and_uses_one_pass():
    # Endpoints have one neighbour and are removed; the interior remains even
    # though filtering survivors a second time would remove them as well.
    points = np.column_stack(([0., 1., 2., 3.], np.zeros(4), np.zeros(4)))
    np.testing.assert_array_equal(radius_keep_mask(points, radius_m=1, min_other_neighbors=2),
                                  [False, True, True, False])
    assert not radius_keep_mask([[0, 0, 0]], min_other_neighbors=1)[0]
    np.testing.assert_array_equal(radius_keep_mask([[0, 0, 0], [0, 0, 0]], min_other_neighbors=1),
                                  [True, True])


@pytest.mark.parametrize("text,byte_order", [(False, "<"), (False, ">"), (True, "<")])
def test_cleanup_preserves_fields_and_original_rows(tmp_path, text, byte_order):
    source = tmp_path / "source.ply"
    original = write_cloud(source, text=text, byte_order=byte_order)
    before = source.read_bytes()
    output = tmp_path / "output"
    receipt = cleanup_pointcloud(source, output)
    assert receipt["kept_points"] == 5 and receipt["removed_points"] == 1
    assert source.read_bytes() == before
    for name, expected in (("kept", [0, 1, 2, 3, 4]), ("removed", [5])):
        np.testing.assert_array_equal(np.load(output / f"{name}_indices.npy", allow_pickle=False), expected)
        ply = PlyData.read(output / f"{name}.ply")
        actual = ply["vertex"].data
        assert actual.dtype == original.dtype
        assert actual.dtype.names == original.dtype.names
        assert ply.comments == ["source metadata"]
        assert ply.obj_info == ["original observations"]
        assert ply["vertex"].comments == ["field metadata"]
        for field in original.dtype.names:
            assert np.array_equal(actual[field], original[field][expected],
                                  equal_nan=original[field].dtype.kind == "f")
    assert json.loads((output / "receipt.json").read_text()) == receipt
    for name, digest in receipt["artifact_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_empty_removed_set_is_an_explicit_artifact(tmp_path):
    source = tmp_path / "source.ply"
    write_cloud(source)
    output = tmp_path / "all_kept"
    receipt = cleanup_pointcloud(source, output, radius_m=10)
    assert receipt["removed_points"] == 0
    assert len(PlyData.read(output / "removed.ply")["vertex"].data) == 0
    assert np.load(output / "removed_indices.npy").size == 0


def test_dangling_output_symlink_is_not_followed(tmp_path):
    source = tmp_path / "source.ply"
    write_cloud(source)
    output = tmp_path / "output"
    target = tmp_path / "missing_target"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        cleanup_pointcloud(source, output)
    assert output.is_symlink() and not target.exists()


def test_all_removed_and_existing_output_are_rejected(tmp_path):
    source = tmp_path / "source.ply"
    write_cloud(source)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="No vertices survive"):
        cleanup_pointcloud(source, output, radius_m=0.001)
    assert not output.exists()
    output.mkdir()
    marker = output / "owned.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        cleanup_pointcloud(source, output)
    assert marker.read_text() == "unchanged"


@pytest.mark.parametrize("radius,count", [(0, 4), (np.nan, 4), (np.inf, 4), (True, 4), (1, 0), (1, 1.5), (1, True)])
def test_invalid_parameters_fail(radius, count):
    with pytest.raises(ValueError):
        radius_keep_mask([[0, 0, 0]], radius_m=radius, min_other_neighbors=count)


@pytest.mark.parametrize("points", [[], [[np.nan, 0, 0]], [[0, np.inf, 0]], [[0, 0]]])
def test_invalid_geometry_fails(points):
    with pytest.raises(ValueError):
        radius_keep_mask(points)


def test_topology_and_vertex_lists_are_rejected(tmp_path):
    source = tmp_path / "source.ply"
    write_cloud(source)
    cloud = PlyData.read(source)
    edge = np.array([(0, 1)], dtype=[("vertex1", "i4"), ("vertex2", "i4")])
    PlyData([cloud["vertex"], PlyElement.describe(edge, "edge")]).write(source)
    with pytest.raises(ValueError, match="topology"):
        cleanup_pointcloud(source, tmp_path / "edge_output")
    values = np.empty(1, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("views", "O")])
    values["x"] = values["y"] = values["z"] = 0
    values["views"][0] = np.array([1, 2], dtype=np.int32)
    PlyData([PlyElement.describe(values, "vertex")]).write(source)
    with pytest.raises(ValueError, match="scalar"):
        cleanup_pointcloud(source, tmp_path / "list_output")


def test_input_change_does_not_emit_success_receipt(tmp_path, monkeypatch):
    import pose_pipeline.pointcloud_cleanup as module
    source = tmp_path / "source.ply"
    write_cloud(source)
    output = tmp_path / "output"
    original = module.radius_keep_mask

    def change_source(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(module, "radius_keep_mask", change_source)
    with pytest.raises(RuntimeError, match="Input changed"):
        cleanup_pointcloud(source, output)
    assert not output.exists()
