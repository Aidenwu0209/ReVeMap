"""Packaging contracts that are essential to an independent repository."""
import json
from pathlib import Path
import subprocess
import sys

from pose_pipeline.semantic_runtime.common import CONFIG_ROOT, SOURCE_ROOT, registry


def test_packaged_configs_match_reviewable_copies():
    root = Path(__file__).resolve().parents[1]
    for name in ("vlm_models.json", "sam3_indoor_v1.json"):
        assert (CONFIG_ROOT / name).read_bytes() == (root / "configs" / name).read_bytes()
    assert "none" in registry()


def test_public_cli_works_outside_checkout_without_gpu_imports(tmp_path):
    for arguments in (["--help"], ["--version"], ["list-vlm-models"], ["gui", "--help"],
                      ["clean-pointcloud", "--help"]):
        code = (
            "import runpy,sys; sys.argv=['revemap', *sys.argv[1:]]; "
            "sys.path.insert(0, " + repr(str(SOURCE_ROOT)) + "); "
            "runpy.run_module('revemap',run_name='__main__')"
        )
        result = subprocess.run([sys.executable, "-c", code, *arguments],
                                cwd=tmp_path, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        if arguments == ["list-vlm-models"]:
            assert "none" in json.loads(result.stdout)


def test_active_core_has_no_legacy_neural_imports():
    import ast
    roots = (SOURCE_ROOT / "pose_pipeline", SOURCE_ROOT / "reconstruction")
    forbidden = {"adapters", "aligner", "engine", "inference", "matching", "safety",
                 "sgf_runtime", "sgf_native", "GeoTransformer"}
    for root in roots:
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    assert not ({n.name.split('.')[0] for n in node.names} & forbidden), path
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    assert node.module.split('.')[0] not in forbidden, path
