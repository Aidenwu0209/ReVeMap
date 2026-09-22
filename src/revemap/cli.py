"""Public ReVeMap entry point with lazy GPU and camera dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__


def main():
    if sys.argv[1:2] == ["evaluate-semantic"]:
        from pose_pipeline.evaluation import main as evaluate_main
        return evaluate_main(sys.argv[2:])
    if sys.argv[1:2] == ["gui"]:
        from pose_pipeline.live_gui import main as gui_main
        sys.argv = [f"{sys.argv[0]} gui", *sys.argv[2:]]
        return gui_main()

    from pose_pipeline.semantic_runtime.__main__ import add_commands
    parser = argparse.ArgumentParser(
        prog="revemap",
        description="ReVeMap: multi-view verified instance recovery for RGB-D semantic mapping",
    )
    parser.add_argument("--version", action="version", version=f"ReVeMap {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    add_commands(commands)
    from pose_pipeline.scene_graph import add_commands as add_scene_commands
    add_scene_commands(commands)
    from .demo import add_command as add_demo_command
    add_demo_command(commands)
    commands.add_parser("evaluate-semantic", help="evaluate a completed artifact; use evaluate-semantic --help")
    commands.add_parser("gui", help="capture/replay GUI; use revemap gui --help for options")
    mapping = commands.add_parser("run-rgbd", help="build geometry and estimated poses from raw RGB-D")
    for name in ("manifest", "output", "provider-root", "gpu-python", "cpu-python"):
        mapping.add_argument(f"--{name}", type=Path, required=True)
    mapping.add_argument("--threads", type=int, default=2)
    mapping.add_argument("--stage-timeout", type=float, default=7200)
    mapping.add_argument("--device", default="0")

    def run_mapping(args):
        from pose_pipeline.rgbd_mapping import run_rgbd_mapping
        result = run_rgbd_mapping(
            manifest_path=args.manifest, output_dir=args.output,
            provider_root=args.provider_root, gpu_python=args.gpu_python,
            cpu_python=args.cpu_python, threads=args.threads,
            stage_timeout_s=args.stage_timeout, device=args.device,
        )
        print(json.dumps(result, indent=2))

    mapping.set_defaults(handler=run_mapping)
    args = parser.parse_args()
    return args.handler(args)
