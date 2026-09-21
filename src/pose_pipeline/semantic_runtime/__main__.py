"""User-facing model tests, raw deployment, and frozen enhancement replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import PROMPT, model_spec, read, registry, sha


def test_model(args):
    from .worker import name_tasks
    config = read(args.runtime) if args.runtime else {"models": {}}
    config.setdefault("models", {}).setdefault(args.model, {})
    if args.weights:
        config["models"][args.model]["weights"] = str(args.weights.resolve())
    if args.llama_server:
        config["models"][args.model]["llama_server"] = str(args.llama_server.resolve())
    if args.index:
        rows = read(args.index)
        crops = []
        for row in rows:
            path = Path(row["file"])
            if not path.is_absolute():
                path = (args.image_root or args.index.resolve().parent) / path
            crops.append({**row, "file": str(path.resolve()), "sha256": row.get("sha256") or sha(path)})
    else:
        crops = [{"file": str(p.resolve()), "sha256": sha(p)} for p in args.images]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        crops = crops[:args.limit]
    if not crops:
        raise ValueError("at least one image is required")
    name_tasks([{"task_id": 0, "crops": crops}], args.model, config, args.output.resolve())
    return read(args.output / "COMPLETE.json")


def add_commands(commands):
    listing = commands.add_parser("list-vlm-models", help="list all retained VLM/API test profiles")
    listing.set_defaults(handler=lambda args: print(json.dumps(registry(), indent=2)))
    test = commands.add_parser("test-vlm", help="run one selected model on real image crops")
    test.add_argument("--model", choices=list(registry()), required=True)
    images = test.add_mutually_exclusive_group(required=True)
    images.add_argument("--images", nargs="+", type=Path)
    images.add_argument("--index", type=Path, help="JSON list with file and optional sha256 fields")
    test.add_argument("--image-root", type=Path)
    test.add_argument("--weights", type=Path)
    test.add_argument("--llama-server", type=Path)
    test.add_argument("--runtime", type=Path)
    test.add_argument("--limit", type=int)
    test.add_argument("--output", type=Path, required=True)
    test.set_defaults(handler=lambda args: print(json.dumps(test_model(args), indent=2)))
    run = commands.add_parser("run-sam3", help="raw RGB-D mapping with serial or stage-parallel semantics")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--runtime", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--schedule", choices=("serial", "parallel"), default="parallel")
    run.add_argument("--vlm", choices=[k for k, v in registry().items() if v["kind"] != "ocr"], default="qwen3vl_2b_bf16")
    run.add_argument("--stride", type=int, default=5)
    def deploy(args):
        from .pipeline import run
        print(json.dumps(run(args), indent=2))
    run.set_defaults(handler=deploy)
    enhance = commands.add_parser("enhance-semantic", help="replay verified P2 and grounded naming on frozen geometry")
    enhance.add_argument("--bundle", type=Path, required=True)
    enhance.add_argument("--output", type=Path, required=True)
    enhance.add_argument("--surface", choices=("off", "verified"), default="verified")
    enhance.add_argument("--vlm", choices=list(registry()), default="qwen3vl_2b_bf16")
    def enhance_run(args):
        from .enhance import run
        print(json.dumps(run(args), indent=2))
    enhance.set_defaults(handler=enhance_run)
    refine = commands.add_parser("refine-semantic", help="quality views, real NF4/SAM3 evidence and unknown-point refinement")
    refine.add_argument("--workspace", type=Path, required=True)
    refine.add_argument("--stage", choices=("all", "prepare", "name", "decide", "ground", "apply"), default="all")
    refine.add_argument("--fragments", action="store_true", help="opt-in three-view unknown-fragment fill")
    def refine_run(args):
        from .refinement.__main__ import run
        run(args.workspace, args.stage, args.fragments)
    refine.set_defaults(handler=refine_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
