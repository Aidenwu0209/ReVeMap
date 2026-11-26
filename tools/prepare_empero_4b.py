"""Prepare the pinned Q8 language GGUF and build its matching F16 vision projector."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from pose_pipeline.semantic_runtime.common import model_spec, sha, write


def prepare(language, checkpoint, converter, destination):
    spec = model_spec("qwen38d_4b_q8_gpu34")
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        stage = Path(scratch)
        source = stage / "vision_source"
        source.mkdir()
        for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
            (source / filename).symlink_to((checkpoint / filename).resolve(strict=True))
        output = stage / "weights"
        output.mkdir()
        shutil.copyfile(language, output / spec["gguf"])
        if sha(output / spec["gguf"]) != spec["weight_sha256"][spec["gguf"]]:
            raise ValueError("Unexpected Q8 language weights")
        subprocess.run([sys.executable, str(converter), str(source), "--mmproj", "--outtype", "f16",
                        "--outfile", str(output / spec["mmproj"])], check=True)
        if sha(output / spec["mmproj"]) != spec["weight_sha256"][spec["mmproj"]]:
            raise ValueError("Projector differs from the tested F16 weights")
        files = [{"file": name, "bytes": (output / name).stat().st_size, "sha256": digest}
                 for name, digest in spec["weight_sha256"].items()]
        write(output / "DOWNLOAD_COMPLETE.json", {"repo": spec["repo"], "revision": spec["revision"], "files": files})
        output.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("language", "checkpoint", "converter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.language, args.checkpoint, args.converter, args.output)
