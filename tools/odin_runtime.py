"""Load ODIN's closed-set ScanNet Swin with the original model parameters and heads."""

import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def output_lock(path):
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def publish(directory, filename, arrays, signature, metadata, seal_name="seal.json"):
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    seal = directory / seal_name
    with output_lock(directory / ".odin-write.lock"):
        if seal.is_file():
            old = json.loads(seal.read_text(encoding="utf-8"))
            if old.get("signature") != signature:
                raise FileExistsError(f"Existing output uses different inputs or settings: {directory}")
            if not path.is_file() or old["files"][filename] != sha256(path):
                raise ValueError(f"Existing output does not match its receipt: {path}")
            return
        with tempfile.TemporaryDirectory(dir=directory, prefix=".odin-staging-") as staging:
            staged = Path(staging)
            np.savez_compressed(staged / filename, **arrays)
            digest = sha256(staged / filename)
            result = dict(metadata, GT_used=False, signature=signature, files={filename: digest})
            (staged / seal_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
            (staged / filename).replace(path)
            (staged / seal_name).replace(seal)


def inference_tree(source, destination):
    # Optional training, language, visualization and evaluator imports are not needed by
    # this closed-set forward. Preserve the vendor checkout and record every changed file.
    edits = {}
    source_hash = hashlib.sha256()
    for path in sorted((source / "odin").rglob("*.py")):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original = path.read_text(encoding="utf-8")
        source_hash.update(relative.as_posix().encode("utf-8") + b"\x00" + bytes.fromhex(sha256(path)))
        code = original
        if relative.as_posix() in ("odin/__init__.py", "odin/data_video/__init__.py"):
            code = ""
        else:
            code = re.sub(r"(?m)^import ipdb\s*\nst\s*=\s*ipdb.set_trace",
                          'def st(*args, **kwargs): raise RuntimeError("ODIN diagnostic breakpoint")', code)
            code = re.sub(r"(?m)^import ipdb\s*$", "import pdb as ipdb", code)
            code = re.sub(r"(?m)^from transformers import ([A-Za-z_, ]+)\s*$",
                          lambda m: "\n".join(n.strip() + " = None" for n in m[1].split(",")), code)
            code = code.replace("import pyviz3d.visualizer as vis", "vis = None")
            code = code.replace("import wandb", "wandb = None")
            code = code.replace("from odin.utils import vis_utils", "vis_utils = None")
            if relative.as_posix().endswith("ops/functions/ms_deform_attn_func.py"):
                # ODIN's pixel decoder already calls its PyTorch MSDA reference routine.
                code = re.sub(r"except ModuleNotFoundError as e:.*?raise ModuleNotFoundError\(info_string\)",
                              "except ModuleNotFoundError:\n    MSDA = None", code, flags=re.S)
        target.write_text(code, encoding="utf-8")
        if code != original:
            edits[relative.as_posix()] = {"source": sha256(path), "inference": sha256(target)}
    return edits, source_hash.hexdigest()


def load_model(source, detectron2, weights, runtime, extra_paths):
    edits, source_hash = inference_tree(source, runtime)
    paths = [*extra_paths, runtime, source / "libs/pointops2", source / "libs", detectron2, source]
    sys.path[:0] = [str(p.resolve()) for p in paths]
    import torch
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from odin.config import add_maskformer2_config, add_maskformer2_video_config
    from odin.global_vars import NAME_MAP20
    from odin.odin_model import ODIN
    import odin.modeling  # register the author's Swin, decoders and attention blocks

    config = get_cfg()
    deeplab_path = detectron2 / "projects/DeepLab/deeplab/config.py"
    spec = importlib.util.spec_from_file_location("revemap_odin_deeplab", deeplab_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.add_deeplab_config(config)
    add_maskformer2_config(config)
    add_maskformer2_video_config(config)
    config.merge_from_file(str(source / "configs/scannet_context/swin_3d.yaml"))
    # Overrides from the released scripts/scannet/scannet_swin.sh.
    options = {
        "INPUT.FRAME_LEFT": 9, "INPUT.FRAME_RIGHT": 9, "INPUT.SAMPLING_FRAME_NUM": 19,
        "INPUT.IMAGE_SIZE": 256, "MODEL.CROSS_VIEW_CONTEXTUALIZE": True,
        "INPUT.CAMERA_DROP": True, "INPUT.STRONG_AUGS": True, "INPUT.AUGMENT_3D": True,
        "INPUT.VOXELIZE": True, "INPUT.SAMPLE_CHUNK_AUG": True,
        "MODEL.MASK_FORMER.TRAIN_NUM_POINTS": 50000, "MODEL.CROSS_VIEW_BACKBONE": True,
        "MODEL.PIXEL_DECODER_PANET": True, "MODEL.SEM_SEG_HEAD.NUM_CLASSES": 20,
        "MODEL.MASK_FORMER.TEST.SEMANTIC_ON": True, "SKIP_CLASSES": [19, 20],
        "USE_GHOST_POINTS": True, "MODEL.FREEZE_BACKBONE": False,
        "SAMPLING_STRATEGY": "consecutive", "USE_SEGMENTS": True, "MAX_FRAME_NUM": -1,
        "MODEL.MASK_FORMER.DICE_WEIGHT": 6.0, "MODEL.MASK_FORMER.MASK_WEIGHT": 15.0,
        "USE_WANDB": False, "USE_MLP_POSITIONAL_ENCODING": True, "MODEL.DEVICE": "cpu",
    }
    for key, value in options.items():
        element = config
        fields = key.split(".")
        for field in fields[:-1]:
            element = getattr(element, field)
        setattr(element, fields[-1], value)
    categories = [NAME_MAP20[i] for i in range(1, 21)]
    if categories[-2:] != ["wall", "floor"]:
        raise ValueError("Expected ODIN's ScanNet20 class order (wall 19 and floor 20)")
    MetadataCatalog.get(config.DATASETS.TRAIN[0]).thing_classes = categories
    model = ODIN(config).eval()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    receipt = {"weight_sha256": sha256(weights), "state_keys": len(state), "strict": True,
               "parameters": sum(p.numel() for p in model.parameters()), "classes": categories,
               "runtime_edits": edits, "source_python_sha256": source_hash, "config": config.dump()}
    return config, model, categories, receipt
