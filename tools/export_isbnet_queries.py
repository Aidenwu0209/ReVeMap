"""Infer box-aware ScanNet object masks on a measured ReVeMap reconstruction."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from refine_native_panoptic import digest, geometric_superpoints

CHECKPOINT = "fd2202d76ee7d8adda0d1f546a3bd7abce169871bcc41c8c0c1a3e1a75ba7576"


def run(args):
    import open3d as o3d
    import torch
    import yaml
    from mmengine.config import ConfigDict
    from isbnet.data.custom import CustomDataset
    from isbnet.model import ISBNet
    from isbnet.util import rle_decode

    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {key: digest(getattr(args, key)) for key in ("map", "mesh", "config", "checkpoint")}
    if inputs["checkpoint"] != CHECKPOINT:
        raise ValueError("Expected the ISBNet ScanNet v2 head_3dyco_scannetv2_val checkpoint")
    source = args.anchor / "result.json"
    anchor = json.loads(source.read_text())
    if any(anchor["inputs"][key] != inputs[key] for key in ("map", "mesh")):
        raise ValueError("The anchor must use this exact reconstruction")
    rotation = np.asarray(anchor["rotation"])
    if rotation.shape != (3,3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise ValueError("Invalid native gravity rotation")
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    xyz, groups = geometric_superpoints(mesh)
    vertex = PlyData.read(args.mesh)["vertex"].data
    measured = np.column_stack([vertex[k] for k in "xyz"]).astype("f4")
    if not np.array_equal(xyz, measured):
        raise ValueError("Mesh vertex order differs between face and color readers")
    colors = np.column_stack([vertex[k] for k in ("red","green","blue")]).astype("f4") / 127.5 - 1.
    coords = CustomDataset.dataAugment(None,(xyz - xyz.mean(axis=0)) @ rotation.T,False,False,False)
    native = PlyData.read(args.map)["vertex"].data
    distance, nearest = cKDTree(xyz).query(np.column_stack([native[k] for k in "xyz"]), workers=2)
    dummy = torch.full((len(coords),), -100, dtype=torch.int64)
    sample = (args.map.stem, torch.from_numpy((coords-coords.min(axis=0))*50).long(),
              torch.from_numpy(coords).float(), torch.from_numpy(colors), dummy.clone(), dummy.clone(),
              torch.from_numpy(groups).long(), 0)
    cfg = ConfigDict(yaml.safe_load(args.config.read_text()))
    model = ISBNet(**cfg.model, dataset_name="scannetv2")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["net"]
    if tuple(state.pop("criterion.empty_weight").shape) != (19,):
        raise ValueError("Unexpected ScanNet training criterion dimensions")
    model.load_state_dict(state, strict=True)
    data = CustomDataset.collate_fn(ConfigDict(voxel_cfg=cfg.data.test.voxel_cfg), [sample])
    torch.manual_seed(942)
    model.eval()
    model.cuda()
    with torch.inference_mode():
        predictions = model(data)["pred_instances"]
    classes, scores, masks = [], [], []
    for pred in predictions:
        mask = rle_decode(pred["pred_mask"]).astype(bool)
        if len(mask) != len(xyz):
            raise ValueError("A box query lost mesh vertex correspondence")
        classes.append(int(pred["label_id"]) + 1)
        scores.append(float(pred["conf"]))
        masks.append(np.packbits(mask[nearest] & (distance <= .12)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as work:
        dest = Path(work) / "result"
        dest.mkdir()
        query = dest / "queries.npz"
        np.savez_compressed(query, classes=np.asarray(classes,"i2"), scores=np.asarray(scores,"f4"),
                            masks=np.asarray(masks,"u1").reshape(len(scores),(len(nearest)+7)//8), points=len(nearest))
        inputs["anchor"] = digest(source)
        audit = dict(inputs=inputs, GT_used=False, label_space="ScanNet20", vertex_count=len(nearest),
                     queries=len(scores), outputs={query.name:digest(query)})
        (dest/"result.json").write_text(json.dumps(audit, indent=2))
        dest.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mesh","map","anchor","config","checkpoint","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(parser.parse_args())
