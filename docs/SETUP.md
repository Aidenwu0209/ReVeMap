# Environment and input setup

## CPU environment

Use Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
revemap --help
python -m pytest -q
```

For TSDF integration and ICP association, install `.[geometry]` in the configured
CPU environment on a platform supported by Open3D. `.[visualization]` provides
Matplotlib for the optional preview script. CUDA/model packages are intentionally
not installed by the base package.

## Separate model environments

The runtime invokes explicit Python interpreters for each stage. Fill in
`configs/semantic_runtime.example.json` and save it as `configs/runtime.local.json`.

| Configuration | Requirement |
|---|---|
| `cpu_python` | CPU environment with ReVeMap dependencies and Open3D for geometry/ICP |
| `gpu_python`, `provider_root` | Existing compatible DROID-W checkout, CUDA environment, native extensions and checkpoint |
| `sam3_python`, `sam3_source`, `sam3_checkpoint`, `sam3_sha256` | SAM3 environment, source, weights and verified checkpoint hash |
| `vlm_python`, `models.<id>.weights` | Model-specific VLM environment and checkpoint directory |
| `models.<id>.python` | Optional interpreter override for a model requiring a different environment |
| `models.<id>.token_env` | Environment-variable name for an explicitly selected API; never an API key value |

The DROID-W provider must expose `src/modules/droid_net`, `src/depth_video.py`,
`src/frontend.py`, `src/backend.py`, `src/trajectory_filler.py`,
`src/motion_filter.py`, `configs/droid_w.yaml`, and `pretrained/droid.pth`.
It is a separate geometry backend, not an SGAligner dependency.

Model/checkpoint versions and historical runtime requirements are recorded in
[the model registry](../configs/vlm_models.json) and
[the source runtime guide](experiments/semantic-runtime-20260915/README.md).
Follow the model-specific setup instead of assuming all models share one
Transformers/CUDA environment. Model directories must contain their real
`DOWNLOAD_COMPLETE.json` receipt with pinned source and file hashes, as required
by the loaders. Models and environments are not downloaded by the CLI.

The runtime model registry and SAM3 taxonomy are packaged in
`src/revemap/resources/`. The root `configs/` copies are kept identical for reading
and review. If intentionally updating either, update both copies and rerun tests.

## RGB-D manifest

Use the `rgbd_sequence_manifest.v1` contract implemented by
`pose_pipeline.contracts`. Existing `developnew` manifests remain compatible.
The example below creates a manifest from registered RGB/depth pairs:

```python
from pathlib import Path
from pose_pipeline.contracts import FrameRecord, SequenceManifest, write_manifest

root = Path('/absolute/path/registered-sequence')
frames = (
    FrameRecord(
        frame_id=0,
        timestamp_us=0,
        color_path=root / 'color/000000.jpg',
        depth_path=root / 'depth/000000.png',
        intrinsics=(500.0, 500.0, 320.0, 240.0),  # replace with real calibration
    ),
)
manifest = SequenceManifest(
    dataset='orbbec', sequence_id='my-sequence', root=root,
    depth_scale=1000.0, frames=frames, source='registered RGB-D capture',
)
write_manifest(root / 'manifest.json', manifest)
```

Supply all actual frames in timestamp order. Depth images are uint16 and
`depth_scale` converts depth values to metres. Intrinsics must correspond to the
actual depth image and registration/rotation convention. The manifest supports
`scannet`, `3rscan` and `orbbec`; a new dataset adapter should establish its own
calibration contract. RGB-D paths must remain inside the sequence root.

Native unregistered RGB/depth cannot be made calibrated by resizing alone. The
direct-refinement input registration options are documented in
[the refinement guide](experiments/semantic-refinement-20260916/README.md).
Estimated trajectories use `T_world_camera` in metres; dataset GT is only for
separate evaluation and is not a pipeline input.

## Starting runs

```bash
revemap run-sam3 \
  --manifest /absolute/path/manifest.json \
  --runtime /absolute/path/runtime.local.json \
  --output /absolute/path/new-output \
  --schedule serial --vlm qwen3vl_2b_nf4 --stride 5
```

Serial and stage-parallel schedules preserve the existing algorithm settings.
Stage parallelism overlaps some CPU work/model loading; it does not keep SAM3
and a local VLM resident on the GPU simultaneously. The example profile is a
historically tested configuration, not a universal model recommendation.

The raw route produces `mapping/`, `semantic/`, `fused/export/map_labeled.ply`,
`fused/instance_names.json`, and `COMPLETE.json`. Completion records the executed
scope and is not a semantic-accuracy verdict.

For `enhance-semantic`, follow the [bundle instructions](experiments/semantic-runtime-20260915/README.md).
For `refine-semantic`, prepare `INPUT_PLAN.json`, `runtime.json` and each scene's
input files following the [refinement contract](experiments/semantic-refinement-20260916/README.md).
