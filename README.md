# ReVeMap

**Multi-View Verified Instance Recovery for RGB-D Semantic Mapping**

ReVeMap is a research prototype for building RGB-D semantic/instance maps and
recovering supported instance ownership and semantic labels from multiple views.
It combines estimated RGB-D geometry, SAM3 masks, measured object association,
multi-view recovery, and optional vision-language object naming.

[中文说明](README_zh.md) · [Installation and inputs](docs/SETUP.md) ·
[Capture GUI](docs/gui/README.md) · [Source and migration](docs/MIGRATION.md)

## Pipeline and scope

| Entry point | Input and operation | Output and boundary |
|---|---|---|
| `run-sam3` | Raw RGB-D → estimated poses/geometry → SAM3 → geometric association and multi-view instance fusion/recovery | Labeled PLY and optional VLM name metadata; does not automatically run frozen-evidence T1/P2 enhancement |
| `enhance-semantic` | Existing map and prediction bundle → T1 anchored association → P2 verified completion → confirmed naming | Updated semantic/instance labels on unchanged XYZ; replays real evidence already present in the bundle |
| `refine-semantic` | Existing map, estimated poses and RGB-D → selected views → fresh VLM/SAM3 evidence → unknown-point assignment | Supported semantic recovery on existing instances; not a general generator of unseen object geometry |
| `gui` | Orbbec capture or registered RGB-D replay → preview → `run-sam3` → optional refinement | Local browser viewer, labeled PLY and trajectory export |

The raw pipeline and frozen-map enhancement have separate input/output contracts.
VLM names from `run-sam3` are metadata; verified enhancement/refinement can write
semantic IDs. Unknown labels are allowed. The GUI's semantic result is produced
after capture; this repository does not claim real-time semantic inference.

## Quick start

Python 3.11 or newer is required. Core checks run on CPU. Raw mapping and model
inference require separately configured GPU environments and checkpoints.

```bash
git clone https://github.com/Aidenwu0209/ReVeMap.git
cd ReVeMap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
revemap --help
revemap list-vlm-models
python -m pytest -q
```

Copy [the runtime example](configs/semantic_runtime.example.json) to
`configs/runtime.local.json`, then fill in real absolute environment, source and
checkpoint paths. See [setup](docs/SETUP.md) for external dependencies and the
RGB-D manifest contract. Local runtime files and credentials are ignored by Git.

```bash
revemap run-sam3 \
  --manifest /absolute/path/rgbd-manifest.json \
  --runtime /absolute/path/runtime.local.json \
  --output /absolute/path/new-map \
  --schedule serial --vlm qwen3vl_2b_nf4 --stride 5

revemap refine-semantic \
  --workspace /absolute/path/prepared-refinement --stage all

revemap enhance-semantic \
  --bundle /absolute/path/BUNDLE.json \
  --output /absolute/path/new-enhanced-map \
  --surface verified --vlm qwen3vl_2b_bf16
```

Output directories must be new. `--vlm none` disables naming in `run-sam3`.
The optional refinement stage uses its own Qwen NF4/SAM3 configuration; disable
that stage as well when using the GUI without VLM inference.

`python -m revemap` is equivalent to `revemap`. The historical
`python -m pose_pipeline.semantic_runtime` entry remains available for existing
runtime configurations. SGF, learned SGAligner, and GeoTransformer are not needed.

## Research status

The project name describes the research direction. There is no associated
published ReVeMap paper or claim of benchmark superiority in this release.

The extraction preserves historical development evidence, including:

- [SAM3 sampling experiments](docs/experiments/sam3-stride5-20260913/README.md).
- [Instance association, P2 completion and confirmed naming](docs/experiments/semantic-runtime-20260915/README.md).
- [Model/runtime comparisons](docs/experiments/rtx4060-models-20260915/README.md).
- [Quality-view unknown-point refinement](docs/experiments/semantic-refinement-20260916/README.md).

These reports include limited development-scene gains and failed variants. They
are not new ReVeMap experiments, full-dataset acceptance, or official ScanNet
AP/mIoU results. Fixed-geometry recovery does not repair missing/folded geometry.
See [migration verification](docs/VERIFICATION.md) for what was checked during
the repository split and what still needs GPU/device validation.

## Layout

```text
src/revemap/                Public CLI and packaged model/taxonomy resources
src/pose_pipeline/          RGB-D, SAM3, association, recovery, VLM and GUI code
src/reconstruction/         Shared RGB-D TSDF fusion
configs/                   Portable configuration examples
tests/                     CPU regression and data-contract checks
scripts/                   GUI launcher, evidence conversion and diagnostics
docs/experiments/          Historical source-version experiment records
SOURCE_PROVENANCE.json     Exact extraction source and per-file hashes
```

## Attribution

Extracted from `SGF-SGAligner/developnew` at
[`18ddeca`](https://github.com/Aidenwu0209/SGF-SGAligner/tree/18ddeca303bf9937e855d3ac3113a90dd852a462).
The original repository is unchanged. The independent repository omits unrelated
legacy alignment/training code, datasets, weights and machine-local configuration.
See [NOTICE.md](NOTICE.md) and [MIT license](LICENSE).
