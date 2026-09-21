# SAM3 image/video tracking mechanism probe

The probe uses the existing ssh44 SAM3 runtime and checkpoint. It does not reproduce all of Any3DIS, generate a new complete scene map, or infer business-specific names. The single-concept clips were selected from raw RGB before inference. All frame IDs and the Orbbec visual boxes are frozen in `run_tracking_probe.py` and the output `PLAN.json`.

- Orbbec curtain: 48 frames 1200 through 1435, stride 5.
- Orbbec cable reel: 16 frames 2400 through 2475, stride 5.
- ScanNet0030 chair: 32 frames 0 through 155, stride 5.
- 3RScan bed: the six preserved frames 0, 1, 6, 7, 30, 31. This is not a complete clip or full-scene result.

Per-frame text SAM3 and text SAM3 video use exactly the same decoded RGB inputs, prompt string, threshold 0.5, original depth, poses and map. The video model uses its standard detector/tracker architecture; the image model has no cross-frame memory. No new frames are added only for the video arm. The frame stride differs from earlier full-span semantic mapping, so these results must not be presented as a drop-in comparison of those complete maps.

The reel assistance experiment uses one positive box on frame 2400. Image-only positive and positive-plus-negative controls are exported. A separate video session uses only that first-frame positive box and propagates without additional human prompts. This is extra supervision and is not automatic object naming. The negative floor box is only used by the image assistance control.

## Runtime and provenance

Runtime: `/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1/env/bin/python`, Python 3.12, PyTorch 2.7.1+cu128, RTX 5070 Ti. The original environment specification is unchanged (file SHA256 `bf541ae39b5f33d05f5ec938944551a96bf0a6dc76c56846c0de8677c1930f96`; canonical JSON SHA256 prefix `d0719190`). No package installation was required. CPU video storage limits GPU memory; the official video model is loaded with strict checkpoint matching and compilation disabled.

Checkpoint: existing local `sam3.pt`, SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`. The runner reads the actual official checkout commit and hashes source files, inputs and checkpoint before and after the run. It fails on drift, incomplete propagation, nonfinite scores or missing checkpoint tensors.

The runner is launched under a detached finite timeout, with `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and `PYTHONUNBUFFERED=1`. Commands on ssh44 WSL:

```sh
/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1/env/bin/python /mnt/d/SGF-SGA-experiments/sam3_guided_20260912_v1/run_tracking_probe.py --mode smoke
/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1/env/bin/python /mnt/d/SGF-SGA-experiments/sam3_guided_20260912_v1/run_tracking_probe.py --mode full
```

The output directory must not exist; to reproduce, copy the runner and the R4 code module into a new experiment directory and invoke that copied runner. Do not delete or overwrite frozen results. Smoke selects two curtain frames only; its original input receipt still lists the frozen 48-frame clip, while its output frame records identify the two actually processed observations. Full inference is gated by a successful seeded CUDA plus video smoke.

## Diagnostics and limits

- Foreground temporal IoU compares predicted union masks on original map points that pass the fixed 5 cm depth visibility test in both adjacent frames. Empty/empty foreground pairs are excluded, not scored as perfect agreement.
- Per-mask matching retains masks with at least 20 jointly visible map points and uses Hungarian IoU assignment, accepting IoU >= 0.25. Report unmatched masks as well as matched IoU.
- Merge/split-like transitions require >= 20 intersection points and intersection/min(area) >= 0.5 for multiple masks. These are prediction transitions, not verified errors.
- Video-ID continuity is evaluated only after geometric mask matching. Model IDs are not used as ground truth.
- Camera movement, imperfect geometry, occlusion and stable incorrect masks can affect these measures. RGB overlays are required for interpretation. Neither increased temporal IoU nor increased segmented coverage alone demonstrates semantic or instance accuracy.
- Timings include each arm's session/preprocessing/export work but exclude model construction and frozen-geometry loading. The single-frame assisted controls are timed separately from the independent-text arm.

Six unit tests cover visibility intersection, empty abstention, foreground disagreement, empty mask resizing, merge events hidden by union agreement, and ID swaps despite unchanged geometry.

## Supplemental blue-equipment and competition probes

`run_tracking_blue_probe.py` freezes four conditions at Orbbec frame4811: machine, blue machine, one positive box [169,0,431,335], and that positive box plus floor negative [440,165,620,300]. It is single-frame, manually assisted where boxes are used, and does not discover a true business category. It runs only after the main GPU process exits.

`run_tracking_blue_competition.py` makes no model calls. It verifies source RGB/depth hashes and identical depth mask shape, reconstructs old31 PixelClaims exactly against the old raw semantic cache, then adds saved blue machine as provisional35 with unchanged threshold/margin. The old production taxonomy and all maps stay unchanged. Its output establishes missing effective candidates at this specific frame, not full-scene semantic accuracy.
