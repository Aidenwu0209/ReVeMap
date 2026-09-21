# SAM3 guided, unknown-object, naming and tracking validation

This opt-in experiment starts at f99f359. It is not promoted to developnew.
Read REPORT_zh.md for outcomes and limits. Evidence root on macOS:
/Users/wu/Desktop/wu/SGF-SGA/comparisons/sam3_guided_20260912_v1
SSH44 evidence root:
/mnt/d/SGF-SGA-experiments/sam3_guided_20260912_v1

Three final CPU arms each processed the same five scene caches; 15 maps preserve
all source vertex properties and all baseline non-instance label arrays. Ten
explicit naming runs write objects_named.json separately without recoloring point
semantics. A real SAM3 GPU probe compares identical selected input frames in
image/video mode; only 102 single-concept observations, plus assisted prompts
and one blue-equipment frame. 3RScan remains six archived frames, not full-scene.

The attached runners were executed from the evidence root, whose inputs/JOBS.json
refers to unchanged R1/R2/R3 caches. They refuse existing prediction destinations.
For a new inference run, deploy the runner, plan, inputs and matching source bundle
to a fresh root; do not execute into a sealed output directory. The final guided
runner expects guided_veto_code/src/pose_pipeline/sam3_guided.py; other CPU
runners expect code/src/pose_pipeline. Historical official SAM3/model paths remain
in the tracking runners and are hash-checked. Per-scene result.json records the
exact executed sources, inputs and environment. No environment install was done.

Recorded SSH44 Python:
/mnt/d/SGF-SGA-experiments/sam3_semantics_20260911_v1/env/bin/python
The Windows SSH endpoint is 100.64.57.44; execute through `wsl -- python3 -` using
Python subprocess input for a remote deployment script. This avoids shell quoting
ambiguity. Scripts/model/weights must not be silently changed for warm reuse.

Local analysis Python:
/Users/wu/Desktop/wu/SGF-SGA/comparisons/sgf_sga_restore_20260910_v1/analysis_env/bin/python

From this worktree, targeted validation was:

```sh
PYTHONPATH=src /Users/wu/Desktop/wu/SGF-SGA/comparisons/sgf_sga_restore_20260910_v1/analysis_env/bin/python -m pytest -q tests/test_sam3_guided.py tests/test_sam3_loss_unknown.py tests/test_sam3_tracking_probe.py tests/test_sam3_multiview.py tests/test_sam3_fusion.py tests/test_sam3_refine.py tests/test_sam3_sga.py tests/test_sam3_object_naming.py
```

82 existing/changed-module checks and 8 naming checks passed. Fixed legacy0030
metrics and additional class-agnostic/naming diagnostics are distinct protocols,
not official ScanNet AP. The secondary evaluator reads exported object names;
it reports any posthoc point-majority analysis under an explicit separate label.
GT enters evaluation only. No full-scene new semantic accuracy claim, no SGA/CLIP/
LLM inference this round, and no new scene-graph relation prediction.
