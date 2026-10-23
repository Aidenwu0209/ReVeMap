# Complete-owner terminal readout — developup

This branch preserves the tested research integration of `owner-lineage-v1`.
It is based on commit `51076f68bdf5d0768242f7cbb3159398257ecbf4`, matching the
candidate's original control. It is not a port to the later `main` revision
`88a15efea9e0464a7bca99230d734cf585c41b65` and does not contain main's subsequent
changes. The feature remains **off by default**.

## Behavior

The existing multi-view fusion call records geometric groups and their original
safe owners. After guided recovery, the terminal readout restores unowned,
semantically unknown points only when the complete owner lineage leads to one
surviving owner. It preserves coordinates, semantic labels, confidence values,
existing positive assignments, and the set of instance IDs. Recovered points
do not become new association anchors or semantic evidence.

Exports bind the base labels, captured groups, point lineage, audit and source
provenance in `GEOMETRY_CONTRACT.json`. Artifact loading checks these bindings
and recomputes the readout. Scene graphs use the complete support for geometry;
category and naming evidence continue to use pre-readout support.

## Usage

Install the package and configure model runtimes as described in the repository
README. Append the independent option to a normal run:

```bash
revemap run-sam3 \
  --manifest /absolute/path/to/manifest.json \
  --runtime configs/semantic_runtime.local.json \
  --output outputs/owner-lineage-run \
  --schedule serial --vlm none --stride 5 \
  --geometry-readout owner-lineage-v1
```

Use `--geometry-readout off`, or omit the option, for the baseline behavior.
Each output directory must be new. Use the same inputs and other options for
comparisons. The example above runs the normal pipeline and requires the
configured mapping and SAM3 runtimes; it is not a cache-only replay command.

The GUI bridge rejects `owner-lineage-v1` combined with optional refinement
before execution. This combination has not been validated. There is no new GUI
toggle. Checkpoint reuse retains the existing strict source-context checks;
changing source versions can invalidate previous checkpoints.

## Source provenance

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records the 15 imported source/test
files and SHA256 digests. They are byte-identical to the sealed
`production_integration_candidate_v1/candidate_source_uncapped` snapshot used
for the historical native replays. The core `geometry_readout.py` SHA256 is
`db5224d0023e2d0552d73536efd16aa94252957996dd29ee05061f2fd97dbf7b`.

Compared with the original candidate, the uncapped snapshot removes one
hard-coded two-thread environment assignment at the fusion entry point.
The historical control received the same resource change. Other entry points
and explicit algorithm worker settings retain their original behavior; this
does not promise full CPU utilization throughout the pipeline.

No thresholds, experimental measurements, or readout algorithm were changed
while preparing this branch. Large datasets, model weights, local runtime
configuration, cache payloads, and unrelated N4 experiments are not included.

## Validation and interpretation

[FOUR_NATIVE_DEVELOPMENT_REPORT.md](FOUR_NATIVE_DEVELOPMENT_REPORT.md) preserves
the original four-scene development report. Its references to adjacent replay
JSON files refer to the local research archive; those payloads are not bundled
here. Its statements that the main repository was not merged describe the
historical experiment and remain true for `main`; this branch publishes the
isolated candidate.

Those cached native replays passed compatibility checks. At the predefined
IoU >= 0.5 threshold, matching counts did not increase; at IoU >= 0.25, one
scene gained one class-agnostic match. Instance purity decreased slightly in
all four scenes. These are development diagnostics, not official benchmark AP
or evidence of independent generalization. The feature remains opt-in.

Run the repository's existing CPU tests after installing development extras:

```bash
python -m pytest -q
```

`tests/test_geometry_readout_runtime.py` covers owner-lineage behavior,
immutable fields, provenance tampering, export/graph/query integration,
argument forwarding, and the explicit refinement guard. Its small native
label fixture is a no-op interface check, not a quality benchmark. CPU tests
do not rerun mapping, SAM3, Qwen, or native-scene quality evaluation.
