# Independent repository extraction

Source: `Aidenwu0209/SGF-SGAligner`, branch `developnew`, commit
`18ddeca303bf9937e855d3ac3113a90dd852a462`.

The source checkout and branch are not edited. ReVeMap starts a separate Git
history and retains the source identity in `SOURCE_PROVENANCE.json`.

## Retained

- Raw RGB-D geometry/estimated-pose pipeline and TSDF fusion.
- SAM3 segmentation, geometric association, multiview fusion and recovery.
- Semantic runtime, T1/P2 evidence replay, model interfaces and refinement.
- Capture/replay GUI and original scientific/data-contract regression tests.
- Relevant historical experiment reports, figures and evidence metadata.

## Decoupling changes

The raw semantic fusion path previously imported geometry helpers from
`sam3_sga.py`, and one of those helpers imported `semantic_mapping.py`. Those
files also contained unrelated SGF/SGAligner neural inference and graph adapters.
The five needed association/consensus functions and `measured_association` are
extracted into `pose_pipeline.instance_association`. Their computations and
thresholds are preserved. The evaluation script also receives its original
standalone JSON writer instead of importing the legacy neural pipeline.

The `revemap` CLI, portable launcher, install metadata and project documentation
are new. The model registry and taxonomy are included as package resources so an
installed wheel can find them outside the source checkout. Root `configs/` copies
remain readable examples and are checked against the packaged copies.

`pose_pipeline` and `reconstruction` remain internal package names so subprocesses,
existing imports and data contracts keep working. Old SGF neural code, learned
SGAligner/training code, GeoTransformer, original example datasets and unrelated
registration experiments are excluded. External model environments still need
to be installed separately.

The GUI title is changed to ReVeMap; its source-version field identifies the
original extraction commit. Historical experiment IDs, result numbers and
machine paths in archived evidence are not new deployment defaults. Links to
omitted historical files point to the exact original GitHub commit.

## Evidence

Original source hashes, destination hashes and function origins are recorded in
[SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json). Historical reports remain
in [experiments](experiments/README.md); the original branch README is retained
as [an archive](archive/developnew_README.md).

See [VERIFICATION.md](VERIFICATION.md) for migration checks. CPU checks cannot
establish GPU inference quality, camera behavior, full-scene performance or
scientific novelty of the extracted research direction.
