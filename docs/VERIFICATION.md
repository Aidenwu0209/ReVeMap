# Repository extraction verification

Checked on 2026-09-21 against `SGF-SGAligner/developnew` commit
`18ddeca303bf9937e855d3ac3113a90dd852a462`.

## CPU regression checks

In an isolated Python 3.12.13 environment on macOS, with system site packages
disabled:

```bash
python -m pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -rs
python -m pip check
```

Result: **135 passed, 1 skipped**, with no broken package requirements. The
skipped test is `RGBDRefusionContractTests.test_default_frame_list_rejects_missing_pose`
because Open3D was not installed in this CPU test environment.

The tests cover projection and occlusion rules, distinct-view support, instance
association, unknown-point recovery, naming evidence, RGB-D contracts, worker
failure handling, GUI data handling and independent package imports. They use
small fixtures and mocked model/API calls where appropriate.

## Source and installation checks

- Seven relocated functions were compared structurally with the pinned source;
  their computations were unchanged. The only removed statement was the local
  import replaced by the new standalone module. See
  [extraction_equivalence.json](../verification/extraction_equivalence.json).
- The source checkout remained clean and its branch/remote were unchanged.
- Both an sdist and a wheel were built. The wheel was installed into a second,
  clean environment and exercised from outside either checkout.
- `--help`, `--version`, `list-vlm-models`, `run-sam3 --help`,
  `enhance-semantic --help`, `refine-semantic --help`, `run-rgbd --help` and
  `gui --help` all exited successfully from the installed wheel.
- The installed wheel contained the model registry, taxonomy and ReVeMap GUI
  HTML. Core semantic imports did not load PyTorch or legacy SGF neural code.
- Per-file source and extracted-content hashes are recorded in
  [SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json).

GitHub Actions runs the CPU suite, CLI checks and package build on Python 3.11.
Its current result is available in the repository's Actions tab; the local
results above do not imply that a hosted run has completed.

## Remaining validation

This extraction did not rerun CUDA model inference, TSDF/ICP execution with
Open3D, a complete RGB-D sequence or a physical camera session. Those require
the external environments, weights and input data described in
[SETUP.md](SETUP.md). CPU tests and successful packaging do not establish
map accuracy, model quality, runtime performance or scientific novelty.

The files under `docs/experiments/` are inherited development records. Their
numbers are not new measurements of the independent ReVeMap repository.
