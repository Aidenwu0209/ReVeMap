# developup packaging validation

Validated locally on 2026-10-23 (Asia/Shanghai), before the branch commit.

- Base: `51076f68bdf5d0768242f7cbb3159398257ecbf4`.
- All 15 imported source/test files match the sealed uncapped candidate hashes
  recorded in `SOURCE_MANIFEST.json`.
- Full CPU suite: `python -m pytest -q` — **321 passed, 1 skipped in 18.63s**.
- Public CLI: `PYTHONPATH=src python -m revemap run-sam3 --help` exits 0 and
  exposes `--geometry-readout {off,owner-lineage-v1}`.
- `git diff --check` passes.

The tests used the existing local ReVeMap Python environment from the new
branch checkout. They did not execute model inference, use a GPU, or rerun
native-scene quality measurements. Historical quality findings remain in
`FOUR_NATIVE_DEVELOPMENT_REPORT.md` unchanged.
