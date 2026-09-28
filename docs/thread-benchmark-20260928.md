# Graph native thread default: 2

The graph stage now applies its own BLAS/OpenMP limit after importing Open3D,
instead of relying solely on environment variables passed to a Python launcher.
Some launchers clear those variables. `graph_threads` in the semantic runtime,
`revemap run-rgbd --graph-threads`, and `run_measured_graph(..., threads=...)`
default to 2 independently of other stages. Explicit overrides remain available.
The graph receipt records the requested limit and observed native pools. Pools
that only support a single thread may stay at 1; the limit is not a GPU setting.
The context restores prior native pool settings even if graph execution fails.
Existing CPU environments need `threadpoolctl>=3.5,<4` (now a package dependency).

## Evidence and scope

2026-09-28, ssh33 (i9-13900HX, 32 logical CPUs): replay the same frozen DROID
output from 256 consecutive frames of `scan_20260924_175726_41844f`. Each setting
has three measurements, in interleaved order; baseline warmup is excluded.
The measured stage includes graph construction and registration, not only the
final global solver. No models, input resolution or algorithm settings changed.

| Setting | Three stage times (s) | Median (s) |
|---|---|---:|
| Original runtime, actual 2 | 12.658 / 12.640 / 12.688 | 12.658 |
| Explicit 2 | 12.651 / 12.772 / 12.754 | 12.754 |
| 8 | 14.405 / 14.394 / 14.367 | 14.394 |
| 16 | 14.290 / 14.381 / 14.372 | 14.372 |
| 32 | 14.312 / 14.416 / 14.226 | 14.312 |
| 56 | 14.334 / 14.261 / 14.185 | 14.261 |
| 64 | 14.303 / 14.325 / 14.313 | 14.313 |

All frozen-input graph pose outputs were bitwise equal. At requested 56/64,
OpenMP reported 56/64 while NumPy/SciPy BLAS reported 32. The final solver alone
took about 0.012 s, less than 0.1% of the baseline stage. These results support
the default on this input, not a universal optimum or full-pipeline speedup.
No GT accuracy claim follows from output equality. Desktop/background work
remained; CPU exclusivity was not established. GPU-stage thread sweeps on ssh33
and RTX 3080 showed no compelling benefit from increasing existing defaults.
The new runtime enforcement is separate from the original environment-variable
sweep; its behavior is checked with loaded-pool and restoration regressions.

An additional isolated ssh33 replay of the new code cleared all four thread
environment variables before importing the numerical libraries. BLAS/OpenMP
initially reported 32; inside the graph they reported 2 (the single-threaded
OpenCV BLAS remained 1). The frozen graph poses were bitwise equal to the old
baseline, and previous pool settings were restored after both success and an
injected failure. This was a correctness check, not an additional speed sweep.
Local regression validation: 436 tests passed, 6 skipped, 24 subtests passed;
three skips cover Apple Accelerate, which threadpoolctl does not manage. Native
Linux pools were exercised by the ssh33 replay. Five JavaScript tests passed.
