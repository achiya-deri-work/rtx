# Local predictor study: first coverage wave

2026-09-13, RTX 5070 Ti, Torch 2.13.0+cu132. Dataset:
`autotune_datasets/predictor_prospective_local_v2`.

All 32 cells completed their eight attempted configurations and passed all
three post-search winner checks. Seven task types had compatible model
contracts; standalone GLU was an incompatible-contract fallback control.
The geometry was M=384, K=640, N/H=1280 (MLP output width 640).

| Candidate | Control | Paired tasks | Geometric mean kernel speedup | Added tuning time |
|---|---|---:|---:|---:|
| Retrieval + local/random | Existing policy | 7 | 1.012x | +1.14 s |
| Retrieval + residual + local/random | Existing policy | 7 | 1.019x | +1.33 s |
| Retrieval + residual + MLP + local/random | Existing policy | 7 | 1.077x | +1.10 s |
| Retrieval + residual + MLP + local/random | Retrieval + local/random | 7 | 1.064x | -0.04 s |

These are measured kernel-winner latencies, not an application speedup. The
timing column is mean additional search wall time per task. Shared persistent
compiler caches and treatment order affect compilation time. There is only one
geometry and one search seed in this wave; no useful generalization confidence
interval can be claimed. No default predictor has been promoted.

The first four-cell development pilot is separately preserved at
`autotune_datasets/predictor_prospective_local_v1`; it is not merged into this
comparison. The frozen v2 plan has 192 cells, adding a larger geometry and
independent 16/32-trial searches. The detached shell did not survive execution
tool cleanup; the continuation was restarted in a persistent execution session
and verified with a live supervisor/worker and another completed cell. Its log
is `autotune_datasets/predictor_prospective_local_v2/continuation.log`.
Rebuild the current results with `python -m tools.analyze_predictor_study` as
documented in [the study guide](predictor_prospective_study.md).

The implementation regression suite passed 69 tests, covering policy budgets,
retrieval-only controls, resume, corruption rejection, paired analysis and docs.
