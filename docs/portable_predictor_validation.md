# Portable predictor validation — 2026-09-13

## Artifact and reproducibility

`tools/export_tournament_predictors.py` refits the two selected experimental
architectures on older accepted observations only. The newer study is used for
serialization verification, never for fitting. An observation-ID overlap check
guards this boundary. Model selection has already inspected these studies, so
this is not a new unbiased prospective evaluation.

```bash
.venv-research/bin/python -m tools.export_tournament_predictors \
  --sources autotune_reports/effort_ml_v2/audit.json \
  --output autotune_priors/tournament_predictors_v1.json.gz \
  --threads 8
```

The source manifest and datasets must be present locally. Research dependencies
are isolated in `requirements-research.txt`. Runtime inference needs only NumPy.
The data-only gzip JSON artifact is approximately 15 MiB; its verification
sidecar records provenance, versions and digest. It remains research-only.

Training used 53,786 aggregated rows. Verification used 35,730 newer rows:

| Head | Maximum absolute score difference from sklearn | Identical top-eight sets |
|---|---:|---:|
| Residual trees | 0 | 482 / 482 contexts |
| Small MLP | 0.000002623 | 482 / 482 contexts |

These are log-latency scores, not microseconds. MLP fits reached the 40-iteration
cap; this does not claim optimizer convergence. The portable MLP preserves
float32 scaling intermediates; tree comparisons use float64 thresholds.
The exporter isolates sklearn's private tree representation from runtime code.

## Runtime verification

Unit tests cover malformed schemas/scalers/trees, research opt-in, feature-view
equivalence, sklearn export equivalence, semantic compatibility, two independent
UCB arms, retrieval, eight-trial accounting and resumed searches.

The local RTX 5070 Ti passed cold `torch.compile(fullgraph=True)` training for
MXFP8Linear and default JIT-region NVFP4Linear at M=128, K=256, N=128. Both heads
and retrieval actually supplied candidates. Forward and both gradients were
compared against independent FP32 references; repeated calls appended no new
tuning observations. Operator-contract tests also passed. This is an integration
smoke test, not a replacement for the broad numerical qualification matrix.

```bash
.venv/bin/python -m unittest tests.test_portable_predictors tests.test_autotune_effort
.venv-research/bin/python -m unittest tests.test_predictor_export
.venv/bin/python -m unittest tests.test_predictor_cuda tests.test_operator_contracts
```

The CUDA predictor test requires the local artifact and skips without it or CUDA.
Screening used only three samples, two calls per sample and one warmup. These
timings are diagnostic: in one run NVFP4's residual proposal measured 7.60 us
against a 10.38 us initial candidate, while its MLP proposal measured 162.35 us.
The backward predictor proposals also lost to the initial/local candidates.
Thus model quality is heterogeneous; no universal speedup or default promotion
is justified. Measured selection retained better candidates.

## Before default deployment

Run a fixed-budget prospective comparison at 8/16/32 trials against the existing
policy and retrieval-only controls, on held-out shapes and all three GPUs. Report
regret, failures, proposal/compilation/measurement time, and tail regressions;
do not choose policy on the same observations used to train it. Retain random
and local-search arms. Current validation establishes export fidelity and real
first-hit integration, not that either head alone is the best search policy.
