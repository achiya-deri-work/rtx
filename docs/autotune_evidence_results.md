# Audited-data and prospective-prior results — 2026-09-14

**Decision: do not promote the learned policy.** All five stages of the
evidence workflow were executed, including a bounded fresh-shape GPU pilot.
The historical model results improve substantially, but the live eight-trial
policy still has a reproduced regression. The existing default remains intact.

## Data and attribution

The nine source bundles contain 199,038 unique observations. The versioned view
retains 116,837 aggregated latency rows. It separately records 5,292 unstable
observations, 13,568 numerical quarantines, 213 correctness errors, 35 compile
errors, and 39 runtime errors. Group-level exclusions include 315 inconsistent
repeat groups, 190 context/config groups with failures, and 720 cross-split
overlaps. Those group counts are not additional raw observations.

Across the three overnight studies, 2,588 policy comparisons were matched;
1,946 had complete budgets, compatible contracts and stable within-cell
verification. Of 153 regressions over 2%:

- 104 lacked the baseline winner in the recorded pool and never measured it.
- 2 generated it but did not measure it.
- 10 measured it but selected something else.
- 37 lacked sufficient pool records.

These historical attribution counts use within-cell checks; repeated-context
instability remains a separate audit. Full proposal-rank events are now recorded
to remove one important ambiguity in future experiments.

## Offline study

All six train-only portfolios were built before model fitting. Twenty-four
fits compare two model families and two feature views over chronological,
whole-M/K, contraction-category, and three held-out-SKU splits. Results cover
8/16/32/64 observed candidates, with shape-cluster bootstrap intervals and a
separate portfolio-plus-random control.

The frozen selection rule chose **ExtraTrees with shape/config features**,
combined with the compact portfolio. Its eight-trial mean regrets are:

| Historical protocol | Portfolio + selected model | Random |
|---|---:|---:|
| Chronological development | 5.71% | 14.16% |
| Unseen whole M/K groups | 3.75% | 6.79% |
| Unseen contraction category | 1.96% | 6.11% |
| Held-out laptop 5070 | 1.73% | 6.57% |
| Held-out 5070 Ti | 2.21% | 7.67% |
| Held-out 5090 | 2.94% | 8.51% |

Regret is relative to the best **observed successful configuration**, not BF16
and not a theoretical optimum. Historical worst-context regrets remain severe
(over 300% in two protocols). Better averages are not sufficient for promotion.
Histogram models benefited more consistently from hardware-interaction features;
the selected model shows that such features are not universally beneficial in
this small, confounded three-SKU dataset.

The exported research artifact is 4.4 MB. Its NumPy implementation matches
sklearn predictions over 2,816 checks with maximum absolute difference
`6.66e-16`. No sklearn/pickle dependency is needed at runtime. Old MLP/backward
revision contracts are not silently upgraded.

## Fresh local GPU pilot

RTX 5070 Ti, Torch 2.13.0+cu132, CuTe 4.7.0, driver 595.84. Two previously
unattempted whole-M/K groups, four treatments each, eight trials per cell.
All 64 candidate evaluations succeeded. The independent initial forward
checks against FP32 GEMM gave 3.735–3.745% relative L2 under the unchanged 7%
gate. This pilot does not qualify every frontend or backward path.

Six balanced forward/reverse confirmation rounds per context used prepared
adapters and ordinary CUDA-event measurements. No profiler-derived speedups
or hardware-counter conclusions are reported.

| Shape M×N×K | Existing | Guarded mixed | Portfolio only | Portfolio + model |
|---|---:|---:|---:|---:|
| 384×768×1152 | 10.213 us | 10.218 us | 10.207 us | 11.869 us |
| 1792×512×1408 | 16.990 us | 16.996 us | 17.014 us | 17.007 us |

The learned policy is **16.21% slower** on the first shape; paired ratios range
from 1.1608 to 1.1637. The other differences are approximately ties at this
measurement scale. Existing tuning took 2.44–2.51 seconds per context versus
3.22–3.68 seconds for the learned policy, excluding setup but including its
search. Total worker times were 4.27–4.37 seconds versus 5.50–5.98 seconds;
process import/startup is not included. Shared persistent compiler caches were
allowed, so this is not a cold-machine startup benchmark.

### Why this regression matters

The baseline winner was now **present in the learned pool**, but ranked 86th
of 128 and was never measured. This is a different failure from an absent pool
candidate. Near-tied initial/retrieved measurements also changed the anchor
explored by the reserved local-search pulls. Reserving an arm is therefore not
equivalent to preserving a baseline trajectory.

Next work should test an explicit initial-anchor neighborhood and tie-aware
incumbent handling, then evaluate ranking against that stronger safeguard.
Use a new evaluation set after policy changes; do not tune on this pilot and
call its reuse prospective evidence. The intermittent MLP numerical case and
changed backward revision still need separate revalidation.

## Artifacts and reproduction

- `autotune_reports/evidence_dataset_v1/`: audit residuals, Parquet shards,
  manifest, exclusions, and software inventory.
- `autotune_reports/selection_diagnostics_v1/`: attribution rows and summary.
- `autotune_reports/portfolio_study_v2/`: six portfolios, 24 fitted development
  models, detailed metrics, bootstrap summaries, CSV and Parquet exports.
- `autotune_reports/portfolio_prospective_v1/`: frozen artifact and plan, eight
  worker journals, per-measurement confirmation residuals, and CSV/Parquet/JSON
  pilot summaries.

The larger 512-cell prospective plan is prepared but **not launched**. No new
runtime winners or default learned policy were installed. No other machine is
needed to inspect these results. See [the workflow](autotune_evidence_pipeline.md)
for exact commands and limitations.
