# First-hit execution-region coverage: HOME qualification

The flat BalancedMoE router had a finite-budget search failure: the opening
changed backward reductions while leaving forward column ownership fixed.
Large token pools could spend eight trials in the wrong execution region.
Predicting latency more accurately inside that region cannot fix coverage.

The new opening derives legal forward/backward ownership contrasts from token
count, expert count and SM count. It tests joint and single-direction changes
and alternative hard-selection backends. Static legality, the independent
evaluator and normal UCB/local search still decide. Unknown hardware, short
reductions and the separate packed-router opening remain protected. There are
no named-SKU winner tables, manual register caps or changes to kernel arithmetic.
All new selection work runs on the cold host autotuning path. Custom-op hot
launchers, compiler-visible tensor allocations and user-owned graph boundaries
are unchanged.

## Implementation

Optional adapter `opening_fn` and `region_fn` hooks make this reusable outside
MoE. Protected representatives survive pool truncation; the eight-trial opening
still leaves actual exploration/adaptation slots. Target-normalized donor IDs
and region-diverse retrieval avoid stale-schema or redundant proposals.
Offline SKU aliases are excluded from portable features; derived reduction
geometry is available before compilation.

Bandit rewards/restored incumbent statistics exclude timing-ineligible
successes. Failed shared-MLP deployment qualification retains detailed
diagnostics and still refuses an unverified incumbent. Evidence identities
change with policy; increasing only the trial cap keeps a compatible
progressive journal.

Named effort presets select progressive/mixed search. Direct construction of
`AutotunePolicy()` retains legacy search for compatibility; explicit legacy
overrides remain supported. No learned artifact is installed automatically.
With a compatible explicit artifact, `prior_search="retrieval"` avoids model
deserialization, `"ranker"` excludes donor retrieval, and `"hybrid"` permits
both. `region_coverage=False` is the ordinary-opening control. These controls
do not override an adapter's explicit reservations or legality.

## Prospective HOME study

Both GPUs are RTX 5090s: this is new-shape evidence, not cross-SKU validation.
The research prior was frozen before these geometries were chosen. Arms have
fresh winner caches, shared compilation caches, reversed treatment order
across GPUs and exclusive per-GPU locks. Outer compilation is user-owned
`fullgraph=True, dynamic=False`, aggressive Inductor tuning, `freezing=False`
and default mode. Timings use ordinary CUDA events on warmed graph replays.

Budgets are **per component**: router, experts and shared MLP each receive
eight actual search attempts; affinity has two alternatives. The 300-second
soft admission allowance permits these counts. Public preset deadlines do
not guarantee that many trials complete. Higher budgets resume the same
journals. Cold cost below means complete caller autotuning invocations, not
total reference/model setup or outer compilation.

### Eight-trial whole forward plus backward

T/D/E/K/H/S means total tokens, model width, installed experts, routes per
token, per-expert hidden width and shared hidden width.

| Flat shape | Old opening | Geometry opening | Whole speedup | Retrieval only | Geometry + retrieval + ranker |
|---|---:|---:|---:|---:|---:|
| 18432/384/72/4/320/256 | 5.277 ms | 4.065 ms | 1.298x | 4.067 ms | 4.062 ms |
| 26624/512/104/6/256/512 | 9.409 ms | 7.756 ms | 1.213x | 7.438 ms | 7.415 ms |

These are **autotuner improvements, not BF16 speedups**. The BF16 control uses
the selected arm's router, so its latency changes too. The learned ranker has
not shown a material gain over retrieval on these cases and remains opt-in.

### Actual cumulative effort

| Flat shape | Old 8 / 16 / 32 trials | Geometry 8 / 16 / 32 trials |
|---|---:|---:|
| 18432/384/72/4/320/256 | 5.277 / 5.215 / 4.236 ms | 4.065 / 4.076 / 3.980 ms |
| 26624/512/104/6/256/512 | 9.409 / 9.048 / 8.036 ms | 7.756 / 7.757 / 7.754 ms |

The new eight-trial choices beat the old 32-trial choices by 4.2% and 3.6%.
More effort is not a promise of a monotonic measured whole-layer time: component
objectives, near ties, compiler choices and visit noise still matter. Most of
the benefit is now obtained early rather than requiring the larger budget.
Routing-only best-observed eight-trial regret against each arm's own deeper
journal drops from 40.5% to 4.2% on D384 and from 32.7% to below 1% on D512.
These are finite observed oracles, not global optima.

The first D384 ordinary-32 shared-MLP deployment qualification refused its
incumbent. One retry on the unchanged source and same journals succeeded,
with zero new router/expert/shared search trials. The initial failure remains
in the audit. Its original diagnostics do not establish the exact cause;
the improved error reporting is a follow-on change, not a retroactive diagnosis.

Summed caller autotuning invocation time at eight trials was 92.67 versus
89.56 seconds on D384 and 104.45 versus 101.25 seconds on D512 (old/geometry).
This excludes outer compilation/reference setup and is not a portable startup
guarantee: compilation caches and order affect cold measurements. Hybrid
geometry+prior runs cost 96.88 and 121.85 seconds. In particular, the small
extra steady-state gain over retrieval alone does not justify automatically
loading a learned ranker here.

Every completed whole-layer result checks independent quantized FP64 expert
references, finite outputs/all optimizer gradients, replay gradients and live
allocation growth. Legal FP32 reduction schedules can change hard assignments
near ties; identity hashes are retained. This is not bitwise routing-policy
identity or a relaxation of the separate strict fixed-route/full-router
gradient gate. There is no training-convergence claim.

### Alternating remeasurement and protected controls

Four alternating visits remeasure the complete selected public MoE. Input,
parameters, upstream gradient and each arm's saved route hashes must match;
all-gradient capture/replay and live-memory checks remain enabled.

| Eight-trial pair | Old | Geometry | Ratio of median times | Visit speedup range |
|---|---:|---:|---:|---:|
| D384 / E72 | 5.251 ms | 4.071 ms | 1.2898x | 1.2888–1.2898x |
| D512 / E104 | 9.380 ms | 7.750 ms | 1.2103x | 1.2100–1.2111x |

The visit range is not a population confidence interval. Both large pairs
have different cross-arm hard assignments; each arm reproduces its own saved
assignments and passes its own independent quantized-expert checks.

The protected packed control (T15360/D384/E40/K4) measured 2.3908 versus
2.3984 ms; the short flat control (T1024/D384/E64/K4) measured 0.8396 versus
0.8427 ms. Changes are below 0.4%, unresolved as gains or regressions by these
separate-process visits. Both have identical cross-arm hard assignments.

All 22 whole-layer result slots completed after the single preserved retry.
The worst independent quantized-FP64 expert error across output/dX/dW/route
gradients was 0.0107% NRMS. This is error against the quantized contract, not
against BF16 training. Five focused CUDA integration tests passed: compiled
8-to-16 refinement, warmed capture/replay, GLU durable reuse, and named quick
mixed-policy MXFP8/NVFP4 first-hit/cache/replay. CPU suites also passed
(`cpu-combined.log`: 128 tests/20 subtests; final focused suite: 69 tests/two
subtests; these overlap and must not be added). `git diff --check` passed.

Matched eight-trial controls retain identical hard assignments:

| Control | Old | Geometry-enabled policy | Change in whole latency |
|---|---:|---:|---:|
| Eight independent packed 1920-token samples, D384/E40/K4 | 2.3908 ms | 2.3984 ms | +0.32% |
| Eight 128-token samples, flat D384/E64/K4 | 0.8396 ms | 0.8427 ms | +0.37% |

There is no resolved regression or gain in these controls. This is two
protected geometries, not universal packed/small-shape qualification.

All 22 planned whole-layer result slots completed, retaining the initial
ordinary-32 failed attempt and its successful single retry. Across these
results, worst normalized RMS against the independent quantized FP64 expert
oracle is 0.0107% (route-weight gradient); this is **not error against BF16**.
All live-allocation growth checks are zero. Focused CPU suites passed, as did
CUDA compiled 8→16 cache/refinement, MXFP8 capture/replay and GLU first-hit
reuse/refinement tests. Named mixed presets additionally have focused compiled
first-hit and warmed replay checks for MXFP8 and NVFP4, not an exhaustive
NVFP4 performance matrix.

The first alternating helper attempt incorrectly requested an unused
attribution reconstruction, whose separate hard-scheduler path did not
reproduce the full router's assignments. Its assertion/log is preserved.
The helper now requests whole-only construction; default attribution mode
retains its strict check. Actual selected-route identity checks are retained
in the successful remeasurements. No production or numerical gate was removed.

## Usage and reproduction

```python
policy = rtx.AutotunePolicy.preset("quick", cache_behavior="refine")
# Pass to a supported module's balanced autotuning path. Compile on the
# application side; do not initiate cold tuning inside manual capture.
```

The supervised `benchmarks.run_moe_region_qualification` driver has immutable
source, shard, prior and output arguments. `benchmarks.qualify_autotune_regions`
provides the causal panel, progressive routing search and alternating
whole-layer remeasurement. `benchmarks.analyze_autotune_regions` verifies
source/input/parameter identities and emits JSON, CSV and Parquet while
retaining failed jobs. Each has `--help`.

Evidence and frozen sources are archived under
[`autotune_reports/autotune_region_opening_20261120/`](../autotune_reports/autotune_region_opening_20261120/ARCHIVE.md),
with [whole-layer CSV](../autotune_reports/autotune_region_opening_20261120/analysis/whole_layer.csv)
and [journal-prefix CSV](../autotune_reports/autotune_region_opening_20261120/analysis/journal_prefixes.csv).
Previous source data and
the frozen prior are not rewritten. See the
[Astra review](mxfp8_moe_autotuning_astra_review_20261120.md) and
[prior discovery study](mxfp8_moe_heuristics_20261120.md).
