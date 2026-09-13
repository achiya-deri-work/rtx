# CPU model tournament for portable autotuning

This is an optional development tool, not a new runtime dependency or an
automatically deployed prior. It runs entirely on CPU against existing
observation journals and ZIP bundles. No GPU collection is started.

See [the completed local results](model_tournament_results_20260913.md).

## Run and resume

Use an isolated environment with `requirements-research.txt`. The local
`.venv-research` already contains these dependencies. From the project root:

```bash
.venv-research/bin/python -m tools.research.tournament \
  --sources autotune_reports/effort_ml_v2/audit.json \
  --output autotune_reports/model_tournament_v1 \
  --threads 8 --views interactions --protocols chronological --auxiliary
```

The source manifest contains `train_paths` and `test_paths`: lists of ZIPs or
directories containing `observations.jsonl`. Paths can point to other studies.
Use `--protocols all` for chronological, unseen-shape and three held-out-device
evaluations. `--models` and `--views` select a subset of the declared matrix.
The same command resumes completed fits. Additional views/protocols can be
added to the same directory when the source/code/environment contract matches.
Changing code or source contents requires a new output directory.

Each completed experiment publishes a separate JSON/Parquet shard atomically;
an interrupted fit does not damage other results. Source contents, code and
package versions are fingerprinted in `manifest.json`. Errors are recorded
separately, never scored as successful model evaluations. No pickle/joblib
model is loaded into the runtime. To regenerate aggregate reports:

```bash
.venv-research/bin/python -m tools.research.tournament \
  --sources autotune_reports/effort_ml_v2/audit.json \
  --output autotune_reports/model_tournament_v1 --summarize
```

## Model registry

`tools/research/models.py` exposes a small registry and fit/predict interface:

| Candidate | Role |
|---|---|
| Retrieval | Exact-config nearest historical workload, bounded family fallback |
| Ridge | Standardized linear baseline |
| Additive histogram boosting | Univariate decision-stump baseline; not a spline GAM |
| ExtraTrees | Family-specific randomized tree ensembles |
| Global/family histogram boosting | Test sharing versus independent heads |
| XGBoost | Family-specific histogram regression |
| CatBoost | Family-specific boosting on the same numeric/one-hot representation |
| Pairwise XGBoost | Within-context ordering objective |
| Small MLP | Standardized 32-unit or 64/32-unit family heads |
| Roofline residual | Family boosting around an estimated memory-time baseline |

Regression labels are log latency centered by training context. Pairwise
labels are within-context ranks; predicted direction is normalized to
lower-is-better. The analytical baseline is **memory-only**, not a complete
compute roofline. CatBoost is not yet given native categorical columns, so
this is a controlled common-representation comparison, not maximal tuning of
each vendor library. MLP convergence warnings are retained in result JSON.

Every learned candidate has two prespecified variants. Select the variant
with lowest mean budget-8 regret on a deterministic **shape-grouped inner
split of older training data**, refit on all permitted older data, and then
evaluate on newer data. HGB and MLP do not use internal row-random early-stop
validation. This is a modest triage grid, not exhaustive hyperparameter search.

## Feature ablations

1. `shape`: configuration, workload, semantic precision/state, cache regime.
2. `hardware`: add numeric architecture/SKU properties (bus width, SM count,
   bandwidth, L2, registers, SMEM, supported execution model, etc.).
3. `interactions`: add existing static kernel-resource estimates, CTA waves,
   arithmetic intensity, working-set/L2 ratios, and estimated memory times.
4. `operating`: add only top-level **pre-evaluation** power limit/draw, SM and
   memory clocks, temperature, and explicit missingness flags.

Device names, UUIDs, software/source identities, measured registers/spills,
post-run telemetry and measured latency are excluded. Configured register
budgets and statically estimated occupancy are allowed, not actual compiled
register counts. Derived features are estimates, not profiler evidence.

Pre-evaluation power draw may reflect the preceding workload and collection
order. It is only a candidate feature for a runtime that takes an equivalent
pre-evaluation snapshot; its inclusion does not establish a causal power
effect. Missing values are not fabricated. `audit.json` inventories ranges,
medians and coverage of recorded hardware/operating fields per device.

## Evaluation and limits

The chronological protocol uses older runs for fitting and newer runs for
evaluation. Device holdouts remove the target device from older fitting data.
Unseen-shape holdouts remove those workload geometries across devices, cache
regimes and numerical policies. Shape grouping hashes numeric workload fields,
not configuration, family or device identity; differing workload schemas may
still describe related geometries and need domain-specific grouping to merge.

Report regret at 8/16/32, median/p90 regret, within-2% frequency, and paired
workload-cluster bootstrap intervals versus identical seeded random subsets.
Prediction overhead includes feature extraction/vectorization. Per-fit reports
include timing for up to ten candidate pools of at most 128 observed entries;
these are not all necessarily full 128-entry pools.
The initial local tournament, feature ablation, and transfer jobs run
concurrently. Their CPU timing fields are diagnostic wall times under varying
contention, not isolated inference-speed rankings. Finalists need isolated
timing before deployment decisions use small overhead differences.

Latency evaluation still uses observed successful accepted candidate pools.
It does **not** simulate adaptive UCB, claim the global optimum, or establish
failure-inclusive time-to-winner. Separate fixed auxiliary heads report:

- failure probability: Brier score, log loss, average precision, prevalence;
- observed evaluator elapsed time: log MAE and multiplicative error versus
  a training-median baseline.

Static rejects and established quarantines are removed before auxiliary
training. Failures are not assigned invented latency values. Elapsed-time
metadata includes the original evaluator's work and cache state: it is not
necessarily cold compilation time. The auxiliary heads are exploratory and
are not calibrated for production pruning. Raw failure counts are essential
for interpretation because failures may be very rare after static filtering.

Only three SKUs are present. Many hardware variables covary; thousands of
kernel measurements do not create thousands of independent hardware samples.
The newer dataset was inspected previously: all results are development
evidence, not untouched release qualification. Final model/policy selection
still needs a new prospective, failure-inclusive budget study. Runtime UCB
and numerical legality checks remain unchanged by this tool.
