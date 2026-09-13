# Autotuning effort: empirical ML development study

## Outcome

Separate boosted-tree models per kernel family are substantially more
promising than a single global model. Hardware descriptors alone do not
reliably fix cross-device transfer. Retain exploration and retrieval;
do not turn predicted slowness into a hard legality rule.

Mean selection regret at eight candidates (lower is better):

| Evaluation | Global shape/config HGB | Global SKU HGB | Family SKU HGB | Random |
|---|---:|---:|---:|---:|
| Older → newer, all devices | 14.65% | 16.17% | **4.76%** | 14.31% |
| Laptop held out | 10.97% | 14.16% | **3.63%** | 8.36% |
| 5070 Ti held out | 15.83% | 2.40% | **2.23%** | 16.06% |
| 5090 held out | 21.36% | 24.64% | **12.40%** | 17.78% |

For chronological family HGB, 77.18% of contexts are within 2% of the
observed oracle; p90 regret is 5.00%. Its paired improvement over random
has a workload-cluster bootstrap 95% interval of 3.53–14.41 percentage
points. The held-out laptop interval narrowly includes zero, and the
5090 interval is wide and includes zero. Only the 5070 Ti transfer result
is clearly separated from random by this interval. This is not proof of
reliable transfer to arbitrary GPUs.

## Data and methodology

The older combined-collection bundles from all three GPUs provide 62,446
raw observations; the newer prior-study bundles provide 79,664. After
status, finite timing, explicit stationarity, numerical quarantine, and
duplicate-context/config aggregation, the study has 53,786 training rows
and 35,730 evaluation rows. Raw inputs remain untouched.

Excluded explicit unstable observations: 1,090 older / 1,330 newer.
Numerically quarantined observations: 5,261 older / 8,264 newer. MLP
quarantine conservatively checks stored component configs even if a
schedule might not execute that component; refining that filter could
recover additional valid data.

Fit log-latency centered by each **training** context's median. Vectorizers
are fit only on training data. Compare shape/config HGB, SKU/derived-feature
HGB, ExtraTrees, and independent family HGB heads. HGB uses 100 iterations,
depth 6, minimum leaf 20, L2=1; ExtraTrees uses 48 trees, depth 18, minimum
leaf 4, feature fraction 0.2. Seed is 1729. No GPU training is needed.

Protocols are chronological older→newer and three device-disjoint variants
which remove the target GPU entirely from older training data. Score top
8/16/32 predicted configurations against 32 seeded random subsets of the
same observed pool. Chronological budget-8 coverage is 482 contexts across
161 workload clusters. Export family/device breakdowns and mean, median,
p90, within-2%, and 2,000-resample paired workload-cluster bootstrap CIs.

## Important limits

- This is **development evidence**, not an untouched final qualification
  set: the prospective results had already been inspected, and family
  heads were added after an exploratory global-model run.
- Pools contain observed successful, accepted configurations. Regret is
  against the best configuration in that pool, not the full search space.
  These are candidate-ranking budgets, not failure-inclusive online trial
  budgets or simulated UCB trajectories.
- No compiler-time saving, end-to-end first-hit speedup, or causal bus-width
  effect is established here. Three SKUs cannot identify every correlated
  hardware effect. Cache regimes are distinct clusters in the CI analysis.
- Failed attempts remain useful for a future separate legality/failure-cost
  model. They must not be assigned arbitrary latency regression targets.
- A single global SKU model has better near-winner frequency but worse
  mean regret than the shape-only model: catastrophic tails matter.
- No sklearn model from this study is promoted into production or silently
  substituted for the frozen prior. No exhaustive optimum is claimed.

## Reproduction and artifacts

Run `tools/research_effort.py --train <older bundles/directories...>
--test <newer bundles/directories...> --output <report-dir> --threads 4`,
then `tools/summarize_effort_research.py <report-dir>` in a CPU environment
with numpy, scikit-learn, pyarrow, scipy, and threadpoolctl. This run used
the isolated `.venv-research`; production dependencies were not changed.

Local completed artifacts are in `autotune_reports/effort_ml_v2/`:
`audit.json`, `environment.json`, `fits.json`, `metrics.parquet`, and
`summary.{json,csv,parquet}`. Audit records exact input paths; environment
records package versions. The partial earlier `effort_ml_v1` experiment is
retained separately and is not the source of the table above.

## Next decisions supported by the evidence

1. Prefer family-specific normalized ranking heads over one global model.
2. Keep an exploration allocation and a known-good fallback on unfamiliar
   hardware; use scores for ordering, not irreversible performance pruning.
3. Build a portable serialization/inference path with feature-contract and
   family/revision validation before shipping the sklearn research heads.
4. Evaluate that frozen candidate on new, shape-disjoint fixed-budget runs
   including failures, compiler/proposal time, and confirmation overhead.
5. Reuse existing failure and sampling records for failure-cost prediction
   and uncertainty-aware timing. Do not collect another broad dataset
   before measuring these narrower hypotheses.

Validation for the new effort API includes CPU budget/resume/quarantine
tests and local CUDA linear forward/backward, packed compiled inference,
and an actual 8→16 attempt extension without rerunning the original eight.
These are integration checks, not full release requalification.
