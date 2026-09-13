# Model tournament results — 2026-09-13

## Decision

Keep three research candidates: nearest-workload retrieval, a small
family-specific MLP, and a family-specific boosted residual around an
analytical memory-time estimate. Plain family histogram boosting remains a
useful baseline. **Do not install one universal learned replacement for
retrieval.** No production prior or runtime winner was changed by this study.

## Chronological comparison

Mean selection regret at eight candidates, using identical observed accepted
pools across 482 contexts / nine workload geometries:

| Approach | Mean regret | Within 2% of observed best |
|---|---:|---:|
| Memory estimate + boosted residual | **2.23%** | 81.12% |
| Small family MLP | **2.56%** | 70.12% |
| Family histogram boosting | 3.89% | 80.08% |
| Ridge | 5.06% | 65.15% |
| Retrieval | 5.14% | 55.19% |
| XGBoost regression | 7.96% | 78.84% |
| Global histogram boosting | 10.30% | 75.73% |
| Family ExtraTrees | 16.67% | 70.54% |
| Pairwise XGBoost ranking | 17.32% | **85.06%** |
| CatBoost regression | 19.01% | 76.97% |
| Additive boosting | 22.14% | 59.54% |
| Seeded random subsets | 14.31% | — |

Against retrieval, the chronological mean gain has a shape-cluster bootstrap
95% interval of **0.15–5.39 percentage points** for the residual model and
**1.06–4.37 points** for the MLP. These are exploratory intervals on a small
number of geometries, not multiple-comparison-adjusted qualification claims.

Pairwise ranking illustrates why average error or near-winner frequency alone
is insufficient: it has excellent typical discovery but extreme bad tails.
The worst recorded top-eight selections include MXFP8 backward contexts with
more than 800% regret. A measured incumbent and UCB fallback remain important;
pure top-eight ranking is not the actual runtime search policy.

## Hardware feature ablation

Family histogram boosting, same model/inner-selection protocol:

| Features | Mean regret at eight |
|---|---:|
| Shape/config/semantic state | 12.00% |
| + raw SKU/architecture properties | 14.47% |
| + static hardware–kernel interactions | **3.89%** |
| + pre-evaluation operating telemetry | 4.51% |

The useful representation is not merely bus width, SM count and power.
It includes CTA waves, resource fractions, working-set/L2 ratios, estimated
traffic and related interactions already supplied by the adapters. This
ablation supports the interaction representation, not a causal coefficient
for any individual hardware statistic.

The source metadata records bus widths 128/256/512 bits and SM counts
36/70/170 for laptop 5070 / 5070 Ti / 5090. Desktop configured power limits
are recorded as 350 W and 575 W; the laptop limit is missing. Do not infer
a laptop limit from its measured draw or fill it from an unrelated SKU.
Pre-evaluation clocks and power draw vary substantially. They can reflect the
preceding workload and are not the candidate's eventual steady-state clocks.

## Transfer checks

Mean regret at eight candidates:

| Evaluation | Retrieval | Family boosting | Small MLP | Memory residual |
|---|---:|---:|---:|---:|
| Unseen geometry subset | **6.08%** | 16.88% | 8.89% | 6.12% |
| Held-out laptop 5070 | **3.14%** | 3.48% | 7.81% | Not run |
| Held-out 5070 Ti | 7.76% | 4.53% | **3.63%** | Not run |
| Held-out 5090 | 8.53% | 13.36% | 4.09% | **3.43%** |

The residual model's paired gain over retrieval on 5090 is 5.10 points
(bootstrap interval 2.75–8.53). On unseen geometries its difference from
retrieval is effectively zero, with a wide interval. The MLP improves on
retrieval for the desktop holdouts but worsens on the laptop and unseen-shape
subset. This does not justify a hard-coded SKU router; it motivates a
retrieval-backed ensemble and uncertainty-aware proposals.

The newer input bundles contain twelve numeric workload geometries, only nine
of which support the eight-candidate comparison after filtering. The
unseen-geometry test covers **two** eligible geometries / 120 contexts. The
older chronological fit has 38 training geometries, split 29/9 for inner
selection. Context count is not independent geometry or hardware diversity.

## Model selection and auxiliary heads

Each learned candidate chooses between two prespecified variants using only
older-data, shape-grouped validation. That inner selection still favors
ordinary family histogram boosting over the MLP and residual model. Their
better newer-data results are not permission to silently promote them.
The MLP and residual transfer follow-ups were chosen after chronological
triage; they are exploratory rather than a preregistered full factorial study.

The MLP chose 64/32-unit hidden layers. With 2,118 input features, that is
approximately 138k parameters per active family head; total storage includes
all family heads. Forty-iteration convergence warnings were recorded, so this
is not a claim that neural training or architecture search is exhausted.
The residual baseline sums available `memory_roofline_ms` estimates; it is a
heuristic memory-time proxy, not proof of a physical latency lower bound.

After current static/quarantine filtering, the auxiliary failure study has
only **four training failures and zero evaluation failures**. Its tiny Brier
score is not evidence of useful failure detection. Keep static legality checks;
do not use this classifier to prune candidates.

The evaluator-time head is also not ready: log MAE **0.689**, worse than the
training-median baseline's **0.622**, with median multiplicative error about
**1.70×**. Elapsed-time labels reflect historical cache/evaluator state, not
isolated cold compilation cost. Better stage/cache-state provenance is needed
before using such predictions to allocate wall-clock budgets.

## Reproducibility and scope

142,110 raw observations were inspected across older/newer bundles; latency
models use 53,786 aggregated training rows and 35,730 evaluation rows.
An independent identity scan found zero overlapping observation IDs between
the two source sets. Input contents and research code/package versions are
fingerprinted. Raw journals are unchanged.

The completed local studies are `model_tournament_v1`, `model_ablation_v1`,
`model_transfer_v1`, `model_neural_transfer_v1`, `model_neural_devices_v1`, and
`model_residual_transfer_v1` under `autotune_reports/`. Merged CSV/Parquet,
paired comparisons against retrieval, inner-selected candidates and fit logs
are in `autotune_reports/model_tournament_review_v1/`. There were no failed
experiment shards. Prediction timing includes feature extraction and was
measured under potentially concurrent CPU load, not as an isolated benchmark.

Budget-8/16/32 headline cohorts contain 482/478/362 contexts. Use the exported
`fixed_cohort_budget_curves.csv` to compare budgets on the same 362-context
chronological cohort, rather than mistaking dropped contexts for improvement.

Six CPU research tests and the three documentation checks passed. This study
does not qualify GPU kernels, measure end-to-end online UCB speedup, or deploy
trained model weights. See [the tournament guide](model_tournament.md) for
commands, model contracts, filtering and holdout details.

## Next implementation step

Freeze/export the residual and MLP candidates with prediction-equivalence
tests and cheap inference; keep retrieval as a first-class proposal source.
Measure disagreement/out-of-distribution behavior and preserve a known-good
incumbent. Then test that fixed proposal portfolio prospectively on new
geometries, including failed attempts, compilation cost and confirmation.
Do not spend another large GPU campaign merely repeating these same shapes.
