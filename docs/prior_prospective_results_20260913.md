# Four-hour prospective prior study: three GPUs

## Decision

The frozen prior improves first-hit candidate selection, but not uniformly across
kernel families. Keep the artifact research-only for now. Use these results to
develop family-specific selection and adaptive budgets; do not deploy the
ranker alone or permanently prune the search space.

No production policy, kernel, numerical tolerance, or runtime winner cache was
changed during this analysis. The prospective observations were not used to
retrain the evaluated prior.

## Inputs and integrity

All runs used commit `95c791a` and frozen prior SHA256
`d8db87b1b492c2af7396a7b8b58dc1a76f32442da0dc565724dd23572ed12cd2`.

- Local: `autotune_datasets/prior_study_v1_achiyas-pc_20260913T082155Z`.
- Laptop: `prior_study_v1_DESKTOP-8GVONEQ_20260913T082441Z.zip`.
- 5090: `prior_study_v1_sphera2-Dell-Pro-Max-Tower-T2-FCT2250_20260913T082219Z.zip`.

Both ZIP CRC checks passed. All three logs end at their allocation deadline.
All 2,640 tasks have compatible priors. The 528 workload/device groups have
matching pools across all five policies. No malformed JSONL lines, incomplete
JSONL tails, duplicate observation IDs, repeated configurations within a task,
or violations of the prescribed candidate ordering were found.

| Device | Observations | Successful | Explicitly unstable successes | Successes without stationarity metadata |
|---|---:|---:|---:|---:|
| 5070 Ti | 36,753 | 36,666 | 44 | 3,341 |
| Laptop 5070 | 9,019 | 9,011 | 1,253 | 838 |
| 5090 | 33,892 | 33,830 | 33 | 3,039 |
| Total | 79,664 | 79,507 | 1,330 | 7,218 |

There were no recorded compile errors. Eleven workers were censored at shutdown;
those interruptions are not compiler failures or completed trials.

## Comparable coverage

Numbers below count workload groups where **all five policies** reached the
given budget. Partial deeper waves must not be mixed with full earlier waves
to claim an improvement from increasing the budget.

| Device | 4 trials | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| 5070 Ti | 176 | 176 | 176 | 176 | 53 |
| Laptop 5070 | 176 | 176 | 49 | 0 | 0 |
| 5090 | 176 | 176 | 176 | 176 | 35 |

## Eight-trial comparison

Regret is selected latency / observed oracle latency minus one, not speedup
against BF16 and not distance from the exhaustive optimum. Failed trials consume
budget. Explicitly unstable successes are excluded; successes lacking a
stationarity field remain eligible but are not certified stable.

The noise-reduced analysis assigns each configuration the median of its eligible
measurements across treatments within the same workload/device group. The oracle
is the best of these medians across all configurations observed in that group.
Only configurations actually measured successfully within a policy's prefix can
be selected. This is an offline schedule-selection diagnostic, not the exact
online winner's measured performance.

| Policy | Mean regret | Contexts within 2% of oracle |
|---|---:|---:|
| Original ordering | 46.18% | 21.0% |
| Random | 22.30% | 36.9% |
| Retrieval | 6.64% | 58.7% |
| Ranker | 11.62% | 51.9% |
| Hybrid | **4.53%** | **61.7%** |

All policies reached eight trials in all 528 contexts. Random, retrieval, and
ranker each lack an eligible successful result in one context; their conditional
mean regrets use 527 contexts. The within-2% denominator remains 528, so an
absence of eligible results is not counted as success.

Using raw per-treatment timings and the raw union oracle gives 5.10% hybrid
versus 23.14% random regret: the overall conclusion is not an artifact of the
median normalization.

| Device | Random regret | Retrieval | Ranker | Hybrid |
|---|---:|---:|---:|---:|
| 5070 Ti | 22.82% | 6.16% | 12.53% | **2.55%** |
| Laptop 5070 | 22.67% | 5.92% | 12.41% | **5.44%** |
| 5090 | 21.43% | 7.83% | 9.94% | **5.62%** |

### Family-specific signal

Each row covers 48 workload/device contexts at eight trials. These are
development recommendations derived from this study, not another unbiased
evaluation of a newly selected family router.

| Family | Random | Retrieval | Ranker | Hybrid | Next policy to validate |
|---|---:|---:|---:|---:|---|
| MXFP8 recomputed MLP | 102.79% | 1.73% | 0.73% | 0.48% | Hybrid |
| MXFP8 backward | 14.89% | 2.09% | 59.30% | 6.56% | Retrieval-first; investigate ranker outliers |
| MXFP8 fused forward | 66.00% | 21.65% | 12.48% | 14.80% | Ranker plus retrieval; retain exploration |
| MXFP8 prequant forward | 4.89% | 1.73% | 4.89% | 2.54% | Retrieval-first |
| MXFP8 weight-prequantized | 21.33% | 6.98% | 2.18% | 2.38% | Ranker/hybrid |
| MXFP8 fully-prequantized | 12.83% | 12.09% | 3.68% | 3.54% | Hybrid |
| NVFP4 JIT-region | 5.36% | 3.29% | 6.60% | 4.66% | Retrieval-first |
| NVFP4 delayed | 3.28% | 5.71% | 12.45% | 4.45% | Preserve random/diversity slots |
| NVFP4 block | 3.96% | 2.64% | 5.46% | 3.05% | Retrieval-first |
| NVFP4 weight-prequantized | 4.66% | 3.76% | 3.48% | 5.02% | Resolve numerical failures before promotion |
| NVFP4 fully-prequantized | 5.00% | 11.28% | 16.44% | 2.39% | Hybrid |

At four trials, retrieval averages 10.22% regret versus hybrid's 13.99%.
This supports trying a short retrieval-first phase before spending the larger
budget on learned ranking and diversity. It does not establish the optimal
switch point without evaluating that composed policy.

## Measurement and operational caveats

- Laptop instability is material: 13.9% of successful observations explicitly
  failed stationarity. Even among eligible repeated measurements, the median
  max/min spread per configuration was 4.34%, versus 0.38% on 5070 Ti and 0.39%
  on 5090. Corresponding p90 spreads were 22.52%, 1.59%, and 2.00%. These are
  cross-treatment spreads, not confidence intervals or proof of a thermal cause.
- At eight trials, mean accumulated observation time was 18.9 seconds for
  hybrid versus 17.0 seconds for random. These numbers include GPU lease waits,
  exclude some worker/harness setup, and share compiler caches. They are **not**
  cold first-hit latency. Fixed accumulated-evaluation-time comparisons are
  exported separately with matched censoring, but do not solve this limitation.
  At a 20-second accumulated-evaluation budget, 469 groups have enough follow-up
  in every arm: retrieval's conditional regret is 5.45%, hybrid's 6.76%, and
  random's 22.77%. They lack an eligible result in five, three, and two groups
  respectively. At 40 seconds hybrid leads, but only 142 groups remain matched.
  This reinforces short-budget retrieval as a candidate, not a universal winner.
- Descriptive bootstrap intervals in the aggregate export resample contexts.
  Reused shapes across devices and a single seeded study limit independence;
  do not interpret them as replicated hardware-population guarantees.
- The prior was frozen before this study, but its training data included older
  results from these GPU models. This is prospective workload evaluation, not
  unseen-SKU validation. `new_shape_candidate` is not proof of training exclusion.

## Failures that must remain visible

- 131 correctness failures in NVFP4 weight-prequantized inference: 65 local,
  60 on 5090, six on laptop. Prior work implicated reciprocal conversion paths;
  this analysis does not establish that every new failure has the same cause.
- Four MXFP8 prequant forward correctness failures, with large absolute errors.
- Two MLP correctness failures; the logged relative-L2 errors were 0.00313 and
  0.00584. Do not loosen the gate just to retain these configurations.
- Twenty MLP runtime failures report an `AccumulateGrad` stale default-stream
  reference during CUDA graph capture (18 local, two laptop).

Failures were excluded from latency selection, retained as consumed trials,
and exported for targeted reproduction. None were silently promoted to winners.

## Next work

1. Reproduce and quarantine numerical/capture failures with exact configuration
   and workload identities. Keep the raw observations immutable.
2. Build a family-aware research policy: retrieval-first short budget, then
   ranking/diversity where this study supports it. Keep fallback exploration.
3. Diagnose MXFP8 backward ranker outliers and evaluate the composed policy on
   a fresh holdout; avoid scoring it on data used to choose its routing rules.
4. Measure actual isolated first-hit overhead, including prior loading, pool
   generation, compilation, confirmation, and cache reuse. Use the desktop
   measurements for tight latency decisions until laptop stability is understood.
5. Promote only validated, revision-safe measured winners. This study does not
   itself authorize a release or runtime-default change.

## Artifacts and reproduction

Generated reports are under
`autotune_reports/prior_prospective_three_device_v1/`:

- `policies.csv` / `policies.parquet`: existing raw timing summary.
- `audit.json`: candidate-pool matching audit.
- `deep_analysis.json`: record/worker audit, aggregates, noise, failure counts.
- `canonical_policies.parquet`: median-normalized per-context results.
- `coverage.parquet`: matched-budget completion indicators.
- `aggregates.parquet`: device/family/policy summaries.
- `evaluation_time_policies.parquet`: accumulated-evaluation-time prefixes.
- `failures.json`: device/family/policy/configuration IDs and errors.
- `analyze.py`: the supplementary analysis script; run with `PYTHONPATH=.` and
  the same three input paths from the repository root to regenerate the deeper
  artifacts. It uses the fixed report output directory above.

Regenerate the raw summary with:

```bash
python -m rtx.autotune.prior_summary \
  prior_study_v1_sphera2-Dell-Pro-Max-Tower-T2-FCT2250_20260913T082219Z.zip \
  prior_study_v1_DESKTOP-8GVONEQ_20260913T082441Z.zip \
  autotune_datasets/prior_study_v1_achiyas-pc_20260913T082155Z \
  --output autotune_reports/prior_prospective_three_device_v1
```
