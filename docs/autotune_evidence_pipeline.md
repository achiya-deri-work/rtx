# From collected data to a first-hit prior

This workflow develops opt-in priors. It does not install runtime winners or
change the default autotuning policy. Historical results are development data;
the new-shape campaign is a separate evaluation set.

## 1. Versioned ingestion

Use the research environment (`requirements-research.txt`) for CPU analysis.
The source JSON maps `train` and `development` to lists of directories or ZIPs.
Keep those inputs unchanged. ZIPs are read in place.

```bash
.venv-research/bin/python -m tools.research.evidence_dataset \
  --sources autotune_reports/evidence_sources_v1.json \
  --output autotune_reports/evidence_dataset_v1
```

Choose a fresh output directory for each ingestion revision. The output contains:

- One audit JSON residual per input journal, including its hash, line-level
  provenance, classifications, failures, and compilation/evaluation costs.
- Compressed Parquet latency shards and an immutable completion manifest.
- Explicit exclusions for repeated context/config failures, cross-split overlap,
  and inconsistent repeated measurements.

Latency admission requires successful outcomes, at least three positive finite
timing samples, CV at most 5%, relative range at most 15%, and no explicit failed
stationarity flag. Repeated observation medians differing by over 20% are
excluded. These are conservative ingestion rules, not claims that all remaining
measurements are noise-free. Unknown timing quality is not silently accepted.
Numerical failures are never encoded as slow runtimes. Existing numerical
quarantines are retained; repaired revisions are not relabeled automatically.

Contexts retain family, kernel revision, device and semantic workload policy.
Raw features and journal hashes remain available for software-stack auditing.
Historical same-revision sources are not thereby proven implementation-equivalent;
this is a research view, not release qualification. In particular, the three
devices also differ in driver/binding versions, so SKU correlations are not
causal estimates of bus width, SM count, or wattage.

## 2. Attribute selection losses

```bash
.venv-research/bin/python -m tools.research.selection_diagnostics \
  --sources LOCAL_STUDY LAPTOP_STUDY.zip RTX5090_STUDY.zip \
  --output autotune_reports/selection_diagnostics_v1
```

The report distinguishes missing pool candidates, generated-but-unmeasured
candidates, measured-but-unselected candidates, and unknown pool membership.
It reports historical failed-evaluation cost and time to first valid result.
It cannot infer the rank of unrecorded proposals or impute their performance.
Its within-cell stability checks are separate from the ingestion view's
cross-observation exclusions; consult both before attributing a particular case.

Future portable-model runs emit `strategy_candidate_ranking` events containing
the complete ranked pool once per arm. Each selected proposal also records its
zero-based `pool_rank`, allowing ranking and scheduler allocation to be studied
separately.

## 3–4. Portfolios, then models

```bash
.venv-research/bin/python -m tools.research.portfolio_study \
  --dataset autotune_reports/evidence_dataset_v1 \
  --output autotune_reports/portfolio_study_v2 --threads 6

.venv-research/bin/python -m tools.research.portfolio_baselines \
  --dataset autotune_reports/evidence_dataset_v1 \
  --study autotune_reports/portfolio_study_v2
```

All train-only portfolios are constructed before fitting any model. A greedy
portfolio covers training contexts within 5% of their best observed latency,
up to 16 configurations per semantic contract/shape category. Unmeasured
configurations supply no coverage evidence; they are not labeled slow.

The fixed model comparison uses family-headed histogram boosting and extra
trees, with shape-only versus hardware-interaction feature views. Loss weights
balance families and contexts. Targets are context-centered log latency, not
absolute cross-device runtimes. Model fitting does not use held-out labels.

The protocols are chronological development, whole-(M,K)-group holdout across
devices/families/output widths, contraction-category holdout, and each SKU held
out in turn. The category holdout may share M/K with other categories; it tests
a different generalization question. Previously inspected data are not called
unbiased prospective validation.

Metrics at 8/16/32/64 candidates include regret against the **best observed**
configuration, near-best hit rate, comparison with random selection, worst
regret, and shape-cluster bootstrap intervals. Compact-portfolio plus random
selection is a separate control for the model's incremental value.

These are observed-successful-pool evaluations. They do not reproduce adaptive
UCB, unknown compilation failures, or first-hit wall time. Missing portfolio
members and unsupported contracts are reported explicitly.

## 5. Freeze and validate prospectively

```bash
.venv-research/bin/python -m tools.research.prospective_campaign prepare \
  --dataset autotune_reports/evidence_dataset_v1 \
  --study autotune_reports/portfolio_study_v2 \
  --sources autotune_reports/evidence_sources_v1.json \
  --output autotune_reports/portfolio_prospective_v1

.venv/bin/python -m tools.research.prospective_campaign run \
  --output autotune_reports/portfolio_prospective_v1 --pilot --max-tasks 8

.venv/bin/python -m tools.research.prospective_campaign compare \
  --output autotune_reports/portfolio_prospective_v1
```

Preparation requires all planned model fits to be complete. The predeclared
development selection rule minimizes the worst protocol's mean eight-trial
portfolio-plus-model regret. It exports the selected chronological training fit,
not a fit using prospective data. The NumPy-only `trees` head is checked against
sklearn predictions before export; runtime needs neither sklearn nor pickle.
The artifact stays research-only and is revision/semantic-contract checked.

The frozen plan excludes every previously attempted M/K pair found in the raw
source inventory, including failures. It records runtime, driver-script, and
artifact hashes. Changes require a fresh plan. Cells run in separate processes,
with shared compiler caches explicitly allowed. Completed cells are skipped
on resumption; failures require inspection rather than silent retries.

The bounded pilot runs two MXFP8 prequant shapes and four treatments at eight
trials: existing, mixed with a local-search reservation, compact prior without
learned ranking, and compact prior with learned ranking. The initial MXFP8
configuration also passes an independent FP32 GEMM numerical gate. Candidate
correctness gates remain active. Balanced finalist replays have separate
per-evaluation JSON residuals and separate measurement cost.

The larger prepared plan covers four forward families, four shapes, four
budgets and two replicates. It is **not launched automatically**. MLP and backward
remain withheld until their changed implementation revisions are revalidated;
the intermittent MLP numerical failure is not waived for data collection.

Only after fresh results demonstrate useful budget/latency tradeoffs without
unacceptable regression tails should a learned policy become a runtime default.
