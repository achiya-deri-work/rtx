# Protected first-hit search and conditional heuristics

This is development evidence, not promotion of a new default or a comparison
against BF16. The prior remains opt-in.

## Consolidated regression gate (2026-09-14)

After the delayed-policy identity and runtime-lifecycle work, the combined
CPU suite passed 62 tests with 14 subtests in 19.15 seconds. It covers effort
configuration, progressive resume, cache contracts/process locking, paired
confirmation and promotion, matched-pool membership/comparisons/auditing,
source-snapshot integrity, and allocator-guard behavior. Evidence:
`/tmp/rtx_consolidated_effort_cpu_20260914.xml`.

Two research tests skipped in the runtime environment because sklearn is not
installed there. Both passed in `.venv-research`, alongside nine other research
tests: matched ranking permutations, portfolio accounting/grouped splits,
forest/histogram export parity, conditional-rule revision/held-out guards,
feature leakage exclusion, and evidence admission/identity separation. Commands:

```bash
.venv-research/bin/python -m unittest tests.test_effort_research tests.test_matched_rankings -v
.venv-research/bin/python -m unittest tests.test_portfolio_study tests.test_portfolio_export tests.test_conditional_heuristics tests.test_evidence_dataset -v
```

This is regression coverage, not new model-quality evidence. The outstanding
promotion gate remains broader independently frozen shape/seed evaluation,
finite-pool references, tail risk, and full first-hit startup cost. Current
forward pilots cannot qualify backward or revision-incompatible historical
models, and no default is promoted by these test results.

## Runtime change

### Paired confirmation of the two fresh shape groups

`tools/research/runtime_prior_compare.py` replays the public compiled frontend
with explicit saved configurations, shared inputs/weights, and ten balanced
forward/reverse rounds. Every distinct configuration is prepared and validated
against FP32 GEMM cast to BF16 before timing; each evaluation uses seven samples
of twenty graph replays and verifies captured output equality afterwards.
Identical serialized configs share one measurement, making policy-selection
ties exact. The original search journals and winners are not modified.

Source admission verifies each original archive in full, then requires identical
current `rtx/` sources and the original input-generation probe. Newly added
analysis code has its own saved source snapshot. Device/software identity is
checked. CPU tests reject mixed/partial selection matrices and execution-source
changes while permitting separately versioned analysis additions.

All 50 small-batch and 60 large-shape confirmation evaluations passed, with no
invalid candidates. Paired relative latency versus the ordinary selection:

| Shape | Budget | Retrieval | Trees |
|---|---:|---:|---:|
| 320×1536×1920 | 8 | exact tie | exact tie |
| | 16 / 32 | exact tie | +0.018% (effectively tied) |
| | 64 | +1.257% [0.973%, 1.545%] | -0.031% [-0.043%, 0.117%] |
| 6144×3072×1280 | 8 | -0.061% [-0.102%, 0.018%] | exact tie |
| | 16 | +0.723% [0.538%, 0.852%] | +0.625% [0.580%, 0.899%] |
| | 32 | +0.723% [0.538%, 0.852%] | -0.298% [-0.483%, -0.088%] |
| | 64 | +0.812% [0.644%, 0.887%] | +0.081% [0.007%, 0.221%] |

Intervals bootstrap balanced timing pairs within one context; they do not
measure generalization across shapes/devices/seeds. The apparent small-batch
16/32-trial ML improvement in the earlier **unpaired** measurements did not
reproduce in this paired protocol. Do not retain that apparent gain as evidence
for promotion. Large-shape differences are small, and retrieval has measured
regressions. This weakens, rather than strengthens, a universal-prior default.

Artifacts are `runtime_prior_fresh_shapes_20260914_v1/narrow-paired/` and
`large-paired/`, each with selected-config/input hashes, numerical evidence,
source snapshot, fsynced measurement journal, confirmation and summary JSON.
Reproduce into a fresh output directory:

```bash
.venv/bin/python -m tools.research.runtime_prior_compare \
  --root autotune_reports/runtime_prior_fresh_shapes_20260914_v1/narrow \
  --output autotune_reports/runtime_prior_fresh_shapes_20260914_v1/narrow-paired-NEW
```

Nine focused CPU comparison/source-integrity tests pass. This still qualifies
only selected forward-inference configurations on one GPU. An all-identical
selection matrix is explicitly rejected as needing a selection-tie report, not
a fabricated paired contest. No runtime winners or policy defaults are promoted.

### Two fresh public-runtime shape groups

The next plan was frozen before timing at `runtime_prior_fresh_shapes_20260914_v1`:
M×N×K=320×1536×1920 and 6144×3072×1280, seed 2718, ordinary/retrieval/trees,
cumulative 8/16/32/64, fresh process per cell, isolated shape/mode caches.
Both (M,K) groups were absent from the saved historical/prospective exclusions.
This is two additional forward-inference contexts on one GPU, not broad
cross-device qualification or a finite-pool regret measurement.

All 24 public compiled runs passed numerical, one-graph, same-stream capture,
repeat-output and no-retuning checks. All 384 search observations passed. The
saved per-cell JSON includes normalized RMS, first-call wall time and all graph
samples; logs, source snapshots, runtime winners and journals are retained.

Descriptive **unpaired** median microseconds:

| Shape | Budget | Ordinary | Retrieval | Trees |
|---|---:|---:|---:|---:|
| 320×1536×1920 | 8 | 21.208 | 21.378 | 21.515 |
| | 16 | 21.560 | 21.413 | 20.550 |
| | 32 | 21.403 | 21.613 | 20.581 |
| | 64 | 20.466 | 20.552 | 20.482 |
| 6144×3072×1280 | 8 | 193.523 | 193.018 | 193.808 |
| | 16 | 193.006 | 193.274 | 193.934 |
| | 32 | 193.163 | 193.120 | 192.528 |
| | 64 | 191.798 | 194.126 | 192.595 |

The small-batch case suggests incremental ML value at 16/32 trials; the large
case does not show a clear prior advantage. The three small-batch eight-trial
selections are the **same configuration**, despite different measured medians:
do not attribute that variation to a search policy. Paired confirmation of
distinct selected configurations is required before interpreting small deltas,
including the apparent large-shape retrieval regression at 64. The result does
not justify a universal learned-prior default.

At eight trials, first-call costs were 3.067/3.658/3.988 seconds for
ordinary/retrieval/trees on the small-batch shape, and 2.952/3.635/3.798 seconds
on the large shape. Prior setup must earn that overhead through reduced search
or repeated execution savings; these costs omit process/import startup and use
uncleared compiler caches. Preserve these shapes as inspected evaluation data,
not training examples for the same prior being qualified.

### Public compiled prior probe (progressive revision 2)

`tools/research/runtime_prior_probe.py` exercises the real `rtx.MXFP8Linear`
materialized inference frontend under `torch.compile(fullgraph=True,
dynamic=False)`. At the already inspected 4096×768×1792 shape, seed 2718,
ordinary, retrieval-only, and retrieval+tree policies each refine cumulatively
through 8/16/32/64 in fresh processes with separate caches. This is development
evidence, not an untouched prospective test and not the finite-pool ablation.

All 12 runs passed independent FP32-GEMM-to-BF16 finite/7%-normalized-RMS checks
(observed ~3.7794%), one-graph compilation, same-stream warmup/capture, repeated
output equality, and no extra tuning or journal changes during replay. All 192
search observations passed. The two retrieval candidates appear at trials four
and five. The ML arm is actually used later; it does not improve selection here.

Unpaired CUDA graph-replay median latency, microseconds (15 samples × 20 replays):

| Budget | Ordinary | Retrieval only | Retrieval + trees |
|---|---:|---:|---:|
| 8 | 55.270 | 49.160 | 49.158 |
| 16 | 52.051 | 49.155 | 49.149 |
| 32 | 52.294 | 49.173 | 49.181 |
| 64 | 49.146 | 49.158 | 49.155 |

Retrieval and trees retain the same winning configuration across all budgets.
Ordinary reaches comparable latency at 64. These separate-process timings are
descriptive, not paired significance claims. They include the public compiled
operation and must not be compared directly to isolated-harness timings above.

First compiled call costs in seconds (8/16/32/64): ordinary
3.149/2.909/5.498/8.712; retrieval 3.435/4.113/6.014/9.657; trees
3.581/4.315/6.158/9.574. Later entries are refinement invocations, not independent
full-budget cold runs. Imports, process startup, module/input construction and
graph capture are excluded; compiler disk caches were not cleared. These are
not clean per-component compiler/model-cost estimates.

Artifacts: `autotune_reports/runtime_prior_probe_20260914_v2/` retains source
snapshots, per-budget results, logs, runtime winners, and raw journals. Version 1
is preserved as a failed harness attempt: it warmed the default stream but
captured a different stream, correctly hitting the runtime's warmup guard.
Version 2 explicitly warms its capture stream; no production guard was relaxed.
Run each budget/mode in a fresh process, sharing only that mode's output root:

```bash
.venv/bin/python -m tools.research.runtime_prior_probe \
  --output autotune_reports/runtime_prior_NEW/retrieval \
  --mode retrieval --budget 8 \
  --artifact autotune_reports/portfolio_prospective_v1/portfolio_predictor.json.gz
```

This supports the usefulness of the integrated retrieval opening on this case,
not causal attribution to the overlap fix, a universal ML win, backward coverage,
or default promotion. The next quality gate remains untouched shapes/seeds with
paired comparisons and separately accounted startup costs.

### Larger-shape matched reference (2026-09-14)

The frozen MXFP8 materialized-forward study at M=4096, N=768, K=1792,
seed 2718, RTX 5070 Ti completed all four treatments at 8/16/32/64 trials.
All 256 search observations and all 386 full-pool reference observations passed
their harness numerical gates. Initial independent FP32-GEMM-to-BF16 normalized
RMS was 3.7405% (7% gate). Ten balanced confirmation rounds covered six distinct
configurations; all 60 evaluations passed. The integrity audit reports no issues.

Paired median latency in microseconds:

| Budget | Ordinary | Portfolio | Rules | Portfolio + ML |
|---|---:|---:|---:|---:|
| 8 | 53.530 | 53.530 | 53.530 | 53.530 |
| 16 | 53.530 | 47.753 | 53.530 | 47.753 |
| 32 | 53.530 | 47.753 | 53.530 | 47.753 |
| 64 | 47.630 | 47.753 | 47.630 | 47.753 |

The slower selection has 12.37% paired excess latency over the confirmed
reference (within-context bootstrap interval 11.99–12.56%). Portfolio's selection
has 0.39% excess, with interval -0.28–0.51%; it is not clearly distinguishable
from that reference here. Portfolio and ML chose identical configurations, so
this study does not show incremental ML value. The reference is the best of the
screened pool's top five after paired confirmation, not a global optimum.

Only four of 386 screened candidates were within 5% of the best. Uniform
sampling without replacement reached that band with probabilities 7.9%,
14.45%, 31.2%, and 51.8% at 8/16/32/64 trials (2,000 simulated draws from the
measured finite pool). These are descriptive screening statistics, not paired
policy superiority probabilities or training labels.

Mechanism: portfolio's near-reference winner was retrieval rank two and was
evaluated at trial nine by the `frozen_ranking` arm. Ordinary/rules reached their
reference winner at trial 64 through local search. **This ranking ablation does
not exercise the runtime's explicit two-candidate retrieval opening.** Do not
infer that the runtime eight-trial policy must also miss this candidate. The
next useful experiment compares that integrated opening against the same frozen
pool, first as development evidence here, then on untouched shapes/seeds.

The shallow rules tied 338/386 candidates; learned ranking produced 265 distinct
scores. Ranking setup took 0.802 seconds. Summed search-evaluation time per
64-trial treatment was 19.83–21.45 seconds, with 14.82–16.43 seconds reported by
the harness as compilation/setup. Those costs exclude confirmation and process
startup and are not clean cold-compiler measurements. No default promotion.

Artifacts: `autotune_reports/matched_large_20260914_v1/` contains the prespecified
plan, exclusions, frozen rankings/pool, source ZIP, per-cell logs and journals,
complete reference, paired results, distribution summary, and integrity audit.
The source ZIP has 221 files and SHA-256
`5b6792c4a0490f245b21c5be9669cf9dc9fffae548f88f8407319a9e720fef16`.
This inspected shape is now development evidence and remains excluded from
prior training and untouched prospective qualification. One new geometry/seed
does not establish cross-shape, cross-device, or backward quality.

The previous `baseline_trials` option reserved adaptive local-search arm pulls,
not the initial configuration's neighborhood. A near-tied retrieved candidate
could change the anchor and consume those pulls on a different path.

Two new, default-off controls address that mechanism:

- `initial_neighborhood_trials`: a deterministic, legal, deduplicated prefix
  anchored to the initial config, before retrieval. It shares normal budgets,
  validation and persistence; resume skips previous attempts.
- `local_anchor_tolerance`: a minimum fractional improvement before changing
  the chronological local anchor. It is a policy margin, not a confidence bound.
  It does not alter raw timing, final selection, or bandit rewards.

The pilot uses two initial neighbors and a 1% anchor margin. Those are tested
policy settings, not claims of optimal allocation at every effort scale.

## Fresh local GPU pilot

RTX 5070 Ti, Torch 2.13.0+cu132. Eight cells: two previously unobserved (M,K)
groups, four policies, eight trials each, one search seed, hot-cache forward.
Previous pilot shapes were explicitly excluded before freezing the new plan.
All 64 candidate outcomes passed. Independent initial-config checks use FP32
GEMM cast to BF16 with finite outputs and relative L2 below 0.07; all other
candidates also pass the adapter's numerical gate.

Each winner was validated again, then all four policies were timed in six
balanced forward/reverse rounds (48 confirmation evaluations). Median of round
medians, microseconds:

| M×N×K | Ordinary search | Protected mixed | Protected portfolio | Protected portfolio + ML |
|---|---:|---:|---:|---:|
| 512×768×1152 | 10.782 | 10.775 | 9.980 | 9.988 |
| 1920×512×1408 | 17.650 | 17.643 | 17.637 | 17.624 |

Paired portfolio/ordinary latency ratio is about 0.926 on the first shape and
0.999 on the second. Portfolio and learned policies chose identical configs
on both shapes; ordinary and protected-mixed also chose identical configs.
This supports further testing of retrieval plus a protected initial prefix,
not a claim that learned ranking adds value or that either guard caused the win.
The combined guard change was not independently ablated.

First-hit search took 2.55–2.70 seconds for ordinary search and 3.12–3.65 seconds
for the portfolio policies. These are descriptive: CPU historical rule fitting
ran concurrently. Compilation/setup/search is excluded from paired graph timing;
compiler disk caches were not cleared. This is not a cold-machine startup test.

Artifacts: `autotune_reports/portfolio_guard_prospective_v1/`, including frozen
plan, per-cell journals, raw paired evaluations and `summary.json`.
Environment: `/tmp/rtx_guard_environment.json`.

## Conditional rule study

`tools/research/conditional_heuristics.py` fits depth-four decision trees with
context-balanced weights and separate revision/semantic-policy heads. Inputs
are pre-execution config, shape, SKU and physical interaction estimates. No
measured compiler resources or test latencies are input features. The target is
within-context centered log latency, not absolute cross-device latency.

Rules are readable JSON text exports. Continuous feature thresholds are signed
log1p transformed; leaf values are centered log-latency predictions. They are
associations from adaptively collected data, not causal hardware laws. In
particular, three SKUs do not identify independent effects of bus width, power,
SM count and software differences.

All six historical splits reuse the existing portfolio study's observed pools
and portfolio prefix. Unmeasured candidates are not assigned slow labels.
This isolates ranking on observed candidates; it cannot simulate the protected
initial neighborhood where the historical data lack those timings.

Artifacts: `autotune_reports/conditional_heuristics_v1/`. No rule hard-prunes
legal candidates or enters the runtime prior automatically.

Completed results at eight trials, relative to each context's best observed
stable configuration (lower regret is better):

| Historical split | Portfolio + random mean regret | Portfolio + rules mean regret | Rules worst regret |
|---|---:|---:|---:|
| Chronological development | 13.84% | 8.67% | 476.13% |
| Whole-(M,K) held out | 4.61% | 2.26% | 63.15% |
| Contraction class held out | 5.40% | 3.24% | 84.33% |
| Laptop 5070 held out | 4.65% | 1.98% | 91.18% |
| 5070 Ti held out | 5.03% | 3.03% | 112.58% |
| 5090 held out | 6.06% | 4.32% | 192.97% |

The simple rules improve mean regret over portfolio + random in every split,
but the tails are unacceptable for hard pruning. On held-out (M,K), they improve
on the previously selected ExtraTrees model's 3.75% mean / 354.5% worst regret;
on chronological data they lose to its 5.71% mean and still fail badly in the
tail. There is no universal winning ranker here. The chronological rules versus
uniform-random shape-cluster bootstrap interval also crosses zero; the apparent
average gain should not be described as uniformly statistically significant.

The minimum tree-leaf size is 32 observations, not 32 independent contexts.
Further rule admission needs distinct-context/SKU support and tail-risk checks.
Next qualification should ablate the two guards, include 8/16/32/64 budgets,
and expand beyond two MXFP8-forward shapes before changing defaults. Paired
runtime finalist promotion and matching protected prefixes across a broader
identical-pool policy study remain outstanding.

## Matched-pool qualification preparation

The existing prospective pilot is not a clean ranker ablation: `existing`
versus guarded treatments changes candidate mixing, initial protection and
exploration together. Its observed win cannot be attributed to ML ranking.
Historical observed-pool comparisons control the available timings, but cannot
reconstruct an unmeasured initial neighborhood. Neither proves the complete
progressive runtime policy's quality.

`tools/research/matched_pool.py` now supplies a research-only finite adapter for
the next study. It enforces membership for random sampling, local neighbors,
static rejection and direct evaluation (including finalist evaluation). Pool
identity includes the base context, initial configuration and sorted candidate
IDs; changing the pool or initial anchor produces a separate journal context.
Input order and duplicate IDs do not change the frozen pool. Its deterministic
initial-neighborhood prefix can be shared across treatments. Model features
remain the base pre-execution features; the pool identifier is bookkeeping,
not a predictor feature. No missing candidate gets an invented latency label.

`make_matched_tuner` now composes a controlled four-treatment ranking ablation
using the production orchestrator. Each treatment gets the same coordinate,
uniform-random and frozen-ranking arms, progressive UCB exploration schedule,
protected opening and paired finalist confirmation. Ordinary uses a seeded
random permutation; portfolio, rules and portfolio+ML require explicit complete
rankings of the same pool. Missing ranks and duplicate/foreign candidates fail
closed. Pool, ranking, treatment, seed and opening enter the resume identity.
Budgets can increase without changing that identity. This evaluates ranking
value under shared search mechanics, not the full deployed policy or its model
loading/feature-extraction startup overhead; that needs separate measurement.

CPU tests use synthetic rankings to verify common openings and pool membership,
reserved exploration counts at each effort, exact 8/16/32/64 resume equivalence
under fixed cost evidence, and rejection of changed rankings on resume. Actual
train-only artifact rankings are now connected for the MXFP8-forward pilot
below; multi-context and other-family qualification remain outstanding.
No default promotion is claimed. Freeze common pools and rankings before measuring, record artifact
construction costs separately, and keep those new evaluation cells held out.

Conditional rules can now be exported as data-only trees with
`conditional_heuristics --export-only`. Export accepts only explicitly `train`
rows, preserves exact semantic/revision heads, and verifies every training-row
prediction against sklearn with absolute tolerance `1e-12`. Tree inputs retain
sklearn's float32 conversion before threshold comparisons. The artifact records
training context IDs and whole-(M,K) geometries; prospective prediction rejects
either overlap and rejects unsupported heads rather than reporting ordinary
fallback as a rule result. These admission checks are necessary but not a
substitute for excluding all previously inspected development geometry in the
campaign plan. Export does not enable runtime deployment or hard pruning.

Local export `autotune_reports/matched_rules_20260914_v1.json.gz` completed
from 52,552 training rows, 1,845 contexts and 37 whole-(M,K) geometries, producing
12 semantic/revision heads in 43,748 compressed bytes. All 52,552 predictions
passed export parity (`atol=1e-12`, `rtol=0`); fitting/export took 78.38 seconds.
SHA256: `829d5628d1c1d67cfd38623d1f80cf4888e3f9bd09fc40655c56ca68e8fe4023`.
This is training/export validation, not held-out ranking quality evidence.

```bash
.venv-research/bin/python -m tools.research.conditional_heuristics \
  --dataset autotune_reports/evidence_dataset_v1 \
  --output autotune_reports/matched_rules_NEW.json.gz --export-only
```

Current delayed evidence revision is 4; revision-3 learned bundles must not be
silently relabeled to participate. Unsupported contracts need a reported
control-only stratum or new training evidence, not a hidden fallback presented
as a learned treatment.

## Matched pilot through 64 trials and paired comparison

All four treatments completed 8→16→32→64 refinement, adding exactly 8/8/16/32
observations. Each journal contains 64 distinct candidate IDs, all successful:
256 successful search evaluations total, with no repeated candidate IDs inside
a treatment. Cross-treatment repetitions are intentional controls. Each used
19–20 local, 20–21 uniform-random and 21 ranked proposals, plus the shared
initial configuration and two protected neighbors. This tests live resumed
budget accounting; exact uninterrupted sequence equivalence remains the CPU
fixed-cost test contract, not a claim about nondeterministic CUDA timings.

`paired/` stores 10 balanced rounds over the four distinct selected configs
(40 successful confirmation evaluations). Each selected config is measured
once per round and shared selections are exact ties, not independently timed
copies. Paired medians in microseconds:

| Effort | Ordinary | Portfolio | Rules | Portfolio + ML |
|---|---:|---:|---:|---:|
| 8 | 16.338 | 15.072 | 15.601 | 15.072 |
| 16 | 16.338 | 15.072 | 15.601 | 15.072 |
| 32 | 16.338 | 15.072 | 15.601 | 15.072 |
| 64 | 15.307 | 15.072 | 15.072 | 15.072 |

At effort 8 the portfolio/ordinary paired ratio is 0.92187, with a balanced-pair
bootstrap 95% timing interval [0.92037,0.92358]. At 64 it is 0.98443
[0.98275,0.98739]. These quantify measurement variation on **one context**, not
policy generalization or a probability of winning on future shapes. This first
comparison preceded the full-pool reference measured below.

The rule scores expose a weakness: 371/401 candidates share one score, with
only four distinct scores total. Thus many rule proposals are ordered by the
config-ID tie-break, not meaningful performance distinctions. ML has 259
distinct scores (largest tie 41) but still adds no selected-winner benefit over
portfolio retrieval in this pilot at any effort. `rankings_diagnostic.json`
adds this audit; its four rankings were verified identical to the original
frozen rankings. The experiment was not retuned after observing its results.

Next evidence needs more contexts/replicates and supported family contracts. Do not respond
to this single result by promoting a prior or changing defaults.

### Complete finite-pool reference

`reference/` independently enumerated all 401 configurations in seeded order
in approximately 125 seconds; all 401 passed. Its screened best was the same
`06e41e62e47329058251` selected by portfolio at trial 8. Only six candidates
were within 2% of its screened latency and eleven within 5%.

`paired_reference/` remeasured the five best screened configurations together
with every policy winner in ten balanced rounds. The portfolio choice remained
the best among those finalists. Paired excess latency over that reference:

| Effort | Ordinary | Portfolio | Rules | Portfolio + ML |
|---|---:|---:|---:|---:|
| 8–32 | 8.48% | 0% | 3.37% | 0% |
| 64 | 1.58% | 0% | 0% | 0% |

Zero means an identical selected configuration, not precision beyond the timing
noise. This establishes observed finite-pool success, not a global optimum:
only the top five screened candidates received this additional confirmation.

The separate screening-only uniform random baseline (2,000 draws without
replacement per budget) had mean regret 8.64/6.85/4.64/2.63% at 8/16/32/64,
and probability of reaching within 5% of the screened best
19.25/36.0/61.15/86.1%. It is not the orchestrated ordinary treatment, which
also protects initial neighbors and runs local search. Missing/failed draws
would count as unsuccessful trials, never imputed latency training labels;
none occurred here. `screening_summary.json` records the exact scope.

The plateau in portfolio performance is genuine within this frozen pool, not
evidence of failed exploration. In contrast, the simple rule tree's ties
explain its weak ordering resolution. Keep this pilot held out: improving the
model using these findings requires new prospective evaluation contexts.

## Reproduction commands

### Source snapshots for new experiments

New `matched_gpu prepare` outputs include a sibling `pool.sources.zip` and its
digest in the pool manifest. The snapshot preserves the exact Python files
under `rtx/`, `tools/`, and `tests/`, plus `setup.py`, `pyproject.toml`,
`requirements.txt` and `requirements-dev.txt` when present. It does not include
datasets, logs, Git history, virtual environments, docs or unrelated projects.
The archive has deterministic member ordering/timestamps and per-file digests.
Verification performs no extraction or execution, rejects unsafe member paths,
symlinks, digest changes and inventory changes, and checks the current workspace
before new frozen studies execute. Snapshots are exclusive-create.

This preserves dirty source state that a Git SHA or source hash alone cannot
recover. It is not an environment/container snapshot: exact installed package,
driver and CUDA versions still belong in the study's context/environment record.
The older pilot and expansion manifests predate this addition and are **not**
retroactively certified as having archived source. Their existing source-hash
guards remain enforced; do not rewrite them to accept a changed worktree.

Local source-snapshot qualification produced
`autotune_reports/source_snapshot_20260914_v1.zip`: 215 files, 3,118,605 source
bytes, SHA256 `bb805fc217d78e60f2d34a3a96e5eb17941fc3f1496a6ce1aad44aacf8b35389`.
Sixteen focused CPU tests passed, including deterministic output, overwrite
refusal, content tampering, path traversal/symlinks and study snapshot guards.
No GPU kernels or runtime policy defaults changed in this addition.

### First matched-pool CUDA pilot

`autotune_reports/matched_pilot_20260914_v1/` freezes 401 legal candidates for
MXFP8 prequant forward, hot `M=640,N=768,K=1664`, seed 1729. Geometry exclusion
covered 199,038 raw observations and 12 earlier plan files (67 distinct M,K
pairs); this M,K was absent. The GPU-validation gate compared an independent
FP32 GEMM cast to BF16: finite, normalized RMS 0.0375673 against the existing
0.07 gate, RMS error 0.00305666, reference RMS 0.0813651, maximum absolute error
0.0150146, normalized maximum 0.0380569. It is a forward-only pilot, not a
backward or full-library qualification.

All four 8-trial treatments completed with valid candidates and paired internal
finalist confirmation. A separate final harness verification measured:

| Treatment | Verified winner (µs) | Worker setup + search + verification (s) |
|---|---:|---:|
| Ordinary | 16.320 | 4.98 |
| Portfolio | 15.063 | 5.10 |
| Rules | 15.590 | 5.25 |
| Portfolio + ML | 15.057 | 5.00 |

The portfolio and portfolio+ML issued identical configurations at this budget;
there is no evidence here of incremental ML value. Timings above come from
sequential treatment processes, not a cross-treatment paired confidence
interval. One context and replicate cannot justify promotion. Artifact loading
and ranking creation took another 0.801 seconds in the CPU research process;
the worker figures exclude that cost and Python startup. Later stages must
measure integrated first-hit startup and include more contexts/replicates and
budgets. Original evidence is unchanged and no runtime winner was installed.

`tools.research.matched_gpu prepare` writes the pool after independent initial
validation; `tools.research.matched_rankings` runs in the CPU research venv and
freezes all orders; `tools.research.matched_gpu run` runs one treatment/budget
and resumes that treatment's isolated journal. Result files are exclusive-create
and repeated completed cells must not be overwritten. The launcher now has
independent FP32-GEMM oracles for MXFP8 prequant forward, NVFP4 block dynamic
forward and NVFP4 JIT-region forward. Other families deliberately fail closed
until their independent oracles are added.

### Expansion follow-up and compatibility boundary

`python -m tools.research.matched_audit <context-root> --output <new-report.json>`
audits completed four-treatment contexts without changing their evidence. It
checks the frozen pool-file digest, ranking permutations, exact context and
family/revision identities, config payloads, duplicate trials, JSONL tails,
the common protected opening, 8/8/16/32 refinement increments and successful
winner membership in each budget prefix. It also requires a durable matching
winner decision at every budget, complete balanced forward/reverse confirmation
rounds, and finite successful measurements for the selected winner in every
round. Missing confirmation cannot pass merely because screening succeeded.
A partial running campaign is reported
incomplete rather than interpreted as completed evidence. Costs distinguish
recorded candidate evaluation time and recorded compile time; missing compile
costs are counted rather than silently treated as measured zeros. Confirmation
and process startup are outside those sums.

The completed first pilot passes this audit with no issues
(`matched_pilot_20260914_v1/integrity_audit.json`). Its four 64-trial searches
recorded 21.66–22.35 seconds of candidate evaluation each, including
16.77–17.45 seconds recorded as compile/setup by the benchmark harness. This is
not a measurement of compiler backend time alone or integrated first-hit cost.

`autotune_reports/matched_expansion_20260914_v1/` freezes the new
`M=896,N=2304,K=1664` geometry for the three oracle-supported families, sharing
seed 1729 and the same four treatments/budgets. Exclusions extend the raw-source
and prior-plan audit with the inspected pilot's M,K, for 68 excluded pairs.
Each family's ranks were frozen before its first candidate timing. Legal pool
sizes are 401 MXFP8, 445 NVFP4 block, and 515 JIT-region candidates.

Initial independent normalized RMS errors are 0.0373642 / 0.161083 / 0.134369
respectively, with unchanged gates 0.07 / 0.20 / 0.20; all are finite.
Absolute maxima are 0.0163574 / 0.0654278 / 0.0545654. This is distribution-level
forward validation, not a gradient or all-distribution qualification. The
progressive campaign is launched low-budget-first, with each treatment in a
separate process and journal. Completion and comparisons must be read from
result artifacts, not inferred from this launch description.

The historical predictor/rules artifacts do **not** support current MXFP8
backward (trained revision 22, current 23) or tensor-delayed NVFP4 (trained 3,
current 4). They must remain explicitly excluded from learned-treatment
comparisons until revision-compatible training evidence exists. Runtime
correctness/cache qualification for these families is a separate gate; it does
not make old latency labels compatible. No backward or delayed fallback may be
reported as an ML treatment.

The expansion follow-up completed all 48 family/treatment/budget cells:
768 successful, unique-within-treatment search evaluations; all three context
audits pass, including paired winner decisions. Cross-treatment confirmation
then performed 150 successful evaluations over 15 distinct family/config
finalists (ten balanced rounds per family). Paired medians in microseconds:

| Family | Effort | Ordinary | Portfolio | Rules | Portfolio + ML |
|---|---:|---:|---:|---:|---:|
| MXFP8 | 8 | 33.907 | 32.592 | 33.907 | 32.592 |
| MXFP8 | 16 | 33.907 | 32.592 | 33.179 | 32.592 |
| MXFP8 | 32 | 32.665 | 32.592 | 33.179 | 30.859 |
| MXFP8 | 64 | 30.859 | 30.859 | 30.991 | 30.859 |
| NVFP4 block | 8–16 | 21.020 | 21.020 | 21.020 | 21.020 |
| NVFP4 block | 32 | 21.020 | 21.020 | 20.630 | 21.020 |
| NVFP4 block | 64 | 20.866 | 20.866 | 20.630 | 20.897 |
| NVFP4 JIT-region | 8 | 28.277 | 28.277 | 28.277 | 28.277 |
| NVFP4 JIT-region | 16–32 | 27.499 | 27.499 | 27.499 | 27.499 |
| NVFP4 JIT-region | 64 | 25.972 | 26.016 | 25.589 | 26.016 |

Here ML reaches the eventual MXFP8 winner at effort 32 instead of 64 for
ordinary/portfolio search. NVFP4 does not show the same benefit; the small tree
selects slightly faster configurations despite its many ties, which does not
prove the ordering is well learned. JIT-region portfolio/ordinary ratio at 64
is 0.99956 with balanced-pair bootstrap timing interval [0.99818,1.00224], so
those selections are not clearly distinguishable in this measurement.

This remains one seed and one additional M,K geometry across three families;
do not bootstrap families sharing geometry as independent shape replicates.
No complete-pool reference was collected for the expansion contexts, and all
numbers here compare selected configurations rather than absolute regret.
The rule model's largest tie groups were 355/401, 383/445 and 489/515 for
MXFP8/block/JIT respectively. Defaults remain unchanged and these observations
remain excluded from prior training. Further work needs independent shapes,
replicates, candidate-pool references and integrated first-hit startup costs,
not promotion based on the favorable MXFP8 slice alone.

```bash
.venv-research/bin/python -m tools.research.conditional_heuristics \
  --dataset autotune_reports/evidence_dataset_v1 \
  --study autotune_reports/portfolio_study_v2 \
  --output autotune_reports/conditional_heuristics_NEW

.venv-research/bin/python -m tools.research.prospective_campaign prepare \
  --dataset autotune_reports/evidence_dataset_v1 \
  --study autotune_reports/portfolio_study_v2 \
  --sources autotune_reports/evidence_sources_v1.json \
  --exclude-plan autotune_reports/portfolio_prospective_v1/plan.json \
  --exclude-plan autotune_reports/portfolio_guard_prospective_v1/plan.json \
  --output autotune_reports/protected_pilot_NEW
.venv/bin/python -m tools.research.prospective_campaign run \
  --output autotune_reports/protected_pilot_NEW --pilot --max-tasks 8
.venv/bin/python -m tools.research.prospective_campaign compare \
  --output autotune_reports/protected_pilot_NEW
.venv/bin/python -m tools.research.summarize_prospective \
  --output autotune_reports/protected_pilot_NEW
```

Use fresh output directories. Runtime/driver changes invalidate frozen plans.
The inspected pilot is now development data, not an unseen qualification set.
# Current-runtime NVFP4 numerical admission (2026-09-14)

The public first-hit probe and paired-confirmation harness now support NVFP4
block (`nvfp4_dynamic_fwd`) and JIT-row-region in addition to MXFP8. Their
numerical policies remain explicit, and comparison matrices reject mixed
families or variants. Twelve focused CPU tests pass. No production default or
numerical gate changed.

The frozen `2304×1536×2176` study stopped at its first block-scaling numerical
gate: normalized RMS was **21.0720%**, above the predeclared 20% limit. Both
the selected schedule and default block schedule exactly matched an independent
TorchAO `NVFP4Tensor.to_nvfp4(..., per_tensor_scale=1,
use_triton_kernel=False)` dequantize→FP32 GEMM→BF16 reference (zero difference).
This localizes the failure to block quantization rather than an autotuned GEMM
schedule. Weight RMS was 0.0124026; independently quantized X/W normalized RMS
errors were 9.5187%/18.7973%. Tensor-current, JIT-region and tensor-delayed each
produced 13.8631% forward error on identical source tensors.

Evidence: `autotune_reports/runtime_nvfp4_quality_20260914_v1/`, including the
failed log, raw eight-trial residual, selected schedule, frozen plan, and
`localization.json` with source snapshot. Reproduce localization with
`python -m tools.research.nvfp4_block_failure --winner <saved-runtime-winner.json>
--output <new-localization.json>`.

The failed block case is not a performance admission or a reason to relax the
gate. Block scaling remains a distribution-dependent policy; schedule tuning
cannot restore precision lost during quantization. The subsequent JIT-only
study preserves the same shape, initialization and inputs, and retains the
block failure rather than changing the workload to obtain a passing score.
# Current-runtime public JIT-region quality result (2026-09-14)

On `2304×1536×2176`, the full public `torch.compile(fullgraph=True)` inference
study completed ordinary/retrieval/ML × 8/16/32/64 cumulative trials using the
new runtime L2 projection and search-space identities. All 192 search outcomes
were successful; all 12 frontend cells had one compiled graph, no repeat tuning
or capture-induced recompilation, and independent forward normalized RMS
13.8631–13.9395%. This is one geometry/seed on the local RTX 5070 Ti, not a
generalization or training-quality claim.

Eight distinct selected configurations were independently validated and compared
in ten balanced paired rounds. Median graph-replay latency (µs):

| Budget | Ordinary | Retrieval | Portfolio + ML |
|---|---:|---:|---:|
| 8 | 57.421 | 57.421 | 57.398 |
| 16 | 55.348 | 55.348 | 57.398 |
| 32 | 55.288 | 55.124 | 55.348 |
| 64 | 53.288 | 53.318 | 54.394 |

Ordinary search's 64-trial winner is about 7.2% lower latency than its 8-trial
winner. Retrieval has no material consistent gain here. ML is 3.71% slower at
16 trials (paired timing 95% interval +3.66–3.73%) and 2.08% slower at 64
(+1.21–2.73%). Its apparent unpaired 8-trial advantage did not reproduce as a
material paired improvement. The intervals concern timing uncertainty within
this context, not policy generalization. Do not promote either prior from this
result.

First real compiled-call/refinement wall times in seconds for budgets 8/16/32/64:
ordinary 5.92/7.11/12.19/20.63; retrieval 7.02/9.08/13.12/22.75; ML
7.42/7.89/14.98/21.10. Each refinement ran in a fresh process, reusing its
mode's accumulated trials; these exclude process startup and do not clear
compiler disk caches. Summed raw search-evaluator wall time through 64 was
35.84/38.77/38.27 seconds, of which recorded `compile_ms` totals were
35.07/38.00/37.50 seconds. These fields overlap: do not add them to first-call
wall time, and do not mistake them for a complete compiler/model/confirmation
startup attribution.

Artifacts: `autotune_reports/runtime_nvfp4_jit_quality_20260914_v1/` includes
the frozen plan, source snapshots, per-budget JSON/logs, durable residuals and
`paired/{plan,confirmation,summary}.json` with raw paired measurements.
Reproduce the independent comparison using
`python -m tools.research.runtime_prior_compare --root <study-root>
--output <new-paired-output>` while its recorded execution sources still match.
# Session-scoped first-hit cost accounting (2026-09-14)

`python -m tools.research.first_hit_costs --root <public-probe-root>
--output <new-costs.json>` now separates new-session search evaluation and paired
finalist evaluation. It reports recorded compile/build times as nested subsets,
not additive costs, counts failed trials separately, excludes prior sessions,
and does not count confirmation copies embedded in winner decisions twice.
Missing compile fields are counted, rather than represented as complete zero
cost. Incomplete/corrupt JSONL tails, duplicate observations, mixed contexts and
impossible timing totals fail closed. Eleven CPU tests passed across this
accounting and the public comparison helpers.

The JIT-region study's report is
`autotune_reports/runtime_nvfp4_jit_quality_20260914_v1/costs_v2.json`.
Search evaluator wall time was 3.76–18.48 seconds per first-call/refinement
session. Finalist evaluator time was usually ~0.03 seconds, reaching
0.66–0.69 seconds in several sessions. Observed proposal time was 0.018–0.297
seconds. This is a lower bound that includes ranking when called, not an
isolated model inference measurement. The 2.10–3.28-second unclassified remainder
is an upper bound and is **not** labeled as model loading, frontend compilation
or Python overhead. Those three fields remain explicitly unknown. Recorded
`compile_ms` may include runner construction and synchronization; it is not an
NVVM-only profiler measurement.

The report hashes its source documents/journals and analyzer. It reads existing
evidence only and does not rerun kernels, rewrite residuals or promote winners.
Full disjoint model-loading/frontend-compilation/setup instrumentation remains
necessary before assigning the unclassified cost to a specific optimization.
# Direct startup tracing and unused-reference cleanup (2026-09-14)

The public probe accepts `--startup-trace` for opt-in research-only wall-time
spans. It measures first call, frontend compilation, NVFP4 harness construction,
tuner construction/run, predictor loading, retrieval and prediction. Same-thread
parent/child spans record inclusive and self time; inclusive values must not be
summed. Hooks restore on exceptions and are never installed by production
launchers. Hook imports prewarm Python modules, so these are diagnostic timings,
not fully cold process-start measurements.

On the qualified JIT-region `2304×1536×2176`, eight-trial ML startup initially
showed harness construction 1.496 s, predictor loading 0.486 s, prediction
0.035 s and frontend compilation 0.270 s. Constructor inspection found that
JIT-region inherited a compiled quantized reference and then discarded it for
its existing FP32-GEMM→BF16 oracle. Reference construction is now dispatched
directly to that same oracle. A regression forbids the discarded runner call;
block/delayed keep their own reference contracts and no validation gate changed.

**This did not demonstrate a substantial first-hit speedup.** In fresh-process
diagnostic samples, ordinary total first-call time was 5.808→5.789 s and ML was
6.839→6.977 s. Most of the former setup compilation moved into the initial
candidate evaluation, where it is needed anyway. Do not claim the 1.496-second
harness span as an end-to-end saving. The improvement is eliminating an unused
reference execution and accurately locating first-hit cost; candidate
compilation remains the main measured cost.

Validation: 18 CPU tests passed, followed by seven NVFP4 CUDA refinement tests
and two custom-policy subtests in 36.89 seconds. Four diagnostic probe runs
(before/after × ordinary/ML) passed independent forward numerics at 13.8631%,
one-graph/repeated-call invariants and graph replay. Artifacts:
`autotune_reports/startup_trace_20260914_v{1,2}/` and
`/tmp/rtx_reference_setup_gpu_20260914.xml`. Source snapshots preserve both
versions. This is one shape and unpaired startup timing, not a universal
latency improvement or a reason to promote a learned prior.
# Runtime-projected matched MXFP8 study (2026-09-14)

`autotune_reports/matched_runtime_20260914_v1` completed the frozen MXFP8
M=1792, N=2048, K=1920, seed-31415 comparison on the local RTX 5070 Ti.
All four treatments share 396 legal runtime-projected candidates and the same
protected initial neighborhood. The 256 cumulative search observations and
all 396 full-pool reference observations passed their numerical gates; the
independent initial FP32-GEMM→BF16 normalized RMS error was 3.7417%.
The completed audit reports no inventory, prefix, resume, identity or paired
winner-decision issues. This is a finite-pool experiment, not unrestricted
public first-hit search or proof of a global optimum.

Paired latency (µs), followed by excess relative to the selected paired reference:

| Budget | Ordinary | Portfolio | Rules | Portfolio + ML |
|---|---:|---:|---:|---:|
| 8 | 65.884 / +7.60% | same | same | same |
| 16 | 65.164 / +6.52% | same | same | same |
| 32 | 65.164 / +6.52% | same | 64.886 / +6.01% | 61.316 / +0.16% |
| 64 | 61.203 / reference | same | same | 61.316 / +0.16% |

The 95% paired timing interval for ML's reference excess is +0.024–0.308%;
ordinary at 32 trials is +6.13–6.61%. These are within-context timing intervals,
not policy-generalization intervals. ML materially helped at 32 trials here;
neither retrieval nor rules fixed the poor 8/16-trial result. Combined with the
current JIT-region regressions, this supports further family/budget-specific
evaluation, **not** a universal prior promotion.

The independent full-pool screening contains nine configurations within 2% and
22 within 5% of its best screened latency. Uniform sampling without replacement
(2,000 simulated draws) reaches within 5% with probability
36.55%/57.8%/85.65%/98.45% at 8/16/32/64. These simulations use complete
screened observations, not fabricated missing labels or paired-generalization
claims. Raw search evaluator totals through 64 trials were 22.29/22.12/20.21/
18.27 seconds for ordinary/portfolio/rules/ML; recorded nested compile totals
were 17.28/17.12/15.23/13.29 seconds. Ranking setup was 0.798 seconds. These are
not full public first-call latency measurements.

The first paired launch failed before measurement because a missing
`source_digest` import affected provenance metadata. The preserved
`paired.log` and `paired_import_repair.json` document a metadata-only `runpy`
global injection that completed the frozen-source comparison in
`paired_repaired/`. No kernels, candidate choices or existing measurements were
modified. The import is now fixed in source; a regression first reproduced its
absence, then the 14-test matched comparison/audit/pool suite passed.
Future experiments require a fresh source snapshot rather than bypassing the
old snapshot's source-change guard.

## Structural-prefix ablation: no benefit on a fresh geometry

`autotune_reports/matched_prefix_20260914_v1` compares the original axis-major
protected prefix with the existing library's structural-interleaved coverage
planner. Both freeze two neighbors, then use unchanged search machinery. This
is a research-only change; the production policy is untouched.

The local RTX 5070 Ti experiment used MXFP8 prequant forward, M=2816, N=2560,
K=1792, seed 42424. The M/K pair was absent from the frozen list of previously
observed geometries, and rankings were frozen before timing. Both arms had exactly
the same 393 legal configurations and identical complete rankings. Each arm
ran ordinary, portfolio, rules and portfolio+ML at cumulative 8/16/32/64 trials.
Treatment/prefix execution order alternated between waves. This is one
geometry/seed/device, not cross-context generalization evidence.

All 512 search outcomes and both 393-candidate reference passes succeeded.
Both integrity audits were complete with no issues. Each arm completed ten
balanced confirmation rounds; the final cross-prefix measurement used ten
balanced rounds over the four distinct selected winners (40 measurements).
Every selected winner separately passed FP32 GEMM cast to BF16, with normalized
RMS error 3.741024% on the fixed test inputs and finite outputs. Gates were not
relaxed. This validates selected forward configurations, not backward or a
distribution-wide numerical guarantee.

The result is an exact selection tie in **all 16 treatment/budget cells**:
changing the two protected neighbors did not change any final configuration.
Cross-prefix ratios are therefore exactly one by shared configuration identity,
not an inference that two separately measured noisy latencies are equivalent.

Within the original-prefix arm, paired reference excess was approximately
7.33% at both 8 and 16 trials for every treatment. ML reached the measured
reference at trial budget 32, while the other treatments remained 7.33% behind.
At 64, portfolio and ML selected the reference, rules was about 0.07% behind,
and ordinary was about 5.74% behind. The interleaved arm selected those same
configurations. These are gaps against a measured finite-pool reference, not
theoretical optimum or BF16 speedups.

Search evaluator time totaled 81.37 seconds for axis-major and 81.86 seconds
for interleaved across their four treatments. Recorded compilation/build time
was 61.18 and 61.66 seconds respectively, nested inside evaluator time. These
totals exclude startup, ranking setup, reference enumeration and confirmation;
they are not end-user first-call latency estimates.

Decision: do not promote the diversified prefix. The low-budget issue remains
proposal coverage: ML's first candidate within 5% appeared at trial 21 in both
arms. Next investigate early retrieval/ranking exposure and candidate diversity,
then freeze a new independent evaluation set for any derived policy. Keep this
inspected case as development evidence. Plans, commands, raw journals, source
snapshots, numerical metrics, audits and paired summaries are preserved in the
experiment directory.

### Ranking quality versus scheduler exposure

`matched_opportunities` now reports both the best full-reference outcome in
the first 1/2/4/8/16 frozen ranking positions and the actual search trial at
which each leading candidate was evaluated. These are screening diagnostics,
not adaptive-search simulations: rank-only probe counts exclude initial,
protected and exploration costs. Failed reference candidates consume rank
positions and never receive imputed latency labels. Tests reject incomplete
rankings, duplicate observations and invalid probe counts.

The preserved `rank_exposure.json` reports for `matched_runtime_20260914_v1`
and `matched_prefix_20260914_v1/axis_major` show portfolio+ML's first within-5%
candidate at rank 6, measured at trial 20 and 21 respectively. The first four
ranked candidates could only reach about 7.1% reference excess; the first eight
could reach 0.19% and 0.28%. Three poor predictions preceded those strong
candidates. The first two positions were explicitly prepended historical
retrieval candidates. This motivates separately testing that forced retrieval
prefix and early model allocation, not promoting aggressive ML allocation
without another controlled evaluation. Both geometries are now development
evidence for such a policy and cannot serve as its held-out evaluation.

The next research comparison can independently set preparation-time
`--ranked-opening 0` or `4` and ranking-freeze `--retrieval-prefix 2` or `0`.
The execution command reads the opening from the frozen pool manifest; the
ranking artifact contains the selected retrieval policy. Four novel opening
candidates plus the initial candidate and two protected neighbors leave one
random trial within budget eight. The same opening allocation applies to each
treatment, including seeded ordinary ranking. This is an explicit matched
ablation, not a change to `AutotunePolicy` defaults or public first-hit behavior.
The CPU suite passed 24 tests and 8 subtests; the separate research-environment
ranking test passed, including pure-model ordering without retrieval and
unchanged rules/ordinary rankings.

## Retrieval/opening factorial: baseline already strong on a fresh shape

`autotune_reports/matched_opening_20260914_v1` completed the 2x2 ablation on
the local RTX 5070 Ti, MXFP8 prequant forward M=3328/N=3072/K=2176, seed 51515.
The four cells use ranked opening 0/4 and forced retrieval prefix 2/0. Candidate
pools (389 configurations), protected neighbors and numerical policy were
identical. Rankings and source snapshots were frozen before timing. Opening
changes did not alter the rankings; retrieval changes only altered prepending.

All 1,024 search trials passed. All four search audits were clean, and a
separate first-eight journal audit verified protected neighbors, unique
attempts and reserved random exploration for all 16 first-hit cells. The
four-ranked opening used exactly initial + two protected + four ranked + one
random at eight trials. One complete 389-candidate reference was shared across
cells via documented relative symlinks; it is not four independent references.

Each cell completed paired winner/reference confirmation. The final shared
cross-factorial comparison measured six distinct selected configurations over
ten balanced rounds (60 measurements). Before that timing, all six passed
independent FP32 GEMM cast to BF16 checks on seeds 51515, 42 and 43. All 18
checks were finite, with normalized RMS 3.7384--3.7435%, below the unchanged
7% gate. Extra numerical seeds are not extra performance replicates.

Shared paired latency change for the **ML treatment**, relative to its
unchanged opening-zero/retrieval-two baseline (negative means faster):

| Variant | Budget 8 | Budgets 16/32/64 |
|---|---:|---:|
| Opening 0, retrieval 0 | +0.578% [0.477%, 1.056%] | -0.063% [-0.132%, 0.009%] |
| Opening 4, retrieval 0 | -0.037% [-0.200%, 0.004%] | same selected configuration and interval |
| Opening 4, retrieval 2 | exact selection tie | exact selection tie |

Intervals are balanced-block bootstrap timing intervals for this single
context, not confidence in generalization across shapes. The baseline was
already approximately 0.24% behind its measured reference at eight trials,
so this geometry offers little headroom. The slight apparent gain from the
four-ranked/no-retrieval option is not established by the paired evidence.
Removing retrieval without extra ranked opening caused a small measured
regression at eight trials. Ordinary search selected the same eight-trial
configuration across variants. Rules and portfolio likewise do not provide
evidence for a universal switch.

Search evaluator totals were 91.33 seconds for opening0/retrieval2, 91.66 for
opening0/retrieval0, 93.41 for opening4/retrieval2 and 92.26 for
opening4/retrieval0, each covering four 64-trial treatments. These exclude
process startup, reference enumeration and confirmation. Raw nested build/
compile costs remain in each `audit.json`; do not add them to evaluator totals.

Decision: retain existing defaults and keep both controls research-only.
The experiment shows why evidence from the two earlier, weaker-baseline
geometries cannot justify removing retrieval globally. Wider family/shape
qualification and current public first-hit cost tests remain required before
promotion. Plans, driver source, exact commands, immutable snapshots, journals,
allocation/integrity audits and cross-factorial metrics are preserved together.
## Completed matched NVFP4 JIT-region case (2026-09-14)

`autotune_reports/matched_nvfp4_jit_20260914_v1/` extends the matched study to
NVFP4 JIT-region on local RTX 5070 Ti, M1920/N3072/K1664, seed 71717. The
historical predictor and rules were unchanged; pool/rankings/source snapshot
were frozen before candidate timing. All four treatments shared 507 candidates
and the same protected opening. No evaluation data entered training and no
runtime winner or policy was installed.

All 256 search trials and all 507 reference candidates passed. The audit found
no identity, prefix, JSONL-tail, resume-increment or paired-decision issues.
Ten distinct selected/reference-leading configurations passed three independent
seeds (30 checks): normalized RMS 13.4384–13.4540% against FP32 GEMM cast to
BF16, below the unchanged 20% gate. Ten balanced rounds then produced 100
separate paired measurements.

Paired latency overhead relative to the best remeasured top-five reference
leader (finite pool, not a proven global optimum):

| Trials | Ordinary | Portfolio | Rules | Portfolio + ML |
|---:|---:|---:|---:|---:|
| 8 | +20.33% | +3.72% | +28.13% | +3.72% |
| 16 | +10.74% | +3.72% | +20.86% | +1.44% |
| 32 | +7.83% | +3.72% | +7.83% | +1.00% |
| 64 | +1.44% | +1.44% | +7.83% | +1.00% |

At eight trials, portfolio and portfolio+ML select the **same** configuration:
the 13.99% paired latency reduction versus ordinary search belongs to retrieval,
not incremental ML ranking. At 16 trials, ML's selected configuration is the
same one ordinary search reaches at 64. ML reduces latency versus the matching
ordinary budget by 8.45% at 16, 6.34% at 32, and 0.61% at 64. Its 16-trial
paired block-bootstrap interval is 8.41–8.54% faster. These intervals describe
timing uncertainty in this context, not cross-context policy generalization.

Rules discriminate poorly here: 466/507 candidates share one score. This is a
diagnostic limitation of the frozen model, not evidence that those candidates
should be pruned or a reason to retrain on this evaluation case.

Cost accounting is preserved separately in `analysis.json`. Across the four
progressive calls per treatment, ordinary/portfolio/rules/ML worker times total
57.93/63.06/59.84/65.04 seconds. Recorded compilation totals are
38.47/39.93/40.82/41.06 seconds **within**, not in addition to, search evaluation.
Their confirmation work adds 3.34/7.09/2.99/7.95 seconds separately from search
observations. Ranking setup took 0.928 seconds; reference evaluation 302.51
seconds; final paired evaluation 13.80 seconds. Worker timers exclude interpreter
startup/import and setup before their timer; these are not complete public
first-hit latency claims. Shared compiler caches were not purged between cells.

Reproduction commands, frozen identities and the exclusive-output numerical
driver are in the study README. `analysis.json` hashes its input evidence and
also reports uniform sampling from the screened finite pool, separately from
paired comparisons. This positive case supports keeping the portfolio/ML path
available for further evaluation; one geometry/seed does not justify a universal
default, tail guarantee, or SKU transfer claim.
# Revision-23 backward held-out study (2026-09-14)

**Timing quarantine:** both cases finished, but the subsequent quality audit
found explicitly nonstationary confirmation measurements (nine in each final
paired comparison, plus search-time confirmations). The descriptive numbers
below are retained, not admitted policy-promotion evidence. Numerical checks
passed. See the immutable `timing_quality_audit.json` sidecar and audit script
in the study root; no original record has been modified.

Case 1 (M6144/N512/K896) completed 256 search trials, 1,802 reference candidates,
and 18 independent dX/dW checks. Ordinary selected the best paired reference
candidate at every budget. Portfolio selected it except at 16 trials, where
the selected alternative was 34.12% slower in the final comparison. Inspection
of that decision shows a challenger/incumbent timing ratio below one in one
round and above one in the other, with an explicitly nonstationary incumbent
measurement. The current promotion code accepts the median ratio without
checking that quality flag. This is a confirmation robustness defect, not
evidence that the candidate space should be pruned or numerical gates relaxed.
Fix stability handling and preserve the incumbent on inconclusive evidence
before collecting another policy comparison.

Artifacts: `autotune_reports/matched_backward_20260914_v1/`. The two evaluation
geometries were reserved before collecting the new revision-23 calibration
data. The fitted prior uses three calibration geometries; a fourth checks
export parity only. Neither evaluation geometry was admitted to training.
All methods retain the same frozen candidate pool, protected initial neighbors,
ranked opening zero, retrieval prefix two, and progressive 8/16/32/64 budgets.

Case 0 (M3072/N896/K640) completed 256 successful search evaluations and its
1,806-candidate reference sweep. The journal/prefix audit and three-seed
independent dX/dW finalist checks passed. Ten balanced paired rounds gave:

| Trials | Ordinary, us | Portfolio, us | Rules, us | Portfolio+ML, us |
|---|---:|---:|---:|---:|
| 8 | 51.707 | 51.679 | 51.679 | 51.679 |
| 16 | 51.288 | 43.746 | 51.679 | 43.746 |
| 32 | 51.288 | 43.746 | 51.679 | 43.746 |
| 64 | 51.288 | 43.746 | 51.679 | 43.746 |

At 16+ trials portfolio and portfolio+ML select the exact same configuration:
14.71% lower paired latency than ordinary, with a timing-bootstrap interval
of 13.37–14.80% lower latency. Their gap to the best paired top-five screened
reference leader is 1.09%; this is not proof of a global optimum. At eight
trials the portfolio difference is indistinguishable from timing noise.
Rules remain 0.76% slower than ordinary at 16+ trials. This supports retrieval
on this context, not an incremental ML benefit or a transferable tail bound.

`case0/analysis.json` records per-budget search, compiler, confirmation and
worker costs plus input hashes; compiler time is included in search time,
not additive. Ranking setup is separate (2.94 seconds). No runtime prior or
winner is promoted. The quality quarantine above supersedes promotion-related
interpretations of these raw descriptive timing ratios.
