# First-hit autotuning goal audit

Audited against the current dirty worktree on 2026-09-14. The final
[six-requirement acceptance audit](first_hit_completion_audit.md) supersedes
the earlier active/pending notes preserved below. This is not a blanket release
declaration: no universal learned prior is promoted, and no commit/push is
authorized by this audit.

### Latest local integration evidence

**Engineering wrap-up:** the final current-code Torch 2.13 GPU/compiler/cache
matrix passed 97 tests and 42 subtests in 511.52s, zero failures/errors/skips
(`/tmp/rtx_first_hit_full_torch213_v3.xml`). Final focused Torch 2.12.1 GPU
compatibility passed 19 tests in 71.27s, zero failures/skips
(`/tmp/rtx_wrapup_torch212_gpu_v1.xml`). CPU compatibility passed 23 tests and
10 subtests; progressive/search/audit checks passed 38 tests and 11 subtests.
Numerical thresholds were not relaxed. No further dataset collection is
planned for this wrap-up and no commit/push was performed.

User-facing API setup, cache behavior, effort caps, measurement boundaries,
and the two-shape reference results are consolidated in
`first_hit_autotuning.md`. Broadly generalizing learned priors and reliable
population tail-regret remain unproven; no universal model/default promotion
is claimed. The implementation/qualification handoff is ready, while that
broader quality objective remains an explicit limitation rather than a hidden
release guarantee.

Both graph-mode full-pool references and paired comparisons are complete.
All 3,607 screened candidates passed, with zero explicitly unstable screening
measurements. Each shape's ten-round, five-candidate paired comparison also
had no invalid or unstable candidates (100 paired measurements total).
The shared 8-trial winner matched the best paired top-five reference on both
shapes: 62.014 us and 113.120 us. Consequently every method and budget has an
exact configuration tie to that finite reference; ML and extra trials showed
no selection benefit. Independent winner checks passed on three seeds each.
Read-only, hashed aggregation: `autotune_reports/matched_backward_graph_20260914_v1/oracle_analysis.json`.
Reference evaluation costs were 285.84/291.01s, including compile costs of
138.69/136.82s and capture costs of 7.89/8.72s; paired checks cost 5.74/5.93s.
These are not additive cost categories. This is evidence for two local
backward cases only, not a universal stopping rule or cross-device prior.
The final current-code integration matrix is the next milestone; priors stay
opt-in and no runtime-policy default is promoted.

First graph-mode full-pool reference is complete: all 1,798 candidates for
M/N/K 3584/768/1152 passed. Ten balanced confirmation rounds over the screened
top five selected the same `8f041a7821c9f584c459` configuration already chosen
at 8 trials by every method. Its paired median was 62.014 us, with no invalid
or explicitly unstable confirmation candidates. Thus 8/16/32/64 have an exact
selection tie with the best paired top-five reference in this context, not
a proven global optimum. Source: `matched_backward_graph_20260914_v1/case0/`
under `autotune_reports`, `reference/result.json` and
`oracle-paired/{confirmation,summary}.json`. The second full-pool reference is
still running. Do not generalize this single-shape result to all effort levels
or SKUs; no policy is promoted.

Graph-mode backward search completed on M/N/K 3584/768/1152 and
7168/640/1024 using frozen host-trained priors: 512/512 trials valid,
zero explicitly unstable search measurements, clean budget/identity audits.
All four methods at all four budgets selected the identical config
`8f041a7821c9f584c459` on both shapes. This is an exact selection tie, not a
fabricated paired timing interval. Each selected winner passed independent
dX/dW checks on three seeds (six checks total). There is no demonstrated ML
benefit; 64-trial ML search cost 20.58/22.30s versus ordinary 15.15/14.42s,
mostly additional compilation. These include warm/shared compiler cache
effects and exclude interpreter startup; confirmation costs are separate.
Artifacts: `autotune_reports/matched_backward_graph_20260914_v1/analysis.json`.
Full-pool screening plus top-reference paired comparison is now the next
step; exact method ties do not prove optimality. No policy is promoted.

Calibrated dataset mode wiring is implemented: MXFP8 backward jobs accept
`tags.backward_measurement_mode="cuda_graph"`; omitted mode retains host
behavior and the old manifest format. Graph contexts include a measurement
revision, and mismatched job/harness modes are rejected. Matched-pool preparation
accepts `--backward-measurement-mode cuda_graph`; the frozen task carries it
through search/reference workers without reinterpreting existing timings.
Other families reject this option. Local CUDA qualification passed 9 tests in
3.38s, including actual research adapters, separate context IDs, independent
both-gradient admission and ragged/rotating replay:
`/tmp/rtx_backward_dataset_modes_gpu_v1.xml`. Mode-specific prospective quality
remains open; no performance study or policy promotion was performed here.

The low-overhead MXFP8 forward/prequant/backward evaluators now report timing
stability using the calibrated protocol's existing limits, without additional
GPU samples or threshold changes. Explicitly unstable samples remain raw
evidence but cannot win paired confirmation. CPU regression batch: 18 passed,
4 skipped; compiled refinement and backward-mode GPU batch: 8 passed in
63.94s (`/tmp/rtx_runtime_timing_quality_gpu_v1.xml`). Unknown quality metadata
from generic adapters is still accepted; this is not universal calibration.

The calibrated backward harness now accepts explicit `measurement_mode` of
`host` (unchanged default) or `cuda_graph`. Graph mode captures complete batches,
including rotating input addresses and stream joins, validates both replayed
gradients against the independent reference, and reports capture setup cost
outside timed events. Graphs belong to prepared candidates, keyed by batch
length and ring offset, not a process-global cache. Component diagnostics stay
host-measured and are labeled accordingly. Initial focused GPU validation:
5 passed (aligned/ragged, hot/rotate, capture reuse), in 3.15s, artifact
`/tmp/rtx_backward_harness_modes_v1.xml`. Dataset campaign/paired study identity
wiring and prospective graph-mode quality remain outstanding; existing data
has not been reinterpreted or recollected.

Opt-in backward measurement `auto` now resolves at compiled joint-backward
lowering from the requested `triton.cudagraphs` setting; eager/partial paths
remain host-measured. Resolved requests embed policy/cache metadata and survive
fresh-process registry loss. Six GPU mode tests passed; a targeted follow-up
also requires an actual Inductor/AOT disk-cache hit on the repeat processes
while tuning is forbidden (`/tmp/rtx_auto_backward_compiler_cache_v1.xml`).
This supersedes earlier automatic-inference-pending entries. Compiler capture
fallbacks, broader mode qualification and prospective mode-specific quality
remain unproven; the default mode is still host.

Graph/host backward cache coexistence now has fresh-process evidence. Four
compiled launches (host, graph, host, graph) retain one forward history and
distinct backward histories; the last two prohibit tuning. All numerical
checks pass, with 24 total observations. A pass-scoping fix prevents the
backward-only option from invalidating forward effort identity. Three GPU
tests passed in 23.69s; see `autotune_effort.md`. This verifies explicit mode
selection only; automatic compiler selection remains outstanding.

Opt-in linear backward measurement mode is implemented:
`AutotunePolicy(backward_measurement_mode='cuda_graph')`. It captures complete
batches, revalidates dX/dW, reports capture setup separately, and uses distinct
search/winner identities without legacy host-winner fallback. Default remains
host. Three focused CUDA tests passed, including compiled ragged 8→16
refinement; CPU contract batch passed 21 tests/12 subtests. Automatic compiler
mode selection, calibrated-dataset mode parity and prospective graph-mode
quality remain open. See `autotune_effort.md` for explicit scope.

The first post-qualification timing diagnosis is complete: fixed-candidate
host-fed versus graph-batched backward measurements and a narrow node-level
Nsight timeline. Graph replay stabilizes the incumbent but makes the challenger
slower by reducing dX/dW overlap (1,180 us host-fed versus 152 us batch-graph
overlap across 32 calls). See `backward_timing_diagnosis.md`. This rules out a
blind switch to graph timing as merely noise removal; execution mode must be
an explicit measurement/cache contract. Current production timing is unchanged.

**Completed local qualification milestone:** Torch 2.12.1+cu132 passed the
full corrected batch (78 tests/42 subtests, zero failures/skips, 486.71s).
Torch 2.13.0+cu132 completed with 78 passes and four NVFP4 cache-instrumentation
subfailures; the test-only suite-order fix then passed 3 tests/14 subtests.
See `autotune_qualification_matrix.md` for exact artifacts and scope. The
historical running/pending entries below are superseded. The goal is still
active: backward timing stability and prospective quality/tail evidence under
the corrected promotion rule remain unresolved. Do not expand model fitting
or promote policies before resolving that measurement issue.

Promotion revision 2 implements the instability fix described below: no
replacement on explicitly nonstationary timing or a reversal between paired
orders; old confirmed-winner metadata is rejected even in reuse mode. Raw
search journals and numerical gates are unchanged. Focused regression batch:
37 tests/14 subtests passed (`/tmp/rtx_promotion_stability_v2_cpu_final.xml`).
The v1 CUDA qualification was intentionally interrupted (4 tests/4 subtests
passed, not a full matrix). The corrected v2 batch is running serially across
both project Torch environments. Do not interpret the older 'currently ignores'
description below as the current implementation; it records the discovery.

Both revision-23 backward held-out studies have completed (512 search trials,
3,608 reference candidates, 45 independent finalist dX/dW checks). Numerical
checks and structural audits passed, but a separate timing-quality audit found
nonstationary confirmations. Case 1 exposed an accepted 16-trial promotion
whose winner later measured 34% slower. `paired_promotion.py` currently ignores
the measurement's stationarity flag when accepting median paired improvement.
The results are quarantined from policy promotion; raw records remain intact.
The next implementation priority is conservative confirmation under unstable
or order-sensitive timing, with focused regressions and no numerical changes.

The consolidated CUDA matrix is running serially on Torch 2.13.0 and 2.12.1
under `/tmp/rtx_first_hit_full_torch{213,212}_v1.{log,xml}`. These are not yet
passes. Do not modify its source contract mid-run or overlap performance work.

The chronological entries below retain their original limitations; some have
since been addressed. `docs/mlp_first_hit_plan_contract.md` records the current
MLP first-hit path (private selected-op revision 3, plan-spec revision 7),
caller-controlled automatic replay, external-capture fallback, and measured
small/decoder cost tradeoff. The current integration batch passed 32 tests and
two subtests in 86.45 seconds, including actual tuning/refinement, cache reuse,
ragged and partial gradients, opcheck, outstanding forwards/reverse backward,
and stable live allocation. This is scoped local Torch 2.13.0+cu132 evidence,
not complete software/device coverage or a claim that replay always wins.

The next evidence expansion is frozen at
`autotune_reports/matched_nvfp4_jit_20260914_v1/`: NVFP4 JIT-region,
M1920/N3072/K1664, seed 71717, common 507-candidate pool, unchanged historical
prior/rules, ordinary/portfolio/rules/portfolio+ML, and progressive 8/16/32/64
budgets. Protected prefixes, ranked opening zero and retrieval prefix two are
unchanged. Whole-(M,K) exclusions and rules' train-geometry checks precede
ranking freeze. Independent initial admission passed at 13.4384% normalized
RMS against FP32 GEMM cast to BF16 (unchanged 20% gate).

The study has completed: 256/256 search trials, 507/507 reference candidates,
a clean audit, 30 independent numerical checks over ten finalists, and 100
balanced paired measurements. Portfolio/ML select the same eight-trial winner;
at sixteen trials ML reaches the configuration ordinary search finds at 64.
Its paired gap to the finite-pool reference is 3.72/1.44/1.00/1.00% at
8/16/32/64 trials. Full results and separate costs are in
`autotune_protected_prior_results.md`. It admits no evaluation rows to training
and installs no runtime winners. One positive NVFP4 geometry/seed adds family
coverage but does not close the broader tail/generalization requirement.

## Requirements and evidence

Current backward calibration is frozen in
`autotune_reports/backward_calibration_20260914_v1/`: four revision-23 training
geometries, 128 candidates each (initial, 31 interleaved structural neighbors,
96 distinct legal random samples), visited in 32/64/128 waves. All four initial
independent dX/dW admissions passed at 3.736–3.746% normalized RMS. Held-out
M3072/N896/K640 and M6144/N512/K896 are reserved and have not been timed; their
(M,K) pairs were absent from the saved matched-study exclusion lists. Candidate
selection uses no old-revision scores or measured ranking. The shared residual
store and reference enumerator are reused, not another measurement engine.

Seven CPU tests passed, including exact 32/64/128 resumed-versus-uninterrupted
candidate order, deterministic distinct pool construction, disjoint training/
held-out geometry and admission failure checks:
`/tmp/rtx_backward_calibration_plan_v2_20260914.xml`. The collection is not yet
admitted training evidence: integrity, numerical and timing-stability audits
must finish before fitting the prespecified model/rules. No runtime policy or
winner is installed, and no old record receives a new revision label.

Calibration has now completed: all 512 trials passed numerical evaluation,
and the collection audit checks exact frozen enumeration order and 32/64/128
increments. The unchanged timing-quality filter excluded 46 unstable records,
retaining 466. A stricter fit allocation (frozen before reading latency outcomes)
withholds calibration context 3 from fitting: 338 rows from contexts 0/1/2 fit
the fixed ExtraTrees interaction model and depth-4 rules; 128 rows from context 3
verify serialized prediction parity only. Both reserved matched-evaluation
geometries remain unmeasured. Original collection records and excluded timings
are preserved rather than rewritten.

The research artifact is
`autotune_reports/backward_calibration_20260914_v1/model/predictor.json.gz`
(SHA256 `07f0ff6274e0f323286c40f185f84d6a577be96abfe5270060324f8529d44e0b`),
with `rules.json.gz` alongside it. The 48-tree fixed model fit took 0.344 seconds
and uses 647 features; portable/sklearn predictions differed by at most
4.997e-16 over all 128 verification rows. This is serialization correctness,
not model-selection or held-out quality evidence. The model has only three
fitting geometries and one SKU; no default or transfer claim is justified yet.

The reserved backward evaluation is now frozen and running in
`autotune_reports/matched_backward_20260914_v1/`: M3072/N896/K640 (seed 91919)
and M6144/N512/K896 (seed 91920). Initial independent dX/dW admission passed
at 3.7413% and 3.7396% maximum normalized RMS. Their common candidate pools
contain 1,806 and 1,802 legal configurations, respectively; no latency pruning
was applied. Four treatments visit each 8/16/32/64 budget across both cases
before advancing. Full references, audits, independent three-seed selected-
winner checks and ten balanced timing rounds are queued afterward.

Both models/rankings are frozen before search; no model adjustment is allowed
using either evaluation case. The first case has only two rule scores (largest
tie: 1,793 candidates) and 209 ML scores (largest tie: 734), a prespecified-model
limitation to retain in the report, not evidence of quality or a pruning rule.
Completion and policy conclusions are still pending.

Matched backward admission is now implemented in `tools/research/matched_gpu.py`:
both dX and dW are checked independently against FP32 GEMMs cast to BF16, using
the unchanged 7% gate. Tests deliberately corrupt each gradient separately and
reject nonfinite references and shape mismatches. Aligned 128/128/128, ragged
17/97/35, and long-reduction 4096/256/128 (M/N/K) passed on local CUDA, with
normalized RMS 3.71–3.80%. The combined matched-study regression passed 21 tests
and four subtests, with one skip, in 12.00 seconds:
`/tmp/rtx_matched_backward_admission_gpu_20260914.xml`.

The saved predictor/rules still cover backward revision 22, whereas current
backward is revision 23. Admission support does not make that prior compatible.
Preparation now rejects an incompatible prior before compiling the independent
kernel check; five CPU tests passed in
`/tmp/rtx_matched_backward_admission_contract_20260914.xml`. A complete four-way
backward comparison therefore needs separately admitted current-revision
training/calibration evidence and a compatible frozen prior. Old observations
must not be relabeled, and new evaluation geometries must remain excluded from
training. This is an available next local task, not a claim of model readiness
or an external-machine blocker.

| Requirement | Evidence inspected | Remaining limitation |
|---|---|---|
| Progressive protected prefixes, exploration, resume | `rtx/autotune/effort.py`, `orchestrator.py`; `test_progressive_effort.py`, `test_progressive_retrieval_opening.py` | Deterministic equivalence assumes identical observation/cost evidence. Independently timed GPU runs need not select identical schedules. Progressive remains opt-in. |
| Paired durable promotion and numerical exclusion | `paired_promotion.py`, `confirmation.py`; `test_winner_promotion.py`, `test_paired_confirmation.py`; compiled refinement XML | Available when `finalist_rounds` is enabled; default remains zero. Numerical gates qualify tested distributions, not all inputs. |
| Effort-aware reuse/refinement and revision isolation | `winners.py`, `runtime_coordinates.py`; cache contract/process and search-identity regressions | Changed domains have separate residual identities. A sampler exhausting proposals does not certify the entire space; no special below-budget exhaustive-space admission is claimed. |
| Matched ordinary/portfolio/rules/ML quality | `matched_pool.py`, `matched_gpu.py`, `matched_rankings.py`; two completed current-contract MXFP8 geometries and the current public JIT-region study | Current matched evidence now exists after runtime L2 projection/search-identity changes, but low budgets still underperform. More families, contexts and replicates are needed; the public JIT study is not a common finite pool and lacks a rules arm. |
| Forward/backward families, policies, shapes, compilation | Progressive/packed/capture/process/concurrency/interruption/training CUDA tests and artifacts listed below; aligned compiled forward/backward eviction now passes across five policies | Broad focused linear coverage exists, not a complete Cartesian qualification. Shared-instance concurrent delayed streams and active-kernel/CUDA-fatal interruption remain unqualified. Unlimited compiler specializations and all ragged/partial-backward eviction cases are not covered. GLU/MLP explicit tuning and earlier numerical quarantine must not be represented as equivalent to qualified linear first-hit support. |
| Promotion supported by prospective quality, tails and startup cost | `autotune_protected_prior_results.md`; current JIT paired measurements, first-hit costs, startup traces | No universal learned-policy promotion is supported. More current-contract matched geometries/replicates and explicit tail/startup acceptance evidence are needed. |

## Verified regression artifacts

The audit's CPU command passed **49 tests and 18 subtests; eight CUDA-dependent
tests were skipped**, not counted as GPU passes:

```bash
.venv/bin/python -m pytest \
  tests/test_autotune_effort.py tests/test_progressive_effort.py \
  tests/test_progressive_retrieval_opening.py tests/test_effort_cache_contract.py \
  tests/test_effort_cache_process.py tests/test_winner_promotion.py \
  tests/test_paired_confirmation.py tests/test_mlp_quality_projection.py \
  tests/test_mlp_precision_api.py tests/test_mxfp8_mlp_frontend.py \
  -q --junitxml=/tmp/rtx_goal_audit_cpu_20260914.xml
```

Existing artifacts were checked for presence, and their scope was matched to
their test sources; this audit did not rerun them or certify every older artifact
against every subsequent source edit:

- `/tmp/rtx_search_space_identity_gpu_20260914.xml`: compiled refinement and packed/scaling integration after search-identity isolation.
- `/tmp/rtx_long_reduction_efforts_20260914.xml`: long-reduction 8/16/32/64 ladder, five linear policies.
- `/tmp/rtx_first_hit_optimizer_common_eval_20260914.xml`: 16-step optimizer integration, per-step independent gradients and common held-out evaluation.
- `/tmp/rtx_fresh_process_training_efforts64_gpu_20260914.xml`: fresh-process training reuse across the ladder, one small shape.
- `/tmp/rtx_concurrent_first_hit_gpu_20260914.xml`: separate-process same-context contention, not shared-module stream safety.
- `/tmp/rtx_interrupted_gpu_verified_candidate_20260914.xml`: kill after CUDA measurement and before publication, not during an active kernel.
- `/tmp/rtx_eager_compiled_eviction_qualified_20260914.xml`: combined eager/compiled eviction, two tests/nine subtests; 60 compiled Y/dX/dW numerical checks and five zero-growth live-memory series. Three aligned shapes, all five linear scaling policies, exactly 48 unchanged forward/backward observations and three graphs per compiled policy. This run was executed after the matched studies, not merely checked for artifact presence.

The current JIT-region quality study retains 192 successful search outcomes and
80 paired measurements in `autotune_reports/runtime_nvfp4_jit_quality_20260914_v1`.
Ordinary search improved with effort; ML regressed at 16/64 trials. The separate
block numerical failure was preserved and localized to quantization by exact
TorchAO agreement. Neither case supports loosening a gate or silently changing
scaling policy.

## Next work, in priority order

Low-budget model diagnosis now has a structural association audit, not just
rank/exposure statistics. `tools/research/matched_structure.py` checks complete
unique pool/reference/ranking identities, rejects invalid successful timings,
keeps failed candidates in ranking counts, and reports conditional group
coverage without causal or pruning claims. Ten tests/13 subtests passed in
`/tmp/rtx_matched_structure_20260914.xml`. The report is
`autotune_reports/matched_transport_associations_20260914_v1.json` and hashes
its input files. On all three inspected MXFP8 geometries, every within-5%
configuration belongs to `(mma128, tma scale role, tma epilogue)`. In contrast,
`(row_major, consumers, direct)` occupies 3/3/2 of the first eight ML ranking
positions while its best measured candidate is 55.98%/62.97%/45.44% behind the
respective pool minimum. The first two rankings include the fixed retrieval
prefix; the third removes it. These are ranking positions, not trial numbers.
The inspected predictor contains the transport feature vocabulary (no missing
config features in the first pool), so simply adding those feature names is
not a fix. Candidate populations are heavily imbalanced/correlated and the
geometries share one device/family: no universal transport rule follows.

Next model work should inspect training support/calibration within these
transport groups and test a soft, context-conditioned correction or ranking
model on fresh contexts. Preserve exploration and the complete legal pool.
Merely assigning more early trials to the current ML ranking spends scarce
budget on systematically poor proposals in these inspected cases. These three
geometries remain development data and cannot validate that derived change.

Training-support follow-up: `transport_training_calibration_20260914_v1.json`
verifies the predictor's exact dataset-manifest hash and audits only its train
split. MXFP8 prequant forward has 4,902 rows, all kernel revision 9; none carry
the newer runtime-search-space tag (absence alone does not prove numerical
incompatibility). Row-major/consumer/direct has 1,164 rows across 116 contexts
and three devices, so global undersampling is not the explanation. However,
the relevant 5070 Ti/hot/ordinary stratum has only 33 rows across three contexts
for that group, versus 128 rows/six contexts for native/TMA/TMA. Its in-sample
context-median log residual is +0.00869, compared with −0.00783 for native/TMA.
That small training residual does not justify applying a 45–63% global penalty
to repair new-context ranking errors. The expansion stratum differs (+0.11927
for row-major/direct), further arguing for conditional evaluation. These are
in-sample residuals, not held-out quality evidence or causal transport effects.
`tools/research/transport_calibration.py` preserves device/shape-class/regime
strata and equal context weight; five tests/11 subtests passed in
`/tmp/rtx_transport_calibration_20260914.xml`. Next: compare training-versus-new
shape support and current feature values, then use shape-held-out calibration
or retraining; do not fit a constant penalty from the inspected pool outcomes.

Whole-(M,K) retraining screening completed in
`autotune_reports/transport_retraining_20260914_v1`: 24 fits, three folds of
historical train-only MXFP8 data, two model families, shape/interaction views,
weighted/unweighted fitting. All devices and output widths for a held-out M/K
stay out of its training fold; no inspected prospective pool is loaded.
Seven tests/15 subtests passed in `/tmp/rtx_transport_retraining_20260914.xml`.
Plans, split memberships, per-candidate predictions, fit/predict costs and
context-level metrics are retained. For portfolio-then-ranker screening:

| Model | Budget | Contexts | Mean observed-pool regret | Worst regret |
|---|---:|---:|---:|---:|
| ExtraTrees shape, unweighted | 8 | 175 | 1.361% | 25.831% |
| ExtraTrees interactions, unweighted | 8 | 175 | 0.708% | 11.320% |
| ExtraTrees shape, unweighted | 16 | 129 | 0.578% | 20.949% |
| ExtraTrees interactions, unweighted | 16 | 129 | 0.275% | 6.338% |

Context weighting helps the shape-only model but does not consistently improve
the interaction model. Interaction features include hardware and derived
features, so this does not isolate which feature block causes improvement.
The interaction ExtraTrees model is the next prospective candidate, not a
promoted default. These are historical observed-pool ranking budgets, excluding
protected-neighbor/exploration/compilation costs and numerical failures; they
are not adaptive first-hit trials. Dataset-level summary intervals compare
against random search, not direct pairwise model differences. Export equivalence,
new-context matched GPU quality and startup costs must be tested next.

The selected unweighted interaction ExtraTrees prior has now been fitted on
the 4,902 historical training MXFP8 rows and exported to
`autotune_reports/transport_prior_20260914_v1/predictor.json.gz` (SHA256
`812d128ed293e51db481ec61a5a3dfff3eba9d6845ad5110e1a20c652b85990f`).
Fit time was 1.93 seconds. Export comparison passes on 256 later historical
rows and all 1,178 configurations from the three inspected pools, maximum
absolute prediction difference 4.44e-16. The pool equivalence check loads no
timing labels and is not model-quality evaluation. NumPy prediction took
0.106–0.112 seconds per approximately 400-candidate pool, excluding load and
feature construction. Artifact remains deployment-disabled and the old prior
is unchanged. Runtime predictor tests passed four tests; the research-only
ranking test skipped there was executed successfully in the research venv
along with two exporter-scope tests. The exporter now restricts bundles to
families supported by the fitted model and rejects missing per-family training
or verification evidence. This prevents an MXFP8-only fit from advertising
unrelated family retrieval support. New-context matched GPU comparison is still
required before replacing any runtime prior.

That first prospective comparison is now complete in
`autotune_reports/matched_transport_prior_20260914_v1`: fresh
M2432/N2816/K2304, seed 61616, one RTX 5070 Ti. Old and interaction priors use
the identical 396-config pool, protected prefix, two retrieved opening
candidates and four search treatments. All 512 search and 396 reference
evaluations succeeded; both audits are clean. A joint ten-round confirmation
of three distinct winners followed six independent numerical checks (two seeds,
maximum normalized RMS 3.741581%). ML's 16-trial winner improves from 112.908
to 110.434 microseconds: paired ratio 0.978084, interval [0.977162, 0.980855].
At 8/32/64 trials the old/new ML winners are identical, as are all non-ML
treatment winners. The first within-5% frozen-ranking candidate is exposed at
trial 12 instead of 21. This comparison uses shared reference screening, with
joint timing in `cross_model/`; intervals describe this context's timing,
not cross-shape generalization. Total candidate evaluation time across each
side's four treatments is 83.282/83.881 seconds, excluding process startup,
full-pool reference and confirmation. The candidate remains research-only;
there is no observed eight-trial improvement here. Source and commands are
archived and both study processes exited successfully. Consolidated CPU
qualification also passed 53 tests/33 subtests in
`/tmp/rtx_first_hit_consolidated_cpu_20260914.xml`.

Runtime early-model allocation is now an explicit policy option:
`ranked_opening_trials=0/1/2`, default zero. Standard progressive allocation
preserves initial + two neighbors + two retrieved + up to two model proposals,
leaving trial eight for exploration. Duplicate/illegal ranks are skipped,
multiple heads are round-robin, and existing arm ranking/audit state is reused.
Disabled-option evidence identities remain unchanged; enabled values separate
checkpoint and winner evidence. Thirty-three focused CPU tests/14 subtests
passed, including the complete 8/16/32/64 resume ladder. Actual compiled MXFP8
M512/N384/K256 training with the exported interaction prior passes 8→16
refinement and Y/dX/dW gates, unchanged repeat-call journals/graphs, and stable
live allocation. The test explicitly verifies model proposals at positions
six/seven: `/tmp/rtx_ranked_opening_compiled_verified_20260914.xml` (13.36s).
This closes integration of the control, not its performance validation. The
previous prospective study used zero opening slots; a newly frozen comparison
must evaluate the changed allocation before recommending it by default.

That actual-runtime allocation experiment is complete in
`autotune_reports/runtime_ranked_opening_20260914_v1`: M2176/N3584/K2816 and
M4608/N1536/K1920, seed 71717, openings zero/two, 8/16/32/64 budgets, unchanged
interaction prior. All 256 search outcomes succeeded. Independent finalist
checks pass (maximum normalized RMS 3.743660%); joint confirmations use ten
balanced rounds per shape. First-eight journals prove the requested model
slots replace later UCB allocation, while preserving both local/retrieval
slots and exploration. There is no supported eight-trial improvement: first
shape +0.111% with interval spanning zero, second shape exact configuration
tie. First shape 32/64 also inconclusive; second shape 64 improves 0.462%
(0.349–0.755% faster timing interval). Keep zero default. This is actual
runtime-scheduler evidence, not finite-pool regret or process-startup timing.
The experiment falsifies the claim that simply forcing more early ML trials
fixes low-budget quality on these contexts. Prioritizing broader family/shape
qualification and better proposal quality is preferable to adding more
scheduler controls from this result.

Composite persistence gap fixed: GLU/MLP progressive explicit tuning now applies
runtime search-domain identity and policy-specific residual namespaces, matching
the linear effort path through a shared `effort_residual_root` helper. Budget
extensions retain their path; measurement/prior/search-domain changes do not
reuse incompatible journals. Legacy locations are unchanged. Thirty-two tests
and 12 subtests passed in `/tmp/rtx_composite_effort_namespace_20260914.xml`,
including mocked entrypoint checks for both frontends, budget extensions,
changed sampling, and changed coordinate domains. This is CPU-verified
persistence plumbing, not a new GLU/MLP GPU or automatic-first-hit qualification.
MLP backward derives execution keys during autograd setup, so automatic tuning
requires deliberate deferred forward/backward configuration ownership rather
than mutating the module during tracing. Its automatic path remains pending.

GLU opt-in first-hit is now implemented through the existing eager/Inductor
launcher boundary: `autotune='balanced'`, `tuning_policy`, and
`autotune_cache_dir`; default off is unchanged. Real-input misses resolve
within the configured backend, use the initial config, share runtime locks and
revisioned winner storage, and cache the resulting runner. Warm launches add
no selection or eager tensor operations. Cache-only mode skips searches;
explicit `autotune_` installs a config and disables implicit selection. The
registered operator schema/fake outputs remain unchanged. Both BF16 and packed
MXFP8 output paths pass compiled 8→16 refinement, eager equality, no repeat-call
retuning/recompilation, durable reuse after runner eviction, and warmed capture
replay. First-hit test shape M=K=H=128; independent FP32 projection/SiLU/product
errors 5.23–5.25% / 5.85–5.86% for BF16/MXFP8 outputs. Consolidated CUDA evidence
at `/tmp/rtx_glu_first_hit_integration_20260914.xml`: 16 tests, 275 subtests,
116.41 seconds. This adds GLU forward first-hit support, not standalone GLU
backward or recomputed-MLP automatic selection. Larger/CTA first-hit and
multi-process qualification remain pending.

Delayed shared-stream inspection found that bootstrap metadata locking ends
before runner launch and history publication; separate stream-local workspace
caches do not protect the shared history buffers. A new `DelayedHistoryExecution`
helper now wraps managed eager and Inductor delayed submissions from bootstrap
through publication; caller-managed functional histories remain caller-owned.
NVFP4 frontend revision is 10. Seventeen CPU tests/four subtests cover wrapper
scope, per-state submission exclusion, no events on warm same-stream calls,
exception cleanup and non-owning storage references. Real CUDA tests verified
pending-write handoff, handoff after partial enqueue failure, and five exact
same-stream graph replays. The combined CUDA ordering/capture/packed-refinement
regression passed 19 tests/four subtests in 68.85 seconds, recorded at
`/tmp/rtx_delayed_guard_capture_regression_20260914.xml`. Shared-module two-stream
forward overlap now passes in eager and compiled training, with gradients
requested after the overlapping forwards. All four history buffers and Y/dX/dW
match ordered execution exactly; 36 independent numerical gates pass, journals
remain at 16 unchanged observations and compiled graph count remains one.
`/tmp/rtx_delayed_shared_module_forward_overlap_20260914.xml` records the three
CUDA tests/four subtests. An isolated guard-disabled control failed final-history
equality in both modes (all 16 entries, maximum difference 0.185546875), preserved
in `autotune_reports/delayed_guard_negative_control_20260914.json`. This proves
the test detects a missing forward dependency rather than relying on autograd
synchronization. The subsequent two-host-thread extension passes eager and
compiled warmed forwards at the same shape: recorded acquisition order
2/0/3/1/5/4 in both modes, exact ordered histories and Y/dX/dW, 36 additional
independent numerical gates, unchanged 16-observation journals and one compiled
graph. `/tmp/rtx_delayed_host_threads_cuda_20260914.xml` records four tests/six
subtests passing in 10.88 seconds. CPU guard/bootstrap regression passed 11
tests/four subtests. Concurrent cold compilation/backward/optimizer updates and
multi-stream performance remain unqualified. Single-stream warm overhead is now
measured in `autotune_reports/delayed_guard_overhead_20260914_v1.json`:
three shapes, eager/compiled, 12 alternating pairs per ordinary/replay mode,
288 raw samples, six independent forward numerical gates and exact ablation
outputs. Median paired host cost is 3.52–3.99 µs; large-shape device work hides
that host cost and captured replay bypasses the guard. This is fixed-config
forward evidence, not end-to-end training or tuned-winner performance.
Python guards
cannot order graph replays that bypass Python; shared-history replay ordering
must remain an explicit caller contract unless separately implemented.

Alignment update: `matched_gpu.build_adapter` now applies the same runtime
coordinate projection for preparation, search, full-pool reference and paired
comparison. Preparation records canonical retrieval IDs; ranking freeze requires
them for projected contexts instead of intersecting incompatible historical IDs.
Sixteen runtime-environment CPU tests plus the research-environment ranking test
passed, including duplicate L2 projection and missing/foreign/duplicate retrieval
ID rejection.

The first aligned study completed in `autotune_reports/matched_runtime_20260914_v1`:
MXFP8 M=1792, N=2048, K=1920, seed 31415, 396 legal configurations and 32
retrieved configurations. The initial independent numerical gate passed at
3.7417% normalized RMS. Its updated exclusion list includes recently inspected
geometries; all four rankings were frozen before candidate timing. All 256
search and 396 reference outcomes passed, the audit found no issues, and 90
paired measurements completed. ML reached within 0.16% of the paired reference
at 32 trials; all methods were still 7.60%/6.52% behind at 8/16 trials.
See `autotune_protected_prior_results.md` for costs, limitations and the
documented metadata-import repair in paired confirmation. This closes the
first current-contract matched case, not the multi-context/tail evidence gap.

Proposal/selection diagnosis of this inspected case is recorded in
`matched_runtime_20260914_v1/opportunities.json`. At 16 trials, every method's
best successfully proposed candidate was still 6.43% behind the full-pool
screening minimum. Both protected neighbors used the same `layout_transport`
coordinate. ML first proposed a candidate within 5% at trial 20. This points to
low-budget proposal coverage as the next development target, rather than simply
increasing confirmation repetitions. These screening gaps are not the paired
timing estimates above. The new diagnostic rejects incomplete/duplicate pools
and prefixes and never treats a failed search candidate as eligible. Any policy
derived from these observations must be evaluated on new, untouched contexts;
this case cannot become its held-out evidence.

The matched research runner now exposes preparation-time
`--prefix-order structural_interleaved` as an explicit ablation. It reuses
`rtx.autotune.coverage.coverage_plan`, fixes two legal unique neighbors before
timing, and records their IDs in the manifest. Execution reads this choice from
the manifest, not from a new runtime override. All four treatments share that
prefix; their checkpoint/evidence identity prevents resuming an axis-major run
as an interleaved run. The existing `axis_major` default and production policy
are unchanged. CPU tests cover each treatment's uninterrupted 64-trial sequence
against resumed 8/16/32/64 prefixes, legality, coordinate diversity and policy
mismatch rejection. GPU quality and comparative benefit remain unproven for
this option. Next comparison must use newly frozen sources and geometry, not
modify the existing study artifacts.

That comparison has now completed in `matched_prefix_20260914_v1`: fresh
M2816/N2560/K1792, identical 393-candidate pools/rankings, 512 successful search
trials, two complete full-pool references, clean audits and shared paired
confirmation. All 16 treatment/budget cells selected identical configurations
across prefixes. Four distinct winners passed the independent forward gate
(3.741024% normalized RMS). This is evidence **against promoting this prefix
change as the low-budget fix**. The remaining target is early proposal/ranking
exposure; its first near-reference ML candidate still arrived at trial 21.
See `autotune_protected_prior_results.md` for scope and costs.

The rank/exposure diagnostic now separates frozen-ranking quality from the
trial at which the scheduler measured each candidate. On both inspected
current-contract geometries, portfolio+ML's first within-5% candidate was rank
6, reached at trial 20/21. Its ranks 3--5 were roughly 56--112% slower than the
reference, so increasing model allocation alone also increases exposure to
poor predictions. The first two positions were forced historical retrieval,
not the model's own top scores. Next controlled development ablation: separate
retrieval-prefix length from early ranking allocation; retain protected initial
neighbors, a reserved random trial and identical finite candidate pools for all
treatments. Do not describe rank-only prefix quality as simulated UCB quality,
and use a new evaluation geometry after choosing the ablation.

Research controls are implemented: `matched_gpu prepare --ranked-opening 4`
freezes four novel ranked candidates after the two protected neighbors, while
`matched_rankings --retrieval-prefix 0` removes forced retrieval prepending
without changing raw model scores or pool membership. Both defaults retain the
previous behavior (opening zero, retrieval prefix two). Opening is capped at
four so trial eight still has room for random exploration. Duplicate initial
or protected candidates are removed before counting the ranked quota; changed
openings receive incompatible evidence identities. CPU tests cover exact first
eight allocation and 8/16/32/64 resume equivalence for all four treatments,
illegal option rejection and independence of the retrieval control. The next
study is a 2x2 ablation (opening 0/4, retrieval 2/0), frozen before timing on a
new geometry, with common pools and protected neighbors. This control has not
yet demonstrated GPU quality or runtime startup benefit and is not promoted.

The 2x2 experiment has completed in `matched_opening_20260914_v1` on fresh
M3328/N3072/K2176. All 1,024 search trials and the shared 389-candidate reference
passed; four audits and the first-eight allocation audit were clean. Six
selected winners passed three independent numerical seeds (18 checks,
3.7384--3.7435% normalized RMS), followed by 60 shared paired measurements.
The baseline was already near the finite-pool reference at eight trials.
Removing retrieval without an early opening regressed ML eight-trial latency
by about 0.58%; the four-ranked/no-retrieval difference was only -0.04% with a
timing interval crossing zero. Neither control is promoted. This adds a third
current-contract MXFP8 evaluation geometry, not broader family or public
startup-cost qualification. Next prioritize remaining lifecycle qualification
and family/shape coverage rather than iterating only on this inspected case.

1. Align the matched research adapter with the runtime's executable coordinates.
   Test projection and retrieval-ID mapping so inactive L2 variants neither
   receive duplicate trial allocation nor erase retrieved candidates. Freeze
   a new manifest; do not relabel old measurements.
2. Freeze untouched geometries and rankings before timing. Run all four
   treatments on identical pools/protected prefixes at 8/16/32/64, obtain a
   measured finite-pool reference and paired confirmations, and audit complete
   trial inventories. Include startup/compile costs and failed numerical cases.
3. Repeat enough independent contexts/seeds to assess tails, not just a mean
   or a fortunate winner. Keep inspected outcomes out of model training for
   that evaluation; any future model revision requires a new evaluation set.
4. Close remaining lifecycle gaps with explicit scoped support or tests. Do
   not imply shared-module delayed concurrency from separate-process tests.
5. Consolidate the growing chronological evidence into a release-facing support
   matrix and rerun the relevant current-code qualification set before claiming
   completion. Publication/commit remains a separate authorization.

There is useful local work available now. Additional machines would strengthen
transfer claims later, but are not currently a blocker.
