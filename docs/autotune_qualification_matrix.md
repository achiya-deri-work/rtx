# First-hit autotuning qualification matrix

## Latest completed local milestone (2026-09-14)

**Final wrap-up gate:** current-code Torch 2.13.0+cu132 passed 97 tests and
42 subtests in 511.52s, with zero failures/errors/skips:
`/tmp/rtx_first_hit_full_torch213_v3.{log,xml}`. This supersedes the earlier
2.13 suite-order failure below. The final focused Torch 2.12.1+cu132 follow-up
passed 19 tests in 71.27s, zero failures/skips:
`/tmp/rtx_wrapup_torch212_gpu_v1.{log,xml}`. Its CPU compatibility batch passed
23 tests/10 subtests (five deliberate GPU skips). Current search-contract CPU
checks passed 38 tests/11 subtests (one GPU and one research-environment skip),
`/tmp/rtx_wrapup_search_contracts_v1.xml`.

The two graph-mode studies and full-pool references are complete, not running:
512 search trials plus 3,607 reference trials passed; 100 final paired
measurements were stable. Every method/budget selected the same configuration
as the best paired top-five reference on both local shapes. This supports
the tested small-budget path, not cross-device generalization or universal
tail-regret guarantees. Learned priors remain opt-in. Start with the
[handoff guide](first_hit_autotuning.md); lower entries are historical evidence.

Subsequent focused qualification added backward host/graph/auto runtime
measurement, fresh-process compiled-cache reuse, and calibrated dataset mode
identities. Latest combined dataset/harness GPU batch passed 9 tests in 3.38s;
CPU dataset/progressive regression passed 36 tests and 11 subtests. The runtime
timing-quality batch passed 8 CUDA tests in 63.94s. These are focused follow-ups,
not another full two-version matrix. See `autotune_goal_audit.md` for artifacts.
The timing diagnosis below is complete, and a new immutable graph-mode study
is running under `autotune_reports/matched_backward_graph_20260914_v1`.
Its models and rankings were fixed before candidate timing; it does not
rehabilitate or reinterpret the earlier quarantined host measurements.

Follow-up: backward runtime coordinate revision 1 canonicalizes the ignored
dual-stream `dw_first` setting to `dx_first`; single-stream orders remain
distinct. New context tags prevent sharing search/scheduler evidence with the
old duplicate space. Refinement requires matching coordinate-revision metadata,
while reuse may retain a numerically valid old winner. Raw datasets and kernel
revision are unchanged. CPU checks passed 26 tests/13 subtests (one CUDA skip),
then the focused CUDA compiled-refinement/coordinate batch passed 10 tests and
three subtests in 11.85s (`/tmp/rtx_backward_coordinate_gpu_v1.{log,xml}`). This
is a follow-up to the broad matrix below, not a new full-matrix run.

On the RTX 5070 Ti, the corrected promotion-revision-2 batch passed **78 tests
and 42 subtests** on Torch 2.12.1+cu132 in 486.71 seconds, with no failures or
skips: `/tmp/rtx_first_hit_full_torch212_v2.{log,xml}`. On Torch 2.13.0+cu132,
the full batch had 78 passes and four NVFP4 cache-test instrumentation failures;
the corrected suite-order reproduction then passed three tests and 14 subtests
in 57.30 seconds. Preserve that distinction rather than claiming a clean
uninterrupted 2.13 batch. Runtime code did not change for the test-only fix.

This covers the named local numerical, compiler, cache, history, interruption,
and replay contracts below. It does **not** close prospective performance
quality or tail-regret: the backward study's timing remains quarantined, and
new priors/progressive policy defaults remain opt-in. Next diagnose the
backward timing instability on fixed validated candidates before another
prospective comparison. No model or runtime winner has been promoted.

The entries below retain the execution sequence and intermediate results;
references to pending runs there are historical, superseded by this section.

This is the consolidated acceptance checklist, not a claim that every row has
passed on every supported environment. Existing detailed results remain in
`autotune_goal_audit.md`, `autotune_protected_prior_results.md`, and
`mlp_first_hit_plan_contract.md`. Trial budgets measure investment, not a
guarantee of a globally optimal kernel.

| Contract | Acceptance tests/evidence | Current boundary |
|---|---|---|
| Protected progressive proposals and exact resume | `test_progressive_effort.py`, `test_progressive_retrieval_opening.py`, `test_backward_calibration.py` | Exact replay assumes identical recorded evidence, not independent noisy timings. |
| Separate paired confirmation and durable promotion | `test_paired_confirmation.py`, `test_winner_promotion.py`, `test_effort_cache_contract.py` | Confirmation has separate cost; failed numerical candidates cannot win. |
| First-hit dynamic linear training and effort refinement | `test_progressive_cache_cuda.py`, `test_first_hit_training_cuda.py` | Includes MXFP8 and retained NVFP4 scaling modes; not every possible shape. |
| Packed-weight and fully packed inference | `test_packed_refinement_cuda.py` | Separate cache identities from dynamic training. |
| New/ragged shapes and manual capture | `test_progressive_capture_cuda.py` | Warm required shapes/paths first; cold capture does not tune. |
| Delayed history with streams and host threads | `test_delayed_history_execution_cuda.py` | Shared optimizer updates and concurrent writes to `.grad` remain caller responsibilities. |
| Contention, interrupted publication, bounded runner caches | `test_concurrent_cache_cuda.py`, `test_interrupted_cache_cuda.py`, `test_shape_churn_cuda.py` | Interruption checks are not proof of recovering an already-faulted CUDA context in-place. |
| GLU first-hit forward | `test_glu_first_hit_cuda.py` | BF16/MXFP8 output; configured-backend search; no standalone GLU backward. |
| MLP selected-plan ownership and replay | `test_mlp_first_hit_cuda.py`, `test_mlp_automatic_graphs.py`, `test_execution_plan_token.py` | Automatic selection currently targets MXFP8 precision; replay follows compiler options and is not always faster. |
| Model serialization and semantic compatibility | Export verification artifacts, `test_portable_predictors.py`, `test_matched_backward_admission.py` | Revision-22 backward models are not revision-23 models. |
| Prospective quality by budget | Frozen common-pool search, independent winner checks, full references, balanced paired measurements | NVFP4 JIT and two revision-23 backward cases completed. Backward timing is quarantined: nonstationary confirmation was accepted. One/few contexts cannot establish SKU transfer or reliable tails. |

## Consolidated execution order

Do not overlap GPU qualification with the frozen performance studies. Once the
studies finish, run the current-code GPU matrix in both `.venv` (Torch 2.13.0)
and `.venv-torch2121` (Torch 2.12.1), using distinct fresh report paths:

```bash
.venv/bin/python -m pytest -q \
  tests/test_progressive_cache_cuda.py tests/test_progressive_capture_cuda.py \
  tests/test_packed_refinement_cuda.py tests/test_first_hit_training_cuda.py \
  tests/test_delayed_history_execution_cuda.py tests/test_concurrent_cache_cuda.py \
  tests/test_interrupted_cache_cuda.py tests/test_shape_churn_cuda.py \
  tests/test_glu_first_hit_cuda.py tests/test_mlp_first_hit_cuda.py \
  tests/test_mlp_automatic_graphs.py tests/test_execution_plan_token.py \
  --junitxml=/tmp/rtx_first_hit_full_torch213_v1.xml
```

Repeat with `.venv-torch2121/bin/python` and a `torch212` report name. Preserve
failures and skip reasons; a skipped CUDA test is not a GPU pass. These suites
cover integration and numerical gates, not a speed comparison against BF16.

On Torch 2.12.1, the CPU-only namespace/precision/token subset has passed ten
tests and two subtests, with six GPU-dependent skips. The new first-hit GPU
matrix on that environment remains outstanding. On Torch 2.13.0, the latest
MLP integration batch passed 32 tests and two subtests; prior linear/GLU results
are recorded individually in the audit and must not be presented as a new
consolidated run.

## Promotion decisions

The held-out backward study exposed a bad refinement under order-sensitive
timing. Promotion revision 2 now excludes explicitly nonstationary timings
from decisions, retains a numerically valid incumbent on inconclusive evidence,
and requires an improvement in both measurement orders (every paired round),
as well as the configured median margin. Timing instability does not blacklist
a numerically valid configuration. Old confirmed-winner metadata is rejected
in both reuse and refine modes, without discarding raw search evidence.
Research comparison rejects nonstationary evidence, including old artifacts
without the new summary flag. Unknown timing-quality metadata from generic
adapters remains supported; this is not a claim of statistical confidence.

The actual failing case-1/portfolio/16-trial confirmation has also been
replayed on CPU with its recorded eight-trial incumbent held fixed. Revision 2
retains `8f041a7821c9f584c459` as `timing_inconclusive`, rather than promoting
`036a60c43daa5cf63fa0`. The immutable
`matched_backward_20260914_v1/bad_promotion_cpu_replay_v2.json` records the
conditional replay, exact input hashes and sibling script. This checks the
observed defect, not an uninterrupted new search trajectory or new timing.

A read-only inventory of all 21 `matched*20260914*` final paired measurement
files found explicit stationarity metadata throughout. Only the two backward
files contain nonstationary samples (nine each); the other 19 have none.
`autotune_reports/paired_quality_inventory_20260914_v1.json` records file hashes,
counts and affected lines, with its reproducible sibling script. This audits
final measurements, not every search-time confirmation, and does not change
any prior admission or promotion decision.

The focused revision-2 batch passed 37 tests and 14 subtests. The original
consolidated v1 run was deliberately interrupted after four tests/four
subtests, before modifying source; it is not a completed qualification. The
full corrected batch is running serially in both Torch environments, with
the same command above plus the three promotion/cache CPU test files, under
`/tmp/rtx_first_hit_full_torch{213,212}_v2.{log,xml}`.

Torch 2.13 v2 finished in 443.43 seconds: 78 tests and 38 subtests passed;
four NVFP4 subcases failed at the cache-eviction test's instrumentation check,
before eviction (the patched forward cache was empty). Existing launcher
instances reused from earlier tests bypassed the test's constructor patch.
The test now patches existing as well as newly constructed launchers and
restores both afterward, without clearing production registries or weakening
eviction/numerical assertions. A targeted suite-order reproduction is running
(`test_first_hit_training_cuda.py` followed by `test_shape_churn_cuda.py`).
The serial batch stopped on those failures; Torch 2.12 v2 did not start.

The targeted suite-order follow-up passed 3 tests and 14 subtests in 57.30
seconds (`/tmp/rtx_churn_suite_order_fix_v1.{log,xml}`), including all four
NVFP4 policies. This closes the observed instrumentation failures without
changing runtime code. The full Torch 2.12.1 v2 matrix has now been launched
separately at its previously unused report paths. Its results remain pending.

Keep new priors, ranked openings and progressive policy changes opt-in unless
prospective comparisons support their quality, tail behavior and startup cost.
Do not infer universal superiority from the positive NVFP4 JIT case or export
parity. Preserve evaluation geometry exclusions for subsequent model revisions.
No commit, push or runtime-winner installation is authorized by this checklist.
