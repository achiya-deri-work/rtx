# User-controlled autotuning effort

`rtx.AutotunePolicy` composes the existing `rtx.autotune` evaluators,
residual journals, strategies, confirmation logic, and cost-aware contextual
UCB scheduler. It does not introduce another timing engine.

## Current support and qualification

### Explicit backward measurement mode

For linear backward tuning, `AutotunePolicy(backward_measurement_mode="cuda_graph")`
measures a complete captured batch instead of host-fed launches. The default
is `"host"`. Opt-in `"auto"` resolves to graph measurement in the compiled
joint-backward lowering when `triton.cudagraphs` is enabled, and to host
measurement otherwise. Eager and partial-gradient paths resolve auto to host.
This is **not** a request to enable application CUDA graphs. Choose the mode
to match the intended consumer; dual-stream scheduling can rank candidates
differently under capture. GLU/MLP and calibrated dataset harness measurement
are not switched by this linear-backward option.

Graph-mode backward uses a separate durable winner variant and search identity,
and cannot fall back to the legacy host-measured winner. dX/dW are checked
after capture/replay before timing. `graph_capture_ms` is reported separately
from compiler time and steady-state samples; capture includes
`calls_per_sample` backward calls and their auxiliary-stream joins. Forward
measurement is unchanged. Existing host-mode kernel winners are preserved.

Focused checks passed 21 CPU tests/12 subtests (one CUDA skip), followed by
three CUDA tests including compiled ragged 8→16 refinement and repeat-call
reuse (`/tmp/rtx_backward_mode_gpu_v1.xml`). This is not yet qualification of
every compiler capture/fallback, cross-mode prior transport or every
backward family/shape under capture. Keep learned priors opt-in; historical
host-measured model quality does not establish graph-mode ranking quality.

Cross-mode fresh-process isolation is also tested: host → graph → host → graph
uses one eight-trial forward history and two separate eight-trial backward
histories (24 total observations). The final two fresh processes forbid tuning
and reuse their respective winners; forward/dX/dW pass independent FP32
references. Switching only backward measurement mode no longer invalidates
forward effort evidence. This follow-up passed three GPU tests in 23.69s
(`/tmp/rtx_mode_pass_isolation_gpu_v1.xml`) after 22 CPU tests/10 subtests.

The joint-backward path also passes an actual compiler-replay check with
`torch.compile(..., options={"triton.cudagraphs": True})`: after warmup the
Python packed-backward launcher is no longer called, while independent dX/dW
gates hold across eight steps (`/tmp/rtx_backward_compiler_replay_v1.xml`).
This proves the tested path replays; it does not mean every compiler graph is
captured. Auto inference uses the compiler's requested setting, not an
after-the-fact guarantee that capture succeeds. Its resolved request embeds
policy/cache data rather than relying on lowering-time Python registry state,
and six focused GPU tests passed in 49.64s, including cold auto host/graph
selection, numerical checks and fresh-process no-retune reuse
(`/tmp/rtx_auto_backward_mode_gpu_v1.xml`). Defaults remain unchanged.

| Surface | Current behavior | Qualification boundary |
|---|---|---|
| MXFP8Linear / NVFP4Linear first hit | A balanced-mode runtime miss can tune on real inputs, including through `torch.compile(fullgraph=True)`, then cache the chosen configuration. | Focused linear CUDA tests exist; this is not a promise that every shape achieves near-optimal performance. |
| Effort levels | Quick/balanced/thorough/extensive supply 8/16/32/64 trial caps. | Wall time is also bounded by a soft admission deadline; failures and abandoned attempts can consume the budget. |
| Compatible cache hit | `cache_behavior="reuse"` reuses an existing winner; `"refine"` permits extending its evidence to a larger budget. | Search and measurement identities must match; a changed policy is not merely a larger budget. |
| Progressive search | Explicit `search_schedule="progressive"` makes the proposal plan budget-independent and resumable. | Experimental; resume equivalence assumes identical observation/cost evidence, not identical noisy GPU measurements in separate runs. |
| CUDA graph replay | Warm the required shapes and paths before capture. | A cold tuning miss inside capture is not a supported way to autotune. |
| GLU | Explicit tuning and opt-in `autotune="balanced"` first-hit selection accept effort policies. | Forward-only; first-hit search stays within the configured backend. See the focused qualification below. |
| Recomputed MLP | Explicit tuning and opt-in `autotune="balanced"` support first-hit selection of a forward-owned recomputation plan. | MXFP8 precision only for automatic selection; compiled replay follows the caller's compiler option. Local lifecycle/effort tests exist, not universal quality or software/device coverage. |
| Learned priors and new opening controls | Priors remain opt-in; factorial opening controls live in research drivers. | No universal learned default or prediction-only hard pruning has been justified. |

The public defaults are still legacy search, cache reuse and zero paired
finalist rounds. The 8/16/32/64 labels describe investment, not guaranteed
quality tiers. See [goal audit](autotune_goal_audit.md) for remaining release
gates and [measured evidence](autotune_protected_prior_results.md) for both
positive and negative results. Later sections describe experimental settings
explicitly; they are not implicitly enabled by importing the library.
The [qualification matrix](autotune_qualification_matrix.md) consolidates the
acceptance tests and the remaining Torch 2.12.1/2.13.0 GPU execution checklist.

### Composite-kernel evidence compatibility

GLU and recomputed MLP explicit tuning now use the same progressive evidence
isolation rule as linears: increasing only trial/time budgets reuses a journal,
while changing measurement/search policy or the initial configuration/search
domain creates a separate namespace. Existing journals are preserved, not
relabeled. This is persistence parity for `autotune_`; implicit selection is a
separate opt-in described below. Legacy search paths retain their existing
locations.

GLU additionally supports opt-in first-hit selection:

```python
glu = rtx.MXFP8GLU(
    768, 1536, device="cuda", autotune="balanced",
    tuning_policy=rtx.AutotunePolicy.preset(
        "balanced", search_schedule="progressive", cache_behavior="refine",
        finalist_rounds=2,
    ),
    autotune_cache_dir=".rtx-cache",
)
compiled_glu = torch.compile(glu, fullgraph=True, dynamic=False)
with torch.no_grad():
    y = compiled_glu(x)
```

The real-input launcher resolves the configuration; tracing/fake execution
does not tune or mutate the module. Warm runners bypass selection entirely.
Misses use shared process locking, progressive residual journals and
revisioned durable winners, with activation/output-format/initial-config
identity. Larger compatible budgets refine; cache-only mode never searches.
Both BF16 output and `output_format="mxfp8", packed_output=True` are supported.
First-hit search stays in the constructor's backend (default prequant), not
an implicit cross-backend competition. Explicit `autotune_` retains that option
and disables implicit selection after installing its result. GLU remains
forward-only; this does not provide standalone GLU autograd.
Warm the shape on the capture stream before capturing it.

Focused first-hit qualification covers compiled/eager equality, BF16 and packed
MXFP8 outputs, 8→16 refinement, durable winner reuse after runner eviction,
unchanged repeat-call journals/graph count, and warmed three-replay capture at
M=K=H=128. Independent FP32 projection/SiLU/product reference normalized RMS
is 5.23–5.25% for BF16 output and 5.85–5.86% for MXFP8 output. The consolidated
GLU CUDA suite (including existing wider kernel/API coverage) passed 16 tests
and 275 subtests in 116.41 seconds:
`/tmp/rtx_glu_first_hit_integration_20260914.xml`. That broader existing suite
does not make the first-hit test a full cross-product qualification; cold
multi-process contention, larger-shape first-hit budgets, and CTA-backend
first-hit coverage remain to be expanded.

Recomputed `MXFP8MLP` now has a separate experimental `autotune="balanced"`
path with the same policy/cache arguments. It selects a complete training plan
on a grad-enabled first call and saves that plan as CPU metadata for recomputed
backward. Explicit shape policies override it. The default remains `off`.
Current evidence includes a real 8→16→32→64 training ladder, fresh-process
durable reuse, caller-requested automatic replay, external warmed capture,
outstanding forwards with reverse backward, and ragged partial-gradient opcheck.
Its private backward bridge uses separate outputs without gradient staging
copies. The full ladder covers one training geometry, not every shape/policy.
Replay is not universally faster: fixed-config measurements improved a small
case but regressed a larger decoder case by 3–4%. Broad performance and
effort-level quality qualification are still pending.
See [MLP integration status and example](mlp_first_hit_plan_contract.md) for the
precise scope and remaining work.

### Evaluated research prior

For a local opt-in experiment with the newly evaluated MXFP8 interaction prior:

```python
policy = rtx.AutotunePolicy.preset(
    "balanced", search_schedule="progressive", cache_behavior="refine",
    finalist_rounds=2, allow_research_prior=True, predictor_heads=("trees",),
    predictor_artifact="autotune_reports/transport_prior_20260914_v1/predictor.json.gz",
)
```

This artifact is generated research output, not bundled release data. It supports
MXFP8 prequant forward; incompatible families fall back to ordinary search.
On one fresh 5070 Ti shape, matched 16-trial search selected a 2.19% faster
winner than the older prior, with identical winners at 8/32/64 trials. This
does not establish portable all-family quality or improve the default implicitly.
Changing the predictor changes the evidence contract; do not expect old-prior
trials to count as a compatible refinement of this experiment.

`ranked_opening_trials=1` or `2` additionally reserves early proposals from the
predictor, before normal UCB selection. It requires progressive search and at
least one predictor head. With standard settings, two local and two retrieved
candidates remain protected; two model slots leave trial eight for exploration.
Duplicate/illegal ranks do not consume the model quota. Multiple heads are
visited round-robin, reusing their frozen ranking and provenance. Oversized
combined openings are rejected. A missing/incompatible predictor still falls
back to ordinary search. Wall-time exhaustion or a small trial cap can stop
before all reserved slots execute.

This new option defaults to zero and preserves existing evidence identities
when disabled. Enabling/changing it creates a different evidence contract.
The 2.19% result above used **zero** ranked opening slots; it must not be
attributed to this new allocation. CPU resume equivalence and compiled
MXFP8 forward/backward 8→16 refinement are verified; eight-trial speed benefits
still require prospective measurement.

The first actual-runtime comparison now covers two fresh MXFP8 geometries,
M2176/N3584/K2816 and M4608/N1536/K1920, with 8/16/32/64 cumulative budgets.
It verifies the proposed first-eight allocation but shows **no supported
eight-trial gain**. On the first shape, opening two versus zero gives +0.111%
paired runtime at 8/16 (interval −0.022% to +0.344%) and −0.017% at 32/64
(interval −0.128% to +0.027%). On the second, winners are identical at 8/16/32;
64 improves by 0.462% (interval 0.349–0.755% faster). Thus zero remains the
default. Both sides use the same interaction prior and production scheduler;
this is not an exhaustive finite-pool reference. All 256 search trials and
independent finalist numerical checks passed. Raw evidence is in
`autotune_reports/runtime_ranked_opening_20260914_v1`.

### Delayed-state stream ownership

Managed NVFP4 delayed forward submissions now hold a per-history lock across
bootstrap, launch and publication. A change of CUDA stream queues an event
dependency on prior submissions; warm calls on the same stream add no event
operations. Stream/shape transitions retain the existing history-bootstrap
policy. Caller-managed functional history buffers remain caller-owned.

The two-stream regression uses one shared module, fixed weights, changing
inputs, noncontiguous upstream gradients, and stream sequence 0/0/1/1/0/1.
It observes pending producer work and enqueues all forwards before asking for
gradients, so autograd synchronization cannot mask a forward-history race.
Eager and compiled paths match ordered execution exactly for Y/dX/dW and all
four history buffers. All 36 independent numerical checks pass (maximum
normalized RMS 13.96% Y, 3.86% dX, 3.77% dW), with no retuning and one compiled
graph. Disabling only the guard in an isolated process makes both modes fail
history equality, confirming the test exercises the dependency.

Reproduce with `tests/test_delayed_history_execution_cuda.py`; current artifact:
`/tmp/rtx_delayed_shared_module_forward_overlap_20260914.xml`. This is a
correctness test, not a throughput benchmark. It uses one host thread and
separate returned gradients, not concurrent updates to a shared `.grad` or
optimizer state. PyTorch emits a retained-AccumulateGrad stream-mismatch warning
during backward; this is preserved in the artifact, and backward starts only
after all forward streams finish.

The additional warmed two-host-thread regression now passes in both modes.
It records acquisition order inside the production history lock and replays
that order on an independent module. Both observed orders were 2/0/3/1/5/4,
not input-index order; histories and Y/dX/dW match exactly. Another 36 independent
numerical checks pass with the same maximum errors above. Journals remain
byte-identical at 16 observations and the compiled graph count stays one.
The combined four-test/six-subtest result is
`/tmp/rtx_delayed_host_threads_cuda_20260914.xml` (10.88 seconds).
CPU guard/bootstrap regression: 11 tests/four subtests passed in
`/tmp/rtx_delayed_thread_cpu_20260914.xml`. This qualifies concurrent warmed
forward submissions at the tested shape, not concurrent cold compilation,
backward execution, optimizer mutation, or all stream schedules. The guard's
warm-path overhead is measured below.

`tools/benchmark_delayed_guard.py` compares the managed guard against a
process-local no-guard ablation on a single stream, with fixed defaults and
stationary inputs. This does not expose a public unsafe option. On RTX 5070 Ti,
12 alternating-order pairs of 100 calls gave these median paired host costs:

| M×N×K | Eager added host time | Compiled added host time |
|---|---:|---:|
| 128×128×256 | 3.63 µs | 3.65 µs |
| 512×1536×1536 | 3.81 µs | 3.99 µs |
| 32768×768×1536 | 3.89 µs | 3.52 µs |

Small-shape CUDA-event intervals include CPU launch starvation and show similar
added cost; they are not isolated kernel-duration measurements. At the large
shape, median paired CUDA-event differences were −0.21/+0.02 µs (eager/compiled),
consistent with host work being hidden. Captured replay bypasses the Python
guard; graph paired differences ranged from −0.20 to +0.12 µs across all cases,
not evidence of a guard-induced device speedup or slowdown. All six cases pass
the unchanged 20% forward normalized-RMS gate and bitwise output equivalence,
including captured outputs. Raw 288 timing samples and environment are retained
in `autotune_reports/delayed_guard_overhead_20260914_v1.json`.
This is paired same-process forward/inference evidence, not backward timing,
first-hit tuning cost, multi-stream throughput, or a Nsight counter diagnosis.
Reproduce using `.venv/bin/python tools/benchmark_delayed_guard.py --output
<new-json-path>` on the local CUDA environment. Removing ordering is not a
deployable optimization; host-overhead reduction must preserve its contract.

CUDA graph replay bypasses Python, so callers must explicitly order replays
that share mutable histories. Do not reset/mutate delayed state, weights or
optimizer state while dependent device work is outstanding. Moving shared
history ownership between streams during capture is rejected; warm same-stream
capture and delayed-history replay regressions pass.

## Experimental protected local search

`initial_neighborhood_trials=2` reserves two legal, distinct neighbors of the
initial configuration, in the adapter's deterministic coordinate order, before
retrieval or bandit search. Unlike `baseline_trials` (adaptive local-arm pulls),
these candidates cannot be redirected by a noisy retrieved incumbent. Their
evaluations use the normal trial budget, correctness gates, and durable journal;
resume skips already attempted configurations. An exhausted neighborhood or a
wall-time deadline can yield fewer evaluations. This is not exhaustive coverage
of every coordinate.

`local_anchor_tolerance=0.01` keeps the earliest local-search anchor unless a
later measured candidate improves its score by more than 1%. This is an explicit
policy margin, **not statistical confidence**. It changes local proposals only:
raw measurements, bandit rewards, and final winner selection are unchanged.
Paired finalist confirmation is available in runtime tuning by explicitly setting
`finalist_rounds` to an even value of at least two; it defaults to zero. It is
separate from the ordinary per-candidate `confirmation_repeats` setting.
`initial_neighborhood_trials` and `local_anchor_tolerance` default to zero in the
legacy schedule. The opt-in progressive schedule interprets a zero neighborhood
count as its default of two protected neighbors.

```python
import rtx

policy = rtx.AutotunePolicy.preset(
    "quick", initial_neighborhood_trials=2, local_anchor_tolerance=0.01,
)
```

The initial trial plus neighborhood, baseline-arm, and exploration reservations
must fit the trial cap. Historical ranking studies cannot reconstruct timings
for unmeasured initial neighbors; test the protected policy on fresh GPU shapes.
See [protected prior results](autotune_protected_prior_results.md) for the local
pilot, historical rule evaluation, limitations, and reproduction commands.

Both linear modules use `AutotunePolicy` on a runtime cache miss,
including when the first invocation is through `torch.compile(fullgraph=True)`.
The defaults remain `search_schedule="legacy"`, `cache_behavior="reuse"`,
and `finalist_rounds=0`; the experimental settings above are not silently enabled.
You do not need an explicit policy unless you want to change effort or behavior.

```python
import rtx

policy = rtx.AutotunePolicy.preset("balanced", max_trials=16)
layer = rtx.MXFP8Linear(
    768, 3072, bias=False, device="cuda",
    autotune="balanced", tuning_policy=policy,
    autotune_cache_dir="autotune_cache",
)
# NVFP4Linear accepts the same tuning_policy argument.
```

Presets are quick (8 trials / 15 seconds), balanced (16 / 30), thorough
(32 / 90), and extensive (64 / 180). These are effort limits, not promises
of a particular performance level. Explicit values override presets.
`forward_trials` and `backward_trials` optionally override the per-pass cap.
Joint MLP training tuning uses the backward override. GLU and MLP explicit
tuning functions accept `tuning_policy=policy`; tune them before capture or
compilation and install the returned configuration using their existing API.

## Budget and cache contract

### Progressive schedule (experimental)

`search_schedule="progressive"` uses a budget-independent search composition
and retrieval opening, with two initial neighbors when no explicit nonzero
neighborhood count is supplied. Exploration milestones target cumulative
1/3/6/10 broad-arm trials by budgets 8/16/32/64; an unavailable arm releases
its opportunity to UCB. `exploration_trials=0` disables these forced milestones.
The schedule is opt-in and has not yet passed the full GPU qualification matrix.
Compiled dynamic MXFP8 training has passed the focused matrix described below.

Each completed observation carries a data-only checkpoint of the proposal RNG,
UCB statistics/cooldowns, exclusions, and a policy/history identity. Reopening
the journal restores this state. Effort/time caps are excluded from the search
identity; changing the seed, search settings or prior contents is not a budget
extension and a conflicting checkpoint is rejected. Old journals lacking a
checkpoint can warm-start from observations but do not establish exact resume
equivalence.

The CPU regression compares an uninterrupted 64-trial run against fresh tuner
instances resumed at 8/16/32/64 using identical measurement/cost evidence. It
does not promise identical outcomes between independently timed GPU runs.
CPU fault-injection tests additionally cover exit immediately after a durable
observation, an issued but uncompleted evaluation, and prior-backed mixed-pool
resume. Uncompleted unique attempts consume budget and are not reissued. A lost
evaluation has no timing evidence, so that case promises safe recovery, not the
same sequence as a run where the evaluation completed. A real process-kill at
the post-CUDA/pre-publication boundary is now qualified below. Killing during an
active kernel and recovery from a CUDA-fatal error remain unqualified.

`tests/test_interrupted_cache_cuda.py` kills a subprocess after its fourth
candidate returns a successful real CUDA measurement but before the outcome is
published. At that boundary exactly three observations exist and no runtime
winner exists. A fresh compiled-inference process resumes against the same cache.
For materialized MXFP8 and default JIT-region NVFP4 at M=N=K=128, the completed
journal prefix remains byte-for-byte unchanged, the abandoned configuration is
not reissued or promoted, and seven distinct observations plus one abandoned
attempt exhaust the eight-trial budget. The resumed winner carries two finalist
confirmation rounds. Independent FP32-GEMM-to-BF16 checks pass (finite, normalized
RMS below 7% MXFP8/20% NVFP4), and repeated execution neither evaluates more
candidates nor recompiles. Acquiring and completing the resumed tuning also
checks that the killed process did not leave its runtime lock held.

The test passed for both families in 19.10 seconds. Reproduce:
`.venv/bin/python -m pytest tests/test_interrupted_cache_cuda.py -q -o junit_logging=all --junitxml=/tmp/rtx_interrupted_gpu_verified_candidate_20260914.xml`.
Only child processes created by the test are killed. This does not inject
filesystem corruption, terminate a running GPU kernel, or qualify backward or
delayed-state interruption. The earlier failed harness run instrumented the wrong
MXFP8 evaluator; the final test instruments the actual evaluator for each family
and requires a successful measured candidate before signaling the kill boundary.

Mutable online model fit state is not yet supported by this progressive
schedule. Local, random/structural and compatible frozen-prior arms are retained;
the legacy schedule and its online model remain available and unchanged.

### Runtime versus experimental L2-fetch coordinates

Runtime residual contexts also fingerprint the serialized initial configuration
and ordered coordinate/value lists, after runtime projection. Reordering an axis
can change the protected neighborhood and random-walk trajectory, so preserving
only the set of values is insufficient for progressive resume. The fingerprint
applies to backward adapters as well as forward adapters. Identical domains
resume; changed domains cannot count the old domain's trials toward their budget.
Inactive per-op L2-only differences still map to the same executable domain.
The effort contract includes search-space identity revision 1, preventing older
unfingerprinted effort evidence from satisfying a new `refine` request; ordinary
compatible-winner `reuse` remains available. Old residuals are not rewritten.

This isolates search evidence; it does not certify exhaustive coverage of a
space. A strategy running out of proposals is not proof that all legal schedules
were visited. Below-budget runtime refinement admission therefore remains
conservative rather than treating the existing `search_exhausted` stop label as
an exhaustive-search certificate.

Validation: the new identity regression first failed for axis reordering, changed
axis values and changed initial configuration. After the fix, 31 CPU tests and
17 subtests passed, including same-store reopen/isolation. The GPU-enabled suite
passed 22 tests and five subtests in 79.42 seconds: compiled aligned/ragged
training refinement, packed/scaling inference, delayed custom policies, singleton
execution and independent numerical/live-memory gates. Reproduce:
`.venv/bin/python -m pytest tests/test_runtime_coordinates.py tests/test_progressive_cache_cuda.py tests/test_packed_refinement_cuda.py -q -k 'not long_reduction' --junitxml=/tmp/rtx_search_space_identity_gpu_20260914.xml -o junit_logging=all`.
This rechecks lifecycle compatibility, not search-quality performance.

Runtime first-hit adapters now canonicalize per-config `l2_fetch_granularity`
to `None` and remove its search axis. Production launchers deliberately ignore
that per-kernel field: only `RTX_L2_FETCH_GRANULARITY` selects one process-wide
setting. Previously, experimental evaluators could temporarily apply a candidate
value while timing a runtime winner that would not apply it in production.
That was a measurement-contract mismatch, not a numerical change.

Canonicalization preserves tile/pipeline/quantization fields and deduplicates
retrieved schedules differing only in the inactive L2 field. Offline adapters
and their explicit per-candidate experiments are unchanged. This is not pruning
on predicted performance. Runtime contexts and effort evidence identify the
measurement revision and explicit process-wide L2 policy; old observations are
preserved, and refinement cannot count incompatible measurement evidence as
current. Compatible-winner reuse remains a separate correctness contract.

Post-change validation: 37 CPU tests with 14 subtests passed, followed by 15
CUDA integration tests and two custom delayed-policy subtests in 77.17 seconds.
The GPU matrix retains independent numerics, compiled 8→16 refinement,
packed/scaling coverage and steady-state live-memory checks. Artifact:
`/tmp/rtx_runtime_l2_alignment_gpu_20260914.xml`. No performance improvement is
claimed yet, and this mismatch does not establish the cause of every prior
search-versus-confirmation discrepancy.

A custom search space containing only the inactive per-kernel L2 coordinate
now becomes a singleton executable space rather than an invalid empty adapter.
Its no-op coordinate is deduplicated normally: one measured initial schedule,
then clean exhaustion. The CPU regression failed on the empty-adapter guard
before this fix. After the fix, a real CUDA test measured one candidate,
reopened its tuning history with zero new evaluations, and executed the selected
config through the compiled frontend with independent finite/7%-normalized-RMS
checks. Four focused tests passed in 2.96 seconds:
`/tmp/rtx_runtime_singleton_cuda_20260914.xml`. This tests explicit tuning-history
resume; it does not claim special runtime-winner admission for an exhausted
search whose attempted count is below a requested refinement budget.

Progressive schedule revision 2 counts the retrieval opening in **distinct legal
candidates not already covered by the initial/protected neighborhood**. Revision
1 truncated retrieval first, then removed neighborhood overlap; its intended
two-candidate retrieval opening could consequently contain fewer candidates or
none. Revision 2 fills the quota from the remaining retrieved list without
changing the protected initial neighborhood or hard-pruning slow predictions.
The CPU regression demonstrates both novel retrieval candidates in the first
five trials when available, plus stable opening composition at 8/16/32/64.
Deadline or pool exhaustion can still stop a run early.

Its evidence contract and scheduler checkpoint are versioned. Progressive
runtime journals now have an evidence-contract subdirectory, preventing old
search observations from being relabeled as evidence for the corrected opening.
Existing files are preserved. Ordinary compatible-winner reuse remains valid;
`cache_behavior='refine'` requires current-contract evidence and may therefore
spend a new budget after this migration. A same-contract 8→16 extension continues
to reuse its journal. Legacy search scheduling is unchanged.

Post-change regression: 35 CPU tests and 14 subtests passed, including the
opening/evidence tests and progressive resume checks. CUDA compiled training at
three MXFP8 shapes plus concurrent MXFP8/NVFP4 first-hit/refinement passed
(4 tests, two family subtests, 76.59 seconds). Artifact:
`/tmp/rtx_progressive_v2_runtime_gpu_20260914.xml`. This verifies the runtime
plumbing without proving integrated prior quality; the prior-backed GPU quality
comparison remains a separate gate.
Effort-aware runtime refinement is wired for dynamic MXFP8 forward/backward,
packed MXFP8/NVFP4 inference, and materialized NVFP4 scaling families. Delayed
NVFP4 now passes the focused no-recompilation regression, described below;
its full state/capture lifecycle is not yet qualified.

### Paired winner confirmation (experimental)

`finalist_rounds=4` enables paired final selection in the shared orchestrator.
The default is zero (off). It compares the prior confirmed incumbent, or the
initial configuration on a fresh study, with up to two strongest challengers.
Numerical failures disqualify candidates and remain recorded as exclusions for
later paired decisions in the same context. All failed finalists raise an error
instead of installing the fastest failed candidate.

`finalist_margin=0.005` requires a median paired improvement exceeding 0.5% to
replace a valid incumbent. This is an effect-size policy, not a confidence
interval. Incomplete deadline-limited comparisons retain the incumbent and may
be retried later. Full decisions are reusable without further GPU measurements
when the evidence and confirmation policy match.

`finalist_reserve_fraction=0.2` reserves 20% of the soft wall-time allowance from
search for final confirmation. An admitted balanced pair may overrun the limit.
Confirmation evaluations do not consume unique candidate trials; their count
and full raw outcomes are stored separately in the residual event journal.
Search observations, including their original screening/contender evidence,
are not rewritten by the final selection. The returned tuning config is the
selected incumbent/challenger, not necessarily the lowest screening median.

These options have CPU regression coverage and focused compiled MXFP8 CUDA
coverage, but are not enabled by runtime defaults. Cache refinement is separate.

### Effort-aware cache refinement (integration in progress)

`cache_behavior="reuse"` is the default: immediately reuse a compatible runtime
winner. In balanced tuning mode, `cache_behavior="refine"` additionally requires matching search and
measurement evidence, enough cumulative attempted trials for the requested pass,
and the requested finalist confirmation. Missing evidence is not interpreted as
having met the budget. Trial/time cap increases do not change evidence identity.

Cache-only mode never tunes and does not enforce a refinement budget, even if
the supplied policy carries `cache_behavior="refine"`. It may reuse a compatible
runtime or legacy-cache configuration. Coordinate mode likewise retains its
existing fallback behavior; only balanced refinement suppresses fallback to
cache entries that lack effort evidence. A regression previously suppressed
legacy fallback in these non-balanced modes merely because the policy said
`refine`; resolver tests now cover forward, backward and delayed fallback, plus
the opposite requirement that balanced refinement must not accept legacy effort.
Post-fix qualification: 22 CPU tests (14 subtests) and 14 CUDA integration tests
passed. The CUDA group covers the three compiled MXFP8 forward/backward shape
cases and the full packed/scaling-policy refinement suite, including initialized
delayed deepcopy and nested delayed training. Artifact:
`/tmp/rtx_cache_mode_refinement_gpu_20260914.xml` (49.03 seconds).
No numerical gates, kernel implementation revisions or policy defaults changed.

Creating a new execution request
with a higher budget can refine; already resolved/captured execution plans remain
fixed. This is not background tuning or a mutation of an active captured graph.

CPU process qualification (`tests/test_effort_cache_process.py`) launches three
independent workers per wave against one cache. It verifies one miss at budget
8, one refinement at 16, no miss when subsequently requesting 8, and a new miss
after changing the kernel revision. A separate test kills a confirmed lock
holder before publishing a winner and verifies another process can acquire the
lock and populate the cache. This qualifies advisory-lock recovery, not recovery
of a killed GPU evaluation or its search checkpoint. The workers use synthetic
effort evidence; they do not benchmark kernels.

Malformed non-object JSON is treated as a cache miss and reported invalid by
inspection. Stored fields cannot override the inspector's compatibility verdict.
The cache/progressive/confirmation/operations regression group passed 49 tests
and 14 subtests (`/tmp/rtx_effort_cache_process_qualification.xml`).

Real concurrent GPU first-hit qualification now passes for materialized MXFP8
and JIT-region NVFP4 training at M=128, N=128, K=256. Two independent compiled
workers share a cache for each of three waves: budget 8, refinement to 16, then
reuse at 8. Each forward/backward tuner runs exactly once across the workers
on each miss/refinement and never on reuse. Each journal contains 16 distinct
trials. Repeated calls do not retune or recompile and reproduce outputs and
both gradients exactly. Independent FP32 GEMMs cast to BF16 pass finite and
normalized-RMS gates (7%, except 20% for NVFP4 forward).

Run: `.venv/bin/python -m pytest tests/test_concurrent_cache_cuda.py -q`.
Result: 1 test, 2 family subtests passed in 46.93 seconds; evidence:
`/tmp/rtx_concurrent_first_hit_gpu_20260914.xml`. This is correctness and cache
coordination evidence, not performance evidence: concurrent work can contaminate
timings. It does not qualify one module shared across threads/streams, delayed
state concurrency, or interrupted GPU evaluations.

Delayed numerical policy is now part of runtime tuning identity. A custom
`NVFP4ScaleConfig` is passed to the benchmark's reference and candidate runners;
history allocation uses its actual length. Nondefault combinations of scale
mode, history length, and history algorithm have separate runtime-winner variants,
search directories, and context tags. They cannot silently reuse default-history
or block-only evidence. The default policy retains its existing identity because
its measured implementation is unchanged. Explicit user-supplied schedules remain
explicit schedules, not claims of policy-specific tuning evidence.

Regression evidence: 15 CPU tests with 10 subtests, plus the expanded 12-test
CUDA packed/refinement suite with two custom-policy subtests. A final focused
CUDA run passed 5 delayed tests and two subtests, including persisted winner
variant assertions (`/tmp/rtx_delayed_policy_identity_gpu_20260914.xml`). Custom
policies exercised compiled inference at budgets 8 and 16 for exact/4/most-recent
and power2/64/window-max, with independent forward normalized-RMS below 20%,
finite outputs, and no repeated tuning or cache-induced recompilation. All 16
policy identities are distinct in CPU tests; this is not exhaustive GPU numerical
qualification of every history policy or changing-distribution sequence.

Fresh-process compiled GPU refinement now passes for materialized dynamic
MXFP8 and default JIT-region NVFP4 inference at M=128, N=128, K=256. Five
independent processes share one temporary winner/journal directory: write at
8 trials, read at 8, refine to 16, read at 16, then read requesting 8 again.
Each writing phase invokes each family's tuner exactly once; all reading
phases forbid tuner entry and pass. Repeated calls retain one compiled graph,
and each winner records sufficient cumulative effort plus two finalist rounds.
This is a lifecycle test, not a search-quality or throughput claim.

Against an independent FP32 GEMM cast to BF16, relative RMS errors were 3.5482%
for MXFP8 and 13.7364% for NVFP4, with maximum absolute errors 0.0859375 and
0.3740234375 respectively. All outputs were finite and repeated outputs were
bitwise equal. Gates remained 7% and 20%; these results do not qualify delayed
scaling, backward in a fresh process, or arbitrary shapes.

Reproduce with `.venv/bin/python -m pytest tests/test_runtime_cache_cuda.py -q -s`.
The run passed 2 tests and 5 subtests on RTX 5070 Ti, driver 595.84, Python
3.13.15, Torch 2.13.0+cu132, and CuTe DSL 4.7.0; result:
`/tmp/rtx_fresh_process_progressive_gpu_20260914.xml`.

The corresponding **training** test now passes the complete 8→16→32→64
effort ladder for both materialized dynamic MXFP8 and JIT-region NVFP4 at
M=128, N=128, K=256. Nine independent processes perform a write/refinement and
cache-only read at each budget, followed by another 8-trial request after 64.
Families use separate caches, so sharing their MXFP8 backward implementation
cannot mask a missing first-hit test. Both forward and backward tuners execute
exactly once per writing phase and are forbidden in every reading phase.
Repeated training calls leave observations unchanged and cause no additional
top-level graph compilation. The upstream gradient is noncontiguous.

All 54 output/gradient metric records pass independent FP32 GEMM references
cast to BF16: output relative RMS is 3.5581% for MXFP8 and 13.8425% for NVFP4;
dX and dW are 3.5570% and 3.7088% for both families. Every output/gradient is
finite and repeated calls are bitwise identical. This proves one-shape lifecycle
and numerical coverage across the effort ladder, not general search quality,
delayed-state correctness, or graph-capture replay.

Reproduce: `.venv/bin/python -m pytest tests/test_runtime_cache_cuda.py -k progressive_training -q`.
The test passed on the same software/device stack in 111.56 seconds. Full
captured trial logs and metrics are retained in
`/tmp/rtx_fresh_process_training_efforts64_gpu_20260914.xml`.

### Compiled optimizer trajectories

Short optimizer trajectories are separately qualified in
`tests/test_first_hit_training_cuda.py`: MXFP8 and NVFP4 block/current/JIT-region/
delayed run 16 SGD steps on a fixed synthetic linear regression problem
(M=256, N=K=128). Parameters, inputs and gradients are BF16; loss reduction and
independent oracle GEMMs use FP32. The linear plus loss is compiled fullgraph;
SGD steps run outside the compiled region. Each path tunes eight forward and
eight backward candidates once, then updates weights without new journal rows
or additional compiler graphs. Every step checks finite forward/dX/dW values
against independent references using the same upstream gradient and unchanged
7%/20% numerical gates.

The final local run passed all five paths in 15.23 seconds. All training losses
decreased by more than 34%. Learned weights were also evaluated through the
same independent FP32-GEMM→BF16 path on training and held-out inputs, avoiding
comparison bias from differently quantized outputs. Held-out loss ratios to
the BF16-trained control were 1.00131 (MXFP8), 1.00958 (block) and 1.00156
(current/JIT/delayed). The block path's lower *quantized training loss* did not
mean better learned weights: its common training-set loss ratio was 1.01477.
These are short synthetic trajectories, not transformer convergence or proof
that block scaling is safe for other distributions. No timing claim is made.

Reproduce: `.venv/bin/python -m pytest tests/test_first_hit_training_cuda.py -q --junitxml=/tmp/rtx_first_hit_optimizer_common_eval_20260914.xml -o junit_logging=all`.
The XML retains all per-step losses and numerical errors, common-evaluation
results, seeds, dtype/optimizer policy, device identity and raw tuning logs.

### Long-reduction training across the effort ladder

Long-reduction training qualification now covers M=8192, N=128, K=256 through
the full 8→16→32→64 refinement ladder for MXFP8 and NVFP4 block, current,
JIT-row-region, and tensor-delayed. Inputs, weights, and noncontiguous upstream
gradients are held fixed across budgets. Independent FP32 GEMMs, cast to BF16,
provide forward, dX, and dW references; gates remain 7% normalized RMS for
MXFP8/gradients and 20% for NVFP4 forward, with separate finiteness checks.
Absolute, RMS, reference-RMS and normalized-maximum errors are also recorded.

On the local RTX 5070 Ti, all five paths passed in 156.76 seconds (two tests,
four NVFP4 subtests). Forward normalized RMS was 3.5529% for MXFP8,
13.7559% for block, and 13.9268% for the other NVFP4 policies. Across all paths,
dX/dW were 3.5566%/3.7289%. Each value was stable across the four budgets.
The 20 workload/budget cases made 400 post-warmup training calls with 100
synchronized memory samples: zero live-allocation growth, no repeated tuning
observations and no added compiler graphs. Final winner evidence reached 64
trials with two paired finalist rounds. This is one long-reduction geometry
and stationary-input delayed-state coverage, not a changing-distribution
training trajectory or proof of search-quality gains.

Reproduce: `.venv/bin/python -m pytest tests/test_progressive_cache_cuda.py -q -k long_reduction --junitxml=/tmp/rtx_long_reduction_efforts_20260914.xml -o junit_logging=all`.
The XML retains captured trial logs, JSON numerical metrics and memory samples.

### Steady-state memory and CUDA graph replay

Steady-state allocator qualification now augments the compiled first-hit tests.
After three lifecycle warmup calls, each case executes 20 calls and samples
synchronized live allocated, reserved, and peak allocated bytes every five calls.
Training releases iteration-local gradients before sampling. No allocator cache
flush is used. Live bytes must not exceed the post-warmup baseline; reserved
growth alone is not classified as a leak. CPU guard tests verify both rejection
of live growth and acceptance of reserve-only growth.

The RTX 5070 Ti run passed 15 CUDA tests and two subtests in 76.28 seconds:
26 workload/budget cases, 520 post-warmup calls, and 130 sample points showed
zero live-allocation growth, with no added tuning observations or compiler
graphs. Coverage is budgets 8/16, MXFP8 training at three aligned/ragged shapes,
MXFP8/NVFP4 packed inference, and NVFP4 current/block/JIT/delayed inference,
including two custom delayed policies. Independent numerical checks remain
enabled. This is not shape-churn eviction, all-budget memory qualification, or
proof of bounded host/compiler metadata across arbitrarily many shapes.

Reproduce: `.venv/bin/python -m pytest tests/test_progressive_cache_cuda.py tests/test_packed_refinement_cuda.py -q -o junit_logging=all --junitxml=/tmp/rtx_first_hit_live_memory_20260914.xml`.
The XML includes JSON `MEMORY` records with workload labels, sample series, and
Torch/CUDA/device identity. These validation runtimes are not performance results.

Eager materialized shape-churn eviction is separately qualified by
`tests/test_shape_churn_cuda.py`: MXFP8 and NVFP4 block/JIT-region/delayed each
use a temporary two-entry runner cache while tuning M=128/256/384, N=K=128
at eight trials per shape. After the third shape, weak references confirm that
the first runner's workspace tensor objects have been released (4/6/7/7 tensors
respectively). Revisiting the first shape rebuilds its runner while both tuning
entry points are patched to fail if called. Each family retains exactly 24
observations, two runner entries, and two evictions. Independent forward
FP32-GEMM-to-BF16 references pass finite and normalized-RMS gates on all visits
(7% MXFP8, 20% NVFP4).

Result: 1 test and four family/policy subtests passed in 24.66 seconds. Reproduce:
`.venv/bin/python -m pytest tests/test_shape_churn_cuda.py -q -o junit_logging=all --junitxml=/tmp/rtx_shape_churn_20260914.xml`.
This qualifies eager forward workspace eviction and in-process configuration
reuse, not compiled-graph ownership eviction, backward-runner eviction, or an
unbounded number of distinct compiler specializations.

Compiled training eviction is now separately qualified in the same test file.
MXFP8 and NVFP4 block/current/JIT-region/delayed use
`torch.compile(fullgraph=True, dynamic=False)` at M=128/256/384, N=K=128,
with noncontiguous upstream gradients. The test limits the actual
launcher-local compiled-forward cache and shared backward cache to two shapes.
After three shapes, weak references prove the first forward workspace
(4/6/6/7/7 tensor objects respectively) and backward workspace (eight objects
in each case) are released. Revisiting the first shape succeeds with every
tuner entry point patched to fail. The 48 durable forward/backward observations
remain byte-identical and the top-level graph count stays at three. Each cache
ends with two entries and two evictions.

All 60 independent Y/dX/dW checks passed: maximum normalized RMS was 3.825%
for MXFP8 Y, 15.328% for NVFP4 block Y, 13.936% for other NVFP4 Y, and
3.811%/3.760% for dX/dW across all policies. Gates remained 7% for MXFP8 and
gradients, 20% for NVFP4 forward. Each policy then completed 20 repeated
compiled training steps after warmup with zero live-allocation growth across
five synchronized samples. Reserved memory is recorded but is not a leak gate.

Combined eager/compiled result: two tests and nine policy subtests passed in
57.81 seconds. Reproduce:

```bash
.venv/bin/python -m pytest tests/test_shape_churn_cuda.py -q \
  -o junit_logging=all \
  --junitxml=/tmp/rtx_eager_compiled_eviction_qualified_20260914.xml
```

The first compiled test attempt mistakenly patched eager-forward caches and
failed its cache-ownership assertion; that artifact is preserved as
`/tmp/rtx_compiled_training_eviction_20260914.xml`. The corrected test patches
the real Inductor launcher-local caches. Production runtime code was not
changed to obtain the pass. This covers three aligned training shapes and
full backward with bounded runner caches, not unlimited distinct Dynamo
specializations, simultaneous streams, or every ragged/partial-backward case.

`tests/test_progressive_capture_cuda.py` passes aligned (128×128×256) and
ragged (257×131×129) M×N×K training for materialized MXFP8 and JIT-region
NVFP4. Each case performs 8-trial tuning and warmup on the capture stream,
captures forward, backward and gradient clearing, then changes X, W and dY
for three replays. All tuner entry points are forbidden during capture/replay;
the journal/cache bytes and top-level compilation count remain unchanged.

The 36 independent-reference checks are finite and within unchanged gates:
MXFP8 output relative RMS 3.684–3.792%, NVFP4 output 13.280–13.532%, and
both families' dX/dW 3.691–3.809%. This does not cover cold capture, delayed
scaling, graph-pool memory growth, or every effort/shape combination.

The first run exposed an unrelated first-hit failure before capture: a ragged
explicit materialized MXFP8 request started with an illegal MMA128 baseline.
Default initial selection and the tuning correctness reference now share the
existing legal row-major fallback. Explicit nondefault initial configurations
are not silently replaced. The failed artifact is retained at
`/tmp/rtx_progressive_capture_gpu_20260914.xml`; the final four-case passing
artifact is `/tmp/rtx_progressive_capture_final_gpu_20260914.xml`.
CPU regressions: 29 tests plus 10 subtests, and 20 prequant tests (2 CUDA tests
skipped in the CPU-only run). Reproduce capture with
`.venv/bin/python -m pytest tests/test_progressive_capture_cuda.py -q`.

A focused RTX 5070 Ti test runs `torch.compile(fullgraph=True, dynamic=False)`
with progressive 8→16 trial budgets, paired final selection, and checks repeated
forward/backward calls add neither observations nor graph compilations:

| M×N×K | Output / dX / dW independent-reference relative L2 range |
|---|---:|
| 128×128×128 | 3.75–3.82% |
| 1024×384×256 | 3.54–3.75% |
| 257×131×129 | 3.75–3.82% |

References are FP32 GEMMs cast to BF16; all tensors must be finite and each
relative L2 must be below 7%. The ragged case exposed and fixed a reference-path
bug: the backward evaluator unconditionally constructed a 128-divisible baseline
instead of using the existing shape-aware baseline selector. No gate was relaxed.

Run: `.venv/bin/python -m pytest tests/test_progressive_cache_cuda.py -v`.
Current evidence: `/tmp/rtx_progressive_cache_matrix_fixed.xml` and
`/tmp/rtx_progressive_environment.json`. This is integration/numerical evidence,
not a first-hit quality benchmark or complete release qualification. Shared-module
concurrency and all-family/shape capture coverage remain gates; the
one-shape fresh-process inference and training evidence is recorded above.

A second focused CUDA matrix now passes 8→16 refinement for MXFP8 and NVFP4
weight-packed/fully-packed inference, and NVFP4 current, block, and JIT-region
dynamic inference. Repeated calls add neither trials nor graph compilations.
An additional delayed-state test compares tuned versus untuned live amax
histories exactly across three changing input magnitudes; it passes.
Evidence: `/tmp/rtx_packed_refinement_other_paths.xml` (8 passed, 1 deselected).

The earlier deselected case failed because lazy Python history initialization
caused a second compilation on the next call. Its failure artifact is retained
at `/tmp/rtx_packed_refinement_isolated.xml`. Fixed-size history buffers now move
cold initialization into the real-runtime launcher. A weak-storage registry
records initialization identity, shape, stream and policy, but holds neither
owning tensor references nor numerical history. Warm lookup executes no eager
tensor operations. Reset allocates fresh same-shaped buffers; the next real
call bootstraps them. The private operator declares possible current-history
mutation as well as next-history mutation. The frontend ABI revision is now 9.

The retained 8→16 delayed refinement regression and a new nested compiled
training test pass. The latter checks exact history rotation against an
independent amax-history construction and finite output/dX/dW within the existing
20%/7% gates, with one graph across three changed inputs. The delayed-only GPU
group passes 11 tests plus 2 subtests, including shape/stream/state-load and
distribution-jump cases (`/tmp/rtx_delayed_fixed_state_matrix_gpu_20260914.xml`).
The complete NVFP4 plus packed-refinement GPU regression then passed 67 tests
and 36 subtests (1 skipped) after isolating compiler state between independent
tests: `/tmp/rtx_nvfp4_bootstrap_full_isolated_gpu_20260914.xml`. The initial
combined run's unrelated per-code specialization-limit failure is retained in
`/tmp/rtx_nvfp4_bootstrap_full_gpu_20260914.xml`. CPU frontend/bootstrap/cache
regressions passed 36 tests plus 20 subtests. No numerical gate was relaxed.
The exact delayed-replay test then exposed a second issue: a Python ping-pong
swap occurs during capture but not on replay, so the second replay read stale
history. Module histories now keep stable addresses. After quantization/GEMM,
the launcher publishes both next histories using stream-ordered device-to-device
copies, with no eager Torch tensor operations. With the default 16-entry history
this adds two 64-byte copies; their latency has not yet been performance-qualified.
Explicit out-of-place private-operator calls retain their separate next-history
contract. The failed after-replay assertion is preserved in
`/tmp/rtx_delayed_history_after_replay_gpu_20260914.xml`.

The fixed test runs 21 changed-input/weight/upstream-gradient replays, exceeding
the complete history window, and verifies every X/W history element exactly.
All 63 independent-reference output/gradient checks pass: output relative RMS
13.285–13.661%, dX 3.680–3.803%, dW 3.679–3.831%. Capture/replay never enters
the tuner or changes cache/journal bytes. The full capture, packed-refinement and
NVFP4 GPU group passes 72 tests plus 36 subtests (1 skipped):
`/tmp/rtx_nvfp4_stable_history_full_gpu_20260914.xml`. CPU publication/bootstrap,
frontend and cache checks pass 37 tests plus 20 subtests.

The delayed benchmark harness now uses the same stable-address publication as
the module, rather than a Python-only swap. Delayed evidence/kernel revision is
4, so revision-3 timings/winners cannot be reused as evidence for this workload.
A CPU regression proves publication occurs after every execution; four focused
delayed tuning/capture GPU tests pass with the revised harness
(`/tmp/rtx_delayed_benchmark_parity_gpu_20260914.xml`).

Publication overhead was measured on RTX 5070 Ti with fixed default configs,
stationary inputs, and paired CUDA graphs. Each graph contains eight executions;
ten alternating-order samples each time 25 replays. Omitting publication is a
diagnostic ablation only, **not** a deployable delayed policy or tuning datum.

| M×N×K | Full median | No-publication ablation | Median paired added cost |
|---|---:|---:|---:|
| 128×128×256 | 10.351 µs | 8.289 µs | 2.057 µs |
| 512×1536×1536 | 30.816 µs | 28.810 µs | 2.006 µs |
| 32768×768×1536 | 598.273 µs | 596.893 µs | 1.538 µs |

Independent-reference relative RMS was 13.314–13.431%; both paths were bitwise
equal for these stationary inputs. Live allocated bytes were unchanged during
measurement for all three shapes. These are not tuned-winner timings, end-to-end
model timings, host launch costs, or Nsight counter measurements. Raw samples,
configs and environment are in `autotune_reports/delayed_publication_20260914_v1.json`;
reproduce with `tools/benchmark_delayed_publication.py --output <new-json-path>`
using the CUDA virtualenv. Coalescing the two history copies is a potential
optimization only with a redesigned state ABI; deleting publication is not.

A direct coalescing experiment (adjacent X/W views in one allocation per
generation) passed its CPU copy test but **failed compiled CUDA capture**.
Functionalization changed the views' storage identities, so the runtime could
not recognize the warmed history during capture. The initialization guard
correctly stopped execution rather than silently resetting state. Evidence:
`/tmp/rtx_delayed_coalesced_history_gpu_20260914.xml`. The allocation/copy change
was reverted; the 12-test delayed GPU matrix passed again with 2 subtests
(`/tmp/rtx_delayed_coalescing_reverted_gpu_20260914.xml`). No performance win
is claimed for the rejected candidate. Managed bootstrap now explicitly rejects
shared-storage history views before initialization; the supported module uses
four independent storage buffers. A packed-history optimization would need
paired state tensors at the operator boundary, with renewed functionalization,
history and graph-replay tests—not merely adjacent allocations.

Initialized-module `copy.deepcopy` now preserves the initialization signature
after independently copying numerical history buffers. Previously the copied
values lost their readiness metadata and would silently rebootstrap. A cold
copy remains cold; resetting a copy does not reset its source. Cold registry
insertion prunes expired storage entries, including repeated copy/destruction.
The CPU copy/cleanup/frontend group passes 35 tests plus 10 subtests.

An initialized delayed copy compiled with `fullgraph=True` matches the original
eager module's output, dX, dW and complete histories bitwise across three changed
inputs, with one graph and unchanged independent-reference numerical gates.
The copied-state plus delayed refinement/capture group passes five GPU tests:
`/tmp/rtx_delayed_copy_and_capture_gpu_20260914.xml`. This is same-device,
same-stream deepcopy evidence, not module-pickle or cross-device migration
qualification.

Concurrent use of one delayed module remains unqualified. Do not treat these
focused results as full stateful-runtime qualification before promoting
the complete first-hit policy. Independent test cases reset compiler state to
avoid exhausting Dynamo's shared per-code specialization limit; no reset occurs
between budget increases or repeated calls within a case.

- Trials count distinct attempted configurations, including compile or
  correctness failures. Static legality rejects do not consume trials.
- The cap includes compatible resumed observations. Explicitly tuning at
  8, then 16, allows eight additional attempts rather than sixteen.
- Confirmation repeats are additional measurements of the same candidate,
  not additional distinct trials. They cost wall time and are recorded.
- Time is a per-invocation **soft admission limit**, including setup and
  proposal generation. An admitted compile/measurement/confirmation is not
  interrupted, so actual elapsed time may exceed it.
- Runtime winner hits bypass tuning. Merely raising a layer's policy does
  not force refinement of an existing runtime winner. Use the explicit
  family tuning function to refine its journal, then install its result.
- Measurement settings are part of history identity; changing those can
  start a separate study. Effort limits themselves do not split history.
- Restored candidates are checked against current static legality. Old
  quarantined winners cannot silently become the returned winner.

## Search composition

Start with the known initial configuration. Reserve one random-exploration
attempt by default (`exploration_trials=0` disables this). When a compatible frozen prior
is explicitly supplied, try a small retrieval prefix, then let UCB allocate
the remaining attempts among frozen ranking, coordinate/local search, and
random exploration. Without a prior, larger budgets also enable the
existing lightweight online boosted cost model. Mandatory arm warmup
blocks are disabled so they do not swallow small budgets.

`pool_size`, `retrieval_trials`, `exploration_trials`, `exploration`, `seed`, timing samples, and
confirmation repeats are configurable. Research artifacts require both
`prior_artifact=...` and `allow_research_prior=True`. No new research model
is installed implicitly. Unknown prior contracts fall back to ordinary
search. This policy is supported by the synchronous tuning entry points;
it is not a replacement for the separate ask/tell runner policy API.

The current defaults use fixed screening measurement settings plus
contender confirmation—not a newly calibrated adaptive sequential timing
algorithm. CUDA smoke tests validate integration, not universal convergence
of these timing statistics.

See [the empirical ML study](effort_ml_study_20260913.md) for what the
collected data does and does not justify.

## Portable research predictors

### Mixed candidate pools (experimental)

Set `candidate_mix="mixed"` on `AutotunePolicy` to reserve candidate-pool
slots for broader structural exploration alongside retrieval and local walks.
The default remains `"legacy"` pending matched multi-device evaluation.
The mixed pool reserves approximately one quarter each for retrieval and
structural candidates, with remaining slots used by local candidates. Empty
quotas are backfilled. Every candidate must pass current static legality;
the adapter's active-schedule identity removes duplicates across sources.

`structural_proposals=512` bounds construction attempts, not GPU trials.
Coupled assignments may cross illegal intermediate states, but only legal
complete configurations can reach evaluation. One structural attempt replaces
the reserved random attempt when a structural pool is available. UCB controls
the remaining allocation, including local/random and optional learned arms.
The pool's origins, counts and rejection summary are recorded in session metadata.
Construction and ranking still cost time; diversity does not guarantee speed.

Explicit caller seeds have priority over ordinary pool fill. The experimental
`baseline_trials=2` option reserves two local-search arm pulls within the
cumulative trial budget; it does not reproduce the baseline's entire adaptive
trajectory. It defaults to zero. Deadlines and exhausted legal neighbors can
still prevent the reservation from being fulfilled.

New audited histogram/forest artifacts use `predictor_heads=("trees",)` for
plain tree scores without a memory-roofline offset. Like the existing heads,
they require research opt-in and an exact compatible implementation contract.
The runtime remains NumPy-only. Full ranked pool IDs are logged once per learned
arm, with the selected candidate's rank attached to each proposal.

See [the evidence pipeline](autotune_evidence_pipeline.md) for immutable ingestion,
portfolio construction, grouped evaluation, and frozen prospective validation.

Explicit searches sharing a residual and exact context now serialize the
read/search/write transaction on Linux, preventing duplicate cumulative-budget
spending by concurrent callers. This is separate from runtime-winner locks.
Different residual directories intentionally remain independent experiments.

The exported residual-tree and small-MLP heads can participate as separate
UCB proposal sources. Inference uses NumPy only: no sklearn, XGBoost, Torch
model, pickle, or extra GPU work is needed to rank candidates.

```python
policy = rtx.AutotunePolicy.preset(
    "balanced", max_trials=16,
    predictor_artifact="autotune_priors/tournament_predictors_v1.json.gz",
    predictor_heads=("residual", "mlp"),
    allow_research_prior=True,
)
layer = rtx.NVFP4Linear(
    768, 3072, bias=False, device="cuda",
    tuning_policy=policy, autotune_cache_dir="autotune_cache",
)
# The first real torch.compile(fullgraph=True) invocation can tune both passes.
```

This is opt-in, not a promoted default. The opening trials include the initial
configuration and nearest-shape historical winners; UCB then selects among
residual ranking, MLP ranking, coordinate/local search and random exploration.
Historical-winner retrieval uses shape distance, not the tournament's separate
KNN predictor. Hardware/configuration interactions enter the learned rankings.
When both artifact types are supplied, `prior_artifact` supplies retrieval.

Family, kernel revision, cache regime and workload/scaling policy must match.
Missing campaign-only policy tags are completed only when exactly one trained
contract remains compatible. Unknown or ambiguous contracts fall back to ordinary
search; malformed artifacts and absent research opt-in raise an error. Matching
a contract does not certify transfer to an unseen GPU. The low-level predictor's
`predict` method only scores rows; adapter-facing `order` performs these checks.

Both heads rank a bounded pool once per search. Their proposals still undergo
normal legality, numerical, timing and confirmation checks. A bad prediction
can waste a trial but cannot replace the incumbent merely because of its score.
Artifact hashes and head names are recorded with observations. Loading an
artifact does not invalidate existing runtime winners or force retuning.

See [export and CUDA validation evidence](portable_predictor_validation.md)
for reproduction, known limitations and the next qualification gate.

The [prospective comparison harness](predictor_prospective_study.md) tests
8/16/32-trial policies on previously unobserved geometries without training on
the resulting observations.
