# Post-merge qualification

Work in progress; this is not a release announcement. The GLU integration was
published at `24d13e6`. Follow-up campaign/setup fixes are on `main`.

## Latest completed qualification

The following supersedes the earlier campaign status recorded below. It is
not a claim that every release gate is closed.

- At `6886304`, the local Torch 2.12.1 suite passes 1,743 tests and 1,064
  subtests (eight skips, 22 optional DINO studies deselected). Its 49-case
  production matrix, prospective Ettin inference and short training also pass.
- The matching Torch 2.13 suite passes 1,741 tests and 1,064 subtests, with
  two failures: intermittent 23,040-byte live-allocation fluctuation in the
  GLU ownership test, and a short-budget persistent-winner assertion. Its
  production, Ettin inference and short-training checks all pass.
- The 5060 Torch 2.12.1 suite at runtime revision `2c1d9e0` passes 1,735 tests
  and 1,057 subtests, with three failures: concurrent tuner-call accounting,
  the short-budget winner assertion, and singleton timing eligibility. These
  failures do not report numerical mismatches. The device is not fully qualified.
- The rebuilt `6886304` wheel passes all 49 production cases in each genuinely
  clean dependency environment, outside the checkout. Build, twine and
  dependency checks pass. Archive inspection finds 136 wheel files and 605
  sdist regular files, with no dataset, checkpoint, log or environment paths.

The cache integration test had overridden balanced tuning with four trials
and three seconds while unconditionally demanding persistent winners. With
the public balanced defaults, three fresh-process repetitions per local stack
pass winner writes and zero-tuner-call cross-process reuse. The 5060 also passes
this corrected test on Torch 2.12.1. Numerical and timing-eligibility gates are
unchanged; forced-noise tests still prohibit persistent unmeasured fallbacks.

The isolated ownership test passes eight traced repetitions, and another eight
after preceding MLP tests. The original 23,040-byte fluctuation was not
reproduced. That preceding-test run instead exposed a deployment memory
assertion failing when allocation decreased. Memory checks now collect
unreachable Python cycles before sampling the retained footprint, without
clearing live owners, graphs or caches. The original equality/4 KiB gates remain
unchanged; the ownership check also records raw pre-collection allocation.
This is a test-measurement correction, not evidence of a repaired kernel leak.
All 34 focused ownership/deployment tests pass on each supported local Torch
stack with the revised protocol; this is separate from the original full run.

Artifacts reside under `autotune_reports/postmerge_20261024/`: the two
`final_6886304_*` bundles, `full_2c1d9e0_212.xml`, `installed_6886304_*.json`,
`dist-6886304-audit.json`, `balanced_default_*` and `ownership_trace_*`.
The 5070 laptop remains unavailable and unqualified; no release tag or upload
has been made. Remaining timing/concurrency checks and memory-test requalification
must not be replaced by the green production matrix alone.

## Matched longer training trajectories

Two RTX 5090 workers completed all eight policies for 256 compiled steps each,
using 1,024 WikiText training paragraphs and 128 validation paragraphs at
sequence length 256. Within a worker, checkpoint, masks, data order, optimizer
and seed are identical across policies. Parameters, gradients and AdamW moments
are BF16; cross entropy is FP32. All checked gradients and optimizer states
remained finite. Validation uses copies to preserve delayed training state.

| Policy | Torch 2.13, seed 4810: initial → final NLL | Torch 2.12.1, seed 5810: initial → final NLL |
|---|---:|---:|
| BF16 | 2.0828 → 2.0489 | 2.1350 → 2.0954 |
| MXFP8 dynamic | 2.0972 → 2.0582 | 2.1427 → 2.1142 |
| NVFP4 JIT-region | 2.3803 → 2.3573 | 2.4392 → 2.4124 |
| NVFP4 delayed | 2.3878 → 2.3573 | 2.4568 → 2.4114 |
| NVFP4 current | 2.3876 → 2.3536 | 2.4568 → 2.4240 |
| NVFP4 block | 2.3757 → 2.3723 | 2.4322 → 2.4470 |
| NVFP4 block Four Over Six | 2.3476 → 2.3312 | 2.4246 → 2.4115 |
| MXFP8 recomputed MLP | 2.0976 → 2.0628 | 2.1541 → 2.1160 |

These are two matched within-run comparisons, not an experiment isolating
Torch-version effects: seed and version differ together. The checkpoint is
already pretrained and this is a short fixed-mask fine-tuning trajectory, not
training from scratch or long-run convergence certification. Block-only worsens
on one validation set. The substantial NVFP4/BF16 gap remains. Per-step timings
include compilation, synchronous gradient/state checks and logging; they must
not be reported as production throughput.

Reproduce using `benchmarks/run_postmerge_validation.sh`; raw JSONL, checkpoint
revision and source identities are retained in the output directory.

## First-hit refinement finding

The small-shape 8/16/32-trial sequence and fresh-process reuse passed both
stacks. The larger `M,K,H,O=2048,768,1536,768` case exposed a runtime bug at
8 → 16 trials: injecting the previously deployed winner into the opening
portfolio changed the progressive checkpoint key. The correction preserves
the opening portfolio and uses the winner only as the deployment comparison
baseline; its measured observation remains in history. Checkpoint compatibility
validation is not weakened. Resuming the actual failed journals at 16 and 32
trials now passes on both Torch stacks, with exactly 16/32 observations and one
compiled graph. A fresh-process 32-trial reuse adds no observations and invokes
no tuner. The targeted MLP first-hit/graph/gradient and progressive-search
regression suites pass 15 tests on each supported Torch stack. The ragged
`511,384,576,384` budget progression and fresh-process reuse also pass both
stacks. Laptop qualification remains in progress.

## Linear first-hit budget sweep

All 54 compiled inference probes passed: three families (MXFP8 materialized,
NVFP4 block, NVFP4 JIT-region), three M,N,K shapes (`256,768,768`,
`2048,768,1536`, `511,576,384`), three progressive budgets (8/16/32), and
both supported Torch stacks on separate RTX 5090s. Each probe checks finite
outputs against a BF16-cast FP32 reference, trial accounting, warm reuse and
CUDA-graph capture/replay. Error gates remain NRMS <0.07 for MXFP8 and <0.20
for NVFP4. This is inference coverage, not backward qualification.

Median first-call times across the nine family/shape cells were 3.98/4.69/6.65
seconds for Torch 2.13 and 4.35/4.75/6.91 seconds for Torch 2.12.1 at
8/16/32 trials respectively. The later budgets resume previous observations;
their times are additional refinement costs, not independent cold-start totals.
Compiler disk caches were not cleared, and process startup is excluded.

Most replay timings were approximately flat with additional effort. One
Torch 2.13 ragged MXFP8 cell changed from 10.24 to 8.18 microseconds (about
1.25x), while most cells changed by less than 2%. These are unpaired
cross-process measurements, not proof of monotonic speedup or global optimality.
The two stacks sometimes selected different schedules with materially different
latencies; that does not isolate a Torch-version effect. No research winner
was installed into the user's default runtime cache.

## Cross-device and installation status

An isolated source snapshot on the RTX 5060 laptop is running release
qualification. The user confirmed the 5070 laptop is unavailable; its interrupted
results do not count as qualification, and no further work is waiting on that
machine. Its cross-device release gate remains outstanding. Initial attempts
had missing test dependencies and an import-path
setup error; those are failed attempts, not kernel passes. The corrected runner
explicitly imports its own checkout, and Pillow is now a declared test dependency.
Both initial Ettin model/short-training subprocesses passed, but their parent
qualification runs did not. The 5060's subsequent partial suite was deliberately
stopped to include the runtime refinement fix. The 5070 WSL instance restarted
externally during its partial suite; the cause is not established. Those partial
logs remain intact. Full suites restarted at `7ec987e` with four compiler workers
by default (caller-overridable), avoiding excessive parallel compiler memory use.
Both local full suites finished with 1,730 passes, eight skips and one failure
each (plus 1,063 passing subtests). Torch 2.13's preparation-refinement assertion
incorrectly required unique configurations even when the earlier measurement
was explicitly nonstationary; the journal proves that the duplicate was a
legitimate retry, not redundant stable exploration. Torch 2.12.1's singleton
coordinate test had no eligible timing window. Test corrections and GPU
requalification are in progress; stationarity and numerical gates are unchanged.
The 5060 full suite finished with 47 failures (including subtest failures),
1,714 passes, 14 skips and 1,021 passing subtests. Its separate 49-case production
matrix, Ettin inference and short-training subprocesses exited successfully.
Open failures include timing eligibility/cache assumptions and a single BF16
output discrepancy repeated across fused MXFP8 schedules. Additional portability
failures were missing optional `nvidia-smi` on PATH, links to unshipped local
reports, and an OOM test whose 2x-VRAM allocation succeeded under WSL. These are
not blanket kernel failures, but the device is not fully qualified.

The two local timing-contract corrections pass three fresh-process GPU repeats
per stack. The OOM observer test now uses a disposable worker's allocator limit
instead of physical VRAM capacity and passes both local stacks. Optional
telemetry now supports WSL's executable location and reports unavailable or
timed-out monitoring without aborting the numerical benchmark; its CPU tests
and both three-way GPU comparison modes pass locally. Laptop retests remain
necessary. The queued second-stack laptop campaign was stopped before launch
to prioritize this triage; no unrelated workload was stopped.

An operational error affected the local shell wrappers: editing their script
while they were running changed the continuation read offsets. This overwrote
the Torch 2.13 text test log and recorded a shell exit of 127 in both status
files. The completed JUnit XML files preserve the actual test results and
tracebacks. Do not interpret the shell exit as a kernel failure or a test pass.
No more runner edits were made while either wrapper remained live.

Both local production matrices separately passed 49/49; their Ettin prospective
inference and short-training processes also exited successfully. These passes
do not erase the outstanding full-suite and laptop gates.

The built wheel imports MXFP8Linear, NVFP4Linear and MXFP8MLP outside the source
tree and passes runtime version validation. That check reused the qualified
environment's dependency directory; it is not a fresh network dependency solve.
Both dependency requirement files have now installed successfully into genuinely
fresh virtual environments. Both Torch stacks pass `pip check`, runtime version
validation, CLI help and all 58 public-export imports outside the checkout.
Both fresh environments subsequently passed the 49-case production GPU matrix
outside the checkout. Separate installed-wheel GLU checks pass eight cases per
stack: SiLU/GELU, aligned/ragged and eager/fullgraph compiled forward plus all
three gradients against the independent quantized recipe (NRMS gate 0.025).
These are small-shape installation checks, not a replacement for full GLU tests.
The `c728aae` wheel/sdist pass build and twine checks;
archive inspection found no ignored source artifacts (136 wheel / 622 sdist
entries). No release tag or package upload has been made.

A subsequent rebuild from `2034ecb` plus this documentation update passes build,
twine, wheel namespace and ignored-file checks (136 wheel files, 605 sdist
regular files; generated egg-info is excluded from the ignored-file audit).
The source archive includes the immediate-failure plugin and qualification
runner. This rebuild is separate from the wheel used by the fresh installs.

Subsequent qualification runs record failures immediately in
`test-events.jsonl`, including collection failures and tracebacks. The optional
pytest plugin refuses existing output files and fsyncs failure/final records.
Its passing/failing-probe and overwrite-protection tests pass on both stacks.
This does not change any numerical or runtime gate.

## Follow-up failure localization

The 5060 portability retests pass on both Torch stacks: 19 tests and 251
documentation subtests each, including the allocator observer and both
replacement-benchmark execution modes.

The MXFP8 discrepancy was a reference error, not a kernel error. At `[91,187]`
the exact quantized dot product is `4538369/8388608 = 0.5410157442092896`.
All five diagnostic schedules return the correctly rounded BF16 `0.54296875`;
the FP32 GEMM reference instead lands on `0.541015625`, a midpoint rounding to
`0.5390625`. Captured inputs, exact rational sums and cross-reference metrics
are retained in `rounding_5060_fd2360a`. The strict schedule test now uses FP64
accumulation and explicit nearest-even BF16 rounding without an intermediate
FP32 conversion. Its zero-tolerance assertion is unchanged. All 34 schedules
pass on the 5060 and 5090 on both supported Torch stacks.

A separate runtime bug arises when a resumed history exhausts its budget with
no stable timings: no seed has been checked on the current invocation. The
runtime now offers lazy, current-input correctness-only seed validation, with
a separate journal session and no extra timing trials or installed winner.
Offline tuning remains strict. GPU regressions pass on both local stacks,
including changed finite operands, unchanged trial journals, and rejection of
NaN operands. Both local 49-case production matrices also pass after this fix.
The laptop's affected runtime cases still require retesting.

The laptop backward fallback regression and the previously failing NVFP4
backward cases now pass. A remaining Torch 2.13 functional-forward case exposed
the same exhausted-history issue in the dimension-based NVFP4 tuner. The lazy
validation mechanism now covers that tuner using its normal harness probe;
it does not claim application-data validation. A deterministic noisy-history
GPU regression passes both local stacks and verifies zero extra timing trials,
unchanged observation journals and no persistent winner.

The remaining-runtime sweep at `a6abc2f` passed 14/17 tests on Torch 2.13 and
15/17 on Torch 2.12.1 on the 5060. It exposed the analogous missing runtime
fallback wiring in packed MXFP8 inference; that path and fused MXFP8 forward
now use the same checked fallback. Forced-noise GPU tests cover packed W,
packed X/W and fused forward without permitting persistent unmeasured winners.
The larger progressive training-cache test passed on both laptop stacks in
this sweep; the short first-hit cache-write test remained timing-inconclusive.

A diagnostic host-timing sweep compared minimum batch windows of 2/8/32 ms,
3/9 samples and three repetitions for forward and backward at 128x128x128.
The original timing gates accepted 8/36 windows on the 5060 versus 32/36 on
the 5090. Longer windows did not consistently improve laptop stability, so an
experimental automatic long-window retry was discarded, not shipped. These
small studies describe measurement reliability, not kernel speedup or SKU-wide
performance. The original timing thresholds and measurement mode remain intact.

At `2c1d9e0`, the 5060 passes all three targeted noisy-history fallback tests
and three subtests on each supported Torch stack. This includes packed MXFP8
W, packed X/W, fused forward, backward current-input validation and the
dimension-based NVFP4 harness fallback. The corresponding XML artifacts are
`autotune_reports/postmerge_20261024/fallback_2c1d9e0_{213,212}.xml`.
These passes establish checked fallback behavior, not stationary timing or
persistent-winner qualification.

Process-only timing experiments on the 5060 accepted 11/36 windows with
Python garbage collection disabled and 10/36 with CPU affinity restricted to
core 0. Neither established a reliable fix relative to the original 8/36.
No library GC, CPU-affinity, timing-threshold or measurement-mode change was
made. The 5070 laptop is unavailable by user confirmation; its unfinished
qualification remains explicitly outstanding.
