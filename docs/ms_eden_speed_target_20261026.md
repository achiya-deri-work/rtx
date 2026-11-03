# NVFP4 backward speed target: execution record

Historical research notebook, consolidated on November 3. Operational status
below is dated evidence, not current instructions. Start with
[the consolidated research index](nvfp4_research.md) for retained interfaces
and subsequent full-model qualification.

Target: `T_MXFP8 / T_NVFP4 >= 1.20` for complete public compiled dX+dW,
including preparation and RNG, on representative decoder projections. This
means 20% greater throughput, not 20% less latency. A candidate must retain
original BF16 operands, FP32 scaling/correction, numerical and allocation
guards. Stable defaults are not changed by this experiment.

## Measurement and current work

`benchmarks.compare_ms_eden_pipeline --baseline-policy stable_mxfp8` compares
against the stable public backward with the same exact NVFP4 forward. It
records the baseline's resolved configuration; omitted `--mxfp8-config-json`
means repository default, not an exhaustively tuned MXFP8 optimum. The speed
acceptance field requires stationary backward timings and confidence lower
bounds at least 1.20, including separate AB/BA order checks. This timing gate
does not replace exact numerical qualification or training comparison.

Initial baseline sweep: four 21M-decoder projection shapes, wide and long
controls, pure paired MS-EDEN schedule. None qualified for the target.
Raw baseline artifacts are in
`autotune_reports/ms_eden_speed20_20261026_5060/baseline/`.

The next wave uses 24 trials per shape, six shapes, complete-backward workers,
3000 ms warmup and persistent JSONL residuals. It searches FP4 schedules;
winners must subsequently be compared directly against MXFP8 in fresh paired
runs. The search portfolio now includes paired+async-2/3 and interleaves dX
and dW exploration instead of exhausting a small pool on dX alone. It also
covers coupled producer/finalizer resource geometry and independent GEMM
challengers. No arithmetic, RNG distribution or precision gate was weakened.

## Remote ownership

- Host: `personal_laptop` (5060), exclusively assigned GPU at launch.
- Snapshot/output: `/home/achiya/rtx-mseden-speed20-20261026.CjAIUF`.
- Python: `/home/achiya/rtx-nvfp4-20261022/.venv/bin/python`.
- Baseline: `run-baseline.sh`, `launch-baseline.log`, `baseline/*.json`.
- Tuning: `run-tune.sh`, `launch-tune.log`, `tune-*.log`, `tuning/*/residuals/`.
- The tuning launcher waits for the baseline sweep before using the GPU.
  It was detached with nohup/disown; do not launch a concurrent benchmark.

Next structural experiment: independently tune A/B preparation geometry
within each gradient leg so small weights and large activations do not share
one resource allocation. This passed the focused 5070 GPU suite (10 tests,
including four GPU cases covering dX/dW and inline/staged scale maxima).
Local development
must not be copied over the active immutable remote snapshot mid-run.

## Parallel 5070 qualification and experiments

The 5070 became reachable and idle during this run. Its isolated snapshot is
`/home/achiya/rtx-mseden-geometry-20261026.JlSpFv`, using
`/home/achiya/vs_code_main/rtx/.venv/bin/python`. Pytest was installed into
snapshot-local `.testdeps`, not the shared environment. Qualification logs are
saved locally in `autotune_reports/ms_eden_speed20_20261026_5070/`.

An additional native FP4 pair-decode experiment passed seven focused tests:
all 256 packed byte pairs against an independent FP32-bit oracle (including
signed zero), plus byte-exact producer/finalizer output and captured
changed-buffer replay. This is opt-in and has no measured performance claim
yet. It changes only the local decode for correction, not FP32 statistics.

The 5070 is now comparing baseline and smaller dX-weight preparation grids
against matched MXFP8 backward on the four decoder shapes, using
`run-comparisons.sh`, `launch-comparisons.log`, and `comparisons/`.
No GPU benchmark runs concurrently with another benchmark on the same host.

Native decode is now wired through immutable schedules, compilation, runtime,
study helpers, and autotune identity/features. `correction_decode="lut"` remains
the default; `"native"` is an explicit alternative. Preparation context revision
11 and feature revision 6 separate the new evidence from older configurations.
A second immutable 5070 snapshot,
`/home/achiya/rtx-mseden-native-20261026.TtVcCh`, queues `run-native.sh` behind
the geometry sweep. It requalifies the integrated path, then measures both
isolated serial LUT-vs-native backward and paired x4/async-2/native against
stable MXFP8 across six shapes. See `launch-native.log`, `qualification.log`,
and `comparisons/` there. Do not infer success before those outputs exist and
pass their guards.

First 5060 shape completed its 24 trials: selected paired x4/async-2 at
141.313 us, versus the initial FP4 trial's 163.581 us. These are search
observations, not an independent winner confirmation or a speed-target win.
The search proceeded to the next shape normally.

### Power-state qualification boundary

A later read-only check found the 5070 reporting AC offline, approximately
1425 MHz SM clock and 9001 MHz memory clock. The 5060 reported AC online,
2265 MHz SM clock and 11001 MHz memory clock. No power settings were changed.
5070 timing artifacts from this sweep are therefore provisional diagnostics,
not acceptance evidence for the speed target; correctness results remain
valid. The first two shapes showed no useful gain from the tested smaller
weight grids. No geometry default is promoted.

The verified 5070 native-sweep launcher (PID 4439 in the native snapshot)
was terminated to prevent additional battery-powered timing jobs. Its
already-started trial was allowed to finish; results were not deleted.
Resume only missing trials into fresh artifact names after confirming AC.
The 5060 six-shape search remains running, unmodified and on AC power.

The user subsequently reported plugging in the 5070 but requested that work
remain on the 5060 while the 5070 undergoes changes. No further 5070 jobs or
remote checks are scheduled.

## Next 5060 confirmation wave

Snapshot `/home/achiya/rtx-mseden-confirm-20261026.CxKq8A` contains an additional
opt-in `partial_max_scope="cta"`: original per-warp traversal is retained,
but a single reduction at each producer CTA's end emits one partial rather
than one per warp. This reduces subsequent repeated partial scans without an
additional global kernel. Defaults remain warp scope. It requires exact GPU
qualification before performance evidence is accepted.

`run-confirm.sh` waits for the existing search launcher, then runs focused
CTA/native/operand-geometry/paired GPU checks. Only if they pass does
`confirm-winners.py` load each selected immutable configuration and compare
four variants (unchanged selection, native decode, CTA partials, both) against
stable MXFP8. Each worker is fresh, logs full configuration provenance, and
uses 12 paired rounds plus 5000 ms warmup. Outputs live in `confirmations/`;
`launch-confirm.log` and `qualification.log` record execution and failures.
This queue is detached and must not overlap another 5060 GPU job. No result
from this queued wave is yet claimed.

The search subsequently completed all 144 trials. The queued exact suite
passed **64 tests** on the 5060; fresh comparisons then began. CTA partials
are therefore GPU-correct over the tested surface, not yet a proven speed win.

## Guarded arithmetic wave (October 27)

Snapshot `/home/achiya/rtx-mseden-arithmetic-20261027.OhrWPy` queues
`run-arithmetic.sh` after the preceding confirmation launcher. It first runs
the explicit GPU probability/refinement tests. Only after success does
`confirm-arithmetic.py` compare control, power-of-two probability scaling,
guarded normalization refinement, and both together against MXFP8.

It selects a prior candidate by the best paired confidence lower bound among
stationary comparisons; if none is eligible it retains the original search
selection. That choice is remeasured, not treated as an accepted winner. The
four narrow decoder shapes, wide and long controls remain. Two additional
normal decoder workloads (`4096×6144×1536`, `8192×1536×1536`) test larger-width
amortization; they do not replace narrow-shape failures. Transferred starting
schedules and exact provenance are explicit in each manifest.

Artifacts: `launch-arithmetic.log`, `qualification.log`, `comparisons/`.
The snapshot includes probability math and refinement options but keeps their
defaults at ordinary division. The exhaustive CPU refinement test is opt-in;
ordinary GPU qualification does not repeat the exhaustive CPU sweep.

The 5090 remains occupied/inaccessible for workload management; local AFRL
GPUs are not used. Training translation and multi-session winner confirmation
remain required before declaring the new speed goal achieved.

## Exact shared-reciprocal normalization experiment (CPU qualified, GPU pending)

`normalization_math="refine"` is a new opt-in producer experiment; the default
remains `"divide"`. It replaces the four per-fragment normalizations with one
**precise** `div.rn.f32(1,m)` shared outside the pair loop, power-of-two
normalization, and two explicit rounded FP32 FMAs per value. It does not change
maximum/6, EDEN correction, Hadamard arithmetic, source policy, packing, or
probability math. Original/packed compilation and runtime/study preparation
forward the option. Preparation context 14 / structural-feature revision 9
separate this evidence from prior configurations.

For a positive normal scale `s=m*2^E`, `m=t/16`, `16<=t<=31`, form the exact
inverse power of two and, for each value, compute:

```text
xN = mul.rn(x, 2^-E)
r  = div.rn(1, m)                 # once for all four fragment values
q0 = mul.rn(xN, r)
e  = fma.rn(-q0, m, xN)
q1 = fma.rn(e, r, q0)
```

The fast path requires scale exponent field 1..253, low 19 mantissa bits zero,
finite original x, and `2^-90 <= abs(xN) <= 16`. Eligible signed zeros return
the original x bits. All other cases use the original explicit `div.rn.f32`.
Exponent field 254 is deliberately excluded: its inverse power of two is
subnormal and cannot be constructed by simply shifting a biased exponent.

Proof sketch: power-of-two normalization is exact for the accepted normal xN.
Let `u=2^-24`, `r=(1+delta)/m`, and `q0=(xN/m)*(1+epsilon)`. Then
`abs(delta)<=u`, `abs(epsilon)<=2u+u^2`. The exact residual is on the lattice
`2^(exponent(q0)-27)` and has at most seven significant bits after cancellation;
its smallest nonzero magnitude is at least `2^-118`. The first FMA therefore
returns it exactly without underflow. Before the final rounding,
`q0+e*r=(xN/m)*(1-epsilon*delta)`, whose error is less than
`(2^-23+2^-48)` quotient ULP. After removing powers of two from t, the remaining
denominator is odd and at most 31: an exact quotient is never an FP32 midpoint,
and its distance from a midpoint is at least `1/62` ULP. The refinement error
cannot cross that boundary, so the final rounded quotient equals precise
division. Negative values follow by sign symmetry; zero is handled explicitly.

CPU evidence obtained with NumPy 2.4.4:

- 805,306,368 comparisons: every one of the 2^23 significands, both signs,
  all 16 denominators, normalized exponents -90, 0, and 3. Zero quotient-bit
  mismatches against FP32 division and zero nonexact residuals. The read-only
  vectorized CPU prototype took 3.15 seconds; this is validation cost, not GPU
  performance evidence.
- 10,487,520 additional original-x/scale cases spanning scale exponents
  -126,-125,-90,-1,0,1,90,125,126,127, original subnormals, signed zeros,
  infinities, NaNs, and fallback cases. Of these, 2,515,680 used refinement;
  zero non-NaN bit mismatches, including zero signed-zero mismatches.
- The CPU sweep uses binary64 intermediates followed by FP32 casts. Treat
  this as supporting evidence, not an independent proof that every modeled
  FMA equals hardware FMA; the explicit GPU division oracle is required.

Reproduce the significand sweep with
`RTX_RUN_REFINE_EXHAUSTIVE_CPU=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_ms_eden_normalization_refine.py`.
The independent GPU oracle uses explicit `div.rn.f32`, exercises all 16
denominators and fallback edges, and checks producer/finalizer bytes plus
changed-buffer graph replay. Run it only on the assigned GPU using
`RTX_RUN_REFINE_CUDA=1 python -m pytest -q tests/test_ms_eden_normalization_refine.py`.
The 5060 arithmetic qualification subsequently passed **19 tests, 1 skipped**
across probability math and normalization refinement. The log is
`/home/achiya/rtx-mseden-arithmetic-20261027.OhrWPy/qualification.log`.
Complete-backward comparisons are running; no promotion is justified.

This is different from the archived negative exact-integer-normalization
prototype: it has no per-value integer quotient/remainder repair or rounded
quotient bit construction. It shares one correctly rounded reciprocal and uses
a short, bounded-error floating-point refinement with exact residuals. Saving
three precise divisions may help, but four normalization multiplies, eight
FMAs, guards, and extra live values can offset it. Accept only exact GPU
qualification followed by stationary, complete public backward comparisons
against stable MXFP8; do not infer the 1.20x target from arithmetic counts.

### October 27 arithmetic results and frontier confirmation

The completed decoder comparisons reject refinement as a universal speed
improvement. On decoder1536, control measured 335.6 us versus refinement
355.5 us; on decoder2048, 421.9 versus 442.8 us; on decoder1024,
208.1 versus 223.0 us. These are separate-session medians, not paired
control-versus-refinement confidence intervals. Their matched MXFP8 ratios
also deteriorated. None passed the 1.20x target. Artifacts are retained under
`autotune_reports/ms_eden_speed20_20261026_5060/arithmetic/`.

The selected wide-shape schedule used a 64-block preparation grid. Several
faster 256-block candidates had passed numerical gates but were excluded by
stationarity checks. They require fresh confirmation, not weaker gates.
Queued in the same qualified snapshot after arithmetic launcher PID 31372:
`run-frontier.sh`, PID 36793, logging to `launch-frontier.log`.
It tests three original wide configurations (x8 async-2, x4 async-2, and
synchronous wide loads), each with LUT/native decoding, twice in fresh
processes with reversed candidate order, 16 timing rounds and 5-second warmup.
Both repetitions use the same data seed; this checks session repeatability,
not independent-data numerical generalization. Results go to
`frontier-confirmations/`. Stable defaults remain unchanged.

Hardware scope: only personal_laptop/5060. The user explicitly excluded the
5070 during ongoing machine changes, even though it is now plugged in.

### Next structural experiments (not yet qualified)

1. **Async copy width.** The transpose preparation loader currently issues
   32-bit cp.async copies. Optional 64/128-bit copies can reduce instruction
   count while keeping the same BF16 source, FP32 arithmetic and packed outputs.
   Wider alignment also changes shared-memory bank conflicts, so fewer copy
   instructions alone do not establish a speedup. Test identical outputs and
   full backward latency; preserve the original width as default.
2. **FP32 split-K dW.** The shared MXFP8 GEMM base has split-reduction support,
   but the NVFP4 compiler currently constructs only unsplit BF16 output.
   With 128x128 CTA tiles, a 512x512 dW has only 16 output CTAs on the 26-SM
   5060. Research-only split-K can increase available CTAs; partials must remain
   FP32, the FP32 outer scale must be applied exactly once, and BF16 conversion
   must occur only after the final reduction. Added workspace/reduction traffic
   may outweigh occupancy gains. This is an implementation hypothesis, not a
   measured speedup, and it does not erase the larger preparation bottleneck.

Copy-width implementation is now frozen in the isolated 5060 snapshot
`/home/achiya/rtx-mseden-copywidth-20261027.Zb91wk`. The launcher
`run-copywidth.sh` (PID 41078) waits for frontier launcher PID 36793, then
runs `RTX_RUN_ASYNC_COPY_WIDTH_CUDA=1 ... pytest -q
tests/test_ms_eden_async_copy_width.py` before any timing. Eighteen GPU cases
cover exact outputs, multiple buffering depths, idle/tail CTAs and changed
graph inputs. The CPU-masked regression was 96 passed, 121 skipped; skipped
GPU cases are not qualification. If qualification passes, compare widths
32/64/128 on decoder512, wide and long, holding the paired x8/stage-2 schedule
fixed. Logs: `launch.log`, `qualification.log`, `comparisons/`.

The training harness now accepts `--ms-eden-schedules path.json`, an explicit
map from canonical `M,N,K` keys to `{dx, dw, backward_pipeline}`. It rejects
missing/extra model shapes, duplicate keys and conflicting CLI overrides.
Resolved contents, not only a filename, enter checkpoint identity; forward
policies and initialized weights are unchanged. This enables actual tuned
schedules to be used in the required matched training follow-up. CPU tests:
36 passed across `test_precision20m_schedule_map.py` and
`test_precision20m_training.py`. This is harness validation, not new GPU
training evidence.

### First repeated timing target pass: wide shape

The frontier confirmation completed on 5060. On `4096x2048x2048`, paired
x8/async-2 preparation with native FP4 correction decoding passed the unchanged
timing target in **two fresh processes**:

| Session | MXFP8 median us | FP4 median us | Paired speed ratio | Cluster-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| r0 | 1025.2 | 848.8 | 1.2047 | [1.2030, 1.2063] |
| r1 | 1009.4 | 837.2 | 1.2083 | [1.2058, 1.2107] |

Both passed all AB/BA stationarity and lower-bound timing checks, alongside
the comparator's current-buffer, gradient, forward-match and allocation gates.
The ratio is the paired geometric estimator, not the ratio of displayed
medians. These are same-seed session repetitions and the baseline is the
recorded default MXFP8 backward, not a newly exhaustively tuned MXFP8 winner.
Artifacts: `autotune_reports/ms_eden_speed20_20261026_5060/frontier/`, filenames
`wide-de88abf02ed36552018ee7e0-native-r{0,1}.json` plus manifests.
This is **one shape**, not completion of the multi-shape/model-training goal.
The x8/LUT alternative was 1.192x and 1.189x; x4/LUT was 1.169x and 1.166x.
No stable policy default changes.

The queued copy-width GPU qualification subsequently completed **24 passed**
in 17.72 seconds (including 18 GPU cases). Its nine complete-backward
comparisons are now running. The first narrow-shape runs still missed the
timing target; final per-shape comparisons remain to be analyzed.

### Copy-width outcome and split-K public qualification

All nine copy-width comparisons completed. Wider copies were not promoted:
128-bit copies regressed, including stationary long-shape ratio 0.808x versus
0.893x for 64-bit copies. The 32-bit long sample was nonstationary (0.906x),
so it is not a certified direct comparison. Wide 64-bit measured 1.179x and
was stationary; other wide copy-width sessions were nonstationary. These
results do not establish a benefit over the retained 32-bit native-decode
winner. Raw records and exact qualification log are in the local `copywidth/`
report subdirectory. Shared-memory bank conflicts are a plausible explanation
for the wide-copy regression, not a counter-verified diagnosis.

Split-K primitive qualification passed **25 tests** on 5060, including
independent unpacked FP64 GEMM references, raw FP32 partials, unequal final
partitions, scale application exactly once, changed-input graph replay and
live allocation checks. Source snapshot:
`/home/achiya/rtx-mseden-splitk-20261027.7xGFHq`.

Public split-K integration then passed **11 tests** (32 deprecation warnings)
in 16.30 seconds. This covers paired compiled backward, current-buffer/RNG
guards, ordinary/graph memory checks and a separate packed-operand numerical
gate. Active immutable snapshot:
`/home/achiya/rtx-mseden-split-public-20261027.yin0fp`.
Launcher `run-public.sh` PID 43945 / driver PID 44140 now measures all six
original decoder shapes with dW split counts 1, 2, 4 and 8, interleaving shapes
within each wave. The native x8/async-2 baseline and 32-bit copies are held
fixed. Splitting changes the dW epilogue to direct FP32 partial output plus a
separate scaled BF16 reduction; dX remains unsplit. Measurements include that
extra workspace/reduction cost and all preparation/RNG. No split-K end-to-end
speed claim is justified yet.

The complete comparator and training harness now also record the NVFP4 and
shared MXFP8 GEMM source hashes. Focused CPU training/comparator regressions:
50 passed. The matched short training follow-up remains pending candidate
selection; one wide-shape timing pass is not evidence of model throughput.

### Split-K interim result and next exact launch reduction

Split-K did not solve narrow-shape latency. Completed stationary paired
MXFP8/FP4 ratios for dW splits 1/2/4 are decoder1536: 0.810/0.744/0.684;
decoder2048: 0.962/0.895/0.805. Decoder512 split-2 is 0.682 versus unsplit
0.722, and split-8 falls to 0.576. Long split-4 reaches 0.955 but still misses
the target (its unsplit session was nonstationary). Remaining split-8 samples
are still running. This rejects split-K as a general default for this paired
workload. A small dW grid alone does not imply the complete paired path lacks
parallelism: dX also runs, and splitting adds workspace traffic/reduction and
changes the epilogue. No isolated-GEMM improvement is being substituted for
the complete-backward result.

The next exact experiment is `outer_factor_mode="pair"` (default remains
`"product_kernel"`). Operand finalizers write two adjacent FP32 slots. GEMM
or the split reducer computes explicit `mul.rn.f32(A_factor,B_factor)` before
its existing accumulator scaling, preserving the separately rounded FP32
product. This removes exactly two product launches from full dX+dW; it does
not fold scales into E8M0 or claim a particular latency saving.
CPU regression: 127 passed, 139 skipped. GPU work is queued behind split-K:
`/home/achiya/rtx-mseden-factorpair-20261027.caa1GY`, launcher PID 47907,
`run-factorpair.sh` / `launch.log`. Exact direct/TMA/split and public graph
qualification gates precede six-shape product_kernel-versus-pair comparisons.
The code snapshot is immutable; current measurements continue on their
previous source snapshots. Stable defaults stay unchanged.

### Factor-pair qualification and matched training launch

Factor-pair qualification passed **13 tests** on 5060 (14 deprecation
warnings), including public compiled/graph checks. Its completed timing wave
shows modest or negligible benefits, not a narrow-shape solution: stationary
paired ratios are 0.727x decoder512, 0.812x decoder1536, 0.889x decoder1024,
0.934x long and 1.204x wide. The decoder2048 pair session was nonstationary.
These are comparisons against MXFP8, not paired pair-versus-product effect
estimates; source records are in local `factorpair/`.

Matched short training is running in
`/home/achiya/rtx-mseden-matched-training-20261027.PSRhJ1`:
`matched-training.py` PID 51403, first worker PID 51436, log
`launch-retry1.log`. Four 512-step runs use seeds 941/942, reversed policy
order, identical per-seed initial weights/data, NVFP4 forward, and either
stable MXFP8 or experimental FP4 backward. Each run processes 2,097,152 byte
tokens with the established 21M model/FP32-sensitive-op/BF16-state recipe and
compiled CUDA graphs. No training outcome is claimed yet.

Per-shape schedules are selected from stationary, numerically guarded public
backward artifacts using the minimum lower confidence bound across whole
rounds and AB/BA orders. Exact selections and source paths are saved in
`selected-schedules.json` and `selection-provenance.json`; this is selection
evidence, not an unbiased performance estimate or a claim of meeting 1.20x.

The first selector invocation stopped before training: the generic
`pipeline_outcome` validator assumed both arms were stochastic FP4 and rejected
the deterministic MXFP8 reference's `[false,false]` RNG flags. It now verifies
an explicit stable-MXFP8 policy, requires its deterministic flags and <=0.06
gradient gate, and still requires fresh RNG and the unchanged <=0.25 gate for
FP4. Unknown/inconsistent policies fail. Focused CPU adapter/tuner regression:
22 passed; a retained real artifact now validates. The failed `launch.log`
was preserved, and only this validator was updated before restarting the
confirmed-terminal pre-training job. Kernel code stayed unchanged.

A larger preparation candidate remains under development: exact FP4 threshold
packing plus scale-free FP32 correction. CPU evidence is promising but the
correction reassociation is **not bit-identical** (18 one-ULP BF16 scale changes
among 1,835,008 tested blocks). It must remain explicit, pass independent
numerical/public gates, and receive GPU qualification before timing or model
use. It is not part of the running training snapshot.

### Completed matched short training: no model speed win yet

All four 512-step runs completed with finite parameters. The analyzer verifies
matching initial-weight hashes, corpus/validation identities, optimizer,
precision recipe and NVFP4 forward projection configurations within each seed.
It now supports `--baseline-mode nvfp4_mxfp8` explicitly rather than relabeling
the hybrid reference as full MXFP8; its 48 CPU tests passed.

| Seed | MXFP8 backward train s | FP4 backward train s | MX/FP4 speed ratio | MX validation loss | FP4 validation loss | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 941 | 9.971 | 10.492 | 0.950 | 0.991845 | 0.993939 | +0.002094 |
| 942 | 10.283 | 10.625 | 0.968 | 0.952835 | 0.957086 | +0.004250 |

Training wall includes data movement, forward/backward, clipping and optimizer,
but excludes compile, validation, logging and checkpoint I/O. Inclusive walls
were 46.25/53.34 s (MX/FP4, seed 941) and 25.86/36.12 s (seed 942); compile/cache
effects matter substantially at this short length. These are two runs per
policy, not a throughput confidence interval or long-convergence qualification.
Each run consumed 2,097,152 byte tokens with the established approximately
21M-parameter model. FP4 is 5.23% and 3.33% slower in measured training wall.

Equal-wall analysis deliberately uses observed checkpoints only. At each
seed's common final training budget, the latest FP4 validation was step 384
versus MXFP8 step 512: losses 1.032818/0.991845 (941) and
1.000864/0.952835 (942). FP4 observations were already 2.02/2.26 seconds old
at those budgets because validation occurs every 128 steps. Do not interpret
those coarse checkpoint gaps as precise equal-wall degradation or interpolate
them into claimed measurements.

Artifacts: local `autotune_reports/ms_eden_speed20_20261026_5060/training/`
contains raw JSONL, selections/provenance and `analysis-seed941.json` /
`analysis-seed942.json`. The remote driver exited normally at
2026-10-26T22:45:29Z. The goal remains incomplete: the repeatable wide-kernel
win has not translated to a training win on the narrower model projections.

### Threshold/scale-free GPU qualification

The explicit `normalization_math="threshold_scale_free"` candidate passed
**9 tests** on 5060 (14 deprecation warnings), including independent threshold
and FP64 correction probes, producer comparisons, changed-buffer replay and
public compiled/RNG/memory/numerical gates. This does not make its reassociated
FP32 correction bit-identical to division, nor establish a global one-ULP
bound beyond the tested cases. It was not used in the training results above.

The first qualification had 6 passes and 3 compilation failures: CuTe needed
`bits` initialized before a staged conditional. Initializing `bits=Int16(0)`
fixed compilation without changing arithmetic or tolerances. Both logs are
preserved locally in `scale-free/qualification-initial.log` and
`scale-free/qualification-fixed.log`. No benchmark ran before the full pass.

Current immutable-after-fix snapshot:
`/home/achiya/rtx-mseden-scale-free-20261027.0ihv2d`, driver PID 53408,
`launch-retry1.log`, `confirm-scale-free.py`. It compares division and the new
mode on six original decoder shapes, retaining paired x8/async-2, native
decoding and 32-bit copies, with pair-factor scaling in both arms. Each
candidate is measured against the stable MXFP8 backward with the same public
guards and 1.20x confidence/stationarity requirements. Timing results are still
pending; stable defaults are unchanged.
## Threshold packing with scale-free FP32 correction (new numerical option)

`normalization_math="threshold_scale_free"` is opt-in; `"divide"` remains the
default. Unlike the exact-but-slower shared-reciprocal experiment, this removes
all four per-fragment normalizing divisions and does not form a reciprocal.
It does **not** preserve corrected-scale bits: it reassociates FP32 EDEN
statistics. Existing original-BF16 Gaussian gates and the independent `.005`
packed-GEMM NRMS gate are unchanged.

Packing uses three branchless bisection comparisons against the exact dyadic
midpoints `.25,.75,1.25,1.75,2.5,3.5,5` times normalized E8M4 scale m. All tie
directions implement even E2M1 codes; input signed zero is preserved on this
path. This also matches `FP4_RN(FP32_RN(v/s))`, not just direct real division:
adjacent FP32 input spacing divided by m is at least `16/31` quotient ULP,
strictly above half an ULP. Thus no additional input rounds onto a midpoint.
The power-of-two predecessor case scales input/output spacing identically;
subnormal original threshold products have a coarser lattice. CPU checks of
507,276 ties and four neighbours on either side, both signs, covered all 16
scale mantissas and exponent fields corresponding to -126..125: zero code
mismatches. Six generic scales with overflowing thresholds were excluded from
that original-domain check; the kernel compares normalized quantities instead.

For accepted blocks, exact power-of-two scaling forms u from v. Correction is
`2^E * (sum(u*u)/sum(u*q))` in FP32, then one BF16 cast. Guards require a
positive normal <=5-significant-bit scale (exponent field 1..253), finite
original inputs, each u zero only if original v is zero or `abs(u)>=2^-60`,
and `abs(u)<=16`. A block-wide guard also verifies that its maximum matches
the unchanged `maximum/6 -> E8M4 RTZ` regime (E8M3 is supported too). Unsafe
blocks retain the original division, FP4 conversion, and correction path.
All-zero/tiny-flushed scales retain the original zero behavior.

This guarded domain prevents dot-product overflow/underflow and guarantees a
nonzero denominator. Terms are nonnegative, so standard FP32 product/tree-sum
relative bounds apply without cancellation. Counting the original divisions,
FP32 products/additions, final ratio and scale multiplication gives a small
multiple of `2^-24` relative error (conservatively below `32*2^-24`); the
reassociated result can therefore cross at most one BF16 rounding boundary.
This supports a **one-BF16-ULP numerical bound, not byte equivalence**. Final
FP32 restoration can be subnormal but is at least `2^-128`, so its rounding
error remains covered; no FP16 statistics or intermediate BF16 values are used.

Bounded CPU study (explicit balanced FP32 multiply/add model, not GPU FMA-IR
identity): 1,835,008 sixteen-value blocks across Gaussian, 32x outliers, 95%
sparse, signed/unsigned zero and extreme dynamic range, with magnitudes
`2^[-120,-80,-40,0,40,80,120]`. All 21,345,872 accepted packed values matched.
There were 18 changed BF16 scales, each by one ULP; worst relative change was
0.7194%. Before the BF16 cast, the largest difference was nine FP32 ULPs / a
relative `5.47e-7`. Gaussian changed 10/524288, outliers 6/524288, sparse
2/524288; zeros were unchanged. The extreme 1e20 dynamic-range case correctly
fell back for 130230/131072 blocks. Validation took about 3.07 CPU seconds.

New tests compare packing to explicit GPU `div.rn` plus native E2M1 conversion,
correction to an independent FP64 formula/direct BF16 rounding oracle, and
producer packed bytes plus changed-buffer replay. Public fullgraph/RNG/current
buffer/ordinary-allocation/replay-memory and packed-GEMM gates remain required.
Run on the assigned idle GPU only:
`RTX_RUN_SCALE_FREE_CUDA=1 python -m pytest -q tests/test_ms_eden_threshold_scale_free.py`.
At implementation time these GPU tests and performance comparisons are pending.
Preparation context/feature revisions are 18/13. No performance win or default
promotion is claimed: extra thresholds, comparisons, guards and register life
can offset the four removed precise divisions.

### Follow-up: measured preparation cost and compact row loops

The threshold/scale-free GPU gate subsequently passed **9 tests** after fixing
a staged-conditional variable initialization. It did not improve the measured
complete backward path. On 4096x1536x512, the stationary comparison was
335.6 us for original division versus 367.3 us for threshold/scale-free.

Actual 5060 cubins, inspected offline, show 40–49% more static instructions in
the three captured buffered producers. Registers changed 48→56, 72→77 and
48→48; all reported zero stack/local storage. These are static counts, not
executed instruction counters, and do not establish an occupancy or instruction
cache bottleneck. Nsight Systems node-level graph traces confirm longer
producer durations while GEMM durations remain similar. The diagnostic traces
include overlap and are not replacements for ordinary event benchmarks.
Evidence is under `autotune_reports/ms_eden_speed20_20261026_5060/profile/`.

The next exact-arithmetic experiment exposes `producer_unroll_rows=False`:
process staged subrows with a compact loop instead of duplicating the body.
The default remains `True`; arithmetic, loads, barriers and stage order are
unchanged. This targets buffered producers, not the dominant single-row
producer. Context/feature revisions are 19/14. CPU regression: 132 passed,
159 skipped. Remote 5060 qualification: **24 passed in 22.64 seconds**.

Six-shape, alternating-order complete-backward comparisons are running in
`personal_laptop:/home/achiya/rtx-mseden-compactrows-20261027.3qSpgN`.
No performance promotion is justified yet. The 5070 is explicitly excluded
from further work while the user changes that machine.

The six-shape compact-loop run has now completed. Candidate/MXFP8 paired
speed ratios (reported as MXFP8 time / FP4 time) for compact versus unrolled:

| Shape label | Compact | Unrolled |
|---|---:|---:|
| decoder512 | 0.7367* | 0.7259 |
| decoder1536 | 0.8184* | 0.8171 |
| decoder2048 | 0.9038 | 0.9084* |
| decoder1024 | 0.8965 | 0.8867* |
| long | 0.9341 | 0.9246* |
| wide | 1.2001 | 1.1963 |

`*` denotes at least one failed stationarity gate. These are separate
baseline/candidate comparisons, not a paired compact-versus-unrolled effect
estimate. None passed the full 1.20 target gate. Wide compact's 95% interval
was [1.1970, 1.2029], so its point estimate is not a qualifying target win.
Artifacts: `autotune_reports/ms_eden_speed20_20261026_5060/compactrows/`.

Next, a sequential seven-candidate sweep isolates only `dx.a_geometry` on
decoder1536: producer (threads, blocks) = (128,256) control, (128,52),
(256,104), (128,104), (128,416), (256,208), (128,208). Original division,
unrolled rows, native correction decode and pair-factor scaling are fixed.
This tests scheduling of the measured dominant row-major dY producer without
changing arithmetic. The same remote snapshot runs `rowgrid-study.py`, PID
59718 at launch, writing `rowgrid-launch.log` and per-candidate `rowgrid/`
manifests/results. Its process was verified live after the compact benchmark
terminated; GPU measurements do not overlap.

The row-grid sweep completed and all seven runs passed stationarity. Ratios
for 128-thread grids of 52/104/208/256/416 CTAs were respectively
0.73538/0.78281/0.81827/0.81541/0.81083. The 256-thread grids at 104/208
CTAs gave 0.80644/0.80037. Thus launch geometry alone did not close the gap;
208 versus 256 CTAs is a small search observation, not a confirmed improvement.
Compare paired ratios rather than raw times across these separate processes.
Final artifacts are under `compactrows/rowgrid/` in the local report root.

Next structural hypothesis: the row-major producer executes 48 grid-stride
iterations per warp on decoder1536 at 256x128 launch geometry, whereas the
x8 transposed path exposes eight tile bodies per outer iteration. An opt-in
row-only tile batch of 2 or 4 could expose instruction-level independence
without changing original tile ownership/order, partial counts, reduction
arithmetic, or SMEM. Exact qualification precedes complete-backward timing.
Explicit 64-bit row loads were not chosen yet because the captured PTX files
contain only buffered producers (filename collision overwrote the row kernel);
there is no compiled row-load evidence to justify that change yet.

Row batching is now implemented as `producer_row_tile_batch=1/2/4`, default 1,
context/feature revisions 20/15. The exact per-warp sequence is preserved even
at incomplete batches and under reduction-major traversal. It is inactive on
transposed operands; packed row-major batching is explicitly rejected.
Focused CPU suite: 28 passed, 32 skipped (agent); direct parent test-file run:
3 passed, 12 GPU cases skipped. GPU qualification is queued/running in
`personal_laptop:/home/achiya/rtx-mseden-rowbatch-20261027.wiw2qP`, launcher
PID 61771. Only after its exactness gate passes will batch 1/2/4 be compared on
decoder1536, decoder512 and wide, with ordinary complete-backward paired
measurement. Defaults are unchanged.

Before that qualification, two existing staged-amax configurations completed
sequentially on decoder1536: staged dX gave ratio 0.81639; staged dX+dW gave
0.82365, both stationary. This is a small diagnostic difference relative to
the earlier inline control (0.81541), not a confirmed causal benefit across
sessions, and nowhere near the target. It does not justify prioritizing
additional reduction machinery over the dominant preparation arithmetic.
Remote artifacts are `rtx-mseden-compactrows-20261027.3qSpgN/staged-diagnostic/`.

The row-batching GPU qualification subsequently passed **15 tests in 15.75s**
(12 GPU cases plus 3 CPU cases). Launcher PID 61771 proceeded to the first
complete-backward comparison, `decoder1536-batch1`; verified live after the
qualification finished. No timing conclusion or default promotion yet.

Initial row-batching timings are negative: decoder1536 batch1/2/4 ratios are
0.81667/0.81388/0.81439, all stationary. Decoder512 gives
0.72234/0.72166/0.71922, all with at least one nonstationarity exclusion.
Wide is still running. There is no justification for promoting larger batches.
Retrieved results and exact test log are under the local `rowbatch/` folder.

A collision-free compiler capture is queued after those benchmarks, PID
62799 at launch: each process compiles exactly one row-major producer, with
batch1/2/4 in separate `row-compiler/batchN` directories. File caching is
disabled for this diagnostic; PTX and cubin are retained. This closes the
earlier row-kernel dump collision before proposing a vector-load change.
It does not run concurrently with performance measurements and is not itself
a performance benchmark.

Both jobs finished. Wide batch1/2/4 ratios were 1.20718/1.20242/1.20202.
Batch1 failed stationarity; batches2/4 were stationary but their lower
confidence bounds fell below 1.20. No batch variant qualifies as an improvement.

The isolated row-producer dump now provides actionable load evidence:
baseline PTX has four `ld.global.b16` instructions at byte offsets 0/2/4/6,
and offline `cuobjdump --dump-sass` confirms four `LDG.E.U16` instructions
inside the tile loop. The four sign loads are outside that loop. Thus the
compiler does not combine the four adjacent BF16 source values even though
they are owned by one lane. Exact artifacts are retained under
`rowbatch/row-compiler/batch1/`. An opt-in aligned 64-bit row-load experiment
is now justified; it must validate alignment and preserve all FP32 operations.
Expected mechanism: fewer load instructions/requests, not fewer useful input
bytes. Complete-backward improvement remains unproven.

Vector loading is implemented as `producer_row_load_bits=16/32/64`, default
16, context/feature revisions 21/16. It applies only to original-BF16
row-major preparation; transposed loading is unchanged. Runtime construction
checks base-pointer and row-stride alignment for opt-in widths. Unsupported
packed-row vector loading is rejected. CPU focused suite: 27 passed,
18 skipped. GPU suite: **9 passed in 12.19 seconds**, including byte-exact
prepared arrays, exact same-draw gradients, changed-buffer graph replay,
aligned nonzero storage offsets, signed zero and outliers.

Isolated actual-cubin inspection confirms the intended mechanism: 16-bit
baseline has four `LDG.E.U16` loads, 32-bit mode two `LDG.E` loads, and
64-bit mode one `LDG.E.64` load. Baseline and 64-bit mode both report 40
registers and zero stack/shared/local storage. These are code-generation
facts, not proof of a runtime speedup. Local evidence is retained under
`autotune_reports/ms_eden_speed20_20261026_5060/rowvector/`.

Complete backward comparisons on decoder1536/decoder512/wide are running
sequentially on the 5060 in `rtx-mseden-rowvector-20261027.ut38t3`, PID 64209
at launch. Each compares against the same stable MXFP8 policy, includes
preparation/RNG, and retains stationarity, graph, memory and numerical gates.

Vector-width comparisons completed. Ratios for 16/32/64-bit loads:

| Shape | 16-bit | 32-bit | 64-bit |
|---|---:|---:|---:|
| decoder1536 | 0.81279 | 0.81064 | 0.80457 |
| decoder512 | 0.72325* | 0.71802* | 0.71960* |
| wide | 1.20725 | 1.20487 | 1.20801* |

`*` denotes a failed stationarity gate. Wide 16/32 passed the full timing
target, but the unchanged scalar control is already as fast; this is not a
vector-load win. All decoder1536 results were stationary and show no gain.
Retain the default; fewer load instructions did not recover the missing time.

The next ranked structural experiment is row-only 8/16 values per lane
versus 4. Each warp handles two/four independent 128-value transforms using
16/8-lane subgroups. More butterfly stages become local FP32 operations;
shuffle instruction count per fixed amount of data falls by 20%/40%, with no
additional FLOPs or useful input bytes. Increased register pressure may erase
the advantage. Preserve the original Hadamard add tree and original four-value
EDEN subtotal tree; reject the candidate if exact final payload/scales/gradient
tests fail. Per-warp partial maxima may differ with ownership, so validate
their exact global maximum rather than claiming per-warp identity. This is
still a design/implementation hypothesis, not a measured speedup.

The wider-lane source is implemented behind `producer_values_per_lane=4/8/16`
(default 4), context/feature revisions 22/17. It preserves each physical
warp's original tile set by assigning subgroup g the tile
`warp + (iteration*groups_per_warp + g)*work_warp_count`. This also permits
exact finite partial-max comparisons, stronger than the earlier proposal.
The old VPL4 kernel branch is unchanged. Initial larger-lane support requires
original BF16, divide normalization and row_tile_batch=1.

CPU design probe: 131,072 FP32 values, including mixed magnitudes 2^-100 to
2^99, had zero bit mismatches between the VPL4/8/16 Hadamard decompositions.
This is not GPU evidence. Focused CPU tests: 4 passed, 10 GPU cases skipped;
agent's wider focused selection: 17 passed, 10 skipped.

Early GPU compilation of VPL8 failed in NVVM before exactness could be tested.
No VPL timing was launched. Isolated snapshot:
`personal_laptop:/home/achiya/rtx-mseden-vpl-20261027.L4WKKC`.
Initial smoke PID 66309 and diagnostic PID 66412 both exited. Retained clean/raw
MLIR and logs are local under `autotune_reports/ms_eden_speed20_20261026_5060/vpl/`.
The clean IR identifies the new grouped row kernel; any PTX alongside it may
be the successful old VPL4 compile, not the failed candidate. An opt-level-0
diagnostic (PID 66659 at launch) is isolating optimizer involvement, not
changing production compilation or weakening numerical acceptance.

The NVVM failure was isolated with compile-only probes: runtime add/multiply/
three-input FMA compiled; passing `Float32(0)`, `Float32(-1)` or `Float32(1)`
as inline-PTX read arguments failed. Repeating versus separating operand
placeholders did not fix it. The grouped kernel's first subtotal used such a
literal-zero input. `grouped_add_zero` now embeds `0f00000000` directly in the
PTX instruction, preserving the exact rounded addition (including zero-sign
behavior) rather than deleting it. No arithmetic policy was changed.

Repaired GPU smoke passed native/LUT decode × VPL8/16, Gaussian/outlier/zero,
with byte-identical packed data, temporary scales, partial maxima, finalized
scales and outer factors. Full 14-case qualification is now followed by nine
timing comparisons only on success, launcher PID 67600 in the same isolated
snapshot. Old failure logs are retained; this is a targeted source fix, not
a compiler optimization downgrade.

Full wider-lane GPU qualification passed **14 tests in 12.67 seconds**.
The driver proceeded to decoder1536 comparisons; baseline VPL4 measured
0.81464× stable MXFP8, with VPL8/16 still pending at this checkpoint.
Successful qualification and repaired smoke logs have been copied to the
local `vpl/` evidence directory. No performance promotion yet.

Final VPL timings: decoder1536 VPL4/8/16 = 0.81464/0.82544/0.80210, all
stationary. Decoder512 = 0.72209/0.71407*/0.70696*. Wide =
1.20458/1.21506/1.19539*, where `*` fails stationarity. Wide VPL8 passes the
full 1.20 timing gate, but narrow-shape target coverage remains incomplete.
Actual row-producer cubins show 40/40/57 registers for VPL4/8/16, all with
zero local/stack/shared storage. This establishes higher register demand for
VPL16, not measured occupancy loss. Artifacts: `vpl/row-compiler/`.

`refine_warp_lut` was implemented for VPL4 with one exact reciprocal entry
per lane initialized before persistent loops and unconditional table shuffles
before eligibility branches. Existing refinement/fallback remains unchanged;
no new tensors or launches. Context/feature revisions 23/18. GPU exactness:
**7 passed, 14 deprecation warnings in 13.06 seconds**. Completed isolated run:
`rtx-mseden-warplut-20261027.ht18r3`, PID 69607 exited. Local `warplut/` retains
all logs/results. Ratios (divide/refine/LUT): decoder1536
0.81598/0.76196/0.76107*, decoder512 0.72100/0.67409*/0.67392*, wide
1.20800/1.15055/1.16470. LUT did not beat ordinary division; no promotion.

Next coverage audit: inherited native-mma128 scale legality currently fixes
M/N/K CTA tiles to 128, but atom_layout_m=2/4/8 and direct epilogue with
store_vec=1 are legal. These produce 160/288/544-thread CTAs. Check historical
GEMM coverage before deciding whether a new preparation implementation is the
best next investment. Do not remove the native-layout restriction without
independent layout qualification.

The controlled GEMM sweep completed in isolated
`rtx-mseden-gemmsweep-20261027.hkTf8s`; PID 71701 exited. Every candidate first
passed full decoder1536 same-draw dX and dW byte-equivalence to the control on
Gaussian, outlier and zero inputs. All eight timing comparisons were stationary:

| M atoms | Epilogue | Stages | MXFP8 / candidate |
|---|---|---:|---:|
| 8 | TMA | 2 | 0.81638 |
| 4 | TMA | 2 | 0.81630 |
| 2 | TMA | 2 | 0.81982 |
| 8 | direct | 2 | 0.73166 |
| 4 | direct | 2 | 0.73434 |
| 2 | direct | 2 | 0.73338 |
| 4 | direct | 1 | 0.63256 |
| 4 | TMA | 1 | 0.69618 |

Direct stores used store_vec=1 as required by legality. Preparation and
quantization were held fixed. No candidate closes the gap; the tiny two-atom
ratio difference is not a confirmed improvement. Local `gemmsweep/` preserves
same-draw qualification logs, commands and all comparison artifacts.

A further orchestration gap is visible in source: `_build_gradient.launch`
serializes full preparation of A (producer and finalizer) before B. Current
paired mode overlaps the two gradients, not their four independent operands.
An opt-in four-operand preparation pipeline is being designed with bounded
caller-stream-scoped side streams, invocation-local event dependencies,
exception-safe joins and allocation lifetime recording. Each GEMM must wait
only for its own two preparations; caller completion must join both gradients.
It changes no arithmetic and adds no eager tensor operations. Capture/cache/
memory qualification is mandatory before performance testing or promotion.

The opt-in `operand_paired` implementation now passes the 5060 qualification:
**12 passed, 14 deprecation warnings in 15.79 seconds**. A separate smoke test
also passed same-draw Gaussian/outlier/zero gradient equivalence, fullgraph,
capture on a non-default caller stream, current-buffer/RNG replay, and stable
allocated memory across replay. CPU CLI/topology checks: 10 passed, 2 GPU tests
skipped. The training probe accepts the new option; serial remains its default.

The implementation schedules four independent preparations across the caller
and three bounded cached side streams, joins each operand pair before its GEMM,
and joins both gradients before returning. Arithmetic is unchanged. An isolated
matched timing study is running at
`/home/achiya/rtx-mseden-operands-20261027.IUMCFF` on personal_laptop (5060),
PID 74063, comparing paired/operand_paired on decoder1536, decoder512 and wide.
The wide comparison uses the previously qualified VPL8 row producer. It does
not use the 5070 laptop, per the user's current restriction. Correctness alone
is not a performance win; do not promote this pipeline before timing evidence.

The six comparisons completed; PID 74063 exited. Local `operands/` retains
the manifests, outputs and logs. Full backward MXFP8/candidate ratios:

| Shape | paired | operand_paired |
|---|---:|---:|
| decoder1536 | 0.81624 | 0.83229 |
| decoder512 | 0.72680* | 0.75480 |
| wide (VPL8) | 1.19952 | 1.21596 |

`*` denotes rejected timing stationarity; other rows passed stationarity.
Wide operand_paired has round-bootstrap 95% interval [1.21360, 1.21823] and
passes the full timing target gate. This is not broad target achievement:
narrow shapes still lose to stable MXFP8, and no new training claim is made.
All comparisons use the repository-default stable MXFP8 backward, not a fresh
exhaustive MXFP8 search. The next controlled admission study changes only the
dW A producer CTA cap (52/104/128/208) from 256, with one paired control at 128.
Every variant must pass full decoder1536 same-draw dX/dW byte equivalence on
Gaussian/outlier/zero inputs before its timing comparison. The hypothesis is
that a smaller first grid permits earlier independent-work admission; it is
not a hardware-counter occupancy claim. Isolated PID 75320, `admission.log`
in the same remote experiment directory. Stable defaults remain unchanged.

Admission sweep completed (PID 75320 exited); all five full-shape exactness
gates passed. Operand-paired ratios at 128/104/52/208 CTAs were respectively
0.84401*/0.81942/0.80440/0.83065; paired at 128 was 0.82292. The apparent best
128 result failed baseline stationarity and is not promoted. Smaller caps
did not reliably outperform the 256-CTA operand-paired control (0.83229).
Raw results and exactness logs are retained under `operands/admission/`.
Next structural experiment: extend the qualified VPL grouping to transposed
preparation, keeping the butterfly and EDEN FP32 reduction order unchanged.

Transposed grouping is implemented as opt-in
`producer_transpose_values_per_lane=4/8/16` (default 4), separate from row VPL.
It currently requires original BF16, coalesced_x8 loads and divide normalization.
Each warp still owns the same eight staged rows; subgroups move butterfly
levels into registers while retaining the original EDEN association. Staging
layout, copies, ring stages and barriers are unchanged. Context/features 24/19.
CPU focused qualification: 24 passed, 42 GPU skips. On the 5060, the new
transpose and existing row suites jointly passed **29 tests in 23.14 seconds**,
including multi-iteration ring/tail, sign reload, CTA maxima, exponent range,
signed zero and changed-buffer graph replay. The shared helper refactor thus
has fresh row-regression coverage, not just new-path tests.

Isolated experiment: `/home/achiya/rtx-mseden-transpose-vpl-20261027.viXokC`,
PID 77002. A nine-case study tests dW transpose VPL4/8/16 at decoder1536,
decoder512 and wide, retaining the prior operand_paired schedules. Each case
first requires full-shape Gaussian/outlier/zero dX/dW same-draw byte equality,
then compares complete compiled backward with stable MXFP8. Results pending;
no promotion or performance claim follows from qualification alone.

The nine-case transpose study completed; PID 77002 exited and all full-shape
same-draw dX/dW qualification gates passed. Local `transpose-vpl/` contains
all outputs and qualification logs. Full-backward MXFP8/candidate ratios:

| Shape | transpose VPL4 | transpose VPL8 | transpose VPL16 |
|---|---:|---:|---:|
| decoder1536 | 0.83224 | 0.86868* | 0.88218* |
| decoder512 | 0.74569* | 0.79765* | 0.80426 |
| wide | 1.21861 | 1.23758 | 1.24026 |

`*` failed candidate stationarity in BA windows and is not a confirmed winner.
Wide VPL16 passes the full timing gate with round-bootstrap 95% interval
[1.23854, 1.24196]. This remains a wide-shape result, not broad achievement or
a model-training speed claim. A follow-up on the same frozen source snapshot
(PID 79149, `followup.log`) tests transposed grouping on both legs and stages
1/2/3, with 15-second warmup and 24 paired rounds. Every candidate again has
full-shape same-draw qualification before measurement. No defaults changed.

First follow-up confirmation is stationary: decoder1536, dW transpose VPL16,
two stages, operand_paired yields **0.88999** MXFP8/candidate with 95% interval
[0.88813, 0.89173]. The prior VPL4 control was 0.83224; this supports a useful
preparation improvement but still does not beat MXFP8 on that shape. Remaining
follow-up candidates are still running; do not aggregate incomplete results.

Follow-up completed (PID 79149 exited), with all exact full-shape gates passing.
For decoder1536, both-leg VPL16/stages2 was 0.88902; dW-only/stages1 was
0.88978*; both-leg/stages1 was **0.90704** with interval [0.90532,0.90877];
dW-only/stages3 was 0.84451*. Both-leg/stages2 on decoder512 was 0.80243*
and on wide **1.24114**, interval [1.23938,1.24282], target timing gate passed.
Starred runs failed stationarity. These results do not establish a universal
winner or satisfy the broad goal. Local `transpose-vpl/followup/` stores evidence.

Fresh Nsight node traces compare decoder1536 VPL4 to grouped-both/stages1.
The first capture failed before profiling because an old remote helper lacked
`--schedule-manifest`; after updating that diagnostic helper, both fresh
`profiles-v2/` captures completed (PID 81460 exited). Raw reports, SQLite,
compiler artifacts and `node-summary.json` are retained locally. Median node
durations in microseconds (diagnostic only, 10 replays): dW dY producer
89.67→76.01; dW X producer 59.24→45.26; dX W producer 20.83→16.03; dX dY
producer 76.52→85.84 while its start advanced 108.68→80.03 after first prep.
dW/dX GEMMs stayed about 38.7/40.3 microseconds. Actual overlap changes mean
individual durations are not additive speedups. The remaining row dY path is
critical; no tensor-pipe, spill-traffic or achieved-occupancy counter claim is made.

`guarded_scale_free` passed **8 GPU tests (14 deprecation warnings, 12.78s)**
on the 5060. It is exact to the existing threshold_scale_free prepared arrays,
not to divide's corrected statistics; observed divide differences were at most
one BF16 ULP, with exact packed bytes and unchanged independent gradient/memory
gates. Isolated `/home/achiya/rtx-mseden-guarded-20261027.pupver`, PID 82290,
is timing divide, threshold and guarded variants on decoder1536. Results pending.

Independent integer scale construction has exhaustive CPU proof over
75,497,472 inputs and a standalone **3-test GPU pass (3.88s)** against literal
PTX RN division, including boundary and graph-replay cases. New opt-in
`producer_scale_math="integer_rtz4"` preserves the old scale exactly on its
nonnegative producer-maximum domain. Default divide is unchanged. Integration
CPU gate: 45 passed, 58 skipped; context/features 26/21. Integrated producer
GPU qualification and timing remain pending; standalone equivalence is not a
speed claim. Proof artifacts: `cpu_scale_rtz_proof/{proof.py,result.json}`.

Guarded packing study completed (PID 82290 exited); all full decoder1536
same-draw gates against the corresponding threshold contract passed. Ratios:
divide 0.90099*; dX threshold 0.84829; dX guarded 0.85975; both threshold
0.76318; both guarded 0.76373. The divide run failed BA candidate stationarity;
the independent preceding ordinary-divide confirmation was 0.90704. Guarded
packing does not close the gap and is not promoted. Local `guarded/` retains
all evidence. It also cannot currently compose with grouped VPL8/16 on an
affected leg, so this measures real eligible candidates, not an isolated
arithmetic shortcut. Integer-scale integration qualification is now running
in `/home/achiya/rtx-mseden-integer-scale-20261027.qATo4q` (`qualification.log`).

Integer-scale integration passed **14 GPU tests in 21.84s**. Its six-case
study completed (PID 83805 exited), all full-shape exact gates passed, with
divide/integer MXFP8 ratios: decoder1536 0.90458/0.89962; decoder512
0.80774*/0.80680*; wide 1.24131/1.23917. Starred cases failed stationarity.
No performance improvement supports promotion; keep ordinary division.
Local `integer-scale/` retains full results and qualification evidence.

Next is a research-only preparation enqueue-order screen, not a production
API expansion. `benchmarks/experiment_ms_eden_operand_order.py` changes only
the order of four independent preparation calls, retaining streams, startup
waits, lifetimes, locks, finish dependencies and exception joins. It records
the explicit order and harness/launcher hashes, and requires full-shape
same-draw fullgraph/changed-buffer exact dX/dW before inherited timing/numerical/
memory gates. CPU topology/failure/parser regression: 220 passed, 2 GPU skips.
Eight orders are running on the 5060 at the integer-scale snapshot, PID 85257,
`orders.log`; candidate arithmetic uses the ordinary-divide grouped-stage1
schedule. Results pending, and production launch order/defaults are unchanged.

The first order attempt exited before timing: the research smoke warmed the
default stream but implicit graph capture selected another stream. The runtime
correctly rejected the unwarmed caller-stream entry. Fixed only the harness:
explicit capture stream, wait for current inputs, triple warmup on that stream,
stream synchronization, explicit graph stream, and Inductor nested cudagraphs
disabled. CPU regression: 206 passed, 2 GPU skips. Fresh `orders-v2/` retry
(PID 85666) passed full-shape exact/graph gates and completed order0 timing at
0.88108, stationary but slower than the prior production-order confirmation.
The remaining seven orders are running; retain the failed `orders/` logs as
diagnostic evidence, not performance observations. Production code unchanged.

All eight `orders-v2` cases completed (PID 85666 exited), all full-shape
exact/current-buffer graph gates and stationarity checks passed. Ratios by
order0..7 were 0.88108, 0.88987, 0.87563, 0.88275, 0.92141, 0.90259,
0.92267, 0.90987. Order7 is the production order. Order6 (dX B, dX A, dW A,
dW B) and order4 (dW A, dW B, dX A, dX B) show small screening gains, not a
confirmed universal policy. Local `integer-scale/orders-v2/` contains all
provenance; no production enum or default was changed.

Finalization study completed (PID 87066 exited), all full-shape exact gates
passed. Control/dX-staged/dW-staged/both-staged/dX-weight-cap26/cap52 ratios:
0.90749/0.91464*/0.90845/**0.92009**/0.90760*/0.90873*. Starred runs failed
stationarity. Both-staged interval [0.91885,0.92129]. Local `integer-scale/finalizer/`
retains artifacts. An eight-case transfer study on decoder2048, decoder1024,
decoder512 and wide now compares inline/staged maxima with the qualified
grouped schedules, not claiming transferred geometry is optimal. PID 88617,
`generalization.log` in the same remote snapshot. This supplies the missing
current decoder-shape evidence before another matched training translation test.

Offline actual-cubin audit rejected a hypothesized duplicated-add problem:
the isolated VPL4 row kernel already uses 20 shuffle, 20 sign-select and 20
unpredicated add instructions across its five warp butterfly stages, with
40 registers and zero static stack/local allocation. Initial signs also fuse
into the first butterfly. No XOR rewrite is justified by those counts alone.
A separate read-only feasibility study is considering tensor-core execution
of initial Hadamard stages on exactly representable original BF16 operands;
no such kernel or numerical/performance claim exists yet.

Decoder transfer study completed (PID 88617 exited), all full-shape exact gates
passed. Inline/staged ratios: decoder2048 1.01257/1.03859; decoder1024
0.98883*/1.02255; decoder512 0.78646*/0.83084*; wide 1.24169/1.24732.
Starred measurements failed stationarity. The wide staged interval is
[1.24592,1.24874], timing target passed; other shapes still miss 1.20.
Local `integer-scale/generalization/` retains evidence.

### Updated matched short training

The v2 training attempt used an old remote helper lacking `--ms-eden-schedules`:
its MXFP8-backward control completed, but full FP4 stopped at argument parsing.
Neither is used for the comparison. A fresh complete local-source archive was
deployed to `/home/achiya/rtx-mseden-training-v3-20261027.pvYKvm`; all four runs
completed (PID 90962 exited). CPU training/schedule-map regression: 36 passed.
Each ~21M decoder ran 512 steps / 2,097,152 byte tokens, with identical NVFP4
forward, compiled CUDA-graph training, original BF16 sources and the existing
sensitive-operation FP32 recipe. Seeds 941/942 used opposite policy order.
Only stationary, validated ordinary-division schedules were selected; research
launch-order patches and scale-free statistics were not used. All parameters
remained finite. Local `training-v3/` has JSONL, logs, manifests, selection
provenance and per-seed analysis with the matched `nvfp4_mxfp8` baseline.

| Seed | MXFP8-bwd training s | FP4-bwd training s | MX/FP4 ratio | MX validation loss | FP4 validation loss |
|---|---:|---:|---:|---:|---:|
| 941 | 10.21325 | 10.23574 | 0.99780 | 0.98980010 | 0.97944283 |
| 942 | 9.92588 | 10.18113 | 0.97493 | 0.95879632 | 0.95106888 |

Training clocks exclude compilation, validation and checkpoint overhead.
Total job walls were 25.8653/42.2336 seconds (MX/FP4, seed941) and
25.5542/34.9351 (seed942). FP4 is roughly tied on one training seed and 2.5%
lower throughput on the other; the larger end-to-end startup cost remains.
Lower FP4 validation loss in these two short runs does not establish superior
convergence. Neither this study nor the wide-kernel win completes the broad goal.

The new pure-transform guarded TF32-H8 prototype passed **14 GPU tests in
8.12s** on the 5060 at `/home/achiya/rtx-mseden-tf32h8-20261027.noVK4C`.
Gaussian/outlier FP64 NRMS was about 3.2e-8, comparable to scalar FWHT, with
6/1 changed FP32 words; cancellation/zero, scalar fallback, changed-buffer
graph replay and allocation tests passed. This is not yet quantization,
backward, training or performance qualification. Before timing, the forced
scalar control is being corrected to cache its signs, matching the production
register-sign policy instead of comparing scalar reloads against TF32 caching.

### Batched transform qualification and integration decision

The cached-sign control was corrected and requalified. The native-BF16 padded
H8 variant then passed the combined **26-test** suite in 8.83s on the 5060.
Both variants retain guarded FP32 accumulation, scalar fallback, dense FP64
oracle checks, changed-buffer graph replay and stable live allocation gates.
Their small FP32 reassociation differences are not a byte-equality claim.

Single-node replay showed a host submission floor on small working sets.
Measurement revision 2 captures 32 transforms per graph for this study,
normalizes CUDA-event time per transform, and retains raw whole-window and
per-replay durations. CPU measurement contracts: **7 passed**. All six runs
completed successfully in snapshot `rtx-mseden-tf32h8-20261027.noVK4C`;
artifacts and qualification log are local under
`autotune_reports/ms_eden_speed20_20261026_5060/tf32-h8/`.

| Transform shape | TF32 scalar/candidate us | TF32 ratio | BF16 scalar/candidate us | BF16 ratio |
|---|---:|---:|---:|---:|
| 512x512 | 2.860 / 2.588 | 1.10149 | 2.857 / 2.514 | 1.13416 |
| 4096x512 | 17.498 / 14.706 | 1.19039 | 17.505 / 13.864 | 1.26110 |
| 4096x1536 | 105.155 / 104.704 | 1.00436 | 105.145 / 104.521 | 1.00603 |

All paired 24-round collections were stationary. BF16 ratio confidence
intervals were [1.12669,1.14065], [1.25624,1.26422], and [1.00566,1.00641],
respectively. Geometry was fixed at 128 threads / 256 blocks, not tuned.
These are repeated-buffer, pure-transform diagnostics, not full preparation
or backward speedups. The output is materialized as FP32; the large-case
near-tie is consistent with traffic hiding arithmetic savings, but no hardware
counters establish that explanation.

Decision: pursue the guarded BF16 H8 **inside** preparation, feeding existing
FP32 quantization/correction directly in registers. Do not add a standalone
transform pass or promote it based on these diagnostics. Integration must
address coalesced logical-transpose/grouped producers as well as row inputs,
and rerun original-source gradient, RNG, graph and memory gates before timing
the complete dX+dW path. Stable defaults are unchanged; the broad 1.20x goal
remains unmet.

### Fused H8 preparation: first complete-backward screen

Implemented `H8PreparationKernel` as a research-only subclass: shared source
loaders and FP32 quantization/correction remain intact, with only the VPL4
transform replaced. No intermediate tensor or launch. Guarded conversion
requires the existing signed FP32 inputs to be exactly BF16-representable
and in the qualified normal range; scalar fallback handles other products.
Grouped producers remain scalar; no default dispatcher selects this class.

Snapshot `/home/achiya/rtx-mseden-fusedh8-20261027.ShEh6J` passed 17 initial
CPU/GPU preparation tests in 10.87s. CPU harness/dispatch tests then passed
10 cases with 16 GPU cases skipped locally. Full compilation and backward
tests ran only on the 5060. The first comparison attempt lacked the snapshot's
`tests/cache_memory_checks.py`; it stopped before timing. The missing helper
was supplied and new `comparisons-v2/` outputs preserve the failed attempt.

All three v2 complete compiled backward runs finished, including FP32 gradient
checks, fresh captured RNG, changed-buffer replay and allocation guards. Shape
4096x1536x512, matched NVFP4 forward, original BF16 sources, 24 paired rounds,
15-second warmup, stable MXFP8 repository-default backward baseline:

| Candidate | MXFP8 / NVFP4 backward | 95% paired interval | Stationary rounds |
|---|---:|---:|---|
| Fresh scalar/grouped control | 0.92202 | [0.91986,0.92423] | yes |
| Row-only H8, grouped transpose retained | 0.92116 | [0.91389,0.93392] | baseline no |
| All H8, transpose reduced to VPL4 | 0.88019 | [0.87780,0.88249] | yes |

Local evidence: `autotune_reports/ms_eden_speed20_20261026_5060/fused-h8/`.
Row-only results cannot support a performance improvement, particularly with
nonstationary baseline timing. Sacrificing grouped-transpose preparation for
the current VPL4 H8 implementation regresses the complete path. Therefore
do not promote or spend a model-training run on this implementation. A further
H8 attempt must retain grouped layout benefits, or directly demonstrate a
preparation critical-path improvement; the pure-transform result alone was
insufficient. Broad 1.20x goal remains open, stable defaults unchanged.

### Grouped exact quotient refinement screen

Tested the already-proven guarded FP32 quotient refinement amortized over
VPL8/16 rather than four values. `ms_eden_grouped_refine.py` is an experimental
subclass, not a schedule/default change. A fragment-wide domain check chooses
the precise shared-reciprocal / two-FMA route or original division for the
entire fragment. Signed zeros are explicitly preserved. Hadamard, scales,
packing, FP32 correction and reductions retain their previous association.

Snapshot `rtx-mseden-groupedrefine-20261027.zcp4Ec` passed **16 byte-exact GPU
tests in 10.01s**, covering row/transpose, VPL8/16, Gaussian/outlier/zero/wide
exponents, changed-buffer graphs and live allocations. Local focused CPU
regression: 18 passed, 22 GPU/exhaustive cases skipped. Complete compiled
backward passed its FP32 gradient, graph RNG/current-buffer and memory gates.

At 4096x1536x512, 24 paired rounds / 15s warmup, all round collections were
stationary. Fresh scalar control: MXFP8/NVFP4=0.91651 [0.91406,0.91869],
median times 274.228/297.878 us. Grouped refinement: 0.89024
[0.88909,0.89145], 269.673/302.809 us. Separate baseline clocks differ, so
these are not a direct paired scalar-versus-refinement ratio; nevertheless
there is no improvement against the required matched baseline. Do not promote.
Artifacts, launcher and selection provenance are under local
`autotune_reports/ms_eden_speed20_20261026_5060/grouped-refine/`.

This eliminates another arithmetic-only hypothesis on the currently qualified
grouped schedule. Exactness is not sufficient for speed: guards, register
lifetimes and issue throughput must be measured before another refinement.
No counters were collected here, so none of these is asserted as the cause.

### Compiled-resource audit and grouped geometry search

Environment-only KEEP/DUMP controls produced no artifacts in three isolated
attempts (including a fresh cache and a direct test call). Explicit compiler
options `--keep-cubin --keep-ptx` succeeded; the Gaussian VPL16 transpose case
again passed its exact output gate. The reason environment-only dumps were
ignored is unverified. Actual binaries are retained locally in
`grouped-refine/compiler-audit-explicit/`.

Offline CUDA 13.2 cuobjdump, without initializing local GPUs:

| Buffered VPL16 producer | Registers/thread | Stack/local bytes | Static SASS instructions |
|---|---:|---:|---:|
| Ordinary division | 96 | 0 / 0 | 2052 |
| Grouped guarded refinement | 128 | 0 / 0 | 2596 |

Both have 1024 static shared bytes (dynamic staging is separate). Whole-function
static counts include fallback and slow-path code, not dynamically executed
instruction totals. Refinement adds 33% register allocation and about 26.5%
static instructions. No achieved-occupancy/counter claim follows from that.
The original contains separate `div.rn.f32` PTX operations for its normalized
values; the cubin contains reciprocal instructions, not a single-instruction
hardware exact divide. These facts motivate either reducing guarded temporary
live ranges or scheduling the existing producer better, not assuming division
removal automatically saves cycles.

A nine-case full-backward screen is running on the 5060 in snapshot
`rtx-mseden-groupedrefine-20261027.zcp4Ec`, `geometry-screen/`, launched by
`geometry-study.py`. It retains ordinary division, VPL16 transpose, staged
maxima, operand-paired streams and existing GEMMs. Only producer thread/block
counts change: (128,256), (64,128), (256,128), (64,256), (256,64), (128,128),
(64,512), (256,256), (128,512). Each fresh process measures complete compiled
dX+dW against stable MXFP8, with existing numerical/RNG/graph/memory guards,
12 paired rounds and 5s warmup. Screening results require stationary independent
confirmation and per-shape transfer before promotion.

The geometry screen subsequently completed all nine configurations, with all
fullgraph gradient/RNG/current-buffer/allocation guards passing. MXFP8/NVFP4
ratios in the listed order: 0.92206, 0.78197, 0.90677, 0.83587, 0.84873,
0.84754, 0.88924, 0.91641, 0.92146. All round collections were stationary.
The inherited 128-thread/256-block schedule remains best in this screen;
128/512 is effectively tied, not a demonstrated improvement. Independent
sessions had different absolute baseline clocks, so rank by paired ratios
and do not present raw candidate-time differences as direct paired gains.
All JSONs, manifests and logs are local under `grouped-refine/geometry-screen/`.
No geometry change is promoted; next work must address more than global
producer thread/block count. Goal remains incomplete.

### Grouped vector FP4 stores

Implemented an experimental grouped producer store-width variant (16/32/64
bits, bounded by each thread's packed fragment). Conversion, Hadamard,
normalization, EDEN statistics, corrected scales and reduction order are
unchanged. Packed bytes are retained in registers and copied contiguously,
without an extra allocation, tensor pass or kernel. Default remains byte
stores; only `ms_eden_grouped_store.py` selects the prototype.

Snapshot `/home/achiya/rtx-mseden-groupedstore-20261027.uzAyDU` passed **21
tests in 10.17s**: store contract plus 20 byte-exact row/transpose, VPL8/16,
32/64-bit store, Gaussian/outlier/zero/wide-range, graph replay and allocation
cases. CPU-only contract: 1 passed, 20 GPU cases skipped locally.

Complete compiled dX+dW at 4096x1536x512, same schedules and NVFP4 forward,
24 paired rounds, 15s warmup, original BF16 / unchanged FP32 policy:

| Grouped stores | MXFP8/NVFP4 | 95% interval | MXFP8 / NVFP4 median us |
|---|---:|---:|---:|
| 8 bits | 0.91874 | [0.91487,0.92137] | 272.761 / 295.942 |
| 32 bits | 0.94027 | [0.93466,0.94423] | 269.833 / 286.104 |
| 64 bits | 0.93121 | [0.92713,0.93428] | 287.155 / 308.757 |

All round collections were stationary and full-path gradient/RNG/current-buffer
and allocation guards passed. These separate sessions do not directly pair
scalar and vector producers, so a same-session complete-backward 8-vs-32-bit
confirmation is running (`paired-study.py`, `paired-store32.log` in the same
snapshot). This confirmation is an FP4 implementation A/B, not a substitute
for the required MXFP8 baseline ratio.

Actual cubin audit, 64-bit VPL16 transpose case: packed stores change from
16 `STG.E.U8` to two `STG.E.64`. Both kernels use 96 registers, zero stack/local
bytes and 1024 static shared bytes. Whole-function static instruction counts
are 2052/2067 (scalar/vector); packing adds instructions while reducing stores.
These are static observations, not hardware traffic counters. Qualification,
timings, source-selection provenance and binaries are locally retained under
`autotune_reports/ms_eden_speed20_20261026_5060/grouped-store/`.

Same-session 8-vs-32-bit confirmation completed with the original numerical,
RNG/current-buffer and live-allocation gates intact. Complete-backward paired
ratio scalar/vector = **1.01934**, clustered 95% [1.01131,1.02557], medians
296.467/290.266 us; round collections stationary. AB ratio 1.02475
[1.02010,1.02956], BA 1.01397 [0.99959,1.02485]. The pooled result supports a
small improvement, but BA's lower bound crosses one: retain it as a promising
experimental tuning option, not a universal/default win. Multi-shape transfer
and independent confirmation remain required. The measured MXFP8/NVFP4 ratio
still misses 1.20; no model-training benefit is claimed from this isolated A/B.

### Vector-store transfer and experimental tuner integration

Eight cross-shape comparisons completed in the same grouped-store snapshot.
Each run first compared complete scalar serial dX/dW to a compiled captured
candidate using identical signs/uniforms, original BF16 inputs and current
Gaussian/outlier/zero buffers. All full-shape outputs were byte-exact. Timing
then used independent current-buffer/fresh-RNG/FP32 and memory gates.

| Shape | Byte-store MXFP8/NVFP4 | 32-bit-store MXFP8/NVFP4 | Round stationarity |
|---|---:|---:|---|
| 4096x512x512 | 0.82247 | 0.85497 | byte candidate failed; vector passed |
| 4096x2048x512 | 1.04331 | 1.07217 | both passed |
| 4096x512x1024 | 1.00824 | 1.03143 | byte candidate and both vector paths failed |
| 4096x2048x2048 | 1.24170 | 1.27362 | both passed |

Local `grouped-store/transfer/` retains results and exact-gate evidence.
Nonstationary comparisons are not accepted performance improvements. Stationary
cross-session ratios support keeping width as a per-shape tuning option, not a
universal default. The broad 1.20 goal remains unmet.

Integrated `producer_grouped_store_bits` into experimental schedules, JSON
serialization, canonical producer compilation, runtime and study dispatch,
structural features and autotuner proposals. Default 8 remains unchanged;
VPL4 retains byte stores, VPL8 caps at 32, VPL16 permits 64. Proposals from a
non-grouped seed enter the existing grouped-transpose regime coherently.
Context revision 27 / feature revision 22 invalidates incompatible identities.
Focused CPU regression: 22 passed, 46 GPU/exhaustive cases skipped. One initial
mechanical test revision edit lost indentation; collection caught it, indentation
was repaired, and the focused suite then passed.

Canonical (not subclass) GPU qualification passed **26 tests in 10.47s** in
`rtx-mseden-store-api-20261027.XBlQPH`, including 16/32/64-bit choices and
effective-width/identity contracts. A final complete compiled backward run
through ordinary schedule JSON, without the research compiler patch, is running
there as `check.py` / `integrated-comparison.log`. No stable frontend default
or numerical policy changed.

The canonical schedule-JSON run subsequently completed: 4096x1536x512
MXFP8/NVFP4=0.93925 [0.93670,0.94158], median 268.992/286.353 us, both round
collections stationary. Fullgraph, fresh RNG/current buffers, independent FP32
gradient and allocation guards passed. Local `grouped-store/integrated/`
retains evidence. This agrees with the research-wrapper result and confirms
the option reaches the actual experimental runtime, not merely a benchmark.

### Updated critical-path profile and dual-dY prototype

Nsight Systems captured ten warmed backward graph replays for integrated
32-bit-store NVFP4 and stable MXFP8. Local artifacts and node-ID checked
summary: `grouped-store/integrated/profiles-current/`. These are diagnostic
node times, not ordinary timing or hardware-counter claims. NVFP4 had 23
kernels/replay (including two graph-related elementwise kernels), MXFP8 three.
Seven RNG kernels totaled about 11.87us median summed duration. Inductor already
batches seed generation; six Python random requests must not be described as
six seed launches. Combining random requests would change the legacy same-seed
contract; no RNG policy was changed.

NVFP4 diagnostic median producer onsets/durations (us): dX weight 19.25/15.13,
dW dY 27.47/74.36, dW X 70.03/42.19, row dX dY 99.14/87.62. The dW GEMM
ended near 206.57us; dX finalization ended near 210.06us and its GEMM near
250.17us. The MXFP8 graph span was 237.22us, including quad preparation
115.66us and GEMM durations 63.45/71.85us. Concurrent durations are not additive.
The launcher gives row dY no data dependency on those other producers, so its
late observed start motivates an admission/scheduling experiment; no counters
prove the underlying scheduler/resource cause.

Added a new-file-only `DualDYPreparationKernel` prototype: physical CTAs
alternate row/transposed dY roles, sharing one BF16 source but independent
signs, outputs and remapped partial indices. It reuses existing arithmetic
helpers and has a single shared-memory allocation. It does not share the input
load between roles. Row roles now pay the common resource envelope, and both
dY finalizers must wait for the whole grid. These costs may outweigh earlier
admission. Initial support deliberately pins the tested VPL4 row / VPL16 async1
coalesced_x8 transpose, native decode, ordinary division, warp maxima, vector32
transpose stores and equal CTA counts/common threads.

Preparation launch closures were split into invocation-owned producer/finalizer
descriptors without changing ordinary execution order, allocation ownership or
stable dispatch. A research launcher starts fused dY first, forks W/X from the
input frontier, and records dY completion before finalization extends the
caller frontier. It retains stream lifetime recording, exception joins and
kernel-only launch closures. CPU baseline regressions: 206 passed, 51 skipped;
new launch/dependency suite plus operand tests: 19 passed, 2 GPU skipped.

In `rtx-mseden-dualdy-20261027.fl1mEN`, 37 preparation/launcher checks passed
before an existing regression needed a missing test-only helper. After copying
that helper, the existing operand fullgraph/changed-buffer/opcheck CUDA test
passed separately. New GPU preparation checks cover idle CTAs, multiple ring
iterations/tails, independent current signs/uniforms and all five prepared
arrays byte-exactly. Full-shape exact-gradient qualification and complete
backward timing are running in `comparisons/` via `study.py`; no performance
claim or default promotion yet.

The two comparisons completed and were retrieved to local `dual-dy/`.
At 4096x1536x512, fused dY passed mandatory full-shape Gaussian/outlier/zero
same-draw byte-exact dX/dW, compiled graph/current-buffer, independent FP32,
fresh RNG and live-allocation checks. Complete backward was 309.656us against
268.108us MXFP8: ratio **0.86623** [0.86507,0.86735], both round collections
stationary. Fresh separate-producer control measured ratio 0.93449
[0.92491,0.94077], 270.001/286.951us; its candidate timing was nonstationary.
The fused result is not a win and does not support promotion. The common
resource envelope and whole-fused-grid finalization frontier remain plausible
costs, not counter-proven explanations. Default scheduling stays unchanged.

The follow-up changed only dY per-role grid counts (W/X unchanged), retaining
mandatory full-shape exact and runtime gates. At 64/128/512 blocks, complete
backward medians were 302.952/316.193/307.673us against MXFP8
269.326/268.127/268.682us; paired ratios 0.88855/0.84925/0.87418.
All reported round/order collections were stationary. Smaller fused grids do
not rescue this experiment. Raw results: `dual-dy/grid-study/` under the
5060 speed-target report root. The fused implementation remains unpromoted.

Next bounded experiment: compiler combo-kernel fusion of the six explicit RNG
requests, without changing their order or consumption. An isolated probe in
`benchmarks/probe_ms_eden_rng_combo.py` checks all output tensors and CUDA RNG
state exactly for three seeds on three representative shapes, plus fresh RNG
under graph replay. Initial exact checks passed. Single-call graph timing was
noisy and is not acceptance evidence; a 32-call graph diagnostic was launched.
This cannot establish complete-backward speed or meet the target on its own.

The batched RNG probe completed on the 5060: compiler-generated kernel counts
fell from six to one for all three shapes; exact six-output and post-call CUDA
RNG-state checks passed for seeds 941/942/943. Baseline/combo isolated medians
were 8.516/4.332us (4096x512x512), 11.033/7.355us (4096x1536x512), and
15.990/12.165us (4096x2048x2048). These are sequential diagnostic timings,
not paired stationary acceptance evidence. Raw results are in
`dual-dy/rng-combo-batched.json`. Scoped complete-backward control/combo
comparisons were launched in the same remote snapshot's `rng-study/`;
`benchmarks/experiment_ms_eden_rng_combo.py` changes compiler config only
inside the experiment, with no runtime/default changes.

Complete backward completed for decoder1536. Control: MXFP8/FP4 medians
270.708/288.191us, paired ratio 0.94099 [0.93893,0.94271]; all timing windows
stationary. Combo: 271.672/282.300us, ratio 0.96115 [0.95566,0.96510], but
BA candidate windows were nonstationary. Full runtime/numerical guards passed
in both processes. The roughly 6us candidate reduction is promising, not a
qualified speedup: rerun paired control/combo and inspect generated backward
before promotion. The 1.20 target remains unmet. Artifacts: `dual-dy/rng-study/`.

Direct interleaved FP4 control/combo confirmation (32 AB/BA rounds) completed:
289.505/282.430us; ratio 1.02498 [1.02149,1.02810]. AB ratio 1.02669
[1.02249,1.03192], BA 1.02327 [1.01749,1.02748]. Both round collections
and all four per-order collections are stationary (per-order status recomputed
from retained raw windows with runtime_timing_metadata). Public builder gates,
post-replay FP32 gradients and zero live-allocation growth passed. This confirms
a bounded ~2.5% compiler-option benefit on decoder1536, not the MXFP8 target.
`benchmarks/compare_ms_eden_rng_combo.py` makes this comparison reproducible;
the result is `dual-dy/rng-study/paired-combo.json`.

Next interaction screen combines combo fusion and grouped 32-bit stores with
row VPL8/16 (only the dX row ownership changes). Full-shape serial-versus-
compiled same-schedule Gaussian/outlier/zero checks precede complete-backward
timing. This rechecks a previously modest row-VPL effect after vector stores,
not a claim that previously rejected larger layouts are now universally useful.

The interaction screen completed: row VPL8 achieved MXFP8/FP4 medians
272.341/272.538us, paired ratio 0.99677 [0.99359,0.99891]; row VPL16 achieved
270.470/269.035us, ratio 1.00444 [1.00102,1.00764]. All round and AB/BA
timing collections passed stationarity; full-shape same-schedule exact checks
and public current-buffer/RNG/FP32/live-allocation gates passed. This closes
the decoder1536 deficit to approximate parity, not 1.20x. Source-control
defaults remain unchanged. Raw evidence: `dual-dy/row-combo/vpl{8,16}.json`.
Next: test transfer across shapes and profile the VPL16 + vector32 + combo
combination to identify the new critical path before another structural edit.

### Combined-setting transfer on the 5060

Eight complete-backward runs finished in `rtx-mseden-dualdy-20261027.fl1mEN`,
with full-shape serial/current-buffer exact gates before timing. Both arms
use combo compiler fusion and vector32 stores; retained uses the prior
per-shape row VPL, row16 changes only dX row ownership. All numerical/runtime
checks passed. Ratios below are MXFP8 time / FP4 time, not a direct paired
comparison of the two FP4 schedules.

| Shape MxNxK | Retained | Row16 | Timing qualification |
|---|---:|---:|---|
| 4096x512x512 | 0.90084 | 0.93413 | Both have a nonstationary order window |
| 4096x2048x512 | 1.09408 | 1.12582 | Both stationary; below target |
| 4096x512x1024 | 1.10050 | 1.12246 | Both candidate runs nonstationary |
| 4096x2048x2048 | 1.28436 | 1.28518 | Both pass complete timing target |

Row16 wide medians: 1010.720/785.652us MXFP8/FP4. Row16 decoder2048:
404.625/358.809us. No universal VPL16 default is justified: wide improvement
over retained is negligible, and two contexts require timing reconfirmation.
Artifacts retrieved under `dual-dy/row-transfer/`. The goal remains incomplete.

Added scoped `--combo-kernels` to the public backward profiling driver;
three CPU tests verify recording, restoration, failure restoration and rejection
outside public mode. Two row4/row16 diagnostic graph-node captures were queued
after all eight transfer results completed, in `profiles-combo/` on that same
snapshot. Only the 5060 is used; stable defaults remain unchanged.

Both profiles completed and were retrieved to `dual-dy/profiles-combo/`.
The summary validates matching node identities across ten graph replays;
two repeated-ID nodes are retained in counts/span but not individually
attributed. Diagnostic graph kernel count is 18/replay versus the historical
23 without combo fusion. The RNG distribution kernel plus fused payload
kernel take about 1.5+5.6us, consistent with fusion occurring in real backward.

Row4 versus row16 diagnostic spans: 245.83/235.67us (not ordinary benchmark
timings). The row dY producer falls from 87.39 to 60.18us, with 40/57 registers,
but still begins late: 92.05/94.71us. Row16 final dX preparation reduction
ends near160.28us, while its finalizer starts176.61us, then GEMM starts194.20us.
The dW GEMM overlaps156.53..196.68us. The profile suggests admission/resource
contention still limits exposed preparation and finalizer overlap; this is
not an occupancy-counter proof. Next ranked experiments are launch-order
interactions with the now-larger row fragment and shortening finalizer work,
not assuming more buffering or broad fusion will help.

### Scheduling interaction and paired finalizer

Five row16/vector32/combo preparation orders were screened with unchanged
stream topology and mandatory full-shape exact gates. Production order ratio
0.99897; row-first orders 0.96235 and 0.99867 (latter baseline nonstationary);
W-then-row order 1.01335 (stationary); dW-first 1.00767 (candidate BA
nonstationary). These are separate stable-MXFP8 comparisons, not direct paired
proof of an order benefit. No launch-order default was changed. Results:
`dual-dy/combo-orders/`.

Added research-only `ms_eden_paired_finalizer.py`: two distinct normalized FP32
scale values share one E4M3x2 conversion. Division, rounding-probability math,
uniform indexing, physical scale layout and FP32 outer factors are unchanged.
Stable finalizer remains intact. Ten focused 5060 GPU cases passed in 5.20s:
row-major/mma128, single/staged partial maxima, ordinary/power2 probability
math, independent positive FP8 format/tie oracle, exceptional input equality,
changed-buffer graph replay, zero scale, ragged rows and multi-iteration grids.
The complete-backward experiment is running in `paired-finalizer/` with the
same numerical/runtime gates; conversion throughput is not yet a speed claim.

Paired-finalizer timing completed: MXFP8/FP4 medians268.351/279.200us,
ratio0.95792 [0.95288,0.96185], with nonstationary candidate BA windows.
Full-shape same-draw and public numerical/runtime guards passed, but this is
worse than the retained ~269us row16 path and fails acceptance. Do not promote
paired conversion. Fewer conversion instructions do not imply better complete
backward performance: this implementation also doubles per-thread work and
live values. That is a source-level tradeoff, not measured occupancy evidence.
Raw comparison, full-shape qualification and ten-case test log are retrieved
under `dual-dy/paired-finalizer*`. The 1.20x broad goal remains unmet.

### Original-finalizer geometry and cooperative partial reduction

Six dX-A-only original-finalizer geometries were screened, retaining row16,
vector32 and combo fusion. (threads,blocks) and MXFP8/FP4 ratios:
(128,224)1.00728; (64,448)1.00235; (128,448)1.00686;
(256,112)0.99741; (256,224)1.00660; (512,112)1.00339.
Only (256,112) had a nonstationary candidate BA window; all numerical/runtime
gates passed. No geometry win to promote. Results: `dual-dy/finalizer-geometry/`.

Research `ms_eden_cta_finalizer.py` now reduces producer maxima cooperatively
across the CTA before the unchanged scalar finalization loop. It replaces
each warp's duplicated scan with disjoint per-thread reads, a warp reduction,
one small SMEM array/barrier, and a shared maximum. It is intended for inline
amax, avoiding separate reduction launches; no stable runtime change.
Twelve GPU cases passed in7.27s across64/128/256/512threads,1/3/1024partials,
changed-buffer graphs, tail maximum, zero and wide exponent range. Final scales
and FP32 outer factors match the original bitwise; outer factors also match
an independent torch maximum oracle. Complete-backward comparison is running
in `cta-finalizer/` after those gates. No performance claim yet.

CTA finalization completed: MXFP8/FP4 medians269.343/272.708us, ratio0.98722
[0.98258,0.99077], candidate BA window nonstationary. Same-draw full-shape,
independent FP32/current-buffer/RNG/memory guards passed, but no complete-path
improvement over the retained staged reduction is demonstrated. Do not promote.
Both finalizer restructurings and geometry tuning failed to close the gap;
next substantial work should return to the dominant rotation/quantization
producers rather than repeatedly polishing this smaller phase.

### Isolated competing producer and streamed quotient refinement

The dW-dY-only grid screen completed, leaving all other producer geometries,
row16/vector32/combo and GEMM schedules unchanged. Blocks32/64/128/256/512
gave MXFP8/FP4 ratios0.89798/0.96971/0.99114/1.00169/0.99716. All timing
windows and numerical/runtime gates passed. Reducing this competing grid did
not demonstrate a benefit; retain256. Artifacts: `dual-dy/producer-reservation/`.

The prior grouped quotient refinement retained a whole normalized fragment
and raised register usage. A research-only `streamed=True` option now calls
the existing guarded exact quotient helper pair-by-pair, avoiding that extra
normalized fragment. The ordinary producer still has refinement disabled.
No approximation or rounding policy was introduced. Thirty-two 5060 GPU
cases passed in8.09s for fragment/streamed, row/transpose, VPL8/16 and
Gaussian/outlier/zero/wide-range inputs with vector32 stores, including
packed/scale/partial byte equality, changed-buffer graph replay and live memory.
Local CPU collection:2passed,56GPUskipped. Fresh control/streamed comparisons
run in `/home/achiya/rtx-mseden-streamed-20261027.en7WaB`; prior snapshot sources
are preserved. Register-pressure reduction remains a hypothesis pending audit.

Fresh comparison completed: ordinary producer 270.929us versus matched
MXFP8 270.484us, ratio 0.99833 (stationary).
Streamed refinement 289.324us versus MXFP8 269.411us, ratio 0.93144;
baseline BA nonstationary. Full numerical/runtime gates passed but the
streamed variant does not improve complete backward and is not promoted.
The focused CPU regression found an old feature-revision assertion21 after
the earlier vector-store revision22; corrected that assertion and reran:
9passed,28GPUskipped. Raw comparisons: `streamed-refinement/` under the
5060 report root. Compiler dump attempts emitted no artifacts even after
documented no-cache controls; a short Nsight launch-resource audit is running
instead. No register or occupancy improvement is claimed without that evidence.

Nsight resource audit completed: the qualified transpose-VPL16 two-stage
probe uses 96 registers/thread for both ordinary and streamed producers, with
17404B dynamic shared memory each. Thus the streamed implementation avoids
the earlier fragment-wide refinement's 128-register allocation on this probe,
but does not beat ordinary division end-to-end. This is launch-resource
evidence for that two-stage probe, not occupancy counters or proof of the
one-stage full-shape schedule's resources. Keep ordinary division. Raw
`resource-audit.sqlite` and `.nsys-rep` plus qualification log are saved in
`streamed-refinement/`.

### Current-setting short transformer translation

Completed four fresh 21,121,024-parameter TinyStories decoder runs on the
5060:512updates/2,097,152byte tokens each, seeds941/942 with opposite policy
orders. Both policies use identical NVFP4 regional-exact forward configs,
original BF16 sources, compiled graphs and explicitly enabled pointwise
combo fusion. FP32 norms/residual sums/loss/clipping and BF16 matrix/optimizer
state recipe are unchanged. The FP4 candidate uses fixed row16/vector32
operand-paired schedules; no rejected refinement/finalizer prototype is active.
Some narrow-shape schedule timing windows remain unqualified, so these are
explicit proposals, not promoted runtime winners.

| Seed | MXFP8 bwd training seconds | FP4 bwd training seconds | Ratio | MXFP8 bwd final val | FP4 bwd final val |
|---|---:|---:|---:|---:|---:|
|941|9.94527|9.93762|1.00077|0.992125|0.994944|
|942|10.00403|9.95216|1.00521|0.949978|0.975585|

Total job wall (includes compile/eval/checkpoint) was43.832/45.877s and
25.810/34.729s for MXFP8/FP4 respectively. All runs completed with finite
parameters. The analyzer verified matched initialization, corpus, validation
offsets and model/optimizer recipes per seed. This is near-parity training,
not the requested speedup and not long-convergence evidence. Raw JSONL/logs,
schedule map and per-seed analyses: `training-combo/` under the5060 report root.
Training CLI now exposes opt-in `--combo-kernels`, records it in checkpoint
identity and passes options explicitly to compilation. Focused CPU suite:
85passed. Legacy default identities are preserved when the flag is absent.

During training, reviewed NVIDIA's [PTX floating-point ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma):
FP32x2 mul/fma were introduced in PTX8.6 and support SM100+. Research
`ms_eden_paired_fp32.py` pairs the norm/dot correction statistics using these
instructions with RN rounding, no FTZ, and the exact existing b-product/a-FMA
tree. Sixteen5060 producer byte/graph/memory cases passed in7.47s, including
row/transpose,VPL8/16,zero,outliers and wide exponents. Complete-backward
timing is running; ISA support alone is not a throughput claim.

Paired FP32 statistics comparison completed:270.425/269.592us MXFP8/FP4,
ratio1.00247 [0.99801,1.00583], all timing windows stationary and full
numerical/runtime guards passed. This does not demonstrate a meaningful gain
over the retained ordinary-statistics row16 path. No promotion. Raw comparison
and16-case qualification log: `paired-fp32/`. The paired-instruction idea
would need application to the larger Hadamard arithmetic region and verified
lowering before a broader performance conclusion; this test changes only
correction statistics.

### Paired FP32 Hadamard arithmetic: exact but rejected on SM120

Extracted the unchanged ordinary grouped butterfly into a helper and added
research-only paired register butterflies and paired shuffle-level arithmetic.
The same seven levels, FP32 RN operations, signs and reduction tree remain.
Forty-eight5060 cases passed in12.71s for statistics/transform/both, row and
transpose,VPL8/16 and four distributions, with packed/scales/partials byte-exact,
changed-buffer graph replay and stable live allocations. CPU:2passed,72skipped.

Complete decoder1536 timing: transform-only338.512us versus MXFP8262.264us,
ratio0.77411 (stationary); combined343.365us versus MXFP8265.566us,
ratio0.76790 (candidate nonstationary). Full runtime gates passed but both
variants clearly fail acceptance. Scalar butterfly remains the default.

Direct `runner.__cubin__` / `runner.__ptx__` extraction with keep flags worked
even though file-dump attempts had not. For the actual4096x1536 transpose,
VPL16,128threads/256blocks,stage1 configuration, offline cuobjdump reports:

| Variant | Registers/thread | Static instructions | MOV | FADD | FFMA |
|---|---:|---:|---:|---:|---:|
| Ordinary |96|1925|147|238|252|
| Paired butterfly |244|2128|333|334|252|
| Paired butterfly + statistics |234|2119|332|334|252|

All report zero local/stack bytes. The f32x2 PTX lowers to scalar FP32 SASS
on this tested SM120 configuration, with substantially more moves and register
demand. Instruction counts are static and include cold/fallback paths, not
executed-instruction counters. This invalidates assuming native paired FP32
throughput merely from PTX acceptance. It supports rejecting this implementation;
no general hardware throughput claim beyond inspected binaries is made.
Raw evidence: `paired-transform/` (comparisons,48-case log,PTX,CUBIN,SASS,
resource dumps and summary). Remote snapshot:
`/home/achiya/rtx-mseden-fp32x2-20261027.zdngBV`.

### dW gradient producer CTA geometry follow-up

Ran `benchmarks/experiment_ms_eden_producer_threads.py` on the 5060 only,
varying dW/dY producer threads and blocks while retaining row16, grouped
store32, scalar butterfly, operand-paired scheduling and scoped RNG combo
fusion. Shape4096x1536x512;24 interleaved AB/BA rounds,16 graph calls per
window,15s warmup. All six cases completed same-draw Gaussian/outlier/zero
checks, current-buffer graph guards and FP32 gates; live replay growth was0.

| Threads / blocks | MXFP8 us | FP4 us | Paired MXFP8/FP4 | All timing windows stationary |
|---|---:|---:|---:|---|
|128 /256 (control)|266.995|267.436|0.99895|yes|
|64 /256|270.651|273.894|0.98723|yes|
|64 /512|274.584|276.101|0.99070|no|
|256 /256|269.191|269.684|0.99446|no|
|256 /128|276.148|272.085|1.01362|yes|
|512 /64|264.091|294.446|0.89756|yes|

The256/128 result is a small within-session advantage over MXFP8, not proof
of improvement over the FP4 control run in a different process. No case
approaches1.20x; no schedule promotion. This further limits the claim that
producer CTA resource sizing alone closes the narrow-shape gap. A structural
reduction in preparation work remains necessary, unless a different phase
reveals substantially more headroom. Raw records and per-case logs are in
`autotune_reports/ms_eden_speed20_20261026_5060/producer-threads/`.

### Grouped exact reciprocal table: first lowering rejected

Added research-only `reciprocal_lut` to `compile_grouped_refine`. It uses the
existing exact lane reciprocal and uniform shuffle before eligibility branches;
the FP32 quotient refinement, fallbacks and signed-zero behavior are unchanged.
All32 producer cases passed on5060 in11.95s (row/transpose,VPL8/16,
Gaussian/outliers/zero/wide-range,fragment/streamed, changed-buffer graphs and
stable live memory). The tests compare every output byte to ordinary division.

Complete4096x1536x512 results,24 rounds:

| Variant | MXFP8 us | FP4 us | Paired ratio | All timing windows stationary |
|---|---:|---:|---:|---|
|Ordinary control|279.087|279.117|0.99838|yes|
|Fragment table|271.157|282.963|0.95898|yes|
|Streamed table|293.384|316.076|0.92561|no|

No promotion. Compiler artifacts show96 registers for ordinary/fragment and95
for streamed, no local/stack bytes. Crucially fragment PTX still contains
lane-table `div.rn` inside the tile loop (line697, after loop label311), followed
by the shuffle. Loop-invariant construction was not hoisted at this IR stage;
this implementation does not realize once-per-warp setup. The next meaningful
variant must explicitly carry the pre-loop reciprocal through grouped row and
transpose helpers, not assume compiler hoisting. These results reject this
implementation, not the untested explicitly hoisted one.
Evidence: `autotune_reports/ms_eden_speed20_20261026_5060/grouped-lut/` including
PTX/CUBIN, timing records and32-case qualification log. No GPU job remains.

### Explicitly hoisted grouped reciprocal: correctly lowered, still rejected

Carried the cached lane reciprocal explicitly from grouped-row and buffered/
unbuffered transpose entry points through the grouped preparation helpers.
The compile-time flag remains false in ordinary kernels; only the opt-in
research refinement factory sets it.64 exact GPU cases passed in14.06s,
including the existing no-table variants, graph input changes and live-memory
checks. No numerical gates or RNG policy changed.

Generated fragment PTX now constructs the lane reciprocal at line43, before
the tile-loop label at316, and reuses it at701/1801. This realizes the intended
pre-loop setup, unlike the previous prototype. Ordinary, fragment and streamed
CUBINs each report96 registers/thread and zero local/stack bytes.

Complete4096x1536x512 backward,24 interleaved rounds on5060:

| Variant | MXFP8 us | FP4 us | Paired ratio |
|---|---:|---:|---:|
|Ordinary control|269.052|267.717|1.00385|
|Hoisted fragment|270.955|281.412|0.96363|
|Hoisted streamed|273.800|290.207|0.94328|

Every aggregate and AB/BA timing window was stationary; all complete-backward
gates passed and replay live-allocation growth was0. No promotion: removing
per-tile reciprocal setup does not overcome the refinement path's total cost.
Retain ordinary division. Further work should target a larger preparation
change, not assume reciprocal amortization alone closes the20% gap.
Raw evidence: `autotune_reports/ms_eden_speed20_20261026_5060/grouped-lut-hoisted/`.
Remote snapshot: `/home/achiya/rtx-mseden-hoisted-20261027.TvUMsu`.
All study and artifact extraction processes completed.

### Grouped threshold packing and scale-free correction

Extended the existing guarded VPL4 scale-free numerical option to research-only
VPL8/16 grouped producers, reusing the exact threshold packing and power-of-two
normalization bounds. This removes per-value division on the eligible path;
correction products/reductions remain FP32. It is a reassociation of correction,
not a byte-identical replacement for division; stable defaults remain unchanged.

New grouped tests cover both layouts and VPL8/16 across zero, Gaussian,
outliers, sparse, tiny1e-30 and large1e30 inputs. Packed codes must match ordinary
division exactly. Corrected scales must remain within one BF16 ULP both of
ordinary preparation and an independent CPU FP64 H128/EDEN oracle. Changed
buffers replay through a captured graph, and repeated replay has no live-memory
growth.36 tests passed in12.14s, including32 grouped reciprocal regressions.
Initial isolated-snapshot dependency and staged-code initialization failures
were fixed before any performance run; logs are retained.

Complete4096x1536x512 backward (24 paired AB/BA rounds,15s warmup):

| Variant | MXFP8 us | FP4 us | Paired ratio |
|---|---:|---:|---:|
|Ordinary control|270.647|268.417|1.00827|
|Grouped scale-free|283.337|321.043|0.88158|

All timing windows stationary, all complete runtime/numerical gates passed,
zero live-allocation growth. CUBIN resource report shows96 registers/thread
for ordinary and103 for scale-free, zero local/stack bytes. This is not a
performance win; do not promote. Fewer divisions alone does not outweigh the
guards, threshold packing and extended lifetimes in this implementation.
Resource counts are compiler evidence, not executed occupancy counters.
Evidence: `autotune_reports/ms_eden_speed20_20261026_5060/grouped-scale-free/`.
Remote: `/home/achiya/rtx-mseden-grouped-scale-free-20261027.M7Epzn`.
No remaining live jobs from this study.

### Streaming scale-free normalized values: no resource or latency win

Added a research-only `scale_free_streamed` switch. The domain guard still
examines the entire16-value scaling block, but normalized values are recomputed
per pair instead of retaining a second fragment. Eight GPU cases passed in
11.98s: both layouts,VPL8/16,both storage variants,six distributions. Streamed
packed/scales/partial outputs are byte-identical to fragment scale-free; both
also pass the independent FP64 corrected-scale bound and replay-memory checks.

Complete4096x1536x512 backward: ordinary control267.194us versus MXFP8268.785us
(paired ratio1.00409); streamed scale-free302.635us versus MXFP8268.872us
(ratio0.88947). All aggregate and AB/BA windows stationary; complete numerical,
RNG/current-buffer and memory gates passed, zero live replay allocation growth.

The intended resource reduction did not happen: CUBIN reports106 registers
for streamed versus103 for fragment and96 for ordinary, all zero local/stack
bytes. Source-level recomputation is not proof of shorter compiled lifetimes.
Reject promotion and stop this scale-free arithmetic line for now; retain the
ordinary producer. Next investigation should return to eliminating preparation
passes or the preparation-to-GEMM dependency chain, not another local arithmetic
variant without new profiler evidence.
Raw records, logs and PTX/CUBIN:
`autotune_reports/ms_eden_speed20_20261026_5060/scale-free-streamed/`.
Remote snapshot `/home/achiya/rtx-mseden-scale-free-streamed-20261027.uH031L`.
All study processes completed; stable defaults unchanged.

### Admission dependencies: retain concurrent dY producers

Added research launcher `benchmarks/experiment_ms_eden_dependency.py`, using
the invocation-owned preparation split to delay dW/dY until a selected dX
frontier. This changes graph edges, not enqueue order alone. Input/RNG waits,
record_stream lifetimes, caller joins and exception cleanup remain. Producer
completion is recorded before caller finalization so that variant allows dW
preparation to overlap dX finalization/GEMM, without accidentally waiting for
the entire caller stream. Stable production launcher is untouched.

Complete4096x1536x512 backward on5060,24 paired rounds:

| dW admission frontier | MXFP8 us | FP4 us | Paired ratio | All windows stationary |
|---|---:|---:|---:|---|
|Original concurrent control|269.770|268.442|1.00179|no|
|After dX/dY producer|268.510|288.066|0.93670|yes|
|After dX/dY finalizer|268.942|296.248|0.90716|yes|
|After dX GEMM|264.816|295.786|0.89298|no|

All exact full-shape serial-versus-paired Gaussian/outlier/zero comparisons,
compiled current-buffer/RNG/FP32 gates passed; replay live growth0. Two runs
had nonstationary order windows and are not eligible winners. Neither stable
delayed variant beats MXFP8. Reject admission delays: overlap is more valuable
than the proposed contention reduction for this tested shape/configuration.
This does not establish that different kernels could not benefit from another
dependency graph. Focus future work on reducing required preparation/finalizer
work rather than artificially delaying the current producers.
Evidence: `autotune_reports/ms_eden_speed20_20261026_5060/dependency/`.
All four remote cases completed; no remaining study process.

### Paired staged reductions and scale finalizers: promising, unpromoted

Implemented research `ms_eden_dual_finalizer.py`: one two-CTA maximum reduction
launch plus one disjoint-CTA finalizer launch for both operands of a GEMM.
It reuses ordinary per-element FP32/SR finalization, caller-owned uniforms,
tensor factors and native physical scale layouts. Four launches become two
per fused gradient by construction. Producers remain independent, but paired
finalization waits for both producer outputs. Invocation-owned preparation
descriptors now expose optional finalizer metadata/buffers; no tensor-global
cache or default launch change. Seeded/inline-amax research modes are rejected.

Eight5060 kernel tests passed in7.29s after resolving CuTe helper ownership:
row-major/mma128,divide/power2,asymmetric shapes/CTA counts,partial counts1/3
and1031/77,zero and changing buffers,byte-exact scale/maximum/outer outputs
against independent original launches,stable live allocations. Nine focused
descriptor/launcher regressions passed in1.39s.

Complete4096x1536x512 backward against MXFP8,24 paired rounds:

| Fused legs | MXFP8 us | FP4 us | Paired ratio | All windows stationary |
|---|---:|---:|---:|---|
|None|294.983|294.912|0.99972|yes|
|dX|276.171|273.141|1.00987|no|
|dW|283.616|282.742|0.99873|no|
|Both|271.617|262.148|1.03530|yes|

All full-shape serial-versus-paired exact checks and complete numerical/RNG/
current-buffer gates passed; replay live growth0. Direct same-session FP4
unfused-versus-both comparisons then gave269.589/260.118us,ratio1.03646
CI[1.03154,1.04063],and repeat295.076/282.275us,ratio1.03965
CI[1.03602,1.04277]. Both direct runs failed at least one order-specific
stationarity window, despite stationary aggregate rounds. These are promising
3–4% point estimates, NOT accepted schedule promotion or the1.20x target.
Do not compare absolute medians between processes. Next: confirm on additional
shapes and obtain acceptable direct timing quality before integrating selection.
Evidence: `autotune_reports/ms_eden_speed20_20261026_5060/dual-finalizer/`.
Remote `/home/achiya/rtx-mseden-dual-finalizer-20261027.hZRkVQ`.
All study processes completed; defaults unchanged.

### Paired finalizer cross-shape transfer

Extended the research guarded launcher with `--shape`. Tested the four other
decoder projections using each shape's retained schedule, rowVPL16/store32,
ordinary FP32 preparation and scoped combo fusion. For every shape, ran both
the complete MXFP8 comparison and a direct interleaved unfused/fused FP4
comparison; did not infer fusion gain from separate-process absolute timings.

| Shape M,N,K | MXFP8/fused FP4 | All windows stationary | Unfused/fused FP4 | All windows stationary |
|---|---:|---|---:|---|
|4096,512,512|0.94443|no|1.01779|no|
|4096,2048,512|1.13649|yes|1.00894|yes|
|4096,512,1024|1.13491|no|0.94804|no|
|4096,2048,2048|1.28933|yes|0.99507|yes|

All exact full-shape and compiled gradient/current-buffer/RNG guards passed;
no live replay allocation growth. Qualified decoder2048 medians:408.241 versus
358.873us MXFP8/fused FP4; direct FP4 control/fused365.790/362.820us. Wide:
1010.052/783.188us MXFP8/fused; direct control/fused844.147/849.386us. These
different absolute medians illustrate why the direct paired comparisons matter.

Fusion is not a universal improvement: a qualified~0.9% gain on decoder2048
and~0.5% loss on wide. Other shapes have unacceptable timing windows and remain
unqualified, including the decoder1024 regression estimate. Keep fusion an
experimental scheduling choice; do not replace stable/default launch behavior.
Wide still exceeds1.20x, but the representative suite and training target do
not. Finalizer fusion alone cannot close the remaining narrow-shape deficit.
Records: `autotune_reports/ms_eden_speed20_20261026_5060/dual-finalizer/cross-shape/`.
All eight comparisons completed on5060; no live job remains from this sweep.

### Refreshed matched narrow-shape timelines

Captured six fresh Nsight Systems node-level traces on5060: decoder512 and
decoder1536, each with stable MXFP8, retained ordinary FP4, and fused-finalizer
FP4. Exact retained schedules,row16/store32,scoped RNG combo fusion; ten warmed
complete backward graph replays per capture, initialization/compilation outside
cudaProfilerApi range. Raw command/selection/result JSON,nsys-rep andSQLite are
in `autotune_reports/ms_eden_speed20_20261026_5060/profiles-finalizers/`.
`node-summary.json` validates graph-node identities across all ten replays;
duplicate-ID nodes are counted in span/node totals but not uniquely attributed.

| Shape | MXFP8 nodes/span us | Ordinary FP4 nodes/span us | Fused FP4 nodes/span us |
|---|---|---|---|
|4096,512,512|3 /90.12|18 /102.31|14 /106.71|
|4096,1536,512|3 /238.51|18 /237.79|14 /228.54|

These spans are profiler diagnostics, NOT fresh ordinary latency acceptance.
Do not add overlapping per-kernel times. Fusion physically removes four nodes,
but tiny-shape trace span worsens; the ordinary event-based evidence remains
authoritative for performance.

For decoder1536 ordinary FP4, row-dY starts95.55us and ends155.58us; its
finalizer ends195.96us and dX GEMM ends237.79us. With fused finalizers row-dY
starts94.46us/ends156.08us, paired finalization ends187.73us and dX ends228.54us.
The MXFP8 quad-operand quantizer ends116.00us; its final GEMM ends238.51us.
FP4 GEMM durations are about39–41us versus64–72us MXFP8 in these captures,
but the preparation frontier erodes that advantage. On decoder512 the
MXFP8 quad quantizer ends34.90us; ordinary FP4 row-dY ends64.56us and dX
finalization ends75.59us. This localizes the next higher-value experiment to
producer consolidation/admission rather than another reciprocal variant.

Hypothesis, not established counter diagnosis: one explicitly partitioned
four-operand producer grid could admit row-dY work earlier than four separate
large producer grids. Previous dual-dY fusion regressed and must be respected;
an all-operand variant needs independently tunable CTA shares and full-path
validation, not assumed benefit from fewer launches. No new tensor-pipe/DRAM/
occupancy counters were collected in this timeline refresh. All captures ended.

### Compact four-operand producer: qualified, no win in first allocation sweep

Implemented research `ms_eden_quad_preparation.py`: CTA-uniform role branches,
one shared-memory allocation, original grouped transform/quantization/FP32
correction helpers, and compact interleaved ownership for four independently
sized grids. Host-computed bands map every physical CTA to exactly one
(role,logical CTA), without inactive padding CTAs. All finalizers wait for the
whole fused grid. This reads dY twice for the distinct logical orientations;
it does not claim cross-role operand reuse or eliminate either quantization.

Five tests passed on5060 in20.02s: exhaustive ownership enumeration for sampled
counts, equal/asymmetric role grids,VPL8/16, exact packed/scales/partial arrays
against independent original launches,Gaussian/outliers/zero/tiny inputs,
changed-sign/input graphs and stable live allocations. The full public compiled
comparisons also passed same-draw serial/paired and FP32/RNG/current-buffer gates.

Complete4096x1536x512 backward,24 paired rounds; allocations are
(dX dY,dX W,dW dY,dW X):

| Allocation | MXFP8 us | FP4 us | Paired ratio | All windows stationary |
|---|---:|---:|---:|---|
|Ordinary independent control|269.297|269.322|0.99969|yes|
|256,256,256,256|267.898|296.502|0.90028|no|
|128,16,128,48|263.879|306.171|0.86068|yes|
|256,32,128,64|266.984|290.214|0.91632|no|
|512,32,256,128|268.119|287.075|0.93017|no|
|128,16,64,32|265.371|300.012|0.88161|no|

No candidate approaches1.20x. Nonstationary runs cannot be promoted. The actual
equal-grid CUBIN reports96 registers/thread, zero local/stack bytes, not a
register explosion. However all roles now share the transpose resource envelope
and all finalizers wait for all producers. Those costs remain plausible causes
of regression; the binary alone does not establish executed occupancy. This
limited allocation sweep does not exhaust every four-role schedule, but supplies
no reason to replace the concurrent independent producers. Stable defaults and
MXFP8 backward remain unchanged.
Evidence: `autotune_reports/ms_eden_speed20_20261026_5060/quad-preparation/`.
Remote `/home/achiya/rtx-mseden-quad-20261027.eHQKUf`; all study processes ended.

### Grouped-producer CTA maxima revisit

Requalified existing CTA maxima on the current VPL8/16 grouped row/transpose
paths before timing: four GPU tests passed in11.29s, exact packed/temporary/
final scales/factors, identical global maxima, changed-source/sign/uniform
graph replay across Gaussian/sparse/outlier/zero inputs. The producer emits
256 rather than1024 partials at128threads/256CTAs, paying an extra CTA barrier.
No new numerical policy or production default was introduced.

Complete4096x1536x512 backward,24 paired rounds:

| CTA-max legs | MXFP8 us | FP4 us | Paired ratio | All windows stationary |
|---|---:|---:|---:|---|
|None/control|269.845|267.969|1.00755|yes|
|dX|274.405|273.877|1.00083|yes|
|dW|270.191|271.612|0.99397|yes|
|Both|278.327|276.677|1.00249|yes|
|Both + paired finalizers|276.062|268.433|1.02292|no|

All complete numerical/RNG/current-buffer gates passed; replay live growth0.
No evidence to promote CTA maxima. The paired-finalizer variant has an unstable
order window and does not establish an incremental benefit from CTA scope.
Keep the retained warp-partial producer. Next higher-value design is genuine
shared dY input-tile reuse across row/transpose preparation; previous fusion
only combined launches while still reading dY twice. It must preserve separate
orientation-specific signs, scales and FP32 corrections, not reuse invalid
transposed quantization blocks.
Evidence: `autotune_reports/ms_eden_speed20_20261026_5060/grouped-cta/`.
All five cases completed on5060; no remaining job from this sweep.

### True shared dY tile reuse: exact first implementation, not yet faster

Implemented research `ms_eden_shared_dy.py`. Each128x128 BF16 source tile is
copied once using128-bit cp.async into a32KiB swizzled SMEM tile; row and
column groups read it logically in both orientations. Signs,H128 transforms,
packing,FP32 EDEN statistics and corrected scales remain distinct. No GMEM
transpose or reuse of incorrectly oriented quantization blocks. This removes
one logical dY source read, unlike earlier dual/quad launch fusion.

Eight tests passed in15.26s for128x128 and384x256,128/256threads,VPL8/16,
three-CTA grids including idle and multi-tile CTAs,Gaussian/outlier/tiny/zero,
changing signs/input graph replay and stable live allocations. Packed/scales
match independent original producers byte-for-byte. Partial assignments differ
by design; their global maxima match exactly and equal maximum corrected scale.

Full4096x1536x512 backward,24 paired rounds, original BF16, ordinary correction:

| Shared producer threads/blocks | MXFP8 us | FP4 us | Paired ratio | All windows stationary |
|---|---:|---:|---:|---|
|Independent control|268.467|267.979|0.99674|no|
|128/256|266.237|289.331|0.91839|yes|
|256/64|260.909|304.780|0.85086|no|
|256/128|268.486|286.331|0.93595|yes|
|256/256|266.656|286.624|0.93133|yes|
|128/64|265.805|290.773|0.91281|yes|

All full-path same-draw/FP32/current-buffer/RNG/memory gates passed. No promotion:
the single-stage32KiB input reuse implementation remains slower despite reduced
logical source loads. No DRAM counter reduction is claimed. Next inspect/test
the SMEM-to-register load pattern (currently scalar), because shared-input
reuse can substitute bank/LSU costs for GMEM traffic. A vectorized row load
and layout alternatives need exact qualification and complete-path measurement;
do not infer benefit simply from fewer global reads.
Evidence: `autotune_reports/ms_eden_speed20_20261026_5060/shared-dy/`.
Remote `/home/achiya/rtx-mseden-shared-dy-20261027.NSVeb7`; all runs completed.

### Shared dY: vector SMEM loads and alternate swizzle (2026-10-27)

Only the authorized 5060 was used; the 5070 remains excluded. The previous
turn recovered 24 passing GPU cases (21.71 seconds): two shapes, two thread
counts, two values-per-lane choices and three load/layout variants. Each case
checks Gaussian, outlier, tiny and zero inputs, distinct changing signs,
exact packed values/corrected scales, exact global maximum, changed-input
graph replay and stable live memory against independent ordinary producers.

Complete compiled backward for 4096x1536x512, shared producer 256 threads,
128 blocks, VPL16, 24 AB/BA paired rounds against stable MXFP8:

| Row load bits / swizzle | MXFP8 us | FP4 us | Paired MXFP8/FP4 | All order windows stationary |
|---|---:|---:|---:|---|
|128 / 1|269.612|275.909|0.97755|yes|
|128 / 0|269.144|274.554|0.98074|yes|
|16 / 0|267.805|285.908|0.93329|no|

All three completed same-draw full-gradient, FP32-reference and replay checks.
The vector path is promising relative to the scalar research variant, but
these are separate sessions against MXFP8, not a paired scalar/vector claim.
Neither vector variant beats MXFP8; no promotion or default change.

Generated CUBIN confirms two LDS.128 instructions replace sixteen scalar row
loads; the column path retains sixteen LDS.U16 instructions. Both variants
use 128 registers, no local/stack allocation and no LDL/STL instructions.
The tile itself is 32 KiB dynamic shared memory; cuobjdump's additional 1024
static shared bytes must not be mistaken for the entire allocation. This is
generated-code evidence, not measured bank-conflict/occupancy counters.

The research wrapper now fingerprints its kernel and launcher sources for
future records. A bounded 128-thread geometry check follows; stable runtime
policy remains unchanged. Evidence and compiled artifacts:
`autotune_reports/ms_eden_speed20_20261026_5060/shared-vector/`.
Remote snapshot: `/home/achiya/rtx-mseden-shared-vector-20261027.5w2IVH`.

The follow-up 128-thread vector/swizzle0 cases are terminal and pass all
same-draw/current-buffer/FP32/RNG/memory gates:

| Producer threads / blocks | MXFP8 us | FP4 us | Paired ratio | All order windows stationary |
|---|---:|---:|---:|---|
|128 / 128|279.725|297.879|0.93942|yes|
|128 / 256|269.076|281.479|0.95559|yes|

Conclusion: do not promote shared-dY ownership or enlarge this geometry
sweep. Reducing source reads and vectorizing row SMEM loads is insufficient
to beat the retained independent producers. Column loads remain scalar and
both orientations share a completion barrier; any future attempt must change
one of those structural costs, not repeat the tested launch geometries.
This is not evidence that all possible shared-input kernels must lose.
No training rerun is warranted for these losing candidates. The broad 1.20x
goal remains unachieved and stable defaults are unchanged.

### Inline stochastic scale finalization inside GEMM (2026-10-27)

Implemented a research-only `ms_eden_inline_scales` GEMM and comparison wrapper.
The existing partial-maximum reduction remains, but each GEMM converts corrected
BF16 temporary scales to E4M3 as it stages scales to SMEM. It uses the original
explicit uniforms by logical index, original FP32 division/probability and the
same separately rounded FP32 outer-factor product. There is no packed-forward
reuse, approximate normalization, changed RNG policy or public dispatch change.

The ordinary GEMM scale-loader hook is an identity; the experimental subclass
overrides it. Launch plans now expose their output explicitly, avoiding closure
introspection. Tensor views and compilation precede the kernel-only launch chain.
The benchmark forces row-major scalar scale loading and selects producer or
consumer staging. The first records' `schedules` describe the input schedule;
the effective overrides are documented here and in the hashed launcher source.
Future records also include those overrides in experiment metadata.

Qualification: **8 GPU cases passed in 10.42 seconds**, covering two shapes,
producer/consumer staging, divide/power2 probability, changing inputs/uniforms,
outliers, tiny and zero scales, graph replay and stable live allocation. Outputs
are bit-identical to independently materialized scales plus the ordinary GEMM.
An independent E2M1 decode and FP64 matrix product also passes the 0.5% RMS gate.
The first run had a test configuration spelling error (`consumer` versus the
existing `consumers` enum); two producer cases passed before that rejection.
No numerical threshold was relaxed. Local CPU regression checks: nine launch
tests and sixteen split-K contract tests passed; nine GPU-gated cases skipped.

Full compiled 4096x1536x512 dX+dW including preparation/RNG, 24 AB/BA rounds:

| Inline scale role | MXFP8 us | FP4 us | Paired MXFP8/FP4 | All order windows stationary |
|---|---:|---:|---:|---|
|consumers|251.782|559.215|0.45090|yes|
|producer|242.607|3824.714|0.06347|yes|

Both full-path experiments pass same-draw Gaussian/outlier/zero dX/dW equality,
FP32 gradient gates, fullgraph changed-input replay and zero live-memory growth.
Both are rejected for performance. Source geometry implies A-scale conversion
is repeated four times in both GEMMs; B-scale conversion repeats 32 times for
dX and 12 times for dW. This replaces a once-per-operand pass with duplicated
work and larger per-tile scale payloads. The producer variant additionally puts
that work on one warp. These are source-derived explanations, not hardware
counter measurements or a quantified attribution of the entire slowdown.

Generated dW artifacts report 70 registers, zero stack/local allocation and no
LDL/STL instructions for both ordinary and inline variants under either staging
role. Thus spills do not explain the observed loss. No training rerun or policy
promotion is warranted. Retain materialized E4M3 scales and focus future work
on reducing once-per-operand preparation cost rather than repeating it in GEMM.
Evidence: `autotune_reports/ms_eden_speed20_20261026_5060/inline-scales/`.
Remote: `/home/achiya/rtx-mseden-inline-scales-20261027.hCQ5V8`; all runs terminal.

### Bounded direct reciprocal sharing (2026-10-27)

The retained VPL8/16 producer repeats division by one five-significant-bit
scale. Prior shared-reciprocal variants normalized each value by a power of
two before refinement. The new research subclass removes that rescaling only
for scale exponents [-30,30] and quotient magnitudes [2^-32,16]; outside the
guard, explicit div.rn remains. Signed zero is preserved. A correctly rounded
reciprocal is shared; no approximate reciprocal or correction reassociation
is introduced. Exact power-of-two equivalence to normalized refinement holds
on this bounded domain without nonzero intermediate underflow/overflow.

Qualification: 17 GPU tests passed in 10.07 seconds. The explicit-div.rn probe
covers every denominator significand, scale-exponent boundaries, signs,
subnormals and exceptional fallback inputs. Sixteen complete producer cases
cover row/transpose, VPL8/16 and Gaussian/outlier/zero/wide-exponent data,
checking byte-exact packed values, corrected BF16 scales, partial maxima,
changed graph inputs and stable allocation. A separate CPU binary64 residual
model passed 1,179,648 quotient comparisons across guard-endpoint exponents,
both signs and representative significands; the GPU division probe remains
the authoritative exact-arithmetic check.

Complete compiled backward 4096x1536x512, same preparation/RNG policy:

| Direct refinement scope | MXFP8 us | FP4 us | Paired ratio | All order windows stationary |
|---|---:|---:|---:|---|
|all operands|269.491|283.038|0.94885|no|
|row dY only|272.931|277.441|0.98165|no|

Both pass full-path gradient/replay/allocation gates but fail timing acceptance.
These are provisional losing observations, not precise confirmed regressions.
Generated code reports row registers 57→56 and transpose registers 96→78,
with no local/stack storage or LDL/STL in either version. Static RCP counts
increase because fallback division remains in device code (row18→19,
transpose35→37); those counts are not dynamic execution counts. Reduced
registers did not establish a complete-backward benefit. No promotion and no
training rerun for these candidates. Further reciprocal-only variants are
deprioritized absent a new measured mechanism.

Evidence: `autotune_reports/ms_eden_speed20_20261026_5060/direct-refine/`.
Remote `/home/achiya/rtx-mseden-direct-refine-20261027.S6EGtp`; runs terminal.

### Joint four-producer grid interaction study (2026-10-27, running)

Previous global-grid and single-competing-grid sweeps do not exhaust coupled
resource allocation across dX-dY, dX-W, dW-dY and dW-X. A new bounded study
uses the existing `rtx.autotune` adapter, first-hit orchestration and residual
store. Its candidate pool is the retained control plus an OA(16,4,4,2) over
GF(4), with producer-block levels64/128/256/512. Every pair of grid coordinates
contains every ordered level pair once. Only per-operand producer blocks vary;
FP32 arithmetic, RNG/source policy, finalizer geometry and GEMMs are fixed.
Two CPU tests passed, proving the pairwise design and coordinate isolation.

The evaluator uses fresh subprocesses, scoped combo-kernel compilation,
same-draw full-shape checks and the public compiled complete-backward comparator
against stable MXFP8. Screening uses8 AB/BA rounds,5s warmup and16 replays;
independent24-round confirmation with all order-window gates is required before
any speed claim. Initial selection retains rowVPL16 and grouped32-bit stores.
The pool is fixed (no local mutations), max17 trials and1800 seconds. The
existing timing-quality/gradient/RNG/memory converter scores candidates; no
runtime winners are installed.

Remote `/home/achiya/rtx-mseden-joint-grids-20261027.I9FptI`, `study-v2.log`
and `study-v2/`. Parent135063 and first worker135101 were verified live.
The initial launch failed before GPU work because the input already contained
`backward_pipeline`; the corrected dictionary override preserves that log.
The first trial is retained control2297b213391451a89945420f. Results pending;
do not restart solely due to a disconnected observer or stale log.

The v2 control measurement completed, but ingestion then failed: the worker
had already produced a validated TrialOutcome and the adapter factory attempted
to convert it again as a raw result dictionary. Fixed by assigning the validated
evaluator directly, matching the existing pipeline tuner. A real first-hit CPU
integration test (synthetic worker evidence, real adapter/orchestrator/residual
store) now verifies two successful persisted outcomes; all3 joint-study tests
pass. No numerical/timing gate was bypassed.

Relaunched detached with nohup/disown into `study-v3/`, log `study-v3.log`.
Parent135394 and active worker135780 were verified live after the first two
successful residuals. Control273.719us; first challenger285.970us. These are
screening latencies only, not a final winner. Prior v2 evidence/logs are retained.

The corrected study completed17 trials in316.06 seconds; all workers passed
their numerical/RNG/replay/memory gates. Fourteen results have stationary
AB and BA windows for both baseline and candidate. The highest paired ratio
is1.01214 at grids(dX-dY,dX-W,dW-dY,dW-X)=(128,64,128,128), candidate271.688us
versus MXFP8274.538us. Lowest candidate median269.820us occurs at
(256,128,512,64), paired ratio1.00972. Retained control is273.719us with
paired ratio1.00302. These are screening observations across separate paired
sessions, not direct confirmed candidate/control wins. No 1.20x target pass.

Balanced-screen descriptive means suggest dX-dY64 is unfavorable (mean ratio
0.9320 versus0.9901–0.9916 at128/256); dW-X64 has the highest mean for that
coordinate0.9872. These are single-shape, single-replicate observations including
unstable records, not transferable heuristics or causal attribution.
All trial records are local under `joint-grids/study-v3/`.

After all tuning workers exited, Nsight environment preflight passed on driver
617.14 and the5060. A fresh single retained transpose-producer counter capture
still failed with ERR_NVGPUCTRPERM. The failure and environment JSON are retained
under `joint-grids/`. No hardware counters were obtained, no driver setting was
changed, and no profiler ran concurrently with timing. Requested host-side
counter access from the user; that is an outstanding profiling dependency, not
proof that every optimization avenue is blocked.

Three fresh24-round/15s-warmup confirmations (control, best paired ratio,
lowest latency) are now sequentially running in the same snapshot under
`confirmations/`, launcher `confirm.py`, log `confirmations.log`. They preserve
the screened numerical policies and require the normal full-path gates.

All three confirmations are now terminal and preserved locally in
`joint-grids/confirmations/`. Full-path same-draw/numerical/RNG/replay/memory
gates pass, with zero replay live-memory growth:

| Schedule | MXFP8 us | FP4 us | Paired ratio [95% interval] | All order windows stationary |
|---|---:|---:|---|---|
|retained control|273.013|272.716|0.99297 [0.98371,1.00092]|no|
|best screened ratio,128/64/128/128|269.565|266.547|1.01118 [1.00642,1.01586]|yes|
|best screened latency,256/128/512/64|292.361|291.584|1.00415 [1.00163,1.00663]|yes|

The best screen ratio repeats a small shape-specific advantage versus MXFP8,
not a confirmed direct advantage versus the retained FP4 schedule. It does
not meet1.20x and does not justify runtime promotion or another training run.
All GPU workers have exited; counter access remains the outstanding next
diagnostic request, with the current permission denial saved locally.

The research outcome converter previously checked aggregate stationarity only.
It now also honors any AB/BA quality records supplied; a strict opt-in rejects
missing order records. The joint study requires that strict mode and uses a
new measurement-contract identity for future runs. Historical records are not
rewritten; the manual all-order checks above already used the stricter gate.
All26 focused adapter/tuner/joint-study CPU tests passed. Stable kernel defaults
and numerical thresholds are unchanged. The broad speed/training goal remains
unachieved; no claim that further optimization is impossible is made.
