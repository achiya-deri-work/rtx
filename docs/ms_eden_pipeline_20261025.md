# MS-EDEN complete-backward pipeline experiments

The best stationary long-workload result is paired execution plus two-stage
asynchronous preparation: 1.143× baseline/candidate, about 12.5% less backward
time. Pure pairing gives 1.123× and serial two-stage buffering gives 1.042×
on that workload. These are bounded single-device/session results, not a
global dispatch rule. Tiny/decoder/wide pure-pairing confirmations have at
least one nonstationary arm. Staged amax alone regresses tiny; joint amax
showed no accepted benefit and was removed. Defaults remain unchanged.
The retrieved final evidence also includes 38 passing GPU tests, a completed
two-trial tuner smoke, and six matched short training runs. None establishes
long-run convergence or a default promotion. The subsequently retrieved
Nsight trace confirms actual caller/side-stream kernel overlap.

## Scope and fixed policy

RTX 5060 Laptop, 26 SMs, 8 GiB; Torch 2.13.0+cu132, CUDA 13.2.
Only the parent-coordinated `personal_laptop` GPU was used for these artifacts.
Local training GPUs and the 5070 were not used by this benchmark work.

Baseline: original BF16 operands, Hadamard-128, E8M4 RTZ and EDEN correction,
stochastic scale rounding, caller-owned explicit RNG, register-cached signs,
128 producer threads/256 CTAs, coalesced-wide loads, inline amax, and
128-thread/224-CTA finalization. dX uses row-major traversal and dW
reduction-major traversal. In the staged-amax substudy alone, the candidate
changes only inline to staged amax; later sections identify their independent
pairing and buffering changes explicitly.
Forward remains the exact JIT-row-region NVFP4 forward.

The comparator reuses the existing public fullgraph forward/autograd builder.
Both configurations share current input/weight/upstream buffers and are built
before timing. Ten randomized AB/BA round clusters use common calibrated
repetition counts and a target event window of at least 2 ms. Compilation,
capture, validation, warmup, and allocation audits are outside timing. These
are CUDA-graph backward latencies, not optimizer/model training throughput.

## Staged amax: complete backward

Microseconds per complete backward, dX and dW including public RNG. The ratio
is the paired geometric baseline/candidate ratio, not the ratio of displayed
medians; a value above one favors staged amax.

| M × N × K | Inline µs | Staged µs | Paired ratio [95% interval] | Timing eligibility |
|---|---:|---:|---|---|
| 512 × 512 × 512 | 60.54 | 62.88 | 0.9657 [0.9610, 0.9711] | Both stationary; regression |
| 4096 × 2048 × 512 | 428.95 | 432.49 | 0.9899 [0.9839, 0.9953] | Both nonstationary; diagnostic only |
| 32768 × 512 × 512 | 1168.78 | 1170.71 | 0.9955 [0.9930, 0.9979] | Both nonstationary; diagnostic only |

Intervals bootstrap complete rounds, not individual graph replays. They
describe this session and do not rescue nonstationary measurements. Tiny
forward+backward also has a nonstationary baseline; no qualified total-step
speedup claim is made. No staged configuration is promoted.

## Validity and memory

The separate identical-draw preparation matrix passed all 12 cases: 128×128
and 4096×512 sources, row and transposed readers, zero/sparse/outlier data,
and 128 versus 1024 partial maxima. Packed FP4 bytes, native scales, and outer
factors matched byte-for-byte under identical signs and uniforms. The
candidate's captured graph read changed source/sign/uniform buffers. This
focused matrix uses row-major producer traversal; it is not an independent
full dW traversal or paired-pipeline qualification.

Public bitwise forward equivalence, fresh RNG for both gradients, changed
current input/weight/upstream checks, finite gradients, and independent FP32
gradient sanity gates passed for all three complete-backward workloads.
Candidate backward NRMS spans 13.70–13.76%, below the fixed 25% Gaussian
algorithm-error gate. This broad gate does not establish convergence or replace
the identical-draw kernel checks.

Ordinary-call and graph-replay live-allocation growth was zero. Combined
paired replay peaks were approximately 40.6, 148.8, and 391.2 MiB for tiny,
decoder, and long respectively. These peaks include retained comparator graph
pools, not a single model training step. No new hardware-counter evidence was
collected; the earlier counter-permission limitation remains separate.

Raw artifacts:
[staged exact](../autotune_reports/ms_eden_pipeline_20261025_5060/staged/staged-exact.json),
[tiny](../autotune_reports/ms_eden_pipeline_20261025_5060/staged/staged-tiny.json),
[decoder](../autotune_reports/ms_eden_pipeline_20261025_5060/staged/staged-decoder2048.json),
[long](../autotune_reports/ms_eden_pipeline_20261025_5060/staged/staged-long.json).
Every result retains commands, source hashes, schedules, numerical checks,
raw timing clusters, stationarity flags, and memory evidence.

## Paired inter-leg execution and asynchronous producer probes

These later probes use 3000 ms preconditioning, ten randomized AB/BA rounds,
and the same current-buffer, explicit-RNG public fullgraph measurement.
Earlier staged-only runs used 200 ms; comparisons across their absolute
latencies are not controlled speedup measurements.

The paired candidate uses **paired execution plus staged amax**, against the
serial inline-amax baseline. The old comparator CLI implicitly selected
staged amax when `--candidate-json` was omitted; this is now corrected to an
identity default and guarded by a CPU regression test. Do not read the table
as isolated pairing speedup or infer a standalone staged benefit by subtracting
different sessions.

| M × N × K | Serial inline µs | Paired staged µs | Paired ratio [95% interval] | Timing eligibility |
|---|---:|---:|---|---|
| 512 × 512 × 512 | 59.77 | 47.01 | 1.2711 [1.2441, 1.2984] | Both stationary; within-session win |
| 4096 × 2048 × 512 | 460.59 | 401.49 | 1.1527 [1.1490, 1.1566] | Baseline nonstationary; no promotion |
| 32768 × 512 × 512 | 1182.87 | 1051.32 | 1.1279 [1.1257, 1.1301] | Both stationary; within-session win |

The corresponding paired reduction in backward time is approximately 21.3%
on tiny and 11.3% on long. These remain single-device/session results, not
cross-device dispatch rules or model/optimizer throughput claims.

The producer-only probes keep serial execution and inline amax, enabling
asynchronous copies with one, two, or three staging slots. All are decoder
4096×2048×512 complete-backward comparisons, not isolated producer timings.

| Async slots | Baseline µs | Candidate µs | Paired ratio [95% interval] | Timing eligibility |
|---|---:|---:|---|---|
| 1 | 452.67 | 456.69 | 0.9938 [0.9899, 0.9978] | Both stationary; slight regression |
| 2 | 447.26 | 448.58 | 0.9950 [0.9881, 1.0001] | Both stationary; no demonstrated win |
| 3 | 452.00 | 455.22 | 0.9907 [0.9881, 0.9935] | Both nonstationary; diagnostic only |

Long 32768×512×512 serial-buffering results are different from the decoder
regime. All three long comparisons pass stationarity for both arms:

| Async slots | Baseline µs | Candidate µs | Paired ratio [95% interval] |
|---|---:|---:|---|
| 1 | 1171.01 | 1167.71 | 1.0028 [1.0009, 1.0046] |
| 2 | 1174.24 | 1126.50 | 1.0419 [1.0388, 1.0446] |
| 3 | 1170.05 | 1128.28 | 1.0364 [1.0346, 1.0381] |

The long artifacts are retained in
[probes/results](../autotune_reports/ms_eden_pipeline_20261025_5060/probes/results/)
as `buffer{1,2,3}-long.json`. Their gains support shape-conditioned tuning,
not enabling buffering indiscriminately on decoder shapes.

All six public probes pass unchanged-forward/current-buffer/fresh-RNG,
finite-gradient and FP32 sanity checks. Candidate backward NRMS remains
13.67–13.74%. Ordinary-call and paired replay live-growth checks report zero.
Raw JSONs are under
[probes](../autotune_reports/ms_eden_pipeline_20261025_5060/probes/):
`paired-{tiny,decoder2048,long}.json` and
`buffer{1,2,3}-decoder2048.json`. Independent paired concurrency/checkpoint
and buffering kernel validation results are not inferred from these public
stochastic checks; those separate artifacts must be included before broader
qualification.

## Explicit inline-amax confirmation and rejected joint reduction

The following confirmation candidates explicitly retain inline amax, removing
the earlier paired+staged attribution ambiguity. Ratios compare each candidate
against its own same-session serial/synchronous baseline. Do not compare raw
medians across rows as if they were one matched experiment.

| Candidate | Shape | Baseline µs | Candidate µs | Paired ratio [95% interval] | Timing eligibility |
|---|---|---:|---:|---|---|
| Pure paired | tiny | 59.73 | 45.69 | 1.2719 [1.2218, 1.3070] | Candidate nonstationary |
| Pure paired | decoder2048 | 463.23 | 401.38 | 1.1479 [1.1401, 1.1546] | Baseline nonstationary |
| Pure paired | long | 1210.20 | 1078.80 | 1.1233 [1.1201, 1.1270] | Both stationary |
| Pure paired | wide | 995.41 | 862.72 | 1.1466 [1.1422, 1.1520] | Candidate nonstationary |
| Paired + async-2 | tiny | 60.25 | 45.84 | 1.3352 [1.2880, 1.3918] | Both nonstationary |
| Paired + async-2 | long | 1192.90 | 1043.13 | 1.1428 [1.1404, 1.1450] | Both stationary |

All six confirmation artifacts report zero paired replay live growth. Their
current-buffer/fresh-RNG/forward/FP32 checks remain separate from timing
eligibility. In particular, a narrow bootstrap interval does not override
failed stationarity. No default promotion follows from the long wins.

Raw artifacts:
[confirmation](../autotune_reports/ms_eden_pipeline_20261025_5060/confirmation/),
`paired-inline-{tiny,decoder2048,long,wide}.json` and
`paired-buffer2-{tiny,long}.json`.

The experimental joint two-maxima/outer-product reduction was evaluated and
archived rather than retained as an API option. Serial joint amax gives a
stationary 0.9852× on long (regression) and 0.9968× on decoder (no demonstrated
win); tiny fails candidate stationarity. Paired+joint gives 1.1139× on long,
but does not establish an improvement over the separate pure-paired 1.1233×
or paired+async-2 1.1428× results. These are not direct pairwise comparisons
between those candidates. Paired+joint tiny also fails candidate stationarity.
No accepted joint-specific benefit was established; its new kernel/API/CLI
option was removed while evidence remains in
[joint](../autotune_reports/ms_eden_pipeline_20261025_5060/joint/) and
`confirmation/paired-joint-{tiny,long}.json`.

## Autotuner integration and remaining work

`rtx.autotune.ms_eden_pipeline` composes the existing per-leg posthoc adapters
with an immutable complete-backward serial/paired schedule. It delegates
component axes, updates, rejection, identities and resource features; it does
not introduce another search engine. Scoring requires the requested workload,
pipeline and schedules to match the result, successful public guards, both
gradients, no live growth, at least four complete-backward rounds and explicit
timing-quality metadata. Nonstationary trials remain feasibility evidence,
not timing-eligible winners.

`benchmarks.tune_ms_eden_pipeline` uses the existing first-hit selector and
append-only JSONL residual store. Trials run sequentially in fresh subprocesses;
initial coverage prioritizes baseline, paired, staged and coupled two/three-
stage async producers before round-robin full-adapter coordinate proposals.
Selections are research artifacts only; runtime defaults are never installed.
The tuner now defaults to 3000 ms warmup; `--warmup-ms` is recorded and remains
configurable. More warmup does not bypass stationarity gates.

The first two-trial frontend smoke stalled before any GPU worker or residual
observation: the CLI redundantly acquired the same-context `flock` already
owned by `AutotuneOrchestrator.tune`. The outer lock was removed. A new CPU-
only subprocess integration test runs the real first-hit/orchestrator/JSONL
store stack with a hard timeout and synthetic guarded evaluator results;
it completes serial and paired observations plus a selection artifact. This
proves the locking/persistence path, not GPU kernel behavior.

The retrieved [final GPU smoke log](../autotune_reports/ms_eden_pipeline_20261025_5060/final/tuner-smoke.log)
now closes that rerun requirement: both trials completed, saving serial
59.800 µs and pure-paired inline-amax 45.734 µs observations and emitting a
paired selection artifact after about 24 s. This is frontend/persistence
smoke evidence, not a new stationarity-qualified speedup study or evidence
that a later identical invocation hits the persistent cache. The final
[qualification log](../autotune_reports/ms_eden_pipeline_20261025_5060/final/qualification.log)
records 38 passed GPU tests (14 deprecation warnings) in 18.15 s. Separately,
the parent-verified historical CPU result was 88 passed, 21 skipped; it is
not a fresh CPU rerun in this final artifact set.

## Matched short model-training evidence

The six [final training JSONLs](../autotune_reports/ms_eden_pipeline_20261025_5060/final/training/)
each complete 512 optimizer steps and 2,097,152 tokens without interruption,
with finite final parameters and finite loss/gradient norm on all 512 logged
steps. They share seed 941, initial-weight/corpus/validation-offset hashes,
model and optimizer configurations, precision settings, and recorded source
hashes. These are repeated same-seed runs, not independent-seed convergence
replicates. Identical initialization does not make stochastic training
trajectories bitwise identical.

The model has 21,121,024 parameters: eight decoder layers, width 512,
intermediate width 1024, eight heads, vocabulary 257, and batch 16 × sequence
256, with FLASH attention. AdamW uses learning rate 3e-4, betas (0.9, 0.95),
32 warmup steps and weight decay 0.1. Matrix/embedding parameters are BF16;
normalization parameters and norm/gradient-norm reductions are FP32.
All runs start with validation loss 5.57462. Identity hash prefixes are
`c5968e2ef598` (initial weights), `c041423d36ce` (corpus), and `8e08204efdb2`
(validation offsets); full hashes and per-projection schedules remain in
each JSONL environment record.

`baseline{1,2}` are serial `nvfp4_full` MS-EDEN; `candidate{1,2}` explicitly
select paired MS-EDEN. Both use 256 producer CTAs, register signs, inline
amax, one stage and synchronous preparation: this training comparison does
**not** test the long-workload paired+async-2 winner. `hybrid{1,2}` record the
`nvfp4_mxfp8` mode (NVFP4 forward/MXFP8 backward comparator), not another
MS-EDEN schedule. Forward projection configurations match across all six.

| JSONL | Backward arm | Training wall s | End-to-end wall s | Final validation loss |
|---|---|---:|---:|---:|
| baseline1 | Serial MS-EDEN | 11.4184 | 38.8983 | 0.97543 |
| baseline2 | Serial MS-EDEN | 11.2421 | 32.2071 | 0.99565 |
| candidate1 | Paired MS-EDEN | 10.4303 | 38.6093 | 0.99896 |
| candidate2 | Paired MS-EDEN | 10.4356 | 31.3966 | 0.99037 |
| hybrid1 | MXFP8 | 9.6023 | 24.5695 | 0.98696 |
| hybrid2 | MXFP8 | 9.6403 | 24.5854 | 0.99224 |

Training wall is the recorded synchronized-block sum including data movement
and optimizer work, excluding compilation, validation, checkpointing, logging
and outside-block checks; end-to-end wall includes those costs. The ratio
of two-run mean training walls is 1.086× serial/paired (7.9% less training
wall), a descriptive short-run result without randomized timing rounds or
stationarity qualification. Paired MS-EDEN remains slower than the MXFP8
comparator here (mean 10.433 versus 9.621 s). End-to-end differences also
reflect compilation and are not steady-state kernel speedups.

Logged gradient norms span 0.919–19.780 across these runs, with no nonfinite
logged losses/norms. Final validation losses overlap at this short horizon;
this supports bounded training health, not statistical equivalence, long-run
stability or convergence qualification. No BF16 training arm is present in
this six-run set.

## Opt-in public controls

The research public API exposes serial/paired inter-leg execution and
independent immutable per-gradient preparation schedules. Defaults remain
serial, synchronous and inline-amax; stable `NVFP4Linear` backward policy is
not replaced. For example, explicit opt-in paired two-stage preparation is:

```python
import json
from rtx.experimental_ms_eden import ms_eden_linear

schedule = {"producer_load": "coalesced_wide", "producer_signs": "register",
            "producer_threads": 128, "producer_blocks": 256,
            "producer_stages": 2, "producer_async_copy": True,
            "finalizer_threads": 128, "finalizer_blocks": 224,
            "amax_reduction": "inline", "rng_mode": "explicit"}
y = ms_eden_linear(x, weight, source_policy="original_bf16",
    backward_pipeline="paired",
    dx_schedule_json=json.dumps(dict(schedule, producer_order="row_major")),
    dw_schedule_json=json.dumps(dict(schedule, producer_order="reduction_major")))
```

Use `schedule_json` for shared overrides or `dx_schedule_json`/`dw_schedule_json`
for independent leg schedules as above. Forward configuration is separate;
this example illustrates backward controls rather than overriding it. Retain
exact source/device/toolchain fingerprints
when applying research selections; do not silently promote one shape's winner.

```bash
python -m benchmarks.compare_ms_eden_pipeline --validate-exact \
  --candidate-json '{"amax_reduction":"staged"}' --output NEW/exact.json
python -m benchmarks.compare_ms_eden_pipeline --shape tiny --warmup-ms 3000 \
  --candidate-json '{"amax_reduction":"staged"}' --output NEW/staged.json
python -m benchmarks.compare_ms_eden_pipeline --shape tiny --warmup-ms 3000 \
  --candidate-pipeline paired --candidate-json '{"amax_reduction":"staged"}' --output NEW/paired-staged.json
python -m benchmarks.tune_ms_eden_pipeline --shape tiny --output NEW/tuning \
  --trials 2 --rounds 4 --pool-size 8
CUDA_VISIBLE_DEVICES='' python -m benchmarks.tune_ms_eden_pipeline \
  --shape tiny --output unused --trials 2 --dry-run
```

Run GPU commands only on an idle explicitly assigned GPU. The two-trial tuner
smoke and short training checks above are complete. Remaining decoder shapes,
stationarity-clean tiny/decoder/wide pure-pairing confirmation, long-run and
broader hardware/training qualification remain follow-up work beyond this
bounded experimental study. The persistent-cache and timeline checks are
complete, as detailed below. No new counter-based
bottleneck attribution or hardware-wide dispatch recommendation follows.
The full Cartesian schedule space is deliberately not benchmarked.

### Follow-up cache regression (October 26)

The real first-hit/JSONL integration regression now executes two fresh CPU-only
processes against the same store. The first records two synthetic guarded
observations; the second forbids evaluator calls and must return the identical
configuration and timing without adding observations. All six tuner tests pass.
This proves persistent cache reuse in the orchestration path, independently of
GPU availability. New 5090 correctness evidence and the current
performance-access gate are recorded in the
[5090 qualification note](ms_eden_pipeline_20261026_5090.md).

### Recovered 5060 cache and timeline evidence (October 26)

The final remote snapshot matches the current local SHA256 fingerprints for
`experimental_ms_eden.py`, `nvfp4_posthoc.py`, and `autotune/posthoc.py`.
`final/tuner-cache-hit.log` records `FIRST_HIT cache` for the original paired
winner at 45.734 us. Its second selection matches the first, and the remote
residual store still contains exactly two observations: no rebenchmarking.

`final/paired-timeline.{nsys-rep,sqlite,json}` and the CSV/log capture ten warmed
public backward graph replays at 4096×2048×512, pure paired execution with
inline amax and synchronous producers. The trace contains 210 kernels across
three streams. Caller stream 142 and side stream 143 overlap for a total
**1785.360 us** within that capture. Summed kernel duration is 5380.585 us;
the union of active kernel intervals is 3595.225 us. This confirms real
concurrent execution, not merely multiple stream identifiers. These profiled
durations are diagnostic and are not a new speedup measurement.

The capture's public guards pass: forward bitwise match, changed input,
weight and upstream buffers, fresh RNG per gradient, and both gradient
reference gates. Ordinary compiled calls and graph replay both report zero
live-allocation growth. No hardware-counter saturation or spill claim is made;
Nsight Compute access remained permission-limited. Original BF16 operands,
FP32 scaling/correction, and stable MXFP8-backward defaults remain unchanged.
