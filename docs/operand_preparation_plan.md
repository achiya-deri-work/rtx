# Operand preparation execution-plan selection

Latest NVFP4-only results and API boundary:
[NVFP4 forward and experimental backward](nvfp4_completion_20261023.md).
The material below retains earlier experiments and their historical caveats.

Status correction: the completed execution-plan matrix below is **MXFP8
forward/backward**, not completion of the requested NVFP4 investigation.
The active task is NVFP4 forward plus experimental NVFP4 backward; MXFP8
backward remains its default/reference. The earlier closeout was premature
for that task. Default runtime selection is unchanged.
An experimental public-frontend hook is available behind
`AutotunePolicy(preparation_selection=True)` or the legacy
`RTX_PREPARATION_SELECTION=1` switch (details below).

## Completion audit (2026-10-23)

### NVFP4 continuation: matched compiled training-path baseline

`compare_nvfp4_backward_policies` uses the selected NVFP4 JIT-region forward
config for both backends and asserts bit-exact forward equality. Original BF16
X/W and identical upstream gradients are used in both backward policies.
Fullgraph compiled forward plus autograd is captured together; MS-EDEN RNG
generation is included. No packed-source reuse is used. Explicit resource
seed 1 for dX and seed 2 for dW are preliminary proposals, not exhaustively
tuned backward schedules. Fresh per-shape processes produced:

| M,N,K | NVFP4 fwd + MXFP8 bwd (us) | NVFP4 fwd + MS-EDEN bwd (us) |
|---|---:|---:|
| 2048,768,1536 | 46.36 | 138.84 |
| 8192,768,1536 | 149.97 | 254.29 |
| 32768,768,1536 | 678.12 | 824.98 |
| 8192,3072,768 | 292.52 | 429.02 |

All eight timing windows passed stationarity gates. Forward NRMS was
13.44-13.45% for both backends, MXFP8 gradient NRMS 3.76-3.77%, and MS-EDEN
gradient NRMS 13.70-13.76%. These are one-seed Gaussian error measurements,
not convergence or unbiasedness evidence. Smallest shape ran on local GPU 0;
other shapes on GPU 1, with both policies paired on the same GPU per shape.
Artifacts: `nvfp4_completion_20261023/matched_backward*.json`.

The experimental API now supports original-source current/JIT-region/block
forward with explicit forward configs and independent dX/dW schedule JSON.
Packed-source reuse still requires fixed tensor-current forward; unsupported
combinations fail explicitly. Stateful delayed is not exposed by this stateless
function. All 148 post-hoc kernel, experimental integration, and schedule tests
passed after the change. This is not completion of backward performance work.

Fresh GPU-0 forward-only finalist confirmation measured JIT/delayed at
13.83/16.69, 37.86/41.06, 165.61/170.14, and 61.81/63.55 us in the table's
shape order. Thus the earlier cross-GPU delayed advantage did not reproduce
in this same-device pass; do not use the cross-GPU table to rank policies.
Reports include phase timings, raw samples, policy/config and device identity.

**Scope warning:** this audit closes only the MXFP8 execution-plan work and
bounded experimental implementation checks. It does not close the NVFP4
forward/matched-backward comparison. New NVFP4 runs are isolated under
`autotune_reports/nvfp4_completion_20261023/`.

The NVFP4 research tuner now supports full quantizer plus GEMM scheduling for
JIT-region and tensor-delayed, holding block selection and outer-scale mode
fixed. Reports separately record preparation, GEMM, and complete-forward
timings and their stability checks. Each candidate is checked against both
original BF16-input FP32 GEMM and an independent decode of its actual packed
FP4 values, E4M3 scales, and FP32 outer factors. The packed oracle must pass
0.5% NRMS; original-source error cannot exceed the fixed-policy initial error
by more than 0.1% relative. Delayed history is explicitly fixed at input amax
for these steady-state schedule experiments, not a changing-distribution
training qualification. The tested JIT geometry is explicitly 8x8 rows and
does not imply numerical equivalence with tensor scaling.

Initial NVFP4-only searches completed 168 successful, timing-eligible trials
across four shapes and both forward policies (12 trials for the smallest,
24 for each larger shape). Selected forward latency, in microseconds:

| M,N,K | JIT 8x8 exact | Tensor-delayed exact |
|---|---:|---:|
| 2048,768,1536 | 14.08 | 16.44 |
| 8192,768,1536 | 38.13 | 39.03 |
| 32768,768,1536 | 167.07 | 160.01 |
| 8192,3072,768 | 62.74 | 59.49 |

These are separately tuned forward policy results, not identical numerical
policies or end-to-end backward comparisons. JIT was measured on local 5090
GPU 0, delayed on GPU 1; the later same-device finalist confirmation above
supersedes this cross-device ranking. At the longest shape, JIT's selected original-source NRMS was
13.444% and packed-GEMM NRMS 0.166%; delayed's were 13.440% and 0.166%.
This single Gaussian distribution does not establish equivalent quality on
outliers or changing training distributions.

The generic GEMM axis list included per-candidate L2 fetch settings, which
production runners deliberately ignore unless a process-wide policy is set.
Those historical trials therefore did not test hardware L2 limit changes.
The NVFP4 research driver now removes that inactive axis. The raw old config
fields remain in artifacts; do not interpret them as a measured L2-policy win.

The implemented contract is policy-preserving first-hit selection, not a
promise of globally optimal schedules, a deployable ML predictor, or universal
GPU qualification. The following requirements have concrete implementations
and evidence:

| Requirement | Implementation and verification |
|---|---|
| Audit/expose preparation alternatives | `preparation_mxfp8.py` describes forward fused/materialized and backward staged/fused-dX/fused-dW/fused plans. Component axes remain searchable; active-leg equivalence and legality are tested. |
| Device/shape-aware prior with local confirmation | Compatible measured-neighbor retrieval orders by shape and known SM/L2/bus capacity; plan-diverse seeds and bounded expansion remain available. Transferred schedules are measured locally, never assigned another device's latency. Runtime journals now include stable measured capacities. |
| Preserve numerical policy | Composed adapters reject mismatched quantization signatures. Original BF16 operands and MXFP8 backward remain the runtime contract; explicit configs stay authoritative. |
| Revision-safe first-hit/cache integration | Component kernel, policy, workload/device/software and measurement identities partition exact caches. Tests cover hits without evaluation, revision isolation, nearby-shape remeasurement, cumulative refinement, and unstable-trial rejection. Separate-process laptop cache replay is recorded below. |
| Forward/dX/dW framework integration | Eleven local CUDA integration cases cover eager/fullgraph, paired/partial gradients, ragged fallback, changed-input graph replay, live allocations, and operand-device restoration. |
| Matched-policy GPU evidence | Four-shape 64-trial baseline plus 48 targeted dW trials, followed by 28 stable phase measurements; original-BF16 FP32 oracles and exact configs accompany the reports below. |
| Post-hoc alignment and fused backward preparation | `nvfp4_posthoc.py` implements extended temporary scales, scale-only finalization, fused rotation/quantization/correction/layout, and optional cooperative transport. Independent oracle, extreme-scale, layout, RNG, and packed-GEMM tests pass. |
| Explicit optional FP4 backward and packed-state reuse | `experimental_ms_eden.py` keeps both outside the default frontend; original-BF16 and forward-quantized source policies have distinct identities/oracles. Tests cover autograd, fake/compile, checkpoint RNG, saved-state ownership and partial gradients. |
| Vendor packed-GEMM diagnostic | `compare_nvfp4_packed_gemm.py` and recorded vendor/native experiments separate preparation from packed multiplication. Actual backend and independent decode references are recorded; not described as a paper reproduction. |

Final combined recheck: **102 CPU/reference tests passed**, **135 experimental
kernel/integration GPU tests passed**; the current runtime selector separately
passed **11 CUDA integration cases**. Test files are `test_preparation`,
`test_first_hit`, `test_preparation_phase_report`, `test_posthoc_autotune`,
`test_ms_eden_reference`, `test_preparation_transfer_study`,
`test_packed_gemm_diagnostic`, `test_nvfp4_posthoc_cuda`,
`test_experimental_ms_eden`, and `test_preparation_runtime_cuda` under `tests/`.

The measured-neighbor prior is the supported result. The analytical traffic
proxy and learned models did not earn deployment and remain rejected. Sparse
MS-EDEN estimator bias, the 5070 Laptop's unstable timing windows, bounded
shape coverage, non-exhaustive schedule tuning, and the lack of full-model
qualification for our new FP4 backward remain explicit limitations, not hidden
success claims. Neither a public default change nor repository publication is
part of this closeout. Historical entries below retain their experiment dates
and describe intermediate states rather than overriding this audit.

## Detailed evidence and historical progress

### Runtime resource-profile wiring audit

The runtime preparation selector now augments its legacy device identity with
positive CUDA-exposed static capacities: L2 bytes, memory bus width when
available, opt-in block/SM shared memory, SM registers, and SM thread/block
limits. It does not record transient clocks/power or invent missing values
from GPU names. This closes a wiring gap: measured-neighbor retrieval already
used L2/bus capacities as shape-distance tie-breakers, but the legacy runtime
fingerprint omitted them.

Retrieval now uses canonical `memory_bus_width_bits` and accepts historical
`memory_bus_width` records. Unknown capacities remain penalized rather than
treated as exact matches. Hardware facts affect proposals, never permit
cross-device latency reuse, and never override numerical-policy/toolchain
compatibility or local measurement. Richer identity intentionally creates a
new exact context; compatible old observations remain retrievable proposals.
No automatic migration of old winners into an unmeasured exact context occurs.

Validation: 47 preparation/first-hit CPU tests and all 11 runtime CUDA
integration cases passed on the local two-5090 machine. The multi-GPU test
also verifies that forward/dX/dW journals contain the operand GPU's measured
L2 capacity while restoring the caller's current device. Additional tests cover
legacy/canonical bus-width matching, unknown resources, and exclusion of
transient or invalid property values. This is functional qualification of a
measured-neighbor prior, not evidence that the rejected analytical cost proxy
or an ML model now generalizes.

### Targeted axis coverage

The research preparation driver now accepts repeatable `--axis-filter`
substrings to target late search coordinates while retaining baseline plans.
For example, `--refinement-plan staged --axis-filter logical_transport
--axis-filter logical_tile --axis-filter scale_store` explores transpose
transport, tile geometry, and native scale stores instead of spending a small
expansion budget on earlier GEMM coordinates. Unmatched filters raise an error.
The shared expansion API accepts an explicit coordinate list without mutating
the adapter; default behavior is unchanged. Two new unit tests cover late-axis
selection, baseline retention, non-mutation, and invalid coordinate/filter
errors (29 preparation tests passed).

This targets an identified search-coverage limitation, not a proven hardware
optimization. One-coordinate expansion still cannot establish whether coupled
load/geometry changes win, and a bounded prefix of a coordinate's values is
not exhaustive coverage. Keep the original numerical policy and legality gates.

Two 24-trial original-BF16/MXFP8 dW studies completed on the local 5090s:
`targeted_v2_32768_dw_transport` and `targeted_v2_wide_dw_transport` under the
report root. At `32768,768,1536`, the register-load baseline remained best at
348.812 us including preparation and dW GEMM. At `8192,3072,768`, changing
only operand B to `cp_async` selected 93.179 us versus the 93.375 us baseline
(0.21%, not a convincing replicated improvement). Packed scale-store and
tested smaller logical tiles did not establish a benefit. These results do
not justify changing production defaults or claiming transport is universally
irrelevant. Coupled geometry/transport and GEMM schedule exploration remain
separate follow-up experiments.

### Matched-policy final matrix measurement protocol

The `final_v2_*` local research searches use eight isolated trials per shape
and direction, original BF16 sources, FP32 quantization arithmetic/amax,
E4M3/E8M0 infinity rounding, and MXFP8 backward. They cover the four named
shapes below. Forward searches run on local GPU 0 and backward searches on
GPU 1; both are 5090s, but results retain device UUIDs and are not a paired
cross-device end-to-end measurement.

The legacy research drivers now use bounded time-based graph warmup and
attach `runtime_timing_metadata` before passing outcomes to the shared tuner.
Previously those research outcomes lacked the explicit stationarity gate;
this change does not alter the already-gated production runtime selector.
Measurement revision 2 is included in component context tags to prevent
reusing revision-1 research selections as equivalent measurements.

The phase profiler can consume completed searches via `--forward-trials`
and `--backward-trials`. It checks shape, completed-session winner identity,
timing eligibility, and staged implementation compatibility before loading
the selected configs. Fused winners are not falsely decomposed into staged
phase timings. Five unit cases cover valid, unfinished, wrong-shape,
unstable, and fused-winner inputs. Phase reports include per-phase timing
quality and source-search paths; the complete sequence remains authoritative.

All 64 search trials completed with numerical gates and timing eligibility
passing. Materialized forward and staged backward won each eight-trial
search. The phase replays below passed all 28 stationarity checks; all twelve
forward/dX/dW NRMS results were between 3.753% and 3.773%. Units are us.

| M,N,K | Fwd prep | Fwd GEMM | dX prep | dX GEMM | dW prep | dW GEMM | Full sequence |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048,768,1536 | 4.37 | 12.62 | 4.09 | 14.94 | 6.17 | 15.62 | 57.76 |
| 8192,768,1536 | 11.26 | 37.66 | 6.65 | 37.59 | 16.76 | 52.38 | 176.47 |
| 32768,768,1536 | 93.60 | 144.44 | 21.28 | 157.75 | 147.97 | 201.84 | 749.43 |
| 8192,3072,768 | 7.42 | 75.71 | 22.17 | 70.92 | 28.08 | 63.84 | 313.75 |

Phase reports are `autotune_reports/preparation_20261023/final_v2_phases_*.json`.
The smallest shape was replayed on GPU 0, the others on GPU 1. Each complete
sequence is measured on one device; isolated medians need not sum because
cache reuse changes. This is an eight-trial execution-plan baseline, not
exhaustive schedule tuning, a whole-model benchmark, or evidence that FP4
backward matches MXFP8 numerical behavior. Long-shape dW preparation remains
a substantial target (147.97 us versus 201.84 us isolated GEMM). More budget
must explore quantizer/load and GEMM schedules rather than merely repeat the
same initial portfolio before asserting a precision-dependent crossover.

### Cross-device cache replay and timing audit (2026-10-23)

The RTX 5070 Laptop completed the public compiled MXFP8 qualification at
`2048,768,1536`, with four trials per forward/backward context. A separate
process reused the exact persistent cache without adding observations; hot
execution also left the journal byte-identical. Against original-BF16-input
FP32 references, forward/dX/dW NRMS were respectively 0.037489, 0.037524,
and 0.037679. This is one-shape compiler/cache/numerical evidence, not a
cross-device performance qualification.

The end-to-end timing was nonstationary both with the original ten-replay
warmup and with bounded time-based warmup (40 replays, approximately 200 ms).
The latter recorded a 0.692235 ms median but 27.95% relative range and 11.27%
split drift, failing the unchanged 10% range / 3% drift gates. Do not promote
this median as a stable performance result or infer the root cause from it.
Only one of the four original backward candidates had eligible timing, so
the cached selection does not establish a well-explored backward winner.

`benchmarks.qualify_preparation_runtime` now explicitly records measurement
contract `graph8x10_samples15_warm200ms_revision2` and requests bounded
time-based warmup, matching the revised research phase benchmark. Old reports
remain unchanged. Local validation of the research warmup metadata passed
(one CUDA test), and the posthoc/first-hit/preparation CPU suite passed all
49 tests. Source snapshots and results on the laptops are isolated from the
user checkouts; no driver, clock, power, or system-service settings changed.

Remote evidence: snapshot `rtx-preparation-transfer-20261023.0irZbV`,
`results/mxfp8_runtime_compiled/result.json` and
`results/mxfp8_runtime_cache_replay_warmup_v2/result.json`.

The RTX 5060 Laptop revision-2 four-trial research repeat at the same shape
completed with all four timing-eligible candidates: 186.013, 156.821,
112.455, and 183.865 microseconds. The winner uses 256 producer threads,
128 producer CTAs, reduction-major cooperative loading, and 224 finalizer
CTAs. The 1.65x baseline-to-selected improvement applies only to experimental
MS-EDEN original-BF16-source dW preparation plus its fixed GEMM, not to public
MXFP8 backward or whole training. It reproduces the earlier revision-1
selection; the two measurement revisions remain separate datasets.

Completed laptop artifacts are mirrored locally under
`autotune_reports/preparation_20261023/transfer_5060/` and
`autotune_reports/preparation_20261023/transfer_5070/`.

The 5070 Laptop revision-2 research repeat finished with four numerically
valid candidates but only one timing-eligible result (178.294 microseconds,
128 producer threads, 96 producer CTAs, 288 finalizer CTAs, cooperative
reduction-major loading). The baseline 222.620 us, 256-thread candidate
150.409 us, and direct-load challenger 198.819 us all failed timing gates.
Consequently this run supplies correctness and rejection evidence, not a
reliable comparative schedule ranking. Increasing warmup did not resolve the
5070 timing problem; do not train a performance prior on its rejected samples.

The original objective is not replaced by the experimental FP4 kernel work.
Current implementation boundaries and remaining acceptance evidence are:

- **MXFP8 runtime plans:** forward fused/materialized and backward staged,
  fused-dX, fused-dW and fully fused alternatives are exposed through the
  explicit preparation-selection opt-in. Numerical-policy signatures and
  legality filter the candidates. Explicit configurations remain authoritative.
- **Cold/hot integration:** local tests cover eager/fullgraph, paired/partial
  gradients, ragged fallback, current-device restoration, changed-input graphs,
  cache refinement/retrieval, and stable live allocations. The public hot path
  does not call the research MS-EDEN driver or change gradient precision.
- **Prior boundary:** runtime MXFP8 uses compatible measured-candidate retrieval
  and policy-matched seeds, not the unsuccessful analytical/ML prior. The new
  resource-coupled portfolio is currently an MS-EDEN research-tuner prior with
  local-5090 evidence. Cross-SKU validation is required before broader claims;
  generic learned-prior artifacts remain rejected by runtime selection.
- **Expanded Quartet work:** post-hoc alignment, fused preparation, explicit
  experimental FP4 backward, packed-state numerical policy, and packed-GEMM
  diagnostics have implementations and bounded numerical evidence below.
  They are not interchangeable with original-BF16/MXFP8 backward. Persistent
  sparse-distribution estimator error is a documented policy limitation.
- **Comparison delivered:** current selected schedules have a phase-resolved,
  matched-policy four-shape report above, alongside cross-device runtime and
  first-hit results. Historical, fixed-grid, profiler-replay and toy training
  results retain their labels rather than becoming release claims.

The execution-plan implementation and validation objective is complete; universal
linear-kernel qualification is not claimed. No public default is changed by
the research findings.

Multi-GPU cold selection explicitly enters the operand device for capture
checks, compilation, graph replay, and timing events, then restores the caller's
current device. A two-5090 regression keeps device 0 current while selecting
forward, dX, and dW plans for device-1 inputs, verifies all three numerical
evaluations succeed, and checks context restoration. This change is confined
to cold selection; hot launchers do not gain extra host arithmetic or tuning.

Nearby-context retrieval scans the journal and retains at most 4,096 recent
compatible successful observations. Previously the oldest 4,096 raw rows
exhausted the scan budget, even if they belonged to unrelated families, hiding
newly appended useful measurements. Regression coverage demonstrates that
failure and verifies the corrected behavior. Retention memory is bounded;
scan I/O still grows with journal length and an indexed store remains a future
scalability improvement. This changes which schedules are proposed, not their
numerical policy or the requirement to remeasure them on the target context.

### Latest research preparation tuning

#### Stochastic gradient diagnostic

On original BF16 Gaussian operands at M,N,K=512,768,256, 256 independent
scale-rounding samples with fixed Hadamard signs left mean-gradient NRMS of
12.647% (dX) and 12.632% (dW). Resampling the shared 128 Rademacher signs as
well reduced these to 0.8565% and 0.8550%; estimated sampling-noise NRMS of
the mean was 0.8542% and 0.8554%. Individual-gradient NRMS remained about
13.67--13.69%. The mean uses FP64 Welford accumulation; the independent
reference is original BF16 operands multiplied with FP32 accumulation.

This separates conditional scale-rounding noise from rotation-dependent
quantization error. Scale SR alone does not remove the observed fixed-rotation
error. The resampled result is consistent with noise averaging at this shape
and distribution, not a proof of zero bias or of convergence on real training
data. Repeated estimates of identical operands are not a larger minibatch.
Both operands share signs to preserve the underlying dot product, but receive
independent scale uniforms. Production RNG generation, checkpoint restoration,
cross-step correlation, and overhead still require qualification.

Reproduce for each `--phase dx` / `--phase dw`:

```bash
.venv/bin/python -m benchmarks.study_ms_eden_backward \
  --shape 512,768,256 --phase dx --stochastic-samples 256 --no-benchmark \
  --output /tmp/ms-eden-stochastic-dx.json
```

Raw results: `autotune_reports/preparation_20261023/ms_eden_stochastic_dx_512.json`
and `ms_eden_stochastic_dw_512.json`. These diagnostics are outside measured
kernel launchers and do not change any public backward policy.

The expanded distribution study gives a more restrictive conclusion. At the
same shape and 256 fresh rotations, mean dX/dW NRMS was 0.894%/1.187% for
variance-normalized Student-t(3), 1.800%/1.774% for 10%-dense Gaussian inputs,
and 1.062%/1.037% for Gaussian inputs with independent 1% sixteen-fold
outliers. Estimated mean-noise NRMS was 0.86--0.90%. Sparse dW extended to
2,048 samples retained **1.584% mean NRMS**, while estimated mean noise fell
to **0.305%**. Thus the Gaussian result cannot be generalized to an unbiased
gradient estimator: this sparse case has a persistent conditional error well
above the measured sampling noise. Its signed gain error was only -0.0099%,
so a scalar gain statistic alone would hide the vector error. Root-cause
attribution (finite Hadamard ensemble, EDEN correction, rounding, or their
interaction) remains open; this is not yet a diagnosis of an implementation bug.

A Gaussian dW reduction of length 8,192 retained the noise-averaging behavior:
0.854% mean NRMS versus 0.854% estimated noise at 256 samples. All studies use
one operand seed (807); distribution/seed flags now allow independent repeats.
Artifacts add `ms_eden_stochastic_{dx,dw}_{student3,sparse,outliers}_512.json`,
`ms_eden_stochastic_dw_sparse_2048samples.json`, and
`ms_eden_stochastic_dw_8192.json` under the same report root. Forty-two GPU
tests pass, including deterministic repeated studies for all four distributions.

The `--decoded-samples` diagnostic localizes this sparse error before GEMM:
every realization is decoded and multiplied independently in FP32, with a
0.005 NRMS gate on the actual packed GEMM. At 2,048 fresh-rotation samples,
seeds 807/808 gave kernel mean NRMS 1.58447%/1.59425% and decoded mean NRMS
1.58448%/1.59426%. Kernel-versus-decoded mean differences were only 0.00373%
for both seeds; worst individual packed-GEMM NRMS was below 0.168%.
The persistent error is therefore present in the prepared operands, not
explained by tensor-core accumulation/output rounding. This does not yet
distinguish a preparation implementation error from the finite rotation and
correction policy. Artifacts: `ms_eden_sparse_decoded_2048.json` and
`ms_eden_sparse_decoded_seed808_2048.json`. The diagnostic is research-only;
its decoded matmuls and statistics are never inside timed launchers.

An independent preparation comparison now narrows that attribution further.
`--oracle-samples` uses a dense FP64 Sylvester Hadamard matrix, the independent
Torch FP4 grid/EDEN oracle, and independently implemented scale brackets/SR
with the same supplied randomness. On sparse M,N,K=128,256,128 dW with 512
fresh rotations, both the decoded fused path and this oracle produced mean
NRMS **1.657589%**, with zero difference between their means. The kernel
output mean was 1.657668%, estimated mean noise 0.599385%. This tested case
reproduces the persistent error in the specified algorithm independently of
the fused implementation. It does not establish a universal bias bound or
justify changing the production backward policy. Artifact:
`ms_eden_sparse_independent_oracle_512.json`. The next policy investigation
should distinguish rotation-family limitations from EDEN/scale correction,
not attempt to repair this error by GEMM scheduling changes.

#### Explicit forward-quantized source ablation

The research driver accepts `--source-policy forward_current_roundtrip_bf16`.
It quantizes X/W with RTX tensor-current forward quantization, decodes to BF16,
and then runs the same backward preparation. dy remains original BF16. This
is a numerical ablation, **not** packed-autograd storage reuse: it materializes
BF16 and makes no memory or performance savings claim. The default remains
`original_bf16`; policy and forward quantizer configuration are recorded.

At Gaussian 512x768x256, 256 fresh rotations gave mean dX/dW NRMS
9.510%/9.574% relative to original BF16 gradients, but 0.854%/0.856% relative
to the forward-quantized source reference. Individual-gradient NRMS was about
16.6--16.7% relative to original BF16. This confirms that source reuse changes
the gradient target and cannot be introduced as a free scheduling optimization.
These are not Quartet's FourOverSix forward policy or full training results.
Artifacts: `ms_eden_forward_roundtrip_dx.json` and
`ms_eden_forward_roundtrip_dw.json` (initial runs predate the added serialized
forward-config field; the source policy was already recorded). Actual packed
state retention, fused decode/requantization, autograd lifetimes, checkpointing,
and training qualification remain required before exposing this publicly.

`compile_packed_hadamard_preparation` now implements the first direct packed
reader: row-major FP4 bytes, E4M3 block scales, and one FP32 tensor scale are
decoded into registers, explicitly rounded to BF16, then passed through the
same rotation/EDEN preparation. Logical transposed loading uses the original
source's scale indexing; no decoded-BF16 or transposed GMEM tensor is produced.
The explicit BF16 rounding preserves the round-trip policy above rather than
silently improving/changing it to FP32 decoding. This reader does not support
regional or tensor-core-native input scales, and is not yet wired into public
autograd or the preparation autotuner.

Four GPU cases (normal/transposed loading x three/four temporary mantissa bits)
match the explicit decode-to-BF16 producer bit-for-bit for packed bytes,
corrected scales, and partial maxima. Captured replay after zeroing packed
inputs and changing signs/outer scale also matches and clears every output.
Performance, saved-state ownership, and full backward integration remain open.

The research gradient driver and preparation tuner now accept `--packed-reader`
only with `--source-policy forward_current_roundtrip_bf16`. Adapter revision 2
partitions original-BF16, decoded-forward, and packed-forward executions;
thread/grid schedules remain independently tunable. Full dX/dW metrics match
the decoded-source path exactly in GPU tests. Public autograd remains unchanged.

On 2048x768x1536, fixed-schedule full dX was 46.57 us with resident decoded
BF16 versus 56.33 us with direct packed reading; dW was 73.74 versus 80.66 us.
**The decoded baseline excludes decoding**, so these numbers isolate reader
cost rather than establish an end-to-end staging winner. A four-trial isolated
packed dW smoke search on the other 5090 had four successful candidates and
selected the default schedule at 85.98 us; it is neither extensive tuning nor
a same-device comparison with the previous numbers. Artifacts:
`ms_eden_reader_{dx,dw}_{False,True}.json` and `ms_eden_packed_reader_tune_dw`.
Research references still retain decoded/source buffers, so these runs do not
demonstrate autograd saved-memory reduction. Fifty-two post-hoc/adapter GPU
and CPU tests passed after integration.

The `--staged-packed-reader` comparison now includes an explicit CuTe
packed-to-BF16 decode on every preparation invocation. Both staged and direct
paths use identical forward-quantized sources and match full-gradient metrics
exactly. With fixed 128-thread/128-block producer/decode geometry, full-gradient
times (direct / staged, us) were 58.64 / 62.25 for small dX, 84.40 / 111.39 for
small dW, 445.18 / 463.49 for large dX, and 1102.35 / 2129.49 for large dW.
Small/large are M=2048/32768 with N=768,K=1536; each pair ran on the same
5090. All full timing windows passed stationarity checks. Artifacts:
`ms_eden_decode_cost_{2048,32768}_{dx,dw}_{False,True}.json`.

These results compare initial implementations, not best schedules. In
particular the standalone scalar decoder currently shares producer geometry;
independent decoder geometry/vectorization could substantially improve it.
Therefore they justify keeping direct and staged candidates, not hard-coding
a crossover or claiming these are tuned speedups. The standalone decode is
also tested against an independent nibble-decode reference including a small
non-CTA-sized input.

The revision-3 research adapter now tunes `reader_mode` (direct/staged) and
independent `decoder_threads`/`decoder_blocks` alongside producer/finalizer
geometry. It reserves execution-plan coverage and deduplicates inactive
decoder settings for direct reading. Staged schedules are rejected for
original-BF16 source contexts. Forward-quantizer configuration is also part
of current context identity; the two initial revision-3 campaigns began just
before that tag was added, so their cached measurements are not silently reused
under the stronger current identity.

A 12-trial small dW campaign selected staged decoding with 256 threads and
1,024 blocks at 80.96 us, versus direct 85.63 us and default staged 111.18 us.
Thus independent decoder tuning reversed the fixed-schedule small-shape
conclusion. The campaign does not exhaust producer/GEMM scheduling and should
not be called a global optimum. Artifacts: `ms_eden_reader_choice_2048`.
Sixty-six numerical/adapter tests passed, including execution-plan equivalence,
independent decoder geometry, inactive-field deduplication and policy isolation.
The matched 12-trial large dW campaign selected direct at 1101.56 us; the best
tested staged decoder improved from 2130.05 to 1157.84 us but remained slower.
All 24 trials across the two campaigns succeeded. This is evidence for retaining
both plans and measuring by shape, not yet a qualified cross-SKU crossover
heuristic. Large artifact: `ms_eden_reader_choice_32768`.

### Experimental tensor-only gradient integration

`rtx.experimental_ms_eden.ms_eden_gradient` exposes one explicit dX or dW
operation with original BF16 operands, FP32 accumulation and BF16 output.
It registers `rtx_experimental::ms_eden_gradient` with a fake implementation;
shared Rademacher signs and independent scale uniforms are caller-owned tensor
inputs. A static `schedule_json` selects the tested preparation geometry.
The module is not re-exported from `rtx`, does not alter NVFP4Linear, and does
not implement automatic RNG generation, packed-source retention, or training
policy selection. Higher-order differentiation raises explicitly.

The opaque operator allocates per-invocation workspaces, compiles through
bounded kernel caches, then launches the preparation/GEMM chain without eager
Torch arithmetic between launches. User compilation can capture it; it does
not create internal CUDA graphs. Warmup before explicit graph capture remains
required. Metadata validation is shared with fake execution. Sign values
must be +/-1 and uniforms in [0,1); these contents are caller preconditions,
not synchronizing host validations.

GPU tests cover dX/dW opcheck (schema, autograd registration, fake and AOT
dispatch), fullgraph compilation, independent dense-Hadamard/quantization
oracle comparison, changed-input/randomness graph replay, no live-allocation
growth after warmup, and explicit higher-order rejection. These establish
an experimental operator boundary, not convergence or a release-qualified
NVFP4 backward policy. Packed-source public autograd remains unfinished.

The same experimental module now provides `ms_eden_linear` for bounded
training qualification: tensor-current NVFP4 forward, original BF16 X/W saved,
and MS-EDEN dX/dW. Fresh signs/uniforms are generated by traceable Torch
operations in its autograd formula, outside the opaque gradient operator,
so compiled training includes RNG generation. It remains outside the public
`rtx` exports and does not change NVFP4Linear. Tests establish eager reentrant
checkpoint equivalence including final RNG state, compiled RNG-state replay,
noncontiguous upstream gradients, and x-only/weight-only gradients. They do
not claim eager and compiled random streams are bitwise identical to each other.

`benchmarks.train_ms_eden_probe` ran 200 compiled AdamW steps at widths 128 and
256 using BF16 weights, gradients, first/second optimizer moments (the step
counter is FP32). Train inputs are Gaussian std 0.25; separate held-out inputs
have std 1.0. Held-out output NRMS against the BF16 teacher:

| Width | BF16 | NVFP4 / MXFP8 backward | NVFP4 / MS-EDEN backward |
|---|---:|---:|---:|
| 128 | 1.287% | 16.967% | 17.030% |
| 256 | 3.922% | 18.763% | 18.738% |

All gradients remained finite. The two backward policies gave close outcomes
in this small fixed-data/single-seed probe, but both quantized paths remain
well above the BF16 error. This does not establish transformer convergence,
statistical equivalence, or absence of the previously measured sparse bias.
Wall times include compilation/optimizer/host checks and are not throughput
comparisons. Artifacts: `ms_eden_training_probe_128_v2.json` and
`ms_eden_training_probe_256.json`. The initial non-v2 128 run allowed automatic
MXFP8 backward tuning; v2 pins the backward configuration for reproduction.

### Packed-state experimental autograd integration

`ms_eden_linear(..., source_policy="forward_current_roundtrip_bf16")` now
retains actual packed forward data/scales and calls the registered packed
gradient operator. `schedule_json` chooses direct or staged packed preparation.
This explicit policy uses exact tensor-current outer scales for forward;
the original-source experimental path uses the existing dynamic power2
forward. Therefore switching source policies is not a schedule-only numerical
comparison, and no such performance claim is made.

The packing/GEMM operator returns only fresh output/payload tensors. Outer
scale reductions remain traceable outside this opaque boundary; all launches
inside it are CuTe kernels. Scale storage crosses the dispatcher as uint8
bytes, avoiding a PyTorch opcheck FP8 arithmetic limitation, and is reinterpreted
as E4M3 at the kernel boundary. Both packed operators have fake implementations,
metadata guards, opcheck coverage, and explicit higher-order rejection.

Saved-tensor inspection for two 128x128 operands reports six payload tensors,
18,440 bytes versus 65,536 bytes for BF16 X/W (about 71.9% less saved tensor
storage). Detached payloads have neither Python `_base_inputs` annotations
nor tensor `_base` references. This is not peak-VRAM qualification: outputs,
gradients, optimizer state, graph pools and temporary workspaces remain.
Tests cover eager checkpoint/RNG replay and partial gradients for both source
policies, fullgraph training for both packed execution plans, packed-gradient
equivalence to explicit BF16 decoding, and independent packed-forward numeric
checks. The path remains experimental and outside public NVFP4Linear defaults.

`forward_outer_mode="exact"` or `"power2"` now explicitly matches forward
policy across original and packed saved sources. Omitting it preserves the
earlier experimental defaults. A GPU regression verifies bitwise-identical
forward outputs between source policies for each outer mode. Constructing
policy dataclasses inside a traced function exposed a Dynamo error; policies
are now immutable objects created outside tracing, and fullgraph tests pass.

Matched-forward 200-step teacher probes produced held-out NRMS:

| Width / outer mode | MXFP8 backward | MS-EDEN original sources | MS-EDEN packed sources |
|---|---:|---:|---:|
| 128 / power2 | 16.967% | 17.030% | 16.799% |
| 256 / exact | 18.791% | 18.704% | 18.556% |

All NVFP4 variants had identical initial losses within each probe; training
uses BF16 optimizer moments as before. These single-seed differences are not
evidence of an accuracy advantage or equivalence. BF16 reference NRMS remained
1.287% and 3.922%. Artifacts: `ms_eden_matched_training_128_power2.json` and
`ms_eden_matched_training_256_exact.json`. Fourteen experimental integration
tests passed after the explicit forward-policy change.

### Joint preparation and GEMM tuning

Research adapter revision 4 includes a serialized NVFP4 GEMM schedule and
reuses existing GEMM axes for geometry, buffering, warp layout, register
budgets, scale movement, ldmatrix, locality/persistence, and epilogues. Shared
legality checks retain quarantines and reject producer/GEMM scale-layout
mismatches. The experimental registered gradient consumes the same schedule;
nondefault 2x2-warp/three-stage dX and dW pass the independent oracle gate.
The wrapper-level global L2-fetch option is not included because these
compile entrypoints do not expose it. Candidate generation is bounded rather
than materializing the expanded Cartesian product. Numerical source/rounding
policies remain separate from the schedule.

Two eight-trial 32768x768x1536 searches succeeded on all candidates:
original-BF16 dX improved from 437.81 to 428.63 us; packed-forward dW improved
from 1141.82 to 1132.55 us. Both selected a 128x128x128, 2x2-warp, three-stage
GEMM. These are small-budget, same-run comparisons and not global winners.
They demonstrate some schedule headroom, but do not eliminate preparation
cost or establish superiority to MXFP8 backward. Artifacts:
`ms_eden_joint_dx_32768` and `ms_eden_joint_packed_dw_32768`.
Sixty-five integration/kernel tests plus three adapter tests passed before
the searches; nineteen integration/adapter tests passed afterward, including
the new nondefault-GEMM numerical cases.

### Profile evidence for the next preparation change

#### Four-shape load matrix and resource-aware seeds (2026-10-23)

Thirty-two fresh-process cases compare both load layouts and both traversals
on all four target shapes for dX/dW. Producer/finalizer grids and threads stay
at 128; GEMM, source policy, operands and RNG stay fixed. Every numerical gate
and full-timing stationarity gate passed. Each shape's comparisons run on one
5090; shapes are split across the two local devices. The execution order is
deterministically shuffled (seed 821). These are schedule controls, not global
optima or cross-SKU generalization evidence.

Full preparation plus one GEMM, microseconds:

| M,N,K / phase | Row/direct | Row/cooperative | Reduction/direct | Reduction/cooperative |
|---|---:|---:|---:|---:|
| 2048,768,1536 / dX | 48.74 | 45.93 | 45.96 | 45.93 |
| 2048,768,1536 / dW | 76.58 | 69.88 | 70.18 | 70.00 |
| 8192,768,1536 / dX | 102.66 | 99.97 | 100.59 | 100.57 |
| 8192,768,1536 / dW | 227.04 | 221.38 | 238.74 | 237.19 |
| 32768,768,1536 / dX | 450.80 | 449.68 | 453.46 | 453.57 |
| 32768,768,1536 / dW | 1021.65 | 1075.20 | 1404.82 | 1371.48 |
| 8192,3072,768 / dX | 306.45 | 305.78 | 316.02 | 316.12 |
| 8192,3072,768 / dW | 410.37 | 402.09 | 439.87 | 434.35 |

Artifacts: `ms_eden_load_matrix_v1_gpu0/` and `ms_eden_load_matrix_v1_gpu1/`,
including each subprocess log and JSON. In particular, cooperative loading
loses at the large default grid despite winning after grid/thread tuning.
Per-axis greedy pruning at a single baseline would discard useful schedules.

The research adapter now records structural features for each candidate:
operand orientation/stride, capped producer and finalizer CTAs, partial counts,
requested shared-memory tensor bytes, SM-relative grid size, temporary/decoded
workspace, packed payload relative to L2, launch count, and logical partial-max
loads. These are model inputs, **not measured DRAM traffic, allocated-SMEM
rounding, occupancy predictions, or a trained cost model**. Missing device
capacities carry explicit known flags. Feature schema revision 1 is in the
context tags; the research tuner records L2 capacity alongside its fingerprint.

`preparation_resource_seeds` proposes the baseline, cooperative 128-thread /
2-SM-wave and 256-thread / 4-SM-wave grids with an 8-wave finalizer, plus a
direct-load challenger. Grids round to 32 CTAs and cap against available work;
dX seeds use row-major and dW seeds reduction-major traversal. Both regimes
are measured rather than selecting one with a fitted shape threshold. The
baseline, original numerical policy, optional reader mode, and GEMM are kept.
Broader axes remain available after these seeds. This is a soft research prior,
not public dispatch or cross-device qualification; no ML performance claim.

Four-trial dW studies on the two intermediate shapes selected 138.167 us from
a 235.480 us baseline (8192,768,1536), and 200.651 us from 408.990 us
(8192,3072,768). A faster wide candidate at about 171.9 us failed one full
timing window's range gate and was not selected; preserve that quality decision
rather than reporting the smallest raw median as the winner. Artifacts:
`ms_eden_resource_prior_{8192,wide}_dw`. Correctness, first-hit confirmation,
and persistent residual recording remain mandatory.

The corresponding four-trial dX searches selected 69.719 us from 106.265 us
at 8192,768,1536 and 149.882 us from 306.589 us at 8192,3072,768. The former
prefers the 128-thread seed, the latter the 256-thread seed, supporting a
portfolio rather than one fixed thread/grid rule. Artifacts:
`ms_eden_resource_prior_{8192,wide}_dx`. All selected winners passed numerical
and stationarity gates. Forty-nine CPU autotuner tests passed, including exact
feature-count checks and preservation of policy/GEMM/reader settings in seeds.

A five-trial refinement retried the rejected wide-dW candidate at 171.828 us,
but again one timing window exceeded the unchanged 10% range gate (10.158%).
The other window passed; confirmation correctly retained the ineligible status
and the 200.651 us winner. No threshold was relaxed. Progress logs now append
`timing_ineligible` to numerically successful observations excluded from winner
selection, avoiding the misleading appearance that every `ok` median can win.
The existing unstable-observation regression now checks that visible marker.

#### Cooperative transposed-source loader (2026-10-23)

`producer_load="coalesced"` adds an optional CTA-local BF16 staging layout to
the experimental preparation kernel. Threads load neighboring source columns
together into a padded shared-memory tile; each warp then reads its logical
128-element Hadamard vector into registers. There is no standalone transpose
pass, globally transposed tensor, or full decoded-BF16 allocation. Packed inputs
decode directly into the tile with the same explicit BF16 rounding as the
original packed reader. Nontransposed operands retain the direct loader.
Two CTA barriers protect tile fill/read/reuse. The mathematical 128-element
transform and subsequent FP32 arithmetic/rounding are unchanged.

The load choice is exposed alongside thread count, grid, traversal, maximum
reduction and GEMM axes; the research adapter is revision 7. Older JSON
schedules default to `direct`. The research tuner accepts `--producer-load`,
and the study driver also accepts `--producer-threads` for controlled sweeps.
CTA width follows the selected warp count; this first implementation uses one
BF16 element of shared-memory stride padding, not an exhaustive layout search.
Production defaults and MXFP8 backward remain unchanged.

At the previous reduction-major winners' grids, full dW event timings were:

| M,N,K | Direct, 128 threads | Coalesced, 128 | Coalesced, 256 | Coalesced, 512 |
|---|---:|---:|---:|---:|
| 2048,768,1536 | 51.200 us (fresh rerun) | 48.972 us | 52.772 us | 76.556 us |
| 32768,768,1536 | 517.221 us (fresh rerun) | 466.440 us | 435.864 us | 467.835 us |

The 64-thread coalesced cases were slower at 57.293/613.772 us. Grid and GEMM
were held fixed; these results are not exhaustive tuning or training-step
speedups. Each full timing passed stationarity gates, with unchanged
independent packed-reference metrics. Artifacts:
`ms_eden_dw_{2048,32768}_coalesced_load.json` and
`ms_eden_dw_{2048,32768}_coalesced_t{64,256,512}.json`.

Validation passed 134 GPU tests and 47 CPU autotuner tests. Coverage includes
bitwise packed values/scales/FP32-factor equality, logical orientations, both
scale layouts, both traversal orders, inline/staged maxima, original/packed
sources, 64/128/256/512 threads, partial CTA-grid iterations, and fullgraph
training RNG replay. Compute Sanitizer racecheck passed the 16 cooperative
loader geometry cases with zero hazards/errors/warnings; raw log:
`ms_eden_coalesced_racecheck.log`.

The first memcheck of those tests exited with 26 **host API lookup errors**:
cuda-python requests CUDA 13.3 symbols from the installed 13.2 driver while
the unrelated forward quantizer discovers hardware capabilities. No invalid
memory access was reported, but that is not a clean sanitizer pass. The loader
test now generates packed bytes/scales directly and uses independent decoding,
which also covers all legal nibble values without relying on that quantizer.
All 16 isolated geometry cases then passed memcheck with **zero errors**.
The isolated cases also passed racecheck with zero hazards, errors or warnings
(`ms_eden_coalesced_racecheck_isolated.log`).
Both original and isolated logs are retained as
`ms_eden_coalesced_memcheck.log` and `ms_eden_coalesced_memcheck_isolated.log`.
No driver, package, or unrelated service was modified.

Large-winner counters (704 CTAs, 256 threads, cooperative loading) show
producer replay time 85/176 us versus 133/261 us for the prior direct winner;
L2 traffic falls from 279/604 to 134/267 MB. Active warps rise from about 33%
to 60%, registers rise from 40 to 41, and local spills remain zero. DRAM reads
are still approximately 50/101 MB. The additional producer warps increase
partial maxima, and finalizer replay rises to 68/77 us. These are joint
load/thread-configuration counter deltas, not isolated load-only effects or
ordinary timing. Raw artifacts: `ms_eden_dw_coalesced_selected_counters.ncu-rep`
and `.csv`, plus `ms_eden_dw_coalesced_selected_profile.json`. The unchanged
GEMM remains about 183 us under replay. Do not infer memory/tensor saturation.

#### Inline versus staged partial-max reduction (2026-10-23)

Schedule axis `amax_reduction` now offers `inline` (unchanged default) and
`staged`. The latter uses a one-warp kernel to reduce each producer's nonnegative
FP32 partial maxima once, then gives the scale finalizer one FP32 scalar. It
adds one launch and four workspace bytes per operand; no decoded tensor,
transpose, dtype change, delayed estimate, or eager Torch arithmetic is added.
The research operator supports both original and packed sources. Adapter
revision 6 records the execution change; old schedule JSONs default to inline.
The research first-hit portfolio includes the choice, and the benchmark/tuner
accept `--amax-reduction inline|staged` for controlled comparisons.

This plausible traffic optimization did **not** give a substantial measured
win. Same-device fresh-process event timings, full preparation plus one GEMM:

| Shape M,N,K / phase | Inline | Staged |
|---|---:|---:|
| 128,256,128 / dW, default schedule | 9.700 us | 10.751 us |
| 2048,768,1536 / dX, default schedule | 47.005 us | 47.869 us |
| 32768,768,1536 / dX, default schedule | 450.834 us | 452.024 us |
| 32768,768,1536 / dW, tuned reduction-major schedule | 517.350 us | 516.811 us |

Small tuned dW staged measured 51.407 us, also no improvement over the prior
51.219 us winner. The large tuned dW inline rerun is important: comparing only
against the older 521.903 us winner would exaggerate the staging benefit.
All full timing samples passed stationarity gates. An independent maximum
reduction does not automatically shorten the critical path: it adds a launch
and still performs the reduction before finalization. This is a mechanism
hypothesis, not a candidate-counter diagnosis. Do not promote staged as a
universal improvement or infer ordinary-time savings from profiler replay.

Artifacts: `ms_eden_dw_{2048,32768}_staged_max.json`,
`ms_eden_dw_32768_inline_max_recheck.json`, and `ms_eden_max_control_*.json`.
Validation passed 100 GPU tests and 47 CPU autotuner/preparation tests. New
coverage checks independent FP32 maxima for 1/3/512/2816/8192 partials, changed
inputs and zero replay, bitwise packed/scales/factor invariance, independent
dX/dW references, and fullgraph compiled training with restored RNG for both
original and packed-source policies. Public MXFP8 backward is unchanged.

A four-trial revision-6 first-hit smoke at 2048,768,1536 dW compared both
traversals and both reduction plans. It selected reduction-major/inline at
67.616 us versus the initial 73.502 us. This low-budget smoke does not recover
the deeper-search 51.219 us grid winner. A fresh-process rerun took the exact
cache hit, performed no new trials, and left every residual-store file's SHA256
unchanged. Artifact: `ms_eden_first_hit_revision6_smoke`.

#### Traversal-order experiment (2026-10-23)

The selected 192-producer / 1,376-finalizer grid was re-profiled on large dW.
Its producers take 226/446 us in Compute replay, 40 registers/thread, no local
spills, and 9.55/9.57% active warps. Finalizers take 18/27 us and roughly 65%
active warps. Producer DRAM reads remain 50.34/100.68 MB; L2 traffic rises to
418/803 MB. Thus the grid speedup is not evidence of lower memory traffic.
Raw artifacts: `ms_eden_dw_grid_selected_counters.ncu-rep/.csv` and
`ms_eden_dw_grid_selected_profile.json`. Profiler times are diagnostic only.

A new `producer_order` schedule axis offers `row_major` (unchanged default)
and `reduction_major`. The latter assigns neighboring logical rows to nearby
warps, potentially reusing source cache lines when logically transposed.
Output tile addresses, transform, rounding, correction and scale policy are
unchanged. This does not physically transpose a tensor or make per-warp
strided loads coalesced. Both original and packed readers support it, including
the experimental registered operator. Adapter revision 5 isolates the new
search context; older schedule JSONs still deserialize with row-major order.
First-hit portfolios balance reader mode and traversal before spending their
remaining budget on grid/GEMM axes. No production default changes.

Fixed-grid fresh-process A/B results: small dW row-major 68.868 us versus
reduction-major 62.775 us; large dW row-major 765.707 us versus reduction-major
893.918 us. All full timings passed stationarity gates. This shape-dependent
regression is why both remain choices, not an unconditional replacement.
Artifacts: `ms_eden_dw_{2048,32768}_{row_order_recheck,reduction_order}.json`.
Separate reduction-major grid searches investigate scheduling interactions.

Both searches completed 25 reduction-major grid pairs plus the row-major
default baseline, with correctness gates and fresh-process confirmations:

| M,N,K | Best earlier row-major | Tuned reduction-major | Producer/finalizer CTAs | Incremental speedup |
|---|---:|---:|---|---:|
| 2048,768,1536 | 68.672 us | 51.219 us | 352 / 1376 | 1.341x |
| 32768,768,1536 | 768.273 us | 521.903 us | 704 / 1376 | 1.472x |

These are full preparation-plus-dW-GEMM timings, fixed threads/GEMM/math,
not training-step speedups. All 52 distinct candidates succeeded. Artifacts:
`ms_eden_reduction_grid_dw_{2048,32768}`. The new traversal loses at the old
large grid but wins after retuning: retain traversal/grid interactions in the
search, and do not prune a new traversal using only another traversal's winner.
No numerical-policy or production default changes follow from these results.

Counter follow-up for the large winner supports the combined traversal/grid
mechanism: producer replay times fall from 226/446 to 133/261 us, active warps
rise from about 9.6% to 33.2%, and L2 traffic falls from 418/803 to 279/604 MB.
DRAM reads remain about 50/101 MB. Registers stay at 40 with zero local spills.
Finalizers become slower (39/47 versus 18/27 us), consistent with more producer
partial maxima to scan; the producer savings outweigh that cost. These counters
compare jointly tuned schedules, not traversal alone. GEMM replay time remains
about 184 us. No bandwidth or tensor-core saturation claim follows.
Artifacts: `ms_eden_dw_reduction_selected_counters.ncu-rep/.csv` and
`ms_eden_dw_reduction_selected_profile.json`; the independent packed-GEMM NRMS
is unchanged at 0.00165356. Further opportunities include genuinely coalesced
transposed loads and a less redundant partial-max reduction, rather than
silently changing the precision policy.

Validation: 81 GPU tests passed, including bitwise random/zero grid-order
invariance in both layouts/orientations, packed-reader BF16-roundtrip equivalence
for both temporary precisions, captured changed-input replay, registered dX/dW
against the independent oracle, and existing compile/checkpoint/autograd tests.
Forty-seven CPU first-hit/adapter/preparation tests passed; all four adapter
tests were rerun successfully after adding the invalid-order assertion.

#### Device/work-size grid proposals (2026-10-23)

The research tuner now seeds producer and finalizer grids independently from
SM count times 1/2/4/8, rounded to 32 CTAs and capped by available work. It
retains the baseline, tests isolated changes first, then crosses the proposed
grids. These are soft proposals requiring measurement, not an occupancy model
or production dispatch rule. General grid axes also reach 2,048 CTAs.
`--grid-only` fixes threads, numerical policy, and GEMM for controlled studies.
No kernel arithmetic, production default, or cache numerical identity changes.

Two local RTX 5090 studies use original BF16 sources, fixed signs/uniforms,
native scales, the same GEMM, graph-event timing, and fresh isolated trial
processes. Each study evaluates 25 distinct grid pairs, with additional
confirmation runs. Small and large shapes use different GPUs; each speedup
must be calculated against its own same-device baseline. Every trial retains
independent decoded-packed GEMM validation and timing stationarity metadata.
Artifacts: `ms_eden_grid_dw_2048` and `ms_eden_grid_dw_32768` below the report
root; per-trial schedule/result JSONs and residual journals are retained.

Completed confirmed selections (microseconds, full preparation plus dW GEMM):

| M,N,K | Baseline 128/128 grids | Selected producer/finalizer grids | Selected time | Speedup |
|---|---:|---|---:|---:|
| 2048,768,1536 | 73.452 | 128 / 1376 | 68.672 | 1.070x |
| 32768,768,1536 | 1021.485 | 192 / 1376 | 768.273 | 1.330x |

All 50 distinct configurations completed successfully. Large selected
preparation measured approximately 625 us and isolated GEMM 126 us; isolated
times need not sum to the full sequence. These are experimental MS-EDEN dW
results, not MXFP8 backward or whole-training speedups, and not a global optimum.
Defaults remain unchanged. The next structural candidate is the transposed
producer's memory-access ownership, after re-profiling the selected grid.

Initial isolated probes already reject a monotonic "more producer blocks is
better" rule. At M=2048,N=768,K=1536, increasing producer blocks from 128 to
1,376 with finalizer fixed at 128 worsens full dW from about 73.45 to 112.78 us.
Changing only finalizer blocks to 1,376 improves it to 68.67 us. More producer
CTAs also enlarge the partial-max reduction read by each finalizer warp;
address scheduling and cache behavior may contribute, so the measurements do
not attribute all regressions to that reduction without follow-up counters.

New GPU invariance tests compare packed bytes, E4M3 scales, and FP32 outer
factors bit-for-bit for normal/logically-transposed operands, row-major/native
scales, 192/1,376-block grids, random inputs, and a subsequent zero-input replay.
These tests passed in all four combinations. Forty-seven CPU adapter,
first-hit, and preparation tests also passed. The full post-hoc kernel and
experimental autograd suite subsequently passed all 71 GPU tests, including
compile, checkpoint/RNG, packed-source, and nondefault-GEMM cases. This is schedule qualification,
not evidence that the experimental FP4 backward matches MXFP8 accuracy.

Reproduce a complete grid study (use a new output directory for a fresh run):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m benchmarks.tune_ms_eden_preparation \
  --shape 32768,768,1536 --phase dw --trials 26 --grid-only \
  --output autotune_reports/preparation_20261023/ms_eden_grid_dw_32768
```

Nsight Systems 2025.6.3 and Nsight Compute 2026.2.1 captured one warmed,
original-BF16 32768x768x1536 dW on the local 5090, default schedule and fixed
explicit randomness. Compilation, allocation, reference computation and 25
warmups precede cudaProfilerStart. Nsight Python is not installed, so the
available Compute CLI was used. Independent packed-reference NRMS was 0.001654.

The Systems capture has six kernels: rotation/quantization takes 69.5% of
summed GPU kernel time, finalization 16.6%, GEMM 13.7%, factor product 0.1%.
There are six launch calls and the expected measurement-boundary synchronize;
the capture does not indicate eager elementwise work between these launches.
Compute reports 40 registers/thread, zero local-load/store sectors, and about
8.3% active warps for both producers (128 CTAs of 128 threads on 170 SMs).
Their L1 global-load sectors are 37,748,736 and 75,497,472: these counters
include sign loads as well as source loads, so they must not be described as
source DRAM bytes. DRAM reads are approximately 50.34/100.68 MB; L2 traffic
in the single-pass DRAM capture is 267.95/515.55 MB. Replay producer durations
are 263/524 us. These profiler times are diagnostic, not replacements for
ordinary event timings, and no peak-bandwidth claim follows from them.

The next evidence-backed experiment is wider grid coverage before a coalesced
transposed loader: no spills were observed, while low active warps and large
sector traffic make register trimming/GEMM-only tuning poor first priorities.
The first counter capture requested obsolete DRAM metric names and omitted
those counters; a second capture uses verified `dram__bytes_op_read.sum` and
`dram__bytes_op_write.sum`. Both raw reports are retained.

Artifacts under the report root: `ms_eden_dw_timeline.nsys-rep`, its SQLite
export and `ms_eden_dw_stats_*.csv`, `ms_eden_dw_counters.ncu-rep/.csv`,
`ms_eden_dw_dram.ncu-rep/.csv`, corresponding run JSONs, and
`ms_eden_profile_environment.json`. Reproduction entrypoint:
`python -m benchmarks.study_ms_eden_backward --shape 32768,768,1536 --phase dw
--no-benchmark --profile --output <new-json-path>`; profiler capture starts
only at the explicit CUDA profiler API range.

The MS-EDEN research preparation now uses the shared first-hit autotuner with
independent producer/finalizer thread and grid sizes, isolated trial processes,
residual journals, and phase/layout/numerical-policy-specific cache identities.
Sixteen-trial RTX 5090 searches selected 256 producer threads / 128 blocks for
`2048x768x1536` dX (44.37 us full gradient), and 128 threads / 256 blocks for
`32768x768x1536` dW (804.45 us). Both selected 128 finalizer threads / 128 blocks.
Earlier fixed-schedule measurements were 48.93 and 996.16 us respectively;
these are exploratory comparisons, not independently replicated speedups.

Artifacts are the `selection-*.json` files and per-trial records in
`autotune_reports/preparation_20261023/ms_eden_tuned_dx_2048` and
`ms_eden_tuned_dw_32768`. This tuning changes schedules only: original BF16
operands, shared-sign Hadamard-128, four retained temporary mantissa bits,
EDEN correction, and stochastic scale rounding remain fixed. Random-input
generation is excluded and GEMM configuration is fixed. The results do not
qualify public FP4 backward, demonstrate a win over MXFP8, or complete the
packed-forward-state reuse and training/RNG-validation requirements below.

## Active goal scope and acceptance order

The user expanded the active goal on 2026-10-23 to include the following
Quartet-derived implementation work. These are required follow-through, not
claims that existing measurements already qualify them. The original
execution-plan selection objective remains in scope.

### First: establish a tuned, phase-resolved baseline

Use M,N,K shapes `2048,768,1536`, `8192,768,1536`,
`32768,768,1536`, and `8192,3072,768`. Tune the existing RTX path with a
recorded budget, then measure forward preparation, forward GEMM, backward
preparation, dX, dW, and total forward-plus-backward. Record overlapping
execution explicitly: isolated phase times need not sum to the total.
Compare packed GEMMs against Quartet's actual installed GEMM backend,
independently of quantization. Use identical packed values/scales and layout
conversions where possible; account for any conversion cost separately.

### Then: implement and qualify the transferable ideas

1. **Post-hoc range alignment.** Investigate extended-range temporary scales
   followed by a scale-only finalization pass once the global maximum is
   available. Target MS-EDEN backward and/or tensor-current preparation. Verify
   the numeric contract, underflow/overflow handling, and stochastic correction;
   do not assume this benefits JIT-region. Measure whether full-tensor rereads
   and repeated rotations are actually removed.
2. **Fused backward preparation.** Combine rotation, quantization or
   requantization, correction statistics, and tensor-core layout emission where
   legal. Compare full-tensor passes and bytes, not merely launch counts. Expose
   implementation alternatives to tuning without changing the policy.
3. **Optional FP4 backward.** Implement and tune MS-EDEN as an explicit
   experimental numerical policy. Preserve MXFP8 backward as the default.
   Validate the actual dX/dW estimators, stochastic replay/RNG restoration,
   finite-sample variance and bias, and training behavior; individual-quantizer
   unbiasedness is not sufficient proof for the complete GEMM.
4. **Explicit packed forward-state reuse.** Offer saving/requantizing packed
   forward operands as a separately identified numerical choice. Distinguish
   its forward-quantized gradient oracle from original-BF16-operand backward.
   Measure saved-state memory and total training cost; test checkpointing and
   autograd lifetimes. Never silently select it as a scheduling optimization.
5. **Vendor GEMM diagnostic baseline.** Keep the packed-GEMM comparison as a
   diagnostic tool to separate schedule quality from preparation cost and
   precision-dependent hardware throughput. Record the actual backend/version,
   not just the paper's backend description.

### Completion evidence

Keep policy-preserving schedule comparisons separate from numerical-policy
ablations. Require raw timings, configs, device/toolchain identities, seeds,
independent numerical references, and correctness gates before accepting speed
claims. Qualify eager/fullgraph compile/CUDA graphs, changing inputs, gradients,
memory ownership, and persistent cache invalidation for integrated paths.

The final analysis must answer separately: (a) schedule headroom, (b) avoidable
preparation traffic, and (c) gains requiring FP4 backward. No promise is made
that tuning alone closes the gap while retaining MXFP8 gradient precision.

Reference supplied with this scope:
[Quartet II, section 7](https://arxiv.org/html/2601.22813v2#S7).

### Post-hoc alignment oracle: numerical groundwork

`benchmarks/reference_ms_eden.py` is an independent Torch reference, not a
runtime kernel. It accepts already-rotated FP32 blocks, rounds FP4 with an
explicit nearest-even grid, calculates EDEN's block correction, stores BF16
temporary scales, and finalizes only those scales into E4M3 using caller-owned
uniform samples and an FP32 outer factor. The zero-tensor convention is
outer=1 with zero local scales. Subnormal temporary scales explicitly flush
to zero; BF16 temporary overflow is rejected.

The paper describes E8M3 intermediate scales. The pinned upstream RHT source
(`had128_utils.cuh`, `eden_rounding`, SHA
`5f2a47e8f464b17ef4b1cac78db8d14edd1f3f7d`) actually masks away 19 FP32
mantissa bits, retaining four with truncation. The oracle exposes three and
four bits separately; a regression confirms they produce different FP4
values. This detail must be matched before claiming numerical parity.

Twelve CPU tests cover FP4 ties/saturation, the pre-BF16-correction
inner-product identity, analytic scale-SR conditional expectation including
FP8 subnormals, zero handling, explicit random replay, and magnitudes from
1e-20 to 1e20. Neither the inner-product identity nor unbiased scale rounding
proves the complete gradient estimator unbiased. Rotation pairing, BF16
temporary-rounding error, cross-operand RNG correlation, GPU fusion, and
training validation remain to be implemented/tested. No public backward
policy changed, and no speed claim is made for this reference.

The experimental CuTe component `rtx/kernels/nvfp4_posthoc.py` now implements
scale-only finalization with configurable 64/128/256/512-thread launches,
bounded grid-stride CTAs and row-major or native mma128 scale emission.
Its inputs are nonnegative finite corrected BF16 scales, their FP32 maximum,
and caller-owned FP32 uniforms. It never reads or rewrites the packed FP4
operand. The global reduction and random-number production are not included;
materializing uniforms is a validation interface, not a production traffic
optimization. It has no public layer or autograd registration yet.

The initial GPU test caught a one-ULP outer-factor difference between division
and reciprocal multiplication. Both reference and kernel now explicitly use
FP32 multiplication by rounded `1/448`; no tolerance was loosened. GPU tests
require bit-identical E4M3 outputs and FP32 factors, including zero, logical
tails, native physical layout, FP8 subnormals, extreme scale magnitudes, and
CUDA graph replay after changing scales and random samples. These component
tests do not qualify the full rotated gradient estimator or its RNG policy.

The next component, `ExtendedPreparationKernel`, consumes already-rotated
FP32 values once and writes packed FP4, BF16 corrected scales, and per-warp
FP32 maxima. Every partial slot is overwritten, including inactive lanes;
no global atomic or separate zeroing launch is required. Scale finalization
can now reduce those partials in each consumer warp, so the complete two-pass
chain contains no Torch reduction between launches. This repeated small
reduction is a baseline implementation, not a demonstrated optimum.

GPU coverage extends to 512x1536 values, three/four-bit temporary policies,
grid-stride iteration, tails, and two-pass CUDA graph replay with nonzero-to-zero
input changes. Packed FP4 values match the independent oracle exactly. At the
largest three-bit test, one of 49,152 BF16 temporary scales differs by one
adjacent representation because FP32 reduction association crosses a BF16
midpoint. The FP64 value is 0.08227539782286816; the midpoint is 0.082275390625.
Tests allow an adjacent BF16 difference only when the independent FP32
correction lies within eight FP32 epsilons of that midpoint. There is no
general BF16-relative-error waiver. Finalization is still bit-exact against
the actual temporary-scale input. This is not bitwise equivalence of the
entire producer to Torch or upstream Quartet. Rotation, counter-based RNG,
gradient GEMMs, and training integration remain outstanding.

### Fused BF16 Hadamard preparation component

`HadamardPreparationKernel` now performs a normalized 128-value butterfly
transform, FP4 quantization, EDEN correction, and partial maximum production
inside one kernel. Four FP32 registers per lane and warp shuffles implement
the transform; it does not materialize a rotated GMEM tensor. Input is BF16,
signs are explicit shared Rademacher inputs, and a `transpose` specialization
loads the original matrix with logical transposed indices (no transpose
buffer). The reduction dimension currently requires divisibility by 128.

Independent dense FP64 Hadamard references check 128x256 and 384x512 inputs,
both orientations and both temporary mantissa policies. Packed values match
exactly. The larger three-bit ordinary-layout case has one adjacent corrected
BF16 scale at a rounding midpoint, covered by the same narrow FP32-midpoint
check rather than a new general tolerance. Partial maxima match the actual
stored scales. The full component/reference suite passes 42 tests on local
5090. This establishes the tested transform/preparation semantics, not FP4
gradient unbiasedness, a training-quality result, or throughput advantage.
Counter-based RNG, gradient GEMM integration, and full kernel tuning remain.

### First FP4 dX/dW GEMM integration (research only)

`benchmarks/study_ms_eden_backward.py` connects original-BF16 preparation to
the existing NVFP4 GEMM. Both operands use the same explicit Hadamard signs
and separate explicit scale-uniform arrays. dX logically transposes W; dW
logically transposes dy and X. Native scales are emitted directly. An explicit
small CUDA factor-product kernel avoids eager Torch arithmetic between
preparation and GEMM. This baseline uses six launches per gradient; dual
preparation/finalization fusion and RNG generation are not yet implemented.

All four target shapes passed the independent decoded-packed GEMM reference
(NRMS 0.00165–0.00166). Original-BF16 gradient NRMS was 0.1368–0.1374 for this
one seeded Gaussian study. This measures one realization, not estimator bias,
variance, convergence, or parity with the approximately 0.0375 MXFP8 error.

| M,N,K | dX prep / GEMM / full (us) | dW prep / GEMM / full (us) |
|---|---:|---:|
| 2048,768,1536 | 38.65 / 10.50 / 48.93 | 63.67 / 9.74 / 73.32 |
| 8192,768,1536 | 80.47 / 25.64 / 106.09 | 194.39 / 32.00 / 226.99 |
| 32768,768,1536 | 252.17 / 101.76* / 449.96 | 855.24 / 122.24 / 996.16 |
| 8192,3072,768 | 261.27 / 43.13 / 311.38 | 322.86 / 35.84 / 404.07 |

*The isolated large dX GEMM failed the timing-stationarity gate; retain its
samples but do not use it as a reliable comparative latency. All other listed
timings passed. Phase measurements are not additive. These fixed schedules
are clearly preparation-bound and are not claimed faster than MXFP8 or vendor
implementations. Stochastic-input generation is excluded from timing, so this
is not full training cost. Artifacts: `ms_eden_dx_*.json`, `ms_eden_dw_*.json`.
Four CUDA integration tests additionally check both output scale layouts and
both gradient orientations on the small shape.

## Contract

Choose execution plans within a numerical policy, never between FP4 and FP8
backward implicitly. Match block geometry, rounding, scales, source operands,
and accumulator policy. Preparation means per-call materialization, not cached
training weights. Physical transpose buffers are not implied.

Candidate plans are fused, stage A, stage B, and stage both, but only paths
actually implemented and legal may enter search. Current MXFP8 forward has
fused and materialized implementations. Backward exposes fused/decomposed
independently for dX and dW. Hybrid preparation needs a separate implementation
audit; the feature representation does not assert those kernels exist.

## Prior and search

`rtx.autotune.preparation` computes logical preparation volume, reuse counts,
packed intermediate volume, CTA waves, and intermediate/L2 ratio. Features
are integrated into the shared hardware traffic features used by adapters.
Unknown device capacities have explicit known flags. These are source-level
estimates, not measured HBM traffic or a calibrated latency prediction.

The portfolio helper reserves one candidate per distinct execution plan before
filling with schedule refinements. It accepts caller-owned scores.
`make_preparation_adapter` now composes existing family adapters, validates
matching workloads/devices/regimes, and rejects numerical-policy changes.
Its cache identity includes all component contexts/revisions and the numerical
policy. `tune_preparation` uses the shared first-hit orchestrator, confirmation,
and residual store. Public linear-launcher integration is experimental, not yet
the default or fully qualified across all effort policies.

Required follow-through:

1. Build competitive, policy-matched schedule portfolios for each existing plan.
2. Measure forward, dX, dW, and joint backward, including shared preparation.
3. Fit/rank from matched observations, retaining held-out shapes and devices.
4. Integrate bounded first-hit confirmation and existing revision-safe caches.
5. Qualify eager, compile, graph replay, gradients, and memory behavior.
6. Add hybrid or NVFP4 implementations only where evidence justifies them.

## Preliminary evidence, 2026-10-23

RTX 5090, Torch 2.13.0+cu132. The independent reference is FP32 GEMM on
identical BF16 inputs. Event timing uses 15 samples of 20 graph replays;
changed-input replay must equal eager. These are single-process measurements,
not exhaustive winners or a trained heuristic.

At M,N,K = 2048,768,1536:

| Schedule | Forward microseconds | Normalized RMS error |
|---|---:|---:|
| Fused scalar default | 530.71 | 0.03760 |
| Fused three-role TMA, FP32 quantization | 149.54 | 0.03760 |
| Materialized default | 18.42 | 0.03760 |

Default-versus-default comparisons are heavily confounded by schedule quality.
The TMA candidate improves fused latency substantially but still loses here.
Do not infer a universal fused/staged crossover from this one shape.

Artifacts: `autotune_reports/preparation_20261023/` (untracked).
The composed first-hit GPU study at `first_hit_2048/` completed six legal
candidates (eight-trial budget) with confirmation of competitive candidates.
Materialized dual preparation won at 18.43 microseconds; separate preparation
measured 18.53. Fused TMA stage counts 1/2/3 measured 174.09/149.51/149.49.
All completed numerical and changed-input graph checks. These timings are
forward only, not backward or training qualification. Thirty-nine CPU tests
cover preparation, hardware features, and existing first-hit behavior.

Run the composed study (rerun the same command to test persistent reuse):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.tune_operand_preparation \
  --output autotune_reports/preparation/new-first-hit --trials 8
```

Reproduction:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.study_operand_preparation \
  --backend fused --schedule tma --shape 2048,768,1536 \
  --output autotune_reports/preparation/new-fused-tma.json
```

Use `--backend materialized` without `--schedule tma` for staging. Output files
are exclusive-create; choose new paths for independent repetitions.

## Backward preparation follow-through

`benchmarks.study_backward_preparation` supports separate dX, dW, and paired
backward, optional fullgraph compilation, independent FP32 references, and
changed-input graph replay. `benchmarks.tune_operand_preparation --phase
backward` compares staged/staged, fused/staged, staged/fused, and fused/fused
through the shared selector. `--phase dx` and `--phase dw` isolate a partial
gradient. Numerical signatures live in `rtx.autotune.preparation_mxfp8` and
include both operand quantizers, orientations, reduction policy, and split
count. These experiments do not change runtime selection defaults.

Paired backward CUDA-graph microseconds, fixed TMA fused schedule versus dual
materialized preparation, FP32 quantization arithmetic, 2026-10-23:

| M,N,K | Both staged | Fused dX | Fused dW | Both fused |
|---|---:|---:|---:|---:|
| 2048,768,1536 | 41.00 | 187.57 | 207.07 | 352.47 |
| 8192,768,1536 | 118.89 | 492.25 | 776.32 | 1149.56 |
| 32768,768,1536 | 533.65 | 1848.41 | 3065.29 | 4397.76 |
| 8192,3072,768 | 208.99 | 960.89 | 797.05 | 1573.85 |

First three rows used GPU0; the wide row used GPU1, both RTX5090s. Each
row's candidates ran serially on its device. These are plan-screening results,
not exhaustive tuned winners. All 16 candidates passed both gradient gates
(NRMS <= 0.06 against original BF16 operands evaluated in FP32), finiteness,
and changed-input graph replay. Staged winners received first-hit confirmation.
Artifacts are `first_hit_bwd_{2048,8192,32768,wide}/` under the study directory.

At 2048,768,1536, staged partial gradients also passed
`torch.compile(fullgraph=True, dynamic=False)` followed by graph capture/replay
on GPU1: dX 18.53 microseconds, dW 22.53 microseconds; NRMS 0.03753 and
0.03770 respectively. Partial timing includes each leg's preparation and must
not be labelled GEMM-only. It does not qualify automatic cold selection inside
the compiled public frontend. Forty CPU tests passed after these additions.

## Phase decomposition and experimental runtime hook

`benchmarks.profile_preparation_phases` directly measures the staged runners'
preparation and GEMM calls, plus their complete sequential execution. All
outputs pass independent FP32-reference gates first. CUDA graphs use eight
unrolled calls, ten replays per sample, and fifteen samples. Isolated phases
have different hot-cache reuse from the full sequence; do not sum them as an
end-to-end prediction. These use default staged schedules, not exhaustive
winners, and are not vendor-backend comparisons.

| M,N,K | Fwd prep | Fwd GEMM | dX prep | dX GEMM | dW prep | dW GEMM | Full sequence |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048,768,1536 | 4.35 | 12.31 | 4.10 | 14.85 | 6.16 | 15.39 | 56.05 |
| 8192,768,1536 | 11.27 | 37.58 | 6.68 | 37.94 | 16.94 | 54.16 | 181.31 |
| 32768,768,1536 | 93.54 | 137.73 | 23.76 | 154.47 | 147.04 | 200.92 | 746.14 |
| 8192,3072,768 | 7.15 | 70.10 | 22.30 | 70.04 | 26.59 | 62.62 | 312.72 |

All values are microseconds; artifacts `phases_*.json`. GPU0 ran the first
two shapes; GPU1 ran the last two. No tensor-utilization or bandwidth-saturation
claim follows from these timings. The local profiler preflight found Nsight
Systems 2025.6.3 and Compute 2026.2.1, but no Nsight Python package.

With `RTX_PREPARATION_SELECTION=1`, dynamic MXFP8 `backend="auto"` plus
`autotune="balanced"` resolves a fused or staged runner at the existing opaque
operator's cold-launch boundary. Explicit configs/backends and autotune-off
remain unchanged. Hot calls use the frontend's stream/shape-bound runner
cache; persistent measurements live under `<autotune_cache_dir>/preparation`.
The current portfolio is deliberately small and the prior is not calibrated.

The opt-in hook passed eager and `torch.compile(fullgraph=True, dynamic=False)`
CUDA tests at 128x128x128: cold selection, finite gradients, repeated hot calls
with unchanged residuals, new-stream persistent reuse, and changed-input graph
replay. This is integration smoke coverage, not production qualification.
External model-prior requests raise explicit errors in this experimental path
until a preparation-specific model is qualified. Broader ragged coverage,
memory qualification, and removal of the opt-in gate remain work.

### Cumulative refinement and component coordinates

The composite adapter now retains every child adapter's schedule axes instead
of exposing only the preparation implementation name. Bounded, policy-filtered
one-coordinate expansion supplies additional candidates for higher budgets.
The first-hit selector accepts cumulative refinement: a 2-to-4 trial increase
evaluates two new candidates, preserves prior evidence and the incumbent, and
another 4-trial request evaluates none. The experimental public frontend honors
`AutotunePolicy(cache_behavior="refine")`, including `forward_trials`, time,
pool size, and confirmation settings. A CUDA test verified 2-to-4 refinement
through the functional API with four unique durable observations.

Refining the existing `32768,768,1536` backward study from 4 to 16 total trials
evaluated 12 additional candidates. The best measured paired backward improved
from 533.65 to **447.59 microseconds**, retaining staged preparation and FP32
quantization arithmetic; the winner uses a dual-stream schedule. This is about
16.1% lower latency, not a precision change or proof that further tuning is
exhausted. The proposal registry label identifies a seed family; expanded
configs can change execution structure, so use the serialized config rather
than the label when classifying these additional trials.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.tune_operand_preparation \
  --phase backward --shape 32768,768,1536 \
  --output autotune_reports/preparation_20261023/first_hit_bwd_32768 \
  --trials 16 --refine
```

### Backward runtime integration

The opt-in runtime selector now handles paired backward, dX-only, and dW-only
requests through their existing launch boundaries. The requested phase is in
both the in-memory selection key and persistent context. Partial requests
benchmark only their requested output and omit alternatives that change only
the unused gradient. The timing/oracle contract (graph unroll, samples, warmup,
and error gate) is also part of persistent identity, preventing accidental
reuse of the differently timed exploratory benchmark observations.

The standalone functional backward initially exposed an integration mismatch:
its ordinary compiled tuple lowering uses two partial output nodes, while
eager execution uses paired backward. In the experimental selection path it
now uses the existing packed-output compiler lowering, preserving one paired
workload without the historical multi-output allocation-aliasing problem.
The dedicated regression test checks that compiled plus eager calls generate
only one paired context's two observations rather than three contexts.

Seven CUDA integration tests passed after this work: forward eager/compiled
first hit and replay; functional budget refinement; module eager/compiled
paired and partial gradients; standalone compiled backward; and a
129x96x160 ragged backward. Ragged input retains the existing scalar fused
fallback when TMA/native-scale seeds are illegal. These are small-shape
integration tests, not broad training or memory qualification. The default
path remains unchanged without `RTX_PREPARATION_SELECTION=1`.

### Transfer identity repair and first prior study

Preparation contract revision 2 removes duplicated shape/device fields from
component semantic tags. The outer exact cache context still contains them,
and composition verifies that components agree. Component kernel revisions,
numeric policy, and measurement settings remain semantic constraints. This
fix enables nearby-shape schedule retrieval and shape-held-out model contracts
without transferring latency observations as exact-cache winners. Tests cover
shape transfer, exact-cache separation, and rejection of changed rounding.
Revision-1 exact keys are not reused by revision-2 runtime selection.

`benchmarks.analyze_preparation_prior` evaluates the existing dependency-free
gradient-boosted model on explicitly allowlisted, pre-execution features. It
rejects mixed numerical/kernel/measurement contracts and uses leave-one-shape-
out folds. The initial dataset has only 28 paired-backward observations across
four shapes on RTX5090. It is not enough for a cross-SKU qualification.

Result: the model and a simple staged-first ordering have identical regret at
1/2/4/8 candidates. Both find the observed winner first on the three
four-candidate shapes. On the 16-candidate long-reduction shape, both have
19.23% top-1 regret and reach the observed best by four candidates. This is
retrospective ranking over recorded candidates, not prospective tuning speedup.
There is **no demonstrated learned lift over the simple ordering**, so model
deployment remains disabled. The prior study does not replace measured runtime
confirmation. Per-SM normalization is a modeling assumption, not evidence of
cross-device generalization.

Raw analysis: `autotune_reports/preparation_20261023/prior_loo_v3.json`.
The source list, train/test counts, random-selection expected regret, and all
fold scores are saved there. Subsequent data should emphasize within-staged
schedule variation and held-out shapes/devices, not just more easy
fused-versus-staged examples.

### Breadth repair and identical-packed NVFP4 GEMM diagnostic

Bounded coordinate expansion now visits coordinates round-robin rather than
letting the first coordinate consume the expansion quota. A regression test
checks coverage across coordinates. Separate 16-trial forward and backward
searches at M,N,K=128,256,256 completed successfully. The observed forward
winner remained materialized (about 6.13 us); backward reached about 10.23 us
with shared-G quad preparation versus its 12.29 us initial candidate. These
are small-shape observations, not proof of a general fused/staged crossover.
Artifacts: `diverse_v2_fwd_128_256_256` and
`diverse_v2_bwd_128_256_256` under the preparation report root.

`benchmarks.compare_nvfp4_packed_gemm` now runs both backends on the **same
Quartet FourOverSix packed operands**. An independent nibble decode, FP32
scale reconstruction, and FP32 matmul provide the reference; quantization
error against the original BF16 tensors is intentionally outside this gate.
All four shapes passed the 0.5% normalized-RMS gate, with actual errors
0.1666–0.1679% for both backends, consistent with BF16 output rounding.

RTX5090, Torch 2.13.0+cu132, fixed RTX default row-major-scale GEMM versus
the isolated Quartet installation's FlashInfer **cuDNN** backend:

| M,N,K | RTX GEMM (us) | FlashInfer/cuDNN GEMM (us) | Scale-layout conversion (us) |
|---|---:|---:|---:|
| 2048,768,1536 | 11.265 | 8.330 | 6.675 |
| 8192,768,1536 | 32.960 | 20.465 | 8.192 |
| 32768,768,1536 | 108.071 | 72.056 | 15.532 |
| 8192,3072,768 | 54.422 | 39.040 | 7.193 |

CUDA-graph timing uses eight calls per graph, ten replays per sample, and
15 samples, after warmup. Compilation and quantization are excluded. Scale
layout conversion is an explicit separate diagnostic, not charged to either
native GEMM; it includes the PyTorch conversion implementation and is not a
lower bound on conversion cost. Raw samples/configs/numerics are in
`autotune_reports/preparation_20261023/packed_gemm_*.json`.

This establishes **fixed-schedule GEMM headroom**, not a tuned-winner gap,
end-to-end speedup, or proof that RTX cannot close it. The next comparison
must tune RTX's packed GEMM with unchanged operands and account for native
scale emission. No numerical policy or production default changes here.

### Execution-plan coverage hardening

Backward first-hit coverage now classifies the actual dX/dW backend pair,
not the registry name of the seed. Coordinate search can change a backend,
so the seed name was not a reliable coverage category. Partial-gradient
selection classifies only the requested leg. Both the runtime selector and
the research tuning driver use the actual-plan classifier.

Partial runtime searches also reject inactive-leg changes during expansion,
before they consume the useful-candidate quota. The bounded examination cap
still applies, so this does not introduce unbounded host search. These are
search-order fixes: arithmetic contracts and exact winner validity are
unchanged, and existing verified winners remain usable.

Validation after these changes: 27 preparation/first-hit CPU tests passed,
and all seven preparation CUDA integration tests passed on the local RTX5090
(eager, compile, replay, refinement, paired/partial gradients, and ragged
backward). The opt-in release boundary remains unchanged.

### Rejected analytical prior

The expanded retrospective study contains 44 paired-backward observations
over five shapes, including 128x256x256. It now compares an explicit work
proxy (BF16 elements prepared plus packed workspace round trips, with CTA
underfill and L2 pressure as tie-breakers) alongside staged-first and GBDT.
The proxy is available as `reuse_prior` for research, **not runtime selection**.

It fails the small-shape ranking: top-1 regret is 380.34% versus 20.11% for
staged-first, and the observed winner appears only within its first eight
candidates. On the four larger shapes it adds no benefit over staged-first.
The GBDT also adds no lift: on the small shape it finds the observed winner
within four candidates, versus two for staged-first. These are retrospective
candidate rankings, not a replay of the coverage-reserving orchestration.

Raw results: `autotune_reports/preparation_20261023/prior_reuse_v4.json`.
The attempted runtime wiring was removed after this negative result. This
is evidence against deploying a traffic-only proxy, not against all hardware-
aware models. Missing costs include the schedule-dependent quantization and
GEMM throughput that a logical byte count does not represent. A qualified
device/shape-aware runtime prior remains unfinished.

### Four-shape forward search and hot-loop memory checks

Completed fresh 16-trial forward searches on each requested shape (64 total,
zero failed trials). Each candidate uses the same original-BF16 source,
FP32 quantization math/amax, infinity scale rounding, and independent FP32
reference. Each isolated trial also checks replay with changed inputs.
The best observed medians, in microseconds, were:

| M,N,K | Initial staged | Best observed | GPU |
|---|---:|---:|---|
| 2048,768,1536 | 18.434 | 18.430 | local 5090 #0 |
| 8192,768,1536 | 49.426 | 49.426 | local 5090 #0 |
| 32768,768,1536 | 213.142 | 212.194 | local 5090 #1 |
| 8192,3072,768 | 77.081 | 77.081 | local 5090 #1 |

All selected plans were staged. Sub-percent differences are not a claimed
speedup; this budget found no material forward improvement. These are 16
trials from a 54-candidate bounded pool, not exhaustive tuning. Raw configs,
logs, numerical metrics, confirmations, and residuals are in
`autotune_reports/preparation_20261023/tuned_fwd_v2_*`.

The runtime CUDA suite now also checks eager and fullgraph-compiled hot
training for live allocation retention: ten warmup iterations followed by
three groups of 25 forward/backward iterations, gradients released between
steps, synchronized live-allocation measurements, and unchanged tuning
observations. This tests fixed-shape hot-path retention, not arbitrary-shape
cache capacity or cold-tuning peak memory.

### Measured transfer and relative-target study

Nearby-context retrieval now requires symmetric agreement on semantic tags,
cache regime, environment, and recorded Torch/CUDA/CuTe versions. Previously,
extra source-side tags could slip through a one-way comparison. It deduplicates
configurations before consuming the retrieval budget. Shape distance remains
primary; equal-distance sources are ordered by available SM-count, L2-capacity,
and bus-width similarity, with missing capacities treated explicitly. No SKU
name or UUID is a performance predictor. Missing capacity fields are not
invented. Every transferred schedule is still revalidated and measured on the
target; source timings never become target cache hits.

An additional 44-observation, leave-one-shape-out experiment trains on latency
relative to each **training** context's best observation, avoiding held-out
normalization leakage. It does not improve the current GBDT: at 32768x768x1536,
top-1 regret rises to 29.98% versus 19.23% for staged-first, and the best observed
candidate is reached within eight rather than four choices. At 128x256x256,
top-four regret remains 20.11%, whereas staged-first finds the observed winner
within two. The three four-candidate shapes remain easy for both approaches.

The relative-target model is not deployed. Raw results are in
`autotune_reports/preparation_20261023/prior_relative_v5.json`. This small,
uneven dataset still cannot justify a generalizable learned prior. Additional
schedule-diverse measurements, rather than more model variants on these same
44 observations, are the next requirement.

### Active-schedule identity repair (preparation revision 3)

Two 32-trial backward searches with a wider, reproducibly shuffled staged
expansion completed at 2048x768x1536 and 8192x768x1536. All baseline plan
families were retained. The research driver now exposes expansion count,
examination cap, refinement family, and portfolio seed, and writes immutable
invocation manifests. Raw results are under `wide_staged_bwd_*`.

These searches revealed that changing inactive fused fields inside a
decomposed leg consumed distinct trial IDs despite executing the same code.
Revision 3 now keys MXFP8 preparation candidates by their active backend
fields: decomposed legs exclude fused schedules, fused legs exclude unused
decomposed GEMM/quantizer schedules, and partial-gradient identities exclude
the unused leg. Full configurations remain serialized for reproducibility.
The revision change isolates old exact caches. This removes known no-op
neighbors rather than claiming different active schedules are equivalent.

The offline study also now offers `--features schedule`, an allowlist of
pre-execution tile, pipeline, quantizer, and launch choices, excluding inactive
backend fields and measured compiler resources. The preliminary 108-row
`prior_schedule_v6.json` remains unqualified: legacy inactive-field duplicates
still overrepresent some schedules, and pooled shape metrics need separate
device/context evaluation before model-selection claims. Do not promote this
artifact. These issues must be corrected before interpreting apparent lift.

Post-repair validation: 35 CPU tests and nine runtime CUDA tests passed,
including eager/fullgraph execution and fixed-shape allocation stability.

### Context-separated evaluation and fresh revision-3 evidence

Offline schema 2 consolidates equivalent active schedules within each exact
context using median latency, not the fastest repeat. It retains separate
devices, revisions, shapes, and regimes. Leave-one-shape-out training excludes
all contexts of the held-out shape; each held-out context gets its own regret
calculation. In the legacy dataset this reduces 108 rows to 78 schedules;
30 rows were repeated active configurations. The corrected artifact is
`prior_context_dedup_v7.json`. It remains retrospective and unqualified.

Fresh revision-3 32-trial searches completed on both local 5090s at
2048x768x1536 and 8192x768x1536 (`active_staged_bwd_*`), with no failures.
The latter's best observed latency was 112.682 us against a 114.685 us initial
candidate; repeat measurements are needed before claiming a stable speedup.

The runtime suite now has ten passing CUDA tests, including a nearby-shape
transfer check: the second shape creates new measured observations rather
than inheriting the first shape's timings, and subsequent calls to either
shape reuse their exact cache. CPU preparation/first-hit tests: 36 passed.

Cross-SKU follow-through has started in isolated directories, using existing
Torch 2.13.0+cu132 environments without editing remote working checkouts:
5060 laptop `/home/achiya/rtx-preparation-20261023.qijAAZ` and 5070 laptop
`/home/achiya/rtx-preparation-20261023.yaL5a3`. Each runs the same 16-trial
2048x768x1536 backward search with portfolio seed 20261023. Completion and
results must be checked before treating these as qualification evidence.

### First completed cross-SKU matched-schedule comparison

Both laptop searches completed: 16 trials each at 2048x768x1536, with no
failed trials. Copies of all residuals, configs, trial logs, and results are
under `remote_5060_bwd_2048` and `remote_5070_bwd_2048` in the preparation
report root. Source artifact paths inside those records still identify the
remote directories. Remote working checkouts and environments were unchanged.

`benchmarks.compare_preparation_transfer` compares identical active candidate
IDs only, requires matched semantics/toolchains/shapes, and computes target
regret using target measurements. It reports both all-plan and staged-only
rankings, with a fixed 2% near-tie threshold (not a confidence interval).
The three-device artifact is `transfer_three_skus_v2.json`.

For the 16 candidates common to the 5090 and both laptops (13 staged), source
ranking transferred as follows. Percentages are excess latency above the
target's best **within the matched set**, not the full possible search space:

| Source → target | First choice regret | Best within first four |
|---|---:|---:|
| 5090 → 5060 laptop | 0.83% | 0.07% |
| 5090 → 5070 laptop | 9.55% | 0.98% |
| 5060 laptop → 5090 | 0.18% | 0.03% |
| 5060 laptop → 5070 laptop | 0% | 0% |
| 5070 laptop → 5090 | 0.18% | 0.18% |
| 5070 laptop → 5060 laptop | 0% | 0% |

This supports transferring **candidate portfolios with target-side measurement**,
not transplanting a single desktop winner or its latency. One shape, a small
matched set, and laptop clock/thermal variability do not establish general
cross-SKU performance. The 9.55% desktop-to-5070 difference merits an interleaved
repeat before attributing it to hardware scheduling. All model artifacts stay
unqualified. Transfer-study plus preparation/first-hit tests: 38 passed.

### Transfer repeat: laptop timing instability and portfolio retrieval

The 5070 repeat used six fresh processes in A/B/B/A/A/B order, where A is
the 5090-selected schedule and B the laptop-selected schedule. Both passed
the numerical and changed-input replay checks. A timings were 113.853,
488.960, and 112.794 us; B timings were 112.030, 505.408, and 117.642 us.
The outliers affect both implementations. This does **not** establish a
repeatable 9.55% scheduling difference, and the original one-pass transfer
ranking must not be interpreted as a stable cross-SKU model target.
No cause (power, thermal, contention, or otherwise) has been established.
Artifacts: `remote_5070_transfer_repeat/{a,b}{1,2,3}.json`.

Preparation retrieval now takes up to four distinct schedules per compatible
source context, interleaving nearby source contexts under the existing global
eight-candidate retrieval cap. The generic first-hit API retains its previous
one-per-context default. Source latency orders proposals only; target-side
legality, numerical checks, measurement, and confirmation remain mandatory.
This is bounded portfolio transfer, not a learned latency predictor. It does
not increase the user trial cap or invalidate already verified exact winners.

Validation after the change: 49 CPU/CUDA tests passed, covering deduplication,
source interleaving, caps, semantic isolation, compile/replay, target-shape
remeasurement, refinement, and fixed-shape live allocations.

### Timing eligibility is separate from numerical feasibility

Preparation runtime measurements now attach the shared `runtime_timing_metadata`
quality result. Measurement-contract revision 2 isolates older unannotated
runtime records. Explicitly nonstationary outcomes remain successful numerical
observations but cannot become the search incumbent, an exact first-hit cache
winner, or a retrieved latency-ranked proposal. Confirmation recomputes quality
over combined samples and preserves a negative quality flag from any repeat.
Offline preparation model/transfer studies also exclude explicitly unstable
successes. Historical unannotated generic outcomes remain compatible; they are
not retroactively claimed stable.

Configurations whose prior observations were only timing-ineligible can be
remeasured in a later session, once per new session and within its budget.
Cumulative refinement still counts previously spent trials; increasing effort
does not erase earlier work. Numerical failures retain their existing handling.
If no timing-eligible candidate exists, selection fails rather than inventing
a winner. This detects within-sample/confirmation instability, not arbitrary
clock shifts between independently stable runs.

Validation: 76 CPU tests (plus two subtests) and all ten preparation CUDA
tests passed after the core eligibility and confirmation changes.

### Packed NVFP4 schedule search

`benchmarks.tune_nvfp4_packed_diagnostic` now uses the shared fully-prequantized
NVFP4 adapter and first-hit orchestrator to tune the identical-packed-input
diagnostic. Trials are isolated subprocesses, compare against the independent
decoded FP32 oracle, record timing-quality metadata, and retain the vendor
GEMM measurement. The initial scope is row-major scales and default global L2
fetch settings; layout conversion stays separately timed.

The first 24-trial searches completed at 2048x768x1536 and 32768x768x1536.
They exposed a reachability problem: K=64 geometry needs none/32-byte shared-
memory swizzles, so changing geometry alone rejected those configurations.
The driver now supplies coupled geometry/swizzle/stage seeds and reserves
coverage across legal tile/warp layouts. A regression test verifies that legal
K=64 seeds are reachable. The two searches are refining to a cumulative 48
trials; completion must be checked before claiming their final result.
Artifacts: `nvfp4_packed_tuned_2048` and `nvfp4_packed_tuned_32768` under the
preparation report root, including immutable invocation and selection files.
The first small-shape selection measured 10.778 us versus 11.264 us initially;
this does not yet close the vendor gap or qualify a production default change.

### Native-scale diagnostic corrects the backend-gap interpretation

Both row-major searches reached 48 cumulative trials with zero numerical
failures, covering seven tile geometries. Their selected times remained
10.778 us and 99.607 us for the small and large shapes. Coupled geometry,
swizzle, and stage proposals are now available in the shared NVFP4 GEMM axes,
not only the diagnostic driver. The original proposals remain available;
quarantines and legality checks are unchanged. The related NVFP4/composable
test run passed 90 tests (38 subtests), with one skip.

An important diagnostic correction: RTX already has a native `mma128` scale
path used by dynamic kernels. A bit-preserving scale-layout conversion now
lets the fixed-packed comparison exercise it. The independent FP32 oracle
still decodes the original row-major scales; a CPU test verifies every native
scale offset over multiple row/K tiles. No FP4 value or FP32 outer scale is
changed. Fresh measurements were:

| M,N,K | RTX native-scale GEMM | FlashInfer/cuDNN GEMM |
|---|---:|---:|
| 2048,768,1536 | 8.262 us | 8.372 us |
| 32768,768,1536 | 78.098 us | 68.658 us |

All outputs passed the same decoded-FP32 gate. The small case is effectively
parity in this run, not a statistically established win. The large case remains
about 13.7% slower. Thus the earlier row-major result cannot be presented as a
general RTX GEMM deficit: native scale layout/TMA movement removes much of it.
Both conversion stages remain measured separately and excluded from GEMM;
native scale emission by a producer is required to realize this advantage
without those conversions in an end-to-end workload.

Artifacts: `packed_native_2048.json`, `packed_native_32768.json`. Sixteen-trial
native-layout refinements are running under `nvfp4_native_tuned_*`; their
results are not yet claimed. The diagnostic driver now uses a distinct
`nvfp4_packed_diagnostic` family so native-layout research results cannot be
mistaken for promotable TorchAO fully-packed frontend winners.

The native refinements subsequently completed: 16 trials per shape, no
numerical failures. Selected medians were **7.959 us** at 2048x768x1536 and
**71.931 us** at 32768x768x1536. Both use a 128x128x128 tile, 2x2 MMA atom
layout, native scales via TMA, and 64-byte operand swizzles; the small winner
has two stages and the large winner three. These remain packed-GEMM-only
measurements, not full linear forward/backward speedups or promoted defaults.

### Native emission: matched-policy preparation plus GEMM

`benchmarks/compare_nvfp4_native_preparation.py` now checks actual delayed
quantizer emission, without a layout-conversion pass. Both paths use the same
FP32 power-of-two outer-scale policy, fixed 16-entry input-amax history,
amax block scaling, quantizer schedule, and 128x128x128 / 2x2 GEMM schedule.
Only the scale layout/transport differs. The largest shape uses three stages;
the others use two. This is a fixed-schedule diagnostic, **not** a comparison
of fully autotuned winners or stateful training performance.

| M,N,K | Row-major prep / GEMM / full (us) | Native prep / GEMM / full (us) |
|---|---:|---:|
| 2048,768,1536 | 10.45 / 16.85 / 28.44 | 14.92 / 8.19 / 22.94 |
| 8192,768,1536 | 22.86 / 50.25 / 73.84 | 34.33 / 23.55 / 57.46 |
| 32768,768,1536 | 114.85 / 160.81 / 286.52 | 135.65 / 79.30 / 222.01 |
| 8192,3072,768 | 16.48 / 71.00 / 86.54 | 22.61 / 44.17 / 66.20 |

All four passed bit-exact packed-value, logical-scale, outer-factor, and
BF16-output equivalence. Packed GEMM outputs also passed an independent
FP32 decoded-operand reference with a 0.005 NRMS gate. The subsequent small
confirmation additionally checks next-history equality and reports error
against original BF16 operands; it measured 26.75 versus 22.63 us on the
second local 5090. Every recorded timing passed the shared stationarity gate.
Phase timings are independently measured and are not additive.

Conclusion: native emission makes preparation slower in this schedule, but
its GEMM savings still reduce full-path latency by roughly 19–24% in the
first four runs. This is evidence to keep layout/transport coupled in
end-to-end autotuning, not to force native layout unconditionally. It also
separates a genuine schedule/layout effect from FP4-backward precision
tradeoffs. No public defaults or backward numerical policies were changed.

Artifacts: `autotune_reports/preparation_20261023/native_delayed_*.json`.

Autotuner reachability audit: the dynamic/delayed adapters already expose
coupled `scale_transport` updates for quantizer emission and GEMM transport.
However, full implementation anchors carried default block-policy fields,
making those seeds illegal for non-amax callers. Adapter anchor updates now
preserve the current block-scale selection and outer-scale mode while taking
the anchor's schedule. Explicit non-anchor numerical coordinates retain their
existing behavior; this is not a change to the regional numerical search API.
Twelve policy combinations (four block policies by three outer modes) verify
that both layout basins remain legal and preserve these policies. No kernel
semantics or serialized winner layout changed, so existing measured winners
remain valid rather than being invalidated merely for a proposal fix.

Delayed-policy identity is now owned by `make_nvfp4_delayed_adapter` through
its `scale_policy` argument, rather than requiring every caller to construct
the correct metadata. Runtime tuning and dataset construction pass the
evaluator's actual policy. Legacy nondefault policy tags produce the same
context identity; conflicting explicit policy/tag declarations fail. Default
policy contexts retain their historical identity. CPU regression coverage
checks all 24 combinations of three outer-scale modes, four history lengths,
and two history algorithms. An arbitrary external evaluator must still honor
its declared policy; the adapter cannot inspect a callable's GPU semantics.

### Compiled public-API qualification on decoder shapes

`benchmarks/qualify_preparation_runtime.py` exercises `MXFP8Linear` with
fullgraph compilation, eight first-hit trials per forward/backward context,
FP32 output/gradient references, and hot CUDA graph replay. Observations must
remain byte-identical during hot replay. The initial harness retained its
cold autograd graph across side-stream capture; that harness lifetime error
was fixed by releasing the cold outputs before capture, without a PyTorch
stream-override workaround.

Completed cases on local 5090s:

| M,N,K | Gradients requested | Cold seconds | Hot forward + requested backward |
|---|---|---:|---:|
| 2048,768,1536 | dX,dW | 9.04 | 56.59 us |
| 8192,768,1536 | dX | 8.90 | 93.51 us |
| 8192,3072,768 | dX,dW | 15.58 | 508.82 us |
| 32768,768,1536 | dW | 30.23 | 800.60 us |

All 64 candidate trials passed numerical gates; output/gradient NRMS was
0.0375–0.0378 against original-BF16 FP32 matmuls. These are not performance
qualification wins: the long case exposed a timing-stability failure in
selection. Fast forward candidates (~226 us) were ineligible because their
individual sampling windows drifted; the cache selected a stable ~448 us
candidate instead. Hot timings are stationary, but that does not establish
the selected plan is good. The wide paired measurement also must not be
inferred from adding independently timed components.

The runtime measurement contract is now revision 3 with ten graph replays
before sampling (in addition to kernel warmup); all ten existing preparation
CUDA tests and 39 preparation/first-hit CPU tests pass. A fresh long-case
repeat still selected ~449 us and ran at 800.74 us: replay warmup alone did
**not** resolve the selection problem. The initial candidate samples rise
from ~222 to ~237 us inside a window, then repeat that pattern on confirmation.
Cause remains unproven. Keep the stability gates unchanged and investigate
time-based workload stabilization before promoting this selector.

Artifacts: `runtime_2048_compiled_v2`, `runtime_8192_dx_compiled_v2`,
`runtime_wide_compiled`, `runtime_32768_dw_compiled`, and
`runtime_32768_dw_compiled_v3` beneath the preparation report root.

### Time-based stabilization resolves the reproduced selection failure

The isolated `study_preparation_stabilization.py` holds the staged runner,
shape, operands, reference gate, graph unroll, and samples fixed, varying
only initial graph replay count. Forty-five samples are evaluated as five
consecutive nine-sample windows. Ten warmup replays produced window medians
224.94, 236.23, 237.07, 237.57, 237.62 us; the first failed stationarity.
With 128 warmup replays, all windows passed at 237.79–238.15 us. This supports
insufficient workload stabilization, but does not establish a specific clock,
power, thermal, or cache mechanism.

Runtime measurement revision 4 therefore pilots two graph replays and chooses
a bounded warmup count targeting 200 ms (minimum 10, maximum 4096 replays).
Pilot time is not a ranking sample. Actual pilot duration and replay count are
recorded in trial metadata; thresholds and numerical policy remain unchanged.
The new contract partitions cached measurements from revisions 2/3.

A fresh compiled 32768x768x1536 forward+dW run with eight trials per context
then had **all 16 trials stationary**, retained the fast ~236.73 us forward
candidate, and measured **583.70 us** hot forward+dW versus **800.74 us** in
revision 3 (27.1% lower latency). NRMS remained exactly 0.037518 forward and
0.037741 dW. Cold time was 32.09 seconds versus 32.88; these timings are one
controlled reproduction, not a multi-SKU generalization. Fifty CPU/CUDA
preparation/first-hit tests passed, including warmup bounds, cache/refinement,
compiled partial/paired gradients, graph replay and stable live allocations.

Artifacts: `stabilization_10.json`, `stabilization_128.json`, and
`runtime_32768_dw_compiled_v4` under the same report root.

Follow-up revision-4 public-API checks covered paired gradients at
128x256x256 and 8192x3072x768. Both passed independent numerical checks and
cached hot replay. Hot forward+backward measured 12.80 and 309.12 us,
respectively; cold times were 11.90 and 18.22 seconds. The wider case had all
16 stationary trials (previous hot result 508.82 us); the small case still
had one unstable forward candidate, correctly excluded. Thus time-based
warmup is not a claim that noise is eliminated on every shape/device.

Forward portfolio seeding now tries the legal policy-matched two-stage TMA
fused implementation before its much slower scalar seed. Three-stage,
one-stage and scalar remain candidates; materialized preparation remains the
first baseline, and both families require target measurement. This seed order
is supported by the four decoder probes plus the small probe, not by a claimed
universal latency model. Adapter legality and policy checks precede selection.
An initial CPU test incorrectly assumed ragged shapes necessarily reject TMA;
the actual implementation supports this logical-padding case, so the test
retains only the valid scalar-fallback invariant.

A fresh compiled 2048x768x1536 two-trial-per-context run measured staged
forward 16.11 us and TMA fused forward 142.67 us (rather than spending the
second trial on scalar). Both forward and paired-backward contexts passed
stationarity and numerical gates, hot replay did not retune, and complete hot
forward+backward was 53.93 us. Cold time was 4.88 seconds. Two trials cannot
cover all four paired-backward combinations; higher budgets retain that
coverage. Artifacts: `runtime_small_compiled_v4`, `runtime_wide_compiled_v4`,
`runtime_2048_prior_two_trials`.

Fresh-process persistent-cache checks rebuilt compiled public modules for
2048x768x1536 paired backward and 32768x768x1536 dW-only. Both passed the
same numerical gates with byte-identical preparation journals across the
first invocation and hot replay. First invocation (including process-local
compiler/runner setup, not search) took 2.21 and 1.76 seconds. Artifacts:
`runtime_2048_persistent_hit` and `runtime_32768_persistent_hit`. The driver
now supports `--cache-dir` plus `--require-cache-hit` for reproduction.

The selector has a per-layer opt-in through
`AutotunePolicy(preparation_selection=True)`, documented in `autotuning.md`.
Default behavior remains unchanged, and the legacy environment switch remains
supported. Compiled/eager paired and partial gradient tests now exercise the
policy-only path with the environment switch removed. The option remains
experimental; adding a public switch does not promote the research priors or
claim completion of the separate NVFP4 backward work.

Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.compare_nvfp4_native_preparation \
  --shape 32768,768,1536 --stages 3 --output /tmp/native-delayed-new.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \
  tests/test_preparation_runtime_cuda.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.profile_preparation_phases \
  --shape 2048,768,1536 --output autotune_reports/preparation/new-phases.json
```
