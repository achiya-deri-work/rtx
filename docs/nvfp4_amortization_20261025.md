# NVFP4 backward amortization goal

Status: completed, with a modest speed improvement and explicit tradeoffs.
Baseline is the qualified **256-CTA** original-BF16-source
MS-EDEN path, not the superseded 64-CTA training seed. See
[the preparation qualification](nvfp4_backward_fix_20261025.md).

## Outcome

- **Keep register sign reuse as an autotunable choice.** It preserves packed
  operands/scales/gradients bit-exactly and improves the tested complete gradient
  paths. The full-model training gain is smaller: **1.63%** in two reversed-order
  repeats on the 5060, not a universal hardware-wide guarantee.
- **Keep fused Philox optional for its measured memory tradeoff, not as a speed
  default.** On the largest projection it reduces peak allocation by 1,637,888
  bytes (1.56 MiB), but slows complete backward by about 5%; total F+B is near
  neutral. Small-shape timings were inconclusive. Persistent capture allocation
  did not decrease.
- **Reject exact integer normalization.** It passed 134,256,140 bitwise checks
  against explicit PTX division plus independent edge checks, yet slowed dX by
  37.8% and dW by 31.7%. Production integration was removed; archived sources and
  results are under `autotune_reports/ms_eden_exact_normalization_20261025_5060/`.
- Stable `rtx.NVFP4Linear` continues using MXFP8 backward. There is no silent
  change to precision, forward-packed source reuse or cross-step caching.

## Speed and numerics

Actual compiled public backward, `(M,N,K)=(4096,2048,512)`, caller RNG included,
paired graph replay with both execution orders:

| Change from 256-CTA baseline | Backward speedup, 95% round-bootstrap interval | F+B speedup |
|---|---:|---:|
| Register signs, explicit RNG | 1.044x [1.041, 1.047] | 1.031x [1.024, 1.041] |
| Register signs, fused Philox | 0.951x [0.947, 0.956] | 0.996x [0.992, 1.000] |

Controlled conditional-gradient checks hold input signs and uniforms identical.
Packed bytes, scales, tensor factors and complete gradients match bit-exactly.
Complete-gradient medians in microseconds (separate measurements from the
public caller-RNG comparator):

| Projection `(M,N,K)` | dX reload/register | dW reload/register |
|---|---:|---:|
| 4096,1536,512 | 137.67 / 132.91 | 174.52 / 164.22 |
| 4096,512,512 | 59.30 / 58.08 | 91.64 / 85.97 |
| 4096,2048,512 | 180.28 / 173.96 | 225.66 / 216.49 |
| 4096,512,1024 | 80.75 / 78.92 | 140.21 / 132.82 |

## Full training confirmation

Same 21.1M decoder, BF16 inputs/outputs/states, FP32 sensitive operations,
exact-FP32 JIT-region forward, original BF16 backward sources, fullgraph/static
compile and CUDA graphs. Each fresh process trains 512 steps / 2,097,152 byte
tokens. Run order: baseline, register, hybrid, hybrid, register, baseline.
Initialization/corpus/held-out/model/optimizer identities match all six runs.

| Policy | Training seconds, two runs | Mean |
|---|---:|---:|
| Full NVFP4, 256 CTAs, reload signs | 11.81961 / 11.71048 | 11.76504 |
| Full NVFP4, 256 CTAs, register signs | 11.53323 / 11.61405 | 11.57364 |
| NVFP4 forward / MXFP8 backward | 9.80081 / 9.83000 | 9.81541 |

This is **1.63% less training time**, reducing the same-session hybrid penalty
from 19.86% to 17.91%. It does **not** establish speed parity. Compilation,
validation and checkpoint time are excluded from training timing; inclusive
clocks remain in the raw JSONL.

All runs completed with finite losses, gradients and parameters. Final held-out
losses were reload 0.99514/0.99591, register 0.99580/0.97244, hybrid
0.99298/0.99397. Fresh-process stochastic training is not bit-exact. These short
runs do not establish a convergence ranking; unchanged kernel arithmetic is
supported by controlled same-draw tests, not by equal training trajectories.

## Autotuning and integration

The existing first-hit tuner now supports a **public-RNG evaluator** with
independent measurement identity. Small budgets probe both RNG modes and both
sign strategies early, regardless of the starting configuration. Explicit-input
benchmarks reject Philox rather than crediting an unexecuted implementation.

The first single-gradient tiny-shape timing attempt failed eligibility for all
four configurations. It was preserved, not promoted. Capturing eight gradients
per replay, normalized per gradient, produced three eligible configurations;
the fourth remained excluded. The four-trial CLI run selected explicit RNG plus
register signs. No stationarity gate was relaxed. Contract:
`public_rng_graph8_samples9_replays16_warm200ms_calibrated_cudaevents_v2`.

Feature revision 3 accounts for packet4's actual finalizer CTA cap and models
logical sign reads; these counts are not measured DRAM traffic. Context identity
includes both feature and measurement revisions. Older feature-revision-2
observations must not be treated as revision-3 structural features unchanged.

Example offline tuning (original BF16 sources, explicit precision policy):

```bash
python -m benchmarks.tune_ms_eden_preparation \
  --shape 4096,2048,512 --phase dw --public-rng --trials 16 \
  --initial-schedule-json '{"producer_threads":128,"producer_blocks":256,"producer_load":"coalesced_wide","producer_order":"reduction_major","finalizer_blocks":224}' \
  --output autotune_reports/ms_eden_public_dw
```

`tools/run_ms_eden_training_ab.sh PYTHON TOKENS OUTPUT_DIR signs` reproduces the
training comparison. Optional `philox` and `combined` experiments exist but are
not recommended speed defaults. Training CLI overrides are stored in run and
checkpoint identities. No global schedule default or model policy changed.

The final focused CPU check passed 73 tests with 5 CUDA cases intentionally
skipped locally (the local 5090s were occupied). Remote GPU evidence includes
31 independent RNG/numerical cases, caller checkpoint/replay tests, 16 previous
finalizer regressions, 8 sign-reuse tests, the paired benchmarks and six actual
training runs. This is focused experimental qualification, not a library-wide
release certification.

## Scope and ownership

- RNG agent: optional Philox uniforms generated in scale finalization, explicit
  small caller-owned seeds, custom-op/fake/compile integration. Preserve the
  existing explicit-uniform API and independent operand streams.
- Preparation agent: arithmetic-preserving sign/index/register reuse and
  geometry, with honest autotuner feature/context contracts.
- Validation agent: independent Philox reference, numerical/replay/checkpoint
  tests and complete public backward comparisons.
- Main agent: exclusive GPU scheduling, profiling, integration review,
  full-model training comparisons and final evidence consolidation.

Only `personal_laptop` (5060) is authorized for this run. Do not access
`laptop_5070` or disturb local AFRL training. Use persistent home-directory
snapshots and copy results locally; `/tmp` was lost during the previous reboot.

## Acceptance

1. Fixed-input explicit-randomness arithmetic changes must match packed/scales
   and complete gradients bit-exactly, or be documented and tested as a distinct
   optional arithmetic choice before use. No silent numerical-policy weakening.
2. Seeded RNG must agree with an independent counter-RNG reference, remain in
   [0,1), separate operand streams, refresh on replay, and restore reproducibly
   under its documented Torch RNG/checkpoint contract. Same-distribution RNG is
   not a promise of the old sample sequence or an unbiased whole gradient.
3. Validate eager, fullgraph compile, CUDA graphs, missing gradients, finite
   zeros/extremes, and stable live allocation on focused cases first.
4. Compare complete public backward with identical source policies/GEMMs and
   include RNG costs. An explicit-uniform benchmark must not rank RNG modes.
5. Measure all four 21M decoder projection shapes, both gradients, alternating
   execution order. Keep raw samples and rejected experiments.
6. Confirm worthwhile candidates in compiled full-model training against the
   256-CTA baseline and NVFP4-forward/MXFP8-backward hybrid. Report remaining
   gap rather than asserting parity from kernel-only improvements.

## Reuse boundaries

dX and dW reduce over different axes, so their Hadamard/quantized dY operands
cannot simply be shared. A joint loader could reduce rereads but would still
need both transforms and scale layouts; its larger working tile can reduce
parallelism. Forward-packed reuse and cross-update caches change numerical
semantics and are outside this goal's default policy. Wider projections or
larger batches can change the crossover, but changing the model is not an
implementation speedup on the fixed training workload.

## Artifacts

Existing baseline artifacts: `autotune_reports/nvfp4_backward_fix_20261025/`.
New artifacts:

- `autotune_reports/nvfp4_amortization_20261025/`: 256-CTA training trace,
  remaining shape measurements, `training-signs/*.jsonl` and launch receipt.
- `autotune_reports/ms_eden_amortization_validation_20261025/`: paired public
  comparisons, memory/error checks and both failed/successful tuner smoke runs.
- `autotune_reports/ms_eden_sign_reuse_20261025_5060/`: producer comparisons/PTX.
- `autotune_reports/ms_eden_exact_normalization_20261025_5060/`: rejected exact
  division prototype, exhaustive proof checks and negative performance results.

Final training snapshot/checkpoints:
`/home/achiya/rtx-amortize-training-20261025.jV6xKa` on `personal_laptop`.
The 5070 was not accessed and local AFRL training was not disturbed.

## Seeded RNG implementation and focused qualification

The optional schedule `rng_mode="philox"` retains the 128 caller-generated
FP32 signs and replaces the two scale-uniform tensors with **one int64[2] key
tensor (16 bytes) per requested gradient**. Each key word is drawn with
`torch.randint(0, 2**32, ..., dtype=torch.int64)` outside the opaque operation,
so AOTAutograd/Inductor and CUDA graphs own the RNG advancement. The kernels
have no hidden counter, mutable generator or host `.item()` extraction.

Philox4x32-10 uses counter `(logical_scale_index//4, operand_stream, 0, 0)`,
key `(seed[0],seed[1])`, streams 0/1 for the two operands and output word
`logical_scale_index%4`. Its top 24 bits yield exact FP32 uniforms on the
24-bit `[0,1)` grid. This keeps the stochastic-rounding probability expression,
but does **not** promise the old sample sequence, RNG consumption, or identical
discretization to every Torch RNG backend. It does not make the entire FP4
gradient unbiased. Key values in `[0,2**32)` are a caller precondition; metadata
validation is non-synchronizing. The implementation bounds logical scale counts
below `2**31` to keep its signed index arithmetic valid.

`philox_packet_size=4` computes one Philox quartet and finalizes four scales
per thread; `1` retains the scalar alternative. The field is autotunable only
in public-RNG contexts and removed from explicit-RNG configuration identities.
The original explicit-uniform API remains available and rejects a misleading
`rng_mode="philox"` request rather than silently ignoring it. The deterministic
`ms_eden_seeded_gradient(..., seed, phase=...)` API directly selects seeded RNG;
its schedule may use the default RNG field because the API input is explicit.
Packed-source seeded backward is explicitly unsupported, not silently decoded.

Initial RTX 5060 Laptop qualification in
`/home/achiya/rtx-rng-fusion-20261025.1IZ9tQ`:

- 24 independent CUDA RNG/SR/gradient tests passed: CPU Philox known-answer
  mapping, both operand streams, scale tails/native layout, zeros and wide
  magnitudes, exact comparison against CPU-generated uniforms, compiled seed
  updates, graph replay, no graph live-allocation growth and higher-order reject.
- 7 caller integration tests passed, including packet1/4 checkpoint RNG restore,
  compiled CUDA-graph fresh draws and RNG-state restoration, and partial-gradient
  RNG consumption. An additional CPU explicit-uniform-contract test was added.
- 16 existing finalizer tests passed after factoring shared scalar-scale logic,
  including independent scale oracles, subnormals and changed-input replay.
- Seeded `opcheck` passed all schema/fake/AOT checks. A negative-test exception
  regex initially mismatched the existing wording; the test was corrected.

These establish focused contracts, **not a speedup**. The full-path comparator
above includes RNG generation, preparation and GEMM and found a memory/speed
tradeoff, not a reason to change the default. Logs are retained under
`autotune_reports/nvfp4_amortization_20261025/rng/`.
