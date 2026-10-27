# NVFP4 backward improvement study

Status: bounded improvement and qualification study completed, 2026-10-25.
Stable `NVFP4Linear` continues to use MXFP8 backward.
This study must not silently change its numerical policy or install research
configurations as runtime winners.

## Questions and acceptance

1. How much time and workspace does FP4 preparation consume relative to GEMM?
   Measure dX and dW independently, then confirm the complete compiled path
   with identical forward, original BF16 operands and upstream gradients.
2. Can existing preparation layouts or a structural change reduce that cost?
   Preserve rotation, correction, FP32 outer factors and scale-rounding
   semantics. Validate against an independent reference before timing.
3. Is mixed backward useful? Explicitly compare FP4/FP4, FP4/MXFP8,
   MXFP8/FP4 and MXFP8/MXFP8 (dX/dW). Do not select numerical policy with a
   performance-only tuner. Packed-source behavior remains a separate policy.
4. Does estimator error average away? Separate single-draw error, mean error
   and sample variance on Gaussian, sparse and outlier inputs, with small and
   long reductions. Finite Monte Carlo means do not prove unbiasedness.
5. Does a promising policy survive short matched training? Use identical
   initialization, data, forward policy, optimizer precision and step budget.
   Exercise both propagated activation gradients and weight gradients.

## Execution

- RTX 5070 laptop: exclusive preparation/performance worker initially.
- RTX 5060 laptop: numerical study and training when SSH is reachable.
- Use isolated remote snapshots; preserve other projects and services.
- Record software versions, device/power observations, source identity,
  shapes, seeds, exact commands and raw timing/numerical samples.
- Keep RNG generation charged in complete-path measurements. In-kernel RNG
  is a hypothesis until its benefit and checkpoint/replay contract are proven.
- Run focused tests per change, then one combined relevant regression check;
  do not repeatedly run the whole repository suite.

## Reporting boundary

Historical 5090 results identify preparation as a likely target, not a laptop
baseline. Compare baseline/candidate on the same device, with no concurrent GPU
job. Isolated phase times are diagnostic and are not assumed additive.
Report failed or inconclusive timing windows rather than promoting their minima.
No speedup claim is accepted by loosening numerical gates or excluding RNG,
quantization, required gradients or preparation from only one side.

## Initial integration evidence

Independent `dx_policy` / `dw_policy` selection is implemented in the research
API, with explicit MXFP8 configuration control. Stable `NVFP4Linear` is unchanged.
On RTX 5070 Laptop / Torch 2.13.0+cu132, ten focused CUDA cases passed:
fullgraph compilation, checkpoint replay, RNG restoration, missing gradients,
noncontiguous upstream gradients and exact phase equivalence with the existing
FP4 and MXFP8 operations. This is integration evidence, not a convergence result.
Artifacts: `autotune_reports/ms_eden_mixed_backward_20261025/`.

The first preparation measurements and their stationarity limitations are in
[the 5070 preparation diagnostic](ms_eden_preparation_20261025_5070.md).
Numerical and matched-training results are in the
[5060 report](nvfp4_backward_numerics_20261025.md). Complete compiled execution,
stable paired-MXFP8 comparison, RNG and allocation evidence are in the
[5070 public-path report](nvfp4_mixed_backward_comparison_20261025.md).

## Completion audit and conclusion

| Requirement | Evidence / result |
|---|---|
| Preparation versus GEMM attribution | Current 5070 phase measurements and a retained Nsight Systems trace identify transposed producers as the dominant dW cost; no unavailable hardware counters are invented. |
| Preparation improvement | Opt-in `coalesced_wide`, exposed to research tuning; independent producer references, packed-source cases and 86 focused CUDA tests pass. Paired isolated-gradient gains are 1.02–1.10x. |
| Complete-path consequence | Public fullgraph CUDA replay includes forward, both gradients and RNG. Wide preparation improves its FP4 predecessor by 1.0805x within the measured session, but full FP4 and tested mixed backward remain slower than stable paired MXFP8 at the representative shape. |
| RNG investigation | Materialized uniform storage and diagnostic RNG cost recorded; fresh randomness, state restoration and checkpoint replay checked. In-kernel RNG was not introduced without a proven graph/checkpoint-safe state contract. |
| Mixed policy API | Explicit dX/dW precision, separate FP4 schedules and an MXFP8 config; original-BF16-source restriction enforced. Stable frontend defaults unchanged. |
| Bias and noise | Nine shape/distribution cases plus a 128-repetition sparse check separate mean error and variance. Sparse residual around 1.55% persists; no unbiasedness claim. |
| Matched training | Four policies completed 100-step decoder runs with identical initial weights/data/forward hashes. Corrected exact-mode runs are distinguished from retained power2 runs. This is one-seed short-training evidence, not convergence certification. |
| Compiler and memory contracts | Final experimental-wrapper 14-case CUDA regression passes after the exact-mode fix. Public-path checks cover changed X/W/dY, fresh RNG, ordinary-call cleanup and graph replay with zero measured live growth. |
| Consolidation | Constructor/function guidance, mode precedence, research restrictions, reproducible CLIs, raw artifacts and source identities documented. Final focused CPU/docs/reference check: 85 tests and 254 subtests pass. |

The exact-mode audit found and fixed an experimental-wrapper forwarding bug;
the stable JIT API's existing `dynamic_config` mode precedence is now stated
explicitly rather than changed. Historical mislabeled artifacts remain intact
with correction manifests. No stable precision default, runtime winner, release
tag or package publication was promoted by this study.

The actionable recommendation is to retain MXFP8 backward in production and
keep the new FP4/mixed paths explicit research options. Further work may target
producer traffic, sparse-estimator bias and multi-seed/longer training; these
results do not establish that FP4 backward is a superior replacement, nor that
the tested schedules exhaust the available search space.
