# MXFP8 BalancedMoE speed work — RTX 5090

**Result: the large routed-expert path beats compiled BF16 by 1.24× for
forward+backward, but the complete layer is only 1.07× faster. The 1.2×
whole-layer target is not met, and smaller experts still need work.**

Date: 2026-11-15. MXFP8 only. See [the API guide](balanced_moe.md).

## Final-source result

Shape: N=2000, D=768, E=48, K=6, H=256, one shared H=512 SwiGLU.
All values are median microseconds; speedup is BF16 / MXFP8.

| Scope | BF16 | MXFP8 | Speedup |
|---|---:|---:|---:|
| Routed experts, forward | 179.94 | 116.86 | 1.54× |
| Routed experts, forward+backward | 591.09 | 477.97 | 1.24× |
| Complete layer, forward | 714.81 | 654.20 | 1.09× |
| Complete layer, forward+backward | 1377.90 | 1285.09 | 1.07× |

Both complete-layer variants use the **same selected CuTe router**, identical
weights/input, one shared expert, and gradients for input and every parameter.
The reference uses compiled batched BF16, not Python per-expert loops.
Native experts/shared MLP recompute; ordinary BF16 autograd saves intermediates.
This is not a matched-memory checkpoint comparison. Optimizer steps are excluded.

Timings use alternating AB/BA order, three rounds, nine samples per measurement,
and warmed CUDA-graph replay. Compilation, tuning, warmup and capture are
excluded. These are graph-replay GPU timings, not an eager CPU-overhead promise.
Automatic whole-expert Inductor replay is still not qualified; the registered
expert op retains its immutable CPU plan token and capture-unsafe tag.

## What changed

1. **Quantize once per expert operand/orientation.** A fused preparation
   portfolio emits packed X/weights and independent leading-axis operands.
   Output tiles reuse these instead of repeating dynamic quantization.
2. **Coalesced logical-transpose preparation and warp-owned gather.** BF16
   transposes are never materialized. Dummy rows remain zero and scales never
   cross expert boundaries.
3. **Asynchronous, coalesced BF16 output epilogues.** Extend the shared GEMM's
   TMA descriptor with the expert coordinate, then use stmatrix→SMEM→TMA for
   selected linear outputs. Without the expert coordinate, batched stores
   would address the wrong matrix. Optional physical capacity padding enables
   full-tile stores without changing logical routes.
4. **Shared autotuner coverage.** Independent schedules for all seven expert
   stages, explicit transport/epilogue choices, stage-aware local proposals
   within the existing UCB orchestration, and numerical checks after fast replay.
   Extending eight trials to sixteen preserves a compatible progressive
   checkpoint; on-chip exploration opens after eight recorded trials.
5. **Prepared execution is the initial default**, not a per-SKU frozen winner.
   Fused-quantization and on-chip families remain selectable.

The large-case winner uses row-major scales, ordinary per-operand quantizer
grids, separate recomputation/preparation, TMA operands/output epilogues, and
physical expert capacity 256 for logical capacity 250. FP32 activation/gating
and mixture reductions, BF16 parameters/I/O, MXFP8 hidden storage, and full
recomputation remain intact. No extra forward hidden state is saved.

## Attribution and negative results

The original large-case expert path took roughly 2637 us. Prepared operands
plus transpose/gather improvements brought it below 700 us. Coalesced output
stores were the next major gain: the frozen 678 us expert schedule fell to
about 591 us with only weight-gradient TMA epilogues; enabling aligned stores
for the other linear outputs and further tuning reached approximately 478 us.
Initial-to-final numbers include multiple changes, not a single controlled A/B.

A warmed diagnostic trace of the retained large-case path attributed roughly
184 us to preparation, 101 us to GLU/recomputation, 125 us to packed GEMMs,
72 us to movement, and 7 us to other work. These are profiler attribution
figures, **not** the headline event timings. No Nsight counters were available;
there is no claim of measured HBM saturation, spills or tensor-core saturation.

Plausible alternatives are retained as choices, not forced as improvements:

- Joint recomputation emits forward hidden and both derivative orientations
  from one projection, preserving exact forward hidden rounding. It passed
  independent gradients but did not win the measured whole expert objective.
- Native128 scale layout/TMA did not beat the selected row-major schedule.
- Cooperative cp.async operands remove the dedicated producer warp, but were
  slightly slower than the frozen TMA path in the large-case A/B.
- Batch-aware quantizer grids emitted **byte-identical data/scales**, but
  expert-capped and whole-portfolio-capped schedules took 498/518 us versus
  their paired ordinary-grid baselines of 477/477 us. Fewer CTAs alone were
  not a performance improvement.

## Smaller designs are not yet BF16 replacements

N=2000, one shared expert in each case. These are the inherited-winner,
48-trial follow-ups with asynchronous-output choices, source `f5560710…`;
they precede the grid-policy-only addition. Times are microseconds.

| D / E / K / H | Expert BF16 fwd+bwd | Expert MXFP8 fwd+bwd | Whole BF16 fwd+bwd | Whole MXFP8 fwd+bwd |
|---|---:|---:|---:|---:|
| 384 / 24 / 6 / 128 | 166.36 | 206.26 | 654.18 | 735.42 |
| 384 / 32 / 4 / 128 | 117.03 | 151.57 | 720.99 | 800.19 |
| 512 / 64 / 8 / 128 | 290.10 | 346.29 | 1025.53 | 1137.17 |

The four initial designs at N=128/512/2000 were exercised during initial
qualification, but this does **not** mean every context received the final
deep search. A large-shape win is not evidence of a universal speedup.

## Correctness and reproduction

Final source digest:
`2d986d94926d9d097961ff6455b8121b33bc0c09b2342e1d57a0596c5b98dbe0`.
Final CUDA matrix: **76 passed, 2 skipped, 63 subtests passed**. Coverage includes
independent quantized forward/all-gradient oracles, ragged/odd dimensions,
expert-boundary TMA stores, first-hit fullgraph selection, opcheck, mutable
inputs/weights under graph replay, bounded live workspace, and focused stable
GLU/MLP regressions. The skips concern deliberately misaligned legacy TMA
strides. CPU documentation checks additionally passed, including generated API
freshness and 314 link subtests.

On the final large-case winner, implementation NRMS versus the independent
FP64 quantized operational oracle was output 2.09e-5, dX 8.61e-5, gate/up dW
4.57e-5, down dW 2.16e-5, and routing-weight gradient 1.82e-5. These are **not**
errors against unquantized BF16 or evidence of learning convergence.

Environment: work_5090, RTX 5090 SM120/170 SMs, Torch 2.13.0+cu132, CuTe 4.7.0.
Resident detector services were left untouched and observed idle; they retained
about 13 GiB. This was not an exclusive bare-GPU or cross-SKU qualification.

Raw JSON, append-only trial/measurement journals, diagnostic traces, test logs,
and exact source archives are stored under
`autotune_reports/balanced_moe_mxfp8_20261115/work_5090/moe_speedup/`, backed by
Storage4. The final wide files are `study/prepared-final-current-source-wide.json`
and `moe-final-current-source-tests.log`; smaller cases are
`study/prepared-final-tma-shapes.json`. Earlier source archives and negative
experiments are retained, not overwritten.

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_mxfp8_moe.py tests/test_mxfp8_moe_prepared.py
PYTHONPATH=. python benchmarks/benchmark_balanced_moe.py \
  --case-ids d768_e48_k6_h256 --tokens 2000 \
  --effort extensive --trials 96 --auxiliary-effort extensive \
  --whole-layer --rounds 3 --samples 9 \
  --output autotune_results/moe/study.json
```

For continuation, pass `--config-study` and `--auxiliary-study` with matching
case/token rows to inherit verified configs. Fresh cold search is not promised
to rediscover an inherited deep winner within the same small budget.

## Remaining target

The whole-layer 1.2× target would require about **1148 us**, another 137 us below
the final measured 1285 us. The next work is reducing expert preparation and
movement/full-recompute passes, improving H=128 expert ownership, and hardening
the composed execution path. The common router and shared path dilute the
routed-expert gain; they must remain in end-to-end reporting. Do not change
precision, save hidden state silently, or swap in a slower router baseline to
manufacture the requested speedup.
