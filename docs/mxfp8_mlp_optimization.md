# Recomputed MLP optimization ledger

The current target is a **1.5× forward+backward speedup** over an aggressively
compiled, reentrant-checkpointed BF16 MLP on the RTX 5070 Ti. Initial target
dimensions are M=512, K=768, H=1536, O=768, with SiLU. **The target was met on
2026-09-12:** three independent compiled comparisons measured 1.5104×,
1.5116×, and 1.5123×. This is a shape/device-specific result, not a blanket
MLP speedup. Do not interpret tuner minima as final compiled comparisons.

The algorithm retains only inputs/weights, launches exactly two forward device
kernels, stores a packed MXFP8 H-width intermediate between them, and recomputes
the projections/activation during backward. Public inputs, outputs and gradients
are BF16. Timing uses warmed CUDA graph replay; compilation is excluded.

## Evidence collected

| Experiment | Whole-operation tuner median | Evidence directory |
|---|---:|---|
| Starting compiled candidate | 323.32 µs | `autotune_reports/mlp/medium_reentrant_comparison.json` |
| Vectorized BF16 operand loads | 256.18 µs | `autotune_results/mlp_goal_gate` |
| Separate A/B quantizer loops | 249.44 µs | `autotune_results/mlp_split_ab` |
| Packed shared-memory quantized stores | 219.81 µs | `autotune_results/mlp_wide_store` |
| Dead operand scratch + warp arrangement | 207.48 µs | `autotune_results/mlp_alias` |
| Down projection quantization retuning | 199.98 µs | `autotune_results/mlp_down_words` |
| Independent cooperative quantizer warps | 188.43 µs | `autotune_results/mlp_extra_quant` |
| Quantization unrolling and proven full-tile bounds | 181.81 µs | `autotune_results/mlp_bounds` |
| Down weights quantized by the first kernel | 158.04 µs | `autotune_results/mlp_producer_quant` |
| Larger gate/up tile using a storage lifetime union | 140.90 µs | `autotune_results/mlp_wide_union` |
| Overlap independent backward branches | 136.07 µs | `autotune_results/mlp_parallel` |
| Scratch swizzle and producer quantizer retuning | 133.95 µs | `autotune_results/mlp_vector_scratch` |
| Quantize-once backward recomputation | 129.64 µs | `autotune_results/mlp_prequant_recompute` |
| Gradient prefetch / coalesced derivative stores | 121.58 µs | `autotune_results/mlp_grad_exchange` |
| Independently retuned forward projection | 118.01 µs | `autotune_results/mlp_independent_gate` |
| Packed hidden reconstruction + SiLU reciprocal policy | 112.90–113.06 µs, compiled repeats | `autotune_reports/mlp/goal_repeat_{1,2,3}.json` |

The original matched compiled checkpointed BF16 measurement was 170.73 µs;
ordinary compiled BF16 was 143.39 µs. A fresh matched comparison is required
after the next optimization round. These are one-shape measurements, not
general performance claims.

The starting Nsight timeline put about 62% of device time in gate/up forward
and recomputation, and 17% in mixed packed-X/BF16-W down projection. The gate/up
kernel issued approximately 39 million warp instructions, including substantial
integer indexing and byte-wide shared stores. It was not tensor-core saturated.
Raw profiles: `autotune_reports/mlp/goal_start_train.nsys-rep` and
`autotune_reports/mlp/goal_glu_detail.ncu-rep`.

## Implemented search dimensions

- `quant_schedule`: combined or separately specialized A/B quantizer loops.
- `quant_store_bits`: 8, 16, 32 or 64 bits, subject to register-vector and
  operand-layout legality. Wide MXFP8 stores explicitly apply the physical
  shared-memory swizzle before emitting the packed store.
- `glu_scratch`: dedicated storage or a lifetime union with dead operands.
  The underlying allocation is enlarged when necessary; the two logical
  lifetimes remain disjoint. Persistent tiles are rejected. Transport variants
  require their corresponding last-reader synchronization before reuse.
  It is independently tunable from CTA dimensions and warp arrangement.
- `cooperative_warps`: zero retains the original MMA-only cooperative schedule;
  4/8/16/32 selects a legal total warp count independently of MMA ownership.
  Extra warps participate in quantization and barriers but not MMA. Initially
  implemented for scalar MXFP8 transport. At the target shape, 64-row down
  tiles with four MMA warps and sixteen total warps improved on either the
  original smaller-tile or larger-tile schedules.
- Logical M=32 tiles preserve native scale addressing with a smaller logical
  scale view. They pass aligned/ragged parity, but did not beat the current
  target-shape winner merely by producing a larger CTA grid.
- `glu_quant_epilogue`: `exchange` or `fragment_registers`. The latter keeps
  GLU values in accumulator-owned registers and exchanges only partial maxima
  between the two N warps sharing an output scale block. It currently requires
  N=128; the attempted N=256 extension failed parity and is statically rejected.
  It was approximately 5–6 µs slower than the best exchange configuration in
  the first target-shape ablation, so it is not the default or current winner.

Packed-store debugging exposed an existing ragged vector-load alignment issue:
alignment must be computed from the physical weight layout, not the logical
interleaved gate/up row number. Aligned and ragged parity tests cover that case.

Separate scope limits remain: the mixed down kernel currently uses scalar
transport, and recomputation stores BF16 derivative/hidden temporaries in
backward. Neither limitation is hidden behind additional eager GPU arithmetic
in the operator launcher.

`down_backend="producer_quantized"` instead generates freshly quantized down
weights inside the gate/up kernel, then uses the stable packed GEMM. This adds
transient packed-weight storage, not a third forward launch or retained state.
It never reuses quantized weights across optimizer updates.

`backward_overlap=True` overlaps dDown with the independent gate/up backward
branch using CUDA stream dependencies. Partial-gradient requests retain the
existing selective schedule. `recompute_backend="prequant"` independently
chooses a quantize-once gate/up recompute followed by a packed GEMM with fused
derivative and hidden reconstruction. It adds a backward-only launch and does
not change forward retention. Both options remain autotunable.

A fresh compiled comparison of the overlap winner measured approximately
134 µs versus 171 µs checkpointed BF16 (1.27×), or 143 µs ordinary BF16
(1.07×), in `autotune_reports/mlp/goal_parallel_comparison.json`.
Subsequent compiled repeats confirmed the final result below.

## Final matched result

| Independent process | Compiled checkpointed BF16 | Compiled MXFP8 | Speedup |
|---|---:|---:|---:|
| 1 | 170.76 µs | 113.06 µs | 1.5104× |
| 2 | 170.79 µs | 112.99 µs | 1.5116× |
| 3 | 170.73 µs | 112.90 µs | 1.5123× |

These repeats used Torch 2.13.0+cu132. The same preset on Torch 2.12.1+cu132
measured 112.38 µs versus 171.68 µs checkpointed BF16 (1.5276×), with identical
reported reference errors (`goal_final_torch2121_comparison.json`).

Ordinary BF16 **without recomputation** was approximately 144 µs; the winner
is 1.27× faster against that different retention policy, not 1.5×. All numbers
exclude compilation/capture and measure warmed CUDA graph replay. Both matched
paths retain only input/weight storage. The benchmark independently checks
forward and all three gradients before timing.

The winner retains on-chip gate/up quantization, quantizes down weights inside
that first kernel, uses the packed down GEMM, recomputes with a separately
tuned packed GEMM, and overlaps independent backward branches. It uses gradient
prefetch, coalesced derivative exchange, native packed hidden reconstruction,
and `glu_silu_math="approx_reciprocal"`. That last choice uses an explicitly
approximate FP32 reciprocal (without flush-to-zero), consistently in forward
and recomputation. Maximum relative L2 against the independent **quantized
MXFP8 reference**, not BF16, was 0.0001601 (0.0161%).

The cooperative-grid forward did not win this comparison. It remains an
experimental option with an additional counter-initialization memset and
transient packed operands; the winning preset does **not** use that path.

The revisioned, device/shape-scoped preset is
`benchmarks/configs/mxfp8_mlp_5070ti_m512_k768_h1536_o768.json`. Reproduction
commands are in `docs/mxfp8_mlp.md`; the preset is not a universal default.

Qualification completed so far:

- 22 CUDA MLP/decoder tests passed on Torch 2.13.0+cu132 and 2.12.1+cu132.
- 21 additional selected config/shared MXFP8, GLU, and NVFP4 regression tests
  passed (this count includes CPU configuration checks).
- Compute Sanitizer memcheck: five selected tests, zero errors.
- Compute Sanitizer racecheck: three selected tests, zero hazards or warnings.
- Native packed reconstruction matched FP32-then-BF16 across every finite
  E4M3 value and all 255 finite E8M0 scale encodings, including extreme scales.
- Compiled graph tests mutate weights between replays; no stale quantized
  weight cache is used. Retention tests report zero non-input saved bytes.
- Nsight Systems counted exactly two device kernels in the final preset's
  compiled forward (`autotune_reports/mlp/goal_final_forward.nsys-rep`).

The tuned speedup is currently qualified for the stated SiLU shape on the
5070 Ti. Other shapes, activations, GPUs, and full training-step speedups need
their own measurements. Optional configurations are not all equally fast;
many rejected/negative experiments are intentionally retained in residuals.

The legal three-role TMA pipeline sweep also did not improve the current
scalar target-shape winner (`autotune_results/mlp_three_role`). These negative
results are retained, rather than treating pipelining, more CTAs, or more
register-resident work as inherently faster.

The fresh compiled comparison after the first scratch/staging round measured
approximately 211 µs MXFP8 versus 171 µs checkpointed BF16
(`autotune_reports/mlp/goal_alias_packed_comparison.json`). Later tuner minima
above still require a fresh matched compiled comparison. Memory checking of
wide stores and dead-operand scratch reuse reported zero errors; additional
warp/fragment variants have separate qualification tests.
