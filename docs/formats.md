# Formats and numerical semantics

## What is low precision?

The dynamic Python API takes BF16 X and W and returns BF16 Y. Low precision
is an internal representation of the GEMM operands, not a request to keep
optimizer state or the entire model in FP8/FP4.

| Layer | Forward payload | Local scale | Outer scale | Backward |
|---|---|---|---|---|
| MXFP8Linear | E4M3 | E8M0 per 32 reduction elements | None | MXFP8 |
| NVFP4Linear | E2M1 (two values/byte) | E4M3 per 16 reduction elements | FP32, depending on policy | MXFP8 |

FP32 accumulators/reductions are retained where required. Gradients returned to
ordinary BF16 inputs/parameters are BF16. Quantization uses a straight-through
backward convention; scale selection and rounding are not trained scalar
parameters. Finite-difference differentiation of the discontinuous quantizer
is not the correct oracle for this convention.

## MXFP8 scales

E8M0 represents powers of two. The training default rounds scale selection
toward infinity to accommodate the block maximum; the floor/OCP conversion
is an explicit low-level ablation that can clip values. Scale policy belongs
to numerical correctness, not just scheduling. Do not assume that different
scale rounding choices produce identical gradients.

For `Y = X @ W.T`, forward reduces over input features K. Backward dX reduces
over output features N; dW reduces over flattened leading dimensions M. A
logical transpose is free metadata, but it does **not** make a forward group of
32 values the right scale group for the new reduction axis. Backward quantizes
the oriented operands independently. Long dW reductions accumulate in FP32;
split-reduction schedules retain FP32 partials before the final BF16 cast.

## NVFP4 scale hierarchy

The approximate reconstruction is `payload * block_scale * outer_scale`.
The native MMA consumes the low-precision payload and block scales. The FP32
outer factors are applied to accumulators in the epilogue; they are not passed
as additional FP32 scale operands to `MmaMXF4NVF4Op`.

- `jit_row_region`: current FP32 factors for bounded row groups; default
  dynamic grouping is X=5 and W=4, independently controllable.
- `delayed`: historical tensor factors, owned by the module; current data
  initializes the first call. Still delayed in inference/no-grad mode.
- `current`: current tensor factors; a useful reference policy.
- `block`: outer factor is one; no FP32 outer normalization. This reduces
  overhead but relies on the data fitting the E4M3 block-scale range.

Region rows are **rows of the flattened operand matrix**, not 2D square blocks
within a weight matrix. `X=5,W=4` does not mean a 5-by-4 scale block along K.
Each row still has its native 16-element block scales. Regional factors cannot
be folded into E4M3 block scales without potentially changing their range and
rounding contract. [NVFP4Linear](nvfp4_linear.md) documents all overrides.

`nvfp4_linear` is stateless and supports current/JIT-region/block only. Use
`NVFP4Linear(scaling="delayed")` for owned history. Packed-weight inference
supports current/block; JIT-region and delayed need dynamic BF16 weights.

## Packed tensors and logical padding

`MXTensor` and `MXFP8Tensor` alias TorchAO's MXTensor; `NVFP4Tensor` aliases its
NVFP4Tensor. A tensor's low-precision dtype alone is insufficient: block scales,
layout metadata, logical shape, and optional outer scale must travel with it.
Use `quantize_mxfp8` / `quantize_nvfp4` instead of constructing private metadata.

Logical K is preserved. MXFP8 storage rounds K to a multiple of 32 values;
NVFP4 also rounds physical K to 32 values for 16-byte packed-row alignment,
while its numerical scale block stays 16 values. Padding values are zero.
Tensor-core-native scale layouts are aligned fast paths; row-major scales
handle ragged storage. Arbitrary transpose/slice/redistribution of packed
TorchAO objects is not guaranteed to preserve a legal RTX kernel layout.

## What to measure

Separate implementation error against an explicitly quantized reference,
total low-precision error against BF16, and training convergence. Report
absolute error and relative L2 with a denominator floor, plus nonfinite counts.
Cosine alone can hide scale errors. Benchmarks must use the same inputs,
initialization, optimizer precision, checkpointing, and compile policy.
See [PRECISION_POLICY](../PRECISION_POLICY.md) and the
[MLP stress example](mxfp8_mlp_numerics.md).
