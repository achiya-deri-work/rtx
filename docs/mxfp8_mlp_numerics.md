# Recomputed MLP numerical stress qualification — 2026-09-12

## Outcome

RTX 5070 Ti, CUDA 13.2, Torch 2.13.0 and 2.12.1:
**430 successful cases and two isolated fatal singleton cases**. This is not
a clean release pass. No kernel implementation or numerical gate was changed
to obtain these results.

The packed-down policy hits a CUDA illegal instruction for M=K=H=O=1 with
both ordinary and tiny inputs. Compute Sanitizer locates the fault in
`MXFP8GemmKernel`, in the down projection during forward, at instruction
offset `0x350`, threads 512–543. This is a launch/kernel correctness blocker,
not an assertion about quantization error. The conservative default passes
the singleton cases. The exact responsible instruction has not been decoded;
do not infer a confirmed register-allocation root cause from its thread index.

## Numerical results

Against the independent explicitly quantized operational reference, worst
relative L2 errors across completed stress cases were:

| Tensor | Maximum relative L2 |
|---|---:|
| Output | 0.3637% |
| Input gradient | 0.0711% |
| Gate/up weight gradient | 0.0523% |
| Down weight gradient | 0.0350% |

All completed cases had finite outputs and gradients. These are implementation
errors, **not total error versus BF16**. On the tuned M512/K768/H1536/O768 SiLU
case with ordinary random inputs, total differences versus an independent
BF16 MLP were 6.585% / 6.653% / 6.666% / 6.591% for output/dX/dW-gu/dW-down.
This includes multiple MXFP8 quantization steps and independently quantized
backward GEMMs. It is not a single-linear error measurement or a convergence
guarantee. Adversarial outliers and activation sign changes can increase that
gap substantially; near-zero references can also produce large relative
errors with negligible absolute error. Every case records both.

The largest forward discrepancies were localized further:

- SiLU, M17/K8192/H256/O129, outliers: four hidden FP8 values differ,
  with no scale-byte differences. Two large values are near FP8 rounding
  midpoints (-136.0011 and 400.0019 after scaling).
- GELU, M64/K128/H256/O64, outliers: one hidden FP8 value differs,
  with no scale-byte differences; the scaled reference is 272.00003,
  adjacent to the midpoint between 256 and 288.
- The down projection is bitwise equal to the reference when supplied with
  the actual packed hidden tensor. An independent FP64 gate/up calculation
  reproduces the hidden-reference mismatch, localizing it upstream of down.
  These observations show rounding-boundary sensitivity, not corrupted scales.

## Matrix and methodology

- SiLU, exact GELU, ReLU; conservative and packed/overlapped recomputation;
  compiled target-shape preset with approximate FP32 reciprocal SiLU.
- Singleton, ragged, aligned, and wide shapes; input reduction up to 8192,
  hidden width up to 8193, dW reduction up to 32768 rows.
- Normal, tiny, large, outlier-heavy, paired cancellation, scale-boundary,
  strongly negative gate, all-zero, zero upstream gradient, sparse, signed
  zero, subnormal BF16, and large upstream-gradient inputs.
- Torch 2.12.1 default-policy matrix uses another seed. Wide cases are also
  repeated with identical seeds/policy on Torch 2.13.0.
- Reference quantization uses PyTorch E4M3/E8M0 operations, independently
  implemented projection and activation arithmetic, and independent gradient
  formulas. No production kernels are called by that reference. Separate
  BF16/FP32 autograd baselines quantify total precision loss.
- TF32 and BF16 reduced-precision reductions are disabled for this numerical
  investigation. This is not the performance benchmark's configuration.
- Each output/gradient records finiteness, maximum absolute error, RMS error,
  reference RMS, relative L2, normalized maximum error, cosine, and worst
  element index. Relative denominators use a floor of 1e-30.
- The predeclared gate is finite and (relative L2 <= 2.5% or maximum absolute
  error <= 1e-12). Absolute tolerance handles vanishing references; recorded
  relative errors remain visible. STE gradients are tested against their
  operational contract, not finite differences of a discontinuous quantizer.

There are 429 passes in completed stress bundles plus one pass before the
initial process encountered the singleton fault. Two fresh isolated processes
reproduced that fault. Failed tensors/configs are saved, and fatal device
errors stop their process rather than contaminating later tests.

## Reproduce

```bash
python -m benchmarks.stress_mxfp8_mlp_numerics \
  --start 27 --output-dir autotune_reports/mlp/new_stress

# Safe default-only matrix, including singleton cases:
python -m benchmarks.stress_mxfp8_mlp_numerics --policy default \
  --output-dir autotune_reports/mlp/new_default_stress

# Known failing case: use a dedicated process; it poisons that CUDA context.
CUDA_LAUNCH_BLOCKING=1 python -m benchmarks.stress_mxfp8_mlp_numerics \
  --policy packed --start 0 --limit 1 \
  --output-dir autotune_reports/mlp/new_singleton_repro
```

Output directories must not already exist. Results are flushed per case in
`cases.jsonl`; policy/environment snapshots and summary JSON accompany them.
Current local bundles are `numerics_stress_v1_remaining`, `numerics_wide_v1`,
`numerics_default_torch2121`, `numerics_exponents_v1`, and
`numerics_wide_same_seed_torch213`, under `autotune_reports/mlp/`.
The early remaining-bundle summary incorrectly counts all 183 planned cases;
its JSONL contains 156 actual cases. The runner now reports selected/attempted
counts explicitly. Consolidated counts are in `numerics_consolidated_v1.json`.

This qualifies numerical behavior on the tested local matrix, not all tuning
configurations, nonfinite inputs, other GPUs, or model-scale convergence.
