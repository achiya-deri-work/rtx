# Production precision policy

This document defines the numerical policy used to evaluate and ship RTX
training kernels. It is the normative contract for convergence studies,
throughput measurements, and release gates.

## Production training contract

The ordinary model representation is BF16 (`torch.bfloat16`):

- model parameters;
- gradients at operator boundaries;
- activations and the residual stream; and
- inputs and outputs of `rtx.MXFP8Linear` and `rtx.NVFP4Linear`.

Low-precision formats replace the arithmetic of eligible GEMMs, not the
surrounding model representation:

- `MXFP8Linear` dynamically quantizes BF16 activations and weights, performs
  the forward and backward linear GEMMs in MXFP8, accumulates as required in
  FP32, and returns BF16 tensors.
- `NVFP4Linear` uses NVFP4 for its forward linear GEMM and MXFP8 for backward,
  with BF16 inputs, parameters, outputs, and returned gradients.

Optimizer-state precision is a separate policy axis. Production convergence
and end-to-end runtime comparisons use FP32 moments and FP32 master weights,
with the same policy for BF16, MXFP8, and both NVFP4 modes. FP32 optimizer
bookkeeping does not turn BF16 model execution into all-FP32 training. The
BF16-optimizer path is retained only as an explicitly labelled kernel-isolation
throughput ablation.

## Operations retained in FP32

FP32 is used locally where range or accumulation error matters more than the
storage bandwidth saved by BF16. This includes:

- tensor-core accumulators and long reductions, especially split-K `dW`;
- normalization statistics and reduction arithmetic;
- RoPE angle/rotation arithmetic;
- softmax and loss reductions;
- sums whose length makes BF16 drift material; and
- selected transcendental functions and other numerically sensitive scalar
  calculations.

The result is cast back to BF16 at the appropriate model boundary. Keeping a
reduction or transcendental calculation in FP32 does not make the surrounding
layer or training run an FP32 workload.

## Correctness, convergence, and performance baselines

The three questions are deliberately separated:

1. **Numerical correctness:** FP32 references may be used as local oracles for
   individual operators, accumulations, and gradients. These tests measure
   implementation error; they do not define the training recipe.
2. **Training convergence:** BF16 GEMMs are the baseline. BF16, MXFP8,
   NVFP4-delayed, and NVFP4-block runs must use identical initialization, data
   order, optimizer and optimizer-state precision, learning-rate schedule, and
   compilation settings. Comparisons against an all-FP32 model training run
   are optional diagnostics, not a release gate.
3. **Training throughput:** speedup is end-to-end tokens per second relative
   to the otherwise identical BF16 run. Production gates must include the
   same optimizer step and state precision for every mode. Production results
   use FP32 optimizer state/master weights; BF16-state ablations are reported
   separately.

The current decoder throughput targets are:

- MXFP8 at least **1.30x** the BF16 baseline; and
- NVFP4 at least **1.35x** the BF16 baseline.

On the measured 70-SM RTX 5070 Ti decoder workload, the steady-state rates
were 186.8k BF16, 243.3k MXFP8, and 255.1k NVFP4 tokens/s, corresponding to
**1.302x** and **1.365x**. These are exact-shape, device-local observations,
not portable performance guarantees, and were measured with the explicitly
labelled BF16-optimizer kernel-isolation policy. Production convergence and
end-to-end runtime artifacts use FP32 optimizer state/master weights.

## Reporting requirements

Every convergence or performance artifact should record at least:

- device identity and architecture;
- precision mode and linear-kernel backend;
- parameter, gradient, optimizer-state, master-weight, and activation dtypes;
- optimizer and schedule;
- compilation settings;
- model geometry, batch size, and sequence length; and
- the set of operations intentionally promoted to FP32.

Reports should say **FP32 reference** only for an oracle calculation and
**BF16 baseline** for the production model comparison. If optimizer moments or
master weights are FP32, report that independently. This avoids conflating
selective FP32 accumulation or optimizer bookkeeping with all-FP32 model
training.
