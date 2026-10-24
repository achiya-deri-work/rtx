# Quartet II: stream/RNG repair, desktop benchmarks, and decoder training

Follow-up to [the initial native-kernel study](quartet2_kernel_study_20261023.md).
This is an **isolated experimental backend**, not a change to RTX defaults or
a public FP4-backward API. Upstream revision remains
`5f2a47e8f464b17ef4b1cac78db8d14edd1f3f7d` from
[IST-DASLab/Quartet-II](https://github.com/IST-DASLab/Quartet-II).

## Concrete integration fixes

`benchmarks/patch_quartet2_research.py` creates a new source tree, preserves
upstream notices, and writes before/after source hashes. It refuses an existing
output directory. Three changes are independently identifiable:

1. Put native launches, extended launches, and scratch resets on the current
   PyTorch stream. Scratch resets use asynchronous memset on that stream.
2. Generate automatic stochastic seeds on GPU and pass a device pointer down
   to the scale-conversion kernel. Native Philox reads the seed on device.
3. Optional `--contiguous-backward`: materialize strided incoming gradients
   before native rotation, which otherwise asserts contiguity.

The stream adapter calls Python through nanobind while retaining the GIL. It
is a research bridge, not an optimized dispatcher/device-guard implementation.
All tests use one visible device per process; multiple devices in one process,
concurrent use of one module's scratch buffers, and general ragged shapes are
not qualified. Explicit CPU seeds are a convenience outside graph capture;
automatic seeds are created on GPU.

### Ablation evidence

| Backend | Changed-input graph replay | Fixed-rotation stochastic scales |
|---|---|---|
| Unmodified upstream | Stale output; 50.98% NRMS against eager after doubling input | Not qualified |
| Stream fix only | Exact eager match | 1 distinct sample in 16 replays |
| Stream + GPU seeds | Exact eager match | 16 distinct samples in 16 replays |

The stream-only path already produced changing **gradients**, because Hadamard
signs changed. That was insufficient: the fixed-rotation probe isolated the
frozen scale RNG. On the 5090, reseeding the CUDA generator reproduced scales
exactly. Stream/RNG checks also passed on the 5060 and 5070 laptops.

Strided-gradient failure was reproduced in the upstream backward. The optional
contiguity adapter passed eager and fullgraph-compiled CUDA-graph execution on
the 5090, including the changed-input check. This did not modify the backend
used for the matched contiguous training/timing matrix.

## Training: 63 runs, not just one-layer regression

Completed **31,500 optimizer updates / 107,520,000 training tokens** across 63
cases. Every run completed with finite parameters and finite gradient norms.
An artifact audit verified step counts, identical initial-weight hashes and
corpus-prefix hashes within each matched seed/architecture group.

All runs use the existing byte-level TinyStories decoder, QK normalization,
pre/post normalization, BF16 embedding/output boundaries, 0.02 truncated-normal
initialization, and depth-scaled residual initialization. Internal attention
and MLP projections are replaced. BF16 fused AdamW, learning rate 3e-4, betas
(0.9, 0.95), weight decay 0.1, gradient norm clipping at 1.0. Fullgraph/static
compilation is enabled. Training timings are diagnostic; automatic CUDA graphs
are disabled in this training harness.

Data: first 10 million cached byte tokens; first 90% for training, remaining
10% for validation. Step-indexed windows are identical across modes. Each run
uses 16 matched validation batches. Seeds: 17, 29, 43. These are short,
single-corpus, byte-level studies, not reproductions of the paper's large-model
training or sufficient evidence for a universal ranking.

Definitions:

- **RTX default:** JIT-region NVFP4 forward, MXFP8 backward.
- **RTX Four-over-Six:** current tensor outer scale with explicit
  `exact_four_over_six` headroom and Four-over-Six block selection; MXFP8 backward.
- **Quartet:** upstream Four-over-Six forward and patched MS-EDEN FP4 backward.
- **Hybrid:** Quartet forward, original-BF16-operand RTX MXFP8 backward.
- **Quartet BF16 STE:** upstream forward with `disable_backward_quant=True`;
  backward uses dequantized forward operands with BF16 GEMMs. This is not the
  same gradient oracle as original BF16 operands.

### RTX 5060 Laptop, Torch 2.12.1+cu132

Two layers; widths 256 and 512; FFN width twice hidden width; batch 16;
sequence 128; 250 steps. Mean validation cross entropy over three seeds:

| Mode | Width 256 | Width 512 |
|---|---:|---:|
| BF16 | 1.488306 | 1.324132 |
| MXFP8 | 1.490344 | 1.327001 |
| RTX default | 1.520956 | 1.336165 |
| RTX Four-over-Six | 1.515694 | 1.337839 |
| Quartet | 1.514761 | 1.344296 |
| Hybrid | 1.519635 | 1.339485 |
| Quartet BF16 STE | 1.514466 | 1.335035 |

Quartet beats RTX default on two of three width-256 seeds, but loses on all
three width-512 seeds. Increasing width also changes capacity and optimization;
this does not establish that width itself fixes quantization error.

### RTX 5090, Torch 2.13.0+cu132

Four layers; width 512; FFN 1024; batch 32; sequence 128; 1,000 steps per run:

| Mode | Mean validation CE | Mean excess over BF16 |
|---|---:|---:|
| BF16 | 0.935898 | — |
| MXFP8 | 0.937910 | +0.002011 |
| RTX default | 0.941906 | +0.006008 |
| RTX Four-over-Six | 0.943775 | +0.007877 |
| Quartet | 0.945291 | +0.009393 |
| Hybrid | 0.941424 | +0.005525 |
| Quartet BF16 STE | 0.944153 | +0.008255 |

Quartet loses to RTX default on all three seeds here. The hybrid's mean
advantage over RTX default is only 0.000482 and is not consistent across all
seeds. Do not promote it as a reliable quality gain. There are only three
training replicates; validation batches are not additional training replicates.

## Repeated graph timing on RTX 5090

Forward plus both gradients, BF16 inputs/outputs. Two fresh processes per
mode/shape; eight steps captured per graph; nine event-timed samples of ten
replays each. Compilation, warmup, capture, and correctness checks are excluded.
Each case verifies changed-input replay against eager before timing. Warmup
must occur on the actual capture stream: an earlier harness revision correctly
hit RTX's cold-runner guard when it warmed only the default stream.

Reported values are the mean of two process medians, in microseconds. Schedules
are fixed, **not exhaustive autotuning winners**. Vendor plan selection and
the RTX default schedules are not identical tuning procedures. GPU 0 ran this
matrix while GPU 1 ran training; within-GPU benchmarks were serialized.

| M×N×K | BF16 | MXFP8 | RTX default NVFP4 | Quartet | Hybrid |
|---|---:|---:|---:|---:|---:|
| 2048×768×1536 | 93.14 | 48.21 | 44.06 | 88.46 | 59.47 |
| 8192×768×1536 | 320.04 | 167.06 | 151.40 | 164.33 | 166.94 |
| 32768×768×1536 | 1187.41 | 771.13 | 715.56 | 498.15 | 747.92 |
| 8192×3072×768 | 566.69 | 334.36 | 309.81 | 242.66 | 294.13 |

Quartet is 1.44× faster than RTX default on the longest reduction and 1.28×
faster on the wide projection; RTX default is about 2× faster on the smallest
of these shapes. Therefore neither a universal replacement nor universal
rejection is supported. Timing variation between the two process medians was
small for these 5090 cases; larger laptop timing variability motivated the
unrolled, repeated-process comparison.

## Numerical and profiler follow-through

Eighteen 5070-laptop stress cases cover three implementations, sparse inputs,
outliers, all-zero operands, magnitudes 0.001 and 1000, and strided gradients.
Seventeen completed; the upstream strided-gradient assertion was the one
failure, subsequently fixed/tested in the isolated 5090 adapter. All-zero
outputs and both gradients were exactly zero. Across the nonzero cases,
Quartet single-sample backward error remained roughly 14–17% NRMS; the
MXFP8-backward paths remained roughly 3.4–3.9% against original BF16 operands.

Averaging MS-EDEN samples reduces noise relative to **forward-quantized**
operands. It does not make that oracle equal to original BF16 operands.
The 64-sample strided-adapter test achieved about 1.74% against the former,
but about 8.9% against the latter. Never compare that 1.74% directly with
MXFP8's original-operand error as though they measured the same target.

Nsight Systems node traces on the 5070 identify three FP4 GEMMs plus rotation,
requantization, and scale-conversion work in Quartet. Scale conversion accounts
for about 4.2% of traced kernel time; forward absolute-max reductions about
10.5%. RTX's backward GEMMs plus backward quantizer account for about 73% of
its traced kernel time. Profiler overhead differs materially from ordinary
timing: use these as attribution, not cross-backend latency comparisons.
Nsight Compute was unavailable; no register, bandwidth-saturation, or
tensor-utilization claim is made.

The patched upstream rounding/linear suite remains **14 passed, 3 failed**.
Failures are stochastic backward thresholds at 64, 256, and 1024 samples
(7.375 vs 7.5, 7.722 vs 8.1, 8.897 vs 8.9 bits). Those failures have not been
relaxed or dismissed. Four CPU tests validate the mechanical patch transforms.

## Decision and next implementation boundary

- Keep the existing MXFP8-backward default. The longer training study does not
  justify replacing it with MS-EDEN globally.
- MS-EDEN merits an **explicit experimental numerical policy** for large
  projections/reductions. Its hardware feasibility and speed opportunity are
  now demonstrated on RTX, despite the lack of native stochastic FP4 rounding.
- Schedule autotuning can optimize within a policy. It should not silently
  switch between these different backward oracles/noise distributions.
- Four-over-Six remains optional; lower local forward error did not translate
  into consistent training superiority.
- Post-hoc scale range alignment is worth a separate RTX implementation
  experiment, not a claimed gain here. Profile evidence prioritizes backward
  work over simply micro-optimizing the final scale-conversion pass.
- Before any public backend: explicit device/stream ABI, concurrent scratch
  ownership, ragged support, checkpoint/RNG restoration, memory qualification,
  more seeds/tasks, and investigation of the upstream threshold failures.

## Artifacts and reproduction

Raw logs, JSONL, build logs and Nsight traces:
`autotune_reports/quartet2_deeper_20261023/{5060,5070,5090}/`.
The 5060 connection later became unavailable; all 42 completed decoder cases
were already copied locally. The complete follow-up stress matrix was run on
the available 5070 instead.

Research entrypoints:

- `benchmarks/patch_quartet2_research.py`
- `benchmarks/probe_quartet2_runtime.py`
- `benchmarks/evaluate_quartet2.py`
- `benchmarks/train_quartet2_decoder.py`

After installing the pinned upstream package in an isolated target and placing
it before other packages on `PYTHONPATH`:

```bash
python benchmarks/patch_quartet2_research.py upstream/kernels new-source \
  --gpu-seed --contiguous-backward
# Build/install new-source separately with CUDAARCHS=120a.
python -m benchmarks.probe_quartet2_runtime --rng-probe
python -m benchmarks.evaluate_quartet2 --mode quartet \
  --shape 32768,768,1536 --compile --graph --graph-unroll 8 --samples 16 \
  --output /tmp/new-kernel-result.jsonl
python -m benchmarks.train_quartet2_decoder --mode quartet --seed 17 \
  --width 512 --layers 4 --batch 32 --steps 1000 \
  --tokens training_data/tinystories-train-00000.byte_tokens.npy \
  --output /tmp/new-decoder-result.jsonl
```

Outputs are exclusive-create. Do not point an experiment at an existing result.
