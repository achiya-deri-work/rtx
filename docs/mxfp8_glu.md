# Experimental fused MXFP8 gate/up projection

Available on `feat/mxfp8-glu`; this is **forward-only**, not yet a training
module or a complete FFN. No down projection or bias is included.

## Module and tensor-only APIs

`rtx.MXFP8GLU` owns the packed BF16 parameter, with truncated-normal
initialization (`std=0.02`, bounds ±0.04). It is an experimental **forward-only**
module, not a drop-in training FFN. `.eval()` does not disable autograd; use
`torch.no_grad()` or `torch.inference_mode()` until backward is implemented.

```python
import torch
import rtx

layer = rtx.MXFP8GLU(
    in_features=768, hidden_features=1536, activation="silu",
    output_format="mxfp8", packed_output=True, device="cuda",
)
# Or copy existing bias-free BF16 projections, without modifying them:
# layer = rtx.MXFP8GLU.from_linears(gate_proj, up_proj, activation="silu")

layer.autotune_(
    example_x, backend="auto", max_trials=64, time_budget_s=120,
    output_dir="autotune_results/my_glu_module",
)
print(layer.config, layer.tuning_result)
forward = torch.compile(layer, fullgraph=True, dynamic=False)
with torch.inference_mode():
    data, scales = forward(example_x)
```

`autotune_()` returns the module and exposes the report as `tuning_result`.
It tunes one representative shape and never starts a search inside forward.
Tuning must finish before compile/capture. A different shape can need a
different configuration; there is no implicit per-shape tuning dispatcher yet.
The module defaults to the conservative quantize-once configuration and exposes
`config=` for full manual control. State dicts contain only the packed weight;
retune after constructing/loading on a different machine. Activation and
output policy remain constructor settings, like other architectural choices.

`rtx.mxfp8_glu_packed(x, weight, activation=..., config=...)` returns ordinary
`(E4M3 data, E8M0 scales)` tensors, with all leading X dimensions preserved.
The final dimensions are `ceil(H/32)*32` and `ceil(H/32)`, respectively; H is
`weight.shape[0]//2`. This keeps logical-width handling explicit when composing
custom kernels and avoids constructing a TorchAO tensor subclass. It does not
dequantize, change the scale policy, or round through BF16. The existing
`mxfp8_glu(..., output_format="mxfp8")` still returns TorchAO's MXTensor.

Eager and compiled paths reuse bounded per-shape/device/stream operand
workspaces. Outputs are freshly allocated, and X/W are requantized on every
invocation. The new API does not introduce prequantized-weight inference.

At `(M,H,K)=(512,1536,768)` on the local 5070 Ti, the tensor-only API measured
**26.39 µs uncaptured versus 29.57 µs** for the compiled baseline (1.12×),
compared with the previous MXTensor frontend's approximately 33.44 µs.
Replay remained **21.63 versus 28.80 µs** (1.33×). These are interface-specific
results, not a new kernel precision policy. Reproduce with
`python -m benchmarks.benchmark_mxfp8_glu --output-format mxfp8 --packed-output`
and an explicit tuned config/history. Raw measurements are in
`autotune_reports/glu_api/packed.json`.
The module/API iteration passed the full 15-test GLU suite on Torch 2.13
and eight targeted configuration, conversion, compiler, and replay tests on
Torch 2.12.1. All 96 additional output-reduction tuning trials passed their
numerical gates; the best sampled configuration continued to use shuffle.

```python
import torch
import rtx

# X: [..., M, K], W: [2*H, K], both contiguous BF16 on an SM12x GPU.
# W[:H] is gate; W[H:] is up/content. No physical interleaving is needed.
with torch.no_grad():
    y = rtx.mxfp8_glu(x, w, activation="silu")  # [..., M, H], BF16
    qy = rtx.mxfp8_glu(x, w, activation="gelu", output_format="mxfp8")
```

`activation` accepts `silu`, `gelu`, and `relu`. GELU uses the erf definition,
not the tanh approximation. Activation is a user-selected mathematical policy;
the tuner cannot change it. Likewise, it cannot change the output format.

## Numerical and storage contract

1. Dynamically quantize each X and W row in blocks of 32 K values to E4M3,
   with round-to-infinity E8M0 power-of-two scales.
2. Compute paired projections with native MXFP8 MMA and FP32 accumulators.
3. Compute activation(gate) times up in FP32 on chip, **before BF16 rounding**.
4. Write only H result columns, or directly quantize the product to MXFP8.

For BF16, the only nonempty output allocation is M*H BF16 values. For MXFP8,
output storage is M*ceil(H/32)*32 E4M3 values and M*ceil(H/32) E8M0 scales.
The wrapper returns TorchAO's MXTensor with RTX logical-width metadata H,
row-major scales, and zero-filled tail values. TorchAO's native `dequantize()`
can retain the padded width; slice to H when using it outside RTX. Output quantization is along H, independently of
input quantization along K. It does not round through BF16 first.

There is no global M*2H projection intermediate, no separate transpose/packing
pass, and no separate activation kernel. A logical CuTe weight layout pairs
gate/up rows. The register epilogue combines paired accumulators in registers;
the exchange epilogue remains a searchable alternative. MXFP8 output uses a
shared product exchange for its 32-value output scale reductions.
Ragged positive M/H/K are supported by predication. Noncontiguous inputs are
rejected rather than silently copied.

## Autotuning

```python
from rtx.autotune.glu import tune_mxfp8_glu

result = tune_mxfp8_glu(
    x, w, activation="silu", output_format="bf16",
    max_trials=64, time_budget_s=180, backend="auto",
    output_dir="autotune_results/my_glu",
)
forward = torch.compile(
    lambda x, w: rtx.mxfp8_glu(x, w, config=result.config),
    fullgraph=True, dynamic=False,
)
```

Tune explicitly before compilation. The returned configuration is bound into
the compiled call. This initial API does **not** automatically tune new shapes
or install winners into the ordinary linear runtime cache.

Standalone launch:

```bash
.venv/bin/python -m rtx.autotune.glu \
  --m 512 --hidden 1536 --k 768 --activation silu --output-format mxfp8 \
  --trials 64 --seconds 180 --output-dir autotune_results/glu
```

This uses the existing composable bandit portfolio, cost-model/local search,
durable per-context residuals, and confirmation machinery. Contexts distinguish shape,
device/software fingerprint, activation, output format, and kernel revision.
The same command resumes completed observations. Timings use CUDA-event
measurements of CUDA-graph replays, with warmup, seven samples, twenty kernel
calls per sample, and two winner confirmations. Every candidate is checked
against an independent quantize/dequantize + FP32 GEMM/activation reference.
The adapter can be composed with other search recipes through
`make_glu_adapter` and `GLU_SEARCH_SPACE`.

`backend="auto"` splits the bounded search budget between two implementations
and returns the lower-latency verified result. Each backend uses its own bandit
portfolio and persistent context; backend allocation is presently equal, not
a learned outer scheduler. Select `cta` or `prequant` to explore one exclusively.

- `MXFP8GLUConfig`: quantize inside each GEMM CTA. No quantized operands in GMEM.
- `MXFP8GLUPrequantConfig`: quantize X/W once (dual or separate launch), then
  reuse E4M3/E8M0 operands across GEMM CTAs. Inputs are still dynamic BF16 on
  every invocation; this is **not** weight PTQ or delayed quantization.

The second implementation materializes quantized operands/scales, but never
the doubled-width projection. Its quantizer emits paired weight rows directly;
there is no extra interleave launch. Its quantization and GEMM configurations
are independently searchable, including native scale layouts and TMA scale
transport. `--seed-history <old-results>` may suggest a previous winner, but
discards its timing and revalidates/rebenchmarks it under the current revision.

Searchable implemented coordinates include CTA/MMA tile geometry, MXFP8
buffers, quantization vector/load width, reduction instruction, ldmatrix widths,
shared-memory swizzles, register ceiling, K unroll, persistence and grid order.
`config.tile_n` is the **paired** projection width: 128 means 64 output columns.
Resource legality includes the FP32 accumulator exchange, not just operands.

The CTA backend inherits the existing fused projection search coordinates:
scalar/cp.async/TMA, cooperative/producer-consumer/three-role pipelines,
buffering, register allocation, swizzles, load instructions, and reuse moves.
The quantize-once backend inherits the prequantized projection search space.
The ordinary GEMM output axes are replaced by GLU-specific choices:
`glu_epilogue={exchange,registers}`, `glu_store={direct,tma}`, and
`glu_store_stages={1,2,3,4}`. TMA stores currently support BF16 output with
H divisible by 8; partial tiles use direct predicated stores. Writes can overlap
subsequent persistent GEMMs, with a final drain before CTA exit. MXFP8 output
currently uses direct stores. The GEMM's stmatrix vector-width knob is not
pretended to apply to the GLU's changed fragment layout.

Some large tiles/deep buffers exceed SMEM and are rejected before compilation.
Process-global L2-fetch-limit tuning is deliberately excluded from this
per-kernel adapter. Exposing a coordinate is not a claim that all combinations
are feasible or beneficial.

`glu_quant_vec={1,2,4,8}` controls MXFP8 output quantization separately from
operand quantization. Each lane owns that many consecutive FP32 products;
subwarp reductions still cover exactly 32 output values per E8M0 scale.
Widths 2/4/8 use paired FP32-to-E4M3 conversion and 16-bit stores. This does
not introduce an intermediate BF16 rounding step. The axis is inactive for
BF16 output.
The prequantized backend additionally exposes `glu_reduction={shuffle,redux}`
(the underlying field is `gemm.reduction`). CTA-local quantization uses its
existing reduction setting. Hardware redux is a choice, not an assumed win.

### Graph replay policy

Graph replay is the default **primary benchmark score** for steady-state
graph-safe workloads; uncaptured timings remain mandatory companion results.
The kernel autotuner already measures graph replay. This is not automatic
per-layer capture: the application should capture the surrounding model or
step after compilation and tuning, so it can amortize launch costs across
operators. See [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)
for static-address/control-flow restrictions and
[CUDA Graph Trees](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
for per-shape recordings and input staging.

Changing X or W **values** at their captured addresses is supported and must
requantize them on every replay. Changing shapes or addresses requires the
appropriate graph-management layer; warmup alone does not make arbitrary
workloads graph-safe. Graph storage must remain alive. Do not clear RTX's
runtime workspaces while a graph references them. This forward-only GLU is
not yet a graph-qualified training FFN.

## Current boundaries

- No backward yet; grad-enabled calls with trainable inputs raise explicitly.
- No prequantized input/weight mode yet. MXFP8 output is supported.
- Custom operator + fake implementation and direct Inductor lowering support
  fullgraph inference. Output allocation is owned by Inductor, including the
  two independently allocated MXFP8 data/scale buffers. The compiled path caches operand workspaces
  per shape/device/stream and avoids the custom-op dispatcher and repeated
  config parsing. Native kernel launches still occur on every invocation.
  `rtx.clear_runtime_caches()` releases these workspaces; do not clear them
  while retaining external CUDA graphs that reference them. A trainable module
  and automatic first-hit tuning remain follow-up work.
- Reported speedups are shape/device-specific, not a universal dispatch claim.

Validation command:

```bash
.venv/bin/python -m unittest tests.test_mxfp8_glu -v
```

## Historical initial RTX 5070 Ti evidence (2026-09-12)

The GLU tests cover all six activation/output combinations, ragged dimensions
including `(M,H,K)=(1,1,17)`, exact output scale bytes on the reference cases,
alternate schedules, zero input, nondefault streams, opcheck, and fullgraph
compilation. They pass on both supported PyTorch stacks. Ordinary MXFP8
forward/reference and schedule regression tests also pass after the base-kernel
epilogue hook was added.

At `(M,H,K)=(512,1536,768)`, a 48-trial BF16 search (including a resumed run)
found 66.04 microseconds; a separate 24-trial MXFP8-output search found 78.37
microseconds. Both winners passed two extra confirmations. All 72 evaluated
candidates passed their numerical gates. These are small, unequal-budget,
shape-local searches—not exhaustive tuning or a comparison with tuned unfused
alternatives. The returned configurations, not the conservative default, give
these timings.

Relative L2 against the **input-quantized FP32 projection/GLU reference** was
0.166% for the BF16-output winner and 2.642% for the MXFP8-output winner. These
measure epilogue/output error, not total error against unquantized BF16 inputs.

Padded-allocation Compute Sanitizer memcheck reported zero device-access
errors over the tails, formats, schedule variants and stream tests. API-error
reporting was disabled for that focused pass: the unfiltered pass reports
CUDA 13.3 binding probes against this machine's 13.2 driver. That environment
diagnostic is not claimed to be resolved.

## Expanded search and aggressive compiled baseline

The subsequent iteration added quantize-once projection, fragment-register
epilogues, and persistent buffered TMA output. All remain search choices;
the conservative default is **not** the measured winner. The selected
quantize-once path still quantizes the current BF16 X and W on every call.

Measured on the local RTX 5070 Ti with PyTorch 2.13.0+cu132, SiLU,
`fullgraph=True`, `dynamic=False`, and the aggressive options shared with
`benchmarks/benchmark_torchao_fp8_rowwise.py`. Freezing is disabled and both
operands are runtime inputs. The baseline is BF16 packed gate/up projection
plus activation/product; for MXFP8 output it also quantizes the result.

| M × H × K / output | Compiled baseline, uncaptured | Tuned GLU, uncaptured | Speedup | External CUDA-graph speedup |
|---|---:|---:|---:|---:|
| 512 × 1536 × 768 / BF16 | 29.28 µs | 23.32 µs | 1.26× | 1.61× |
| 2048 × 3072 × 768 / BF16 | 193.91 µs | 93.35 µs | 2.08× | 2.08× |
| 512 × 1536 × 768 / MXFP8 | 29.49 µs | 32.90 µs | 0.90× | 1.21× |
| 32 × 1536 × 768 / BF16, transferred config | 21.77 µs | 22.75 µs | 0.96× | 0.51× |

Uncaptured numbers are median synchronized wall time across 11 samples of
50 calls, after warmup. The external graph diagnostic uses 21 samples of
20 calls; it is **not** the requested `triton.cudagraphs=False` end-to-end
result. The small-shape configuration was transferred, not locally tuned.
MXFP8 output and small-batch performance remain open work. The MXFP8
allocation change did not remove its uncaptured overhead.

Relative L2 versus the unquantized BF16 computation was approximately 5.37%
for the medium BF16 result, 5.38% for the large result, and 6.48% for MXFP8
output. These are reduced-precision comparisons, not exact-equivalence
claims. Candidate acceptance separately checks the input-quantized FP32
projection/epilogue reference.

Reproduce or extend the search and comparison:

```bash
python -m rtx.autotune.glu --m 512 --hidden 1536 --k 768 \
  --backend auto --trials 128 --seconds 180 \
  --output-dir autotune_results/my_glu
python -m benchmarks.benchmark_mxfp8_glu --shape 512,1536,768 \
  --history autotune_results/my_glu \
  --output autotune_reports/my_glu.json
```

Local raw measurements are under `autotune_reports/glu_iteration/`, with
revision-separated observations under `autotune_results/glu_v3_*`.
They are intentionally not shipped as portable runtime winners.

The expanded searches recorded 495 successful evaluations and one static
implementation rejection, with no recorded compile or numerical failures.
The final correctness matrix passed 16 tests on PyTorch 2.12.1; on 2.13 the
10-test GLU matrix passed, followed by seven final compiler/shared-kernel
regressions. These cover both epilogues, all activations and output formats,
ragged K/interleaved loads, native scale TMA, dynamic weight updates, and
1–4 output buffers with persistent tail tiles.
The final padded-allocation memcheck also passed the prequantized GLU and
persistent TMA recycling/tail matrices with zero device-access errors, using
the same API-probe-reporting caveat described above.

Nsight Compute supports the structural explanation: the selected GEMM's
tensor-pipe active metric rose from roughly 11% for the original scalar
implementation to 50%, while per-CTA shared storage fell from approximately
100 KB to 28 KB. The quantize-once implementation adds its own quantizer
launch and does not reduce measured total L2 traffic (about 46 MB versus
40 MB). Its benefit is avoiding repeated conversion work in the GEMM and
improving the MMA schedule, not eliminating all memory traffic. Profiler
durations are diagnostic and are not used in the speedup table.

## Revision 4: vectorized output quantization

A 192-trial prequantized-backend search selected `glu_quant_vec=4` with the
register GLU epilogue. A fresh matched SiLU benchmark at
`(M,H,K)=(512,1536,768)` measured **28.80 µs baseline versus 21.62 µs GLU**,
or **1.332× graph-replay speedup**; the repeated baseline was 28.84 µs.
Uncaptured wall timing remained slower (29.58 versus 33.44 µs). Thus the
1.3× target is met for this replay case, not for uncaptured MXFP8 execution.

The small `(32,1536,768)` shape received its own 256-trial search across both
backends. It found approximately 7.69 µs, versus the previous transferred
configuration's 9.55 µs. This is still slower than the approximately 4.84 µs
compiled BF16 baseline; **the all-shapes 1.3× target is not met**. Changing
measurement policy does not turn this into a win. These searches are bounded,
not proof of a global optimum or a hardware lower bound.

Fresh replay comparisons after this change:

| M × H × K / output | Baseline | GLU | Speedup | ≥1.3× |
|---|---:|---:|---:|---|
| 512 × 1536 × 768 / BF16 | 28.58 µs | 17.73 µs | 1.612× | yes |
| 2048 × 3072 × 768 / BF16 | 192.12 µs | 92.12 µs | 2.086× | yes |
| 512 × 1536 × 768 / MXFP8 | 28.80 µs | 21.62 µs | 1.332× | yes |
| 32 × 1536 × 768 / BF16 | 4.85 µs | 7.70 µs | 0.629× | no |

A disposable-process probe of 64-column CTA tiles failed compilation with a
block-scale source/destination fragment-size mismatch. That restriction was
not relaxed in the supported search space. Narrower tiles need their own
validated scale layout; silently accepting them is not a tuning solution.

The new searches recorded 256 valid small-shape trials and 191 valid MXFP8
trials. One MXFP8 candidate failed the correctness gate: `mma64x128` native
scales had incorrectly been allowed with row-major consumer staging. The
failure reproduced with both output formats, vector width 1 as well as 4,
and separate as well as dual quantizers. The shared legality contract now
requires TMA for that native scale layout, rejecting the invalid combination
before compilation. The selected row-major winner is unaffected.

The 12-test GLU suite passed on PyTorch 2.13, and the new vectorization and
graph-replay tests passed on 2.12.1. Replay tests verify changing input/weight
values and stable live GPU allocation over 50 replays. Padded memcheck passed
all output vector widths and tails with zero device-access errors (API probe
reporting disabled for the previously documented binding/driver mismatch).

Reports live in `autotune_reports/glu_target13/`, and residual observations in
`autotune_results/glu_target13_*`. The benchmark emits `primary_mode`,
`primary_speedup`, `target_speedup`, and `target_met` while retaining both
timing modes. Use `--primary-mode uncaptured` to change the primary target.
