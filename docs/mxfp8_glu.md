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
That explicit call tunes one representative shape before compile/capture.
For per-shape first-hit selection, instead construct the module with
`autotune="balanced"`, `tuning_policy`, and `autotune_cache_dir`. The default
remains off; see [shared frontend controls](api.md#shared-frontend-controls).
The module defaults to the conservative quantize-once configuration and exposes
`config=` for full manual control. State dicts contain the BF16 combined gate/up weight;
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

The functional example above tunes explicitly before compilation and binds
the returned configuration into the compiled call. `MXFP8GLU` also supports
opt-in per-shape first-hit selection with `autotune="balanced"`, `tuning_policy`,
and `autotune_cache_dir`; its default remains off. This does not install
winners into the ordinary linear runtime cache. See [shared frontend controls](api.md#shared-frontend-controls).

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
  remains follow-up work. Automatic first-hit forward tuning is available as
  an opt-in; it does not add autograd support.
- Reported speedups are shape/device-specific, not a universal dispatch claim.

Validation command:

```bash
.venv/bin/python -m unittest tests.test_mxfp8_glu -v
```

## Historical initial RTX 5070 Ti evidence (2026-09-12)

See [historical measurements and qualification](mxfp8_glu_history.md).

## Expanded search and aggressive compiled baseline

See [historical measurements and qualification](mxfp8_glu_history.md).

## Revision 4: vectorized output quantization

See [historical measurements and qualification](mxfp8_glu_history.md).
