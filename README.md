# rtx-blackwell low-precision linear layers

`rtx-blackwell` is an experimental Python distribution, imported as `rtx`,
for low-precision training and
inference on NVIDIA RTX Blackwell GPUs. Its two public linear
frontends are `rtx.MXFP8Linear` and `rtx.NVFP4Linear`. The
project's normative numerical and benchmarking contract is documented in
[`PRECISION_POLICY.md`](PRECISION_POLICY.md). The current implementation
contains:

- fused BF16 input/weight quantization and MXFP8 forward GEMM, including
  persistent three-role TMA producer/quantizer/MMA schedules and staged async
  TMA epilogues; E8M0 scales use training-safe round-to-infinity by default,
  while the clipping-prone OCP floor conversion remains an explicit ablation;
- materialized BF16-to-NVFP4 training forward with packed E2M1 operands, E4M3
  block scales, block-only/current/row-region-JIT/delayed FP32 outer-scale
  policies,
  rolling in-kernel amax telemetry, and a native K=64 block-scaled MMA;
  NVFP4 training reuses the MXFP8 backward kernels;
- materialized dynamic MXFP8 quantization plus GEMM, including autotunable
  one-to-four-stage mainloops and epilogues, locality scheduling, and up to
  eight output tiles per persistent CTA;
- fused dynamic MXFP8 backward for `dX` and the FP32-accumulating `dW`,
  including logical-transpose cp.async/TMA transport, persistent multi-output
  split-FP32 workspace/atomic schedules, continuous cross-output pipeline
  state, and clustered cp.async load/quantization elision with native-tile
  DSMEM publication; dX-only and dW-only autograd requests compile and retain
  only their selected matmul runner;
- one-launch four-operand backward quantizers, including a shared-G family
  that emits row and transposed MXFP8 encodings from one BF16 SMEM tile, with
  concurrent dX/dW GEMMs,
  plus autotunable fused and quantize-once FP32 workspace/atomic split-K,
  fused and prequantized generation-counted cluster-local DSMEM FP32 split
  reduction,
  per-matmul, interleaved, asynchronous logical-transpose transport,
  wide-store, wide-CTA, and CTA-cluster reuse families;
- PyTorch custom-op and `nn.Module` frontends, with allocation-free direct
  Inductor lowerings for fused training/inference, materialized dynamic,
  AOT-weight, and fully prequantized MXFP8 execution;
- persistent random, gradient-boosted cost-model, bandit, and local search;
- backend-neutral conditional spaces, staged tasks, and resumable ask/tell
  workers suitable for MoE and attention kernel projects;
- calibrated hot/rotating-cache measurements and paired finalist races; and
- portable cross-device datasets exported as CSV or Parquet.

The executable native kernels target SM120/SM121. SM110/Jetson Thor is outside
the project scope because its TCGen05/TMEM execution model requires a separate
kernel family. This is research software:
kernel, packed-operand, and dataset schemas are versioned, but APIs may still
change.

## Requirements

- Linux
- Python 3.11+
- an RTX Blackwell GPU and a compatible NVIDIA driver
- PyTorch 2.12.1 or 2.13.0 built for CUDA 13.2
- TorchAO 0.18.0
- CUDA Python 13.x
- NVIDIA CUTLASS Python DSL 4.7.0 with its `cu13` runtime extra
- PyArrow, Apache TVM FFI 0.1.13.post2, and Einops

`requirements.txt` selects the supported PyTorch 2.13.0 CUDA 13.2 stack and
installs the full runtime:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The alternate supported PyTorch 2.12.1 stack is explicit:

```bash
python -m pip install -r requirements-torch212-cu132.txt
python -m pip install --no-deps -e .
```

Other PyTorch or CUDA minor versions are intentionally rejected when a native
kernel is first loaded. The top-level `import rtx` remains architecture-neutral.
Install the CUDA 13.2 PyTorch stack before this project. A bare
`pip install rtx-blackwell` cannot make PyPI choose a CUDA-local PyTorch wheel
and is therefore not a supported native installation route.

PyArrow is a core dependency, so CSV and Parquet export are available in every
supported install.
For development tools:

```bash
python -m pip install -e '.[dev]'
```

The equivalent dependency lists are in `requirements.txt` and
`requirements-dev.txt`. An editable install is recommended while kernels are
changing because every dataset bundle records a hash of the installed source.

## Repository layout

```text
rtx/
├── rtx/
│   ├── configs/        # immutable kernel and inference configuration models
│   ├── formats/        # TorchAO-backed MXFP8/NVFP4 tensor contracts
│   ├── kernels/        # lazily imported CuTe DSL kernel implementations
│   ├── models/         # decoder-only convergence reference model
│   └── autotune/       # adapters, bandits, cost models, stores, and campaigns
├── autotune_manifests/ # immutable portable campaign specifications
├── benchmarks/         # focused developer benchmarks and tuning utilities
└── tests/              # CPU contracts plus CUDA-gated kernel tests
```

The public runtime surface is `rtx.MXFP8Linear`, `rtx.NVFP4Linear`, their
functional forms, and packed tensor types. Kernel modules remain lazy so
importing `rtx` on a non-Blackwell or CPU-only machine does not initialize
CuTe. Legacy coordinate tuners remain as compatibility modules; new search
policy work belongs under `rtx.autotune`.

The top-level namespace is reserved for the stable runtime, tensor, and config
API. Legacy tuner names remain directly importable for compatibility but are
not included by `from rtx import *`; new tuning code should import from
`rtx.autotune`.

The portable boundary is independent from CuTe and MXFP8: external projects
define a `ConditionalSearchSpace` and `FunctionKernelTask`, then use
`StagedTaskAdapter` with either the synchronous orchestrator or an
`AskTellSession`. Existing project-specific `KernelAdapter` implementations can
be exposed through `AdapterKernelTask` without being rewritten. See
[`rtx/autotune/README.md`](rtx/autotune/README.md) for a complete example.

See [`autotune_manifests/README.md`](autotune_manifests/README.md) for campaign
status and [`benchmarks/README.md`](benchmarks/README.md) for focused tools.

## Python frontend

```python
import torch
import rtx

x = torch.randn(512, 1536, device="cuda", dtype=torch.bfloat16)
w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)

y = rtx.mxfp8_linear(x, w)
layer = rtx.MXFP8Linear(
    1536, 1536, bias=False, device="cuda", dtype=torch.bfloat16
)
y2 = layer(x)

nv_layer = rtx.NVFP4Linear(
    1536,
    1536,
    bias=False,
    device="cuda",
    dtype=torch.bfloat16,
    backend="auto",
)
nv_y = nv_layer(x)  # JIT-regional NVFP4 forward, MXFP8 backward
```

The dynamic NVFP4 default uses asymmetric current scale regions (`X=5`,
`W=4`). Every scaling, geometry, backend, inference, and autotuning constructor
option is described in [the NVFP4Linear API guide](docs/nvfp4_linear.md).
The corresponding MXFP8 states, backends, conversion methods, and tuning
controls are described in [the MXFP8Linear API guide](docs/mxfp8_linear.md).
See the [runtime/checkpoint contract](docs/runtime_contract.md) and
[troubleshooting guide](docs/troubleshooting.md) for deployment behavior. The
[1.0 release checklist](docs/release_checklist.md) defines the exact source,
GPU, autotuning, numerical, and packaging gates used for a release tag.

TorchAO-style MXFP8 PTQ is available for existing no-bias BF16 models:

```python
model.eval().to(device="cuda", dtype=torch.bfloat16)
model = rtx.quantize_(
    model,
    rtx.MXFP8WeightOnlyConfig(autotune="cache"),
    filter_fn=lambda module, fqn: isinstance(module, torch.nn.Linear)
    and fqn.startswith("decoder.layers."),
)
```

This packs selected weights once while retaining dynamic MXFP8 activation
quantization. RTX linear layers do not support bias; selected biased linears
are rejected before any module replacement occurs.

The same top-level conversion surface supports NVFP4 PTQ and dynamic training
for both formats:

```python
mxfp8_training = rtx.convert_to_mxfp8_training(
    model,
    module_filter_fn=filter_fn,
    config=rtx.MXFP8TrainingConfig(autotune="cache"),
)
nvfp4_training = rtx.convert_to_nvfp4_training(
    model,
    module_filter_fn=filter_fn,
    config=rtx.NVFP4TrainingConfig(autotune="cache"),
)
nvfp4_ptq = rtx.quantize_(
    model,
    rtx.NVFP4WeightOnlyConfig(scaling="current", autotune="cache"),
    filter_fn=filter_fn,
)
```

See the [model conversion guide](docs/model_conversion.md) for parameter,
optimizer, TorchAO-rowwise, weight-sharing, bias, and torchvision ViT
contracts.

For an optional pretrained, model-level numerical and `torch.compile`
validation across 48 transformer projections, see the
[DINOv3 regression suite](docs/dinov3_regression.md). DINOv3 source and weights
are local test assets, separately licensed, and ignored by Git.

### Decoder-only convergence model

`rtx.models` contains a decoder-only language model for controlled BF16,
MXFP8, and NVFP4 training comparisons. It uses fused QKV and gate/up
projections, QK normalization, FP32-reduction RMSNorm and RoPE, and separate
pre/post sublayer norms. Token embeddings, the tied vocabulary head, attention,
and loss remain BF16/FP32 as appropriate; only transformer-internal linear
projections change precision. Base projections retain truncated-normal 0.02
initialization, while attention-output and MLP-down residual projections use
depth-progressive residual scaling, `0.02 / sqrt(2 * current_depth)`, with
one-based depth. Early residual branches are affected less than later ones,
counteracting progressive pre-norm residual dilution at initialization.

```python
from dataclasses import replace

import torch

from rtx.models import DecoderConfig, DecoderOnlyTransformer, LinearSpec

base = DecoderConfig()  # 8x768, SwiGLU-1536, 12 heads, context 512
bf16 = DecoderOnlyTransformer(base, device="cuda")
mxfp8 = DecoderOnlyTransformer(
    replace(base, linear=LinearSpec(precision="mxfp8")), device="cuda"
)
nvfp4 = DecoderOnlyTransformer(
    replace(base, linear=LinearSpec(precision="nvfp4")), device="cuda"
)

# All three dynamic models share state-dict keys.
initial_state = bf16.state_dict()
mxfp8.load_state_dict(initial_state, strict=True)
nvfp4.load_state_dict(initial_state, strict=True)

compiled = torch.compile(mxfp8, fullgraph=True, dynamic=False)
```

The low-precision frontend default is `autotune="balanced"`: exact
device/shape/policy winners are loaded first, while a missing context receives
a bounded 24-trial/30-second first-hit search and an atomically persisted
winner. Repeated calls reuse the process-local selection. Reproducible
convergence runs can still select `autotune="cache"` to forbid benchmarking.
The model is a validation scaffold, not a pretrained architecture or checkpoint
format.

Production convergence and throughput runs use BF16 model parameters,
gradients, and activations for all four training modes. FP32 is retained
locally for sensitive reductions, RoPE, softmax, selected transcendentals, and
accumulation. Optimizer-state precision is recorded independently: the
production convergence/runtime comparison uses FP32 AdamW moments and master
parameters, while `--optimizer bf16` is retained as a separately labelled
kernel-isolation throughput ablation. Neither policy is an all-FP32 model
baseline. See
[`PRECISION_POLICY.md`](PRECISION_POLICY.md) for the complete contract. Use
`--stop-after-step N` to pause a run at an absolute checkpoint while preserving
the original `--steps` learning-rate schedule and resume contract.

Check a completed production-policy run with:

```bash
python benchmarks/check_decoder_throughput.py <run-directory>
```

The default thresholds are MXFP8 >= 1.30x and NVFP4 >= 1.35x BF16. On the
70-SM RTX 5070 Ti decoder run used for this change, the measured steady-state
rates were 186.8k BF16, 243.3k MXFP8, and 255.1k NVFP4 tokens/s: 1.302x and
1.365x. That ratio is the explicitly labelled BF16-optimizer kernel-isolation
ablation; the four-mode production run reports end-to-end behavior with FP32
AdamW state/master weights. These are exact-shape, device-local results rather
than portable guarantees. BF16 is also the convergence baseline; FP32
references are local numerical oracles, not a separate production training
configuration.

[`benchmarks/train_decoder.py`](benchmarks/train_decoder.py) provides the
resumable fixed-token TinyStories comparison for BF16, MXFP8, NVFP4 delayed
scaling, and NVFP4 block-only scaling. Scaling policy is part of each run's
checkpoint identity, so the two NVFP4 results cannot overwrite one another.
The production comparison uses a physical batch of 24 (12,288 tokens per
optimizer step), 37,500 steps, and 375 warmup steps: exactly 460.8M training
tokens and 4.608M warmup tokens. The batch was selected at the measured
low-precision throughput knee rather than from maximum memory occupancy. The
exact invocation and artifact contract are documented
in [`benchmarks/README.md`](benchmarks/README.md).

Shape/stream-specific dynamic runners retain quantized workspaces for reuse.
These caches are bounded to eight entries per execution family by default.
Use `RTX_MXFP8_RUNNER_CACHE_ENTRIES`, or a family-specific variable such as
`RTX_MXFP8_BACKWARD_CACHE_ENTRIES`, to change the bound. Applications which
finish a variable-shape phase can release every retained workspace safely:

```python
released = rtx.clear_runtime_caches()  # synchronizes before releasing buffers
```

Both modules return BF16 and expose the usual no-bias
`nn.Linear(in_features, out_features, bias=False, ...)` shape convention.
All positive M/N/K shapes are supported. Dynamic fused kernels keep the BF16
GMEM tensors at their exact logical shape and predicate or TMA-zero-fill the
final tensor-core tile on chip. Materialized and AOT operands allocate only the
smallest native-MMA storage tail: `ceil(K / 32) * 32` E4M3 values for MXFP8
and `ceil(K / 32) * 32` logical E2M1 values for NVFP4. NVFP4 still uses one
E4M3 scale per 16 values; the 32-value physical boundary keeps every packed
FP4 row 16-byte aligned, appending a zero block when necessary. Original logical
shape is retained as RTX metadata while GEMM sees the physical extent;
there is no `torch.pad`, padded BF16 temporary, or explicit transpose in a
registered forward/backward path. Native blocked scale layouts remain an
aligned fast path, while ragged packed operands use row-major block scales.
Quantization state is independent from training mode:

| State | Activation | Weight | Training | Per-call work |
| --- | --- | --- | --- | --- |
| Dynamic | BF16 | BF16 | Yes | Quantize X and W, then GEMM |
| AOT weight | BF16 | Packed | Inference only | Quantize X, then GEMM |
| Prequantized | Packed | Packed | Inference only | GEMM only |

Dynamic execution is equally valid under `torch.inference_mode()`; inference
does not imply packed weights. Explicit conversion performs weight
quantization exactly once:

```python
dynamic = rtx.MXFP8Linear(1536, 1536, device="cuda")

with torch.inference_mode():
    y_dynamic = dynamic(x)  # BF16 X and W are dynamically quantized

    packed_weight = dynamic.to_quantized_weight()
    y_weight_aot = packed_weight(x)  # only X is dynamically quantized

    packed_x = rtx.quantize_mxfp8(x)
    y_prequantized = packed_weight(packed_x)  # pure packed GEMM
```

`rtx.MXTensor` and the compatibility spelling `rtx.MXFP8Tensor` are direct
aliases of TorchAO's
[`MXTensor`](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/mx_tensor.py).
Likewise, `rtx.NVFP4Tensor` is TorchAO's
[`NVFP4Tensor`](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/nvfp4_tensor.py).
RTX-produced E4M3/E8M0 storage is wrapped without copying; TorchAO blocked
scales are exposed to CuTe through zero-copy kernel-shape views. NVFP4 uses
TorchAO's packed qdata, E4M3 block scale, and optional FP32
`per_tensor_scale` fields; an absent global scale is treated as exactly one.
Packed module weights remain persistent raw buffers and do not retain a BF16
master weight.

Both formats implement all three state boundaries. The decoder and dynamic
BF16 `NVFP4Linear` default to JIT row-region scaling; block-only remains an
explicit throughput-oriented mode that selects materialize-once quantizers and
native NVFP4 GEMM without an outer-scale reduction.
delayed tensorwise scaling remains an explicit compatibility/performance
policy. The first delayed call bootstraps from current amax; later calls
use the configured 16-entry history while producing its successor. Delayed
execution uses the same materialize-once architecture as block scaling: one
persistent dual quantizer reads X and W once, computes native-layout 1x16 E4M3
scales, observes amax during those same reads, and then launches the native
NVFP4 GEMM. Only one atomic maximum per warp and operand is emitted, the old
history is rotated on device, and the power-of-two X/W outer-scale product is
published for the GEMM. The alternate history buffers are cleared with
stream-ordered CUDA operations, so there are no eager tensor reductions or
scale-preparation kernels in the compiled training graph.

Set `scaling="current"` to recompute tensor scales just in time. Set
`scaling="jit_row_region"` (the dynamic default) to compute independent current
FP32 outer scales for bounded X and W row regions. The portable seed is
asymmetric: five X rows and four weight rows. `scale_region_rows=N` requests a
symmetric policy, while `x_scale_region_rows` and
`weight_scale_region_rows` tune the axes independently. The CuTe quantizer gives
one CTA ownership of each region, performs a cooperative current-amax pass,
then rereads warm cache lines to emit packed E2M1 values and native E4M3 1x16
scales. The register-cached producer retains each thread's BF16 fragment across
the cooperative reduction, so current amax and quantization need only one
operand read. The native GEMM multiplies the corresponding X and weight region
factors in FP32 before the sole BF16 conversion. Its portable TMA seed follows
the SM120 accumulator fragment: each thread loads two X factors once, loads one
W factor per four accumulators, and reuses the two FP32 products across adjacent
columns. Direct per-value scaling and dedicated epilogue-warp schedules remain
autotunable alternatives for shapes or devices where layout and register
pressure reverse that advantage. No eager Torch reduction or additional
pointwise output kernel is present.
Region sizes, warp/CTA ownership, observer vector width/unroll/grid/order, quantizer schedule,
native scale transport, and GEMM schedule are jointly tuned by the
`nvfp4_jit_row_region_fwd` family. Set
`scaling="block"` to use an implicit global scale of one and rely only on the
1x16 E4M3 block scales, eliminating both tensor-wide reductions. Block-only is
the fastest policy when values stay inside the E4M3 scale exponent range;
current scaling remains the tensorwide numerical reference for extreme/tiny ranges.
For dynamic BF16 operands, `backend="auto"` selects a pointer-free materialized
block path for block-only execution: either one dual X/W quantizer or two
independent/concurrent quantizers, followed by native NVFP4 GEMM. Its quantizer launch topology,
vector/reduction/register/scale-math schedule, native or row-major scale
transport, GEMM SMEM/RMEM geometry, TMA/mainloop/epilogue schedule, balanced
SM-count persistence, locality, and L2 fetch policy form the
`nvfp4_dynamic_fwd` autotuning family. Native-layout one-wave and three-wave
anchors plus a row-major fallback are measured early, then remain fully
mutable; they prevent a learned search from spending its entire budget in the
older consumer-staged or non-persistent basins.
`backend="auto"` selects this materialized delayed path and its dedicated
`nvfp4_delayed_fwd` winner family.
An explicit `NVFP4ScaleConfig(tensor_scale_mode="exact")` retains the exact
TorchAO tensorwise FP32 scale for controlled studies; power-of-two is the
default because its reciprocal is exact and cheaper. A dynamic module honors
its selected policy during inference; AOT-weight dynamic-X inference supports
current or block-only X scaling (regional prequantized-W epilogues are not yet
implemented), and fully packed inference uses the tensor
scale stored in TorchAO's `NVFP4Tensor`.

On the 70-SM RTX 5070 Ti used for the revision-4 study, putting E4M3 scales
directly in tensor-core-native physical layout and moving them through the
three-stage operand TMA pipeline changed the winning basin. Against the current
materialized MXFP8 baseline, verified hot-input `N=K=1536` timings were 10.30
vs 15.62 us at M=128 (1.516x), 11.51 vs 16.69 us at M=512 (1.450x), 24.71 vs
40.58 us at M=1536 (1.642x), and 109.89 vs 170.90 us at M=8192 (1.555x).
Rotated-input speedups on those shapes were 1.506x, 1.360x, 1.465x, and
1.549x. A 15-sample rotated `(1536, 6144, 1536)` confirmation used the
three-wave anchor and measured 85.07 vs 130.15 us (1.530x). The NVFP4 output
had cosine similarity about 0.991 and normalized RMSE about 0.134 against
FP32. These are device/shape-specific measurements, not portable defaults;
runtime winners remain keyed by the exact hardware/software fingerprint,
shape, and kernel revision.

All four NVFP4 runtime paths support
`torch.compile(fullgraph=True, dynamic=False)`. Tensor-current scale arithmetic
remains compiler-visible FX which Inductor may fuse; JIT-regional observation
and quantization execute inside the registered CuTe producer. Prequantized
modules pass their persistent raw qdata/scales to their
launchers without rebuilding a TorchAO wrapper. AOTAutograd likewise owns the
MXFP8 dX/dW result allocations and calls allocation-free out variants; CuTe
constructs logical transpose layouts from the original contiguous G/W/X
pointers inside its JIT entry, never through eager `.T`, `contiguous`, or
transpose-copy kernels.

MXFP8 uses the public backend terms `auto`, `fused`, and `materialized`;
NVFP4 supports `auto` and `materialized`. Materialized dynamic execution quantizes BF16 operands into
global memory on every call before launching the GEMM; it is not an AOT weight
state. MXFP8 still accepts the historical `prequant` spelling as a compatibility
alias, but modules normalize it to `materialized`.

The frontends also share runtime-tuning controls: `autotune="off"` uses an
explicit or built-in configuration, `"cache"` consumes only an installed
winner, `"balanced"` performs bounded cache-miss tuning, and `"coordinate"`
launches the deep tuner. `"online"` remains an alias for `"balanced"`.
Both accept `tuning_policy` and `autotune_cache_dir`. NVFP4 exposes independent
schedule overrides for its three materialized states:

```python
layer = rtx.NVFP4Linear(
    1536,
    1536,
    device="cuda",
    scaling="block",
    backend="materialized",
    autotune="cache",
    dynamic_config=rtx.NVFP4DynamicConfig(),
    weight_prequant_config=rtx.NVFP4WeightPrequantConfig(),
    fully_prequant_config=rtx.NVFP4FullyPrequantConfig(),
)
```

An explicit state configuration takes precedence over runtime winners. Cached
and tuned winners remain keyed by device/compiler fingerprint, exact shape,
cache regime, operand state, and packed layout. Internal dataset family names
such as `mxfp8_prequant_fwd` are stable schema identifiers and are not renamed
with the public backend vocabulary.

See `rtx/fp8.py` and `rtx/fp8_bwd.py` for backend, configuration, and explicit
backward controls, and `rtx/fp4.py` for NVFP4 scaling and packing policies.

## One-command dataset campaign

Validate a manifest without a GPU launch:

```bash
rtx-autotune validate autotune_manifests/cross_device_dataset_v2.json
```

Probe a machine before a long run:

```bash
rtx-autotune probe --device cuda:0
```

Create a machine-local calibration once. It measures L2/DRAM copy bandwidth,
BF16 tensor throughput, and the native MXFP8 quantization/GEMM pipeline while
retaining the raw timing samples:

```bash
rtx-autotune calibrate --device cuda:0 \
  --output hardware_calibration.json
```

Run or resume the complete assigned campaign:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_v2.json \
  --device cuda:0 \
  --output-dir autotune_datasets \
  --format both \
  --calibration hardware_calibration.json
```

The same campaign engine accepts `nvfp4_dynamic_fwd`, `nvfp4_delayed_fwd`,
`nvfp4_jit_row_region_fwd`, `nvfp4_weight_prequant_fwd`, and
`nvfp4_fully_prequant_fwd` jobs. The delayed family measures the full stateful
observer/quantizer plus GEMM path and always uses one dual quantization launch;
block dynamic and inference families exclude inactive policy coordinates and
one-time AOT operand packing.

New campaigns can exercise the same portable lease/worker boundary used by
external projects while retaining the existing residual stores and campaign
verification:

```bash
rtx-autotune run manifest.json --device cuda:0 \
  --execution-engine ask_tell --output-dir autotune_datasets
```

Long campaigns may also add `--reuse-deterministic-failures`. This writes an
append-only ledger for exact architecture, compiler, kernel revision, workload,
and configuration matches. It reuses only static/compile failures, never
latencies, correctness failures, or runtime failures, and keeps prospective
treatments and replicates isolated.

Use `--format parquet` or `--format both` to emit Parquet datasets.
Every accepted observation is fsync'd to JSONL before the next candidate, so
rerunning the same command resumes after interruption.

For rented or preemptible machines, use the anytime schedule. It visits a
representative, interleaved set of forward, prequant-forward, and backward
contexts breadth-first at absolute trial milestones. Longer runs deepen the
same contexts; shorter runs still cover all kernel families and cache regimes.

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_v2.json \
  --device cuda:0 \
  --output-dir autotune_datasets \
  --format both \
  --calibration hardware_calibration.json \
  --wall-time 12h \
  --context-slice 2m
```

The default milestones are 32, 96, 192, 384, and 512 total trials per
context—not new trials per invocation. The first breadth pass verifies two
finalists, and fully searched contexts use the manifest's full confirmation
policy. The global wall time and every context visit are checkpoints, and
Ctrl-C preserves all observations already printed as `SAVE`.

For new campaigns, the hierarchical bandit manifest makes random exploration,
gradient-boosted global search, and model-guided local search competing,
resumable arms. Adding the context allocator distributes wall time across
shapes while retaining a 32-trial coverage floor and a bounded milestone lead:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_bandit_v1.json \
  --device cuda:0 \
  --output-dir autotune_datasets \
  --format both \
  --calibration hardware_calibration.json \
  --wall-time 12h \
  --context-slice 2m \
  --context-orchestration bandit
```

Use `--strategy-orchestration bandit --strategy-bandit-exploration 0.35` with
an existing manifest to override its per-context scheduler without changing
that manifest's digest. Both levels replay their append-only history after
interruption. Strategy decisions live in each store's `events.jsonl`; context
decisions live in `context_allocations.jsonl` and are exported to CSV/Parquet.

### Prospective 5070 optimizer study

The dedicated study compares pure random coverage, random followed by
coordinate-local search, and the online learned/bandit portfolio. It uses two
replicates and rotates the launch order so every interrupted prefix remains
balanced across optimizer treatment, kernel family, shape category, cache
regime, and replicate.

On the RTX 5070 laptop:

```bash
git pull
./benchmarks/run_5070_autotuner_study.sh laptop-3h
```

On the RTX 5070 Ti:

```bash
git pull
./benchmarks/run_5070_autotuner_study.sh ti-6h
```

Each tuning context owns independent `observations.jsonl`, `sessions.jsonl`,
`events.jsonl`, and `verification.jsonl` residuals. A truncated tail is skipped
and isolated before the next append, so one damaged file cannot corrupt the
campaign or the first resumed observation. `SAVE` records are fsync'd. Pressing
Ctrl-C and running the same command resumes at absolute 4/8/16/32/64-trial
milestones. The script intentionally uses `--format none`; raw residuals are
authoritative and no large monolithic export is rewritten during collection.
Every candidate's ID, complete serialized configuration, features, and search
provenance are fsync'd before evaluation. Sticky CUDA faults such as illegal
instructions are saved against the candidate that caused them, then terminate
only that worker process. A parent watchdog also detects silent generated-kernel
deadlocks, terminates the isolated CUDA process group, and returns restartable
status 75. On resume, any issued candidate without a durable completion is
reported by `audit` and excluded at exact context/config scope rather than
launched again. The launch script automatically starts a fresh CUDA context
with the remaining global wall-time and resumes the existing residuals. Set
`RTX_AUTOTUNE_STALL_TIMEOUT` to increase the inactive threshold. The watchdog
also samples CPU time across the compiler process tree, so an output-silent but
active NVVM compile is retained (subject to a separate 20-minute hard ceiling)
while a blocked CUDA synchronization is restarted. The launcher also preserves compatible context
identity across these runner-only upgrades. The launcher pins `CUDA_HOME` to
the available CUDA 13.2 toolkit when the calling shell omitted it, keeping
libNVVM discovery and the compiler/machine fingerprint stable across restarts.
It also restores the toolkit's `nvcc` path and unbuffers progress output.

After copying the two `autotune_datasets/` trees together, create the optimizer
comparison only once:

```bash
rtx-autotune audit \
  autotune_datasets/mxfp8_autotuner_prospective_5070_v1 \
  laptop-results.zip \
  --output autotune_reports/5070_prospective_v1.audit.json

rtx-autotune summarize-tuners \
  autotune_datasets/mxfp8_autotuner_prospective_5070_v1 laptop-results.zip \
  --output autotune_reports/5070_prospective_v1 \
  --format both
```

After an anytime campaign, compare the best stationary candidate from every
treatment under one higher-resolution protocol before promotion:

```bash
rtx-autotune tournament-winners \
  autotune_manifests/blackwell_prospective_v2.json \
  --device cuda:0 --output-dir autotune_datasets \
  --calibration hardware_calibration.json

rtx-autotune install-winners autotune_datasets --dry-run
```

The tournament is append-only and resumable at each confirmation and paired
race. Partial family runs merge into the same summary. Promotion rejects an
explicitly nonstationary confirmation unless `--allow-nonstationary` is used.

The report includes validity and compiler-waste rates, evaluator time to first
valid candidate, best latency and observed-oracle regret at 1/4/8/16/32/64
trials, and treatment aggregates per machine and kernel family. Schema v3 also
records minimum/median/maximum context coverage, exact shape/cache-regime
splits, compile-failure and wasted-compiler-time rates, matched deltas and the
empirical probability of beating random search with deterministic bootstrap
confidence intervals. A context which silently exhausts early is therefore
visible in the primary report rather than only in its residual journal.

`audit` is read-only. It verifies JSONL tails, interior corruption, machine and
manifest identities, unit coverage, observation/config duplicates, and
repeated confirmation keys before any bundle is ingested. A single malformed
final line is reported as a recoverable crash tail; malformed interior records
or conflicting identities make the command fail.

If this scheduler is pulled while a v2 run made by an older checkout already
exists, explicitly adopt that bundle's context identity so the old observations
count toward the same milestones:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_v2.json \
  --device cuda:0 \
  --output-dir autotune_datasets \
  --format both \
  --calibration hardware_calibration.json \
  --wall-time 12h \
  --context-slice 2m \
  --adopt-existing-context-identity
```

That flag is intentionally opt-in and should only cross runner/autotuner-only
changes. Do not use it after changing kernel implementations or kernel revision
numbers. Reuse the exact same calibration JSON, output directory, manifest, and
shard arguments as the original run.

For multiple processes on one machine, split whole workload contexts:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_v2.json \
  --shard-index 0 --shard-count 2
```

Use the same manifest without sharding on each different GPU when the goal is
to measure identical workload coverage across devices.

## Copying and collecting results

Each run produces one self-contained directory:

```text
autotune_datasets/<campaign>/<machine-id>/shard-000-of-001/
├── manifest.json
├── machine.json
├── summary.json
├── verification.jsonl
├── context_allocations.jsonl
├── dataset.csv
└── stores/
    └── <kernel>/<regime>/
        ├── observations.jsonl
        ├── sessions.jsonl
        └── events.jsonl
```

Copy the entire machine or shard directories back to the main machine. Then
deduplicate and export all copied bundles:

```bash
rtx-autotune collect copied_datasets/ \
  --output merged/mxfp8_blackwell_v2 \
  --format both
```

This writes `merged/mxfp8_blackwell_v2.csv`,
`merged/mxfp8_blackwell_v2.parquet`, and an export report. Rows contain exact
configs, resource/occupancy/wave estimates, traffic and L2 ratios, memory-bus
and calibrated roofline data, raw timing arrays, compiler latency and available
compiled-resource attributes, correctness results, telemetry, proposal
provenance, device/environment context, confirmation measurements, and
paired-race decisions. The JSONL files remain authoritative.

New machine snapshots expose independent `architecture_id`, `device_id`,
`compiler_id`, `environment_id`, `calibration_id`, and `kernel_source_id`
fields under `identities`. The compatibility `machine_id` still selects the
bundle directory. Offline models should group by the narrowest identity needed
for a claim instead of treating a compiler PATH or calibration change as a new
physical GPU.

Campaigns using `storage_mode: residual_context` place the same four journals
under one directory per family/treatment/replicate/category/regime/shape/context
instead of the shared `stores/<kernel>/<regime>` path shown above.

## Pretraining the autotuner

The CPU-only pretrainer accepts copied directories and ZIP archives directly.
Filter to compatible campaigns so obsolete or interrupted kernel revisions
cannot enter the artifact accidentally. Repeat `--campaign` when a device used
an earlier campaign name but the same kernel revisions and feature schema:

```bash
rtx-autotune pretrain autotune_datasets autotune_datasets.zip \
  --campaign mxfp8_blackwell_cross_device_bandit_v1 \
  --output autotune_models/mxfp8_blackwell_bandit_v1 \
  --seed 20260809
```

It writes revision-scoped absolute-latency and context-ranking models, exact-SKU
heads validated on unseen contexts, an end-to-end feasibility classifier,
confidence-qualified paired-coordinate rules, and leave-one-device-out
validation metrics. Unknown SKUs only receive models that passed cross-device
replay; an exact SKU model must separately beat random replay in at least three
of four context-held-out folds. Use the artifact as a soft proposal prior:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_bandit_v1.json \
  --device cuda:0 --output-dir autotune_datasets --format both \
  --wall-time 2h --context-orchestration bandit \
  --pretrained-artifact autotune_models/mxfp8_blackwell_bandit_v1
```

Only a head whose median and p90 regret both beat matched random catalogue
replay on every held-out device
is allowed to propose from the first model-guided trial. Otherwise the family
keeps its ordinary online model and uses only validated feasibility/rule priors.
Four initial random trials retain local exploration for an enabled model, and
measured local data refits it after the configured adaptation interval.
Conditional rules only adjust ranking; they never reject a legal candidate or
install an unverified winner.
Artifacts are ignored by Git because they are reproducible from the source
JSONL datasets.

Keep a prospective study out of training, then evaluate the frozen artifact on
it explicitly:

```bash
rtx-autotune evaluate-pretrained \
  autotune_models/mxfp8_blackwell_bandit_v1 \
  prospective-5070-ti/ prospective-5070-laptop.zip \
  --output autotune_reports/mxfp8_blackwell_bandit_v1.heldout.json
```

The command rejects identical dataset digests or overlapping source files by
default and records both input identities. It never changes the artifact's
deployment gates.

After audit and numerical verification, preview and install runtime winners:

```bash
rtx-autotune verify-winners \
  autotune_manifests/mxfp8_bwd_revision19_calibration_v1.json \
  --device cuda:0 --output-dir autotune_datasets \
  --calibration hardware_calibration.json --promote 4
rtx-autotune install-winners copied_datasets/ laptop-results.zip \
  --minimum-support 2 --dry-run
rtx-autotune install-winners copied_datasets/ laptop-results.zip \
  --minimum-support 2
```

Promotion is atomic and keyed by kernel family, exact device fingerprint,
shape, cache regime, and packed scale-layout variant. Existing entries are not
overwritten unless `--force` is supplied.
`verify-winners` is a post-search GPU pass for interrupted anytime campaigns:
it confirms the current residual-journal finalists, records paired races, and
refreshes bundle-local winners without consuming additional search trials.

## Dataset manifest

The portable manifest can contain forward and backward jobs. Each job has its
own shapes, cache regimes, benchmark protocol, search policy, and finalist
count:

```json
{
  "schema_version": 2,
  "name": "example",
  "seed": 20260808,
  "jobs": [
    {
      "family": "mxfp8_prequant_fwd",
      "shapes": [{"m": 512, "n": 1536, "k": 1536}],
      "regimes": ["hot", "rotate"],
      "promote": 4,
      "tuning": {
        "max_trials": 128,
        "time_budget_s": 1800,
        "cost_model_trials": 80
      },
      "protocol": {
        "samples": 7,
        "confirm_samples": 15,
        "race_rounds": 15,
        "target_batch_ms": 50
      }
    }
  ]
}
```

Supported families are `mxfp8_fused_fwd`, `mxfp8_prequant_fwd`,
`mxfp8_weight_prequant_fwd`, `mxfp8_fully_prequant_fwd`, `mxfp8_bwd`,
`nvfp4_dynamic_fwd`, `nvfp4_delayed_fwd`, `nvfp4_jit_row_region_fwd`,
`nvfp4_weight_prequant_fwd`, and `nvfp4_fully_prequant_fwd`. Persistent
inference families never time their AOT
packing work and do not expose inactive quantizer coordinates. Additional
kernel families can register a `DatasetBackend` with
`rtx.autotune.register_dataset_backend`; campaign orchestration, persistence,
verification, and export do not need to change.

After the active v2 campaign is collected, run the bounded inference pilot:

```bash
rtx-autotune run autotune_manifests/inference_states_pilot_v1.json \
  --device cuda:0 --output-dir autotune_datasets --format both
```

Every verified winner is also written below the bundle's `runtime_winners/`
directory. Point a packed layer's `autotune_cache_dir` at that bundle to use a
matching hot-regime winner. Cache identity includes exact device/software
fingerprint, M/N/K, operand state, and physical scale layouts.

## Tests

CPU-safe contract and serialization tests:

```bash
python -m unittest discover -s tests -v
```

GPU kernel tests are skipped automatically when a compatible CUDA target is
not available.

Additional kernel design and autotuning details are in `AUTOTUNING.md` and
`rtx/autotune/README.md`. The RTX-specific hardware boundary and upstream
attribution are documented in `sm120_hardware.md`.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
