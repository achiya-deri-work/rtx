# `rtx.MXFP8Linear`

`MXFP8Linear` is a no-bias BF16 linear for SM120/SM121. Dynamic training keeps
a BF16 master weight, quantizes X and W to E4M3 with E8M0 block scales, executes
MXFP8 forward and backward GEMMs, and returns BF16 tensors.

## Quick start

```python
import torch
import rtx

layer = rtx.MXFP8Linear(1536, 3072, device="cuda")
x = torch.randn(64, 512, 1536, device="cuda", dtype=torch.bfloat16)

compiled = torch.compile(layer, fullgraph=True, dynamic=False)
y = compiled(x)
y.float().square().mean().backward()
```

The default uses `backend="auto"` and `autotune=None`. Auto selects the
materialized dynamic quantize-once pipeline when legal and otherwise the fused
kernel. Cache mode loads a compatible verified winner without benchmarking.

## Constructor

```python
rtx.MXFP8Linear(
    in_features,
    out_features,
    bias=False,
    *,
    device=None,
    dtype=torch.bfloat16,
    forward_config=None,
    config=None,  # compatibility alias
    autotune=None,
    tuning_policy=None,
    autotune_cache_dir=None,
    backend="auto",
    dynamic_config=None,
    prequant_config=None,  # compatibility alias
    backward_config=None,
    packed_weight=None,
)
```

- `in_features`, `out_features`, and leading input dimensions follow
  `nn.Linear`.
- `bias` must be false. The module is a drop-in replacement only for no-bias
  BF16 linears.
- `dtype` must be `torch.bfloat16` and is the master-weight dtype.
- `forward_config` fixes a fused dynamic forward schedule. `config` is its
  compatibility alias; passing both is an error.
- `dynamic_config` fixes the materialized dynamic quantizer/GEMM schedule.
  `prequant_config` is its compatibility alias; passing both is an error.
- `backward_config` fixes the MXFP8 dX/dW schedule.
- `packed_weight` creates an inference-only module with no BF16 Parameter.

Low-level configs should normally remain unset. They exist for reproducible
kernel studies and promoted winners.

## Backends

| Backend | Dynamic per-call work | Intended use |
| --- | --- | --- |
| `auto` | Selects the legal production path | Default |
| `materialized` | Quantize X/W once into global storage, then GEMM | Production and broad autotuning |
| `fused` | CTA-local fused BF16 quantization and GEMM | Kernel studies and fallback |

`prequant` is a deprecated compatibility spelling for `materialized`. It does
not mean the weight was quantized ahead of time.

## Operand states

| State | X | W | Training | Timed work |
| --- | --- | --- | --- | --- |
| Dynamic | BF16 | BF16 | Yes | Quantize X/W and GEMM |
| Weight prequantized | BF16 | MXFP8 | No | Quantize X and GEMM |
| Fully prequantized | MXFP8 | MXFP8 | No | GEMM only |

```python
dynamic = rtx.MXFP8Linear(1536, 3072, device="cuda").eval()

with torch.inference_mode():
    y_dynamic = dynamic(x)
    packed_weight = dynamic.to_quantized_weight()
    y_aot_weight = packed_weight(x)
    packed_x = rtx.quantize_mxfp8(x)
    y_fully_packed = packed_weight(packed_x)
```

`MXFP8Linear.from_float(module)` converts a no-bias BF16 `nn.Linear` directly
to the inference-only packed-weight state. Packed modules reject training and
dtype conversion; move them with `.to(device=...)`.

## TorchAO-style PTQ conversion

For inference, convert selected existing BF16 `nn.Linear` modules in place:

```python
import torch
import rtx

model.eval().to(device="cuda", dtype=torch.bfloat16)
model = rtx.quantize_(
    model,
    rtx.MXFP8WeightOnlyConfig(autotune="cache"),
    filter_fn=lambda module, fqn: (
        isinstance(module, torch.nn.Linear)
        and fqn in {"decoder.layers.0.mlp.up_proj", "decoder.layers.0.mlp.down_proj"}
    ),
)
compiled = torch.compile(model, fullgraph=True, dynamic=False)
```

Each selected weight is quantized exactly once to E4M3 with E8M0 block scales;
its BF16 master copy is removed. BF16 activations are still quantized just in
time on every invocation, so "weight-only" describes persistent checkpoint
storage rather than the GEMM operand precision.

The default filter converts `nn.Linear` instances and subclasses. A custom filter takes
`(module, fully_qualified_name)`. RTX linear layers never support bias, so a
selected biased linear is rejected during preflight. Selected weights must
already be BF16 CUDA tensors, or pass `device="cuda"` after converting the
model to BF16. The function returns the model because selecting a root linear
requires replacing the root object.

For dynamic training conversion, NVFP4 PTQ parity, TorchAO-rowwise models, and
optimizer/weight-sharing guarantees, see [the model conversion guide](model_conversion.md).

## Autotuning

- `autotune=None` follows `RTX_MXFP8_AUTOTUNE`/`RTX_AUTOTUNE`, defaulting to
  `balanced`.
- `off` uses an explicit or portable configuration.
- `cache` loads compatible verified winners and never benchmarks unexpectedly.
- `balanced` checks the exact runtime cache first, explores at most 24 candidates
  or 30 seconds on a miss, confirms the winner, and persists it atomically.
  Repeated calls use the process-local selection without another lookup.
- `online` is a compatibility alias for `balanced`.
- `coordinate` is the explicit deep-search mode and retains the legacy
  30-minute default when no custom policy is supplied.
- `tuning_policy` and `autotune_cache_dir` customize online orchestration and
  winner storage.

The balanced limits can be changed with `RTX_BALANCED_AUTOTUNE_TRIALS` and
`RTX_BALANCED_AUTOTUNE_SECONDS`. Screening defaults to five adaptive samples;
`RTX_BALANCED_AUTOTUNE_WARMUP`, `RTX_BALANCED_AUTOTUNE_SAMPLES`, and
`RTX_BALANCED_AUTOTUNE_CALLS_PER_SAMPLE` expose the remaining first-hit cost.
Set `RTX_AUTOTUNE_PRETRAINED_ARTIFACT` to a validated portable model bundle to
seed the learned arms; without it, the same policy learns only from the current
context and any resumable local journal.

Exact winner selection is keyed by device/software identity, kernel revision,
shape, operand state, and physical scale layouts. It remains opaque through
`torch.compile` and resolves inside the registered runtime launcher.

Use side-effect-free introspection before running a shape:

```python
decision = layer.explain(x)
print(decision.family, decision.backend, decision.selection_source)
```

`selection_source="deferred_runtime"` means cache lookup or online tuning will
occur later; `explain()` itself never compiles or benchmarks.

## Packed tensor contract

`rtx.MXFP8Tensor` is an alias of TorchAO's `MXTensor`. RTX preserves the
logical ragged shape while allocating the minimal K multiple of 32. E8M0 scales
may be row-major or tensor-core-native. Packed weights serialize as raw qdata,
scale, and versioned layout metadata buffers.

## Initialization and checkpoints

Dynamic modules use the same Kaiming-uniform weight initialization as a
no-bias `nn.Linear` and expose the ordinary `weight` state-dict key. This lets
BF16, MXFP8, and NVFP4 dynamic models exchange a common checkpoint. Packed
modules use a distinct inference checkpoint schema and do not retain the BF16
master weight.
