# Model conversion

RTX provides TorchAO-style top-level conversion for bias-free BF16 linear
layers. A filter receives each module and its fully qualified name, and only
selected `nn.Linear` instances are replaced.

RTX linear layers never implement bias. If a selected module has a bias, the
whole conversion fails during preflight before any module is replaced. Leave
such modules unselected or change the source architecture intentionally; RTX
does not silently drop or separately add the bias.

## Dynamic training

```python
import torch
import rtx

def transformer_mlp(module: torch.nn.Module, fqn: str) -> bool:
    return (
        isinstance(module, torch.nn.Linear)
        and module.bias is None
        and ".mlp." in fqn
    )

model = model.to(device="cuda", dtype=torch.bfloat16)
model = rtx.convert_to_mxfp8_training(
    model,
    module_filter_fn=transformer_mlp,
    config=rtx.MXFP8TrainingConfig(
        backend="auto",
        autotune="cache",
    ),
)
compiled = torch.compile(model, fullgraph=True, dynamic=False)
```

For NVFP4 forward with MXFP8 backward:

```python
model = rtx.convert_to_nvfp4_training(
    model,
    module_filter_fn=transformer_mlp,
    config=rtx.NVFP4TrainingConfig(
        scaling="jit_row_region",
        backend="auto",
        autotune="cache",
    ),
)
```

Training conversion retains the exact source `Parameter` object. Existing
optimizer references, weight sharing, `requires_grad`, initialization, and
the ordinary `weight` state-dict key are preserved. Constructing the optimizer
before or after conversion is therefore supported.

TorchAO `Float8Linear` is an `nn.Linear` subclass and normally retains its BF16
master `Parameter`, so a rowwise-trained TorchAO model can be converted
directly. FSDP-specific tensor-wrapped weights and distributed placements are
outside the current RTX contract; load their BF16 checkpoint into an ordinary
local model before conversion.

## Post-training quantization

MXFP8 PTQ packs the weight once while quantizing BF16 activations dynamically:

```python
model.eval().to(device="cuda", dtype=torch.bfloat16)
model = rtx.quantize_(
    model,
    rtx.MXFP8WeightOnlyConfig(autotune="cache"),
    filter_fn=transformer_mlp,
)
```

NVFP4 PTQ offers current tensor scaling or block-only scaling:

```python
model = rtx.quantize_(
    model,
    rtx.NVFP4WeightOnlyConfig(
        scaling="current",  # or "block"
        autotune="cache",
    ),
    filter_fn=transformer_mlp,
)
```

`WeightOnlyConfig` describes persistent storage, not GEMM precision. The BF16
weight is removed, while a BF16 activation is dynamically quantized into the
selected low-precision format at inference time. Packed modules are
inference-only and reject dtype conversion.

## Replacement behavior

- The default filter selects every `nn.Linear` subclass, including TorchAO
  `Float8Linear`.
- Passing a filter is strongly recommended for architectures containing bias.
- A root linear cannot be replaced in place; always use the returned module.
- Shared module aliases remain shared. A filter that selects only some aliases
  of the same object is rejected.
- PTQ requires selected BF16 weights to be on CUDA. Pass `device="cuda"` or
  move the model before conversion.
- Training conversion may happen on CPU before a later `.to(device="cuda")`.
- All selected modules and all replacements are prepared before the module
  tree is changed, preventing partially converted models after an error.

## Torchvision ViT boundary

Torchvision Vision Transformers use biases in their attention output, MLP,
and classifier linears. A blanket RTX conversion therefore fails by design.
Select only projections made bias-free by the source architecture. The test
suite constructs a one-layer small ViT, selects two intentionally bias-free
MLP projections, and validates:

- MXFP8 dynamic training under fullgraph compilation;
- MXFP8 packed-weight PTQ under fullgraph compilation;
- NVFP4 packed-weight PTQ under fullgraph compilation;
- parameter identity and state-dict compatibility for both training formats.
