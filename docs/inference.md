# Inference, PTQ, and checkpoints

## Three operand states

| State | X | W | Per-call work | Backward |
|---|---|---|---|---|
| Dynamic | BF16 | BF16 | Quantize both and GEMM | Supported |
| Packed weight | BF16 | Packed | Quantize X and GEMM | Not supported |
| Fully packed | Packed | Packed | GEMM | Not supported |

All return BF16. Dynamic inference is valid: `.eval()` does not imply weight
packing. `WeightOnlyConfig` refers to stored weights, not W8A16/W4A16 arithmetic;
the BF16 X is still quantized for GEMM. Do not infer W4A8 from that API name.

## Convert

For one layer, `MXFP8Linear.from_float(bf16_linear)` creates an inference-only
packed-weight module; it is not the training conversion. For whole models,
use `quantize_` with `MXFP8WeightOnlyConfig` or `NVFP4WeightOnlyConfig`.
Always assign the return value, since the selected root may be replaced.

```python
import torch
import rtx

source = torch.nn.Linear(128, 256, bias=False, device="cuda",
                         dtype=torch.bfloat16).eval()
x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    packed = rtx.MXFP8Linear.from_float(source, autotune="cache")
    y = packed(x)
    xq = rtx.quantize_mxfp8(x)
    yq = packed(xq)
```

No BF16 master weight is kept by the packed module. Keep your source checkpoint
if you will train again or requantize. Standalone packing is explicit setup
work; precompute packed weights outside compiled model execution.

NVFP4 packed weights use current or block scaling. Converting a dynamic
JIT-region/delayed module with `to_quantized_weight()` selects current scaling;
block stays block. Explicit `scaling="block"` packs an outer factor of exactly
one. Fully packed execution consumes the encoded scales. See
[NVFP4Linear](nvfp4_linear.md) for policy overrides.

## Save and restore

Dynamic linears use the ordinary `weight` state-dict key and can share BF16
checkpoints with a matching no-bias architecture. Packed modules use versioned
raw buffers and layout metadata instead. Load packed state into an architecture
constructed in the same packed state, not a fresh dynamic module.

```python
torch.save(packed.state_dict(), "packed_weights.pt")
# Construct a same-shape packed template; its initial quantization is replaced.
with torch.inference_mode():
    restored = rtx.MXFP8Linear.from_float(source, autotune="cache")
restored.load_state_dict(torch.load("packed_weights.pt", weights_only=True))
restored.eval()
```

The application must save architecture, activation choices, scaling policy,
and software provenance separately. Do not assume arbitrary Python module
pickles or private TorchAO tensor attributes are a stable checkpoint format.
Use trusted checkpoints; `weights_only=True` limits object deserialization but
does not establish provenance or make arbitrary artifacts trustworthy.

Packed modules reject training and dtype conversion. Move device only with
`.to(device=...)`; do not cast their byte/scale buffers to BF16. Runtime winners
are not checkpoint weights and should be reselected for the destination GPU.
NVFP4 delayed history is nonpersistent and resets after loading dynamic state.

## Shape and layout

The last dimension is input features; leading dimensions flatten to M and are
restored at output. Positive ragged dimensions use logical padding rather than
an eager BF16 pad. Packed physical storage and its scale layout must match the
kernel ABI; arbitrary packed views, transposes or sharding are not supported
just because the object remains a TorchAO tensor subclass.

Use the public quantizers to construct RTX-compatible tensors. A packed-X /
dynamic-W mixture is not one of the three documented inference states.
See [formats](formats.md) and [checkpoint schema details](runtime_contract.md).

## Deploy

Warm representative shapes, validate outputs on representative data, install
verified device-specific winners, then compile/capture the inference workload.
Keep the original BF16 baseline and report error distributions, not only mean
cosine. Optional [DINOv3 regression](dinov3_regression.md) demonstrates a model
test; its source and weights are external assets, not bundled dependencies.
