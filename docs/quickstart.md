# Quickstart

Run these after [installation](installation.md). CUDA examples assume an
available supported GPU. Start with MXFP8, establish a BF16 baseline, then
evaluate NVFP4 on representative data.

## One linear with gradients

```python
import torch
import rtx

layer = rtx.MXFP8Linear(128, 256, device="cuda", autotune="cache")
x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
compiled = torch.compile(layer, fullgraph=True, dynamic=False)
y = compiled(x)
y.float().square().mean().backward()
assert y.dtype == x.grad.dtype == layer.weight.grad.dtype == torch.bfloat16
```

Substitute `rtx.NVFP4Linear` for NVFP4 forward with MXFP8 backward. No manual
activation quantization is needed. Both dynamic layers store BF16 parameters,
return BF16, and reject bias. `autotune="cache"` loads a verified winner or
uses a portable configuration; omit it for the bounded balanced first-hit
search. First use still includes kernel compilation in cache mode.

## Convert selected model linears

```python
model = torch.nn.Sequential(
    torch.nn.Linear(128, 256, bias=False),
    torch.nn.SiLU(),
    torch.nn.Linear(256, 128, bias=False),
).to(device="cuda", dtype=torch.bfloat16)

model = rtx.convert_to_mxfp8_training(
    model, config=rtx.MXFP8TrainingConfig(autotune="cache")
)
```

Use `module_filter_fn(module, fqn)` to select projections in a larger model.
Training conversion preserves the source `Parameter` objects and optimizer
references. Selected biased linears fail preflight; RTX never discards bias.
See [conversion](model_conversion.md) before converting tied or sharded models.

## Pack weights for inference

```python
dynamic = rtx.MXFP8Linear(128, 256, device="cuda", autotune="cache").eval()
with torch.inference_mode():
    packed = dynamic.to_quantized_weight()
    y = packed(x.detach())                 # dynamic activation quantization
    xq = rtx.quantize_mxfp8(x.detach())
    yq = packed(xq)                        # both operands already packed
```

Packing is explicit; `.eval()` alone does not pack weights. The packed module
has no BF16 master parameter and is inference-only. NVFP4 offers the same
three operand states; see its [scaling restrictions](nvfp4_linear.md).

## Fully recomputed GLU MLP (feature branch)

```python
mlp = rtx.MXFP8MLP(128, 256, device="cuda", activation="silu")
# Explicit search, before compile/capture; optional:
# mlp.autotune_(x.detach(), max_trials=32, time_budget_s=120)
compiled_mlp = torch.compile(mlp, fullgraph=True, dynamic=False)
compiled_mlp(x).float().square().mean().backward()
```

This saves input/weight references, not hidden activations. It is not the
forward-only `MXFP8GLU`. Read the [MLP contract and known blocker](mxfp8_mlp.md).
Unlike linears, its default is `autotune="off"`. For opt-in per-shape first-hit
selection, pass `autotune="balanced"`, `tuning_policy=rtx.AutotunePolicy.preset("quick")`,
and `autotune_cache_dir=".rtx-cache"` to the constructor. Automatic selection
currently requires MXFP8 precision; explicit tuning remains available.

## Runnable examples

From the repository root:

```bash
python docs/examples/basic_workflows.py --mode training
python docs/examples/basic_workflows.py --mode inference
python docs/examples/basic_workflows.py --mode mlp
```

These are finite-value/gradient smoke tests, not convergence or throughput
benchmarks. The [training guide](training.md) explains the evidence needed
before deploying low precision in a real model.
