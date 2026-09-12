# RTX Blackwell

**Distribution:** `rtx-blackwell` · **Import:** `rtx` · **License:** Apache-2.0

Bias-free MXFP8 and NVFP4 linear layers for NVIDIA RTX Blackwell, with a
persistent, composable GPU autotuner. Dynamic model tensors remain BF16;
kernels quantize operands internally and accumulate with the required FP32
precision.

**[Read the documentation](docs/README.md)** ·
[Quickstart](docs/quickstart.md) · [API reference](docs/api.md) ·
[Troubleshooting](docs/troubleshooting.md)

## What is supported?

| Frontend | Forward | Backward | Torch-side dynamic tensors |
|---|---|---|---|
| `rtx.MXFP8Linear` | MXFP8 E4M3/E8M0 | MXFP8 | BF16 |
| `rtx.NVFP4Linear` | NVFP4 E2M1/E4M3 + policy-dependent FP32 outer scales | MXFP8 | BF16 |

Both replace **bias-free BF16** `nn.Linear` modules, not arbitrary linears.
They support dynamic training/inference, prequantized-weight inference, and
fully prequantized inference. Leading dimensions are restored at output;
positive ragged shapes use logical padding.

Native kernels target **SM120/SM121** on Linux. SM110/Jetson Thor and datacenter
Blackwell are outside this implementation's scope. This remains a development
project: read the [release checklist](docs/release_checklist.md) before treating
local qualification as a released support guarantee.

## Install from this checkout

Supported stacks are Torch **2.13.0+cu132** or **2.12.1+cu132**, TorchAO **0.18.0**,
and `nvidia-cutlass-dsl[cu13]==4.7.0`. Python metadata requires 3.11+.
The requirements include PyArrow, Apache TVM FFI, CUDA Python, NumPy and Einops.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
rtx-autotune probe
```

Use `requirements-torch212-cu132.txt` in a separate environment for Torch
2.12.1. A bare package install cannot choose the required CUDA-local Torch
wheel for you. See [installation](docs/installation.md).

## Train

```python
import torch
import rtx

layer = rtx.MXFP8Linear(128, 256, device="cuda", autotune="cache")
x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
compiled = torch.compile(layer, fullgraph=True, dynamic=False)
compiled(x).float().square().mean().backward()
```

Use `NVFP4Linear` for NVFP4 forward and MXFP8 backward. Its dynamic default
is current JIT-regional scaling with independent X=5/W=4 row groups.
Tensor-delayed, block-only and tensor-current policies remain explicit options.
See the [NVFP4 constructor guide](docs/nvfp4_linear.md) for their numerical
tradeoffs and packed-weight restrictions.

To convert selected existing modules, use
`convert_to_mxfp8_training` / `convert_to_nvfp4_training`.
Training conversion preserves source parameter objects; selected biased modules
fail preflight. [Conversion guide](docs/model_conversion.md).

## Inference and PTQ

```python
dynamic = rtx.MXFP8Linear(128, 256, device="cuda", autotune="cache").eval()
with torch.inference_mode():
    packed = dynamic.to_quantized_weight()
    y = packed(x.detach())                    # dynamic X, packed W
    xq = rtx.quantize_mxfp8(x.detach())
    yq = packed(xq)                           # both packed
```

Whole-model PTQ uses `rtx.quantize_(model, rtx.MXFP8WeightOnlyConfig(), ...)`
or `NVFP4WeightOnlyConfig`. “Weight-only” describes persistent storage:
activations are still quantized for the low-precision GEMM. Packed modules
are inference-only; keep your BF16 source checkpoint for future training.
[Inference and checkpoint guide](docs/inference.md).

## Autotune

Linear defaults use bounded balanced first-hit search: exact cache lookup,
then up to 24 trials / 30 seconds of search budget on a miss, with confirmation
and persistent winners. These budgets do not interrupt an in-flight compiler.
`autotune="cache"` forbids searches; `"off"` selects explicit/portable settings.
Compilation is still lazy in both modes.

The reusable `rtx.autotune` framework includes conditional spaces, staged
validation, random/model/local search, bandit allocation, durable residual
journals, resumable campaigns, ask/tell workers, and CSV/Parquet exports.
[Runtime and dataset workflow](docs/autotuning.md) ·
[Framework extension reference](rtx/autotune/README.md).

## GLU / recomputed MLP feature branch

- `MXFP8GLU`: fused gate/up + SiLU/GELU/ReLU; BF16 or MXFP8 output;
  **forward only**. [Guide](docs/mxfp8_glu.md).
- `MXFP8MLP`: two-kernel forward, packed MXFP8 hidden connection, BF16 output,
  fully recomputed first-order backward. Supports `from_linears`, explicit
  `autotune_`, and revisioned JSON policies. [Guide](docs/mxfp8_mlp.md).

Unlike linears, GLU/MLP tuning is explicit, not automatic per new shape.
These feature APIs are not in the stable wildcard-export surface.
**Known blocker:** the optional packed-down MLP schedule faults on
M=K=H=O=1; the conservative default passes. Read the
[numerical stress report](docs/mxfp8_mlp_numerics.md).

## Evidence and development

Low-precision implementation error, precision loss versus BF16, model
convergence, and speedup are separate measurements. FP32 references are
numerical oracles, not an all-FP32 training baseline. Optimizer-state precision
is selected by the training application. [Precision policy](PRECISION_POLICY.md).

Documentation examples can be run on a supported GPU:

```bash
python docs/examples/basic_workflows.py --mode all
python -m unittest tests.test_documentation -v
```

[Developer guide](docs/development.md) ·
[Benchmarks and decoder training](benchmarks/README.md) ·
[Campaign catalogue](autotune_manifests/README.md).

Pretrained DINOv3 source/checkpoints are optional, separately licensed test
assets and are not shipped. [DINOv3 tests](docs/dinov3_regression.md).
The [SM120 hardware notes](sm120_hardware.md) credit Tri Dao's Quack library.

See [LICENSE](LICENSE) and [CHANGELOG](CHANGELOG.md).
