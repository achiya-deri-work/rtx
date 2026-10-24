# RTX Blackwell documentation

Install **rtx-blackwell**, import **rtx**. Bias-free MXFP8/NVFP4 linear layers
for RTX Blackwell, with a reusable GPU autotuner. Dynamic Torch-side inputs,
parameters, outputs and gradients are BF16; NVFP4 uses MXFP8 backward.

## User manual

Start with [installation](installation.md) and [quickstart](quickstart.md).
The [API map](api.md) owns public names and shared frontend controls;
the [generated reference](api_reference.md) owns exact signatures and defaults.

| Task | Canonical guide |
|---|---|
| Use the linear layers | [MXFP8Linear](mxfp8_linear.md) · [NVFP4Linear](nvfp4_linear.md) |
| Convert a model for training or PTQ | [Model conversion](model_conversion.md) |
| Calibrate NVFP4 and select numerical policies | [Quantization policy](quantization_policy.md) |
| Review measured NVFP4 strengths, weaknesses and external recipes | [Broad evaluation, 2026-10-23](nvfp4_broad_evaluation_20261023.md) |
| Choose dynamic, packed-weight or fully packed inference | [Inference](inference.md) |
| Understand precision | [Training](training.md) · [Formats](formats.md) |
| Configure effort and cache reuse/refinement | [First-hit autotuning](first_hit_autotuning.md) |
| Compile and use graphs, streams and caches | [Execution](execution.md) |
| Collect datasets and deploy winners | [Autotuning operations](autotuning.md) |
| Diagnose errors and limitations | [Troubleshooting](troubleshooting.md) · [Runtime contract](runtime_contract.md) |

Linears replace **bias-free** nn.Linear under the documented dtype/device
constraints, not every PyTorch/distributed configuration. Calling eval() does
not pack weights. Operand state, numerical policy, backend and tuning effort
are separate choices.

## Feature APIs

- [MXFP8GLU](mxfp8_glu.md): paired projection and activation, BF16/MXFP8 output,
  forward only.
- [MXFP8MLP](mxfp8_mlp.md): packed hidden connection with recomputed backward.

Both default to tuning off, with opt-in first-hit selection. They remain explicit
imports, outside the stable wildcard surface. The optional packed-down MLP
singleton failure is recorded in [numerical qualification](mxfp8_mlp_numerics.md).

## Development and evidence

- [Development](development.md), [autotuner framework](../rtx/autotune/README.md),
  [manifest catalogue](../autotune_manifests/README.md), [benchmarks](../benchmarks/README.md).
- [Research and qualification index](research.md): experiments and historical
  results, separate from current API instructions.
- [Release checklist](release_checklist.md), [precision policy](../PRECISION_POLICY.md).
- [MXFP8/GLU merge qualification](mxfp8_glu_merge_qualification.md): current
  scheduling transfer and cross-stack validation status.

Native kernels target SM120/SM121 on Linux with the supported CUDA 13.2 stack.
SM110/Thor and datacenter Blackwell are not supported. Import is CPU-safe;
native execution is CUDA-only. Source: [Apache-2.0](../LICENSE).
[SM120 notes](../sm120_hardware.md) credit Tri Dao's Quack project.
Optional [DINOv3 tests](dinov3_regression.md) need separately obtained assets.
The current primary application target is [Ettin](ettin_qualification.md);
its checkpoint is downloaded separately and is not included in RTX.
