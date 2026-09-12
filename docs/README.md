# RTX Blackwell documentation

Install **`rtx-blackwell`**, import **`rtx`**. This library supplies bias-free
low-precision linear layers for RTX Blackwell and a reusable GPU autotuner.
Torch-side dynamic parameters, activations, outputs and gradients are BF16.
MXFP8 executes forward/backward GEMMs in MXFP8; NVFP4 executes forward in NVFP4
and backward in MXFP8. No library-wide speedup or convergence guarantee is made.

## Start here

1. [Install and verify the environment](installation.md).
2. [Run a layer, convert a model, and compile it](quickstart.md).
3. Choose [MXFP8Linear](mxfp8_linear.md) or [NVFP4Linear](nvfp4_linear.md).
4. Read [training and numerical policy](training.md) or
   [inference, packing, and checkpoints](inference.md).
5. [Tune, collect datasets, and deploy winners](autotuning.md).

## User reference

| Topic | Guide |
|---|---|
| Every top-level export and its purpose | [API index](api.md) |
| Exact callable signatures and configuration defaults | [Generated API reference](api_reference.md) |
| Selective training conversion and PTQ | [Model conversion](model_conversion.md) |
| Formats, scale hierarchy, and backward reduction axes | [Numerical formats](formats.md) |
| Compilation, CUDA graphs, streams, and caches | [Execution guide](execution.md) |
| Checkpoint schemas and distributed limitations | [Runtime contract](runtime_contract.md) |
| Errors, slow first calls, and debugging | [Troubleshooting](troubleshooting.md) |
| Tests, contributions, and adding a kernel family | [Development guide](development.md) |

## Feature-branch APIs

These APIs are explicitly importable as `rtx.<name>` but are not part of the
stable wildcard-export surface. They have separate qualification gates.

- [MXFP8GLU](mxfp8_glu.md): fused gate/up and SiLU/GELU/ReLU, BF16 or packed
  MXFP8 output; **forward only**.
- [MXFP8MLP](mxfp8_mlp.md): two-kernel forward, fully recomputed first-order
  backward, weight conversion, explicit autotuning and policy persistence.
- [MLP numerical stress results and known singleton failure](mxfp8_mlp_numerics.md).
- [Bounded-input divergence and precision interventions](mxfp8_mlp_divergence.md).
- [TorchAO-quantized MLP recipe parity and residual-stream error](mxfp8_mlp_recipe_parity.md).
- [Selective gate and hidden precision experiments](mxfp8_mlp_selective_precision.md).
- [Measured mixed-precision MLP schedules and compiler caveats](mxfp8_mlp_quality_runtime.md).
- [SiLU/GELU multi-shape numerics and long-reduction tuning](mxfp8_mlp_multishape.md).
- [Selective-precision API, ragged schedules, and decoder training](mxfp8_mlp_precision_api.md).
- [MLP optimization evidence](mxfp8_mlp_optimization.md).

The optional packed-down MLP policy currently faults for M=K=H=O=1. The default
policy passes this case. Do not interpret logical shape support as proof that
every experimental schedule is safe.

## Autotuner developers and experiment operators

- [Portable adapters, conditional spaces, bandits, and ask/tell](../rtx/autotune/README.md)
- [Campaign manifest catalogue](../autotune_manifests/README.md)
- [Benchmark and decoder-training commands](../benchmarks/README.md)
- [Cross-SKU analysis](autotune_cross_sku_study.md) and
  [evidence methodology](autotune_evidence_methodology.md)
- [Historical kernel-tuning notebook](../AUTOTUNING.md)

## Scope and evidence

Native execution targets SM120/SM121 on Linux, with the pinned CUDA 13.2 stack.
SM110/Thor and datacenter Blackwell are not supported by these kernels.
Importing the package does not require a GPU. CPU execution of native layers,
FSDP/DTensor integration, and general dynamic-shape export are not promised.

The [precision policy](../PRECISION_POLICY.md) defines numerical and benchmark
terminology. The [release checklist](release_checklist.md) tracks release gates;
dated reports describe evidence at that revision, not timeless performance.
Optional [DINOv3 tests](dinov3_regression.md) need separately obtained source
and weights. Those assets are not shipped by this project.

The source package is Apache-2.0; see [LICENSE](../LICENSE). Third-party test
assets keep their own licenses. The [SM120 hardware notes](../sm120_hardware.md)
credit Tri Dao's Quack project.
