# Python API map

Prefer `import rtx` and explicit names. Importing is lazy and CPU-safe; native
execution is CUDA-only. [The generated reference](api_reference.md) lists all
top-level exports, callable signatures and locally declared configuration
fields directly from source, including compatibility names. Use the guides
below for semantics rather than treating a signature as a support guarantee.

## Model and functional entry points

| API | Purpose |
|---|---|
| `MXFP8Linear`, `mxfp8_linear` | Dynamic/AOT-weight/fully packed MXFP8; [guide](mxfp8_linear.md) |
| `NVFP4Linear`, `nvfp4_linear` | NVFP4 forward, MXFP8 backward; [guide](nvfp4_linear.md) |
| `quantize_mxfp8`, `quantize_nvfp4` | Explicit BF16-to-packed conversion; [formats](formats.md) |
| `convert_to_mxfp8_training`, `convert_to_nvfp4_training` | Preserve BF16 parameters while replacing linears |
| `quantize_` | In-place selected-module PTQ; always use returned root |
| `mxfp8_linear_backward` | Advanced explicit dX/dW API; ordinary users use autograd |

The conversion policies are `MXFP8TrainingConfig`, `NVFP4TrainingConfig`,
`MXFP8WeightOnlyConfig`, `NVFP4WeightOnlyConfig`. See
[conversion](model_conversion.md). Functional NVFP4 does **not** accept delayed
history; delayed mode belongs to `NVFP4Linear`.

## Types, configuration and introspection

- `MXTensor` / `MXFP8Tensor`, `NVFP4Tensor`: TorchAO-backed payload/scale types.
- `LinearOperandState`: dynamic, weight-prequantized, fully-prequantized state.
- `LinearExecutionDecision`: result of linear `.explain(x)`.
- `AutotuneMode`, `LinearBackend`, `MXFP8Backend`, `NVFP4Backend`,
  `NVFP4ScalingMode`: user-facing policy type aliases.
- `MXFP8FwdConfig`, `MXFP8PrequantConfig`, `MXFP8QuantConfig`,
  `MXFP8GemmConfig`: fused/dynamic quantizer and GEMM scheduling.
- `MXFP8WeightPrequantConfig`, `MXFP8FullyPrequantConfig`: inference states.
- `MXFP8BwdConfig`, `MXFP8BwdMatmulConfig`: backward orchestration/matmul policy.
- `NVFP4ScaleConfig`, `NVFP4QuantConfig`, `NVFP4GemmConfig`,
  `NVFP4DynamicConfig`, `NVFP4WeightPrequantConfig`,
  `NVFP4FullyPrequantConfig`: scaling and state-specific scheduling.
- `MXFP8Problem`, `NVFP4Problem`: low-level M/N/K problem descriptors.

Named `DEFAULT_*` exports supply portable seeds, not universal winners. Explicit
configuration changes can alter numerical policy or become illegal for another
shape. Immutable dataclasses and `.rejection(...)` checks support kernel studies;
users normally leave low-level settings unset and use verified selection.

`validate_runtime_environment()` checks the supported dependency contract.
`clear_runtime_caches()` releases process runtime resources after synchronization;
it does not delete source datasets or model checkpoints.

## Feature-branch GLU and MLP

`MXFP8GLU`, `mxfp8_glu`, `mxfp8_glu_packed` expose forward-only paired projection
with activation. `MXFP8GLUConfig`, `MXFP8GLUGemmConfig`,
`MXFP8GLUPrequantConfig` configure it. The packed function returns a tensor-only
data/scales pair; it is not a BF16 hidden tensor.

`MXFP8MLP`, `mxfp8_mlp`, `MXFP8MLPConfig` add fully recomputed training. Public
helpers include `MXFP8MLP.from_linears`, `.autotune_`, and config
`.to_dict/from_dict/to_json/from_json`. Read [MLP limitations](mxfp8_mlp.md).
These feature APIs are explicitly importable but excluded from `rtx.__all__`.

## Subpackages

`rtx.autotune` is the extensible strategy/task/store framework; see its
[reference](../rtx/autotune/README.md). Legacy coordinate-tuner exports remain
directly accessible for compatibility, but are excluded from wildcard exports.
New projects should use the composable interfaces, not depend on private
runtime launchers, `_rtx_*` tensor fields or underscored operator schemas.

`rtx.models` contains the reference decoder, configuration/linear specification,
normalization helpers, attention/MLP layers and causal LM loss. This is an
evaluation scaffold rather than a general transformer framework.
