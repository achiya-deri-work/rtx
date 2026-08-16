# `rtx.NVFP4Linear`

For TorchAO-style whole-model training conversion and packed-weight PTQ, see
[the model conversion guide](model_conversion.md).

`NVFP4Linear` is a no-bias replacement for `torch.nn.Linear` on supported
Blackwell RTX GPUs. It stores dynamic training weights in BF16, executes the
forward GEMM in NVFP4, returns BF16, and computes dX and dW with MXFP8 kernels.

## Default contract

```python
layer = rtx.NVFP4Linear(1536, 3072, device="cuda")
```

For dynamic BF16 activations and weights this means:

- `bias=False` and `dtype=torch.bfloat16`;
- `scaling="jit_row_region"`;
- independent current outer scales for five X rows and four W rows;
- `backend="auto"`, which selects the production materialized pipeline;
- runtime winner lookup through the autotuning cache;
- NVFP4 forward and MXFP8 backward.

The `5×4` geometry is a portable seed, not a universal winner. X and W remain
independent autotuning coordinates, including every asymmetric pair from 2–8
and wider regions. An installed per-device, per-shape winner can replace the
portable schedule and geometry.

The default changes for a prequantized weight: it uses current activation
scaling because its weight scale is already materialized.

## Constructor

```python
rtx.NVFP4Linear(
    in_features,
    out_features,
    bias=False,
    *,
    device=None,
    dtype=torch.bfloat16,
    scale_config=None,
    backward_config=None,
    scaling=None,
    scale_region_rows=None,
    x_scale_region_rows=None,
    weight_scale_region_rows=None,
    backend="auto",
    packed_weight=None,
    autotune=None,
    tuning_policy=None,
    autotune_cache_dir=None,
    dynamic_config=None,
    weight_prequant_config=None,
    fully_prequant_config=None,
)
```

### Linear and storage arguments

- `in_features` and `out_features` have the same meaning as `nn.Linear`.
- `bias` must be `False`. Bias belongs in a later fused epilogue rather than
  this base GEMM API.
- `device` chooses the initial parameter/buffer device.
- `dtype` must be `torch.bfloat16`. Dynamic training retains a BF16 master
  weight; NVFP4 is produced just in time and is not an optimizer state.
- `packed_weight` accepts an `rtx.NVFP4Tensor` with logical shape
  `[out_features, in_features]`. The resulting module is inference-only,
  contains no BF16 `Parameter`, and rejects training and dtype conversion.

Inputs may have any leading dimensions. The last dimension must equal
`in_features`; leading dimensions are logically flattened into GEMM M and the
output is restored to `[..., out_features]`.

### Scaling policies

`scaling=None` is recommended. It resolves to the best valid default for the
operand/backend state. To hold numerical policy fixed, choose explicitly:

| Policy | Scale source | State | Main tradeoff |
| --- | --- | --- | --- |
| `jit_row_region` | Current amax for bounded independent X/W row regions | Stateless | Default quality/performance balance; reacts immediately to distribution shifts |
| `delayed` | Previous amax history | Stateful | Avoids a current global reduction but can be stale for one step after a shift |
| `current` | Current tensorwide X and W amax | Stateless | Tensorwide numerical reference; global reduction/synchronization is expensive |
| `block` | No FP32 outer scale; native E4M3 block scales only | Stateless | Lowest scale overhead, but least protection for extreme or tiny exponent ranges |

Delayed mode initializes from current data, keeps device-resident amax history,
and resets that history after state loading, a relevant shape/stream change, or
re-entering training. It remains delayed during `eval()`, `no_grad()`, and
`inference_mode()`; delayed scaling is a forward policy, not an autograd switch.

Block scaling is appropriate only when the data distribution safely fits the
native block-scale exponent range. It is not a substitute for JIT regional
scaling when convergence or abrupt distribution changes matter.

### Region geometry

The three region arguments are optional and matter only to
`jit_row_region`:

```python
# Portable asymmetric default: X=5, W=4
rtx.NVFP4Linear(1536, 3072, device="cuda")

# Force symmetric 4x4 regions
rtx.NVFP4Linear(1536, 3072, device="cuda", scale_region_rows=4)

# Explicit asymmetric policy
rtx.NVFP4Linear(
    1536,
    3072,
    device="cuda",
    x_scale_region_rows=8,
    weight_scale_region_rows=3,
)

# Shared value plus one operand override => X=8, W=4
rtx.NVFP4Linear(
    1536,
    3072,
    device="cuda",
    scale_region_rows=4,
    x_scale_region_rows=8,
)
```

`scale_region_rows=N` supplies the value for both operands. Either independent
argument overrides the shared value for that operand. When all three are
`None`, X uses five rows and W uses four.

Larger regions amortize observer and scale traffic but share one FP32 scale
across more rows. Smaller regions track local dynamic range more precisely but
produce more scales and observer work. X/W asymmetry is intentional: activation
and weight reuse, dimensions, and distributions differ. Region geometry
controls outer-scale grouping; GEMM CTA tile shapes and persistent scheduling
separately control actual operand reuse.

The low-level `NVFP4GemmConfig` also exposes
`regional_epilogue_schedule`, `regional_epilogue_warps`, and
`regional_epilogue_registers`. `warp_specialized` assigns independent warps to
the FP32 regional-scale/BF16-store epilogue and overlaps it with the next
persistent tile's MMA. It requires a one-stage operand pipeline and at least two
tiles per CTA because its 128x128 FP32 handoff occupies 64 KiB of SMEM. The
portable seed enables the measured eight-warp/eight-tile basin only for large
output grids at K<=1024; other shapes retain `mma`. These are advanced schedule
coordinates and normally should be left to runtime autotuning.

### Backend

- `auto` is the production default. It chooses the valid materialized pipeline
  and its cached/tuned configuration.
- `materialized` explicitly selects separate quantization/observation and
  native NVFP4 GEMM launches. This is the supported JIT-regional backend.

“Materialized” means the quantized operands exist between CuTe launches. It
does not mean eager PyTorch reductions: the registered path remains compatible
with `torch.compile(fullgraph=True)` and keeps scale work inside kernels or
compiler-visible FX where specified.

### Autotuning and low-level overrides

- `autotune=None` follows `RTX_NVFP4_AUTOTUNE`/`RTX_AUTOTUNE`, defaulting to
  cache lookup.
- `autotune="off"` uses supplied or portable configurations.
- `autotune="cache"` loads an installed verified winner and otherwise uses a
  portable seed.
- `autotune="online"` permits tuning for a missing context. `"coordinate"`
  remains its compatibility spelling.
- `tuning_policy` supplies a custom orchestration/search policy.
- `autotune_cache_dir` selects a non-default winner/cache root.
- `scale_config` selects exact/power-of-two outer scaling and delayed-history
  policy; `backward_config` overrides the low-level MXFP8 backward schedule.
- `dynamic_config` fixes the dynamic-X/dynamic-W materialized schedule.
- `weight_prequant_config` fixes dynamic-X/prequantized-W inference.
- `fully_prequant_config` fixes prequantized-X/prequantized-W inference.

Low-level configs are primarily for reproducible benchmarking and installed
winners. Most users should leave them unset so runtime selection can account
for GPU SKU, shape, cache regime, and kernel revision.

## Training

```python
layer = rtx.NVFP4Linear(1536, 3072, device="cuda").train()
x = torch.randn(64, 512, 1536, device="cuda", dtype=torch.bfloat16)

compiled = torch.compile(layer, fullgraph=True, dynamic=False)
y = compiled(x)
y.float().square().mean().backward()
```

The weight, activation presented by the model, output, and optimizer-visible
gradients remain BF16 according to the surrounding training policy. NVFP4 is
used for forward GEMMs; MXFP8 is used for backward GEMMs; FP32 remains suitable
for reductions, loss accumulation, and optimizer state when the training recipe
requires it.

## Prequantized inference

```python
packed_w = rtx.quantize_nvfp4(weight_bf16)
layer = rtx.NVFP4Linear(
    weight_bf16.shape[1],
    weight_bf16.shape[0],
    device=weight_bf16.device,
    packed_weight=packed_w,
).eval()

# Dynamic BF16 X, prequantized W
y = layer(x_bf16)

# Fully prequantized X and W
packed_x = rtx.quantize_nvfp4(x_bf16.reshape(-1, x_bf16.shape[-1]))
y_packed = layer(packed_x)
```

JIT row-region scaling currently requires dynamic BF16 X and W. A packed W
therefore defaults to current activation scaling; requesting JIT regional with
a packed weight raises a clear error. Fully packed execution consumes the
scales encoded in both tensor subclasses.

The normal conversion API resolves unsupported dynamic policies safely:

```python
dynamic = rtx.NVFP4Linear(1536, 3072, device="cuda")
packed = dynamic.to_quantized_weight()  # scaling becomes "current"

block_dynamic = rtx.NVFP4Linear(
    1536, 3072, device="cuda", scaling="block"
)
block_packed = block_dynamic.to_quantized_weight()  # remains "block"
```

Block conversion writes an FP32 tensor scale of exactly one and absorbs the
entire range choice into the E4M3 1x16 block scales. It does not retain a
current tensorwise weight scale under a block-only policy.

Pass `to_quantized_weight(scaling="current")` or `scaling="block"` to choose
the packed activation-scale policy explicitly. Delayed and JIT-regional
policies cannot be transferred because both observe a dynamic BF16 weight.

## Choosing a policy

- Start with the default JIT regional policy.
- Use installed autotuning winners for production shapes and devices.
- Select delayed explicitly when compatibility or its measured schedule wins
  and one-step scale staleness is acceptable.
- Select current for a tensorwide numerical comparison, not as the expected
  fastest production path.
- Select block only after representative numerical validation confirms that
  its exponent-range assumptions hold.
- Override region geometry only for controlled numerical experiments or when
  autotuning evidence supports it; do not assume a symmetric region is best.
