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

### Experimental adaptive block scales

Outer scaling and block-scale selection are separate decisions. The default
`NVFP4QuantConfig.block_scale_selection="amax"` retains the original quantizer.
Three explicit alternatives keep FP32 outer scales and native E4M3 block scales:

- `lsq1`: try one least-squares scale update, requantize, and accept only if
  reconstructed block squared error improves.
- `mse5`: evaluate the baseline plus four scaled candidates and retain the
  lowest-error reconstruction. It costs substantially more arithmetic.
- `four_over_six`: compare scales targeting the block maximum at E2M1 values
  six and four, retaining the lower reconstructed squared error (baseline on
  ties). This is a bounded variant: the existing FP32 outer scale is unchanged,
  and both candidates are clamped to the existing E4M3 range `[2**-6, 448]`.
  Peak blocks may therefore lack headroom for the four-target candidate. It
  does not claim exact equivalence to recipes that enlarge the outer scale.

All operate on the loaded values inside the quantizer, including the cached
regional implementation. Neither inserts eager Torch operations or changes
MXFP8 backward. Lower operand reconstruction error does **not** guarantee
lower GEMM error, better convergence, or a speedup. In particular, `mse5` can
erase NVFP4's speed advantage. These are experimental opt-in policies, not
new defaults or interchangeable schedule choices.

`lsq1` is an algebraic one-step least-squares fit, **not** learned-scale
quantization: it adds no trainable scale parameters or scale gradients.

To reserve headroom for both Four Over Six candidates, explicitly choose
`tensor_scale_mode="exact_four_over_six"` in **both** `NVFP4ScaleConfig` and
the supplied `NVFP4DynamicConfig`. The FP32 outer decode factor is
`amax * FP32(1 / (6 * 256))`, with one for all-zero regions. The maximum
block's two raw E4M3 scales become 256 and 384 instead of 448 and 672.
This mode can also be paired with `amax` for a same-outer control experiment.
Existing `exact` and `power2` behavior is unchanged; changing outer scale
changes the numerical policy and is not an interchangeable schedule option.
The tuning adapter pins this mode, and default runtime-winner promotion
rejects it. This does not reproduce every upstream recipe detail: RTX retains
its minimum-normal E4M3 scale floor and native round-to-nearest-even conversion.

For standalone packing with reserved headroom, pass an explicit FP32
`tensor_scale` to `quantize_nvfp4`; its default outer scaling remains unchanged.
The experimental module/functional mode requires matching explicit scale and
dynamic configs and dynamic BF16 operands. Packed-weight and block-only calls
reject it instead of silently ignoring the requested outer policy.

The [Four-over-Six study](nvfp4_completion_20261023.md#four-over-six-policy-fixed-search-and-training)
includes native-scale tuning, full-forward comparisons, held-out numerics and
matched short decoder training. It remains opt-in: lower forward error did not
consistently improve validation loss. The research tuner accepts
`--selection four_over_six --outer-mode exact_four_over_six`, with
`--scaling current` or `--scaling jit_row_region` and `--scope full`.
These searches preserve numerical policy and do not promote default winners.

To use an explicitly selected, matching-shape research configuration:

```python
import json
from pathlib import Path
import rtx

record = json.loads(Path("path/to/winner.json").read_text())
raw = dict(record["config"])
raw["quant"] = rtx.NVFP4QuantConfig(**raw["quant"])
raw["gemm"] = rtx.NVFP4GemmConfig(**raw["gemm"])
config = rtx.NVFP4DynamicConfig(**raw)
_, n, k = record["shape"]
layer = rtx.NVFP4Linear(
    k, n, device="cuda", autotune="off",
    scaling=record["scaling"],
    dynamic_config=config,
    scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode=config.tensor_scale_mode),
)
```

This fixes the execution schedule explicitly; it is not an automatic
cross-shape/device performance guarantee. Inputs and outputs remain BF16 and
backward remains MXFP8. Reconfirm schedules on the intended GPU.

```python
quant = rtx.NVFP4QuantConfig(block_scale_selection="lsq1")
config = rtx.NVFP4DynamicConfig(
    quant=quant,
    x_scale_region_rows=8,
    weight_scale_region_rows=8,
    tensor_scale_mode="exact",
)
layer = rtx.NVFP4Linear(
    1536, 768, device="cuda",
    scaling="jit_row_region", scale_region_rows=8,
    scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode="exact"),
    dynamic_config=config, autotune="off",
)
```

The example is a legal explicit configuration, not a tuned winner. Standalone
packing also accepts `rtx.quantize_nvfp4(weight, config=quant)`. Adaptive modes
require `scale_reciprocal="direct"`. Schedule updates preserve the selected
policy; the dynamic tuning adapter rejects candidates that change it. Adaptive
results cannot be installed into the default amax runtime-winner context.
`adaptive_reuse_packed=True` is a separate schedule choice: retain the selected
FP4 bytes from error evaluation instead of converting again at the store.
It preserves the numerical policy but increases register live ranges, so it
remains tunable rather than assumed faster on every device.
`packed_store_bits=32` combines four packed bytes into each aligned store
instead of the default 8-bit stores. It requires `values_per_lane` of 8 or 16,
preserves the encoded values and scales exactly, and is a scheduling choice
available to the quantizer autotuner, not a different precision policy.
For adaptive block policies, `adaptive_decode="native"` uses the hardware
FP4-pair decoder during candidate error evaluation. Widening E2M1 through FP16
to FP32 is exact; scale products, reconstruction, and error accumulation still
use FP32. The default `"lut"` decoder remains available for tuning. This
coordinate is inactive (and rejected) for the non-adaptive `"amax"` policy.

For a policy-pinned search that records durable tuning residuals without
installing a runtime winner:

```bash
python -m benchmarks.tune_nvfp4_adaptive_scale \
  --selection lsq1 --shape 8192,768,1536 --trials 32 \
  --output autotune_reports/nvfp4_lsq1
```

The search keeps region geometry and GEMM fixed and varies quantizer schedules.
Validate its exported explicit configuration through the public frontend and
your training workload before adopting it.

See [the numerical-policy study](nvfp4_precision_study.md) for headroom controls,
independent reference tests, pretrained-model evaluation and offline
activation-aware calibration commands.

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

Region geometry changes the numerical policy, not only performance. The
autotuner therefore explores asymmetric and power-of-two candidates—including
8x4 and 8x8—but never silently changes an explicit user selection. On SM120's
128x128, atom-N=2 path, the default epilogue hoists repeated X/W FP32 factors
into registers and reuses each product across its matching accumulator
fragment. Direct scaling remains in the search space and both paths perform
the same FP32 multiplication before the final BF16 conversion.

Larger regions amortize observer and scale traffic but share one FP32 scale
across more rows. Smaller regions track local dynamic range more precisely but
produce more scales and observer work. X/W asymmetry is intentional: activation
and weight reuse, dimensions, and distributions differ. Region geometry
controls outer-scale grouping; GEMM CTA tile shapes and persistent scheduling
separately control actual operand reuse.

The low-level `NVFP4GemmConfig` also exposes
`regional_epilogue_schedule`, `regional_epilogue_warps`,
`regional_epilogue_registers`, and `regional_epilogue_values`.
Physical 64-row NVFP4 GEMM tiles are quarantined after large-grid numerical
failures; 128/256-row tiles remain available. Logical 64-row and ragged inputs
are supported through tail masking and do not require changing the user tensor.
`warp_specialized` is currently rejected by static validation following
reproduced GPU hangs during first-use autotuning. Its implementation is retained
for repair; use the default `mma` schedule with direct or fragment-register
epilogues. The quarantined implementation assigns independent
warps to the FP32 regional-scale/BF16-store epilogue and overlaps it with the
next persistent tile's MMA, but its 128x128 FP32 handoff consumes 64 KiB of
SMEM. Revision 8 instead seeds the faster TMA/direct-MMA basin. Its CTA
quantizer retains each thread's small BF16 quantization fragment in registers
while the CTA resolves the current regional amax, then quantizes those exact
values without rereading X or W. Regional scales, accumulator scaling, and the
final BF16 rounding remain FP32, FP32, and BF16 respectively. The MMA-thread
epilogue also loads its two X factors once per fragment, loads one W factor per
four accumulators, forms two products in registers, and reuses each across two
adjacent columns.

On the development RTX 5070 Ti, the register-cached producer reached 70.8 us
for `32768x2304x768`, versus 69.5 us for block-only quantization. Register
hoisting reduced the measured regional GEMM by about 1.6--6.8 us across the
decoder shapes, with larger output widths benefiting most. End-to-end JIT
regional forward remains slower than tensor-delayed scaling because exact
current regional observation and varying per-output factors retain costs that
epilogue reuse cannot remove. Shared-memory operand staging, expanded
factor/product tables, and a
separate output-rescale pass were measured and rejected because they increased
end-to-end latency. These schedule details remain autotunable and are not part
of the public module contract.

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
  `balanced`.
- `autotune="off"` uses supplied or portable configurations.
- `autotune="cache"` loads an installed verified winner and otherwise uses a
  portable seed.
- `autotune="balanced"` checks the exact cache first, explores at most 16
  candidates or 30 seconds on a miss, confirms the winner, saves it atomically,
  and reuses it for every subsequent call with the same context.
- `autotune="online"` is a compatibility alias for `"balanced"`.
- `autotune="coordinate"` requests the full deep-search policy.
- `tuning_policy` supplies a custom orchestration/search policy.
- `autotune_cache_dir` selects a non-default winner/cache root.
- `scale_config` selects tensor-current/delayed outer scaling and delayed-history
  policy. For stable JIT-region execution, the effective outer mode belongs to
  `dynamic_config.tensor_scale_mode`; passing only `scale_config` does not
  override the preferred regional configuration. Pin matching configurations
  as in the examples above when selecting exact regional scaling.
  `backward_config` overrides the low-level MXFP8 backward schedule.
- `dynamic_config` fixes the dynamic-X/dynamic-W materialized schedule.
- `weight_prequant_config` fixes dynamic-X/prequantized-W inference.
- `fully_prequant_config` fixes prequantized-X/prequantized-W inference.

Low-level configs are primarily for reproducible benchmarking and installed
winners. Most users should leave them unset so runtime selection can account
for GPU SKU, shape, cache regime, and kernel revision.

Set `RTX_BALANCED_AUTOTUNE_TRIALS` or `RTX_BALANCED_AUTOTUNE_SECONDS` to change
the bounded first-hit budget. The shared effort policy defaults to 16 trials;
older 24-trial campaign results are not a guarantee for this smaller budget.
Use `tuning_policy=rtx.AutotunePolicy(...)` to customize effort or explicitly
supply a prior. The runtime never silently trains on copied campaign data.

## Training

The stateful `delayed` policy is available on this module, not on the stateless
`rtx.nvfp4_linear` function. Functional scaling accepts only `current`,
`jit_row_region`, or `block`.

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

## Experimental FP4 backward

`rtx.NVFP4Linear` still uses MXFP8 backward. For explicit research use only:

```python
from rtx.experimental_ms_eden import ms_eden_linear

y = ms_eden_linear(
    x, weight,
    forward_scaling="jit_row_region",  # also "current" or "block"
    forward_outer_mode="exact",
    forward_config=nvfp4_config,       # optional NVFP4DynamicConfig
    dx_schedule_json=dx_schedule_json,
    dw_schedule_json=dw_schedule_json,
)
```

This saves original BF16 operands and uses MS-EDEN FP4 gradients with fresh
rotation signs and stochastic scale rounding. dX and dW can use different
execution schedules. No unbiasedness guarantee is made; sparse-input bias
has been observed. Supported inputs are contiguous rank-two BF16 CUDA tensors
with positive dimensions divisible by 128; higher-order gradients are not
supported. Eager and `torch.compile(fullgraph=True)` are tested.

Experimental overlap is opt-in: `backward_pipeline="paired"` schedules dX and
dW on joined CUDA streams for original-BF16, two-gradient MS-EDEN execution.
Partial/mixed-policy backwards retain their individual paths. Warm the actual
caller stream before CUDA graph capture. Independent preparation schedules
also expose `producer_async_copy` and `producer_stages` (1/2/3); asynchronous
buffering requires a supported coalesced source reader. These are performance
choices, not changes to FP32 scale/correction semantics or a switch away from
the original BF16 gradient sources. Serial execution and synchronous loading
remain defaults: overlap helps some shapes and can add contention on others.
See the [consolidated complete-backward guide](ms_eden_composable_search.md) for
validated configurations, bounded training results, and the opt-in research
tuner (`python -m benchmarks.tune_ms_eden_pipeline`). Its selections do not
automatically replace stable layer defaults.

`source_policy="forward_current_roundtrip_bf16"` instead saves packed forward
operands and changes the gradient's source values. It only supports the fixed
tensor-current forward; combining it with regional/block forward or a custom
forward config raises an error. Stateful delayed forward is not supported by
this stateless experimental function. None of these choices is enabled by
ordinary performance autotuning or changes the stable layer's backward policy.

See the [NVFP4 evaluation](nvfp4_completion_20261023.md) for matched forward/
backward timings, numerical error, and the boundary of current qualification.

### Independent experimental gradient policies

For this research function's JIT-region forward, an explicit
`forward_outer_mode="exact"` or `"power2"` controls the effective dynamic
configuration, even if a supplied schedule carries a different outer mode.
When the argument is omitted, a supplied JIT configuration retains its mode
(including Four-over-Six); without a supplied configuration, the historical
research default remains power-of-two. This does not change the stable layer's
configuration precedence. Earlier research runs requesting exact mode without
pinning the dynamic configuration used power-of-two scaling; corrected evidence
labels are recorded in the backward study report.

The research function also accepts `dx_policy` and `dw_policy`, each either
`"ms_eden"` (the existing experimental FP4 estimator) or `"mxfp8"`:

```python
y = ms_eden_linear(
    x, weight,
    forward_scaling="jit_row_region",
    forward_outer_mode="exact",
    dx_policy="ms_eden",
    dw_policy="mxfp8",
)
```

This example investigates FP4 activation gradients while retaining MXFP8 weight
gradients; it is not a recommended training default. Both policy arguments
default to `"ms_eden"`, preserving this experimental function's previous
behavior. Stable `rtx.NVFP4Linear` remains MXFP8 backward for both gradients.

Mixed policies require `source_policy="original_bf16"`. Packed-forward-source
reuse is rejected with any MXFP8 phase, rather than silently changing the
saved-state or gradient-source contract. Missing gradients perform no work or
random-number generation. MXFP8 phases consume no RNG.

`dx_schedule_json` / `dw_schedule_json` configure FP4 phases only. An optional
`mxfp8_backward_config=MXFP8BwdConfig(...)` controls the MXFP8 phases, including
its independent dX/dW matrix-multiplication schedules; otherwise the existing
deterministic default is used. This research API does not perform implicit
online tuning. Choose precision explicitly, then tune schedules within that
fixed numerical policy. The active qualification work is recorded in the
[backward study plan](nvfp4_backward_workplan_20261025.md); do not interpret API
availability as demonstrated mixed-policy convergence or a speedup.

The research all-MXFP8 option invokes the existing separate dX/dW operators;
it is not a substitute performance baseline for the stable layer's paired
backward implementation. Use `NVFP4Linear` for the stable training path.
The [estimator and short-training report](nvfp4_backward_numerics_20261025.md)
separates stochastic variance, residual sparse-input error, and corrected
exact-forward training evidence.

### Experimental preparation reuse and RNG choices

`PosthocPreparationSchedule` exposes `producer_signs="reload" | "register"`.
Register mode loads each lane's four fixed Hadamard signs once per producer
launch and reuses them across its tiles. The controlled GPU comparisons are
bit-exact; it is a scheduling choice, not a different gradient estimator.
Producer grids and thread counts remain independently configurable. A good
grid on one SKU/shape is not a universal default.

`rng_mode="explicit"` retains the original caller-generated uniform vectors.
The opt-in `rng_mode="philox"` instead draws a tiny key using PyTorch RNG and
generates scale-rounding uniforms inside the finalizer. `philox_packet_size`
may be `1` or `4`; four reuses the four words from each Philox calculation.
This mode only supports original-BF16 backward sources. It uses a documented
24-bit uniform grid and changes the random sequence/consumption; it is not a
promise of bit-identical training trajectories or unbiased gradients. It passed
the focused seed/reset/checkpoint and compiled graph tests, but measured slower
backward on the tested large 5060 projection while reducing peak allocation.
Keep it an explicit memory/performance tradeoff, not an assumed speed upgrade.

For direct deterministic tests, `rtx.experimental_ms_eden.ms_eden_seeded_gradient`
accepts a caller-owned contiguous CUDA `int64[2]` key and explicit signs. Its
two operands use separate Philox counter streams. The older
`ms_eden_gradient` still requires explicit uniform vectors and rejects a
misleading Philox schedule.

The preparation tuner supports `--public-rng` to measure caller randomness and
the complete compiled gradient. Without that flag it measures fixed supplied
random inputs and deliberately excludes/rejects RNG implementation choices.
Use `--initial-schedule-json` to start from a known schedule, and `--trials` to
bound effort. These experiments do not change the stable layer's backward
policy. See [the amortization study](nvfp4_amortization_20261025.md) for measured
tradeoffs and qualification boundaries.

Grouped preparation also exposes `producer_grouped_store_bits` in the
experimental `PosthocPreparationSchedule` / schedule JSON: `8` (default),
`16`, `32`, or `64`. This is a maximum FP4 packed-store width, not a quantization
precision choice. It applies to VPL8/16 producers; VPL4 keeps byte stores, and
VPL8 caps a requested 64-bit width at its 32-bit packed fragment. All arithmetic
and output bytes remain unchanged. For grouped transpose use
`producer_transpose_values_per_lane=16` with `producer_load="coalesced_x8"`.
The autotuner exposes this axis and records effective widths per operand;
schedule context revision 27 / feature revision 22 separates new records.
Initial 5060 results favor 32-bit stores on some shapes, not universally.
Stable `NVFP4Linear` still uses MXFP8 backward, and no default is promoted.
