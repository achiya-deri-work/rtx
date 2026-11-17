# BalancedMoE (experimental, MXFP8 only)

`rtx.BalancedMoE` combines fixed-K, exactly balanced routing, fine-grained
routed SwiGLU experts, and an ordinary always-on shared SwiGLU. It is a
experimental implementation with correctness qualification, **not yet a
general performance replacement for compiled BF16 MoE**. Quantize-once expert
preparation substantially improves the original fused-quantization schedules;
forward-only wins do not establish a complete training-step speedup.

```python
import torch
import rtx

moe = rtx.BalancedMoE(
    d_model=384, num_experts=24, expert_hidden_features=128,
    top_k=6, shared_hidden_features=256, num_shared_experts=1,
    device="cuda", dtype=torch.bfloat16,
    autotune="balanced",
    tuning_policy=rtx.AutotunePolicy.preset(
        "balanced", search_schedule="progressive", candidate_mix="mixed",
    ),
)
model = torch.compile(moe, fullgraph=True, dynamic=False)
x = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16)
y = model(x)
y.float().square().mean().backward()
```

Only MXFP8 expert forward/backward is implemented. No NVFP4 mode is accepted
or silently substituted. Native execution uses the supported RTX SM120/SM121
stack; importing the API remains CPU-safe. `backend="torch"` is an explicitly
chosen batched BF16 reference, not an automatic fallback.

## Constructor and tensor contract

- `d_model`, `num_experts`, `expert_hidden_features` and `top_k` define routed
  experts. Hidden width is **after gating**, not the concatenated gate/up width.
- `shared_hidden_features` and `num_shared_experts` independently define the
  always-on shared path. The initial four test designs all use **one** shared
  expert. Setting its count to zero disables shared computation.
- `router_dim` and `router_options` configure [BalancedRouter](balanced_router.md).
  They control model policy independently of expert kernel schedules.
- `config=rtx.MXFP8MoEConfig(...)` pins independent expert schedules and defaults
  expert tuning to off. Without an explicit config, tuning defaults to balanced.
  `shared_config` separately controls the existing MXFP8 SwiGLU implementation.
- `autotune`, `tuning_policy` and `autotune_cache_dir` are inherited by routed
  experts, router and shared MLP. Their contexts and trial allowances are
  **separate**: a 16-trial policy is not a 16-trial total module budget.

Input/output are contiguous BF16 `[N,D]`. Flatten batch/sequence explicitly
to choose the balancing pool; this API does not infer causal routing pools.
Weights and their returned gradients are BF16. There are no biases, implicit
residuals, token drops or cross-device expert communication.

The router selects K distinct experts per real token. Minimal dummy padding
makes scheduled expert loads exactly equal, with
`Nscheduled = ceil(N / (E/gcd(E,K))) * (E/gcd(E,K))` and
`capacity = Nscheduled*K/E`. Dummies contribute nothing, including to
quantization and gradients. Equal scheduled loads do not imply equal real-only
loads when padding is needed. No global E-divisible-by-8 restriction is imposed.
Native expert grids are limited to 65535 experts and signed-32-bit element
address ranges; unsupported sizes are rejected before launching a kernel.

Gate/up/down are separate optimizer-visible 2D Parameters via
[VirtualLayer](virtual_layers.md), with contiguous packed banks
`[E,2H,D]` and `[E,D,H]`. Optimizers see each expert's gate and up separately.
Do not replace the logical Parameters with another trainable packed Parameter.
Expert/shared reset uses truncated normal, std 0.02, bounds -0.04/+0.04;
the router retains its separate zero-projection initialization.

## Precision and backward

All GEMMs quantize BF16 operands to E4M3 with one E8M0 scale per contiguous
32 values. Gate activation and gating multiply use FP32 accumulators; the
forward hidden connection is MXFP8, not BF16. Expert outputs are BF16 before
the FP32 mixture sum. Shared addition is compiler-visible FP32, followed by
the final BF16 conversion.

Backward uses an explicit MXFP8 straight-through recipe and **fully recomputes**
gathered inputs, gate/up projections, hidden and expert outputs. Forward saves
original input/weights, route metadata and an immutable configuration, not
hidden activations or gathered buffers. Bound prepared execution keeps that
configuration as static autograd metadata; only the cold first-hit/legacy
compatibility operation returns a CPU configuration token. The recomputed packed hidden is
decoded to BF16 for down-weight gradients, matching the retained two-pass MLP
recipe. Quantization reduction blocks never cross expert boundaries.

Hard QB/repair assignment has no gradient. FP32 route-weight gradients flow
through the selected Sinkhorn values and its saved normalization state. Sinkhorn
is saved, not recomputed. This is first-order custom autograd; higher derivatives
are not qualified.

## Real implementation families and autotuning

`MXFP8MoEConfig.family` selects:

The initial schedule now uses `grouped_prepared` with warp-owned gathering;
this is a starting configuration, not a cached claim of optimality. First-hit
tuning can still choose either of the other families.

| Family | Forward hidden | Current scope |
|---|---|---|
| `grouped_staged` | Packed MXFP8 data/scales in GMEM | Positive feature widths and logical tails |
| `grouped_prepared` | Packed MXFP8 data/scales in GMEM | Quantize each expert operand/orientation once and reuse across output tiles |
| `grouped_onchip` | MXFP8 handoff entirely in SMEM | H=128; one CTA owns gate/up and streams down output tiles |

These families materialize expert-slot outputs before a native combine. Prepared
execution can gather packed rows instead of materializing gathered BF16 X.
None fuses the router, shared expert or final token combine into the expert
GEMM. The on-chip family's backward currently uses staged recomputation.

The on-chip family exposes serial/warp-specialized down execution,
`onchip_down_stages=1/2/3/4`, and independent down N=64/128 tiles, subject to
real shared-storage bounds. Other config fields independently control gate/up,
down, recomputation, dX and both weight gradients: tiles, scalar/cp.async/TMA
transport, buffering, quantization packing, layouts, register-role budgets,
raster order and persistence. Unsupported combinations are rejected rather
than becoming no-op flags. Expert clustering and native split-dW families
remain follow-up work.

The prepared family's schedules live in
`MXFP8MoEConfig(prepared=rtx.MXFP8PreparedMoEConfig(...))`. Each of `gate_up`,
`down`, `recompute`, `down_dx`, `down_dw`, `gate_dx`, and `gate_dw` has an
independent packed-GEMM tile/pipeline/layout/resource schedule; `quant` controls
the quantization portfolio. `transpose_mode="staged"` uses coalesced shared
staging and logical CuTe transposes where legal, with a predicated direct-load
alternative. Neither mode materializes a BF16 transpose.

`input_preparation="gather_bf16"|"quantize_once"` is an independent autotuner
choice. The default retains the original gather-then-quantize path. The latter
quantizes original-token X once alongside all expert weights, then routes its
packed FP8 rows and scales to the expert slots. Forward needs no gathered BF16
X allocation. Backward still constructs expert-local BF16 rows when required
for independent leading-axis quantization; transposing feature-axis scales is
never substituted for that operation. Native scale layouts and separate/joint
backward remain supported. These paths must emit identical numerical values;
the different launch/memory tradeoff remains measured, not assumed faster.

`quant_grid="operand"|"expert"|"portfolio"` controls persistent quantizer
grid sizing. Operand mode retains each quantizer's original standalone grid;
expert mode divides the SM-wave target across the expert batch; portfolio mode
also apportions it across operands by size. The latter two avoid multiplying
a standalone SM-sized grid by every expert. More work per CTA can hurt latency
or occupancy, so all three remain autotunable. This changes scheduling only:
the same FP8 bytes and per-expert scales must be emitted.

`backward_preparation="separate"` and `"joint"` are selectable implementations,
not different activation precisions. The joint path recomputes gate/up only
once, emitting hidden FP8 values and both independently scaled derivative
orientations in that epilogue. It does not save forward intermediates or
reroute tokens. It currently requires complete 32-value hidden blocks and
row-major scales; expert-local capacity is zero-padded to 32. Fewer passes do
not guarantee a faster kernel, so the tuner measures both implementations.

Row-major scales can be staged by producers or consumers. The optional
`prepared.with_scale_layout("mma128")` compound schedule emits tensor-core-native
scales and uses scale TMA; it requires D/H divisible by 128 and zero-pads
physical expert capacity to 128. Only the small hidden scale array is converted
after GLU, not its FP8 payload. Logical routing capacities/assignments never
change. Native scale transport is not assumed universally faster.

Each ordinary linear stage also exposes cooperative 16-byte cp.async operand
loading versus TMA. Use `prepared.with_transport("down_dw", "cooperative")`
for an explicit schedule; it constructs the required single-buffer direct-store
configuration. Cooperative loading uses the MMA warps rather than a dedicated
producer, so producer/consumer role-register controls apply only to TMA.
Tile shapes, swizzles, ldmatrix widths and compiler register cap remain active.
This alternative passed the independent oracle but did not beat TMA in the
large measured case; it is an option, not a forced optimization.

BF16 linear outputs independently support direct stores or stmatrix→SMEM→TMA
stores with configurable store width/buffers. Full-tile TMA output is rejected
for ragged physical matrices. `capacity_multiple=1/32/64/128/256` controls an
optional **expert-local physical** token-capacity padding so aligned stores can
be considered; default 1 adds no such padding. Actual storage rounds up to the
largest requirement of this option, joint backward, and native scale layout.
Padding does not add real routes or change the balancing pool. The tuner
compares its extra work/memory against coalesced-store savings. GLU outputs
remain packed MXFP8, not BF16.

SiLU can use FP32 division or an FP32 shared reciprocal. The latter reuses one
reciprocal between activation and derivative; minor rounding differences are
subject to the same independent numerical gate. Approximate reciprocals are
not admitted. Gate products, activation derivatives, mixture sums and routing
adjoints stay FP32. BF16-pair conversion is a quantizer option, not BF16 gating.

Selection uses `rtx.autotune`'s existing UCB orchestration, local/random/structural
search, append-only residual journals, timing-quality gates and schema-v2
revision/device-safe winners. Complete execution families remain proposed by
the exploration arm; they are not merely recorded in pool metadata. Each
candidate is checked against an independent quantized oracle before timing and
again after repeated graph replay. A stage-aware local arm changes active
prepared components while still measuring the complete expert path, alongside
the existing local/random/structural UCB arms.
Cold budgets of eight trials or fewer keep GMEM-hidden execution, including
the prepared family;
larger budgets admit on-chip families automatically. A compatible deeper
cached winner or an explicitly supplied on-chip initial config remains usable.
The progressive proposal gate admits on-chip exploration after eight recorded
trials. The pool/checkpoint no longer changes just because a budget extends
from 8 to 16, so partial search work can be resumed.

For explicit pre-compilation tuning:

```python
result = moe.autotune_experts_(
    x, phase="train",  # or "forward"
    tuning_policy=rtx.AutotunePolicy.preset("thorough"),
    output_dir="autotune_results/my_moe", progress=print,
)
```

This tunes the entire routed expert path together, excluding router and shared
MLP, and populates the first-hit cache. Raising the cumulative trial allowance
resumes compatible observations; runtime refinement requires the usual
`cache_behavior="refine"`. Statistical timing failure may retain a
correctness-checked process-local seed, never a fabricated durable timing winner.

## Functional use and compilation boundaries

`rtx.mxfp8_routed_experts(x, gate_up_weight, down_weight, route, ...)` accepts
`BalancedRouter` metadata and returns an FP32 routed sum. Shared computation is
not included. Functional tuning defaults to off; module tuning defaults differ
as described above.

For all three bound expert families, the traceable autograd wrapper allocates
outputs and scratch **outside** the opaque custom op. Inductor sees these
allocations and logical shape views. A private mutation op receives the buffers
and launches a prebuilt CUDA sequence; it does no eager allocation, gather,
quantization, activation or numerical reduction. Scratch uses liveness-planned
typed arenas; returned outputs/gradients own separate storage and do not retain
those arenas. FP8 scales cross the private boundary as UINT8 storage carrying
unchanged E8M0 bits, interpreted by CuTe pointer layouts. There are no Torch
cross-dtype aliases or scale conversions at that boundary.

First-hit tuning still uses the **actual input**, not synthetic substituted
data. Its cold operation publishes a bounded per-shape selection cell. On the
next call, one automatic Dynamo guard transition compiles exact selected
allocations and static backward metadata. Other shapes do not invalidate an
existing hot cell. Pending cold/hot backwards retain their original selection
across cache clearing or explicit refinement. Offline tuning uses the same
execution/storage plan as the selected runtime family. Staged/on-chip families
retain their original numerical kernels; they share this allocation contract.

There is one grouped launch across all experts per GEMM, never a Python loop
launching individual experts. Users do not allocate workspace or manage streams.
There is no nested `torch.compile` or internally captured per-layer replay graph.
Default-mode fullgraph forward/backward is tested for all three families;
caller-owned CUDA graphs and user-enabled Inductor CUDAgraph capture are also
tested for prepared execution. The outer model's capture
eligibility also depends on its router/shared-MLP policies; expert qualification
alone is not a whole-model graph guarantee.

Complete cold selection and the guarded transition before manual capture.
Graph-replay benchmark results exclude compilation, tuning, capture and warmup;
ordinary compiled host latency is reported separately.

## Qualification and reproduction

Run the compact CUDA contracts:

```bash
python -m pytest -q tests/test_mxfp8_moe.py tests/test_moe_compiler_execution.py \
  tests/test_moe_global_input.py tests/test_moe_plan.py tests/test_moe_selection.py
```

The [case fixture](../benchmarks/balanced_moe_cases.json) covers all four initial
designs at 128/512/2000 tokens. Compare frozen-route experts with compiled batched
BF16, excluding router/shared work:

```bash
PYTHONPATH=. python benchmarks/benchmark_balanced_moe.py \
  --effort balanced --trials 16 --output autotune_results/moe/study.json
```

Add `--whole-layer` for complete compiled-layer comparisons with the **same
CuTe router** and shared count on both sides. The BF16 baseline saves ordinary
autograd intermediates; native experts recompute. This difference is explicit,
not a matched-memory recomputation claim. Add `--profile` for a separate warmed
diagnostic trace, not profiler-derived headline latency.

See the [current speed study](balanced_moe_speedup_20261115.md),
[initial work-5090 qualification](balanced_moe_mxfp8_20261115.md), and
[longer design](balanced_moe_design.md) for measured limitations and follow-ups.
