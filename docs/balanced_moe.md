# BalancedMoE (experimental, MXFP8 only)

`rtx.BalancedMoE` combines fixed-K, exactly balanced routing, fine-grained
routed SwiGLU experts, and an ordinary always-on shared SwiGLU. It is an
experimental implementation with correctness qualification, **not yet a
general performance replacement for compiled BF16 MoE**. Quantize-once expert
preparation substantially improves the original fused-quantization schedules;
forward-only wins do not establish a complete training-step speedup.

## Start here

Use one layer. Default first-hit tuning, persistent cache reuse, scratch and
execution plans are automatic. Parameters and input/output are BF16; the
kernels perform MXFP8 quantization internally.

```python
import torch
import rtx

moe = rtx.BalancedMoE(
    384, 32, 320, top_k=4, shared_hidden_features=256, device="cuda",
)
model = torch.compile(moe, fullgraph=True, dynamic=False)
x = torch.randn(8, 1024, 384, device="cuda", dtype=torch.bfloat16)
y = model(x)  # same shape; each batch sample is an independent routing pool
y.float().square().mean().backward()
```

There is no required prepare call, routing object, workspace, stream or graph
API. Compile your outer model normally; RTX does not compile one model per
expert/sample or create private per-layer replay graphs. First use of a new
shape/phase is slower while it tunes; compatible subsequent calls reuse it.

| Input | Routing semantics | Output |
|---|---|---|
| `[N,D]` | One pool across all N tokens | `[N,D]` |
| `[B,T,D]` | B independent T-token pools, inferred automatically | `[B,T,D]` |
| `[N,D]`, `cu_seqlens=cu` | Independent packed variable-length pools | `[N,D]` |

Dense flattening is a zero-copy view. Regular offsets are compiler-visible;
there is no per-sample Python loop or activation packing/copy. Inputs must be
nonempty contiguous BF16. Do not pass offsets with 3D input: they are only
for explicitly packed 2D input. Dense CUDA batches share the packed router's
E<=128/K<=32 limits. A shared expert always processes all real tokens once.

### One optional tuning knob

`autotune="quick"`, `"balanced"` (default), `"thorough"`, or `"extensive"`
requests 8/16/32/64 trials **per component**, subject to soft deadlines.
`"off"` disables search, not quantization; `"cache"` loads a compatible winner
without searching on a miss. `autotune_cache_dir` optionally chooses storage.
No `AutotunePolicy` object is needed for ordinary use.

To invest more **before** compiling or manually capturing a CUDA graph:

```python
moe.autotune_(x, effort="thorough")  # optional: all components, returns moe
model = torch.compile(moe, fullgraph=True, dynamic=False)
```

This uses the same first-hit selectors and durable caches, permits compatible
history refinement, and warms the selected paths. Existing parameter gradients,
input `requires_grad` and train/eval state are preserved. Each shared MLP shape
uses the normal common cache; users do not tune every expert separately.
Use `phase="forward"` to prepare no-grad inference. Inference itself is normal:

```python
model.eval()
with torch.inference_mode():
    y = model(x)
```

`eval()` alone does not disable autograd. Manual CUDA capture still requires
warmup; normal first-hit tuning is supported through caller-owned compile.
Advanced `tuning_policy`, `config`, `shared_config`, `router_options` and
`autotune_experts_` remain available, but are not required setup. Do not combine
quick/thorough/extensive shorthand with a custom policy; use `autotune="balanced"`
and that policy instead. The standard `reset_parameters()` restores the router
and std=0.02 truncated-normal expert recipe without replacing Parameters.

Frontend checks on Torch 2.13/CUDA 13.2/RTX 5090 cover cold compiled first-hit
training without any prepare call, dense versus explicit
packed outputs/all gradients, changed-input graph replay, training/inference
preparation, preserved existing gradients and compiled hot reuse with selectors
forbidden. Virtual-bank checks cover no-grad, inference mode and frozen-weight
input gradients, plus existing MXFP8/NVFP4 CUDA consumers. Static CUDA compile
is qualified here; CPU dense reference is eager, not a dynamic/export promise.

## Workload priorities

The primary optimization target is independent packed training batches of
**8, 16 or 32 samples, each containing at least 1024 tokens**. Thin and wide
model/expert projections, many/few experts and many/few selected experts all
matter within this envelope. This is a performance-development priority,
**not a new runtime minimum or supported-shape restriction**.
Here B counts samples in one GPU's MoE invocation, not a distributed global batch.

| Tier | Workloads | Performance emphasis |
|---|---|---|
| Primary training | B=8/16/32; every sample L>=1024 | Complete forward+backward, with separate forward/backward accounting |
| Secondary training | B>=8; every sample L>=512, plus other legal batches/shapes | Coverage and regression controls; lower optimization priority |
| Tiny/general fallback | Small B/T, short/empty packed samples and logical tails | Preserve current forward performance and correctness; backward must stay correct/stable but is not the main speed target |

At the primary lower bound the total token counts are 8192, 16384 and 32768.
Those totals must not be substituted for the independent-sequence contract:
8x4096, 16x2048 and 32x1024 have equal T but different routing and padding
behavior. Regular and genuinely uneven packs should both be represented;
primary uneven packs still keep every individual length >=1024.

Use a bounded, stratified matrix rather than a full Cartesian sweep: retain
the original four model designs as anchors, then vary D, expert H, E and K
with matched-axis and approximately equal-work controls. In particular,
track both mean useful rows per expert `T*K/E` and per-sample rows
`L_i*K/E`, scheduled/group-padded rows, physical capacity and actual GEMM tile
rounding. More tokens do not make activation preparation free, and increasing
E at fixed K does not increase useful expert FLOPs.

Longer samples do not eliminate every scale-padding problem. For example,
L=1024/E=128/K=2 gives 16 useful rows per expert per sample but a 32-row
sequence-local backward scale group. Increasing B does not remove that 2x
group-padding ratio. Preserve the existing backward grouping/numerical recipe;
forward compaction and safe bounded producers remain separate choices.

Rank future training improvements on the primary matrix, broken down by
geometry/E/K regime, not by one pooled average dominated by large GEMMs.
Keep actual light-effort first-hit selection and cache replay in qualification,
alongside fixed-schedule comparisons. Tiny forward remains an explicit
no-regression control; do not spend the main optimization budget closing
tiny backward gaps at the expense of the intended training workloads.

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

Input/output are contiguous BF16 `[N,D]` or `[B,T,D]`. Two-dimensional inputs
without offsets form one pool; dense batch samples are independent pools.
Use cumulative offsets only for packed 2D sequences. The layer does not infer
causal routing pools or add causal masking.
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

### Packed variable-length batches

```python
# Four independent sequences, not one global balancing pool.
cu = torch.tensor([0, 512, 2512, 3280, 7376], device="cuda", dtype=torch.int32)
x = torch.randn(7376, 384, device="cuda", dtype=torch.bfloat16)
y = model(x, cu_seqlens=cu)  # [7376, 384]
```

The router balances each sequence independently. Expert GEMMs batch the packed
assignments across sequences using the same expert weights; the shared SwiGLU
processes all real tokens once. CUDA uses segmented kernel grids and one
GPU-controlled repair loop, not a Python sequence loop or B copies padded to
the longest sequence. Empty sequences are supported when total tokens > 0.

Each sequence gets minimal logical routing padding. Expert slots are separately
aligned to 32 per sequence so MXFP8 backward quantization does not join its
reduction blocks with a companion sequence. Public metadata/workspace uses a
T/B-dependent upper bound so boundary values may change at fixed shapes under
graph replay without reading them back to Python. This physical slot padding
can be costly for very short examples; no universal speedup is claimed.

Router first-hit tuning records the cold length profile; expert tuning also
namespaces packed sequence count/alignment separately from a flat pool.
The same bounded `AutotunePolicy` and persistent caches apply. Explicit tuning
accepts `router.autotune_(x, cu_seqlens=cu, ...)` and
`moe.autotune_experts_(x, cu_seqlens=cu, ...)`. Warm both cold selection and the
bound hot call before manual CUDA capture. No caller workspace is required.

Packed native routing currently supports E<=128, K<=32, contiguous device int32
offsets and the existing SM120/SM121 stack. The CPU packed reference can use
`backend="torch"`. To compare CUDA BF16 experts with the same native packed
router, explicitly set `router_options={"backend": "cute"}`; the router's
`backend="torch"` mode is a CPU packed reference, not a hidden native fallback.

Hard QB/repair decisions are nondifferentiable. Small eager/compiled differences
in FP32 affinities can change near-tie assignments; comparing gradients from
different assignments is not a same-graph numerical check. Kernel-gradient
qualification fixes assignments and independently checks the segmented router
Jacobian. Both modes preserve sequence-local policy and exact scheduled loads.
See [the packed routing contract](balanced_router.md#packed-independent-sequences).
The [packed qualification study](balanced_moe_finish_20261117.md#packed-independent-sequences-2026-11-1718)
records the four-design RTX 5090 forward/backward matrix, source-matched
improvements, numerical checks and remaining BF16 gaps. It is not a universal
speedup or cross-SKU qualification.

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
than becoming no-op flags. Expert clustering remains follow-up work.

Forward combine and backward token dX have independent movement alternatives:
`combine_reduce_engine="scalar"|"token"` and
`dx_reduce_engine="scalar"|"token"` (both default to `"scalar"`). Each token
engine exposes its own `*_vector_width=2|4|8` and
`*_lanes_per_token=8|16|32`; `movement_threads=64|128|256|512` remains shared.
The token engine shares routing metadata across adjacent features without
changing rank-order FP32 accumulation or output casts. It is not universally
faster, so autotuning measures complete paths, including one-leg alternatives.
Explicit token schedules require feature width divisible by their vector width
and at least as many lanes as selected experts. These options add no buffers,
saved activations, eager tensor operations, or stream-management requirements.

For independent packed sequences, `prepared.active_prefix=True` can also
compose with the fragment GLU epilogues and unsplit retained-panel persistent
dW engines. It uses one private device bound, keeps physical allocation and
descriptor strides unchanged, and skips inactive computation rather than
promising smaller scratch allocations. Static legality still excludes split
dW, indexed dW, cooperative transport and split-warp recomputation from this
composition. Low-budget search includes complete compatible challengers on
the incumbent; sufficiently padded workloads receive early prefix coverage.

Prepared producers expose independent `bounded_gather`, `bounded_dy` and
`bounded_xt` booleans (all default `False`). They skip the unused suffix while
retaining physical pitches and the last tiles that consumers can read. These
are compatible with active-prefix row-major TMA execution. Training gather
bounds currently require indexed input rather than a materialized BF16 gather;
dY bounds require `dy_preparation="fused"`; X-transpose bounds require its
separate indexed producer. The full producers remain controls and fallbacks.
Turning on one producer does not assume that the other readers are bounded:
the index map is initialized fully when an unbounded reader still needs it.

`forward_row_layout="original"|"compact"` is a separate prepared-forward
choice (default `"original"`). Compact execution gathers into a private dense
row layout, uses the same fragment GLU/down arithmetic, and combines through a
translated inverse in the original rank order. Backward still recomputes from
the original routing layout and preserves its 32-value scale groups. No map
is saved into autograd. Forward scratch can shrink; whole F+B peak need not.
Initially compact forward requires complete 32-feature blocks and independent
row-major TMA fragment tiles. It is unavailable when its safe bound does not
reduce storage. Pool bounds come automatically from `BalancedMoE`'s regular
or packed input shapes; `compact_pool_count` is internal derived metadata,
not a tuning knob or a caller-provided proof. Opaque functional or externally
supplied tuning routes cannot enable the compact arm by copying that field.

On eligible padded inputs, eight-trial coverage protects prefix/token,
bounded-producer, compact-forward and combined alternatives before ordinary
exploration. Actual timings select among them; none is universally forced.
See [iteration-4 qualification](mxfp8_moe_iteration4_20261119.md) for the measured
gains and remaining small/irregular-shape gaps.

The prepared family's schedules live in
`MXFP8MoEConfig(prepared=rtx.MXFP8PreparedMoEConfig(...))`. Each of `gate_up`,
`down`, `recompute`, `down_dx`, `down_dw`, `gate_dx`, and `gate_dw` has an
independent packed-GEMM tile/pipeline/layout/resource schedule; `quant` controls
the quantization portfolio. `transpose_mode="staged"` uses coalesced shared
staging and logical CuTe transposes where legal, with a predicated `"direct"`
load alternative. `transpose_mode="warp"` adds a register-shuffle/byte-permute
implementation without a CTA shared-memory transpose or CTA barriers.
`transpose_warp_tokens=2/4/8` distributes a 32-token block over that many lanes;
the remaining lanes cover feature pairs. `transpose_warp_raster="features"`
or `"tokens"` changes traversal locality. All three modes keep independent
32-value reduction-axis scales; none materializes a BF16 transpose or reuses
feature-axis scales as transposed scales. These choices are autotuned, not
assumed faster on every SKU or length.

`forward_engine="packed"|"onchip"` decouples forward fusion from backward
preparation. The default packed engine writes an MXFP8 hidden tensor between
the two expert projections. For H=128, the on-chip engine keeps that hidden
tile on chip using the existing fused expert kernel, while retaining the
independently selected prepared backward. Its raw `MXFP8MoEConfig.gate_up`
and `down` schedules, `onchip_down_stages=1/2/3/4`, and
`onchip_pipeline="serial"|"warp_specialized"` control the fused forward;
`prepared.*` still controls backward. Unsupported combinations are rejected,
not silently run as a different precision policy. Both paths recompute in
backward; neither saves forward hidden activations.

The structural search cycles execution engines before deepening each
engine's layouts and traffic policies. Otherwise many variants of one
prepared family could consume a small budget without trying another engine.
Automatic search defers on-chip fusion until after eight recorded trials,
unless an explicit on-chip initial configuration was supplied. This keeps
small-budget exploration on GMEM implementations and larger-budget searches
able to investigate fused pipelines. These are search-coverage decisions,
not manually selected winning schedules. When a tuning seed inherited from
another shape is illegal (for example a full-tile TMA output store), the
tuner validates a legal starting alternative and records the rejection;
explicit untuned configurations retain their ordinary legality checks.

`down_dw_splits` and `gate_dw_splits` independently select `1/2/4/8/16`
reduction partitions. One is the unchanged unsplit path. Larger values use
balanced nonempty K-tile partitions, FP32 tensor-core accumulators, a
compiler-visible FP32 workspace, and a deterministic FP32 tree reduction before
one final BF16 cast. No BF16 partial-sum accumulation or atomic nondeterminism
is introduced. `dw_reduce_threads=64/128/256/512` and
`dw_reduce_vector=1/2/4/8` tune the reduction launch. Split paths require TMA
operand transport, direct FP32 stores and no simultaneous output persistence;
illegal combinations are rejected. The tuner proposes the complete legal
combination, so users do not need to coordinate these fields. Splitting costs
workspace and an extra launch; it is optional, especially for short sequences.

`input_preparation="gather_bf16"|"quantize_once"` is an independent autotuner
choice. The default retains the original gather-then-quantize path. The latter
quantizes original-token X once alongside all expert weights, then routes its
packed FP8 rows and scales to the expert slots. Forward needs no gathered BF16
X allocation. Backward still constructs expert-local BF16 rows when required
for independent leading-axis quantization; transposing feature-axis scales is
never substituted for that operation. Native scale layouts and separate/joint
backward remain supported. These paths must emit identical numerical values;
the different launch/memory tradeoff remains measured, not assumed faster.

`input_preparation="indexed_transpose"` extends quantize-once with an indexed
backward path. Routing emits a compiler-owned int32 token map (including -1
for dummy/gap slots); the transpose quantizer reads original BF16 X through
that map and emits the same independent expert-local 32-slot FP8 blocks.
No replicated expert BF16 X buffer is allocated. Weighted dY and other
backward operands still have their own storage and scales. This currently
requires joint backward preparation and the warp transpose implementation;
the tuner proposes that complete combination. Fewer bytes moved can cost
additional indexing and a launch, so materialized alternatives are retained.

`dy_preparation="materialized"|"fused"|"fused_all"` controls weighted upstream-gradient
preparation. The materialized default writes expert-local BF16 dY before
quantizing its two orientations. The fused alternative requires
`input_preparation="indexed_transpose"` (joint backward with warp X-transpose
quantization), reuses its validated token map, and loads each routed FP32 dY
tile once. It applies the FP32 route weight, rounds to BF16 **before** scale
selection exactly as the materialized path does, then quantizes feature-axis
and leading-axis 32-element blocks independently on chip. Dummy/gap values
remain positive zero. It removes a logical BF16 buffer and one launch, without
changing public gradient precision or saving forward activations.

`"fused_all"` additionally reuses that shared tile and its slot metadata to
emit the leading-axis quantization of original BF16 X, replacing the separate
indexed-X transpose launch. It does **not** reuse dY scales for X or
feature-axis scales for a leading-axis reduction. This optional family
improved short-pack preparation in qualification but regressed at a tested
long pack; larger shared tiles also lost. Both fusion levels and the original
materialized path therefore remain measured alternatives, not a forced policy.

For fused dY, `quant.transposed_tile_rows` controls feature rows,
`quant.transposed_tile_k` controls routed slots, and
`quant.transposed_smem_padding` controls their shared tile's padding. These
coordinates remain active even when the separate X transpose uses the warp
implementation. Quantization vector/load/store widths, reduction/math, warp
count, register cap, tile raster and grid policy are also tuner choices.
The tuner proposes the complete legal dependency bundle and can return to
materialized, separate-backward or native-scale families; fusion is not a
one-way search decision. Buffer elimination is not a promise of an equal
peak-VRAM reduction because the compiler-visible scratch arenas reuse
nonoverlapping lifetimes.

`route_gradient_epilogue="separate"|"fused"` independently controls the joint
backward's recomputed down output. The default writes ordinary expert outputs
then reduces their dot product with dY. The fused alternative first rounds
each accumulator to BF16 exactly as the ordinary output does, multiplies by
FP32 dY, and produces row/column-tile FP32 partials. A deterministic FP32
reduction supplies route-weight gradients without writing/rereading the
full expert-output tensor. It needs joint preparation and TMA down transport.
The forward down-store schedule stays independent, including its TMA store
choice. No precision change, saved forward state, atomics, or caller-managed
workspace is involved; FP32 summation order can differ within the numerical
contract. The tuner measures the separate and fused choices.

The fused route adjoint has its own optional `route_gradient_gemm` schedule
(otherwise it inherits `down`) and `route_gradient_loader`:
`"accumulator"` retains scalar accumulator-owned loads; `"warp_vec2"` broadcasts
the row map and pairs FP32 upstream loads; `"shared_vec4"` stages a coalesced
tile. Misaligned contiguous upstream storage falls back to scalar accesses
without a copy. Smaller adjoint tiles can win even when forward down prefers
a larger output tile. Extra shared staging is not automatically faster.

`gate_dw_engine` / `down_dw_engine` default to `"packed"`. The optional
`"persistent"` family shares one operand panel among two or three independent
64x64 accumulator teams, controlled by each stage's `*_reuse_axis="m"|"n"`
and `*_reuse_tiles=2|3`. It is currently unsplit, row-major, TMA, static-capacity
execution with nonsingleton output dimensions; it adds no GMEM workspace.
This is real operand-panel sharing, not the generic `tiles_per_cta` loop.

Gate dW also has experimental `gate_dw_engine="indexed"`, with
`indexed_dw_token_lanes=2|4|8`, `indexed_dw_gather_pairs=1|2`, and cooperative
or dedicated `indexed_dw_producer_warps=0|2|4|8`. It computes each independent
32-slot X scale inside the GEMM and omits the global X-transpose buffer. It
requires joint `indexed_transpose` input, no active prefix/split, and dY
preparation other than `fused_all`. Repeated quantization across output tiles
lost the tested producer+consumer comparisons, so this remains an explicit/
deeper-search alternative, not a protected light-tuning candidate.

The autotuner proposes complete legal combinations and ranks panel grouping
by padded tile work/operand-panel count, not a SKU winner table. Long-reduction
eight-trial openings retain the incumbent and store/direct controls while
admitting complete ownership and preparation challengers. Explicit invalid policies are
rejected before scratch allocation even with autotuning off. See the
[backward implementation and qualification report](mxfp8_moe_backward_ownership_20261119.md).

`forward_glu_engine="legacy"|"fragment_direct"|"fragment_packed"` selects
the packed gate/up epilogue. Fragment ownership keeps FP32 products in
registers and exchanges compact scale maxima; packed mode also coalesces FP8
stores. Legacy remains the default. The alternatives currently require static
capacity, qualified M64/M128 accumulator layouts, row-major scales, SiLU with
division, physical K32 alignment and direct stores. They can lose on small
pools, so neither is an unconditional replacement.

`weight_preparation="separate"|"dual_orientation"` controls backward weight
preparation. Dual orientation reads each BF16 weight tile once and independently
quantizes both reduction directions. It requires joint recomputation and
original-token input preparation; it does not reuse forward quantized weights.
`weight_tile_rows`, `weight_tile_cols`, `weight_load_values`, `weight_store_bits`,
`weight_warps` and `weight_raster` expose its scheduling choices. It adds no
launch or global buffer. See the [additive qualification and negative controls](mxfp8_moe_iteration2_20261119.md).

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

`joint_recompute_engine="legacy"|"operand_reuse"|"dual_operand"|"split_warp"` selects
the joint epilogue's storage/ownership. The default is `"legacy"`.
The alternatives reuse TMA operand storage only after its last consumer has
finished; `"dual_operand"` also shortens the live register lifetime of hidden
products and derivatives. FP32 activation/products, BF16-rounded derivatives,
and both independent MXFP8 quantization directions remain unchanged. These
schedules require producer scale staging, sufficient retired operand storage,
and one nonpersistent output tile per CTA. The tuner proposes complete legal
tile/pipeline combinations. Lower register/SMEM use alone does not establish
a speedup: compact schedules won qualified H=128 cases, while tested wider
schedules tied or regressed. See the [matched qualification](balanced_moe_joint_recompute_20261118.md).

Experimental `"split_warp"` divides derivative and hidden-output teams using
`joint_recompute_gradient_warps`. It preserves the BF16 decoded-hidden boundary
and independent transpose quantization, but needs additional shared storage
and static capacity. It lost the measured comparison with the existing
vector-4/XOR schedule and is not a protected light-tuning candidate.

Tiny expert pools also admit optional independent K32/K64 weight-gradient
staging via `prepared.with_short_dw("gate_dw", 32)` or `"down_dw"`.
These templates preserve block-32 scales and FP32 accumulation, and avoid
scheduling an empty K128 macro tile where useful reduction is short. They
require row-major scales, unsplit weight gradients and direct outputs; the
current implementation does not combine with active-prefix trimming.
Both stages and their inherited K128 alternatives remain autotuner choices.
See [short-dW qualification](balanced_moe_short_dw_20261118.md).

`active_prefix=False|True` controls optional device-side work trimming.
It is off by default. For independent packed pools, the tuner can measure
the active alternative without any new caller arguments: allocations and
fake shapes retain their static bound, while a device-computed prefix skips
unused output/reduction tiles. A narrow tail initialization preserves every
partial reduction tile that can still be read. Sequence-local 32-row scale
groups are not compacted together. Supported training schedules require
row-major packed TMA, joint recomputation and unsplit weight gradients;
legal retired-operand engines can compose with the prefix. This does not
save the statically allocated VRAM and may lose when there is little unused
work. Users never provide the prefix tensor, scratch, or a stream. See
[active-prefix qualification](balanced_moe_active_prefix_20261118.md).

For ordinary tuning budgets of at least eight trials, legal new liveness
and packed-prefix families receive measured opening coverage alongside the
inherited initial config and existing quantize-once/indexed alternatives.
This is tested through the real shared orchestrator, not just by inspecting
its candidate pool. It remains subject to the time deadline and does not
override an explicitly selected research-prior opening or force a winner.

Small-budget tuning also compares **compound output-transport recipes**.
On a legal packed row GEMM, changing direct stores to TMA may require a
64-row tile and two operand buffers together; changing each coordinate alone
can be illegal. The proposal generator derives these combinations from the
actual physical expert capacity and output width, without adding padding or
changing quantization, activation arithmetic, register budgets or dW reduction
policies. Direct stores remain a measured alternative. The common packed
MXFP8 GEMM also supports `epilogue="hybrid"`: complete interior tiles use TMA,
and only boundary tiles use masked direct stores. This is ordinary dense
GEMM tail handling, not sequence-aware shared-expert processing. Output rows
must retain the TMA-required 16-byte stride alignment and at least one full
interior tile; otherwise direct stores remain necessary. No output padding or
additional GMEM intermediate is introduced. Multi-buffer persistent schedules
drain pending bulk reads at a boundary before an output stage can be reused.

The ordinary eight-trial prefix measures the incumbent, up to six distinct
complete structural proposals, and one scheduler-selected trial. For routed
experts this includes combined down/dX stores and their preparation control;
for the shared MLP, forward-only stores compete with combined forward/dX
stores, including when cold selection starts from an on-chip configuration.
The shared opening also crosses stores with a second existing legal dW plan:
agreement between two budgets is insufficient if both miss that combination.
These are soft search priorities, not shape-specific winners or guarantees
of a global optimum. Explicit reservation settings and compatible research
priors retain control. A time cap can still admit fewer than eight trials:
reference setup and an in-flight compilation are additional wall time.
Progressive 8-to-16 refinement reuses the same prefix. The shared MLP winner
namespace is revision 48. Shape-derived transpose tiles restore legal
prequantized proposals for half-aligned operands, rather than forcing 128-row
tiles that reject them before measurement. This invalidates older first-hit
results with that coverage gap. See [eight-trial qualification](mxfp8_moe_eight_trial_prior_20261119.md),
[hybrid-tail follow-up](mxfp8_hybrid_tails_20261119.md) and the
[fresh half-aligned 8/16-trial study](mxfp8_moe_iteration2_20261119.md).

These improvements compose in the same `BalancedMoE`, not in a separate
experimental frontend. Router and expert winners have independent persistent
contexts and bind together into the caller's compiled graph. Expert local
search also rebases complete compact recompute recipes onto its measured
incumbent, preserving its input preparation, routed-gradient and other traffic
choices instead of restoring an older seed. Every combination is measured;
the [45-context challenge profile](balanced_moe_challenge_20261119.md) separates
small/large expert batches, input distributions, router-search failures and
padding-sensitive BF16 comparisons. It is not a universal speedup guarantee;
there is no automatic promotion based on isolated percentages. Persistent
Sinkhorn currently belongs to the **packed** router (`cu_seqlens`, including
a single explicit pool); the no-offset router keeps its existing execution.
See the [combined qualification](balanced_moe_next_priorities_20261118.md#combined-path-qualification).

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
