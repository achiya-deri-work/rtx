# BalancedMoE design

Status: initial MXFP8 implementation, 2026-11-15. Experimental
[BalancedMoE](balanced_moe.md) now combines the existing
[BalancedRouter](balanced_router.md), grouped forward/recomputed backward,
VirtualLayer expert banks and shared SwiGLU. H=128 also has a real on-chip
hidden handoff with serial or 1/2/3/4-buffer down scheduling. Shared UCB tuning,
first-hit compilation and independent numerical gates are implemented.

This document describes the broader target, not a claim that all execution
families below exist. Prepare-once reuse, multi-expert clustering, split-dW,
automatic whole-expert graph replay and model-learning qualification remain
open. Measured grouped experts are still slower than compiled batched BF16.
Use the [API guide](balanced_moe.md) for implemented names and limitations.
Current development is **MXFP8 only**; NVFP4 discussion below is deferred.

## 1. Scope and terminology

**BalancedMoE** assigns exactly **K distinct routed experts to every real
token**, with exactly equal scheduled capacity per expert. It uses the BiMoE
regular fixed-top-k router, not the variable-count router.

**ConstrainedMoE** has minimum/maximum/mean variable token allocation. Its
bounded KL-Dykstra transport, expert-choice initialization and coverage repair
are a different design and are not part of this implementation.

BalancedMoE combines:

- The existing fixed-K routing policy, with FP32 routing arithmetic.
- Fine-grained routed SwiGLU experts using purpose-built batched/routed kernels.
- Independently configured always-on shared experts using our usual SwiGLU.
- BF16 Torch-side weights, inputs, outputs and parameter gradients, with
  explicit expert precision policies and the existing autotuning framework.
- Separate optimizer-visible projection matrices over contiguous storage,
  using [VirtualLayer](virtual_layers.md).

Initial scope is single-device SM120/SM121 on RTX's supported stack. There is
no token dropping, variable token degree, expert-parallel communication,
attention implementation or implicit residual connection. The host model
owns attention, normalization, residual addition and any residual scaling.

## 2. Mathematical contract

Let N be real tokens, D model width, E routed experts, K experts per token,
H routed hidden width, S shared experts and Hs shared hidden width.
Require N,D,E,H > 0 and 1 <= K <= E. S may be zero.

For a real token t:

```text
expert_e(x) = Wdown_e @ (silu(Wgate_e @ x) * (Wup_e @ x))
y_t = sum_{e in selected(t)} a_te * expert_e(x_t)
      + sum_{s=0..S-1} shared_s(x_t)
```

The K selected weights are finite, nonnegative FP32 values summing to one.
Shared experts process every real token, are added independently, and are
excluded from K and routed-weight normalization. Their count and width are
independent controls. Active FFN work depends approximately on
`K*H + S*Hs`, not expert counts alone.

### Equal capacity and nondivisible inputs

Equal real-only loads require `N*K` divisible by E. Preserve the handoff's
minimal dummy-padding policy for arbitrary positive N:

```text
token_multiple = E / gcd(E, K)
Nscheduled = ceil(N / token_multiple) * token_multiple
C = Nscheduled * K / E
```

Every expert receives exactly C scheduled assignments. Every real token still
has exactly K distinct experts. Dummy assignments have exactly zero mixture
weight and no contribution to outputs or expert gradients. Equal scheduled
loads are **not** a promise of equal real-only loads in padded cases.

A proposed strict padding option rejects nondivisible shapes instead of adding
dummies. It does not change K or silently approximate balance. Capacity is
derived from shape and policy, not an independent capacity-factor knob.

## 3. Routing policy to preserve

```text
BF16 tokens / learned router parameters
  -> FP32 normalized latent log-sigmoid affinities
  -> minimal constant-affinity dummy rows
  -> log Sinkhorn, 5 iterations
  -> exponentiated transport scores
  -> quantile/dual pricing (QB), 5 iterations
  -> score-aware residual repair, maximum 64 rounds
  -> expert-major IDs, selected soft weights, token-major inverse map
```

Preserve the original RMS normalization, learned directions, FP32 scale/bias,
zero projection initialization, deterministic perturbations and tie ordering.
Initially all affinities are 0.5; zero direction/scale gradients at this point
are not by themselves a missing-gradient bug. Expert initialization remains
the user's existing reset recipe, including 0.02 truncated normal where used.
Neither packing nor routing substitutes a different initialization.

QB supplies an approximate integer candidate. Repair supplies exact capacity:
move overloaded assignments to underloaded experts under source and target
quotas, without selecting an expert already held by the token. Preserve the
16 greedy rounds followed by the existing diversified proposal policy and
minimum necessary final route changes relative to QB.

The finite budget is statistically qualified, not a convergence proof.
Exhaustion must report failure. Do not return a round-robin replacement,
drop assignments, substitute a reference result or relax exactness.

Integer selection is detached. Gradients flow through the selected,
per-token-renormalized **soft transport weights**, including the differentiable
affinity and Sinkhorn computations. Do not replace these weights with raw
sigmoid values or detach the whole router.

The integer QB/repair score domain remains exponentiated transport, not logits.
Changing it, reducing Sinkhorn/QB/repair budgets, adding exchanges, or changing
the balancing pool is a model-policy change—not an autotuner schedule choice.

### Execution, not a different policy

Retain a fixed64 reference and the same-policy adaptive and segmented paths:

- GPU conditional stopping omits completed repair rounds while retaining the
  maximum budget. Balanced assignments are fixed points of the original body.
- Segmented quota ranking preserves FP32 priorities and original proposal-index
  ties, including the original nonfinite and signed-zero ordering behavior.
- Preserve precomputed int64 token indices; rebuilding them inside a compiled
  body previously caused int32 intermediate overflow in diversified rounds.
- Preserve the short-sort alternative for small proposal sets; segmented
  execution is not automatically faster at every size.

Reuse and requalify these paths on RTX's stack before attempting a new solver.
Affinity crowding and ConstrainedMoE's coverage repair are excluded.

## 4. Tensor and parameter layout

The reference routing contract is:

| Tensor | Layout | Meaning |
| --- | --- | --- |
| X | `[N,D]`, BF16 | Real token activations |
| expert_ids | `[E,C]`, int64 | Safe real-token addresses, including dummy slots |
| route_weights | `[E,C]`, FP32 | Selected weights; dummies are zero |
| inverse | `[N,K]`, int64 | Flattened expert-slot indices per real token |

Kernel-private validity metadata may be derived from `inverse`. Dummy status
cannot be inferred from `expert_ids`: their addresses are deliberately safe
real-token indices. A zero coefficient alone is also insufficient to classify
an assignment as dummy, since a real soft weight can underflow.

Keep int64 metadata initially to preserve the reference and its hash behavior.
Any future narrower representation requires overflow guards and exact parity.

Routed weights use two contiguous VirtualLayer banks:

| Physical bank | Logical optimizer matrices |
| --- | --- |
| Wgu `[E,2H,D]` | Separate `[H,D]` gate and up Parameters for each expert |
| Wdown `[E,D,H]` | Separate `[D,H]` down Parameter for each expert |

Within each expert, Wgu order is `[gate; up]`. There is no runtime weight
stack/cat, pointer table, or second trainable packed Parameter. Muon operates
on the logical 2D matrices; kernels consume the packed views. Checkpoints
contain logical matrices, not a duplicate packed bank.

Equal C makes this a **uniform batched routed GEMM problem**, not a general
variable-M grouped GEMM. Expert/token/output tiles can be scheduled over one
grid without a Python loop launching a GEMM for every expert.

Padding C/H/D for kernel legality is a separate, masked physical-layout issue.
Do not add real routes, change C or materialize padded BF16 tensors merely to
satisfy a convenient tensor-core tile.

## 5. Forward execution families

The common semantic path is:

```text
route X -> grouped gate/up + FP32 SwiGLU -> H-wide hidden
        -> grouped down -> FP32 weighted routed sum -------+
shared SwiGLU(X) -> usual BF16 shared outputs -> FP32 add --+-> BF16 Y
```

Gate activation and gate/up multiplication are evaluated in FP32 on chip.
Do not write a 2H-wide projection merely to apply SwiGLU afterward. Shared
experts initially reuse the existing ordinary SwiGLU implementation rather
than a new approximation or a per-expert routed kernel.

### A. Materialized batched reference and baseline

Gather `[E,C,D]` activations, run batched gate/up and down, then combine in
fixed token/K order. Use contiguous VirtualLayer weights and batched matmuls.
A per-expert loop is useful only as an independent correctness oracle; it is
not the performance baseline to beat.

This family also provides a possible execution candidate when preparing
contiguous expert inputs enables better GEMM transport. If native code wins,
gather/preparation must be implemented as kernels, not eager Torch operations
hidden in a registered launcher.

### B. Direct routed execution

The first kernel reads X through `expert_ids`, uses the corresponding expert
weights, computes gate/up and SwiGLU, and emits only the H-wide connection.
Down consumes that connection as a batched bank.

Compare direct BF16 gather/quantization with preparing reusable quantized X
once in original token order. Preparing X once can avoid repeating its
quantization across K experts; it adds a launch and global traffic and must
not be forced universally. Weight preparation similarly competes with
on-chip conversion, especially when C is small.

Hidden MXFP8 data/scales may be emitted directly by the producer, following
the existing MLP pattern. A BF16 hidden option is a separate numerical policy.
The hidden is transient between forward kernels, not saved for backward.

### C. Fully on-chip expert pipeline: first-class tuning family

This is a required execution family for fine-grained experts, **not a later
optional optimization**. Compare the complete gate/up -> SwiGLU -> down path
with no explicit GMEM gate/up or hidden tensor against families A/B. H=128
after gating is a primary target; also cover H=64 when "128 total" denotes
the combined gate/up width 2H.

Expose distinct implementations for register-resident, SMEM-resident and
hybrid register/SMEM intermediates. A GEMM accumulator fragment is not
automatically a valid operand fragment for the next GEMM: register exchanges,
quantization, scale packing and any required SMEM layout conversion are
part of the implementation, resource model and tuning cost.

For perspective, with 16 token rows and H=128, the logical FP32 hidden is
8 KiB, or an MXFP8 payload of 2 KiB plus 64 scale bytes. The FP32 gate/up
accumulators alone contain 16 KiB. These are payload counts, **not** the
complete register/SMEM requirement: weight/input stages, tensor-core layouts,
down accumulators, barriers, alignment and buffering all matter. Stream
weight tiles rather than assuming a whole expert's weights fit on chip.

The template must expose:

- Complete-H retention versus legal blocked-H execution, with explicit
  accumulator lifetimes and scale-block ownership.
- Independent input/weight, gate/up, activation/quantization, and down stages;
  warp-role specialization versus shared-role execution.
- Independent legal buffering depths, including 1/2/3/4 where resource limits
  permit, rather than one global "stages" value for the whole FFN.
- Token, hidden, input-reduction and down-output tile shapes; register and
  SMEM layouts, exchange/load/store paths and swizzles.
- One or multiple down-output tiles per CTA, with H retained/reused across
  those tiles rather than recomputing gate/up for every output tile by default.
- One or multiple expert/token microtiles per CTA and static/persistent work
  scheduling, without forcing all work for one expert into a single CTA.

Pipeline **independent work**: input/weight movement for a future microtile,
gate/up for one microtile, activation/packing for another, and down for an
already-ready microtile. A given microtile's down GEMM must wait for its own
gate/up and activation/scale dependencies. Double/triple buffering is a
search choice, not permission to overlap dependent arithmetic incorrectly.
Both GEMMs compete for tensor-core resources; overlapping schedules must
actually outperform sequential reuse, not just look more pipelined.

Retain the chosen hidden precision boundary even without a GMEM store. For
an MXFP8-hidden recipe, perform the same E4M3/E8M0 rounding on chip and feed
the resulting values/scales to down. Using unrounded FP32 hidden instead
would change numerics and cannot be promoted as a free scheduling win.

Blocked-H pipelines must also respect the amax domain. MXFP8's complete
feature-axis scale blocks can be finalized locally; an NVFP4 outer scale
covering an entire hidden region may need all its H values first. Retain that
region on chip or perform a justified additional pass before its down
consumption. Do not replace that scale with an earlier partial maximum or
silently split the numerical region just to make the pipeline legal.

One CTA per expert may underfill a large GPU; more persistence can also reduce
parallelism. The tuner must weigh saved traffic and conversion work against
redundant computation, occupancy, tensor-core tile utilization and device-wave
coverage. Compiler spill metadata and, when available, local-memory traffic
must distinguish an intended on-chip design from register spills to GMEM.
Do not claim physical GMEM-free intermediates from source-level fusion alone.

### D. Expert fusion versus whole routed-output fusion

Keep two promises distinct:

1. **Fused expert FFN:** no GMEM gate/up/hidden connection, but expert-slot
   outputs may still be stored for a later deterministic token combine.
2. **Fused routed-output tile:** retain the FP32 weighted sum across all K
   contributions owned by that tile, avoiding expert-slot output storage as
   well. Write the final routed sum, or final layer output when shared outputs
   are available under the chosen combine schedule.

The second is another autotunable ownership family, not an assumption that
all token rows share the same experts. Token-major ownership can reuse X
across selected expert projections, but different tokens select different
experts. A tensor-core tile normally shares its expert weight operand across
rows: heterogeneous assignments need explicit local grouping/compaction,
masked microbatches or smaller tiles. Include that cost and utilization loss.
Do not quietly execute every expert for every token or change selected routes.

Expert-major multi-expert CTAs can reuse X only where their routed token sets
actually overlap. Fixed K does not imply identical C-row expert batches.
Any scheduling metadata must preserve the original inverse map and the
declared token/K reduction order. Cross-CTA K contributions still need a
correct combine/workspace unless one CTA genuinely owns the complete sum.

"No intermediate GMEM" therefore names a specific fusion domain. It does
not mean no global X/weight/route reads, no router buffers, or no separately
computed shared-expert outputs. Do not market family C as full-layer fusion
if it still writes `[E,C,D]` expert outputs.

### E. GMEM-staged families: tune reuse, not just their GEMMs

Materializing H remains a valid competitor, especially when down needs many
output tiles or retaining H on chip would reduce occupancy. Its search must
include the producer, storage layout, consumer and reuse schedule together.

| Data | Reuse opportunity | Required choices |
| --- | --- | --- |
| Original X / feature-axis quantized X | Several selected experts and both gate/up projections | Prepare once versus direct gather; expert/token grouping; coalesced row loads; CTA order and locality |
| One expert's hidden `[C,H]` | Multiple down-output tiles of that same expert | Native packed layout/scales; one/multiple output tiles per CTA; persistent down consumers; producer/consumer ordering |
| One expert's weights | Several token tiles assigned to that expert | CTA raster/persistence; preload/packing versus on-chip conversion; buffered weight tiles |
| Routed outputs / gradients | Token combine and backward consumers | Explicit ownership; combine fusion; bounded reusable scratch; appropriate precision |

**Different experts do not share their hidden activations.** Their gate/up
weights differ. Reuse across experts is primarily X and its compatible
quantization, not a supposedly common SwiGLU hidden. Reuse of H is within an
expert across down-output tiles; gate and up share their input, not weights.

Compare original-token packs indexed by routes with contiguous expert-major
packs, and layouts tiled/interleaved across expert/token/output groups where
useful. If preparing packs/scales, emit the physical layout consumed by GEMM
directly rather than requiring another transpose/repacking pass. Row grouping
and scale semantics must remain unchanged when comparing execution layouts.

Expose expert-group width, token-group width, raster/swizzle, output persistence
and work ordering. CTA locality can encourage simultaneous consumers of the
same X/weight/hidden cache lines; it does not guarantee L2 residency or a
near-100% hit rate. No implicit cross-CTA SMEM sharing is assumed. Include
all preparation launches, bytes written/read, repeated quantization and any
grouping/worklist cost in the end-to-end comparison.

For a GMEM hidden, writing MXFP8 rather than BF16 saves payload traffic only
under its explicit numerical recipe. The on-chip and staged implementations
must be comparable at that same recipe before attributing a win to fusion.

### F. Reuse-oriented route ordering and expert groups

Expert-group widths **1/2/4/8** and the number of input rows/work items handled
per group are internal tuning axes. The user does not choose them. Sort or
bucket the **route/work metadata**, not the trainable projection matrices on
every call. VirtualLayer banks, expert identities and optimizer state retain
their physical ownership. No selected route is added or removed.

Compare no additional ordering, cheap bucketing/merging, and radix-style
ordering by expert group and token block. The reference's stable expert-major
metadata already preserves original token order within each expert before
dummy clipping. Exploit that ordering when possible instead of automatically
paying for a second global sort. Compatible worklist construction can also be
fused with routing metadata generation.

Candidate grouping policies include adjacent expert IDs for contiguous weight
access and affinity/co-occurrence-guided groups for overlapping input rows.
The latter must justify its construction/update cost. Rebuild data-dependent
worklists on the GPU when routing changes; do not reuse a frozen batch's
assignment list or perform per-step CPU score readback/clustering. Cache the
execution strategy, not the assumption that future route values are identical.

Within a CTA, reuse a loaded/quantized X row for the experts in the group that
actually select it. Record useful assignments per unique input row, mask/tile
utilization, resident weight bytes, grid size and ordering overhead. Expert
proximity by itself says nothing about shared tokens. Under an idealized
uniform random K-of-E assignment, grouping G fixed experts gives:

```text
p_hit = 1 - choose(E-G, K) / choose(E, K)
expected assignments per token in group = K*G/E
expected uses per distinct input row = (K*G/E) / p_hit
```

For E=128/K=8/G=8 this is approximately 1.21 uses per loaded row, not eight.
This is an analytical baseline, not a claim about learned routing; clustered
co-occurrence can yield substantially greater reuse. Time representative
changing routes so group selection does not overfit one easy fixture.

Preserve canonical expert-slot IDs and update inverse/worklist mappings when
changing execution order. In particular, do not silently renumber the C-axis
quantization blocks or NVFP4 regions: row reordering can otherwise change
forward scaling and leading-axis backward quantization.

### G. Weight-resident streaming for genuinely small experts

Add a weight-stationary family: retain a whole expert's prepared weights, or
a useful tile/subset, and stream successive assigned input microbatches through
gate/up, activation and down. Reuse weights/scales and their preparation across
those microbatches within the invocation. Pipeline the next input/metadata
loads against current compute. Compare it with input-stationary multi-expert
groups and hybrids; neither ownership direction is fixed by the frontend.

Small H alone does not imply the complete expert fits. A bias-free SwiGLU
expert contains `3*D*H` weight elements. Logical weight storage, excluding
native-layout padding, temporary preparation and other CTA state, is:

| D / H | BF16 weights | MXFP8 payload + E8M0 scales |
| --- | ---: | ---: |
| 128 / 128 | 96 KiB | 48 + 1.5 = 49.5 KiB |
| 768 / 128 | 576 KiB | 288 + 9 = 297 KiB |

Thus even the smaller complete expert leaves limited room for multiple resident
experts and deep buffering; the wider case requires weight tiles, distribution
across workers, or another execution family rather than a whole expert in one
CTA's SMEM. Register ownership can complement SMEM, but fragment replication,
accumulators and spilling must be accounted for. Quantized preparation on chip
also needs temporary state. Use actual device/resource limits, not payload
size alone, to admit candidates.

Tune whole versus partial residency, resident expert count, token tiles streamed
per residency window, weight/input buffering and warp roles jointly. When C is
small there may be little streaming work to amortize residency; when C is large,
too few persistent CTAs can limit device utilization. Prequantized weights may
be reused during a launch, not silently cached across optimizer updates.

### Combine strategy

Initially write expert-slot outputs and use the inverse map for a deterministic
FP32 weighted gather/reduction over K. Fuse weighting, accumulation and the
chosen shared-output addition where the ownership permits it.

Avoid global BF16 atomics as the default combine. A split FP32 workspace or
atomic alternative must declare its determinism/reduction-order behavior and
compete on whole-layer latency and memory. Do not move mixture weights before
SwiGLU; that changes the function.

Fix the initial output recipe explicitly: unweighted expert outputs have the
usual BF16 FFN boundary, then weights and accumulation operate in FP32. Add
the BF16 shared outputs in FP32 before the final BF16 layer output. Fusing the
down/weighting/combine stages must retain the expert-output BF16 rounding in
registers; removing that boundary would be a separate numerical policy. An
initial implementation can expose an internal FP32 routed sum and let
Inductor fuse shared addition/final casting; a later combine kernel can consume
shared outputs directly without changing the result's precision contract.

## 6. Precision and quantization boundaries

First qualify BF16 reference execution and MXFP8 forward/backward. Add NVFP4
forward with MXFP8 backward after the grouped foundation is correct. FP4
backward remains a separate experimental policy, not part of the initial plan.

External tensors and expert Parameters remain BF16. Routing and route-weight
gradients are FP32. GEMMs/reductions use FP32 accumulation, with BF16 casts at
declared boundaries. The exact operational oracle must record those casts.

Shared SwiGLU is independent of routed precision. Existing MXFP8MLP can supply
the ordinary shared path. Existing NVFP4MLP currently implements **SiTU**, not
ordinary SwiGLU; using it unchanged would violate this design. NVFP4 SwiGLU
requires an explicit activation implementation and qualification, not a rename.

Fix quantization semantics before tuning movement or fusion:

- MXFP8 feature-axis X scales can be shared across routed consumers when
  blocks are defined in original token rows.
- dW reduces expert-local capacity C. Its X/gradient scales group routed rows
  along C and cannot generally reuse original-token-axis or forward scales.
- W, hidden and gradient scales must respect expert boundaries.
- NVFP4 row-regional scaling in original token order is not equivalent to
  scaling after expert gathering. Choose an explicit policy; do not time-tune
  between these as though numerics were identical.
- Hidden format, scale rounding, region geometry and activation arithmetic
  are numerical policy, separate from tile/pipeline tuning.
- Canonically zero dummy rows before expert-local quantization. Although
  their weighted contribution is zero, repeated safe-token values could
  otherwise alter a block's amax and change real-token quantization.

Recomputation must reproduce the selected forward recipe, including scale
grouping and hidden rounding. It must not substitute MXFP8 projections for an
NVFP4 forward when differentiating the recomputed activation.

## 7. Backward and recomputation

Save original X, weight references, immutable routing assignments/inverse,
selected soft weights, and the selected execution/numerical metadata. Reuse
the hard schedule: **never reroute in backward**. Do not save gate/up, hidden,
or expert outputs solely to avoid recomputation in the default expert path.
Differentiable router autograd saves the final Sinkhorn log-transport and every
normalization probability for its exact finite-iteration adjoint. **Do not
recompute Sinkhorn in backward.** Expert recomputation does not eliminate this
state: it costs `(2 * sinkhorn_iterations + 1) * N_scheduled * E * 4` bytes in
the current native implementation, excluding temporary workspace/input scores.

For expert-slot output o_ec, token t and upstream gradient g_t:

```text
dO_ec = a_ec * g_t
dA_ec = dot_FP32(g_t, o_ec)
```

The derivative of the mixture weight needs the **unweighted expert output**.
Recompute it; weighting the output first and dividing by a is invalid for
zero/tiny weights. Dummy validity masks exclude dummy contributions explicitly.

The grouped backward then performs:

1. Recompute gate/up and the forward-rounded hidden, and recreate expert
   outputs as needed for FP32 dA.
2. Compute down dX (the hidden gradient) and down dW using expert-local rows.
3. Differentiate SwiGLU in FP32, emitting required independent gradient packs.
4. Compute gate/up dX and gate/up dW with FP32 accumulators.
5. Reduce K routed input-gradient contributions in a defined FP32 order,
   returning BF16 at the grouped operator's X-gradient boundary. Shared and
   router input gradients join through ordinary autograd under their own
   dtype contracts.

dA must cross the custom-op boundary as FP32 and flow into route-weight
normalization/Sinkhorn/affinity autograd. Frozen router or expert parameters
can skip unnecessary work without losing required X gradients.

Do not claim that the final cross-branch X gradient is accumulated entirely
in FP32 merely because each kernel uses FP32 internally. BF16 autograd inputs
can cast branch gradients before they meet. A future layer-wide FP32 gradient
merge needs an explicit, tested boundary; casting already-rounded BF16
partials back to FP32 is not equivalent. Record this distinction in the oracle.

Fine-grained dW reductions are often short: C is not N. At N=1024/E=128/K=8,
C=64; at E=256 it is 32. Conversely large N can make C long. Reuse the existing
FP32 split-reduction machinery only when beneficial; neither heavy split-K
nor a dense-MLP schedule is the universal starting point.

Save/recompute choices beyond this default can be added explicitly later.
They must identify saved-state bytes and numerical equivalence rather than
quietly changing the memory contract during timing selection.

## 8. Public frontend and compiler boundaries

Current experimental API (explicit balancing pool `[N,D]`):

```python
moe = rtx.BalancedMoE(
    d_model=768, expert_hidden_features=128, num_experts=128, top_k=8,
    num_shared_experts=1, shared_hidden_features=256,
    autotune='balanced', device='cuda', dtype=torch.bfloat16,
)
compiled = torch.compile(moe, fullgraph=True, dynamic=False)
y = compiled(x)  # [N, D] -> [N, D]
```

One module owns routing, routed banks and shared modules. Expose the usual
first-hit effort/cache controls and a deliberate offline `autotune_` path.
Keep router policy separate from kernel execution configuration. No public
stream, pointer-table, scratch-buffer, native plan ID or mandatory `prepare`
argument is required. An explicit routing-inspection method can return the
three metadata tensors without changing ordinary forward arity.

Proposed internal separation:

- Compiler-visible FP32 affinity and selected-weight math, with native CuTe
  soft transport forward/backward and explicitly saved normalization state.
- A CUDA-only private scheduling operator with detached FP32 scores and static
  policy inputs, returning nondifferentiable integer metadata. Its fake path
  derives output shapes from N/E/K and never builds a native plan.
- Tensor-only grouped expert operators consuming X, the two packed weight
  banks, IDs, FP32 weights and inverse metadata. They own expert autograd and
  return packed bank gradients plus FP32 route-weight gradients.
- Existing shared SwiGLU functional operations and a traceable or fused
  final addition, with a declared BF16/FP32 boundary.

Use truthful alias/mutation schemas, fake/meta functions and custom autograd.
Private workspace is not a user-facing output. Do not place eager gather,
casts, activation, quantization, scale processing or reductions inside opaque
launchers; keep them compiler-visible or implement them in kernels.

Port routing components without requiring the BiMoE expert/Helion package.
Keep a separate CPU/Torch correctness backend instead of registering an
unimplemented native CPU kernel. There is no per-expert Python launch loop
and no `torch.compile` instance constructed per invocation.

## 9. Automatic runtime ownership

Adapt prepared routing to RTX's automatic resource management. Cold plan
creation occurs at a real runtime setup boundary, not during fake execution,
Dynamo tracing or active manual capture. Warmed contexts reuse their plans.

Key contexts by device/architecture, dtype, logical/physical shape, routing
policy and budgets, source revision, stream and necessary execution lane.
Native build keys must include toolchain/runtime/architecture, not only source
bytes. Native sources must be included in installed package artifacts.

Plans own stable buffers; saved metadata has independent storage. Concurrent
or reentrant calls must not overwrite outstanding results. Use internal
per-context ownership/serialization and distinct overlapping lanes rather
than assuming every same-shape call can share one mutable workspace.

Caller-owned CUDA graphs remain optional. When capture is active, append
conditional routing nodes with correct capture dependencies; do not launch a
conditional graph into capture. Retain native/compiled resources for the
lifetime of each live capture, including after ordinary cache cleanup.

Cold manual capture may require ordinary warmup first, as with the existing
kernels. Users must not need to allocate buffers or select preparation streams.
Module movement, checkpoint assignment and policy changes must invalidate or
rebind contexts safely; they cannot leave stale native pointers behind.

## 10. Autotuner-first execution design

Use `rtx.autotune` adapters, persistent residuals, revision-safe cache identities,
budgeted orchestration and existing model/local/random search. Do not add a
separate MoE tuning engine or manually install a tile/register winner.

Expose legal families rather than only scalar tweaks:

- Materialized contiguous expert inputs versus indirect routed loads.
- Fully on-chip gate/up/activation/down with register, SMEM and hybrid hidden
  ownership versus GMEM hidden staging; these are primary family choices.
- On-chip operand quantization versus prepare-once reusable packs.
- Expert-major versus legal token-major/local multi-expert ownership, grouping
  overhead, grouped/batched tiling and expert/token/output CTA locality.
- Register/shared-memory tile shapes, legal load/store instructions, swizzles,
  buffering, pipeline stages, warps/roles, and register-budget configurations.
- Independent whole-FFN pipeline stages/roles/buffering, static versus
  persistent work distribution, and down-output-tile persistence/reuse.
- Expert-output staging versus complete K-contribution ownership/combine,
  with their different resource, scratch and determinism requirements.
- Fused activation/hidden quantization, derivative packing and consumption
  overlap under explicit operand lifetimes.
- Appropriate full/split FP32 dW and input/combine reduction implementations.
- Same-policy fixed/adaptive/segmented routing and legal completion-check
  intervals, preserving assignments and exhaustion semantics.

The context includes N, Nscheduled, E, K, C, D, H, S, Hs, precision policies,
gradient mask, phase, GPU specs and cache regime. Equal GEMM dimensions alone
are insufficient: routing reuse/locality and the full batch context matter.
Evaluate representative frozen routing fixtures and changed-batch sequences;
do not select a router schedule using only an already-balanced easy input.

### Effort controls exploration; users do not choose hardware ownership

The client selects a familiar effort preset or trial/time budget. Placement,
fusion, route ordering, grouping and residency are internal search decisions;
do not require a questionnaire about GMEM, SMEM, radix sorting or expert tiles.
Use the existing quick/balanced/thorough/extensive effort API rather than a
second set of MoE-only effort names. The intended cold-search progression is:

| Effort | Cold-search behavior |
| --- | --- |
| Light / quick | Establish and tune a proven GMEM-hidden staged path using a small bounded opening; avoid compiling broad fused/resident/grouping families |
| Balanced | Start from that incumbent, retrieve compatible evidence and try a bounded, prioritized set of higher-value structural alternatives |
| Deep / thorough / extensive | Sweep broadly across staged/on-chip families, register/SMEM ownership, 1/2/4/8 expert groups, token blocks, route-ordering policies, weight residency and pipeline configurations, then refine finalists |

Deep effort does not force fusion: a measured staged winner remains valid.
Light cold starts deliberately retain intermediate GMEM writes to bound search
cost. Exact compatible verified cache hits can reuse a previously qualified
winner at any effort without rerunning its deep search. A higher requested
budget resumes compatible evidence instead of discarding earlier trials.

Broad searches must obtain a valid incumbent first, then cover structural
families before spending most trials in one local neighborhood. Preserve that
incumbent on interruption, compilation/resource failures or failed timing gates.
Generate candidates lazily, use legality/resource pruning before compilation,
and reserve budget for correct paired confirmation. Increasing effort changes
search coverage, not the model's precision, routing policy or correctness gates.

Record the actual family, intermediate placement, fusion domain, grouping,
route-ordering cost, measured overlap, resident bytes and buffering in residuals.
Priors should learn these tradeoffs rather than hide them behind a tile ID.

Backward/recompute uses the same principle: consume recomputed hidden and
derivative tiles on chip where ownership permits it, but retain required
FP32 reductions/workspaces for gradients spanning CTAs. An on-chip forward
does not prove that all dW/dX and routing-gradient work can fit in one CTA.

Component timings diagnose where time goes. Winner selection measures the
complete intended forward/backward, including routing gradients, preparation,
expert recomputation, combining and shared work. Exclude setup/tuning time,
record cold cost separately, and retain memory costs and failed observations.
Keep the existing balanced effort ladder and correctness-checked fallback;
do not publish unmeasured winners when timing is unstable.

## 11. Qualification and performance baselines

### Initial user workloads

The machine-readable [initial workload suite](../benchmarks/balanced_moe_cases.json)
crosses these four configurations with **128, 512 and 2000 real tokens**,
giving 12 cases. Each has one always-on shared SwiGLU. Intermediate widths
below are **after gating**, not the combined gate/up projection width 2H.

| D | Shared Hs | Routed K | Routed H | Total routed E | C at N=128 / 512 / 2000 |
|---|---:|---:|---:|---:|---|
| 384 | 256 | 6 | 128 | 24 | 32 / 128 / 500 |
| 384 | 256 | 4 | 128 | 32 | 16 / 64 / 250 |
| 512 | 384 | 8 | 128 | 64 | 16 / 64 / 250 |
| 768 | 512 | 6 | 256 | 48 | 16 / 64 / 250 |

All twelve have integral exact capacity without dummy routing rows. The
2000-token cases deliberately exercise capacities 250/500 rather than rounding
the routing pool to a convenient GEMM tile. A kernel may mask physical capacity
tails (for example 250 into a 256-row tile), but must not add routes, alter C,
mix experts' quantization blocks or let masked rows contribute to amax/dW.
The 128-token cases emphasize small per-expert batches and grid occupancy;
compare grouped/multi-expert scheduling rather than launching one GEMM per
expert. Shared/routed paths and complete-layer time must all be reported.

These are regression/performance priorities, **not kernel design constants**.
Keep D, H, Hs, E, K and N as workload dimensions in legality, tuning and cache
identity. No new global "even dimensions" or "E divisible by 8" restriction is
implied: this suite has convenient aligned feature widths and E multiples of
8, but that does not establish a hardware necessity. Restrict an individual
optimized family only where its instruction/layout requirements demand it;
use masked logical padding or another eligible implementation otherwise.
Additional ragged N, odd E/K and feature-width tails remain separate boundary
tests. Minimal routing padding still follows `E/gcd(E,K)`, not a fixed 8/16/128.

The current comparison driver accepts `--cases benchmarks/balanced_moe_cases.json`
to measure the **router** on all 12 workloads. Shared/expert widths are retained
as suite metadata, but the router benchmark does not execute those FFNs; their
expert timings are provided separately by `benchmark_balanced_moe.py`, whose
optional `--whole-layer` comparison uses the same CuTe router on both sides.
Neither benchmark establishes whole-model convergence.

### Required gates

Port the retained routing fixtures and independent validators before changing
the implementation. Required gates include:

1. Exact K distinct routes, exact scheduled loads, inverse roundtrip, finite
   normalized real weights and zero dummy contributions. Include N=1,
   nondivisible N, K=1, K=E, ties, sharp/quantized and low-rank cases.
2. Exact fixed-versus-fast assignment parity, budget exhaustion, diversification
   hashes, stable quota ties and retained routing-score behavior. Feasibility
   alone is insufficient for a policy-preserving rewrite.
3. Independent FP32 frozen-route expert/mixture references for outputs, dX,
   every gate/up/down/shared gradient and FP32 route-weight gradients.
4. A quantization-aware operational oracle, separate from total errors versus
   BF16/FP32. Include expert boundaries, dummy/tile tails and long C reductions.
5. Gradients through the soft router, including scalar cancellation-sensitive
   parameters, zero initialization and several matched optimizer updates.
6. VirtualLayer/Muon independence, state-dict loading, moves, frozen gradients,
   fake/meta, opcheck and default-mode fullgraph compilation without graph
   breaks or repeated lazy-initialization recompilation.
7. Changed-input/weight graph replay, multiple outstanding forwards, owned
   overlapping contexts, cleanup while captures remain live, bounded caches
   and synchronized live-allocation measurements.

Use a compiled **batched BF16** expert implementation with the same routing
policy as the principal runtime baseline. Also retain the BiMoE fast-router
implementation and simple per-expert FP32 oracle for attribution/correctness.
Do not headline a speedup against Python expert loops or wasted fixed64 rounds
when comparing quantized expert kernels.

Cover capacity ladders from tiny/padded C through 32/64 and long reductions,
fine-grained H, varying E/K, and large token dimensions around 40K with
D=128/256/384 as well as wider transformer projections. Report phase times,
complete layer latency, peak workspace, cold cost and numerical error.
Match saving/recomputation policies in performance comparisons, or clearly
label their different memory/compute contracts. No speedup is promised before
these measurements.

Include H=64/128/256 and heterogeneous learned-style route fixtures when
comparing fused and staged families. Verify on-chip versus GMEM numerical
equivalence at fixed rounding boundaries; measure kernel/launch count, global
intermediate bytes, grouping overhead and spill/resource evidence in addition
to time. A repeated easy routing pattern must not be the only evidence for a
multi-expert reuse schedule. Report whether a winner removes just hidden
traffic or also expert-output staging.

## 12. Implementation sequence, when work resumes

1. **Policy and reference:** extract the fixed-K router plus tests/provenance,
   define metadata/dummy semantics, and build the batched BF16 reference over
   VirtualLayer banks. No new routing heuristic.
2. **Runtime integration:** automatic same-policy prepared routing, truthful
   operator contracts, compiler/graph ownership and gradient checks on RTX's
   supported stack. No mandatory user preparation API.
3. **Grouped MXFP8:** use a materialized path for bring-up, then implement
   complete forward/backward/recompute templates with **on-chip register/SMEM
   FFN fusion as a required initial native family**, alongside direct staged
   and prepare-once/reuse alternatives. Admit families only after independent
   correctness. Do not call fine-grained tuning complete while it can select
   only component GEMMs or GMEM-hidden paths. Complete routed-output fusion
   additionally requires the ownership qualifications in section 5D.
4. **NVFP4 forward (deferred, outside current scope):** only after the MXFP8
   execution/precision baseline is useful and qualified.
5. **End-to-end qualification:** balanced multi-shape whole-pipeline tuning,
   matched optimizer trajectories and whole-model learning, then further
   expansion of the already-supported fusion/reuse families.

Do not import the entire research package, rewrite routing and expert math
simultaneously, or port its old stream-management burden into the public API.

## 13. Limits and provenance

Flattening `[...,D]` defines the initial balancing pool. All flattened tokens
compete for capacity. Changing that pool changes policy, not just scheduling.
Global balancing/transport can couple tokens; this is not a causally qualified
autoregressive-decoder replacement. Causal routing and distributed all-to-all
need separate designs. Export, external autocast, sharding and arbitrary
dynamic-shape deployment are not initially qualified.

Source handoff: `routing-research-port-20261111`, currently located at
`/home/achiya/vs_code_main/work/BiMoE/routing-research-port-20261111/`.
The design follows its `DESIGN.md` section A, `PORTING.md` regular contract,
`RESEARCH_INDEX.md`, `VALIDATION.md`, and regular statistical/stage/QB/repair,
adaptive and segmented-quota campaign reports. Constrained/crowding reports
are background, not the implementation specification for BalancedMoE.

Historical same-policy routing gains were measured on the handoff's recorded
Torch 2.15/CuTe 4.8.1 environments. They are not new RTX qualifications or a
speedup forecast for our supported stack. Preserve source hashes, rejected
experiments, failure tensors and attribution when extracting that code.

Current foundations: [virtual layers](virtual_layers.md),
[MXFP8 MLP](mxfp8_mlp.md), [NVFP4 SiTU MLP](nvfp4_mlp.md),
[autotuning](autotuning.md), [execution](execution.md), and
[automatic resources](execution_resources.md). This document adds no runtime
API or new support claim to those components.
