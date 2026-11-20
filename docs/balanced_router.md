# BalancedRouter (experimental)

`rtx.BalancedRouter` implements the routing front end of BalancedMoE. It does
not execute expert or shared FFNs. BalancedMoE means **K distinct experts per
real token and equal scheduled capacity per expert**. It is not ConstrainedMoE
and does not vary the number of experts per token.

```python
import torch
import rtx

router = rtx.BalancedRouter(
    d_model=768, num_experts=128, top_k=8, router_dim=64,
    device="cuda", dtype=torch.bfloat16,
)
route = torch.compile(router, fullgraph=True, dynamic=False)(x)  # x: [N, 768]
token_ids, weights, inverse = route
```

Inputs must be nonempty `[N, D]`; flatten a batch/sequence explicitly to choose
the balancing pool. Parameters and external activations default to BF16.
Affinities, Sinkhorn and mixture weights use FP32. Projection weights start
at zero, giving uniform initial sigmoid affinities of 0.5; consequently some
parameter gradients are initially zero by design.

## Returned metadata

| Field | Shape/dtype | Meaning |
|---|---|---|
| `expert_ids` | `[E, C]`, int64 | Real input token to gather for each expert slot |
| `route_weights` | `[E, C]`, FP32 | Differentiable normalized mixture coefficient |
| `inverse` | `[N, K]`, int64 | Flat expert-slot indices for each real token |

Here `C = N_scheduled * K / E`. With `padding="minimal"` (default), the router
adds the smallest number of dummy routing rows needed for integral capacity.
Each dummy receives constant log-affinity `-log(2)` and zero mixture weight.
Its returned token index is clamped to a safe real token. **Mask these slots
before expert-local quantization**: a safe gather index does not turn a dummy
into a real token. The valid slots are exactly `inverse.flatten()`.
`padding="error"` instead rejects nondivisible pools.

The weights selected by `route_weights.flatten()[inverse]` sum to one for
every real token. A caller later gathers expert inputs and combines weighted
expert outputs using these tensors. Hard metadata is owned by each invocation;
a subsequent routing call must not overwrite an earlier result.

## Packed independent sequences

```python
cu = torch.tensor([0, 512, 2512, 3280, 7376], device="cuda", dtype=torch.int32)
route = router(x, cu_seqlens=cu)  # x: [7376, D]
```

Sinkhorn, QB prices, repair quotas and token tie indices are independent for
the four sequences. Every real token has K distinct experts; each sequence
gets its own minimal logical dummy rows and exact scheduled expert capacity.
Empty sequences are allowed; total tokens must remain positive.
`padding="error"` checks divisibility separately for every sequence.

Offsets must be contiguous int32 `[B+1]` on the input device, start at zero,
be nondecreasing and end at T. CUDA validates values on device. The hot path
has no boundary readback, Python sequence loop or `[B,max_length,D]` activation
allocation. Normalization/quantile grids are segmented; one conditional GPU
repair loop serves all pools. Its FP32 exponentials are emitted in the transport
epilogue, not a separate eager/Inductor exponential at the decision boundary.
The existing packing kernel also checks every real affinity for finiteness;
validation is not disabled, and no separate full-score reduction serializes
the affinity epilogue. Its status is reset on every invocation/replay.

The returned fields are unchanged: IDs address global real-token rows and
`inverse` is `[T,K]`. Within each expert, sequences own consecutive segments
whose physical capacity is rounded to 32. This keeps MXFP8 reduction-scale
blocks from straddling sequences. Dummy/tail slots have zero mixture weight
and are absent from `inverse`. Validity still comes from `inverse`, not merely
from a safe ID or a coefficient that might underflow.

To permit compiler-visible allocation without data-dependent output shapes,
write `A=E/gcd(E,K)`, `q=K/gcd(E,K)`, `S=min(T,B)`. The routing row bound is
`A*floor((T+S*(A-1))/A)`: at most S sequences can be nonempty. If q divides 32,
set `P=A*32/q`; the tight worst-case per-expert slot bound is
`32*(S+floor((T-S)/P))`. Other q conservatively round
`routing_row_bound*K/E + 31*S` up to 32. This is bounded routing/slot padding,
not padding BF16 activations to the longest example. Very short sequences can
still have substantial slot overhead. Native support is E<=128, K<=32 on
SM120/SM121. `auto`/`cute` use CUDA; `torch` is the CPU packed reference.

Packed first-hit tuning uses the same shared UCB orchestration and effort
levels, caller plus sigmoid-normal calibration, an independent segmented FP64
normalization/gradient oracle, residual journals under `varlen_routing_effort`,
and source/revision-safe `balanced_varlen_routing` winners. Train/inference,
T/B, policy and the cold input's length profile have separate identities.
Only cold selection/explicit tuning reads lengths on the host. The subsequent
binding is shape-local: boundary values may change at fixed T/B without
retuning or recompilation. Explicit refinement accepts the new profile:

```python
result = router.autotune_(x, cu_seqlens=cu, phase="train",
                         tuning_policy=rtx.AutotunePolicy.preset("quick"))
```

Warm cold selection and the bound hot call on the intended stream before
manual graph capture. Backward owns its normalization history, mapping and
immutable execution policy; it never reinterprets old state using new offsets
or the latest winner. No user scratch or per-sequence graph management is needed.

Packed routing also searches structural alternatives:

- `RoutingKernelConfig(packed_row_backend="warp")` keeps each row's FP32
  values in registers and uses warp-local Sinkhorn reductions instead of the
  original CTA/shared-memory reduction. Forward/backward thread counts remain
  independent; `row_groups` is inactive for this backend and canonicalized by
  the tuner. Column reduction geometry is unchanged.
- `packed_column_cache_items=4/8/16/32/64` caches that many exact radix keys per
  thread for QB column selection. Zero streams as before. Columns larger than
  the cache stream in full; no quantile approximation or truncation is used.
  With the Torch QB backend this field is inactive.

Both choices are optional, exposed to the shared autotuner and encoded in
immutable execution tokens. They are inactive for a flat pool without offsets.
The defaults remain CTA rows and streaming columns; automatic tuning may choose
either implementation. Packed runtime winners use revision 7/source-safe keys;
older routing tokens still decode with newly appended defaults.
Early structural candidates cover ungrouped column grids as well as grouped
ones: B*E is often too small to fill an RTX's SMs after grouping. This is a
measured candidate preference, not a forced dispatch. Residual model features
record row/column CTA counts, saved-state bytes and column-cache capacity.
Packed backward saves only normalization probabilities and mapping tensors it
actually uses, not an unused final log-transport allocation.

## What is differentiated

The pipeline is:

1. Ordinary Torch FP32 projected/RMS-normalized sigmoid affinities.
2. Five finite log-Sinkhorn iterations by default.
3. Detached QB assignments and quota-constrained balance repair.
4. Gather selected **Sinkhorn log-values**, then per-token softmax.

Gradients follow step 4 through Sinkhorn and the affinity model. QB and repair
have **no backward**. There is no identity straight-through surrogate for
their integer decisions, and no rerouting in backward. The configurable
`sinkhorn_iterations`, `dual_iterations` and `repair_rounds` are algorithmic
budgets (defaults 5/5/64), not performance-autotuning axes. Exhausting repair
without exact capacity raises a device-side assertion; no token-drop or
round-robin fallback silently changes the policy.

The native Sinkhorn saves its final log-transport and every row/column
normalization probability. Backward applies the saved normalization Jacobians;
it does **not** rerun Sinkhorn or recompute those exponentials. Saved state
costs `(2 * iterations + 1) * N_scheduled * E * 4` bytes, excluding incoming
scores and temporary workspace. For 1,024 scheduled tokens, 128 experts and
five iterations this is 5.5 MiB; for 40,000 tokens it is about 215 MiB.
This deliberate no-recomputation policy is not memory-free. Native gradients
are first-order; higher-order differentiation is not qualified.

## Execution and dependencies

`backend="auto"` uses the CPU Torch reference for CPU inputs and native routing
on supported RTX CUDA devices. `backend="torch"` explicitly selects the
portable policy reference; `backend="cute"` requires CUDA. There is no silent
native-to-reference fallback after a compiler or capacity failure.

Sinkhorn, QB order statistics, quota selection and conditional predicates use
CuTe DSL kernels. Host graph control uses `cuda-python`. **Production routing
does not build CUDA C++, use our CUB implementation, or require NVCC/CCCL.**
The optional historical CUB baseline lives under
`benchmarks/routing_reference`, outside the installed runtime package.

QB margin arithmetic and final/proposal top-k indexing retain
the original policy as compiler-visible Torch/Inductor operations. Metadata
can use that reference composition or exact stable CuTe construction. They have
not all been replaced with handwritten CuTe code. GPU conditional graphs stop
repair when balanced or the unchanged budget is exhausted. Hot replay has no
Python per-round loop or host load predicate. Active quota proposals are packed;
this is not a claim that all proposal construction is variable-length/fused.

Use the normal supported RTX stack plus CUDA Python graph bindings. The
qualified bindings are `cuda-python==13.3.1`; the optional `routing` extra
ensures this minimum without changing the other kernel families' requirements:

```bash
python -m pip install -e '.[routing]'
```

Older bindings must expose the required conditional-node/graph-parameter APIs
or native routing fails with an actionable message. Normal Torch/Inductor
tooling still applies. Import and fake tracing do not compile or initialize CUDA.

The first ordinary call also prepares private GPU plans automatically. No
caller stream management, scratch allocation or explicit prepare API is
required. Warm required shapes **on the intended capture stream** before
manual CUDA graph capture. Cold `torch.compile(fullgraph=True, dynamic=False)`
is tested. Shape changes create separate cached plans; symbolic shape/export,
distributed training and concurrent replay of the same captured graph on
different streams are not qualified. Independent current-stream plans are
tested; this is not an unrestricted graph-concurrency guarantee.

For fixed or bound policies, the traceable wrapper allocates Sinkhorn outputs,
saved history and temporary FP32 workspaces outside a private launch-only
mutation operation. Inductor can see those allocations and logical views.
Forward/backward geometry is immutable metadata saved with the forward;
backward never consults the latest winner. Automatic first-hit selection still
uses the actual input. Its cold invocation carries an immutable CPU token;
one subsequent shape-local guard transition specializes the selected hot path
without creating another token. Complete both calls before manual capture.

The hard scheduler's returned IDs are also compiler-allocated outside its
launch-only operation. The conditional repair graph still owns bounded,
stream-local fixed-address state prepared on the cold path. This state is not
an output alias or a user workspace. Changing output ownership does **not**
claim that all conditional graph-pool storage becomes compiler-reusable.
No eager Torch numerical operations or per-round Python loop run inside the
hot hard-scheduling launch boundary.

RTX's normal cache/resource management retains plans needed by live compiled
callables or captured graphs. Routing plan count follows
`RTX_ROUTING_HARD_CACHE_ENTRIES` (default 8). See
[execution resources](execution_resources.md) for lifetime and cleanup controls.
Byte accounting covers explicit plan tensors; captured graph-pool allocations
are not all included in that tensor count. Live owners can retain a plan beyond
the unowned cache entry limit.

## Bounded first-hit autotuning

Without an explicit kernel config, `autotune` defaults to `"balanced"`. The
first CUDA input selects and records a schedule, including when the first call
arrives through `torch.compile(fullgraph=True, dynamic=False)`. Subsequent calls
reuse it. An explicit `config` or `sinkhorn_config` defaults to `autotune="off"`;
set the mode explicitly to use it as a tuning seed instead.

```python
router = rtx.BalancedRouter(
    768, 128, top_k=8, device="cuda", autotune="balanced",
    tuning_policy=rtx.AutotunePolicy.preset("quick"),
    autotune_cache_dir="/path/to/rtx-cache",
)
compiled = torch.compile(router, fullgraph=True, dynamic=False)
route = compiled(x)  # automatic, bounded first-hit selection
```

Modes are `off` (explicit/portable seed), `cache` (reuse or seed on a miss), and
`balanced` (reuse or search on a miss). The shared `AutotunePolicy` effort levels
are quick/balanced/thorough/extensive: 8/16/32/64 unique trials, with soft time
allowances of 15/30/90/180 seconds. An in-flight compile/measurement is not
preempted. Confirmations can add evaluations of an already measured config.
`cache_behavior="refine"` resumes compatible history to a larger effort cap.

Train (forward plus metadata/Sinkhorn backward) and no-grad inference are distinct
contexts; use `torch.no_grad()` or `torch.inference_mode()` for inference.
Policy budgets 5/5/64 are never tuning axes. Numerical gates and exact hard
policy checks cannot be relaxed to win a benchmark.

Search uses the shared UCB orchestration/local/structural search infrastructure
under `rtx.autotune`, not a router-specific search engine. It records independent
context residual JSONL journals under `routing_effort` and revision/source-safe
schema-v2 winners under `runtime_winners/balanced_routing`. Identity includes
device/toolchain, real and padded pool sizes, E/K, model/router widths, phase,
policy budgets, numerical settings and measurement contract. Inactive settings
are canonicalized so an eight-trial budget is not spent on duplicate schedules.

### Flat-pool transport choices

Long flat pools can expose too few column-owning CTAs to occupy an RTX GPU.
`RoutingKernelConfig` therefore also offers:

- `flat_column_shards_forward` / `flat_column_shards_backward`: independently
  choose 0 (original column owners) or 4/8/16/32/64/128 hierarchical shards.
  Partials and saved probabilities stay FP32. The sharded implementation
  assigns four adjacent experts per column group for coalesced row-major reads.
- `flat_row_backend="cta"` / `"warp"`: original CTA rows or register-resident
  warp rows (currently E<=128). Warp rows canonicalize inactive `row_groups`;
  row thread counts and column geometry remain independent tuning choices.

These fields are inactive for packed independent pools. Explicit configs keep
the original zero-shard/CTA defaults. With progressive tuning and default
opening reservations, eligible long underfilled flat pools contrast both
families within the first eight trials. A shape/SM-count heuristic proposes
roughly two column-CTA waves; it does not force a winner. Short pools, unknown
SM counts and explicit opening reservations retain their ordinary behavior.

Beyond eight trials, a coupled `flat_transport` candidate generator competes
under the existing UCB scheduler with structural, local and random search.
It contrasts column coalescing, shard depth, row ownership and hard-routing
backends around timing-eligible completed 8/16/24-trial prefixes. This is not
prediction-only pruning or a separate timing engine. Increasing effort keeps
the same compatible journal prefix. `region_coverage=False` disables this
structural opening and its associated generator for ablation.

Flat routing uses kernel revision 3 and full source-safe identities. Legacy
tokens migrate with zero shards and CTA rows. Cold selection, durable cache
hits, bound execution and backward all honor the complete selected token;
refining the winner cannot change a pending backward's saved plan. Both Torch
and native selected-softmax adjoints center the upstream before their FP32
reduction to avoid cancellation on nearly constant gradients. The Torch
composition remains visible to Inductor, not hidden inside a launch custom op.
See the [measured qualification](autotune_cycle3_20261120.md) for gains and limits.

### Affinity projection and scalar-gradient layout

`affinity_layout="auto"` (default), `"scalar"`, or `"row_broadcast"` controls
the compiler-visible FP32 affine/log-sigmoid part of the router. The row option
uses logical broadcasts to expose per-token reductions for `logit_scale` and
`score_bias`; it does not materialize an expanded parameter tensor or change
the mathematical affinity. Only the FP32 scalar-gradient sum association may
differ. Explicit `scalar` and `row_broadcast` pin the implementation.

With `auto`, trainable scalars, eligible CUDA inputs and tuning enabled, a
**separate two-candidate affinity context** measures the complete projection,
RMS normalization, affinity, and every requested adjoint. It uses the shared
autotuner, independent residuals and `balanced_router_affinity` runtime winners.
Its budget is separate from the transport/QB study, like the router/expert/
shared contexts in `BalancedMoE`; cold-start wall time is not free. Row layout
must preserve compiled scores exactly on caller and nonzero-projection probe,
pass gradient/replay checks and win paired timing before promotion. Tiny/tied
cases retain scalar. `off`, no-grad and autocast auto paths retain scalar;
`cache` reuses a compatible affinity winner or scalar on a miss.

The cold compiled call contains a no-output ordered preparation effect, with
no affinity tensor copy or surrogate gradient. A normal shape-local guard
transitions to the selected hot graph, where the affinity remains visible to
Inductor. No private per-layer graph or persistent tensor workspace is added.
Explicit `router.autotune_(..., phase="train")` also prepares this context.
Recorded benchmark router sidecars include the frozen affinity layout so
switching benchmark tuning off cannot accidentally discard the chosen plan.
See [the follow-up study](balanced_moe_search_and_kernels_20261118.md).

Candidates are checked against an independent FP64 normalization/gradient oracle
and the frozen hard-routing policy. Timings use graph replay after calibrated
warmup, with compilation and reference construction excluded. Calibration uses
both the caller's scores and a sigmoid-normal fixture: uniform initial routing
alone must not choose the repair schedule. The search objective covers Sinkhorn,
hard routing, stable expert metadata and their differentiable adjoints; affinity
projection is checked separately in full-router benchmarks. Routing revision 2
invalidates the earlier objective's winners rather than comparing timings that
excluded metadata. It is not a guarantee of a global optimum or a whole-MoE win.

When no timing samples are eligible, runtime can retain a correctness-checked
seed process-locally. It does not persist unstable timing as a latency winner.
Offline explicit tuning fails instead of labeling that result a measured win.
Enable INFO logging for `rtx.routing.autotune` to inspect cold selection progress.

For deliberate pre-compilation tuning:

```python
result = router.autotune_(
    x, phase="train", tuning_policy=rtx.AutotunePolicy.preset("thorough"),
    output_dir="/path/to/rtx-cache", progress=print,
)
```

This populates the same durable cache; it does not impose one shape's winner
on every later shape. Bound backwards retain immutable scalar configuration
metadata from their own forward; a cold forward instead retains its immutable
CPU token. Cache retirement or later refinement cannot change which policy
interprets that saved state. No device scalar is read back.

For standalone collection on another machine:

```bash
python -m rtx.autotune.routing --tokens 4096 --d-model 384 \
  --experts 128 --top-k 8 --effort balanced --output routing-data
```

Re-run with a higher effort cap to resume compatible residuals. Copy the whole
output directory to retain observations, session provenance and deployed winners.
`--trials` and `--seconds` permit custom budgets; `--phase forward` tunes no-grad
inference. This collects synthetic router data, not end-to-end MoE training data.

## Explicit execution configuration

```python
from rtx.routing import RoutingKernelConfig, SinkhornKernelConfig

router = rtx.BalancedRouter(
    768, 128, top_k=8, device="cuda",
    config=RoutingKernelConfig(
        forward=SinkhornKernelConfig(threads=128, row_groups=1, column_groups=4),
        backward=SinkhornKernelConfig(threads=128, row_groups=1, column_groups=4),
        qb_backend="native", qb_threads=128,
        segmented_quotas=True, quota_threshold=1024,
        metadata_backend="cute", metadata_threads=256,
        quota_cache_items=8, quota_threads=128, quota_early_exit=True,
    ),
)
```

Threads may be 64/128/256, and independent row/column reduction groups may be
1/2/4/8. These are implementation geometry, separate from numerical budgets.
Forward and backward geometry tune independently. `qb_backend` is `torch`
(reference order statistics), `native_column` (CuTe expert quantiles),
`native` (CuTe radix token/expert quantiles), or `native_warp` (warp-local row
order statistics, with the same radix column quantiles). The optional
`native_warp` family supports E<=128: one warp owns a row and sorts its ordered
FP32 keys in registers, with no shared histogram or CTA-wide synchronization.
It preserves radix signed-zero/NaN/Inf ordering; it does not change QB rounds
or assignment policy. All variants write only the selected statistic, not a
sorted value matrix. `qb_threads` is 64/128/256; for the warp family this means
2/4/8 independent rows per CTA. The default remains unchanged and the
autotuner decides whether the new family is worthwhile.

`segmented_quotas=True` selects packed CuTe radix thresholds when proposal count
N*K reaches `quota_threshold`; otherwise it uses CuTe direct ranking. False
always selects direct ranking. Both preserve `(priority, original position)`
ties and source-before-target acceptance. The direct implementation is bounded
to 4,096 proposals; native repair currently supports at most 1,024 experts.
These limits are rejected before compiling a tuning candidate.

`quota_threads` is independently selectable at 64/128/256. For the packed
implementation, `quota_cache_items=4/8/16` caches that many keys per thread.
Segments exceeding the cache stream all their keys; none are truncated.
The default `0` retains streaming radix selection. `quota_early_exit=True`
can stop histogram passes once the selected prefix contains exactly one key,
recovering its complete priority-and-position key before acceptance. Extra
registers and recovery work are not always beneficial; these are separate
autotuner axes, not assumed wins. Inactive cache/early-exit settings are
canonicalized for direct ranking.

`metadata_backend="cute"` replaces sorting with chunk-local ranks, per-expert
prefix counts and stable emission. It preserves exactly the reference's
token-major assignment order within every expert, including dummy slots.
Selected-log softmax and its saved-weight Jacobian remain FP32; dummy rows
have exactly zero gradient. Output and scratch allocations are outside
launch-only mutation operators. The implementation supports E<=128 and K<=32;
`metadata_threads` is 128/256/512. `metadata_backend="torch"` retains the
compiler-visible reference composition and is still the default seed.

For packed independent pools, `packed_column_shards_forward` and
`packed_column_shards_backward` independently select `0/4/8/16/32` column
reduction shards. Zero preserves the legacy single-group reduction. Sharded
columns use a coalesced partial pass and deterministic FP32 final reduction
and apply pass, while retaining all saved Sinkhorn normalization states.
Forward and backward shard counts, threads and `column_groups` tune
independently; each sharded column group covers four neighboring experts.
Individual empty sequences and runtime-mutable packed boundaries remain
supported. Scratch allocation is compiler-visible, outside the launch-only
mutation operator; users do not manage workspaces or streams.

`packed_qb_column_backend="row_major"|"expert_keys"` selects the original
QB column traversal or an ordered-key construction pass in expert-major
layout followed by contiguous exact radix selection. This adds a key buffer
and a launch but can reduce repeated strided column traffic. Rank, signed-zero
ordering, tie rules and repair policy are unchanged. Cache capacity and thread
count remain autotunable, and oversized segments stream rather than truncate.
This does not transpose model activations or change the numerical policy.

For bounded small pools, `packed_sinkhorn_backend="persistent"` keeps all
Sinkhorn iterations in one CTA per sequence. The forward writes the original
FP32 row/column probabilities and final transport; backward uses those saved
probabilities and writes directly to real-token gradients. Independent
`"persistent_forward"` and `"persistent_backward"` choices mix the persistent
and staged directions. No policy budget, QB/repair logic, or physical layout
changes. When the static bound fits one CTA (at most 512 scheduled rows and
80 KiB), every pool is eligible. Otherwise, a multi-pool pack can use a
hybrid: pools with at most 128 scheduled rows use persistent CTAs; longer
pools use the configured staged implementation. Empty pools write no
logical rows. A CUDA Graph conditional node skips the complete staged
transport for an all-short replay. Eager execution launches both masked
paths without reading boundaries back. Workspace allocation and graph
topology remain shape-static; a replay may change which pools run each
path. The optional `packed_persistent_threads=512/1024` override widens
persistent CTAs only; zero uses the direction's ordinary thread count.
This trades parallelism for fewer launches; it is not necessarily faster.

The packed search always checks its inherited initial configuration. Its
two opening proposals on all-short eligible training shapes include both-direction
and backward-only persistence; inference considers two forward CTA sizes.
With a mixed cold profile, one opening remains staged. A cold profile with
no eligible pools keeps the original large-pool openings.
Persistent forward candidates must match hard metadata exactly against a
staged transport with the same FP32 reduction order on both calibration
fixtures. On larger shapes the opening proposals remain sharded transport
with row-major and expert-major QB layouts. These openings are subject to
the same trial budget and soft deadline. Local/random/model-guided tuning
and all legacy families remain available.

Immutable execution tokens retain their original first ten fields. The
current 22-field token adds metadata/cache/thread/early-exit and packed
row/column/layout choices; the compatibility decoder supplies defaults for
older 10/14/15/17/19/20-field tokens. Runtime winners additionally require the current routing
revision and source/measurement identity; token decoding is not permission
to reuse incompatible old timing evidence.

The defaults are portable seeds, not universal winners. Grouped expert
execution and whole-MoE autotuning remain in the
[BalancedMoE design](balanced_moe_design.md).

## Qualification and provenance

See [CuTe/autotuner qualification](balanced_router_cute_autotune_20261115.md)
and [initial bring-up evidence](balanced_router_qualification_20261115.md).
The FP32 fixed-K policy/reference and native quota/control code were ported
from the standalone BiMoE handoff `routing-research-port-20261111`, not the
variable-route ConstrainedMoE implementation. The extracted policy source
SHA256 is recorded in `rtx/_routing_reference.py`. Numerical tests use an
independent FP64 normalization oracle; hard-policy parity freezes input scores
and compares against the original compiled finite-round algorithm.
