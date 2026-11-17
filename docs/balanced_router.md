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

QB margin arithmetic, final/proposal top-k indexing and metadata math retain
the original policy as compiler-visible Torch/Inductor operations. They have
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

Train (forward plus Sinkhorn backward) and no-grad inference are distinct
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

Candidates are checked against an independent FP64 normalization/gradient oracle
and the frozen hard-routing policy. Timings use graph replay after calibrated
warmup, with compilation and reference construction excluded. Calibration uses
both the caller's scores and a sigmoid-normal fixture: uniform initial routing
alone must not choose the repair schedule. The search objective covers native
routing stages; unchanged affinity/metadata work is checked separately in full
router benchmarks. It is not a guarantee of a global optimum or a whole-MoE win.

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
on every later shape. Backward retains an immutable CPU execution token from
its own forward, so cache retirement or later selection cannot change which
configuration interprets its saved state. No device scalar is read back.

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
    ),
)
```

Threads may be 64/128/256, and independent row/column reduction groups may be
1/2/4/8. These are implementation geometry, separate from numerical budgets.
Forward and backward geometry tune independently. `qb_backend` is `torch`
(reference order statistics), `native_column` (CuTe expert quantiles), or
`native` (CuTe token/expert quantiles). CuTe uses register-cached keys for short
vectors and streaming radix selection for long vectors; it writes only the
selected order statistic, not a sorted value matrix. `qb_threads` is 64/128/256.

`segmented_quotas=True` selects packed CuTe radix thresholds when proposal count
N*K reaches `quota_threshold`; otherwise it uses CuTe direct ranking. False
always selects direct ranking. Both preserve `(priority, original position)`
ties and source-before-target acceptance. The direct implementation is bounded
to 4,096 proposals; native repair currently supports at most 1,024 experts.
These limits are rejected before compiling a tuning candidate.

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
