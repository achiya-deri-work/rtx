# MLP first-hit execution-plan handoff

Status: connected to an experimental opt-in MLP launcher and kernel-only backward.
Existing explicit MLP execution remains unchanged. This is not yet a
performance-qualified replacement for explicit tuning.

Current integration: private selected-op revision 4, replay-op revision 3 and
plan-spec revision 8 route the caller's `torch.compile` CUDA graph option to
cached **CUDA launch graphs**, without internal `torch.compile` calls.
The host metadata boundary stays outside capture. Old Inductor artifacts
cannot bypass the new routing; durable kernel winners keep their numerical
revision because the kernel/config contract did not change. No cache deletion is
required. `autotune='off'` and ordinary compiler defaults remain unchanged.
The dated experiment sections below preserve the progression and former limits.

Each replay entry is keyed by policy, activation, device, tensor shapes/strides/
dtypes, caller stream and (for backward) gradient mask. The forward/backward
caches each hold at most 32 entries. Cold setup warms the kernel sequence and
captures once; repeated invocations only stage operands, replay and publish
independent outputs. New input/weight values are copied every call; weights
are never silently frozen. Private replay adds memory and device copies, so
it is not guaranteed faster than host feeding for compute-bound shapes.

Graph-owned resources retain their scratch runners even if lower-level caches
evict them. `clear_runtime_caches()` synchronizes by default and clears private
MLP graphs too. It must still not be used while an external captured graph owns
those runtime resources. The existing post-search cleanup remains in place.

Manual outer capture records the already-warmed kernel sequence directly,
without staging or launching a nested graph. Warm on the capture stream first.
Without requested replay, the selected backward invokes kernels directly;
there is no hidden compiler in either path. User-side `torch.compile` remains
responsible for the surrounding model. Explicit-config MLPs and linears continue
to use the user's outer compiler/capture rather than private per-layer compilers.

Current local validation: 21 integration tests plus five kernel-only replay
tests passed on RTX 5070 Ti / Torch 2.13.0+cu132. These cover changed operands,
SiLU/GELU, ragged/aligned shapes, partial/full gradients, overlapping backward,
multiple live outputs, external capture, scratch eviction, stable live memory,
and a guard that forbids internal `torch.compile`. Historical timings below
describe the previous nested-compiler implementation, not this replacement.
The same five kernel-only tests plus nine automatic/lifetime CUDA tests also
pass on Torch 2.12.1+cu132 (14 tests, 56.02 seconds) on this device.

Kernel-only replay measurements on the same GPU, using fresh processes,
identical **untuned default configs**, 20 CUDA-event samples × 10 calls after
warmup (no optimizer):

| M/K/H/O | Direct host-fed fwd+bwd | Private launch graph | Outer CUDA graph |
|---|---:|---:|---:|
| 64/128/256/64 | 336.1 µs | 343.5 µs | 323.9 µs |
| 512/768/1536/768 | 670.0 µs | 706.9 µs | 665.0 µs |

These are not speedups over the previous compiled implementation or tuned
BF16 comparisons. Private staging regresses these compute-heavy defaults;
outer capture avoids its copies. A narrow Nsight API trace of ten replayed
fwd+bwd calls records 20 graph launches, no individual host kernel launches,
and 110 device-to-device copies. Capture/compilation is absent from that range.
Kernel-node timing was not enabled in this trace, so it establishes the launch
contract, not per-kernel utilization. Reproduce timings with
`tools/benchmark_mlp_kernel_replay.py --mode direct|replay|outer --shape M,K,H,O
--output NEW.json` (one mode/process at a time).

Use `torch.compile(..., mode="reduce-overhead", fullgraph=True, dynamic=False)`
to request automatic replay. It is a choice, not a universal performance win.
On the local 5070 Ti / Torch 2.13.0+cu132, fixed-default-config, host-fed training
measurements (30 samples × 5 iterations, compile/search/warmup excluded) were:

| M/K/H/O | Plain compiled | Automatic replay | Scope |
|---|---:|---:|---|
| 64/128/256/64 | 457.4 µs | 372.2 µs | one fresh process per mode |
| 512/768/1536/768 | 686.0–687.1 µs | 706.1–710.1 µs | two processes per mode, reversed order |

These are CUDA-event end-to-end times including host launch gaps, not isolated
kernel latencies or comparisons against BF16. The small case is about 18.6%
faster; the decoder case is 2.9–3.5% slower. Cold first forward+backward took
2.852 vs 2.955 seconds (small) and 3.784–3.830 vs 3.930–3.953 seconds (decoder).
Inductor caches were isolated per process; shared CuTe/CUDA caches were not
purged. No search was run. Warm compilation counters stayed unchanged and all
four numerical outputs passed the existing independent recipe gate.

Raw samples, config, environment, and commands are in
`autotune_reports/mlp_replay_cost_20260914_v1/`. Reproduce one mode at a time:

```bash
TORCHINDUCTOR_CACHE_DIR=/tmp/rtx-replay-cost-fresh .venv/bin/python \
  tools/benchmark_mlp_automatic_replay.py --mode replay \
  --shape 512,768,1536,768 --output /tmp/rtx-replay-cost.json
```

Use a distinct cache/output for `--mode plain`. There is no claim of statistical
cross-shape or cross-device superiority, no counter/timeline attribution, and no
promotion of a learned search policy from these measurements. The integration
honors the explicitly requested compiler mode rather than forcing replay.

Revision-3 integration passed **32 tests and two subtests in 86.45 seconds**:

```bash
.venv/bin/python -m pytest -q tests/test_mlp_automatic_graphs.py \
  tests/test_mlp_first_hit_cuda.py tests/test_execution_plan_token.py \
  tests/test_composite_effort_namespace.py tests/test_mlp_precision_api.py \
  --junitxml=/tmp/rtx_mlp_replay_integration_20260914.xml
```

The ordinary first-hit `reduce-overhead` test now requires replay without any
research-hook call. Tests retain actual tuning/refinement, durable reuse,
ragged/masked gradients, opcheck, external capture, outstanding-output lifetime,
steady memory, and unchanged explicit precision APIs. Qualification was on
Torch 2.13.0+cu132 / local 5070 Ti, not all supported software/device combinations.

```python
mlp = rtx.MXFP8MLP(
    768, 1536, device="cuda", autotune="balanced",
    tuning_policy=rtx.AutotunePolicy(
        max_trials=16, max_seconds=90,
        search_schedule="progressive", cache_behavior="refine",
        finalist_rounds=2,
    ),
    autotune_cache_dir="autotune_cache",
)
compiled = torch.compile(mlp, fullgraph=True, dynamic=False)
```

`off` remains the default. `cache` loads compatible durable results without
searching and falls back to the constructor config. `balanced` uses the shared
autotuner, residual evidence namespace, paired promotion, and runtime winner
locking. Grad-enabled calls select for complete forward/backward; no-grad calls
select for forward. Explicit `configure_shape_` / `autotune_` results override
implicit selection for their shapes. Automatic selective-precision and BF16
paths are not supported; explicit precision APIs retain their existing behavior.

The cold tuning harness runs on a dedicated host thread using the caller CUDA
stream, outside the custom operator's below-autograd dispatcher state. Candidate
tensors are explicit compiled arguments, not closure-captured inputs. This
measurement contract has revision 35; prior revision-34 records remain intact.
No work thread is created for warm calls or compatible durable cache hits.

The backward bridge compiles existing registered operators and gives each plan
an isolated Dynamo frame cache. Its bounded caches retain metadata/compiled
functions, not caller input tensors. Existing explicit paired backward uses one
packed allocation. The selected-plan bridge instead uses a private registered
operator that gives the **same paired runner** separate output buffers. It
preserves the selected quantization/execution schedule without staging copies,
splitting the paired GEMM workflow, or changing precision. Intermediate
recomputation remains compiled. The original explicit path is unchanged.

Runtime selection cannot be repeated independently in backward. Another forward
may refine a durable winner while an older forward still has an outstanding
backward. That older backward must recompute with its own selected configuration,
including joint/two-pass recomputation, precision, and overlap decisions.

`rtx/_execution_plan_token.py` provides a private, self-contained CPU uint8
metadata output containing the complete revisioned configuration string. It uses
a fixed 64 KiB extent so fake implementations and Inductor can allocate it before
selection. The versioned header bounds the payload and checks its CRC. Oversized
configurations are rejected, never truncated. This is not a security boundary.

The token owns no CUDA storage, intermediate activation, runner, or model. It
does not depend on an in-process ID registry. Tensor aliases, copies, and ordinary
tensor serialization preserve the plan. Its host memcpy/read helpers perform no
eager Torch tensor operations or device synchronization. The surrounding private
operator must allocate a fresh token per outstanding forward and must not mutate
it after publication; retaining saved-tensor semantics remains its responsibility.

## Evidence

On local Torch 2.13.0+cu132 / RTX 5070 Ti:

```bash
.venv/bin/python -m pytest -q tests/test_execution_plan_token.py
```

Five tests passed. Coverage includes actual MLP configuration round-trip,
storage alias lifetime, clone/serialization, concurrent independent tokens,
malformed metadata, and rejection of fake tensor data access. A synthetic private
operator returns a CUDA result and CPU token under `torch.compile(fullgraph=True)`.
Two outstanding forwards choose different plans and backward in reverse order
uses the correct original plans with one Dynamo compilation. `opcheck` passes on
CPU and CUDA. Warmed CUDA graph capture and three forward/backward replays retain
the captured plan after the Python selector changes.

JUnit evidence: `/tmp/rtx_execution_plan_token_capture_20260914.xml`.

The later consolidated integration run passed **21 tests and two subtests** in
35.44 seconds on the same GPU/software stack:

```bash
.venv/bin/python -m pytest -q -s \
  tests/test_mlp_first_hit_cuda.py tests/test_execution_plan_token.py \
  tests/test_composite_effort_namespace.py tests/test_mlp_precision_api.py
```

Artifact: `/tmp/rtx_mlp_first_hit_milestone_v2_20260914.xml`.
Actual MLP coverage includes M=K=H=O=128, SiLU, full training, real 8→16 trial
refinement, journal immutability on repeat/cache reuse, and three capture
replays. On that case output/dX/dDown matched the independent quantized oracle
exactly; dGateUp normalized RMS was 0.000046899 (maximum absolute error
0.00000190735). These are **quantized-recipe errors**, not errors versus an
unquantized BF16 model. Additional GELU two-pass tests use M=17/K=35/H=97/O=41,
each single-gradient mask, noncontiguous upstream gradients, and full opcheck.
All were below the unchanged 0.025 normalized-error gate. The suite also covers
the existing precision API and the updated 64 KiB metadata ABI.

## Copy-free bridge qualification

The subsequent direct-output run passed 12 tests in 34.73 seconds:
`/tmp/rtx_mlp_first_hit_direct_outputs_lifecycle_20260914.xml`. It repeats
first-hit/refinement/capture/partial-gradient checks and includes the existing
frontend suite. Additional two-pass and joint/overlapping all-gradient cases
retain outputs across later invocations, check distinct storage identities, and
verify unchanged values. Four synchronized allocation readings across 20 warmed
iterations were constant for each case (68,707,328 and 68,639,744 bytes,
respectively). Those totals include other process-resident test allocations;
they are not per-layer workspace sizes.

Fresh-process, fixed-default-config replay comparison on the local 5070 Ti:

| M/K/H/O | Previous copy bridge | Direct-output bridge | Observed reduction |
|---|---:|---:|---:|
| 64/128/256/64 | 325.47 µs | 324.20 µs | 1.27 µs (0.39%) |
| 512/768/1536/768 | 669.47 µs | 662.31 µs | 7.16 µs (1.07%) |

Each result uses 30 CUDA-event samples of ten forward/backward graph replays,
excluding compilation, warmup, capture and autotuning. Both variants check all
outputs against the independent quantized oracle before timing. These are single
fresh-process comparisons, not replicated confidence intervals, an exhaustive
shape study, or a BF16 speedup claim. Host first-hit/repeated-call overhead is
not measured by graph replay.

Raw samples, exact configs, numerical errors and environment are under
`autotune_reports/mlp_first_hit_bridge_20260914_v1/`. Reproduce with distinct
output paths (the driver refuses overwrites):

```bash
.venv/bin/python tools/benchmark_mlp_first_hit_bridge.py \
  --bridge direct --shape 512,768,1536,768 --output /tmp/mlp-direct.json
.venv/bin/python tools/benchmark_mlp_first_hit_bridge.py \
  --bridge copy --shape 512,768,1536,768 --output /tmp/mlp-copy.json
```

The copy variant reconstructs only the previous bridge inside that process;
production sources and numerical policy are not changed. Broader startup,
effort-level, shape-churn and multi-device qualification remain open.

A subsequent [fresh-process effort qualification](mlp_first_hit_effort_qualification.md)
completed the actual 8/16/32/64 training ladder on one additional shape and
tuner-forbidden durable reuse, plus an eight-trial ragged inference case. That
extends lifecycle evidence; it does not establish universal effort-level quality.

## Automatic versus manual CUDA graphs

Automatic `torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)`
is a separate contract from manual warmed capture. A direct probe on Torch
2.13.0+cu132 found that the explicit MLP invoked its Python forward launcher on
the first two steps and zero times on steps 2–7: automatic replay was active.

The initial first-hit integration instead crashed on step 1 (the second call).
Inductor split the CPU plan extraction from the GPU output, carrying the complete
mixed-output tuple across a captured partition boundary. CUDA graph trees then
asserted while processing that tuple as an output. This was not a numerical
kernel failure or a missing warmup.

The private selected forward/backward operators are now tagged
`torch.Tag.cudagraph_unsafe` for automatic partitioning. Their private names are
versioned to avoid reusing previously cached unsafe partition code. The tag is
scoped to these host-plan boundaries, not the entire model or process. It does
not prevent manual warmed CUDA capture, which still passes.

After this correction, first-hit reduce-overhead executes correctly but its
forward launcher still runs on every step. **Do not claim automatic replay or
reduce-overhead performance for this boundary.** Capturing the GPU-only selected
execution independently of the host-plan handoff remains an optimization task.

Evidence:

- `autotune_reports/mlp_automatic_graphs_20260914_v1/explicit.json`
- `autotune_reports/mlp_automatic_graphs_20260914_v1/first_hit_guarded.json`
- `/tmp/rtx_mlp_automatic_graph_guard_20260914.xml`: nine tests passed in 42.13 s.

The regression suite uses separate processes, updates X and down weights after
warmup, and checks all gradients against the independent quantized recipe. It
also reruns real first-hit refinement, ragged gradients/opcheck, manual capture,
and repeated-allocation tests. The probe counts forward launcher calls; it does
not separately establish backward replay counts or measure performance.

```bash
.venv/bin/python -m pytest -q \
  tests/test_mlp_automatic_graphs.py tests/test_mlp_first_hit_cuda.py
```

### Nested replay: corrected probe and ongoing qualification

A GPU-only nested compilation prototype separates CPU plan handling from the
forward/backward device functions. It is **not registered by production code**.
The probe can enable it explicitly in its own process using `--nested-replay`.

The original probe on Torch 2.13.0+cu132 passed the first two numerical
checks, but the third failed CUDA graph trees' live-allocation validation:
one pool allocation was not tracked as an output. Disabling only backward replay
did not resolve it: the forward-only experiment failed the same check. Allocation
history did not provide an origin stack. Subsequent inspection found the owner:
the probe's `MagicMock(side_effect=...)` retained launcher arguments, including
graph-pool output buffers. Replacing it with a non-retaining wrapper resolves
the failure without disabling any pool check. This artifact is instrumentation-
contaminated evidence, **not evidence that the runtime is memory-unsafe**.

The failure is preserved in
`autotune_reports/mlp_automatic_graphs_20260914_v2/rejected_nested.json`, including
completed numerical steps and the traceback. Reproduce only as a diagnostic:

```bash
.venv/bin/python tools/probe_mlp_automatic_graphs.py \
  --selection first_hit --nested-replay --output /tmp/mlp-nested-rejection.json
```

Append `--forward-only-replay` for isolation or `--allocation-history` for
allocation diagnostics. Reports refuse overwrites and are saved on failure.
Normal first-hit execution still retains the guarded host boundary and warmed
manual-capture support; the nested lowering remains research-only while the
broader matrix runs.

The corrected probe counts both forward and backward launchers and supports
shape, activation, gradient-mask, inference, and real eight-trial tuning modes.
`autotune_reports/mlp_automatic_graphs_20260914_v3/tuned_uncached.json` passes
eight training iterations with changing X and down weights; after iteration 2,
both launcher counts are zero. A ragged GELU 17/35/97/41 case also passes with
both counts zero. The worst relative error in the tuned case is 2.331e-5 against
the independent quantized recipe, below the unchanged 0.025 gate.

An important compiler-cache control: the earlier `tuned_training.json` reused
pre-lowering Inductor code for backward and kept three Python launches per
iteration. Disabling Inductor caches in `tuned_uncached.json` resolves that
specific observation. Research tests use isolated `TORCHINDUCTOR_CACHE_DIR`
directories. A production rollout must revision its private boundary, rather
than requiring users to clear caches or claiming stale code proves a kernel
limitation. Reproduce the cold compiler control with:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 .venv/bin/python tools/probe_mlp_automatic_graphs.py \
  --selection first_hit --nested-replay --tune --output /tmp/mlp-nested-tuned.json
```

After disabling the prototype, the guarded path passed 14 tests in 42.72 s,
covering automatic-mode correctness, first-hit tuning/refinement, ragged
gradients/opcheck, manual capture, metadata lifetime and repeated live allocation:
`/tmp/rtx_mlp_replay_rejection_regression_20260914.xml`.

These token tests validate the handoff architecture, not MLP kernel numerics or
runtime cost. Actual MLP tests are in `tests/test_mlp_first_hit_cuda.py`; broader
effort/shape/precision coverage and memory-growth qualification are still needed.
No new policy or learned prior is promoted by these integration tests.

The corrected batched replay qualification passed **19 tests in 67.31 seconds**:

```bash
.venv/bin/python -m pytest -q tests/test_mlp_automatic_graphs.py \
  tests/test_mlp_first_hit_cuda.py tests/test_execution_plan_token.py \
  --junitxml=/tmp/rtx_mlp_nested_matrix_20260914.xml
```

Nested replay cases use separate processes and isolated compiler caches:
ragged GELU full training (17/35/97/41), X-only / gate-up-only / down-only
gradients (64/128/256/64, SiLU and GELU), and decoder-size inference
(512/768/1536/768). All require zero forward and backward Python kernel-launch
calls after warmup and independently check updated operands. This proves replay
for those paths, not timing quality. The existing first-hit, manual-capture,
plan-token, and memory tests in this batch exercise the normal guarded path;
they do **not** establish nested replay's outstanding-output lifetime or
external-capture compatibility. Those remain gates before enabling the lowering
in production. No default was changed.

### External capture and outstanding outputs

`tools/probe_mlp_replay_lifetime.py` adds independent subprocess checks for
two outstanding forwards followed by reverse-order backward, unchanged first
output storage, and steady live allocation across repeated steps. The external
capture case warms on its capture stream and checks both the original and
updated X/down-weight values after graph replay.

The original external test failed with `cudaErrorStreamCaptureUnsupported`:
CUDA does not permit the nested replay launch used here during an enclosing
stream capture. The boundary now detects external capture. Forward records its
already-warmed kernel runner directly; backward uses an additional graph-free
compiled body primed on the first noncapturing backward. This keeps intermediate
arithmetic compiled, does not disable pool checks, and does not change precision.
There is one extra backward execution/compilation on this experimental cold path,
not on warm replay. Its startup cost still needs measurement.

The pre-fix failure is preserved at
`autotune_reports/mlp_replay_lifetime_20260914_v1/external.json`;
the first corrected external test passes with zero recipe error in
`autotune_reports/mlp_replay_lifetime_20260914_v2/external.json`.
The first outstanding-forward probe also passes (maximum relative recipe error
1.744e-7) in the v1 directory. Later regression tests add changing captured
operands and a live-memory plateau check; these standalone artifacts do not
claim that expanded coverage.

The expanded consolidated run passed **21 tests in 78.16 seconds**:

```bash
.venv/bin/python -m pytest -q tests/test_mlp_automatic_graphs.py \
  tests/test_mlp_first_hit_cuda.py tests/test_execution_plan_token.py \
  --junitxml=/tmp/rtx_mlp_replay_lifetime_matrix_20260914.xml
```

In the outstanding-output subprocess, all four live-allocation readings were
587,776 bytes after warmup (20 measured steps). This is live allocator usage,
not reserved graph-pool memory or a many-shape cache-eviction guarantee.
External replay matched the quantized recipe exactly before and after operand
updates. The nested replay shape/mask matrix continued to require zero Python
kernel-launch calls after warmup. The prototype still needs a revision-safe
rollout and cold/warm cost comparison; these results do not promote an autotune
policy or claim cross-device performance.
## Torch 2.12.1 CPU qualification

The isolated `.venv-torch2121` uses Torch 2.12.1+cu132 and Python 3.12. CPU
construction of the selected MLP succeeds and `torch.Tag.cudagraph_unsafe` is
available. Installing pytest 9.1.1 added only test-runner dependencies in that
venv; Torch/CUDA packages were unchanged.

With CUDA explicitly hidden to avoid interfering with the frozen GPU study:

```bash
CUDA_VISIBLE_DEVICES='' .venv-torch2121/bin/python -m pytest -q \
  tests/test_composite_effort_namespace.py tests/test_mlp_precision_api.py \
  tests/test_execution_plan_token.py \
  --junitxml=/tmp/rtx_first_hit_torch212_cpu_20260914.xml
```

Ten tests and two subtests passed in 2.31 seconds; six CUDA-dependent tests
were skipped, not counted as GPU qualification. This establishes CPU-side
metadata/plan/namespace behavior on the second environment. Its current
first-hit/replay GPU matrix remains pending until the benchmark GPU is free.
