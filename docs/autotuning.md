# Autotuning: runtime selection to portable datasets

## First-hit linear tuning

| Mode | Behavior |
|---|---|
| `None` | Environment override, otherwise balanced |
| `off` | Explicit or portable configuration |
| `cache` | Compatible verified winner; portable fallback; no search |
| `balanced` | Exact cache lookup, bounded search on miss, confirmation/persistence |
| `online` | Compatibility alias for balanced |
| `coordinate` | Explicit deeper search; legacy default can be 30 minutes |

Balanced defaults to the shared `AutotunePolicy`: 16 trials / 30 seconds per new context. These are search
budgets, not a hard interruption of an in-flight compiler or CUDA operation.
Backward and additional operand states can create their own contexts. Repeated
matching calls reuse selection. Filesystem locking and atomic publication avoid
duplicate same-context local work when processes share the cache.

Compile first and call normally—neither eager warmup nor an explicit tuning
call is required for `MXFP8Linear` and `NVFP4Linear`:

```python
layer = rtx.MXFP8Linear(768, 3072, device="cuda")
compiled = torch.compile(layer, fullgraph=True)
y = compiled(x)  # Tunes a missing forward context using real inputs.
y.sum().backward()  # Tunes a missing backward context.
```

Fake/meta tracing never benchmarks. The registered runtime launcher resolves
the request on real tensors; subsequent matching calls reuse its selection
without a graph recompile caused by cache mutation. A new shape can require
new tuning (and normal Torch shape specialization). Inductor-managed graph
warmup precedes replay; manually capturing a completely cold kernel is not
a supported way to initiate tuning. See [effort controls](autotune_effort.md).

Persistent runner scratch uses a separate CUDA memory pool rather than
Inductor's output-tracked graph pool. This avoids untracked live allocations
during `reduce-overhead` warmup. Scratch follows the existing bounded,
stream-specific runner caches; `rtx.clear_runtime_caches()` also drops scratch
pool ownership. Do not clear caches while a captured graph is still in use.
Cold compiled forward/backward, two static shapes, stable graph counts, and
reduce-overhead inference are tested on Torch 2.13.0+cu132 / RTX 5070 Ti.

`autotune_cache_dir` selects the winner root. `tuning_policy` provides advanced
policy control. Environment overrides include `RTX_AUTOTUNE`,
`RTX_MXFP8_AUTOTUNE`, `RTX_NVFP4_AUTOTUNE`,
`RTX_BALANCED_AUTOTUNE_TRIALS`, `RTX_BALANCED_AUTOTUNE_SECONDS`, and the
`RTX_BALANCED_AUTOTUNE_WARMUP/SAMPLES/CALLS_PER_SAMPLE` measurement settings.
Measurement defaults can differ by family; use the implementation's runtime
policy rather than assuming one iteration count everywhere.

Selection is scoped to device/software, kernel revision, shape, operand state
and policy/layout. Never copy a raw fastest config from another SKU and assume
equivalent performance. Runtime schema-v1 winners are invalidated; schema v2
includes the kernel revision. Preserve their source datasets for analysis.

## GLU/MLP distinction

Feature-branch modules default to tuning off, but accept the same
`autotune="balanced"`, `tuning_policy`, and `autotune_cache_dir` constructor
controls for opt-in first-hit selection. GLU tunes forward only within its
configured backend. Automatic MLP selection requires MXFP8 precision and
selects a forward-owned recomputation plan; it is not independent per-GEMM
linear winner installation. See [frontend controls](api.md#shared-frontend-controls).

Explicit `module.autotune_(example_x, ...)` is still available before
compile/capture. MLP can search whole forward or forward+recomputed-backward
time and install a shape policy. Keep exported policies separate from weights
and revalidate them on other workloads/devices.

## Run a campaign

Choose a current manifest from [the catalogue](../autotune_manifests/README.md),
then validate it before allocating GPU time:

```bash
rtx-autotune validate autotune_manifests/cross_device_dataset_v1.json
rtx-autotune probe
rtx-autotune run autotune_manifests/cross_device_dataset_v1.json \
  --device cuda:0 --output-dir autotune_datasets --format both \
  --wall-time 6h --context-slice 2m \
  --context-orchestration breadth_first
```

The named manifest is a command example, not a recommendation to rerun an old
study without checking its catalogue status. Same revision + manifest + output
root resumes compatible journals. Do not force identity adoption across kernel
or numerical-policy changes. The explicit adoption flags are for verified
runner-only changes, not a way to erase revision differences.

Breadth-first milestones provide coverage before deep refinement; the optional
context bandit allocates additional slices. A separate strategy bandit chooses
random, learned-global and local proposals inside a context. Both layers charge
time and preserve provenance; a configuration failure is not a zero-latency win.
`score=inf` during minimum-coverage allocation denotes exploration priority,
not a measured kernel result.

Use `--stall-timeout 180s` to enable the parent-process watchdog if appropriate.
Inactive and CPU-active compilation stalls have distinct deadlines. A watchdog
exit 75 is resumable; ordinary manifest/dependency errors need diagnosis.
An illegal memory access/instruction can poison CUDA context: do not continue
benchmarking in that same process. Shell backgrounding is application-owned;
use the repository's campaign launch scripts where supplied, and monitor logs.

## Copy, audit, analyze, deploy

JSONL journals are authoritative and flushed incrementally. CSV/Parquet are
rebuildable analysis exports, not the only copy to keep. Copy complete bundle
directories or ZIPs, retaining metadata and residual records.

```bash
rtx-autotune audit copied_bundle/ laptop.zip --output audit.json
rtx-autotune collect copied_bundle/ laptop.zip --output merged --format both
rtx-autotune summarize-tuners copied_bundle/ laptop.zip \
  --output autotune_reports/comparison --format both
rtx-autotune install-winners copied_bundle/ laptop.zip --dry-run
rtx-autotune install-winners copied_bundle/ laptop.zip
rtx-autotune list-winners --summary
```

Installation promotes verified compatible linear winners, not every raw minimum.
Use `verify-winners` / `tournament-winners` on the target GPU for additional
confirmation; inspect their `--help` for budgets and family/treatment filters.
Do not use `--force` or permit nonstationary results merely to make promotion
succeed. Schema/layout/numerical compatibility is distinct from performance.

## Learn from the corpus

`pretrain` fits revision-scoped cost/feasibility heads; `evaluate-pretrained`
checks a frozen artifact on held-out data. `study-sku` analyzes hardware and
configuration relationships; `study-evidence` analyzes paired changes and
held-out ranking. Keep shapes/devices/studies used for evaluation out of policy
selection/training. Adaptive-search associations are not causal evidence.
Deploy only artifacts that pass their recorded gates; configure
`RTX_AUTOTUNE_PRETRAINED_ARTIFACT` deliberately.

Hot replay and rotating-address regimes answer different locality questions.
Report end-to-end latency, compile time, failures, regret by trial/time budget,
raw timing distributions and confirmation uncertainty. Component winners are
not necessarily end-to-end winners. See [evidence methodology](autotune_evidence_methodology.md).

## Port to another kernel project

Use `ConditionalSearchSpace` for legal conditional parameters and
`FunctionKernelTask`/`StagedTaskAdapter` for compile, correctness and benchmark
stages. Strategies propose; the scheduler allocates; stores persist outcomes.
`AskTellSession` supports serializable leases and out-of-order worker replies;
it is a transport boundary, not a managed cluster service. The
[extension reference](../rtx/autotune/README.md) includes a full example and
durable local runner. New projects must provide their own independent numerical
oracle and workload-specific resource/measurement logic.
