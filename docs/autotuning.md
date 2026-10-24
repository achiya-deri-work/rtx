# Autotuning: runtime selection to portable datasets

## Local evidence consolidation

Use one **logical collection**, not one enormous mutable file. The CPU-only
storage command writes checksummed Parquet shards and a source/line occurrence
index. It deduplicates exact record bytes only: repeated measurements,
conflicting observation IDs, errors, malformed rows and raw metadata remain.
It never averages timings or marks observations safe for training.

```bash
python -m tools.research.storage build --output research_data/collection_v1 \
  --sources autotune_datasets copied_datasets autotune_reports autotune_results
python -m tools.research.storage verify --output research_data/collection_v1
```

Root ZIP bundles may also be passed as sources. Read `records/*.parquet` for
unique payloads and join `occurrences/*.parquet` on `record_hash` for source
locations. `manifest.json` defines source IDs and checksums. `payload` retains
the original JSONL record bytes. This is a lossless storage layer, not the
curated training view; preserve study split/quarantine sidecars as well.

Inactive original trees can be packed independently:

```bash
python -m tools.research.storage archive --sources copied_datasets \
  --output research_data/archives/copied_datasets.zip
```

The archive is fully read back and compared by SHA-256. Optional `--retire`
removes the checked originals only after verification; **do not use it on an
active campaign**. Restore by extracting the archive into the original source
parent. Resume old campaigns only after restoration. Keep receipts, original
ZIP bundles and metadata, and never retire model checkpoints as part of this
operation. The archive and collection are ignored by Git.

To retire a tree whose archive already exists, without recompressing it:

```bash
python -m tools.research.storage retire --sources copied_datasets \
  --output research_data/archives/copied_datasets.zip
```

This rechecks the archive and every original file before removing anything.
The separate `.retired.json` receipt records retirement; the archive's original
`.receipt.json` records the initial verified snapshot.

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

### Shared packed-GEMM outer-loop scheduling

`MXFP8GemmConfig.k_tile_unroll` and `NVFP4GemmConfig.k_tile_unroll` are
performance-only outer K-loop controls:
`1` (the backward-compatible default), `2`, `3`, `4`, `6`, `8`, or `12`.
All NVFP4 GEMM search spaces expose it as `gemm_k_unroll`, independently of
operand-buffer count (`stages`) and the inner MMA schedule. Larger factors
can reduce loop bookkeeping but increase compile time and code size; they
are choices for the autotuner, not universal recommendations. Quantization
policy, FP32 accumulation order, and the MXFP8 backward path do not change.
Older explicit configs omit the field and retain `1`; newly saved configs
record the chosen factor. Research benchmark winners are not installed into
the runtime cache automatically.

The MXFP8 prequantized-forward space exposes the same `gemm_k_unroll` axis.
Backward exposes independent `dx_gemm_k_unroll` and `dw_gemm_k_unroll` axes;
never assume that a forward winner is suitable for the long dW reduction.
`MXFP8GLUGemmConfig` inherits the control, including BF16 and packed MXFP8
GLU outputs and backward derivative recomputation. The MLP search forwards
the axis to gate projection (`cooperative_gemm`), down projection
(`down_gemm`), recomputation (`recompute_gemm`), and both backward groups.
Existing on-chip BF16-to-MXFP8 fused paths retain their distinct `k_unroll`
control. Neither control changes the activation, scale policy, or FP32
accumulation contract. Changing this scheduling axis alone is not permission
to choose a less accurate quantization recipe.

The host-launched MXFP8 preparation evaluator calibrates repetitions to a
2-ms timing window (bounded at 4096 calls) while retaining the requested
sample count. Records include requested and actual calls; calibration is not
part of the measured samples. This remains host timing, not implicit graph
replay. Revision 11 separates these measurements from old short-window records.
Backward uses the same batch calibration (revision 26); its explicit
`host` versus `cuda_graph` choice remains intact, and graph batch counts are
reported in complete backward calls rather than replay counts.
If a runtime MXFP8 forward/backward or NVFP4 forward search has no stationary
candidate but its initial schedule passed numerical checks, execution may use that seed with a
warning for the current process. No persistent winner is written. Explicit
offline tuning still fails without eligible timing evidence, and numerical or
compilation failures never authorize this fallback.
For MXFP8 tuning resumed from an exhausted, noisy history, a runtime caller
revalidates the initial schedule on the current operands before using it.
This is a separately recorded correctness-only session: it collects no timing
samples, adds no search trials, and does not create a persistent winner.

## Experimental MXFP8 preparation selection

For dynamic BF16-input MXFP8 linears, opt in per layer to measured fused versus
staged preparation selection:

```python
policy = rtx.AutotunePolicy(
    preparation_selection=True,
    max_trials=8,
    max_seconds=30,
    confirmation_repeats=1,
)
layer = rtx.MXFP8Linear(
    768, 3072, device="cuda", backend="auto", autotune="balanced",
    tuning_policy=policy, autotune_cache_dir=".rtx-cache",
)
compiled = torch.compile(layer, fullgraph=True, dynamic=False)
```

This is experimental and off by default. Forward needs `backend="auto"`
without an explicit forward kernel config. Explicit backward configs retain
their meaning; automatic backward selection compares staged, fused-dX,
fused-dW, and fully fused execution, with separate contexts for partial
gradients. The gradients still use original BF16 operands and MXFP8 GEMMs:
this flag does not enable FP4 backward or change quantization policy.

Each forward/backward context has its own budget; `forward_trials` and
`backward_trials` can override it. A two-trial budget cannot cover all four
paired-backward plans. The selector uses policy-matched seeds and compatible
nearby measured candidates, then measures on the target device; it does not
deploy the unqualified analytical/ML priors. `prior_artifact` and
`predictor_artifact` are rejected on this path. Its fixed, revisioned graph
measurement protocol includes bounded time-based warmup and stationarity
checks; `warmup`, `samples`, and `calls_per_sample` do not override that
experimental protocol.

Evidence lives under `<autotune_cache_dir>/preparation/`. Compatible exact
hits reuse measured schedules, including across processes; kernel, numerical,
and measurement revisions partition the evidence. `cache_behavior="refine"`
allows a larger cumulative trial budget. The legacy process-wide opt-in
`RTX_PREPARATION_SELECTION=1` remains supported and enables selection even
when the policy flag is false. Warm before manual CUDA graph capture.

Local tests cover eager/fullgraph execution, paired/partial gradients,
logical tails, changed-input graph replay, persistent cache hits and stable
live allocations. These are bounded qualification results, not universal
shape/SKU certification. See [evidence and limits](operand_preparation_plan.md).

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
# MLP deployment qualification

See [MLP deployment-mode qualification](mlp_deployment_qualification.md) for
execution-mode-aware first-hit selection, paired promotion, logging, extra
qualification costs, and current validation limits.
