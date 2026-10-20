# Backward scratch-cache byte budgets

## Public startup policy

Configure both backward caches together, before model execution/capture:

```python
import rtx

rtx.configure_workspace(rtx.WorkspacePolicy(
    paired_bytes=256 * 1024**2,
    partial_bytes=128 * 1024**2,
    paired_entries=8,
    partial_entries=16,
))
before = rtx.workspace_report()  # take after warmup for a steady interval
# Run the application's interleaved training/checkpoint/teacher workload.
after = rtx.workspace_report(previous=before)
print(after)
```

These sizes are an **explicit example**, not a validated full-model budget.
Omit either entry argument (default `None`) to make that cache byte-only: an
adequate byte budget then has no hidden eight-entry ceiling. Paired runners
own both dX/dW legs; partial runners own one requested leg. A five-shape paired
working set can therefore need ten partial entries, with additional entries
for different streams/configurations. Neither limit replaces the other unless
the entry limit is explicitly disabled. Existing environment/default behavior
is unchanged until this API is called.

The policy overrides import-time environment limits for these two caches only.
It requires empty caches and must not run concurrently with execution; it never
silently clears live caches. For deliberate reconfiguration, first perform
explicit synchronized runtime cleanup. Retained graphs can still own storage.

Reports show resolved limits, interval hits/misses/insertions/evictions, which
entry/byte limit triggered evictions, and the last 16 evicted keys (including
matrix dimensions, gradient leg, device, stream and config identity). A bounded
recent list is not a complete shape histogram. `insertions` counts successful
runner constructions in these caches, unlike double-checked lookup misses.
After warmup, insertion deltas expose reconstruction cost in counts, not time.
Diagnostics recommend adjusting the binding limit; they do not automatically
change policy or infer a performance benefit. Both limits can bind one eviction.

`unique_cached_storage_bytes` deduplicates explicit tensor storages across both
cache owners. Per-cache byte limits remain conservative sums over entries.
This report excludes allocator reserve, construction peaks, outputs, opaque
closure ownership, other frontend caches and resources held only by live graphs.
It is neither a full-device limit nor an OOM guarantee. Reporting does not
synchronize the GPU, so take synchronized application measurements separately.

## Environment configuration (legacy)

The paired MXFP8 backward runner cache and the individual dX/dW runner cache
support optional byte caps in addition to their existing LRU entry limits:

```bash
export RTX_MXFP8_BACKWARD_CACHE_BYTES=268435456
export RTX_MXFP8_BACKWARD_MATMUL_CACHE_BYTES=134217728
```

Set these before importing the backend. `RTX_MXFP8_RUNNER_CACHE_BYTES` supplies
the fallback for both caches; it is a **per-cache**, not process-wide budget.
Unset preserves existing behavior. Zero prevents retention of nonempty scratch;
zero-byte executable wrappers can still fit. Entry-count limits still apply.

Stats from each cache include `resident_bytes`, `peak_resident_bytes`,
`max_bytes`, and `oversized`, in addition to hits/misses/evictions. Sizes count
explicit tensor storage in runner dataclasses, deduplicating aliased views
within one owner. Executables and arbitrary closures are not traversed.
The peak is retained storage after eviction, not the transient construction
peak. Oversized runners still execute for their caller but are not retained.

`rtx.runtime.runtime_cache_stats()` reads counters from loaded frontends without
loading kernels, synchronizing, or clearing caches. Check `bytes_accounted`:
other frontend caches without an explicit sizing callback report false, and
their zero byte counter must not be interpreted as zero GPU ownership.

## Scratch-only cleanup

`rtx.clear_runtime_caches(preserve_compiled=True)` releases cached scratch and
private replay graphs while preserving cheap selections, MLP configuration
parsing, direct-backward closures, and compiled MLP helpers. First-hit tuning
uses this mode after search. The default `preserve_compiled=False` retains the
previous full-cleanup behavior. Disk winners are untouched in either mode.

Cleanup is process-wide; the legacy `device` argument is **not** an eviction
filter. Safe cleanup now synchronizes all visible devices before dropping
global caches. Callers disabling synchronization must ensure completion on
every affected device themselves. The multi-device synchronization routing has
a mocked test; this machine does not qualify actual multi-GPU execution.

Caller-owned graph resources survive cache cleanup through
`retain_runtime_resources()`. Autograd may access runners on a worker thread,
so retention is not limited to the thread entering the scope. Worker-thread
accesses conservatively pin into every active scope, and scopes retain existing
scratch allocator pools. Concurrent scopes can consequently retain more than
their individual kernel working sets. This is explicit graph ownership, not
cache-owned bytes; release graphs and their owner mappings to release it.

This cross-thread requirement was caught by an actual illegal-access replay
regression: the former thread-local-only retention missed backward runners.
It is not evidence that the older AFRL capacity OOM had the same cause.

September 17 validation: the 36-test cleanup/byte-budget/first-hit/deployment
suite passes on both Torch 2.13 and 2.12.1. Reports are
`autotune_reports/scratch_cleanup_regression_torch{213,2121}_20260917.xml`.
The previously failing real-MLP case replays after cleanup with changed input
and gate/up weights, compares output and all gradients, and checks that
scratch-only cleanup preserves executable cache misses and selection identities.

## Boundaries

September 17 checkpoint/teacher regression: five compiled selected-direct MLP
shapes (M128/256/384/512/640, K128/H256/O128) are interleaved inside one
caller-compiled workload with no-grad teacher forwards and reentrant checkpoint
training. Inputs/weights change between warm passes. Eight partial entries cause
30 successful runner reconstructions, 30 entry-limit evictions and 60 reported
misses over three passes; sixteen entries cause zero reconstructions/evictions
and 30 hits. Both policies have flat post-warmup live allocation and pass an
independent MXFP8 output/all-gradient gate. Unique explicit cached storage rises
from 3,429,888 to 3,801,600 bytes (363 KiB); eliminating churn costs retention.

Both Torch versions pass. Evidence is in
`autotune_reports/workspace_checkpoint_teacher_torch213_v2_20260917.xml` and
`workspace_checkpoint_teacher_torch2121_20260917.xml`. The first 2.13 harness
failed Dynamo's eight-specialization limit when compiling layers separately;
its report is preserved. The passing harness compiles the enclosing workload,
without changing compiler limits. This is not an AFRL timing result. Absolute
process allocation across the two parametrized tests is not an isolated-process
memory A/B; only within-test stability and explicit cache storage are compared.

- CuTe executable compilation caches are independent and are not cleared by
  these LRU evictions. This does not yet constitute a complete separation of
  every frontend's executable metadata and scratch ownership.
- Values must have immutable storage ownership while cached; accounting occurs
  at insertion. Existing backward runner dataclasses satisfy this contract.
- Budgets do not include CUDA allocator reserve, private pool reserve, outputs,
  inputs, forward scratch, or resources pinned by live graphs/callers. Aliases
  shared across distinct entries may conservatively count more than once.
- `retain_runtime_resources()` pins graph owners beyond eviction. Dropping a
  cache entry must not invalidate graph addresses; graph-owned memory can
  therefore exceed the cache budget. It remains live until its owners release.
- Very small budgets can prevent the warm runner working set from fitting.
  Manual CUDA capture must still warm all needed runners and keep them resident
  until capture. Budgeting is not permission to initialize runners inside capture.
- This is a retained-scratch limit, not a capacity preflight or an OOM guarantee.
  Allocation still occurs before insertion. Near-capacity first-hit tuning and
  device-wide accounting remain separately required.

Coverage lives in `tests/test_runtime_byte_budget.py`: alias accounting, LRU
replacement/deletion/oversize behavior, retained-owner lifetime and CUDA graph
replay after cache eviction. MLP integration coverage remains separate.
## Optional serial backward operand reuse

`MXFP8BwdConfig.operand_storage` accepts `"separate"` (unchanged default),
`"serial_reuse"`, or `"serial_reuse_partials"`. Reuse is a memory choice, not a
different quantization policy. It is legal only for two decomposed matmuls,
`quant_schedule="per_matmul"`, `stream_schedule="single"`, and execution order
`"dx_first"` or `"dw_first"`. Quad quantization, interleaved quantization and
dual-stream execution are rejected, never silently changed.

Each quantized operand/scale role shares one allocation across the two GEMMs;
different roles never alias within one GEMM. Allocation uses the larger required
storage and creates each GEMM's layout views during cold runner construction.
CUDA stream ordering keeps the second quantizer after the first GEMM/reducer.
Both orientations are still independently quantized with their own block
scales: only storage is reused, never previously quantized/transposed values.
Runner/graph owners retain the views, and byte accounting counts their shared
storage once. There is no cross-runner, cross-shape or cross-stream arena.
FP32 split workspaces are unchanged.

The backward autotuner exposes `operand_storage`; the MLP tuner inherits it as
`gate_backward.operand_storage` and `down_backward.operand_storage`.
`serial_reuse` applies only to an already **paired** dX/dW runner. Joint MLP
recomputation launches down dX and dW independently, so that legacy mode is
inactive and deduplicated for the down projection.

`serial_reuse_partials` explicitly routes partial requests through a paired
runner, enabling reuse across those serial down dX/dW calls. Both legs are
constructed on the first request. **The allocation is charged to the paired
cache's byte/entry limits**, not the partial cache, and no secondary owner is
inserted into the partial cache. A zero/oversized paired budget may cause
reconstruction rather than retention, while the caller retains the current
runner until its launch completes. Graph resource pinning uses the same paired
runner ownership as ordinary full backward.

For a workload requesting only dX or only dW, this opt-in may allocate more
than the ordinary partial path. Keep `separate` or `serial_reuse` for that
allocation-minimal behavior. Requalify actual phase/gradient masks and budget
mix; there is no automatic conversion of existing plans to this mode. Its MLP
training identity is distinct even for joint recomputation, so the autotuner
can measure the storage/performance tradeoff instead of deduplicating it away.

Existing separate-storage serialized identities are preserved. Old winner files
are not modified, and no default or automatic cache policy is changed.

At M20480/K256/H768/O256 on the local 5070 Ti, separate-process measurements of
the same selected-direct MLP configuration reduced cold peak, warm peak and
warm residency by 31.32 MiB each. Warm peak was 337.64 -> 306.32 MiB. Timing
was essentially unchanged and did not pass the speed-promotion rule. This is an
optional capacity tradeoff, not a measured full-application memory saving.
Those absolute warm peaks included reference-only cuBLAS workspace; subsequent
benchmark cleanup before target rewarm measures 242.32 MiB for the reuse
candidate versus 213.50 MiB BF16. The reuse delta reproduces on both supported
Torch versions. Do not run this process-global library cleanup in applications.
See [replacement-level qualification](mlp_replacement_comparison.md) for the
measurement contract. The referenced local research artifacts are not
distributed with the library.
