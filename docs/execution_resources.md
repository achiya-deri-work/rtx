# Execution resources and scratch lifetime

Kernel plans, mutable scratch, and captured CUDA addresses have different
lifetimes. A bounded runner LRU alone is not a graph lifetime contract or a
process-wide VRAM cap. These APIs preserve numerical policies and work with
MXFP8/NVFP4 linears, packed-weight launchers, GLU and recomputed MLP caches.

## Ordinary usage: library-owned resources

```python
compiled = torch.compile(model, fullgraph=True, dynamic=False)
y = compiled(x)
```

This is the primary contract: default mode, without `reduce-overhead`, forced
CUDA graphs, explicit resource scopes, stream management or scratch allocation.
Each compiled RTX callable owns its accessed runner working set independently
of the global bounded LRU. Ownership ends with that callable or an explicit
between-phase cache clear. First-hit autotuning does not attach trial scratch
to the enclosing model owner. Compatible FP32 write/reduce chains automatically
reuse size-classed storage per device, stream and execution lane.

CUDA graphs remain an application choice. The version-pinned Inductor
integration observes compiled-callable creation and cold Graph Trees warmup/
recording; it does not create extra graphs or wrap replay. Completed captures
retire unpinned duplicate preparation contexts automatically. No user-side
retirement boundary or cache-entry override is required for normal compilation.

## Automatic graph ownership

The first CUDA runner warmup installs a small integration around the public
`torch.cuda.CUDAGraph.capture_begin`, `capture_end` and `reset` methods. Public
manual capture and Inductor Graph Trees therefore retain the RTX runners they
actually access. Autograd-worker accesses are conservatively retained by active
capture scopes as well. Nothing is wrapped in the steady-state replay path.

Graph reset/destruction keeps resources alive until outstanding GPU work is
complete, then releases its pins. Private scratch tensors own their allocator
pool, including after runner eviction. Pools are routed per device/stream with
weak routing references; unused stream pools do not accumulate indefinitely.
Graph scopes no longer pin every preparation pool on the device.

Warm the required kernels before capture. This integration does not own the
application's inputs, outputs, weights or inter-stream dependencies. Raw C++/
external CUDA graph APIs bypassing the public PyTorch capture methods require
`rtx.runtime.retain_runtime_resources()` and explicit ownership of its returned
dictionary for the whole replay lifetime.

## Optional advanced controls

The following is a diagnostic/advanced override, **not** a required setup step.
Normal compiled working-set retention and scratch reuse above are automatic.

```python
resources = rtx.ExecutionResources(split_workspace_bytes=16 << 20)
preparation_stream = torch.cuda.current_stream()

with resources.activate():
    # Application-owned startup: warm all real forward/backward shapes, then
    # warm/capture the application's torch.compile CUDA graphs as usual.
    prepare_and_capture_model()
    torch.cuda.synchronize()
    released = resources.retire(streams=[preparation_stream], device="cuda:0")

report = rtx.runtime_memory_report()
```

`prepare_and_capture_model` above denotes the application's normal warmup and
capture, not an RTX function. Avoid retiring a stream still used for direct
eager execution unless rebuilding its caches is intended. The CUDA graph
capture stream is ordinarily different from the preparation stream.

`activate()` retains accessed runner contexts independently of the global LRU;
this lets a finite model working set survive small entry limits until capture.
It is an explicit owner, not a bounded cache: scope its lifetime to the intended
working set. Worker factories inherit an owner only when unambiguous; multiple
concurrent owners conservatively fall back to independent scratch. Captured
graphs retain their own resources after this owner is dropped.

`retire()` removes that owner's selected stream contexts, matching entries in
all registered caches (including nested compiled launchers), arena routing and
preparation-pool routing. Compiled code/selections and unrelated streams are
not globally cleared. The standalone `rtx.retire_runtime_contexts(streams=...,
device=...)` is available without an explicit execution owner.

Retirement requires a boundary with **no concurrent host submissions**.
Synchronization completes queued GPU work but is not a lock on another host
thread. Live graphs retain their pins even when their cache entries are retired.
`clear_runtime_caches()` remains a process-wide operation, not this selective API.

## Optional split-workspace arena

An explicit `ExecutionResources` owner defaults to zero fixed arena capacity;
ordinary execution instead uses automatic size classes. A positive fixed
`split_workspace_bytes` supplies FP32 views to compatible fused/decomposed split
and atomic backward runners. Requests larger than capacity fall back to private
storage. This is per parent stream and execution lane, not total process memory.

Serial write/reduce chains reuse an ordered lane. Dual-stream paired backward
uses separate dX/dW lanes, joined by the existing caller-stream dependency.
A host-side submission guard spans each complete workspace write/reduce chain,
not individual kernels. It does not insert CUDA synchronization into the hot
path. Configure/activate before creating the runners whose workspaces should
share; existing runners do not change storage after capture.

Quantized operands are **not** blindly aliased across different runners:
recomputed MLPs can prepare gate-dX and down-dW before consuming either. Existing
`operand_storage="serial_reuse"` / `"serial_reuse_partials"` remain explicit,
legality-checked search choices for their respective call topologies. Shared
source tensors also do not imply interchangeable dX/dW quantized values, because
their block-scale reduction axes differ.

## Memory accounting

`runtime_memory_report()` counts unique explicit tensor storages across every
registered top-level/nested cache, execution owners, private replay resources,
and live graph pins. Aliased views count once globally. Context rows include
stream identity and explicit role/address/size inventories. Optional
`include_allocator=True` includes allocator segment snapshots; call at a
quiescent diagnostic boundary, not in a hot step.

Do not add cached, execution-owned and graph-pinned byte totals: they overlap.
Retired bytes are ownership removed, not a guaranteed device-memory reduction.
Measure synchronized allocator snapshots and `torch.cuda.mem_get_info()` after
cleanup. Graph-pool reserve can be inactive between replays while its addresses
are still essential. `WorkspacePolicy` continues to control backward cache entry/
byte limits, not external pins or allocator reserve.

## Backward search coverage

Linear and MLP searches include complete, legal fused/decomposed transitions,
shape-derived split chunks and narrow-feature quantizer tiles. Opening trials
try split dW and combined materialized dX/dW before incidental local moves;
they are candidates, not universal winners. Inactive fused quantizer/GEMM fields
no longer spend duplicate trials. Backward tuning evidence reports observed
backend, reduction, tile geometry, split count and geometric CTA count.

For a ragged token reduction, row-major, register-loaded 32-value transpose
quantization zero-fills only the final quantized block. BF16 operands retain
their logical shape, with no padded BF16 copy or physical transpose. Ragged
dual/quad quantization remains rejected; separate preparation is the supported
materialized path. Search/kernel revision changes invalidate incompatible
coverage; historical evidence remains on disk.
