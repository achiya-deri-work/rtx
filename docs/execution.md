# Compilation, CUDA graphs, and runtime caches

## Execution boundaries

Python frontends validate/rout operands, registered `torch.library` operators
provide fake implementations and custom autograd, and native CuTe launchers
execute CUDA kernels. Direct Inductor lowerings are version-pinned. A fullgraph
compile does not imply Inductor can fuse arithmetic inside an opaque CuTe GEMM.
Operations outside the custom boundary can be optimized by the surrounding
compiled graph where supported.

The production compiler contract is
`torch.compile(module, fullgraph=True, dynamic=False)`. A new static shape can
compile and select another launcher. General symbolic dynamic shapes, export
to unrelated runtimes, higher-order differentiation, and arbitrary tensor
subclasses are not guaranteed by success on this contract.

## Warmup and capture order

1. Install/load the model and choose its precision/scaling policy.
2. Convert/pack weights outside the compiled step if using PTQ.
3. Explicitly tune GLU/MLP if desired; linears may do balanced first-hit tuning.
4. Compile the surrounding forward/training workload.
5. Execute representative warmup shapes, including backward for training.
6. Only then capture a manual CUDA graph, or use the application's compiled
   graph facility. Do not perform searches inside an active capture.

Graph replay reduces repeated host submission overhead. It requires stable
captured addresses/control flow and application-owned input/output lifetimes;
it is not universally automatic. Changing a module config, packed weights,
shape policy, or runtime launcher after capture requires rebuilding the
corresponding graph. Two forward kernels remain two kernels under replay.

Do not time compilation, first-hit tuning, observer initialization, or capture
as steady-state kernel latency. Report them separately when startup matters.
Do not enable inference freezing for dynamic-weight training comparisons.

## Three different caches

| Cache | Stores | Lifetime |
|---|---|---|
| Runtime winner | Revision/device/shape/policy-scoped configuration | Persistent files |
| Compiled launcher | Native code / launcher references | Process and compiler caches |
| Runner workspace | Shape/stream-specific temporary buffers | Bounded process cache |

`autotune="cache"` avoids benchmarking but may still compile selected kernels.
Clearing runner caches is not the same as deleting datasets or installed
winners. `rtx.clear_runtime_caches()` synchronizes before releasing retained
runtime resources; use it between phases, not in a hot training loop. Do not
clear caches backing an active captured graph and expect that graph to remain
valid. Inspect its docstring for the returned release accounting.

Dynamic runners use bounded caches (normally eight entries per family).
`RTX_MXFP8_RUNNER_CACHE_ENTRIES` and family-specific bounds such as
`RTX_MXFP8_BACKWARD_CACHE_ENTRIES` adjust retention. Allocator-reserved memory
remaining high after a release is not itself a tensor leak; compare synchronized
live allocated bytes after equivalent warmed iterations.

## Streams and threads

Use normal PyTorch stream dependencies for inputs produced on another stream.
The application owns inter-stream ordering around calls and parameter updates.
Runtime runners key stream-specific state; MLP's optional backward overlap
joins its child work before returning. NVFP4 delayed history can reset on a
relevant shape/stream change; do not assume one uninterrupted history across
arbitrary routing. Native kernels do not implement distributed collectives.

## Introspection and deterministic operation

Linear modules expose `layer.explain(x)` without compiling or benchmarking.
`selection_source="deferred_runtime"` means selection occurs at execution.
GLU/MLP do not expose the same runtime-winner interface; inspect their explicit
`.config` and `.tuning_result` after tuning instead.

For reproducible comparisons, select explicit configs or cache-only mode and
record software revision, GPU, shape, cache regime, and reduction schedule.
FP32 accumulation does not guarantee bitwise identity across split/atomic
reduction schedules. Test determinism under the exact selected policy rather
than promising it for the entire autotuning search space.
