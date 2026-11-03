# Tuned full-model backward discrepancy: diagnostic findings

**Follow-up:** the current-tool traces and A/B/A controls now localize the
penalty to full-graph TMA preparation scheduling. See
[the graph-execution diagnosis](precision200m_graph_execution_20261028.md).
The profiler limitation described below was resolved with a task-local CLI;
the original failed captures remain preserved.

Scope: existing batch-72, 205.85M SiTU model and exact selected schedules.
No production kernel, policy, numerical tolerance or training default changed.
Artifacts: `autotune_reports/precision200m_gap_20261028/`.

## Reproduction

Four fresh 64-update pilots on GPU 0 in hybrid/full/full/hybrid order gave
post-first-block mean step times 215.706, 231.165, 231.476, 217.031 ms.
Thus the original 0.9% cross-board training difference is not a robust estimate
of the isolated backward penalty. These are short pilots, not convergence runs.

Uninjected CUDA-event attribution over the last 48 updates:

| Mean ms | NVFP4 forward / MXFP8 backward | Full NVFP4 |
|---|---:|---:|
| Forward | 50.294 | 50.117 |
| Backward | 163.649 | 178.492 |
| Clip | 0.974 | 0.973 |
| AdamW | 1.920 | 1.920 |

The regression is in backward, not forward, clipping or optimizer work.
Requested TMA implementations and explicit schedule maps are recorded in the
original training artifacts. Dispatch records alone do not prove timing or
overlap behavior.

## Graph perturbation control

Adding external CUDA timing-event nodes around each paired linear backward
changed full-model backward mean to **163.633 ms**. Instrumenting internal
preparation/finalization stages similarly yielded 164.134 ms. Removing the
in-graph events and repeating the ordinary phase measurement restored
**178.405 ms**. Forward/clip/optimizer remained essentially unchanged.

This is strong evidence of sensitivity to captured-graph execution structure,
not permission to report instrumented subpart timings as an unbiased profile
or promote event insertion as a production optimization. The measured 64
paired-linear spans sum to 88.006 ms in the perturbed graph. Concurrent dX/dW
subpart durations must not be summed as end-to-end latency. The precise graph
scheduling/dependency mechanism remains unresolved.

## Cache control

On GPU 1, exact tuned gate/up projection backward `(36864,5632,1024)` retained
its advantage with or without a preceding 512 MiB device fill, excluded from
timing. Forty randomized rounds produced medians:

| Replay condition | MXFP8 ms | NVFP4 ms |
|---|---:|---:|
| Immediately repeated | 2.600 | 2.383 |
| Cache-disrupted | 2.640 | 2.345 |

This does not reproduce all full-model memory interactions, but simple
hot-cache residency alone does not explain the lost advantage on this shape.

## Compiler-boundary audit

`_ExperimentalLinear.backward` puts `dy.contiguous()`, random signs and uniform
generation in the traceable autograd formula. They are outside the opaque
paired-gradient operator. The inspected original-BF16 launch path creates
workspace tensors/views and launches CuTe kernels; it does not hide a chain
of eager Torch reductions, matmuls or scale arithmetic inside registration.

The paired gradient **is** an opaque custom-op boundary: Inductor cannot fuse
the outside uniform generation or surrounding activation derivatives into its
CuTe preparation kernels. That is a real optimization boundary, but it is not
evidence that eager Python is executing between its kernels on every captured
training step. Fullgraph compilation and CUDA replay are enabled. Nor is this
boundary yet proven to cause the observed regression.

## Profiler limitations and next diagnostic

Nsight Systems 2025.6.3 on the current driver/runtime stalled at runtime capture
start; a second launch-time capture completed training but stalled at profiler
teardown. Only the task-owned profiler services were stopped. Logs were kept;
no usable kernel timeline/counter capture is claimed. Ordinary timing and
CUDA-event probes completed normally.

Next: inspect/export paired graph dependencies and compare a minimal,
correctness-preserving graph scheduling change under an A/B/A protocol, before
changing kernels or adding a production workaround. Also audit paired stream
cache fallback counts rather than inferring overlap from the requested policy.
