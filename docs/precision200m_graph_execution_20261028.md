# Why isolated NVFP4 backward speed did not transfer to the decoder

## Conclusion

The demonstrated penalty is **captured-graph execution order and resource
interference around TMA preparation**. It is not an FP4 GEMM arithmetic
slowdown, an eager Python gap, or a hidden serial fallback. The current
isolated replay objective does not reproduce this full-model scheduling
context. The exact internal CUDA scheduling algorithm is not observable here;
no specific hardware queue bug or hardware-counter bottleneck is claimed.

Production defaults, kernel arithmetic and runtime policies were not changed.
The event boundaries and reordered launches are diagnostic variants only.

## Same-board controls

Home GPU 0, batch 72, 205.85M decoder, exact previously selected configs,
fresh processes, 64 updates; first eight-update block discarded:

| Graph representation | Mean training step |
|---|---:|
| Ordinary paired graph, first control | 230.15 ms |
| Two untimed external event nodes per linear backward | 217.33 ms |
| Ordinary paired graph, repeat | 230.51 ms |
| Untimed-event replication without graph export | 217.84 ms |

Thus the effect repeats in A/B/A controls and is not explained by clock drift,
the graph-export implementation, or timestamp collection. Both modes have the
same peak allocated memory (26,858,755,584 bytes). These short runs establish
finite training/performance behavior, not model convergence or bitwise-equal
stochastic training trajectories.

Stream audit: all 64 ordinary captured paired calls use a side stream; none
falls back to serial. The per-device caller-stream cache has exactly one entry.

## Topology and execution evidence

Exported backward graphs:

- Ordinary: 1,178 kernels + 97 memset nodes.
- Untimed-event variant: the same kernels/memsets + 128 event-record nodes.
- Kernel multiset is identical. Compressing the event nodes yields exactly
  the same 1,401 kernel/memset dependency edges.

Only graph representation changed; neither the arithmetic work nor its
required kernel partial order changed. CUDA is free to choose different
execution orders among ready nodes and different internal replay streams.

A task-local **Nsight Systems 2026.5.1** succeeded where the installed 2025.6.3
capture had stalled. It was extracted under `/tmp/rtx-nsys-20261028.rHakZr`;
no installed tools, drivers or system configuration were modified.

One warmed step captured for each variant, both on physical GPU 0:

| Backward diagnostic | Ordinary graph | Pre-event graph |
|---|---:|---:|
| GPU backward graph span | 176.63 ms | 166.21 ms |
| Cumulative TMA preparation kernel duration | 56.34 ms | 36.86 ms |
| QKV TMA preparation mean, 16 instances | 1.321 ms | 0.420 ms |
| Cumulative FP4 GEMM duration | 42.57 ms | 43.67 ms |

**Cumulative durations overlap across streams: do not add them as wall time.**
Profiler timings are diagnostic, not the unobstructed training speed claim.
The large change is in TMA preparation; GEMMs did not get faster.

The traces show the competing row-preparation kernel getting a head start in
the slow graph. In the fast representation, TMA starts first. On the initial
cross-board trace pair, the QKV TMA instance took 1.343 ms versus 0.347 ms;
same-board captures above confirm the large effect is not a board difference.
The same-board QKV TMA kernel uses 72 registers/thread, 8,200 shared bytes,
128 threads and 256 CTAs; the competing row producer uses 40 registers/thread,
no shared memory, 128 threads and 256 CTAs. These resource counts alone do not
establish spilling or occupancy as the cause.

The slow trace used three internal replay stream IDs for backward, while the
initial fast trace used two, despite both user paths owning only the caller
and one side stream. This is execution evidence, not a changed semantic DAG.

## Negative controls

- Exact gate/up isolated backward remained faster with NVFP4 even after a
  preceding 512 MiB device fill excluded from timing. Simple hot-cache reuse
  did not explain the lost advantage.
- On GPU 1, default / 32 / 128 requested CUDA connections produced roughly
  220.93 / 220.76 / 221.21 ms. No useful connection-count fix was demonstrated.
  NVIDIA documents hardware-channel false dependencies, but these experiments
  do not establish that increasing the limit solves this workload.
- A pre-event boundary on GPU 1 gave 210.52 ms; a post-event boundary gave
  235.65 ms. Boundary location matters.
- A tiny ordinary pre-kernel gave 221.23 ms on GPU 1: merely adding launch
  delay does not reproduce the event effect.
- Reversing host submission order regressed to 245.35 ms on GPU 0.
- Prefilling dW's first preparation before forking regressed to 230.64 ms on
  GPU 1. Serial execution was also slower. Blanket serialization/reordering
  is not a solution across the four projection families.

## Why our autotuning missed it

The search measured one projection's complete compiled backward repeatedly,
including preparation and RNG. It did **not** measure the 16-layer model's
combined captured execution graph. Selection then minimized a sum of the four
isolated costs, constrained to one common implementation family. The full-model
pilots checked the chosen composition but did not feed a model-level objective
back into schedule selection.

Consequently a valid isolated TMA/paired schedule can win selection and still
encounter unfavorable co-scheduling in the actual application. Increasing
trials on that same isolated objective is not sufficient to fix this mismatch.

## Implications

The next optimization should expose capture boundary, preparation ownership
and launch-order choices as schedule axes and score them on a representative
compiled projection chain/block, with whole-model confirmation. Keep isolated
timing as a cheap first-stage filter rather than the final selection objective.
No universal event insertion, environment override or register cap is promoted
by this diagnosis.

The paired custom op remains opaque to Inductor; external RNG and activation
derivatives cannot fuse into its CuTe preparation kernels. That is a real
optimization boundary, but the graph controls above localize this particular
regression without changing that boundary. Native ATen reductions also appear
as captured GPU nodes; their presence is not evidence of eager Python gaps.

## Artifacts and reproduction

Root: `autotune_reports/precision200m_graph_20261028/` (ignored, local).

- `summary.json`: ordinary controls, trace groups, topology equality.
- `normal-topology/graph-003.{dot,json}` and
  `untimed-topology/graph-003.{dot,json}`: full backward DAGs.
- `nsys-normal.nsys-rep` and `nsys-before-gpu0.nsys-rep`: same-board traces,
  with SQLite exports and logged commands.
- Each variant retains its full command-derived training JSONL, finite
  terminal record, selected map where changed, and stream audit.

Driver: `benchmarks/investigate_precision200m_graph.py`.
Analyzer: `benchmarks/analyze_precision200m_graph.py`.
Snapshots and CLI paths are diagnostic artifacts, not installation requirements.

Sources for execution semantics:
[CUDA graph best practices](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html),
[CUDA execution environment controls](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).
