# CuTe routing and shared autotuning — 2026-11-15

Production routing now uses CuTe DSL for Sinkhorn, radix order statistics,
direct/packed quota selection and conditional predicates. Host graph control
uses `cuda-python`; **no production CUDA C++ build, NVCC or custom CUB backend**
is required. The earlier CUB code is isolated in `benchmarks/routing_reference`
and excluded from the wheel. See the [API guide](balanced_router.md).

QB margin/proposal arithmetic, final/proposal top-k indexing and metadata remain
compiler-visible Torch/Inductor operations. This is not a claim that every
device instruction or every routing suboperation is handwritten CuTe.

## Qualified performance

RTX 5070 Laptop, Torch 2.13.0+cu132, CuTe 4.7.0, CUDA Python 13.3.1. Same inputs,
parameter values and 5/5/64 policy budgets; all external tensors/parameters BF16,
routing arithmetic FP32. Standard `torch.compile(fullgraph=True, dynamic=False)`
is stitched into a CUDA graph for steady-state timing; compilation/search are
excluded. These are **router-only** measurements, not expert/whole-MoE speedups.

The shared tuner selected schedules with 16 trials, including local, structural,
random and cost-guided proposals. Searches took approximately 22 and 27 seconds.
No per-shape winner was manually chosen or hardcoded into dispatch.

Final alternating A/B: four rounds, 21 samples per measurement, calibrated
750-ms requested graph warmup. The unchanged stationarity gates apply; the
paired lower-quartile and aggregate-median gain must both exceed 1%.

| Pool / model / experts / K | Portable seed F+B | Tuned F+B | Speedup | Promotion |
|---|---:|---:|---:|---|
| 1024 / 384 / 128 / 8 | 1.3475 ms | 1.0208 ms | **1.32×** | Passed |
| 4096 / 384 / 128 / 8 | 4.2488 ms | 2.9246 ms | **1.45×** | Passed |

The 1024-token forward-only comparison also passed: 1.2039 → 0.8663 ms,
about **1.39×**. The high-resolution 4096-token forward-only comparison retained
the incumbent because one challenger window was unstable. An earlier four-round
comparison passed at 3.4176 → 2.1167 ms, but did not repeat a clean promotion in
the higher-resolution run. Both runs are retained; do not describe every shape
and phase as unconditionally qualified.

These results compare against the current portable seed, already using CuTe
quota kernels. They are not comparisons against BF16 dense linear/MLP layers.
The separate [fixed-64 compiled Torch comparison](balanced_router_vs_torch_20261115.md)
measures the original BiMoE baseline on this same laptop, including both
forward-only and forward/backward timings and their stability limitations.

## What changed

- CuTe quantiles use warp-aggregated radix histograms and return only the
  required FP32 order statistic. Short vectors keep keys in registers; long
  vectors stream them. No full top-(capacity+1) result matrix is emitted.
- Packed quota selection counts/compacts active proposals per expert, selects
  a unique `(priority, original position)` threshold and marks original slots.
  Small bounded proposal sets can instead use CuTe direct ranking. Source
  acceptance still precedes target acceptance, including exact ties.
- A CuTe predicate calls CUDA's conditional-handle device builtin from graph
  nodes. CUDA Python constructs the conditional loop, preserves captured
  driver kernel handles/attributes and advances outer-capture dependencies.
- Private fixed routing bodies lower from static FX directly to Inductor.
  Preparing shape/stream plans no longer consumes the application's Dynamo
  per-code-object recompile budget. No global recompile limit is increased.
- Shared `rtx.autotune` UCB orchestration, residual JSONL journals and schema-v2
  runtime winners own search/resumption. Forward/backward geometry is independent;
  QB and quota execution families are selectable. Inactive choices are
  canonicalized rather than measured repeatedly.
- Autograd retains immutable CPU execution metadata and saved normalization
  probabilities. Backward never retunes, reroutes, recomputes Sinkhorn or reads
  a CUDA scalar to determine execution configuration.

## Correctness and integration

The focused matrix passed **49/49** on the 5070/Torch 2.13.0 and on the
5060/Torch 2.12.1, both CUDA 13.2 stacks. Earlier 5060/Torch 2.13.0 runs passed
47 cases before adding two long streaming-order-statistic cases.

Coverage includes exact order statistics with ties, exact frozen hard-policy
parity, ragged/dummy metadata, independent FP64 Sinkhorn/gradient gates, real
`opcheck`, cold first-hit fullgraph compilation, disk winner reuse, changed-input
graph replay, independent streams and repeated live-memory stability. The long
streaming cases include 8193×7 and 40000×2 matrices.

The first-hit test forbids subprocess NVCC builds and supplies a nonexistent
toolkit path. CUDA execution metadata is constrained to CPU int64 tensors;
putting it on CUDA is rejected instead of introducing hidden readback.
Normal Torch/Inductor tooling remains necessary.
The final startup/autotune subset also passed all six tests after adding these
guards. Its no-build check permits Inductor's diagnostic `nvcc --version`
query, but rejects NVCC compilation. Local CPU/resource/documentation checks
passed 71 tests and 299 subtests, with 39 CUDA-only tests skipped locally.

A clean wheel build includes the Python/CuTe routing implementation and excludes
the old native loader, `.cu` production source and benchmark-only CUB reference.
Routing's optional installation extra selects the tested CUDA Python APIs.

## Profiling and operational limits

A node-level Nsight Systems trace of three 1024-token hard-route graph replays
attributes approximately 46.7% of GPU kernel duration to remaining Torch top-k
and bitonic kernels, 20.1% to CuTe quota kernels and 21.5% to CuTe quantiles.
Those are kernel-duration shares, not whole-layer utilization or bandwidth.
This identifies proposal/final-index selection as useful remaining kernel work.
Nsight Compute/Python were not available; no DRAM/L2 traffic, occupancy or spill
counter claim is made. Ordinary CUDA-event timings, not profiler timings,
establish the speedups above.

An eight-trial standalone CLI smoke run completed and wrote strict result JSON,
residuals and a runtime winner. A separate two-trial run had no eligible timing
winner: numerical evidence was retained and no latency winner was promoted.
The CLI reports this as status/exit code 2. Runtime first-hit tuning can retain
its independently correctness-checked seed process-locally; explicit offline
tuning still fails rather than turning unstable timings into deployment data.

Cold tuning needs temporary calibration/reference/capture memory. Prepopulate
the cache and use `autotune="cache"` when full-model allocation leaves little
headroom. Symbolic public deployment/export, distributed execution, concurrent
replay of one captured graph on different streams and whole-model convergence
remain unqualified. Expert FFNs are not implemented by this work.

## Evidence and reproduction

The 5070 studies, both paired runs, JUnit evidence, the `.nsys-rep`, SQLite and
CSV exports are preserved in the existing storage4-backed project location:

`autotune_reports/balanced_router_cute_20261115/5070`.

The completed 5060 test/CLI logs were inspected remotely, but its SSH connection
timed out during evidence retrieval. Those files remain in the 5060 snapshot;
they have not yet been copied into the local artifact folder.

The 5070 `balanced.json`, `paired.json` and `paired-highres.json` record configs,
timings and failed eligibility. `cute-hard-node-stats_cuda_gpu_kern_sum.csv`
contains the attribution above. Logs and journals also remain in isolated
snapshots `/tmp/rtx-routing-20261115.cowvat` (5070) and
`/tmp/rtx-routing-20261115.MRErRT` (5060), not required library paths.

```bash
python -m pytest -q tests/test_routing.py tests/test_routing_autotune.py
python -m rtx.autotune.routing --tokens 4096 --d-model 384 \
  --experts 128 --top-k 8 --effort balanced --output routing-data
PYTHONPATH=. python benchmarks/benchmark_balanced_router.py \
  --shapes 1024:384:128:8 4096:384:128:8 --tune balanced --output study.json
PYTHONPATH=. python benchmarks/qualify_balanced_router.py \
  --study study.json --output paired.json --samples 21 --warmup-ms 750
```

CUDA's conditional-graph API constraints are documented in the
[CUDA graph guide](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/04-special-topics/cuda-graphs.html).
