# NVFP4 training backward optimization — 5060 qualification

**Update after the reboot:** the measured fix is producer parallelism, not
removing Python/Inductor gaps. A 256-CTA producer cap reduced full-model training
time by **10.1%** in reversed-order fresh-process runs, preserving the existing
MS-EDEN arithmetic. It reduced the penalty versus NVFP4-forward/MXFP8-backward
from **32.0% to 18.7%**. This does not establish parity with MXFP8 backward.

## Actual training profile

The new persistent snapshot is `/home/achiya/rtx-bwd-fix-20261025.L8bAWr` on
`personal_laptop`. The reboot erased the earlier `/tmp` snapshot. The completed
100M training JSONL files had already been copied locally; the new measurements,
checkpoints and profiler files use the persistent snapshot.

One warmed full-model update was captured with Nsight Systems 2025.6.3 on the
RTX 5060 Laptop (26 SM, Torch 2.13.0+cu132):

- Three `cudaGraphLaunch` calls plus eleven ordinary kernel launches; this is
  not a Python launch per backward-preparation kernel.
- Hadamard producers: 8.975 ms, **37.5%** of total GPU kernel time.
- FP4 backward GEMMs: 2.347 ms, **9.8%**.
- Scale finalizers: 0.802 ms, **3.3%**.
- Total recorded kernel time: 23.953 ms. These are **profiler diagnostics**, not
  the ordinary timing results below. The explicit end-of-capture synchronize
  is instrumentation, not evidence of a production synchronization bottleneck.
- Nsight Compute/Python were unavailable. No measured occupancy, register-spill
  or DRAM-byte claim is made.

## Schedule comparison

The old training recipe selected the two-wave resource seed: 128 threads,
64 producer CTAs, `coalesced_wide`. The confirmed schedule keeps the threads,
load geometry, arithmetic and GEMM, increasing producer CTAs to 256. Finalizers
remain 128 threads / 224 CTAs. dX uses row-major traversal, dW reduction-major.

Complete conditional gradients, CUDA graph replay, fixed matched randomness:

| Projection `(M,N,K)` | dX old/new, us | dW old/new, us |
|---|---:|---:|
| 4096,1536,512 | 179.19 / 137.57 | 222.54 / 171.75 |
| 4096,512,512 | 69.21 / 59.42 | 111.12 / 91.67 |
| 4096,2048,512 | 241.09 / 182.05 | 290.20 / 221.56 |
| 4096,512,1024 | 92.91 / 80.89 | 174.55 / 140.30 |

Both execution orders and changed inputs/signs were tested. Packed bytes,
scales, FP32 factors and complete gradient outputs matched **bit-exactly**.
These are single-session schedule results, not cross-device guarantees.

Further thread/grid probes on the largest projection did not give a compelling
additional improvement over 128 threads / 256 CTAs. Wider x4/x8 tiles helped the
old under-parallelized schedule modestly but did not beat the selected schedule
consistently after increasing its grid. They remain optional autotune choices.

## Full-model confirmation

21.1M decoder, batch16, sequence256, 512 updates per run, fullgraph/static
compilation and CUDA graphs, unchanged precision policy. Fresh-process order:
old, new, hybrid, hybrid, new, old. Initialization/data/model/optimizer identities
were checked across all six runs. Training time excludes compile/validation.

| Policy | Run 1 / run 2 training, seconds | Mean |
|---|---:|---:|
| Full NVFP4, old schedule | 13.0284 / 12.9425 | 12.9854 |
| Full NVFP4, 256-CTA schedule | 11.7107 / 11.6386 | 11.6747 |
| NVFP4 forward / MXFP8 backward | 9.8287 / 9.8394 | 9.8341 |

All completed with finite losses, gradients and final parameters. Full-model
training is **not bit-exact** across fresh processes: old-schedule final losses
were 0.99537/0.98324, new 0.99594/0.99834, hybrid 0.99217/0.99363. These short,
two-repeat stochastic runs do not establish a convergence ranking or prove
equal long-run quality. The unchanged-arithmetic claim is supported by the
controlled bit-exact kernel comparisons, not by equal training trajectories.

76 focused CPU tests and 64 GPU producer-geometry tests passed. The latter
cover original BF16 and packed sources, four thread counts, both traversal
orders and all four cooperative load modes.

## Integration and reproduction

- The research autotuner now explores higher-grid seeds early, including wide
  x4/x8 alternatives. All remain optional; no universal SKU policy is inferred.
- Existing public/experimental default schedules are not silently replaced.
  The confirmed training setting is explicitly selected with
  `--ms-eden-producer-blocks 256`; loader and grid overrides are recorded in the
  run identity, and incompatible checkpoint resumes are rejected.
- `tools/run_ms_eden_training_ab.sh PYTHON TOKENS OUTPUT_DIR` reproduces the
  reversed-order full-model experiment.
- `benchmarks/tune_ms_eden_training_shapes.py` has `loads`, `grids`, `resources`
  and `confirmation` portfolios for controlled paired measurements.
- Local artifacts: `autotune_reports/nvfp4_backward_fix_20261025/`, including
  the raw `.nsys-rep`, profiler CSV/environment, eight-shape/phase measurements,
  geometry-test log, resource probes, and `training-ab/*.jsonl`.

Remaining: reduce preparation math/traffic further and tune the backward GEMMs.
The current evidence does **not** justify replacing MXFP8 backward as default.

Only the 5060 was used. The 5070 remained off-limits; the local 5090s' AFRL
training was not disturbed. The initial connectivity interruption was resolved
by reboot, after which all profile/qualification work above was rerun.
