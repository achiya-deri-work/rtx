# Event-free NVFP4 backward scheduling fix

## Result and scope

The captured-model TMA scheduling regression has a real, event-free fix:
`backward_pipeline="paired_dw_first"`. dW owns the caller stream and its launch
chain is submitted first; dX runs on the joined side stream. Both choices are
necessary. Merely reversing submission order or caller ownership alone did
not improve the decoder.

No quantization, FP32 scaling/correction, accumulator policy, input source or
RNG draw order changes. No profiling-event, no-op kernel or eager tensor
arithmetic is added to the launch path. Side-owned dX storage is explicitly
recorded, and joins remain exception-safe. Existing serial/paired paths and
stable `NVFP4Linear` MXFP8-backward defaults remain unchanged.

## Correct selection objective

The new per-shape mode is an autotuner coordinate with early protected seeds,
not a universal hardware heuristic. Pipeline/design context revisions are
4/7. `make_ms_eden_model_adapter` refines qualified kernel designs using the
complete compiled application cost; it composes with existing orchestrators,
UCB scheduling, search strategies and residual storage.

The batch-72 decoder search used eight fresh full-model candidates with
matching initialization, data, numerical recipe and kernel designs. It chose
`paired_dw_first` for all four projection shapes. Search score was 215.29 ms
per step versus 228.88 ms initial, with a separate 64-update confirmation at
216.72 ms. These short-search scores are not the long-run speed claim.
No winner is silently installed into the public runtime cache.

`benchmarks.study_precision200m_tuned` now performs model-context refinement
before final training. `--model-trials` and `--model-seconds` control this
additional effort. Search/qualification time remains outside training clocks.

## Correctness and integration

- 181 focused CPU tests passed (18 GPU-only skips in that run).
- The current paired-pipeline test module passed all 50 cases on GPU 1,
  including 18 real-CUDA integration cases: fixed-draw exactness, seeded and
  explicit RNG, opcheck, fullgraph compilation, graph replay, checkpointing,
  allocation growth and independent caller streams.
- All four actual `(36864,N,K)` decoder projection shapes matched serial
  gradients byte-for-byte with fixed signs/uniforms. Changed inputs, weights
  and uniforms replayed correctly; zero dY gave zero gradients; repeated replay
  had zero live-allocation growth.
- Independent FP32 GEMMs gave gradient relative-L2 errors around 13.7%, the
  existing MS-EDEN FP4 error. Tolerances were not loosened. This scheduling
  change does not make FP4 gradients as accurate as MXFP8.

## Trace confirmation without instrumentation

A new Nsight Systems trace of the integrated choice on GPU 0 contains the
same 1,178 backward kernels and two internal replay streams. The helper audit
recorded 64 captured pairs, no stream-cache fallback and **zero external
event nodes**.

| Diagnostic, one profiled step | Old paired graph | Integrated fix |
|---|---:|---:|
| Backward GPU span | 176.63 ms | 161.23 ms |
| QKV TMA preparation, mean of 16 | 1.321 ms | 0.354 ms |
| Cumulative TMA kernel duration | 56.34 ms | 33.93 ms |
| Cumulative GEMM duration | 42.57 ms | 44.07 ms |

Cumulative kernel times overlap across streams and must not be summed as wall
time. These captures establish the mechanism, not ordinary long-run speed.
GEMMs did not get faster; preparation is no longer penalized by the unfavorable
captured execution order. The original diagnosis and negative controls remain
in [the graph investigation](precision200m_graph_execution_20261028.md).

## Sustained matched training on both home 5090s

Each process trained the same 205.85M SiTU decoder for 2,730 updates at batch
72, sequence 512: **100,638,720 byte tokens**. Same seed-941 CPU initialization,
corpus, validation samples, forward config, optimizer and precision recipe.
GPU 0 ran hybrid then full NVFP4; GPU 1 ran the opposite order. The analyzer
verified matched provenance within each board. No power/clock settings changed.

| Board | Backward | Training seconds | Final validation loss |
|---|---|---:|---:|
| GPU 0 | MXFP8 | 604.24 | 0.37959 |
| GPU 0 | NVFP4 fixed | 603.99 | 0.39085 |
| GPU 1 | MXFP8 | 572.09 | 0.38630 |
| GPU 1 | NVFP4 fixed | 570.19 | 0.38638 |

**Sustained throughput is effectively tied**, not a meaningful FP4 speed win.
The scheduling regression is removed, but isolated-backward gains still must
be judged against the complete training workload. The 0.04%/0.33% differences
are too small to claim a robust advantage from these runs.

All runs finished with finite parameters and audited gradients. Full NVFP4
had ~0.95 GiB less peak allocated memory. Final loss was nearly equal on GPU
1 and 0.01125 worse on GPU 0. This is one seed on two boards, not evidence of
generally better full-NVFP4 quality, nor a multi-seed convergence study. The
same-draw kernel contract is preserved; whole-training stochastic trajectories
are not asserted bitwise identical across policies or graph layouts.

The training clock includes data movement, forward/backward, clipping and
AdamW. Tuning, compilation, validation and checkpoints are excluded and logged
separately. Full process wall times were ~644.66/651.56 s on GPU 0 and
610.71/616.23 s on GPU 1 (hybrid/full), so cold-start full NVFP4 remains more
expensive to compile in these runs.

## Artifacts

Local ignored root: `autotune_reports/precision200m_fix_20261028/`.

- `model-refinement/`: residuals, complete commands, confirmed `designs.json`.
- `fixed-draw-model-shapes.json`: actual-shape equality and FP32 errors.
- `paired-gpu-tests.log`, `final-gpu-contracts.log`: GPU contract checks.
- `qualification-gpu0/`, `qualification-gpu1-v2/`: complete matched runs and
  per-board comparison reports.
- `nsys-integrated.{nsys-rep,sqlite}` and `integrated-trace-summary.json`.

The first GPU-1 qualification collided with an optional probe and OOMed before
training. It is preserved in `qualification-gpu1/` and excluded from results.
The restarted launcher uses a GPU-UUID file lease and checks for existing
compute processes before launching. No other user's processes were stopped.

The reusable option is in `rtx.experimental_ms_eden.ms_eden_linear`; the model
refinement API is under `rtx.autotune`. Neither is silently promoted to a new
public precision default or a universal performance policy.
