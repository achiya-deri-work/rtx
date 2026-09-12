# Measured MLP quality/runtime choices

## Result

The best new compromise in this local search is **BF16 gate forward, MXFP8 up
and down, and a combined MXFP8 gate/up backward**. Activation/product remain
FP32, with BF16 linear/hidden boundaries. It improves numerical fidelity while
remaining faster than recomputed BF16, but does not beat our fastest all-MXFP8
training path. This is an experimental composition in the benchmark module,
not a new production constructor option or a replacement default.

RTX 5070 Ti, Torch 2.13.0+cu132, SiLU, `(M,K,H,O)=(512,768,1536,768)`:

| Policy | Forward + backward | Output error | dX error |
|---|---:|---:|---:|
| Recomputed BF16 | 172.9–173.1 µs | 0% | 0.287%* |
| Existing tuned fused MXFP8 | 112.4–112.5 µs | 6.325% | 6.335% |
| BF16 gate forward + combined MXFP8 backward | 130.9–131.3 µs | 5.202% | 5.791% |
| BF16 gate forward and backward | 144.3 µs | 5.202% | 5.144% |

The first three runtime ranges are three fresh-process confirmation medians,
not extrema of a single timing distribution. The last row is a search result.
Errors are relative L2 against a BF16 reference with FP32 activation/product,
using identical inputs and std=0.02 truncated-normal weights. *The reference
adds separate BF16 gate/up dX contributions; the BF16 baseline uses a combined
projection reduction. Their small gradient difference is not FP8 error.

The new compromise gives **1.32× BF16 training speed**, at about **17% extra
time versus fused MXFP8**. Its gate/up and down weight-gradient errors are
5.668% and 5.052%, versus 6.218% and 6.191% for fused MXFP8.

## Broader checks

- ReLU at the same initialized shape: the new policy has 5.785% dX error,
  substantially below the approximately 12% all-MXFP8 behavior seen in the
  matched studies. Runtime was 130.6 µs. GELU was 134.7 µs with 5.804% dX error.
- Uniform `[-0.5,0.5]` inputs/weights, independent seed 31: SiLU/GELU/ReLU
  output errors were 4.907%/4.887%/4.885%; dX errors
  5.462%/5.458%/5.442%. No large-magnitude or trained-model guarantee follows.
- At M=2048 with the same feature dimensions and transferred schedules:
  BF16 570.3 µs, fused MXFP8 325.3 µs, new policy 379.6 µs. The new policy is
  about 1.50× BF16 here. These are not independently exhaustive shape tunes.
- The benchmark verifies no non-input saved storage after compiled forward.
  All candidates use full recomputation. The hybrid is **not** the original
  two-forward-kernel implementation; it materializes temporary BF16 values.
- CUDA reference tests passed on both Torch 2.13 and 2.12.1. Fullgraph compiled
  forward/backward and CUDA graph replay ran on both stacks. This does not
  constitute an eager checkpoint API, export, allocator-leak, or optimizer
  convergence qualification.

## Winning schedule and search

The leading hybrid uses materialized MXFP8 up/down with row-major scales,
64×128×128 GEMM tiles, 2×2 MMA warp layout, one GEMM stage, no persistent
waves, and eight-value quantization with 128-bit loads and 32-bit stores.
Quantization uses BF16x2 arithmetic and BF16-bit amax. Its backward uses the
existing revision-8 MLP preset's gate configuration and shape-default down
configuration. Each result JSON serializes the exact parameters, including
the MLP preset, rather than relying on these abbreviated notes. Final v2
results additionally materialize the inherited backward/default configurations.

Tested alternatives included scalar versus materialized forward, native
mma128 and asymmetric mma64x128 scale layouts, 64/128-row tiles, wider
256-column tiles, 1/2/3 stages, persistence, packed backward quantization,
default versus preset backward, and compiler GEMM/epilogue choices.

Important negative results:

- The initial scalar hybrid cost 332.7 µs; materialization brought it to
  147.5 µs before further tuning. This is not a speedup over the existing
  optimized fused MLP.
- Row-major 64×128/one-stage and native mma128 128×128/two-stage schedules
  were nearly tied around 144.3 µs for the full-BF16-gate policy.
- One-stage backward regressed the forward-only-gate policy to about 149 µs;
  two stages stayed near 131 µs. Persistence/packed arithmetic did not give
  a robust further win.
- A valid 64×256 forward tile cost about 145 µs end to end. Some wider/deeper
  layouts exceeded SMEM or violated native scale-layout contracts and were
  rejected. Initial experiment-driver configuration errors are retained in
  logs; they are not kernel correctness failures or timing observations.
- Forcing ATen GEMMs or disabling epilogue fusion did not improve the 2.13
  full-BF16-gate candidate.

This is a targeted search, not proof of the global optimum. Numerical policies
remain explicit: a latency search must not exchange the full-BF16-gate and
forward-only-BF16-gate recipes behind the user's back.

## Compiler caveat and strict reference

Torch 2.12.1's aggressive epilogue-fused candidate had about 1.15% relative L2
disagreement with the independent recipe in output and dW-down. Switching to
ATen GEMMs or disabling epilogue fusion removed it. This isolates the effect
to a compiler-path choice, not a fully diagnosed generated-code root cause.
Do not promote that aggressive 2.12 candidate merely because its BF16 error
looks similar.

The final benchmark uses a dedicated TorchAO-quantized reference for the
**combined** backward and a stricter 0.3% implementation gate for the new
policy. Final 2.13 errors against it were at most **0.0033%**. The conservative
2.12.1 configuration (`--compiler-mode no_epilogue`) passed the strict gate
at about 134 µs. Earlier JSONs retain their original broader gate/reference;
use `final_strict*.json` for final implementation evidence.

## Profiling and measurement

Each timing uses five warmups, three operations per captured CUDA graph,
and 15 event-timed replays. Compile, autotune, input creation, validation,
and capture are excluded. We used fresh processes per candidate and interleaved
BF16/fused/hybrid measurements in a fixed sequence across confirmation rounds.
No clock locking or cross-device portability claim is made.

Nsight Systems shows that the hybrid's BF16 gate work is a meaningful cost.
The forward-only-gate version replaces the separate BF16 gate backward GEMMs
with the existing combined MXFP8 backward, and the timeline contains fewer
kernels (16 versus 18 in the initial full-gate capture, which also used a
different forward layout). The captures support this structural explanation;
they are not the performance timer. We did not collect tensor-pipe, spill,
DRAM or L2 counters, and make no saturation/bandwidth diagnosis from this trace.

## Reproduce

```bash
python -m benchmarks.benchmark_mlp_quality \
  --policy bf16_gate_fwd --backend materialized \
  --materialized-layout row_major --tile-m 64 --stages 1 \
  --output autotune_reports/mlp/my_quality_run.json
```

On Torch 2.12.1, add `--compiler-mode no_epilogue`. Use a fresh output path.
`--policy bf16` and `--policy fused` run the matched baselines. `--help` lists
the explicit schedule, shape, seed, distribution and compiler controls.
The default preset is intentionally a 5070-Ti training-tuned seed; it is not
a universal best forward/inference configuration.

```bash
python -m unittest tests.test_mlp_quality_projection \
  tests.test_mlp_precision_study -v
```

Local artifacts: `autotune_reports/mlp/quality_runtime_v1/`, with per-run JSON,
raw Nsight Systems reports, SQLite exports, and logs. Best strict configurations
are `final_strict_v2.json` and `final_strict_torch2121.json`; confirmations are
`confirm_*.json`. No production defaults, runtime winner caches, or public
module constructor contracts were changed. Next promotion requirements are
matched training/update evidence and a qualified frontend for the new policy.
