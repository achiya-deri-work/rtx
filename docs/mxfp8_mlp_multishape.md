# SiLU/GELU multi-shape quality and performance

## Outcome

This study expands the experimental BF16-gate-forward/MXFP8-backward MLP to
seven shapes. It finds a reproducible **10.1% SiLU / 9.5% GELU runtime reduction**
on the long-reduction case without changing its numerical policy. The original
all-MXFP8 path also improves by **5.8% / 5.4%** with its own backward schedule.
Other aligned shapes gain much less; tiny and ragged cases still favor BF16.
No ReLU benchmark was run in this study.

These are explicit experimental schedules, not universal defaults. Production
module behavior and runtime winner caches remain unchanged. The all-MXFP8
configurations are usable with the existing MLP API; the hybrid remains the
benchmark composition described in [the previous runtime study](mxfp8_mlp_quality_runtime.md).

## Numerical coverage

All shapes below were tested with both SiLU and GELU. Dimensions are
`(tokens M, input K, hidden H, output O)`:

| Shape | Coverage purpose |
|---|---|
| `128,256,512,256` | Small aligned, 2× expansion |
| `512,768,1536,768` | Previous target, 2× expansion |
| `2048,512,2048,512` | Longer reduction, 4× expansion |
| `1024,1024,4096,1024` | Wider features, 4× expansion |
| `8192,256,768,512` | Long dW reduction, 3× expansion, unequal input/output |
| `129,128,256,128` | Ragged token count |
| `129,136,272,120` | Ragged tokens, reduction and output features |

The first five shapes use std=0.02 truncated-normal weights (seed 17) and
uniform `[-0.5,0.5]` inputs/weights (seed 31). The token-ragged case uses
initialized weights; the feature-ragged case uses both distributions. The
optimized long-shape winner additionally uses initialized weights at seed 31.
There are **28 distinct hybrid shape/activation/distribution/seed cases**.
The 93 successful hybrid runs include repeats and schedule alternatives, not
93 independent samples. Across all policies there are 146 successful timing
reports in the initial and follow-up artifact directories.

Relative L2 error against the matched BF16 reference, across hybrid runs:

| Quantity | Error range |
|---|---:|
| Output | 4.887–5.304% |
| dX | 5.432–5.819% |
| dW gate/up | 5.442–5.732% |
| dW down | 4.890–5.097% |

The independent TorchAO-quantized reference models the combined gate/up
backward, not a differently rounded sum of two gradients. The largest native
discrepancy from that reference was **0.091%**, below the unchanged **0.3%**
implementation gate. Absolute error, RMS, reference RMS, cosine, worst-element
location and finiteness are retained per output/gradient in every result.
This is not a training-convergence study or a claim of BF16-equivalent branch
accuracy. The BF16 reference uses BF16 GEMMs with FP32 activation/product.

## Performance

RTX 5070 Ti, Torch 2.13.0+cu132. All times are forward plus fully recomputed
backward, compiled with `fullgraph=True`, `dynamic=False`, aggressive options,
and CUDA graph replay. Warmup, compilation, autotuning and capture are excluded.

Representative SiLU results in microseconds:

| Shape M,K,H,O | BF16 | Hybrid initial | Hybrid selected |
|---|---:|---:|---:|
| 128,256,512,256 | 21.2 | 39.6 | 39.6 |
| 512,768,1536,768 | 173.0 | 130.8 | 130.6 |
| 2048,512,2048,512 | 517.1 | 344.0 | 336.1 |
| 1024,1024,4096,1024 | 975.9 | 653.8 | 649.0 |
| 8192,256,768,512 | 587.8 | 462.5 | **415.9** |
| 129,128,256,128 | 16.4 | Rejected TMA store | 173.4 |
| 129,136,272,120 | Not timed | Not timed | 233.9 |

Only the long-shape selected value is the median of three fresh-process
confirmations. Other selected values are search observations; small differences
are not established improvements. The original fused MXFP8 baseline uses the
existing transferred training preset, not exhaustive fresh tuning on every
shape. This is deliberately not an average-speedup headline across workloads.

The long-shape confirmations are:

| Policy | SiLU | GELU |
|---|---:|---:|
| Hybrid initial | 462.5 µs | 467.9 µs |
| Hybrid confirmed range | 415.3–415.9 µs | 422.8–424.2 µs |
| All-MXFP8 initial | 445.6 µs | 452.9 µs |
| All-MXFP8 confirmed range | 419.5–420.1 µs | 428.3–428.8 µs |

The hybrid is about **1.41× BF16 for SiLU / 1.40× for GELU** here. Its runtime
is close to the retuned all-MXFP8 path while its output error is approximately
5.21%, versus 6.33% for all-MXFP8. This relationship does not transfer unchanged
to the other shapes.

## What improved runtime

The initial Nsight Systems trace attributed approximately 101.8 µs to the
two dW GEMMs. Their output grids are small despite an 8192-element reduction.
We tested 2/4/8/16-way splitting, retaining FP32 partials and FP32 combination
before the final BF16 cast. Two-way splitting won; more splits regressed.

The selected hybrid schedule uses:

- materialized row-major up/down operands, 64×128×128 GEMM tiles and one
  forward GEMM stage;
- two-way dW splitting for both projections, reduction tile 4096;
- `split_fp32_workspace`, tree combination and direct FP32 partial stores;
- per-matmul quantization and dual-stream backward execution.

The selected all-MXFP8 schedule uses the same split count but **single-stream**
per-matmul backward. The runtime policies therefore must remain independently
searchable. Independent up/down tile sizes and stages are also exposed in the
benchmark rather than forcing both GEMMs to share a schedule.

In the optimized hybrid trace, dW GEMM time falls to about 60.8 µs and the two
FP32 combination kernels cost about 5.9 µs. Quantization/compute overlap also
changes. Summed kernel durations are not the critical-path runtime; all speed
claims above come from ordinary CUDA-event graph measurements, not profiler
timings. No tensor-pipe, DRAM/L2 or spill counters were collected. The remaining
large costs include gate recomputation and activation/gradient movement.

Negative results are retained: cooperative native forward regressed on this
long shape at the two tested resident-grid settings; larger split counts,
some persistent schedules and wide tiles also lost. This is a targeted search,
not proof of the globally optimal configuration.

## Ragged cases and limitations

Four initial ragged-hybrid attempts were rejected before execution because
the selected TMA epilogue required full output tiles. Direct stores fix that
schedule limitation for ragged M. Feature-ragged tests use the existing fused
forward/backward paths. An optimized existing fused-backward configuration
was selected for those tests; no BF16 tensor is padded or transposed in GMEM
by the benchmark.

The ragged paths pass numerics but are slow. Do not interpret successful
logical-tail support as efficient small-shape training. The benchmark checks
that compiled forward retains no non-input hidden storage, but this study
does not establish transient-memory parity, an allocator leak bound, optimizer
convergence, eager checkpoint ergonomics or export readiness for the hybrid.

## Reproduction and artifacts

```bash
python -m benchmarks.sweep_mlp_quality \
  --output-dir autotune_reports/mlp/new_multishape --seconds 2700
```

The bounded study runs coverage before search, with one isolated process per
trial and individual result/outcome/log files. Run one driver per output root.
Current resume keys include repository Python sources, package versions and
GPU/driver identity; changed code or devices do not inherit old outcomes.
The recorded v1 run predates this expanded fingerprint and used its worker
source hash. Partial result/outcome files are preserved under an incomplete
filename when recovery is necessary.

Reproduce the long-shape hybrid winner:

```bash
python -m benchmarks.benchmark_mlp_quality \
  --shape 8192,256,768,512 --activation silu --policy bf16_gate_fwd \
  --backend materialized --materialized-layout row_major --tile-m 64 --stages 1 \
  --dw-splits 2 --bwd-stream dual_stream --bwd-quant-schedule per_matmul \
  --output autotune_reports/mlp/new_long_hybrid.json
```

For the native MXFP8 winner, use `--policy fused --bwd-stream single` instead.
Its `mlp_preset` result field can be loaded using `MXFP8MLPConfig.from_dict` and
passed to the existing MLP API, explicitly scoped to this shape/device. **The
hybrid's `mlp_preset` alone does not encode its BF16-gate policy**; do not load
that field and assume the production MLP has changed numerical recipes.

Local evidence:

- `autotune_reports/mlp/quality_multishape_v1/`: 100 scheduled trials, 96
  successful timings and four static tail-store rejections.
- `autotune_reports/mlp/quality_multishape_followup/`: split/overlap experiments,
  ragged corrections, independent-input verification and confirmations.
- `autotune_reports/mlp/quality_multishape_final.json` and `.csv`: consolidated
  results; the JSON's best-search observations are not confidence intervals.
- `quality_multishape_long_{baseline,optimized}.nsys-rep` and SQLite exports
  under `autotune_reports/mlp/`: raw timeline evidence.

```bash
python -m benchmarks.summarize_mlp_quality_sweep \
  autotune_reports/mlp/quality_multishape_v1 \
  --extra-root autotune_reports/mlp/quality_multishape_followup \
  --output autotune_reports/mlp/quality_multishape_final.json
```
