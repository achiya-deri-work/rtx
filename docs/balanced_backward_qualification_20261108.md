# Long-token MXFP8 backward qualification, 2026-11-08

Status: implemented and laptop-qualified. Real G2DETR/5090 requalification is
pending availability; no HOME or work 5090 workload was touched.

## Diagnosis

The supplied G2DETR report captures an older application on HOME 5090, Torch
2.14.1+cu132, effective batch 64. That stack is not the laptop test stack.
Its copied runtime's `mlp.py` and `_bwd_search.py` matched this task's starting
source, main `fe7a48c`; the application copy itself has no RTX git identity.
Source-tree digest in its manifest:
`4225e30a8b6cd5a62a549bd6c1197213860ad2ae052d5a5eac51ee53f367d454`.

Reading its saved source-attributed kernel records for the first fine-stream
MLP gives the following kernel sums per 64-image update (two microbatches):

| Backward phase | GPU kernel ms | Share |
|---|---:|---:|
| Two weight-gradient GEMMs | 2.063 | 40.6% |
| Quantization/layout preparation | 1.382 | 27.2% |
| Gate/up recomputation and derivative | 0.873 | 17.2% |
| Down-projection dH | 0.474 | 9.3% |
| Gate/up dX | 0.282 | 5.6% |
| Workspace reduction | 0.007 | 0.14% |

Total: 5.081 ms. The two dW grids have only 32 and 16 CTAs, despite their long
reductions. This identifies scheduling and preparation as useful experiments;
it does **not** measure achieved occupancy, spills or bandwidth saturation.
CPU launch gaps cannot account for these GPU-kernel sums. Recomputed backward
also performs extra arithmetic that scheduling cannot remove.

## Changes

See [the scheduling contract](balanced_backward.md) for API details.

- Add nonempty, balanced K-tile partitions and 64/128-way decomposed FP32
  workspace reduction. Keep fused/atomic/cluster support unchanged.
- Expose compound dW tile/split proposals through shared linear and MLP
  autotuning, including NVFP4 frontends using MXFP8 backward. Bound automatic
  workspace proposals and prioritize device-wave coverage without forcing it.
- Put fused gradient/hidden preparation in normal whole-MLP opening proposals.
  This is an existing execution family made easier to discover, not a new
  precision policy or a manually installed winner.
- Retain short/ragged complete 4/8-way openings; preserve both operands'
  quantization arithmetic, scale rounding and amax policies.
- Preserve legacy uniform JSON semantics and isolate expanded search results
  with backward search 30/frontend 6 and MLP search 42.

No application-side stream, scratch or CUDA-graph management is added.

## Complete-workload timing

Synthetic BF16 X/weights/upstream gradients use seed 81110 and standard
deviations 0.2/0.02/0.1. Shapes are `[M,K,H,O]`, gate weight `[2H,K]`.
Torch 2.13.0+cu132, default-mode fullgraph/static compilation, caller-owned
outer CUDA graphs, CUDA events; warmup 3, nine timing batches of three
workloads. A workload includes forward and all three gradients. Compilation,
search and qualification setup are excluded from steady-state medians.

Finalists use alternating AB/BA rounds. Promotion requires both aggregate
median speedup and the paired lower-quartile decision statistic to exceed 1%.
This is a consistency rule, not a statistical confidence interval.

| Device / shape | Trials | Paired rounds | Incumbent ms | Selected ms | Speedup |
|---|---:|---:|---:|---:|---:|
| 5060 laptop / `68320,256,512,256` | 8 | 6 | 9.151 | 8.509 | 1.0755× |
| 5070 laptop / `208192,128,256,128`, resumed study | 16 total | 12 | 11.817 | 11.072 | 1.0673× |
| 5070 laptop / same shape, final normal shared proposals | 8 | 12 | 11.316 | 10.690 | 1.0586× |

The final row uses search revision 42 with **no diagnostic portfolio**. Both
earlier studies used revision 41 with explicit diagnostic challengers. Those
records retain their original hashes; they were not relabeled as final-source
evidence. Resuming the fine study reused eight trials and added eight.

All promoted winners fuse gradient and hidden quantization into recomputation,
use separate output storage and keep overlap off. Their dW remains uniform,
eight-way, 128×128. Balanced splitting is therefore a qualified search-space
extension, **not the cause of these measured wins**. Its tested first challenger
was slower on the laptops. More CTAs alone are not an optimization guarantee.

The initial eight-trial fine study had a 5.7% aggregate improvement but failed
the unchanged six-round consistency gate; it retained the incumbent. The
extended and final-source studies passed. Laptop clocks drifted substantially,
so do not compare absolute timings across studies or extrapolate to 5090.

## Correctness, graphs and memory

- Initial 19-case balanced suite passed on both laptops: 5060 (26 SMs) and
  5070 (36 SMs). Stack: Torch 2.13.0+cu132 / CuTe 4.7.0.
- Updated 5070 regression: **49 passed, 9 skipped**. The skipped opt-in NVFP4
  split suite was then explicitly enabled: **25 passed, no skips**.
- Final opening/compiled regression: **21 passed** on 5070.
- Final balanced suite: **21 passed** on 5070, including a compiled NVFP4 MLP
  using the new shared MXFP8 backward partitions for both weight gradients.
- CPU compatibility/search/documentation regression: **159 passed, 91 CUDA
  cases skipped, 300 subtests passed**. Skips are not CUDA qualifications.

Independent FP32 arithmetic on explicitly quantized Torch operands checks dW
at 11296, 11712, 65568 and 208192 rows, including noncontiguous upstream
gradients and ragged K tails; normalized RMS error must be below 0.2%.
Exact integer workspace tests check every partial, including the last, for
64/128 partitions and all three reducer algorithms. A compiled recomputed MLP
checks output, dX and both weight gradients against the separate quantized
recipe oracle with a 2.5% gate. These gates measure implementation agreement
with the quantized recipe, **not** quantization error versus an ordinary BF16
MLP or detector convergence.
The separate compiled NVFP4 MLP uses its independent SiTU/quantization recipe
oracle and the existing 3% implementation gate for output and all gradients.

Changed-buffer graph replay must use current inputs. Eight warmed repeated
replays must have zero live PyTorch allocation growth, and replay after cache
cleanup must preserve the retained result exactly. This is not a claim of zero
graph-pool/driver allocation or lower total VRAM.

Across all 16 fine-study observations, maximum relative-L2 implementation
errors were output 0.00128%, dX 0.00226%, gate/up dW 0.01933% and down dW
0.01872%; all observations passed. These are synthetic recipe comparisons.

## Artifacts and remaining qualification

Local retained results and available raw residuals:
`autotune_reports/g2detr_balanced_split_20261108/` (ignored, Storage 4).
`5070-fine-final/result.json` records final RTX source hashes, benchmark hash,
arguments, selected config and paired samples; its residual journal records
per-trial numerical observations. Earlier
results remain in `5060-coarse`, `5070-fine` and `5070-fine-v2`. The 5060 result
was copied locally; its detailed remote journal was not retrieved after SSH
became unavailable again. Do not treat that journal as locally archived.

Reproduce normal proposal selection with
[the driver command](balanced_backward.md#qualification-and-deployment-boundary).
Use `--diagnostic-portfolio` only to add explicit diagnostic challengers.
The executed GPU regression file set was:

```bash
python -m pytest -xq tests/test_balanced_split.py \
  tests/test_mlp_complete_seeds_cuda.py tests/test_split_dw_long_cuda.py \
  tests/test_split_gemm_proxy_order_cuda.py tests/test_scratch_cleanup.py \
  tests/test_nvfp4_split_k.py
RTX_RUN_NVFP4_SPLIT_CUDA=1 python -m pytest -xq tests/test_nvfp4_split_k.py
```

Use an assigned idle CUDA device before promoting application winners.

Next: retune the exact saved fine/coarse/query shapes on an available 5090,
then measure the real packed G2DETR fixture, peak memory and full-step timing.
Preserve the live application's source/stack, numerical gates and previous
selection registry. No detector-step speedup, training convergence improvement,
new hardware-counter result or 1.0 release qualification is asserted here.
