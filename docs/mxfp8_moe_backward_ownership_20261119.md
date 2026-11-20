# Routed MXFP8 backward: implementation and measured results

Three Sol 6.2 high agents implemented the [Astra review](mxfp8_moe_1p2_review_20261119.md)
priorities. Two changes compose profitably; on-chip indexed-X quantization is
correct but loses the target workload and remains experimental. **The 1.2x
whole-MoE target is not achieved.**

## Whole-path result

HOME RTX 5090 GPU0, 170 SMs, Torch 2.13.0+cu132/CuTe 4.7.0. D768, H256,
E48, K6, shared H512, packed independent lengths 4097+8223, physical expert
capacity 1600. Four alternating visits, seven ordinary event samples per
visit, warmed outer graphs; fullgraph/static compilation and aggressive
dynamic-weight Inductor settings for both precisions. Same reentrant BF16
checkpoint control, no freezing/PTQ. Exact X, dY, parameter and hard-route
hashes match the source study.

Final guarded source:
`f2a63f5465c46d62c24b13e3b0b74b6d0c76e002241bbbd06e6c188b25499f81`.

| Final paired whole forward+backward | Median |
|---|---:|
| Original MXFP8 schedules | 5.46794 ms |
| Composed backward schedules | 5.27036 ms |
| Same-router compiled checkpointed BF16 | 5.69191 ms |

The composed change saves **197.58 us / 3.61%** of whole-layer time, or
**1.0375x** versus the old MXFP8 path. BF16/MXFP8 improves from **1.041x to
1.080x**. Another **527 us of native-only saving** would be needed at this
BF16 time to reach 1.2x. This is not a sum of isolated kernel percentages.

A preceding four-cell paired factorial measured original / persistent /
route / combined at 5.47342 / 5.36906 / 5.38771 / 5.27524 ms. Interaction was
-8.12 us: the gains survive composition rather than merely replacing one
another. The final result above repeats the composed comparison after the
static safety/identity audit.

All outputs and gradients pass the unchanged comparison gates. Persistent
dW alone is bitwise identical to packed dW. Composed maximum normalized RMS
difference against the original schedule is **0.00214%**, from changed FP32
route-adjoint summation grouping. Changed-dY replay passes; five synchronized
live-allocation samples are identical in each variant. This is numerical and
optimization qualification, not model convergence or optimizer-step timing.

## Implemented families and controls

### Retained operand panels for unsplit dW

One TMA producer loads a common panel once per K stage. Two or three warp
teams retain separate 64x64 accumulator tiles while sharing that panel.
Neighbor-M and neighbor-N reuse are independent choices for gate/down dW.
There is no global partial workspace or intermediate BF16 reduction. FP32
accumulation spans the complete K reduction before the final BF16 cast.

Fresh paired packed/candidate graph timings on the source D768 fixture:

| Kernel | Packed | Retained panel |
|---|---:|---:|
| Gate dW | 305.20 us | 226.92 us, N-neighbors x3 |
| Down dW | 141.61 us | 116.01 us, M-neighbors x3 |

Both results are bitwise equal to packed; sampled independent quantized FP64
NRMS is 0.00641% / 0.00419%. Wider output-tile and split-K controls lose.
For these divisible shapes, generated code schedules one third fewer input
TMA instructions overall, with one third as many CTAs. CUBIN actual register
counts rise 64->79 and 63->77; stack/local usage remains zero. These are
lowering/resource observations, not measured bandwidth or occupancy counters.

Fields: `gate_dw_engine`/`down_dw_engine="persistent"`, per-stage
`*_reuse_axis="m"|"n"` and `*_reuse_tiles=2|3`. Current legality is unsplit,
static-capacity row-major TMA, 64x64/K128 geometry, scalar scales, sufficient
neighbor tiles and SMEM. Generic persistence/tiles-per-CTA are separate.
Unused role/register-cap/cache coordinates are canonicalized, not hand-tuned.

### Independent route-adjoint tiling and paired upstream loads

The optional `route_gradient_gemm` no longer needs forward down's output
tile. `route_gradient_loader="warp_vec2"` broadcasts slot mappings and uses
paired FP32 upstream loads in the accumulator's ownership layout. BF16
rounding of the recomputed down result stays **before** its dot with dY;
partials and deterministic final reduction stay FP32. The alignment check is
pointer metadata in the compiled CuTe launcher; misalignment selects scalar
loads without a copy or eager Torch arithmetic.

On exact D768 routes, the guarded route phase changes from **225.02 to
165.79 us**. Of this, the independent N64 tile accounts for 48.90 us, and
the loader another 10.33 us. Frozen fullgraph expert F+B changes **2.31246
to 2.24358 ms**. Same-tile loaders are bitwise equivalent; cross-tile route
NRMS is about 1.01e-7. The independent two-upstream FP64 oracle's worst
expert-gradient NRMS is 0.0137%.

`"accumulator"` remains available, as does `"shared_vec4"` coalesced shared
staging. Shared staging is a retained negative control: it is slower at both
tested widths, so the protected candidate uses warp loads rather than shared.
Smaller tiles enlarge only the FP32 partial array (D768: 1.84->3.69 MB), not
a full expert-output tensor. Logical workspace is not peak allocator usage.

### Indexed X quantization inside gate-dW: do not promote this as a win

`gate_dw_engine="indexed"` removes the global transposed-X pair and computes
its independent E8M0 blocks on chip. Cooperative and dedicated producer
variants expose token-lane groups, paired gathers and pipeline staging.
All seven target candidates match packed results bitwise. However, producer
plus consumer takes **747.43 us best versus 360.10 us** for separate indexed
transpose plus packed GEMM. This is a **2.08x slowdown**, not a fast-path win.

It removes 60.83 MB logical storage, but an M64 gate-dW repeats each indexed
gather/amax eight times across 512 output rows. Eliminating a buffer did not
eliminate its work. A one-output-tile thin probe only ties at 6.568 us; a
second thin case loses 8.192->12.200 us. Thus the engine is explicit/deeper
search only, not protected in light tuning. No quantization or gradient
precision was relaxed to rescue it.

## First-hit autotuning and a fresh workload

MoE kernel revision **12**, portfolio **22**; new kernel/helper source files
are included in the cache fingerprint. Existing serialized configs default
to packed engines and inherited adjoint tiling. Incumbent, direct/store
controls and shared UCB exploration remain. For long reductions, one complete
ownership challenger gets early coverage. Reuse axes/groups are ranked by
padded output work and scheduled operand panels, **not per-SKU winners**.

Fresh shape: D640, H192, E40, K5, shared H384; independent lengths 6137+9591,
physical capacity **2016**. The fourth actual trial selects gate M-neighbor
and down N-neighbor sharing, reversing the D768 directions. All 16 expert
and 16 shared observations pass numerical and timing eligibility.

| Actual progressive prefix | Expert best | Shared best | Whole MX / BF16 |
|---|---:|---:|---:|
| 8 trials per component | 1.71200 ms | .441792 ms | 5.58643 / 5.70584 ms |
| 16 trials per component | 1.71200 ms | .441792 ms | 5.57706 / 5.70141 ms |

Eight-versus-16 search regret is **zero** for both components. Worst
independent quantized FP64 expert NRMS is **0.00582%**; no live allocation
growth, no compile/correctness/timing failures. Full cold calls at eight
trials take **22.8 s expert + 18.7 s shared**. Setup and confirmation are
included in those numbers; the search deadline is soft.

Both prefix runs were frozen at source
`1ca42e0e39d7ab0d9d1ce364053711581b143bbead211f38e5861745a758ebcd`.
Final review then tightened explicit-policy guards and inactive-parameter
deduplication. Every one of the 16 recorded expert proposals was rechecked
against final source: legality and configuration IDs still match. Final
whole-module and focused GPU qualification use the final source digest above.
Neither finite-budget agreement nor this one held-out implies a global optimum.

## Safety and validation

- Final targeted GPU0 suite: **32 passed**, covering all three new kernel
  families plus the common hybrid epilogue. Tests include independent FP64
  quantized oracles, M/N/K tails, padded/invalid slots, multiple independent
  pools, fullgraph backward, noncontiguous/shifted-contiguous dY, canaries,
  changed-input/dY graph replay, and synchronized allocation stability.
- Final CPU plan/selection/compiler suite: **171 passed, one skipped**,
  including explicit-policy guard and inactive-identity tests.
- Final review fixes explicit `autotune="off"` bypasses: incompatible input
  maps, active prefixes, splits and scale layouts now fail before allocating
  scratch. Independent adjoint schedules also check active-prefix ownership
  and initial register-role bounds. New paths use compiler-visible existing
  buffer plans and allocation-free launchers, without user streams/workspaces.
- An extra N=1 probe initially faulted. Offline IR/PTX identified a singleton
  tensor-map rank mismatch. Persistent dW now rejects singleton outputs before
  launch; symmetric two-element tails pass. The failed process ended, a fresh
  GPU0 context passed, and **no reset or driver change was performed**.
- GPU1 and unrelated workloads were untouched. No fresh Nsight injection or
  hardware-counter claims; only generated-code inspection and ordinary timing.

## Evidence and reproduction

`autotune_reports/mxfp8_moe_backward_ownership_20261119/` is on the existing
Storage4 symlink. It retains `integration/` (paired/final whole results,
prefix residuals, summaries, XML), `persistent/`, `route/`, `indexed/`, and a
final source snapshot with a SHA-256 manifest. Earlier exploratory runs and
the singleton failure remain labeled, not promoted as final qualification.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \
  tests/test_moe_persistent_dw.py tests/test_moe_indexed_dw.py \
  tests/test_moe_route_adjoint.py tests/test_mxfp8_hybrid_epilogue.py

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.qualify_moe_store_prior \
  --output /tmp/rtx-ownership-check --cases ownership_heldout \
  --budgets 8 16 --max-autotune
```

The numerical policy, equal-load/fixed-K routing contract, BF16 public ABI,
MXFP8 forward/backward and full recomputation are unchanged. Better route
and dW schedules are real but insufficient for 1.2x. The remaining next target
is measured joint-recompute/preparation cost, not more replicated indexed
quantization or wider dW tiles.
