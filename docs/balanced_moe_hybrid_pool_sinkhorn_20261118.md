# Packed BalancedMoE: per-pool persistent Sinkhorn

The optional persistent transport no longer requires the entire packed total
to fit one CTA. If that original direct path does not fit, a multi-pool
invocation can use, within the same shape and call:

- One persistent CTA per nonempty pool with at most 128 scheduled rows.
  The limit is rounded down to a valid capacity multiple and still obeys
  the shared-memory bound.
- The configured staged row and column implementation for longer pools.
  The two paths write disjoint regions of the same FP32 log-transport,
  saved history and gradient buffers.

Eligibility comes from device-resident scheduled boundaries. In a CUDA
Graph, a device predicate guards the staged launch sequence, so an all-short
replay executes only the persistent kernel and predicate per enabled
direction. The conditional body captures with the actual compiler-owned
addresses; neither forward nor backward copies transport tensors through
a fixed-address side buffer. Eager calls run both masked paths and do not
read boundaries back. Conditional bodies use the CUDA driver
[capture-to-graph API](https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__CONDITIONAL__NODE__PARAMS.html).

The public config/token, fixed iteration budgets, hard routing, and saved
normalization adjoint do not change. The shape-static graph supports replay
with short, long, or empty pools in different positions. The packed autotuner
checks the cold profile for useful pools; it does not freeze that profile into
the runtime branch. Mixed contexts keep a staged contender and all-long
cold profiles retain their original openings. No default was changed.

## Correctness

On the RTX 5070 Laptop the focused suite passed 17 tests. It included
both-direction and backward-only FP64 mixed-pool tests (including E=7,
ragged K-capacity multiples, empty pools, and noncontiguous upstream
gradients), fullgraph hard-route/input/all-router-gradient comparison, and
zero postwarm live-allocation growth. One replay changes the boundaries
from all-short to mixed to all-long and back with identical shape. A
poisoned staged workspace proves the all-short graph does not launch its
staged branch, including after a mixed replay. The saved FP32 state and tail
values are compared independently of the native geometry. Local CPU/fake
suite: 28 passed, 2 GPU skips, 76 deselected.

## Timing: the CUDA conditional does not pay on this laptop

The frozen-schedule RTX 5070 Laptop comparison used Torch 2.13.0+cu132,
BF16 D=384/E=32/K=4/H=128/shared H=256, identical hard/experts/shared
configurations and upstream tensors, and warmed ordinary graph replay. Only
the persistent transport mode changed. The repeated-short case used four
fresh processes in staged/hybrid/hybrid/staged order. The mixed and
all-long controls each used one process per mode. All visits kept two
seven-sample rounds. Compilation, capture and checks are excluded.

| T=1024, B=8 pack | Router staged → hybrid | Whole step staged → hybrid |
| --- | ---: | ---: |
| Eight × 128 | 0.580 → 0.567 ms | 1.846 → 1.782 ms |
| [768, 128, 119, 9, 0, 0, 0, 0] | 0.723 → 0.756 ms | 1.852 → 1.934 ms |
| [512, 512, 0, 0, 0, 0, 0, 0] | 0.452 → 0.547 ms | 1.570 → 1.890 ms |

The short staged full-step round medians spanned 1.689–1.970 ms, versus
1.742–1.819 ms hybrid. This WSL drift prevents claiming a full-step win.
All eight visits passed exact hard route and operand identities, all-leaf
and fresh-gradient replay checks, and zero measured live-allocation growth.
Worst relative L2 versus the matched staged reference was 1.46e-5.

The paired transport-only graph probe used alternating order in one process,
with identical packed operands and the same fixed addresses. It distinguishes
the raw 1024-thread kernel (feasible only for this known all-short fixture
without a fallback) from the real graph path:

| All-short pack | Staged fwd/bwd | Raw persistent fwd/bwd | Safe hybrid fwd/bwd |
| --- | ---: | ---: | ---: |
| Eight × 128 | 34.5 / 26.1 µs | 22.0 / 18.4 µs | 80.6 / 75.1 µs |
| Sixteen × 64 | 37.0 / 29.7 µs | 16.1 / 10.4 µs | 74.2 / 71.6 µs |

The raw kernel wins, but the conditional path (predicate plus graph branch)
adds about 57–61 µs per direction on this WSL device. The poisoned-workspace
test verifies this is not simply the staged kernels running on all-short
input. This is a measured reason not to promote the hybrid as a speed
default. The next experiment would isolate that branch cost on native Linux,
or find a cheaper GPU-only fallback; a static assumption about the lengths
would not be safe under replay.

A two-slot real router tuner with the generic exploration slot disabled
measured both staged (529.17 µs) and hybrid (524.84 µs). Both passed exact
hard metadata and independent FP64 calibration; the tuner picked hybrid by
less than 1%. That threshold-sized result demonstrates search coverage, not
a deployment decision. The normal bounded defaults and staged fallback
remain unchanged.

## Reproduction

[Raw samples, replay checks, tuner journals, frozen configs, and source](../autotune_reports/moe_next_priorities_20261118/routing_hybrid_5070/)
use runtime SHA256
`d3d03b35304cb818473d1b71d3831caee2d96a2b455f993426f08dd31e1e42c4`.
Archive checksums are included. Use the
[fresh-process launcher](../tools/qualify_hybrid_pool_sinkhorn.sh) for the
eight whole-layer visits and the
[transport probe's `--hybrid` mode](../benchmarks/benchmark_persistent_pool_sinkhorn.py)
for the paired graph control. Both keep the policy budgets fixed.
