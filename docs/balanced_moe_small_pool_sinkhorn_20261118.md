# Packed BalancedMoE: persistent small-pool Sinkhorn

This report covers the original total-bounded path. The
[per-pool hybrid follow-up](balanced_moe_hybrid_pool_sinkhorn_20261118.md)
extends it to larger packed totals containing short pools.

The packed router now has a bounded, optional Sinkhorn family that keeps one
independent pool in one CTA. It preserves the five default finite FP32
row/column iterations, every saved normalization probability, and the hard
QB/repair path. It does not change the affinity calculation, K, capacity,
tie-breakers, or any solver budget.

At each direction the staged path launches one kernel per normalization and
separate initialization/finalization or unpack kernels. The new direction is
one launch, with the same history shape. For the sharded staged configuration
used in the RTX 5070 comparison, the count changes from 17 forward + 16
backward launches to 1 + 1, excluding the unchanged setup, metadata and hard
route kernels. Direct writes to real-token score gradients also avoid two
packed-size backward intermediates.

The choice is only available for a conservative static bound of 512 scheduled
rows per pool and 80 KiB shared memory per CTA, so captured boundary values can
change without changing the kernel plan. The tuner tests staged, persistent
forward, persistent backward and both directions. It compares CTA sizes, with
two protected opening proposals on small pools; the larger-pool candidate
prefix and staged fallback are preserved. Forward persistent candidates have
an exact hard-metadata gate against a staged transport with matched FP32
reduction order on both the caller and synthetic calibration scores.

## Measured mechanism

RTX 5070 Laptop (SM120), Torch 2.13.0+cu132; D=384, E=32, K=4, H=128 and
shared H=256 for whole-MoE runs. All timings use warmed ordinary CUDA-event graph
replay; compile, correctness checks and capture are excluded. Candidate-only
operator probes measured the same packed FP32 scores, saved history,
upstream, and staged config across four alternating-order rounds. Each round
retained seven raw samples.

| Pools | Staged forward/backward | 256 threads | 512 threads | 1024 threads |
| --- | ---: | ---: | ---: | ---: |
| [128] | 26.6 / 20.5 µs | 40.9 / 38.8 µs | 24.7 / 22.5 µs | 21.6 / 18.1 µs |
| [255, 0, 129] | 26.7 / 20.5 µs | 73.6 / 71.7 µs | 43.2 / 38.9 µs | 33.7 / 26.6 µs |

Packing is excluded from this table. At 128 tokens, the wider CTA reduces
isolated forward latency by about 19% and backward by about 12%. At 384 tokens,
even the widest CTA loses both isolated comparisons. Fewer launches are not
enough when work per SM rises; this is not a production size threshold or a
hand-pinned device schedule.

## Whole layer

The 128-token confirmation ran four fresh processes in persistent / staged /
staged / persistent order. Each visit used two rounds of seven raw samples,
frozen baseline expert, shared, and hard-routing schedules, identical BF16
inputs and upstream, and real compiled forward/backward to every leaf. There
was no retuning in these comparisons. Reported values are the medians of the
per-visit medians. Standalone phase times do not add to whole-layer time.

| 128-token phase | Staged | Persistent, 1024 threads | Change |
| --- | ---: | ---: | ---: |
| Transport forward, with packing | 35.3 µs | 27.4 µs | −22% |
| Transport backward | 24.7 µs | 17.4 µs | −29% |
| Router forward + backward | 419 µs | 351 µs | −16% |
| MXFP8 layer forward + backward | 875 µs | 754 µs | −14% |

The staged whole-layer round medians spanned 0.769–0.915 ms, compared with
0.741–0.771 ms persistent. One staged visit moved about 17% between rounds.
The direction is supported by both persistent visits and by the better
controlled transport probe; the WSL effect size is descriptive.

The 384-token ragged control did **not** show a full-layer win:
1.048 ms staged versus 1.090 ms at 1024 threads with both persistent directions.
The selected-width router phase alone was 0.524 versus 0.507 ms, with more than
10% inter-round spread on the staged visit; the isolated transport probe
confirms a regression. The 256-thread variant was slower still on both
directions. A 2,000-token larger-shape fixed staged early control was 1.452
versus 1.451 ms whole layer, with identical hard metadata and zero live
growth. The persistent family is not eligible there.

## Native Linux RTX 5090 confirmation

The HOME GPU 0 controlled run used the same integrated source for both modes,
Torch 2.13.0+cu132, and frozen 5090 expert, shared, and hard-routing seeds.
No tuner ran. Tiny `[128]` used staged / persistent / persistent / staged
fresh processes, and ragged `[255,0,129]` used one fresh process per mode.
The 1024-thread persistent variant and staged baseline each ran two rounds
per visit, with seven samples per round. Inductor was not max-autotuned.

| Pools | Staged router step | Persistent router step | Staged full step | Persistent full step |
| --- | ---: | ---: | ---: | ---: |
| [128] | 0.2827 ms | 0.2724 ms (−3.6%) | 0.4957 ms | 0.4858 ms (−2.0%) |
| [255,0,129] | 0.3913 ms | 0.4037 ms (+3.2%) | 0.6411 ms | 0.6520 ms (+1.7%) |

Tiny full-step round medians were 0.4955–0.4960 ms staged versus
0.4855–0.4864 ms persistent. That confirms the small-pool benefit without
the larger WSL effect size. The independent transport replay probe confirmed
the mechanism: tiny forward/backward went from 22.5/20.5 to 18.4/14.3 µs,
but ragged went from 22.7/20.5 to 32.6/24.6 µs. The ragged pool remains a
counterexample to forcing the new plan.

All six visits completed with exact input/parameter/upstream and hard-route
hashes within each shape and no live-allocation growth. Each persistent
visit passed the all-leaf numerical comparison and fresh-gradient replay;
worst relative L2 versus the staged reference was 1.96e-5. The focused
ragged FP64 saved-state/adjoint and exact compiled mutable-boundary replay
tests both passed on GPU 0. It returned to 0% utilization and its 15 MiB
desktop baseline; the other GPU was not used.

Tested runtime SHA256:
`c0a5ef1ab72fed895ff4168c3f9c3cd64879ef2254d5a5f60e2d11f658887054`.
[Raw samples, source, frozen configs, logs, and reference tensors](../autotune_reports/moe_next_priorities_20261118/routing_home_5090/).
The `same-source` mode in the [fresh-process launcher](../tools/qualify_persistent_pool_sinkhorn.sh)
reproduces the six visits from that snapshot and the archived seed directory.

## Qualification

- On the 5070, 10 focused tests passed. Six independent FP64 cases compared
  final log/probability output, every saved state, and score backward with
  noncontiguous upstream. They cover E=7/32/64/128, K=2/4/8/32, odd or empty
  pools, directional mixing and multiple iteration/CTA settings. Value/history
  gates were 5e-5 relative + 2e-6 absolute; score adjoint 5e-4 + 8e-6.
- Compiled-vs-compiled fixed-shape graph replay mutated pool boundaries across
  empty and nonempty positions without recompilation. IDs and inverse slots
  matched exactly; mixture weights and all router/input gradients passed their
  numerical gates, and postwarm live allocations stayed flat. Comparing a
  compiled candidate to eager affinities was not an exact hard-route control:
  FP32 affinity rounding exposed a known near-tie sensitivity. Both sides use
  the same affinity execution mode in the successful comparison.
- The selected-route operator passed real-device schema, fake tensor,
  autograd registration and AOT opcheck. Local CPU/fake suite: 35 passed,
  12 skipped, 72 deselected. The trial-order tests exercise the actual shared
  orchestrator with a stub timing evaluator, not a measured hardware winner.
- All completed whole-layer controls passed exact hard metadata, all-leaf
  finite/source comparisons, fresh-gradient replay and zero live-allocation
  growth. The largest tiny whole-layer relative-L2 delta was 0.0033%
  (router projection gradient); the ragged maximum was about 0.60% (one
  quantized expert weight gradient). The per-tensor results are in JSON.

Reproduction: [operator probe](../benchmarks/benchmark_persistent_pool_sinkhorn.py),
[fresh-process launcher](../tools/qualify_persistent_pool_sinkhorn.sh), and
[tests](../tests/test_routing_persistent_pool.py). The reproduction baseline
snapshot is archived at
`autotune_reports/moe_amortization_scaling_20261118/final-source.tar.gz`
(runtime SHA256
`6a3c7a776a921f8e6a9886a3efe62ea99230be3fc7627b0026a34e9ae3243155`).
The final wide candidate used runtime SHA256
`4e609c65b1048211c350707be9a89c93795a3f51fbfd55f165214fbe9d901ffb`.
The subsequent CPU-checked host guard for signed index bounds in the
pathological million-empty-pool regime does not change any measured launch.
Raw rounds, captured replay checks, commands, matched references, seeds and
the tested candidate snapshot are under
[`autotune_reports/moe_next_priorities_20261118/routing_5070`](../autotune_reports/moe_next_priorities_20261118/routing_5070/).
