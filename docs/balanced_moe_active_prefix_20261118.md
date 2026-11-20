# Packed BalancedMoE: device-side active prefixes

The packed expert kernels can skip unused rows without shrinking the
compiler-visible tensors or reading sequence lengths back to the host.
The optional `prepared.active_prefix` schedule is off by default. On the
RTX 5070 Laptop, it passed the correctness checks, but the small packed
timing gains were not reproducible enough to promote it. The five-trial
search measured the candidate and kept active prefixes off.
On bare-metal RTX 5090 GPU0, active prefixes also worked with dual-operand
recomputation. They shortened packed expert forward, but did not improve
the full step; the one-pool control regressed.

## What changes

Packed sequences are still balanced independently. Each sequence keeps
its own 32-row quantization alignment, and experts still allocate the
same shape-only upper bound. A device kernel finds the largest
expert-local position in the inverse route map and rounds it up to a
32-row group. It uses true assignments, including assignments with zero
route weight. This bound changes at fixed tensor addresses on graph
replay.

For the tested pack `[128,512,360,1000]` (D384/E32/K4/H128), 250 real
assignments per expert occupy 288 physical rows. The static capacity is
352; 64-row scheduling formerly computed six row tiles. The active
schedule computes five. Producer extents include every complete input
tile a consumer can load. In backward the 128-K dW descriptor can still
read rows 320–351 after recomputation stops at 320, so a CuTe kernel writes
canonical zero FP8 bytes and the E8M0 code for an all-zero group into
just that transposed slice. Whole K tiles after the last active reduction
tile are omitted.
Combines use real inverse routes, and invalid route-gradient slots do
not read skipped output rows.

No public routing metadata changes, no client-managed buffers or
streams, and no CUDA allocations inside the launchers. The same MXFP8
quantization and BF16 public output/weight gradients are preserved; the
reductions and route adjoints remain FP32. The qualified 5070 policy is
row-major packed TMA with one output tile per CTA and an unsplit backward
using the legacy joint recompute engine. The untrimmed path is retained.

## RTX 5070 Laptop results

Frozen baseline runtime:
`6a3c7a776a921f8e6a9886a3efe62ea99230be3fc7627b0026a34e9ae3243155`.
Active snapshot:
`e099e9ee7f2ae2c45f4a9bef4463722dd5167c8eefb04a083ef47087f61e9ec9`.
Same archived router, expert tiles, shared schedules, inputs, parameters,
upstream gradients and hard routes. Timing used `torch.compile` fullgraph,
real backward to every trainable leaf, and warmed ordinary CUDA-event
graph replay. It includes prefix preparation and the scrub launch; it
excludes initialization, compilation, capture and tuning. Expert-only
timings use frozen routes; whole-layer timings also include router and
shared MLP. These standalone times are not additive.

Milliseconds, medians of the recorded round medians:

| Pack | Measurement | Frozen | Active | Active change |
| --- | --- | ---: | ---: | ---: |
| Four sequences | Expert forward + backward | 0.935 | 0.903 | −3.4% |
| Four sequences | Whole-layer forward + backward | 1.722 | 1.697 | −1.5% |
| One pool | Expert forward + backward | 0.737 | 0.767 | +4.1% |
| One pool | Whole-layer forward + backward | 1.697 | 1.660 | −2.2% |

The packed pair and its reverse-order repeat each contribute two rounds
of seven samples per phase. The one-pool control contributes two rounds
per policy. Several round medians drift by more than 10%, including the
active packed repeat and the one-pool controls. The table is descriptive,
not evidence of a reliable whole-layer improvement or a precise one-pool
overhead. The one-pool case has no full unused CTA/K tile to skip. The
five-trial packed autotune independently measured the active candidate
at 0.944 ms versus 0.905 ms for the inactive initial schedule, and
returned a config with `active_prefix=False`.

## Qualification

- **11 tests passed on RTX 5070 Laptop.** Six poisoned-scratch oracle
  cases cover a 192-row active prefix crossing a partial 128-K tile and
  a 128-row prefix that omits later tiles, with capacity deliberately
  oversized to 288. Cases include empty and one-token pools, gather-BF16,
  quantize-once, indexed transpose with fused dY/X, and both fused and
  separate route-gradient epilogues. Max normalized L2 error against
  the independent quantized FP64 recipe was `1.11e-7` across these tests.
- The matched N2000 packed and one-pool runs had exactly equal outputs
  and all optimizer-visible gradients versus the frozen source
  (maximum absolute difference zero). The FP64 expert oracle includes
  dX, both dW banks and route-weight gradients; worst normalized L2 was
  `2.42e-5`. Normalized L2 divides the error norm by the reference norm
  floored at `1e-12`.
- Fake/fullgraph and mutation-only ABI checks passed. The compiled
  graph accepted changed `cu_seqlens` with the same shape and address
  and replayed all gradients. Eight postwarm live allocation samples
  were identical; all benchmark records also had zero postwarm
  live-allocation growth. Static allocation is intentional; this is
  not a claim that the active schedule saves VRAM.
- The local focused CPU suite passed 152 tests, with 18 CUDA cases
  skipped. The 5070 was released with no compute apps or allocations.

Raw timings, numerical records, executable reference tensors, test logs,
the tuning journal and source archives are in
[`active_tail_5070`](../autotune_reports/moe_next_priorities_20261118/active_tail_5070).
The evidence archive SHA-256 is
`cd5aebadf08fb56af2a2a10dd8bcb75c3bbb31bb7b74fe60c10ce8484dc08490`.
To repeat the matched run, unpack `baseline-source.tar.gz` into both
baseline and candidate directories, then apply `active-overlay.tar.gz`
to the candidate. Use the archived router/expert/shared config files
with the existing benchmark:

```bash
python benchmarks/benchmark_moe_varlen.py --source . --output result.json \
  --case-id d384_e32_k4_h128 \
  --case-spec '{"d_model":384,"num_experts":32,"top_k":4,"expert_hidden_dim":128,"shared_hidden_dim":256}' \
  --lengths 128 512 360 1000 --expert-phases \
  --phases expert_forward expert_step mxfp8_step --rounds 2 --samples 7 \
  --router-config /path/to/fixture/result.router.json \
  --expert-config /path/to/fixture/result.expert.json \
  --shared-config /path/to/fixture/result.shared.json
```

For the matched candidate ablation, add
`--expert-overrides '{"prepared":{"active_prefix":true}}' --oracle`.
Use `--lengths 2000` for the one-pool control. The separate tuner visit
left the override unset and used `--tune-expert-trials 5 --tune-seconds 50`.

## HOME RTX 5090 GPU0: composed dual-operand check

The runtime snapshot
`2c69f64c65c74fd891bb38fbb6c1d440eacf21936808e491b44844d67353b1a8`
supports both active-tail and dual-operand recompute in the same plan.
The autotuner can switch either coordinate without silently resetting the
other when the combination is legal. This does not enable cooperative
transport, persistent M grids or split-K dW for active tails. Defaults
remain unchanged.

Both timing policies use the same dual-operand engine and recompute
schedule: M64/N64/K128, one stage, producer scales, exchange derivative
stores, interleaved gradient loads and XOR scratch layout. The only
timed ablation is `prepared.active_prefix`. Each item is the median of
two round medians, seven ordinary CUDA-event graph samples per round:

| Pack | Measurement | Dual only (µs) | Dual + active (µs) | Active change |
| --- | --- | ---: | ---: | ---: |
| Four sequences | Expert forward | 59.57 | 57.36 | −3.7% |
| Four sequences | Expert forward + backward | 209.34 | 210.62 | +0.6% |
| Four sequences | Whole-layer forward + backward | 687.57 | 689.19 | +0.2% |
| One pool | Expert forward + backward | 161.93 | 180.30 | +11.3% |
| One pool | Whole-layer forward + backward | 665.33 | 685.88 | +3.1% |

The packed forward improvement did not survive the full step, which
includes prefix discovery and partial-K tail initialization. The
one-pool result shows the cost of enabling this policy when it has
nothing to skip. These are fixed-schedule results for these workloads,
not an argument for enabling active prefixes by default.

**15 tests passed** on GPU0, including a dual-operand active-tail
poisoned-buffer oracle at both 192 and 128 active rows, and compiled
mutable-boundary replay with each recompute engine. The larger packed
and one-pool runs matched their fixed-dual controls numerically exactly
(all model output and gradient maximum differences zero). Max FP64
operational-oracle normalized L2 was `1.64e-5`; live allocation growth
was zero. GPU0 was handed back idle; GPU1 was not used.

The [HOME GPU0 evidence](../autotune_reports/moe_next_priorities_20261118/active_tail_home0)
contains the source snapshot, exact resolved configs, test logs,
reference tensors and raw samples. In the reproduction command above,
substitute the HOME source and use the resolved
`evidence/packed_dual/result.expert.json` and
`evidence/packed_dual_active/result.expert.json` files as the two expert
configs, without any additional override. The one-pool records have the
same comparison with `--lengths 2000`.
