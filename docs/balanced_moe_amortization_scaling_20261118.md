# BalancedMoE: where MXFP8 overhead amortizes

## Conclusion

This is not just a quantizer-launch problem. Small token pools suffer from
routing/graph fragmentation and padded expert tiles. After increasing tokens,
per-token movement/quantization and a poorly utilized fused recompute schedule
remain. Increasing the expert hidden width helps more reliably than increasing
tokens alone. Light first-hit searches also leave meaningful scheduling headroom.

This is a bounded diagnosis, not an exhaustive search or a claim that every
shape is faster than BF16. No production numerical policy or default changed.
There are **35 completed numerical/replay/memory-valid shape/device contexts**,
one excluded 5090 failure, two additional refinement controls and four ABBA
confirmation visits. All use runtime source digest
`6a3c7a776a921f8e6a9886a3efe62ea99230be3fc7627b0026a34e9ae3243155`.

## Protocol

The core sweep holds D=384, H=128, E=32, K=4 and shared H=256, then changes
one axis: tokens 128/512/2000/7376/29504; D=256/512/768; H=64/256/512;
K=2/8; E=16/64. There is a same-total-token packed comparison, plus two
large-token width/hidden interactions. The laptops repeat selected axes;
width-128 and bounded tuning-headroom controls are separate.

- RTX 5090: 170 SMs; laptop 5070: 36; laptop 5060: 26. Torch
  2.13.0+cu132, CUDA 13.2; exact environments and source hashes are archived.
- BF16 operands/parameters, MXFP8 expert and shared GEMMs, FP32 gating and
  router arithmetic. BF16 comparison checkpoints FFNs, never the router.
- `torch.compile(fullgraph=True, dynamic=False)`, Inductor max autotune,
  warmed outer CUDA Graph, real `.backward()` to every optimizer-visible leaf.
  Compilation, tuning, capture and initialization are excluded. No optimizer.
- Two timing rounds, seven calibrated CUDA-event samples per phase, a 200 ms
  graph-warmup target (bounded at 4096 replays). Separate whole layer, router and frozen-route
  expert measurements. **Standalone phase times are not additive.**
- Per new point: at most 12 expert / 8 router / 8 shared trials, with a
  30-second soft limit per tuner. Actual counts are recorded; cold compilation
  can exhaust that time before the requested trial count.
- Hidden-width comparisons preserve exact X, dY, router/shared parameters and
  hard-routing hashes. Packing means independent routing pools, not one merged
  pool. The packed comparison uses `[128,512,360,1000]` versus `[2000]`.
- Completed contexts check finite outputs/all gradients, fresh-gradient replay
  and repeated live allocations. Prior kernel oracle qualifications remain
  applicable to the unchanged runtime. The targeted fan-out follow-up also
  checks the independent quantized FP64 oracle.

Raw samples are retained. CSV flags identify >10% inter-round drift or >5%
sample relative MAD; these are reporting flags, not relaxed correctness gates
or runtime-promotion rules. Several WSL whole-layer timings are nonstationary;
small percentage differences there are not reliable optimization evidence.
After checking both native and matched-baseline round spread, 23/35 pass
the descriptive whole-layer flag and 23/35 the expert flag. Numerical/replay
validity and timing stationarity are deliberately separate. All 35 completed
contexts have zero measured postwarm live-allocation growth.

## What increasing tokens does—and does not—fix

Approximate 5090 light-search results, milliseconds:

| Tokens | Whole MXFP8 | Whole checkpoint BF16 | MXFP8 experts | BF16 experts | Standalone router |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.483 | 0.337 | 0.092 | 0.050 | 0.275 |
| 2,000 | 0.643 | 0.537 | 0.176 | 0.129 | 0.354 |
| 7,376 | 1.517 | 1.247 | 0.548 | 0.380 | 0.654 |
| 29,504 | 5.480 | 4.652 | 2.287 | 2.046 | 2.277 |

These are bounded first-hit schedules, **not the earlier deeply tuned winners**.
The 512-token 5090 point failed BF16 compiled-backward warmup and is excluded.
Increasing tokens reduces fixed cost per token, but does not remove the router,
make narrow GEMMs automatically efficient, or eliminate activation traffic.

At N=128, mean real rows per expert are only `N*K/E=16`. Storage pads to
32 rows and the selected 64-row gate tiles imply 4x modeled row-tile work.
The gate grid has 64 CTAs for 170 SMs. This is grid underfill plus padding,
not evidence that the GPU's advertised compute or bandwidth is exhausted.

The weight gradients have a different problem: large dense outputs but very
short useful reductions. In the 5070 N=128 trace, gate dW takes about 52 us
and down dW 26 us, versus 25 us for joint recompute and 9 us for gate dX.
At N=7376, joint recompute instead takes about 514 us, versus 138 us gate dW
and 74 us down dW. Compute-stage labels are assigned only when the isolated
trace exactly matches the selected plan's expected compute-launch count/order.
The implemented 128-value reduction tiles can imply 8x modeled gate-dW
reduction work at only 16 real rows/expert. This is an implementation tile
restriction, not a claim that the hardware requires that minimum.

## Hidden width is an important lever

5060, N=2000 / D=384 / E=32 / K=4, same X/router/shared initialization:

| Expert H | MXFP8 experts | Checkpoint BF16 experts | BF16/MXFP8 |
| ---: | ---: | ---: | ---: |
| 64 | 0.608 ms | 0.488 ms | 0.80x |
| 128 | 0.733 ms | 0.830 ms | 1.13x |
| 256 | 1.254 ms | 1.522 ms | 1.21x |

Schedules are independently selected within the same bounded budget; this is
not a fixed-schedule causal estimate of tensor-core throughput. Width alone
does not guarantee a win either: it changes several GEMM dimensions and
weight/preparation volume. All per-axis rows and selected schedules are in CSV.

The 5090 confirms that larger D is not a monotonic cure. At N=2000, the
expert-only BF16/MXFP8 ratios for D=256/384/512/768 are about
0.65/0.73/0.85/0.64. Increasing H=64/128/256/512 gives about
0.65/0.73/0.77/0.97 instead. These independently selected light-search
schedules must not be interpreted as an intrinsic hardware ordering.

At N=29504, **D=384/H=512 experts reach 1.23x**, while the complete layer
reaches only **1.09x** (7.733 versus 8.415 ms). With D=768/H=128, the
experts are still about 0.96x and the complete layer 0.89x. Merely increasing
tokens/model width while keeping the experts narrow does not reach the target.
In the large-H case, fused joint recompute is about 40% of expert GPU kernel
duration, versus explicit quantization about 6%: optimize that mixed stage,
not just standalone quantization.

The additional D=128 laptop probes at 2000 and 29504 tokens also fail to
establish a speedup, but their whole-layer timings are nonstationary. They
are preserved as descriptive data, not precise ratio or threshold evidence.

K/E changes are also schedule-dependent. At N=2000 on the 5090, K=2/4/8
gives roughly 0.66/0.73/0.63x expert ratios; E=16/32/64 gives
0.64/0.73/0.75x. More E can improve grid parallelism while worsening weight
traffic, rows/expert and routing cost. The 5070's K=8 experts instead reach
about 1.31x in the light sweep, demonstrating why a universal SKU-independent
threshold from these few points would be premature.

## A useful cost model, not a fitted predictor

Let N be tokens, D model width, H post-gating expert width, E experts and K
assignments per token. Ignoring padding and constant factors:

```text
expert GEMM work       ~ N*K*D*H
activation prep/moves  ~ N*K*(D+H)
weight preparation    ~ E*D*H

prep / GEMM ~ launch_cost/(N*K*D*H) + a*(1/D + 1/H) + b*E/(N*K)
```

Consequences:

- More tokens amortize launches and weight preparation, but the activation
  term does not shrink with N. Large batches cannot fix everything.
- At D=384,H=128, `D*H/(D+H)=96`; doubling D gives about 110 (+14%),
  whereas doubling H gives about 154 (+60%). This activation-traffic proxy
  explains why increasing H can help more; it is not a measured roofline.
- Increasing E at fixed N/K adds weight preparation and routing work while
  shrinking rows per expert. More independent CTAs can help underfill, so
  the net effect is not universally negative.
- Increasing K amortizes weight/router costs but also increases activation
  traffic and expert work. It is not a free reuse improvement.
- Sinkhorn transport also scales with N*E, router projection with N*D*E,
  and repair depends on the assignments. Routing is not purely fixed cost.

## The hardware bottleneck is not just quantization

At N=2000 on the 5090, the expert replay has 18 kernels and about 99% GPU
coverage. Explicit quantization accounts for roughly 12% of summed expert
kernel duration, gather/scatter 11%, and fused recompute/activation/quantization
18%; GEMMs and their fused epilogues account for most of the remainder.
These are diagnostic GPU-duration shares, not additions to event timing.

The complete layer has 166 kernels, 118 below 3 microseconds in that trace,
and about 79% GPU coverage. Because it is a graph replay, the uncovered
span cannot simply be blamed on Python launching each internal kernel.
Routing child graphs and dependencies need separate treatment.

Kernel count alone also needs correction: in the separate 5070 K=8 capture,
native MXFP8 experts have **18** kernels versus **24** for checkpointed BF16.
Complete-layer counts are 171 versus 166. Thus quantization does not imply
that the fused MXFP8 path always launches more expert kernels. Efficiency,
traffic, fusion elsewhere in the graph and dependency structure matter.

Selective 5070 Nsight Compute probes used the real saved route, without
changing clocks, and left cache control disabled:

| Joint recompute probe | Active-warps occupancy | Tensor-pipe active | Registers/thread | L2 bytes | DRAM read + write |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=128 | 10.38% | 4.37% | 66 | 6.34 MB | 0.051 MB |
| N=7376 | 10.40% | 5.53% | 123 | 120.14 MB | 28.86 MB |

The larger selected recompute schedule uses about 68 KB SMEM per CTA,
limiting residency to one CTA on this device. It has enough grid work, but
still very low active warps/tensor activity. Observed large-probe traffic is
about 50 GB/s DRAM and 207 GB/s L2 in counter replay; these are not evidence
of peak-memory saturation. Tensor-pipe activity is not a peak-FLOP percentage.

Quantization-only portfolio probes reach roughly 81-92% active-warps occupancy
and 0.5-1.0 TB/s observed L2 traffic rate. That does not establish converter
saturation or optimality. DRAM counters were absent in those first portfolio
captures, so no DRAM bandwidth or cache-hit claim is made for them. Mixed
recompute/activation/quantization is intentionally not labeled pure GEMM or
pure quantization.
Local-memory spill counters were not collected; high register use is not
presented as proof of spills. The residency diagnosis uses measured active
warps together with the selected CTA's SMEM footprint.

## Packing adds bounded dummy work

For the four independent lengths totaling 2000 tokens, actual per-expert
physical slots sum to 288, versus 250 real assignments per expert. The
GPU-boundary-agnostic static bound is 352; the selected 64-row tiles round
this to 384 modeled rows. Thus storage work amplification is 1.408x and
gate tile-work amplification is 1.536x, before other tile dimensions.

The static bound is necessary for allocation without CPU synchronization;
**it need not force arithmetic on every unused tile**. A useful next design
is compiler-visible static allocation plus a device active-prefix operand
that skips globally unused CTA/K tiles. This must preserve per-sequence
32-value scale groups, independent pools, zero tails and backward semantics.
It is a proposed kernel change, not implemented by this profiling study.

## Search headroom versus structural limits

On the 5060, holding router/shared schedules fixed and seeding from the
light winner, an at-most-48-trial / 90-second expert refinement evaluated
45 trials at N=2000 and 42 at N=7376. Observed expert times improved from
0.733 to 0.595 ms and 2.861 to 2.387 ms. These are cross-run observations;
the larger case's BF16 control also moved, so the entire raw delta must not
be attributed to the selected config without a paired confirmation.

A fresh-process **refined / light / light / refined** confirmation at N=2000
then measured median expert times **0.658 versus 0.783 ms**: about **16% less
expert latency** for the refined config. The corresponding BF16 expert controls
were 0.856 versus 0.858 ms (about 0.25% difference). Complete-layer medians
were 1.633 versus 1.743 ms, about 6.4% less native latency; whole-layer laptop
variability warrants more caution than the expert result. These are medians
of the two process medians per policy, not extra autotuning trials. Exact
routes, parameters and inputs were preserved, with no further tuning in ABBA.

This is enough to reject the claim that the light search found a hardware
limit. It is not evidence that more generic trials alone will solve the
router, padding or low-residency recompute problems.

## Priorities

1. Small-pool routing: fewer device launches/graph dependencies, with
   shape-appropriate fused saved-state Sinkhorn schedules. Do not change
   routing/repair budgets or drop assignments to make a timing look better.
2. Recompute schedules: smaller SMEM/register tile footprints and alternate
   producer/consumer decomposition, selected by the autotuner. Do not
   manually cap registers or assume deeper pipelines always help.
   Tiny expert pools additionally need short-reduction dW alternatives;
   their bottleneck is not the same as long-token recompute.
3. Packed execution: skip unused global-capacity tails on device while
   retaining compiler-visible allocations and independent scale groups.
4. First-hit selection: protect genuinely distinct low-footprint and reuse
   families early; use phase/shape features rather than treating 12 trials
   as a universal quality guarantee. Keep stronger effort levels available.
5. Preparation/movement: retain fused dY and optional combined X preparation,
   but rank further fusion by whole-layer contribution, not kernel count alone.

## Limitations and reproduction

The first three-consecutive-capture WSL probe failed during BF16 profiler
injection after ordinary checks/timing completed. Fresh-process single BF16
captures succeeded. A later paired probe was nonstationary and also faulted
in its BF16 capture; it is not promoted as timing evidence. Hardware counter
injection around the router raised a repair-budget assertion; the successful
expert counter probes instead load the identical real route prepared in an
uninjected process. Whole-model NCU replay is not qualified.

The 5090 512-token compiled BF16 warmup failure remains an excluded point,
not a numerical/performance pass. None of these failures was worked around
by increasing repair rounds, relaxing numerics or changing GPU settings.

Artifacts: `autotune_reports/moe_amortization_scaling_20261118/`, backed by
Storage4. Per-context JSON, raw samples, selected configs, residual tuning
records, trace JSON, NCU reports/CSV, source snapshots, environment and
power/clock telemetry are retained. The ordinary sweep never runs concurrently
with another timing/counter workload on the same GPU.

- [Main 35-context CSV](../autotune_reports/moe_amortization_scaling_20261118/analysis/summary.csv)
- [Kernel groups, per-GEMM stages and raw timing identities](../autotune_reports/moe_amortization_scaling_20261118/analysis/summary.json)
- [Refinement and ABBA control CSV](../autotune_reports/moe_amortization_scaling_20261118/controls_analysis/summary.csv)

The focused CPU benchmark/analysis suite passed 36 tests; the generated API
reference and whitespace checks passed. This is profiling-tool validation,
not a claim of a new whole-library GPU qualification. Study GPU processes
finished; unrelated workloads subsequently present on the 5090 were left alone.

```bash
python -m benchmarks.profile_moe_amortization --source . \
  --output autotune_reports/moe_scaling --max-autotune \
  --contexts tokens_2000 tokens_128 tokens_7376 width_768 hidden_256 experts_64

python -m benchmarks.analyze_moe_amortization <gpu1-sweep> <gpu2-sweep> \
  --output autotune_reports/moe_scaling_analysis
```

For the optional counter checks, use `tools/profile_moe_amortization_counters.sh`
only after the timing sweep finishes. Exact executed commands are preserved
with each result. This study is not a transformer convergence or full-library
release qualification.
