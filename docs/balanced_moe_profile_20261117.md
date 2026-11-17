# BalancedMoE: launch, traffic, reuse and token-scaling diagnosis

The wide expert path is primarily a **tensor-traffic problem**, not a lack of
CUDA-core or tensor-core instructions. The narrow public API also has a **host
dispatch/launch problem**. More buffering, a different raster, or fewer quantizer CTAs is not a
universal solution. Joint backward preparation has a measured token-dependent
crossover and should remain an autotuner choice.

This is a diagnosis of the current MXFP8 implementation, not a production
kernel change or a claim that every design beats BF16.

## Scope and measurement contract

- Source digest: `2d986d94926d9d097961ff6455b8121b33bc0c09b2342e1d57a0596c5b98dbe0`.
- RTX 5090 (170 SMs), RTX 5070 Laptop (36 SMs), RTX 5060 Laptop (26 SMs).
  Torch 2.13.0+cu132 and CuTe DSL 4.7.0 throughout; separate compatible
  environments and isolated source snapshots, not edits to other projects.
- Four designs from `benchmarks/balanced_moe_cases.json`: D/E/K/H of
  384/24/6/128, 384/32/4/128, 512/64/8/128, 768/48/6/256; one shared SwiGLU.
  Token counts: 128, 512, 2,000, 8,192 and 32,768. Frozen inherited schedules
  separate scaling effects from a different autotuner winner.
- Parameters/input are BF16; GEMMs/hidden use MXFP8, activation/gating and
  mixture accumulation retain FP32. All required gradients are requested.
- Expert attribution freezes real router outputs. Complete-layer comparisons
  use the same CuTe router and weights; the reference is compiled batched BF16,
  not Python per-expert loops. Native recomputes; BF16 saves intermediates.
  This is **not** a matched-memory checkpoint benchmark.
- Ordinary GPU timing uses warmed manual CUDA graphs, alternating AB/BA order,
  three rounds and nine samples. Compilation, search, warmup and capture are
  excluded. An optional 128 MiB overwrite before the event boundary probes
  cache sensitivity; it is not a guarantee of an entirely cold cache or a
  rotating-input workload. Host-call timing is recorded separately.
- Systems and Compute run in separate processes, never alongside ordinary
  timing on the same GPU. Instrumented durations below are diagnostic, not
  the headline performance measurements.

Full independent FP64 quantized operational references check output, input,
both weights and route-weight gradients through 2,000 tokens. At larger sizes,
bounded-memory independent references check the full token reduction for the
first/last expert's two weight gradients, plus finiteness of all results.
Structural A/Bs additionally compare **all** outputs/gradients with the
qualified frozen result before and after repeated fast replay. These are
implementation-error checks, not error against an unquantized model or learning
convergence evidence.

Shape retargeting is explicit in each JSON: aligned TMA outputs can require
larger physical expert capacity; shared-MLP dW split counts must change when
the token reduction changes. The latter fixes benchmark legality, not the
production algorithm. Neither change is silently called an identical schedule.

## Ordinary performance, correctness and memory

**46 complete composed timing contexts:** all 20 design/token combinations on
both 5090 and 5060; five wide-token contexts plus the narrow 2K context on 5070.
Six earlier expert-only 5070 contexts are also retained, separately by path.
No capacity OOM prevented the main 46-context matrix from finishing.

Wide N=2,000 comparison, median milliseconds:

| GPU | Expert MXFP8 | Expert BF16 | Whole MXFP8 | Whole BF16 |
|---|---:|---:|---:|---:|
| 5090 | 0.482 | 0.594 | 1.288 | 1.383 |
| 5070 Laptop | 2.984 | 3.268 | 4.428 | 5.422 |
| 5060 Laptop | 3.198 | 3.466 | 5.198 | 5.508 |

5090 wide token scaling, median milliseconds:

| Tokens | Expert MXFP8 | Expert BF16 | Whole MXFP8 | Whole BF16 |
|---|---:|---:|---:|---:|
| 128 | 0.334 | 0.101 | 0.729 | 0.480 |
| 512 | 0.328 | 0.179 | 0.793 | 0.603 |
| 2,000 | 0.482 | 0.594 | 1.288 | 1.383 |
| 8,192 | 1.707 | 1.625 | 3.606 | 3.433 |
| 32,768 | 6.972 | 7.222 | 14.780 | 14.322 |

This reproduces the earlier roughly 1.24x expert / 1.07x whole-layer 2K win;
it does **not** demonstrate a universal speedup. The inherited wide schedule
pads logical capacity 16 to 128 at N=128 (8x) and 64 to 128 at N=512 (2x).
A separate 16-trial small-shape search did not escape the padded basin: about
0.318 ms expert F+B versus the frozen 0.334 ms and BF16 0.101 ms. That is a
limitation of this starting policy/search budget, not proof that the full
small-expert design space is exhausted.

Laptop medians are noticeably less stable. For wide expert F+B, across-round
range is about 2–22% on the 5070 and 4–11% on the 5060, versus 0.2–1.5% on the
5090. Other laptop phases fluctuate more. Each JSON retains samples, MAD,
round order, wall timestamps and the timing campaigns' power/clock telemetry.
Do not promote 1–2% laptop differences into dispatch rules, or treat a host
measurement below an earlier graph median as a causal speedup.

The maximum full-oracle implementation NRMS in the main matrix is about
1.82e-4, below the output/gradient gates. All 46 contexts have **zero live
allocated-memory growth over eight repeated expert steps**. The largest measured
temporary expert peak is about **1,292 MiB**; that is not whole-model peak or
allocator-reserved VRAM, and this short check does not prove an indefinite
training run cannot leak.

## Every expert launch: wide 2K case on the 5070

N=2,000, D=768, E=48, K=6, H=256. Logical expert capacity 250, physical 256.
One fresh Nsight Systems capture contains three iterations: **5 forward +
14 backward launches per iteration**, all correlated to their helper range.
The table gives median instrumented microseconds across those three iterations.

| Phase | Launch | us |
|---|---|---:|
| Forward | Gather routed BF16 X | 58.9 |
| Forward | Quantize X, gate/up W, down W | 299.3 |
| Forward | Gate/up GEMM, FP32 SiLU/gating, MXFP8 hidden | 134.0 |
| Forward | Down GEMM | 61.7 |
| Forward | FP32 weighted combine | 47.6 |
| Backward | Regather X | 57.5 |
| Backward | Requantize forward operands | 300.9 |
| Backward | Recompute gate/up and packed hidden | 134.0 |
| Backward | Recompute down output for route-weight gradient | 60.3 |
| Backward | Slot dY and route-weight gradient | 140.5 |
| Backward | Decode hidden to BF16 | 37.6 |
| Backward | Prepare dY row/transpose, down-W transpose, hidden transpose | 198.2 |
| Backward | Down dX / hidden gradient | 69.6 |
| Backward | Down dW | 78.1 |
| Backward | Recompute gate/up derivative | 207.6 |
| Backward | Prepare derivative row/transpose, gate-W transpose, X transpose | 283.8 |
| Backward | Gate/up dX | 134.9 |
| Backward | Gate/up dW | 145.2 |
| Backward | Combine expert dX | 68.2 |

BF16 transpose accesses are logical layouts. Producing a new MXFP8 orientation
still costs a quantization/write pass because its reduction blocks/scales differ.

### How the shares change

| 5070 capture | Quant/preparation | Movement | Packed linear GEMMs | GLU/recomputed hidden | Gate derivative | Hidden decode | GPU coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| D768/H256, 2K tokens | 43.0% | 14.8% | 21.8% | 10.6% | 8.3% | 1.5% | 96.6% |
| D768/H256, 8K tokens | 31.6% | 24.0% | 22.4% | 10.7% | 9.4% | 1.9% | 99.4% |
| D384/H128, 2K tokens | 24.7% | 18.1% | 39.4% | 8.0% | 8.4% | 1.4% | 49.7% |

Shares are of summed GPU kernel duration; coverage is the interval union inside
the first-to-last GPU launch, not CPU wall time. These helper-annotated traces
add CPU work, especially visible in narrow kernels. Compilation is outside both.

Fresh warmed **outer-NVTX-only** captures qualify that annotation overhead:

| 5070 path | GPU coverage | Kernel sum/iteration | First-to-last span/iteration |
|---|---:|---:|---:|
| Narrow 2K, raw expert core | 87.9% | 0.795 ms | 0.905 ms |
| Narrow 2K, compiled public expert API | 66.9% | 0.891 ms | 1.330 ms |
| Wide 2K, compiled complete layer | 83.8% | 4.330 ms | 5.165 ms |

Thus the detailed trace's 50% coverage overstates the raw core's launch gaps.
The **public binding/composed path** still has a real gap, supported separately
by uninstrumented host-call timing below. All captures retain their scope and
annotation mode; no profiler-derived duration is substituted for event timing.

## Hardware counters: why quantization costs so much

Actual routes/inputs were saved in an ordinary process, then replayed unchanged
under Compute. This avoids executing conditional routing graphs under Compute
injection. Application replay preserves the real producer/consumer sequence;
caches/clocks are not forcibly changed. Missing counters are unknown, not zero.

Initial wide 2K forward capture:

| Launch | DRAM read+write | DRAM GB/s | L2 hit | Active-warp occupancy | Tensor-pipe elapsed activity | Shared/CTA |
|---|---:|---:|---:|---:|---:|---:|
| Gather | 9.8 MB | 149 | 86.4% | 83.8% | 0% | 1 KiB |
| Quantization portfolio | 105.7 MB | 322 | 6.6% | 85.2% | 0% | 1 KiB |
| Gate/up GLU | 43.9 MB | 271 | 78.7% | 36.3% | 40.0% | 43 KiB |
| Down GEMM | 19.4 MB | 227 | 64.5% | 18.5% | 36.9% | 68 KiB |
| Combine | 6.8 MB | 106 | 99.3% | 91.1% | 0% | 1 KiB |

The separately measured 5070 streaming-add reference is **320 GB/s**. This is
a measured reference, not a guaranteed physical peak; cross-run numbers can
slightly exceed it. The quantizer is close to this reference and reports about
91% of the profiler's sustained DRAM throughput. This is evidence of bandwidth
pressure, not a hypothesis based on utilization or marketing specifications.

The 2K forward/backward phase totals are approximately **186/627 MB** of DRAM
traffic. At 8K they become **543/1,813 MB**. These are measured traffic including
cache behavior, not sums of tensor sizes. The 2K backward's three preparation
launches achieve roughly 306–326 GB/s; movement also becomes important as N grows.
The narrow 2K design moves only about **24/113 MB**, and its first portfolio has
about 40% L2 hits instead of 6.6%. Its bottleneck mixture is different.

No local-load or local-store sectors were measured in these expert captures:
there is no evidence that register spilling explains the slowdown. Register
count alone would not establish a spill problem.
Some reported stall ratios exceed 100% in this tool/stack. The CSV preserves and
flags them; they are not used as normalized probabilities or bottleneck proof.

A second capture after at least 300 ms of repeated workload warmup confirms
the bandwidth conclusion: first-portfolio throughput is about **328 GB/s** in
both forward and backward, and backward's other preparation launches achieve
about **319/326 GB/s**. The low down-GEMM occupancy remains roughly **18.5%**.
These warmed reports are separate from the initial captures, not overwrites.

### Reuse we have, and reuse we do not have

- Quantization is shared across output CTAs **within each expert and required
  orientation**; it is not repeated independently by every GEMM tile.
- X is still gathered and quantized for its K expert assignments. We do not yet
  quantize original-token X once globally and reuse its packed rows/scales
  across those assignments. This is a genuine remaining reuse opportunity.
- Packed forward operands are not retained for backward: recomputation repeats
  their preparation. The same orientation is reused within backward where legal.
- GLU/down consumer L2 hit rates are much better than the streaming quantizer's,
  but are not 100%. L2 traffic/hit metrics include multiple operand/output
  accesses and cannot assign a separate hit rate to W from these counters alone.
- The wide case has about 56.6 MB of BF16 routed weights, larger than the laptop's
  32 MiB L2 before gathered activations/output traffic are considered. Making
  the entire preparation portfolio nearly 100% L2-resident is not realistic.
  Packed weights alone are about 27.8 MiB including scales, but packed X adds
  about 9.3 MiB at 2K, before hidden/output traffic. Packing does not make the
  complete producer/consumer working set fit either.
- Current `same_a`/`same_b`/serpentine work ordering changes cache locality;
  it does not by itself prove that an operand remains in SMEM across tiles.

H=128 also does not mean a whole expert's weights fit on chip. Already at
D=384, the gate/up/down packed matrices plus scales total about 148.5 KiB per
expert, exceeding the 99 KiB per-CTA shared-memory budget before X/hidden or
pipeline storage. Hidden tiles can still remain on chip; a fully resident
expert needs a partitioned/register/distributed design, not just a smaller H.

## Buffering, occupancy and locality A/Bs

The down GEMM's 68 KiB shared allocation permits only one 288-thread CTA per
102,400-byte SM. Nine active warps out of a 48-warp limit explain the measured
roughly 18.5% occupancy. GLU's 43 KiB permits two such CTAs, consistent with
roughly 36% measured occupancy. Tensor activity is not zero, but these are not
measurements of peak tensor FLOP utilization.

Existing schedule options were varied, not hand-selected register values:
1/2/3/4 buffers, producer/consumer scale roles, cooperative versus TMA linear
transport, independent stage changes, raster/swizzle, four multi-tile locality
orders, quantizer grids and joint backward. Illegal four-buffer cases were
rejected for shared-memory capacity before compilation.

5090 exploratory whole-expert F+B results:

- At 2K, a locality/depth change improves about 1–1.4%; this is small and not a
  robust new dispatch rule from one sweep. Cooperative linear transport is
  substantially slower than the inherited TMA schedule.
- At 8K, one down buffer improves about 1.7%, while joint backward improves
  about 8.4%. Three buffers is not automatically better than one.
- At D384/H128, 8K, the best changes in this sweep are only about 0.5%.
- Globally capping the portfolio/expert quantizer grids is consistently slower
  in the tested wide cases. Fewer CTAs is not synonymous with less traffic.

Three alternating confirmation rounds on the 5090, median microseconds:

| Tokens | Separate preparation | Joint backward | Speedup |
|---|---:|---:|---:|
| 2,000 | 483.3 | 500.6 | 0.965x |
| 8,192 | 1,702.5 | 1,572.4 | 1.083x |
| 32,768 | 6,963.1 | 6,528.1 | 1.067x |

These are same-policy expert F+B objectives with correctness checked after
fast replay. They are not whole-MoE speedups and have not been promoted into
runtime winners. Joint preparation changes the number of passes/launches,
not the required MXFP8 backward precision or FP32 activation recipe.

The same three-round 8K A/B on the laptops:

| GPU | Separate preparation | Joint backward | Speedup |
|---|---:|---:|---:|
| 5060 Laptop | 10.130 ms | 9.462 ms | 1.071x |
| 5070 Laptop | 8.949 ms | 8.186 ms | 1.093x |

This is cross-device evidence for the **8K regime**, not a tested crossover
threshold on every GPU. All-gradient checks pass before/after repeated replay.

## Graph replay is not the compiled public API

The public expert forward/backward ops currently retain
`torch.Tag.cudagraph_unsafe` because they carry an immutable CPU selection token.
`fullgraph=True` does not remove their Python launch work or make the internal
19 launches an automatically replayed graph. Manual graph GPU timing must not
be sold as ordinary compiled-call latency. The JSON records both separately.

| Ordinary measurement | Public GPU graph | Compiled public host call |
|---|---:|---:|
| 5090, wide 2K experts | 0.480 ms | 1.005 ms |
| 5090, wide 2K complete layer | 1.288 ms | 2.536 ms |
| 5070, narrow 2K experts | 0.935 ms | 1.398 ms |

Host samples synchronize at the boundary and include dispatcher/autograd/Python
work. Graph timing excludes that repeated host work. These are separate sample
sets, not exact additive decomposition; the low-annotation timeline corroborates
the public API gap.

The detailed whole-layer captures contain **206 native versus 193 BF16 GPU
launches per step**, including about 73 router-classified launches in both.
The whole layer is not merely the 19 expert launches: shared FFN, router,
projection-gradient bridges and framework work remain. Conditional-graph nodes
without a helper correlation stay explicitly unmapped. Standalone profiler
captures do not preserve the ordinary BF16/native timing ranking on this laptop;
only the alternating uninstrumented measurements support speedup comparisons.

A focused 5090 CPU profile also distinguishes the raw and registered paths:
20 instrumented narrow expert steps take about 9 ms in the raw core versus
44 ms through the compiled public path. Much of the additional attributed time
is in `torch.library`'s backend wrapper and autograd/runtime dispatch. The token
writer/reader together account for only about **0.54 ms across all 20 steps**;
alias checking about **0.09 ms**. Neither a slow CPU token copy nor the alias
check alone explains the gap. This is diagnostic Python attribution, not
unobstructed latency or an exact causal split of opaque C++ runtime overhead.
No eager CUDA pointwise arithmetic was found in the expert launchers.

The compiled BF16 reference and native benchmarks explicitly disable automatic
Inductor cudagraphs for attribution, then use the same outer manual graph for
GPU timing. Public API and complete-layer Systems captures separately examine
the real direct-launch behavior. A future graph-safe execution plan must retain
correct first-hit selection, stream-safe scratch, immutable backward policy and
bounded allocation; removing a safety tag alone is not a fix.

## Operational findings and reproducibility

- 5070 counter access initially failed with `ERR_NVGPUCTRPERM`; a host setting
  change/update temporarily removed WSL GPU access. Completed contexts were
  preserved. CUDA and real counter collection were verified after reboot.
- Compute injection into conditional routing hit a device assertion; actual
  saved-route fixtures isolate unchanged expert execution instead.
- Multi-process kernel replay stalled at GLU. Its partial report/log are kept;
  single-process compilation and bounded application replay completed.
- An initial whole Torch-profiler capture after manual graph benchmarks hit
  illegal memory access. A fresh trace-only process and subsequent separate
  Systems captures are used instead. This mixed-execution/instrumentation
  failure is retained, not declared a solved production-lifetime bug.
- The work GPU's `grounding-dino-school.service` was stopped with authorization;
  unrelated CPU services were left alone. Work-host commands/transfers reuse a
  single live SSH master, with proxy fallback disabled for subsequent commands.
  The detector service was restored after testing; the SSH master was then
  closed, rather than left reconnecting or idle indefinitely.

Raw artifacts, exact fixtures, commands/configs, environment reports, journals,
telemetry and exports are under `autotune_reports/balanced_moe_profile_20261117/`
on the existing Storage4-backed symlink. Early exploratory captures and later
warmed confirmations are separate directories. Keep the raw reports/fixtures
for replay; do not replace them with only these tables.

```bash
# Run from an isolated current-source snapshot, one GPU workload at a time.
bash tools/profile_moe_campaign.sh results/token-matrix --whole-layer \
  --seed-studies seeds/prepared-final-current-source-wide.json \
                 seeds/prepared-final-tma-shapes.json
bash tools/profile_moe_captures.sh results/hardware-captures
bash tools/profile_moe_confirm.sh results/warmed-confirmation
bash tools/profile_moe_confirm.sh results/minimal-confirmation --minimal-only
bash tools/profile_moe_followups.sh results/structural-variants
python -m benchmarks.analyze_moe_profile \
  autotune_reports/balanced_moe_profile_20261117 --output /tmp/moe-profile-tables
python -m pytest -q tests/test_moe_profile_analysis.py
```

The seeds are the frozen 2026-11-15 study results, not portable dispatch defaults.
Driver/source archives accompany the evidence; `--config` accepts a selected
profiling config, while `--tune-trials 16` runs the existing tuner before timing.
Small-budget tuning is not claimed to recover an exhaustive winner.

CSV exports include ordinary `timings.csv`, per-launch `kernels.csv`, raw-metric
`counters.csv` and per-round `variants.csv`; `trace-summary.json` records coverage,
unmapped launches and parsing failures. Six CPU analysis tests cover correlation,
overlap, units/missing metrics, padding and non-mutating partition adaptation.

## Next work, ranked by measured bottleneck

1. **Wide cases: eliminate passes/bytes.** Preserve joint preparation as a
   shape/SKU-dependent choice. Investigate global-token X quantization with
   packed-row reuse across routed assignments and fusing gather/preparation or
   derivative/orientation emission. Measure complete F+B and memory, not just
   the reduced launch count. Packed forward-state retention would be a separate
   memory/recomputation policy, not a free change to the current contract.
2. **Narrow/small cases: graph-safe composed dispatch and less padding.** Do not
   ask users to allocate scratch/manage streams. A bound execution plan should
   let user-side compilation/replay stitch the layer normally. Keep on-chip
   fine-expert families in deeper searches; preparation-to-GMEM is not assumed
   optimal for H=128.
3. **Bias, do not hard-code, schedule proposals.** Use physical-capacity waste,
   expert weight working set versus L2, stage SMEM/residency and token reduction
   length as features. Keep pipeline/locality/transport choices searchable.
   This study does not establish a universal tile or buffer-depth rule.
4. **Always retain the common router/shared path.** An expert-only win can be
   diluted or reversed by complete-layer costs. No precision changes, extra
   saved forward tensors, or weakened baseline should be hidden in a speedup.

A particularly useful reuse experiment is expert-group preparation/consumption:
try 2/4/8 experts per cache-sized chunk, with quantization and GEMMs close in
time. This could avoid evicting early experts' packed operands before their
consumers start, but adds launches and can underfill a 170-SM GPU. Treat group
size, token tile and buffering as joint autotuner choices, not a fixed rule.
