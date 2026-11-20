# RTX 5090: packed BalancedMoE overhead study

## Outcome

On the two controlled complete-layer cases, native MXFP8 forward/backward
latency improved **18.39%** (1.478 to 1.206 ms) and **23.05%** (5.559 to
4.277 ms). Against checkpointed BF16 with the same tuned router, the final
ratios are **0.9994x** and **1.0184x**, not a 1.2x advantage. These are
compiled, warmed graph measurements, not training/convergence results.

The implemented search choices are fused weighted-dY quantization,
optionally combined X-transpose preparation, sharded Sinkhorn column
reductions, and expert-major quantile keys. Existing defaults remain
available; bounded router searches now guarantee trials of both new
layout families rather than spending their structural openings on legacy
aliases. Allocations remain compiler-visible and callers manage neither
scratch buffers nor streams. The row-broadcast affinity helper remains a
private experiment because tiny-shape performance can regress.

Cross-device component evidence is stored alongside the 5090 artifacts:

- **5060:** eight preparation cases passed byte-exact comparisons;
  fused dY was 1.407-2.166x faster for the measured preparation component.
  Combined X/dY preparation helped some cases but regressed long/wide
  cases, so it remains optional. Fullgraph ragged/empty-sample backward,
  independent FP64 checks, first-hit tuning, disk-cache reuse and explicit
  budget refinement passed.
- **5070:** sharded column reductions were 1.76-3.02x faster across eight
  component cases; expert-major quantile keys were 1.087-1.311x faster
  including key construction, with exact rank/tie/repair results. Nsight
  evidence supports less L2 traffic for the profiled column reduction;
  it does not establish whole-layer DRAM savings.
- **Local consolidation:** 190 CPU tests passed, 141 CUDA-only cases
  were skipped locally; the focused CUDA qualifications ran remotely.
  The generated API reference and whitespace checks also passed.

Raw evidence is under
`autotune_reports/balanced_moe_amortize_20261118/{expert_5060,routing_5070,acceptance_5090}`.
Component speedups must not be read as whole-model speedups.

## Frozen selected-winner baseline

The current `rtx/*.py` runtime initially matched the archived `h15` source
digest `bfb6fa23def59bbef323ad6e79109cb0d731ce5d68f2c02564f43dc226a58b66`.
These two cases use its actual selected expert, shared and routing schedules,
not inherited seeds. The narrow small-pack winner uses joint prepared
backward, `quantize_once` input and warp transpose. The wide large-pack
winner instead uses the indexed input transpose with independently split
down dW and a fused route-gradient epilogue.

- Both cases: D=384, E=32, K=4, shared H=256, seed 17391.
- Small: expert H=128, lengths `[512,2000,768,4096]`, 7,376 tokens,
  physical expert capacity 1,024.
- Large: expert H=192, lengths `[2048,8000,3072,16384]`, 29,504 tokens,
  physical expert capacity 3,808. Only routed H changes; the benchmark
  preserves the wider study's initialization and input hashes.
- RTX 5090 (SM120), driver 595.99.02, Torch 2.13.0+cu132, CuTe 4.7.
- Full layer and actual `.backward()` to every leaf with gradients reset;
  `torch.compile(fullgraph=True, dynamic=False)`, Inductor max autotune
  isolated in subprocesses, warmed outer CUDA Graph. Three rounds of nine
  ordinary CUDA-event samples; median of round medians. No optimizer.
- BF16 uses the same native router and checkpointed FFNs. Saved-state BF16
  is not the comparison. Timings exclude compile/tune/capture/warmup.

| Geometry | MXFP8 step | BF16 checkpoint step | BF16/MXFP8 | MXFP8 peak allocated | Postwarm live growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| H128 / 7,376 | 1.477680 ms | 1.442736 ms | 0.9764× | 378,407,424 B | 0 B |
| H192 / 29,504 | 5.558624 ms | 5.487232 ms | 0.9872× | 1,257,721,856 B | 0 B |

MXFP8 sample ranges were `[1.47510, 1.47979]` ms and
`[5.53613, 5.56512]` ms respectively. Every output/gradient was finite;
both fresh-gradient captured replays matched their uncaptured MXFP8 result
exactly. Independent quantized FP64 expert oracle worst relative L2 was
`7.377e-5` small and `1.269e-4` large across output, input adjoint, both
expert weight adjoints and the route-weight adjoint.

## Device work on the selected winners

The following are time sums for identified kernels in a separate single
warmed CUDA Graph replay trace, not extra timing phases to add to the
ordinary step. The small trace spans 1,495.35 µs with 155 kernel records
(1,309.98 µs total); the large trace spans 5,572.73 µs with 160 kernel
records (5,219.42 µs total). Conditional repair graph activity uses a child
stream; the gap between timestamp span and summed duration is not evidence
of CPU launch starvation. Nsight CLI/Python were unavailable, so there are
no hardware counter claims about HBM, L2, spills or tensor-pipe utilization.

| Trace contribution | H128 / 7,376 | H192 / 29,504 |
| --- | ---: | ---: |
| Sinkhorn column forward, five launches | 104.00 µs | 394.72 µs |
| Sinkhorn column backward, five launches | 180.67 µs | 415.17 µs |
| Hard-route column quantile, five launches | 54.46 µs | 642.76 µs |
| Small-affinity one-CTA scalar reduction | 122.69 µs | Already split: ~10 µs for all corresponding pointwise/reduction kernels |
| Expert `slot_dy` movement | 27.90 µs | 144.19 µs |
| Expert quantization portfolio with dY | 37.06 µs (also includes staged X transpose) | 120.00 µs (dY only) |
| Separate indexed X transpose | — | 64.26 µs |
| Expert joint hidden/derivative recompute | 101.28 µs | 618.43 µs |

For the large case, materializing the weighted BF16 `[E,C,D]` dY writes
93,585,408 logical bytes before both quantized orientations read it. Removing
that intermediate and both reads can avoid 280,756,224 logical buffer bytes;
that is **not** measured DRAM traffic and does not count the new kernel's
source reads and output writes. Router improvements save time in both
precisions, so a lower absolute native latency need not improve the ratio
against matched BF16. The column quantile is separate from Sinkhorn: a
Sinkhorn-only optimization cannot make its ~643 µs disappear.

## Compiler-visible router affinity experiment

Both learned scalars still use the same FP32 affine and `logsigmoid`.
Giving them logical `[N,1]` broadcast views exposes per-token expert sums
to autograd before the final scalar sums. It does not allocate an `[N,E]`
broadcast or register a custom operator. In the small case Inductor changed
the 122.688 µs one-CTA reduction into a 922-CTA 1.888 µs pass and two
final reductions totaling 2.560 µs. A 16-row chunk variant also parallelized
this reduction, but tied the row layout and requires divisibility.

| Geometry | Affinity transform | MXFP8 step | BF16 checkpoint step | BF16/MXFP8 |
| --- | --- | ---: | ---: | ---: |
| H128 / 7,376 | Row broadcast | 1.363040 ms | 1.324032 ms | 0.9714× |
| H128 / 7,376 | 16-row chunk probe | 1.362608 ms | 1.324104 ms | 0.9717× |
| H192 / 29,504 | Row broadcast | 5.531424 ms | 5.495136 ms | 0.9934× |

The row option reduced the small step by 7.76% (native) and 8.23%
(checkpointed BF16), and made no material difference on the large pack.
Both paired comparisons retained identical hard route metadata and exact
MXFP8 output, input gradient and all nonscalar parameter gradients. Only
`logit_scale` and `score_bias` gradients differed, as expected from the
FP32 sum association: row worst relative L2 `1.995e-6`, max absolute
`2.981e-8`; live-allocation growth remained zero. A separate independent
FP64 affine/adjoint test passed three CPU cases and two compiled 5090 CUDA
cases, including both learned scalar gradients and strided input/upstream.

The new `rtx/_routing_affinity.py` helper is not a default schedule change.
The current transport autotuner starts from detached scores, so it cannot
measure affinity layout. These are explicit full-layer A/Bs; they are
not evidence that every geometry should choose row broadcasting.
An isolated graph F+B probe at N={1,128,512,2000}, E={32,64} passed all
eight FP64 derivative and replay checks. Row broadcasting tied at N=1/E32
and N=128/E32, and regressed N=1/E64 from 2.24 to 3.07 µs. The other
isolated cases improved, e.g. 22.57→6.14 µs at N=2000/E32 and
41.08→6.14 µs at N=2000/E64. These kernel-scale results do not include
surrounding affinity projection fusion or routing. A fixed universal policy
is not selected based on these probes.

## Fused dY: fixed expert and router policy

Candidate CUDA source digest
`4bd66291f36949f694e03ca05d0388c6887348f08034d93b4f9c04c460aff599`.
The same frozen selected schedules were accepted with the materialized
backward as a source-drift control. The small winner's input retarget from
`quantize_once` to `indexed_transpose` is shown separately; the large
winner already uses the indexed input map. No other expert stage, router,
shared schedule or precision baseline changes in this table. All values
are ordinary CUDA-event complete layer F+B measurements.

| Geometry | dY/input policy | MXFP8 step | BF16 checkpoint step | MXFP8 peak allocated |
| --- | --- | ---: | ---: | ---: |
| H128 / 7,376 | Stock, quantize-once | 1.481264 ms | 1.443008 ms | 378,407,424 B |
| H128 / 7,376 | Indexed, materialized dY | 1.468688 ms | 1.442840 ms | 378,538,496 B |
| H128 / 7,376 | Indexed, fused dY | 1.444880 ms | 1.442760 ms | 361,761,280 B |
| H192 / 29,504 | Stock indexed, materialized dY | 5.554496 ms | 5.498656 ms | 1,257,721,856 B |
| H192 / 29,504 | Indexed, fused dY | 5.433632 ms | 5.496992 ms | 1,211,060,224 B |

Fused dY lowers the directly matched small native step by 1.62% and
the large one by 2.17%; the large ratio against checkpointed BF16 is
1.0117×. The large trace observed `RoutedDYQuant` at 138.50 µs replacing
the 144.19 µs movement and 120.00 µs two-orientation quantization launches.
The trace spans 5,449.51 µs across 159 kernel records. The ~125.7 µs trace
delta is consistent with the ordinary full-layer 120.9 µs improvement.

All five results matched the frozen hard metadata, output and every leaf
gradient exactly, passed independent quantized FP64 and fresh-gradient
replay checks, and had zero live-allocation growth. On the same candidate
snapshot, two full-expert FP64 checks on H192 and H384, and four compiled
packed-layer all-gradient checks with frozen differentiable routing, passed.
The full-layer fixed-policy benchmarks themselves compile the real router
and verify its optimizer-visible gradients, rather than freezing routing.

## Production tuner: sharded transport and expert preparation

On the same frozen candidate, the production autotuner was allowed 24
router and 32 expert trials per context, with up to 180 seconds per
search; shared-MLP config remained frozen. The experimental affinity hook
was not used. The tuned router (including any chosen column sharding) is
also used by checkpointed BF16. Any speedup versus the previous native
step is therefore distinct from the MXFP8/BF16 precision ratio.

| Geometry | Native step, before → after | BF16 checkpoint step, after | BF16/MXFP8 |
| --- | ---: | ---: | ---: |
| H128 / 7,376 | 1.477680 → 1.271816 ms | 1.272448 ms | 1.0005× |
| H192 / 29,504 | 5.558624 → 5.351520 ms | 5.433952 ms | 1.0154× |

The small search selected 32-way forward/8-way backward column sharding
and fused dY. Hard routes remained identical; the independent oracle,
all-leaf source comparison and replay gates passed. Worst per-leaf
relative L2 versus the old selected policy was `1.704e-3`. This contrasts
with the fixed-configuration, fused-dY-only experiment's bitwise parity.
Live allocation growth was zero and warm peak allocation was 361,761,280 B.
The large search selected four forward shards, the legacy backward column
schedule and fused dY. Hard routes again matched exactly, worst selected
per-leaf relative L2 was `1.271e-3`, and all oracle/replay checks passed.
Warm peak was 1,210,407,936 B with no live growth. Both searches evaluated
their full 24 router and 32 expert trial budgets.

## Production tuner: exact column quantile layout

A separate source snapshot adds the optional expert-major ordered-key
column quantile, while retaining row-major quantile as an alternative.
Its runtime source digest is
`23e16e311c7e1dbae6cb64d85281cbf92f6178c51e617854d8d24f7193dd36ab`.
On this snapshot the router received 20 trials seeded with the above
sharded-router winner; the selected expert and shared configurations
were held fixed and the affinity hook was off. Eleven targeted RTX 5090
CUDA cases passed exact ordered statistic, hard-route and all-gradient
checks, including ties, empty segments and streaming fallback.

| Geometry | Native step, shard-only → QB follow-up | BF16 checkpoint step, after | BF16/MXFP8 |
| --- | ---: | ---: | ---: |
| H128 / 7,376 | 1.271816 → 1.240685 ms | 1.240096 ms | 0.9995× |
| H192 / 29,504 | 5.351520 → 4.794432 ms | 4.878336 ms | 1.0175× |

The large winner selected expert-major keys plus four forward Sinkhorn
shards and fused dY. It lowers native latency by 13.75% versus the initial
frozen run, compared against the unchanged-work checkpointed BF16 path
with its same faster router. It also changed backward/quotas thread
geometry versus the shard-only winner; the 0.557 ms delta is a selected-
policy result, not an isolated column-key kernel claim. Hard route IDs
remained exact, the maximum all-leaf source relative L2 was `1.271e-3`,
all oracle/replay checks passed, and live allocation growth was zero.
The small winner retained row-major quantile and selected 16/8-way
forward/backward shards. Native time improved 16.04% versus the frozen
baseline; parity with BF16 remains effectively unchanged in this geometry.
All hard routes, oracle, full gradients and replay gates passed, with
worst all-leaf relative L2 `1.704e-3` and zero live growth.

### Controlled QB-only comparison

Holding the selected large router, expert and shared configs fixed, one
field toggled row-major vs expert-major keys. Four fresh processes ran
keys / row / row / keys; each retained three rounds of nine samples per
precision. The following are medians across the six round medians per
layout. Both layouts' native hard routes, output and every leaf gradient
matched the newly saved keys fixture bitwise. Fresh-gradient replay gates
passed independently in both precisions. Each native layout used
1,211,060,224 B warm peak allocated memory with zero live growth.

| Column quantile layout | MXFP8 step | BF16 checkpoint step |
| --- | ---: | ---: |
| Row-major | 5.177696 ms | 5.258816 ms |
| Expert-major keys | 4.794888 ms | 4.873856 ms |

The isolated layout change saves 0.3828 ms / 7.39% on MXFP8 and
0.3850 ms / 7.32% on checkpointed BF16. A separate MXFP8 replay trace
shows five row-major quantiles totaling 645.02 µs versus 247.97 µs
for five expert-major quantiles plus 10.88 µs for five key transposes.
Complete trace spans were 5,203.81 µs (164 kernel records) and
4,816.26 µs (169). This demonstrates less device work despite five extra
launches, and attributes most of the full-layer delta to the exact key
layout, not a timer or scheduler artifact.

## Final source: protected router-family coverage

The earlier ordinary tuner often spent both protected structural openings
on legacy variants. In particular, the first large shard-only search did
not time any 32-way structural seed. The final source keeps the supplied
winner as the initial trial, then protects two genuinely competing column
layout proposals with the same sharded schedule and cache capacity. Both
are ordinary correctness/timing trials, not forced winners. Cache capacity
is based on the maximum scheduled sequence length. The global scheduler
and its budget are unchanged.

Final runtime digest:
`6a3c7a776a921f8e6a9886a3efe62ea99230be3fc7627b0026a34e9ae3243155`.
The protected-opening CPU suite passed five tests. Final timed router
searches use 20 trials with the previously qualified fused-dY expert and
shared schedules fixed and flat affinity. The source also contains the
separately qualified fused-all alternative; this experiment does not select
or benchmark that alternative.

| Geometry | Frozen native → final native | Final BF16 checkpoint | BF16/MXFP8 |
| --- | ---: | ---: | ---: |
| H128 / 7,376 | 1.477680 → 1.205882 ms | 1.205158 ms | 0.9994× |
| H192 / 29,504 | 5.558624 → 4.277296 ms | 4.356048 ms | 1.0184× |

The final large winner chooses expert-major keys, cache 64 and 32-way
forward/backward column sharding. It selects 1/4 column groups rather than
the 8/8 proposed initial geometry; the protected trials did not pin its
schedule. Native absolute latency improves by 23.05% versus the frozen
baseline, without meeting a 1.2× speedup over the matched BF16 baseline.
Hard routes match, the quantized FP64 oracle maximum relative L2 is
`1.255e-4`, selected-policy all-leaf source relative L2 is at most
`1.209e-3`, fresh replay passes, and live growth is zero.
The small winner retains row-major quantile, cache 16 and 32/32-way
sharding, improving absolute native latency by 18.39%. It passes exact
hard routes, oracle, all-leaf and replay gates, with zero live growth.
All 20 trials in each context passed: small measured 13 row-major and
7 expert-major cases; large measured 14 row-major and 6 expert-major
cases. The protected row/key proposals were the second and third
evaluations, after the inherited incumbent. Their large raw router
forward plus score-adjoint timings were 1.1395 / 1.1190 ms, versus
1.9765 ms for the inherited starting router. This component timing is
not the whole-model latency.

## Reproduction

Each benchmark publishes an exact `.command` file, ordinary samples, memory
and validation JSON, frozen config sidecars and separate Chrome trace JSON.
The raw directories on the 5090 are
`/tmp/rtx-moe12x-20261118.LU2siN/results/5090-amortize-baseline.JJlnU6/`
and `/tmp/rtx-moe12x-20261118.LU2siN/results/5090-affinity-probe.6mWPvW/`.
The latter also preserves the isolated probe, both launch scripts and
Inductor-generated code from `TORCH_LOGS=output_code`. The baseline's
`*.reference.pt` contains exact hard metadata and all MXFP8 leaf gradients
for fixed-policy parity; it is not a replacement for the FP64 oracle.
The corresponding Storage4-backed bundles are under
`autotune_reports/balanced_moe_amortize_20261118/acceptance_5090/`:
`5090-baseline-affinity-evidence.tar.gz` and `5090-experts-fixed-evidence.tar.gz`.
The latter includes the frozen candidate source and test logs.
`5090-shard-retune-evidence.tar.gz` contains the production sharded-route
and expert tuning commands, selected configs and full trial histories.
`5090-qb-retune-affinity-micro-evidence.tar.gz` keeps the first QB source,
its exact tests and selection evidence, and the tiny affinity probe.
`5090-qb-only-abba-evidence.tar.gz` keeps the controlled layout commands,
paired samples, exact comparison fixture, and the two full CUDA traces.
`5090-final-protected-coverage-evidence.tar.gz` keeps the final runtime
source snapshot, final benchmarks, trial histories and exact commands.
The archived initial source is in `autotune_reports/balanced_moe_h15_20261118/`
as `rtx-moe-h15-source-20261118.tar.gz`. Both final runs completed; the
5090 was idle afterward (68 MiB, 0% GPU utilization, no compute processes).
