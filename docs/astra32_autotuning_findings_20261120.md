# What did another 24 real trials buy?

Follow-up: the flat transport family is now integrated and qualified on two
fresh shapes; see [production integration](autotune_cycle3_20261120.md).
This document preserves the earlier research observations. Phase-specialized
hard repair below remains research-only.

**Finding:** a missing parallel reduction was more valuable than a better
ranker for the tested long flat routers. The previous search could give an
expert column its own CTA but could not distribute that same long column
across CTAs. At E72 or E104, even the best unsharded column ownership gave
only 72 or 104 CTAs across a 170-SM 5090; changing the candidate ordering
cannot create independent work in that missing direction.

The honest marginal number is **1.029× whole layer** on E88 when the
eight-trial policy already has the new programs. The ≈1.10–1.13× numbers
below compare against a policy or implementation that did not yet try them.

This is research, not an installed kernel or preset. New flat-only options
are in copied snapshots; neither the packed implementation nor the routing
policy, iteration count, quantization, or hard capacity was changed. No ranker
was trained, and there is no named-shape or SKU winner table.

## Isolate the new executable from the search policy

The experiment specializes the existing packed two-launch column reduction
for a static, single global pool, with no packed lengths, token repartitioning
or changed saved-state contract. The first CTA grid writes FP32 per-shard
log-sum-exp (forward) or sum (backward); a second combines partials and writes
each shard's own history/adjoint. Both directions remain independently
selectable, including zero. The additional scratch is `4 × E × shards`
bytes per direction (9,216 / 13,312 bytes with 32 shards in the two pilots).
The research search also exposes a register-resident, contiguous warp-owned
row implementation adapted from the packed program.

The following uses the *already-selected* region-32 configurations at two
spent development shapes, freezing experts, shared MLP, affinity and all
other hard/soft options. The two-fixture router evaluator retains its FP64
transport and adjoint, hard/reference, metadata and timing-eligibility gates.

| D/E/K/T | Unsharded router | Forward-only 32 shards | Backward-only 32 shards | Both 32 | Whole serial → both, ms | Whole ratio |
|---|---:|---:|---:|---:|---:|---:|
| 384/72/4/18432 | 2.1444 | 1.9464 | 1.9537 | 1.7611 | 3.9441 → 3.5495 | 1.111× |
| 512/104/6/26624 | 3.8058 | 3.4444 | 3.4589 | 3.0883 | 7.6876 → 6.9683 | 1.103× |

Two small-shape oracle gates passed before the full panels. Both 16 and 64
shards were also close to joint 32 (E72 1.7610/1.7693; E104 3.0970/3.1093
ms). This is a broad useful family, not a uniquely magic shard count. Four
alternating complete graph visits on E104 gave 7.7312 → 7.0174 ms, **1.102×**.
The cross-arm hard assignments differ; both arms reproduce their own saved
IDs and pass unchanged whole checks. Corresponding compact-checkpoint BF16
using each new router was 3.8189 / 7.7812 ms vs 3.5495 / 6.9683 ms MXFP8.
The ≈10–11% here measures introducing a family, not spending more trials
inside the old search space or universally beating BF16 by that amount.

Nsight Compute confirms the mechanism on E72's final column operation.
The single 72-CTA launch becomes two 576-CTA launches. Summed forward L2
traffic dropped 186.01→46.16 MB; backward 170.09→43.43 MB. Active warps
rose from 16.6% to 51–55%; all four configurations reported zero local
load/store sectors. Counter-replay duration was 58.9→18.7 µs forward and
55.9→16.6 µs backward. The focused warm sample reported no DRAM reads; do
not generalize that cache state or confuse traffic reduction with omitted
work. Whole ordinary replay, not counter time, is the performance claim.

## Does a real 32-trial search find it?

Router trials are actual unique admitted attempts, **not a model-wide
budget**. On fresh whole shapes eight expert and eight shared trials first
selected their own plans. The +24 experiment freezes those plans, resumes
the same router journal, and rechecks the complete layer with actual routes.
The affinity has its own two-way selection. Both RTX 5090s were initially idle
and separately locked. Kernels use Torch 2.13.0+cu132, CuTe on SM120; ordinary
warm graph timing, fullgraph static shapes, aggressive Inductor, freezing off.

| Experiment | Eight | Cumulative 32 | Interpretation |
|---|---:|---:|---|
| Fresh E112/D512/K6/H256/S512/T28672, protected former opener | 8.7224 ms | 7.7000 ms | New families admitted only after trial 8; **not** the strongest new-space 8-trial control. Paired 8.7347 → 7.7412, 1.128×. |
| Fresh E88/D384/K4/H320/S256/T19712, new family offered immediately | 3.9894 ms | 3.8725 ms | Genuine strong 8 vs 32: the combined family already wins at attempt 4. Paired 3.9912 → 3.8776, 1.029× (visit ratios 1.0274–1.0301×). |

At E88, the strong eight-trial router best was 1.9461 ms; the deeper race got
1.8236 ms in the router evaluator. Its 32-trial recipe used warp rows, adjacent
column groups, 64 shards, native warp QB and cached quota. These settings
describe an outcome, **not** a suggested row in a deployment table. E72's
fresh, protected-prefix 32-trial search finished in 54.0 s and its chosen
whole timed 3.5856 ms: it found the useful family, but was not faster than
simply adding sharding to the previously qualified winning recipe (3.5495
ms). A sophisticated selector must be compared against this cheap baseline.
All three router journals have 32 unique, 32 timing-eligible attempts. At
32 the separately matched compact-checkpoint BF16/whole MX ratios were
1.023× on E88 and 1.118× on E112. The hard IDs differ across budget arms on
both shapes; neither paired comparison demonstrates hard-assignment identity.

## Another executable that is actually worth a trial

Flat repair historically computes a diversity hash and a second top-k each
round, selecting those results only from round 16 onward. A separate prototype
uses two device-conditional loops: the first retains the identical top-k
candidate pool but omits its unused hash/selection; the second executes the
unchanged late choice. Both check the same true load and original total round
limit. There is no host readback, early stopping approximation, precision
change or reduction in allowed rounds.

Exact checks covered one-step phases 0, 15, 16 and 23, and compared the whole
conditional graph to an independent functional reference on deliberately
imbalanced starts (0→4, **15→18**, 16→19). Changed-score hard assignments
matched unspecialized execution. On E88 the actual schedule needed four
rounds: matched hard graph visits reduced 1.524→1.377 ms. On the *same strong
32* expert/shared/router recipe, normal whole runs in fresh processes measured
**3.8757→3.7854 ms (1.024×)**. All hard IDs/inverse matched; all 273 saved
output/gradient tensors had max absolute difference zero. This adds an
implementation option, not another 2.4% from the search algorithm. The
research source uses a process-wide switch: production needs an explicit
plan field and cache identity if the alternative is to become selectable.

## Where the old trials really went

The [earlier evidence](autotune_region_opening_20261120.md) shows that 8→32
largely stagnated *within the executed spaces*, not that the legal space was
exhausted. The E72/E104 expert journals each contain five post-eight
`prepared_components` proposals, all on the same backward-ownership recipe
coordinate; almost all coordinate-local trials changed movement threads. No
deep pipeline/depth sweep took place in those 24 attempts. Both new-shape
MLP journals spent all late coordinate-local selections on
`gate_up.smem_rmem_tile`. On fully prequantized forward/recompute that field
is not consumed at all: 32 distinct old IDs collapse to 22/27 active schedules
in the E72/E104 region arms, wasting 11.64/16.25 s on duplicates. The expert
tuner replaces the generic structural pool and wraps the arms; a seed-pool
count therefore cannot substitute for the attempted journals. Past 96-trial
geometry/transport panels showed little gain on their scopes; they did not
eliminate other operator families. Shared-MLP active identity is a separate
production-side fix, not responsible for the sharded-router measurements.

## Portable allocation to test

- Attempts 1–8: keep the incumbent, one proven joint column-ownership
  alternative and protected escape slots. For an obviously underfilled long
  column, test at least one sharded representative now, not artificially on
  attempt nine. Pick a starting shard count near two CTA waves using SM count
  and CTAs per shard, *then measure*. Cover at least two hard-selection
  implementations on complete compatible recipes. Do not change the packed
  or short opener on the strength of these long flat experiments.
- 9–16: contrast both directions and row ownership from a fixed eligible
  prefix anchor; one-sided/backward-only is valuable as a hard-route
  preservation control. Do not burn four attempts on aliases under a noisy
  new leader.
- 17–24: rerank the first tranche, freeze that anchor for the next tranche,
  and vary coalescing, shard count, and hard quota cost on the *same* good
  transport. Rank by qualified latency, allow rejected/failed admitted trials
  to count, and release exhausted quotas to legal alternatives.
- 25–32: measure coupled hard backends/threads/cache vs the good transport,
  preserve exploration, and remeasure structurally distinct finalists. Charge
  repeats explicitly: this study charged 32 distinct admissions per router;
  the separate four-visit whole validations are not free search trials.

Do not use spatial similarity between candidate IDs as a proxy for different
programs. Avoid always taking the first coordinate of a newly rebased
neighborhood. For a high-reuse layer, use paired whole evaluation of a few
eligible plans, and compare against retrieval, incumbent-plus-shards and the
strong eight; if near-tied, deploy the simpler validated candidate. Neither
this nor the earlier offline ranks shows convergence or a cross-SKU optimum.

## Qualification, deployment boundary and handoff

All eleven whole-arm outputs completed normal finiteness, replay, independent
quantized-FP64 expert and zero live-growth checks. The largest expert normalized
RMS error was 0.0107%. Five extra warp/empty-shard/tail fixtures passed, from
1×1 up to 208×104. The strict full-router gate also passed on these tested
flat shapes: E72/E104 serial vs sharded and E88/E112 8 vs 32, each with
normal → changed input/nearly constant gradient → restored graph replay.
All **24 records** passed the original scalar tolerance (rtol 2e−4, atol
1e−5), tensor gradient/route metadata/weight checks. The worst scalar absolute
error was 9.13e−6. Fixed-route oracle refers to each arm's own actual IDs; it
does not force two distinct transports to select the same hard routes. This
does not repair or dismiss the [earlier separate failed fixture](mxfp8_moe_autotuning_astra_review_20261120.md).

**Recommendation:** port the column family as an opt-in flat implementation
and expose it early on long, underfilled reductions; then test the exact
repair split independently. The research cold router path
`selected_sinkhorn{,_backward}` still invokes legacy geometry-only launchers
on its first unbound call. Route both cold and bound dispatch to the same
plan, carry the full immutable forward/backward token, version winner keys
and serialization, canonicalize packed/inactive fields, and test lazy cold
vs bound execution, saved backward across refinement, custom-op graphs,
capture and reload before changing a default. Keep legacy, small/packed and
fixed-forward paths. Qualify independent hardware and additional length/
score regimes; the two boards here are the same SKU. The repair split also
needs explicit plan selection and bounded graph-cache/lifetime tests.

Raw [archive and replay directions](../autotune_reports/astra32_autotuning_20261120/ARCHIVE.md),
[machine-readable timings and journal coverage](../autotune_reports/astra32_autotuning_20261120/summary.json)
and [incremental research patches](../autotune_reports/astra32_autotuning_20261120/patches/)
are separate from the live package. To reproduce a source delta use the
documented predecessor snapshot, not the current dirty working tree.

## Parent control: real programs and active-phase coverage

The shared-MLP identity bug is fixed in the live tree, separately from Astra's
experimental kernels. Autotune revision 50 drops unused `gate_up` fields when
both selected paths are prepared, but retains the common SiLU numerical policy
and the full on-chip schedule whenever either executed path needs it. Old
MLP tuning identities are invalidated, not silently mixed. Twelve CPU
backend/phase cells and sensitivity checks pass; a compiled CUDA test confirms
bitwise-identical output, dX and both weight gradients for the collapsed aliases.
The wider focused CPU suite passed 51 tests with five CUDA skips.

We also froze a separate fresh case, T20480/D384/E80/K4/H320/S256, eight
2560-token samples, **fixed router/affinity** and identical inputs, parameters,
upstream and hard IDs. Both controls use the corrected active identities.
Experts and shared MLP each receive real cumulative 8→32 trials; every result
passes independent quantized-expert, all-gradient replay and live-memory gates.

| Search policy | Eight, whole F+B | 32, whole F+B |
|---|---:|---:|
| Active IDs, ordinary proposal order | 4.3932 ms | 4.3694 ms |
| Phase-balancing prototype v1 | 4.3941 ms | 4.3872 ms |
| Corrected phase-balancing v2 | 4.3920 ms | 4.3929 ms |

Version 1 mistakenly treated flattened dX/dW coordinate aliases as different
phase groups; its immutable evidence is retained. Version 2 corrects this
and tests both nested and flattened names. Its actual later proposals cover
all seven expert GEMM roles and four shared-MLP backward legs, instead of
repeating the inactive gate tile. **Coverage improved, performance did not.**
These sub-percent separate-process changes do not establish a material win;
the phase wrapper remains a research driver, not a production default or proof
that the entire implemented space is exhausted.

This argues for adaptive stopping and allocating deeper effort to measured
headroom, not spending 32 attempts on every component regardless of return.
Current caps are per component; a future whole-layer budget allocator must
explicitly account for cross-component admissions and finalist remeasurements.
Promoting a model or a more elaborate scheduler on candidate diversity alone
would not be justified by this control.

The independent [phase-control archive](../autotune_reports/autotune_phase_coverage_20261120/ARCHIVE.md)
contains frozen v1/v2 sources, the alias audit, raw journals, six qualified
whole results, test logs, and CSV/Parquet exports. It does not modify Astra's
checksum-verified original archive.
