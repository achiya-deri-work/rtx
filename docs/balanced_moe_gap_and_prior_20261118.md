# BalancedMoE: clean evidence, search coverage and remaining gap

## Outcome

HOME GPU0 was used alone. The clean collection is finalized, the bounded
search now reaches an important GEMM-geometry/pipeline basin, and focused
qualification found a real but modest complete-layer gain. **This is not a
1.2× BF16 result.** Checkpointed BF16 remains faster in the representative
thin-expert cases measured here.

No numerical policy, register budget, public allocation/stream requirement,
or default kernel configuration was changed. The runtime change is soft
candidate coverage: candidates still need the normal oracle and measurement.
GPU1 and unrelated workloads were not used or stopped.

Evidence is preserved under
`autotune_reports/moe_gap_and_prior_20261118/` (local Storage4), including raw
journals, exact fixtures, traces, rejected runs, NCU report, research models
and a source snapshot. Earlier quarantined studies remain excluded.

## Finalized collection

The source is the qualified prefix of
`autotune_reports/moe_clean_recollection_20261118_home0_v2/`, not the older
fault-contaminated studies. Its frozen runtime SHA256 is
`4c0064a38513c8fbeb73b856227cb2827cb5d3c88cc18a08e2d56c4e726f116e`.

- 32 architecture/packing contexts reached eight trials; eight reached 16.
- 40 visits passed the numerical, replay, allocation and log-health gates.
- 1,024 raw observations: 320 experts, 320 routing, 320 shared MLP, 64 affinity.
- Eight observations after the last qualified checkpoint are excluded, not
  deleted from the original journals.
- Four independently checksummed JSONL partitions, lineage, audit,
  CSV and Zstd Parquet include the previously separate shared-MLP records.

`benchmarks/finalize_moe_collection.py` checks the exact primary observation
identities against the last qualified export and completion-time fence.
It refuses existing destinations and publishes a negative health marker
while constructing partitions. Failures remain labels, not successful data.

`collect_moe_clean.py` now recognizes a campaign-clipped visit timeout as a
normal deadline stop, writes `completion.json`, and retains the last qualified
checkpoint. An ordinary per-visit timeout with campaign time remaining, a
fatal CUDA warning, or failed qualification still stops with an error.

## What the traces actually show

The focused cases use D=384, E=32, K=4, expert H=128 and one shared H=256 MLP:
128 flat tokens, independent packed lengths `[512,2000,768,4096]`, and 29,504
flat tokens. Inputs, all parameters, upstream gradients and hard routes are
hash-matched to the frozen source study. Ordinary CUDA Graph timings are
separate from profiler diagnostics.

The large native expert step has 19 launches versus 23 for checkpointed BF16,
but takes approximately 1.91 ms versus 1.46 ms in the diagnostic trace.
Both expert traces have approximately full device-time coverage; CPU launch
gaps are not the explanation for this isolated expert deficit. The BF16
trace includes recomputation; it is not a saved-intermediate comparison.

Large native expert diagnostic costs are approximately:

| Work | Time |
|---|---:|
| Pure GEMMs | 716 µs |
| Gate/GLU | 162 µs |
| Joint recomputation/preparation | 192 µs |
| Route-gradient GEMM/reduction | 244 µs |
| Quantization | 241 µs |
| Movement | 318 µs |
| Cast/copy | 39 µs |

These categories describe one instrumented trace, not independently
subtractable or additive optimization promises. Forward down projection and
gate-dX are particularly weak: approximately 218 versus 77 µs, and 238 versus
140 µs, respectively. Joint recomputation is not uniformly bad; the matched
BF16 recompute/activation/derivative pieces together cost more than its
192 µs here.

Tiny shapes have a different limitation. For N=128, mean expert load is 16
but storage capacity is 32. Modeled slot amplification is 2×, forward tile
amplification 4× and gate-dW tile amplification 8×. These are layout/work
models, not measured instruction counts. The large flat case's capacity
amplification is only about 1.0065×. One universal sampling preference would
therefore be inappropriate.

### Hardware counters

The old large forward-down GEMM measured approximately 5.8% tensor-pipe
activity, 18.2% active warps, 64 registers/thread and zero local-load/store
sectors. Its grid has 11,136 CTAs; this is not a single-wave grid shortage.
The report records 558.7 MB L2 traffic. Requested DRAM read/write metrics were
not returned and must be treated as missing, not zero. NCU's instrumented
233 µs duration is not used as an ordinary latency measurement.

Low tensor activity without spills supports investigating tile shape and
latency/scheduling. It does not establish HBM saturation. The CLI NCU report
is retained because the Nsight Python interface was unavailable.

## Search starvation and the implemented correction

All **320** old expert trials kept down and gate-dX at the same 64×64×128
TMA geometry. The collection could identify preparation associations but
could not learn a TMA/cooperative or geometry boundary it never sampled.

Two focused 24-trial searches used the existing adapter, legality checks,
shared UCB/global/local engine, FP64 expert oracle and residual store:

- Geometry/transport: 1.8606 → 1.7444 ms complete expert evaluator latency.
- Pipeline/locality follow-up: 1.6997 ms, about 8.6% below the same initial
  expert evaluator. The measured best uses 128×128×128, one stage, preload.

These are entire expert-evaluator measurements, not just one GEMM. Their
FP32 upstream-gradient scope differs from the public BF16-gradient step;
do not compare their numbers directly to the public-path timings.

`geometry_growth_seeds()` derives legal coupled geometries from the existing
axes, keeps register/precision/traffic choices inherited, and offers shallow
preload when there is only one physical K tile. It does not remove deeper
pipelines, cooperative transport, or any existing candidate family.

The component arm rebases proposals onto the measured incumbent. The ordinary
eight-trial opening protects one geometry proposal for non-tiny expert
capacities while retaining quantize-once, indexed and compact-recompute
basins. Tiny short-dW coverage remains protected. Portfolio revision is 17;
cache/checkpoint identities change rather than reinterpreting old budgets.
Explicit user/research ranked openings retain their existing priority.

Fresh eight-trial expert searches completed for large and mixed packing in
39.0 and 35.6 seconds respectively, with numerical, full-leaf replay and
steady live-memory checks passing. The large winner selected the offered
128×128, one-stage/preload down path. These tests tune experts while freezing
router/shared schedules; their durations are not a cold entire-model compile
or three-family first-hit time promise.

### Paired complete-layer qualification

The geometry-only candidate was compared under fullgraph max-autotune,
outer graph replay, four alternating visits and seven samples per visit.
Every leaf, hard assignment, replay and steady allocation check passed.

| Case | Old schedule | Geometry candidate | Result |
|---|---:|---:|---|
| 29,504 flat | 4.8731 ms | 4.7660 ms | 2.2% speedup |
| Packed 7,376 | 1.2603 ms | 1.2585 ms | Effectively tied |
| 128 flat | 0.4920 ms | 0.4919 ms | Effectively tied |

The actual new eight-trial large-case winner was separately qualified against
the old schedule: **4.8917 → 4.7388 ms, 1.0323× speedup**. All four alternating
visits favored it; maximum relative sample MAD was below 0.07%. Worst
normalized difference across all compared leaves was 8.76e-5. Both variants
had five identical live-allocation samples. The allocation totals are
cumulative while both graphs/models remain alive, not independent per-layer
peak-memory estimates.

The new eight-trial packed winner also passed: **1.2579 → 1.2390 ms,
1.0153× speedup**, with every alternating visit favoring it and worst
normalized leaf difference 1.18e-5. This is a small measured gain, not the
large-shape tile's universal benefit. Both results use the entire selected
expert recipe, not a claim that one coordinate caused all improvement.

The first attempt was rejected before timing because the unchanged baseline
alternated by about 32 MiB in its five live samples. A separate warm-cache
retry passed the unchanged gate. The rejected log is retained; its root cause
is not proven and the retry does not establish that cold compiler cleanup is
universally stable. No threshold was loosened or failed sample omitted.

No measured configuration was silently installed as a universal runtime
winner. New first-hit results and rejected qualification attempts are
preserved beside the geometry-only controls.

## Priors and rejection analysis

The finalized dataset produced 3,983 context-adjusted coordinate effects,
but zero portable effects: it contains only one SKU. Good within-context
associations include indexed input preparation, fused route-gradient work
and compact/dual-operand recomputation. These are coupled observational
associations, not independent causal gains that may be summed.

Small research cost/ranking/feasibility models were trained on only the clean
1,024-row prefix. The new focused searches and first-hit tests are not in
their training data. Every family still has deployment cost head `none`;
feasibility/rule deployment is disabled. No research model was installed as
a portable runtime prior. The SKU report now prints the actual device count
and marks unavailable hardware measurements `n/a`, rather than physical zero.

Five distinct affinity context/config pairs were rejected by the existing
strict non-scalar gradient comparison after replicate aggregation (nine raw
rejections). Forward score agreement alone does not qualify these backward
paths. BF16 rounding/association is a possible explanation, not established
causality. Scalar fallback and all thresholds remain unchanged. One routing
candidate's CUDA illegal-state error remains a failure label; a selected
fallback subsequently passed the full visit gates.

## Remaining work

1. Expand geometry/pipeline coverage prospectively across the four intended
   architectures; compare fixed trial budgets, including negative controls.
2. Reduce the large expert movement/quantization cost and down/gate-dX
   scheduling deficit, measuring the complete composed path each time.
3. For tiny pools, prioritize executed-work/capacity amplification and routing
   rather than assuming the large down-GEMM tile is beneficial.
4. Collect held-out device evidence before enabling learned portable heads.
   More data from the same unvaried geometry would not answer that question.

The closeout CPU suite passed **55 tests** covering study/priority coverage,
trace classification, SKU reporting, finalization/deadline gates and frontier
templates. The generated API reference matches source and `git diff --check`
passes. GPU qualification is the targeted oracle, replay, memory and paired
matrix above, not a rerun of the entire library or a convergence experiment.

No new broad collection, training-convergence claim or release qualification
is implied by this focused investigation.
