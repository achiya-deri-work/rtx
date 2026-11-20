# Beating checkpointed BF16 with bounded MXFP8 tuning

The recommendations have a measured
[implementation follow-up](balanced_moe_priority_execution_20261118.md),
including the negative shared-fusion result and bounded first-hit changes.

**Verdict.** Put roughly equal engineering attention into routed experts and
the dense shared MLP. On the large H=128 fixture they explain about 0.30 and
0.26 ms respectively of the 0.57 ms complete-layer deficit. A better expert
opening is worthwhile but cannot, at the demonstrated scale, fix both. Parity
is a credible research target; 1.2× here is not supported by any current
search or component result.

## New evidence: cross the two FFNs in the same complete layer

HOME RTX 5090 physical GPU 0 only, device UUID
`GPU-09db6472-cc08-c4cb-90db-560de232e5aa`.
One independent pool, N=29,504, D=384, E=32, K=4, H=128, one shared H=256.
All inputs, upstream gradient, parameters and hard routes hash-match the
[eight-trial winner fixture](../autotune_reports/moe_gap_and_prior_20261118/first-hit/large/result.json).
The native schedules are frozen, not retrained; the same native router runs
in every cell. Each BF16 FFN is independently checkpointed with the benchmark's
existing reference, never the router. The mixed cells **only attribute cost**;
they are not proposed precision fallbacks.

| Routed experts | Shared MLP | Complete F+B (ms) |
|---|---|---:|
| MXFP8 | MXFP8 | 4.7317 |
| MXFP8 | Checkpointed BF16 | 4.4731 |
| Checkpointed BF16 | MXFP8 | 4.4313 |
| Checkpointed BF16 | Checkpointed BF16 | 4.1653 |

The expert penalty is 0.3005–0.3078 ms holding the shared branch fixed.
The shared penalty is 0.2587–0.2660 ms holding experts fixed. The interaction
is −0.0073 ms; these are contrasts of compiled layer variants, not the sum
of independent kernel timings. Parity requires removing **0.5664 ms**.
1.2× requires reaching **3.4711 ms**, cutting **1.2607 ms (26.6%)** from native.
Even BF16 parity on *both* branches would still leave 0.6942 ms to find.
Optimizing the shared router alone can improve absolute latency for both
but cannot flip their ordering if the saving is the same.

Torch 2.13/cu132, fullgraph max-autotune with subprocess GEMM tuning,
set-to-none backward, all gradients, four counterbalanced visits × seven
batched CUDA-event samples on warmed outer graphs. No optimizer. Worst
within-visit relative MAD 0.171%; output plus 104 gradient tensors checked
on two upstream values against fresh execution. Native-expert replay matched
exactly; BF16-expert maximum NRMS 0.000251 (0.0251%), not bitwise equality.
All five final live-allocation samples identical for each variant; absolute
totals are cumulative while multiple graphs remain live, not peak estimates.
This tests replay, not cross-precision numerical equivalence or convergence.
Compile/warm/capture excluded; its per-cell wall times (6–34 s) are
order/cache-dependent and are not first-hit search timings.

[Raw samples, schedule provenance, per-leaf identities and checks](../reports/moe_precision_factorial_review_20261118/large-v1.json)
and [reproducible evaluation](../benchmarks/review_moe_precision_factorial.py).
Runtime package hash `5f0417fd1666264dd5cc7d828e8b84bf27f53081d490314603aa42bf8cbd275f`
matches the focused closeout, not the older clean recollection. Run from
the corresponding source, idle GPU0, with a **new** output path:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.review_moe_precision_factorial \
  --source-result autotune_reports/moe_gap_and_prior_20261118/first-hit/large/result.json \
  --output reports/moe_precision_factorial_review_20261118/large-repeat.json
```

## Ranked work

1. **Routed adjoint preparation and movement.** Both actual eight-trial
   winners still materialize expert-slot BF16 weighted dY; **none** of their
   trials exercised the implemented fused dY path. At large capacity the
   BF16 temporary is 91.2 MB per write, before rereads; this is a shape-based
   byte count, not measured DRAM traffic. Rebase the existing fused dual-axis
   dY quantizer on the winning indexed/compact/down recipe before writing
   another kernel. Then try warp-owned, vectorized token/rank reduction for
   combine and dX, loading route IDs/weights once per group of features and
   keeping FP32 accumulation in rank order. Keep standalone indexed-X and
   fused-all as competing templates, not compulsory fusion. Compare whole
   F+B, L2 bytes and elapsed movement, not launch count. Preserve BF16 rounding
   of weighted dY and independently quantize the two orientations. See
   [expert plan](../rtx/_moe_execution.py#L252), [dY kernel](../rtx/kernels/mxfp8_moe_dy.py#L117),
   [movement](../rtx/kernels/mxfp8_moe.py#L149), [prior negative controls](balanced_moe_amortization_5090_20261118.md).
2. **Shared MLP training path, now a measured ~46% of the loss.** The frozen
   winner uses prequant gate/down, joint recompute, decomposed gradient GEMMs,
   but *separate* derivative and hidden quantization. Benchmark the existing
   legal compound `recompute_both` plus hidden-quantization exchange against
   it, including required compatible layouts and dW policy. Then test within-
   phase reuse of the per-token X quantization between expert and shared gate
   producers **only when** axes, scale rules and layout agree. Do not persist
   stale packed weights across optimizer updates or cache forward activations
   into backward. Shared forward still creates scratch behind its launcher;
   use compiler-visible output/arena allocation for new work, as in routed
   [plan/launch separation](../rtx/_moe_execution.py#L397). See [shared
   forward](../rtx/mlp.py#L220), [shared recompute](../rtx/mlp.py#L676).
   First measure selected shared stages before redesigning them: this
   factorial establishes branch cost, not its internal cause.
3. **A shallow, few-K row-GEMM/route-epilogue family.** Existing down
   geometry also feeds the expensive backward route-gradient projection.
   The large frozen trace has ~218 µs down, 244 µs route-gradient GEMM,
   238 µs gate-dX. Legacy down launches 11,136 CTAs with 5.8% tensor-pipe
   activity, 18.2% active warps, 64 registers/thread, no spills and 558.7 MB
   L2 traffic; DRAM metrics absent. This is not proof of HBM saturation,
   register pressure or global grid starvation. Test output-tile/raster
   ownership, direct vs staged store, and one/two-K-tile producer/consumer
   vs cooperative transport. A single K-tile cannot overlap its next load;
   gate-dX has two, not one. Add a separately tunable route-gradient
   epilogue only if controlling it together with forward down blocks gains.
   Preserve the BF16-rounded expert output before computing its route adjoint.
   Challenge gate-dX vs a strong control: 11 isolated stage candidates in the
   first geometry frontier achieved only ~1.839 versus ~1.861 ms expert
   baseline; the down frontier was more productive. Never choose register
   caps or declare one geometry a SKU dispatch rule. [Builder](../rtx/_moe_execution.py#L306),
   [frontier journals](../autotune_reports/moe_gap_and_prior_20261118/frontier-large-v2/residuals),
   [NCU counters](../autotune_reports/moe_gap_and_prior_20261118/down-baseline.csv).

No single one of these has evidence for 1.2× at H=128. The historical
[width control](balanced_moe_amortization_scaling_20261118.md) reached
1.23× *experts* and 1.09× *layer* at H=512/N=29,504 with different source/
schedules. That makes wider, compute-heavier FFNs a better place to pursue
1.2×; do not silently widen the model or import old training data. For
N=128, 16 real rows/expert, storage 32, 4× modeled forward tile work, 8×
gate-dW reduction work, and only 64 modeled forward gate CTAs for 170 SMs:
prefer short-K dW and exact small-pool routing. Even eliminating the entire
isolated native expert time (~0.092 ms) is less than the observed whole-layer
gap (~0.165 ms); it is not a literal additive upper bound across graphs.

## First-hit policy and acceptance

The clean dataset is 1,024 observations, **not** 1,024 expert trials. All 320
old expert attempts used one down/gate-dX geometry; one SKU cannot train a
portable geometry selector. On the two revision-17 eight-trial runs the only
non-default down proposal was trial **7**, won both, and no gate-dX geometry
changed. [Opening](../rtx/autotune/moe.py#L981) takes just the first legal
growth yield (down before gate-dX). Positive reachability is not useful
coverage under a wall deadline: these eight-trial searches had **120 s**
available; winners finished at 38.68 / 35.29 s. Preset quick=8/15 s and
balanced=16/30 s can miss them entirely. A trial's `compile_ms` includes
first run, oracle comparison, warm/capture, not pure JIT; the per-context
FP64 expected result is computed *before* the search timer. [Evaluator](../rtx/autotune/moe.py#L436),
[presets and evidence contract](../rtx/autotune/effort.py#L110).

| Cumulative effort per context | Proposed policy (not deployed) |
|---|---|
| 8, service first hit | Load exact compatible winner first; otherwise try legal seed and one compact+indexed+geometry compound immediately. Protect one alternative traffic/dY path, compact/legacy subfamily controls, and a genuinely different transport/ownership recipe. For tiny capacity substitute short-K dW, not large-row geometry. Reallocate redundant initial neighbors; keep one exploration slot. Report *admitted* configurations and time to first useful winner; no requirement to finish all eight before serving an incumbent. |
| 16, deliberate refinement | Resume the same eight observations; rebase fused dY, one gate-dX basin and a down/route-epilogue challenger on the best feasible recipe; spend the rest on competition/confirmation. In eight historical clean expert contexts, best eligible latency ratio at 8 vs 16 has median 1.047×, range 1.000–1.151×; not an unbiased oracle or a whole-layer gain. |
| 32, hot shapes/offline | Only if lifetime warrants it: compare joint expert/shared finalists on full F+B; spend residual exploration on another engine and structurally different candidate, not 32 near-duplicates. There is **no** clean 32-trial empirical regret curve here. Separate pilot transfer proposals from exact winners, and time/validate them on the target. |

Keep device/source/measurement/effort/shape and packing-mode identity;
use exact disk hits with `cache_behavior='reuse'`, explicit `refine` to
extend compatible progressive journals. Packed expert [key](../rtx/moe.py#L114)
contains sequence count and alignment, not actual offsets: same total and
sequence count may have quite different active/padded work. Test the winner
on radically different same-key offset distributions before broad reuse;
avoid per-value guards, CPU scalar reads or unbounded cache keys. Router,
experts and shared have separate first-hit contexts; time all three plus the
cold→hot graph transition, both empty-cache and warm restart. Identify
identical layer shapes that can amortize one choice across a stack. A
39 s/35.6 s *entire* expert search paid solely for the marginal measured
revision-17 advantage over a free old winner would need ~255k large / 1.87M
packed **layer invocations** to break even; incremental refinement cost and
the actual fallback can change that materially.

**Next bounded experiment.** Frozen large+mixed H128 positives, tiny and
empty/ragged negatives, a wider H256/H512 positive, then all four actual
architectures and a held-out device before any portable prior. First A/B
the independent expert-dY and shared-exchange changes, then their composition;
only then spend 16/32 trials on row-GEMM frontier expansion. Record best
eligible regret vs the same finite prospective envelope (not an optimum),
coverage/time at 8/16/32 and at 15/30/60 s, context setup, kernel compilation,
capture, persistence/reload and end-to-first-hot-step wall time. Report zero
completed coverage as missing, not zero latency.

Require exact hard routes; independent quantized FP64 expert leaves and
shared oracle; changed-dY replay of every leaf, poisoned padding/empty pools,
flat warm live memory and fresh-process isolated peak. Keep BF16 checkpointing,
all gradients, fixed seeds and fair max-autotune in every timing. Promote only
when all four paired visits agree and the whole-path delta exceeds noise
(e.g. >1% and >2× relative MAD); report ties/regressions and keep fallbacks.
Require BF16/native ≥1.00 or ≥1.20 on the **specified target** before calling
it parity/a win; no small-shape universal, portable-head or convergence claim.
The original intermittent 32 MiB steady-allocation failure remains unresolved;
repeat the unchanged gate after cold as well as warm setup.
