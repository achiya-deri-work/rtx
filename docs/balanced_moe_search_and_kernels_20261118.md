# BalancedMoE: search quality and structural follow-up

> **Qualification correction, 2026-11-18:** The subsequent GPU0 investigation
> found fatal CUDA warnings in this collection and in its older training
> input. Both `moe_search_and_kernels_20261118` and
> `moe_amortization_scaling_20261118` are quarantined. The first-hit/regret,
> prior and timing claims derived from those bundles below are historical,
> **not qualified deployment evidence**. Passing child exit codes and JSONL
> integrity audits did not establish device health. Independent per-feature
> controls are separate artifacts, not automatically recertified by this fix.
> See the [incident and structural fix](balanced_moe_gpu0_incident_20261118.md).

This continues the [composed-path qualification](balanced_moe_next_priorities_20261118.md).
Only HOME physical GPU 0 and the authorized laptops were used. HOME GPU 1
and unrelated workloads were not modified. No optimizer/convergence claim or
universal advantage over checkpointed BF16 follows from these kernel studies.

## Measured structural results

| Change | Measured result | Decision |
| --- | --- | --- |
| Compiler-visible affinity layout, HOME 5090 | Matched complete native packed MoE F+B 1.29308 → 1.17308 ms, 9.28% lower latency | Integrated, measured binary choice with scalar fallback |
| Short K32/K64 weight-gradient scheduling, HOME 5090 | 1.8–5.4% lower routed-expert latency; 0.4–0.9% lower complete-layer latency across four paired shapes | Optional shared-autotuner family |
| Per-pool persistent Sinkhorn, laptop 5070 | Raw kernels win, but safe conditional graph adds 57–61 µs per direction on WSL; mixed/long controls regress | Optional only, no speed/default promotion |

These are separate controlled experiments, not additive percentages or BF16
speedup ratios. The affinity comparison uses N=7376,D=384,E=32,K=4,H=128
and frozen shared/expert/router schedules. Every one of its four paired
timing rounds exceeds 1.101x. Hard metadata, output and all nonscalar gradients
are bitwise equal; changed FP32 scalar reduction association has maximum
absolute gradient difference 7.92e-8. Live allocations remain flat.
The independent tiny N=1/E64 first-hit control retains scalar; real
N7376/E32 affinity F+B measures 154.00 → 83.14 microseconds and selects rows.
Fresh cache-only processes reuse both decisions. Thirteen focused GPU tests
and 91 CPU routing tests pass. Exact source and evidence:
[affinity archive](../autotune_reports/moe_next_priorities_20261118/affinity_home0/affinity-evidence.tar.gz).
Archive SHA256: `7d7722db3207e67adaf6e88a8b06b769fb92a5bc20130e28103c77febfdd2756`.

Affinity auto selection uses the existing shared adapter/orchestrator,
paired-promotion, residual and winner machinery. The cold compiled graph
contains only a no-output ordered preparation effect; it does not copy an
affinity scalar or change its autograd edge. A failed scalar-copy prototype
is retained in the evidence because it changed cold compiled scores by one
ULP. The final effect-marker path preserves exact cold/hot scores. Hot
affinity math remains compiler-visible and no private graph is retained.
See [constructor controls](balanced_router.md#affinity-projection-and-scalar-gradient-layout).

The [short-dW report](balanced_moe_short_dw_20261118.md) includes a matched
cooperative K128→K32 control: gate dW improves 10.2%, while down dW changes
inside timing drift. Six HOME qualification tests and three 5060 standalone
poison/oracle tests pass. The 5060 had active display work, so its timings
were not collected or promoted. Existing K128/default kernels remain intact.

The [per-pool report](balanced_moe_hybrid_pool_sinkhorn_20261118.md) includes
17 passing 5070 tests, mutable small/mixed/long boundaries under graph replay,
saved FP64 history/gradient checks, exact hard metadata, and flat live memory.
A poisoned workspace proves the all-short graph skips staged work. The WSL
overhead is not therefore explained by accidentally executing both paths.
The two-trial tuner check measures both choices, but its <1% margin is not
promotion evidence. The native-Linux control now has ordinary paired event
results too (HOME5090, microseconds; saved history and transport gradients
checked against staged before repeated replay):

| Pools | Staged F/B | Raw persistent F/B | Safe hybrid F/B |
| --- | ---: | ---: | ---: |
| Eight × 128 | 24.570 / 20.487 | 20.470 / 14.344 | 24.647 / 20.542 |
| Sixteen × 64 | 24.570 / 20.491 | 12.295 / 10.237 | 18.492 / 14.357 |

The native branch cost is materially smaller than WSL. The safe path ties
staged at eight pools, but saves about 25% forward and 30% backward at sixteen
shorter pools. These are **transport-only** gains, not a qualified whole-layer
hybrid speedup. Geometry and execution platform must remain part of subsequent
evidence; raw-kernel wins alone do not justify dispatch.

## Learning-path corrections

The audit found problems in representation, not just model capacity:

- MoE shapes are named `tokens/d_model/experts/capacity/top_k/hidden`, not
  GEMM `m/n/k`. Held-out validation and contextual-UCB similarity now use
  those dimensions. Packed routing and affinity receive their own shape keys.
- Packing sequence count is retained as a semantic feature rather than being
  discarded with source/provenance tags. Early affinity tensor signatures
  can be interpreted into explicit dimensions on ingestion.
- Named-SKU and flat-spec metadata could describe the same GPU with different
  labels. A conservative loader pass reconciles aliases only when positive
  SM/bus/L2 specs match exactly one observed named SKU. It does not guess a
  GPU from SM count or merge two explicit SKUs with ambiguous specs.
- Portable pre-compilation proxies now include rows/expert, slot amplification,
  weight bytes/L2, experts/SM, activation-to-compute and weight-to-compute terms.
  dW adapters also expose padded K, modeled tile amplification and CTA waves.
- Explicitly nonstationary samples are excluded from SKU effect/winner
  transfer analysis. Source residuals remain authoritative; schema/audit tools
  should precede ingestion. The pretrained trainer revision is 7.

Historical export contains **755 MoE/router observations across 69 contexts**
over three physical device profiles, including the earlier bounded-eight
control. Shared-MLP observations are kept in the existing pretraining input,
not mislabeled as expert kernels: total historical training input is 1027
observations, with nonstationary rows excluded from latency fitting.
The old/new metadata audit reconciles 486 historical row aliases without
an ambiguous match. CSV and Zstd Parquet exports retain config and portable
feature payloads, outcome, setup time and identities.

The existing gradient-boosted latency/ranking/feasibility and conditional-rule
tools are reused, not replaced by another model framework. Training uses the
older study only; fresh collection is held out. Artifacts stay offline. Small
historical catalogs often contain no more than eight candidates, so zero
retrospective regret@8 is not proof of great first-hit search. Conditional
associations also do not establish causal benefits from coupled coordinates.

## Balanced, resumable collection

`benchmarks/study_moe_priors.py` defines 32 contexts: four intended designs
times eight token/packing profiles. It covers 128/2000/29504-token flat pools,
the original mixed pack, many short pools, and same-total/empty/boundary
controls. Architecture coverage comes before deeper effort.

Each context has cumulative **8→16→32** expert/router trials. Every wave
shares the same residual directory and original seed; seeding the next wave
from its winner would change context and defeat true resume. Initial shared
MLP scheduling is fixed by default (`--shared-trials 0`); use frozen shared
sidecars or explicit shared tuning before making production BF16 speed claims.
Every completed point checks the independent quantized FP64 expert oracle,
all-leaf fullgraph backward, fresh-gradient replay and steady live memory.
Selected representative traces are captured only after ordinary event timing.
Per-context JSON and append-only residuals survive interruption independently.
In-flight work respects a soft admission deadline and an explicit per-visit
timeout; completed evidence cannot be silently resumed under another source.

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.study_moe_priors \
  --source /path/to/immutable-rtx-snapshot \
  --output /path/to/study --max-autotune --max-seconds 3600

python -m benchmarks.analyze_moe_priors /path/to/study \
  --output /path/to/analysis

rtx-autotune pretrain /path/to/older-training-data --output /path/to/prior
rtx-autotune evaluate-pretrained /path/to/prior /path/to/fresh-study \
  --output /path/to/heldout-evaluation
```

Candidate-budget curves count unique trials and report missing budget points
honestly. Their reference is each context's best **observed** eligible result,
not a global optimum. Candidate elapsed time excludes proposal and whole-model
compilation overhead; actual cold-start totals remain in the result files.
No learned model, heuristic pruning rule or study winner is installed into
the user's default runtime cache by these tools.

## Completed first-hit pilot and practical limits

The first HOME wave covers all four designs at N=128 and N=2000: eight
complete points with 8/8 actual router/expert trials per point. Both 5070
visits also completed on the **byte-identical** frozen runtime (D384/E32/K4
at 128 tokens and D512/E64/K8 at 2000). Together with the guided control and
the effort extension, these produce 204 residual observations in 31 tuning
contexts. The 5070 medium row-affinity candidate failed its strict gradient
gate and retained scalar; that failure is evidence, not silently discarded.
All qualified complete points pass independent quantized FP64 expert checks,
all-leaf backward/replay, and zero measured live-allocation growth.
[5070 report](balanced_moe_prior_pilot_5070_20261118.md).

The frozen historical ranking head, evaluated without retraining on the four
fresh HOME tiny catalogs, finds the best observed expert config within four
choices in all four cases. Matched random order has 6.05% median regret@4;
ranked top-one p90 regret is 0.55%. This is catalog replay over eight measured
candidates/context, not prospective search over the full hypercube. Observation
and source identities are disjoint; not every shape is unseen in training.

An actual opt-in portable-prior run at D512/E64/K8/H128, N128 holds router/shared
schedules and **all** input/upstream/parameter/hard-route hashes fixed. Its
eight-trial expert result is 155.46 versus 157.76 microseconds (about 1.5%
lower). Reported expert-tuner time increases 20.70→22.45 seconds; whole-layer
rounds drift and do not establish a win. Thus good offline ranks alone are
insufficient to turn on the prior globally. `pretrain` produces an evaluation
model directory; the runtime `--prior-artifact` comparison uses the separate
portable-prior JSON(.gz) format and explicit `--allow-research-prior`.
Neither artifact has been enabled as a default.

The real N128/D384/E32 8→16 extension reuses the same journals: each core tuner
adds eight candidates, leaving exactly sixteen unique configs and two unchanged
affinity observations. Expert medians are 92.19→83.99 microseconds, full layer
527.53→518.13 microseconds. These are same-input cross-visit effort observations,
not paired causal speedup qualifications.

Cold CPU profiling exposed repeated normalization/serialization of large frozen
configs. Per-search 4096-entry normalization/ID memoization, immutable subtree
reuse and precomputed problem geometry reduce one matched setup control from
10.82 to 5.49 seconds. **All 128 pool IDs and nine protected opening IDs/order
match exactly.** This is portfolio setup, not total first-hit wall time or hot
GPU speed. No serialized mutable dict or tensor is cached; tables die with
the search adapter. Incompatible priors now retain normal family coverage
instead of suppressing it merely because an artifact path was supplied.

The focused final integration CPU suite passes 202 tests (38 CUDA skips);
the search-only subset passes 175 tests. Separately, HOME passes three native
mutable-boundary/cache tests and two user-side automatic CUDA Graph full-MoE
tests. The effect-marker API and six CPU/fake affinity checks also pass on
the installed Torch 2.12.1+cu132 environment; this is not a 2.12 CUDA matrix.
The numerical GPU results here use Torch 2.13.0+cu132/CuTe4.7.

The collection intentionally fixes untuned shared-MLP defaults for its
router/expert study. Those can be expensive, so the recorded full-layer BF16
ratios are **not** production first-hit qualification. Neither 5070 full layer
beats BF16; medium experts alone are about 14% lower latency. Remaining work
includes shared-path selection and more held-out candidate diversity, not a
claim that these incremental gains solve all MoE regimes. Statistical
coordinate effects remain exploratory associations with coupled search choices,
not multiplicity-corrected causal pruning rules.

## Evidence and bounded continuation

The [evidence bundle](../autotune_reports/moe_search_and_kernels_20261118/README.md)
preserves frozen collection/final source archives, residuals, CSV/Parquet
exports, held-out studies, audit and test logs. It resides in the existing
Storage4-backed project artifact directory. Original per-feature evidence
remains linked above; derived exports do not replace source journals.

A now-finished HOME GPU0-only continuation was bounded to eight additional visits
and a 45-minute admission window, with a 15-minute in-flight visit timeout.
It resumes the exact collection runtime and journals, moving to mixed and
many-short-pool architecture coverage before deeper budgets. Progress is in
`autotune_reports/moe_search_and_kernels_20261118/continuation.log`.
The laptops are released and HOME GPU1 is untouched. The 204-observation
figures above describe the completed, audited pre-continuation snapshot;
later work requires another audit/export, not an implicit statistical update.
The continuation has exited and has **not** been restarted. Its successful
exit codes did not override the subsequently discovered CUDA faults; its
results remain in the quarantined bundle for diagnosis.
