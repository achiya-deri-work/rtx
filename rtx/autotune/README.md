# RTX composable autotuning

This package separates five concerns that were previously fused inside each
kernel-specific coordinate tuner:

1. `KernelAdapter` describes configuration identity, serialization, legality,
   features, mutation, sampling, and evaluation.
2. `SearchStrategy` proposes candidates. Random exploration, gradient-boosted
   cost-model search, coordinate/beam local search, and strategy pipelines are
   provided.
3. `StrategyScheduler` assigns the next trial. `SequentialScheduler` implements
   staged search, `UCB1Scheduler` remains available as a small baseline, and
   `AdaptiveBanditScheduler` is the production discounted contextual bandit.
4. `AutotuneOrchestrator` owns budget, deduplication, evaluation, rewards,
   progress, and session lifecycle.
5. `TuningStore` records sessions, orchestration decisions, and observations.
   The JSONL backend is append-only, locked, flushed after every observation,
   and resumable after interruption.

The existing `CoordinateDescentTuner` API is re-exported unchanged from the
package compatibility module.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `core.py` | Context, proposal, observation, budget, and adapter contracts |
| `bandit.py` | Reusable arm state, UCB policies, rewards, and contextual scoring |
| `strategies.py` | Random, learned-global, coordinate, and model-local proposals |
| `cost_model.py` | Small latency and feasibility gradient-boosted models |
| `orchestrator.py` | Single-context evaluation loop and strategy routing |
| `store.py` | Append-only JSONL and in-memory persistence |
| `dataset.py` | Multi-context campaign harnesses, verification, and CLI |
| `dataset_export.py` | CPU-only bundle normalization and CSV/Parquet export |
| `adapters.py` | RTX kernel configuration spaces behind the generic contracts |
| `hardware.py` | Architecture/SKU features and resource estimates |
| `legacy.py` | Compatibility coordinate tuner; no new orchestration code |

Policy mathematics must not import dataset harnesses or kernel implementations.
Campaign code composes public policies and adapters rather than embedding a
second private tuner.

## Default learned-global then local composition

```python
from rtx.autotune import (
    DeviceFingerprint,
    HybridTuningPolicy,
    JsonlTuningStore,
    make_hybrid_autotuner,
    make_mxfp8_bwd_adapter,
)
from rtx.bwd_autotune import MXFP8BwdEvaluator

evaluator = MXFP8BwdEvaluator(grad_output, x, weight, measurement_policy)
adapter = make_mxfp8_bwd_adapter(
    evaluator.problem,
    evaluator,
    device=DeviceFingerprint.current(x.device),
    regime="hot",
)
tuner = make_hybrid_autotuner(
    adapter,
    JsonlTuningStore("autotune_results/unified"),
    HybridTuningPolicy(
        orchestration="sequential",
        cost_model_trials=320,
        max_trials=512,
    ),
    progress=print,
)
result = tuner.tune()
```

The learned stage uses random-walk warmup, then trains a small bagged
gradient-boosted CART ensemble on log latency and ranks a broad candidate pool
with a lower-confidence-bound acquisition. A separate classifier learns from
`compile_error` observations and estimates compilation feasibility. Its
optimistic probability adjusts ranking but never hard-prunes an uncertain
region. The local stage explores complete coordinate neighborhoods around the
best measured configurations and refits both shared models as labels arrive.

## Architecture and SKU features

Dataset campaigns pass a complete SM120/SM121 hardware profile into every
kernel context. It separates ISA facts, physical SKU limits, and optional
empirical calibration. Model features include actual/persistent
grid CTAs, wave fullness at estimated multi-CTA residency, SMEM/register/thread
occupancy limits, logical-transpose quantizer resources, instruction-work
proxies, operand traffic, L2 fit/reuse, memory bus width, theoretical bandwidth,
and measured rooflines. Requested register budgets are estimates; compiled
resource attributes are attached to outcomes whenever the active CuTe wrapper
exposes them.

Generate and use a portable calibration with:

```bash
rtx-autotune calibrate --device cuda:0 --output hardware_calibration.json
rtx-autotune run manifest.json --device cuda:0 \
  --calibration hardware_calibration.json
```

Set `confirmation_repeats` to make apparent incumbents earn promotion through
additional independent evaluator runs. Their raw samples are merged into one
median and every screen/confirmation outcome is retained in observation
metadata. `confirm_initial=True` gives the session a stable starting score;
`confirmation_ratio` can also confirm near-ties rather than only apparent
wins. Confirmation is disabled by default so existing integrations retain
their measurement cost.

For online strategy allocation, set `orchestration="bandit"`. Random, learned,
and local strategies become discounted-UCB arms. The scheduler performs an
explicit random/model warmup, forces a real active-context sample from every
arm, then allocates by bounded incumbent improvement, validity, information
gain, evaluator cost, and uncertainty. Similar workload/device observations
are capped virtual pulls: they influence cold-start ordering but cannot replace
a measurement on the current GPU and shape.

Bandit state is reconstructed from append-only observations on every resume;
it is not reset at process or context-slice boundaries. Every decision records
scores, effective discounted counts, cumulative counts, rewards, cooldowns,
and transfer priors in `events.jsonl`.

Time-budgeted dataset campaigns can compose a second bandit over workload
contexts. It first reaches a configurable coverage floor, then uses contextual
discounted UCB to decide which shape/family/cache regime receives the next
slice. A milestone-lead bound prevents starvation. Its decisions are durable
in `context_allocations.jsonl` and are included in CSV/Parquet exports.

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_bandit_v1.json \
  --device cuda:0 --output-dir autotune_datasets --format both \
  --wall-time 12h --context-slice 2m \
  --context-orchestration bandit
```

To continue an existing manifest without changing its digest, add
`--strategy-orchestration bandit --strategy-bandit-exploration 0.35`; the
override is recorded in the run's anytime policy. Use
`--adopt-existing-context-identity` only for a compatible runner-only upgrade
as described in the repository README.

## Recorded schema

Every observation includes:

- kernel family and revision;
- device, software environment, workload, cache regime, and tags;
- exact serialized configuration and stable ID;
- sparse raw and derived model features;
- strategy, parent configuration, coordinate, and acquisition metadata;
- status for rejection, compile, correctness, runtime, or success;
- raw timing samples, median, compilation latency, and numerical error;
- session sequence, timestamps, and full evaluator duration.

Strategy-selection events and context-allocation rows store the state and score
of every scheduler arm. This makes offline replay and counterfactual
orchestration analysis possible.

When `transfer_history=True`, the cost model receives observations from other
contexts with the same family and kernel revision. Deduplication and incumbent
selection remain context-local, so a configuration measured on a 5090 is still
benchmarked on a 5070 before it can win there.

## Adding another kernel

Implement `KernelAdapter`, or instantiate `DiscreteKernelAdapter` with the
kernel's axes and existing evaluator. No strategy, scheduler, model, or store
code needs to change. The provided adapter factories cover:

- fused MXFP8 forward;
- dynamically prequantized MXFP8 forward;
- the complete MXFP8 backward pair.

Standalone quantizers/GEMMs can use the same generic adapter with their own
problem context and evaluator.

Historical atomic-JSON databases, including the backward tuner's nested
outcome schema, can be imported with
`import_legacy_json_database`. Imported trials use a transfer-only context, so
they train the model without preventing fresh local measurements. For stable
sub-percent comparisons, `CalibratedPrequantEvaluator` and
`CalibratedBwdEvaluator` connect the generic engine to calibrated multi-call
harnesses with paired-race verification.
