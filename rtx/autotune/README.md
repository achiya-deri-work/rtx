# RTX composable autotuning

This package separates eight concerns that were previously fused inside each
kernel-specific coordinate tuner:

1. `ConditionalSearchSpace` describes dependent parameters, normalization,
   legality, mutation, and sampling in a backend-neutral schema.
2. `PortableKernelTask` describes workload context, analytical features, and a
   multi-fidelity evaluation plan.
3. `KernelAdapter` remains the compatibility contract consumed by existing
   strategies and synchronous campaigns.
4. `SearchStrategy` proposes candidates. Random exploration, gradient-boosted
   cost-model search, coordinate/beam local search, and strategy pipelines are
   provided.
5. `StrategyScheduler` assigns the next trial. `SequentialScheduler` implements
   staged search, `UCB1Scheduler` remains available as a small baseline, and
   `AdaptiveBanditScheduler` is the production discounted contextual bandit.
6. `AskTellSession` issues serializable leases and accepts out-of-order worker
   responses; its complete optimizer state can be moved between processes.
7. `AutotuneOrchestrator` owns the compatibility synchronous evaluation loop,
   budget, rewards, progress, and session lifecycle.
8. `TuningStore` records sessions, orchestration decisions, and observations.
   The JSONL backend is append-only, locked, flushed after every observation,
   and resumable after interruption.

The existing `CoordinateDescentTuner` API is re-exported unchanged from the
package compatibility module.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `outcomes.py` | Backend-neutral result and failure records |
| `core.py` | Context, proposal, observation, budget, and adapter contracts |
| `space.py` | Declarative conditional parameter spaces and constraints |
| `task.py` | Portable staged tasks, fidelities, and adapter bridges |
| `ask_tell.py` | Serializable requests/responses, leases, promotion, and resume |
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

## Portable project plugin

The smallest external integration uses dictionaries as configurations. A
project may instead implement `SearchSpace[YourConfig]` to retain its own
immutable configuration type.

```python
from rtx.autotune import (
    AskTellSession,
    Condition,
    ConditionalSearchSpace,
    DiscreteParameter,
    EvaluationPlan,
    EvaluationStage,
    FunctionKernelTask,
    KernelContext,
    LocalTrialWorker,
    RandomSearch,
    StageKind,
    StageResult,
    StagedTaskAdapter,
    UCB1Scheduler,
)

space = ConditionalSearchSpace((
    DiscreteParameter("tile", (32, 64, 128), default=64),
    DiscreteParameter("load", ("vector", "tma")),
    DiscreteParameter(
        "stages",
        (1, 2, 3, 4),
        default=2,
        active_if=(Condition("load", "eq", "tma"),),
    ),
))
plan = EvaluationPlan((
    EvaluationStage("compile", StageKind.COMPILE, 0.2),
    EvaluationStage("correctness", StageKind.CORRECTNESS, 0.5),
    EvaluationStage("benchmark", StageKind.BENCHMARK, 1.0),
))

def run_stage(config, stage):
    # Invoke CuTe, Triton, CUDA C++, ROCm, or a project-specific generator.
    # Results and artifact references must be serializable.
    if stage.kind == StageKind.BENCHMARK:
        return StageResult("ok", {"latency_ms": benchmark(config)})
    return StageResult("ok")

task = FunctionKernelTask(
    KernelContext("custom_attention", 1, {"sequence": 8192}),
    space,
    plan,
    run_stage,
)
adapter = StagedTaskAdapter(task)
study = AskTellSession(
    adapter, [RandomSearch()], UCB1Scheduler(), seed=17,
)
worker = LocalTrialWorker(adapter, {"worker_id": "gpu-0"})

request = study.ask()[0]
study.tell(worker.evaluate(request))
```

For a single local GPU with campaign-grade append-only persistence, wrap the
same policy in `DurableLocalAskTellRunner`, or use
`make_hybrid_ask_tell_runner` for the standard random/model/local portfolio.
The runner restores compatible observations from the normal `TuningStore`,
records each issued lease and completed response, and enforces total resumed
trial and wall-time budgets. It is the migration boundary between current
synchronous campaigns and future queue/RPC workers; the optimizer itself does
not change when execution becomes remote.

`TrialRequest.as_dict()` and `TrialResponse.as_dict()` are the transport
boundary for a queue, RPC service, or database. Requests include stable context
and configuration identities, fidelity, strategy provenance, and a renewable
lease. `AskTellSession.state_dict()` includes observations, pending work, RNG,
and arm statistics; `from_state_dict()` restores it after interruption.

Compile or correctness-only work can be requested with `ask(fidelity=0.5)` and
later advanced through `promote(config_id, 1.0)`. A partial successful trial is
retained as such and is not mistaken for either a latency result or a runtime
failure.

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

For cross-machine warm starts, `rtx-autotune pretrain` fits three distinct
revision-scoped heads from directories or ZIP archives:

- absolute log latency residualized by calibrated analytical rooflines;
- configuration ranking centered within each device/workload/cache context;
- feasibility over compile, launch, correctness, and implementation failures.

The ranking and latency heads compete against matched random catalogue replay;
a head drives proposal order only when it wins on every held-out device.
Otherwise both remain diagnostic outputs. Parent-linked local moves also
produce bootstrap-qualified conditional-effect rules. Rules record
their support, confidence interval, contexts, and devices and act only as a
small ranking adjustment. Leave-one-device-out catalogue replay reports regret
after 1, 4, 8, 16, and 32 proposals before an artifact is deployed.

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

For interruption-prone prospective comparisons, manifests may define named
optimizer `treatments`, independent `replicates`,
`storage_mode: residual_context`, and
`rotation_mode: balanced_categories`. The residual store writes locally while
reading transferable sibling contexts from only the same treatment/replicate.
Use `rtx-autotune summarize-tuners` to generate fixed-budget regret and failure
summaries after collection rather than maintaining a monolithic dataset during
the run.

Sticky accelerator failures use a supervisor boundary. A worker durably records
the responsible proposal, raises `FatalDeviceContextError`, and the CLI exits
75. Supervisors may restart that code only; ordinary Python, manifest, and
dependency failures remain terminal. This prevents one illegal kernel from
turning all later contexts into correlated setup failures.

For a new campaign, `--reuse-deterministic-failures` avoids recompiling exact
known static/compiler failures across cache regimes. Its fsync'd JSONL ledger is
scoped by architecture, compiler, kernel revision, workload, configuration,
treatment, and replicate. The flag is deliberately opt-in so it cannot change
the sampling distribution of an already-running prospective comparison.

## Recorded schema

Every observation includes:

- kernel family and revision;
- device, software environment, workload, cache regime, and tags;
- exact serialized configuration and stable ID;
- sparse raw and derived model features;
- strategy, parent configuration, coordinate, and acquisition metadata;
- candidate-pool size/rank and the available proposal-probability semantics;
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
