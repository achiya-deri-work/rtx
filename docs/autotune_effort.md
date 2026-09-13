# User-controlled autotuning effort

`rtx.AutotunePolicy` composes the existing `rtx.autotune` evaluators,
residual journals, strategies, confirmation logic, and cost-aware contextual
UCB scheduler. It does not introduce another timing engine.

Both linear modules use this policy by default on a runtime cache miss,
including when the first invocation is through `torch.compile(fullgraph=True)`.
You do not need the explicit policy below unless you want to change effort.

```python
import rtx

policy = rtx.AutotunePolicy.preset("balanced", max_trials=16)
layer = rtx.MXFP8Linear(
    768, 3072, bias=False, device="cuda",
    autotune="balanced", tuning_policy=policy,
    autotune_cache_dir="autotune_cache",
)
# NVFP4Linear accepts the same tuning_policy argument.
```

Presets are quick (8 trials / 15 seconds), balanced (16 / 30), thorough
(32 / 90), and extensive (64 / 180). These are effort limits, not promises
of a particular performance level. Explicit values override presets.
`forward_trials` and `backward_trials` optionally override the per-pass cap.
Joint MLP training tuning uses the backward override. GLU and MLP explicit
tuning functions accept `tuning_policy=policy`; tune them before capture or
compilation and install the returned configuration using their existing API.

## Budget and cache contract

- Trials count distinct attempted configurations, including compile or
  correctness failures. Static legality rejects do not consume trials.
- The cap includes compatible resumed observations. Explicitly tuning at
  8, then 16, allows eight additional attempts rather than sixteen.
- Confirmation repeats are additional measurements of the same candidate,
  not additional distinct trials. They cost wall time and are recorded.
- Time is a per-invocation **soft admission limit**, including setup and
  proposal generation. An admitted compile/measurement/confirmation is not
  interrupted, so actual elapsed time may exceed it.
- Runtime winner hits bypass tuning. Merely raising a layer's policy does
  not force refinement of an existing runtime winner. Use the explicit
  family tuning function to refine its journal, then install its result.
- Measurement settings are part of history identity; changing those can
  start a separate study. Effort limits themselves do not split history.
- Restored candidates are checked against current static legality. Old
  quarantined winners cannot silently become the returned winner.

## Search composition

Start with the known initial configuration. Reserve one random-exploration
attempt by default (`exploration_trials=0` disables this). When a compatible frozen prior
is explicitly supplied, try a small retrieval prefix, then let UCB allocate
the remaining attempts among frozen ranking, coordinate/local search, and
random exploration. Without a prior, larger budgets also enable the
existing lightweight online boosted cost model. Mandatory arm warmup
blocks are disabled so they do not swallow small budgets.

`pool_size`, `retrieval_trials`, `exploration_trials`, `exploration`, `seed`, timing samples, and
confirmation repeats are configurable. Research artifacts require both
`prior_artifact=...` and `allow_research_prior=True`. No new research model
is installed implicitly. Unknown prior contracts fall back to ordinary
search. This policy is supported by the synchronous tuning entry points;
it is not a replacement for the separate ask/tell runner policy API.

The current defaults use fixed screening measurement settings plus
contender confirmation—not a newly calibrated adaptive sequential timing
algorithm. CUDA smoke tests validate integration, not universal convergence
of these timing statistics.

See [the empirical ML study](effort_ml_study_20260913.md) for what the
collected data does and does not justify.

## Portable research predictors

### Mixed candidate pools (experimental)

Set `candidate_mix="mixed"` on `AutotunePolicy` to reserve candidate-pool
slots for broader structural exploration alongside retrieval and local walks.
The default remains `"legacy"` pending matched multi-device evaluation.
The mixed pool reserves approximately one quarter each for retrieval and
structural candidates, with remaining slots used by local candidates. Empty
quotas are backfilled. Every candidate must pass current static legality;
the adapter's active-schedule identity removes duplicates across sources.

`structural_proposals=512` bounds construction attempts, not GPU trials.
Coupled assignments may cross illegal intermediate states, but only legal
complete configurations can reach evaluation. One structural attempt replaces
the reserved random attempt when a structural pool is available. UCB controls
the remaining allocation, including local/random and optional learned arms.
The pool's origins, counts and rejection summary are recorded in session metadata.
Construction and ranking still cost time; diversity does not guarantee speed.

Explicit searches sharing a residual and exact context now serialize the
read/search/write transaction on Linux, preventing duplicate cumulative-budget
spending by concurrent callers. This is separate from runtime-winner locks.
Different residual directories intentionally remain independent experiments.

The exported residual-tree and small-MLP heads can participate as separate
UCB proposal sources. Inference uses NumPy only: no sklearn, XGBoost, Torch
model, pickle, or extra GPU work is needed to rank candidates.

```python
policy = rtx.AutotunePolicy.preset(
    "balanced", max_trials=16,
    predictor_artifact="autotune_priors/tournament_predictors_v1.json.gz",
    predictor_heads=("residual", "mlp"),
    allow_research_prior=True,
)
layer = rtx.NVFP4Linear(
    768, 3072, bias=False, device="cuda",
    tuning_policy=policy, autotune_cache_dir="autotune_cache",
)
# The first real torch.compile(fullgraph=True) invocation can tune both passes.
```

This is opt-in, not a promoted default. The opening trials include the initial
configuration and nearest-shape historical winners; UCB then selects among
residual ranking, MLP ranking, coordinate/local search and random exploration.
Historical-winner retrieval uses shape distance, not the tournament's separate
KNN predictor. Hardware/configuration interactions enter the learned rankings.
When both artifact types are supplied, `prior_artifact` supplies retrieval.

Family, kernel revision, cache regime and workload/scaling policy must match.
Missing campaign-only policy tags are completed only when exactly one trained
contract remains compatible. Unknown or ambiguous contracts fall back to ordinary
search; malformed artifacts and absent research opt-in raise an error. Matching
a contract does not certify transfer to an unseen GPU. The low-level predictor's
`predict` method only scores rows; adapter-facing `order` performs these checks.

Both heads rank a bounded pool once per search. Their proposals still undergo
normal legality, numerical, timing and confirmation checks. A bad prediction
can waste a trial but cannot replace the incumbent merely because of its score.
Artifact hashes and head names are recorded with observations. Loading an
artifact does not invalidate existing runtime winners or force retuning.

See [export and CUDA validation evidence](portable_predictor_validation.md)
for reproduction, known limitations and the next qualification gate.

The [prospective comparison harness](predictor_prospective_study.md) tests
8/16/32-trial policies on previously unobserved geometries without training on
the resulting observations.
