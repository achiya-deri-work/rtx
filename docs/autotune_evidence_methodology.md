# Autotuning evidence methodology

Autotuning observations are not independent random samples. A bandit or local
search deliberately measures promising neighborhoods more often, configurations
are coupled, and the same schedule can have very different absolute latency on
different GPUs. This document defines the evidence hierarchy used by RTX.

## Evidence hierarchy

From strongest to weakest:

1. Prospective evaluation on a campaign excluded from model and policy design.
2. Complete-shape-held-out catalogue replay with matched random baselines.
3. Exact parent→child comparisons inside the same context.
4. Context-normalized associations with whole-context bootstrap intervals.
5. Raw marginal configuration frequency or latency correlations.

Only the first two can authorize a learned latency/ranking head. Deterministic
correctness failures can authorize a revision-scoped legality rule after a
boundary regression test reproduces the contract. Lower evidence levels may
change proposal order but must preserve exploration.

Pretrained trainer revision 3 applies the same complete-shape separation to
exact-SKU latency/ranking heads. Revision-2 artifacts fail closed and must be
retrained; their context-fold scores may contain hot/rotate or replicate leakage.

## Paired mutation analysis

Coordinate and model-local strategies record the parent configuration, declared
coordinate, and child value. The evidence study resolves both configurations
and measures

```text
log(child latency / parent latency)
```

inside the exact device, shape, cache regime, and campaign context. Results are
aggregated by context before bootstrapping so a heavily sampled context cannot
dominate the interval. Moves which alter multiple serialized fields are marked
as coupled/composite and are not presented as single-coordinate evidence.

## Pairwise ranking

The pairwise model asks whether configuration A beats configuration B inside the
same context. Its features contain the shared hardware/workload context, both
configurations, and signed differences in config and derived resource features.
Every training pair is emitted in both orientations.

Evaluation hashes complete `(family, revision, M, N, K)` groups into folds. All
devices and cache regimes for a held-out shape remain outside training. Exact-SKU
models use the same shape separation within one device. The model ranks a
catalogue by tournaments against deterministic reference configurations.

Deployment requires the model's median and p90 regret to beat a matched random
catalogue at the declared trial budget in at least 75% of folds. Accuracy, AUC,
or average prediction error cannot override this gate.

## Archetypes and bottlenecks

Schedules are summarized along independent, interpretable dimensions:

- shallow, balanced, or deep pipeline;
- SMEM-heavy, register-heavy, or occupancy-friendly resource use;
- single, few, or many persistent waves;
- scale transport and epilogue strategy;
- fine, middle, or coarse regional scaling geometry.

A transparent diagnostic classifier identifies launch/grid underfill, resource
pressure, memory/scale traffic pressure, tensor-core/compute pressure, or a
balanced roofline. This is not measured hardware-counter attribution. Archetype
effects are normalized within exact contexts and are used to choose promising
search neighborhoods, never as legality rules.

## Failure mining

Failures retain their distinct types: architecture rejection, implementation
rejection, compilation, runtime, correctness, and numerical-contract failures.
The study reports configuration and tail-feature risk ratios separately for
each family and SKU. A high risk ratio can reveal a missing packed-tail or scale
indexing contract, but correlated coordinates remain possible. Before turning
it into static rejection, add deterministic tests on both sides of the proposed
boundary and scope the rule to the affected kernel revision.

## Timing convergence

Prefix medians are compared with the full retained timing sequence. Reports
include all available candidates plus fixed `common_15` and `common_20`
cohorts. The recommendation uses `common_20` when it retains at least five
contexts and otherwise falls back to `common_15`. A fixed cohort is required
when comparing sample counts because adaptive measurement gives noisy or
competitive candidates more samples.

The relevant metrics are p90 median error and the regret of the winner selected
by each prefix—not just variance of individual measurements. Screening,
confirmation, and final paired races can therefore use different budgets.

## Runtime policy

The safe default policy is:

- apply deterministic legality before compilation;
- use qualified feasibility predictions next;
- warm-start from exact-SKU winners;
- use portable conditional rules and archetypes to order candidates;
- use a latency/ranking head only when its matching deployment scope passed;
- retain 15–25% unconstrained exploration;
- revalidate cached winners after compiler, kernel revision, or hardware changes.

Generated evidence reports and model artifacts remain ignored build outputs.
The source datasets are immutable inputs and should be audited before analysis.
