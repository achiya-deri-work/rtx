# Portable prior and four-hour prospective study

## What is ready

`rtx.autotune.prior.FrozenPrior` combines transferred schedules, a frozen
context-centered latency ranker, an optional feasibility score, a preparation
cost proxy, and explicit diversity slots. It operates on the generic kernel
adapter interface. It only orders a bounded pool; it never returns an unmeasured
runtime winner or changes a kernel's numerical policy.

The supplied [research artifact](../autotune_priors/README.md) has 24 bundles and
is about 2.1 MB compressed. Research loading is explicit. No production default
or runtime winner cache was changed.

```python
from rtx.autotune.prior import FrozenPrior

prior = FrozenPrior.load("autotune_priors/blackwell_prior_v1.json.gz", research=True)
retrieved = prior.retrieval(adapter)
ordered = prior.order(adapter, legal_candidates, policy="hybrid",
                      mix=(2, 4, 2), feasibility_penalty=0.25,
                      preparation_penalty=0.05)
# Benchmark and validate the chosen prefix; do not assume ordered[0] is a winner.
```

The feature view uses adapter-supplied configuration/workload/device geometry,
while excluding measured register/spill telemetry and software provenance.
Current semantic fields cover RTX precision, phase, activation, operand state,
scaling and cache regime. Porting to a different operation requires defining its
semantic contract too; do not treat attention visibility policies, for example,
as interchangeable merely because their shapes match.

## Composing a short-budget research policy

The [three-device prospective results](prior_prospective_results_20260913.md)
support investigating retrieval-first budgets and family-specific routing.
`StagedPriorPolicy` and `FamilyPriorPolicy` provide this composition without
embedding RTX family names into the generic prior implementation:

```python
from rtx.autotune.prior_policy import (
    PriorStage, StagedPriorPolicy, FamilyPriorPolicy,
)

staged = StagedPriorPolicy((
    PriorStage("retrieval", trials=4),
    PriorStage("hybrid"),  # Retain every remaining candidate; no hard pruning.
))
policy = FamilyPriorPolicy(staged, routes={
    "mxfp8_bwd": StagedPriorPolicy((PriorStage("retrieval"),)),
    "nvfp4_jit_row_region_fwd": StagedPriorPolicy((PriorStage("retrieval"),)),
})
shortlist = policy.order(prior, adapter, legal_candidates, max_candidates=8)
```

This example is a **new research hypothesis**, not a qualified deployment
recipe. Its routing rules were chosen after inspecting the study; evaluate it
on fresh workloads before claiming an unbiased win. No runtime default changes.

The caller still owns legality checks, numerical verification, confirmations,
trial/time budgets, and measured-winner installation. Stages count unique
proposals, not successful trials. Repeated IDs in the supplied pool are rejected.
Without `max_candidates`, ordering retains the complete pool. A budget of four
in the example never invokes the hybrid stage; retrieval itself skips model
deserialization and per-candidate model feature extraction. The frozen artifact
must still be loaded, so this is not a claim of zero first-hit setup cost.
Unknown semantic contracts preserve caller ordering. Family routing does not
bypass kernel revision, numerical-policy, or cache-regime compatibility.

The existing four-hour launcher continues to compare the original five arms.
It does not silently substitute this newly composed policy into a resumed study.

## Development evidence

`autotune_reports/portable_prior_v1/evaluation.json` records all 24 bundle
evaluations with device-disjoint train/test context lists. It includes recorded
failed candidates in trial budgets, removes static rejections as the adapter
would, and omits conflicting outcomes and explicitly unstable successful timings.

Mean regret at eight candidates, relative to each observed pool's best valid
candidate (percent, lower is better):

| Family | Original ordering | Ranker | Hybrid | Random |
|---|---:|---:|---:|---:|
| MXFP8 recomputed MLP | 29.84 | 0.42 | 0.14 | 43.76 |
| MXFP8 backward | 20.10 | 2.87 | 3.11 | 5.76 |
| MXFP8 fused forward | 69.05 | 3.93 | 5.78 | 19.61 |
| MXFP8 prequant forward | 3.53 | 2.54 | 1.86 | 2.27 |
| MXFP8 weight-prequantized | 5.38 | 2.34 | 1.57 | 3.19 |
| MXFP8 fully-prequantized | 5.58 | 3.22 | 2.64 | 3.37 |
| NVFP4 delayed | 2.63 | 1.65 | 1.18 | 1.66 |
| NVFP4 dynamic block | 1.55 | 2.75 | 1.66 | 1.49 |
| NVFP4 JIT-region | 2.42 | 2.21 | 2.27 | 1.71 |
| NVFP4 weight-prequantized | 2.53 | 1.50 | 1.38 | 1.42 |
| NVFP4 fully-prequantized | 3.65 | 5.99 | 3.32 | 2.05 |

Random averages use 20 seeded subsets. Hybrid ordering CPU time in this replay
was about 8.6 ms median / 40.7 ms p95 for the observed pools; this excludes loading
the artifact and generating features/pools and is not a general runtime SLA.

These results justify **soft shortlist pruning**, not permanent removal. For
example, hybrid's eight-candidate shortlist retained a configuration within 2%
of the observed best in 95.9% of MLP contexts, but only 64.2% of JIT-region
contexts. Nor is a single selector universally superior: retain random and
ranker-only treatments in the next experiment.

The model was not trained on its held-out device during replay, but this remains
development data used to choose policies. Adaptive observed pools, shared
compiler caches and sparse failures prevent claims about cold first-hit time,
unseen-candidate coverage, calibrated failure probability, or release quality.

## Launch a fresh four-hour run

From the repository checkout on each machine:

```bash
git fetch origin
git switch feat/mxfp8-glu
git pull --ff-only
bash benchmarks/launch_prior_study.sh 4
```

The script provisions `.venv` using pinned requirements, validates imports/CUDA,
creates a plan, and launches with `nohup`, `setsid` and `disown`. It prints PID,
log and output directory. Installation is outside the four-hour collection clock.
Set `RTX_INSTALL_DEPS=0` to preserve an already validated supported environment,
`RTX_PYTHON` to select its Python, `RTX_DEVICE` to select a GPU, and
`RTX_AUTOTUNE_CPU_WORKERS` to set the CPU-worker cap (default four).

Never update packages/source in an environment with an active collection.

## Experimental design

- 880 tasks: 176 workload contexts × five treatments (`ordered`, `random`,
  `retrieval`, `ranker`, `hybrid`).
- All ten retained linear families plus MLP forward/training in MXFP8 and
  BF16-gate modes. Linear contexts include hot/rotating-cache regimes.
- Eight linear shapes and four MLP shapes include anchor-like and new-shape
  candidates. The cohort label is not a proof of unseen training membership;
  check the frozen source dataset when reporting strict unseen-shape results.
- Families are round-robin interleaved; treatment order rotates deterministically
  by context. Trial milestones are 4/8/16/32/64/128 across all tasks before the next
  wave. If every wave finishes early, new seeded replicates fill remaining time.
- Every treatment within a workload gets the same frozen-prior-enriched candidate
  pool. The prior supplies candidate schedules, never inherited timing values.
  Pool hashes and exact orders are recorded. The orchestrator's initial candidate
  is explicitly the treatment's first choice, so it does not consume a hidden
  default trial before the chosen ordering begins.
- Parallel CPU compilation, conservative VRAM admission, one shared GPU execution
  lease. This does not lock unrelated applications. The supervisor also shares
  the combined collector's device lock to prevent overlapping RTX campaigns.
- JSONL observations and per-task metadata/milestone files are independent.
  Incomplete workers can resume; `reached_target` distinguishes a completed trial
  budget from a slice that ended early. No model updates occur during the study.
- A 60-second outstanding CuTe-compile watchdog and total wall deadline bound
  workers. Short cleanup grace is allowed. Compiler disk caches remain shared:
  elapsed worker time can include lease waits and is **not cold compile latency**.

Restart with the same explicit output directory and an appropriate remaining
hours budget to resume. Restarting grants the supplied duration from that restart;
the launcher does not infer how much of a previous allocation remains. Source,
plan and prior changes require a fresh output directory.

## Inspect results

```bash
python -m rtx.autotune.prior_summary <run-directory-or-zip> \
  --output autotune_reports/prior_prospective
```

This emits CSV/Parquet and a pool-mismatch audit. Conditional fixed-trial regret
compares `budget_reached=true` rows in complete matched policy groups. Report
completion rates, censored workers and timeout rates alongside that comparison;
do not discard unfinished budgets and call the survivors an end-to-end win.
Supervisor journals retain worker start epochs and censored shutdowns, including
workers interrupted at the overall deadline. The oracle is the union of observed
stable valid configurations, not the exhaustive search-space optimum. Runtime
promotion remains a separate, explicit qualification step.

Local smoke evidence: 128 valid observations across MLP forward, NVFP4 packed
weight and JIT-region tasks; matching pools across treatments; continuation into
new replicates at finite-plan exhaustion. Separate native/BF16-gate MLP training
and a two-to-four-trial resume passed existing numerical gates. These are
collector checks, not a full release numerical/performance matrix.
