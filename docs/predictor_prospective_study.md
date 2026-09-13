# Prospective first-hit predictor comparison

This study freezes the exported residual/MLP artifact and compares real GPU
searches, not retrospective ranking over the configurations that happened to
be measured previously. It never trains a model or installs runtime winners.

## Treatments and budgets

Four treatments use the shared `rtx.autotune` orchestrator and existing numerical
evaluators:

- `existing`: prior effort policy, without the new reserved random pull.
- `retrieval`: historical-winner prefix, random exploration and local search;
  no learned ranking. This is a retrieval-backed control, not pure nearest-neighbor search.
- `residual`: the same control plus a residual-tree UCB arm.
- `residual_mlp`: the same control plus both learned UCB arms.

Each 8/16/32-trial budget is an independent search, not a prefix of a 32-trial
run. Failed attempted configurations consume trials. Time-limited cells are
reported as partial, not credited as having finished their trial budget.
The new default opening uses one retrieved candidate at 8 trials, two at 16,
and three at 32; it also reserves one random attempt. One opportunity per other
arm is preserved when limiting the prefix. Deadlines, static exhaustion and
duplicate proposals can prevent a reserved attempt from executing.

`AutotunePolicy(exploration_trials=0)` disables the explicit random reservation;
`retrieval_trials=0` disables the prefix. Empty `predictor_heads=()` selects
retrieval without learned ranking. User controls are not a guarantee of speedup.

## Matrix and scope

The default plan uses two unseen `(M,K)` pairs and two independent seeds. Plan
creation reads every observation—including failures—from both older training
and previously inspected evaluation datasets, refusing a previously seen pair.
This is stricter than requiring only a new `(M,N,K)` shape.

Tasks cover dynamic MXFP8 forward, joint dX/dW backward, NVFP4 block-only,
tensor-delayed and JIT-region forward, BF16-gated MLP forward/training, and
standalone MXFP8-output GLU. The artifact contains no standalone GLU head: GLU
is an explicitly reported fallback control. The artifact's compatible MLP
policy is `bf16_gate`, not the fully MXFP8-gated policy. Tensor-current and
packed inference are not included in this first dynamic-workload matrix.
This first study is hot-cache only; it does not establish rotating-cache wins.

Each task uses a fresh process. Persistent compiler caches remain shared, so
compilation time is warm-cache/mixed-cache evidence, not a guaranteed cold
installation measurement. Treatment order rotates across contexts/replicates.
The low-budget wave precedes higher budgets. GPU measurement is serialized.
CPU BLAS and PyTorch threads are capped at four to avoid oversubscription.
Workers also perform an artifact compatibility probe outside the timed search.
Its time is recorded; total process time includes that diagnostic overhead.
The loader cache is cleared afterward, so learned-policy setup includes loading
the artifact. This represents a cold predictor, not its cheaper reuse on later
shapes in one application process.

The calibrated linear harness uses seven samples, 2 ms target batches and
5–128 calls per sample; MLP/GLU use seven graph-replay samples of five calls.
All treatments within a family use the same protocol. Compilation, setup,
proposal generation and confirmation have separate records from kernel timing.
After search, each selected winner is measured independently three more times.
Those repeats do not influence selection, and verification failures are retained.

## Run and resume

The artifact, source audit manifest and source datasets must exist locally.

```bash
.venv/bin/python -m tools.predictor_study plan \
  --output autotune_datasets/predictor_prospective_v1 \
  --shapes 2 --replicates 2 --task-seconds 120

.venv/bin/python -m tools.predictor_study run \
  --output autotune_datasets/predictor_prospective_v1 \
  --device cuda:0 --hours 2

.venv/bin/python -m tools.analyze_predictor_study \
  autotune_datasets/predictor_prospective_v1
```

Repeat `run` to continue unfinished tasks. A task without a final result resumes
its durable journal; finalized failures are retained rather than automatically
rerun until they succeed. Use a new study for changed sources or artifacts.
The source/model hashes and machine identity prevent accidental mixed runs.
Only one supervisor can own an output directory. Use distinct outputs per GPU.
Time limits stop admission between tasks; an active task may overshoot the
overall deadline. A worker has a timeout of its tuning allowance plus 180 seconds.

For a detached launch of an existing plan, with `nohup`, `disown`, a timestamped
log and automatic final analysis:

```bash
bash run_predictor_study.sh autotune_datasets/predictor_prospective_v1 2 cuda:0
```

Each task has a log, append-only residual journal, result JSON and process timing.
CSV/Parquet exports are rebuildable. Keep this directory excluded from training
inputs until the policy decision has been made. No automatic ingestion occurs.

## Interpreting results

Use matched complete cells for speed comparisons, and inspect failures/partial
coverage alongside them. Report verified winner latency, tuning wall time,
time to first valid candidate, compile failures, wasted compilation, model
loading/proposal time, confirmation cost and tail regressions.

"Observed regret" is relative to the best verified winner within the matched
four-treatment cell, not the unknown global optimum. Geometric mean speedup
is paired; exploratory bootstrap intervals cluster on geometry. Two geometries
cannot establish broad generalization. Incompatible model contracts are excluded
from learned-policy aggregates. Do not promote a default based only on the
average, one device, or one narrow shape set.
