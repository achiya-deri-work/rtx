# Mixed-pool qualification

The runtime default is unchanged. `candidate_mix="mixed"` is an opt-in
candidate-generation change, not a new numerical policy or a promoted model.
The generic structural generator lives in `rtx._sampling` (Torch-free), with
runtime composition in `rtx.autotune.sampling`.

## Experiment

The follow-up compares four treatments: legacy without frozen predictors,
mixed without frozen predictors, legacy with residual/MLP heads, and mixed
with those identical frozen heads. Each family receives all four treatments
before proceeding; low budgets complete first. Budgets are 8/16/32/64.
Trials include failed compilation and numerical checks. Confirmations and
three post-search verification repeats are recorded separately.

Two shapes across eight families cover MXFP8 forward/backward, retained
dynamic NVFP4 policies, GLU and MLP forward/training. This follows up on
previously examined shapes: it is not a newly unseen-shape evaluation.
Independent task journals and atomic result files make partial runs useful.
Existing results are not overwritten on resume. Compiler caches are shared;
these measurements do not represent fully cold compiler-cache startup.

## Local reproduction

```bash
.venv/bin/python -m tools.predictor_study plan \
  --output autotune_datasets/mixed_pool_local_v1 --mixed \
  --replicates 1 --budgets 8 16 32 64 --task-seconds 120
bash run_mixed_pool_study.sh autotune_datasets/mixed_pool_local_v1 2
```

Plan creation needs the historical source inventory for its geometry audit.
Do not recreate a plan that already exists; simply launch to resume it.

## Other machines

Use the identical source revision, copy only the frozen `plan.json` and the
frozen `autotune_priors/tournament_predictors_v1.json.gz` artifact, then:

```bash
bash run_mixed_pool_study.sh autotune_datasets/mixed_pool_laptop_v1 2 /path/to/plan.json
# Use a different output directory for the 5090.
```

The launcher uses nohup/disown and prints its PID and log. Artifact relocation
is allowed but its SHA256 must match; source changes invalidate resume. Do not
copy the local machine's task results or machine identity into the remote output.
No historical raw dataset is required for replay. These changes must be
published/synchronized before a remote checkout can run this launcher.

## Related CPU study

`tools.research.balancing_study` compares equal-context weighting against
ordinary fitting for histogram boosting and a small MLP. It uses only the
older training/holdout inventory, preserves existing quarantines, and writes
one atomic residual per fit. It neither ingests this GPU study nor installs
a runtime model. Repeated historical holdout analysis is exploratory evidence.

Promotion requires repeatable improvements by budget and family, acceptable
worst-case regression and first-hit cost, and laptop/5090 confirmation.
Broader categorical coverage alone is not a deployment criterion.
