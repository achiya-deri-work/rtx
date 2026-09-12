# First-hit autotuning and the three-GPU anchor study

This is an **opt-in first-hit path**, not a newly enabled library-wide default.
The selector works with any existing `KernelAdapter`; the initial matched
campaign targets recomputed MXFP8 MLPs, native and BF16-gate precision, forward
and training. Existing linear-family campaign observations can also be used by
the offline evaluator. This campaign does not yet collect NVFP4/linear families.

## Use the bounded selector now

```python
mlp.autotune_(example_x, search="balanced", max_trials=8,
              time_budget_s=30, output_dir="autotune_results/my_mlp")
compiled = torch.compile(mlp, fullgraph=True, dynamic=False)
```

`search="balanced"` caps measurements at eight configurations and the search
budget at 30 seconds. It bypasses broad coverage planning and online model
fitting. It measures a baseline, retrieved configurations from nearby contexts,
and complete structural schedules. Every candidate uses the existing numerical
checks and graph replay timing. A compatible cached winner is returned without
evaluating candidates again. The module installs that winner for the example's
shape/device/dtype, using its existing bounded shape-policy map.

This is explicit preparation before compile/capture. An unseen shape in an
ordinary module forward still uses the constructor policy; **it does not yet
automatically invoke this tuner from an Inductor first-compilation hook**.
Adapter construction currently creates validation tensors before the selector
checks its persistent cache. The selector's cache path itself does no GPU
evaluation. Steady-state module forwards do neither.

The wall-time cap is checked between candidates. A compiler invocation may
overrun it. The offline campaign provides a process watchdog; the in-process
runtime selector cannot safely preempt a compiler or hung GPU launch.

Use `ranker_path="...json"` to supply a qualified frozen model. Unqualified
artifacts are rejected by default. Precision and scaling semantics never change
through ranking. No model is trained on the first-hit path.

## Collect matched data on all three devices

For the combined **six-hour MLP + linear** campaign:

```bash
bash benchmarks/launch_combined_collection.sh 6
```

This alternates 15-minute foreground collection slices, never launching the two
GPU collectors concurrently. It persists the next category and resumes each
dataset independently. Once MLP anchors finish, remaining time goes to linear
collection. The linear side uses the existing 10-family diversity atlas:
MXFP8 dynamic/packed forward and backward, NVFP4 block/delayed/JIT-region
forward, and packed inference. Its adaptive observations are supplementary
data, not the matched fixed-portfolio MLP experiment.

Output defaults to `autotune_datasets/combined_first_hit_v1_<hostname>` with
`mlp/`, `linears/`, per-visit logs, and an append-only visit journal. The launcher
uses nohup/disown and prints its PID/log. Keep the same source revision when
resuming; use a fresh output directory after source changes. The total deadline
includes startup and collection but permits a short shutdown grace. Export is
explicit afterward so it cannot consume the GPU collection time budget:

```bash
python -m rtx.autotune.first_hit_study export <combined-output> \
  --output autotune_reports/combined_export
```

For an MLP-only run, use the following launcher instead.

Use the **same source tree** on the RTX 5070 Ti, RTX 5090 and laptop 5070:

```bash
bash benchmarks/launch_first_hit_anchors.sh 4
```

The first argument is hours (decimal values are supported). Optional second
argument selects the output directory. Default output is
`autotune_datasets/first_hit_anchors_<hostname>`.

The launcher uses `nohup` and `disown`, printing the PID and log path. Set
`RTX_PYTHON` to choose a virtual environment, or `RTX_DEVICE` to choose CUDA
device. `RTX_GIT_PULL=1` requests `git pull --ff-only` before starting; it is
off by default to protect dirty/running experiments. Do not update source files
during a campaign. No git commit/push is performed by the launcher.

The manifest contains eight shapes × two precision policies × two phases:
32 contexts. Milestones are 4/8/16/32/64 trials. Each milestone visits all
contexts before the next; precision and phase alternate within a shape.
Candidates are generated deterministically **before device legality checks**.
This preserves matching configurations across devices, including rejection
evidence. Each anchor records its portfolio hash, source hash and software/GPU
context. Export rejects mismatched portfolios for an otherwise matched task.

Each context has independent residual journals and worker logs. The supervisor
uses an output-directory lock and a total deadline. Up to four isolated CPU
workers prepare candidates. A conservative VRAM estimate limits admission;
large contexts may run alone. The worker wall-time watchdog is 180 seconds
times the requested concurrency; an outstanding CuTe compilation has a separate
60-second watchdog (`RTX_AUTOTUNE_COMPILE_TIMEOUT`). A failed worker does not
poison the next context's CUDA process. Only owned worker groups are terminated.
Three consecutive worker failures stop the campaign for inspection. SIGTERM
to the supervisor is handled so its active workers are also stopped.

`RTX_AUTOTUNE_CPU_WORKERS` controls concurrency (default four). A worker-only
hook releases a shared GPU execution lease around `cute.compile`, after draining
that worker's stream work. Tensor creation, Torch tracing, correctness checks,
graph capture and measurement hold the lease. Thus compilation can overlap
compilation or another worker's measurements without concurrent benchmark
workloads. This is not a system-wide lock against unrelated GPU applications.
MLP winner confirmations reuse the last validated CUDA graph instead of
recreating and recompiling it; observations record `prepared_benchmark_reused`.

The combined collector also pipelines linear collection through independent
context shards. Its concurrency is capped at one worker per eight GB of VRAM
(two on a 16 GB card, four on a 32 GB card), with the same GPU lease. Linear
shards currently use the overall slice deadline, not the MLP compile watchdog.
Changing this shard count changes the residual shard layout: old data remains
available but is not automatically repartitioned. CPU compilation still costs
time; this pipeline does not guarantee continuous GPU utilization.

Rerun the same command/output to resume. Manifest/source changes require a new
output directory. For reviewed execution-only source changes, the combined
collector's explicit `--adopt-source` preserves source history while requiring
the workload and portfolio identities to match. Do not use it to mix changed
kernel semantics into an existing experiment. Completed candidates are skipped by the existing residual
orchestrator; abandoned attempts are excluded on recovery. The campaign exits
when its finite plan is exhausted, even if time remains. It does not silently
start an unbounded search or repeat measurements as additional independent data.
Do not run another benchmark on the same GPU concurrently; the output lock
does not arbitrate other applications or campaigns using another directory.

## Copy, export, and compare models

Copy each output directory here, or ZIP the whole directory including
`anchor.json`, journals and metadata. ZIPs are read without extraction.

```bash
python -m rtx.autotune.first_hit_study export \
  <5070ti-bundle> <5090-bundle> <laptop-bundle> \
  --output autotune_reports/first_hit_export

python -m rtx.autotune.first_hit_study fit \
  <5070ti-bundle> <5090-bundle> <laptop-bundle> \
  --split shape --output autotune_reports/first_hit_shape

python -m rtx.autotune.first_hit_study fit \
  <5070ti-bundle> <5090-bundle> <laptop-bundle> \
  --split device --output autotune_reports/first_hit_device
```

Export writes CSV and Parquet with raw outcome/config/features JSON columns.
The original journals remain authoritative and contain timing samples,
compilation/validation cost, numerical errors and failures.

The builtin backend uses our dependency-free context-relative boosted trees.
For the optional XGBoost pairwise-ranking comparator:

```bash
python -m pip install -r requirements-first-hit.txt
python -m rtx.autotune.first_hit_study fit <bundles...> \
  --backend xgboost --split shape --output autotune_reports/first_hit_xgb
```

XGBoost/scikit-learn are optional offline dependencies, not prerequisites for
the builtin runtime. An XGBoost artifact currently also requires XGBoost to
load it; builtin artifacts load without it. Artifacts are JSON, not pickle.

## What the evaluator does—and does not establish

It separates families, revisions, categorical workload policies and benchmark
tags. Shape holdouts keep M/N/K/H/O together across devices and replicates.
Device holdouts use numeric hardware identity. It compares learned ranking,
nearby-context retrieval, ordered-portfolio and random baselines at 1/2/4/8
trials, relative to the best **observed** candidate in each context.

Measured compiler registers/spills and software identity are excluded from
the ranker's pre-compilation inputs. Hardware metadata includes SM count,
memory capacity, architecture/SKU properties and bus width where available.
MLP features add per-role tile counts, tails and single-residency wave estimates;
these are estimates, not measured occupancy or calibrated throughput.

The initial evaluation is **offline replay among successfully observed
candidates**, not a prospective first-hit benchmark. It cannot account for
unmeasured candidates, cold compiler caches, or failed candidate selection.
It reports insufficient groups rather than fitting and testing the same group.
It does not yet supply shape-family holdouts, bootstrap context confidence
intervals, compile-cost/feasibility ranking, or a process-wide tuning budget.

Training writes frozen JSON rankers when anchor contract metadata is present,
but all carry `deployment_enabled=false`. A prospective campaign on unseen
contexts is still required before enabling them. No automatic deployment gate
is implemented yet. Keep the held-out prospective observations separate from
the data used to choose/refit the winning model.

## Local checks

The native and hybrid four-trial runtime smoke tests passed the independent
forward/backward numerical gates, and repeated calls returned cached winners
with zero candidate evaluations. The native anchor worker also passed four
trials. CPU tests cover bounded pools, cache reuse, artifact gates, feature
filtering, grouped evaluation, and CSV/Parquet export. The optional XGBoost
installation could not be tested here because PyPI DNS resolution failed.

Final checks: 44 CPU tests and 13 CUDA frontend tests passed. A real 43-second
supervisor smoke run produced 24 passing candidate observations, one static
implementation rejection, and stopped its final worker at the deadline.
Smoke datasets are not sufficient to qualify a learned policy.
