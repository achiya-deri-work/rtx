# First-hit autotuning: use and current evidence

The first real input can tune a new workload, persist the winner, and reuse it
on later calls or in a fresh process. This works through the tested
`torch.compile(fullgraph=True)` linear paths; no separate tuning call is needed.
Compilation and first backward can each incur setup cost.

## Start without a learned model

```python
import torch
import rtx

policy = rtx.AutotunePolicy.preset(
    "quick",                         # 8 cumulative trials per pass/context
    search_schedule="progressive",
    cache_behavior="refine",
    finalist_rounds=2,                # separate paired finalist measurements
    backward_measurement_mode="auto",
)
layer = rtx.MXFP8Linear(
    256, 512, device="cuda", autotune="balanced",
    tuning_policy=policy, autotune_cache_dir=".rtx-cache",
)
model = torch.compile(
    layer, fullgraph=True, dynamic=False,
    options={"triton.cudagraphs": True},
)
x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
y = model(x)
y.float().square().mean().backward()  # warms/tunes backward too
```

The policy controls effort; `autotune="balanced"` enables the first-hit path.
The same policy/cache arguments apply to `NVFP4Linear`. Switching precision or
scaling policy is a different numerical contract, not a compatible refinement.
Both linear APIs remain bias-free. GLU and recomputed MLP support the policy
surface with narrower boundaries documented in the [full effort guide](autotune_effort.md).

| Preset | Cumulative trial cap | Soft time allowance per tuning invocation |
|---|---:|---:|
| `quick` | 8 | 15 s |
| `balanced` | 16 | 30 s |
| `thorough` | 32 | 90 s |
| `extensive` | 64 | 180 s |

These are investment limits, not speed guarantees. Compilation already in
flight is not interrupted at the deadline; confirmation has additional
measurements. Forward and backward have separate budgets and caches. Use
`forward_trials` / `backward_trials` to give them different caps.

## Reuse versus refinement

Keep the cache directory and compatible policy settings. `cache_behavior="reuse"`
accepts an existing compatible winner without extending its search.
`"refine"` lets a later higher trial cap continue from durable observations.
For example, instantiate the same workload with the same policy except
`max_trials=32` to request more trials; it does not restart at trial zero.
Use a newly compiled callable for the changed static policy rather than
expecting an existing compiled graph to observe arbitrary Python mutations.

Budget-only changes preserve compatible evidence. Changes to kernel revision,
device, workload, numerical policy, search/measurement contract or predictor
can invalidate reuse or require a separate history. Do not edit cache JSON to
force compatibility. Interrupted runs retain completed residual observations;
this is not recovery from a poisoned CUDA context in the same process.

Progressive search protects initial local coverage and reserves exploration.
Compatible predictors can additionally supply diverse retrieved configurations.
Candidate predictions never replace numerical validation or hard-prune a
candidate merely because it is predicted to be slow. Paired winner confirmation
is separate from raw search observations; unstable timing cannot establish an
improvement when explicitly flagged by the evaluator.

## Graphs and timing modes

`backward_measurement_mode="auto"` uses the requested Inductor graph setting
for compiled joint backward. Eager and partial-gradient paths use host timing.
Explicit `"host"` and `"cuda_graph"` modes are also available; the public
default remains `"host"`. This option does not enable application capture,
switch forward measurement, or govern GLU/MLP measurement.

Graph and host backward histories are distinct because multi-stream schedules
can rank differently. Graph capture cost is reported separately from compiler
and steady-state event timings. Auto follows the compiler's requested setting,
not a guarantee that Inductor will actually capture every graph.
Warm every required forward/backward shape before manual CUDA graph capture:
a cold tuning miss inside capture is unsupported.

## What the experiments support

The latest local RTX 5070 Ti graph-mode backward study compared ordinary search,
protected portfolio, conditional rules and portfolio+ML at 8/16/32/64 trials
on two reserved geometries. All 512 search trials and 3,607 full-pool reference
trials passed, with no explicitly unstable search/reference timings. Each
selected winner also passed independent dX/dW checks on three input seeds.

| M × N × K | Common winner's paired median | Selection relative to paired top-five full-pool reference |
|---|---:|---|
| 3584 × 768 × 1152 | 62.014 µs | Identical at every method/budget |
| 7168 × 640 × 1024 | 113.120 µs | Identical at every method/budget |

The 100 final paired measurements were valid and stable. On these shapes,
8 trials sufficed; more trials and ML incurred cost without improving selection.
This is not proof of a global optimum, a universal eight-trial stopping rule,
or transfer to another GPU. Earlier forward studies include small positive
results and negative results; earlier unstable backward timing is quarantined.
Learned priors therefore remain opt-in, not the default recipe.

Raw, immutable evidence and cost breakdowns are in
`autotune_reports/matched_backward_graph_20260914_v1/{analysis,oracle_analysis}.json`.
Generated research artifacts are not bundled release dependencies.
See the [qualification matrix](autotune_qualification_matrix.md) for integration
tests and [goal audit](autotune_goal_audit.md) for current executed results.
Final local integration checks passed: Torch 2.13.0+cu132 ran 97 tests and
42 subtests with zero failures/skips; Torch 2.12.1+cu132 ran 19 focused tests
for the newer backward/measurement/refinement paths with zero failures/skips.
The earlier broad Torch 2.12.1 matrix passed 78 tests and 42 subtests; it is
historical coverage, not a second current-code full-matrix run. This qualifies
the exercised paths, not an unconditional release certification.
