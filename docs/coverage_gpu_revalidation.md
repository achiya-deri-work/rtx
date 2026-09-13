# GPU coverage revalidation

`tools.coverage_gpu_study` turns the inactive-field audit and frozen structural
panel into isolated, resumable GPU tasks. It does not train a model, install a
runtime winner, or automatically reinstate quarantined observations.

## Scope

Recovery chooses representative configurations from each available
precision/phase/gate/down/recompute execution group, adding a far-apart
configuration when a group contains more than one. Each is tested at:

- M=128, K=256, hidden=512, output=256;
- M=512, K=768, hidden=1536, output=768.

The current older-data selection produces nine fully MXFP8-gated configurations
and 18 recovery tasks: six forward and twelve training tasks. These are targeted
execution-path tests, not a complete shape/seed or release matrix.

The independent recipe oracle explicitly quantizes with PyTorch and computes
forward and dX/dGateUp/dDown. Tests record finiteness, maximum absolute error,
RMS error, reference RMS, normalized RMS and normalized maximum error.
Compiled execution uses `fullgraph=True, dynamic=False`. Existing numerical
gates are retained: normalized RMS 0.025 for fully MXFP8-gated and 0.003 for
BF16-gated policies. No performance result is accepted after a numerical failure.

After passing that check, the existing shared MLP evaluator checks numerics
again before measuring graph replay. It collects seven samples of three calls,
followed by two repeated evaluations. This is not an optimizer convergence test
or a memory-leak qualification run; peak allocated bytes are recorded only.

The panel comparison uses 16 unique structural candidates and 16 unique
default/local-walk candidates at M=384,N=1280,K=640. Shared configurations are
measured once and credited to both panels. The frozen selection currently has
31 distinct tasks. Existing calibrated linear evaluators own correctness and
timing, with seven samples and two verification evaluations.

## Commands

Preparation reads the old training sources from the audit manifest and the
previously generated raw panel. It does not use prospective evaluation labels.

```bash
.venv/bin/python -m tools.coverage_gpu_study prepare \
  --output autotune_datasets/coverage_gpu_revalidation_v4

.venv/bin/python -u -m tools.coverage_gpu_study run \
  autotune_datasets/coverage_gpu_revalidation_v4 --limit 2

# Resume all remaining tasks after inspecting the smoke gate.
.venv/bin/python -u -m tools.coverage_gpu_study run \
  autotune_datasets/coverage_gpu_revalidation_v4

.venv/bin/python -m tools.coverage_gpu_study summarize \
  autotune_datasets/coverage_gpu_revalidation_v4
```

Use a new output directory to create a new plan. Preparation freezes source and
worker hashes. Each candidate has its own disposable process, a 240-second
timeout, log and atomic result JSON. Completed failures are preserved, not
retried until they happen to pass. Repeating `run` resumes unfinished tasks.
One lock prevents concurrent supervisors for the same output.

The development v1/v2 plans were superseded before GPU qualification; v3 caught
a dataset-versus-public configuration serialization error in its smoke gate.
Those results are not kernel numerical failures. v4 uses the adapter's dataset
deserializer, without bypassing its static legality checks.

## Interpretation

Passing two new shapes provides execution-path evidence; it does not certify
every historical shape, device, seed or observation. Recovery records remain
pending until a versioned training view specifies the exact admission contract.
Structural sampling may uncover slow configurations as well as fast ones.
Coverage and useful negative examples are separate goals from a fastest-winner
search. Compare measured panel winners, failures and latency distributions;
do not interpret field-distance diversity as a speedup prediction.

## Completed local results — 2026-09-13

RTX 5070 Ti, Torch 2.13.0+cu132: **49/49 tasks completed**.
Sixteen recovery cases passed; two were rejected before execution by the
existing active dX single-buffer/consumer-scale quarantine. That configuration
remains blocked. All 31 distinct panel configurations passed correctness.

| Recovery result | Cases | Worst normalized RMS vs quantization-aware recipe |
|---|---:|---:|
| Output | 16 | 4.40e-5 |
| dX | 10 | 9.17e-5 |
| dGateUp | 10 | 5.26e-5 |
| dDown | 10 | 2.22e-5 |

These errors measure implementation agreement with the explicitly quantized
reference, **not** total quantization error against unquantized BF16/FP32.

The two 16-candidate panels shared one candidate:

| Panel | Best latency | Median candidate latency |
|---|---:|---:|
| Structural | 8.225 us | 22.192 us |
| Default/local walk | 8.583 us | 8.818 us |

Broader sampling found a better winner and many slower candidates. It does not
justify replacing all local proposals with distant configurations.
On the fully measured panel, both frozen predictors placed the structural
winner in their top eight. The local panel could not reach that winner;
its pool penalty was 1.0435x relative to the best measured union candidate.
This is post-collection ranking diagnosis, not another online UCB experiment.

Twelve randomized-order paired confirmation rounds on identical inputs then
measured 8.2205 us structural versus 8.5791 us local. Structural won 12/12 rounds;
median paired speedup was **1.0437x**, with round ratios 1.0401–1.0486x. This
confirms the small local panel win, not a cross-shape or cross-device guarantee.

Artifacts in `autotune_datasets/coverage_gpu_revalidation_v4`:

- `plan.json`, `environment.json`, and per-task logs/results;
- `coverage_analysis.json`: numerical summaries and pool/ranker decomposition;
- `paired_winner_confirmation.json`: raw confirmation outcomes and selected IDs.

Rebuild the analysis with:

```bash
.venv/bin/python -m tools.analyze_coverage_gpu \
  autotune_datasets/coverage_gpu_revalidation_v4
```

The recovered records have not been automatically admitted to training. These
passes support an execution-aware admission audit; they do not certify all
1,998 historical observations or override active backward legality checks.
