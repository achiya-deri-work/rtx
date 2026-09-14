# MLP first-hit effort qualification — 2026-09-14

This is lifecycle and numerical qualification on one RTX 5070 Ti, not evidence
that a particular effort budget or learned prior wins across workloads. The
automatic MLP path remains opt-in. Existing explicit execution is unchanged.

## Training ladder

Shape M/K/H/O = 256/256/512/256, SiLU, dynamic BF16 operands, MXFP8
forward/backward, `torch.compile(fullgraph=True, dynamic=False)`. Each level ran
in a fresh process against one shared durable cache. The progressive policy used
five measurement samples, three calls/sample, one warmup, two paired finalist
rounds, and a 180-second ceiling per invocation. Seed: 946.

| Requested cumulative trials | Recorded trials | First forward | First backward | Dynamo compilations |
|---:|---:|---:|---:|---:|
| 8 | 8 | 10.94 s | 85 ms | 1 |
| 16 | 16 | 23.57 s | 76 ms | 1 |
| 32 | 32 | 31.50 s | 44 ms | 1 |
| 64 | 64 | 44.52 s | 44 ms | 1 |

Times are **per invocation**, not cumulative. First forward includes the
incremental joint forward/backward search and compiler setup; first backward is
the caller's first real backward after selection. Imports are excluded. Fresh
processes still share filesystem compiler caches, so these are not cold-machine
startup guarantees.

All 64 candidates compiled and passed the numerical gate. Independent quantized
recipe validation checked output, dX, dGateUp and dDown after every effort level.
Maximum normalized RMS error was 0.000046817, below the unchanged 0.025 gate.
This is agreement with a separately implemented MXFP8 recipe, **not** an error
measurement against an unquantized BF16 model.

The 16-, 32- and 64-trial selections were identical. More exploration did not
improve this context after trial 16. This single case does not justify shrinking
the universal budget or search space.

Every level verified identical repeated outputs, one Dynamo compilation, and
unchanged journals after warm calls. Hash checks against the final journal
verified preservation of all earlier byte prefixes. Source hashes were identical
across the four processes. This checks actual append/resume behavior, not exact
search-trajectory equivalence to an uninterrupted run with different timing
observations.

## Fresh-process reuse and ragged inference

A fifth process requested 64 trials with calls to the tuning function forbidden.
It passed with **zero tuner calls**, one Dynamo compilation, unchanged journal
hashes, and the same numerical checks. Its first compiled forward took 1.37 s.

A separate no-grad cache used ragged M/K/H/O = 17/35/97/41. Eight real forward
trials completed in a 5.71-second first call; the selected output exactly matched
the independent quantized reference on that input. Fresh-process reuse completed
in 1.41 s with zero tuner calls and unchanged journals.

These runs use ordinary compiled execution. Their ten synchronized warm-call
wall samples include Python, launch, GPU execution and synchronization costs;
they are diagnostics, not CUDA-graph-replay performance. Manual CUDA graph
qualification is recorded separately in
[the execution-plan contract](mlp_first_hit_plan_contract.md). Automatic Inductor
cudagraph mode is not qualified by these runs. Subsequent testing found and
guarded a mixed-output partition crash; see the execution-plan contract for
the current limitation: first-hit host-plan boundaries remain uncaptured by
automatic Inductor CUDA graphs, although warmed manual capture works.

## Artifacts and reproduction

Reports, source hashes, configs, numerical metrics, timing samples and persistent
tuner journals: `autotune_reports/mlp_effort_ladder_20260914_v1/`.

The driver refuses to overwrite reports. Use a new report directory to repeat
the study; preserve the cache between progressive levels:

```bash
for budget in 8 16 32 64; do
  .venv/bin/python tools/qualify_mlp_first_hit.py \
    --cache-root autotune_reports/my_mlp_ladder/cache \
    --output "autotune_reports/my_mlp_ladder/train_${budget}.json" \
    --budget "$budget"
done
.venv/bin/python tools/qualify_mlp_first_hit.py \
  --cache-root autotune_reports/my_mlp_ladder/cache \
  --output autotune_reports/my_mlp_ladder/reuse.json \
  --budget 64 --reuse-only
```

For the ragged inference case, use a separate cache and append
`--phase forward --shape 17,35,97,41 --budget 8`.

Broader shape/SKU coverage, held-out effort-quality comparisons, selective
precision first-hit support, and automatic cudagraph integration remain open.
No learned policy is promoted from this inspected qualification context.
