# MLP deployment-mode qualification

First-hit MLP selection measures the selected operator in the caller's execution
mode: direct selected calls when automatic CUDA graphs are disabled, or private
kernel replay when enabled. RTX does not compile another model inside the runtime
launcher. The tuning harness compiles candidate callers to reproduce that boundary.

Offline `tune_mxfp8_mlp` also accepts `measurement_mode`:

| Mode | Timed execution |
| --- | --- |
| `outer_capture` (offline default) | Replay of a captured sequence of compiled explicit MLP calls |
| `compiled_direct` | Compiled explicit MLP calls, without automatic CUDA graphs |
| `selected_direct` | Compiled selected operator, including host plan publication |
| `private_replay` | Compiled selected operator with private replay, including staging and output publication |

Residual context identities and first-hit runtime keys distinguish execution
modes. Older frontend-revision winners are not silently reused under the new
execution contract. Keep old result files for comparison; do not delete them.
Pass an archived config as `initial` (CLI `--initial`) to qualify against it.
When refining a compatible runtime winner, that winner is the incumbent.
The original configured search anchor stays fixed when the effort budget grows;
the previous winner is added to the candidate portfolio and remains the paired
qualification baseline. Increasing a compatible 8-trial effort to 16 therefore
resumes its journal rather than starting 16 new trials merely because the winner
changed. Changing the workload, numerical policy, search domain, or measurement
contract can still legitimately require a separate journal.

Structural proposal revision 2 adds complete MXFP8 training candidates:
prequantized gate/down/recompute, row-major materialized backward, prefetched
128×64 recompute tiles, and four/eight-way FP32 dW splits. Each proposal goes
through normal legality, numerical, timing, and promotion checks; none changes
the module's kernel default. They enter the candidate pool but are not promised
mandatory trials at every effort level, especially when a retrieved prior
controls opening proposals. Candidate availability is not a claim that an
eight-trial search finds the best measured schedule. The revision is included
in residual context identity; old research journals are preserved, not merged
across changed search domains. Runtime winner validity remains separately
governed by its existing execution/numerical contract.

After search selects a challenger, six paired rounds alternate incumbent-first
and challenger-first. Promotion requires the lower-quartile paired speedup to
exceed 1%. This is a conservative decision heuristic, not a confidence interval.
An invalid challenger cannot replace the incumbent; an invalid incumbent causes
qualification to fail rather than returning an unverified fallback. Numerical
checks use the explicit reduced-precision recipe, not equivalence to BF16.

Qualification requires **additional time and memory**: two prepared finalists and
12 evaluations beyond the soft search budget. They share the independent
reference tensors within this tuning invocation. `qualification_rounds=0`
disables this extra gate for offline collection. Time budgets remain soft;
neither a Python timeout nor this change cancels an in-flight compiler/CUDA call.
The returned elapsed time includes qualification. Immutable
`deployment-qualification-*.json` reports preserve both configurations, samples,
failures, and the selected configuration; search observations remain unchanged.

Enable first-hit messages using standard Python logging before compilation:

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("rtx.autotune.mlp").setLevel(logging.INFO)
```

Messages include the disk-cache decision, search progress, qualification, final
selection, and initial-config fallback on a cache-only miss. These are operational
messages, not a per-compiler-stage watchdog or a full structured telemetry API.

## Qualification workload

The offline tuner accepts an explicit `MLPExecutionMix` from
`rtx.autotune.execution_mix`, or `--execution-mix PAIRS,GRAD,INFERENCE` on the
CLI. For example, `--phase train --execution-mix 1,2,3` measures one complete
training forward/backward pair, two additional grad-enabled forwards, and three
inference forwards. Extra forwards are executed before the training pairs, with
their results discarded. They are **not** additional backward operations.
Timings are milliseconds per complete workload, not milliseconds per forward.

The exact counts and ordering are part of the residual context identity and are
retained for finalist qualification. Default runtime first-hit selection still
uses its existing one-pair objective. Opt in to a workload-aware runtime search
at construction:

```python
import rtx

layer = rtx.MXFP8MLP(
    256, 768, autotune="balanced",
    training_mix=rtx.MLPExecutionMix(training_pairs=1, grad_forwards=2,
                                    inference_forwards=3),
    inference_mix=rtx.MLPExecutionMix(training_pairs=0, inference_forwards=4),
)
```

Training and inference mixes select separate objectives; inference cannot
request backward or grad-enabled calls. These options affect cold tuning, not
the semantics or number of calls in `layer(x)`. They require `autotune="cache"`
or `"balanced"`; silently ignored options with `"off"` are rejected. The
constructor serializes immutable settings before user-side compilation.

Explicit mixes have objective-specific disk-winner identities. Default users
retain legacy keys. Changing a mix cannot silently reuse the other objective's
winner; cache-only mode retains its documented initial-config fallback on a
miss. No precision fallback is introduced. BF16 replacement comparisons remain
pending.

This isolates execution cost for one shape with fixed operands. It does not
simulate full checkpoint nesting, saved-tensor overlap, separate teacher weights,
or full-model peak memory. Do not infer those from this benchmark. Select counts
from per-module call evidence, not aggregate counts spanning unrelated modules.

September 16 validation: all 20 deployment GPU cases passed on both Torch
2.13.0 and 2.12.1, including four repeated-forward mixed-workload cases.
Tests check finite outputs/all gradients against the independent MXFP8 recipe,
prepared-runner reuse, and stable post-warmup live allocation. NVML reported a
595.84/595.91 driver/library mismatch while CUDA itself remained usable; these
runs establish correctness, not new performance comparisons.

The AFRL feedback identifies four contexts: `M=9216` and `11296`, each forward-only
and forward/backward, with `K=256,H=1536,O=256`. For example, when the GPU is free:

```bash
python -m rtx.autotune.mlp --shape 11296,256,1536,256 --phase train \
  --measurement-mode selected_direct --trials 16 --seconds 120 \
  --initial path/to/incumbent.json --qualification-rounds 6 \
  --output-dir autotune_results/afrl_deployment
```

Repeat for both row counts and phases; qualify private replay separately when
used. See [replacement-level qualification](mlp_replacement_comparison.md)
for the BF16 comparison workflow. Historical local qualification reports are
not distributed. CPU policy tests alone are not evidence of a speedup or
CUDA correctness.

## Remaining feedback work

The existing selected-backward lifetime fix, graph-owned scratch, independent
outputs, and post-tuning cleanup are preserved. Complete legal materialized dW/dX
families and combined gradient-leg proposals now precede tile refinements in the
structural seed list; rejected and duplicate seeds are filtered. Explicit portfolio/prior candidates may still
precede these seeds; this is not a guarantee that a tiny budget covers all legs.

Resident-byte budgets, tuning-scoped cleanup, per-stage watchdogs, and checkpoint
replay-drift isolation remain separate work. In particular,
`clear_runtime_caches(device=...)` is not device-local eviction: it still clears
global frontend caches/pools. Do not call it during concurrent multi-device work
without establishing completion on every affected device. No new performance or
selective-checkpoint numerical claim is made by this change.

## Phase-specific application installation

For independent grad/no-grad plans on exact application input shapes, see
[public phase binding](mlp_phase_binding.md). Synthetic preflight and binding
do not confer application/BF16 replacement approval.
