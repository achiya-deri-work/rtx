# BalancedMoE application-side review — 2026-11-20

The normal BF16 training/checkpoint workflow is usable without private RTX
APIs. It is not polished yet: cold `inference_mode()` autotuning and enclosing
BF16 autocast both expose real integration bugs. No runtime fixes were made
as part of this review.

## What I actually ran

[The example](../examples/balanced_moe.py) puts a `BalancedMoE` inside a small
residual token classifier, with FP32 RMSNorm/reductions, BF16 embeddings/head,
and ordinary AdamW. It compiles the outer model with
`torch.compile(fullgraph=True, dynamic=False)` using default compile mode.
There are no custom streams, scratch buffers, private APIs or expert configs
in the application.

The MoE has D=128, E=16, K=2, H_expert=192, H_shared=128. Its active hidden
budget is 4D. Inputs are two independent 64-token samples. This deliberately
small usability workload is **not a throughput, numerical-equivalence or
convergence benchmark**. Eight updates repeatedly use one synthetic batch.

Environment: HOME RTX 5090, PyTorch 2.13.0+cu132, CUDA 13.2 and CuTe DSL 4.7.0.
The executable exercise is
[benchmarks/exercise_moe_user_journey.py](../benchmarks/exercise_moe_user_journey.py).

## Working workflows

| User action | Observed result |
|---|---|
| Outer-model fullgraph compilation and AdamW | Eight updates completed; all parameter gradients present and finite |
| Parameter updates | All 55 MoE parameters changed; toy loss 3.4755 → 0.8570 |
| Checkpoint model and optimizer, load on CPU, move to CUDA | Passed without repacking or allocation management |
| Fresh-process inference and resumed training | Identical saved/reloaded logits (max absolute difference 0); next loss 0.7344 |
| Persistent tuning cache in a fresh process | No additional observation-journal entries during inference and resumed training |
| Cold inference using `no_grad()` | Passed, including forward-phase tuning |
| `inference_mode()` using a populated forward cache | Passed in a fresh process |
| CPU construction followed by `.cuda()` | Passed compiled inference |
| Packed `[128,128]`, offsets `[0,48,48,128]` | Compiled forward/backward passed, including an empty sample |
| Frozen MoE with trainable upstream input | Finite input gradients; no parameter gradients |
| `deepcopy` for an EMA copy | Independent parameter storage and identical output |
| Meta construction, `load_state_dict(assign=True)`, CUDA move | Forward passed |
| One compiled layer with lengths 64 → 80 → 64 | Forward/backward passed; static compilation may specialize per shape |

The example preserves the constructor's mixed-precision recipe: 55 BF16
parameter tensors and four FP32 tensors (two norms and two router scalars).
AdamW's two state slots followed their parameter dtypes: 110 BF16 and eight
FP32 tensors, excluding step counters. This checks optimizer integration,
not the suitability of this optimizer recipe for a real task.

## Two integration bugs to fix

### 1. Cold implicit autotuning fails inside `inference_mode()`

The initial training run succeeded. Switching that compiled model to eval
and invoking it under `torch.inference_mode()` caused a new forward-phase
router tuning miss and failed with:

```text
RuntimeError: Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.
```

The router evaluator constructs calibration state using
`scores.detach().requires_grad_()` in
`rtx/autotune/varlen_routing.py`. The tuning worker runs outside the caller's
inference-mode context, but the detached scores are still inference tensors.
The evaluator needs to own suitable ordinary calibration tensors; callers
should not have to understand this distinction.

This is a **cold-miss** problem: ordinary `no_grad()` populated the forward
cache successfully, and a separate process subsequently passed
`inference_mode()` using that cache. Previous warm/prepared inference checks
therefore did not cover this failure.

### 2. BF16 autocast breaks the router's FP32 contract

Even with contiguous **BF16** input and `autotune="off"`, wrapping the layer
in `torch.autocast("cuda", dtype=torch.bfloat16)` makes router affinities
BF16. The native packed router expects FP32 and raises an FFI argument-dtype
error. Casting projection inputs to FP32 alone does not protect a linear
operation from autocast.

The router should preserve its intended FP32 calculation/dispatch contract
inside a user's autocast region. Regression coverage should include both
eager and fullgraph compiled execution, plus gradients. This review directly
reproduced the BF16-input failure in eager execution; it does not claim that
the compiled AMP path is qualified.

## Friction rather than kernel failures

- **Startup needs clearer feedback.** The first balanced compiled training
  update took 95.25 seconds, the next took 3.92 seconds, then updates were
  approximately 2–3 ms in this tiny host-fed application. Cold forward-only
  tuning under `no_grad()` took 49.40 seconds. A cache-hit fresh-process
  inference still took 8.31 seconds for compiler/startup work. A tuning-cache
  hit is not a compiler-cache or zero-startup guarantee. The user sees many
  framework warnings instead of a concise explanation of current tuning
  phase and progress. These are one-run wall times, not latency estimates.
- **Whole-model dtype conversion changes the recipe.** `.bfloat16()` also
  downcasts the router's FP32 scalar parameters. Forward still runs, but the
  constructor's numerical policy is no longer intact. Prefer device-only
  movement for the supplied mixed-precision example.
- **Input restrictions are explicit but stricter than a generic module.**
  FP32 input, even under BF16 autocast, and transposed noncontiguous BF16
  input both raise the documented contiguous-BF16 validation error. Those
  are current contracts, not the AMP bug above. Any future convenience copy
  or cast must remain compiler-visible rather than hidden in a custom op.
- **Effort is per component, not a layer-wide trial count.** The default
  balanced run admitted 16 expert and 16 shared-MLP trials; router/affinity
  trials were fewer under their deadlines. Startup duration cannot be read
  as exactly 16 trials for the whole layer.

## Reproduce

Run each command in a separate process on an otherwise available CUDA GPU:

```bash
python -m examples.balanced_moe

# Use a new empty output path to exercise the cold inference-mode failure.
python -m benchmarks.exercise_moe_user_journey train --output /tmp/moe-cold-review

# Independent successful no_grad workflow, then cache-hit inference_mode.
python -m benchmarks.exercise_moe_user_journey train \
  --output /tmp/moe-no-grad-review --no-grad-inference
python -m benchmarks.exercise_moe_user_journey restore \
  --output /tmp/moe-no-grad-review

# Includes intentional contract rejections and the autocast reproduction.
python -m benchmarks.exercise_moe_user_journey edges --output /tmp/moe-edges-review
```

The edge runner records each outcome, including expected errors; a stage
completion marker is not a claim that all probes passed. Stop on CUDA
device assertions/illegal accesses rather than continuing in a bad context.

Evidence, cache journals, the replayable checkpoint, source snapshots and
logs are preserved at
`autotune_reports/moe_user_journey_20261120/` (Storage4-backed).

**Next priority:** fix cold inference-mode calibration ownership and router
autocast isolation, add targeted regression tests, then improve first-hit
status reporting. More kernel tuning is not needed to establish these
application-level issues.
