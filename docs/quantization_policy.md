# Quantization policy and accuracy selection

RTX separates numerical policy from execution tuning:

| Mechanism | Chooses | When |
|---|---|---|
| `NVFP4CalibrationConfig` | Packed weight scales | Offline PTQ conversion |
| `rtx.autotune.select_accuracy_policy` | User-defined per-group numerical policies | Offline model-quality evaluation |
| `AutotunePolicy` / `autotune` | Kernel schedules for a chosen policy | Warmup/first-hit tuning and cached execution |

Runtime autotuning does not decide whether your model tolerates FP4 or silently
change training precision. Run calibration and accuracy selection before
`torch.compile` and outside CUDA graph capture. Neither adds policy search or
collection hooks to a converted module's forward.

## Calibrated NVFP4 PTQ

```python
import rtx
import torch

def selected(module, name):
    return isinstance(module, torch.nn.Linear) and module.bias is None

model.eval().to(device="cuda", dtype=torch.bfloat16)
model = rtx.quantize_(
    model,
    rtx.NVFP4WeightOnlyConfig(
        scaling="current",
        weight_calibration=rtx.NVFP4CalibrationConfig(method="mse"),
        autotune="cache",
    ),
    filter_fn=selected,
)
compiled = torch.compile(model, fullgraph=True, dynamic=False)
```

Omit `weight_calibration` to retain native packing. Explicit calibration runs
a CPU search and produces native packed NVFP4 weights, not decoded BF16 runtime
weights. Public inputs/output are BF16, but activations are dynamically
quantized to NVFP4: this is not W4A16.

Offline methods:

- `max`: block amax/6 scaling.
- `four_over_six`: compare encoded amax/6 and amax/4 by weight SSE.
- `mse`: search supported normal E4M3 scales by weight reconstruction SSE.
- `local_hessian`: search using uncentered activation second moments within
  each 16-element block. This approximates local error, not full-model loss.

`rows_per_chunk=64` bounds CPU working memory, not numerical region size.
`hessian_shrinkage` is in `[0,1]`; nonzero values require `local_hessian` and
use `H' = (1-alpha) H + alpha mean(diag(H)) I`. Zero uses observed moments;
one reduces scale selection to weight MSE. Neither method guarantees better
downstream accuracy.

FP32 outer scales remain intact: `current` defaults to weight amax/2688;
`block` fixes the weight outer scale to one. Activation quantization is
unchanged. `quant_config` conflicts with calibrated weight packing and is
rejected rather than ignored. Calibration options are NVFP4-only.

## Activation moments

On the unconverted BF16 model, using representative non-test data:

```python
with rtx.NVFP4CalibrationCollector(
    model, filter_fn=selected, max_tokens_per_batch=1024,
) as collector:
    with torch.inference_mode():
        for inputs in calibration_batches:
            model(inputs)  # Caller owns preprocessing, device and invocation.

moments = collector.second_moments()
model = rtx.quantize_(
    model,
    rtx.NVFP4WeightOnlyConfig(
        weight_calibration=rtx.NVFP4CalibrationConfig(
            method="local_hessian", hessian_shrinkage=0.5,
        ),
    ),
    filter_fn=selected,
    calibration_moments=moments,
)
```

Collection does not change model mode or gradient mode. It samples at most
the configured number of evenly spaced flattened token rows per invocation,
accumulates detached CPU FP64 statistics, and returns CPU FP32 matrices shaped
`[ceil(K/16),16,16]`. No full activation history is retained. Hooks and transfers
make this an offline operation. `sample_counts` reports sampled rows.

The collector is single-use; hooks are removed even on failure. Failed,
incomplete or unobserved collection cannot return results. Every selected
module must execute. Shared modules have one collector and canonical fully
qualified name (FQN) keys; a root linear uses `""`.

## Per-module overrides and conversion contracts

Pass `calibration_overrides={fqn: NVFP4CalibrationConfig(...)}` to `quantize_`.
Overrides replace the default calibration policy for selected modules. Without
a default, other selected modules keep native packing. All keys must identify
selected modules. Unknown keys, missing Hessian moments, invalid moments and
conflicting shared-alias policies fail before replacements are installed.
One shared-alias moment entry suffices; multiple entries must agree. Valid
moments for non-Hessian policies are accepted but unused.

Always assign the returned model. PTQ removes selected BF16 weights; retain a
checkpoint or independent source copy for comparisons. The no-bias, BF16-weight
and shared-alias [conversion contracts](model_conversion.md) still apply.

For custom packing, `rtx.calibrate_nvfp4_weight(...)` returns a native tensor
for `NVFP4Linear(packed_weight=...)`. Its method argument is called `policy`;
the high-level config calls it `method`. `rtx.NVFP4SecondMomentCollector`
provides streaming statistics for custom loops.

## Whole-model accuracy selection

Local reconstruction error is only a screening metric. Our DINO experiments
showed locally better scales can hurt the full model. The generic offline API
uses complete policy states and a separate guard:

```python
from rtx.autotune import select_accuracy_policy

result = select_accuracy_policy(
    groups=["block0", "block1"],
    candidates=["max", "mse", "lh50"],
    baselines=("max", "mse"),
    evaluate=score_on_search_data,
    guard_evaluate=score_on_independent_guard_data,
    max_passes=1,
)
chosen = dict(result.selected_state)
audit_record = result.to_dict()
```

Each callback receives the complete `{group: candidate_label}` mapping,
applies it to the model, and returns a finite nonnegative score (lower is
better). Callers own candidate construction, group-to-FQN mapping, teacher
outputs and dataset partitioning. Expand `chosen` into calibration overrides
on a fresh source model, or apply your own packed candidates explicitly.
The API does not apply its winner, benchmark latency, or write runtime caches.

Search starts from the best homogeneous baseline and runs bounded coordinate
passes with strict improvement. The independent guard compares the searched
state with all homogeneous baselines, preferring baselines on ties.
`guard_choice` is `search` or `baseline:<label>`; `selected_state` can differ
from `search_state` after fallback. The immutable result records both phases.
Scores on different datasets are not directly comparable.

Keep moment fitting, search, guard and final test data separate; freeze policy
before testing. Separate callbacks alone cannot prove dataset independence.
Save the JSON-friendly record alongside checkpoint/data identities. The DINO
runner in `benchmarks/select_nvfp4_dinov3_model.py` uses the shared helpers but
remains model-specific research, not a dependency of the public library.
See [measured evidence and limitations](nvfp4_precision_study.md).

## Training policy

Packed-weight calibration is inference-only. Training keeps BF16 master
weights and dynamically quantizes changing operands; NVFP4 forward uses
MXFP8 backward. Choose numerical policy explicitly through
`NVFP4TrainingConfig`, then autotune its execution schedules.

Compare policies with matched short training runs: identical checkpoint,
optimizer, data order and token budget. Evaluate validation trajectory,
gradients, stability, throughput and memory. Static teacher fidelity does not
establish training convergence. JIT-region, delayed and block scaling have
different numerical/state contracts, not merely different schedules.
Existing quarantine/device qualification boundaries remain unchanged.
