# Autotuner coverage repair: offline development tools

These tools address selection bias and lost training coverage without modifying
the frozen GPU predictor comparison, runtime search defaults or exported models.
They live under `tools/research` while the comparison is running: adding them
to `rtx/autotune` now would change its frozen source identity and halt it.

## Active-path audit

```bash
.venv-research/bin/python -m tools.research.coverage_audit \
  --output autotune_reports/coverage_audit_v2/train.json
```

The revision-33 MLP rules follow `rtx/mlp.py` and `_mlp_precision.py`:

- BF16-gated forward uses both up/prequant and down GEMMs, regardless of the
  unused on-chip/cooperative selector fields stored in the MLP configuration.
- Fully MXFP8-gated forward uses the prequant gate only for cooperative execution,
  and the prequant down GEMM only for producer-quantized down execution.
- Prequant recomputation is checked for training when that backend is active.
- Unknown revisions or execution selectors keep conservative handling.

Effective identities omit proven inactive fields, but retain opaque backward
sub-configurations. This intentionally under-deduplicates rather than merging
potentially different kernels. BF16-gate GEMM projection uses the declared
MXFP8 GEMM fields, matching the frontend's conversion.

The older training-data audit found 1,998 successful MLP records quarantined
only for inactive fields. Active-path filtering leaves 131 distinct configurations
before stationarity and timing aggregation, versus eight in the old retained
training set. They are **pending revalidation**, not automatically reinstated.
Successful old measurements alone do not justify removing numerical safeguards.

The report lists observation IDs for follow-up revalidation. Neither the old
dataset loader's default nor the exported artifact is changed by this audit.

## Broader, matched structural panels

`raw_portfolio(adapter, policy, anchors=...)` builds label-independent proposals
from full assignments, deep assignments, and sparse coupled changes. It does not
require each intermediate state to be legal: only completed configurations are
checked. Normalize-time failures are counted. It preserves supplied structural
seeds; it does not invent new backend implementations.

`select_portfolio` applies target-specific static legality, optional effective
deduplication, and max-min categorical diversity. Rejected configurations and
accepted-but-unselected configurations remain in the audit. This is not uniform
sampling of the feasible Cartesian space and is not performance pruning.

Prepare a dynamic-linear panel without GPU execution, using saved hardware metadata:

```bash
.venv/bin/python -m tools.research.structural_sampling \
  --context-result PATH/TO/COMPLETED_TASK/result.json \
  --output autotune_reports/coverage_audit_v2/panel.json \
  --pool-size 64 --proposals 1024
```

The CLI supports MXFP8 dynamic forward/backward and NVFP4 block, delayed and
JIT-region forward. The generic adapter API also accepts GLU/MLP adapters;
building those from their tensor-based factories is not part of this CPU CLI.

Freeze and copy the **raw panel** to each device. Do not regenerate a different
panel from each device's accepted subset. Rejections on one SKU are observations,
not reasons to silently remove a configuration from all devices.

Local static MXFP8 check at M=384,N=1280,K=640, using 1,024 construction attempts:

| Sampler | Accepted selected configs | Mean field distance from initial | Maximum distance |
|---|---:|---:|---:|
| Existing legal random walk | 64 | 0.0510 | 0.1509 |
| Structural + diversity selection | 64 | 0.1749 | 0.7101 |

The raw structural panel had 818 target-static rejections, 64 selected configs,
and 121 other accepted configs. Field-distance diversity is a diagnostic, not
proof of architectural independence or faster kernels. High rejection rates
motivate explicit coupled structural seeds rather than unbounded brute force.

## Learning and pool diagnostics

`FittedCandidate.fit(rows, balance_contexts=True)` is an opt-in experiment that
gives each family equal total loss weight and each context equal weight within
its family. Supported scaler/estimator steps receive sample weights explicitly.
Pairwise ranking rejects this option until query-group weighting is implemented;
weights are not silently ignored. Existing fits/export remain unweighted by default.

`pool_diagnostics(latencies, pool_ids, selected_ids)` decomposes observed regret:

`selected / reference = (pool best / reference) × (selected / pool best)`

The first factor diagnoses proposal coverage; the second diagnoses selection.
Missing labels are reported, never imputed. Reference means the supplied measured
reference set—not a global optimum. Choose pools and predictions before exposing
evaluation labels to this diagnostic.

## Next qualification steps

The first local [GPU recovery and panel validation](coverage_gpu_revalidation.md)
has now completed. It confirms representative recovery paths and a small panel
win; historical-data admission and broader qualification remain separate steps.

1. Finish the frozen study unchanged.
2. Revalidate representative recovered MLP configurations across execution paths,
   forward/backward and shapes before creating a versioned training view.
3. Measure frozen structural panels with shared residual evaluators and balanced
   SKU/shape coverage; retain local-search and uncertainty-selected portions too.
4. Compare weighted versus unweighted fits on structural and shape holdouts.
5. Promote the generic sampler into `rtx/autotune` only in the subsequent source
   revision, retaining the old generator as a controlled baseline.
