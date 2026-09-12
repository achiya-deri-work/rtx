# Development and validation

## Structure and boundaries

`rtx/fp8.py` and `rtx/fp4.py` own linear frontends and operator integration;
`rtx/configs` owns immutable policies; `rtx/formats` owns packed contracts;
`rtx/kernels` owns CuTe implementations; `rtx/autotune` separates generic search
from RTX adapters; `rtx/models` is the convergence scaffold. GLU/MLP frontends
and direct lowerings are separate modules. Packaging installs only `rtx` and
its package data, not test assets or campaign output.

Kernel contributions should expose actual legal implementation alternatives,
not no-op tuning coordinates. Static legality must account for architecture,
shape, scale layout, memory, thread/register budgets and schedule prerequisites.
Do not let a known invalid candidate reach a GPU just to collect an error.
Generic search-policy code should not import a particular CUDA kernel.

## Test in layers

1. CPU/API tests: config validation, serialization, routing, conversion,
   aliases, package metadata and documentation contracts.
2. Independent CUDA numerical references: output plus every requested
   gradient, finite checks, absolute/relative metrics, ragged and extreme data.
3. `torch.library.opcheck`, fake behavior, fullgraph compile, graph replay,
   selective gradients, noncontiguous upstream gradients and parameter updates.
4. Training/optimizer smoke, gradient accumulation, streams, synchronized live
   memory stability, and checkpoint roundtrip.
5. Compute Sanitizer memcheck/racecheck on risky schedules, then matched
   end-to-end performance and model-level convergence.

```bash
python -m unittest tests.test_documentation -v
python -m unittest discover -s tests -v
python -m unittest tests.test_mxfp8_mlp_frontend tests.test_mxfp8_mlp \
  tests.test_mxfp8_mlp_decode -v
```

The full suite can take substantial time and optional suites require assets or
devices. CUDA-gated tests skipped on CPU are **not** GPU passes. Use both pinned
Torch environments for integration changes. The numerical stress script is
documented in [the MLP report](mxfp8_mlp_numerics.md); its known failing singleton
case must run in a separate process. Stop after a poisoned CUDA context.

No blanket correctness claim follows from a benchmark returning finite output.
Use an independent mathematical/quantized reference, preserve failing inputs,
and do not widen gates to hide missing contributions. FP32/FP64 oracles are
test instruments, not a change to the BF16 training policy.

## Benchmarking and datasets

Record source revision, dependency versions, device, configuration, logical and
physical shapes, scaling policy, cache regime, compile/capture policy, timing
samples, correctness gates and baseline. Compare equally checkpointed models
when studying recomputation; also report ordinary BF16 separately. A single
shape's speedup is not a portable library guarantee.

Keep journals, logs, profiles, checkpoints, third-party model assets and
machine-specific results out of public commits unless intentionally curated.
CSV/Parquet are derived artifacts; retain their source residual bundles.

## Documentation maintenance

[docs/README.md](README.md) is the manual entry point. Update the appropriate
workflow guide when behavior changes. Generated [API reference](api_reference.md)
must match source; `python tools/doc_reference.py` prints its current content
and `python tools/doc_reference.py --check` checks the checked-in version.
`tests.test_documentation` checks reference freshness, local links and example
syntax. CUDA examples are in `docs/examples/basic_workflows.py` and should be
executed on a supported GPU after changing their APIs.

Do not promote dated performance reports into API guarantees. Document known
blockers prominently. Bump relevant kernel/cache/schema revisions when a
change makes old schedules or packed layouts unsafe. Preserve user checkpoint
and optimizer contracts, and document migrations rather than silently casting.

Before publishing, consult [the release checklist](release_checklist.md), run
package build/checks from `requirements-dev.txt`, and inspect the wheel/source
distribution contents. No commit, tag or upload is implied by a local test pass.
