# Experimental mixed NVFP4 backward: public compiled graph comparison

## Boundary and outcome

This is a research-only comparison, not a stable `NVFP4Linear` change or release
qualification. All five paths use identical original BF16 inputs and the same
explicit **exact** JIT row-region NVFP4 forward configuration. The fifth path is
the public stable `nvfp4_linear` paired-MXFP8 backward baseline; it is not replaced
with the research wrapper's separate partial-gradient implementation.

At `(M,N,K)=(2048,768,1536)` on an RTX 5070 Laptop GPU, the wide FP4 preparation
paths did not show a descriptive end-to-end advantage over stable paired MXFP8.
Every wide-run independent total-time sample set failed the repository
stationarity gate, so those medians alone do not establish a measured winner.
A subsequent bounded within-session paired experiment found wide preparation
faster than the previous FP4 preparation, but both full-FP4 and the selected
mixed path slower than stable paired MXFP8. No default dispatch was changed,
and no longer-shape search was used to rescue that result.

## Instrumentation

`benchmarks/compare_mixed_nvfp4_backward.py` starts one fresh worker process per
policy. Each worker runs the public functional API through
`torch.compile(fullgraph=True, dynamic=False)` and `torch.autograd.grad` for both
dX and dW. Forward, backward (`retain_graph=True`), and complete forward+backward
are independently captured CUDA graphs. The total is measured directly, not
inferred by adding isolated phases.

Timings are steady-state CUDA graph replay latencies, excluding compilation,
capture, input generation, and eager Python/autograd launch overhead. They are
not optimizer-step or training wall-throughput measurements. The existing
`measure_cuda_batches` calibration and `runtime_timing_metadata` stationarity
gates are used without relaxed thresholds or retries. Raw samples, calibrated
calls per sample, actual configs, Torch/CUDA/device identity, commands, and dirty
runtime source SHA256 hashes are retained in each result.

The exact configuration is constructed before compilation, including
`dynamic_config.tensor_scale_mode='exact'`. Constructing the stable scale-policy
dataclass inside the traced closure failed Torch 2.13's constant-result handling;
preconstructing that immutable policy fixed the benchmark without changing the
stable runtime.

## Correctness and allocation guards

All five paths passed the `(128,128,128)` smoke and the representative wide run:

- Forward is bitwise identical to the stable public exact-JIT reference.
- Total replay observes changes to X, W, and upstream dY. Each perturbation is
  checked against fresh independent FP32 dX/dW GEMMs on the current BF16 inputs.
  Gaussian algorithm-quantization NRMS limits are 0.25 for FP4 and 0.06 for MXFP8.
  These are stale-buffer guards, not tight packed-kernel numerical qualification.
- FP4 phases produce fresh stochastic gradients on identical graph replays;
  MXFP8 phases remain deterministic. The separately captured backward graph is
  checked for the same RNG policy.
- After matching side-stream and default-stream warmup, 30 ordinary compiled
  calls release outputs and gradients. Three synchronized ten-call samples and
  the existing bounded-idle cleanup helper retain raw allocation observations;
  final live allocation must not exceed the original post-warmup baseline.
- Twenty total-graph replays have zero live allocated-byte growth. No reserved
  memory measurement is used as leak evidence.

Both the smoke and representative wide run had zero ordinary-call and graph
replay live growth for all five policies. Initial diagnostic allocation growth
was caused by auditing a default stream warmed only on a side stream; the final
harness warms the exact audited lifecycle and preserves its original bound.

## Representative wide-preparation measurements

RTX 5070 Laptop GPU, Torch 2.13.0+cu132, BF16 inputs, seed 941, 15 samples.
Only `producer_load='coalesced_wide'` differs from the default FP4 preparation
schedule. All entries below are descriptive medians in microseconds.

| dX / dW | Forward | Backward | Direct total | Total stationary |
|---|---:|---:|---:|---|
| FP4 / FP4 | 52.17 | 205.52 | 247.96 | no |
| FP4 / MXFP8 | 51.72 | 153.19 | 207.07 | no |
| MXFP8 / FP4 | 46.16 | 189.56 | 235.86 | no |
| MXFP8 / MXFP8, research separate phases | 51.96 | 119.72 | 182.90 | no |
| MXFP8 / MXFP8, stable paired baseline | 51.48 | 119.79 | 185.74 | no |

The observed capture peak above the pre-capture live baseline was 24.97 MiB for
FP4/FP4 and 22.50 MiB for stable paired MXFP8. This instrumentation retains three
independent graph pools and contexts; those numbers are not a single training
step's incremental workspace or evidence of a memory advantage.

Prior `producer_load='coalesced'` runs, with every other FP4 schedule field
unchanged, gave direct total medians of 272.30 us for FP4/FP4 (nonstationary),
209.84 us for FP4/MXFP8 (stationary), and 250.52 us for MXFP8/FP4 (stationary).
Separate process medians are not used to establish a wide-producer speedup.

## Paired complete-public-graph result

The `--paired` experiment reuses the same public graph builder and its numerical,
current-buffer, RNG, and allocation guards. Stable paired MXFP8, full FP4 wide,
FP4-dX/MXFP8-dW wide, and full FP4 coalesced graphs share the same static X/W/dY
storage. All four are built and warmed before the first timing sample. Each pair
uses 12 rounds, each containing both AB and BA windows (four timed batches), with
the window order alternating across rounds. Each batch has 40 complete
forward+backward graph replays. FP4 RNG generation is inside those graphs. No
configuration changed within a graph during timing.

The reported ratio is reference latency / candidate latency; greater than one
favors the candidate. The geometric-mean ratio's percentile 95% interval is
bootstrapped from 4,000 resamples of whole rounds, never individual replays or
individual order windows. Each round aggregates its AB and BA ratios
geometrically. Per-order estimates expose residual timing-order bias.

| Reference / candidate | Paired ratio | Round-cluster 95% interval | AB / BA ratios |
|---|---:|---:|---:|
| Stable paired MXFP8 / full FP4 wide | 0.6468 | [0.6183, 0.6746] | 0.6399 / 0.6538 |
| Stable paired MXFP8 / FP4-dX, MXFP8-dW wide | 0.8228 | [0.7958, 0.8508] | 0.7988 / 0.8475 |
| Full FP4 coalesced / full FP4 wide | 1.0805 | [1.0340, 1.1301] | 1.0523 / 1.1095 |

Thus the producer improvement survives complete public graph execution against
its own FP4 predecessor in this session, but it does not make the full-FP4 or
tested mixed path competitive with the stable MXFP8 baseline at this shape.
These are within-session replay results, not cross-session confidence bounds,
eager host-overhead measurements, training wall throughput, or release
qualification. All four paired graph builders also had zero ordinary-call and
graph-replay live allocated-byte growth. Raw log retains a Torch shared-leaf
AccumulateGrad stream warning emitted during graph construction, outside timing;
the numerical/current-buffer guards passed.

## Reproduction and artifacts

```bash
.venv/bin/python -m benchmarks.compare_mixed_nvfp4_backward \
  --policy all --m 2048 --n 768 --k 1536 --samples 15 \
  --schedule-json '{"producer_load":"coalesced_wide"}' \
  --output autotune_reports/exact_mixed_comparison.json

.venv/bin/python -m benchmarks.compare_mixed_nvfp4_backward \
  --paired --m 2048 --n 768 --k 1536 --rounds 12 --replays 40 \
  --output autotune_reports/exact_mixed_paired.json
```

`--dx-schedule-json` and `--dw-schedule-json` independently override the shared
FP4 schedule. `--mxfp8-config-json` supplies an explicit MXFP8 backward config;
otherwise the repository's deterministic shape-specific default is used. Both
the research and stable baseline use the same selected MXFP8 config. The tool
refuses to overwrite existing outputs.

These are controlled schedule comparisons, not exhaustive tuning of either
precision's GEMM and preparation space. No global-optimum or universal
precision-crossover claim follows from this one public-path shape.

Ignored raw evidence is under
`autotune_reports/ms_eden_mixed_comparator_20261025/`:

- `exact_smoke_v2_wide.*.json`: four research-policy smoke results.
- `exact_smoke_v3_stable*.json`: stable paired smoke result.
- `exact_representative_wide*.json` and `.log`: complete five-policy comparison.
- `exact_representative_coalesced_*.json` and `.log`: three prior-load FP4 paths.
- `exact_paired_v2_representative.json` and `.log`: final paired raw rounds,
  both order windows, ratios, per-order estimates, confidence intervals,
  graph-builder guards, and frozen runtime hashes.
- `exact_paired_representative.*`: superseded initial paired diagnostic with
  only one AB or BA window per round, retained unchanged.
- Initial compile-failure logs are retained rather than overwritten.

The frozen exact runtime snapshot is
`/tmp/rtx-mixed-exact-20261025.TIXRgS` on the 5070 host. Earlier 5060 diagnostic
artifacts named `smoke_wide_v3*`, `representative_wide*`, and
`representative_coalesced*` were incorrectly labeled exact: their serialized
dynamic config proves they actually used power2 forward scaling. They are
retained unchanged and excluded from this exact comparison; see
`forward_policy_correction.json` beside them.

CPU orchestration/configuration tests: 53 passed across
`test_mixed_nvfp4_comparator.py`, `test_ms_eden_backward_policy.py`, and
`test_ms_eden_forward_policy.py`. CPU tests are not CUDA numerical evidence.
