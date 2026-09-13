# Prior-study follow-up: validation and research policy

## Changes

- NVFP4 weight-prequantized inference now rejects `e4m3_lut` and `rcp_approx`
  activation reciprocals before compilation, including candidates imported from
  old datasets. Its tuning axis retains only `direct`. The tensor-current
  TorchAO packing contract is unchanged; tolerances were not relaxed. Other
  NVFP4 scaling families are not disabled by this scoped quarantine.
- MLP autotuning gives each prepared candidate fresh autograd leaves. Its first
  correctness run, warmup, and capture all use the same stream; the prepared
  graph retains those leaves and stream. The measurement tag changes to
  `candidate_leaves_same_stream_v2`, preserving the distinction from old data.
- MXFP8 GEMM now rejects the combination `tile_m=64`, `stages=1`, and
  `scale_role="consumers"`. A stress test reproduced intermittent errors in
  that path with exact quantized operands. Two-buffer 64-row and 128-row
  alternatives remain legal. The guard also covers BF16-gate MLP schedules
  using the same GEMM. This contains the demonstrated failure, not its yet
  unidentified low-level root cause.
- The default prequantized MLP gate schedule and its low-latency seed now use
  two buffers so that default BF16-gate execution remains legal. This is a
  correctness-driven schedule change, not promotion of a new search policy.
- Retrieval ordering no longer initializes predictors or builds candidate model
  features. Staged and family-routed research policies can request a short
  prefix without evaluating later stages. They remain explicit and do not
  change runtime defaults or retrain the frozen prior.

## Local evidence

All GPU checks here ran on the local RTX 5070 Ti, not on the laptop or 5090.
The raw prospective study was left unchanged.

- A recorded `rcp_approx` NVFP4 case reproduced with maximum absolute error
  2.5, and an `e4m3_lut` case reproduced with error 3.25 against the existing
  harness oracle. Changing only the reciprocal to `direct` passed both cases.
- All 32 distinct local NVFP4 failing schedule/workload cases passed after
  that one-field change, with zero error against the harness reference.
  This is a control experiment, not proof that the historical candidates
  themselves became correct: they are quarantined.
- The standalone TorchAO NVFP4 quantization-reference test and calibrated
  packed-inference harness tests passed, providing independent-reference
  coverage beyond the schedule-to-schedule comparison.
- All 11 distinct historical MLP capture cases passed three evaluations each,
  interleaved with baseline preparation. The historical MLP forward numerical
  case also passed three evaluations. Forward and all three input/weight
  derivatives were checked against the existing independent quantized FP32
  reference for training cases.
- The capture exception did not reproduce in the first isolated **pre-fix**
  replay either. These passes validate the revised lifecycle but do not prove
  exclusive causality for every historical capture exception.
- Both local MXFP8 prequant numerical cases initially passed isolated replay.
  Subsequent stress reproduced them; see the ablation table below. The MLP
  numerical case uses the same quarantined schedule combination. Numerical
  gates were not relaxed and kernel arithmetic was not changed.
- CPU regression: 50 tests ran, five GPU tests skipped in the sandbox.
  GPU focused suite: three tests passed; the strengthened candidate-switching
  regression then passed separately with distinct configurations.
  After the MXFP8 quarantine, an expanded CPU run completed 65 tests with
  seven GPU tests skipped.
  Final expanded regression: 87 tests passed with nine GPU-only skips; the
  separate retained-scale-transport, independent GEMM-reference, and repeated
  capture GPU tests all passed.

The archived local failure set contains 46 distinct schedule/workload cases.
The final follow-up produced 70 recorded evaluations: 32 NVFP4 direct controls,
36 MLP evaluations, and two MXFP8 replays; all passed. For the final run,
quarantined MXFP8/MLP single-buffer schedules were explicitly changed to two
buffers and their new configuration IDs were recorded. These are correctness
and lifecycle diagnostics, not performance benchmarks or full release approval.

### MXFP8 stress localization

Each cell reports failures / 128 launches. Both packed operands were compared
with an independent E4M3/E8M0 quantization implementation and matched exactly
in every launch. Outputs were compared with FP32 GEMM on those independently
quantized operands, cast to BF16, using the unchanged rtol=0.05/atol=0.5 gate.
Runs rotated operand buffers and used an explicit non-default CUDA stream.

| Change from recorded candidate | 1536×1536×1536 | 6144×3072×1024 |
|---|---:|---:|
| None: 64 rows, one buffer | 4/128 | 27/128 |
| Two buffers | 0/128 | 0/128 |
| 128 rows | 0/128 | 0/128 |
| Scale loads after pipeline wait | 3/128 | 15/128 |
| Preload MMA schedule | 3/128 | 20/128 |

Failures reached maximum absolute error 40.0; this is not normal BF16 rounding.
The passing two-buffer and 128-row variants had maximum absolute error 0.5.
One instrumented racecheck launch passed with zero reported hazards; that does
not disprove an intermittent or uninstrumented hardware/compiler ordering issue.
The quarantined single-buffer path is not eligible for runtime-cache promotion.

## Reproduction and artifacts

```bash
.venv/bin/python -m unittest \
  tests.test_portable_prior tests.test_first_hit tests.test_compile_pipeline \
  tests.test_nvfp4.NVFP4ConfigTests tests.test_mlp_precision_api \
  tests.test_documentation -q

.venv/bin/python -m unittest \
  tests.test_mlp_precision_api.PrecisionAPITests.test_autotune_candidate_capture_uses_fresh_leaves \
  tests.test_nvfp4.NVFP4CudaTests.test_current_matches_torchao_quantization_reference \
  tests.test_nvfp4.NVFP4CudaTests.test_calibrated_inference_harnesses_execute -v
```

Machine-readable evidence is in
`autotune_reports/prior_prospective_three_device_v1/`:
`failure_cases_local.json`, `replay_*.json`, and `fix_validation.json`.
The localization records are `mxfp8_stress.json` and
`mxfp8_stress_{stages2,tile128,after_wait,preload}.json`.
The cases retain original task, seed, shape, schedule, and error; replay records
identify modified configurations explicitly. No historical records were rewritten.
Supplementary scripts `validate_prior_fixes.py`, `stress_prior_mxfp8.py`, and
`replay_prior_case.py` are saved alongside the reports. From the repository root,
run them with `PYTHONPATH=.` and the project Python. The stress script's original
baseline now rejects by design; use `RTX_STRESS_VARIANT=stages2` or `tile128` to
validate retained paths. Reproducing the quarantined baseline requires the old
source revision, in a separate checkout/environment rather than bypassing the
production safety guard.

## Remaining gates

1. Find the low-level cause of the quarantined 64-row single-buffer GEMM and
   test any proposed repair under long-lived candidate sequences. Do not
   re-enable it based on isolated success. Other historical failures must
   still pass their own numerical checks.
2. Run the revised capture harness on other GPUs when available.
3. Qualify staged family routing on a fresh holdout and measure isolated
   first-hit wall time, including loading, compilation and confirmation.
4. Diagnose laptop timing stability before using sub-percent differences there.
