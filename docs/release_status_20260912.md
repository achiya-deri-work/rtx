# Local v1 hardening qualification

This is release-candidate evidence, not a `1.0.0` release declaration. The
blocking checklist remains [release_checklist.md](release_checklist.md).

## Changes requiring fresh qualification

- NVFP4 subgroup shuffle reductions and leader broadcasts now name only the
  participating subgroup. Partial-warp work previously used a full-warp mask.
  On the reproduced delayed candidate, packed values, block scales, and GEMM
  output now exactly match the default quantizer. Regression coverage spans
  four values-per-lane settings and both scale-computation policies.
- Physical 64-row NVFP4 GEMM tiles are quarantined. A saved large-grid
  configuration produced a maximum absolute mismatch of 43.6875 on fresh
  seeded inputs; its 128-row control matched exactly. This is an unresolved
  implementation defect, not a quantization tolerance adjustment. Logical
  64-row and ragged inputs remain supported with safe tiles.
- Warp-specialized regional epilogues remain quarantined after reproduced
  hangs. Direct and fragment-register MMA epilogues remain supported.
- TMA physical strides, register-allocation limits, transposed quantizer
  partitions, and packed SMEM swizzle widths are rejected before compilation.
- Failed confirmation invalidates a candidate, including when its initial
  screen passed. Nonfinite candidate/reference comparisons cannot qualify.
- Cold fused MXFP8 tuning no longer assumes compound search coordinates have
  config attributes. This failure was hidden by warm caches. All three legacy
  coordinate tuners also recheck current legality before cached trial reuse,
  and discard an invalid cached best instead of selecting it as the seed.

NVFP4 runtime kernel revisions are now dynamic **8**, delayed **3**, JIT-region
**10**, and packed inference **5**. Old NVFP4 winners/model heads are not
release evidence and must not be relabeled with these revisions. MXFP8
revisions are unchanged; loaded winners are checked against current legality.
No historical observations were rewritten or winners promoted in this work.
Cache invalidation does not repair already packed tensor payloads. Re-quantize
from the original BF16 weights if a saved packed operand was produced with an
affected partial-warp shuffle/leader-broadcast configuration; loading it cannot
recover quantization information that was already lost.

## Reproduction and artifacts

The local device is an RTX 5070 Ti, driver 595.84. Supported environments are
`.venv` (PyTorch 2.13.0+cu132) and `.venv-torch2121` (2.12.1+cu132).
The 2.12.1 environment also passes `pip check`. The existing 2.13 environment
has an unrelated optional `quack-kernels==0.6.3` installation whose metadata
requests CuTe 4.6 rather than RTX's required 4.7. RTX does not depend on Quack;
that extra was not removed. These are fresh-process GPU checks, not a claim
that the existing 2.13 environment is dependency-clean. GPU CI creates isolated
environments from the supported requirements files.
Reports carry package/kernel source hashes, runtime identity, validator hashes,
and the dirty-worktree flag; they are not represented as tagged-commit runs.

```bash
bash benchmarks/run_release_qualification.sh .venv/bin/python
bash benchmarks/run_release_qualification.sh .venv-torch2121/bin/python
```

Run these serially on a device. The script creates separate test/matrix logs,
reports exit status, rejects evidence-directory reuse, and runs the production
matrix in a fresh process after the test suite. GPU CI uses the same script.

Package build, metadata validation, and a clean-environment wheel import passed
locally. Python source compilation and CPU contract checks passed (245 tests,
111 GPU skips).

Final local GPU qualification passed on **both** supported PyTorch stacks:

| Stack | Full unit suite | Production matrix |
|---|---|---|
| 2.13.0+cu132 | 356 run, 355 passed, 1 explicit quarantine skip | 49/49 passed |
| 2.12.1+cu132 | 356 run, 355 passed, 1 explicit quarantine skip | 49/49 passed |

Artifacts: `autotune_reports/v1_torch213_final/` and
`autotune_reports/v1_torch212_final/`. Both scripts returned zero for tests and
matrix, and both report the same final package source hash:
`cd8c481ee5f4dd2f7b0c31f59cf8c7461742be27d83fc5679284b5a5cba47eef`.
The suites include real-device `torch.library.opcheck`, first-hit tuning,
same-process reuse and cross-process persistent-winner loads, compiled DINO/ViT
integration, gradients, PTQ, ragged shapes and the partial-warp regression.

The production matrix includes aligned/ragged default JIT-region, delayed
inference, partial gradients, multiple streams and bounded caches. At M=8192,
the four tested FP32 dW reduction schemes had approximately 0.03773 relative L2
error against their FP32 reference. This measures MXFP8 approximation error,
not FP32 accumulation drift in isolation. The variable-shape runner caches
remained at their configured eight-entry bounds and recorded evictions.

Fresh-seed replay of the remaining NVFP4 historical failures covered 14 distinct
configuration/shape pairs: five delayed and two weight-prequantized cases passed
with maximum absolute error zero against the harness reference; seven packed
cases were rejected by the 64-row quarantine. Fresh inputs cannot prove that a
historical different-input failure is solved; the two weight-prequantized cases
therefore remain conservative non-reproductions, not retroactively cleared data.
Artifacts: `autotune_reports/v1_nv_replays/`.

## Matched compiled training check

All five modes completed 200 steps, 1,638,400 tokens each, with identical seed,
initialization, corpus windows, optimizer and schedule. The 2.69M-parameter
decoder uses four layers, width 256, intermediate width 512, four heads, batch
64 and sequence length 128. The model and loss are compiled fullgraph; the
optimizer uses FP32 master parameters/state with BF16 execution weights and
gradients. Autotuning was disabled for this integration check.

| Mode | Final validation loss | Diagnostic step ms | Peak allocated GiB |
|---|---:|---:|---:|
| BF16 | 1.44452 | 3.238 | 0.447 |
| MXFP8 | 1.45329 | 3.451 | 0.502 |
| NVFP4 delayed | 1.48480 | 3.436 | 0.528 |
| NVFP4 JIT-region | 1.48863 | 3.422 | 0.497 |
| NVFP4 block | 1.50829 | 3.365 | 0.503 |

Diagnostic step time is 8192 tokens divided by median interval throughput at
steps 40–200; compilation, validation and checkpoint I/O are excluded by the
trainer. This is not a stabilized, tuned kernel benchmark. BF16 is fastest for
this tiny model. These results do not demonstrate long-run equivalence; NVFP4
has a larger short-run loss gap than MXFP8. No acceptance tolerance was changed.

```bash
python benchmarks/train_decoder.py --steps 200 --batch-size 64 \
  --sequence-length 128 --layers 4 --hidden-size 256 --intermediate-size 512 \
  --heads 4 --warmup-steps 20 --validation-batches 4 --validation-interval 100 \
  --checkpoint-interval 200 --log-interval 20 --autotune off --compile \
  --optimizer fp32_master --no-resume --output training_results/v1_local_qualification
```

Artifacts are under `training_results/v1_local_qualification/` and
`autotune_logs/v1_compiled_training.log`. The manifest's package source hash
matches the repaired source used for final qualification.

## Remaining release requirements

### Laptop archive received after local qualification

`release_20260911T213951Z.zip` is from the RTX 5070 Laptop GPU with PyTorch
2.13.0+cu132, clean commit `f325b25`, and matching package/kernel/validator
hashes. Its production matrix passed 49/49, but `tests_exit=1`: all 34 schedule
subcases in `test_real_schedule_variants_match_reference` disagreed at the
same element `(91, 247)` of a 257x385 output. Each disagreement was one BF16
ULP (0.00390625). There were 22 skips: 21 missing-DINO-checkpoint tests and the
quarantined epilogue. This archive does **not** clear the laptop release gate.

Local replay with the same seed did not reproduce the discrepancy and gives a
different value at that coordinate, so the laptop's actual inputs are needed.
Do not loosen the zero-tolerance test or blame a kernel based only on this log.
Run this small diagnostic on the laptop; it saves `inputs.pt` and `report.json`:

```bash
python benchmarks/diagnose_mxfp8_rounding.py \
  --output-dir autotune_reports/mxfp8_rounding_laptop_v1
```

Copy that directory back. It can be replayed without depending on device RNG:

```bash
python benchmarks/diagnose_mxfp8_rounding.py \
  --inputs autotune_reports/mxfp8_rounding_laptop_v1/inputs.pt \
  --output-dir autotune_reports/mxfp8_rounding_laptop_replay
```

The diagnostic compares five schedules with FP32 and CPU FP64 references,
records exact rational dot products at discrepant cells, and checks BF16
rounding without an intermediate FP32 cast. It is deliberately not a release
gate: legal FP32 accumulation need not equal an exactly rounded FP64 dot
product. No numerical tolerance or production kernel was changed.

Laptop PyTorch 2.12.1 qualification remains outstanding.

The same final source needs qualification on the laptop 5070, and the 5090
when available. Historical cross-device datasets cannot satisfy a new-kernel
correctness gate. Rebuild and remeasure before promoting new NVFP4 winners.

The quarantined implementations are excluded from the supported v1 search
space; they do not need to be enabled to release. They do need independent
large-grid numerical/synchronization validation before any future restoration.

Short matched training can catch integration regressions but does not establish
long-run convergence equivalence or a universal throughput advantage. A release
must not promise the earlier requested 1.3x/1.35x model speedups without measured
evidence for the specific workload.
