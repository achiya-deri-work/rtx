# NVFP4 backward and training research

This is the consolidated entry point for the retained NVFP4 research. It
separates reusable experimental integration from kernel prototypes and dated
execution records. It does **not** promote an experimental backward policy or
schedule into stable runtime defaults.

## Choose the right interface

| Use | Interface | Contract |
|---|---|---|
| Ordinary linear training/inference | [`rtx.NVFP4Linear`](nvfp4_linear.md) | NVFP4 forward, MXFP8 backward |
| Fused, recomputed SiTU MLP | [`rtx.NVFP4MLP`](nvfp4_mlp.md) | Experimental NVFP4 forward, FP32 SiTU, MXFP8 backward |
| Explicit FP4 backward studies | `rtx.experimental_ms_eden.ms_eden_linear` | Opt-in MS-EDEN; distinct numerical and RNG policies |
| Isolated backward tuning | `rtx.autotune.ms_eden_pipeline` / `ms_eden_design` | Correctness-checked research schedules, no automatic promotion |
| Whole-model schedule refinement | `rtx.autotune.make_ms_eden_model_adapter` | Fixed kernel designs, per-shape stream ownership, full-model evaluator |

Experimental MS-EDEN backward is not a claim of unbiased gradients for every
distribution. It retains original BF16 operands by default; packed-forward
state reuse and mixed dX/dW policies are explicit numerical choices. FP32
scales/corrections and FP32 accumulation are retained. Do not substitute this
policy for MXFP8 backward merely because an isolated benchmark is faster.

## Consolidated implementations

`paired_dw_first` is the reusable orchestration addition: dW owns the caller
stream and is submitted before side-stream dX. Serial, paired, and
operand-paired alternatives remain available. Independent RNG draws, explicit
storage lifetimes and exception-safe joins are unchanged. Pipeline/design
context revisions 4/7 distinguish the expanded search domain. Choose using
the complete application objective, not an assumed universal winner.

The model adapter composes with the existing residual stores, UCB scheduler,
portfolio and coordinate search. Batch, sequence, source/runtime provenance,
model recipe and exact starting designs participate in identity. It does not
install a public runtime winner. See [composable search](ms_eden_composable_search.md).

Additional `rtx/kernels/ms_eden_*.py` implementations are **research-only**:

| Family | Purpose | Qualification boundary |
|---|---|---|
| CTA finalizer | Cooperative reduction of producer maxima | Same final scales with identical uniforms |
| Dual-dY / quad preparation | Interleaved preparation roles and fewer launches | Not necessarily fewer tensor reads or faster execution |
| Shared-dY roles | Reuse a staged input between row/transposed preparation | Resource envelopes can erase traffic savings |
| Inline scales | Finalize stochastic scales at GEMM load | Fixed uniforms and already-computed global maxima required |
| Grouped/direct refinement | Amortize reciprocal/refinement work | Guarded domain; precise fallback outside it |
| Paired FP32 | FP32x2 transform/statistics instructions | Same-policy exactness checked separately from speed |
| H8/H16 and TF32 transforms | Tensor-core-assisted Hadamard experiments | Guarded source/sign domains; changed association is not bitwise scalar equivalence |

These prototypes are not silently dispatched by `NVFP4Linear`, `NVFP4MLP`, or
the stable backward. Their `benchmarks/experiment_ms_eden_*.py` wrappers scope
any research dispatch changes. Keep negative experiments: they explain why
apparently attractive launch/traffic reductions were not promoted.

## Training and profiling tools

- `benchmarks.train_precision20m`: matched configurable decoder study, BF16
  boundaries, FP32 sensitive operations, SiLU/SiTU, explicit precision policies,
  frozen per-shape configs, resume identity and separate training/total clocks.
- `benchmarks.study_precision200m_tuned`: fresh component tuning, batch-aware
  preparation, full-model refinement, then training; tuning is outside the
  training clock.
- `benchmarks.refine_ms_eden_model`: fresh-process application-context search
  and independent confirmation before exporting a design map.
- `benchmarks.run_precision200m_long`: historical two-board study recipe,
  confirmed-config reuse, GPU lease/busy checks, frozen manifests and terminal
  completion checks. It is not a general multi-GPU training launcher.
- `benchmarks.profile_backward_subparts` and `tools/profile_backward_*.sh`:
  component/timeline/counter diagnosis. Profiler durations are not speed claims.
- Other `probe_*`, `compare_*`, `validate_*` and `experiment_*` programs retain
  the associated reproduction and negative-control work. Some consume a prior
  study bundle; their historical batch/GPU assumptions are not runtime APIs.

The decoder SiTU helper is a compositional reference/training recipe. Adding
SiTU to `DecoderConfig` does not automatically replace its MLP with the new
fused `NVFP4MLP` implementation.

## Evidence, not universal performance claims

The captured-model scheduling correction is documented in
[the October 28 qualification](precision200m_schedule_fix_20261028.md).
Its matched 100M-byte-token crossover runs on both home 5090s showed effectively
tied full-model throughput, not a robust FP4-backward speed win. Isolated kernel
improvements do not imply the same gain in training.

The later [600M-byte-token study](precision200m_600m_home_20261028.md) completed
both arms. Full NVFP4 ended at validation loss 0.40467 versus hybrid 0.37308.
Different physical boards confound a direct speed comparison; this single-seed
study does not establish general convergence equivalence. It supports retaining
MXFP8 backward as the stable default.

Dated documents describe the state **at their recorded time**. Statements such
as "running", "pending", or "not committed" in historical notebooks are not
current job/status instructions. Start with this page and the qualified reports,
not an old launch record. Key records:

- [Initial backward component profile](nvfp4_backward_profile_20261025.md)
- [21M training results](precision20m_results_20261025.md)
- [Guarded TF32/H8](ms_eden_tf32_h8_hadamard.md) and
  [scale-free exactness scope](ms_eden_guarded_scale_free.md)
- [Historical speed-target log](ms_eden_speed_target_20261026.md) and
  [October 27 status](ms_eden_speed20_status.md)
- [Fresh batch-aware tuning](precision200m_fresh_tuning_20261028.md)
- [Model gap](precision200m_backward_gap_20261028.md),
  [graph investigation](precision200m_graph_execution_20261028.md), and
  [integrated fix](precision200m_schedule_fix_20261028.md)

Raw datasets, profiles, logs and checkpoints remain ignored under the existing
`autotune_reports/` and `training_results/` paths, backed by Storage 4 locally.
They are not shipped in the Python package or committed to Git.

## Consolidation checks

The November 3 consolidation ran the related CPU suite with CUDA masked:
**996 passed, 853 skipped** in 10.81 seconds. Skips are predominantly explicitly
gated GPU research cases; they are not proof that every prototype is qualified
on every GPU. Reproduce the CPU surface with:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_ms_eden* tests/test_experimental_ms_eden.py \
  tests/test_precision20m* tests/test_precision200m* \
  tests/test_situ_decoder.py tests/test_compare_ms_eden_tf32_hadamard.py
```

The focused GPU-enabled consolidation suite then passed **122 tests in 35.10
seconds** on the idle work RTX 5090, Torch `2.13.0+cu132` / CUDA 13.2. It covers
the public experimental paired backward and RNG/compile/graph/checkpoint/memory
contracts, representative retained preparation prototypes and compiled SiTU.
This is a smoke matrix, not every prototype's Cartesian numerical domain or
performance qualification. The decoder/API regression subset additionally
passed 24 tests and 12 subtests with one CUDA-only skip locally.

Reproduce the focused GPU smoke from the repository root, on an assigned idle
SM12x GPU, with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/qualify_nvfp4_research.sh \
  .venv/bin/python autotune_reports/nvfp4_research_check
```

The executed evidence is under
`autotune_reports/nvfp4_consolidation_20261103/{pytest.log,results.xml}`;
the clean remote snapshot was staged from `3905bfe` plus this consolidation.

Real CUDA tests opt in per family, for example `RTX_RUN_PAIRED_CUDA=1` for the
public experimental backward/graph/RNG contract. Run only on an assigned idle
GPU. Historical tests, newly executed checks, numerical equivalence and
performance qualification are separate claims.
