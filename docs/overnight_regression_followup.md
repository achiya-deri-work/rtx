# Overnight autotuner regression follow-up

Development diagnosis on the RTX 5070 Ti, 2026-09-14. These selected worst cases
are not a new held-out policy evaluation. No learned policy has been promoted
to the runtime default, and no broad collection was started.

## Paired winner replays

`tools.regression_replay` reconstructs the saved winners, validates them, and
uses separate prepared adapters with the same input seed. Twelve randomized
AB/BA rounds compare their existing CUDA-event/graph measurements. The ratio
below is baseline latency / candidate latency: larger is better. These are
comparisons against the existing autotuner, **not BF16**.

| Winner source / workload (M,N,K) | Original ratio | Local paired median | Paired range | Baseline winner in learned pool? |
|---|---:|---:|---:|---|
| Local MLP forward (32,768,640), 32 trials | 0.856 | 0.872 | 0.861–0.874 | No |
| Local MLP training (8192,128,896), 32 trials | 0.903 | 0.907 | 0.906–0.909 | No |
| Laptop MLP training (128,1280,896), 8 trials | 0.500 | 0.950 | 0.948–0.952 | Yes, also measured |
| Laptop tensor-delayed (257,1280,896), 8 trials | 0.600 | 1.114 | 1.101–1.118 | No |

Both local regressions reproduce. Their better configurations were never
available to the learned selector. This establishes a candidate-coverage gap;
it does not establish that including those configurations would guarantee
selection within the same trial budget.

The laptop MLP records include roughly 145 us and 700 us measurements for the
same initial configuration and input context. That is an execution/timing
stability warning, not evidence of a specific thermal, power, or compiler cause.
Replaying laptop winners on a desktop cannot validate their laptop performance.

Artifacts: `autotune_reports/regression_replay_v1/plan.json` and per-case JSON
files. They include full outcomes and pool-membership evidence. Regenerate with:

```bash
.venv/bin/python -m tools.regression_replay prepare \
  --sources PATH_TO_LOCAL_STUDY PATH_TO_LAPTOP_ZIP --output NEW_REPORT_DIR
.venv/bin/python -m tools.regression_replay replay \
  --output NEW_REPORT_DIR --case CASE_ID
```

## Kernel contract repairs

### Contiguous views are not necessarily 16-byte aligned

Splitting gate/up weights of odd dimensions produces contiguous BF16 views
whose base pointers can be only two-byte aligned. Quantizer fake tensors had
promised 16-byte alignment. Relative-offset vector-load guards alone did not
repair that promise.

Dynamic prequant launchers now specialize on actual source alignment (2/4/8/16
bytes), include it in runner-cache keys, and check absolute vector-load alignment
where necessary. Aligned inputs retain the original fast path. No intermediate
copy or numerical tolerance relaxation was added. The MXFP8 frontend compiler
token revision is incremented to 8.

Independent quantizer tests compare FP8 values and E8M0 bytes exactly, across
offsets 1/2/4/8 BF16 elements and K=128/129/777. Compiled MLP forward/backward
checks cover (17,97,65) and the observed failure shape (513,1543,777), using the
existing numerical gates.

### Singleton TMA source dimensions

The M=1 probe completed quantization but faulted on the first TMA X load.
Preserving singleton source extents as symbolic CuTe dimensions fixes the
tested path while keeping their runtime value one. This is not physical
padding. The helper is shared by prequant MXFP8 and NVFP4 GEMM compilers.

An independent dequantize-plus-FP32-GEMM reference for M=1,N=1280,K=896 gives
relative RMS error 0.00172036 and finite output after the fix. Seven adapter
paths were checked at that shape: MXFP8 prequant forward/backward, NVFP4
dynamic/delayed/JIT-region forward, and compiled MLP forward/training.
This does not qualify every fused backend, singleton axis, or autotune config.
The TMA-store epilogue remains statically illegal for incomplete output tiles.

## Remaining work before policy promotion

1. Protect baseline search coverage within a fixed budget; evaluate coverage
   and ranking separately rather than relying on a more complex predictor.
2. Add or evaluate paired incumbent/challenger confirmation for unstable timing
   contexts. Do not use desktop replays to relabel laptop observations.
3. Investigate the five numerical-failure observations from the local overnight
   study. They remain quarantined; these repairs do not clear them.
4. Run fresh, held-out shape comparisons at 8/16/32/64 trials after freezing
   the next policy. Existing worst cases are now development data.

Original bundles remain untouched. Old frozen collection plans should not be
reused with changed kernel sources; generate a new plan for subsequent runs.

## Second follow-up: row-crossing quantization and search safeguards

The two large backward-error observations shared one configuration:
M=4096,N=640,K=2560, oriented dual quantization with `quant_vec=8`.
For the dX operand reducing over 640 elements, each row contains 20 scale
blocks. The old warp-task mapping decoded the row before adding the lane's
scale-block index. Work spanning a row boundary consequently skipped valid
blocks and wrote invalid logical positions. The corrected mapping decodes
both row and scale index from the complete linear block index.

Independent row-major and logically transposed quantizer checks at reduction
widths 384 and 640 now match FP8 values and scale bytes exactly. An additional
partial-warp test exposed a hang; that test process was stopped. The low-level
oriented compiler now rejects incomplete warp scale groups before compilation
instead of silently accepting them. This restriction is not applied to the
ordinary row quantizer. BWD frontend revision is 4; backward autotune kernel
revision is 23 to invalidate old backward winners.

Fresh-adapter replay (not timing-only replay of an already validated graph)
was run three times for each of the five saved failure observations:

- Both saved backward observations pass after the fix. The repeated backward
  reference-relative L2 is 3.7778%, down from roughly 37% before the fix.
- The MLP (6144,2304,1920) case improves from approximately 20–21% dX error
  against the quantized reference to 0.0167%; all four output/gradient errors
  are below 0.018% in all three checks.
- MLP (2048,4608,1152) passes three checks, but its errors vary between checks.
- MLP (4096,640,2560) still fails one of three checks: output relative L2
  0.3387% exceeds the unchanged 0.3% gate. It remains quarantined. No claim of
  full numerical qualification is made for the other variable case either.

Artifacts are in `autotune_reports/numerical_failure_replay_fixed_v1/`.
The original observations remain unchanged. Subsequent MLP failures report
named output/dX/gate-up-gradient/down-gradient relative L2, maximum absolute
error, RMS error, reference RMS, and finiteness instead of stopping at the
first failing output.

### Opt-in search safeguards

Explicit seeds supplied to `mixed_pool` now receive priority before ordinary
retrieval/structural/local fill. Previously only the initial seed had a reserved
slot. Legal seeds are deduplicated and bounded by pool capacity.

`AutotunePolicy(baseline_trials=2)` reserves two `coordinate_local` arm pulls
inside the existing cumulative trial budget. Retrieval opening respects this
reservation. This is an opt-in coverage floor, **not** a guarantee of reproducing
the baseline autotuner's winner or trajectory. Deadlines or exhausted legal
neighbors can still end search early. The default remains zero; learned policy
promotion needs new fixed-budget evidence.

### Balanced finalist confirmation

`rtx.autotune.confirmation.confirm_finalists` is a reusable, separately tested
primitive. It alternates randomized forward/reverse finalist orders, emits
every measurement to a caller-provided recorder, disqualifies failed/nonfinite
candidates, and chooses using within-round normalized latency. A deadline
prevents admission of another balanced pair; an admitted pair may overrun it.
It does not silently rewrite old observations or promote runtime cache winners.

The replay tool's `--balanced` option exercises this primitive and writes an
individual JSON residual for each measurement. On the local MLP-forward
regression it selected the baseline winner after 24 evaluations. Default
first-hit orchestration still uses its existing confirmation scheme; integrated
promotion/resume semantics and the extra wall-time cost remain to be evaluated.

The combined CUDA/frontend/autotuner regression suite passes 42 tests:

```bash
.venv/bin/python -m unittest \
  tests.test_quant_source_alignment tests.test_autotune_sampling \
  tests.test_autotune_effort tests.test_paired_confirmation \
  tests.test_prequant_autotune tests.test_mlp_precision_api -q
```

This suite exposed cached MLP wrappers retaining old backward configuration
keys after the backward revision change. MLP configuration revision is now 9
and the MLP tuning context revision is 34, invalidating those wrappers/history
contracts. Frozen revision-33 predictor artifacts must not be silently relabeled
as revision 34; validate and rebuild their compatibility contract before reuse.
