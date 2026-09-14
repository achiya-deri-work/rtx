# MXFP8 GLU historical measurements and qualification

Historical, revision-specific evidence extracted from the user guide. Commands
and pending-work statements describe the recorded experiment, not current
API defaults. Use [the current guide](mxfp8_glu.md) and
[shared frontend controls](api.md#shared-frontend-controls) for integration.

## Historical initial RTX 5070 Ti evidence (2026-09-12)

The GLU tests cover all six activation/output combinations, ragged dimensions
including `(M,H,K)=(1,1,17)`, exact output scale bytes on the reference cases,
alternate schedules, zero input, nondefault streams, opcheck, and fullgraph
compilation. They pass on both supported PyTorch stacks. Ordinary MXFP8
forward/reference and schedule regression tests also pass after the base-kernel
epilogue hook was added.

At `(M,H,K)=(512,1536,768)`, a 48-trial BF16 search (including a resumed run)
found 66.04 microseconds; a separate 24-trial MXFP8-output search found 78.37
microseconds. Both winners passed two extra confirmations. All 72 evaluated
candidates passed their numerical gates. These are small, unequal-budget,
shape-local searches—not exhaustive tuning or a comparison with tuned unfused
alternatives. The returned configurations, not the conservative default, give
these timings.

Relative L2 against the **input-quantized FP32 projection/GLU reference** was
0.166% for the BF16-output winner and 2.642% for the MXFP8-output winner. These
measure epilogue/output error, not total error against unquantized BF16 inputs.

Padded-allocation Compute Sanitizer memcheck reported zero device-access
errors over the tails, formats, schedule variants and stream tests. API-error
reporting was disabled for that focused pass: the unfiltered pass reports
CUDA 13.3 binding probes against this machine's 13.2 driver. That environment
diagnostic is not claimed to be resolved.

## Expanded search and aggressive compiled baseline

The subsequent iteration added quantize-once projection, fragment-register
epilogues, and persistent buffered TMA output. All remain search choices;
the conservative default is **not** the measured winner. The selected
quantize-once path still quantizes the current BF16 X and W on every call.

Measured on the local RTX 5070 Ti with PyTorch 2.13.0+cu132, SiLU,
`fullgraph=True`, `dynamic=False`, and the aggressive options shared with
`benchmarks/benchmark_torchao_fp8_rowwise.py`. Freezing is disabled and both
operands are runtime inputs. The baseline is BF16 packed gate/up projection
plus activation/product; for MXFP8 output it also quantizes the result.

| M × H × K / output | Compiled baseline, uncaptured | Tuned GLU, uncaptured | Speedup | External CUDA-graph speedup |
|---|---:|---:|---:|---:|
| 512 × 1536 × 768 / BF16 | 29.28 µs | 23.32 µs | 1.26× | 1.61× |
| 2048 × 3072 × 768 / BF16 | 193.91 µs | 93.35 µs | 2.08× | 2.08× |
| 512 × 1536 × 768 / MXFP8 | 29.49 µs | 32.90 µs | 0.90× | 1.21× |
| 32 × 1536 × 768 / BF16, transferred config | 21.77 µs | 22.75 µs | 0.96× | 0.51× |

Uncaptured numbers are median synchronized wall time across 11 samples of
50 calls, after warmup. The external graph diagnostic uses 21 samples of
20 calls; it is **not** the requested `triton.cudagraphs=False` end-to-end
result. The small-shape configuration was transferred, not locally tuned.
MXFP8 output and small-batch performance remain open work. The MXFP8
allocation change did not remove its uncaptured overhead.

Relative L2 versus the unquantized BF16 computation was approximately 5.37%
for the medium BF16 result, 5.38% for the large result, and 6.48% for MXFP8
output. These are reduced-precision comparisons, not exact-equivalence
claims. Candidate acceptance separately checks the input-quantized FP32
projection/epilogue reference.

Reproduce or extend the search and comparison:

```bash
python -m rtx.autotune.glu --m 512 --hidden 1536 --k 768 \
  --backend auto --trials 128 --seconds 180 \
  --output-dir autotune_results/my_glu
python -m benchmarks.benchmark_mxfp8_glu --shape 512,1536,768 \
  --history autotune_results/my_glu \
  --output autotune_reports/my_glu.json
```

Local raw measurements are under `autotune_reports/glu_iteration/`, with
revision-separated observations under `autotune_results/glu_v3_*`.
They are intentionally not shipped as portable runtime winners.

The expanded searches recorded 495 successful evaluations and one static
implementation rejection, with no recorded compile or numerical failures.
The final correctness matrix passed 16 tests on PyTorch 2.12.1; on 2.13 the
10-test GLU matrix passed, followed by seven final compiler/shared-kernel
regressions. These cover both epilogues, all activations and output formats,
ragged K/interleaved loads, native scale TMA, dynamic weight updates, and
1–4 output buffers with persistent tail tiles.
The final padded-allocation memcheck also passed the prequantized GLU and
persistent TMA recycling/tail matrices with zero device-access errors, using
the same API-probe-reporting caveat described above.

Nsight Compute supports the structural explanation: the selected GEMM's
tensor-pipe active metric rose from roughly 11% for the original scalar
implementation to 50%, while per-CTA shared storage fell from approximately
100 KB to 28 KB. The quantize-once implementation adds its own quantizer
launch and does not reduce measured total L2 traffic (about 46 MB versus
40 MB). Its benefit is avoiding repeated conversion work in the GEMM and
improving the MMA schedule, not eliminating all memory traffic. Profiler
durations are diagnostic and are not used in the speedup table.

## Revision 4: vectorized output quantization

A 192-trial prequantized-backend search selected `glu_quant_vec=4` with the
register GLU epilogue. A fresh matched SiLU benchmark at
`(M,H,K)=(512,1536,768)` measured **28.80 µs baseline versus 21.62 µs GLU**,
or **1.332× graph-replay speedup**; the repeated baseline was 28.84 µs.
Uncaptured wall timing remained slower (29.58 versus 33.44 µs). Thus the
1.3× target is met for this replay case, not for uncaptured MXFP8 execution.

The small `(32,1536,768)` shape received its own 256-trial search across both
backends. It found approximately 7.69 µs, versus the previous transferred
configuration's 9.55 µs. This is still slower than the approximately 4.84 µs
compiled BF16 baseline; **the all-shapes 1.3× target is not met**. Changing
measurement policy does not turn this into a win. These searches are bounded,
not proof of a global optimum or a hardware lower bound.

Fresh replay comparisons after this change:

| M × H × K / output | Baseline | GLU | Speedup | ≥1.3× |
|---|---:|---:|---:|---|
| 512 × 1536 × 768 / BF16 | 28.58 µs | 17.73 µs | 1.612× | yes |
| 2048 × 3072 × 768 / BF16 | 192.12 µs | 92.12 µs | 2.086× | yes |
| 512 × 1536 × 768 / MXFP8 | 28.80 µs | 21.62 µs | 1.332× | yes |
| 32 × 1536 × 768 / BF16 | 4.85 µs | 7.70 µs | 0.629× | no |

A disposable-process probe of 64-column CTA tiles failed compilation with a
block-scale source/destination fragment-size mismatch. That restriction was
not relaxed in the supported search space. Narrower tiles need their own
validated scale layout; silently accepting them is not a tuning solution.

The new searches recorded 256 valid small-shape trials and 191 valid MXFP8
trials. One MXFP8 candidate failed the correctness gate: `mma64x128` native
scales had incorrectly been allowed with row-major consumer staging. The
failure reproduced with both output formats, vector width 1 as well as 4,
and separate as well as dual quantizers. The shared legality contract now
requires TMA for that native scale layout, rejecting the invalid combination
before compilation. The selected row-major winner is unaffected.

The 12-test GLU suite passed on PyTorch 2.13, and the new vectorization and
graph-replay tests passed on 2.12.1. Replay tests verify changing input/weight
values and stable live GPU allocation over 50 replays. Padded memcheck passed
all output vector widths and tails with zero device-access errors (API probe
reporting disabled for the previously documented binding/driver mismatch).

Reports live in `autotune_reports/glu_target13/`, and residual observations in
`autotune_results/glu_target13_*`. The benchmark emits `primary_mode`,
`primary_speedup`, `target_speedup`, and `target_met` while retaining both
timing modes. Use `--primary-mode uncaptured` to change the primary target.
