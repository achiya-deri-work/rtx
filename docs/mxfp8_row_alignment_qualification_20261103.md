# MXFP8 half-tile dispatch qualification

Date: 2026-11-03. Device: `work_5090`, RTX 5090, 170 SMs, 32 GB.

## Result

The packed G2DETR B48 reproduction improved from **342.85 to 301.54 ms/step**,
or **140.00 to 159.18 images/s** (+13.7%). It uses the original 58,368 patch
capacity, not the enlarged alignment-control capacity. No BF16 activation
padding, quantization-policy change, or required caller-managed resources were
introduced.

This fixes the reported materialized-family eligibility cliff. It does not
establish perfectly linear application scaling: on the patched source, B32 is
164.54 images/s and B48 is 159.18 images/s, a remaining 3.3% throughput dip.

## Diagnosis and implementation

Previously, automatic dynamic dispatch tested the aligned MMA128 default and
treated that configuration's rejection as rejection of the whole materialized
family. The eight reported `M % 128 == 64` geometries therefore fell back to
fused execution despite existing legal native MMA64/MMA128 implementations.

Dynamic default selection now checks, in order:

1. The unchanged aligned native MMA128 configuration.
2. An existing native MMA64 activation / MMA128 weight configuration.
3. An existing masked row-major/direct-store configuration for other tails.

All normal legality checks remain active. Explicit schedules are not rewritten.
The shared prequant tuner uses the same legal seed selection. These are starting
configurations, not forced universal winners: preparation search retains fused
and materialized alternatives and the component configuration spaces remain
available.

Backward opening coverage also includes a legal native-scale dX plus existing
split-FP32 dW candidate. Native layouts are added as compound backend moves, not
hard-coded winners. Eight-trial application tuning independently selected
native64 dX for six eligible half-aligned cases; dW retained its split reduction.
The separate N=64 coarse projection uses the masked family and is not one of the
eight newly half-aligned geometries.

Forward compiler identity is revision 9, preparation identity 4, and backward
search identity 29. Retune affected contexts; do not transplant old winner JSON
across source/runtime contexts. Packed-inference layout contracts are unchanged.

## Same-tensor isolation

The forward comparison below uses default-mode `torch.compile(fullgraph=True,
dynamic=False)`, followed by CUDA-graph replay. Each measurement includes dynamic
X and W quantization and GEMM. The old fused schedules are read explicitly from
the application's recorded winners, not installed into the new runtime cache.
Times are medians in microseconds, seven samples of ten replays after warmup.

| M × N × K | Recorded fused | Patched automatic materialized |
|---|---:|---:|
| 3648 × 128 × 256 | 22.54 | 6.14 |
| 3648 × 512 × 256 | 43.21 | 8.14 |
| 76608 × 128 × 256 | 182.04 | 32.97 |
| 76608 × 256 × 256 | 362.60 | 49.08 |
| 76608 × 384 × 256 | 544.16 | 75.66 |
| 76608 × 1024 × 256 | 1709.50 | 152.30 |
| 310080 × 256 × 128 | 980.17 | 217.42 |
| 310080 × 256 × 256 | 1786.08 | 333.15 |

Adjacent aligned automatic controls are 48.72 us at `76672 × 256 × 256` and
322.09 us at `310144 × 256 × 256`. The isolated small alignment change no longer
forces a different, much slower family.

An additional direct-launch/graph-replay backward probe held dW's recorded
schedule fixed and changed only dX to the existing native layout:

| Shape | dX before → after, us | Complete backward before → after, us |
|---|---:|---:|
| 76608 × 256 × 256 | 121.34 → 51.17 | 350.58 → 281.56 |
| 310080 × 256 × 256 | 540.96 → 335.87 | 1676.69 → 1457.57 |

dW stayed approximately 224 us and 1130 us respectively. Backward totals include
preparation and finalization; they need not equal independently measured sums.

## Complete application reproduction

The preserved `/home/achyad/g2detr-perf-20261103` worker was unchanged. The baseline
uses RTX `9aef47fc5832f23ddca787a0cd0f9d1041958139`; the candidate overlays this
patch on that exact source without unrelated experimental working-tree edits.
Original report:
`work/G2DETR/docs/reports/rtx_mxfp8_batch_scaling_20261103/README.md`.

Both B48 runs use 24 measured steps, three rotating resident synthetic cases,
random frozen backbones, packed mixed lengths, batched matching and the same
8-trial/20-second soft admission budgets. Candidate tuning is fresh and
source-context-matched. Compilation, tuning and warmup are excluded from timing.
This is execution and short-step numerical qualification, not convergence.

| Run | Patch capacity | Median step, ms | Images/s | Device-used GiB after warmup |
|---|---:|---:|---:|---:|
| B48 baseline | 58,368 | 342.845 | 140.005 | 27.242 |
| B48 patched | 58,368 | 301.540 | 159.183 | 27.459 |
| B32 patched | 38,912 | 194.479 | 164.542 | 18.658 |

The materialized path trades about **222 MiB** more device memory for less
repeated work. Live PyTorch allocation was constant over all 24 measured steps:
4,047,857,152 bytes for patched B48 and 2,907,271,680 bytes for patched B32.
Device-used memory and allocator live memory are different metrics.

For B48, GPU phase medians changed from 128.13 to 94.75 ms for forward and
198.55 to 190.56 ms for loss/backward. Matching and optimizer remained about
3.15 and 12.87 ms. Both builds used one model graph and one loss graph with no
steady-state recompilation; required gradients were finite and frozen backbones
had no gradients. Default compile mode was used, not `reduce-overhead`.

CUPTI traces separately show RTX GEMM/linear/reduction sums decreasing from
110.26 to 63.86 ms per profiled step while quantization increases from 34.84 to
37.99 ms. These family sums include backward, may overlap, and are not wall
time. Fresh tuning can change other schedules too; the isolated probes establish
the dispatch mechanism without attributing every application improvement to it.

The original report's aligned-capacity B48 control was 299.73 ms/160.14 images/s.
That historical result is close to the patched original-capacity result, but is
not a fresh same-source A/B. The remaining B32/B48 difference must not be
presented as eliminated or attributed to a hardware bottleneck without evidence.

## Validation and limitations

- Supported-stack GPU tests use Torch `2.13.0+cu132` and CuTe DSL 4.7.0.
  The full application uses its existing diagnostic override for Torch
  `2.14.1+cu132`; that is not certification of an additional supported release.
- Final focused GPU suite: **147 passed, 1 skipped, 73 subtests passed**. The skip
  requires two GPUs. Existing stream-lifetime warnings remain visible in the log.
- New regression coverage includes all eight dispatch decisions, adjacent
  aligned controls, N=64, fully ragged `17 × 29 × 33`, forced materialized mode,
  explicit-invalid-config rejection, input/weight mutation, first-hit tuning,
  fullgraph forward/backward, graph replay and stable live allocation.
- Independent FP32 GEMMs over the original BF16 operands validate forward, dX
  and dW. The gate is normalized L2 below 0.08 plus finiteness. Measured forward
  errors in the switching-shape matrix are 0.03719–0.03738; recorded fused and
  native outputs have identical reported errors. Absolute and RMS errors are
  retained in JSON. Neither E4M3/E8M0 policy nor FP32 dW accumulation changes.
- Preparation first-hit testing verifies both fused and materialized trials are
  admitted and subsequent calls do not append new observations. Explicit native
  cached configurations are honored. General `autotune='cache'` does not newly
  consume the experimental preparation journal: the application harness still
  explicitly reapplies its recorded native preparation winners.
- Nsight Systems/Compute were not installed on this worker. Event timings and
  PyTorch/CUPTI traces support the results; no hardware-counter, occupancy,
  saturation or L2-hit-rate claim is made.

## Artifacts and reproduction

Local raw evidence is under
`autotune_reports/mxfp8_row_alignment_20261103/` (the existing Storage 4 symlink).
Remote evidence is under
`/home/achyad/rtx-row-alignment-20261103/evidence/` on `work_5090`.
Keep the JSON, logs, source hashes, traces and source-matched tuning directory
together. They are intentionally not committed.

Key files: `baseline.json`, `compiled-forward-matrix.json`,
`backward-probe.json`, `b48-baseline.json`, `b48-candidate.json`,
`b32-candidate.json`, both B48 trace summaries, `environment.json`,
`final-gpu-regressions.log`, and `g2-fresh-tuning/`. The portable forward CLI
and compiled dX/dW/complete-backward CLI were also executed successfully;
see `portable-cli-smoke.json` and `compiled-backward-cli.json`.

Final supported-stack GPU regression command (on the single-GPU worker):

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_mxfp8_row_alignment.py tests/test_bwd_structural_coverage.py \
  tests/test_mxfp8_bwd.py tests/test_mxfp8_quant.py \
  tests/test_preparation_runtime_cuda.py tests/test_managed_execution.py \
  tests/test_scratch_cleanup.py tests/test_workspace_mlp_interleaved.py \
  tests/test_mlp_kernel_replay.py tests/test_execution_resources.py --maxfail=1
```

Portable isolation without application dependencies or historical caches:

```bash
python benchmarks/qualify_mxfp8_row_alignment.py \
  --shape 76608,256,256 --shape 310080,256,256 \
  --variants auto,fused,row_major,native64 --compiled \
  --output autotune_reports/row_alignment_repro.json
```

The portable `fused` variant uses the explicit portable fused seed, not the
historical tuned schedules in the comparison table. Use `--historical-cache`
with `--variants auto,historical_fused` for the recorded-schedule comparison.
`--backward-selections` accepts the application's explicit schedule list for
isolated dX/dW/backward probes; it does not import those schedules as cache
qualification evidence. CUDA graph capture is a benchmark implementation detail,
not an additional caller obligation for `MXFP8Linear`.
