# BalancedMoE short weight-gradient reductions

Small expert pools make weight gradients unusually wide and shallow. For
`T=128,E=32,K=4`, there are just 16 real slots/expert on average.
Independent block-32 quantization requires 32 physical slots, but the
original prepared GEMM scheduled a K128 macro tile. That is 8× modeled
reduction work relative to 16 real rows. K32 lowers it to 2× without
changing route policy, data, block precision, or public types.

The K128 floor was in the RTX prepared GEMM's staging and scale-layout
assumptions, not the ISA. SM120 supports
`mma.sync...m16n8k32` for E4M3 operands, UE8M0 block scales and FP32
accumulation. See [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
and the native warp op used by the [short-dW kernel](../rtx/kernels/mxfp8_moe_short_dw.py).
The generated PTX shows **8 / 16 / 32** static K32 MMA instruction sites
for the K32/K64/K128 cooperative stages at the same 64×64 output geometry.
This counts instruction sites, not an occupancy or hardware throughput
measurement. The explicitly staged FP8 operand bytes fall from 16 KiB
at K128 to 4/8 KiB at K32/K64, excluding scales and allocator alignment.

## Execution and selection

`MXFP8PreparedMoEConfig.with_short_dw("gate_dw", 32)` and the analogous
`down_dw` option accept K32 or K64. Either stage can be changed alone.
The kernel uses four ordinary MMA warps for predicated cp.async loads;
the scale layout broadcasts one independent byte over each 32-token block.
All accumulators are FP32 before the one public BF16 cast. There is no new
workspace, eager arithmetic inside the custom op, or register cap tuning.

This implementation admits unsplit prepared dW only, row-major scales,
one to four physical blocks, no active-prefix or output persistence, and a
direct BF16 output. The generic prepared GEMM and runtime defaults are
unchanged. The existing autotuner proposes both stages together and
independently, protects a legal short-dW trial early at tiny shapes, and
can rebase the short recipe onto the incumbent. It also explores the
regular path. The first short seed minimizes staging rounds without
adding an empty K32 instruction where possible; it is not a selected SKU
winner. Adapter features record nominal padded K, tile-work amplification,
grid CTAs and single-residency CTA waves for prior/predictor work. Those
are schedule proxies, not a claim about achieved occupancy.

## HOME RTX 5090 GPU 0 timing and qualification

Finite, compiled train workloads use identical inputs, hard routes, upstream
gradients, router/shared schedules, and expert schedules except the named
weight-gradient transport/geometry. Torch 2.13.0+cu132, CUDA Graph ABBA
visits, seven calibrated event samples per visit; compilation and capture
are excluded. Each row is a fresh process. Units are microseconds,
legacy → short dW. Phases are independent timings, not additive.

| Tokens / D / H | Candidate | Gate/up dW | Down dW | Routed expert step | Whole layer |
| --- | --- | ---: | ---: | ---: | ---: |
| 128 / 384 / 128 | K32 | 20.726 → 16.492 | 12.290 → 10.245 | 90.752 → 86.293 | 497.164 → 493.364 |
| 512 / 384 / 128 | K64 | 20.173 → 18.453 | 12.287 → 10.287 | 96.834 → 93.962 | 478.448 → 475.531 |
| 128 / 512 / 256 | K32 | 43.038 → 37.541 | 24.531 → 20.490 | 148.772 → 140.766 | 587.097 → 584.778 |
| 512 / 768 / 256 | K32 | 64.141 → 62.330 | 34.023 → 32.665 | 253.860 → 249.281 | 820.523 → 813.000 |
| 128 / 384 / 128 | Cooperative K128 control → K32 | 18.477 → 16.585 | 10.311 → 10.241 | 87.200 → 85.997 | 496.973 → 492.301 |

The four legacy-schedule comparisons improve the expert step by
**1.8–5.4%**; complete-layer improvements are **0.4–0.9%** and should not be
generalized as a material win for every routing/shape policy. Inter-visit
drift reached 1.37% in an individual phase (max relative sample MAD 0.73%).
At the same cooperative transport, output tile and settings, smaller K
still lowers gate dW by 10.2%; down dW's 0.7% change is inside its drift.
The wider path is not dominated by padded arithmetic alone. Retain both
transport families and measure the complete expert objective.

All five paired runs have bitwise-equal dW components, expert and whole
outputs/all gradients, and zero measured postwarm live-allocation growth.
The small D=384 case also matches the independent quantized FP64 oracle:
worst gradient NRMS 0.000605, identical for both candidates. Six CUDA/CPU
qualification tests passed on the device snapshot; GPU cases include
poison overwrite, expert isolation, ragged output dimensions, mixed
exponent scales, K96/K64 partial staging, all adjoints, fullgraph and
replay. The final focused CPU suite passed 163 tests (32 GPU skips).

RTX 5060 laptop: the approved four-file overlay onto an existing isolated
snapshot passed **three standalone poison/oracle tests** in 7.42 s, with
Torch 2.13.0+cu132. This was component correctness only, not an integrated
source-matched replay or throughput result. GPU preflight showed 43%
utilization from non-CUDA activity; unrelated display applications were
left alone and no laptop performance was measured.
[Overlay and JUnit](../autotune_reports/moe_next_priorities_20261118/short_dw_5060/).
The checked overlay SHA-256 is
`1656e501f886395c8eadc05c6856484c3c076559ec2906ddfc9d5f8acbe9cbe0`;
the resulting sorted Python-file checksum manifest hashes to
`89de40f3acb031a1438bf4581df1b966b9604a527b6e75f2cf9e0490855f6da9`.

## Reproduction

[Raw results, legacy settings, exact source archives, PTX/CUBINs and JUnit](../autotune_reports/moe_next_priorities_20261118/short_dw_home0/)
include timing runtime digest
`f5e1e6dd94e21905001a592788048274319743527d06cac3524146686ef54afb`.
Subsequent changes to derived features and protected proposal ordering do
not change the measured kernels. The progressive synthetic resume test
keeps exact 8→16 proposal checks; it now fixes synthetic evaluator and
proposal cost because otherwise tied dummy rewards can diverge at the
16th choice due to real CPU wall-time jitter.

```bash
PYTHONPATH=. python -m pytest -q -x tests/test_moe_short_dw.py
PYTHONPATH=. python benchmarks/benchmark_moe_joint_recompute.py \
  --baseline-result /path/to/frozen-legacy.json --short-dw 32 --oracle \
  --output /path/to/paired.json
```

Use `--tokens 512 --short-dw 64` for the second row, the corresponding
`--d-model`/ `--hidden` overrides for the wider profiles, and
`--cooperative128-control` for the same-transport control. No default
precision or performance policy was changed.
