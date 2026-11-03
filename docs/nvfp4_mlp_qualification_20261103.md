# NVFP4 SiTU MLP qualification — 2026-11-03

## Scope

Experimental `rtx.NVFP4MLP`, `rtx.nvfp4_mlp` and `rtx.NVFP4MLPConfig`.
Both forward GEMMs use NVFP4; SiTU and its derivatives retain FP32 gate/up
accumulators. The default H-wide connection is packed E4M3/E8M0 MXFP8.
Backward recomputes the original NVFP4 projection and hidden policy and uses
MXFP8 GEMMs with FP32 accumulation. Only X and the BF16 weights are saved.

This qualification used `work_5090`, RTX 5090, Torch `2.13.0+cu132`, CuTe DSL
4.7.0. Source was staged from `7b4424a` with only this feature's changes;
unrelated local MS-EDEN and decoder experiments were not copied. Runtime and
source identities are recorded in each benchmark JSON. No other machine's
performance or long-run convergence is established by this report.

## Correctness and integration

- NVFP4 MLP plus existing MXFP8 MLP/GLU regression run: **70 passed, 282
  subtests passed**. Default-mode fullgraph compilation, changing weights and
  inputs, registered-op checks, and both first-hit replay settings passed.
- Earlier NVFP4 MLP/native-linear/scratch-cleanup matrix: **91 passed, one
  skipped, 48 subtests passed**. Later frontend changes were rerun in the
  regression matrix above; these are overlapping suites, not unique totals.
- CPU contract/structural/managed-execution run: **26 passed, 25 CUDA cases
  skipped, two subtests passed** with CUDA masked.
- Independent quantization-aware FP32 oracle checks cover all three gradients,
  aligned and half-aligned widths, ragged dimensions, and the singleton
  `1×1×1×1` case. BF16 and packed hidden policies are independently exercised.
- All packed-input preparation ownerships (warp/CTA/cached CTA), BF16 exchange
  epilogue, optional backward overlap, partial gradient masks, zero inputs, and
  amax/Four-Over-Six/MSE5/LSQ1 block policies have coverage.
- Two outstanding first-hit forwards support reverse-order backward without
  retargeting their plans; gradients match the selected explicit policy.
- The dedicated live-MLP-graph check passes after clearing runtime scratch
  caches, changing X, and replaying forward plus all gradients. Automatic graph
  ownership retains the required buffers; callers allocate no scratch manually.
- Final runtime-contract/scratch-cleanup checks: **seven passed, two subtests
  passed**, on the final runtime integration.
- A short ten-step compiled AdamW lifecycle decreases its toy loss and has
  stable live PyTorch allocation in the final four iterations. This is a
  lifecycle/leak check, **not** training convergence or total-VRAM evidence.

The singleton test exposed a collapsed logical GLU axis; preserving the dynamic
descriptor extent fixed the illegal access. A late metadata LRU wrapper also
exposed a Dynamo tracing regression; separating the constant-result wrapper
from the cached implementation fixed it, with a CPU Dynamo regression test.

## Matched 8-trial timing

Default-mode `torch.compile(fullgraph=True, dynamic=False)` on both sides.
The BF16 baseline uses **reentrant checkpoint recomputation**, not a saved-hidden
MLP. Its GEMMs use max autotuning and epilogue fusion. Both implementations are
timed by outer CUDA graph replay; these are GPU execution timings, not CPU wall
latency or training throughput. Compilation and tuning are excluded. Seven
samples of ten replays follow warmup. Inputs have standard deviation 1 and
weights use the module's 0.02 truncated-normal initialization.

Shapes are `M × K → H → O`; H is post-gating width. Policy: caps 4/25, current
8-row X/W regions on both projections, power-of-two FP32 outer scaling, amax
FP4 block policy, packed MXFP8 hidden. Numerical policy is fixed during tuning.

| Shape | NVFP4 forward | BF16 forward | NVFP4 forward+backward | BF16 forward+backward | Training speedup |
|---|---:|---:|---:|---:|---:|
| `512×768→1536→768` | 75.1 µs | 30.7 µs | 177.8 µs | 98.5 µs | 0.55× |
| `8192×768→1536→768` | 251.9 µs | 280.8 µs | 825.9 µs | 1098.1 µs | 1.33× |
| `8128×768→1536→768` | 260.1 µs | 281.7 µs | 1466.7 µs | 1080.8 µs | 0.74× |

Each shape used eight measured trials and about 11–13 seconds of tuning. These
are modest-budget winners, not exhaustive optima. The small case loses to BF16;
the half-aligned case is supported but is not a performance win.

Before the backward-family fix, the half-aligned case took **8.32 ms**. It
defaulted to duplicated fused backward preparation and the opening candidates
changed only forward schedules. Complete paired backward families plus a legal
materialized seed reduced this to **1.47 ms**, about **5.7×** faster. No native
alignment validator was relaxed, and no padded BF16 operands were introduced.
Aligned shapes remain the main optimization target.

## Modest 32-trial refinement

A second, two-shape run used the same precision recipe, measurement procedure,
and matched per-run BF16 inputs. Tuning took **38 seconds aligned** and
**80 seconds half-aligned**. This is not an exhaustive search or a repeated-seed
statistical study. Because shape order differs, the 8- and 32-trial studies do
not use bit-identical tensors; each speedup uses its own matched BF16 baseline.

| Shape | NVFP4 forward | BF16 forward | NVFP4 forward+backward | BF16 forward+backward | Training speedup |
|---|---:|---:|---:|---:|---:|
| `8192×768→1536→768` | 233.6 µs | 282.9 µs | 787.1 µs | 1091.9 µs | 1.39× |
| `8128×768→1536→768` | 205.0 µs | 283.6 µs | 1408.8 µs | 1080.4 µs | 0.77× |

Half-aligned forward is competitive, but its backward remains a limitation.
Do not interpret correctness support or the large cliff reduction as a claim
that half-aligned training matches aligned training or beats BF16.

## Numerical interpretation

On these three standard-distribution benchmark shapes, the maximum relative L2
implementation discrepancy against the **same quantization policy** is 0.122%.
This is not the quantization error against BF16. Against the matched BF16
checkpoint baseline, relative L2 is approximately **23.6% output, 15.2–15.3%
dX/dGateUp, and 19.7–19.8% dDown**. These tensor-level measurements do not
establish model convergence or a quality advantage over another FP4 recipe.

## Limits and artifacts

The retained forward has four CUDA launches (two preparations, two GEMMs), not
the strict two-launch on-chip MXFP8 MLP family. No tensor-delayed, tensor-current,
block-only, FP4-backward, bias, dropout, or higher-order MLP mode is qualified.
Torch 2.12.1, other GPUs, broad shape performance, and transformer convergence
remain unqualified for this feature.

Raw evidence is retained under
`autotune_reports/nvfp4_situ_mlp_20261103/` on the existing Storage 4 symlink:
`benchmark.json` preserves the initial half-aligned failure;
`benchmark_v2.json` records the corrected 8-trial run;
`benchmark_effort32.json` records the 32-trial refinement; `tuning/` contains durable
per-context residual observations; qualification logs retain warnings and
completion counts. Reproduce with:

```bash
python benchmarks/benchmark_nvfp4_mlp.py \
  --trials 8 --seconds 120 --output autotune_reports/situ/benchmark.json
```

See [the API and numerical contract](nvfp4_mlp.md) for usage and controls.
