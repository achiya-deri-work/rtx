# RTX kernel autotuning

The MXFP8 frontend has three selection modes:

- `off`: always use the built-in baseline unless an explicit config is passed.
- `cache`: use a compatible saved winner, otherwise use the baseline. This is
  the default.
- `coordinate`: run/resume coordinate descent on first use and then execute the
  winner.

An explicit `MXFP8FwdConfig` or `MXFP8PrequantConfig` always wins over
autotuning.

## Executable search space

Kernel revision 10 has real implementations for:

- scalar, 128-bit cp.async, or TMA BF16 input loading;
- cooperative, TMA warp-specialized, or three-role scheduling. Three-role
  kernels use disjoint TMA producer, quantizer, and MMA-consumer warps;
- BF16 transport tile K of 32/64/128/256, independently of the MXFP8 SMEM
  macro tile, plus independent BF16 none/32B/64B/128B SMEM swizzling;
- 1/2/4/8-way subwarp-vectorized MXFP8 quantization;
- one to four circular MXFP8 operand/scale stages, subject to SMEM capacity;
- independent none/32B/64B/128B shared-memory swizzles for A and B;
- independent A/B `ldmatrix.m8n8.b16` x1/x2/x4 issue widths, plus auto or
  explicit byte-wide E8M0 scale-fragment S2R copies;
- CTA M/N/K tile shape, including a padded-hardware-layout 64-row CTA and the
  required tile/warp-layout coupling;
- MMA atom layout, warp count and thread count for legal SM120 fragments;
- K-loop unrolling;
- PTXAS maximum register count;
- M-major or N-major CTA rastering;
- CTA swizzle groups 1, 2, 4 and 8;
- direct stores or staged SMEM/TMA stores with x1/x2/x4 `stmatrix` width and a
  mandatory final bulk-group drain;
- one-to-four-wave persistent scalar CTA grids, with X- or weight-locality
  work ordering.

Each admitted coordinate changes generated code, launch geometry, or compiler
resource policy and is compiled, checked and timed independently. The TMA path
uses `PipelineTmaAsync` transaction barriers and circular BF16 storage. In the
warp-specialized variant, consumers release the BF16 buffer as soon as
quantization finishes, allowing the DMA warp to fetch the next K tile while
consumers execute MMA. K=32 permits up to four BF16 stages for a 128x128x128
MXFP8 macro tile, exposing small-transport/deep-pipeline schedules without
duplicating the full MMA tile in SMEM. The register M/N mapping remains
controlled independently by the MMA atom-layout axes. The three-role path adds
a second circular mbarrier pipeline so next-tile quantization can overlap the
current MMA. Quantization uses both halves of `cvt...e4m3x2` for
`quant_vec >= 2` and multiplies by the exact power-of-two E8M0 reciprocal.
SM120's unsupported FP32 `redux.sync.max.abs` is represented by an exact
unsigned-bit alternative: clear the FP32 sign bit, issue
`redux.sync.max.u32`, then bitcast the maximum back to FP32. This remains an
autotuned option because its latency can exceed the two-shuffle reduction used
by four-lane `quant_vec=8` subgroups.

The cp.async path requires complete M/N/K tiles. TMA output stores require the
overall M/N problem to be divisible by the CTA output tile; ragged problems use
direct predicated stores. Persistent execution currently targets the
scalar/cooperative/direct path. `reuse=x/weight` changes persistent work
ordering to improve L2 locality; it does not retain a full-K operand in SMEM.
Ping-pong scheduling remains explicitly rejected.

## Command line

Tune one actual flattened linear shape for up to thirty minutes:

```bash
.venv/bin/python -m rtx.autotune \
  --m 8192 --n 4096 --k 4096 \
  --seconds 1800 --passes 4 --restarts 8 \
  --correctness baseline
```

Use `--correctness torch` for an independent, more expensive MXFP8 reference.
Use `--force` to recompile and remeasure trials already present in the JSON
database.

The default database root is `$XDG_CACHE_HOME/rtx/autotune`, falling back to
`~/.cache/rtx/autotune`. Override it with `RTX_AUTOTUNE_CACHE_DIR` or the CLI
`--cache-dir` option.

## Python API

```python
import torch
from rtx import CoordinateDescentPolicy, tune_mxfp8_fwd

x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

result = tune_mxfp8_fwd(
    x,
    w,
    policy=CoordinateDescentPolicy(
        time_budget_s=1800,
        max_passes=4,
        restarts=8,
        samples=9,
        calls_per_sample=5,
    ),
)
print(result.config, result.median_ms, result.database_path)
```

First-use tuning through the module:

```python
from rtx import MXFP8Linear

layer = MXFP8Linear(4096, 4096, device="cuda", autotune="coordinate")
y = layer(x)  # tunes or resumes this exact [M,N,K] problem
```

Environment-controlled first-use tuning is also supported:

```bash
export RTX_MXFP8_AUTOTUNE=coordinate
export RTX_MXFP8_AUTOTUNE_SECONDS=1800
export RTX_MXFP8_AUTOTUNE_PASSES=4
export RTX_MXFP8_AUTOTUNE_RESTARTS=2
export RTX_MXFP8_AUTOTUNE_WARMUP=10
export RTX_MXFP8_AUTOTUNE_SAMPLES=11
export RTX_MXFP8_AUTOTUNE_CALLS_PER_SAMPLE=50
```

Set `RTX_MXFP8_AUTOTUNE=cache` for inference/training jobs that must never
benchmark unexpectedly.

## Persistence and resumption

Each JSON document is specific to:

- GPU properties and UUID;
- compute capability;
- Torch, CUDA, CUDA Python and CuTe DSL versions;
- Python/platform version;
- kernel revision;
- exact `[M,N,K]` problem;
- search-space digest.

Every candidate is written atomically under an advisory file lock immediately
after it is processed. Trial statuses include legal rejection, unimplemented
rejection, compile failure, correctness failure, runtime failure and successful
timings. Interrupted jobs reuse these records on the next invocation.

`restarts > 1` adds deterministic random legal seeds. Each seed is assembled
incrementally so coupled tile/warp and shared-memory constraints remain valid,
then ordinary coordinate descent runs to convergence from that basin. Trials
from every basin share the same persistent database and are reused after an
interruption.

Coordinate descent recomputes dependent launch values such as MMA warp count,
producer warps and total threads whenever atom layout or schedule changes. A
coordinate that is represented but not yet implemented is recorded as rejected;
it is never benchmarked as a no-op.

Two block coordinates prevent coupled regions from being hidden behind a slow
one-field intermediate. `smem_rmem_tile` moves the CTA SMEM shape and per-warp
RMEM M/N ownership together; its values are `(smem_m, smem_n, smem_k, rmem_m,
rmem_n)`. `bf16_pipeline` moves `(load_engine, schedule, bf16_tile_k,
bf16_swizzle, bf16_stages)` together and explicitly includes TMA and cp.async
K=32 schedules with one through four stages, including three-role TMA
schedules. The primitive fields remain in the
coordinate order for refinement after a block move. Thus a block coordinate is
a direct jump to a separately generated kernel, not a meta-parameter or no-op.

The database keeps a provisional raw minimum while a session is running, then
persists the configuration accepted by coordinate descent when the session
finishes. This ensures the configured minimum-improvement threshold is also
respected by later cache-only module calls.

Because timings describe the current system load, do not promote a database
created while the GPU is busy. Use `--force` during a quiet tuning run to replace
those measurements.

## Native-scale prequant backend

The production materialize-once backend has a joint end-to-end tuner. It times
X quantization, W quantization, and GEMM in the same event interval; component
minima are never added together. Run the complete search for thirty minutes:

```bash
.venv/bin/python -m rtx.prequant_autotune \
  --m 512 --n 1536 --k 1536 \
  --seconds 1800 --passes 4 --restarts 2 \
  --warmup 10 --samples 11 --calls-per-sample 50 \
  --log-file autotune_logs/mxfp8_prequant_512x1536x1536.log
```

The joint search has real coordinates for:

- row-major M64/M128/M256 scale staging through consumer or producer warps;
- M64 hybrid and M128 tensor-core-native physical scales moved by TMA;
- one combined quantizer launch or independent X/W launches and schedules;
- independent X/W vector/load widths, arithmetic, amax/reduction path,
  warps, persistent waves, register caps, and native scale-store width;
- CTA/RMEM geometry, operand stages, independent A/B LDSM issue width and
  SMEM swizzle, and independent scale-fragment S2R width;
- scale load placement, vector width, L1 eviction/cache modifier, and L2
  prefetch size;
- producer/consumer/PTXAS register budgets, direct or TMA output, store width,
  raster direction, CTA grouping, and CUDA's global L2 fetch granularity.

Coupled transitions are tested as one candidate when their intermediate states
would be illegal or misleading. In particular, a W-only schedule can cross
from dual to separate quantization in the same trial, native physical layouts
move quantizer/GEMM scale layouts and transport together, and large CTA tiles
move to their required low-SMEM stage/epilogue at admission. Values which fail
shape, tensor-core layout, SMEM, or launch legality are recorded as rejections;
they are not benchmarked as no-ops.

The L2 fetch limit is process-global. Each tuning trial restores the previous
value; if a non-default value wins, the frontend applies it when building that
shape's runner. Use separate processes when comparing this coordinate against
unrelated concurrent workloads.

`mxfp8_linear(..., autotune="coordinate")` and `MXFP8Linear(...,
autotune="coordinate")` use this tuner for the prequant backend. The request is
resolved in the runtime launcher, so first-use tuning also works through
`torch.compile` tracing. `autotune="cache"` loads a compatible winner without
benchmarking and otherwise uses the built-in native-scale schedule.

Every trial and the provisional winner are written atomically under a file lock.
The key includes the exact shape, GPU/software fingerprint, kernel revision, and
the digest of all joint coordinates. Interrupted runs reuse successful,
rejected, and failed trials.

The older component tuners remain useful for diagnosis, but their independent
winners are not production selection. Tune native-layout E4M3/E8M0
quantization alone:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/tune_mxfp8_native_quant.py \
  --m 512 --n 1536 --k 1536 --passes 2
```

Tune the prequantized native-scale GEMM independently:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/tune_mxfp8_native_gemm.py \
  --m 512 --n 1536 --k 1536 --passes 2
```

The quantizer sweep covers vector/load width, FP32 versus BF16x2 arithmetic,
amax implementation, shuffle versus hardware redux, warp/wave launch shape,
register cap, and scalar versus packed physical-scale stores. The GEMM sweep
covers pipeline depth and MMA warp geometry, independent A/B LDSM width,
independent SMEM swizzles, scale S2R width, producer/consumer register budgets,
TMA versus direct epilogue, store width, raster direction, and CTA grouping.

Use the component benchmark to separate quantization, GEMM, and launch effects:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_mxfp8_prequant.py \
  --scale-layout mma128 --component e2e
```

Use `benchmarks/benchmark_mxfp8_frontend.py` to measure the registered
dynamic-weight op through `torch.compile(fullgraph=True, dynamic=False)`. Kernel
timings and standalone compiled-op timings are deliberately reported separately:
the latter also includes graph-output allocation and host submission gaps.
