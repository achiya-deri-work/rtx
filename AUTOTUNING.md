# RTX kernel autotuning

The legacy name `prequant` in APIs, cache keys, and v1/v2 datasets means a
*materialized dynamic* pipeline: BF16 X and BF16 W are quantized every call,
then consumed by a separate GEMM. It must not be confused with the persistent
AOT-weight or fully packed inference states introduced by the public packed
operand API. Historical family names remain frozen so active datasets resume
without relabeling observations.

Persistent inference uses separate families:

- `mxfp8_weight_prequant_fwd` measures X quantization plus GEMM and treats W
  packing as untimed AOT work;
- `mxfp8_fully_prequant_fwd` measures GEMM only and treats both packing steps
  as untimed AOT work.

Their search spaces structurally remove inactive quantizer axes rather than
relying on rejection or hoping the learned model ignores no-op coordinates.

The MXFP8 frontend has three selection modes:

- `off`: always use the built-in baseline unless an explicit config is passed.
- `cache`: use a compatible saved winner, otherwise use the baseline. This is
  the default.
- `coordinate`: run/resume coordinate descent on first use and then execute the
  winner. Persistent inference states use the composable learned-global plus
  local-search engine under the same public mode name.

An explicit `MXFP8FwdConfig` or `MXFP8PrequantConfig` always wins over
autotuning.

Packed-state requests remain opaque through `torch.compile`; exact device,
shape, operand-state, and physical-layout cache selection occurs inside the
custom-op runtime on first execution rather than being frozen during tracing.

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
rtx-autotune-legacy \
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

## Hierarchical bandit campaigns

The composable tuner has two independent allocation levels. Inside one
device/shape/regime context, `AdaptiveBanditScheduler` treats random search,
gradient-boosted global search, and model-guided local search as arms. Its
reward is dimensionless and bounded: incumbent improvement and model
uncertainty are balanced against invalid candidates and actual evaluator wall
time. Discounting makes the policy adapt as the cost model improves. A fixed
warmup and one mandatory local pull per arm prevent transfer priors from
short-circuiting direct evidence.

Across contexts, anytime mode can use a second discounted contextual UCB. It
completes broad minimum coverage first, limits how many milestones any context
may lead the shallowest unfinished context, then allocates slices using recent
improvement per visit and similarity across family, shape, and cache regime.
This layer decides *where* to spend the next slice; the strategy bandit decides
*how* to search inside that slice.

Both layers are crash-resumable. Strategy state is replayed from
`observations.jsonl`, context state from `context_allocations.jsonl`, and every
selection stores all arm scores and sufficient statistics for offline replay.
The original sequential and breadth-first schedulers remain selectable control
policies.

## Interruption-safe prospective optimizer studies

`storage_mode: residual_context` gives every family/treatment/replicate/shape/
regime context its own append-only journals. Writes are flushed and fsync'd per
accepted observation. A malformed crash tail is ignored and terminated before
the next append. Consequently, corruption is bounded to one residual and does
not consume the first record written after resume.

`rotation_mode: balanced_categories` greedily orders contexts so useful
prefixes stay balanced across treatment, family, named shape category, cache
regime, and replicate. Breadth-first absolute milestones then deepen that same
balanced order. Treatment and replicate are included in context identity;
cross-context transfer is scoped inside a single treatment/replicate to prevent
prospective leakage.

The 5070 study exposes `random`, `random_local`, and `online_bandit` as explicit
portfolios rather than inferring optimizer behavior after collection. Summarize
copied residuals with `rtx-autotune summarize-tuners`; it reports success,
compiler waste, time-to-valid, and regret curves at fixed trial budgets.

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

## Rigorous multi-shape experiments

`rtx.prequant_experiments` builds persistent datasets rather than selecting a
first-use production configuration. It adds facilities which ordinary
coordinate descent deliberately does not pay for:

- a conditional legal catalogue with an interpretable one-factor structural
  basis followed by coverage-guided compound schedules;
- adaptive event batches targeting a fixed amount of GPU time;
- explicit `hot` and rotating-input regimes, where the rotation ring targets
  twice the device L2 size when memory permits;
- low-fidelity screening, high-fidelity confirmation, and AB/BA paired races;
- bootstrap confidence intervals and a practical-equivalence threshold;
- separate quantizer and materialized-GEMM diagnostics for confirmed
  candidates while end-to-end timing remains authoritative;
- NVIDIA clock, power, temperature, P-state, and utilization snapshots;
- derived CTA-wave, operand-reuse, tail, byte-size, and L2-fit features;
- append-only JSONL observations, deterministic hash sharding, atomic summary
  JSON, and flat CSV export.

Run the bounded three-shape pilot:

```bash
.venv/bin/python -m rtx.prequant_experiments \
  --manifest autotune_manifests/pilot_rigorous.json \
  --output-dir autotune_results/experiments \
  --export-csv autotune_results/experiments/pilot.csv
```

Probe each machine before distributing work:

```bash
.venv/bin/python -m rtx.prequant_experiments --probe
```

The probe records L2, SMEM, registers, SM count, memory, software fingerprint,
and telemetry availability. The native kernel campaign admits SM120/SM121
only; other compute capabilities are rejected before dataset collection.

Inspect catalogue size and shard allocation without requiring CUDA:

```bash
.venv/bin/python -m rtx.prequant_experiments \
  --manifest autotune_manifests/blackwell_cross_device_v1.json \
  --dry-run
```

The cross-device manifest is an initial common anchor corpus. Copy the same
repository revision and manifest to every machine. Device fingerprints place
each machine in a separate result directory. To split one device's catalogue
across processes or machines, give each copy a distinct `shard_index` in
`[0, shard_count)`; configuration IDs are assigned by a deterministic hash.

JSONL measurement records retain raw timing vectors, confidence summaries,
configuration dictionaries, protocol, device environment, telemetry and
features. Resume checks observation keys and never overwrites measurements.
Completed paired-race decisions are replayed when rebuilding the atomic shard
summary. Compilation and correctness failures are dataset rows rather than
reasons to terminate a campaign.

### Offline model and heuristic artifacts

Once compatible bundles have returned, train on the authoritative JSONL rather
than flattened CSV columns:

```bash
rtx-autotune pretrain copied_datasets/ laptop-results.zip \
  --campaign mxfp8_blackwell_cross_device_bandit_v1 \
  --output autotune_models/mxfp8_blackwell_bandit_v1
```

`--campaign` is repeatable for known-compatible collections. This is preferable
to omitting the filter, which may silently admit obsolete or interrupted runs.

Models are split by `(family, kernel_revision)`. Provenance, UUIDs, raw timing
vectors, and device-name categories are removed from features; physical device
limits and calibrated rooflines remain. Strategy/context balancing limits
adaptive-sampling dominance, while paired `parent_config_id` transitions feed
conditional rules. A rule becomes active only when its bootstrapped median
effect excludes zero with the requested support. This remains observational
evidence, so it changes proposal priority rather than legality.

Deployment is hierarchical. A portable model is gated by leave-one-device-out
catalog replay. If it fails that gate, an exact-SKU model may still be used when
it beats matched random replay in at least three of four held-out-context folds
on that SKU. Neither model is used on an unrecognized device unless the
cross-device gate passed. Feasibility and conditional-rule gates are independent
from latency-head selection.

After copying device result directories to one machine, merge and flatten all
shards with:

```bash
.venv/bin/python -m rtx.prequant_experiments \
  --merge-root autotune_results/experiments/mxfp8_blackwell_cross_device_v1 \
  --merged-jsonl autotune_results/experiments/cross_device.jsonl \
  --export-csv autotune_results/experiments/cross_device.csv \
  --analysis-json autotune_results/experiments/cross_device_analysis.json \
  --portfolio-tolerance 0.01
```

The merger deduplicates by device and observation key, retains the source
journal path, and preserves session records. CSV columns flatten configs,
protocols, device properties, telemetry and derived features; raw sample
vectors remain JSON-encoded in one column.

The analysis report defines one context as `(device, shape, cache regime)`,
uses confirmed measurements when available, and greedily finds the smallest
observed configuration portfolio that covers contexts within the requested
regret tolerance. This gives us an empirical selector baseline and a concrete
top-k target before fitting a learned ranker.

The cache regimes answer different questions. `hot` is the repeated-layer
microbenchmark limit. `rotate` cycles equal-valued tensors at distinct
addresses over a working set larger than L2, approximating newly produced
activations and recently written dynamic weights. It is intentionally not
called "cold": quantized scratch buffers and tensor-core execution still have
their normal within-launch locality.

## MXFP8 backward

The initial backward is an explicitly decomposed end-to-end pipeline. For a
forward problem `Y[M,N] = X[M,K] @ W[N,K].T`, it executes:

- `dX = dY[M,N] @ W.T[K,N].T`, where `W.T` is a metadata-only view;
- `dW = dY.T[N,M] @ X.T[K,M].T`, where both transposes are metadata-only views.

CuTe layouts carry every source orientation. A logical transpose keeps the
original row-major allocation and presents shape `[rows, K]`, stride
`(1, rows)` to the compiled kernel. Its physical `[32, tile_rows]` bytes are
loaded contiguously to SMEM once; a second CuTe layout views those same bytes
as logical `[tile_rows, 32]`. There is no GMEM transpose allocation, transpose
kernel, SMEM transpose copy, or BF16 orientation workspace.

Each oriented operand is dynamically quantized to E4M3 with its own E8M0
scales. Forward scales are not reused because the backward reduction axes are
different. The baseline dW GEMM performs the entire token reduction in FP32
accumulators and converts only its final result to BF16.

The executable dual quantizer also accepts mixed source orientations with
independent schedules. dX can therefore quantize row-major `dY` and logical
`W.T` in one persistent launch without forcing the row and transpose paths to
share vector width, arithmetic, tile, or register choices. dW similarly emits
both logical-transpose operands in one launch. Separate launches remain an
autotuned option because fusion can lose on some shapes.

`rtx.bwd_autotune` times the whole backward, including logical-layout
quantization, four quantized operands, dX, and dW. Its persistent device/shape database saves
successful measurements, named legality/implementation rejections, raw timing
samples, compilation time, correctness results, sessions, and the selected
winner. The search keeps dX and dW schedules independent and includes:

- logical-layout staging tiles and padding, vector widths, warps, persistence
  depth, and register caps;
- dual versus separate quantization, independent A/B vector arithmetic,
  reductions, launch shapes, scale stores, and register caps;
- native scale layout/transport, GEMM SMEM/RMEM geometry, stages, LDSM and S2R
  widths, swizzles, cache controls, register partitions, epilogues and raster;
- full FP32, split-workspace FP32, FP32 atomic and cluster reduction families;
- persistent/multi-output tile scheduling, operand reuse/locality, dX/dW order,
  and stream scheduling.

Families that are represented but do not generate distinct code yet are saved
as `implementation_rejected`; they are never benchmarked as no-op knobs. The
current executable baseline uses row-major and logical-transpose quantizers
plus the MXFP8 GEMM. Tune directly with:

```python
from rtx import CoordinateDescentPolicy, tune_mxfp8_backward

result = tune_mxfp8_backward(
    grad_output,
    x,
    weight,
    policy=CoordinateDescentPolicy(time_budget_s=1800),
)
```

Or let the public backward API load a cached result or tune on a cache miss:

```python
from rtx import mxfp8_linear_backward

dx, dw = mxfp8_linear_backward(
    grad_output,
    x,
    weight,
    autotune="coordinate",
)
```

The calibrated composable search is available as:

```bash
.venv/bin/python -m benchmarks.tune_composable_bwd \
  --store autotune_results/composable_bwd \
  --legacy-db path/to/legacy-backward.json \
  --trials 400 --model-trials 220 --model-pool 8192
```

`BwdBenchmarkHarness` uses auto-sized multi-call batches, an FP32 normalized-L2
accuracy gate, component diagnostics, and AB/BA paired races. Historical
backward databases with nested outcome records are imported as transfer-only
cost-model data.

## Composable autotuning layer

`rtx.autotune` is now an importable package shared by fused forward,
prequantized forward, and backward. The original forward coordinate tuner is
available through a compatibility module, while new searches use five
independent interfaces:

- kernel adapters own configuration identity, legality, features, mutation,
  sampling, and evaluation;
- strategies propose candidates, including random walks, a dependency-free
  gradient-boosted cost model, and beam coordinate-local search;
- schedulers compose strategies sequentially or allocate trials with UCB1;
- the orchestrator owns budgets, deduplication, reward accounting, and session
  lifecycle;
- the append-only JSONL store durably records observations and orchestration
  decisions after every trial.

The default hybrid recipe runs learned global search and then coordinate-local
search. It can instead expose random, learned, and local strategies as bandit
arms:

```python
from rtx.autotune import (
    HybridTuningPolicy,
    JsonlTuningStore,
    make_hybrid_autotuner,
    make_mxfp8_bwd_adapter,
)

adapter = make_mxfp8_bwd_adapter(problem, evaluator, device=fingerprint)
tuner = make_hybrid_autotuner(
    adapter,
    JsonlTuningStore("autotune_results/unified"),
    HybridTuningPolicy(orchestration="bandit", max_trials=512),
    progress=print,
)
result = tuner.tune()
```

Cross-device observations with the same family/revision are training data for
the cost model. Incumbents and duplicate suppression remain device/workload/
regime-local, so transferred predictions never replace actual measurement on
the target. See `rtx/autotune/README.md` for the schema and extension contract.

### Cross-project portability boundary

Version 0.7 adds a kernel-generator-neutral layer above the existing adapters:

- declarative conditional parameter spaces with named hard constraints;
- staged static, compile, correctness, benchmark, and application evaluation;
- partial-fidelity observations and explicit promotion;
- serializable trial leases and worker responses;
- out-of-order `ask`/`tell` completion; and
- optimizer snapshots containing observations, pending requests, RNG, and arm
  statistics.

Existing RTX adapters can be wrapped with `AdapterKernelTask`; MoE, attention,
and unrelated projects can define `FunctionKernelTask` or implement the
`PortableKernelTask` protocol. Transport is intentionally not prescribed: the
same request/response dictionaries can travel through multiprocessing, an RPC
service, or a database queue.

This is the portability foundation, not the final optimizer portfolio. The
next framework milestones are a durable distributed coordinator, separate
failure-type models, a learned multi-fidelity promotion policy, TPE/SMAC-style
conditional-space search, and prospective regret-versus-wall-time evaluation
across RTX linear, bidirectional MoE, and custom attention reference tasks.

### New all-time 512×1536×1536 forward result

The composable tuner imported the 1,646-trial legacy database, trained the GBT
on those observations, performed legal global sampling, and then ranked full
coordinate neighborhoods around its measured beam. It found a previously
untested compound point that changes the best-ever schedule's GEMM producer
register allocation from 64 to 40 while retaining the 232-register consumer,
255-register compilation cap, `raster="m"`, and `grid_swizzle=1`.

- previous best search median: 16.678080 µs;
- new search median: 16.663946 µs;
- first interleaved race, 51 rounds: 0.2622% median speedup, 95% bootstrap CI
  0.2582%–0.2705%;
- independent race, 75 rounds: 0.2738% median speedup, 95% bootstrap CI
  0.2694%–0.2793%.

Both races selected the challenger with a zero practical threshold. The saved
winner and raw verification records are under
`autotune_results/composable_fwd_best/`.

### New 512×1536×1536 backward result

Mixed-layout dual quantization plus two confirmation-aware composable search
waves reduced the complete backward from the previous saved 42.059 µs search
winner to the new configuration `08bd05f84afe8f34e7ba`. A fresh 31-sample
measurement recorded 40.158 µs versus 42.145 µs for the legacy winner. The
authoritative 75-round paired races found:

- Wave 2 versus Wave 1: 0.1174% median speedup, 95% bootstrap CI
  0.1103%–0.1220%;
- Wave 2 versus the previous saved winner: 4.9987% median speedup, 95% CI
  4.9957%–5.0031%.

The two outputs have identical measured normalized-L2 error. The complete
configuration, tuning journals, and raw verification samples are under
`autotune_results/composable_bwd_wave1/`.

Load a saved winner directly with `rtx.load_mxfp8_bwd_config(path)` and pass it
as the `config=` argument to `mxfp8_linear_backward`.
