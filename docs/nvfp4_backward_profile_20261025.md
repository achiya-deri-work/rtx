# Why experimental NVFP4 backward is slower than MXFP8

## Conclusion

The main penalty is **operand preparation, not FP4 matrix multiplication**.
MS-EDEN rotates, normalizes, quantizes, computes correction statistics, reduces
range statistics, stochastically rounds scales, and constructs native layouts.
MXFP8's paired quantizer does substantially less work. FP4 GEMMs are faster in
the measured cases, but their savings usually do not repay preparation cost.

This concerns the **experimental full-FP4 backward**. Stable `NVFP4Linear`
continues to use MXFP8 backward. No production kernels or numerical policies
were changed for this profiling study.

## Scope and controls

RTX 5060 Laptop, 26 SMs, 8 GiB, Torch 2.13.0+cu132, Python 3.13.16.
Only `personal_laptop` was used; neither the 5070 nor local AFRL GPUs was used.

- BF16 sources and outputs; original BF16 backward operands, not saved FP4
  forward round trips. Same exact JIT-region NVFP4 forward in public comparisons.
- FP4: 128 producer threads, 256 producer CTAs, coalesced-wide loads,
  register-cached signs, explicit caller RNG, 128-thread/224-CTA finalizers,
  inline partial-amax reduction. dX row-major and dW reduction-major traversal.
- MXFP8: actual default paired quantizer and dual-stream GEMMs, matching the
  training recipe. Neither direction uses split-K here. Full configurations
  are retained in every result, including GEMM scheduling.
- Compiled fullgraph/static public forward/autograd, manually captured backward
  graphs. Warmup, compilation, correctness checks, and allocation audits are
  outside timing. These are backward graph latencies, not optimizer/model wall
  times, and not a fresh exhaustive autotuning comparison.
- Eight shapes, 40 initial public/phase workers, eight narrow Systems captures,
  and randomized AB/BA complete-backward confirmation on all eight shapes.
- Public current-input/current-weight/upstream-gradient and fresh-RNG guards
  passed. Gaussian FP32-reference NRMS gates are 6% for MXFP8 and 25% for FP4;
  these broad algorithm-error gates are not a convergence qualification.
  Isolated FP4 packed-GEMM reference checks passed the separate 0.5% NRMS gate.
  Across paired-run input-change guards, observed gradient NRMS was
  3.75–3.78% for MXFP8 and 13.66–13.77% for MS-EDEN FP4.

Raw data: `autotune_reports/backward_subparts_20261025_5060/`.
Persistent remote snapshot: `/home/achiya/rtx-bwd-profile-20261025.dTK2a0`.

## Complete backward: paired confirmation

Microseconds per backward, both dX and dW, caller RNG included. Ten randomized
AB/BA rounds; intervals bootstrap whole rounds. The percentage column uses the
paired geometric ratio, not the ratio of the two displayed medians. Intervals
describe this session, not cross-device or cross-day certainty.

| M × N × K | MXFP8 µs | FP4 µs | FP4 time penalty | 95% paired interval |
|---|---:|---:|---:|---:|
| 512 × 512 × 512 | 21.0 | 59.6 | +184.4% | +181.4% to +187.7% |
| 4096 × 512 × 512 | 96.7 | 167.4 | +74.2% | +72.0% to +76.4% |
| 4096 × 1536 × 512 | 258.1 | 360.8 | +39.5% | +39.0% to +40.1% |
| 4096 × 2048 × 512 | 387.4 | 461.3 | +19.5% | +19.0% to +20.0% |
| 4096 × 512 × 1024 | 171.9 | 245.8 | +44.2% | +42.6% to +46.8% |
| 32768 × 512 × 512 | 913.6 | 1182.8 | +29.3% | +29.0% to +29.6% |
| 4096 × 2048 × 2048 | 999.6 | 979.1 | −2.0% | −2.1% to −1.9% |
| 8192 × 1536 × 1536 | 1160.9 | 1213.0 | +3.4% | +2.2% to +4.6% |

Several initial independent-process samples were nonstationary. They remain
recorded with quality flags; the paired table supersedes their speed ratios.
In particular, the initial approximately 9% wide-shape advantage shrank to 2%.

## What consumes FP4 GPU time?

Systems attribution below is percentage of **summed kernel duration**, not
wall-clock latency. Profiling perturbs execution; do not subtract this table
from CUDA-event timings or add isolated hot-phase medians to predict latency.

| Shape | Rotation + quantization + correction | Scale finalization | GEMMs | RNG/state/plumbing | Factor product |
|---|---:|---:|---:|---:|---:|
| 512 × 512 × 512 | 30.5% | 39.0% | 15.4% | 13.5% | 1.6% |
| 4096 × 2048 × 512 | 60.4% | 11.0% | 25.0% | 3.3% | 0.3% |
| 32768 × 512 × 512 | 63.2% | 8.2% | 26.4% | 2.1% | 0.1% |
| 4096 × 2048 × 2048 | 47.3% | 6.7% | 43.8% | 2.0% | 0.1% |

At `4096×2048×512`, summed per-backward GPU durations are approximately
254 µs producers, 46 µs finalizers, 105 µs GEMMs, and 14 µs RNG/state/plumbing.
MXFP8's corresponding separate capture spends approximately 164 µs in its
paired quantizer and 193 µs in GEMMs. These are diagnostic attribution numbers,
not a matched counterfactual or exact decomposition of the paired event gap.

### dX versus dW

The retained phase JSONs separate both producers, both finalizers, factor
product, RNG, GEMM, preparation as a whole, and complete gradient.
For `4096×2048×512`, isolated hot-phase medians are approximately:

| Component | dX µs | dW µs |
|---|---:|---:|
| Combined preparation | 126.9 | 161.0 |
| GEMM | 54.9 | 57.2 |

The dW penalty is mainly preparation: both `dY.T` and `X.T` require the
transposed-source preparation path. dX uses row-oriented dY plus transposed W.
The latter is often much smaller than the activation tensor. These paths
stage contiguous BF16 data through SMEM; there is no full transposed GMEM copy.

## Why preparation costs so much

1. **More arithmetic per input value.** Hadamard-128 butterflies, block amax,
   E8M4 scale construction, FP32 normalization, FP4 conversion/decode, EDEN
   dot/norm correction, and correction-scale output are fused into a producer.
   Source/lowering inspection shows precise divisions and many shuffle
   operations absent from ordinary MXFP8 quantization. This explains the extra
   work; it does not, by itself, prove which instruction pipeline is limiting.
2. **Transposed producers have an additional staging dependency.** The current
   coalesced-wide path loads SMEM, synchronizes, processes two rows per warp,
   then synchronizes before reuse. There is no producer-stage double buffering.
   Three of the four operand preparations use this path.
3. **Finalization has duplicated reduction work.** With 256 producer CTAs and
   four warps/CTA, there are 1024 partial maxima. The inline finalizer has each
   warp scan those partials before converting its scales. For small operands,
   this fixed work outweighs the GEMM: four finalizers consume 39% of FP4 GPU
   kernel time. A separate staged reduction already exists as an optional
   schedule, but was not selected in this baseline.
4. **More graph nodes and dependency edges.** MXFP8 has three GPU kernels per
   backward: paired quantization and two GEMMs. FP4 has 21 observed kernels:
   four producers, four finalizers, two factor products, two GEMMs, and nine
   RNG/state/plumbing kernels. The FP4 compute stages are serial in this graph;
   MXFP8 also has a dual-stream scheduling option enabled in the baseline.

The two dY preparations cannot simply share one packed operand: dX and dW
reduce different axes, with different rotations and block scales.

## Why a longer batch does not solve it

Modeled prepared-element count is `E = 2MN + NK + MK`; two GEMMs perform
`4MNK` conventional FLOPs. Thus work per prepared element is `4MNK/E`.
At fixed N and K, this ratio approaches a finite limit as M grows. Widening
the layer increases compute relative to preparation much more effectively.
This is an amortization model, not a DRAM-counter measurement or a universal
dispatch heuristic. Tile count, cache behavior, and schedules also matter.

## Launch gaps and memory

Large captured FP4 workloads show approximately 95.4–98.8% kernel-union coverage
within the kernel activity span. Their slowdown is not mostly idle GPU time
waiting for Python. The terminal `cudaDeviceSynchronize` belongs to the
profiler boundary, not the production backward.

There are two ordinary launch calls per FP4 graph replay for Torch RNG-state
bookkeeping. The producers/GEMMs themselves are graph nodes, not a Python loop
launching each subpart. A whole-model graph can amortize per-graph bookkeeping;
a separate linear graph benchmark charges it once per linear.

All public ordinary-call allocation audits and graph replay live-growth checks
passed. The comparator's incremental capture peak ranges from 3.0 to 129.0 MiB
for MXFP8 and 3.5 to 144.2 MiB for FP4. These include its separately retained
forward, backward, and total graphs—not a single training step's peak.

Systems launch metadata for the representative narrow shape reports 38
registers/thread for the row producer and 56 for transposed producers, with
0 reported per-thread local allocation. Transposed producers use 2302 bytes
dynamic SMEM; finalizers use 25 registers/thread. These metadata do not measure
spill traffic or achieved occupancy.

### Recovered graph-amortization checks

After the connection was restored, the 5060 was verified idle with no surviving
profiler job. Only the missing checks were resumed. Both eight-backwards-per-
graph comparisons completed and passed finite-gradient, FP32-reference and
fresh-RNG guards; artifacts are now local in `paired_graph8/`.

| Shape | MXFP8 µs/backward | FP4 µs/backward | Paired FP4 time penalty |
|---|---:|---:|---:|
| 512 × 512 × 512 | 18.0 | 53.2 | +195.4% |
| 4096 × 2048 × 512 | 376.5 | 433.8 | +15.1% |

These capture eight complete backwards, not eight replays of an inner graph;
Torch graph/RNG bookkeeping is charged once per eight backwards. Ten randomized
AB/BA rounds compare the two precisions. This confirms that the preparation
penalty remains even when per-graph overhead is amortized. The one- and eight-
backward measurements are from different sessions, so their difference is not
a controlled measurement of the speedup from batching. Repeated identical
operands also do not represent eight distinct model layers' cache working sets.

### Hardware-counter limitation

Nsight Compute 2026.1.1 exists under `/opt/nvidia/nsight-compute/2026.1.1/`, outside
PATH. Its first fixed-randomness capture was rejected with **ERR_NVGPUCTRPERM**:
the account cannot access NVIDIA GPU performance counters. The launcher stopped
there; no driver or system permission was changed. `counters/fp4-dx.log` retains
the error. The associated JSON is ordinary benchmark/validation output, not
successful counter collection. No DRAM/L2 throughput, achieved occupancy,
spill-counter, or tensor-pipeline saturation claim is made.

The retained `environment.json` is the initial PATH-based preflight and therefore
reports Compute missing; the later direct executable check found version
2026.1.1.0. Counter profiling can resume after the machine owner enables access.

## Next experiments, in priority order

1. Improve the producer: transpose staging/barrier amortization and its
   normalization/correction instruction path. Preserve the numerical contract;
   the previously tested exact integer-division replacement was slower.
2. Measure inline versus staged amax reduction by shape; consider sharing one
   range reduction rather than repeating it in every finalizer warp. Small
   shapes particularly need fewer stages and less fixed work.
3. Expose paired FP4 preparation and inter-leg scheduling to amortize launches
   and potentially overlap independent work. Do not conflate shared source
   loading with reusing an incompatible rotated/quantized representation.
4. Keep GEMM tuning, especially wide shapes, but do not treat GEMMs as the
   main explanation for the narrow training-model gap.
5. RNG fusion is a secondary target here. The prior fused-Philox experiment
   saved workspace but was slower; this profile does not justify making it
   the default. Removing factor-product launches alone is a very small win.

## Reproduction

```bash
python -m benchmarks.profile_backward_subparts --output NEW_RUN/results
bash tools/profile_backward_timelines.sh "$VIRTUAL_ENV/bin/python" NEW_RUN/timelines
bash tools/profile_backward_counters.sh "$VIRTUAL_ENV/bin/python" NEW_RUN/counters /path/to/ncu
python -m benchmarks.summarize_backward_subparts NEW_RUN --output NEW_RUN/summary.json
```

Run serially on an idle GPU. Result files are exclusive-create; use a fresh
directory. The suite skips already completed result JSONs on resume and keeps
failed attempt logs. The summary includes component CSVs, timing quality,
kernel attribution, launch resources, and overlap-aware timeline coverage.
