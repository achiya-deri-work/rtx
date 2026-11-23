# G2DETR long-reduction MXFP8 dW — 2026-11-23

The reported `dY.T @ X` geometry (`X = dY = [138304,384]`) now has a
native TMA-scale path. On the same HOME RTX 5090, an ordinary eight-trial
search selected a schedule that reduced complete dW **0.759 → 0.278 ms
(2.73×)**, and paired dX+dW **0.974 → 0.500 ms (1.95×)**. Stock BF16 dW
was **0.178 ms**: this fixes the synchronization cliff, not every remaining
cost of low-precision backward.

## Cause and fix

The G2DETR report used RTX `2d396bd3` and a 128×128×128 partial GEMM with
row-major scale staging. Its 138304-token reduction ends 64 values into
a 128-value tile, so native blocked scales could not be selected. One
producer warp loaded E8M0 scales through ordinary loads, staged them in
shared memory and rendezvoused with all MMA warps at every K tile.

Current-source matched measurements reproduced the old slowdown. Changing
the producer's scale vector width did not help (0.759–0.760 ms). Moving the
same staging to consumer warps cut it to 0.379 ms. A mechanism-only control
with native TMA scales and setup-time BF16 padding reached 0.293 ms.
That padding control is **not** the implementation.

The implementation uses the original contiguous BF16 operands with their
logical transpose layouts. The register-loaded native transpose quantizer
predicates BF16 tail loads and writes an entire last packed tile: zero FP8
values and finite E8M0 scale bytes for fully missing blocks. No separate
padding or transpose launch, BF16 materialization, output-shape change,
precision change, or additional application API is introduced. Only the
internal packed operand K is rounded to 128. Data and scales share the TMA
copy pipeline and its existing stage-release ordering. FP32 split partials
and a final FP32 reduction with one BF16 cast are retained.

Support is initially limited to decomposed transpose/transpose dW with
separate quantizers, FP32 workspace splits, register-loaded quantizers and a
legal MMA128 or asymmetric MMA64×128 scale layout. Other source orientations,
quad quantization, cp.async tail loads, and incompatible M/N geometries
retain their previous contracts. Scratch sharing uses the physical packed
extent, not the logical K.

The shared backward search introduces native transport as an early compound
candidate and measures more than one split count. It is not a manually
installed winner. The tested search ran all eight trials including two
native choices; the selected dW uses 16 balanced splits and two TMA stages.
Old explicit configs still execute as supplied. RTX's backward winner
revision is 32; G2DETR's own explicitly pinned cache is separate and needs
an updated RTX checkout and retuning before an application run benefits.

## Measurements

Same 5090, Torch 2.13.0+cu132, CUDA 13.2, driver 595.99.02. Seed 730,
contiguous BF16 synthetic operands, matched ordinary CUDA graphs, nine
shuffled rounds of 20 replays. No clock or power changes. The probe
includes both operand quantizers, partial GEMM and FP32 workspace reduction;
setup and compilation are excluded. The paired-backward figures are the
fresh quick-search observations, not full detector or optimizer-step time.

| Complete dW | Median |
|---|---:|
| Stock BF16, confirmation | 0.1783 ms |
| Original 32-trial application schedule, confirmation | 0.7585 ms |
| Native scales, 32 splits (same split count as original) | 0.2934 ms |
| Native scales, selected by 8-trial paired search, confirmation | 0.2778 ms |

Nsight Compute CLI (2026.2.1) provides diagnostic counters, not the event
timing above. Nsight Python was not installed. Original producer, consumer,
and native paths were profiled in one narrow profiler-API range after warmup:

| Partial GEMM | NCU duration | Barrier stall per active warp | Tensor-pipe active |
|---|---:|---:|---:|
| Stock BF16 | 184.45 µs | 0.277% | 98.91% |
| Producer staging, 32 splits | 639.74 µs | 76.676% | 10.25% |
| Consumer staging, 32 splits | 195.23 µs | 3.815% | 28.16% |
| Native TMA scales, 16 splits | 84.45 µs | 0.025% | 58.54% |

The original two quantizers were 85.7 and 83.0 µs; the native pair was
82.4 and 77.0 µs. Workspace reducer fell from 13.6 to 7.6 µs with the
smaller split count. The profile shows why optimizing the reducer first
would not have fixed the report, and why improving preparation is now
more important than optimizing the partial GEMM alone.

Same-split PC sampling attributes 38,474 original barrier-stall samples
to the consumer instruction immediately after the named scale-ready
barrier region (barrier 1, 288 threads), before the subsequent
operand-pipeline phase check. The native trace records two barrier
samples near CTA initialization, with no scale-ready barrier in the hot
loop. These are sampled warp states, not percentages of wall time;
neither the counter result nor source adjacency alone measures how much
of the remaining latency is attributable to global load instructions.

## Correctness and scope

- The selected long dW agrees with an independent quantize/dequantize
  then FP32 GEMM oracle: **0.00634% normalized RMS**, max absolute error
  4.0 at the large synthetic output scale. All outputs finite. The
  predeclared implementation gate is 0.2% NRMS, not an assertion of
  equivalence to a BF16 GEMM. The confirmation graph loop has no net live
  PyTorch allocation growth.
- Native tail tests poison the buffers first, then compare packed values
  and scales byte-for-byte with independently zero-padded source operands
  using the established aligned kernel. They cover 32/64/128 quantizer
  tiles, MMA64 and MMA128, scalar and packed scale stores, and a partial
  32-value block. Fullgraph module tests check dX and dW, including
  noncontiguous dY and sequential scratch aliasing.
- The older `test_complete_compiled_recomputed_mlp_balanced_all_gradients`
  asserts a particular structural prefix and fails before compilation on
  both unchanged `8c5467c` and the candidate. It is excluded from the
  final focused regression, not recorded as passing. The remaining suite
  passed **88 tests and 397 documentation subtests** on the 5090.
- Claims are restricted to the supplied long/narrow gradient geometry on
  the 5090 and the tested synthetic regression shapes. The G2DETR checkout,
  run outputs and deployed cache were not modified. Full training speed
  and application quality have not been requalified.

## Reproduce

Raw results, selected config, tuner journal and both NCU reports are under
`research_data/g2detr_long_dw_20261123/` (ignored, local). Using an assigned
idle GPU, with `winner.json` the original 32-trial backward config:

```bash
python -m benchmarks.diagnose_long_dw --winner winner.json \
  --output confirmation.json --cases bf16,reported,tuned --tune-cache fresh-cache

ncu --profile-from-start off --clock-control none --cache-control none \
  --section SourceCounters --kernel-name 'regex:.*mxfp8_gemmMXFP8GemmKernel.*' \
  --export scale-pc python -m benchmarks.diagnose_long_dw --winner winner.json \
  --output unused.json --cases reported,native_tail --counters
```
