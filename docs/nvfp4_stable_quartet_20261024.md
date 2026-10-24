# Stable NVFP4 forward comparison against Quartet

Status: comparison-matrix objective achieved on the tested RTX 5090, using
explicit qualified configurations. No default or runtime-cache promotion.
Earlier investigation checkpoints below are historical; this summary supersedes
their open-gap status. This is not a universal shape/device or training claim.

## Final confirmed result

Fresh public `torch.compile(fullgraph=True, dynamic=False)` forward runs, all
quantization included, changed-input CUDA-graph replay verified. Means of two
fresh-process medians with reversed backend order; shape order is M,N,K.

| Shape | RTX us | Quartet us | Speedup |
|---|---:|---:|---:|
| 2048,768,1536 | 13.630 | 19.508 | 1.431x |
| 8192,768,1536 | 35.428 | 40.850 | 1.153x |
| 32768,768,1536 | 165.656 | 204.225 | 1.233x |
| 8192,3072,768 | 54.138 | 56.274 | 1.039x |

The formerly losing wide sparse holdout now wins all four pairs across seeds
41 and 73: mean 53.738 vs 54.394 us (1.012x). Even the slowest RTX process
is faster than the fastest Quartet process across those observations (1.008x).
Wide outliers average 54.915 vs 56.492 us (1.029x). The sparse margin is small;
it should not be extrapolated to other devices, distributions, or power regimes.
All 28 final comparison windows are stationary and both changed/restored-input
replay checks pass. RTX normalized RMS error is within 1.001 times Quartet's
in each matching run, often lower. Normal-data NRMS is approximately 0.123
for both; sparse approximately 0.0973 and outliers approximately 0.1253.

The decisive final change is `NVFP4GemmConfig.k_tile_unroll=6` for the wide
shape, with the existing native decoder and retuned quantizer. The former
fixed outer K-loop `unroll=1` was a missing scheduling coordinate. Values
1/2/3/4/6/8/12 are now exposed through `gemm_k_unroll` in the shared NVFP4
autotuner. Default 1 and legacy config interpretation remain unchanged;
this does not force unrolling onto other shapes or alter MXFP8 backward.

Qualified configuration records under the artifact root:

- Small: `small_native_final/confirmation.json`.
- Medium: `medium_native_decode/confirmation.json`.
- Long: `long_native_final/confirmation.json`.
- Wide: `wide_k_unroll6_quant_finalist/confirmation.json`.

Final validation: 135 tests plus 260 subtests passed (six intentional skips),
followed by a separate passing compiled forward/backward unroll equivalence
test. The 36 new GPU schedule cases cover non-dividing unroll factors,
different K lengths, pipeline depths, warp layouts, incomplete persistent
CTA work groups, and changing operands. Output and stable MXFP8 dX/dW are
bitwise equal between unroll-one and unroll-six in the public compiled test.
The source-derived API reference and `git diff --check` pass.

An integrated 16-trial search on GPU1 completed with the unroll-six initial
configuration retained; its 52.171-us search timing is not mixed into the
GPU0 public comparison table. No research winners were automatically installed.

## Contract

Compare dynamic BF16-input, BF16-output compiled forward, including all
quantization and current amax work. Keep Four-over-Six selection, FP32 outer
scales, and MXFP8 backward unchanged. JIT-region has explicit 8-by-8 row
geometry; tensor-current is a separate numerical policy, not a scheduling
choice. Quartet uses the isolated current-stream/GPU-seed research patch
documented in the previous study.

Final acceptance uses fresh processes, repeated/reversed ordering, ordinary
CUDA graph event timings, and changed-input replay checks through the public
module API. Profiling times and search minima are diagnostic, not final wins.
Independent decoded-packed FP32 GEMM NRMS must remain below 0.005; search
output NRMS must not worsen beyond 1.001 times its fixed-policy baseline.
Held-out inputs include separate seeds, sparse values, and outliers.

## Initial evidence

Artifacts: `autotune_reports/nvfp4_stable_quartet_20261024/`.
Hardware: two idle RTX 5090s; Torch 2.13.0+cu132, CuTe DSL 4.7.0.

The medium `8192 x 768 x 1536` baseline Nsight Systems capture contains two
kernels: quantization averages 23.60 us and GEMM 23.62 us across five calls.
This is a phase attribution, not the final graph-replay latency.

Nsight Compute reports 128 registers/thread for quantization and 66 for GEMM
(72 allocated), with no local-memory load/store sectors in either captured
kernel. Active-warp ratios are approximately 31% and 32%; these do not prove
bandwidth or tensor-core saturation. Native NCU was used because the optional
Nsight Python module is unavailable. Requested DRAM counters were absent from
the report and cannot support a bandwidth claim.

## Work sequence

1. Separate quantizer-only, GEMM-only, and joint scheduling searches to prevent
   the larger GEMM axis set from consuming the whole preparation budget.
2. Compare identical packed operands with the vendor GEMM before attributing
   a public-forward gap to preparation.
3. Use measured bottlenecks to select implementation experiments.
4. Confirm selected configurations through the public API, then test held-out
   numerics and the existing adaptive-quantization regression suite.

The diagnostic now accepts shape-checked forward winner records directly;
native scale preparation is outside the packed-only GEMM timing by design.

## Correctness finding

The wider search rejected two atom-M=2/4 fragment-register epilogue candidates.
A fresh-process atom-M=2 reproduction had 9.68% NRMS against the decoded-packed
oracle, while its direct epilogue passed at 0.166%. The fragment implementation
incorrectly hoisted two row factors across row groups owned by smaller MMA
layouts. Reloading X factors per four-accumulator group for atom-M != 8 fixes
the reproduction to 0.166%; the atom-M=8 fast path remains unchanged.

The direct/fragment equivalence test now covers atom-M 2/4/8 across 5x4, 8x4,
and 8x8 regional geometries: all nine CUDA subtests pass. JIT-region runtime
revision advances from 12 to 13 to invalidate old winner identities.

The medium identical-packed diagnostic measured RTX GEMM 22.78 us versus
cuDNN 20.48 us, with identical 0.168% oracle NRMS. That explains only part of
the original public-forward gap; both scheduling and preparation matter.

## First measured improvement

The selected medium quantizer moves from 8 to 16 values/lane with a 96-register
cap. Optional 32-bit packed stores combine four bytes without changing any
quantization arithmetic. Tests cover byte identity for all four block-scale
policies, ragged tails, and cached/uncached/warp-owned native regional paths.
The combined CUDA regression run passes 49 tests and 67 subtests.

Two fresh public compiled medium runs measure 39.50 and 39.35 us versus
Quartet 40.86 and 40.92 us. Both have approximately 12.30% original-BF16 oracle
NRMS; all four measurement windows are stationary and changed-input replay
passes. This reverses the prior roughly 15% loss but establishes only a ~4%
lead, not the desired robust margin or a multi-device claim.

Geometry proposals now couple non-fragment tiles with the general epilogue
and row-major scale transport. Merely changing geometry from a native
fragment seed made those configurations unreachable. The CPU reachability
test requires legal 256-row and 256-column alternatives without changing the
regional numerical policy. Those alternatives still need measured value;
their less efficient scale transport can outweigh the larger tile.

## Confirmed four-shape snapshot

Mean of two fresh-process medians, GPU0 only, reversed treatment ordering on
repeat two; all 16 windows stationary and changed-input replay passed.
These are explicit JIT-region Four-over-Six schedules, not installed defaults.

| M,N,K | RTX us | Quartet us | Quartet / RTX |
|---|---:|---:|---:|
| 2048,768,1536 | 15.25 | 19.53 | 1.28x |
| 8192,768,1536 | 39.42 | 40.89 | 1.04x |
| 32768,768,1536 | 166.96 | 202.94 | 1.22x |
| 8192,3072,768 | 59.47 | 56.53 | 0.95x |

Original-BF16 oracle NRMS remains approximately 12.3% for both implementations.
Searches run on either 5090, but cross-GPU search minima are not compared:
GPU1 runs cooler and boosts higher. Final comparisons above use GPU0.

Final configurations are `search_small_quant/winner.json`,
`medium_store32/confirmation.json`, `search_long_quant/winner.json`, and
`wide_store32/confirmation.json`, respectively, under the artifact root.
The large native-layout GEMM remains 128x128; general-layout geometry searches
did not improve either losing target. Tensor-current quantizer searches also
did not overtake JIT-region; no numerical-policy switching is installed.

A paired quantizer counter capture with otherwise identical configuration
shows global store requests falling from 243,040 to 81,760 with 32-bit stores.
Registers/thread remain 128 and active-warp ratios about 32%. Cold profiler
DRAM reads remain 27.60 MB; this is an instruction/request reduction, not a
claim that useful output bytes decreased. Profiler latency is not substituted
for graph timing.

The wide identical-packed diagnostic on GPU1 measures RTX GEMM 40.87 us and
cuDNN 37.36 us, with identical 0.168% decoded-packed NRMS. Regional epilogue
overhead is additional. The goal remains open: medium needs a larger margin,
wide still loses, and these results do not establish cross-SKU or untuned-shape
qualification.

## Native candidate decoder experiment

An optional `adaptive_decode="native"` replaces scalar nibble-LUT decoding
inside adaptive candidate evaluation with `cvt.rn.f16x2.e2m1x2`, then exact
widening to FP32. E2M1 values are exactly representable in both wider formats;
no scale/error arithmetic is moved to FP16. NVIDIA documents this instruction
for SM120a in the [PTX conversion reference](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt).

The medium fixed-schedule diagnostic improves preparation from approximately
15.4 to 11.5 us and full runner from 39.6 to 35.3 us, with unchanged output
metrics. Public confirmation and retuning follow; this is not yet the final
comparison table. Candidate packed-byte/scale identity tests passed before
timing. Native decode remains an optional scheduling choice, not a default
numerical-policy change.

Seed-41 sparse/outlier holdouts of the preceding LUT-decoder winners passed
all numerical and replay checks. RTX errors were slightly lower in those four
cases. Importantly, the medium sparse case was 40.47 us versus Quartet 39.91
us, illustrating why the earlier ~4% normal-input lead was not robust enough.

### Public confirmation of native decoding

Two fresh public processes per treatment, same GPU0, reversed second order:

| Shape | RTX native decode us | Quartet us | Speedup |
|---|---:|---:|---:|
| 8192,768,1536 | 35.15 | 40.68 | 1.16x |
| 8192,3072,768 | 55.39 | 55.61 | 1.004x |

All windows are stationary, replay checks pass, and output metrics are unchanged
from LUT decoding. Wide is a near-tie, not a stable victory. Its selected GEMM
uses M raster, swizzle 2, same-B persistence, two tiles and two waves.
The native tensor-current counterfactual remains slower (59.40 us runner on
GPU0) and is not promoted.

The wide GEMM counter capture reports 104 registers/thread, 17.55% active-warps
and 53.34% tensor-pipe activity. These are diagnostic occupancy/activity ratios,
not peak-throughput estimates. The current native scale transport confines
tiles to 128x128; general row-major larger tiles lost in the measured sweep.

Latest packed-store/native-decoder suite: 47 passed, six intentionally skipped
inactive amax/native combinations. It includes byte/scale identity, all three
regional ownership modes, and public fullgraph forward/backward equivalence
with unchanged MXFP8 gradients. Documentation and geometry reachability tests
also pass. Default policies and runtime winners remain unmodified.

### Native-decoder distribution holdouts and remaining boundary

Seed 41, one fresh process per treatment on GPU0; these are holdout checks,
not repeated-process win qualification:

| Shape / distribution | RTX us | Quartet us | RTX NRMS | Quartet NRMS |
|---|---:|---:|---:|---:|
| medium / sparse | 36.43 | 39.71 | 0.09730 | 0.09750 |
| medium / outliers | 35.14 | 40.70 | 0.12520 | 0.12536 |
| wide / sparse | 55.14 | 53.88 | 0.09738 | 0.09747 |
| wide / outliers | 55.63 | 55.75 | 0.12529 | 0.12550 |

All numerical, changed-operand/restored-operand replay, and stationarity checks
passed. These support the medium improvement, but disprove a stable wide win.
Artifacts are `native_holdout_{medium,wide}_{sparse,outliers}_{rtx_jit,quartet}.json`.

The final 64-trial wide native quantizer search completed on GPU1. Its winner
uses 64-bit input loads and six quantizer persistent waves; its search median
is 53.57 us. This GPU1 timing must not be compared with GPU0 public timings.
A GPU0 sparse public check measures 54.79 us with unchanged NRMS and passing
replay checks (`native_quant_holdout_wide_sparse.json`), still short of the
earlier Quartet sparse result. The winner remains experimental and uninstalled.

Next structural hypothesis: reduce the full-tile BF16 epilogue shared-memory
footprint while retaining the correct 128x128 native-scale MMA tile. A chunked
epilogue could permit more resident CTAs for low-stage schedules, but is not
implemented or established as a win. Existing larger general-layout tiles and
additional quantizer/locality tuning have not closed the wide gap. Native
64-row layouts remain quarantined; their correctness issue must not be bypassed.

The goal remains active. This study establishes useful forward improvements,
not a universal Quartet win, cross-SKU qualification, or a training-quality
claim. Stable MXFP8 backward and public default numerical policies are unchanged.

## Rejected workspace-reuse experiment

Two prototypes overlay dead operand shared storage with the BF16 TMA epilogue:
A-only reuse, then combined A/B/block-scale reuse. Both restrict execution to
one output tile per CTA and one output stage, drain the producer pipeline, and
use a whole-CTA rendezvous before overwriting operands. They retain the native
128x128 scale layout and unchanged FP32 accumulator scaling.

The initial A-only version failed the independent packed oracle because its
output swizzle was represented incorrectly. Putting the output swizzle on the
recast pointer fixed it: packed NRMS returned to 0.001662. No timings from the
incorrect version are accepted. A nested-conditional DSL compile issue in the
combined-storage prototype was also corrected before final measurements.

Each prototype received 64 GEMM trials on GPU0. Best full-runner medians were
60.13 us (A-only) and 56.95 us (combined), still slower than the existing wide
configuration. Every timed candidate passed the independent packed oracle.
The combined winner has two operand stages, atom-M=4, interleaved MMA, and
one output tile per CTA. These experiments do not establish that occupancy
improved: no new counter capture was collected for them. Lower shared-memory
reservation alone did not produce the required end-to-end improvement.

Both prototypes were removed from production code. Residuals remain in
`search_wide_reuse_a` and `search_wide_reuse_operands`. The combined kernel diff
is archived as `rejected_operand_reuse_kernel.patch` for research, not as a
supported patch to install. It includes the preceding fragment correctness fix;
the prototype also required a temporary `NVFP4GemmConfig.epilogue_reuse_a=False`
field and rejection of non-TMA, multi-output-stage, or multi-tile configurations.
Old prototype winner files containing that removed field are not runtime configs.

### Remaining native-decoder public confirmations

After removing the rejected prototypes, repeated fresh-process GPU0 public
comparisons also confirm native decoding for small and long shapes:

| Shape | RTX us | Quartet us | Speedup |
|---|---:|---:|---:|
| 2048,768,1536 | 13.643 | 19.544 | 1.433x |
| 32768,768,1536 | 163.863 | 202.836 | 1.238x |

Each number averages two fresh-process medians; treatment order is reversed
for the second pair. All eight timing windows are stationary. Compiled/eager
equality, changed-input replay, and restored-input replay pass. RTX/Quartet
NRMS is 0.122975/0.122880 (small), 0.122923/0.122842 (long), unchanged by native
candidate decoding. These are forward-only, distribution-specific comparisons,
not convergence or full training speedups.

Explicit configs: `small_native_final/confirmation.json` and
`long_native_final/confirmation.json`; public artifacts follow
`native_public_{small,long}_{rtx_jit,quartet}_{1,2}.json`.

Post-removal regression command:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q \
  tests/test_nvfp4_packed_stores.py tests/test_packed_gemm_diagnostic.py \
  tests/test_documentation.py tests/test_nvfp4.py \
  -k 'packed or vector_store or fragment_register or documentation or native_scale_inverse or independent_fp4'
```

Result: 98 passed, 6 intentionally skipped, 47 deselected, 260 subtests passed.
The current remaining comparison gap is wide, especially sparse inputs; neither
the negative workspace experiments nor these additional wins close that gap.

## Follow-up wide experiments: no new winner

The native-decoder wide GEMM received another 256 trials on GPU0
(`search_wide_native_gemm`). Its starting config remained the winner. The
initial median was 56.48 us; many subsequent neighboring configs measured
about 57.5 us as the device warmed. The GPU reached 74 C and 2737 MHz in an
in-run observation. This search is not a new public comparison or proof that
sub-percent alternatives are worse; thermal drift requires paired finalist
confirmation. It did not produce a candidate worth promoting.

### Packed FP32 instructions

An explicit `mul.rn.f32x2` prototype preserved FP32 arithmetic in regional
factor products and accumulator pairs. NVIDIA documents this instruction for
SM100+ in the [PTX reference](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul).
Nine fragment-layout/region subtests passed bitwise equality. However, two
fresh-process GPU1 A/B pairs gave full runner times 53.306/53.357 us (scalar)
and 53.281/53.338 us (packed), with no consistent GEMM benefit. Applying packed
multiplication to adaptive quantizer normalization/reconstruction also left
preparation at 7.43 us. Both experimental knobs were removed; the archived
`rejected_packed_fp32_kernel.patch` contains research code and preceding
uncommitted changes, not an installable standalone release patch.

### Larger native-scale tiles

Implemented experimental native-scale transport for 256x128 and 128x256 CTAs.
The initial contiguous-1024-byte-block assumption failed the packed oracle
(about 30% NRMS). CuTe's SM120 scale layout interleaves 512-byte blocks of
128 rows and four scale values, so K=128 NVFP4 requires two separately placed
512-byte transfers per row group. Correcting this yielded the expected packed
NRMS 0.001662 with unchanged original-input error for both shapes.

Performance did not follow correctness: wide-forward runner times were
101.63 us for 256x128 on GPU1 and 142.37 us for 128x256 on GPU0 with one-stage
TMA epilogues. The 64-KiB output tile prevents a two-stage operand schedule
within the shared-memory limit. Three-stage direct-store alternatives measured
160.83/166.27 us respectively. Extending fragment-register scale reuse remained
correct but did not materially change these results (101.87/142.16 us).
These are representative probes, not exhaustive tuning of the new geometry.

The prototype and relaxed legality were removed from production. Its transport
and fragment extension are archived in `rejected_large_native_tiles.patch`
(the archive includes the preceding fragment correctness fix). Results use
`wide_native256{m,n}_{probe,fixed,interleaved,direct,fragment}` directories where
present. Failed prototypes were never timed as valid or promoted.

### Chunked BF16 TMA epilogues: implemented, tested, not retained

Tested 32/64-row output chunks for 128-row CTAs and 128-row chunks for
256-row CTAs, keeping the FP32 regional numerical policy unchanged. The
first 64-row prototype used an STMatrix atom spanning 128 rows and caused
an illegal access; a static legality guard and smaller STMatrix atom fixed
that ownership error. It was never accepted as a valid timing result.
The corrected implementation passed 17 tests covering two shapes, four
tile/atom/chunk geometries, two operand-stage counts, heterogeneous/zero
rows, and repeated changes to both operands with bitwise output agreement.

The large-tile prototype's slowness was measured, not attributed merely to
occupancy: `chunk256_counters.ncu-rep` reports 96 registers/thread,
91.136 KB shared memory/block, 34.49% active warps, 5,818,820 local-load
sectors and 5,345,280 local-store sectors. This demonstrates substantial
register spilling for that specific configuration. Register/schedule changes
reduced the full runner from a 91.83-us GPU1 probe to 61.94 us on GPU0;
these are different devices and not a controlled speedup ratio. Neither
result beat that device's existing wide winner. Spills were not reprofiled
for the revised configuration.

Two 96-trial searches also found no replacement winner:

- GPU0 `search_wide_native_chunk256_v2`: best 58.173 us, reverting to a
  128-row CTA with a full 128-row output epilogue.
- GPU1 `search_wide_chunk32_v2`: best 54.386 us with a 64-row output chunk,
  versus the earlier same-device full-epilogue probes around 53.3 us.

These search minima are not fresh paired public comparisons and do not
support fine-grained ranking across devices or thermal states. They provide
no evidence to replace the confirmed winner. Chunk fields, kernel branches,
experimental search entries, and widened native-layout legality were removed
from production. The prototype kernel/config diffs and tests are recoverable
under `rejected_chunked_*` in the artifact directory. Patches include earlier
uncommitted work and are research snapshots, not standalone release patches.

The goal remains active: three tested shapes have confirmed forward wins,
while wide is near parity on normal data and still loses the sparse holdout.
Do not claim a stable across-the-matrix win. Before another structural change,
reprofile the retained wide winner against an identical-packed GEMM baseline;
the large-tile spill result does not establish the retained winner's bottleneck.

After removing the prototype, the retained regression command above was rerun:
98 passed, 6 skipped, 47 deselected, 260 subtests passed in 24.98 seconds.
`git diff --check` also passed. The shared GEMM's remaining modification is
the independently tested fragment row-factor correctness fix; MXFP8 geometry
and shared-memory accounting are restored to their prior implementation.

## Retained-wide diagnosis and direct-store follow-up

The retained wide schedule was remeasured against identical Quartet-packed
operands (`wide_retained_packed_comparison.json`): RTX GEMM 43.519 us,
cuDNN 38.910 us, both NRMS 0.00167869 against decoded packed FP32 matmul.
This scalar-outer diagnostic removes regional epilogue differences; it is
not a substitute for the public dynamic-forward comparison.

`wide_retained_counters.ncu-rep` captures the actual regional GEMM: 104
registers/thread, 91.136 KB shared memory/block, 17.53% active warps,
51.03% tensor-pipe activity, zero local load/store sectors, and 222.94 MB
L2 traffic. It is a one-CTA-per-SM resource regime without register spills.
Removing aligned-tile fragment bounds checks changed full runner timing
56.816 to 56.723 us, insufficient evidence of a meaningful win; it was reverted.

Native K=64 scale transport was also tested by allowing its contiguous
512-byte half-block mapping. The independent packed oracle passed unchanged
at NRMS 0.00166219. Full runner times were 90.061 us (one stage, GPU0),
71.525 us (three stages, GPU0), and 66.983 us (four stages, GPU1).
A five-stage request was statically rejected before launch. The legality
extension was reverted: smaller K tiles did not beat retained schedules.

An experimental paired BF16 direct-store epilogue reduced a fixed direct
configuration from 149.154 to 72.535 us on GPU0. Reusing the FP32 factor
within an aligned two-column region pair gave 72.124 us. Eighteen tests pass
bitwise scalar/paired agreement across atom-M 2/4/8, aligned/ragged shapes,
odd/even regional widths, and changing operands. A 96-trial GPU1 search
reached 65.165 us, still behind that device's retained TMA result near 53.3 us.
These direct-path probes are not accepted public wins.

The paired-store counter capture explains a remaining constraint: 125
registers/thread, 39.936 KB shared memory, 18.33% active warps, zero spills,
36.69% tensor-pipe activity, and 273.67 MB L2 traffic. Reduced SMEM alone
did not increase residency because register allocation still limits it.
Do not infer that a requested register cap equals the actual compiled count.
A minimum-two-CTA compiler launch bound regressed the conservative schedule
to 81.765 us. Applying it to the search winner with 224 consumer registers
stalled: the worker remained active with GPU1 at 100% and no result. It was
terminated (exit 143); a fresh-process CUDA allocation, addition, reduction,
and synchronization passed, and both GPUs returned to idle. Conflicting
register redistribution and residency requirements are a suspected cause,
not a verified device-level diagnosis. This combination must not be searched
without a resource-feasibility guard and isolated timeout enforcement.

The paired-store implementation and minimum-CTA knob were removed from the
runtime and archived as `rejected_direct_pairs_*`. While they improve a weak
direct baseline, they do not beat the retained TMA path on the target problem.
The retained regression suite then passed 98 tests and 260 subtests, six skips.

### Deferred TMA-store wait

Tested moving the output-buffer acquire from immediately after TMA commit
to immediately before the next shared-buffer write. The final read-completion
drain and the MMA-warp buffer-reuse barrier remain. The current two-tile
winner measured 56.822 us versus 56.816 us before the change. With eight
tiles per CTA, deferred/immediate timings were 68.425/68.758 us: neither was
competitive. This does not demonstrate a meaningful exposed output-wait cost
in the retained schedule. The experiment is archived, not a promoted knob.
Twelve bitwise tests passed across two aligned shapes, atom-M 2/4/8, one/two
output buffers, four tiles per CTA, and repeated operand changes. Requested
ragged TMA test cases were rejected by existing legality before launch;
these tests do not extend the supported TMA shape contract.

At that checkpoint, next work was a genuinely smaller native-scale output tile or a
different GEMM mainloop, with independent packing/layout validation first.
The evidence does not support further register-cap or output-wait searches
as the main route to closing the wide-shape gap.

## Final scheduling investigation

A native 64-column output tile passed 18 bitwise GPU cases, but its 96-trial
search remained slower. Counter evidence explains why occupancy alone was
misleading: active warps rose from 17.53% to 34.40%, while L2 traffic rose
from 222.94 to 317.32 MB and tensor-pipe activity fell to 44.44%. The smaller
tile sacrificed operand reuse. Its implementation was removed and archived.

Prefetching regional FP32 factors before MMA also failed to improve the
retained schedule. Ordinary, forced-volatile, and unique-factor variants
measured approximately 56.9, 156.3, and 74.2 us respectively. These are
diagnostic runner timings, not public comparisons. Bitwise tests passed,
but all experimental paths were removed rather than expanding the API.

Outer K-loop unrolling was the successful alternative. The unroll-six NCU
capture reports 132 registers/thread versus 104, unchanged 91.136 KB shared
memory, zero local load/store sectors, and approximately unchanged active
warps (17.44%). Tensor-pipe activity rises from 51.03% to 53.31%; diagnostic
kernel duration falls from 53.248 to 51.072 us with nearly unchanged L2
traffic. This supports a scheduling improvement, not an occupancy or reduced
memory-traffic claim. Final acceptance remains the public comparison above.

The final comparison audit records matched configurations, stationary windows,
changed/restored replay, numerical gates, source identities, and repeated
process wins in `final_comparison_audit.json` under the artifact root. A final
cleanup removes an unreachable rejected-prefetch branch; benchmark source
hashes remain the identities of the measured code, not of that later cleanup.
After cleanup, all 38 unroll tests pass again, including public compiled
forward/backward equivalence. API-reference freshness and whitespace checks pass.
