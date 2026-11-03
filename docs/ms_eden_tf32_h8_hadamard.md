# Guarded TF32-H8 Hadamard prototype

This is a **new-file-only pure-transform experiment**, not a production
MS-EDEN change. It does not quantize, compute EDEN statistics, generate RNG,
register autograd, or alter existing kernel/adapter dispatch. The launcher
consumes original contiguous BF16 values and caller-owned Float32 signs and
produces full normalized H128 FP32 values in original element order, plus one
Int32 route flag per 128-value block.

## Instruction and bit mapping

Installed CuTe provides `warp.MmaTF32Op((16,8,8))`, wrapping
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`. Its verified local traits:

```text
A TV: ((4,8),(2,2)):((16,1),(8,64))
B TV: ((4,8),2):((8,1),32)
C TV: ((4,8),(2,2)):((32,1),(16,8))
```

For A(m,k) and C(m,n), map the four M bits to logical positions {4,5,6,1}
and the three K/N bits to {0,2,3}. Then the C fragment at physical lane L,
register R is exactly original element `4*L+R`. There is no post-MMA
permutation. A's four original source offsets per lane are
`base+{0,2,8,10}`, where `base=(L&1)+((L&2)<<1)+((L&28)<<2)`: direct gathers,
not a GMEM transpose. Those gathers are not the baseline's contiguous vector
load; instruction/sector-request costs must be measured.

The one MMA computes H8 on original bits {0,2,3}, with static H8 coefficients
`(-1)^popcount(k&n)` represented exactly in TF32. Finish bit1 locally with
four FP32 add/subtract operations, then logical bits4/5/6 using lane XOR
distances4/8/16. Apply the existing FP32 normalization constant once afterward.
An all-128-basis integer proof verifies this factorization equals H128 in real
arithmetic. CuTe GEMM construction and rank-(V,1,1) fragment verification pass
locally without GPU compilation; subsequent 5060 device qualification is recorded below.

## Precision and fallback contract

Each warp checks all 128 signs once per kernel invocation; every sign must
be exactly Float32 +1 or -1. For every 128-value block, all original BF16
inputs must be zero or have magnitude in `[2^-96,2^96]`. Both checks are
warp-uniform before issuing MMA. Otherwise the full original scalar sign
multiply/seven-stage FWHT route is used. Generic Float32 signs are not narrowed
to sign bits or silently rejected. A `force_scalar=True` compiler configuration
provides the same scalar route as a control.

On the fast domain, BF16 signed values are exactly representable in TF32;
there is no TF32 rounding of an intermediate FP32 transform. The smallest
possible nonzero exact sum has grid `2^-103`, safely normal, and an absolute
sum of all 128 terms is at most `2^103`, safely finite. Thus the guard excludes
subnormal inputs/partial sums and overflow-prone inputs from the MMA route.
Zero results and cancellation are still explicitly tested.

This **does not imply bit equality to scalar FWHT**. It changes both the order
of bit stages and the dot-product accumulation association. The
[official PTX MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
promises at least single-precision accumulation for TF32 but leaves ordering,
rounding and subnormal handling unspecified. No stronger hardware IEEE
rounding guarantee is assumed here. GPU quality must be measured against an
independent FP64 H128 matrix oracle, alongside the original scalar result.

A conditional numerical estimate, *not a hardware guarantee*: under ordinary
FP32 nearest/directed-rounding models, an eight-term dot contributes a small
multiple of the FP32 unit roundoff, followed by four scalar butterfly stages.
An L2 estimate is a few times `1e-6`; cancellation precludes a uniform
per-coordinate relative-error claim. Tests use an explicit pure-transform
NRMS/normalized-max gate of `1e-5` and report max absolute error plus scalar
association differences. These are new primitive gates, not altered gradient
or packed-GEMM gates. Production integration would still require all existing
original-BF16/FP32, gradient, RNG, capture and memory qualification.

## Cost hypothesis and current evidence

One warp-level MMA performs 2048 conventional tensor FLOPs for 128 outputs.
It replaces one local butterfly stage and two lane-shuffle stages: eight
shuffles, eight sign-selects and their scalar adds disappear. It needs only
four FP32 accumulator registers per lane, unlike the previous 64-register
multi-row H16 proposal. Remaining costs include TF32 operand conversion, four
unit sign products, static B fragments, range voting, changed A load addresses
and tensor-pipe latency. No performance win follows from operation counts.

Read-only CPU association model (serial FP32 H8, **not actual MMA simulation**):
2048 Gaussian/outlier/sparse blocks each gave NRMS about `3.1e-8` versus an
independent FP64 matrix oracle, with 18/17/1 changed FP32 words versus scalar
FWHT. This supports feasibility, not GPU numerical or timing qualification.

Run the new tests only on the assigned idle GPU:

```bash
RTX_RUN_TF32_H8_CUDA=1 python -m pytest -q -s tests/test_ms_eden_tf32_hadamard.py
```

The original TF32 prototype subsequently passed 14 assigned-5060 tests
(parent-reported, 8.12 seconds), with FP64 NRMS about `3.2e-8`, six/one changed
FP32 words for Gaussian/outlier cases, and fallback/capture/memory passing.
This does not qualify production MS-EDEN. Standalone large-case timing was
approximately tied at 106.9/106.6 microseconds; materialized output traffic can
mask any transform arithmetic improvement. Smaller working sets require
batched graph replay to avoid host-launch floors.

## Optional native BF16 padded H8

`TF32H8Config(mma_kind="bf16_padded")` is a second pure-transform experiment;
the default remains `"tf32"`. The installed atom is
`MmaF16BF16Op(BFloat16,Float32,(16,8,16))`, corresponding to
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. Verified traits are:

```text
A TV: ((4,8),(2,2,2)):((32,1),(16,8,128))
B TV: ((4,8),(2,2)):((16,1),(8,64))
C TV: ((4,8),(2,2)):((32,1),(16,8))
```

The same logical bit mapping applies. For lane L, the first four BF16 A
elements are precisely `source[4*L+0:4]`; its other four elements have K>=8
and are zero. A is therefore populated with two packed Uint32 source loads,
two packed sign-mask XOR operations, and two zero packed registers. B's first
packed register holds H8 coefficients at `k=2*(L&3)+{0,1}, n=L>>2` and its
second packed register is zero. C is unchanged: `4*L+register`.
The CPU test enumerates installed A/B/C traits, constructs both padded
matrices over every H128 basis vector, and proves the complete factorization.
It also verifies rank-(V,1,1) GEMM IR with the packed A register view.

The four contiguous Float32 signs are loaded once per launch, checked for
exact ±1 across the warp, and reduced to two packed sign masks. XOR is used
only on the fast branch after that check; arbitrary Float32 signs still use
the original scalar multiplication fallback. BF16 range checking uses the
exact unpacked original bit patterns, with the identical zero/[2^-96,2^96]
guard. The scalar control retains its cached contiguous signs; no extra sign
cache is added to the TF32 fallback. Both MMA kinds run the same FP64 quality,
fallback, capture, changed-buffer and live-memory gates. FP32 accumulation
ordering/rounding remains bounded only by the PTX contract above, not by an
asserted IEEE equivalence to scalar or TF32 MMA.

The hypothesis is fewer source-load/conversion instructions than gathered
TF32, at the cost of twice the MMA K work (half is padding). Transform GMEM
bytes/output materialization are unchanged, and no performance benefit is
claimed. The benchmark accepts `--mma-kind bf16_padded`; its scalar reference
and correctness gates are otherwise unchanged. At BF16-variant freeze, four
CPU tests passed before deployment. The combined suite subsequently passed
all 26 CPU/GPU cases on the 5060 in 8.83 seconds, including both MMA kinds.
Gaussian/outlier NRMS remained about 3.2e-8, with six/one changed FP32 words
relative to scalar FWHT for either MMA kind. This qualifies only the pure
transform, not fused preparation or backward. The prior 14-test
result applies only to the older TF32 snapshot.

## Fused preparation experiment

`rtx/kernels/ms_eden_h8_preparation.py` subclasses the existing preparation
kernel and replaces only its four-values-per-lane H128 transform. Scalar H128
was extracted into an inlined method without changing its butterfly order.
Loaders (including logical transpose and async staging), normalization,
quantization, FP32 EDEN statistics, corrected BF16 scales and partial maxima
remain shared. There is no extra launch or transformed GMEM tensor.

Unlike the standalone packed-load experiment, this prototype receives the
existing FP32 sign products. Every lane checks that these values are exactly
BF16-representable (low 16 bits zero), zero or in `[2^-96,2^96]`; a warp vote
guards the MMA path. This preserves arbitrary FP32 signs without rounding
their product down to BF16: nonrepresentable products use the scalar route.
Representable products may use H8 even for non-unit signs. No FP32 intermediate
of the transform or correction is narrowed. H128 accumulation association still
changes, so scalar byte equality is not the general contract.

On the 5060, 17 initial tests passed in 10.87s: Gaussian/outlier/zero/arbitrary
signs, row/direct transpose/coalesced transpose (one and two async stages),
FP64 transform-oracle decoded error comparison, changed-buffer graph replay,
partial-maximum consistency and stable allocated memory. Candidate decoded
error may not exceed scalar error by more than 1e-4; candidate/scalar decoded
NRMS must be <=0.005. These preparation checks do not replace complete-gradient
or training qualification. Two additional CPU tests check scoped compiler
dispatch and restoration on exceptions.

Grouped VPL8/16 producers are explicitly unsupported by the new compiler;
they retain their original implementation in the research comparison wrapper.
`benchmarks/experiment_ms_eden_h8_preparation.py` can substitute row-only or
all VPL4 producers, records actual dispatch and source hashes, and retains
the complete-backward comparison's independent FP32 gradient/graph/memory
gates. Stable API, schedule space and default dispatch do not select H8.
