# Guarded scale-free packing: exactness scope

`normalization_math="guarded_scale_free"` is a research-only VPL4 option.
Default `"divide"` is unchanged. The required exact comparison is against
`"threshold_scale_free"`, **not against divide's corrected-scale statistics**.

The two scale-free modes share exactly the existing power-normalized FP32
values, block-wide safe/eligibility tests, EDEN dot products, reduction order,
final exponent restoration, and unsafe fallback path. No approximate value is
used in the statistics. Only the safe E2M1 packing mechanism changes.

## Proposal and proof

Let `u` be the existing exact power-of-two-normalized value and let
`m=t/16`, `16<=t<=31`. A warp computes all sixteen `r=RN32(1/m)` entries once
at startup, one entry per lane with duplicate entries in lanes 16..31.
An unconditional full-warp shuffle retrieves r before eligibility branches.
There is no table tensor, allocation, HBM read, or hardware approximate
reciprocal instruction.

The proposed packing input is `p=RN32(u*r)`, which need not equal precise
`a=RN32(original/scale)`. On the existing nonzero safe domain,
`2^-60<=abs(u)<=16`, the true quotient, proposal and precise quotient are
normal and finite. With `eps=2^-24`, the reciprocal relative error is at most
eps, the proposed quotient relative error is at most `2*eps+eps^2`, and the
precise quotient relative error is at most eps. Their difference is therefore
bounded by `(3*eps+eps^2)*abs(u/m)`. Allowing an adjacent binade's smaller ULP
gives fewer than eight FP32 magnitude-word steps between p and a. W=16 is
deliberately conservative.

Every positive E2M1 midpoint `.25,.75,1.25,1.75,2.5,3.5,5` has an FP32
magnitude word divisible by `2^21`. Thus, with `z=absbits(p)`,

```text
near = ((z + 16) & 0x1fffff) <= 32
```

detects a sixteen-word neighbourhood on both sides of every midpoint. It also
detects harmless additional boundaries; those only cause extra fallbacks.
If p and a could produce different E2M1 codes, their interval would contain a
midpoint and this test must fire. Near cases use the original explicit
`div.rn.f32 original,scale` before native E2M1 conversion. This preserves ties
to even as well as ordinary rounding. Signed zero bypasses the unnecessary
division: positive scale and positive reciprocal preserve its exact sign.

The [official PTX reciprocal specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp)
distinguishes IEEE rounded reciprocal from the bounded approximate operation.
This implementation uses the stronger rounded contract, not an assumed MUFU
error. PTX literals are encoded in the assembly string rather than supplied as
literal Float32 read operands, avoiding the installed NVVM wrapper bug.

## Validation and numerical distinction

Tests exercise all sixteen reciprocal mantissas, midpoint ties and neighbours,
both signs, subnormals, exponent extremes and nonfinite fallback cases.
The GPU producer comparison requires **all five prepared arrays byte-exact**
against threshold_scale_free under changed-source/sign/uniform graph replay.
It also checks exact packed bytes against divide and reports the corrected
BF16-scale differences against divide, with the established one-BF16-ULP bound.
That bound is a new scale-free numerical contract, not bit equality to divide.

Public original-BF16/FP32 Gaussian checks, packed-GEMM NRMS <= .005, current
buffers, fresh RNG, forward matching and ordinary/replay allocation checks
remain unchanged. No acceptance tolerance or old exact test is relaxed.

Run on the assigned idle GPU only:

```bash
RTX_RUN_GUARDED_SCALE_FREE_CUDA=1 python -m pytest -q -s tests/test_ms_eden_guarded_scale_free.py
```

GPU qualification and ordinary complete-backward performance remain pending
at implementation time. Passing packing equivalence alone is not a speed win.
