# MS-EDEN register-sign reuse: 5060 diagnostic

An optional `producer_signs="register"` schedule loads each thread's four
Hadamard signs once before its persistent grid-stride loop, instead of loading
them for every 128-value tile. No rotation, normalization, correction, packing
or rounding arithmetic changes. It trades four live FP32 registers for fewer
repeated sign loads; the reload fallback remains available.

On the RTX 5060 Laptop (Torch 2.13.0+cu132), original BF16 sources and
`(M,N,K)=(4096,2048,512)`, 128 producer threads / 256 CTAs, coalesced-wide loads:

| Complete conditional gradient | Reload / register medians, us | Paired speedup | 95% round-bootstrap interval |
|---|---:|---:|---:|
| dX | 180.28 / 173.96 | 1.0314x | 1.0264–1.0358x |
| dW | 225.66 / 216.49 | 1.0383x | 1.0230–1.0573x |

Six rounds, both orders per round, four gradients per CUDA graph and forty
replays per timing sample. Sources/signs change between rounds but are matched
between candidates. All packed operands, final scales, FP32 factors and complete
gradient outputs were bit-exact. The intervals bootstrap six round-level paired
ratios, not independent launches; they are within-session evidence only. Full
public backward/training and other projection shapes need separate acceptance.

Eight additional GPU tests passed for original/packed sources, transpose and
nontranspose, direct/coalesced loads, changed signs/source values, zero inputs
and CUDA graph replay. Public integration forwards the static `producer_signs`
selection; the research autotuner exposes it independently of geometry.

Generated PTX confirms the producer still has six `div.rn.f32` operations per
four-value fragment (maximum/6, four normalizations and EDEN correction). This
is source-lowering evidence, not a measured instruction-throughput bottleneck.
No approximate reciprocal replacement was applied.

## Rejected exact-normalization experiment

The observed repeated divisions motivated a separate exact quotient prototype.
E8M4 scales have a five-bit significand: an integer denominator in [16,31]. The
prototype shared a reciprocal estimate across the four values, recovered the
integer quotient by remainder correction, checked that the remainder certified
the exact quotient, then used integer ties-to-even rounding to construct FP32
bits. Unsupported exponents/subnormals/exceptional values fell back to division.
This did not simply replace division with approximate multiplication.

Validation passed **134,256,140** comparisons: all 2^23 numerator significands
against all sixteen denominators at representative normal exponents, plus
38,412 signed/exponent-boundary/fallback cases. The final sweep used explicit
inline PTX `div.rn.f32` as its independent GPU oracle; the edge matrix also
matched a CPU Float64-to-Float32 reference, preserving signed zero. NaN payload
differences were excluded. Seventeen producer/replay/edge tests also passed.

It was nevertheless slower on the same full-gradient comparison, with register
sign reuse enabled for both sides:

| Conditional gradient | Ordinary division / exact integer normalization, us |
|---|---:|
| dX | 172.95 / 238.34 |
| dW | 212.52 / 279.88 |

The additional integer control/arithmetic did not pay for the saved divisions.
The prototype, schedule field and tests were **removed from production sources**;
it is not an autotuner choice. Archived source, proof results and paired timing
remain under `autotune_reports/ms_eden_exact_normalization_20261025_5060/` and
the intact remote snapshot `/home/achiya/rtx-exact-normalization-20261025.c6RaZm`.
This negative result should prevent repeating the same transformation without
a materially different implementation.

Artifacts: `autotune_reports/ms_eden_sign_reuse_20261025_5060/` contains paired
JSON, GPU test log and generated producer PTX. Remote persistent snapshot:
`/home/achiya/rtx-sign-reuse-20261025.hM5U3X`. Only the 5060 was used.

Reproduce with the same snapshot environment:

```bash
python -m pytest -q tests/test_ms_eden_sign_reuse.py
python -m benchmarks.tune_ms_eden_training_shapes --phase dx --portfolio signs --output dx.json
python -m benchmarks.tune_ms_eden_training_shapes --phase dw --portfolio signs --output dw.json
```
