# Bounded-input MLP divergence: where precision is lost

2026-09-12, RTX 5070 Ti, Torch 2.13.0+cu132. This is a numerical
investigation, not a performance benchmark or a kernel change.

## Experiment

54 cases: two M/K/H/O shapes (64/128/256/128 and 512/768/1536/768),
SiLU/GELU/ReLU, three seeds, and three distributions. All source tensors are
BF16. Distributions are uniform [-0.5,0.5], the same samples divided by four,
and uniform inputs/upstream gradients with truncated-normal weight std=0.02,
bounds +/-0.04. BF16 rounding may include interval endpoints.

Each case evaluates 19 arithmetic ablations and validates the full quantized
simulation against actual native forward and all three gradients. It also
compares actual approximate-reciprocal and division policies with every other
kernel setting fixed. Independent BF16/FP32 autograd baselines use the same
inputs and dY. TF32 and BF16 reduced-precision reductions are disabled.

Metrics include finite checks, absolute/RMS/relative error, normalized maximum,
cosine and worst element. All recorded comparisons are finite. Tables below
are arithmetic means across three seeds at M512/K768/H1536/O768; they are
not confidence bounds or a corpus-wide guarantee.

## Small magnitudes do not remove E4M3 precision loss

For uniform [-0.5,0.5] operands, reconstruction error is approximately 2.37%
relative L2 per operand. E4M3 retains three explicit fraction bits. E8M0
power-of-two scaling controls range but does not restore discarded fraction
bits. Dividing every source tensor by four leaves operand and gate/up
projection relative errors unchanged in this experiment.

Even with std=0.02 weights and reference gate RMS around 0.141, appreciable
end-to-end error remains. Thus this is not primarily overflow or large-gate
saturation. Relative proximity to BF16 cannot be inferred from bounded inputs.

### SiLU forward error growth

Each row compares that stage to its unquantized FP32-arithmetic counterpart,
with the same explicitly modelled BF16 boundary casts where applicable.

| Stage | Relative L2 |
|---|---:|
| Quantized X | 2.371% |
| Quantized gate/up W | 2.371% |
| Gate/up projection | 3.355% |
| SiLU(gate) | 3.447% |
| SiLU(gate) * up | 4.825% |
| Quantized hidden product | 5.501% |
| Final output | 5.993% |

Both multiplicands of the gated product contain projection noise. Hidden
quantization adds another rounding step, then down weights add another source
of error. The observed growth is consistent with compounding operand noise;
this is an interpretation of the measurements, not an independence proof.

## Full forward/backward versus ordinary BF16

| Activation | Output | dX | dW gate/up | dW down |
|---|---:|---:|---:|---:|
| SiLU | 6.004% | 6.361% | 6.370% | 6.002% |
| GELU | 5.988% | 6.597% | 6.606% | 5.991% |
| ReLU | 5.935% | 11.849% | 11.872% | 5.941% |

BF16 itself differs from the unquantized FP32 autograd reference by roughly
0.33–0.38% on these cases. A 6% MLP error is therefore not "near BF16" on this
metric, despite good cosine similarity or a small implementation residual.

About 1.068% of gate signs flip under quantization. ReLU's discontinuous
derivative turns those small gate perturbations into different gradient masks.
This explains why its backward is much worse despite similar forward error.
Correcting only backward GEMMs cannot restore the original mask.

Holding each backward GEMM's actual inputs fixed isolates its added quantization
error: approximately 3.37% for dH, 3.56% for dX, 3.57% for dW-gu and 2.38% for
dW-down. Total gradient error also includes upstream gate/product noise.

## Kernel implementation versus representation policy

Across the 54 cases, the worst native-kernel mismatch against the independent
quantized simulation is 0.0315% relative L2. The worst actual approximate-SiLU
reciprocal versus division mismatch is 0.0154%. These are substantially smaller
than the 6–12% BF16 discrepancy. The approximate reciprocal is not the main
cause on these inputs. This does not excuse failures outside this matrix;
the separate packed singleton fault remains documented in the stress report.

## Controlled precision interventions

Uniform [-0.5,0.5], SiLU. "Exact" means FP32 reference arithmetic over the
unquantized operand at that site, not a timed BF16 kernel implementation.

| Intervention | Output | dX | dW-gu | dW-down |
|---|---:|---:|---:|---:|
| Current policy | 6.004% | 6.361% | 6.370% | 6.002% |
| Exact gate/up projection only | 3.582% | 4.910% | 4.920% | 3.579% |
| Exact gate branch only | 4.907% | 5.480% | 5.482% | 4.903% |
| Exact hidden storage only | 5.391% | 6.361% | 6.370% | 6.012% |
| Exact backward GEMMs only | 6.004% | 4.050% | 4.054% | 5.509% |
| Residual-corrected gate/up only | 3.582% | 4.911% | 4.920% | 3.580% |
| Corrected projections, single-FP8 hidden in dW-down | 0.427% | 0.440% | 0.440% | 2.684% |
| Corrected projections and corrected hidden reconstruction | 0.417% | 0.440% | 0.440% | 0.467% |

The last two rows are **arithmetic counterfactuals**, not production-ready
autograd recipes. They approximate matrix products with three terms:

`A0 @ B0.T + Q(A-A0) @ B0.T + A0 @ Q(B-B0).T`, where `A0=Q(A)`.

The residual-residual product is omitted. These simulations execute FP32 GEMMs
over reconstructed FP8 operands; they do not benchmark three native FP8 MMAs.
Corrected hidden reconstruction adds a second quantized component. This changes
storage, quantization and recomputation requirements. A real implementation
must define and validate a consistent forward/backward numerical contract;
the ablations are not proof of that contract or of convergence.

ReLU remains more difficult: even the fully corrected simulation has about
1.38% dX/dW-gu error due to residual gate-mask sensitivity. Exact gate-only
arithmetic reduces its dX error from 11.85% to 5.46%, making gate fidelity an
especially promising intervention for ReLU.

Simply keeping the forward hidden tensor in higher precision does not fix
dW-down: that operand is quantized again along backward's different reduction
axis. The reconstructed hidden value used by dW-down must be addressed too.

## Options that did not address the main error

- Changing hypothetical E8M0 block size 32 to 16 is effectively unchanged on
  these bounded distributions. Standard MXFP8 hardware uses groups of 32;
  the size-16 experiment is not an available native configuration.
- Hypothetical arbitrary FP32 block scales make only modest improvements
  (SiLU output 6.00% to 5.83%) and can worsen dW-down. This is not the standard
  E4M3/E8M0 contract and is not a proposed drop-in scale change.
- Exact down weights alone only reduce SiLU output to about 5.51%.
- Exact backward GEMMs leave substantial inherited forward error.

An additional isolated dW reduction ladder uses M=128,512,2048,8192,32768 and
N=K=128. The reconstructed-FP8 FP32 GEMM differs from FP64 accumulation by at
most 1.482e-7 relative L2 (0.00001482%); FP8-versus-BF16 remains about 3.35–3.38%.
This is a PyTorch arithmetic diagnostic, not a counter measurement of native
split-K schedules. It shows why upgrading already-FP32 accumulation is a poor
first target compared with operand precision. Prior native long-reduction
coverage is in [the stress report](mxfp8_mlp_numerics.md).

## Recommended implementation order

Follow-up: the [independent recipe comparison](mxfp8_mlp_recipe_parity.md)
found essentially the same branch error with TorchAO conversion and separate
BF16 linear boundaries. Residual-stream errors at std=0.02 initialization are
much smaller. Before implementing the costly precision interventions below,
measure matched training behavior; these remain optional quality experiments,
not fixes justified by a demonstrated converter or accumulator bug.

1. Define accuracy targets separately for output and all gradients, relative to
   BF16; retain a stricter independent quantized-reference implementation gate.
2. Establish a higher-precision gate/up projection as a quality baseline while
   keeping the down and recompute interfaces comparable. Measure its actual
   runtime rather than assuming the current speedup survives.
3. Prototype selective residual correction at gate/up as a separate numerical
   policy. It captures the largest early error source and especially helps
   ReLU mask fidelity. Compare a mixed-precision gate branch as another policy.
4. If targeting sub-percent error, investigate correction of backward operands
   and the hidden reconstruction feeding dW-down, not just its FP32 accumulator.
5. Autotune legal schedules **within** a chosen accuracy policy; do not let a
   latency winner silently change the user-selected numerical contract.
6. Require model-level convergence and matched end-to-end timing before choosing
   a new default. Extra FP8 products may erase the performance gain; no speed
   result was collected for these hypothetical algorithms.

No kernel/default was changed in this investigation. The existing all-MXFP8
policy can remain a speed-oriented option without being described as BF16-like
numerical accuracy on bounded inputs.

## Reproduce and artifacts

```bash
python -m benchmarks.analyze_mlp_divergence \
  --output-dir autotune_reports/mlp/new_divergence
python -m benchmarks.analyze_mlp_divergence --reduction-ladder \
  --output-dir autotune_reports/mlp/new_reduction
```

Use new output directories. The complete stage metrics and counterfactuals
are flushed to `cases.jsonl`; policy definitions and environment are in
`manifest.json`. Local final evidence is in
`autotune_reports/mlp/divergence_bounded_v2/` and
`autotune_reports/mlp/divergence_reduction_v1/`. The earlier v1 run is exploratory
and is not counted again as independent evidence.
