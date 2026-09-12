# Selective MLP precision study

## Findings

Protecting the gate is the strongest targeted intervention tested here for
backward accuracy, particularly ReLU. Protecting just the hidden operand has
a smaller effect and does not improve dX under the tested recipe. Neither
intervention makes the whole MLP sub-percent accurate against BF16.

On RTX 5070 Ti, 36 cases × seven policies were finite. Cases cover ragged
`(M,K,H,O)=(17,35,97,35)` and `(512,768,1536,768)`, three activations, three
seeds, and uniform bounded inputs/weights or std=0.02 initialized weights.
These are numerical recipe simulations, **not new production kernels, native
mixed-format tensor-core instructions, throughput measurements, or training
convergence evidence**.

## Controlled contracts

All recipes have BF16 linear outputs and hidden boundaries, FP32 activation
and multiplication, and independently oriented backward quantization from
saved BF16 sources. The baseline is therefore the separate-linear recipe,
not the production fused hidden-source contract; see
[recipe parity](mxfp8_mlp_recipe_parity.md) for that distinction.

The BF16 oracle uses actual BF16 GEMMs and PyTorch activation autograd.
Quantized products use TorchAO RCEIL conversion and reconstructed FP32 GEMMs.
TF32 and reduced-precision BF16 reductions are disabled. Reported errors are
relative L2 against the matched oracle; results must not be pooled with the
previous experiment's different seeds/tensor generation and fused boundaries.

- **MXFP8:** all products quantized. This already stores hidden as BF16,
  then requantizes it for the down projection. Storage alone is not precision
  preservation in that GEMM.
- **BF16 gate:** gate forward, dX contribution, and weight gradient use BF16
  GEMMs; up/down remain MXFP8.
- **Corrected gate:** each gate GEMM uses `Q(A)Q(B) + Q(A-Q(A))Q(B) +
  Q(A)Q(B-Q(B))`, omitting the residual-residual product. Three products per
  corrected GEMM; no performance advantage over BF16 is assumed.
- **BF16 hidden/mixed down:** forward preserves H but quantizes W-down;
  dW-down preserves H but quantizes dY. dH remains MXFP8. This is a diagnostic
  mixed-operand computation, not a claim that SM120 supports such an MMA.
- **BF16 down:** forward and both backward products use BF16.
- The two combined policies protect gate plus hidden, or gate plus down.

Separate gate/up paths add their BF16 input-gradient contributions, unlike a
single combined projection GEMM. Every policy shares this decomposition to
isolate precision effects. The BF16 recipe matches independent ordinary
PyTorch autograd exactly in the reference test for all three activations.

## Results

Mean percentages across three seeds, larger shape, uniform `[-0.5,0.5]`:

| Policy | SiLU output | SiLU dX | ReLU dX | ReLU dW-gate |
|---|---:|---:|---:|---:|
| MXFP8 | 6.023 | 6.384 | 11.983 | 15.868 |
| BF16 gate | 4.921 | 4.860 | 4.861 | 4.780 |
| Corrected gate | 4.924 | 4.862 | 5.079 | 5.213 |
| BF16 hidden/mixed down | 5.402 | 6.384 | 11.983 | 15.868 |
| BF16 down | 4.847 | 5.411 | 11.502 | 15.504 |
| BF16 gate + hidden | 4.135 | 4.860 | 4.861 | 4.780 |
| BF16 gate + down | 3.380 | 3.487 | 3.491 | 3.381 |

For initialized SiLU weights, MXFP8 output/dX errors are 6.337%/6.346%;
BF16 gate gives 5.212%/5.147%; BF16 gate+down gives 3.600%/3.701%.
The corresponding residual-output errors are 0.241%, 0.217%, and 0.178%.
Thus even substantial branch-error improvements can be small changes to the
initialized residual stream. No optimizer-step or trained-distribution claim
follows from this result.

The full JSONL reports output, dX, separate gate/up/down weight gradients,
residual output/input gradient, absolute errors, RMS, relative L2, and cosine.
Three tests cover the BF16 oracle, mixed-hidden gradient contract, and
three-product correction arithmetic. Finiteness of an experimental policy is
not an accuracy acceptance gate for production.

## Recommendation

Retain the production fast path. If adding a selectable quality policy, start
by measuring a BF16 gate implementation, especially for ReLU. Do not begin
with correction GEMMs or a BF16 hidden buffer alone: the former adds compute;
the latter still requantizes unless the consumer actually changes.
Before promoting any policy, measure matched training updates/loss and native
end-to-end runtime. No new public constructor option is exposed by this study.

## Reproduction

```bash
python -m unittest tests.test_mlp_precision_study -v
python -m benchmarks.study_mlp_precision \
  --output-dir autotune_reports/mlp/new_selective_precision
```

Use a fresh directory. Local evidence:
`autotune_reports/mlp/selective_precision_v1/` contains a manifest, flushed
per-case JSONL, and summary. Stack: Torch 2.13.0+cu132, TorchAO 0.18.
