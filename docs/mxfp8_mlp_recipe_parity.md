# MXFP8 MLP recipe parity

The independent TorchAO-quantized comparison does **not** identify a kernel
implementation error explaining the roughly 6% branch-level error. Fresh
high-precision sources for backward quantization and separate BF16 linear
boundaries produce essentially the same error against BF16 on this matrix.
This is numerical evidence, not a training-convergence or performance result.

## Method

`benchmarks/compare_mlp_recipes.py` uses TorchAO 0.18's actual `to_mx`
conversion with explicit RCEIL, E4M3, E8M0, and 32-element blocks. It reconstructs
quantized operands and multiplies them with FP32 GEMMs (TF32 disabled). This
tests quantization recipes independently of our converter; it does **not** run
Transformer Engine or native TorchAO MXFP8 GEMM kernels on RTX.

Five paths share identical BF16 inputs, weights, and upstream gradients:

- Native RTX fused MLP.
- TorchAO conversion reproducing the current fused numerical contract.
- The same reference, but dW-down quantizes freshly recomputed BF16 hidden
  values rather than the BF16 reconstruction of forward-quantized hidden values.
- Separate BF16-output linear operations with FP32 activation/product.
- Separate BF16-output linear operations with BF16 activation/product.

The separate-linear references quantize each backward reduction orientation
from its original high-precision source. Their activation/product gradients
come from PyTorch autograd. Native fused activation/product remain FP32.
The fresh-hidden intervention changes only dW-down, not forward or the other
gradients; a regression test verifies that isolation.

The matrix contains 72 cases: four `(M,K,H,O)` shapes
`(17,35,97,35)`, `(64,128,256,128)`, `(512,768,1536,768)`,
`(8192,128,256,128)`; SiLU/GELU/ReLU; two seeds; and uniform `[-0.5,0.5]`,
truncated-normal weight initialization, and outlier distributions.
Initialization uses BF16 weights with std=0.02 and truncation at ±0.04.

RTX 5070 Ti, Torch 2.13.0+cu132: all 72 cases passed the finite/implementation
gate. The largest native-versus-matched-reference relative L2 error was
0.0594%, across outputs and all three gradients. Three CUDA reference tests
also passed. The implementation gate is distinct from BF16 approximation error.

## Matched SiLU results

Relative L2 error versus BF16, in percent, averaged over two seeds for
`M=512,K=768,H=1536,O=768`, uniform `[-0.5,0.5]` inputs and weights:

| Recipe | Output | dX | dW gate/up | dW down |
|---|---:|---:|---:|---:|
| Native fused | 6.007 | 6.362 | 6.376 | 6.008 |
| Fresh hidden for dW-down | 6.007 | 6.362 | 6.376 | 6.015 |
| BF16 linear boundaries, FP32 activation | 6.020 | 6.362 | 6.375 | 6.016 |
| BF16 linear boundaries, BF16 activation | 6.024 | 6.363 | 6.374 | 6.020 |

Fresh hidden changes dW-down by about 1.65% relative to native, but does not
systematically improve its distance from BF16. That is a meaningful recipe
difference, not the explanation for the overall 6% gap. It must not be changed
silently as a scheduling optimization.

ReLU remains more sensitive in backward: native dX and dW gate/up errors are
about 11.92% and 11.94% on this shape. Separate BF16 boundaries do not fix that.
See the [stage-level divergence study](mxfp8_mlp_divergence.md) for gate-mask
sensitivity and operand-error accumulation.

## Branch versus residual stream

With std=0.02 initialized weights on the same SiLU shape:

| Quantity | Native relative L2 error versus BF16 |
|---|---:|
| MLP output | 6.323% |
| MLP dX | 6.336% |
| dW gate/up | 6.219% |
| dW down | 6.197% |
| BF16 `X + MLP(X)` | 0.241% |
| BF16 residual input gradient `dY + dX` | 0.296% |

The identity branch dominates the residual-stream norm at this initialization.
It does not reduce parameter-gradient errors, and this experiment does not
establish that the same ratio persists after training or through many layers.
With the much larger uniform random weights, the MLP branch dominates and
the residual output error remains about 6%.

## Decision

Keep the current kernel contract unchanged based on this experiment. There
is no measured benefit here from inserting BF16 activation boundaries or
silently replacing the backward hidden source. Next evaluate matched training
loss, updates, and residual-stream drift before paying for correction GEMMs.
If a sub-percent **isolated branch** target is required, the previous precision
interventions remain relevant, but their runtime cost needs measurement.

## Reproduce

```bash
python -m unittest tests.test_mlp_recipe_reference -v
python -m benchmarks.compare_mlp_recipes \
  --output-dir autotune_reports/mlp/new_recipe_parity
```

Use a fresh output directory. Each case is flushed to `cases.jsonl`, with
`environment.json` and `summary.json` alongside it. Local evidence is in
`autotune_reports/mlp/recipe_parity_v2/`; v1 is the same experiment before the
residual input-gradient metric was added and is not independent evidence.
