# Pretrained DINOv3 regression suite

RTX has an optional, local model-level regression suite built around Meta's pretrained
DINOv3 ViT-S/16. It exercises 48 transformer projection layers over twelve
blocks, making it substantially stronger than isolated GEMM checks: numeric
error can accumulate through attention, residual paths, normalization, and the
MLP before the final 384-dimensional representation is compared.

## Assets and licensing

The DINOv3 source is not distributed by RTX. For local development, place a
separately obtained checkout under `fp8dinov3/`; the complete directory is
ignored by Git. DINOv3 remains governed by Meta's license rather than RTX's
Apache-2.0 license.

Pretrained checkpoints are intentionally ignored by Git. By default the suite
looks for:

```text
fp8dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

The expected file is 83 MiB and has SHA-256
`08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d`.
Set `RTX_DINOV3_CHECKPOINT` to test the same checkpoint from another location.
The loader uses a strict state-dict load and verifies the architecture contract
in the test suite: 21,601,152 parameters, 12 blocks, and 48 transformer
linears.

## Why a bias adapter exists

RTX linear frontends deliberately do not implement bias. DINOv3's transformer
linears were nevertheless constructed with bias, so blindly converting them
must fail atomically. The regression harness instead builds a bias-free RTX
core and preserves the original bias and DINO QKV `bias_mask` in a thin model
adapter. Training conversion retains the exact original `Parameter` objects,
so existing optimizer references remain valid.

Moving the bias outside BF16 `F.linear` introduces one additional BF16 rounding
per projection. The suite therefore uses two references:

- the untouched pretrained BF16 model, measuring complete end-to-end drift;
- a bias-separated BF16 model, isolating quantization from the adapter's
  rounding floor.

On the RTX 5070 Ti validation run, the bias-separated reference differed from
the original by 0.0204 relative L2 while retaining 0.99979 mean cosine. Every
low-precision result is reported against both references.

## Coverage

`tests/test_dinov3_regression.py` covers:

- checkpoint identity, parameter inventory, projection shapes, and strict load;
- DINO QKV bias-mask behavior and learned MLP biases;
- atomic rejection when a user tries to convert biased linears directly;
- parameter identity through the training adapter;
- pretrained BF16 adapter fidelity;
- layer-by-layer and final-output MXFP8 PTQ drift at 224x224 resolution;
- NVFP4 current-scale and block-scale PTQ finiteness and numeric envelopes;
- repeatable eager PTQ execution;
- exact batch-decomposition invariance for BF16/MXFP8 and an explicit stability
  envelope for batch-sensitive tensor-current NVFP4 scaling;
- representation-geometry preservation over a 12-image cosine-similarity
  matrix;
- packed whole-model state-dict round trips with exact output preservation;
- one-patch, rectangular, transposed-rectangular, multi-batch, and ragged large
  image shapes, covering token-row counts far from native GEMM tile sizes;
- `torch.compile(fullgraph=True, dynamic=False)` for MXFP8 and NVFP4 PTQ at
  both minimal and full 224x224 resolution;
- compiled training backward for MXFP8 and delayed, tensor-current,
  JIT-region, and block-only NVFP4, including finite input, weight, and bias
  gradients;
- every dynamic NVFP4 scale policy under fullgraph: delayed, tensor-current,
  JIT-region, and block-only, plus both packed-current and packed-block PTQ;
- delayed-history rotation through a 512x activation-range shift, with exact
  eager/compiled outputs and device-resident history at every step.

The standalone harness adds TorchAO rowwise FP8 and all RTX training/PTQ modes:

```bash
python benchmarks/dinov3_regression.py \
  --device cuda:0 \
  --batch-size 2 \
  --image-size 224 \
  --output autotune_reports/dinov3_vits16_regression.json
```

Add `--compile` to compile each complete model, `--no-blocks` to omit the twelve
intermediate block comparisons, or `--variants ...` to select modes. Reports
record final relative L2, RMS error, maximum absolute error, per-sample cosine,
per-block drift, packed-weight bytes, checkpoint identity, and CUDA environment.
The reported first-call duration includes lazy kernel compilation and is a
diagnostic—not a steady-state performance benchmark.

The separate [W4A16/W4A8 study](dinov3_mixed_precision.md) isolates weight and
activation precision and records which candidate formats map to actual SM120
mixed-precision MMA instructions.

## Current RTX 5070 Ti reference observation

Using PyTorch 2.13.0+cu132, CUDA 13.2, batch 2, and 224x224 inputs, the initial
uncompiled validation produced the following final-output metrics against the
bias-separated BF16 model:

| Mode | Relative L2 | Mean cosine |
|---|---:|---:|
| TorchAO rowwise FP8 | 0.1196 | 0.99285 |
| RTX MXFP8 dynamic training path | 0.1894 | 0.98218 |
| RTX MXFP8 weight-only PTQ | 0.1894 | 0.98218 |
| RTX NVFP4 delayed training path | 0.4693 | 0.88902 |
| RTX NVFP4 tensor-current training path | 0.4693 | 0.88902 |
| RTX NVFP4 JIT-region training path | 0.4693 | 0.88902 |
| RTX NVFP4 block training path | 0.4547 | 0.89574 |
| RTX NVFP4 current-scale PTQ | 0.4289 | 0.90752 |
| RTX NVFP4 block PTQ | 0.4547 | 0.89574 |

These are regression observations for a model trained with rowwise FP8, not
claims of task accuracy or universal tolerance recommendations. Downstream
accuracy evaluation remains necessary before deployment.

Additional behavioral checks found MXFP8 PTQ exactly invariant when the same
three inputs were evaluated as one batch or separately. Tensor-current NVFP4
is intentionally batch-composition-sensitive because its current scale is
shared more broadly; the observed batched-versus-separate minimum cosine was
0.9297. At 224x224, compiled-versus-eager tensor-current NVFP4 produced 0.9332
cosine and 0.3654 relative L2. These explicit envelopes make future scaling or
lowering changes visible instead of reducing the test to a finiteness check.

Across the dynamic NVFP4 modes, tensor-current, JIT-region, and block-only were
bit-exact under batch decomposition for the tested inputs. Packed-block was
also exact. Delayed scaling retained 0.9726 minimum cosine because changing
from a three-image batch to single-image calls changes its stateful problem
history. Its 16-entry amax window rotated correctly through a 512x range shift,
and compiled execution matched eager outputs and history buffers exactly.
