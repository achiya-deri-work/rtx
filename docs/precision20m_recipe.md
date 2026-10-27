# Approximately 20M-parameter transformer precision recipe

This is a source audit and one attention smoke check, not a training result.
The proposed controlled workload is a TinyStories byte-level causal LM with
8 layers, width 512, SwiGLU hidden width 1024, 8 attention heads of width 64,
sequence length 256, and batch size 16. The flattened projection batch is 4096;
all residual-branch linear dimensions are divisible by 128.

With bias-free SwiGLU blocks, the residual branches have 20,971,520 weights.
The instantiated configuration has **21,121,024 parameters**: vocabulary 257,
tied BF16 embedding/head, four RMSNorms per block, QK normalization and a final
RMSNorm. This count was verified by constructing the actual model on the meta
device. There are 32 quantized residual-branch linear modules.

## Matched precision boundary

The three runs use MXFP8 forward/backward, NVFP4 forward/MXFP8 backward, and
NVFP4 forward/experimental MS-EDEN FP4 backward. Only residual-branch projections
change: attention QKV/output and SwiGLU up/gate/down. Embedding, output
head, and stored activations remain BF16. Use identical architecture, inputs,
initial weights, optimizer settings, backend, validation data, and step/token
budgets. Record step-matched and wall-time-matched results separately.

Norm calculations, RoPE arithmetic, cross entropy, gradient norm, and scalar
diagnostic reductions use FP32. Cast normalized/rotated activations back to
BF16 at the documented projection/attention boundaries. Retain the project's
BF16 AdamW parameter/gradient/moment recipe for matrix and embedding parameters;
normalization parameters, their gradients and moments are FP32. No separate
FP32 master-weight copy is introduced. Audit these dtypes during training.
No claim here that all
intermediate attention arithmetic or stored probabilities are FP32.

## Verified native FLASH SDPA precision

The installed Torch is `2.13.0+cu132`, git
`cf30153c4c131c8164ee7798e5022d810682e2cb`. Local
`torch/nn/attention/_registry.py` identifies default native FA2; the 5070
process reported `_FLASH_ATTENTION_ACTIVE=None`. Installed wheel headers
expose the flash API but omit the implementation headers, so the following
read-only audit used official source at the exact installed commit, not an
assumed generic FlashAttention implementation:

1. PyTorch's exact [third-party entry](https://github.com/pytorch/pytorch/tree/cf30153c4c131c8164ee7798e5022d810682e2cb/third_party/flash-attention)
   pins FlashAttention `6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`.
   [ATen CMake lines 196–197 and 350–356](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/CMakeLists.txt#L196)
   build/include that submodule's FA2 source.
2. [kernel_traits.h lines 26–33](https://github.com/Dao-AILab/flash-attention/blob/6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15/csrc/flash_attn/src/kernel_traits.h#L26)
   sets `ElementAccum=float` and selects BF16-input/FP32-accumulating MMA.
3. [softmax.h lines 131–180](https://github.com/Dao-AILab/flash-attention/blob/6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15/csrc/flash_attn/src/softmax.h#L131)
   stores row maxima/sums in float, uses FP32 max/sum and `exp2f`, and normalizes
   with FP32 sums/logarithms.
4. [flash_fwd_kernel.h lines 343–347](https://github.com/Dao-AILab/flash-attention/blob/6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15/csrc/flash_attn/src/flash_fwd_kernel.h#L343)
   performs softmax on FP32 score accumulators, then converts probabilities to
   the BF16 operand type before the probability/value MMA.

Thus BF16 Q/K/V with native FLASH SDPA satisfies the intended FP32 softmax
arithmetic/statistics boundary while retaining efficient BF16 attention GEMMs.
It is not equivalent to retaining the entire probability matrix in FP32.
There is no need to force all attention GEMMs to FP32 to obtain FP32 softmax.

### Executed 5070 smoke

On 2026-10-25, after verifying no other compute applications, the existing
`/home/achiya/vs_code_main/rtx/.venv/bin/python` ran one small forward/backward:
BF16 Q/K/V shape `(2,8,256,64)`, causal, dropout zero, seed 20261025, under
`sdpa_kernel(SDPBackend.FLASH_ATTENTION)` with no fallback backend enabled.
Output and all Q/K/V gradients were finite. Output versus explicit FP32
QK/softmax/PV had NRMS 0.0019482359 and maximum absolute error 0.0080816746.
This verifies actual native FLASH execution on this SM120 laptop/shape, not
the complete training shape or a 5060 run. No power/clock setting was changed.

The training harness should force and record the same backend in every mode.
If a target rejects FLASH, fail explicitly or select the same documented strict
math fallback across all modes; do not silently give different modes different
attention implementations. A full training-shape smoke remains necessary.

## Correct FP32 gradient norm without full gradient copies

The installed `torch/nn/utils/clip_grad.py` calls `_foreach_norm` (or
`vector_norm`) without a dtype override, then stacks those norms and reduces
again without promotion. A local CPU check confirmed BF16 input returns BF16
scalar norms for both APIs. Converting those already-rounded norms to FP32
afterward does not restore their precision.

For dense gradients on one device, use:

```python
grads = [p.grad for p in parameters if p.grad is not None]
norms = [torch.linalg.vector_norm(g, ord=2, dtype=torch.float32) for g in grads]
total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
torch._foreach_mul_(grads, coefficient)
```

Guard the empty list and check nonfinite loss/norm before updating parameters.
For mixed gradient dtypes/devices, group scaling appropriately; this recipe
does not cover sparse gradients. The final stored gradients remain their
original dtype, as intended, but the norm and clipping coefficient are FP32.

The exact [CUDA norm source lines 40–44](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/ReduceNormKernel.cu#L40)
implements BF16 input with FP32 accumulator/output for a float dtype override,
explicitly fusing input conversion and reduction in one kernel. Local installed
`ATen/native/SharedReduceOps.h` lines 366–383 show squaring, summation, and root
performed in the accumulator type. This avoids allocating FP32 copies of all
gradients merely to measure their norm. If the optimizer already owns FP32
master-gradient buffers, calculate and apply clipping to those buffers instead
and keep the contract identical across precision modes.
