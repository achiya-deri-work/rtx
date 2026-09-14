# Recomputed MXFP8 GLU MLP

This feature implements the strict two-device-kernel forward requested for
the GLU branch. **Qualification is in progress; this is not yet a production
performance recommendation.** A conventional separate quantize-then-GLU
forward is not silently substituted: it would violate the two-kernel contract.

## Public contract

See [coverage-first MLP autotuning](mxfp8_mlp_autotuning_coverage.md) for search
controls, benchmark contracts and how to audit what was actually explored.
For first-use effort, cache controls and the current qualification boundaries, see
[first-hit autotuning](first_hit_autotuning.md).

```python
import torch
import rtx

mlp = rtx.MXFP8MLP(768, 1536, device="cuda", activation="silu")
x = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
example_x = x.detach()
# Optional explicit whole-training tuning, before compilation/capture:
mlp.autotune_(example_x, max_trials=32, time_budget_s=120,
              output_dir="autotune_results/my_mlp")
compiled = torch.compile(mlp, fullgraph=True, dynamic=False)
y = compiled(x)
y.backward(torch.randn_like(y))
```

The functional API is
`rtx.mxfp8_mlp(x, gate_up_weight, down_weight, activation="silu", config=...)`.
X and both weights are contiguous BF16 tensors on SM12x. X is `[..., M, K]`;
weights are `[2H,K]` (gate first, up second) and `[O,H]`. Output is BF16
`[...,M,O]`. O defaults to K in the module. SiLU, exact-erf GELU, and ReLU
are supported. Bias, dropout, residual addition, and higher-order derivatives
are outside this operator. Both parameters use truncated-normal initialization
with std 0.02 and bounds ±0.04. State dicts contain the two BF16 parameters;
tuning configs are explicit hardware/workload policy, not weight tensors.

### Converting an existing GLU MLP

```python
# Source projections implement down(silu(gate(x)) * up(x)).
mlp = rtx.MXFP8MLP.from_linears(
    old_mlp.gate_proj, old_mlp.up_proj, old_mlp.down_proj,
    activation="silu",
)
# Replace the source module in your model, then create a NEW optimizer.
```

This copies weights, not optimizer state or hooks, and leaves the source
modules untouched. All three sources must be bias-free BF16 `nn.Linear`s
on one device with matching training modes. Gate/up become a single packed
parameter, so they must have matching `requires_grad`; down can be frozen
independently. Arbitrary model graphs, adapters, custom forward behavior,
residuals and dropout are not inferred or converted. Exact GELU is supported,
not its tanh approximation. CPU construction/conversion is supported for
loading; actual execution requires an SM12x CUDA GPU.
Inputs must have rank at least two and nonzero dimensions. Ragged dimensions
are supported through logical padding; noncontiguous input/weight tensors are
rejected rather than copied inside the launcher. Noncontiguous upstream
gradients are handled by the compiler-visible backward graph.

### Checkpoints and tuning policies

```python
from pathlib import Path

torch.save(mlp.state_dict(), "mlp_weights.pt")
selected_config, precision = mlp.policy_for(example_x)
Path("mlp_policy.json").write_text(selected_config.to_json())
policy = rtx.MXFP8MLPConfig.from_json(Path("mlp_policy.json").read_text())
restored = rtx.MXFP8MLP(768, 1536, device="cuda", config=policy, precision=precision)
restored.load_state_dict(torch.load("mlp_weights.pt", weights_only=True))
```

Save architecture/activation alongside the checkpoint in your application.
JSON policies reject stale revisions and unknown top-level fields. A raw
policy is an explicit selection, **not** a device/shape-scoped runtime cache:
retune on another GPU or workload. The benchmark preset loader additionally
checks its device/shape metadata. `autotune_` installs a schedule for the
example shape/device/dtype only, defaults to total training time with all three gradients, and accepts
`phase="forward"` for inference. It never runs from `forward`, including on a
new shape. `eval()` changes neither quantization nor parameter mutability;
use `torch.inference_mode()` when no backward is needed.

### Explicit numerical policies and shape dispatch

`precision="mxfp8"` remains the default native fused path. `precision="bf16_gate"`
uses a BF16 gate forward GEMM, MXFP8 up/down, and combined MXFP8 gate/up
backward. Activation/product are FP32, with BF16 projection/hidden boundaries.
This quality recipe materializes transient BF16 tensors; it is **not** the
two-kernel forward described below. `precision="bf16"` is an explicit BF16
projection fallback with the same FP32 activation/product boundaries.
Autotuning never changes precision or the native SiLU arithmetic policy.

```python
mlp = rtx.MXFP8MLP(768, 1536, device="cuda", precision="bf16_gate")
mlp.autotune_(example_x, max_trials=32, time_budget_s=120)
# Optional user-authorized small-shape fallback; not an implicit heuristic:
small_x = torch.empty(8, 768, device="cuda", dtype=torch.bfloat16)
mlp.configure_shape_(small_x, precision="bf16")
compiled = torch.compile(mlp, fullgraph=True, dynamic=False)
```

The map retains at most 128 entries, oldest first, and holds no tensors. Unseen
shapes retain constructor settings. Entries are process-local, not state_dict
state; reinstall on another device and recompile/recapture after mutation.
Use `policy_for(example_x)` to inspect the effective configuration and precision.
Persist architecture, activation and precision separately from configuration JSON.

Hybrid/BF16 recomputation uses `checkpoint(use_reentrant=True)`: eager training
supports `loss.backward()`, but not `autograd.grad()` or backward's `inputs=`.
Compiled fullgraph first-order gradients are tested. Higher-order gradients,
export and dynamic=True are not qualified for these added precision paths.
The hybrid uses `gate_prequant` quantizer/GEMM fields for up, `down_quant` and
`down_gemm` for down, and independent `gate_backward`/`down_backward` schedules.
Native-only GLU/recompute fields do not alter its numerical recipe. The MLP
tuner removes inactive native axes for hybrid search and exposes shape-relative
`dw_split_count` independently for both backward projections.

## Native on-chip forward: exactly two device kernels

1. Quantize BF16 X/W on chip to E4M3/E8M0. Compute FP32 gate/up accumulators,
   then `activation(gate)*up` in FP32. Quantize that product directly to
   MXFP8 along H and write only H-width data plus scale bytes to GMEM.
2. Load those MXFP8 data/scales directly, dynamically quantize the BF16 down
   weight on chip, accumulate the down GEMM in FP32, and store BF16 output.

There is one **transient** MXFP8 hidden tensor and its scales between the two
kernels. It is impossible to communicate this result through GMEM without
such storage. There is no 2H projection tensor, BF16 hidden tensor, separate
activation/quantization kernel, or retained hidden tensor in forward.

The optional `down_backend="producer_quantized"` quantizes down weights once
inside kernel 1 and uses the packed GEMM in kernel 2. It allocates transient
packed down weights/scales, refreshed every invocation. It does not cache
weights across optimizer updates.

Experimental `gate_backend="cooperative_prequant"` instead quantizes X, gate/up
weights, and down weights once across a cooperatively launched grid. A grid
barrier precedes packed GEMM tiles within that same kernel. This requires
`down_backend="producer_quantized"`, K divisible by 32, complete M/N output
tiles (partial tiles are rejected after a ragged hang), and legal resident-grid
resources. It has **two compute kernels plus one CUDA memset graph node** to
initialize a private barrier counter; it is not a two-device-operation path.
It adds transient packed operands and is not enabled by default. All setup
costs are included in end-to-end tuning. No eager Torch numerical operations
are used in these launchers.

Autograd saves **only X, gate_up_weight, and down_weight**, by reference through
`save_for_backward`. This preserves normal PyTorch version-counter checks.
The hidden data and scales are not returned as autograd auxiliaries, saved in
context, or used by backward. Allocator/graph pools may retain storage capacity;
that is different from keeping a saved activation tensor.

## Backward: full recomputation

Let `G,U = MX(X,W_gu)`, `A=activation(G)`, and `Hq=Q(A*U)`. Quantization uses
the library's straight-through convention: block-scale selection and rounding
do not have differentiable scale parameters. Each backward GEMM independently
quantizes along its own reduction dimension; forward scale blocks cannot just
be transposed and reused for a different reduction axis.

The default `backward_recompute="joint"` schedule is:

1. Compute `dH = MX(dY, W_down.T)`. This needs no hidden activation.
2. Recompute G/U once on chip, and immediately form
   `dG=dH*U*activation'(G)` and `dU=dH*activation(G)` in FP32. Store their BF16
   derivatives in packed gate/up order. In the same kernel, quantize `A*U`
   exactly as in forward, decode the rounded MXFP8 values, and emit a temporary
   BF16 Hq for dW-down. Do not store G/U themselves. Derivatives and Hq occupy
   independently contiguous, 16-byte-aligned regions of one allocation.
3. Compute `dW_down = MX(dY.T, Hq.T)`,
   `dX = MX([dG,dU], W_gu.T)`, and
   `dW_gu = MX([dG,dU].T, X.T)`, as needed.

The alternative `backward_recompute="two_pass"` first recomputes packed Hq,
decodes it in a separate kernel, computes the down backward, and recomputes
G/U again to form the GLU derivatives. This recomputes gate/up **twice in
backward**, but retains the existing jointly scheduled down-backward path.
Both schedules are autotuner choices; neither retains forward hidden state.
Independently, `recompute_backend="prequant"` quantizes the recomputed gate/up
operands once and runs a packed GEMM with the same fused derivative epilogue.
This adds a backward quantizer launch and permits a separate recompute GEMM
configuration. `backward_overlap=True` overlaps dW-down with the independent
gate/up backward branch when all three gradients are requested; selective
partial-gradient paths retain their existing dependency-aware schedule.
Backward temporaries exist only during backward. dW reductions use the existing
FP32-accumulating MXFP8 backward implementations, including their split-reduction
choices, and round gradients to BF16 at their public boundary. A mathematically
exact FP32 gradient of the quantized forward is not claimed; the operational
reference includes these independently quantized backward GEMMs.

Partial gradients skip unneeded GEMMs. The joint path avoids the first hidden
recomputation for input-only gradients, although its derivative epilogue still
emits the unused hidden values. Down-weight-only gradients use only hidden
recomputation and dW-down. Noncontiguous dY is made contiguous in
the compiler-visible backward graph, not inside a hidden eager helper.

## Search space and compilation

`MXFP8MLPConfig` exposes `gate_up`, `down`, `gate_backward`, `down_backward`,
`backward_recompute`, `backward_overlap`, `recompute_backend`, and `recompute`.
Forward alternatives have independent `gate_backend`, `gate_prequant`,
`cooperative_ctas_per_sm`, `down_backend`, `down_quant`, and `down_gemm` policies.
Gate/up inherits the CTA-local GLU choices. Down currently supports scalar
transport with 16/32/64/128-bit weight loads and direct packed activation
staging; pretending TMA/cp.async already supports
the mixed input types would be incorrect. Other implemented projection
coordinates remain searchable. Both full backward config spaces are available.
Illegal combinations are rejected before compilation.

Derivative transport is also searchable: `glu_grad_load` selects interleaved
loads, register prefetch, or cooperative shared-memory staging;
`glu_grad_store` selects direct or coalesced exchange stores. Shared-gradient
loads cannot overlap the store-exchange scratch lifetime. The two-pass or
fragment-register paths use direct transport where the exchange workspace
does not exist. Geometry and transport winners are measured, not assumed.

`gate_up.glu_silu_math` controls SiLU arithmetic consistently across forward
and all recompute backends: `division` retains the original evaluation;
`shared_reciprocal` shares a correctly rounded FP32 reciprocal;
`approx_reciprocal` uses native `rcp.approx.f32` without flush-to-zero. The last
option is an explicit approximation, not an accuracy-neutral rewrite. GELU
and ReLU are unchanged. The tuner applies its independent numerical gate to
each choice. `glu_hidden_decode="bf16x2"` uses native packed reconstruction
instead of separate FP32 decode/multiply operations; exhaustive finite
E4M3/E8M0 tests verify the final BF16 result, including extreme scales.

Policies can be saved using `config.to_dict()` and restored with
`rtx.MXFP8MLPConfig.from_dict(...)`. Stale schema revisions are rejected.
Policies do not belong in model tensor state and are not portable performance
guarantees across shapes or GPUs.

## Reproduce the target-shape result

See [historical measurements and qualification](mxfp8_mlp_history.md).

## Qualification and remaining release gates

See [historical measurements and qualification](mxfp8_mlp_history.md).
