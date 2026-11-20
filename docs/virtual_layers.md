# Virtual layers: separate optimizer matrices, contiguous physical weights

`rtx.VirtualLayer` wraps named, bias-free projection modules. Their existing
2D `weight` Parameters remain the optimizer's parameters, but the weight data
is placed into one contiguous allocation. `layer.weight` exposes that allocation
as one differentiable Tensor view for a fused or batched kernel.

This lets Muon treat Q, K and V—or each fine-grained expert projection—as
separate matrices without a `cat`, `stack`, pointer table or weight copy on
each forward. Backward splits the packed gradient into the logical matrices.
The packed Tensor is **not** an extra trainable Parameter.

## QKV, including grouped-query attention

```python
import torch
from torch import nn
import rtx

device, dtype = 'cuda', torch.bfloat16
d, kv = 768, 256
projections = {
    'q': nn.Linear(d, d, bias=False, device=device, dtype=dtype),
    'k': nn.Linear(d, kv, bias=False, device=device, dtype=dtype),
    'v': nn.Linear(d, kv, bias=False, device=device, dtype=dtype),
}
# Or pass your already-initialized projections. Wrapping never reinitializes.
for projection in projections.values():
    nn.init.trunc_normal_(projection.weight, std=0.02)
qkv = rtx.VirtualLayer(projections)
optimizer = torch.optim.Muon(qkv.parameters(), lr=0.003)

def forward(x):
    packed_output = rtx.mxfp8_linear(x, qkv.weight, autotune=False)
    return packed_output.split((d, kv, kv), dim=-1)

compiled = torch.compile(forward, fullgraph=True, dynamic=False)
```

The optimizer sees `projections.q.weight` `[768,768]`,
`projections.k.weight` `[256,768]`, and `projections.v.weight` `[256,768]`.
The kernel receives one contiguous `[1280,768]` weight. Autotuning belongs to
the normal RTX functional API; this example disables it to keep the storage
example focused. NVFP4 works the same way with `rtx.nvfp4_linear`.

`qkv['q']` is the original projection module and remains individually callable.
`qkv(x)` is a **Torch dense linear reference**, not an automatically selected
RTX backend. Use the RTX functional API explicitly for low-precision execution.
Do not register the same projections elsewhere in the model as additional
owners: that can produce duplicate checkpoint paths.

For ordinary BF16 QKV, no low-precision kernel is required:

```python
compiled_qkv = torch.compile(qkv, fullgraph=True, dynamic=False)
q, k, v = compiled_qkv(x).split((d, kv, kv), dim=-1)
```

This executes one dense projection while preserving three independently
optimized matrices. Weights, activations and gradients remain BF16 when the
model/input are BF16; packing itself introduces no dtype conversion.

## SwiGLU gate/up

Gate and up can share one physical projection while remaining two independent
optimizer matrices, exactly like QKV:

```python
D, H = 768, 2048  # H is the post-gating hidden width
gate_up = rtx.VirtualLayer({
    'gate': nn.Linear(D, H, bias=False, device=device, dtype=dtype),
    'up': nn.Linear(D, H, bias=False, device=device, dtype=dtype),
})
down = nn.Linear(H, D, bias=False, device=device, dtype=dtype)
optimizer = torch.optim.Muon(
    [*gate_up.parameters(), *down.parameters()], lr=0.003,
)

def swiglu(x):
    return rtx.mxfp8_mlp(x, gate_up.weight, down.weight, activation='silu')

compiled_swiglu = torch.compile(swiglu, fullgraph=True, dynamic=False)
```

The fused/recomputed MXFP8 MLP reads one `[2H,D]` packed weight; Muon updates
gate `[H,D]`, up `[H,D]` and down `[D,H]` separately. Insertion order must be
**gate, then up**, matching the MLP's `[gate; up]` contract. Existing initialized
gate/up linears can be passed directly instead of constructing new ones.

For plain Torch/BF16, use `gate, up = gate_up(x).chunk(2, dim=-1)`, followed
by `down(silu(gate) * up)` under your chosen activation precision recipe.
Packing is the same and does not choose the activation or quantization policy.

## Expert banks

Packing order is the mapping's insertion order. `packed_shape` can reinterpret
the same elements as an expert-major bank. Gate and up can themselves be
separate optimizer matrices:

```python
E, D, H = 128, 768, 128
gate_up = rtx.VirtualLayer({
    f'expert_{e}_{part}': nn.Linear(D, H, bias=False, device=device, dtype=dtype)
    for e in range(E) for part in ('gate', 'up')
}, packed_shape=(E, 2 * H, D))
down = rtx.VirtualLayer({
    f'expert_{e}': nn.Linear(H, D, bias=False, device=device, dtype=dtype)
    for e in range(E)
}, packed_shape=(E, D, H))
```

For large banks, construct the child linears on `device='meta'`, wrap them,
then call `bank.to_empty(device=device)` and run your initialization on its
logical weights. This materializes one packed allocation rather than keeping
all old device allocations alive while initially packing a second copy.

Muon sees `2E` separate `[H,D]` gate/up matrices and `E` separate `[D,H]`
down matrices. Future grouped/routed kernels can consume the 3D `weight`
views directly. This does **not** implement routing, quantization, a grouped
RTX GEMM, or the constrained MoE itself. The dense linear APIs still accept
2D weights, not a 3D expert bank. `gate_up(expert_inputs)` is a Torch batched
projection for `[E,C,D]` input; it is useful as a reference, not a routed kernel.

For equal input widths, omitted `packed_shape` defaults to concatenated output
rows. Mixed input widths default to a flat allocation for a custom consumer.
An explicit shape must contain exactly the total weight elements. There is
no extra alignment padding; consumers retain their own kernel legality rules.

## Autograd and precision

Simply reading an oversized view from the first Parameter would send gradients
only to that Parameter. The virtual view instead has an explicit autograd
mapping to **every** logical weight. Frozen projections, individual child calls,
gradient accumulation and checkpointed recomputation remain supported.

Packing is precision-neutral: weights are copied exactly once at setup; no
activation, scale, weight or gradient quantization policy is introduced.
The consumer determines accumulation and result dtypes. Backward may need a
contiguous gradient temporary if the consumer supplies a noncontiguous packed
gradient; ordinary per-Parameter gradient accumulation can also allocate.
There is no promise that `.grad` tensors share one allocation.

One fused low-precision GEMM is not necessarily numerically identical to
separate low-precision calls: quantization blocks, NVFP4 outer-scale regions
and reduction order follow the **consumer's physical matrix contract**.
Logical optimizer boundaries do not automatically create quantization
boundaries. Expert-aware quantization must be defined by the future grouped
kernel; this tool does not silently share or split its scales.

There is no new dispatcher/custom-kernel boundary. The view's autograd mapping
is traced by AOTAutograd; default-mode Inductor compiles the surrounding model.
There are no library-owned execution graphs or user-managed scratch buffers.

The owner retains a **non-persistent full-span storage alias** so compiler
benchmark workers see the complete physical allocation, not only the first
small logical Parameter. This is metadata over the same storage: no extra
GPU allocation, forward copy, optimizer Parameter or state-dict entry.
The alias is refreshed with packing, moves, deepcopy and assign-loading.
This also supports `torch.compile` max-autotuning in subprocesses; older
first-Parameter `as_strided` exposure could give those workers undersized
scratch allocations. See the [GPU0 incident](balanced_moe_gpu0_incident_20261118.md).

## Initialization, moves and checkpoints

- Construct the virtual owner **before** optimizer creation, compilation and
  graph capture. Original module/Parameter identities and initialization are
  preserved by construction. Keep your own reset/initialization policy.
- Move/cast the owner with `.to(...)`, `.cuda()`, `.cpu()`, `.bfloat16()`, etc.
  It transforms the packed allocation once and preserves each logical view.
  Parameter replacement follows normal PyTorch conversion semantics; create
  optimizers after materializing/moving the model.
- State dicts contain only logical projection weights—no duplicate packed
  buffer. Both copy loading and `load_state_dict(assign=True)` restore packing,
  including when loaded through a parent module or from a meta-device model.
  With `assign=True`, follow PyTorch's optimizer-after-loading rule.
- `.to_empty(...)` preserves packing but leaves values uninitialized, just as
  for ordinary modules. Load or initialize before execution.
- `copy.deepcopy` creates a separate allocation and restores its internal
  packing. Checkpoint loading never changes the declared logical shapes.
- `validate()` explicitly checks packing without GPU synchronization.

## Structural changes and limits

Each child must own only its ordinary contiguous 2D floating `weight` Parameter
and no bias. Different devices/dtypes and tied or overlapping logical weights
are rejected. Parameters may have different `requires_grad` settings.

Do not move children independently, replace their storage with `.data`/`set_`,
or mutate structure during compiled execution. If replacing a weight deliberately,
call `repack_()` outside live autograd graphs/compilation/capture, then recreate
affected optimizers and compiled/captured functions. Names, order and logical
shapes are fixed; construct a new owner to change those. Identity/shape guards
catch weight replacement during tracing, while eager `weight`/`validate()` also
check physical pointers and offsets. Pointer checks do not run inside compiled
graphs: direct storage mutation there is unsupported, not silently repaired.

Distributed sharding/flattening, external parametrizations, tensor subclasses,
`torch.func`/vmap, export and concurrent mutation are not qualified. This
foundation must be requalified alongside the eventual grouped quantized
expert kernels.

## Qualification

`tests/test_virtual.py` compares outputs and every input/weight gradient with
independent ordinary projections, checks separate SGD/AdamW/Muon updates,
storage aliases, checkpointing, partial freezing, dtype moves, meta loading,
deepcopy, gradcheck, fullgraph AOTAutograd and actual Inductor. CUDA cases use
existing MXFP8 and NVFP4 functional kernels, compare all leaf gradients against
an independently owned packed Parameter, and replay after changing inputs and
weights. These are integration checks, not new kernel-performance or MoE
training-quality claims.

Qualification on 2026-11-15: Torch 2.13.0+cu132 local frontend/documentation
regression **61 passed, 2 CUDA skips**; Torch 2.12.1+cu132 virtual/documentation
regression **33 passed, 2 CUDA skips**. On the RTX 5070 laptop with Torch 2.13
and CuTe 4.7, the complete virtual-layer suite **29 passed**, including both
native precisions and graph replay. The GPU run reports a PyTorch
AccumulateGrad stream-mismatch warning in the side-stream capture test; all
replays and exact gradient comparisons pass. This does not qualify distributed
or arbitrary concurrent-stream execution.

Standalone QKV/SwiGLU release checks on 2026-11-15 used an isolated tree with
no BalancedMoE/router changes: **35 passed, 3 CUDA skips** on each of Torch
2.12.1 and 2.13.0. Focused RTX 5090 checks **2 passed**: compiled MXFP8 QKV
with changed-input/weight graph replay, and the recomputed MXFP8 SwiGLU MLP.
Both compared every logical weight gradient with an independent packed
Parameter; the SwiGLU check also compared its down-projection gradient exactly.
The same documented side-stream AccumulateGrad warning remains in the QKV test.
