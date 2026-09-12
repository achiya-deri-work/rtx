# Training integration

## Choose what to replace

Start from a working BF16 model and checkpoint. Replace selected bias-free
projections, keep the rest of the model's numerical recipe unchanged, and
compare convergence before broadening conversion. RTX supplies linear GEMMs,
not an attention, normalization, optimizer, or distributed-training framework.

Dynamic `MXFP8Linear` and `NVFP4Linear` keep ordinary BF16 `weight` parameters.
MXFP8 uses MXFP8 forward/backward; NVFP4 uses NVFP4 forward and MXFP8 backward.
Input, output, and returned parameter gradients are BF16. Float32 weights under
autocast are not a substitute for this BF16 parameter contract.

Use [top-level training conversion](model_conversion.md) to retain existing
parameter objects, ties and optimizer references. Dynamic linear constructors
use ordinary no-bias `nn.Linear`-style Kaiming initialization. They do not
reinitialize imported parameters. The recomputed MLP is different: its new
weights use truncated normal std=0.02, and its `from_linears` creates new
parameters. Construct a new optimizer after that MLP conversion.

## Optimizer and precision policy

RTX does not create an optimizer or choose its state dtype. Our production
comparison policy uses BF16 model tensors with FP32 optimizer moments/master
weights; the BF16-state optimizer is a separately labelled throughput ablation.
Using `torch.optim.AdamW` directly on BF16 parameters is not a promise of FP32
master weights. Choose the optimizer implementation deliberately.

Keep sensitive reductions, normalization statistics, RoPE, softmax and loss
arithmetic in FP32 as required by the surrounding model. QK normalization and
high-precision model input/output projections are useful parts of the provided
decoder recipe, not features silently inserted by linear conversion.

## Compilation and warmup

Convert the model and select policies before constructing the compiled callable:

```python
compiled = torch.compile(model, fullgraph=True, dynamic=False)
```

Linear `autotune="balanced"` can resolve a new runtime context during first
execution. It is not free: warm all production shapes before measurement and
manual CUDA graph capture. Use `autotune="cache"` when runtime searches are
unacceptable. MLP tuning is explicit via `autotune_`, before compilation.
See [execution](execution.md) for capture, streams, and cache lifetimes.

Ordinary autograd accumulates returned gradients in parameter `.grad` buffers.
Gradient accumulation therefore follows the surrounding BF16/optimizer policy;
FP32 dW accumulation inside a GEMM does not make repeated `.grad` accumulation
FP32. Higher-order derivatives are not a promised native operator contract.

## Delayed NVFP4

Use the module for `scaling="delayed"`; the functional API has no history
owner. The first call bootstraps, later calls observe/update history, and
checkpoint loading resets it. `.eval()` and no-grad do not disable delayed
scaling. Do not assume a resumed delayed run is bitwise identical across the
history reset. Compare identical warmup and history policies in experiments.

## Recomputed GLU MLP

`MXFP8MLP` computes `down(activation(gate(X))*up(X))`, quantizing the hidden
product directly to MXFP8. Its forward uses two compute kernels in the normal
on-chip path; backward recomputes hidden intermediates. Autograd still retains
the original input and weight references. There is transient hidden storage
between forward kernels, but no hidden tensor saved for backward.

It already recomputes; wrapping it in another checkpoint normally adds work.
For a matched benchmark, checkpoint the whole BF16 comparator with
`use_reentrant=True`, and report ordinary BF16 separately. Standalone
`MXFP8GLU` is forward-only and must not replace a trainable GLU without a
backward implementation. Read [the MLP guide](mxfp8_mlp.md), especially its
known packed singleton failure and explicit numerical-policy options.

## Reference decoder

`rtx.models.DecoderOnlyTransformer`, `DecoderConfig`, and `LinearSpec` provide
a convergence scaffold, not a pretrained model API. It includes QK norm,
FP32-reduction RMSNorm/RoPE, pre/post sublayer norms, tied BF16 vocabulary
weights, and a configurable internal projection precision. Residual attention
output and MLP down projections initialize with
`std=0.02/sqrt(2*current_depth)`, with one-based depth; other projections retain
the base truncated-normal initialization.

Use [the benchmark guide](../benchmarks/README.md) and
`python benchmarks/train_decoder.py --help` for the exact dataset, resume,
optimizer and token-budget controls. Keep initialization, data order, schedule,
effective batch and optimizer precision identical between formats. A toy loss
decrease is a smoke test, not transformer convergence qualification.

## Distributed boundary

The kernels run locally and perform no collectives. BF16 parameters/gradients
can participate in surrounding synchronization, but FSDP, DTensor, packed
tensor-parallel redistribution and distributed packed-checkpoint conversion
are not qualified here. See [the runtime contract](runtime_contract.md).
