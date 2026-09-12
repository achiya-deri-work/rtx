"""Traceable selective-precision MLP composition over registered linear ops.

No opaque launcher contains Torch pointwise operations. Dynamo sees the BF16
GEMM, activation, product and gradient concatenation; native linear dispatch
continues to own quantization and MXFP8 backward. Reentrant checkpointing owns
recomputation, so eager callers use backward(), not autograd.grad().
"""
from dataclasses import fields

import torch
from torch.utils.checkpoint import checkpoint


def prequant_config(config, down=False):
    from .fp8 import MXFP8PrequantConfig
    from .configs.mxfp8 import MXFP8GemmConfig
    gemm = config.down_gemm if down else config.gate_prequant.gemm
    return MXFP8PrequantConfig(
        quant=config.down_quant if down else config.gate_prequant.quant,
        weight_quant=config.down_quant if down else config.gate_prequant.resolved_weight_quant(),
        quant_launches="dual" if down else config.gate_prequant.quant_launches,
        gemm=MXFP8GemmConfig(**{f.name: getattr(gemm, f.name) for f in fields(MXFP8GemmConfig)}),
    )


@torch.compiler.assume_constant_result
def execution_keys(config,m,k,h,o):
    from .fp8 import _intern_prequant_config
    from .fp8_bwd import _intern_bwd_config, _default_bwd_config
    from .kernels.mxfp8 import MXFP8Problem
    up,down=prequant_config(config),prequant_config(config,True)
    gp,dp=MXFP8Problem(m,2*h,k),MXFP8Problem(m,o,h)
    reason=up.rejection(MXFP8Problem(m,h,k)) or down.rejection(dp)
    if reason:
        raise ValueError(f"BF16-gate schedule is unavailable: {reason}")
    return (_intern_prequant_config(up),_intern_prequant_config(down),
            _intern_bwd_config(config.gate_backward or _default_bwd_config(gp)),
            _intern_bwd_config(config.down_backward or _default_bwd_config(dp)))


class BF16GateProjection(torch.autograd.Function):
    """BF16 gate, MXFP8 up, and a single combined MXFP8 gate/up derivative."""
    @staticmethod
    def forward(ctx, x, weight, prequant, backward):
        from .fp8 import _run_prequant
        ctx.save_for_backward(x, weight)
        ctx.backward_config = backward
        gate, up = weight.chunk(2, 0)
        return (torch.nn.functional.linear(x, gate),
                _run_prequant(x, up, prequant))

    @staticmethod
    def backward(ctx, dg, du):
        from .fp8_bwd import _mxfp8_linear_bwd_op
        x, weight = ctx.saved_tensors
        dx, dw = _mxfp8_linear_bwd_op(torch.cat((dg, du), -1), x, weight,ctx.backward_config)
        return dx, dw, None, None


def recomputed_mlp(x, gu, down, activation, config, precision):
    from .fp8_bwd import _mxfp8_linear_train_op
    if precision!="bf16":
        up_key,down_key,gate_bwd,down_bwd=execution_keys(config,x.shape[0],x.shape[1],gu.shape[0]//2,down.shape[0])

    def body(x, gu, down):
        if precision == "bf16":
            gate, up = torch.nn.functional.linear(x, gu).chunk(2, -1)
        else:
            gate, up = BF16GateProjection.apply(x, gu, up_key, gate_bwd)
        hidden = (getattr(torch.nn.functional, activation)(gate.float()) * up.float()).bfloat16()
        if precision == "bf16":
            return torch.nn.functional.linear(hidden, down)
        return _mxfp8_linear_train_op(hidden,down,down_key,down_bwd)

    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, gu, down)):
        return checkpoint(body, x, gu, down, use_reentrant=True, preserve_rng_state=False)
    return body(x, gu, down)
