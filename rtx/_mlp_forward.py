"""Compiler-visible storage for the prepared ordinary MXFP8 MLP forward.

Torch owns the output and typed scratch. Scales stay UINT8 across the opaque
boundary; CuTe launch adapters interpret those bytes as E8M0 without a copy,
extra launch, or Torch dtype alias.
"""
from dataclasses import replace

import torch


@torch.library.custom_op('rtx::mlp_prepared_forward', mutates_args=('buffers',), device_types='cuda')
def _execute(x: torch.Tensor, gu: torch.Tensor, down: torch.Tensor,
             buffers: list[torch.Tensor], activation: str, key: str) -> None:
    from .kernels.mxfp8 import MXFP8Problem
    from .kernels.mxfp8_gemm import compile_mxfp8_gemm
    from .kernels.mxfp8_glu import compile_mxfp8_glu
    from .kernels.mxfp8_glu_prequant import compile_prequant_glu
    from .kernels.mxfp8_mlp import compile_mixed_down
    from .kernels.mxfp8_quant import compile_mxfp8_dual_quant, compile_mxfp8_quant
    from .mlp import _parse
    from .runtime import validate_runtime_environment

    c = _parse(key)
    validate_runtime_environment()
    if torch.cuda.get_device_capability(x.device)[0] != 12:
        raise RuntimeError('MXFP8 MLP requires SM12x')
    m, k = x.shape
    n = gu.shape[0]
    o, h = down.shape
    out, hidden, scales = buffers[:3]
    with torch.cuda.device(x.device):
        down_start = 3
        if c.gate_backend == 'onchip':
            compile_mxfp8_glu(MXFP8Problem(m, n, k), c.gate_up, activation, 'mxfp8')(
                x, gu, hidden, scales, scales, scales, scales)
        else:
            conf = c.gate_prequant
            weight = conf.resolved_weight_quant()
            qx, qw, sx, sw = buffers[3:7]
            down_start = 7
            if conf.quant_launches == 'dual':
                compile_mxfp8_dual_quant(m, n, k, conf.quant, weight.scale_layout,
                    interleave_weight=True, _byte_scales=True)(x, gu, qx, qw, sx, sw)
            else:
                compile_mxfp8_quant(m, k, conf.quant, _byte_scales=True)(x, qx, sx)
                compile_mxfp8_quant(n, k, weight, interleave_rows=True, _byte_scales=True)(gu, qw, sw)
            gemm = replace(conf.gemm, glu_silu_math=c.gate_up.glu_silu_math)
            compile_prequant_glu(MXFP8Problem(m, n, qx.shape[1]), gemm, activation, 'mxfp8',
                                 _byte_scales=True)(qx, qw, sx, sw, hidden, scales)
        if c.down_backend == 'onchip':
            compile_mixed_down(MXFP8Problem(m, o, h), c.down, _byte_scales=True)(
                hidden, down, out, scales, scales, scales, scales)
        else:
            qw, sw = buffers[down_start:]
            compile_mxfp8_quant(o, h, c.down_quant, _byte_scales=True)(down, qw, sw)
            compile_mxfp8_gemm(MXFP8Problem(m, o, hidden.shape[1]), c.down_gemm, _byte_scales=True)(
                hidden, qw, scales, sw, out)


@_execute.register_fake
def _execute_fake(x, gu, down, buffers, activation, key):
    pass


def uses_prepared_forward(c):
    return c.gate_backend == 'prequant' or (c.gate_backend == 'onchip' and c.down_backend == 'prequant')


@torch.compiler.assume_constant_result
def _rejection(key, m, n, k, o):
    from .configs.glu import glu_rejection
    from .kernels.mxfp8 import MXFP8Problem
    from .kernels.mxfp8_mlp import mixed_down_rejection
    from .mlp import _parse
    c = _parse(key)
    h = n // 2
    gp = MXFP8Problem(m, n, k)
    if c.gate_backend == 'prequant':
        if c.down_backend not in ('onchip', 'prequant'):
            return 'prequant gate requires onchip or prequant down backend'
        reason = c.gate_prequant.rejection(gp, 'mxfp8')
    else:
        reason = glu_rejection(gp, c.gate_up, 'mxfp8')
    if reason is not None:
        return reason
    if c.down_backend == 'onchip':
        return mixed_down_rejection(MXFP8Problem(m, o, h), c.down)
    if (c.down_quant.scale_layout != 'row_major' or c.down_gemm.scale_layout != 'row_major'
            or c.down_quant.scale_rounding != 'infinity'):
        return 'prequant down requires row-major infinity-rounded scales'
    return c.down_quant.rejection(o, h) or c.down_gemm.rejection(MXFP8Problem(m, o, (h + 31)//32*32))


def _allocate(x, gu, down, c):
    m, k = x.shape
    n, o = gu.shape[0], down.shape[0]
    storage_k = (k + 31)//32*32
    storage_h = (n//2 + 31)//32*32

    def empty(shape, dtype):
        return torch.empty(shape, device=x.device, dtype=dtype)

    def scales(rows, layout):
        shape = ((rows, storage_k//32) if layout == 'row_major' else
                 (rows//(64 if layout == 'mma64' else 128), k//128, 512))
        return empty(shape, torch.uint8)

    gate = []
    if c.gate_backend == 'prequant':
        q = c.gate_prequant
        w_layout = q.weight_scale_layout or (q.weight_quant or q.quant).scale_layout
        gate = [empty((m, storage_k), torch.float8_e4m3fn),
                empty((n, storage_k), torch.float8_e4m3fn),
                scales(m, q.quant.scale_layout), scales(n, w_layout)]
    down_scratch = []
    if c.down_backend == 'prequant':
        down_scratch = [empty((o, storage_h), torch.float8_e4m3fn),
                        empty((o, storage_h//32), torch.uint8)]
    return empty((m, storage_h), torch.float8_e4m3fn), empty((m, storage_h//32), torch.uint8), gate, down_scratch


def forward(x, gu, down, activation, key, *, out=None):
    from .mlp import _resolved_policy, _validate
    _validate(x, gu, down, activation)
    c = _resolved_policy(key)
    m, k = x.numel()//x.shape[-1], x.shape[-1]
    reason = _rejection(key, m, gu.shape[0], k, down.shape[0])
    if reason is not None:
        raise ValueError(reason)
    shape = (*x.shape[:-1], down.shape[0])
    out2 = (torch.empty((m, down.shape[0]), device=x.device, dtype=x.dtype) if out is None else
            out.view(m, down.shape[0]))
    x2 = x.view(m, k)
    hidden, scales, gate_scratch, down_scratch = _allocate(x2, gu, down, c)
    _execute(x2, gu, down, [out2, hidden, scales, *gate_scratch, *down_scratch], activation, key)
    return out2.view(shape) if out is None else out


class PreparedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gu, down, activation, key):
        from .mlp import _setup_context
        y = forward(x, gu, down, activation, key)
        _setup_context(ctx, (x, gu, down, activation, key), y)
        return y

    @staticmethod
    def backward(ctx, dy):
        from .mlp import _backward
        return _backward(ctx, dy)
