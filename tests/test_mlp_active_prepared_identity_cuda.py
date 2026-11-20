"""Prove that the collapsed prepared-gate identities execute identical math."""
from dataclasses import replace

import pytest
import torch

from rtx.autotune.mlp import active_schedule
from rtx.fp8_bwd import _default_bwd_config
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.mlp import MXFP8MLPConfig,mxfp8_mlp


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_compiled_prepared_gate_inactive_tiles_are_bitwise_aliases():
    torch.manual_seed(5032)
    x=(torch.randn(128,256,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
    gate=(torch.randn(256,256,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
    down=(torch.randn(256,128,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
    dy=torch.randn_like(x)
    base=replace(MXFP8MLPConfig(),gate_backend='prequant',down_backend='prequant',
        recompute_backend='prequant',
        gate_backward=_default_bwd_config(MXFP8Problem(128,256,256)),
        down_backward=_default_bwd_config(MXFP8Problem(128,256,128)))
    changed=replace(base,gate_up=replace(base.gate_up,tile_m=128,
        maxrregcount=128,quant_vec=8,a_swizzle='64b'))
    assert active_schedule(base,'mxfp8','train')==active_schedule(changed,'mxfp8','train')
    def run(config):
        fn=torch.compile(lambda a,b,c:mxfp8_mlp(a,b,c,config=config),
                         fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        y=fn(x,gate,down)
        grads=torch.autograd.grad(y,(x,gate,down),dy)
        return (y,*grads)
    expected=tuple(t.detach().clone() for t in run(base))
    actual=run(changed)
    torch.cuda.synchronize()
    for a,b in zip(actual,expected):
        assert bool(a.isfinite().all())
        torch.testing.assert_close(a,b,rtol=0,atol=0)
