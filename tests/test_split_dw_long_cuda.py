"""Independent quantized oracle for long, non-power-of-two split chunks."""
from dataclasses import replace
import pytest
import torch

from rtx.autotune.mlp import materialized_dx_config, materialized_dw_config
from rtx.fp8_bwd import mxfp8_linear_backward, _default_bwd_config
from rtx.kernels.mxfp8 import MXFP8Problem


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def quantized_reference(tensor):
    width=tensor.shape[1]
    blocks=torch.nn.functional.pad(tensor.float(),(0,(-width)%32)).reshape(tensor.shape[0],-1,32)
    maximum=blocks.abs().amax(-1,keepdim=True)
    scale=torch.exp2(torch.ceil(torch.log2((maximum/448).clamp_min(2.**-126))))
    return ((blocks/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale).flatten(1)[:,:width]


@pytest.mark.parametrize('rows,parts',[(11296,4),(11712,4),(11712,8),
                                      (32768,4),(32800,8),(65568,16)])
def test_long_split_matches_independent_recipe(rows,parts,record_property):
    torch.manual_seed(191)
    n,k=256,128
    problem=MXFP8Problem(rows,n,k)
    config=materialized_dx_config(materialized_dw_config(_default_bwd_config(problem)))
    tile_k=config.dw.gemm.tile_k
    chunk=((rows+parts*tile_k-1)//(parts*tile_k))*tile_k
    config=replace(config,dw=replace(config.dw,
        reduction='split_fp32_workspace',split_reduction=parts,reduction_tile=chunk,
        workspace_epilogue='tree',gemm=replace(config.dw.gemm,
            epilogue='direct',store_vec=1,epilogue_stages=1,tiles_per_cta=1)))
    assert config.implementation_rejection(problem) is None
    x=torch.randn(rows,k,device='cuda',dtype=torch.bfloat16)*.2
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
    # Deliberately noncontiguous upstream gradient.
    dy=torch.randn(n,rows,device='cuda',dtype=torch.bfloat16).t()*.1
    actual=mxfp8_linear_backward(dy,x,w,config=config,autotune='off')
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        expected=((quantized_reference(dy)@quantized_reference(w.T).T).bfloat16(),
                  (quantized_reference(dy.T)@quantized_reference(x.T).T).bfloat16())
    finally:
        torch.backends.cuda.matmul.allow_tf32=old
    for name,a,b in zip(('dx','dw'),actual,expected):
        assert bool(a.isfinite().all())
        difference=a.float()-b.float()
        rms=float(difference.square().mean().sqrt())
        reference_rms=float(b.float().square().mean().sqrt())
        error=rms/max(reference_rms,1e-12)
        maximum=float(difference.abs().max())
        record_property(name+'_nrms',error)
        record_property(name+'_max_abs',maximum)
        record_property(name+'_reference_rms',reference_rms)
        assert error<.002,(name,error,maximum,rows,parts,chunk)
    record_property('chunk',chunk)
