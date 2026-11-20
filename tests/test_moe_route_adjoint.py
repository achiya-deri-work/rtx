"""SM120 route adjoint: independent down rounding and route-mask checks."""
from dataclasses import replace

import pytest
import torch


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='SM120 CUDA required')


@cuda
@pytest.mark.parametrize('d,indexed,active', [(96,True,False),(95,False,False),(96,True,True)])
def test_route_adjoint_matches_packed_fp64_oracle(d,indexed,active):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_packed_gemm
    from rtx.kernels.mxfp8_moe_route_epilogue import (
        compile_expert_route_gradient,compile_route_gradient_reduce,
    )

    n,e,c,cs,k,h=11,2,73,96,2,128
    rng=torch.Generator().manual_seed(19425)
    def packed(rows):
        # Generate the bit-level inputs independently of the production quantizer.
        q=(torch.randn((e,rows,h),generator=rng)*7).to(torch.float8_e4m3fn)
        s=torch.randint(117,123,(e,rows,h//32),generator=rng,dtype=torch.uint8)
        decoded=q.double()*torch.exp2(s.double().repeat_interleave(32,dim=-1)-127)
        return q.flatten(0,1).cuda(),s.flatten(0,1).cuda(),decoded
    qx,sx,x=packed(cs)
    qw,sw,w=packed(d)
    raw=x@w.transpose(1,2)
    rounded=raw.float().bfloat16()
    cfg=replace(DEFAULT_PREPARED_MOE_CONFIG.down,tile_n=128,epilogue='direct',store_vec=1)
    problem=MXFP8Problem(cs,d,h)
    dy=(torch.randn((d,n),generator=rng)*1.3).T.contiguous().cuda()

    # Dummy rows alias a valid token ID but aren't named in inverse. Some
    # real routes lie across the 64-row tile boundary in the non-prefix test.
    ids=torch.zeros((e,c),dtype=torch.int64)
    inverse=torch.empty((n,k),dtype=torch.int64)
    rows=torch.arange(n)*3 if active else torch.arange(n)*7
    for expert in range(e):
        ids[expert,rows]=torch.arange(n)
        inverse[:,expert]=expert*c+rows
    mapping=torch.full((e,cs),-1,dtype=torch.int32)
    mapping[:,rows]=torch.arange(n,dtype=torch.int32)
    dummy=mapping[:,:c]<0
    ids,inverse,mapping=ids.cuda(),inverse.cuda(),mapping.reshape(e*cs,1).cuda()
    limit=(int(rows.max())+32)//32*32
    prefix=(torch.tensor([limit],device='cuda',dtype=torch.int32),) if active else ()
    expected=torch.zeros((e,c),dtype=torch.float64)
    expected[:,rows]=(rounded[:,rows].double()*dy.cpu().double()).sum(-1)
    unrounded=(raw[:,rows]*dy.cpu().double()).sum(-1)
    assert (unrounded-expected[:,rows]).abs().max()>1e-4, 'Fixture must detect missing BF16 rounding'
    bf16=torch.empty((e*cs,d),device='cuda',dtype=torch.bfloat16)
    compile_expert_packed_gemm(e,problem,cfg)(qx,qw,sx,sw,bf16,bf16)
    torch.testing.assert_close(bf16.cpu().view(e,cs,d),rounded,rtol=0,atol=0)
    columns=(d+cfg.tile_n-1)//cfg.tile_n
    partial=torch.empty((e*cs,columns),device='cuda',dtype=torch.float32)
    result=torch.empty((e,c),device='cuda',dtype=torch.float32)
    reduce=compile_route_gradient_reduce(e,c,cs,columns,128,active)
    shifted=torch.empty(dy.numel()+1,device='cuda',dtype=dy.dtype)[1:].view_as(dy).copy_(dy)
    assert shifted.is_contiguous() and shifted.data_ptr()%8==4
    baseline=None
    for loader in ('accumulator','shared_vec4','warp_vec2'):
        launch=compile_expert_route_gradient(n,e,c,k,problem,cfg,indexed,active,loader)
        for upstream in (dy,shifted):
            partial.fill_(float('nan'))
            result.fill_(float('nan'))
            launch(qx,qw,sx,sw,partial,mapping if indexed else ids,inverse,upstream,*prefix)
            reduce(partial,result,*prefix)
            actual=result.cpu()
            assert bool(actual.isfinite().all()),loader
            assert (actual[dummy]==0).all(),loader
            torch.testing.assert_close(actual.double(),expected,rtol=2e-6,atol=1e-6)
            if baseline is None:baseline=actual
            else:torch.testing.assert_close(actual,baseline,rtol=0,atol=0)


@cuda
@pytest.mark.parametrize('loader',['shared_vec4','warp_vec2'])
def test_route_adjoint_fullgraph_noncontiguous_dy_and_all_gradients(loader):
    from rtx._moe_reference import operational_reference
    from rtx.moe import DEFAULT_MOE_CONFIG,mxfp8_routed_experts
    from rtx.routing import BalancedRouting
    from tests.test_mxfp8_moe import make_route

    n,d,e,k,h=45,96,3,2,64
    torch.manual_seed(40541)
    x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    gu=(torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    down=(torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    ids,weights,inverse=make_route(n,e,k,'cuda')
    weights=weights.detach().requires_grad_()
    inputs=x,gu,down,weights
    base=DEFAULT_MOE_CONFIG
    p=base.prepared
    adjoint=replace(p.down,tile_n=128)
    candidate=replace(base,prepared=replace(p,backward_preparation='joint',
        recompute=replace(p.recompute,glu_grad_store='exchange'),route_gradient_epilogue='fused',
        route_gradient_loader=loader,route_gradient_gemm=adjoint))
    base=replace(candidate,prepared=replace(candidate.prepared,
        route_gradient_loader='accumulator',route_gradient_gemm=None))
    def model(config):
        def f(a,g,w,r):
            return mxfp8_routed_experts(a,g,w,BalancedRouting(ids,r,inverse),config=config,autotune='off')
        return f
    compiled=torch.compile(model(candidate),fullgraph=True,dynamic=False,
                           options={'triton.cudagraphs':False})
    dy=torch.randn(d,n,device='cuda').T
    assert not dy.is_contiguous()
    for upstream in (dy,dy*.37+.21):
        reference=operational_reference(x,gu,down,ids,weights,inverse,upstream)
        y=compiled(*inputs)
        actual=(y,*torch.autograd.grad(y,inputs,upstream))
        y0=model(base)(*inputs)
        control=(y0,*torch.autograd.grad(y0,inputs,upstream))
        for name,a,b,c0 in zip(('output','dx','dgu','ddown','droute'),actual,reference,control):
            assert a.isfinite().all(),name
            err=(a.detach().double()-b.double())
            nrms=float(err.norm()/b.double().norm().clamp_min(1e-12))
            maximum=float(err.abs().max())
            assert nrms<(.003 if name=='output' else .006) or maximum<2e-7,(name,nrms,maximum)
            if name!='droute':torch.testing.assert_close(a,c0,rtol=0,atol=0)
