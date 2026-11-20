"""Global-token MXFP8 preparation, exact routed bytes and native scale tails."""
from dataclasses import replace

import pytest
import torch

cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')


def test_global_input_is_optional_roundtrips_and_has_a_search_axis():
    from rtx.moe import DEFAULT_MOE_CONFIG,MXFP8MoEConfig
    from rtx.autotune.moe import make_moe_adapter,structural_seeds
    p=replace(DEFAULT_MOE_CONFIG.prepared,input_preparation='quantize_once')
    config=replace(DEFAULT_MOE_CONFIG,prepared=p)
    assert DEFAULT_MOE_CONFIG.prepared.input_preparation=='gather_bf16'
    assert MXFP8MoEConfig.from_dict(config.to_dict())==config
    shape=(2000,768,48,250,6,256)
    adapter=make_moe_adapter(shape,None,initial=DEFAULT_MOE_CONFIG)
    assert adapter.axes['prepared.input_preparation']==('gather_bf16','quantize_once','indexed_transpose')
    selected=adapter.update_fn(DEFAULT_MOE_CONFIG,'prepared.input_preparation','quantize_once')
    assert adapter.config_id(selected)!=adapter.config_id(DEFAULT_MOE_CONFIG)
    assert any(c.prepared.input_preparation=='quantize_once' for c in structural_seeds(shape))
    with pytest.raises(ValueError,match='input_preparation'):
        replace(p,input_preparation='automatic_bf16_transpose')


def test_packed_input_metadata_legality():
    from rtx.kernels.mxfp8_moe_input import packed_input_gather_rejection
    assert packed_input_gather_rejection(13,35,5,6,2,32,128,'row_major') is None
    assert packed_input_gather_rejection(13,128,5,6,2,128,128,'mma128') is None
    assert 'truncate' in packed_input_gather_rejection(13,128,5,6,2,5,128,'row_major')
    assert 'Native128' in packed_input_gather_rejection(13,35,5,6,2,128,128,'mma128')


def _quantize(experts,specs,values,config,grid='operand'):
    from rtx.kernels.mxfp8_moe_prepared import compile_expert_quant
    packed=[];scales=[]
    for spec in specs:
        rows,k=spec[:2];batches=spec[4] if len(spec)==5 else experts
        width=(k+31)//32*32
        packed.append(torch.empty((batches*rows,width),device='cuda',dtype=torch.float8_e4m3fn))
        scales.append(torch.empty((batches*rows,width//32),device='cuda',dtype=torch.float8_e8m0fnu))
    compile_expert_quant(experts,tuple(specs),config,grid_mode=grid)(
        values[0],tuple(values),tuple(packed),tuple(scales))
    return tuple(zip(packed,scales))


@cuda
@pytest.mark.parametrize('copy_bf16',[False,True])
@pytest.mark.parametrize('n,d,e,k,storage_c,threads,layout',[
    (1,1,4,1,8,64,'row_major'),
    (13,35,5,2,32,128,'row_major'),
    (128,384,4,3,128,256,'row_major'),
    (130,128,2,1,128,512,'mma128'),
    (17,256,5,2,128,128,'mma128'),
])
def test_packed_input_gather_is_byte_exact_and_zeros_dummy_rows(n,d,e,k,storage_c,threads,layout,copy_bf16):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8_moe_input import compile_packed_input_gather
    from tests.test_mxfp8_moe import make_route
    torch.manual_seed(839)
    # Offset one BF16 element deliberately; the global path must not impose
    # an accidental 16-byte source alignment requirement on ordinary X.
    raw=torch.randn(n*d+1,device='cuda',dtype=torch.bfloat16)*.3
    x=raw[1:].view(n,d)
    ids,_,inverse=make_route(n,e,k,'cuda');c=ids.shape[1]
    config=replace(DEFAULT_PREPARED_MOE_CONFIG.quant,scale_layout=layout)
    row_config=replace(config,scale_layout='row_major',native_scale_store='scalar')
    global_q,global_sf=_quantize(1,[(n,d,'row',False,1)],[x],row_config)[0]
    width=global_q.shape[1]
    qout=torch.empty((e*storage_c,width),device='cuda',dtype=torch.float8_e4m3fn)
    sout=torch.empty((e*storage_c,width//32),device='cuda',dtype=torch.float8_e8m0fnu)
    gathered=torch.empty((e*storage_c,d),device='cuda',dtype=torch.bfloat16) if copy_bf16 else x
    compile_packed_input_gather(n,d,e,c,k,storage_c,threads,layout,copy_bf16)(
        global_q,global_sf,x,ids,inverse,qout,sout,gathered)
    slots=torch.arange(e*c,device='cuda').view(e,c)
    valid=inverse[ids].eq(slots[...,None]).any(-1)
    expected_bf16=torch.zeros((e,storage_c,d),device='cuda',dtype=torch.bfloat16)
    expected_bf16[:,:c]=x[ids].masked_fill(~valid[...,None],0)
    expected_q,expected_sf=_quantize(e,[(storage_c,d,'row',False)],
        [expected_bf16.flatten(0,1)],config)[0]
    assert torch.equal(qout.view(torch.uint8),expected_q.view(torch.uint8))
    assert torch.equal(sout.view(torch.uint8),expected_sf.view(torch.uint8))
    if copy_bf16:assert torch.equal(gathered,expected_bf16.flatten(0,1))
    # Every real assignment also independently copies precisely its token's
    # original packed row; expert-local zero/padding cannot leak another row.
    qlogical=qout.view(e,storage_c,width)[:,:c].view(torch.uint8)
    expected_logical=global_q.view(torch.uint8)[ids].masked_fill(~valid[...,None],0)
    assert torch.equal(qlogical,expected_logical)


@cuda
@pytest.mark.parametrize('layout',['row_major','mma128'])
@pytest.mark.parametrize('grid',['operand','expert','portfolio'])
def test_mixed_portfolio_quantizes_global_x_once_and_all_expert_weights(layout,grid):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx._moe_reference import quantize_decode
    torch.manual_seed(840)
    n,d,e,h=13,128,3,128
    raw=torch.randn(n*d+1,device='cuda',dtype=torch.bfloat16)*.3;x=raw[1:].view(n,d)
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05
    config=replace(DEFAULT_PREPARED_MOE_CONFIG.quant,scale_layout=layout)
    specs=[(n,d,'row',False,1),(2*h,d,'row',True),(d,h,'row',False)]
    result=_quantize(e,specs,[x,gu.flatten(0,1),down.flatten(0,1)],config,grid)
    row=replace(config,scale_layout='row_major',native_scale_store='scalar')
    baseline_global=_quantize(1,[(n,d,'row',False,1)],[x],row)[0]
    baseline_weights=_quantize(e,specs[1:],[gu.flatten(0,1),down.flatten(0,1)],config)
    for actual,expected in zip(result,(baseline_global,*baseline_weights)):
        for a,b in zip(actual,expected):assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    decoded=result[0][0].float()*result[0][1].float().repeat_interleave(32,dim=-1)
    torch.testing.assert_close(decoded,quantize_decode(x),rtol=0,atol=0)


@cuda
def test_glu_accepts_typed_e8_aux_without_torch_dtype_aliasing():
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx._moe_prepared import prepare,packed_gemm
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_prepared import compile_expert_packed_gemm
    torch.manual_seed(841)
    e,c,d,h=2,64,128,128
    x=torch.randn(e,c,d,device='cuda',dtype=torch.bfloat16)*.3
    w=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05
    qx,qw=prepare(((x,'row',False),(w,'row',True)))
    out=torch.empty((e,c,h),device='cuda',dtype=torch.float8_e4m3fn)
    scales=torch.empty((e,c,h//32),device='cuda',dtype=torch.float8_e8m0fnu)
    expected=torch.empty_like(out);expected_sf=torch.empty_like(scales)
    config=DEFAULT_PREPARED_MOE_CONFIG.gate_up
    packed_gemm(qx,qw,expected,config,'glu',expected_sf.view(torch.uint8))
    compile_expert_packed_gemm(e,MXFP8Problem(c,2*h,d),config,'glu',typed_aux_scales=True)(
        qx[0].flatten(0,1),qw[0].flatten(0,1),qx[1].flatten(0,1),qw[1].flatten(0,1),
        out.flatten(0,1),scales.flatten(0,1))
    assert torch.equal(out.view(torch.uint8),expected.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8),expected_sf.view(torch.uint8))


@cuda
@pytest.mark.parametrize('compiled',[False,True])
@pytest.mark.parametrize('shape,joint',[
    ((32,128,4,2,64),False),
    ((13,35,5,2,49),False),
    ((32,128,4,2,64),True),
])
def test_global_input_public_all_gradients(shape,joint,compiled):
    from rtx.moe import DEFAULT_MOE_CONFIG,mxfp8_routed_experts
    from rtx._moe_reference import operational_reference
    from rtx.routing import BalancedRouting
    from tests.test_mxfp8_moe import make_route
    n,d,e,k,h=shape
    torch.manual_seed(942)
    x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    gu=(torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    down=(torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    ids,weights,inverse=make_route(n,e,k,'cuda')
    weights=weights.detach().requires_grad_()
    p=replace(DEFAULT_MOE_CONFIG.prepared,input_preparation='quantize_once')
    if joint:p=replace(p,backward_preparation='joint',recompute=replace(p.recompute,glu_grad_store='exchange'))
    config=replace(DEFAULT_MOE_CONFIG,prepared=p)
    def call(x,gu,down,weights):
        return mxfp8_routed_experts(x,gu,down,BalancedRouting(ids,weights,inverse),config=config)
    if compiled:call=torch.compile(call,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    dy=torch.randn_like(x,dtype=torch.float32)
    result=call(x,gu,down,weights)
    actual=(result,*torch.autograd.grad(result,(x,gu,down,weights),dy))
    expected=operational_reference(x,gu,down,ids,weights,inverse,dy)
    for name,value,reference in zip(('out','dx','dgu','ddown','dweights'),actual,expected):
        error=value.detach().float()-reference.float()
        nrms=float(error.norm()/reference.float().norm().clamp_min(1e-12))
        maximum=float(error.abs().max())
        assert bool(value.isfinite().all()) and (nrms<(.003 if name=='out' else .006) or maximum<2e-7),(name,nrms,maximum)


@cuda
@pytest.mark.parametrize('compiled',[False,True])
def test_global_input_native_scale_public_all_gradients(compiled):
    from rtx.moe import DEFAULT_MOE_CONFIG,mxfp8_routed_experts
    from rtx._moe_reference import operational_reference
    from rtx.routing import BalancedRouting
    from tests.test_mxfp8_moe import make_route
    n,d,e,k,h=17,128,5,2,128
    torch.manual_seed(943)
    x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    gu=(torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    down=(torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    ids,weights,inverse=make_route(n,e,k,'cuda');weights=weights.detach().requires_grad_()
    p=replace(DEFAULT_MOE_CONFIG.prepared.with_scale_layout('mma128'),input_preparation='quantize_once')
    config=replace(DEFAULT_MOE_CONFIG,prepared=p)
    def call(x,gu,down,weights):
        return mxfp8_routed_experts(x,gu,down,BalancedRouting(ids,weights,inverse),config=config)
    if compiled:call=torch.compile(call,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    dy=torch.randn_like(x,dtype=torch.float32);result=call(x,gu,down,weights)
    actual=(result,*torch.autograd.grad(result,(x,gu,down,weights),dy))
    for name,value,reference in zip(('out','dx','dgu','ddown','dweights'),actual,
            operational_reference(x,gu,down,ids,weights,inverse,dy)):
        delta=value.detach().float()-reference.float()
        nrms=float(delta.norm()/reference.float().norm().clamp_min(1e-12));maximum=float(delta.abs().max())
        assert bool(value.isfinite().all()) and (nrms<(.003 if name=='out' else .006) or maximum<2e-7),(name,nrms,maximum)
