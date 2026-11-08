"""Both derivative quantization axes remain independent and BF16-rounded."""
from dataclasses import replace

import pytest
import torch

from rtx.mlp import MXFP8MLPConfig,_gradient_quant_rejection,_hidden_quant_rejection


def policy(m,k,h,o):
    from rtx._bwd_search import materialized_bwd_seed
    from rtx.autotune.mlp import materialized_dx_config
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG as base
    c=MXFP8MLPConfig()
    parts=4 if m>=512 else 1
    return replace(c,gate_backend='prequant',down_backend='prequant',recompute_backend='prequant',
        gradient_quantization='recompute_both',hidden_quantization='recompute',
        joint_output_storage='separate',recompute=replace(c.recompute,
            gemm=replace(c.recompute.gemm,glu_grad_store='exchange')),
        gate_backward=materialized_dx_config(materialized_bwd_seed(MXFP8Problem(m,2*h,k),base,parts)),
        down_backward=materialized_bwd_seed(MXFP8Problem(m,o,h),base,parts))


def test_policy_roundtrip_and_disjoint_lifetime_contract():
    c=policy(40000,128,256,128)
    assert MXFP8MLPConfig.from_dict(c.to_dict())==c
    assert _gradient_quant_rejection(c,40000,128,256) is None
    assert _hidden_quant_rejection(c,40000,128,256,128) is None
    for storage in ('serial_reuse','serial_reuse_partials'):
        bad=replace(c,gate_backward=replace(c.gate_backward,operand_storage=storage))
        assert 'disjoint' in _gradient_quant_rejection(bad,40000,128,256)
    assert _gradient_quant_rejection(c,40001,128,256)
    assert _gradient_quant_rejection(replace(c,hidden_quantization='separate'),40000,128,256)


def test_benchmark_retargets_only_shape_dependent_ranges():
    from benchmarks.qualify_mxfp8_long_mlp import initial_schedule
    c=policy(68320,256,512,256)
    result=initial_schedule({'initial':c.to_dict(),'result':{'config':c.to_dict()}},(40000,256,512,256))
    assert result.gate_backward.dw.split_reduction==c.gate_backward.dw.split_reduction
    assert result.gate_backward.dw.reduction_tile!=c.gate_backward.dw.reduction_tile
    assert result.gate_backward.dw.quant_a==c.gate_backward.dw.quant_a
    assert result.gate_backward.dw.workspace_epilogue==c.gate_backward.dw.workspace_epilogue


def test_native_one_stage_materialization_keeps_a_legal_scale_owner():
    from rtx._bwd_search import materialized_bwd_seed
    from rtx.autotune.mlp import materialized_dx_config
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    p=MXFP8Problem(40000,512,128)
    c=materialized_bwd_seed(p,DEFAULT_FUSED_MXFP8_BWD_CONFIG)
    assert c.dx.gemm.tile_m==64 and c.dx.gemm.stages==1
    c=materialized_dx_config(c)
    assert c.dx.gemm.scale_role=='producer' and c.implementation_rejection(p) is None


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='assigned SM12x CUDA required')


@cuda
@pytest.mark.parametrize('shape',[(128,128,256),(192,256,384)])
@pytest.mark.parametrize('swizzle',['none','xor_16b'])
@pytest.mark.parametrize('math,amax',[('fp32','fp32'),('bf16x2','bf16_bits')])
@pytest.mark.parametrize('activation',['silu','gelu'])
def test_exact_bf16_derivative_transpose_data_scales_and_graph(shape,swizzle,math,amax,activation):
    from rtx.kernels.mxfp8_glu_prequant import build_prequant_glu_runner
    torch.manual_seed(81112)
    m,k,h=shape
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.2
    w=torch.randn(2*h,k,device='cuda',dtype=torch.bfloat16)*.02
    dh=torch.randn(m,h,device='cuda',dtype=torch.bfloat16)*.1
    dp=torch.full((m,h),123,device='cuda',dtype=torch.bfloat16)
    expected=torch.empty(2*m,h,device='cuda',dtype=torch.bfloat16)
    hidden=torch.empty_like(dh)
    qg=torch.empty(2*m,h,device='cuda',dtype=torch.float8_e4m3fn)
    sg=torch.empty(2*m,h//32,device='cuda',dtype=torch.float8_e8m0fnu)
    qh=torch.empty(h,m,device='cuda',dtype=torch.float8_e4m3fn)
    sh=torch.empty(h,m//32,device='cuda',dtype=torch.float8_e8m0fnu)
    def guarded(rows,cols,dtype):
        storage=torch.full((rows*cols+32,),123,device='cuda',dtype=torch.uint8)
        return storage,storage[16:-16].view(dtype).view(rows,cols)
    qguard,qt=guarded(2*h,m,torch.float8_e4m3fn)
    sguard,st=guarded(2*h,m//32,torch.float8_e8m0fnu)
    c=policy(m,k,h,k).recompute
    c=replace(c,gemm=replace(c.gemm,glu_scratch_swizzle=swizzle))
    common=dict(config=c,activation=activation,output_format='mxfp8',bind_inputs=False,
                derivative=True,emit_hidden=True,separate_hidden=True)
    plain=build_prequant_glu_runner(x,w,expected,(dh,hidden),**common)
    aux=((((dh,hidden),qg,sg),qh,sh),qt,st)
    fused=build_prequant_glu_runner(x,w,dp,aux,emit_gradient_quant=True,
        emit_hidden_quant=True,emit_gradient_transpose_quant=True,
        gradient_transpose_quant_policy=(math,amax),**common)
    def run():
        plain(x,w,expected,(dh,hidden));fused(x,w,dp,aux)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run();run()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        x.add_(.001);dh.mul_(.99)
        graph.replay()
        assert bool((dp==123).all()),'both-orientation path wrote the unused BF16 derivative'
        blocks=expected.view(m,2*h).T.float().reshape(2*h,-1,32)
        maximum=blocks.abs().amax(-1,keepdim=True)
        scale=torch.exp2(torch.ceil(torch.log2((maximum/448).clamp_min(2.**-126))))
        want=(blocks/scale).to(torch.float8_e4m3fn).reshape(2*h,m).view(torch.uint8)
        torch.testing.assert_close(qt.view(torch.uint8),want,rtol=0,atol=0)
        codes=(scale.view(torch.int32)>>23).to(torch.uint8).squeeze(-1)
        torch.testing.assert_close(st.view(torch.uint8),codes,rtol=0,atol=0)
        for guard in (qguard,sguard):
            assert bool((guard[:16]==123).all()) and bool((guard[-16:]==123).all())


@cuda
@pytest.mark.parametrize('k',[128,256,384])
def test_large_narrow_compiled_all_gradients_and_first_hit_choice(k,monkeypatch):
    from rtx.autotune.mlp import make_mlp_adapter,mlp_structural_seeds,reference
    from rtx.mlp import mxfp8_mlp
    from rtx.fp8_bwd import _MatmulRunner
    torch.manual_seed(81113);torch.set_float32_matmul_precision('highest')
    m,h,o=40000,2*k,k
    c=policy(m,k,h,o)
    values=tuple((torch.randn(*s,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
        for s,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    dy=torch.randn(o,m,device='cuda',dtype=torch.bfloat16).T*.1
    adapter=make_mlp_adapter(*values,initial=replace(c,gradient_quantization='recompute'),
        phase='train',warmup=1,samples=3,calls_per_sample=1)
    changed=adapter.update_fn(adapter.initial_config,'gradient_quantization_path','recompute_both')
    assert adapter.rejection(changed) is None
    assert changed.gate_backward==adapter.initial_config.gate_backward
    assert changed.down_backward==adapter.initial_config.down_backward
    assert any(s.gradient_quantization=='recompute_both'
               for s in mlp_structural_seeds(adapter,'mxfp8','train')[:2])
    assert {s.recompute.gemm.glu_scratch_swizzle
            for s in mlp_structural_seeds(adapter,'mxfp8','train')[:2]}=={'none','xor_16b'}
    original=_MatmulRunner.quantize
    def no_gradient_requantization(self,a,b):
        assert self.quantized_a.shape!=(2*h,m),'gate dW derivative was requantized'
        return original(self,a,b)
    monkeypatch.setattr(_MatmulRunner,'quantize',no_gradient_requantization)
    fn=torch.compile(lambda *a:mxfp8_mlp(*a,config=c),fullgraph=True,dynamic=False)
    y=fn(*values);grads=torch.autograd.grad(y,values,dy)
    with torch.no_grad():want=reference(*values,dy,'silu')
    for a,b in zip((y,*grads),want):
        assert bool(a.isfinite().all())
        error=float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        assert error<.025,error


@cuda
@pytest.mark.parametrize('selected',[False,True])
def test_registered_op_partial_masks_and_live_graph_cleanup(selected):
    import json
    import weakref
    from rtx import mlp as impl
    from rtx._mlp_autotune import selected_mlp
    from rtx.fp8_bwd import _intern_bwd_config
    from rtx.runtime import clear_runtime_caches,retain_runtime_resources
    torch.manual_seed(81115)
    m,k,h,o=128,128,256,128
    c=policy(m,k,h,o)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.2
    w=torch.randn(2*h,k,device='cuda',dtype=torch.bfloat16)*.02
    down=torch.randn(o,h,device='cuda',dtype=torch.bfloat16)*.02
    dh=torch.randn(m,h,device='cuda',dtype=torch.bfloat16)*.1
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    key=impl._config_key(c);gkey=_intern_bwd_config(c.gate_backward);dkey=_intern_bwd_config(c.down_backward)
    if not selected:
        checked=torch.library.opcheck(impl._recomputed_all_bwd_op,(x,w,dh,dy,down,'silu',key,gkey,dkey))
        assert all(v=='SUCCESS' for v in checked.values())
        transient=dh.clone();ref=weakref.ref(transient)
        prepared=impl._prepare_recomputed_all_bwd(x,w,transient,dy,down,'silu',key,gkey,dkey)
        assert prepared[0][1] is None,'BF16 derivative placeholder retained'
        del transient
        assert ref() is None,'dH retained through remaining gradient GEMMs'
        impl._consume_recomputed_all_bwd(prepared)
    spec=json.dumps(dict(mode='fixed',initial=c.to_dict()),sort_keys=True)
    def forward(*v):
        if selected:return selected_mlp(*v,'silu',spec,True)[0]
        return impl.mxfp8_mlp(*v,config=c)
    fn=torch.compile(forward,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    for mask in ((True,True,False),(False,True,True),(True,False,True)):
        v=tuple(t.detach().requires_grad_(flag) for t,flag in zip((x,w,down),mask))
        y=fn(*v)
        actual=torch.autograd.grad(y,tuple(t for t in v if t.requires_grad),dy)
        baseline=replace(c,gradient_quantization='recompute')
        z=impl.mxfp8_mlp(*v,config=baseline)
        expected=torch.autograd.grad(z,tuple(t for t in v if t.requires_grad),dy)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        v=tuple(t.detach().requires_grad_() for t in (x,w,down))
        def run():
            y=fn(*v)
            return y,*torch.autograd.grad(y,v,dy)
        for _ in range(3):run()
        with retain_runtime_resources() as owners:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):actual=run()
    torch.cuda.current_stream().wait_stream(stream)
    clear_runtime_caches(device='cuda:0',preserve_compiled=True)
    with torch.no_grad():x.add_(.001);w.add_(.0001);dy.mul_(.99)
    graph.replay();torch.cuda.synchronize()
    with torch.cuda.stream(stream):expected=run()
    stream.synchronize()
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    live=torch.cuda.memory_allocated()
    for _ in range(8):graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==live and owners
    graph.reset()
