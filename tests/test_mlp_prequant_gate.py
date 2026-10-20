from dataclasses import replace
import pytest
import torch

from rtx import MXFP8MLPConfig, mxfp8_mlp


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape',[(128,128,256,128),(11712,256,768,256)])
@pytest.mark.parametrize('activation',['silu','gelu'])
@pytest.mark.parametrize('quant_policy',[('fp32','fp32'),('bf16x2','bf16_bits')])
def test_all_recomputed_quantized_consumers(shape,activation,quant_policy,monkeypatch):
    import weakref
    import rtx.mlp as impl
    from rtx.autotune.mlp import make_mlp_adapter
    from rtx.fp8_bwd import _intern_bwd_config,_default_bwd_config,_launch_dw
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.runtime import clear_runtime_caches,retain_runtime_resources
    m,k,h,o=shape
    torch.manual_seed(971)
    x,w,down=[torch.randn(*s,device='cuda',dtype=torch.bfloat16)*.02
              for s in ((m,k),(2*h,k),(o,h))]
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    dh=torch.randn(m,h,device='cuda',dtype=torch.bfloat16)*.1
    adapter=make_mlp_adapter(x,w,down,phase='train',measurement_mode='selected_direct')
    cfg=adapter.update_fn(adapter.initial_config,'gradient_quantization_path','recompute')
    gc=cfg.gate_backward
    cfg=replace(cfg,gate_backward=replace(gc,dx=replace(gc.dx,
        quant_a=replace(gc.dx.quant_a,quant_math=quant_policy[0],quant_amax=quant_policy[1]))))
    dc=cfg.down_backward or _default_bwd_config(MXFP8Problem(m,o,h))
    qb=replace(dc.dw.resolved_quant_b(),scale_layout='row_major',quant_math=quant_policy[0],
               quant_amax=quant_policy[1],scale_rounding=cfg.recompute.gemm.scale_rounding)
    dc=replace(dc,quant_schedule='per_matmul',stream_schedule='single',dw=replace(dc.dw,backend='decomposed',quant_launches='separate',quant_b=qb,
        quant_a=replace(dc.dw.quant_a,scale_layout='row_major'),gemm=replace(dc.dw.gemm,scale_layout='row_major',scale_role='consumers')))
    cfg=replace(cfg,down_backward=dc,recompute=replace(cfg.recompute,
                gemm=replace(cfg.recompute.gemm,glu_hidden_decode='fp32')))
    key=impl._config_key(cfg);gkey=_intern_bwd_config(cfg.gate_backward);dkey=_intern_bwd_config(dc)
    if m==128:
        checked=torch.library.opcheck(impl._recomputed_all_bwd_op,
            (x,w,dh,dy,down,activation,key,gkey,dkey))
        assert all(v=='SUCCESS' for v in checked.values())
    def baseline():
        dx,dw,hidden=impl._recomputed_gate_bwd_launch(x,w,dh,activation,key,gkey)
        return dx,dw,_launch_dw(dy,hidden,down,dkey)
    def fused():
        prepared=impl._prepare_recomputed_all_bwd(x,w,dh,dy,down,activation,key,gkey,dkey)
        return impl._consume_recomputed_all_bwd(prepared)
    # The prepared tuple must not own dH after the final recomputation read.
    transient=dh.clone();ref=weakref.ref(transient)
    import rtx.fp8_bwd as bwd_impl
    original_resolve=bwd_impl._resolve_bwd_context
    resolutions=[]
    def counted_resolve(dy_arg,x_arg,w_arg,key_arg):
        resolutions.append(tuple(w_arg.shape))
        return original_resolve(dy_arg,x_arg,w_arg,key_arg)
    monkeypatch.setattr(bwd_impl,'_resolve_bwd_context',counted_resolve)
    prepared=impl._prepare_recomputed_all_bwd(x,w,transient,dy,down,activation,key,gkey,dkey)
    assert resolutions.count(tuple(down.shape))==1
    monkeypatch.setattr(bwd_impl,'_resolve_bwd_context',original_resolve)
    del transient
    assert ref() is None
    impl._consume_recomputed_all_bwd(prepared);del prepared
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):baseline();fused()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with retain_runtime_resources() as owners:
        with torch.cuda.graph(graph,stream=stream):actual=fused()
    clear_runtime_caches(device='cuda:0',preserve_compiled=True)
    assert owners
    for _ in range(3):
        x.add_(.001);w.add_(.0001);dh.mul_(.99);dy.mul_(.99)
        expected=baseline()
        graph.replay();torch.cuda.synchronize()
        for a,b in zip(actual,expected):
            assert bool(a.isfinite().all())
            torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape',[(128,128,256),(17,128,96),(192,128,384),(17,128,97)])
@pytest.mark.parametrize('activation',['silu','gelu'])
@pytest.mark.parametrize('gradient',[False,True])
@pytest.mark.parametrize('quant_policy',[('fp32','fp32'),('bf16x2','bf16_bits')])
def test_recompute_emits_independent_hidden_transpose_quant(shape,activation,gradient,quant_policy):
    from rtx.kernels.mxfp8_glu_prequant import build_prequant_glu_runner
    m,k,h=shape;storage_m=(m+31)//32*32
    torch.manual_seed(967)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(2*h,k,device='cuda',dtype=torch.bfloat16)*.02
    dh=torch.randn(m,h,device='cuda',dtype=torch.bfloat16)*.1
    dp=torch.empty(2*m,h,device='cuda',dtype=torch.bfloat16)
    expected_dp=torch.empty_like(dp)
    expected_hidden=torch.empty(m,h,device='cuda',dtype=torch.bfloat16)
    untouched=torch.full_like(expected_hidden,123)
    def guarded(rows,cols,dtype):
        storage=torch.full((rows*cols+32,),123,device='cuda',dtype=torch.uint8)
        return storage,storage[16:-16].view(dtype).view(rows,cols)
    q_guard,q=guarded(h,storage_m,torch.float8_e4m3fn)
    s_guard,s=guarded(h,storage_m//32,torch.float8_e8m0fnu)
    cfg=MXFP8MLPConfig().recompute
    cfg=replace(cfg,gemm=replace(cfg.gemm,glu_grad_store='exchange',
                               glu_hidden_decode='fp32_packed' if gradient else 'fp32'))
    common=dict(config=cfg,activation=activation,output_format='mxfp8',bind_inputs=False,
                derivative=True,emit_hidden=True,separate_hidden=True)
    aux=(dh,untouched)
    if gradient:
        qg=torch.empty(2*m,h,device='cuda',dtype=torch.float8_e4m3fn)
        sg=torch.empty(2*m,h//32,device='cuda',dtype=torch.float8_e8m0fnu)
        aux=(aux,qg,sg)
    aux=(aux,q,s)
    if gradient and h%32:
        with pytest.raises(ValueError,match='gradient quantization requires'):
            build_prequant_glu_runner(x,w,dp,aux,emit_gradient_quant=True,emit_hidden_quant=True,**common)
        return
    fused=build_prequant_glu_runner(x,w,dp,aux,emit_gradient_quant=gradient,emit_hidden_quant=True,
                                  gradient_quant_policy=quant_policy,hidden_quant_policy=quant_policy,**common)
    plain=build_prequant_glu_runner(x,w,expected_dp,(dh,expected_hidden),**common)
    def run():
        fused(x,w,dp,aux)
        plain(x,w,expected_dp,(dh,expected_hidden))
    for _ in range(3):run()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    def reference(src):
        blocks=src.float().reshape(src.shape[0],-1,32)
        bits=blocks.abs().amax(-1).view(torch.int32)
        exponent=((bits>>23)&255)-127-8+((bits&0x7fffff)>0x600000)
        codes=(exponent.clamp(-127,128)+127).to(torch.uint8)
        scale=(codes.int()<<23).view(torch.float32).clamp_min(2**-126)
        data=(blocks/scale[...,None]).clamp(-448,448).to(torch.float8_e4m3fn)
        return data.reshape(src.shape).view(torch.uint8),codes
    for generation in range(3):
        x.add_(.001);w.add_(.0001);dh.mul_(.99)
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(dp,expected_dp,rtol=0,atol=0)
        assert bool((untouched==123).all()),'fused path wrote the BF16 hidden output'
        # New scales reduce over M, after the original row-wise round/decode.
        logical=torch.zeros(h,storage_m,device='cuda',dtype=torch.bfloat16)
        logical[:,:m]=expected_hidden.T
        rq,rs=reference(logical)
        torch.testing.assert_close(q.view(torch.uint8),rq,rtol=0,atol=0)
        torch.testing.assert_close(s.view(torch.uint8),rs,rtol=0,atol=0)
        # Match the selected standalone instruction policy, not just FP32 math.
        from rtx.kernels.mxfp8_quant import MXFP8QuantConfig,compile_mxfp8_quant
        qc=MXFP8QuantConfig(quant_math=quant_policy[0],quant_amax=quant_policy[1])
        standalone_q=torch.empty_like(q)
        standalone_s=torch.empty_like(s)
        compile_mxfp8_quant(h,storage_m,qc)(logical,standalone_q,standalone_s)
        torch.testing.assert_close(q.view(torch.uint8),standalone_q.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(s.view(torch.uint8),standalone_s.view(torch.uint8),rtol=0,atol=0)
        if gradient:
            rq,rs=reference(expected_dp)
            torch.testing.assert_close(qg.view(torch.uint8),rq,rtol=0,atol=0)
            torch.testing.assert_close(sg.view(torch.uint8),rs,rtol=0,atol=0)
        for storage in (q_guard,s_guard):
            assert bool((storage[:16]==123).all()) and bool((storage[-16:]==123).all())


def test_gradient_quant_contract_rejects_incompatible_recompute():
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_mlp import compile_prequant_glu_derivative
    cfg=MXFP8MLPConfig().recompute.gemm
    for problem,config,joint in [
        (MXFP8Problem(17,194,128),replace(cfg,glu_grad_store='exchange'),True),
        (MXFP8Problem(128,512,128),replace(cfg,glu_grad_store='exchange'),False),
        (MXFP8Problem(128,512,128),replace(cfg,glu_grad_store='direct'),True),
    ]:
        with pytest.raises(ValueError,match='gradient quantization requires'):
            compile_prequant_glu_derivative(problem,config,'silu',emit_hidden=joint,
                                           emit_gradient_quant=True)
    for decode,separate,joint in [('bf16x2',True,True),('fp32',False,True),('fp32',True,False)]:
        with pytest.raises(ValueError,match='hidden quantization requires'):
            compile_prequant_glu_derivative(MXFP8Problem(128,512,128),
                replace(cfg,glu_hidden_decode=decode),'silu',emit_hidden=joint,
                separate_hidden=separate,emit_hidden_quant=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape',[(128,128,256),(17,128,96),(192,128,384)])
@pytest.mark.parametrize('activation',['silu','gelu'])
@pytest.mark.parametrize('separate',[False,True])
@pytest.mark.parametrize('quant_policy',[('fp32','fp32'),('bf16x2','bf16_bits')])
def test_recompute_emits_bf16_rounded_gradient_quant(shape,activation,separate,quant_policy):
    from rtx.kernels.mxfp8_glu_prequant import build_prequant_glu_runner
    m,k,h=shape
    torch.manual_seed(947)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(2*h,k,device='cuda',dtype=torch.bfloat16)*.02
    dh=torch.randn(m,h,device='cuda',dtype=torch.bfloat16)*.1
    prefix=((2*m+7)//8)*8
    out=torch.empty((2*m if separate else prefix+m),h,device='cuda',dtype=torch.bfloat16)
    baseline=torch.empty_like(out)
    hidden=torch.empty(m,h,device='cuda',dtype=torch.bfloat16)
    baseline_hidden=torch.empty_like(hidden)
    def guarded(rows,cols,dtype):
        storage=torch.full((rows*cols+32,),123,device='cuda',dtype=torch.uint8)
        return storage,storage[16:-16].view(dtype).view(rows,cols)
    q_guard,q=guarded(2*m,h,torch.float8_e4m3fn)
    s_guard,s=guarded(2*m,h//32,torch.float8_e8m0fnu)
    cfg=MXFP8MLPConfig().recompute
    cfg=replace(cfg,gemm=replace(cfg.gemm,glu_grad_store='exchange'))
    common=dict(config=cfg,activation=activation,output_format='mxfp8',bind_inputs=False,
                derivative=True,emit_hidden=True,separate_hidden=separate)
    aux=(dh,hidden) if separate else dh
    baseline_aux=(dh,baseline_hidden) if separate else dh
    fused=build_prequant_glu_runner(x,w,out,(aux,q,s),emit_gradient_quant=True,
                                  gradient_quant_policy=quant_policy,**common)
    plain=build_prequant_glu_runner(x,w,baseline,baseline_aux,**common)
    def run():
        fused(x,w,out,(aux,q,s))
        plain(x,w,baseline,baseline_aux)
    for _ in range(3):run()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for generation in range(3):
        x.add_(.001);w.add_(.0001);dh.mul_(.99)
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(out[:2*m],baseline[:2*m],rtol=0,atol=0)
        if separate:
            torch.testing.assert_close(hidden,baseline_hidden,rtol=0,atol=0)
        else:
            # Alignment padding between the outputs is intentionally unwritten.
            torch.testing.assert_close(out[prefix:],baseline[prefix:],rtol=0,atol=0)
        # Independent PyTorch block quantization, explicitly AFTER BF16 rounding.
        blocks=baseline[:2*m].float().view(2*m,h//32,32)
        amax=blocks.abs().amax(-1)
        bits=amax.view(torch.int32)
        exponent=((bits>>23)&255)-127-8+((bits&0x7fffff)>0x600000)
        codes=(exponent.clamp(-127,128)+127).to(torch.uint8)
        decoded=(codes.int()<<23).view(torch.float32).clamp_min(2**-126)
        expected=(blocks/decoded[...,None]).clamp(-448,448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(q.view(torch.uint8),expected.view(2*m,h).view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(s.view(torch.uint8),codes,rtol=0,atol=0)
        for storage in (q_guard,s_guard):
            assert bool((storage[:16]==123).all()) and bool((storage[-16:]==123).all())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape',[(128,128,256),(192,128,256),(17,128,96)])
@pytest.mark.parametrize('activation',['silu','gelu'])
@pytest.mark.parametrize('store,decode',[('direct','fp32'),('exchange','fp32_packed')])
def test_prequant_derivative_separate_hidden(shape,activation,store,decode):
    from rtx.kernels.mxfp8_glu_prequant import build_prequant_glu_runner
    m,k,h=shape
    torch.manual_seed(731)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(2*h,k,device='cuda',dtype=torch.bfloat16)*.02
    dh=torch.randn(m,h,device='cuda',dtype=torch.bfloat16)*.1
    prefix=((2*m+7)//8)*8
    packed=torch.empty(prefix+m,h,device='cuda',dtype=torch.bfloat16)
    def guarded(rows):
        base=torch.full((rows*h+16,),123.,device='cuda',dtype=torch.bfloat16)
        return base,base[8:-8].view(rows,h)
    dp_base,dp=guarded(2*m)
    hidden_base,hidden=guarded(m)
    cfg=MXFP8MLPConfig().recompute
    cfg=replace(cfg,gemm=replace(cfg.gemm,glu_grad_store=store,glu_hidden_decode=decode))
    common=dict(config=cfg,activation=activation,output_format='mxfp8',
                bind_inputs=False,derivative=True,emit_hidden=True)
    packed_runner=build_prequant_glu_runner(x,w,packed,dh,**common)
    split_runner=build_prequant_glu_runner(x,w,dp,(dh,hidden),separate_hidden=True,**common)
    def run():
        packed_runner(x,w,packed,dh)
        split_runner(x,w,dp,(dh,hidden))
    for _ in range(3): run()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run()
    for _ in range(3):
        x.add_(.001);w.add_(.0001);dh.mul_(.99)
        graph.replay()
        torch.cuda.synchronize()
        assert bool(dp.isfinite().all()) and bool(hidden.isfinite().all())
        torch.testing.assert_close(dp,packed[:2*m],rtol=0,atol=0)
        torch.testing.assert_close(hidden,packed[prefix:],rtol=0,atol=0)
        for base in (dp_base,hidden_base):
            assert bool((base[:8]==123).all()) and bool((base[-8:]==123).all())


def test_prequant_gate_roundtrip_and_identity():
    from rtx.autotune.mlp import active_schedule
    base=replace(MXFP8MLPConfig(),gate_backend='prequant')
    assert MXFP8MLPConfig.from_dict(base.to_dict())==base
    other=replace(base,gate_prequant=replace(base.gate_prequant,
                    gemm=replace(base.gate_prequant.gemm,stages=3)))
    assert active_schedule(base,'mxfp8','forward')!=active_schedule(other,'mxfp8','forward')


def test_joint_storage_serialization():
    base=MXFP8MLPConfig()
    assert 'joint_output_storage' not in base.to_dict()
    assert MXFP8MLPConfig.from_dict(base.to_dict())==base
    separate=replace(base,recompute_backend='prequant',joint_output_storage='separate')
    assert MXFP8MLPConfig.from_dict(separate.to_dict())==separate
    with pytest.raises(ValueError,match='joint_output_storage'):
        replace(base,joint_output_storage='unknown')
    assert 'gradient_quantization' not in base.to_dict()
    quantized=replace(base,gradient_quantization='recompute')
    assert MXFP8MLPConfig.from_dict(quantized.to_dict())==quantized
    with pytest.raises(ValueError,match='gradient_quantization'):
        replace(base,gradient_quantization='unknown')
    assert 'hidden_quantization' not in base.to_dict()
    hidden=replace(quantized,hidden_quantization='recompute')
    assert MXFP8MLPConfig.from_dict(hidden.to_dict())==hidden
    with pytest.raises(ValueError,match='hidden_quantization'):
        replace(base,hidden_quantization='unknown')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape',[(128,128,256,128),(11712,256,768,256),
                                  (11296,128,384,128),(17,128,96,128)])
@pytest.mark.parametrize('activation',['silu','gelu'])
@pytest.mark.parametrize('selected',[False,True])
@pytest.mark.parametrize('hidden_quant',['separate','recompute'])
def test_integrated_recompute_gradient_graph(shape,activation,selected,hidden_quant,monkeypatch):
    import json
    from rtx.autotune.mlp import make_mlp_adapter,reference,active_schedule
    from rtx._mlp_autotune import selected_mlp
    from rtx.fp8_bwd import _MatmulRunner
    from rtx.runtime import clear_runtime_caches,retain_runtime_resources
    m,k,h,o=shape
    torch.manual_seed(953)
    values=tuple((torch.randn(*s,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
                 for s in ((m,k),(2*h,k),(o,h)))
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    adapter=make_mlp_adapter(*values,phase='train',measurement_mode='selected_direct')
    cfg=adapter.update_fn(adapter.initial_config,'gradient_quantization_path','recompute')
    if hidden_quant=='recompute':
        cfg=adapter.update_fn(cfg,'hidden_quantization_path','recompute')
        if m%32:
            assert adapter.rejection(cfg) is not None
            return
    assert adapter.rejection(cfg) is None
    assert adapter.rejection(replace(cfg,backward_overlap=True)) is not None
    assert adapter.rejection(replace(cfg,joint_output_storage='packed')) is not None
    incompatible=replace(cfg,gate_backward=replace(cfg.gate_backward,dx=replace(
        cfg.gate_backward.dx,quant_a=replace(cfg.gate_backward.dx.quant_a,quant_math='bf16x2'))))
    assert adapter.rejection(incompatible) is not None
    old=replace(cfg,gradient_quantization='separate',hidden_quantization='separate')
    assert active_schedule(cfg,'mxfp8','train')!=active_schedule(old,'mxfp8','train')
    assert active_schedule(cfg,'mxfp8','forward')==active_schedule(old,'mxfp8','forward')
    original=_MatmulRunner.quantize
    def no_redundant_gate_dx_quant(self,a,b):
        assert self.quantized_a.shape!=(m,2*h),'gate dX operand was requantized'
        return original(self,a,b)
    monkeypatch.setattr(_MatmulRunner,'quantize',no_redundant_gate_dx_quant)
    if selected and hidden_quant=='separate':
        import weakref
        import rtx.mlp as mlp_impl
        prepare=mlp_impl._prepare_recomputed_gate_bwd
        consume=mlp_impl._consume_recomputed_gate_bwd
        last_dh=[]
        def checked_prepare(x,w,dh,*args):
            last_dh[:]=[weakref.ref(dh)]
            return prepare(x,w,dh,*args)
        def checked_consume(prepared):
            assert last_dh[0]() is None,'dH retained through gate GEMMs'
            return consume(prepared)
        monkeypatch.setattr(mlp_impl,'_prepare_recomputed_gate_bwd',checked_prepare)
        monkeypatch.setattr(mlp_impl,'_consume_recomputed_gate_bwd',checked_consume)
    spec=json.dumps(dict(mode='fixed',initial=cfg.to_dict()),sort_keys=True)
    def forward(*v):
        if selected:return selected_mlp(*v,activation,spec,True)[0]
        return mxfp8_mlp(*v,activation=activation,config=cfg)
    torch._dynamo.reset_code(forward.__code__)
    fn=torch.compile(forward,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def step():
        y=fn(*values)
        return y,*torch.autograd.grad(y,values,dy)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):actual=step()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with retain_runtime_resources() as owners:
        with torch.cuda.graph(graph,stream=stream):actual=step()
    clear_runtime_caches(device='cuda:0',preserve_compiled=True)
    assert owners
    for generation in range(3):
        with torch.no_grad():
            for v in values:v.add_(.0001)
            dy.mul_(.99)
            expected=reference(*values,dy,activation)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):graph.replay()
        stream.synchronize()
        for a,b in zip(actual,expected):
            a=a.detach();b=b.detach()
            assert bool(a.isfinite().all())
            error=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
            assert error<.025,error


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('mask',[(True,False,True),(False,True,True),(True,True,False),
                                (True,False,False),(False,True,False),(False,False,True)])
@pytest.mark.parametrize('selected',[False,True])
@pytest.mark.parametrize('hidden_quant',['separate','recompute'])
@pytest.mark.parametrize('reuse_partials',[False,True])
def test_recompute_gradient_partial_requests(mask,selected,hidden_quant,reuse_partials):
    import json
    from rtx.autotune.mlp import make_mlp_adapter,reference
    from rtx._mlp_autotune import selected_mlp
    torch.manual_seed(959)
    values=tuple(torch.randn(*s,device='cuda',dtype=torch.bfloat16)*.02
                 for s in ((128,128),(512,128),(128,256)))
    adapter=make_mlp_adapter(*values,phase='train',measurement_mode='selected_direct')
    cfg=adapter.update_fn(adapter.initial_config,'gradient_quantization_path','recompute')
    if hidden_quant=='recompute':
        cfg=adapter.update_fn(cfg,'hidden_quantization_path','recompute')
    if reuse_partials:
        from rtx.autotune.mlp import materialized_dx_config,materialized_dw_config
        dc=materialized_dw_config(materialized_dx_config(cfg.down_backward))
        cfg=replace(cfg,down_backward=replace(dc,operand_storage='serial_reuse_partials',
            quant_schedule='per_matmul',stream_schedule='single',execution_order='dx_first'))
        assert adapter.rejection(cfg) is None
    spec=json.dumps(dict(mode='fixed',initial=cfg.to_dict()),sort_keys=True)
    for v,need in zip(values,mask):v.requires_grad_(need)
    def forward(*v):
        if selected:return selected_mlp(*v,'silu',spec,True)[0]
        return mxfp8_mlp(*v,activation='silu',config=cfg)
    torch._dynamo.reset_code(forward.__code__)
    fn=torch.compile(forward,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    dy=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
    y=fn(*values)
    gradients=torch.autograd.grad(y,tuple(v for v,need in zip(values,mask) if need),dy)
    with torch.no_grad():expected=reference(*values,dy,'silu')
    targets=(expected[0],*(v for v,need in zip(expected[1:],mask) if need))
    for a,b in zip((y,*gradients),targets):
        assert bool(a.isfinite().all())
        error=float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        assert error<.025,error


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape',[(128,128,256,128),(17,35,97,41)])
@pytest.mark.parametrize('activation',['silu','gelu'])
def test_separate_joint_compiled_gradients(shape,activation):
    from rtx.autotune.mlp import reference
    torch._dynamo.reset()
    m,k,h,o=shape
    torch.manual_seed(937)
    values=tuple((torch.randn(*s,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
                 for s in ((m,k),(2*h,k),(o,h)))
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    cfg=MXFP8MLPConfig(recompute_backend='prequant',joint_output_storage='separate')
    fn=torch.compile(lambda *v:mxfp8_mlp(*v,activation=activation,config=cfg),
                     fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    for _ in range(3):
        with torch.no_grad():
            for v in values: v.add_(.0001)
        y=fn(*values)
        actual=(y,*torch.autograd.grad(y,values,dy))
        with torch.no_grad(): expected=reference(*values,dy,activation)
        for a,b in zip(actual,expected):
            assert bool(a.isfinite().all()) and bool(b.isfinite().all())
            assert float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))<.025


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape',[(128,128,256,128),(17,35,97,41)])
@pytest.mark.parametrize('activation',['silu','gelu'])
@pytest.mark.parametrize('gate_backend,down_backend',[
    ('prequant','onchip'),('prequant','prequant'),('onchip','prequant')])
def test_prequant_gate_compiled_gradients_and_changed_weights(shape,activation,gate_backend,down_backend):
    from tests.test_mxfp8_mlp import reference
    # Independent configurations share the lambda code object across pytest
    # cases; do not consume Dynamo's per-frame recompile budget across tests.
    torch._dynamo.reset()
    m,k,h,o=shape
    torch.manual_seed(448)
    values=tuple((torch.randn(*s,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
                 for s in ((m,k),(2*h,k),(o,h)))
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    config=replace(MXFP8MLPConfig(),gate_backend=gate_backend,down_backend=down_backend)
    fn=torch.compile(lambda *v:mxfp8_mlp(*v,activation=activation,config=config),
                     fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    for step in range(3):
        with torch.no_grad():
            values[1].add_(.001)
            values[2].add_(.0001)
            values[0].mul_(.99)
        y=fn(*values)
        actual=(y,*torch.autograd.grad(y,values,dy))
        with torch.no_grad():
            yr,gr=reference(*values,dy,activation)
        for name,a,b in zip(('output','dx','dgate','ddown'),actual,(yr,*gr)):
            assert bool(a.isfinite().all()),name
            difference=a.float()-b.float()
            error=difference.norm()/b.float().norm().clamp_min(1e-12)
            assert error<.025,(name,float(error),float(difference.abs().max()),step)
