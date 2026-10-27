import pytest
import torch

from rtx.experimental_ms_eden import _gradient, ms_eden_gradient


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def inputs(phase):
    torch.manual_seed(901)
    dy=torch.randn(128,256,device="cuda",dtype=torch.bfloat16)*.25
    operand=torch.randn(256 if phase=="dx" else 128,128,device="cuda",dtype=torch.bfloat16)*.1
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    ua=torch.rand(dy.numel()//16,device="cuda")
    ub=torch.rand(operand.numel()//16,device="cuda")
    return dy,operand,signs,ua,ub


def independent_reference(args,phase):
    from benchmarks.reference_ms_eden import prepare_extended,finalize_scales
    dy,operand,signs,ua,ub=args
    h=torch.ones(1,1,device=dy.device,dtype=torch.float64)
    for _ in range(7):
        h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    norm=torch.tensor(128**-.5,device=dy.device,dtype=torch.float32).double()
    def quantize(source,uniforms):
        rotated=((source.double().reshape(-1,128)*signs.double())@h*norm).float().reshape(source.shape)
        blocks=prepare_extended(rotated,mantissa_bits=4)
        scales,outer=finalize_scales(blocks.corrected_scales,uniforms.reshape(blocks.corrected_scales.shape))
        return (blocks.values*scales.float().unsqueeze(-1)*outer).reshape(source.shape)
    a=quantize(dy.T if phase=="dw" else dy,ua)
    b=quantize(operand.T,ub)
    return a@b.T


@pytest.mark.parametrize("phase",["dx","dw"])
def test_schema_fake_and_fullgraph(phase):
    args=inputs(phase)
    checks=torch.library.opcheck(_gradient,(*args,phase,"{}"))
    assert set(checks.values()) == {"SUCCESS"}
    eager=ms_eden_gradient(*args,phase=phase)
    torch.compiler.reset()
    compiled=torch.compile(lambda *a:ms_eden_gradient(*a,phase=phase),fullgraph=True,dynamic=False)
    for _ in range(3):
        torch.testing.assert_close(compiled(*args),eager,rtol=0,atol=0)
    dy,operand,*_=args
    ref=dy.float()@operand.float() if phase=="dx" else dy.float().T@operand.float()
    assert torch.isfinite(eager).all()
    assert (eager.float()-ref).norm()/ref.norm() < .3
    packed_reference=independent_reference(args,phase)
    assert (eager.float()-packed_reference).norm()/packed_reference.norm() < .005


@pytest.mark.parametrize("phase",["dx","dw"])
def test_graph_replay_changed_randomness_and_no_live_growth(phase):
    args=inputs(phase)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ms_eden_gradient(*args,phase=phase)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        out=ms_eden_gradient(*args,phase=phase)
    for _ in range(3):
        args[0].mul_(.5)
        args[2].neg_()
        args[3].uniform_(); args[4].uniform_()
        graph.replay()
        expected=ms_eden_gradient(*args,phase=phase)
        torch.testing.assert_close(out,expected,rtol=0,atol=0)
    del expected
    torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated()
    for _ in range(20):
        value=ms_eden_gradient(*args,phase=phase)
        del value
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == baseline


def test_higher_order_gradient_is_explicitly_unsupported():
    args=inputs("dx")
    args[0].requires_grad_()
    value=ms_eden_gradient(*args,phase="dx")
    with pytest.raises(RuntimeError,match="higher-order"):
        value.float().sum().backward()


def test_metadata_validation():
    args=inputs("dx")
    with pytest.raises(ValueError,match="signs"):
        ms_eden_gradient(*args[:2],args[2][:64],*args[3:],phase="dx")
    with pytest.raises(ValueError,match="original BF16"):
        ms_eden_gradient(*args,phase="dx",schedule_json='{"reader_mode":"staged"}')


@pytest.mark.parametrize("source_policy",["original_bf16","forward_current_roundtrip_bf16"])
def test_linear_checkpoint_rng_restore_and_partial_gradients(source_policy):
    from rtx.experimental_ms_eden import ms_eden_linear
    from torch.utils.checkpoint import checkpoint
    x=(torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25).requires_grad_()
    weight=(torch.randn_like(x)*.1).requires_grad_()
    dy=torch.randn(128,256,device="cuda",dtype=torch.bfloat16)[:,::2]
    rng=torch.cuda.get_rng_state()
    def linear(a,b):
        return ms_eden_linear(a,b,source_policy=source_policy)
    def step(recompute):
        torch.cuda.set_rng_state(rng)
        y=checkpoint(linear,x,weight,use_reentrant=True) if recompute else linear(x,weight)
        y.backward(dy)
        result=(y.detach().clone(),x.grad.clone(),weight.grad.clone(),torch.cuda.get_rng_state())
        x.grad=weight.grad=None
        return result
    eager=step(False)
    replay=step(True)
    for a,b in zip(eager,replay):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for x_grad,w_grad in ((True,False),(False,True)):
        xx=x.detach().requires_grad_(x_grad)
        ww=weight.detach().requires_grad_(w_grad)
        linear(xx,ww).backward(dy)
        assert (xx.grad is not None) == x_grad
        assert (ww.grad is not None) == w_grad


@pytest.mark.parametrize("scaling", ["current","jit_row_region","block"])
@pytest.mark.parametrize("compiled", [False,True])
def test_matched_forward_and_independent_backward_schedules(scaling,compiled):
    import rtx
    from rtx.experimental_ms_eden import ms_eden_linear
    from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
    torch.compiler.reset()
    x=(torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25).requires_grad_()
    w=(torch.randn_like(x)*.02).requires_grad_()
    cfg=preferred_jit_row_region_config(rtx.NVFP4Problem(128,128,128)) if scaling=='jit_row_region' else None
    policy=rtx.NVFP4ScaleConfig(tensor_scale_mode='exact')
    def call(a,b):
        return ms_eden_linear(a,b,forward_scaling=scaling,forward_config=cfg,forward_outer_mode='exact',
            dx_schedule_json='{"producer_blocks":64}',
            dw_schedule_json='{"producer_blocks":96,"producer_load":"coalesced"}')
    fn=torch.compile(call,fullgraph=True,dynamic=False) if compiled else call
    out=fn(x,w)
    # Stable regional execution takes its mode from dynamic_config, whereas
    # the research API's explicit forward_outer_mode overrides that mode.
    # Match the requested exact policy, not the preferred config's power2.
    from dataclasses import replace
    reference_cfg=replace(cfg,tensor_scale_mode='exact') if cfg is not None else None
    ref=rtx.nvfp4_linear(x.detach(),w.detach(),scaling=scaling,scale_config=policy,
                         dynamic_config=reference_cfg,autotune='off')
    torch.testing.assert_close(out,ref,rtol=0,atol=0)
    grads=torch.autograd.grad(out,(x,w),torch.ones_like(out))
    assert all(torch.isfinite(g).all() for g in grads)


def test_packed_source_cannot_silently_change_forward_policy():
    from rtx.experimental_ms_eden import ms_eden_linear
    x=torch.empty(128,128,dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='fixed tensor-current'):
        ms_eden_linear(x,x,source_policy='forward_current_roundtrip_bf16',forward_scaling='jit_row_region')


@pytest.mark.parametrize("source_policy,schedule_json",[
    ("original_bf16","{}"),("forward_current_roundtrip_bf16","{}"),
    ("original_bf16",'{"amax_reduction":"staged","producer_order":"reduction_major"}'),
    ("original_bf16",'{"producer_load":"coalesced","producer_order":"reduction_major"}'),
    ("forward_current_roundtrip_bf16",'{"producer_load":"coalesced","producer_order":"reduction_major"}'),
    ("original_bf16",'{"producer_load":"coalesced_wide","producer_order":"reduction_major"}'),
    ("forward_current_roundtrip_bf16",'{"producer_load":"coalesced_wide","producer_order":"reduction_major"}'),
    ("forward_current_roundtrip_bf16",'{"amax_reduction":"staged","producer_order":"reduction_major"}'),
    ("forward_current_roundtrip_bf16",'{"reader_mode":"staged","decoder_threads":256,"decoder_blocks":512}')])
def test_linear_compiled_training_rng_reproducible(source_policy,schedule_json):
    from rtx.experimental_ms_eden import ms_eden_linear
    torch.compiler.reset()
    x=(torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25).requires_grad_()
    weight=(torch.randn_like(x)*.1).requires_grad_()
    loss=torch.compile(lambda a,b:ms_eden_linear(a,b,source_policy=source_policy,schedule_json=schedule_json).float().square().mean(),fullgraph=True,dynamic=False)
    loss(x,weight).backward()
    x.grad=weight.grad=None
    state=torch.cuda.get_rng_state()
    loss(x,weight).backward()
    expected=x.grad.clone(),weight.grad.clone()
    x.grad=weight.grad=None
    torch.cuda.set_rng_state(state)
    loss(x,weight).backward()
    for actual,reference in zip((x.grad,weight.grad),expected):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)


def test_packed_gradient_opcheck_and_saved_state_ownership():
    from rtx.experimental_ms_eden import ms_eden_linear,_packed_gradient,_pack_forward
    from rtx.fp4 import quantize_nvfp4
    dy,operand,signs,ua,ub=inputs("dx")
    packed=quantize_nvfp4(operand)
    data=packed.qdata.view(torch.uint8).detach()
    sf=packed.scale.view(torch.uint8).detach(); outer=packed.per_tensor_scale.detach()
    checks=torch.library.opcheck(_packed_gradient,(dy,data,sf,outer,signs,ua,ub,"dx","{}"))
    assert set(checks.values()) == {"SUCCESS"}
    actual=_packed_gradient(dy,data,sf,outer,signs,ua,ub,"dx","{}")
    reference=ms_eden_gradient(dy,packed.dequantize(torch.bfloat16),signs,ua,ub,phase="dx")
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    weight=torch.randn_like(x,requires_grad=True)
    scale=torch.tensor([.01],device="cuda")
    assert set(torch.library.opcheck(_pack_forward,(x.detach(),weight.detach(),scale,scale.clone())).values()) == {"SUCCESS"}
    from benchmarks.compare_nvfp4_packed_gemm import decode
    output,px,sx,pw,sw=_pack_forward(x.detach(),weight.detach(),scale,scale)
    ref=decode(px,sx.view(torch.float8_e4m3fn),scale)@decode(pw,sw.view(torch.float8_e4m3fn),scale).T
    assert (output.float()-ref).norm()/ref.norm() < .005
    saved=[]
    def pack(value):
        saved.append(value)
        return value
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value):
        y=ms_eden_linear(x,weight,source_policy="forward_current_roundtrip_bf16")
    assert len(saved) == 6
    assert all(value.dtype != torch.bfloat16 for value in saved)
    assert all(value._base is None and not hasattr(value,"_base_inputs") for value in saved)
    assert sum(value.numel()*value.element_size() for value in saved) == 18440
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(weight.grad).all()


@pytest.mark.parametrize("outer_mode",["exact","power2"])
def test_forward_policy_matched_across_saved_sources(outer_mode):
    from rtx.experimental_ms_eden import ms_eden_linear
    torch.manual_seed(902)
    x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.25
    w=torch.randn_like(x)*.1
    original=ms_eden_linear(x,w,source_policy="original_bf16",forward_outer_mode=outer_mode)
    packed=ms_eden_linear(x,w,source_policy="forward_current_roundtrip_bf16",forward_outer_mode=outer_mode)
    torch.testing.assert_close(original,packed,rtol=0,atol=0)


@pytest.mark.parametrize("phase",["dx","dw"])
@pytest.mark.parametrize("order",["row_major","reduction_major"])
@pytest.mark.parametrize("reduction",["inline","staged"])
def test_gradient_nondefault_gemm_schedule(phase,order,reduction):
    import json
    from dataclasses import asdict,replace
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    schedule=PosthocPreparationSchedule(producer_order=order,amax_reduction=reduction)
    schedule=replace(schedule,gemm=replace(schedule.gemm,atom_layout_m=2,atom_layout_n=2,stages=3))
    args=inputs(phase)
    output=ms_eden_gradient(*args,phase=phase,schedule_json=json.dumps(asdict(schedule)))
    reference=independent_reference(args,phase)
    assert (output.float()-reference).norm()/reference.norm() < .005


@pytest.mark.parametrize("dx_policy,dw_policy",[
    ("ms_eden","mxfp8"),("mxfp8","ms_eden"),("mxfp8","mxfp8")])
@pytest.mark.parametrize("compiled",[False,True])
def test_mixed_backward_policy_rng_checkpoint_and_missing_gradients(dx_policy,dw_policy,compiled):
    from rtx.experimental_ms_eden import ms_eden_linear
    from torch.utils.checkpoint import checkpoint
    torch.compiler.reset()
    x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    w=torch.randn_like(x,requires_grad=True)
    dy=torch.randn(128,256,device="cuda",dtype=torch.bfloat16)[:,::2]
    def linear(a,b):
        return ms_eden_linear(a,b,dx_policy=dx_policy,dw_policy=dw_policy)
    fn=torch.compile(linear,fullgraph=True,dynamic=False) if compiled else linear
    # Compile/warm before recording RNG state.
    torch.autograd.grad(fn(x,w),(x,w),dy)
    state=torch.cuda.get_rng_state()
    first=torch.autograd.grad(fn(x,w),(x,w),dy)
    after=torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    second=torch.autograd.grad(fn(x,w),(x,w),dy)
    for a,b in zip(first,second):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(after,torch.cuda.get_rng_state())
    if dx_policy==dw_policy=="mxfp8":
        assert torch.equal(state,after)
    if not compiled:
        torch.cuda.set_rng_state(state)
        replay=torch.autograd.grad(checkpoint(linear,x,w,use_reentrant=False),(x,w),dy)
        for a,b in zip(first,replay):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    for need_x,need_w in ((True,False),(False,True)):
        a=x.detach().requires_grad_(need_x)
        b=w.detach().requires_grad_(need_w)
        fn(a,b).backward(dy)
        assert (a.grad is not None)==need_x and (b.grad is not None)==need_w


@pytest.mark.parametrize("dx_policy,dw_policy",[
    ("ms_eden","ms_eden"),("ms_eden","mxfp8"),
    ("mxfp8","ms_eden"),("mxfp8","mxfp8")])
def test_backward_policy_matches_independent_phase_calls(dx_policy,dw_policy):
    from rtx.experimental_ms_eden import ms_eden_linear
    from rtx.fp8_bwd import mxfp8_linear_backward
    from rtx.kernels.mxfp8_bwd import DEFAULT_MXFP8_BWD_CONFIG
    x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    w=torch.randn_like(x,requires_grad=True)
    dy=torch.randn_like(x)
    config=DEFAULT_MXFP8_BWD_CONFIG
    state=torch.cuda.get_rng_state()
    actual=torch.autograd.grad(ms_eden_linear(x,w,dx_policy=dx_policy,dw_policy=dw_policy,
        mxfp8_backward_config=config if "mxfp8" in (dx_policy,dw_policy) else None),(x,w),dy)
    actual_state=torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    stable=mxfp8_linear_backward(dy,x.detach(),w.detach(),config=config,autotune="off")
    expected=[]
    for phase,policy,operand,reference in (("dx",dx_policy,w,stable[0]),("dw",dw_policy,x,stable[1])):
        if policy=="mxfp8":
            expected.append(reference)
        else:
            signs=torch.where(torch.rand(128,device="cuda")<.5,-1.,1.)
            ua=torch.rand(dy.numel()//16,device="cuda",dtype=torch.float32)
            ub=torch.rand(operand.numel()//16,device="cuda",dtype=torch.float32)
            expected.append(ms_eden_gradient(dy,operand.detach(),signs,ua,ub,phase=phase))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(actual_state,torch.cuda.get_rng_state())


@pytest.mark.parametrize("compiled",[False,True])
@pytest.mark.parametrize("explicit_config",[False,True])
def test_jit_exact_outer_request_reaches_quantizer(compiled,explicit_config):
    from dataclasses import replace
    import rtx
    from rtx.experimental_ms_eden import ms_eden_linear
    from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
    torch.compiler.reset()
    torch.manual_seed(194)
    x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.27
    w=torch.randn_like(x)*.021
    power2=preferred_jit_row_region_config(rtx.NVFP4Problem(128,128,128))
    exact=replace(power2,tensor_scale_mode="exact")
    def forward(a,b):
        return ms_eden_linear(a,b,forward_scaling="jit_row_region",forward_outer_mode="exact",
            forward_config=power2 if explicit_config else None)
    fn=torch.compile(forward,fullgraph=True,dynamic=False) if compiled else forward
    actual=fn(x,w)
    reference=rtx.nvfp4_linear(x,w,scaling="jit_row_region",dynamic_config=exact,
        scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode="exact"),autotune="off")
    old_power2=rtx.nvfp4_linear(x,w,scaling="jit_row_region",dynamic_config=power2,autotune="off")
    torch.testing.assert_close(actual,reference,atol=0,rtol=0)
    assert not torch.equal(actual,old_power2),"test input must distinguish actual exact/power2 quantization"
