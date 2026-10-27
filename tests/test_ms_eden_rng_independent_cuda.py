"""Independent Philox/SR checks; no production helper generates the oracle RNG."""
import json

import pytest
import torch

from benchmarks.reference_ms_eden_rng import uniforms

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("stream",[0,1])
@pytest.mark.parametrize("count",[1,7,129,2049])
def test_uniform_kernel_matches_independent_cpu_words(stream,count):
    from rtx.kernels.nvfp4_rng import compile_philox_uniforms
    run=compile_philox_uniforms(count,stream)
    out=torch.empty(count,device="cuda")
    seed=torch.empty(2,device="cuda",dtype=torch.int64)
    for key in ((0,0),((1<<32)-1,(1<<32)-1),(941,826)):
        seed.copy_(torch.tensor(key,dtype=torch.int64))
        run(seed,out)
        expected=torch.tensor(uniforms(key,count,stream),device="cuda")
        torch.testing.assert_close(out,expected,atol=0,rtol=0)


@pytest.mark.parametrize("stream",[0,1])
@pytest.mark.parametrize("layout,rows,groups",[("row_major",129,17),("mma128",128,8)])
@pytest.mark.parametrize("packet_size", [1, 4])
def test_seeded_finalizer_matches_materialized_uniforms(stream,layout,rows,groups,packet_size):
    from rtx.kernels.nvfp4_rng import compile_philox_posthoc_scales
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig,compile_posthoc_scales
    config=PosthocScaleConfig(128,3,layout)
    explicit=compile_posthoc_scales(rows,groups,config)
    fused=compile_philox_posthoc_scales(rows,groups,config,1,stream,packet_size)
    key=(941,826)
    seed=torch.tensor(key,device="cuda",dtype=torch.int64)
    u=torch.tensor(uniforms(key,rows*groups,stream),device="cuda")
    out=torch.empty(rows*groups,device="cuda",dtype=torch.float8_e4m3fn)
    expected=torch.empty_like(out)
    outer=torch.empty(1,device="cuda")
    expected_outer=torch.empty_like(outer)
    # Same finite scale mathematics, covering zero and wide source magnitudes.
    for magnitude in (0.,1e-20,1.,1e20):
        src=(torch.linspace(.001,3.,rows*groups,device="cuda")*magnitude).bfloat16()
        maximum=src.float().amax().reshape(1)
        explicit(src,maximum,u,expected,expected_outer)
        fused(src,maximum,seed,out,outer)
        torch.testing.assert_close(out.view(torch.uint8),expected.view(torch.uint8),atol=0,rtol=0)
        torch.testing.assert_close(outer,expected_outer,atol=0,rtol=0)
        assert out.float().isfinite().all() and outer.isfinite().all()


def test_seeded_stochastic_rounding_frequency():
    from rtx.kernels.nvfp4_rng import compile_philox_posthoc_scales
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig
    count=32768
    # FP32 tensor factor is exactly one; 1.0625 is midway between E4M3 1/1.125.
    src=torch.full((count,),1.0625,device="cuda",dtype=torch.bfloat16)
    maximum=torch.tensor([448.],device="cuda")
    seed=torch.tensor([941,826],device="cuda",dtype=torch.int64)
    out=torch.empty(count,device="cuda",dtype=torch.float8_e4m3fn)
    outer=torch.empty(1,device="cuda")
    run=compile_philox_posthoc_scales(128,count//128,PosthocScaleConfig(),1,0)
    run(src,maximum,seed,out,outer)
    values=out.float()
    expected_high=torch.tensor([v<.5 for v in uniforms((941,826),count,0)],device="cuda")
    assert torch.equal(values==1.125,expected_high)
    assert torch.all((values==1.)|(values==1.125))
    assert abs(float((values==1.125).float().mean())-.5)<.015
    assert outer.item()==1.


@pytest.mark.parametrize("phase",["dx","dw"])
@pytest.mark.parametrize("magnitude",[0.,1e-10,1.,1e10])
def test_seeded_gradient_matches_independent_randomness(phase,magnitude):
    from rtx.experimental_ms_eden import ms_eden_gradient,ms_eden_seeded_gradient
    from tests.test_experimental_ms_eden import independent_reference
    torch.manual_seed(827)
    dy=torch.randn(128,256,device="cuda",dtype=torch.bfloat16)*(.25*magnitude)
    operand=torch.randn(256 if phase=="dx" else 128,128,device="cuda",dtype=torch.bfloat16)*(.02*magnitude)
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    key=(941,826)
    seed=torch.tensor(key,device="cuda",dtype=torch.int64)
    ua=torch.tensor(uniforms(key,dy.numel()//16,0),device="cuda")
    ub=torch.tensor(uniforms(key,operand.numel()//16,1),device="cuda")
    schedule=json.dumps(dict(producer_load="coalesced_wide",producer_blocks=256))
    args=(dy,operand,signs,ua,ub)
    expected=ms_eden_gradient(*args,phase=phase,schedule_json=schedule)
    actual=ms_eden_seeded_gradient(dy,operand,signs,seed,phase=phase,schedule_json=schedule)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert actual.isfinite().all()
    if magnitude==0:
        assert torch.count_nonzero(actual)==0
    elif magnitude==1.:
        oracle=independent_reference(args,phase)
        assert (actual.double()-oracle.double()).norm()/oracle.double().norm()<.005


@pytest.mark.parametrize("phase",["dx","dw"])
def test_seeded_fullgraph_replay_reads_current_seed(phase):
    from rtx.experimental_ms_eden import ms_eden_seeded_gradient
    dy=torch.randn(128,256,device="cuda",dtype=torch.bfloat16)*.25
    operand=torch.randn(256 if phase=="dx" else 128,128,device="cuda",dtype=torch.bfloat16)*.02
    signs=torch.ones(128,device="cuda")
    seed=torch.tensor([941,826],device="cuda",dtype=torch.int64)
    def call(a,b,s,k): return ms_eden_seeded_gradient(a,b,s,k,phase=phase)
    compiled=torch.compile(call,fullgraph=True,dynamic=False)
    expected=call(dy,operand,signs,seed)
    torch.testing.assert_close(compiled(dy,operand,signs,seed),expected,atol=0,rtol=0)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): compiled(dy,operand,signs,seed)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        out=compiled(dy,operand,signs,seed)
    graph.replay()
    first=out.clone()
    seed.copy_(torch.tensor([942,827],device="cuda"))
    graph.replay()
    expected=call(dy,operand,signs,seed)
    torch.testing.assert_close(out,expected,atol=0,rtol=0)
    assert not torch.equal(first,out)
    seed.copy_(torch.tensor([941,826],device="cuda"))
    graph.replay()
    torch.testing.assert_close(out,first,atol=0,rtol=0)
    torch.cuda.synchronize()
    before=torch.cuda.memory_allocated()
    for _ in range(20): graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before


def test_seeded_gradient_rejects_higher_order():
    from rtx.experimental_ms_eden import ms_eden_seeded_gradient
    dy=torch.ones(128,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    operand=torch.ones_like(dy,requires_grad=False)
    signs=torch.ones(128,device="cuda")
    seed=torch.tensor([941,826],device="cuda",dtype=torch.int64)
    out=ms_eden_seeded_gradient(dy,operand,signs,seed,phase="dx")
    with pytest.raises(RuntimeError,match="higher-order"):
        out.float().sum().backward()


def test_seeded_opcheck_and_seed_metadata():
    from rtx.experimental_ms_eden import _seeded_gradient,ms_eden_seeded_gradient
    dy=torch.ones(128,128,device="cuda",dtype=torch.bfloat16)
    operand=torch.ones_like(dy)
    signs=torch.ones(128,device="cuda")
    seed=torch.tensor([941,826],device="cuda",dtype=torch.int64)
    checked=torch.library.opcheck(_seeded_gradient,(dy,operand,signs,seed,"dx","{}"))
    assert set(checked.values())=={"SUCCESS"}
    invalid=(seed.to(torch.int32),seed[:1],torch.arange(4,device="cuda",dtype=torch.int64)[::2])
    for bad in invalid:
        with pytest.raises(ValueError,match="seed"):
            ms_eden_seeded_gradient(dy,operand,signs,bad,phase="dx")
    with pytest.raises(ValueError,match="share a device"):
        ms_eden_seeded_gradient(dy,operand,signs,seed.cpu(),phase="dx")


@pytest.mark.parametrize("phase",["dx","dw"])
@pytest.mark.parametrize("distribution",["student3","sparse","outliers"])
def test_seeded_non_gaussian_matches_same_draw_oracles(phase,distribution):
    from benchmarks.study_ms_eden_backward import study_operand
    from rtx.experimental_ms_eden import ms_eden_gradient,ms_eden_seeded_gradient
    from tests.test_experimental_ms_eden import independent_reference
    torch.manual_seed(828)
    dy=study_operand(128,256,.25,distribution)
    operand=study_operand(256 if phase=="dx" else 128,128,.02,distribution)
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    key=(0xFFFFFFFF,0x243F6A88)
    seed=torch.tensor(key,device="cuda",dtype=torch.int64)
    ua=torch.tensor(uniforms(key,dy.numel()//16,0),device="cuda")
    ub=torch.tensor(uniforms(key,operand.numel()//16,1),device="cuda")
    args=dy,operand,signs,ua,ub
    expected=ms_eden_gradient(*args,phase=phase)
    actual=ms_eden_seeded_gradient(dy,operand,signs,seed,phase=phase)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    oracle=independent_reference(args,phase)
    assert actual.isfinite().all() and oracle.isfinite().all()
    assert (actual.double()-oracle.double()).norm()/oracle.double().norm()<.005
