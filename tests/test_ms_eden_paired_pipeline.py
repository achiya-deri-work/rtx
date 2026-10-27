"""Paired orchestration contracts; CPU tests are not CUDA qualification."""
import itertools
import json
import os
from contextlib import contextmanager, nullcontext
import threading

import pytest
import torch

import rtx.experimental_ms_eden as eden


def inputs(device="meta"):
    dy=torch.empty(128,256,device=device,dtype=torch.bfloat16)
    x=torch.empty(128,384,device=device,dtype=torch.bfloat16)
    w=torch.empty(256,384,device=device,dtype=torch.bfloat16)
    ds=torch.empty(128,device=device,dtype=torch.float32)
    ws=torch.empty_like(ds)
    return (dy,x,w,ds,torch.empty(dy.numel()//16,device=device),
            torch.empty(w.numel()//16,device=device),ws,
            torch.empty(dy.numel()//16,device=device),torch.empty(x.numel()//16,device=device))


@pytest.mark.parametrize("pipeline",["serial","paired","operand_paired"])
def test_fake_and_fullgraph_metadata(pipeline):
    args=inputs()
    def fn(*values):
        return eden.ms_eden_paired_gradient(*values,pipeline=pipeline)
    for result in (fn(*args),torch.compile(fn,backend="eager",fullgraph=True)(*args)):
        assert [v.shape for v in result]==[torch.Size((128,384)),torch.Size((256,384))]
        assert all(v.dtype==torch.bfloat16 and v.device.type=="meta" for v in result)


@pytest.mark.parametrize("dx_seeded,dw_seeded",list(itertools.product((False,True),repeat=2)))
def test_rng_consumption_matches_independent_serial(monkeypatch,dx_seeded,dw_seeded):
    dy,x,w,*_=inputs("cpu")
    schedules=[json.dumps({"rng_mode":"philox"}) if seeded else "{}"
               for seeded in (dx_seeded,dw_seeded)]
    seen=[]
    def capture(*values,**kwargs):
        seen.append((values,kwargs))
        return torch.empty_like(x),torch.empty_like(w)
    monkeypatch.setattr(eden,"ms_eden_paired_gradient",capture)
    state=torch.get_rng_state()
    eden._random_paired_gradient(dy,x,w,*schedules,"paired")
    paired_state=torch.get_rng_state()
    values,kwargs=seen.pop()
    assert kwargs["dx_seeded"]==dx_seeded
    assert kwargs["dw_seeded"]==dw_seeded
    torch.set_rng_state(state)
    expected=[]
    def single(dy,operand,signs,a,b,**kwargs):
        expected.append((signs,a,b))
        return torch.empty_like(x)
    def seeded(dy,operand,signs,key,**kwargs):
        return single(dy,operand,signs,key,key,**kwargs)
    monkeypatch.setattr(eden,"ms_eden_gradient",single)
    monkeypatch.setattr(eden,"ms_eden_seeded_gradient",seeded)
    eden._random_gradient(dy,w,"dx",schedules[0])
    eden._random_gradient(dy,x,"dw",schedules[1])
    assert torch.equal(paired_state,torch.get_rng_state())
    for actual,reference in zip(values[3:],(*expected[0],*expected[1])):
        torch.testing.assert_close(actual,reference,atol=0,rtol=0)


@pytest.mark.parametrize("need_x,need_w,dx_policy,dw_policy",[
    (True,True,"ms_eden","ms_eden"),(True,False,"ms_eden","ms_eden"),
    (False,True,"ms_eden","ms_eden"),(True,True,"mxfp8","ms_eden"),
    (True,True,"ms_eden","mxfp8"),
])
@pytest.mark.parametrize("pipeline",["paired","operand_paired"])
def test_pair_admission_and_partial_fallback(monkeypatch,need_x,need_w,dx_policy,dw_policy,pipeline):
    import rtx.fp4 as fp4
    monkeypatch.setattr(fp4,"nvfp4_linear",lambda x,w,**kw:x@w.T)
    calls=[]
    def pair(dy,x,w,*args):
        calls.append("paired")
        return dy@w,dy.T@x
    def single(dy,x,w,phase,*args):
        calls.append(phase)
        return dy@w if phase=="dx" else dy.T@x
    monkeypatch.setattr(eden,"_random_paired_gradient",pair)
    monkeypatch.setattr(eden,"_linear_gradient",single)
    x=torch.randn(128,128,dtype=torch.bfloat16,requires_grad=need_x)
    w=torch.randn_like(x,requires_grad=need_w)
    out=eden.ms_eden_linear(x,w,backward_pipeline=pipeline,dx_policy=dx_policy,dw_policy=dw_policy)
    out.sum().backward()
    paired=need_x and need_w and dx_policy==dw_policy=="ms_eden"
    assert calls==(["paired"] if paired else (["dx"] if need_x else [])+(["dw"] if need_w else []))


def test_invalid_pipeline_and_incompatible_shapes():
    with pytest.raises(ValueError,match="pipeline"):
        eden.ms_eden_paired_gradient(*inputs(),pipeline="auto")
    x=torch.empty(128,128,dtype=torch.bfloat16)
    with pytest.raises(ValueError,match="backward_pipeline"):
        eden.ms_eden_linear(x,x,backward_pipeline="auto")
    args=list(inputs())
    args[2]=torch.empty(256,512,dtype=torch.bfloat16,device="meta")
    args[5]=torch.empty(args[2].numel()//16,device="meta")
    with pytest.raises(RuntimeError):
        eden.ms_eden_paired_gradient(*args)


@pytest.mark.parametrize("failure",[None,"dx","dw"])
def test_stream_fork_join_and_lifetime_even_on_failure(monkeypatch,failure):
    calls=[]
    class Stream:
        def __init__(self,name):
            self.name=name
        def wait_stream(self,other):
            calls.append(("wait",self.name,other.name))
    caller,side=Stream("caller"),Stream("side")
    @contextmanager
    def side_context(stream):
        assert stream is side
        calls.append("enter")
        yield
        calls.append("exit")
    class Storage:
        def record_stream(self,stream):
            assert stream is side
            calls.append("record")
    monkeypatch.setattr(torch.cuda,"current_stream",lambda device:caller)
    monkeypatch.setattr(torch.cuda,"stream",side_context)
    monkeypatch.setattr(eden,"_paired_stream",lambda device,caller:(side,threading.Lock()))
    def run(phase):
        calls.append(phase)
        if failure==phase:
            raise RuntimeError(phase)
    if failure:
        with pytest.raises(RuntimeError,match=failure):
            eden._launch_paired(lambda:run("dx"),lambda:run("dw"),[Storage(),Storage()],None)
    else:
        eden._launch_paired(lambda:run("dx"),lambda:run("dw"),[Storage(),Storage()],None)
    assert calls[:3]==[("wait","side","caller"),"record","record"]
    assert calls[-1]==("wait","caller","side")
    assert calls.index("dw")<calls.index("dx") if failure!="dw" else "dx" not in calls


def test_side_stream_cache_is_bounded_and_capture_isolated(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(eden,"_PAIRED_STREAMS",{})
    monkeypatch.setattr(eden,"_PAIRED_STREAMS_PER_DEVICE",2)
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:False)
    created=[]
    def make_stream(device):
        stream=object()
        created.append(stream)
        return stream
    monkeypatch.setattr(torch.cuda,"Stream",make_stream)
    device=torch.device("cuda",0)
    a=eden._paired_stream(device,SimpleNamespace(cuda_stream=100))
    b=eden._paired_stream(device,SimpleNamespace(cuda_stream=101))
    assert a[0] is not b[0]
    assert eden._paired_stream(device,SimpleNamespace(cuda_stream=100)) is a
    assert eden._paired_stream(device,SimpleNamespace(cuda_stream=102)) is None
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:True)
    assert eden._paired_stream(device,SimpleNamespace(cuda_stream=100)) is a
    with pytest.raises(RuntimeError,match="warm paired"):
        eden._paired_stream(torch.device("cuda",1),SimpleNamespace(cuda_stream=100))
    assert len(created)==2


def test_plan_compiles_before_launch_and_tracks_all_storage(monkeypatch):
    import rtx.fp4 as fp4
    import rtx.kernels.nvfp4_posthoc as posthoc
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    calls=[]
    launched_tensors=[]
    def compile_kernel(name):
        calls.append(("compile",name))
        def launch(*args):
            calls.append(("launch",name))
            launched_tensors.extend(v for v in args if isinstance(v,torch.Tensor))
        return launch
    monkeypatch.setattr(torch.cuda,"device",lambda device:nullcontext())
    monkeypatch.setattr(fp4,"_check_sm12x",lambda device:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda value:value)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *args:compile_kernel("gemm"))
    monkeypatch.setattr(posthoc,"compile_hadamard_preparation",lambda *args,**kwargs:(compile_kernel("producer"),4))
    monkeypatch.setattr(posthoc,"compile_posthoc_scales",lambda *args:compile_kernel("finalizer"))
    monkeypatch.setattr(posthoc,"compile_posthoc_scale_product",lambda:compile_kernel("product"))
    dy,x,w,signs,ua,_,_,_,ub=inputs("cpu")
    plan,out,live=eden._build_gradient(dy,x,signs,ua,ub,"dw",PosthocPreparationSchedule(),(256,384,128))
    assert all(kind=="compile" for kind,name in calls)
    assert out.shape==(256,384)
    assert all(any(value is owned for owned in live) for value in (dy,x,signs,ua,ub,out))
    def no_allocation(*args,**kwargs):
        raise AssertionError("launch chain allocated")
    monkeypatch.setattr(torch,"empty",no_allocation)
    plan()
    assert [name for kind,name in calls if kind=="launch"]==[
        "producer","finalizer","producer","finalizer","product","gemm"]
    # Views may differ as Python objects, but all launched buffers must have
    # a tracked storage owner for cross-stream allocator lifetime recording.
    owners={value.untyped_storage().data_ptr() for value in live}
    assert all(value.untyped_storage().data_ptr() in owners for value in launched_tensors)


requires_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_PAIRED_CUDA")!="1",
                               reason="requires exclusive authorized SM12x GPU")


@requires_cuda
@pytest.mark.parametrize("dx_seeded,dw_seeded",list(itertools.product((False,True),repeat=2)))
def test_cuda_fixed_rng_serial_pair_capture_and_opcheck(dx_seeded,dw_seeded):
    args=list(inputs("cuda"))
    for value in args[:3]:
        value.normal_()
    for index in (3,6):
        args[index].fill_(1)
    for index in (4,5,7,8):
        args[index].uniform_()
    for seeded,a,b in ((dx_seeded,4,5),(dw_seeded,7,8)):
        if seeded:
            args[a]=torch.tensor([12345,67890],device="cuda",dtype=torch.int64)
            args[b]=args[a]
    kwargs=dict(dx_seeded=dx_seeded,dw_seeded=dw_seeded)
    serial=eden.ms_eden_paired_gradient(*args,pipeline="serial",**kwargs)
    paired=eden.ms_eden_paired_gradient(*args,pipeline="paired",**kwargs)
    for a,b in zip(serial,paired):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.library.opcheck(eden._paired_gradient,(*args,"{}","{}",dx_seeded,dw_seeded,"paired"))
    compiled=torch.compile(lambda *v:eden.ms_eden_paired_gradient(*v,pipeline="paired",**kwargs),fullgraph=True)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled(*args)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        captured=compiled(*args)
    graph.replay()
    for a,b in zip(serial,captured):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    args[0].mul_(.5)
    graph.replay()
    changed=eden.ms_eden_paired_gradient(*args,pipeline="serial",**kwargs)
    for a,b in zip(changed,captured):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


@requires_cuda
@pytest.mark.parametrize("compiled",[False,True])
@pytest.mark.parametrize("checkpointed",[False,True])
def test_cuda_public_same_draw_checkpoint_and_allocation_growth(compiled,checkpointed):
    from torch.utils.checkpoint import checkpoint
    from tests.cache_memory_checks import wait_for_cuda_allocation_cleanup
    x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    w=torch.randn_like(x,requires_grad=True)
    dy=torch.randn_like(x)
    schedules=dict(dx_schedule_json='{"rng_mode":"philox"}',dw_schedule_json="{}")
    def evaluate(pipeline):
        def fn(a,b):
            return eden.ms_eden_linear(a,b,backward_pipeline=pipeline,**schedules)
        call=torch.compile(fn,fullgraph=True) if compiled else fn
        def run():
            # Reentrant checkpoint requires backward rather than autograd.grad.
            x.grad=None; w.grad=None
            out=checkpoint(call,x,w,use_reentrant=True) if checkpointed else call(x,w)
            out.backward(dy)
            return out,x.grad,w.grad
        return run
    baseline,candidate=evaluate("serial"),evaluate("paired")
    torch.manual_seed(3456)
    serial=baseline()
    expected_state=torch.cuda.get_rng_state()
    torch.manual_seed(3456)
    paired=candidate()
    assert torch.equal(expected_state,torch.cuda.get_rng_state())
    for a,b in zip(serial,paired):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    del serial,paired
    # Explicit release and sync outside timing. Retained x/w grads are cleared.
    for _ in range(4):
        result=candidate()
        del result
        x.grad=None; w.grad=None
    torch.cuda.synchronize()
    before=torch.cuda.memory_allocated()
    for _ in range(20):
        result=candidate()
        del result
        x.grad=None; w.grad=None
    torch.cuda.synchronize()
    samples=wait_for_cuda_allocation_cleanup(before,observed=torch.cuda.memory_allocated())
    assert samples[-1]<=before


@requires_cuda
def test_cuda_independent_caller_streams_concurrent_captures():
    from concurrent.futures import ThreadPoolExecutor
    args=list(inputs("cuda"))
    for value in args[:3]:
        value.normal_()
    for index in (3,6):
        args[index].fill_(1)
    for index in (4,5,7,8):
        args[index].uniform_()
    baseline=eden.ms_eden_paired_gradient(*args,pipeline="serial")
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            eden.ms_eden_paired_gradient(*args,pipeline="paired")
        stream.synchronize()
    barrier=threading.Barrier(2)
    def capture(stream):
        graph=torch.cuda.CUDAGraph()
        barrier.wait()
        with torch.cuda.graph(graph,stream=stream,capture_error_mode="thread_local"):
            output=eden.ms_eden_paired_gradient(*args,pipeline="paired")
        return graph,output
    with ThreadPoolExecutor(max_workers=2) as pool:
        bundles=list(pool.map(capture,streams))
    for graph,output in bundles:
        graph.replay()
        for expected,actual in zip(baseline,output):
            torch.testing.assert_close(expected,actual,atol=0,rtol=0)
