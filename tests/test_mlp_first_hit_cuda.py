from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import rtx

from tests.test_mxfp8_mlp import reference


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_runtime_execution_mix_isolates_disk_winners(tmp_path):
    from rtx._mlp_autotune import SELECTIONS
    import rtx.autotune.mlp as tuning
    torch.manual_seed(942)
    policy=rtx.AutotunePolicy(max_trials=1,max_seconds=60,warmup=1,samples=3,
        calls_per_sample=1,confirmation_repeats=0,finalist_rounds=0)
    mixes=[rtx.MLPExecutionMix(inference_forwards=n) for n in (1,2)]
    layers=[rtx.MXFP8MLP(128,128,device='cuda',autotune='balanced',
        tuning_policy=policy,autotune_cache_dir=tmp_path,training_mix=m) for m in mixes]
    layers[1].load_state_dict(layers[0].state_dict())
    x=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
    dy=torch.randn_like(x)*.1
    functions=[torch.compile(layer,fullgraph=True,dynamic=False,
        options={'triton.cudagraphs':False}) for layer in layers]
    def run(i):
        y=functions[i](x)
        return y,*torch.autograd.grad(y,(x,*layers[i].parameters()),dy)
    original=tuning.tune_mxfp8_mlp
    observed=[]
    def tune(*args,**kwargs):
        observed.append(kwargs['execution_mix'])
        return original(*args,**kwargs)
    with patch.object(tuning,'tune_mxfp8_mlp',side_effect=tune):
        actual=[run(i) for i in range(2)]
    assert observed==mixes
    expected_y,expected_grads=reference(x,layers[0].gate_up_weight,layers[0].down_weight,dy,'silu')
    for values in actual:
        for a,b in zip(values,(expected_y,*expected_grads)):
            assert a.isfinite().all()
            assert (a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)<.025
    journals={str(p):p.read_bytes() for p in tmp_path.rglob('observations.jsonl')}
    assert len(journals)==2
    SELECTIONS.clear()
    with patch.object(tuning,'tune_mxfp8_mlp',side_effect=AssertionError('disk winner retuned')):
        for i in range(2):
            for a,b in zip(run(i),actual[i]):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert journals=={str(p):p.read_bytes() for p in tmp_path.rglob('observations.jsonl')}


@pytest.mark.parametrize("mode", ["cache", "balanced"])
def test_compiled_mlp_selection_and_refinement(tmp_path, mode, monkeypatch):
    from rtx._mlp_autotune import SELECTIONS, BACKWARDS
    from torch._inductor.compile_fx import compile_fx
    import rtx.autotune.mlp as tuning
    tune_calls=[]
    original_tune=tuning.tune_mxfp8_mlp
    def record_tune(*args,**kwargs):
        tune_calls.append(kwargs.copy())
        return original_tune(*args,**kwargs)
    monkeypatch.setattr(tuning,'tune_mxfp8_mlp',record_tune)
    torch.manual_seed(932)
    policy = rtx.AutotunePolicy(max_trials=8, max_seconds=90, search_schedule="progressive",
        cache_behavior="refine", finalist_rounds=2, confirmation_repeats=0,
        warmup=1, samples=3, calls_per_sample=2)
    layer = rtx.MXFP8MLP(128,128,device="cuda",autotune=mode,
        tuning_policy=policy,autotune_cache_dir=tmp_path)
    x = (torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_()
    dy = torch.randn_like(x)*.1
    inputs = x, layer.gate_up_weight, layer.down_weight
    graphs=[]
    def backend(gm, args):
        graphs.append(gm)
        return compile_fx(gm,args)
    fn=torch.compile(layer,fullgraph=True,dynamic=False,backend=backend)
    def run():
        y=fn(x)
        return y,*torch.autograd.grad(y,inputs,dy)
    def journals():
        return {str(p):p.read_bytes() for p in Path(tmp_path).rglob("observations.jsonl")}
    actual=run()
    with torch.no_grad():
        expected_y,expected_grads=reference(*inputs,dy,"silu")
    for name,a,b in zip(("y","dx","dgu","ddown"),actual,(expected_y,*expected_grads)):
        assert a.isfinite().all()
        delta=a.float()-b.float()
        rms=delta.square().mean().sqrt().item()
        ref=b.float().square().mean().sqrt().item()
        print("MLP_FIRST_HIT",json.dumps(dict(mode=mode,output=name,rms=rms,reference_rms=ref,
            max_abs=delta.abs().max().item(),normalized_rms=rms/max(ref,1e-12))),flush=True)
        assert rms/max(ref,1e-12)<.025
    before=journals()
    if mode=="balanced":
        assert sum(len(v.splitlines()) for v in before.values())==8
    else:
        assert not before
    again=run()
    for a,b in zip(actual,again):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert len(graphs)==1 and journals()==before
    SELECTIONS.clear()
    with patch("rtx.autotune.mlp.tune_mxfp8_mlp",side_effect=AssertionError("cache hit retuned")):
        run()
    assert len(graphs)==1 and journals()==before
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    from rtx._mlp_autotune import selected_mlp
    # Fresh capture leaves avoid AccumulateGrad nodes retained by the earlier
    # default-stream comparisons. All uses of these leaves share this stream.
    with torch.cuda.stream(stream):
        capture_inputs=tuple(t.detach().requires_grad_() for t in inputs)
    capture_fn=torch.compile(lambda *v:selected_mlp(*v,"silu",layer._first_hit_spec,True)[0],
                             fullgraph=True,dynamic=False)
    def capture_run():
        y=capture_fn(*capture_inputs)
        return y,*torch.autograd.grad(y,capture_inputs,dy)
    with torch.cuda.stream(stream):
        capture_run()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        captured=capture_run()
    with torch.cuda.stream(stream):
        for _ in range(3):graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    for a,b in zip(actual,captured):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert len(graphs)==1 and journals()==before
    if mode=="balanced":
        from rtx._mlp_autotune import deferred_spec
        layer._first_hit_spec=deferred_spec(layer.config,"mxfp8",mode,replace(policy,max_trials=16),tmp_path)
        # Explicitly new policy: a fresh compiled wrapper may specialize once.
        fn=torch.compile(layer,fullgraph=True,dynamic=False)
        run()
        assert sum(len(v.splitlines()) for v in journals().values())==16
        assert len(journals())==1
        assert len(tune_calls)==2
        assert tune_calls[0]['initial']==tune_calls[1]['initial']
        assert tune_calls[1]['portfolio'] == tune_calls[0]['portfolio'] == ()


@pytest.mark.parametrize("mask", [(True,False,False),(False,True,False),(False,False,True)])
def test_ragged_partial_gradients_and_opcheck(tmp_path,mask):
    from rtx._mlp_autotune import selected_mlp, deferred_spec
    torch.manual_seed(933)
    config=replace(rtx.MXFP8MLPConfig(),backward_recompute="two_pass")
    spec=deferred_spec(config,"mxfp8","cache",None,tmp_path)
    inputs=tuple((torch.randn(*s,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_(need)
                 for s,need in zip(((17,35),(194,35),(41,97)),mask))
    dy=torch.randn(41,17,device="cuda",dtype=torch.bfloat16).T
    fn=torch.compile(lambda x,g,d:selected_mlp(x,g,d,"gelu",spec,True)[0],fullgraph=True,dynamic=False)
    y=fn(*inputs)
    requested=tuple(t for t in inputs if t.requires_grad)
    actual=torch.autograd.grad(y,requested,dy)
    with torch.no_grad():
        ref_y,ref_grads=reference(*inputs,dy,"gelu")
    for a,b in zip((y,*actual),(ref_y,*(g for g,need in zip(ref_grads,mask) if need))):
        assert a.isfinite().all()
        assert float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))<.025
    torch.library.opcheck(selected_mlp,(*inputs,"gelu",spec,True))


@pytest.mark.parametrize("recompute,overlap", [("two_pass",False),("joint",True)])
def test_disjoint_gradients_retention_and_memory(tmp_path,recompute,overlap):
    from rtx._mlp_autotune import selected_mlp, deferred_spec
    torch.manual_seed(934)
    config=replace(rtx.MXFP8MLPConfig(),backward_recompute=recompute,backward_overlap=overlap)
    spec=deferred_spec(config,"mxfp8","cache",None,tmp_path)
    inputs=tuple((torch.randn(*shape,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_()
                 for shape in ((64,128),(512,128),(64,256)))
    dy=torch.randn(64,64,device="cuda",dtype=torch.bfloat16)*.1
    fn=torch.compile(lambda *v:selected_mlp(*v,"silu",spec,True)[0],fullgraph=True,dynamic=False)
    def run():
        y=fn(*inputs)
        return y,*torch.autograd.grad(y,inputs,dy)
    with torch.no_grad():
        ref_y,ref_grads=reference(*inputs,dy,"silu")
    first=tuple(t.detach() for t in run())
    snapshot=tuple(t.clone() for t in first)
    for a,b in zip(first,(ref_y,*ref_grads)):
        assert a.isfinite().all()
        assert float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))<.025
    for _ in range(3):run()
    assert len({t.untyped_storage()._cdata for t in first})==4
    for a,b in zip(first,snapshot):torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.cuda.synchronize()
    live=[]
    for _ in range(4):
        for _ in range(5):run()
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    print("MLP_DIRECT_OUTPUT_MEMORY",recompute,overlap,live,flush=True)
    assert max(live)==min(live)
