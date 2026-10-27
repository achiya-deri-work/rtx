"""Factor-pair ABI removes a launch, not an FP32 rounding operation."""
from contextlib import nullcontext
from dataclasses import asdict,replace
import json
import os
from types import SimpleNamespace

import pytest
import torch

from rtx.autotune.posthoc import (PosthocPreparationSchedule,make_posthoc_preparation_adapter,
    preparation_schedule_features)
from rtx.configs.nvfp4 import NVFP4GemmConfig,NVFP4Problem


def test_pair_schedule_identity_resources_and_invalid_modes():
    base=PosthocPreparationSchedule()
    candidate=replace(base,outer_factor_mode="pair")
    adapter=make_posthoc_preparation_adapter((512,512,512),"dx","mma128",lambda _:None)
    assert adapter.axes["outer_factor_mode"]==("product_kernel","pair")
    assert adapter.config_id(base)!=adapter.config_id(candidate)
    assert adapter.deserialize(adapter.serialize(candidate))==candidate
    assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(candidate))))==candidate
    a,b=(preparation_schedule_features((512,512,512),"dx",c) for c in (base,candidate))
    assert a["preparation_launches"]==b["preparation_launches"]+1
    assert a["scale_product_launches"]==1 and b["scale_product_launches"]==0
    assert a["packed_payload_bytes"]==b["packed_payload_bytes"]
    from rtx.kernels.nvfp4_gemm import NVFP4GemmKernel,NVFP4SplitWorkspaceReduceKernel
    for mode in (None,True,"folded"):
        with pytest.raises(ValueError,match="outer factor mode"):
            PosthocPreparationSchedule(outer_factor_mode=mode)
        with pytest.raises(ValueError,match="outer factor mode"):
            NVFP4GemmKernel(NVFP4Problem(128,128,128),NVFP4GemmConfig(),outer_factor_mode=mode)
        with pytest.raises(ValueError,match="outer factor mode"):
            NVFP4SplitWorkspaceReduceKernel(128,128,2,outer_factor_mode=mode)


def test_compiler_factor_shapes_and_cache_keys_are_distinct(monkeypatch):
    from rtx.kernels import nvfp4_gemm as kernels
    records=[]
    monkeypatch.setattr(kernels.cute.runtime,"make_fake_tensor",
        lambda dtype,shape,**kwargs:SimpleNamespace(dtype=dtype,shape=shape))
    def compile(kernel,*args,**kwargs):
        records.append((kernel,args[-2].shape))
        return object()
    monkeypatch.setattr(kernels.cute,"compile",compile)
    kernels.compile_nvfp4_gemm.cache_clear();kernels.compile_nvfp4_split_workspace_reduce.cache_clear()
    try:
        problem=NVFP4Problem(128,128,512)
        old=kernels.compile_nvfp4_gemm(problem)
        new=kernels.compile_nvfp4_gemm(problem,outer_factor_mode="pair")
        assert old is not new
        assert kernels.compile_nvfp4_gemm(problem,outer_factor_mode="pair") is new
        kernels.compile_nvfp4_split_workspace_reduce(128,128,2)
        kernels.compile_nvfp4_split_workspace_reduce(128,128,2,outer_factor_mode="pair")
        assert [shape for _,shape in records]==[(1,),(2,),(1,),(2,)]
    finally:
        kernels.compile_nvfp4_gemm.cache_clear();kernels.compile_nvfp4_split_workspace_reduce.cache_clear()


@pytest.mark.parametrize("mode",["product_kernel","pair"])
@pytest.mark.parametrize("split",[1,2])
def test_runtime_factor_views_lifetime_and_no_product_launch(monkeypatch,mode,split):
    import rtx.experimental_ms_eden as eden
    import rtx.fp4 as fp4
    from rtx.kernels import nvfp4_posthoc as posthoc,nvfp4_gemm as kernels
    from tests.test_ms_eden_split_integration import split_config
    events=[];slots=[];compiler_options=[];scales=[]
    monkeypatch.setattr(torch.cuda,"device",lambda _:nullcontext())
    monkeypatch.setattr(fp4,"_check_sm12x",lambda _:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda value:value)
    monkeypatch.setattr(posthoc,"compile_hadamard_preparation",lambda *a,**kw:((lambda *a:None),4))
    def finalizer(*args):
        def launch(*args):
            slots.append(args[-1]);args[-1].fill_(len(slots)+1)
        return launch
    monkeypatch.setattr(posthoc,"compile_posthoc_scales",finalizer)
    def product():
        events.append("compile_product")
        def launch(a,b,out):
            events.append("product");out.copy_(a*b)
        return launch
    monkeypatch.setattr(posthoc,"compile_posthoc_scale_product",product)
    def consumer(*args,**kwargs):
        compiler_options.append(kwargs)
        def launch(*args): scales.append(args[-1])
        return launch
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",consumer)
    monkeypatch.setattr(kernels,"compile_nvfp4_split_gemm",lambda *a,**kw:lambda *a:None)
    monkeypatch.setattr(kernels,"compile_nvfp4_split_workspace_reduce",consumer)
    source=torch.empty(512,128,dtype=torch.bfloat16)
    config=replace(PosthocPreparationSchedule(),outer_factor_mode=mode)
    if split>1: config=split_config(config,split)
    launch,out,live=eden._build_gradient(source,source,torch.ones(128),torch.zeros(source.numel()//16),
        torch.zeros(source.numel()//16),"dw",config,(128,128,512))
    launch()
    if mode=="pair":
        assert events==[]
        assert slots[0].untyped_storage().data_ptr()==slots[1].untyped_storage().data_ptr()
        assert slots[1].data_ptr()-slots[0].data_ptr()==4
        assert scales[0].tolist()==[2.,3.]
        assert any(value is scales[0] for value in live)
        assert all(any(value is slot for value in live) for slot in slots)
        assert compiler_options[0]["outer_factor_mode"]=="pair"
    else:
        assert events==["compile_product","product"] and scales[0].tolist()==[6.]
        assert "outer_factor_mode" not in compiler_options[0]


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_OUTER_FACTOR_PAIR_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")

FACTOR_PAIRS=[(1.0000001192092896,1.000000238418579),(2.**-100,2.**90),(2.**100,2.**-90),
    (2.**-126,.5),(2.**-126,2.**-23),(2.**-126,2.**-24),(0.,1.),(-0.,1.)]


@run_cuda
@pytest.mark.parametrize("epilogue",["direct","tma"])
def test_unsplit_direct_tma_exact_changed_factor_graph(epilogue):
    from rtx.kernels.nvfp4_gemm import compile_nvfp4_gemm
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scale_product
    from tests.test_nvfp4_split_k import gpu_operands
    problem=NVFP4Problem(128,128,512)
    config=NVFP4GemmConfig(epilogue=epilogue,store_vec=1 if epilogue=="direct" else 4,
        scale_layout="mma128",scale_role="tma")
    a,b=gpu_operands(problem,"mma128",exact=True)
    baseline=compile_nvfp4_gemm(problem,config)
    candidate=compile_nvfp4_gemm(problem,config,outer_factor_mode="pair")
    product=compile_posthoc_scale_product()
    factors=torch.ones(2,device="cuda");alpha=torch.empty(1,device="cuda")
    expected=torch.empty(128,128,device="cuda",dtype=torch.bfloat16);actual=torch.empty_like(expected)
    def run(): candidate(a[1],b[1],a[3],b[3],actual,factors)
    run();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for pair in FACTOR_PAIRS:
        factors.copy_(torch.tensor(pair,device="cuda"))
        product(factors[:1],factors[1:],alpha)
        baseline(a[1],b[1],a[3],b[3],expected,alpha);graph.replay()
        assert torch.equal(expected.view(torch.uint8),actual.view(torch.uint8))
        assert torch.isfinite(actual).all()


@run_cuda
@pytest.mark.parametrize("algorithm",["serial","tree","persistent_tree"])
def test_split_reducer_exact_with_extreme_and_subnormal_factor_products(algorithm):
    from rtx.kernels.nvfp4_gemm import compile_nvfp4_split_workspace_reduce
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scale_product
    torch.manual_seed(961)
    workspace=torch.randn(2*128*128,device="cuda")
    factors=torch.ones(2,device="cuda");alpha=torch.empty(1,device="cuda")
    expected=torch.empty(128,128,device="cuda",dtype=torch.bfloat16);actual=torch.empty_like(expected)
    baseline=compile_nvfp4_split_workspace_reduce(128,128,2,algorithm=algorithm)
    candidate=compile_nvfp4_split_workspace_reduce(128,128,2,algorithm=algorithm,outer_factor_mode="pair")
    product=compile_posthoc_scale_product()
    candidate(workspace,actual,factors);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):candidate(workspace,actual,factors)
    for pair in FACTOR_PAIRS:
        workspace.normal_()
        if pair[0]<=2.**-100 and pair[1]<1:
            workspace.mul_(2.**90) # Amplify subnormal factor products before the BF16 cast.
        factors.copy_(torch.tensor(pair,device="cuda"))
        product(factors[:1],factors[1:],alpha)
        baseline(workspace,expected,alpha);graph.replay()
        assert torch.equal(expected.view(torch.uint8),actual.view(torch.uint8))
        assert torch.isfinite(actual).all()


@run_cuda
@pytest.mark.parametrize("split",[1,2])
def test_public_pair_forward_rng_current_buffer_graph_and_memory_guards(split):
    from benchmarks.compare_ms_eden_pipeline import schedule_pair,options
    from benchmarks.compare_mixed_nvfp4_backward import run_worker
    from benchmarks.study_ms_eden_backward import run_study
    from tests.test_ms_eden_split_integration import split_config
    shape=(512,512,128)
    schedules={phase:replace(PosthocPreparationSchedule(**json.loads(encoded)),outer_factor_mode="pair")
        for phase,encoded in schedule_pair().items()}
    if split>1: schedules={phase:split_config(config,split) for phase,config in schedules.items()}
    encoded={phase:json.dumps(asdict(config)) for phase,config in schedules.items()}
    record,bundle=run_worker(options(shape,encoded,pipeline="paired"),capture_only=True)
    assert record["guards"]["forward_bitwise_matched"]
    assert record["guards"]["fresh_rng_per_phase"]==[True,True]
    assert record["memory"]["ordinary_compiled_allocation_audit"]["live_growth_bytes"]<=0
    assert record["memory"]["replay_live_growth_bytes"]==0
    assert all(torch.isfinite(value).all() for value in bundle["grads"])
    study=run_study(shape,"dw","mma128",benchmark=False,schedule=schedules["dw"])
    assert study["metrics"]["packed"]["nrms"]<=.005
