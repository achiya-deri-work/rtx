"""Research split-K schedule/runtime contracts and opt-in public GPU guards."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import os

import pytest
import torch

from rtx.autotune.posthoc import (PosthocPreparationSchedule,
    make_posthoc_preparation_adapter, preparation_schedule_features)


def split_config(base=None, split=2):
    base = base or PosthocPreparationSchedule()
    return replace(base, gemm_split_reduction=split,
        gemm=replace(base.gemm,epilogue="direct",store_vec=1,epilogue_stages=1,
            persistent_waves=0,tiles_per_cta=1,regional_epilogue_schedule="mma"))


def test_split_geometry_axes_identity_resources_and_inactive_knobs():
    adapter = make_posthoc_preparation_adapter((4096,512,512), "dw", "mma128", lambda _: None)
    base = adapter.initial_config
    assert base.gemm_split_reduction == 1 and base.gemm_reduction_tile(4096) == 0
    candidate = adapter.update_fn(base, "gemm_split_reduction", 2)
    assert candidate == split_config(base)
    assert candidate.gemm_reduction_tile(4096) == 2048
    assert adapter.rejection(candidate) is None
    assert adapter.deserialize(adapter.serialize(candidate)) == candidate
    assert adapter.config_id(candidate) != adapter.config_id(base)
    assert adapter.config_id(base) == adapter.config_id(replace(base,gemm_reduction_threads=64))
    assert adapter.config_id(candidate) == adapter.config_id(replace(candidate,gemm_reduction_waves=8))
    assert adapter.config_id(candidate) != adapter.config_id(replace(candidate,gemm_reduction_algorithm="serial"))
    features = preparation_schedule_features((4096,512,512), "dw", candidate)
    assert features["gemm_output_tiles"] == 16 and features["gemm_split_ctas"] == 32
    assert features["gemm_workspace_bytes"] == 2*512*512*4
    assert features["gemm_reduction_launches"] == 1 and features["gemm_launches"] == 2
    assert features["preparation_and_gemm_scratch_bytes"] == (
        features["preparation_scratch_bytes"] + features["gemm_workspace_bytes"])
    assert adapter.context.tags["schedule_features_revision"] == 22
    assert candidate.operand_schedule("a").gemm_split_reduction == 2
    assert candidate.operand_schedule("b").gemm_split_reduction == 2
    assert split_config(split=4).gemm_reduction_tile(896) == 256
    with pytest.raises(ValueError, match="empty"):
        split_config(split=4).gemm_reduction_tile(640)
    assert adapter.rejection(replace(base,gemm_split_reduction=2))[0] == "implementation_rejected"


@pytest.mark.parametrize("changes", [dict(gemm_split_reduction=True),dict(gemm_split_reduction=3),
    dict(gemm_reduction_algorithm="atomic"),dict(gemm_reduction_threads=True),
    dict(gemm_reduction_vector=16),dict(gemm_reduction_waves=0)])
def test_invalid_split_controls_rejected(changes):
    with pytest.raises(ValueError): PosthocPreparationSchedule(**changes)


def test_static_split_contract_rejects_incompatible_epilogue_before_launch():
    import rtx.experimental_ms_eden as eden
    dy = torch.empty(512,256,dtype=torch.bfloat16)
    operand = torch.empty(512,128,dtype=torch.bfloat16)
    signs = torch.ones(128)
    ua, ub = torch.zeros(dy.numel()//16), torch.zeros(operand.numel()//16)
    invalid = replace(PosthocPreparationSchedule(),gemm_split_reduction=2)
    with pytest.raises(ValueError, match="direct epilogue"):
        eden._contract(dy,operand,signs,ua,ub,"dw",json.dumps(asdict(invalid)))
    valid = split_config()
    schedule, shape = eden._contract(dy,operand,signs,ua,ub,"dw",json.dumps(asdict(valid)))
    assert schedule == valid and shape == (256,128,512)


def test_runtime_keeps_workspace_live_and_scales_only_final_reducer(monkeypatch):
    import rtx.fp4 as fp4
    import rtx.experimental_ms_eden as eden
    import rtx.kernels.nvfp4_posthoc as preparation
    import rtx.kernels.nvfp4_gemm as gemm
    calls = []
    monkeypatch.setattr(torch.cuda,"device",lambda device:nullcontext())
    monkeypatch.setattr(fp4,"_check_sm12x",lambda device:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda value:value)
    monkeypatch.setattr(preparation,"compile_hadamard_preparation",lambda *args,**kwargs:((lambda *args:None),4))
    monkeypatch.setattr(preparation,"compile_posthoc_scales",lambda *args,**kwargs:lambda *args:None)
    monkeypatch.setattr(preparation,"compile_posthoc_scale_product",lambda:lambda *args:calls.append(("factor",args)))
    def partial(problem, config, **kwargs):
        assert kwargs == dict(split_reduction=2,reduction_tile=128)
        return lambda *args:calls.append(("partial",args))
    def final(m,n,split,**kwargs):
        assert (m,n,split)==(128,128,2)
        assert kwargs == dict(algorithm="tree",threads=256,vector=4,persistent_waves=4)
        return lambda *args:calls.append(("final",args))
    monkeypatch.setattr(gemm,"compile_nvfp4_split_gemm",partial)
    monkeypatch.setattr(gemm,"compile_nvfp4_split_workspace_reduce",final)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *args:pytest.fail("split path used unsplit GEMM"))
    dy = torch.empty(256,128,dtype=torch.bfloat16)
    signs, uniforms = torch.ones(128), torch.zeros(dy.numel()//16)
    launch, out, live = eden._build_gradient(dy,dy,signs,uniforms,uniforms,"dw",split_config(),(128,128,256))
    launch()
    assert [name for name,_ in calls] == ["factor","partial","final"]
    partial_args, final_args = calls[1][1], calls[2][1]
    workspace = partial_args[-1]
    assert len(partial_args)==5 and len(final_args)==3
    assert workspace.dtype==torch.float32 and workspace.numel()==2*128*128
    assert any(item is workspace for item in live)
    assert final_args[0] is workspace and final_args[1] is out
    assert final_args[2] is calls[0][1][-1]


run_cuda = pytest.mark.skipif(os.environ.get("RTX_RUN_MS_EDEN_SPLIT_CUDA") != "1",
                              reason="explicit assigned-GPU integration qualification")


@run_cuda
@pytest.mark.parametrize("split",[2,4])
def test_public_compiled_paired_split_forward_rng_graph_and_memory_guards(split):
    from benchmarks.compare_ms_eden_pipeline import schedule_pair, options
    from benchmarks.compare_mixed_nvfp4_backward import run_worker
    from benchmarks.study_ms_eden_backward import run_study
    shape=(512,512,128)
    pair=schedule_pair()
    schedules={phase:split_config(PosthocPreparationSchedule(**json.loads(encoded)),split)
               for phase,encoded in pair.items()}
    encoded={phase:json.dumps(asdict(config)) for phase,config in schedules.items()}
    record,bundle=run_worker(options(shape,encoded,pipeline="paired"),capture_only=True)
    assert record["guards"]["forward_bitwise_matched"]
    assert record["guards"]["fresh_rng_per_phase"]==[True,True]
    memory=record["memory"]
    assert memory["ordinary_compiled_allocation_audit"]["live_growth_bytes"]<=0
    assert memory["replay_live_growth_bytes"]==0
    assert all(torch.isfinite(value).all() for value in bundle["grads"])
    # Independent packed-operand gate remains .005, not the broader Gaussian
    # original-source quantization/stale-buffer gate inherited above.
    study=run_study(shape,"dw","mma128",benchmark=False,schedule=schedules["dw"])
    assert study["metrics"]["packed"]["nrms"]<=.005
    assert study["gemm_workspace_bytes"]==split*shape[1]*shape[2]*4
