from dataclasses import replace
import pytest
import torch
import rtx.experimental_ms_eden as eden
from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
from rtx.configs.nvfp4 import NVFP4Problem


@pytest.mark.parametrize('requested,config_mode,expected',[
    ('exact',None,'exact'),('power2',None,'power2'),(None,None,'power2'),
    ('exact','power2','exact'),('power2','exact','power2'),
    (None,'exact','exact'),(None,'power2','power2'),
    (None,'exact_four_over_six','exact_four_over_six')])
def test_jit_mode_is_pinned_in_consumed_config(requested,config_mode,expected):
    import rtx.fp4 as fp4
    config=replace(preferred_jit_row_region_config(NVFP4Problem(128,128,128)),
        tensor_scale_mode=config_mode) if config_mode is not None else None
    mode=eden._resolve_outer_mode('original_bf16',requested,'jit_row_region',config)
    key=eden._jit_forward_request(128,128,128,config,mode)
    resolved=fp4._AUTOTUNE_REQUESTS[key].dynamic
    assert mode==resolved.tensor_scale_mode==expected
    if config is not None:
        assert config.tensor_scale_mode==config_mode


@pytest.mark.parametrize('source,expected',[
    ('original_bf16','power2'),('forward_current_roundtrip_bf16','exact')])
def test_tensor_current_defaults_unchanged(source,expected):
    assert eden._resolve_outer_mode(source,None,'current',None)==expected


def test_explicit_four_six_outer_stays_outside_tensor_contract():
    with pytest.raises(ValueError):
        eden._resolve_outer_mode('original_bf16','exact_four_over_six','current',None)


def test_four_six_config_reaches_matching_public_scale_policy(monkeypatch):
    import rtx.fp4 as fp4
    seen={}
    def fake(x,w,**kwargs):
        seen.update(kwargs)
        return x@w.T
    monkeypatch.setattr(fp4,'_nvfp4_dynamic_linear_with_keys',fake)
    x=torch.zeros(128,128,dtype=torch.bfloat16)
    config=replace(preferred_jit_row_region_config(NVFP4Problem(128,128,128)),
                   tensor_scale_mode='exact_four_over_six')
    eden.ms_eden_linear(x,x,forward_scaling='jit_row_region',forward_config=config)
    assert seen['tensor_scale_mode']=='exact_four_over_six'
    assert fp4._AUTOTUNE_REQUESTS[seen['materialized_request_key']].dynamic.tensor_scale_mode=='exact_four_over_six'
    assert eden._FORWARD_POLICIES['exact_four_over_six'].tensor_scale_mode=='exact_four_over_six'
