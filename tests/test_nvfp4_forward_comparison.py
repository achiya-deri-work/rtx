import json
from dataclasses import asdict,replace

import pytest
import torch
from rtx.configs.nvfp4 import NVFP4DynamicConfig
from benchmarks.compare_nvfp4_forward import load_config,replay_check


@pytest.mark.parametrize('fault',[None,'shape','geometry'])
def test_forward_config_contract(tmp_path,fault):
    cfg=replace(NVFP4DynamicConfig(),x_scale_region_rows=8,weight_scale_region_rows=8)
    if fault=='geometry': cfg=replace(cfg,x_scale_region_rows=0)
    value={'shape':[128,256,384],'config':asdict(cfg)}
    if fault=='shape': value['shape'][0]=256
    path=tmp_path/'winner.json'; path.write_text(json.dumps(value))
    if fault:
        with pytest.raises(ValueError): load_config(path,(128,256,384))
    else:
        assert load_config(path,(128,256,384))==cfg


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_changed_input_replay_restores_operands():
    x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)
    w=torch.randn_like(x)
    a,b=x.clone(),w.clone()
    result=replay_check(lambda:x@w.T,x,w)
    assert all(result.values())
    torch.testing.assert_close(x,a,rtol=0,atol=0)
    torch.testing.assert_close(w,b,rtol=0,atol=0)


def test_current_policy_cannot_load_regional_winner(tmp_path):
    cfg=NVFP4DynamicConfig()
    path=tmp_path/'winner.json'
    path.write_text(json.dumps({'shape':[128,256,384],'config':asdict(cfg),'scaling':'current'}))
    assert load_config(path,(128,256,384),'current')==cfg
    with pytest.raises(ValueError): load_config(path,(128,256,384))
    value=json.loads(path.read_text()); value['scaling']='delayed'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): load_config(path,(128,256,384),'current')
