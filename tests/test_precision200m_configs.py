import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from benchmarks import train_precision20m as study
from rtx.models import DecoderConfig


def test_explicit_configs_roundtrip_and_exact_coverage(tmp_path):
    config=DecoderConfig(hidden_size=128,intermediate_size=256,num_attention_heads=4,
        num_layers=1,max_seq_len=16,mlp_activation='situ')
    model=study.build_model(config,'nvfp4_mxfp8',8)
    modules=[m for m in model.modules() if isinstance(m,study.PrecisionProjection)]
    fwd={','.join(map(str,m.shape)):asdict(m.forward_config) for m in modules}
    bwd={','.join(map(str,m.shape)):asdict(m.backward_config) for m in modules}
    fp,bp=tmp_path/'fwd.json',tmp_path/'bwd.json'
    fp.write_text(json.dumps(fwd));bp.write_text(json.dumps(bwd))
    args=SimpleNamespace(mode='nvfp4_mxfp8',forward_configs=fp,mxfp8_backward_configs=bp)
    study.apply_projection_configs(model,args)
    assert all(m.forward_config.gemm.tile_m>0 for m in modules)
    assert all(m.backward_config is not None for m in modules)
    fp.write_text('{}')
    with pytest.raises(ValueError):study.apply_projection_configs(model,args)
    for value in fwd.values(): value['tensor_scale_mode']='power2'
    fp.write_text(json.dumps(fwd))
    with pytest.raises(ValueError,match='exact FP32'):study.apply_projection_configs(model,args)


def test_tuned_backward_is_really_dispatched(monkeypatch):
    import rtx
    original=torch.nn.Linear(128,128,bias=False,dtype=torch.bfloat16)
    module=study.PrecisionProjection(original,'nvfp4_mxfp8',128,1)
    marker=object()
    module.backward_config=marker
    def fake(x,w,**kwargs):
        assert kwargs['backward_config'] is marker
        return x@w.T
    monkeypatch.setattr(rtx,'nvfp4_linear',fake)
    assert module(torch.zeros(128,128,dtype=torch.bfloat16)).shape==(128,128)
