import pytest
import torch
from benchmarks.train_quartet2_decoder import ResearchBackwardLinear, RESEARCH_POLICIES


@pytest.mark.parametrize('mode', RESEARCH_POLICIES)
def test_research_decoder_wrapper_preserves_weight_and_shape(monkeypatch,mode):
    import rtx.experimental_ms_eden as experimental
    seen={}
    def fake(x,w,**kwargs):
        seen.update(kwargs)
        return x@w.T
    monkeypatch.setattr(experimental,'ms_eden_linear',fake)
    linear=torch.nn.Linear(128,256,bias=False)
    wrapper=ResearchBackwardLinear(linear,mode)
    value=torch.randn(2,3,128)
    assert wrapper.weight is linear.weight
    torch.testing.assert_close(wrapper(value),linear(value))
    assert seen['source_policy']=='original_bf16'
    assert seen['forward_outer_mode']=='exact'
    assert seen['forward_scaling']=='jit_row_region'
    assert (seen['dx_policy'],seen['dw_policy'])==RESEARCH_POLICIES[mode]
    assert (seen['mxfp8_backward_config'] is None)==(mode=='research_fp4_fp4')


def test_research_decoder_rejects_bias():
    with pytest.raises(ValueError):
        ResearchBackwardLinear(torch.nn.Linear(128,128),'research_mx_mx')
