"""Byte-exact grouped reciprocal sharing, including inactive/zero groups."""
import os
import pytest


@pytest.mark.skipif(os.getenv('RTX_RUN_GROUPED_LUT_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('transpose',[False,True])
@pytest.mark.parametrize('vpl',[8,16])
@pytest.mark.parametrize('distribution',['gaussian','outliers','zero','wide_range'])
@pytest.mark.parametrize('streamed',[False,True])
def test_grouped_lut(monkeypatch,transpose,vpl,distribution,streamed):
    from rtx.kernels import ms_eden_grouped_refine as module
    from tests.test_ms_eden_grouped_refine import test_exact_producer_and_changed_graph
    original=module.compile_grouped_refine
    def compile_lut(*args,**kwargs):
        return original(*args,reciprocal_lut=True,**kwargs)
    monkeypatch.setattr(module,'compile_grouped_refine',compile_lut)
    test_exact_producer_and_changed_graph(transpose,vpl,distribution,streamed)
