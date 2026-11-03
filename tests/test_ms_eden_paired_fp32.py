"""Reuse full producer byte/graph/memory gates for the FP32x2 variant."""
import os
import pytest


@pytest.mark.skipif(os.getenv('RTX_RUN_PAIRED_FP32_CUDA')!='1',reason='assigned GPU opt-in')
@pytest.mark.parametrize('transpose',[False,True])
@pytest.mark.parametrize('vpl',[8,16])
@pytest.mark.parametrize('distribution',['gaussian','outliers','zero','wide_range'])
@pytest.mark.parametrize('mode',['statistics','transform','both'])
def test_exact_paired_statistics(monkeypatch,transpose,vpl,distribution,mode):
    from rtx.kernels import ms_eden_grouped_refine as refinement
    from rtx.kernels.ms_eden_paired_fp32 import compile_paired_fp32
    from tests.test_ms_eden_grouped_refine import test_exact_producer_and_changed_graph
    def compile_variant(*args,streamed=False,**kwargs):
        return compile_paired_fp32(*args,paired_transform=mode!='statistics',paired_statistics=mode!='transform',**kwargs)
    monkeypatch.setattr(refinement,'compile_grouped_refine',compile_variant)
    test_exact_producer_and_changed_graph(transpose,vpl,distribution,False)
