"""Exact-division and complete-producer gates for bounded direct refinement."""
import os
import pytest


def test_direct_refine_cpu_power_scaling_equivalence():
    """Independent binary64 residual model over guard endpoints/significands."""
    import numpy as np
    mantissa=np.arange(4096,dtype=np.uint32)*np.uint32(2048)
    for exponent in (-30,0,30):
        for relative in (-32,0,3):
            words=mantissa|np.uint32((127+relative+exponent)<<23)
            for sign in (0,0x80000000):
                x=(words|np.uint32(sign)).view(np.float32)
                for numerator in range(16,32):
                    scale=np.ldexp(np.float32(numerator/16),exponent,dtype=np.float32)
                    reciprocal=np.divide(np.float32(1.),scale,dtype=np.float32)
                    estimate=np.multiply(x,reciprocal,dtype=np.float32)
                    residual=x.astype(np.float64)-estimate.astype(np.float64)*np.float64(scale)
                    np.testing.assert_array_equal(residual.astype(np.float32).astype(np.float64),residual)
                    actual=(residual*np.float64(reciprocal)+estimate.astype(np.float64)).astype(np.float32)
                    expected=np.divide(x,scale,dtype=np.float32)
                    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


@pytest.mark.skipif(os.getenv('RTX_RUN_DIRECT_REFINE_CUDA')!='1',reason='assigned GPU only')
def test_direct_refine_division_probe(monkeypatch):
    import numpy as np
    import test_ms_eden_normalization_refine as original
    import rtx.kernels.nvfp4_posthoc as module
    from rtx.kernels.ms_eden_direct_refine import DirectRefinePreparationKernel
    monkeypatch.setattr(module,'HadamardPreparationKernel',DirectRefinePreparationKernel)
    edge=np.float32(2.**-32)
    original.test_gpu_refinement_matches_independent_explicit_division_all_denominators_and_fallbacks(
        scale_exponents=(-126,-90,-31,-30,-1,0,1,30,31,90,126,127),
        normalized_extra=(edge,-edge,np.nextafter(edge,np.float32(0.)),np.nextafter(edge,np.float32(1.))))


@pytest.mark.skipif(os.getenv('RTX_RUN_DIRECT_REFINE_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('transpose',[False,True])
@pytest.mark.parametrize('vpl',[8,16])
@pytest.mark.parametrize('distribution',['gaussian','outliers','zero','wide_range'])
def test_direct_refine_producer(monkeypatch,transpose,vpl,distribution):
    import test_ms_eden_grouped_refine as original
    import rtx.kernels.ms_eden_grouped_refine as module
    from rtx.kernels.ms_eden_direct_refine import compile_direct_refine
    def compile(*args,streamed=False,**kwargs):
        return compile_direct_refine(*args,**kwargs)
    monkeypatch.setattr(module,'compile_grouped_refine',compile)
    original.test_exact_producer_and_changed_graph(transpose,vpl,distribution,True)
