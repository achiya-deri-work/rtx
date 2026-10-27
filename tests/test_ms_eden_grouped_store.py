"""Vector stores must preserve every output byte and graph lifetime."""
import os
import pytest


def test_store_width_contract():
    from rtx.kernels.ms_eden_grouped_store import GroupedStorePreparationKernel
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig
    for width in (16,32):
        kernel=GroupedStorePreparationKernel(256,384,PosthocScaleConfig(),4,False,producer_values_per_lane=8,store_bits=width)
        assert kernel.grouped_store_bits==width and not kernel.grouped_refinement
    for width in (8,64,128,True):
        with pytest.raises(ValueError,match="store width"):
            GroupedStorePreparationKernel(256,384,PosthocScaleConfig(),4,False,producer_values_per_lane=8,store_bits=width)


def test_integrated_autotune_axis_identity_and_effective_width():
    from dataclasses import replace
    from rtx.autotune.posthoc import PosthocPreparationSchedule,make_posthoc_preparation_adapter
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig
    adapter=make_posthoc_preparation_adapter((512,512,128),"dx","mma128",lambda _:None)
    base=PosthocPreparationSchedule()
    assert base.producer_grouped_store_bits==8
    assert adapter.axes["producer_grouped_store_bits"]==(8,16,32,64)
    candidate=adapter.update_fn(base,"producer_grouped_store_bits",64)
    assert candidate.producer_transpose_values_per_lane==16
    assert candidate.producer_load=="coalesced_x8"
    assert adapter.rejection(candidate) is None
    assert adapter.deserialize(adapter.serialize(candidate))==candidate
    assert adapter.config_id(base)!=adapter.config_id(candidate)
    features=adapter.extra_features_fn(candidate)
    assert features["a_producer_grouped_store_bits"]==8
    assert features["b_producer_grouped_store_bits"]==64
    for vpl,width in ((4,8),(8,32),(16,64)):
        kernel=HadamardPreparationKernel(256,384,PosthocScaleConfig(),4,False,
            producer_values_per_lane=vpl,producer_grouped_store_bits=64)
        assert kernel.grouped_store_bits==width
    for width in (0,128,True):
        with pytest.raises(ValueError,match="store width"):
            replace(base,producer_grouped_store_bits=width)


@pytest.mark.skipif(os.getenv("RTX_RUN_GROUPED_STORE_CUDA")!="1",reason="assigned GPU opt-in")
@pytest.mark.parametrize("transpose,vpl,bits",[(False,8,32),(False,16,64),(True,8,32),(True,16,16),(True,16,32),(True,16,64)])
@pytest.mark.parametrize("distribution",["gaussian","outliers","zero","wide_range"])
def test_exact_vector_store(monkeypatch,transpose,vpl,bits,distribution):
    # Reuse the existing independent scalar producer / changed-source graph /
    # allocation test. Only its candidate compiler is replaced, never baseline.
    from tests.test_ms_eden_grouped_refine import test_exact_producer_and_changed_graph
    from rtx.kernels import ms_eden_grouped_refine as candidate
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation
    def compile_store(*args,**kwargs):
        return compile_hadamard_preparation(*args,producer_grouped_store_bits=bits,**kwargs)
    monkeypatch.setattr(candidate,"compile_grouped_refine",compile_store)
    test_exact_producer_and_changed_graph(transpose,vpl,distribution)
