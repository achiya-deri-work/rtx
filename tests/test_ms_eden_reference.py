import pytest
import torch
from benchmarks.reference_ms_eden import fp4_rne, prepare_extended, scale_brackets, finalize_scales


def test_fp4_ties_even_and_saturation():
    x = torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.,100.])
    expected = torch.tensor([0.,1.,1.,2.,2.,4.,4.,6.])
    torch.testing.assert_close(fp4_rne(x),expected,rtol=0,atol=0)
    torch.testing.assert_close(fp4_rne(-x),-expected,rtol=0,atol=0)


@pytest.mark.parametrize("bits", [3,4])
def test_extended_correction_inner_product_and_scale_only_finalization(bits):
    torch.manual_seed(701)
    x = torch.randn(32,128)*.25
    prepared = prepare_extended(x,mantissa_bits=bits)
    blocks = x.reshape(32,8,16)
    reconstructed = prepared.values*prepared.fp32_corrected_scales.unsqueeze(-1)
    # EDEN's conditional inner-product identity is NOT vector unbiasedness.
    torch.testing.assert_close((blocks*reconstructed).sum(-1), blocks.square().sum(-1),rtol=3e-6,atol=1e-7)
    before = prepared.values.clone()
    scales,outer = finalize_scales(prepared.corrected_scales,torch.rand(32,8))
    assert torch.equal(before,prepared.values)
    assert torch.isfinite(scales.float()).all() and outer.dtype == torch.float32
    assert prepared.corrected_scales.numel()*16 == x.numel()


def test_scale_sr_has_exact_conditional_expectation_including_subnormals():
    values = torch.tensor([0.,2**-12,2**-9,.003,1.03,1.95,10.3,430.,448.])
    lo,hi,p = scale_brackets(values)
    torch.testing.assert_close(lo*(1-p)+hi*p,values,rtol=1e-7,atol=1e-9)
    assert ((p >= 0) & (p <= 1)).all()


def test_zero_and_explicit_rng_replay():
    zero = prepare_extended(torch.zeros(2,32))
    scales,outer = finalize_scales(zero.corrected_scales,torch.zeros(2,2))
    assert outer == 1 and not scales.float().any()
    x = prepare_extended(torch.arange(32,dtype=torch.float32).reshape(1,32))
    u = torch.tensor([[.1,.9]])
    a,s = finalize_scales(x.corrected_scales,u)
    b,t = finalize_scales(x.corrected_scales,u)
    assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)) and s == t
    with pytest.raises(ValueError):
        finalize_scales(x.corrected_scales,torch.ones_like(u))


@pytest.mark.parametrize("magnitude", [1e-20,1e-8,1.,1e8,1e20])
def test_extended_range_remains_finite(magnitude):
    torch.manual_seed(702)
    prepared = prepare_extended(torch.randn(4,128)*magnitude)
    scales, outer = finalize_scales(prepared.corrected_scales,torch.full((4,8),.5))
    assert torch.isfinite(scales.float()).all() and torch.isfinite(outer)
    assert (prepared.corrected_scales > 0).all()


def test_subnormal_temporary_scales_explicitly_flush_to_zero():
    prepared = prepare_extended(torch.full((1,16),torch.finfo(torch.float32).tiny))
    assert not prepared.values.any()
    assert not prepared.corrected_scales.any()


def test_three_and_four_bit_temporary_policies_are_distinct():
    torch.manual_seed(703)
    x=torch.randn(32,128)
    a,b=(prepare_extended(x,mantissa_bits=bits) for bits in (3,4))
    assert not torch.equal(a.values,b.values)
