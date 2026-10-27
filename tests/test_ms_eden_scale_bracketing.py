"""CPU proof for a proposed positive-scale E4M3 bracketing implementation.

Not a kernel implementation or GPU qualification. The reference obtains
neighbors by actual FP8 round-to-nearest conversion, independently of FP32 bits.
"""
import torch


def test_positive_e4m3_brackets_from_fp32_bits_match_conversion():
    codes=torch.arange(127,dtype=torch.uint8)
    levels=codes.view(torch.float8_e4m3fn).float()
    midpoints=(levels[:-1]+levels[1:])*.5
    edges=torch.cat((levels,midpoints))
    gen=torch.Generator().manual_seed(982)
    samples=torch.exp2(torch.rand(100000,generator=gen)*150-140)
    values=torch.cat((edges,torch.nextafter(edges,torch.full_like(edges,float('inf'))),
                      torch.nextafter(edges,torch.full_like(edges,-float('inf'))),samples))
    values=values.clamp(0,448)
    nearest_codes=values.to(torch.float8_e4m3fn).view(torch.uint8).int()
    nearest=values.to(torch.float8_e4m3fn).float()
    expected_low=nearest_codes-(nearest>values).int()
    expected_high=nearest_codes+(nearest<values).int()

    bits=values.view(torch.int32)
    normal_codes=(bits>>20)-960
    subnormal_codes=torch.trunc(values*512).int()
    low_codes=torch.where(values>=.015625,normal_codes,subnormal_codes)
    normal_low=((low_codes+960)<<20).view(torch.float32)
    low=torch.where(low_codes>=8,normal_low,low_codes.float()/512)
    high_codes=low_codes+(low<values).int()
    normal_high=((high_codes+960)<<20).view(torch.float32)
    high=torch.where(high_codes>=8,normal_high,high_codes.float()/512)

    torch.testing.assert_close(low_codes,expected_low,rtol=0,atol=0)
    torch.testing.assert_close(high_codes,expected_high,rtol=0,atol=0)
    ref_low=expected_low.byte().view(torch.float8_e4m3fn).float()
    ref_high=expected_high.byte().view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(low,ref_low,rtol=0,atol=0)
    torch.testing.assert_close(high,ref_high,rtol=0,atol=0)
    mask=high>low
    probability=torch.zeros_like(values)
    reference=torch.zeros_like(values)
    probability[mask]=(values[mask]-low[mask])/(high[mask]-low[mask])
    reference[mask]=(values[mask]-ref_low[mask])/(ref_high[mask]-ref_low[mask])
    assert torch.equal(probability,reference)
