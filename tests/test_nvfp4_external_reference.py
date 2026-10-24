"""CPU-only format/oracle cross-checks against installed TorchAO 0.18.

These tests do NOT invoke RTX production kernels or establish CUDA parity.
TorchAO and RTX both clamp stored E4M3 scales to [2**-6, 448]. The official
Four Over Six reference instead allows E4M3 subnormals/zero, and reserves
outer-scale headroom: A/(6*256), not RTX's bounded variant A/(6*448).

Sources inspected 2026-10-23, pinned for reproducibility:
https://github.com/mit-han-lab/fouroversix/blob/dadfad6901d473a734fe71e0b082e70ee993e23a/src/fouroversix/quantize/pytorch/reference.py
https://github.com/mit-han-lab/fouroversix/blob/dadfad6901d473a734fe71e0b082e70ee993e23a/src/fouroversix/utils.py
https://github.com/IST-DASLab/Quartet-II/blob/5f2a47e8f464b17ef4b1cac78db8d14edd1f3f7d/kernels/csrc/round_four_six.cu

Local external oracle: torchao/prototype/mx_formats/nvfp4_tensor.py
(nvfp4_quantize) and kernels.py (FP4 encode/decode and nibble packing).
Quartet-II's CUDA path uses native nearest rounding, but its separate Triton
simulation rounds midpoint magnitudes upward and is not a bitwise RNE oracle.
"""

import unittest

import torch

from benchmarks.study_nvfp4_precision import fp4_round

try:
    from torchao.prototype.mx_formats.kernels import (
        f32_to_f4_unpacked, f4_unpacked_to_f32, pack_uint4, unpack_uint4,
    )
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
except ImportError:
    nvfp4_quantize = None


def external_candidate(blocks, outer, target):
    """TorchAO native-format primitives; no RTX quantizer/decomposition."""
    scale=(blocks.abs().amax(-1,keepdim=True)/(target*outer)).clamp(2**-6,448)
    scale=scale.to(torch.float8_e4m3fn).float()
    normalized=blocks*((1/outer)/scale)
    codes=f32_to_f4_unpacked(normalized)
    decoded=f4_unpacked_to_f32(codes)*(scale*outer)
    return scale,decoded,(blocks-decoded).square().sum(-1,keepdim=True)


@unittest.skipIf(nvfp4_quantize is None, 'TorchAO reference unavailable')
class NVFP4ExternalReferenceTests(unittest.TestCase):
    def test_e2m1_midpoints_saturation_and_packing(self):
        x=torch.tensor([[.25,.75,1.25,1.75,2.5,3.5,5.,100.,
                         -.25,-.75,-1.25,-1.75,-2.5,-3.5,-5.,-100.]])
        expected=torch.tensor([[0.,1.,1.,2.,2.,4.,4.,6.,
                                0.,-1.,-1.,-2.,-2.,-4.,-4.,-6.]])
        codes=f32_to_f4_unpacked(x)
        decoded=f4_unpacked_to_f32(codes)
        torch.testing.assert_close(decoded,expected,rtol=0,atol=0)
        torch.testing.assert_close(fp4_round(x),expected,rtol=0,atol=0)
        packed=pack_uint4(codes)
        torch.testing.assert_close(unpack_uint4(packed),codes,rtol=0,atol=0)
        self.assertTrue(torch.equal(packed,codes[:,::2] | (codes[:,1::2]<<4)))

    def test_native_scale_floor_and_ceiling(self):
        x=torch.tensor([[0.]*16,[1e-8]*16,[1e5]*16])
        scales,packed=nvfp4_quantize(x,per_tensor_scale=torch.tensor(1.))
        torch.testing.assert_close(scales.float()[:,0],
            torch.tensor([2**-6,2**-6,448.]),rtol=0,atol=0)
        decoded=f4_unpacked_to_f32(unpack_uint4(packed))*scales.float().repeat_interleave(16,-1)
        self.assertTrue(bool(decoded.isfinite().all()))
        self.assertEqual(float(decoded[:2].abs().max()),0.)

    def test_baseline_reference_distributions_ragged_and_outer_modes(self):
        generator=torch.Generator().manual_seed(741)
        for distribution in ('zero','normal','outliers','tiny'):
            x=torch.randn(7,33,generator=generator).bfloat16()
            if distribution=='zero': x.zero_()
            if distribution=='outliers': x[:,::17]*=20
            if distribution=='tiny': x*=1e-8
            padded=torch.nn.functional.pad(x.float(),(0,31)).contiguous()
            amax=padded.abs().amax()
            for mode in ('exact','power2'):
                outer=torch.where(amax==0,torch.ones_like(amax),amax/2688)
                if mode=='power2': outer=torch.exp2(torch.ceil(torch.log2(outer)))
                with self.subTest(distribution=distribution,mode=mode):
                    scales,packed=nvfp4_quantize(padded,per_tensor_scale=outer)
                    expected_scales=((padded.reshape(7,4,16).abs().amax(-1)/6)/outer)
                    expected_scales=expected_scales.clamp(2**-6,448).to(torch.float8_e4m3fn).float()
                    torch.testing.assert_close(scales.float(),expected_scales,rtol=0,atol=0)
                    scale=expected_scales.repeat_interleave(16,-1)
                    expected=fp4_round(padded*((1/outer)/scale))*(scale*outer)
                    actual=f4_unpacked_to_f32(unpack_uint4(packed))*(scale*outer)
                    self.assertTrue(bool(actual.isfinite().all()))
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    self.assertEqual(float(actual[:,33:].abs().max()),0.)

    def test_four_over_six_candidates_agree_with_external_fp4_rounding(self):
        blocks=torch.randn(100,16,generator=torch.Generator().manual_seed(62))
        blocks[0]=0
        blocks[1]*=1e-8
        blocks[2,::3]*=100
        outer=torch.tensor(2**-6)
        for target in (6.,4.):
            scale,decoded,error=external_candidate(blocks,outer,target)
            expected=fp4_round(blocks*((1/outer)/scale))*(scale*outer)
            torch.testing.assert_close(decoded,expected,rtol=0,atol=0)
            torch.testing.assert_close(error,(blocks-expected).square().sum(-1,keepdim=True),rtol=0,atol=0)

    def test_official_headroom_is_not_the_bounded_policy(self):
        blocks=torch.tensor([[6.]+[4.5]*15])
        amax=blocks.abs().amax()
        bounded=[external_candidate(blocks,amax/2688,target) for target in (6.,4.)]
        headroom=[external_candidate(blocks,amax/1536,target) for target in (6.,4.)]
        self.assertEqual(float(bounded[0][0]),448.)
        self.assertEqual(float(bounded[1][0]),448.)
        self.assertEqual(float(headroom[0][0]),256.)
        self.assertEqual(float(headroom[1][0]),384.)
        self.assertEqual(float(bounded[0][2]),3.75)
        self.assertEqual(float(bounded[1][2]),3.75)
        self.assertEqual(float(headroom[1][2]),0.)

    def test_headroom_study_zero_and_ragged_regions(self):
        from benchmarks.study_nvfp4_four_over_six_headroom import quantize
        for rows in (1,8):
            for policy in ('amax','bounded_4over6','headroom_4over6'):
                decoded,stats=quantize(torch.zeros(7,33),rows,policy)
                self.assertEqual(decoded.shape,(7,33))
                self.assertEqual(stats['sse'],0.)
                self.assertTrue(stats['finite'])
        x=torch.randn(7,33,generator=torch.Generator().manual_seed(82)).bfloat16()
        _,base=quantize(x,8,'amax')
        _,bounded=quantize(x,8,'bounded_4over6')
        self.assertLessEqual(bounded['sse'],base['sse'])


if __name__ == '__main__':
    unittest.main()
