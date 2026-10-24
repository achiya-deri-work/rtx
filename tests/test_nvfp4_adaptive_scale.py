"""Explicit adaptive E4M3 block-scale policy; independent E2M1 decoder."""
from dataclasses import replace
import unittest

import torch

import rtx
from rtx.configs.nvfp4 import NVFP4QuantConfig
from rtx.nvfp4_inference_autotune import (
    dynamic_config_from_dict, dynamic_config_to_dict, preferred_jit_row_region_config,
    update_dynamic_config,
)


def decode(value):
    packed = value.qdata.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=packed.device)
    decoded = levels[(codes & 7).long()] * torch.where(codes < 8, 1., -1.)
    return decoded * (value.scale.float().repeat_interleave(16, dim=-1) * value.per_tensor_scale)


def oracle_block_error(padded, outer, policy):
    from benchmarks.study_nvfp4_precision import fp4_round
    blocks=padded.reshape(-1,16)
    raw=blocks.abs().amax(-1,keepdim=True)/(6*outer)
    baseline=raw.clamp(2**-6,448).to(torch.float8_e4m3fn).float()
    def reconstruct(scale):
        return fp4_round(blocks*((1/outer)/scale))*(scale*outer)
    base=reconstruct(baseline)
    best=(blocks-base).square().sum(-1)
    if policy=='mse5':
        scales=[(raw*f).clamp(2**-6,448).to(torch.float8_e4m3fn).float()
                for f in (.75,.875,1.125,1.25)]
    elif policy=='four_over_six':
        scales=[(blocks.abs().amax(-1,keepdim=True)/(4*outer)).clamp(2**-6,448)
                .to(torch.float8_e4m3fn).float()]
    else:
        codes=fp4_round(blocks*((1/outer)/baseline))
        fitted=(blocks*codes).sum(-1,keepdim=True)/codes.square().sum(-1,keepdim=True).clamp_min(1e-20)
        scales=[(fitted/outer).clamp(2**-6,448).to(torch.float8_e4m3fn).float()]
    for scale in scales:
        best=torch.minimum(best,(blocks-reconstruct(scale)).square().sum(-1))
    return best.reshape(padded.shape[0],-1)


def four_over_six_oracle(blocks, outer):
    """Independent nearest-level oracle, including native scale saturation.

    Even E2M1 code magnitudes come first, so argmin resolves exact midpoint
    ties to even without reusing the kernel's nibble decoder or PTX packing.
    """
    levels=blocks.new_tensor([0.,1.,2.,4.,.5,1.5,3.,6.])
    candidates=[]
    for target in (6.,4.):
        scale=(blocks.abs().amax(-1,keepdim=True)/(target*outer)).clamp(2**-6,448)
        scale=scale.to(torch.float8_e4m3fn).float()
        normalized=blocks*((1/outer)/scale)
        nearest=(normalized.abs().unsqueeze(-1)-levels).abs().argmin(-1)
        reconstructed=levels[nearest]*normalized.sign()*(scale*outer)
        error=(blocks-reconstructed).square().sum(-1,keepdim=True)
        candidates.append((scale,reconstructed,error))
    six,four=candidates
    use_four=four[2]<six[2]
    return tuple(torch.where(use_four,b,a) for a,b in zip(six,four))


class AdaptiveScaleConfigTests(unittest.TestCase):
    def test_explicit_headroom_outer_mode_is_independent_and_serialized(self):
        self.assertEqual(rtx.NVFP4ScaleConfig().tensor_scale_mode,'power2')
        self.assertEqual(rtx.NVFP4ScaleConfig(tensor_scale_mode='exact').tensor_scale_mode,'exact')
        self.assertEqual(rtx.NVFP4ScaleConfig(tensor_scale_mode='exact_four_over_six').tensor_scale_mode,
                         'exact_four_over_six')
        problem=rtx.NVFP4Problem(128,128,128)
        base=preferred_jit_row_region_config(problem)
        for policy in ('amax','four_over_six'):
            config=replace(base,tensor_scale_mode='exact_four_over_six',
                quant=replace(base.quant,block_scale_selection=policy))
            self.assertIsNone(config.rejection(problem))
            self.assertEqual(dynamic_config_from_dict(dynamic_config_to_dict(config)),config)
            self.assertEqual(update_dynamic_config(config,{'quant':{'num_warps':4}}).tensor_scale_mode,
                             'exact_four_over_six')
            separate=replace(config,gemm=replace(config.gemm,regional_scale_epilogue='separate'))
            self.assertIn('power-of-two',separate.rejection(problem))
        with self.assertRaises(ValueError):
            rtx.NVFP4ScaleConfig(tensor_scale_mode='wrong')

    def test_policy_is_explicit_and_serialized(self):
        self.assertEqual(NVFP4QuantConfig().block_scale_selection, 'amax')
        cfg = preferred_jit_row_region_config(rtx.NVFP4Problem(128,128,128))
        for policy in ('amax','mse5','lsq1','four_over_six'):
            selected = replace(cfg,quant=replace(cfg.quant,block_scale_selection=policy))
            self.assertEqual(dynamic_config_from_dict(dynamic_config_to_dict(selected)),selected)
            updated = update_dynamic_config(selected,{'quant':{'num_warps':4}})
            self.assertEqual(updated.quant.block_scale_selection,policy)
        self.assertIsNotNone(NVFP4QuantConfig(block_scale_selection='wrong').rejection(128,128))
        self.assertIsNotNone(NVFP4QuantConfig(block_scale_selection='mse5',scale_reciprocal='rcp_approx').rejection(128,128))
        for reciprocal in ('e4m3_lut','rcp_approx'):
            self.assertIsNotNone(NVFP4QuantConfig(block_scale_selection='four_over_six',scale_reciprocal=reciprocal).rejection(128,128))

    def test_four_over_six_oracle_native_scales_and_ties(self):
        blocks=torch.tensor([[6.]+[4.5]*15, [6.]+[0.]*15, [0.]*16])
        scale,decoded,error=four_over_six_oracle(blocks,torch.tensor([1.]))
        torch.testing.assert_close(scale[:,0],torch.tensor([1.5,1.,2**-6]),rtol=0,atol=0)
        torch.testing.assert_close(decoded,blocks,rtol=0,atol=0)
        self.assertEqual(float(error.sum()),0.)

    def test_four_over_six_oracle_saturation_and_padded_tails(self):
        # Exact outer scaling gives the peak block no amax/4 headroom:
        # raw candidates 448 and 672 both encode 448, retaining the baseline.
        blocks=torch.tensor([[2688.]+[2016.]*15, [1920.]+[1440.]*15,
                             [1e-8,-1e-8]+[0.]*14, [0.]*16])
        outer=torch.tensor([1.])
        scale,decoded,error=four_over_six_oracle(blocks,outer)
        self.assertEqual(float(scale[0]),448.)
        self.assertTrue(bool(decoded.isfinite().all()))
        self.assertEqual(float(decoded[-2:].abs().max()),0.)
        # A partially clipped candidate is evaluated after encoding, too.
        expected=oracle_block_error(blocks,outer,'four_over_six')
        torch.testing.assert_close(error[:,0],expected[:,0],rtol=0,atol=0)

    def test_default_winner_promotion_and_sampling_cannot_change_policy(self):
        from rtx.autotune.adapters import make_nvfp4_jit_row_region_adapter
        from rtx.autotune.outcomes import TrialOutcome
        from rtx.autotune.promotion import _config_rejection
        from rtx.fp4 import _default_runtime_schedule_rejection
        p=rtx.NVFP4Problem(128,128,128)
        base=preferred_jit_row_region_config(p)
        for policy in ('mse5','lsq1','four_over_six'):
            adaptive=replace(base,quant=replace(base.quant,block_scale_selection=policy))
            self.assertIn('explicit config',_default_runtime_schedule_rejection(adaptive,p))
            self.assertIn('explicit config',_config_rejection('nvfp4_jit_row_region_fwd',dynamic_config_to_dict(adaptive),p))
            adapter=make_nvfp4_jit_row_region_adapter(p,lambda _: TrialOutcome('ok',median_ms=1.),initial=adaptive)
            self.assertIsNotNone(adapter.rejection(base))
            self.assertIsNone(adapter.rejection(adaptive))
        headroom=replace(base,tensor_scale_mode='exact_four_over_six')
        self.assertIn('explicit config',_default_runtime_schedule_rejection(headroom,p))
        self.assertIn('explicit config',_config_rejection('nvfp4_jit_row_region_fwd',dynamic_config_to_dict(headroom),p))
        adapter=make_nvfp4_jit_row_region_adapter(p,lambda _: TrialOutcome('ok',median_ms=1.),initial=headroom)
        self.assertIsNotNone(adapter.rejection(base))
        self.assertIsNone(adapter.rejection(headroom))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class AdaptiveScaleCudaTests(unittest.TestCase):
    def test_explicit_headroom_regional_quantization_matches_oracle(self):
        from benchmarks.study_nvfp4_precision import fp4_round
        from rtx.kernels.nvfp4_quant import compile_nvfp4_jit_region_dual_quant
        x=torch.zeros(7,33,device='cuda',dtype=torch.bfloat16)
        w=torch.zeros(5,33,device='cuda',dtype=torch.bfloat16)
        for value in (x,w):
            value[:,:16]=torch.tensor([6.]+[4.5]*15,device='cuda')
            value[:,-1]=-3.
        x[2:4]=0
        w[3:]=0
        x[4:]*=2
        for policy in ('amax','four_over_six'):
            for owner in ('warp','cta','cta_cached'):
                with self.subTest(policy=policy,owner=owner):
                    cfg=NVFP4QuantConfig(block_scale_selection=policy,adaptive_reuse_packed=True)
                    kernel=compile_nvfp4_jit_region_dual_quant(7,5,33,cfg,2,3,
                        'exact_four_over_six',region_ownership=owner)
                    qx=torch.empty(7,32,device='cuda',dtype=torch.uint8)
                    qw=torch.empty(5,32,device='cuda',dtype=torch.uint8)
                    sx=torch.empty(7,4,device='cuda',dtype=torch.float8_e4m3fn)
                    sw=torch.empty(5,4,device='cuda',dtype=torch.float8_e4m3fn)
                    factors=torch.empty(6,device='cuda',dtype=torch.float32)
                    kernel(x,w,qx,qw,sx,sw,factors)
                    offset=0
                    for value,packed,scales,region_rows in ((x,qx,sx,2),(w,qw,sw,3)):
                        padded=torch.nn.functional.pad(value.float(),(0,31))
                        for begin in range(0,value.shape[0],region_rows):
                            end=min(begin+region_rows,value.shape[0])
                            region=padded[begin:end]
                            maximum=region.abs().amax()
                            outer=torch.where(maximum==0,torch.ones_like(maximum),
                                maximum*torch.tensor(1/1536,device='cuda',dtype=torch.float32))
                            torch.testing.assert_close(factors[offset],outer,rtol=0,atol=0)
                            offset+=1
                            blocks=region.reshape(-1,16)
                            if policy=='four_over_six':
                                expected_scale,expected,_=four_over_six_oracle(blocks,outer)
                            else:
                                expected_scale=(blocks.abs().amax(-1,keepdim=True)/(6*outer))
                                expected_scale=expected_scale.clamp(2**-6,448).to(torch.float8_e4m3fn).float()
                                expected=fp4_round(blocks*((1/outer)/expected_scale))*(expected_scale*outer)
                            torch.testing.assert_close(scales[begin:end].float(),
                                expected_scale.reshape(end-begin,4),rtol=0,atol=0)
                            codes=torch.stack((packed[begin:end]&15,packed[begin:end]>>4),-1).flatten(1)
                            levels=region.new_tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
                            actual=levels[(codes&7).long()]*torch.where(codes<8,1.,-1.)
                            actual*=scales[begin:end].float().repeat_interleave(16,-1)*outer
                            torch.testing.assert_close(actual,expected.reshape_as(region),rtol=0,atol=0)
                    self.assertEqual(factors.dtype,torch.float32)

    def test_four_over_six_native_saturation_zero_and_ragged_tail(self):
        x=torch.zeros(7,33,device='cuda',dtype=torch.bfloat16)
        x[0,:16]=torch.tensor([6.]+[4.5]*15,device='cuda')
        x[1,0]=6.
        x[3,:16]=torch.tensor([2688.]+[2016.]*15,device='cuda')
        x[4,:16]=torch.tensor([1920.]+[1440.]*15,device='cuda')
        x[5,:2]=torch.tensor([1e-8,-1e-8],device='cuda')
        x[6,-1]=-6.
        padded=torch.nn.functional.pad(x.float(),(0,31))
        for outer_value in (1.,1.125):
            outer=torch.tensor([outer_value],device='cuda',dtype=torch.float32)
            expected_scale,expected,_=four_over_six_oracle(padded.reshape(-1,16),outer)
            for vpl in (2,4,8,16):
                for reuse in (False,True):
                    with self.subTest(outer=outer_value,vpl=vpl,reuse=reuse):
                        result=rtx.quantize_nvfp4(x,tensor_scale=outer,
                            config=NVFP4QuantConfig(values_per_lane=vpl,
                                block_scale_selection='four_over_six',adaptive_reuse_packed=reuse))
                        torch.testing.assert_close(decode(result),expected.reshape_as(padded),rtol=0,atol=0)
                        torch.testing.assert_close(result.scale.float(),expected_scale.reshape(7,4),rtol=0,atol=0)
                        self.assertEqual(result.per_tensor_scale.dtype,torch.float32)
                        torch.testing.assert_close(result.per_tensor_scale.reshape(()),outer.reshape(()),rtol=0,atol=0)

    def test_reconstruction_never_worsens_block_sse(self):
        for m,k,distribution in ((7,33,'normal'),(128,128,'normal'),
                                 (257,385,'normal'),(128,128,'tiny'),
                                 (128,128,'outliers')):
            g = torch.Generator(device='cuda').manual_seed(924)
            x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16,generator=g)
            if distribution == 'tiny': x = x * 1e-5
            if distribution == 'outliers': x[:,::17] *= 20
            x[0] = 0
            exponent = -25 if distribution == 'tiny' else -5 if distribution == 'outliers' else -9
            outer = torch.tensor([2.0**exponent],device='cuda')
            padded = torch.nn.functional.pad(x.float(), (0,(-k)%32))
            for vpl in (2,8,16):
                cfg = NVFP4QuantConfig(values_per_lane=vpl)
                base = decode(rtx.quantize_nvfp4(x,tensor_scale=outer,config=cfg))
                err0 = (base-padded).square().reshape(m,-1,16).sum(-1)
                for policy in ('mse5','lsq1','four_over_six'):
                    with self.subTest(shape=(m,k),distribution=distribution,vpl=vpl,policy=policy):
                        result = rtx.quantize_nvfp4(x,tensor_scale=outer,
                            config=replace(cfg,block_scale_selection=policy))
                        y = decode(result)
                        err = (y-padded).square().reshape(m,-1,16).sum(-1)
                        self.assertTrue(bool(y.isfinite().all()))
                        self.assertTrue(bool((err <= err0 + 1e-5 * err0 + 1e-8).all()))
                        expected=oracle_block_error(padded,outer,policy)
                        torch.testing.assert_close(err,expected,rtol=2e-5,atol=1e-7)
                        self.assertLess(float(err.sum()),float(err0.sum()))
                        self.assertEqual(float(y[0].abs().max()),0.)
                        reused = rtx.quantize_nvfp4(x,tensor_scale=outer,
                            config=replace(cfg,block_scale_selection=policy,adaptive_reuse_packed=True))
                        torch.testing.assert_close(reused.qdata.view(torch.uint8),result.qdata.view(torch.uint8),rtol=0,atol=0)
                        torch.testing.assert_close(reused.scale.float(),result.scale.float(),rtol=0,atol=0)

    def test_regional_cached_and_uncached_and_compile(self):
        torch.manual_seed(17)
        m,n,k = 128,256,256
        x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        weight = torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
        grad = torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
        cfg = preferred_jit_row_region_config(rtx.NVFP4Problem(m,n,k))
        for policy in ('mse5','lsq1','four_over_six'):
            ys=[]
            for owner in ('cta_cached','cta','warp'):
                selected = replace(cfg,region_ownership=owner,
                    quant=replace(cfg.quant,block_scale_selection=policy,adaptive_reuse_packed=True))
                layer = rtx.NVFP4Linear(k,n,device='cuda',dynamic_config=selected,
                    scaling='jit_row_region',autotune='off',
                    backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
                with torch.no_grad(): layer.weight.copy_(weight)
                y = layer(x)
                grads = torch.autograd.grad(y,(x,layer.weight),grad)
                self.assertTrue(all(bool(t.isfinite().all()) for t in (y,*grads)))
                ys.append(y.detach())
                if owner == 'cta_cached':
                    compiled = torch.compile(layer,fullgraph=True,dynamic=False)
                    cy = compiled(x)
                    cg = torch.autograd.grad(cy,(x,layer.weight),grad)
                    torch.testing.assert_close(cy,y,rtol=0,atol=0)
                    for a,b in zip(cg,grads): torch.testing.assert_close(a,b,rtol=0,atol=0)
            for y in ys[1:]: torch.testing.assert_close(y,ys[0],rtol=0,atol=0)

    def test_ragged_regional_forward_and_both_gradients(self):
        torch.manual_seed(93)
        for m,n,k in ((37,65,97),(1,1,17)):
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
            g=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
            common_grads=None
            for selection in ('amax','lsq1','mse5','four_over_six'):
                outputs=[]
                for ownership in ('cta_cached','cta'):
                    cfg=rtx.NVFP4DynamicConfig(
                        quant=NVFP4QuantConfig(block_scale_selection=selection),
                        x_scale_region_rows=3,weight_scale_region_rows=2,
                        region_ownership=ownership,tensor_scale_mode='exact')
                    layer=rtx.NVFP4Linear(k,n,device='cuda',scaling='jit_row_region',
                        x_scale_region_rows=3,weight_scale_region_rows=2,
                        scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode='exact'),
                        dynamic_config=cfg,autotune='off',
                        backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
                    with torch.no_grad(): layer.weight.copy_(w)
                    y=layer(x)
                    grads=torch.autograd.grad(y,(x,layer.weight),g)
                    self.assertTrue(all(bool(t.isfinite().all()) for t in (y,*grads)))
                    if common_grads is None: common_grads=grads
                    for actual,expected in zip(grads,common_grads):
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    outputs.append(y.detach())
                torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
