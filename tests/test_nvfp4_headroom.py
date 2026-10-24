"""Explicit outer-scale headroom must not leak into default cache policy."""
from dataclasses import replace
import unittest

import torch

import rtx
from rtx.fp4 import _outer_tensor_scale, _default_runtime_schedule_rejection, _validate_experimental_outer_contract
from rtx.autotune.adapters import make_nvfp4_jit_row_region_adapter
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.promotion import _config_rejection
from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config, dynamic_config_to_dict


class HeadroomContractTests(unittest.TestCase):
    def test_unsupported_or_mismatched_public_policy_rejected(self):
        policy = rtx.NVFP4ScaleConfig(tensor_scale_mode='exact_four_over_six')
        cfg = replace(preferred_jit_row_region_config(rtx.NVFP4Problem(128,128,128)),
                      tensor_scale_mode='exact_four_over_six')
        _validate_experimental_outer_contract(policy,cfg,'jit_row_region',False)
        for public,explicit,scaling,packed in ((policy,None,'current',False),
                (None,cfg,'current',False),(policy,cfg,'block',False),
                (policy,cfg,'current',True)):
            with self.assertRaises(ValueError):
                _validate_experimental_outer_contract(public,explicit,scaling,packed)

    def test_outer_formula_zero_and_dtype(self):
        x = torch.tensor([0., -2688., 1536.], dtype=torch.bfloat16)
        torch.testing.assert_close(_outer_tensor_scale(x, 'exact'), torch.tensor([1.]))
        y = _outer_tensor_scale(x, 'exact_four_over_six')
        torch.testing.assert_close(y, torch.tensor([1.75]), rtol=0, atol=0)
        self.assertEqual(y.dtype, torch.float32)
        torch.testing.assert_close(_outer_tensor_scale(torch.zeros(3), 'exact_four_over_six'), torch.ones(1))

    def test_headroom_cannot_be_promoted_even_with_amax(self):
        p = rtx.NVFP4Problem(128,128,128)
        cfg = replace(preferred_jit_row_region_config(p), tensor_scale_mode='exact_four_over_six')
        self.assertIsNone(cfg.rejection(p))
        self.assertIsNotNone(_default_runtime_schedule_rejection(cfg,p))
        self.assertIsNotNone(_config_rejection('nvfp4_jit_row_region_fwd',dynamic_config_to_dict(cfg),p))

    def test_traceable_outer_formula_fullgraph(self):
        fn = torch.compile(lambda x: _outer_tensor_scale(x, 'exact_four_over_six'),
                           fullgraph=True, backend='eager')
        for x in (torch.zeros(3), torch.tensor([1., -2688., 0.])):
            torch.testing.assert_close(fn(x), _outer_tensor_scale(x,'exact_four_over_six'),rtol=0,atol=0)

    def test_adapter_pins_headroom_and_separates_contexts(self):
        p = rtx.NVFP4Problem(128,128,128)
        base = preferred_jit_row_region_config(p)
        cfg = replace(base,tensor_scale_mode='exact_four_over_six')
        evaluator = lambda _: TrialOutcome('ok',median_ms=1.)
        ordinary = make_nvfp4_jit_row_region_adapter(p,evaluator,initial=base)
        experimental = make_nvfp4_jit_row_region_adapter(p,evaluator,initial=cfg)
        self.assertIsNotNone(experimental.rejection(base))
        self.assertIsNotNone(ordinary.rejection(cfg))
        self.assertIsNone(experimental.rejection(cfg))
        self.assertNotEqual(ordinary.context,experimental.context)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class HeadroomCudaTests(unittest.TestCase):
    def test_fullgraph_forward_and_both_gradients(self):
        torch.manual_seed(431)
        for m,n,k in ((128,256,256),(37,65,97)):
            torch.compiler.reset()
            x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            grad = torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
            cfg = replace(preferred_jit_row_region_config(rtx.NVFP4Problem(m,n,k)),
                          tensor_scale_mode='exact_four_over_six')
            cfg = replace(cfg,quant=replace(cfg.quant,block_scale_selection='four_over_six'))
            if cfg.rejection(rtx.NVFP4Problem(m,n,k)):
                cfg = rtx.NVFP4DynamicConfig(quant=rtx.NVFP4QuantConfig(block_scale_selection='four_over_six'),
                    tensor_scale_mode='exact_four_over_six',x_scale_region_rows=3,weight_scale_region_rows=2)
            layer = rtx.NVFP4Linear(k,n,device='cuda',dynamic_config=cfg,
                scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode='exact_four_over_six'),
                scaling='jit_row_region',autotune='off',backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
            y = layer(x)
            grads = torch.autograd.grad(y,(x,layer.weight),grad)
            cy = torch.compile(layer,fullgraph=True,dynamic=False)(x)
            cg = torch.autograd.grad(cy,(x,layer.weight),grad)
            self.assertTrue(all(bool(t.isfinite().all()) for t in (y,cy,*grads,*cg)))
            torch.testing.assert_close(cy,y,rtol=0,atol=0)
            for a,b in zip(cg,grads):
                torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
