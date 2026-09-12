"""Experimental mixed gate projection: native kernels versus TorchAO operands."""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class QualityProjectionTests(unittest.TestCase):
    def test_eager_forward_and_combined_backward(self):
        from benchmarks.benchmark_mlp_quality import BF16GateProjection
        from benchmarks.compare_mlp_recipes import ao_quantize
        from rtx.fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG
        from rtx.fp8_bwd import _default_bwd_config
        from rtx.kernels.mxfp8 import MXFP8Problem, normalize_fwd_config
        torch.manual_seed(94)
        x=(torch.rand(128,128,device="cuda")-.5).bfloat16().requires_grad_()
        w=(torch.rand(256,128,device="cuda")-.5).bfloat16().requires_grad_()
        g,u=BF16GateProjection.apply(x,w,normalize_fwd_config(),
            DEFAULT_MXFP8_PREQUANT_CONFIG,_default_bwd_config(MXFP8Problem(128,256,128)),"materialized")
        dg,du=torch.randn_like(g),torch.randn_like(u)
        dx,dw=torch.autograd.grad((g,u),(x,w),(dg,du))
        dp=torch.cat((dg,du),-1)
        expected=(x@w[:128].T,(ao_quantize(x)@ao_quantize(w[128:]).T).bfloat16(),
                  (ao_quantize(dp)@ao_quantize(w.T).T).bfloat16(),
                  (ao_quantize(dp.T)@ao_quantize(x.T).T).bfloat16())
        for a,b in zip((g,u,dx,dw),expected):
            self.assertTrue(bool(a.isfinite().all()))
            relative=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)
            self.assertLess(float(relative),.003)


if __name__ == "__main__":
    unittest.main()
