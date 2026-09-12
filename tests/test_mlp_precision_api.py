"""Precision contracts and bounded shape policies for recomputed MLPs."""
from dataclasses import replace
import unittest

import torch
from rtx import MXFP8MLP, MXFP8MLPConfig


class PrecisionAPITests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_inactive_schedules_share_identity(self):
        from rtx.autotune.mlp import active_schedule
        from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
        config=replace(MXFP8MLPConfig(),gate_backward=DEFAULT_FUSED_MXFP8_BWD_CONFIG,
                       down_backward=DEFAULT_FUSED_MXFP8_BWD_CONFIG)
        inactive=replace(config,down_gemm=replace(config.down_gemm,stages=4))
        self.assertEqual(active_schedule(config,"mxfp8","train"),active_schedule(inactive,"mxfp8","train"))
        self.assertNotEqual(active_schedule(config,"bf16_gate","train"),active_schedule(inactive,"bf16_gate","train"))
        inactive=replace(config,gate_up=replace(config.gate_up,glu_store_stages=4))
        self.assertEqual(active_schedule(config,"mxfp8","train"),active_schedule(inactive,"mxfp8","train"))

    def test_hybrid_honors_weight_quantization_schedule(self):
        from rtx._mlp_precision import prequant_config
        config=MXFP8MLPConfig()
        weight=replace(config.gate_prequant.quant,num_warps=4)
        config=replace(config,gate_prequant=replace(config.gate_prequant,weight_quant=weight,
                                                   quant_launches="separate"))
        actual=prequant_config(config)
        self.assertEqual(actual.quant_launches,"separate")
        self.assertEqual(actual.resolved_weight_quant(),weight)

    def test_cooperative_ragged_rejected_before_launch(self):
        from rtx.configs.glu import cooperative_glu_rejection
        from rtx.kernels.mxfp8 import MXFP8Problem
        config=MXFP8MLPConfig().gate_prequant.gemm
        self.assertIsNone(cooperative_glu_rejection(MXFP8Problem(128,512,128),config))
        self.assertIn("ragged",cooperative_glu_rejection(MXFP8Problem(129,512,128),config))
        self.assertIn("ragged",cooperative_glu_rejection(MXFP8Problem(128,514,128),config))

    def test_explicit_policy_and_bounded_shapes(self):
        with self.assertRaisesRegex(ValueError,"precision"):
            MXFP8MLP(32,64,precision="auto")
        module=MXFP8MLP(32,64,precision="bf16_gate",device="meta")
        default=module.config
        for m in range(1,131):
            module.configure_shape_(torch.empty(m,32,device="meta"),precision="bf16")
        self.assertEqual(len(module._shape_configs),128)
        self.assertEqual(module.precision,"bf16_gate")
        self.assertEqual(module.config,default)
        self.assertNotIn(module._shape_key(torch.empty(1,32,device="meta")),module._shape_configs)
        self.assertEqual(set(module.state_dict()),{"gate_up_weight","down_weight"})

    @unittest.skipUnless(torch.cuda.is_available(),"requires CUDA")
    def test_compiled_policies_and_gradients(self):
        from rtx.autotune.mlp import reference
        for m,k,h,o in ((128,128,256,128),(129,128,256,128),(64,256,512,256)):
            for act in ("silu","gelu"):
                torch.manual_seed(71)
                module=MXFP8MLP(k,h,o,precision="bf16_gate",activation=act,device="cuda")
                x=(torch.rand(m,k,device="cuda")-.5).bfloat16().requires_grad_()
                dy=(torch.rand(m,o,device="cuda")-.5).bfloat16()
                fn=torch.compile(module,fullgraph=True,dynamic=False)
                y=fn(x)
                actual=(y,*torch.autograd.grad(y,(x,module.gate_up_weight,module.down_weight),dy))
                with torch.no_grad():
                    expected=reference(x,module.gate_up_weight,module.down_weight,dy,act,"bf16_gate")
                for a,b in zip(actual,expected):
                    self.assertTrue(a.isfinite().all())
                    self.assertLess(((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item(),.003)
                module.zero_grad(set_to_none=True)
                x.grad=None
                module(x).backward(dy)
                for a,b in zip((x.grad,module.gate_up_weight.grad,module.down_weight.grad),expected[1:]):
                    self.assertLess(((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item(),.003)

    @unittest.skipUnless(torch.cuda.is_available(),"requires CUDA")
    def test_explicit_fallback_and_inference(self):
        module=MXFP8MLP(128,256,precision="bf16_gate",device="cuda")
        x=torch.randn(2,3,128,device="cuda",dtype=torch.bfloat16)
        module.configure_shape_(x,precision="bf16")
        self.assertEqual(module.policy_for(x)[1],"bf16")
        other=torch.randn(128,128,device="cuda",dtype=torch.bfloat16)
        self.assertEqual(module.policy_for(other)[1],"bf16_gate")
        fn=torch.compile(module,fullgraph=True,dynamic=False)
        def expected(x):
            g,u=torch.nn.functional.linear(x,module.gate_up_weight).chunk(2,-1)
            return torch.nn.functional.linear((torch.nn.functional.silu(g.float())*u.float()).bfloat16(),module.down_weight)
        with torch.no_grad():
            torch.testing.assert_close(fn(x),expected(x),rtol=.01,atol=1e-4)
            self.assertTrue(fn(other).isfinite().all())
        # Parameters-only gradients exercise checkpointing when X is frozen.
        fn(x).float().square().mean().backward()
        self.assertTrue(module.gate_up_weight.grad.isfinite().all())
        self.assertTrue(module.down_weight.grad.isfinite().all())

    @unittest.skipUnless(torch.cuda.is_available(),"requires CUDA")
    def test_hybrid_registered_contract_and_live_memory(self):
        from rtx._mlp_precision import execution_keys
        from rtx.fp8_bwd import _mxfp8_linear_train_op
        module=MXFP8MLP(128,256,precision="bf16_gate",device="cuda")
        module.config=replace(module.config,gate_prequant=replace(module.config.gate_prequant,
            quant_launches="separate",weight_quant=replace(module.config.gate_prequant.quant,num_warps=4)))
        x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        keys=execution_keys(module.config,128,128,256,128)
        hidden=torch.randn(128,256,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        torch.library.opcheck(_mxfp8_linear_train_op,(hidden,module.down_weight,keys[1],keys[3]))
        fn=torch.compile(module,fullgraph=True,dynamic=False)
        def step():
            module.zero_grad(set_to_none=True);x.grad=None
            fn(x).float().square().mean().backward()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        before=torch.cuda.memory_allocated()
        for _ in range(30):
            step()
        torch.cuda.synchronize()
        self.assertLessEqual(torch.cuda.memory_allocated(),before+4096)

    @unittest.skipUnless(torch.cuda.is_available(),"requires CUDA")
    def test_tuning_space_policy_and_shape_reductions(self):
        from rtx.autotune.mlp import make_mlp_adapter
        module=MXFP8MLP(128,256,device="cuda")
        x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16)
        adapter=make_mlp_adapter(x,module.gate_up_weight,module.down_weight,precision="bf16_gate")
        self.assertNotIn("gate_up.glu_silu_math",adapter.axes)
        self.assertIn("gate_backward.dw_split_count",adapter.axes)


if __name__=="__main__":
    unittest.main()
