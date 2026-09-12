"""Independent TorchAO conversion/reference contracts, CUDA gated."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(),"requires CUDA")
class RecipeReferenceTests(unittest.TestCase):
    def test_torchao_quantizer_matches_independent_reference(self):
        from benchmarks.compare_mlp_recipes import ao_quantize
        from tests.test_mxfp8_mlp import quantized
        for shape in ((17,35),(64,128)):
            torch.manual_seed(93)
            x=(torch.rand(*shape,device="cuda")-.5).bfloat16()
            for value in (x,x.T,torch.zeros_like(x)):
                torch.testing.assert_close(ao_quantize(value),quantized(value),rtol=0,atol=0)

    def test_linear_fresh_orientation_gradients(self):
        from benchmarks.compare_mlp_recipes import AOReferenceLinear
        from tests.test_mxfp8_mlp import quantized
        x=torch.randn(17,35,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        w=torch.randn(41,35,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        dy=torch.randn(17,41,device="cuda",dtype=torch.bfloat16)
        y=AOReferenceLinear.apply(x,w)
        dx,dw=torch.autograd.grad(y,(x,w),dy)
        expected=((quantized(x)@quantized(w).T).bfloat16(),
                  (quantized(dy)@quantized(w.T).T).bfloat16(),
                  (quantized(dy.T)@quantized(x.T).T).bfloat16())
        for a,b in zip((y,dx,dw),expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_hidden_intervention_only_changes_down_weight_gradient(self):
        from benchmarks.compare_mlp_recipes import fused_reference
        torch.manual_seed(91)
        inputs=[(torch.rand(*s,device="cuda")-.5).bfloat16()
                for s in ((64,128),(512,128),(128,256))]
        dy=torch.randn(64,128,device="cuda",dtype=torch.bfloat16)
        original=fused_reference(inputs,dy,"silu",False)
        fresh=fused_reference(inputs,dy,"silu",True)
        for a,b in zip(original[:3],fresh[:3]):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(torch.equal(original[-1],fresh[-1]))


if __name__=="__main__":
    unittest.main()
