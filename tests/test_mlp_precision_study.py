import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PrecisionStudyTests(unittest.TestCase):
    def test_corrected_product(self):
        from benchmarks.study_mlp_precision import product
        from benchmarks.compare_mlp_recipes import ao_quantize
        torch.manual_seed(91)
        a = torch.randn(17,35,device="cuda",dtype=torch.bfloat16)
        b = torch.randn(41,35,device="cuda",dtype=torch.bfloat16)
        aq,bq = ao_quantize(a),ao_quantize(b)
        expected = aq@bq.T + ao_quantize(a.float()-aq)@bq.T
        expected = (expected + aq@ao_quantize(b.float()-bq).T).bfloat16()
        torch.testing.assert_close(product(a,b,"corrected"),expected,rtol=0,atol=0)

    def test_bf16_recipe_matches_autograd(self):
        from benchmarks.study_mlp_precision import evaluate
        for act in ("silu", "gelu", "relu"):
            torch.manual_seed(17)
            inputs = [torch.randn(*s,device="cuda",dtype=torch.bfloat16,requires_grad=True)
                      for s in ((17,35),(97,35),(97,35),(35,97))]
            x,g,u,d = inputs
            dy = torch.randn_like(x)
            hidden = (getattr(torch.nn.functional,act)((x@g.T).float())*(x@u.T).float()).bfloat16()
            y = hidden@d.T
            expected = (y,*torch.autograd.grad(y,inputs,dy))
            actual = evaluate(inputs,dy,act,"bf16","bf16","bf16")
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_mixed_hidden_contract(self):
        from benchmarks.study_mlp_precision import RecipeLinear
        from benchmarks.compare_mlp_recipes import ao_quantize
        x = torch.randn(17,35,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        w = torch.randn(41,35,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        dy = torch.randn(17,41,device="cuda",dtype=torch.bfloat16)
        y = RecipeLinear.apply(x,w,"hidden")
        dx,dw = torch.autograd.grad(y,(x,w),dy)
        expected = ((x.float()@ao_quantize(w).T).bfloat16(),
                    (ao_quantize(dy)@ao_quantize(w.T).T).bfloat16(),
                    (ao_quantize(dy.T)@x.float()).bfloat16())
        for a,b in zip((y,dx,dw),expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
