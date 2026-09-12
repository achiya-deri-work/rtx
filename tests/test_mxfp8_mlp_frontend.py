"""Public conversion, persistence, and compiled training contracts."""
import unittest
import tempfile
from pathlib import Path

import torch

from rtx import MXFP8MLP, MXFP8MLPConfig


def projections(device="cpu"):
    return tuple(torch.nn.Linear(i,o,bias=False,device=device,dtype=torch.bfloat16)
                 for i,o in ((35,97),(35,97),(97,41)))


class FrontendTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_conversion_and_checkpoint(self):
        gate,up,down=projections()
        for layer in (gate,up,down):
            layer.eval().requires_grad_(False)
        rng=torch.random.get_rng_state().clone()
        mlp=MXFP8MLP.from_linears(gate,up,down,activation="gelu")
        torch.testing.assert_close(torch.random.get_rng_state(),rng)
        torch.testing.assert_close(mlp.gate_up_weight,torch.cat((gate.weight,up.weight)))
        torch.testing.assert_close(mlp.down_weight,down.weight)
        self.assertNotEqual(mlp.down_weight.data_ptr(),down.weight.data_ptr())
        self.assertFalse(mlp.training)
        self.assertFalse(mlp.gate_up_weight.requires_grad)
        self.assertFalse(mlp.down_weight.requires_grad)
        self.assertEqual(set(mlp.state_dict()),{"gate_up_weight","down_weight"})
        restored=MXFP8MLP(35,97,41,activation="gelu")
        restored.load_state_dict(mlp.state_dict())
        torch.testing.assert_close(restored.gate_up_weight,mlp.gate_up_weight)

    def test_conversion_rejections(self):
        gate,up,down=projections()
        with self.assertRaisesRegex(TypeError,"nn.Linear"):
            MXFP8MLP.from_linears(None,up,down)
        gate.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError,"requires_grad"):
            MXFP8MLP.from_linears(gate,up,down)
        gate.weight.requires_grad_(True)
        up.eval()
        with self.assertRaisesRegex(ValueError,"training mode"):
            MXFP8MLP.from_linears(gate,up,down)
        up.train().float()
        with self.assertRaisesRegex(TypeError,"BF16"):
            MXFP8MLP.from_linears(gate,up,down)
        gate,up,down=projections()
        gate.bias=torch.nn.Parameter(torch.zeros(97,dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError,"biases"):
            MXFP8MLP.from_linears(gate,up,down)
        with self.assertRaisesRegex(ValueError,"meta"):
            MXFP8MLP.from_linears(*projections("meta"))

    def test_policy_json_validation(self):
        config=MXFP8MLPConfig()
        self.assertEqual(config,MXFP8MLPConfig.from_json(config.to_json()))
        values=config.to_dict()
        values["typo"]=True
        with self.assertRaisesRegex(ValueError,"unknown"):
            MXFP8MLPConfig.from_dict(values)
        del values["typo"]
        del values["gate_up"]
        with self.assertRaisesRegex(ValueError,"missing"):
            MXFP8MLPConfig.from_dict(values)

    @unittest.skipUnless(torch.cuda.is_available(),"requires CUDA")
    def test_autotune_installs_persistable_policy(self):
        mlp=MXFP8MLP(32,64,32,device="cuda")
        x=torch.randn(8,32,device="cuda",dtype=torch.bfloat16)
        before={k:v.clone() for k,v in mlp.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            result=mlp.autotune_(x,max_trials=1,time_budget_s=30,
                                output_dir=directory,progress=None)
            self.assertIs(result,mlp)
            self.assertEqual(mlp._shape_configs[mlp._shape_key(x)][0],mlp.tuning_result.config)
            self.assertEqual(mlp.config,MXFP8MLPConfig.from_json(mlp.config.to_json()))
            self.assertTrue(list(Path(directory).rglob("*.jsonl")))
        for name,value in mlp.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(mlp(x)).all())

    @unittest.skipUnless(torch.cuda.is_available(),"requires CUDA")
    def test_converted_compiled_training(self):
        from tests.test_mxfp8_mlp import reference
        for activation in ("silu","gelu","relu"):
            mlp=MXFP8MLP.from_linears(*projections("cuda"),activation=activation)
            fn=torch.compile(mlp,fullgraph=True,dynamic=False)
            x=torch.randn(2,17,35,device="cuda",dtype=torch.bfloat16,requires_grad=True)
            dy=torch.randn(2,17,41,device="cuda",dtype=torch.bfloat16)
            inputs=(x,mlp.gate_up_weight,mlp.down_weight)
            expected,grads=reference(*inputs,dy,activation)
            y=fn(x)
            actual=(y,*torch.autograd.grad(y,inputs,dy))
            for a,b in zip(actual,(expected,*grads)):
                self.assertTrue(torch.isfinite(a).all())
                error=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)
                self.assertLess(error.item(),.025)


if __name__=="__main__":
    unittest.main()
