"""Real-device dispatcher contracts, independent of numerical error gates."""

import unittest

import torch

import rtx


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "requires RTX Blackwell CUDA",
)
class OperatorContractTests(unittest.TestCase):
    def test_training_operator_contracts(self):
        torch.manual_seed(719)
        for family in ("mxfp8", "block", "jit_row_region"):
            with self.subTest(family=family):
                layer = (
                    rtx.MXFP8Linear(256, 128, device="cuda", autotune="off")
                    if family == "mxfp8"
                    else rtx.NVFP4Linear(
                        256, 128, device="cuda", autotune="off", scaling=family
                    )
                )
                name = (
                    "mxfp8_linear_train" if family == "mxfp8"
                    else f"nvfp4_linear_{family}_train"
                )
                x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16,
                                requires_grad=True)
                result = torch.library.opcheck(
                    getattr(torch.ops.rtx, name).default,
                    (x, layer.weight,
                     layer._prequant_config_key if family == "mxfp8"
                     else layer._forward_config_key,
                     layer._backward_config_key),
                )
                self.assertTrue(all(value == "SUCCESS" for value in result.values()),
                                result)


if __name__ == "__main__":
    unittest.main()
