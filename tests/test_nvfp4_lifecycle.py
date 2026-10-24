"""Bounded public training lifecycle checks; no performance claims."""
import unittest

import torch
from torch.utils.checkpoint import checkpoint

import rtx


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NVFP4LifecycleTests(unittest.TestCase):
    def test_reentrant_checkpoint_noncontiguous_gradient_and_live_memory(self):
        for mode in ("current", "delayed", "jit_row_region", "block"):
            with self.subTest(mode=mode):
                torch.manual_seed(20261023)
                layer = rtx.NVFP4Linear(128, 128, device="cuda", scaling=mode,
                                        autotune="off")
                x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16,
                                requires_grad=True)
                upstream = torch.randn(128, 256, device="cuda",
                                       dtype=torch.bfloat16)[:, ::2]
                self.assertFalse(upstream.is_contiguous())

                def step(recompute):
                    layer.zero_grad(set_to_none=True)
                    x.grad = None
                    y = checkpoint(layer, x, use_reentrant=True) if recompute else layer(x)
                    y.backward(upstream)
                    return y

                # Settle delayed telemetry and all shape-specific runtime caches.
                for _ in range(4):
                    step(False)
                    step(True)
                y = step(False)
                expected = tuple(t.detach().clone() for t in (y, x.grad, layer.weight.grad))
                y = step(True)
                for actual, reference in zip((y, x.grad, layer.weight.grad), expected):
                    self.assertTrue(bool(actual.isfinite().all()))
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                del y, expected, actual, reference
                layer.zero_grad(set_to_none=True)
                x.grad = None
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                samples = []
                for _ in range(3):
                    for _ in range(10):
                        y = step(True)
                        del y
                    layer.zero_grad(set_to_none=True)
                    x.grad = None
                    torch.cuda.synchronize()
                    samples.append(torch.cuda.memory_allocated())
                self.assertEqual(samples, [baseline] * 3,
                                 f"live allocations grew for {mode}: {baseline}, {samples}")
                del layer, x, upstream, step
                rtx.clear_runtime_caches(device="cuda")


if __name__ == "__main__":
    unittest.main()
