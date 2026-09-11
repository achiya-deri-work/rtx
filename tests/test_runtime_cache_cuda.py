"""Cold first-hit tuning and cross-process cache reuse on a real device."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


_WORKER = r'''
import sys
from unittest.mock import patch
import torch
import rtx
import rtx.prequant_autotune as mx
import rtx.nvfp4_inference_autotune as nv

root, phase = sys.argv[1:]
def forbidden(*args, **kwargs):
    raise AssertionError("cache hit unexpectedly entered the autotuner")
for family in ("mxfp8", "nvfp4"):
    module, name = ((mx, "tune_mxfp8_prequant") if family == "mxfp8"
                    else (nv, "tune_nvfp4_inference_state"))
    original = getattr(module, name)
    with patch.object(module, name, side_effect=forbidden if phase == "read" else original) as tuner:
        cls = rtx.MXFP8Linear if family == "mxfp8" else rtx.NVFP4Linear
        layer = cls(256, 128, device="cuda", autotune="balanced",
                    backend="materialized", autotune_cache_dir=root).eval()
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            first = layer(x)
            calls = tuner.call_count
            second = layer(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert bool(torch.isfinite(second).all())
        assert tuner.call_count == calls, "same-process cache miss"
        assert calls == (1 if phase == "write" else 0), (family, phase, calls)
print("PASS", phase)
'''


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "requires RTX Blackwell CUDA",
)
class RuntimeCacheCudaTests(unittest.TestCase):
    def test_balanced_first_hit_and_cross_process_load(self):
        # Exercise orchestration, not search quality; production uses its normal
        # larger budget. Keep this integration test bounded on a fresh machine.
        env = dict(os.environ, RTX_BALANCED_AUTOTUNE_TRIALS="4",
                   RTX_BALANCED_AUTOTUNE_SECONDS="3")
        with tempfile.TemporaryDirectory(prefix="rtx-runtime-cache-") as root:
            for phase in ("write", "read"):
                run = subprocess.run(
                    [sys.executable, "-c", _WORKER, root, phase],
                    env=env, capture_output=True, text=True, timeout=180,
                )
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertGreaterEqual(
                    len(list((Path(root) / "runtime_winners").rglob("*.json"))), 2
                )


if __name__ == "__main__":
    unittest.main()
