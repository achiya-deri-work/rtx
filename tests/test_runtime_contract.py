from __future__ import annotations

from importlib.metadata import PackageNotFoundError
import unittest
from unittest.mock import patch

import torch

from rtx import runtime


class RuntimeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        runtime.validate_runtime_environment.cache_clear()

    def tearDown(self) -> None:
        runtime.validate_runtime_environment.cache_clear()

    def test_supported_native_stack(self) -> None:
        versions = {
            "torchao": "0.18.0",
            "nvidia-cutlass-dsl": "4.7.0",
            "apache-tvm-ffi": "0.1.13.post2",
        }
        with (
            patch.object(torch, "__version__", "2.13.0+cu132"),
            patch.object(torch.version, "cuda", "13.2"),
            patch.object(runtime, "package_version", side_effect=versions.__getitem__),
        ):
            resolved = runtime.validate_runtime_environment()
        self.assertEqual(resolved["cuda"], "13.2")
        self.assertEqual(resolved["nvidia-cutlass-dsl"], "4.7.0")

    def test_reports_every_stack_mismatch_together(self) -> None:
        def missing_or_old(distribution: str) -> str:
            if distribution == "apache-tvm-ffi":
                raise PackageNotFoundError(distribution)
            return "0.0.0"

        with (
            patch.object(torch, "__version__", "2.11.0+cu130"),
            patch.object(torch.version, "cuda", "13.0"),
            patch.object(runtime, "package_version", side_effect=missing_or_old),
        ):
            with self.assertRaisesRegex(RuntimeError, "PyTorch 2.11.0") as raised:
                runtime.validate_runtime_environment()
        message = str(raised.exception)
        self.assertIn("CUDA 13.0", message)
        self.assertIn("torchao 0.0.0", message)
        self.assertIn("nvidia-cutlass-dsl 0.0.0", message)
        self.assertIn("apache-tvm-ffi is not installed", message)


if __name__ == "__main__":
    unittest.main()
