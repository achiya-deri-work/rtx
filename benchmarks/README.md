# Benchmarks and tuning utilities

These scripts are development tools, not public Python APIs. Run them from the
repository root so imports and output paths are predictable.

| Script | Purpose |
| --- | --- |
| `benchmark_mxfp8_frontend.py` | End-to-end `torch.compile` MXFP8 frontend benchmark |
| `benchmark_mxfp8_prequant.py` | Dynamic BF16 quantization plus native-scale MXFP8 GEMM |
| `benchmark_torchao_fp8_rowwise.py` | TorchAO rowwise FP8 comparison baseline |
| `tune_composable_prequant.py` | Composable learned/global plus local forward tuning |
| `tune_composable_bwd.py` | Composable MXFP8 backward tuning |
| `verify_composable_bwd.py` | Independent backward winner verification and racing |
| `tune_mxfp8_native_quant.py` | Standalone native quantizer coordinate sweep |
| `tune_mxfp8_native_gemm.py` | Standalone prequantized GEMM coordinate sweep |

Portable multi-shape and cross-device campaigns belong in
`autotune_manifests/` and should be launched with `rtx-autotune`; these scripts
are for focused kernel investigation.

All generated JSON, JSONL, logs, datasets, and compiled artifacts must go to an
ignored output directory such as `autotune_results/`, `autotune_logs/`, or
`autotune_datasets/`.
