# Benchmarks and tuning utilities

These scripts are development tools, not public Python APIs. Run them from the
repository root so imports and output paths are predictable.

| Script | Purpose |
| --- | --- |
| `benchmark_mxfp8_frontend.py` | End-to-end `torch.compile` MXFP8 frontend benchmark |
| `benchmark_mxfp8_prequant.py` | Dynamic BF16 quantization plus native-scale MXFP8 GEMM, with mainloop/epilogue stage and persistent-locality controls |
| `benchmark_torchao_fp8_rowwise.py` | TorchAO rowwise FP8 comparison baseline |
| `validate_mxfp8_production.py` | Eager/compiled, training/inference, stream, cache, and long-dW readiness matrix |
| `benchmark_mxfp8_bwd.py` | Fused and quantize-once full/workspace/atomic, ordinary and one-BF16-load shared-G quad quantization, dual-stream, and wide-CTA operand-reuse backward families |
| `tune_composable_prequant.py` | Composable learned/global plus local forward tuning |
| `tune_composable_bwd.py` | Composable MXFP8 backward tuning |
| `verify_composable_bwd.py` | Independent backward winner verification and racing |
| `tune_mxfp8_native_quant.py` | Standalone native quantizer coordinate sweep |
| `tune_mxfp8_native_gemm.py` | Standalone prequantized GEMM coordinate sweep |
| `run_5070_autotuner_study.sh` | Resumable 3-hour laptop / 6-hour desktop prospective optimizer study |

Portable multi-shape and cross-device campaigns belong in
`autotune_manifests/` and should be launched with `rtx-autotune`; these scripts
are for focused kernel investigation.

After pulling the same commit on each 5070 machine, launch with
`./benchmarks/run_5070_autotuner_study.sh laptop-3h` or
`./benchmarks/run_5070_autotuner_study.sh ti-6h`. The script calibrates once,
uses the pulled checkout even with a non-editable environment install, and
resumes the same residual journals when rerun.

All generated JSON, JSONL, logs, datasets, and compiled artifacts must go to an
ignored output directory such as `autotune_results/`, `autotune_logs/`, or
`autotune_datasets/`.

Run the complete frontend matrix before promoting a release:

```bash
python benchmarks/validate_mxfp8_production.py \
  --output autotune_reports/mxfp8_production_matrix.json
```

`--quick` retains every category but reduces the long-reduction check from
M=8192 to M=1024.

Use `benchmark_mxfp8_bwd.py --reuse-sweep` for paired races across the legal
128x256 and 256x128 dW reuse basins, BF16 stage counts, and register budgets.
