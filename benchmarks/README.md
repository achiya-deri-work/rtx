# Benchmarks and tuning utilities

These scripts are development tools, not public Python APIs. Run them from the
repository root so imports and output paths are predictable.

| Script | Purpose |
| --- | --- |
| `benchmark_mxfp8_frontend.py` | End-to-end `torch.compile` MXFP8 frontend benchmark |
| `benchmark_nvfp4_training.py` | Paired full delayed-scale NVFP4 training-forward versus fused MXFP8 performance gate |
| `benchmark_nvfp4_end_to_end.py` | Paired forward-plus-MXFP8-backward layer benchmark |
| `validate_nvfp4_convergence.py` | Controlled BF16/current/delayed/exact scale-policy convergence study |
| `benchmark_mxfp8_prequant.py` | Dynamic BF16 quantization plus native-scale MXFP8 GEMM, with mainloop/epilogue stage and persistent-locality controls |
| `benchmark_torchao_fp8_rowwise.py` | TorchAO rowwise FP8 comparison baseline |
| `validate_mxfp8_production.py` | Eager/compiled, training/inference, stream, cache, and long-dW readiness matrix |
| `benchmark_mxfp8_bwd.py` | Fused and quantize-once full/workspace/atomic, persistent split-FP32, ordinary and one-BF16-load shared-G quad quantization, dual-stream, wide-CTA, and TMA/cp.async clustered operand-reuse backward families |
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

Gate the complete single-launch NVFP4 delayed-training forward against verified
MXFP8 runtime winners with paired AB/BA rounds:

```bash
python benchmarks/benchmark_nvfp4_training.py \
  --winner-root /path/to/audited-bundle/shard-000-of-001 \
  --output autotune_reports/nvfp4_training_gate.json
```

The command exits nonzero if either default training shape is below 1.5x.

The end-to-end benchmark supports eager and `--compile` runs. It verifies that
both frontends produce bit-identical MXFP8 dX/dW and uses device-wide fences so
dual-stream backward work cannot escape the measurement. Add `--profile` for a
per-kernel CPU/CUDA breakdown while investigating launch overhead.

Use `benchmark_mxfp8_bwd.py --reuse-sweep` for paired races across the legal
128x256 and 256x128 dW reuse basins, BF16 stage counts, and register budgets.
