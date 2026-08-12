# Benchmarks and tuning utilities

These scripts are development tools, not public Python APIs. Run them from the
repository root so imports and output paths are predictable.

| Script | Purpose |
| --- | --- |
| `train_decoder.py` | Resumable byte-level TinyStories BF16/MXFP8/NVFP4 decoder convergence comparison |
| `benchmark_mxfp8_frontend.py` | End-to-end `torch.compile` MXFP8 frontend benchmark |
| `benchmark_nvfp4_training.py` | Paired full delayed-scale NVFP4 training-forward versus fused MXFP8 performance gate |
| `benchmark_nvfp4_end_to_end.py` | Paired forward-plus-MXFP8-backward layer benchmark |
| `benchmark_nvfp4_frontend.py` | Paired fullgraph dynamic-forward latency and normalized numerical-error comparison against MXFP8 |
| `validate_nvfp4_convergence.py` | Controlled BF16/current/rowwise-JIT/delayed/exact scale-policy convergence study |
| `benchmark_mxfp8_prequant.py` | Dynamic BF16 quantization plus native-scale MXFP8 GEMM, with mainloop/epilogue stage and persistent-locality controls |
| `benchmark_torchao_fp8_rowwise.py` | TorchAO rowwise FP8 comparison baseline |
| `validate_production.py` | Unified MXFP8/NVFP4 eager/compiled, training/inference, packed-state, stream, cache, and long-dW readiness matrix |
| `validate_mxfp8_production.py` | Legacy MXFP8-only readiness matrix |
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

For the current-revision release-candidate campaign, run the same command on
the RTX 5070 Ti and RTX 5070 Laptop:

```bash
./run_5070_4h.sh
```

The root launcher fast-forward pulls, detaches itself, runs the CUDA and unified
production gates, calibrates the exact GPU/compiler checkout, and spends the
remaining four-hour deadline on balanced residual collection across all nine
MXFP8/NVFP4 production families. It then verifies finalists, audits the
journals, exports CSV/Parquet, and installs verified device-local winners.

All generated JSON, JSONL, logs, datasets, and compiled artifacts must go to an
ignored output directory such as `autotune_results/`, `autotune_logs/`, or
`autotune_datasets/`.

Run the fixed-token decoder convergence comparison with:

```bash
python benchmarks/train_decoder.py \
  --precision bf16 mxfp8 nvfp4 \
  --steps 300000 \
  --warmup-steps 3000 \
  --log-interval 100 \
  --validation-interval 5000 \
  --checkpoint-interval 5000
```

The default optimizer keeps master parameters and AdamW moments in FP32, then
copies each update to the BF16 execution model. `--optimizer bf16` is an
explicit reduced-state-precision ablation. A bounded study can use
`--stop-after-step 5000` without changing the 300,000-step cosine schedule;
rerunning the same command without that bound resumes from its atomic
checkpoint.

The first invocation downloads one official TinyStories training parquet
shard and creates a byte-token cache. All precisions receive the same initial
state and deterministic step-indexed windows. Metrics are append-only JSONL;
each precision has an atomic checkpoint and resumes by default. A clean
300,000-step run presents 460.8 million tokens to each model, approximately
one shard-equivalent token pass. Use a new `--output` directory when changing
the model geometry or training schedule.

Run the complete frontend matrix before promoting a release:

```bash
python benchmarks/validate_production.py \
  --output autotune_reports/production_matrix.json
```

`--quick` retains every category but reduces the long-reduction check from
M=8192 to M=1024. Use `--frontend mxfp8` or `--frontend nvfp4` for focused
diagnosis; the release gate uses the default `both`.

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
