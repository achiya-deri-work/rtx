# Benchmarks and tuning utilities

These scripts are development tools, not public Python APIs. Run them from the
repository root so imports and output paths are predictable.

| Script | Purpose |
| --- | --- |
| `train_decoder.py` | Resumable byte-level TinyStories BF16/MXFP8/NVFP4-delayed/NVFP4-block decoder comparison |
| `run_decoder_four_mode.sh` | Exact 460.8M-token four-mode release convergence/runtime launcher |
| `run_decoder_batch64_autotune.sh` | Two bounded 12-minute searches plus verification, winner installation, and batch-24/64 model races |
| `benchmark_mxfp8_frontend.py` | End-to-end `torch.compile` MXFP8 frontend benchmark |
| `benchmark_nvfp4_end_to_end.py` | Production backend paired forward-plus-MXFP8-backward layer benchmark |
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
| `launch_nvfp4_full_power_6h.sh` | Clock-driven six-hour NVFP4 release/topology/deep-JIT/overflow campaign for SM120 SKUs |

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

For the matched NVFP4 revision-8 campaign on the RTX 5070 Ti, RTX 5070 Laptop,
and RTX 5090, pull the same commit and run this command on each machine:

```bash
./benchmarks/launch_nvfp4_full_power_6h.sh 6h
```

The launcher identifies the machine by hostname; set a stable explicit label
when hostnames are ambiguous:

```bash
RTX_AUTOTUNE_NODE=rtx5070ti ./benchmarks/launch_nvfp4_full_power_6h.sh 6h
RTX_AUTOTUNE_NODE=rtx5070_laptop ./benchmarks/launch_nvfp4_full_power_6h.sh 6h
RTX_AUTOTUNE_NODE=rtx5090 ./benchmarks/launch_nvfp4_full_power_6h.sh 6h
```

The worker reserves the final five minutes for journal audit. Before that it
uses 25% of the clock for matched release contexts, 35% for broad M/N/K and
ragged topology coverage, 30% for deep JIT-region exploration, and all
remaining time for two-replicate overflow contexts. A phase that exhausts its
search space early advances immediately; watchdog/CUDA-context exits resume
from append-only residual journals. Each phase emits independent CSV and
Parquet reports and never installs exploratory winners automatically.

Run the fixed-token decoder convergence comparison with:

```bash
python benchmarks/train_decoder.py \
  --precision bf16 mxfp8 nvfp4_delayed nvfp4_block \
  --optimizer fp32_master \
  --batch-size 24 \
  --steps 37500 \
  --warmup-steps 375 \
  --log-interval 25 \
  --validation-interval 625 \
  --checkpoint-interval 625
```

The equivalent reproducible launcher is
`./benchmarks/run_decoder_four_mode.sh`. Set `RTX_DECODER_OUTPUT` to choose a
new artifact directory; additional CLI arguments are forwarded verbatim. A
completed run writes `summary.json` with per-mode runtimes, speedups, final
losses, complete validation-loss curves, and the honest throughput-gate
status. An unmet performance target is reported without discarding an
otherwise valid convergence campaign.

Production model execution uses BF16 parameters, gradients, and activations.
FP32 is reserved for sensitive reductions and calculations as defined by
[`PRECISION_POLICY.md`](../PRECISION_POLICY.md). Optimizer-state precision is
reported separately: the production convergence/runtime comparison uses FP32
AdamW moments and master parameters uniformly across all four modes. The
`--optimizer bf16` path is a kernel-isolation throughput ablation and must be
reported as such. A bounded study can use
`--stop-after-step 5000` without changing the 37,500-step cosine schedule;
rerunning the same command without that bound resumes from its atomic
checkpoint.

Check the append-only convergence and runtime results with:

```bash
python benchmarks/check_decoder_throughput.py <run-directory>
```

This reports final convergence and steady-state runtime for all four modes. It
enforces the default 1.30x MXFP8 and 1.35x NVFP4-block speedup thresholds
against the otherwise identical BF16 production baseline; delayed scaling is
reported without a default speed gate.

The first invocation downloads one official TinyStories training parquet
shard and creates a byte-token cache. All precisions receive the same initial
state and deterministic step-indexed windows. Metrics are append-only JSONL;
each precision has an atomic checkpoint and resumes by default. A clean
37,500-step run at physical batch 24 presents exactly 460.8 million tokens to
each model, approximately one shard-equivalent token pass. The 375 warmup
steps likewise preserve the original 4.608-million-token warmup. This is a
real 12,288-token optimizer batch, not gradient accumulation.

The RTX 5070 Ti saturation sweep measured these steady-state medians with
FP32 AdamW state/master weights (thousands of tokens/s):

| Physical batch | BF16 | MXFP8 | NVFP4 delayed | NVFP4 block | Peak allocation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 202.1 | 279.8 | 204.3 | 251.6 | 1.94 GiB |
| 16 | 215.3 | 295.1 | 226.0 | 267.0 | 3.00 GiB |
| 24 | 216.9 | 296.7 | 216.8 | 268.1 | 3.99 GiB |
| 32 | 219.3 | 294.3 | 222.6 | 267.3 | 5.02 GiB |
| 48 | 220.9 | 293.7 | 224.5 | 266.8 | 7.07 GiB |
| 64 | 222.0 | 294.1 | 227.2 | 269.1 | 9.19 GiB |

Batch 24 is the selected low-precision throughput knee: increasing to 64
consumes another 5.2 GiB without improving MXFP8 or block-NVFP4 throughput.
Use a new `--output` directory when changing the model geometry or training
schedule.

Run the complete frontend matrix before promoting a release:

```bash
python benchmarks/validate_production.py \
  --output autotune_reports/production_matrix.json
```

`--quick` retains every category but reduces the long-reduction check from
M=8192 to M=1024. Use `--frontend mxfp8` or `--frontend nvfp4` for focused
diagnosis; the release gate uses the default `both`.

Collect the paired all-kernel/all-scaling autotuning dataset in a detached,
resumable process with:

```bash
./benchmarks/launch_all_kernel_autotune.sh 4h
```

The launcher performs `git pull --ff-only`, refreshes the editable install,
checks SM12x and the exact runtime contract, calibrates the GPU when needed,
then starts the campaign with `nohup` and `disown`. It prints the PID, durable
residual dataset directory, report directory, and log path. Follow progress
with the exact `tail -f` command it prints. Set `RTX_AUTOTUNE_NODE` to give
each machine a stable label; rerunning with the same label resumes its residual
contexts. Override `RTX_AUTOTUNE_OUTPUT_DIR`, `RTX_AUTOTUNE_REPORT_DIR`, or
`RTX_AUTOTUNE_LOG_DIR` when collecting multiple independent replicates.

The worker rotates breadth-first across MXFP8 forward/inference, shared MXFP8
backward, NVFP4 current/block/delayed/JIT-region forward, and
NVFP4 packed inference. On exit it audits the journals and emits both CSV and
Parquet without automatically installing exploratory winners.

Benchmark the production materialized delayed forward and its MXFP8 backward
against `MXFP8Linear` with paired AB/BA rounds:

```bash
python benchmarks/benchmark_nvfp4_end_to_end.py \
  --compile --nv-scaling delayed --nv-backend auto \
  --output autotune_reports/nvfp4_delayed_end_to_end.json
```

The end-to-end benchmark supports eager and `--compile` runs. It verifies that
both frontends produce bit-identical MXFP8 dX/dW and uses device-wide fences so
dual-stream backward work cannot escape the measurement. Add `--profile` for a
per-kernel CPU/CUDA breakdown while investigating launch overhead.

Use `benchmark_mxfp8_bwd.py --reuse-sweep` for paired races across the legal
128x256 and 256x128 dW reuse basins, BF16 stage counts, and register budgets.
