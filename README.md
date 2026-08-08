# RTX low-precision linear layers

`rtx` is an experimental Python library for low-precision training and
inference on NVIDIA RTX and Jetson Blackwell GPUs. Its two public linear
frontends are `rtx.MXFP8Linear` and `rtx.NVFP4Linear`. The
current implementation contains:

- fused BF16 input/weight quantization and MXFP8 forward GEMM;
- materialized dynamic MXFP8 quantization plus GEMM;
- MXFP8 backward for `dX` and the FP32-accumulating long-reduction `dW`;
- PyTorch custom-op and `nn.Module` frontends;
- persistent random, gradient-boosted cost-model, bandit, and local search;
- calibrated hot/rotating-cache measurements and paired finalist races; and
- portable cross-device datasets exported as CSV or Parquet.

The executable native kernels currently target SM120/SM121. Architecture
discovery and lazy dispatch also recognize SM110/Jetson Thor, whose separate
TCGen05/TMEM kernels remain under development. This is research software:
kernel, packed-operand, and dataset schemas are versioned, but APIs may still
change.

## Requirements

- Linux
- Python 3.11+
- an RTX Blackwell GPU and a compatible NVIDIA driver
- a CUDA-enabled PyTorch build with Blackwell support
- CUDA Python 13.x
- NVIDIA CUTLASS Python DSL 4.7.x

Install the correct CUDA-enabled PyTorch build for the machine first. Then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For Parquet export and development tools:

```bash
python -m pip install -e '.[parquet,dev]'
```

The equivalent dependency lists are in `requirements.txt` and
`requirements-dev.txt`. An editable install is recommended while kernels are
changing because every dataset bundle records a hash of the installed source.

## Python frontend

```python
import torch
import rtx

x = torch.randn(512, 1536, device="cuda", dtype=torch.bfloat16)
w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)

y = rtx.mxfp8_linear(x, w)
layer = rtx.MXFP8Linear(
    1536, 1536, bias=False, device="cuda", dtype=torch.bfloat16
)
y2 = layer(x)
```

Both modules return BF16 and expose the usual no-bias
`nn.Linear(in_features, out_features, bias=False, ...)` shape convention.
Quantization state is independent from training mode:

| State | Activation | Weight | Training | Per-call work |
| --- | --- | --- | --- | --- |
| Dynamic | BF16 | BF16 | Yes | Quantize X and W, then GEMM |
| AOT weight | BF16 | Packed | Inference only | Quantize X, then GEMM |
| Prequantized | Packed | Packed | Inference only | GEMM only |

Dynamic execution is equally valid under `torch.inference_mode()`; inference
does not imply packed weights. Explicit conversion performs weight
quantization exactly once:

```python
dynamic = rtx.MXFP8Linear(1536, 1536, device="cuda")

with torch.inference_mode():
    y_dynamic = dynamic(x)  # BF16 X and W are dynamically quantized

    packed_weight = dynamic.to_quantized_weight()
    y_weight_aot = packed_weight(x)  # only X is dynamically quantized

    packed_x = rtx.quantize_mxfp8(x)
    y_prequantized = packed_weight(packed_x)  # pure packed GEMM
```

`rtx.MXFP8Tensor` stores E4M3 values, E8M0 block scales, logical shape,
orientation, physical scale layout, and packing schema version. The analogous
`rtx.NVFP4Tensor` stores packed E2M1 values, E4M3 scales per 16 values, and its
FP32 tensor scale. Packed module weights are persistent buffers and do not
retain a BF16 master weight.

MXFP8 implements all three state boundaries. `NVFP4Linear` and
`NVFP4Tensor` expose the same dispatcher/fake/module contract, but execution
still raises clearly until the NVFP4 quantizer and GEMM are implemented.

The historical MXFP8 backend name `prequant` means *materialized dynamic*: it
quantizes both BF16 operands into global memory on every call before launching
the GEMM. It is an implementation strategy for the first row above, not an AOT
weight state.

See `rtx/fp8.py` and `rtx/fp8_bwd.py` for backend, configuration, and explicit
backward controls.

## One-command dataset campaign

Validate a manifest without a GPU launch:

```bash
rtx-autotune validate autotune_manifests/cross_device_dataset_v2.json
```

Probe a machine before a long run:

```bash
rtx-autotune probe --device cuda:0
```

Create a machine-local calibration once. It measures L2/DRAM copy bandwidth,
BF16 tensor throughput, and the native MXFP8 quantization/GEMM pipeline while
retaining the raw timing samples:

```bash
rtx-autotune calibrate --device cuda:0 \
  --output hardware_calibration.json
```

Run or resume the complete assigned campaign:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_v2.json \
  --device cuda:0 \
  --output-dir autotune_datasets \
  --format both \
  --calibration hardware_calibration.json
```

Use `--format parquet` or `--format both` when the Parquet extra is installed.
Every accepted observation is fsync'd to JSONL before the next candidate, so
rerunning the same command resumes after interruption.

For rented or preemptible machines, use the anytime schedule. It visits a
representative, interleaved set of forward, prequant-forward, and backward
contexts breadth-first at absolute trial milestones. Longer runs deepen the
same contexts; shorter runs still cover all kernel families and cache regimes.

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_v2.json \
  --device cuda:0 \
  --output-dir autotune_datasets \
  --format both \
  --calibration hardware_calibration.json \
  --wall-time 12h \
  --context-slice 2m
```

The default milestones are 32, 96, 192, 384, and 512 total trials per
context—not new trials per invocation. The first breadth pass verifies two
finalists, and fully searched contexts use the manifest's full confirmation
policy. The global wall time and every context visit are checkpoints, and
Ctrl-C preserves all observations already printed as `SAVE`.

If this scheduler is pulled while a v2 run made by an older checkout already
exists, explicitly adopt that bundle's context identity so the old observations
count toward the same milestones:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_v2.json \
  --device cuda:0 \
  --output-dir autotune_datasets \
  --format both \
  --calibration hardware_calibration.json \
  --wall-time 12h \
  --context-slice 2m \
  --adopt-existing-context-identity
```

That flag is intentionally opt-in and should only cross runner/autotuner-only
changes. Do not use it after changing kernel implementations or kernel revision
numbers. Reuse the exact same calibration JSON, output directory, manifest, and
shard arguments as the original run.

For multiple processes on one machine, split whole workload contexts:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_v2.json \
  --shard-index 0 --shard-count 2
```

Use the same manifest without sharding on each different GPU when the goal is
to measure identical workload coverage across devices.

## Copying and collecting results

Each run produces one self-contained directory:

```text
autotune_datasets/<campaign>/<machine-id>/shard-000-of-001/
├── manifest.json
├── machine.json
├── summary.json
├── verification.jsonl
├── dataset.csv
└── stores/
    └── <kernel>/<regime>/
        ├── observations.jsonl
        ├── sessions.jsonl
        └── events.jsonl
```

Copy the entire machine or shard directories back to the main machine. Then
deduplicate and export all copied bundles:

```bash
rtx-autotune collect copied_datasets/ \
  --output merged/mxfp8_blackwell_v2 \
  --format both
```

This writes `merged/mxfp8_blackwell_v2.csv`,
`merged/mxfp8_blackwell_v2.parquet`, and an export report. Rows contain exact
configs, resource/occupancy/wave estimates, traffic and L2 ratios, memory-bus
and calibrated roofline data, raw timing arrays, compiler latency and available
compiled-resource attributes, correctness results, telemetry, proposal
provenance, device/environment context, confirmation measurements, and
paired-race decisions. The JSONL files remain authoritative.

## Dataset manifest

The portable manifest can contain forward and backward jobs. Each job has its
own shapes, cache regimes, benchmark protocol, search policy, and finalist
count:

```json
{
  "schema_version": 2,
  "name": "example",
  "seed": 20260808,
  "jobs": [
    {
      "family": "mxfp8_prequant_fwd",
      "shapes": [{"m": 512, "n": 1536, "k": 1536}],
      "regimes": ["hot", "rotate"],
      "promote": 4,
      "tuning": {
        "max_trials": 128,
        "time_budget_s": 1800,
        "cost_model_trials": 80
      },
      "protocol": {
        "samples": 7,
        "confirm_samples": 15,
        "race_rounds": 15,
        "target_batch_ms": 50
      }
    }
  ]
}
```

Supported families are `mxfp8_fused_fwd`, `mxfp8_prequant_fwd`, and
`mxfp8_bwd`. Additional kernel families can register a `DatasetBackend` with
`rtx.autotune.register_dataset_backend`; campaign orchestration, persistence,
verification, and export do not need to change.

## Tests

CPU-safe contract and serialization tests:

```bash
python -m unittest discover -s tests -v
```

GPU kernel tests are skipped automatically when a compatible CUDA target is
not available.

Additional kernel design and autotuning details are in `AUTOTUNING.md` and
`rtx/autotune/README.md`.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
