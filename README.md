# RTX low-precision linear layers

`rtx-mxfp8` is an experimental Python library for trainable MXFP8 linear
layers and empirical kernel autotuning on NVIDIA RTX Blackwell GPUs. The
current implementation contains:

- fused BF16 input/weight quantization and MXFP8 forward GEMM;
- materialized dynamic MXFP8 quantization plus GEMM;
- MXFP8 backward for `dX` and the FP32-accumulating long-reduction `dW`;
- PyTorch custom-op and `nn.Module` frontends;
- persistent random, gradient-boosted cost-model, bandit, and local search;
- calibrated hot/rotating-cache measurements and paired finalist races; and
- portable cross-device datasets exported as CSV or Parquet.

The native kernels target SM120/SM121. This is research software: kernel and
dataset schemas are versioned, but APIs may still change.

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
from rtx import MXFP8Linear, mxfp8_linear

x = torch.randn(512, 1536, device="cuda", dtype=torch.bfloat16)
w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)

y = mxfp8_linear(x, w)
layer = MXFP8Linear(1536, 1536, bias=False, device="cuda", dtype=torch.bfloat16)
y2 = layer(x)
```

The final public layer pair is `rtx.MXFP8Linear` and `rtx.NVFP4Linear`.
Both accept BF16 activations and weights, return BF16, expose the usual
`nn.Linear(in_features, out_features, bias=False, ...)` parameter layout, and
use the registered MXFP8 backward kernels. `NVFP4Linear` has a registered
forward/fake/autograd boundary, but its NVFP4 forward kernel is intentionally
not implemented yet.

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
