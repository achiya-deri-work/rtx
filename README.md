# RTX low-precision linear layers

`rtx` is an experimental Python library for low-precision training and
inference on NVIDIA RTX Blackwell GPUs. Its two public linear
frontends are `rtx.MXFP8Linear` and `rtx.NVFP4Linear`. The
current implementation contains:

- fused BF16 input/weight quantization and MXFP8 forward GEMM, including
  persistent three-role TMA producer/quantizer/MMA schedules and staged async
  TMA epilogues;
- materialized dynamic MXFP8 quantization plus GEMM, including autotunable
  one-to-four-stage mainloops and epilogues, locality scheduling, and up to
  eight output tiles per persistent CTA;
- fused dynamic MXFP8 backward for `dX` and the FP32-accumulating `dW`,
  including logical-transpose TMA transport and split-FP32 workspace reduction;
- a one-launch four-operand backward quantizer with concurrent dX/dW GEMMs,
  plus autotunable fused and quantize-once FP32 workspace/atomic split-K,
  generation-counted cluster-local DSMEM FP32 split reduction,
  per-matmul, interleaved, asynchronous logical-transpose transport,
  wide-store, wide-CTA, and CTA-cluster reuse families;
- PyTorch custom-op and `nn.Module` frontends;
- persistent random, gradient-boosted cost-model, bandit, and local search;
- backend-neutral conditional spaces, staged tasks, and resumable ask/tell
  workers suitable for MoE and attention kernel projects;
- calibrated hot/rotating-cache measurements and paired finalist races; and
- portable cross-device datasets exported as CSV or Parquet.

The executable native kernels target SM120/SM121. SM110/Jetson Thor is outside
the project scope because its TCGen05/TMEM execution model requires a separate
kernel family. This is research software:
kernel, packed-operand, and dataset schemas are versioned, but APIs may still
change.

## Requirements

- Linux
- Python 3.11+
- an RTX Blackwell GPU and a compatible NVIDIA driver
- a CUDA 13.x-enabled PyTorch build with Blackwell support (CUDA 13.2 preferred)
- TorchAO 0.18.0 or newer
- CUDA Python 13.x
- NVIDIA CUTLASS Python DSL 4.7.x with its `cu13` runtime extra
- PyArrow, Apache TVM FFI 0.1.13.post2, and Einops

For discrete Blackwell GPUs, `requirements.txt` selects PyTorch's official
CUDA 13.2 wheel channel and installs the full runtime:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

CUDA 13.0 is available as a fallback for machines which cannot yet use 13.2:

```bash
python -m pip install -r requirements-cu130.txt
python -m pip install -e .
```

PyArrow is a core dependency, so CSV and Parquet export are available in every
supported install.
For development tools:

```bash
python -m pip install -e '.[dev]'
```

The equivalent dependency lists are in `requirements.txt` and
`requirements-dev.txt`. An editable install is recommended while kernels are
changing because every dataset bundle records a hash of the installed source.

## Repository layout

```text
rtx/
├── rtx/
│   ├── configs/        # immutable kernel and inference configuration models
│   ├── formats/        # TorchAO-backed MXFP8/NVFP4 tensor contracts
│   ├── kernels/        # lazily imported CuTe DSL kernel implementations
│   └── autotune/       # adapters, bandits, cost models, stores, and campaigns
├── autotune_manifests/ # immutable portable campaign specifications
├── benchmarks/         # focused developer benchmarks and tuning utilities
└── tests/              # CPU contracts plus CUDA-gated kernel tests
```

The public runtime surface is `rtx.MXFP8Linear`, `rtx.NVFP4Linear`, their
functional forms, and packed tensor types. Kernel modules remain lazy so
importing `rtx` on a non-Blackwell or CPU-only machine does not initialize
CuTe. Legacy coordinate tuners remain as compatibility modules; new search
policy work belongs under `rtx.autotune`.

The portable boundary is independent from CuTe and MXFP8: external projects
define a `ConditionalSearchSpace` and `FunctionKernelTask`, then use
`StagedTaskAdapter` with either the synchronous orchestrator or an
`AskTellSession`. Existing project-specific `KernelAdapter` implementations can
be exposed through `AdapterKernelTask` without being rewritten. See
[`rtx/autotune/README.md`](rtx/autotune/README.md) for a complete example.

See [`autotune_manifests/README.md`](autotune_manifests/README.md) for campaign
status and [`benchmarks/README.md`](benchmarks/README.md) for focused tools.

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

Shape/stream-specific dynamic runners retain quantized workspaces for reuse.
These caches are bounded to eight entries per execution family by default.
Use `RTX_MXFP8_RUNNER_CACHE_ENTRIES`, or a family-specific variable such as
`RTX_MXFP8_BACKWARD_CACHE_ENTRIES`, to change the bound. Applications which
finish a variable-shape phase can release every retained workspace safely:

```python
released = rtx.clear_runtime_caches()  # synchronizes before releasing buffers
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

`rtx.MXTensor` and the compatibility spelling `rtx.MXFP8Tensor` are direct
aliases of TorchAO's
[`MXTensor`](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/mx_tensor.py).
Likewise, `rtx.NVFP4Tensor` is TorchAO's
[`NVFP4Tensor`](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/nvfp4_tensor.py).
RTX-produced E4M3/E8M0 storage is wrapped without copying; TorchAO blocked
scales are exposed to CuTe through zero-copy kernel-shape views. NVFP4 uses
TorchAO's packed qdata, E4M3 block scale, and optional FP32
`per_tensor_scale` fields; an absent global scale is treated as exactly one.
Packed module weights remain persistent raw buffers and do not retain a BF16
master weight.

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

New campaigns can exercise the same portable lease/worker boundary used by
external projects while retaining the existing residual stores and campaign
verification:

```bash
rtx-autotune run manifest.json --device cuda:0 \
  --execution-engine ask_tell --output-dir autotune_datasets
```

Long campaigns may also add `--reuse-deterministic-failures`. This writes an
append-only ledger for exact architecture, compiler, kernel revision, workload,
and configuration matches. It reuses only static/compile failures, never
latencies, correctness failures, or runtime failures, and keeps prospective
treatments and replicates isolated.

Use `--format parquet` or `--format both` to emit Parquet datasets.
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

For new campaigns, the hierarchical bandit manifest makes random exploration,
gradient-boosted global search, and model-guided local search competing,
resumable arms. Adding the context allocator distributes wall time across
shapes while retaining a 32-trial coverage floor and a bounded milestone lead:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_bandit_v1.json \
  --device cuda:0 \
  --output-dir autotune_datasets \
  --format both \
  --calibration hardware_calibration.json \
  --wall-time 12h \
  --context-slice 2m \
  --context-orchestration bandit
```

Use `--strategy-orchestration bandit --strategy-bandit-exploration 0.35` with
an existing manifest to override its per-context scheduler without changing
that manifest's digest. Both levels replay their append-only history after
interruption. Strategy decisions live in each store's `events.jsonl`; context
decisions live in `context_allocations.jsonl` and are exported to CSV/Parquet.

### Prospective 5070 optimizer study

The dedicated study compares pure random coverage, random followed by
coordinate-local search, and the online learned/bandit portfolio. It uses two
replicates and rotates the launch order so every interrupted prefix remains
balanced across optimizer treatment, kernel family, shape category, cache
regime, and replicate.

On the RTX 5070 laptop:

```bash
git pull
./benchmarks/run_5070_autotuner_study.sh laptop-3h
```

On the RTX 5070 Ti:

```bash
git pull
./benchmarks/run_5070_autotuner_study.sh ti-6h
```

Each tuning context owns independent `observations.jsonl`, `sessions.jsonl`,
`events.jsonl`, and `verification.jsonl` residuals. A truncated tail is skipped
and isolated before the next append, so one damaged file cannot corrupt the
campaign or the first resumed observation. `SAVE` records are fsync'd. Pressing
Ctrl-C and running the same command resumes at absolute 4/8/16/32/64-trial
milestones. The script intentionally uses `--format none`; raw residuals are
authoritative and no large monolithic export is rewritten during collection.
Sticky CUDA faults such as illegal instructions are saved against the candidate
that caused them, then terminate only that worker process. The launch script
automatically starts a fresh CUDA context with the remaining global wall-time
and resumes the existing residuals. It also preserves compatible context
identity across these runner-only upgrades. The launcher pins `CUDA_HOME` to
the available CUDA 13.2 toolkit when the calling shell omitted it, keeping
libNVVM discovery and the compiler/machine fingerprint stable across restarts.
It also restores the toolkit's `nvcc` path and unbuffers progress output.

After copying the two `autotune_datasets/` trees together, create the optimizer
comparison only once:

```bash
rtx-autotune audit \
  autotune_datasets/mxfp8_autotuner_prospective_5070_v1 \
  laptop-results.zip \
  --output autotune_reports/5070_prospective_v1.audit.json

rtx-autotune summarize-tuners \
  autotune_datasets/mxfp8_autotuner_prospective_5070_v1 laptop-results.zip \
  --output autotune_reports/5070_prospective_v1 \
  --format both
```

The report includes validity and compiler-waste rates, evaluator time to first
valid candidate, best latency and observed-oracle regret at 1/4/8/16/32/64
trials, and treatment aggregates per machine and kernel family. Schema v3 also
records minimum/median/maximum context coverage, exact shape/cache-regime
splits, compile-failure and wasted-compiler-time rates, matched deltas and the
empirical probability of beating random search with deterministic bootstrap
confidence intervals. A context which silently exhausts early is therefore
visible in the primary report rather than only in its residual journal.

`audit` is read-only. It verifies JSONL tails, interior corruption, machine and
manifest identities, unit coverage, observation/config duplicates, and
repeated confirmation keys before any bundle is ingested. A single malformed
final line is reported as a recoverable crash tail; malformed interior records
or conflicting identities make the command fail.

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
├── context_allocations.jsonl
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

New machine snapshots expose independent `architecture_id`, `device_id`,
`compiler_id`, `environment_id`, `calibration_id`, and `kernel_source_id`
fields under `identities`. The compatibility `machine_id` still selects the
bundle directory. Offline models should group by the narrowest identity needed
for a claim instead of treating a compiler PATH or calibration change as a new
physical GPU.

Campaigns using `storage_mode: residual_context` place the same four journals
under one directory per family/treatment/replicate/category/regime/shape/context
instead of the shared `stores/<kernel>/<regime>` path shown above.

## Pretraining the autotuner

The CPU-only pretrainer accepts copied directories and ZIP archives directly.
Filter to compatible campaigns so obsolete or interrupted kernel revisions
cannot enter the artifact accidentally. Repeat `--campaign` when a device used
an earlier campaign name but the same kernel revisions and feature schema:

```bash
rtx-autotune pretrain autotune_datasets autotune_datasets.zip \
  --campaign mxfp8_blackwell_cross_device_bandit_v1 \
  --output autotune_models/mxfp8_blackwell_bandit_v1 \
  --seed 20260809
```

It writes revision-scoped absolute-latency and context-ranking models, exact-SKU
heads validated on unseen contexts, an end-to-end feasibility classifier,
confidence-qualified paired-coordinate rules, and leave-one-device-out
validation metrics. Unknown SKUs only receive models that passed cross-device
replay; an exact SKU model must separately beat random replay in at least three
of four context-held-out folds. Use the artifact as a soft proposal prior:

```bash
rtx-autotune run autotune_manifests/cross_device_dataset_bandit_v1.json \
  --device cuda:0 --output-dir autotune_datasets --format both \
  --wall-time 2h --context-orchestration bandit \
  --pretrained-artifact autotune_models/mxfp8_blackwell_bandit_v1
```

Only a head that beats matched random catalogue replay on every held-out device
is allowed to propose from the first model-guided trial. Otherwise the family
keeps its ordinary online model and uses only validated feasibility/rule priors.
Four initial random trials retain local exploration for an enabled model, and
measured local data refits it after the configured adaptation interval.
Conditional rules only adjust ranking; they never reject a legal candidate or
install an unverified winner.
Artifacts are ignored by Git because they are reproducible from the source
JSONL datasets.

Keep a prospective study out of training, then evaluate the frozen artifact on
it explicitly:

```bash
rtx-autotune evaluate-pretrained \
  autotune_models/mxfp8_blackwell_bandit_v1 \
  prospective-5070-ti/ prospective-5070-laptop.zip \
  --output autotune_reports/mxfp8_blackwell_bandit_v1.heldout.json
```

The command rejects identical dataset digests or overlapping source files by
default and records both input identities. It never changes the artifact's
deployment gates.

After audit and numerical verification, preview and install runtime winners:

```bash
rtx-autotune install-winners copied_datasets/ laptop-results.zip \
  --minimum-support 2 --dry-run
rtx-autotune install-winners copied_datasets/ laptop-results.zip \
  --minimum-support 2
```

Promotion is atomic and keyed by kernel family, exact device fingerprint,
shape, cache regime, and packed scale-layout variant. Existing entries are not
overwritten unless `--force` is supplied.

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

Supported families are `mxfp8_fused_fwd`, `mxfp8_prequant_fwd`,
`mxfp8_weight_prequant_fwd`, `mxfp8_fully_prequant_fwd`, and `mxfp8_bwd`.
The two persistent inference families never time their AOT packing work and do
not expose inactive quantizer coordinates. Additional kernel families can
register a `DatasetBackend` with
`rtx.autotune.register_dataset_backend`; campaign orchestration, persistence,
verification, and export do not need to change.

After the active v2 campaign is collected, run the bounded inference pilot:

```bash
rtx-autotune run autotune_manifests/inference_states_pilot_v1.json \
  --device cuda:0 --output-dir autotune_datasets --format both
```

Every verified winner is also written below the bundle's `runtime_winners/`
directory. Point a packed layer's `autotune_cache_dir` at that bundle to use a
matching hot-regime winner. Cache identity includes exact device/software
fingerprint, M/N/K, operand state, and physical scale layouts.

## Tests

CPU-safe contract and serialization tests:

```bash
python -m unittest discover -s tests -v
```

GPU kernel tests are skipped automatically when a compatible CUDA target is
not available.

Additional kernel design and autotuning details are in `AUTOTUNING.md` and
`rtx/autotune/README.md`. The RTX-specific hardware boundary and upstream
attribution are documented in `sm120_hardware.md`.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
