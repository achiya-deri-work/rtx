# Runtime, checkpoint, and distributed contract

## Supported runtime

| Component | Supported contract |
| --- | --- |
| GPU | NVIDIA RTX Blackwell SM120/SM121 |
| OS | Linux |
| Python | 3.11–3.13 |
| PyTorch | 2.12.1 or 2.13.0 with CUDA 13.2 |
| TorchAO | 0.18.0 |
| CUTLASS DSL | 4.7.0, `nvidia-cutlass-dsl[cu13]` |
| Input/master dtype | BF16 |
| Output dtype | BF16 |
| Bias | Unsupported in the base linear API |
| `torch.compile` | `fullgraph=True, dynamic=False` production contract |

Importing `rtx` is CPU-safe and lazy. Native execution validates the runtime
stack and compute capability when a kernel family is first loaded.

## Dynamic checkpoints

Dynamic `MXFP8Linear` and `NVFP4Linear` retain a BF16 `weight` Parameter with
the same name and shape as a no-bias `nn.Linear`. Their nonpersistent runtime
caches and scale telemetry are excluded from `state_dict()`. Consequently a
single dynamic BF16 checkpoint can be loaded strictly into BF16, MXFP8, or
NVFP4 variants with matching module structure.

NVFP4 delayed amax history is intentionally not checkpointed. Loading a state
dict resets it and the next call bootstraps from current data.

## Packed checkpoints

Packed modules are inference-only and intentionally use a different schema:

- MXFP8 stores qdata, E8M0 scales, and versioned layout metadata buffers.
- NVFP4 stores packed E2M1 qdata, E4M3 block scales, the FP32 tensor scale, and
  versioned layout metadata buffers.
- no BF16 master weight is retained;
- dtype conversion is rejected; device-only movement is supported;
- a packed checkpoint must be loaded into a module constructed in the same
  packed operand state.

The packed operand schema is versioned independently from kernel revisions.
Requantize from the BF16 master checkpoint when changing numeric format or
when a future release announces an incompatible packed schema.

RTX schema v3 keeps NVFP4's numerical block size at 16 but requires physical K
storage to be a multiple of 32 so every packed FP4 row has the 16-byte stride
alignment assumed by the SM120 native load path. If standalone TorchAO packing
ends on an odd block, repack from the BF16 source with `rtx.quantize_nvfp4`
before using the native GEMM. The extra zero block costs at most eight bytes per
row.

Direct serialization of a standalone TorchAO tensor subclass is governed by
TorchAO. For durable RTX inference checkpoints, prefer the packed module's raw
buffers and metadata rather than relying on private `_rtx_*` tensor attributes.

## Distributed execution

Dynamic modules use ordinary BF16 Parameters and gradients at module
boundaries, so they are structurally compatible with DDP-style gradient
synchronization after local backward. The native kernels themselves perform no
collectives.

The 1.0 release contract does not yet claim validated FSDP parameter sharding,
DTensor placement propagation, tensor parallel packed-layout redistribution,
or distributed checkpoint conversion for packed modules. Treat those as
integration work requiring an explicit production matrix rather than assuming
that tensor-subclass metadata survives arbitrary redistribution.

Autotuning winners are device/SKU/software/shape specific. Every rank may read
the same cache root. Balanced first-hit tuning takes an advisory per-context
filesystem lock, rechecks the cache after acquiring it, and atomically publishes
the winner, so local ranks sharing that filesystem do not compile the same miss.
Independent hosts without a shared cache still tune independently. Production
distributed runs may use `autotune="cache"` with preinstalled winners when any
runtime benchmarking is undesirable.

Runtime-winner schema v2 also keys every entry by the current kernel revision.
Revision changes select a different cache path, so a legal-looking schedule
from an older implementation cannot silently survive an upgrade. Schema-v1
winners did not carry this information and are intentionally invalidated;
reinstall winners from verified source datasets after upgrading.

## Compilation and shape behavior

The production compile target is:

```python
torch.compile(module, fullgraph=True, dynamic=False)
```

Every positive M/N/K is logically supported. Ragged problems use predication,
zero-filled physical tails, or direct stores without eager BF16 padding or an
explicit transpose. A new static shape may select a different runtime winner
and compile a distinct launcher.

Use `module.explain(x)` before execution to inspect routing without compiling
or tuning. Use `rtx.clear_runtime_caches()` when intentionally testing cold
selection behavior.
