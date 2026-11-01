# Scratch lifecycle and backward-search fixes, 2026-11-01

## Result

Reproduced the FlashDETR scratch/autotuning reports on `work_5090`, then tested
upstream fixes in a separate checkout. Existing application sources, qualified
selection files, previous evidence, services, drivers and GPU power policy were
not replaced. No change to NVFP4 precision policy or experimental backward was
included in this patch.

The initial proof below used an explicit startup execution owner/retirement
boundary. The follow-up removes that requirement for ordinary compilation;
explicit controls remain optional diagnostics, not required application code.

## Automatic default-mode follow-up

The primary contract is ordinary
`torch.compile(module, fullgraph=True, dynamic=False)`, without
`mode="reduce-overhead"`. Neither user-owned scratch nor RTX stream/resource
scopes are required. Compiled callables retain their accessed working sets;
compatible ordered FP32 reduction workspaces reuse size-classed storage.
Cold tuning trials are excluded from model ownership. CUDA graphs, if the
application enables them, transfer ownership and retire duplicate preparation
contexts at completed capture boundaries.

On supported Torch 2.13.0+cu132, the targeted CUDA matrix passed **98 tests and
49 subtests**: default-mode and separately graph-enabled MXFP8/NVFP4 twelve-
projection models, independent FP32 forward/gradient references, GLU and
recomputed MLP execution, first-hit linear tuning, interleaved checkpoint/
teacher execution, graph lifetime and scratch cleanup. Twelve contexts exceed
the normal eight-entry runner cache; the compiled working set avoids hot-step
cache insertion/eviction churn. CPU policy tests also verify that tuning failure
restores ownership and that explicit/active owners survive cleanup.
The separate compiled-MLP cache/balanced first-hit selection and refinement
checks passed both cases, including subsequent cache reuse and capture.

The unchanged batch-16 synthetic FlashDETR application passed six measured
steps over three packed cases with **CUDA graphs disabled**, normal cache limit
eight, no `ExecutionResources`, no retirement callback and no allocator cleanup.
Model/loss each compiled once, required gradients were finite, and frozen
backbones had no gradients. Median step time was **255.93 ms**. Device usage
was **14.037 GiB** at both warm and final checkpoints; unique explicit RTX
scratch was **1.584 GiB** at warmup. Per-step Torch allocation values vary with
the application cases/temporary lifetimes and are not claimed constant.

The full application still uses its benchmark-only Torch 2.14.1 override;
it is not supported-stack release certification or convergence evidence.
The separate graph-enabled compatibility run also passed six measured steps,
using the same ordinary API and cache limit eight. Device usage stayed at
**13.947 GiB**, live Torch allocation was identical across all six steps,
and median step time was **244.07 ms**. Model/loss each compiled once.
Preparation ownership was retired internally at capture completion, without
an explicit resource scope, retirement callback or `empty_cache()` call.
This compatibility run enabled the application's graph option, not a
`reduce-overhead` compilation mode; it is not the primary default-mode result.
Artifacts are under `autotune_reports/flashdetr_automatic_resources_20261101/`.
Copied application test trees borrowed Git metadata for the harness's revision
field; the recorded source SHA-256s and isolated snapshot paths identify the
actual tested source, rather than that inherited revision string.

## Initial explicit-owner memory proof: physical batch 16

| Measurement | Baseline | Fixed |
|---|---:|---:|
| Steady device usage | 15.814 GiB | 13.910 GiB |
| Steady allocator active blocks | 4.120 GiB | 2.228 GiB |
| Graph-pool reserve | 10.533 GiB | unchanged |

Within the fixed process, selective retirement removed 68 preparation-context
entries (1,696,137,216 bytes of unique referenced storage). Explicit persistent
storage fell from 3.111 to 1.531 GiB. The execution arena retained two 16 MiB
lanes instead of retaining each context's split workspace. Live graph pins and
execution/cache owners overlap: their byte totals must not be added.

The first implementation released live tensors but left most reserve in a
shared private pool. This negative result was preserved. Per-stream pool
ownership, selective pool retirement and weak routing references resolve that
additional retention mechanism. Private scratch tensors retain their own pool
through live graph lifetime, including after eviction.

The final model ran twelve measured steps after warmup over three packed cases.
Model/loss each compiled once; there was no steady-state recompilation or growth
in the measured live-allocation counter. All required gradients were present
and finite, and frozen DINO/Ettin had no gradients. Step times stayed around
236–249 ms, comparable to the 238–250 ms instrumented baseline; this study does
not claim a throughput improvement from the ownership change.

The application is a real architecture with random frozen backbones and
synthetic annotations. These are execution/memory results, not convergence or
accuracy qualification. Its Torch 2.14.1+cu132 environment uses the explicit
benchmark-only runtime override, not expanded library support. The upstream
CUDA regression suite runs on supported Torch 2.13.0+cu132 / CuTe 4.7.0.

The original application's `device_used_after_warmup` field is sampled before
the retirement callback. Use the callback's explicit before/after measurements
and the final allocator snapshot for post-retirement memory, not that field.

## Backward: complete dX+dW, including preparation and FP32 reduction

Ordinary CUDA-event graph-replay timing on the idle RTX 5090, using two warmup
calls, five samples and five replays/sample:

| Forward problem M×N×K | Original fused unsplit | Complete materialized candidate |
|---|---:|---:|
| 8500×256×256 | 2726.9 µs | 100.4 µs |
| 8512×256×256 | 2706.4 µs | 100.4 µs |
| 8192×64×256 | 2379.3 µs | 38.8 µs |
| 136000×256×256 | 44816.9 µs | 686.0 µs |

A fresh eight-trial progressive search selected about 98.5 µs on both 8500 and
8512 rows. Its first proposals reach complete materialized dW, combined
materialized dX/dW and fused split schedules. Observed geometric CTA coverage is
recorded in the tuning evidence; it is not measured occupancy or tensor-pipe
utilization. These are bounded searches, not claims of global optimality.

No padded BF16 input or explicit transpose is needed for 8500 rows. Separate
row-major transpose quantizers predicate the final K tile and zero-fill the
final E4M3 block. Register-loaded 32-value transport is the supported ragged
materialized path; unsupported ragged dual/quad paths fail static legality.

Independent FP32 GEMM references give approximately 3.7–3.8% normalized L2 error
for both gradients, unchanged in scale from the original MXFP8 policy.
Candidate-versus-original normalized L2 is at most 0.000306 in this matrix;
dX was byte-identical. Retirement/alias-lifetime tests require byte-exact replay.
No tolerance was relaxed to accept a missing or nonfinite contribution.

## Coverage and safety boundaries

The isolated clean-source CUDA suite passed 85 tests plus 45 subtests, covering
ordinary/ragged/narrow backward, split and atomic reduction, dual streams,
GLU/MLP private replay, cache cleanup, allocation pressure, graph reset, multiple
live graphs, one-entry-LRU warmup, orphan-pool reclamation and concurrent host
submissions on independent streams. NVFP4 block, current, JIT-region and delayed
forward graph ownership are included. Focused CPU/search tests additionally
exercise budget/resume, active-coordinate identity, schema invalidation and
shape-derived legacy-cache lookup. A final capture-retirement rejection guard
has a dedicated regression ensuring it fails before synchronization.

Do not alias arbitrary cross-context quantized operands: MLP preparation can
keep multiple operands live. Existing serial/partial reuse policies remain
explicit, legality-checked tuning candidates. Manual preparation retirement
requires a quiescent host-submission boundary; automatic capture retirement
uses library-owned submission guards. Raw CUDA graph capture bypassing PyTorch's
public capture methods still requires explicit resource retention.

Nsight Systems/Compute/Python were unavailable on this host. The evidence uses
CUDA events, complete-path tests, allocator snapshots and explicit storage
ownership; it makes no hardware-counter claims. The earlier isolated cold-tune
illegal-memory-access report was not reproduced here and is not claimed fixed
solely by these results.

## Reproduction and artifacts

```bash
python -m pytest -q \
  tests/test_managed_execution.py tests/test_managed_execution_cpu.py \
  tests/test_execution_resources.py tests/test_bwd_structural_coverage.py \
  tests/test_scratch_cleanup.py tests/test_workspace_mlp_interleaved.py \
  tests/test_mlp_kernel_replay.py tests/test_mxfp8_bwd.py

mkdir -p autotune_reports/backward_resources_check
python -m benchmarks.qualify_backward_search \
  --output autotune_reports/backward_resources_check/results.json --tune
```

`benchmarks/qualify_execution_memory.py` runs the existing FlashDETR diagnostic
application without modifying it; its CLI documents the application path,
optional benchmark-only Torch override and arena size. By default the wrapper
only records diagnostics: it does not supply a resource owner or perform
cleanup. `--explicit-resources` reproduces the historical manual control at
the application's `graphs_warm` boundary. See [execution resource
usage](execution_resources.md) for ordinary integrations.

Raw logs, source identities, tuning residuals and allocator evidence are kept
under `autotune_reports/flashdetr_runtime_fix_20261101/` locally and the isolated
`/home/achyad/rtx-resources-release-20261101` / application artifact directories
on `work_5090`. They remain ignored rather than being committed as public data.
