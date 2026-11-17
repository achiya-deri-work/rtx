# BalancedRouter qualification — 2026-11-15

Historical initial bring-up: this page records the earlier CUB-backed graph
implementation. Current production uses CuTe DSL plus cuda-python; see the
[CuTe/autotuner qualification](balanced_router_cute_autotune_20261115.md).

The experimental routing foundation passes the focused suite on both laptops.
This qualifies fixed-K routing metadata and its gradients, **not** expert FFNs,
whole-model learning, peak routing performance or first-hit autotuning.

## Executed matrix

| Environment | Result |
|---|---|
| RTX 5070 Laptop, driver 610.88, Torch 2.13.0+cu132, CuTe 4.7.0 | 35 passed, no skips |
| RTX 5060 Laptop, driver 621.14, Torch 2.13.0+cu132, CuTe 4.7.0 | 35 passed, no skips |
| RTX 5060 Laptop, driver 621.14, Torch 2.12.1+cu132, CuTe 4.7.0 | 35 passed, no skips |
| Local CPU documentation/routing/VirtualLayer/managed-execution/resource regression | 64 passed, 293 documentation subtests; 28 CUDA-only cases skipped |

Native builds used NVCC 13.2.86. Existing environments and unrelated workloads
were not changed. Wheel construction succeeded and includes both the CuTe
Sinkhorn source and `rtx/kernels/routing_schedule.cu`.

The 35-case suite comprises 21 CPU policy/configuration tests and 14 actual
CUDA cases, including:

- Native Sinkhorn forward/backward on `(N,E,K,iterations)` of `(1,1,1,5)`,
  `(14,7,3,5)`, `(64,8,1,1)`, `(68,16,4,3)`, `(192,32,8,5)` and
  `(4096,128,8,5)`, with multiple reduction geometries.
- Exact hard-policy parity on frozen random and uniform scores, including
  repair-budget exhaustion and K=E. An exhausted private scheduler returns its
  finite-round assignment; the public router rejects it if not balanced.
- Real-device `opcheck`: schema, autograd registration, fake tensors and AOT
  dynamic-dispatch checks for private Sinkhorn and scheduling operators.
- Cold `torch.compile(fullgraph=True, dynamic=False)`, forward/backward graph
  capture, changed-input/parameter replay and live-graph cache cleanup.
- A 1,024-token/128-expert router with all input/parameter gradients finite;
  repeated live-memory measurements and independent-stream output isolation.

Symbolic-size public deployment/export and concurrent replay of one captured
graph on different streams are not established by these checks.

## Numerical evidence

Sinkhorn is compared against an independent FP64 matrix-normalization oracle,
not the native kernel's row/column-potential decomposition. Upstream gradients
are noncontiguous. Across the six cases on both GPUs:

| Quantity | Largest observed error |
|---|---:|
| Log-transport maximum absolute error | 1.002e-6 |
| Log-transport normalized RMS error | 8.861e-8 |
| Score-gradient maximum absolute error | 1.039e-6 |
| Score-gradient normalized RMS error | 1.083e-7 |

Normalized RMS is RMS error divided by `max(reference RMS, 1e-12)`. The 1×1
case is exactly zero in forward and backward. Gates are elementwise
`rtol=5e-5, atol=5e-6` for transport and `rtol=5e-4, atol=8e-6` for score
gradients. Seed is 744. These FP32 errors are not quantized-expert accuracy
measurements.

Saved-tensor hooks verify final transport and normalization history remain
available. Backward runs with forward kernels replaced by functions that raise
if called, proving the tested backward does not recompute Sinkhorn.

Full-router gradient checks freeze actual hard assignments, then differentiate
an ordinary Torch transport/gather/softmax reference. CPU checks additionally
construct affinities independently in FP64. Native integration compares every
input/parameter gradient with `rtol=0.015, atol=0.001`, allowing BF16 rounding
and cancellation-sensitive scalar gradients. No gradient is propagated through
integer assignment or repair.

## Memory and runtime correction

After three warmup steps, eight additional forward/backward steps leave live
allocated bytes unchanged at **68,320,256** on each Torch 2.13 laptop run.
This is total process live allocation for that fixture, not a universal
workspace size. Stable allocator reserve alone was not used as leak evidence.

Bring-up exposed a 5060 crash during instantiation of a conditional graph
containing a captured child graph. A minimal empty-body conditional worked;
linking the same CUDA runtime as Torch and pre-instantiating children did not
alone solve it. The retained construction populates the conditional body
directly, copying captured nodes through **driver-level** graph APIs while
preserving dependencies and kernel launch attributes. Runtime-level copying
incorrectly interpreted driver kernel handles as host symbols and was rejected
with `invalid device function`; that construction is not retained. The final
driver-level construction passes the complete matrix on both laptops.

This localizes and avoids the failing construction; it does not prove a general
NVIDIA driver defect. Unsupported graph-body node types fail explicitly.

## Reproduction and remaining work

From the repository with the supported stack and a CUDA 13.2 NVCC/CCCL toolkit:

```bash
CUDA_TOOLKIT_PATH=/path/to/cuda-13.2 \
TORCHINDUCTOR_COMPILE_THREADS=2 \
python -m pytest -xq tests/test_routing.py \
  -o junit_family=xunit1 --junitxml=routing-results.xml
```

JUnit records per-case absolute/RMS/normalized errors, seed, GPU, Torch version
and before/after live allocations. Final logs/XML are also in the isolated
test snapshots `/tmp/rtx-routing-20261115.cowvat` on `laptop_5070` and
`/tmp/rtx-routing-20261115.MRErRT` on `personal_laptop`. Their prefixes are
ephemeral reproduction artifacts, not required library paths. Torch 2.13
results use `routing-final.log`/`routing-final.xml`; the additional 5060
Torch 2.12.1 run uses `routing-torch212.log`/`routing-torch212.xml`.

Next work remains explicit: measured phase baselines; shared persistent
autotuner integration; further handwritten QB/compacted-repair work if profiles
justify it; grouped expert execution; and whole-model learning qualification.
No performance speedup is claimed from pytest wall times.
