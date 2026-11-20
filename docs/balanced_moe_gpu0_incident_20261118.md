# GPU0 collection fault: cause and correction

## What happened

The HOME GPU0 collection emitted 175 Xid 31 MMU read faults from 17:21:13 to
17:46:27 local time on 2026-11-18. Nested TorchInductor tuning workers failed,
but Inductor ignored their candidates, selected fallbacks and let the parent
exit zero. The old collection driver accepted that exit and `status=complete`.
That was an evidence-validation failure, not proof of GPU health.

The collector has exited. No reset, driver repair, reboot or GPU1 operation
was performed. No full sweep has been restarted.

## Concrete cause

`VirtualLayer` used its first independently optimized Parameter as the input
to an `as_strided` view extending over the entire shared packed allocation.
This is valid on its real storage. However, Inductor's subprocess benchmark
request recreated only the first Parameter's logical tensor extent.

The captured failing BMM has batch=24, M=1952, N=256 and K=384. Its kernel
needs 2,359,296 BF16 weight elements. The request describes a `[128,384]`
logical projection: only 49,152 elements, **48 times too small**. The generated
kernel then reads beyond that scratch allocation.

This was localized with the preserved generated kernel, full baseline and
actual benchmark-request metadata, not inferred merely from an Xid code:

- The same generated candidate on correctly sized tensors passes Compute
  Sanitizer with zero errors and relative L2 0.001655 against FP32 BMM.
- A standalone equivalent BMM with subprocess max-autotune also passes.
- The original packed BF16 checkpoint baseline reproduces the illegal read.
  The new guard stops its entire process group with code 76.
- Disabling generated-template caching does **not** fix the metadata error.
- Current-process tuning is a useful diagnostic workaround, but it is not the
  final solution and is not required by the fixed VirtualLayer.

## Structural fix

`VirtualLayer` now retains its existing complete allocation as a non-persistent
buffer and passes that **full-span input** through its custom autograd view.
The original logical Parameters remain the optimizer leaves. The buffer is
not a new allocation or Parameter and is excluded from state dicts. Forward
does not concatenate, copy or launch a storage kernel. Backward still returns
separate logical gradients and checks every saved Parameter's version.

Packing, device/dtype moves, deepcopy and assign-loading refresh the alias.
There is no caller-side memory/stream API and no global Inductor monkeypatch.
Max-autotune, its candidate set and subprocess isolation remain enabled in
the benchmark. The standalone driver also creates/warmups/captures autograd
on the same side stream, avoiding Torch 2.13's stale-default-stream capture
error without disabling that check or setting a process-global override.

## Collection now fails closed

The shared process supervisor streams child and nested-worker output and
recognizes fatal device-context errors, including markers split across reads.
It terminates its owned process group, including compiler grandchildren, on
fault (76), stall (75) or the optional wall limit (124). A child's zero exit
does not override a fatal warning, including warnings in its final output.
Other workloads/process groups are never targeted.

The MoE study uses that supervisor, checks complete result/source identity,
scans logs and writes separate guarded-completion records. It stops rather
than moving to another context after a fatal fault. Schema 2 includes runtime,
driver and guard identities; old unguarded studies cannot be resumed as if
they had been qualified. Quarantine markers stop accidental ingestion through
directories, directly selected nested journals or ZIPs; audit reports them
as errors. Raw records and failure logs are never rewritten or deleted.

CPU-only retrospective scan:

```bash
python -m benchmarks.audit_collection_health /path/to/study \
  --output /path/to/health-report.json
# Explicitly mark confirmed contaminated data, preserving every raw record:
python -m benchmarks.audit_collection_health /path/to/study \
  --quarantine --output /path/to/health-report.json
```

## Evidence and qualification

The older `moe_amortization_scaling_20261118` has 14 fault-containing logs;
`moe_search_and_kernels_20261118` has 16. Both roots are quarantined, including
their derived models/exports. Earlier model/regret studies using these data
are not deployment evidence and need clean collection/retraining. This does
not assert that every individual candidate result is wrong; it prevents
silently treating a fault-contaminated bundle as clean.

On HOME GPU0, Torch 2.13.0+cu132/CuTe 4.7:

- New 2D and expert-bank VirtualLayer GPU regressions pass with reentrant
  checkpointing, subprocess max-autotune, all logical gradients and updates.
- The full originally failing mixed pack `[512,2000,768,4096]`, D384/E24/K6/H128,
  completes with max-autotune **and** subprocess tuning enabled. MXFP8 and
  checkpointed BF16 full/expert paths pass FP64 expert and replay checks.
  Five synchronized live-allocation samples are exactly 131,206,656 bytes
  (zero growth); peak allocated is 355,809,280 bytes.
- Three additional GPU tests pass through native MXFP8/NVFP4 linears and
  MXFP8 SwiGLU, including graph replay and independent logical gradients.
- The final focused CPU regression is 86 passed, five CUDA skips and seven
  subtests. Torch 2.12.1+cu132 separately passes 47 CPU tests with five CUDA
  skips; that is not a Torch 2.12 GPU qualification.
- No new kernel Xid entries occur after 18:00 during the fixed-code checks.
  GPU0 is released after testing. GPU1's independent workload is untouched.

These are fault/correctness qualifications, not a new performance claim.
The exact qualified runtime digest is
`b107d81ac29f0d71ba9880dbe2f7343f043962ccdc2902b4c1300b368e2f79fe`;
`qualified-source.tar.gz` preserves it before offline-only follow-up edits.
Raw reproduction logs, request metadata, generated source and test evidence
are retained separately from contaminated studies in
`autotune_reports/moe_gpu0_fault_20261118/`.
