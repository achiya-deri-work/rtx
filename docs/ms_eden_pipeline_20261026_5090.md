# MS-EDEN pipeline: work_5090 qualification

## Current status

The isolated RTX snapshot passed all 38 focused paired/producer cases across
two invocations: 36 initially passed, while two compiled checkpoint/allocation
tests could not find a C compiler. After provisioning a user-local compiler,
those exact two cases passed. No kernel code or numerical gate was changed.
This is correctness evidence, **not a performance result**: unrelated root-owned
Triton inference containers were active throughout.

The existing [5060 study](ms_eden_pipeline_20261025.md) remains the performance
evidence: long-shape paired+async-2 backward was 1.143x faster; paired execution
alone reduced short-run model training wall time by 7.9%. MXFP8 backward still
won the short training comparison. Neither result is a 5090 claim.

## Environment and artifacts

- SSH alias: `work_5090`; hostname `g2-ps-1000`; account `achyad`.
- RTX 5090, GPU UUID `GPU-0b61630a-ee35-5691-1f49-b678c6a11bdc`, driver 595.99.02.
- Isolated source: `/home/achyad/rtx-pipeline-20261026.LPss9H`.
- Python 3.13.17, Torch 2.13.0+cu132, TorchAO 0.18.0+cu132, CuTe DSL 4.7.0.
- User-local Zig 0.17.0 supplies `CC=$PWD/rtx-5090-cc` and
  `CXX=$PWD/rtx-5090-cxx` for Inductor. System Python/services were not modified.
- Local evidence: `autotune_reports/ms_eden_pipeline_20261026_5090/` contains
  `focused.log`, `compiled-recheck.log`, and `environment.txt` (dependencies,
  source hashes, GPU identity and concurrent load).

Tests cover identical-draw serial/paired gradients, async producer geometry,
streams, graph replay, checkpointing, and allocation behavior. These are a
focused experimental-backward qualification, not the whole library's release
matrix. The local CPU-only tuner regression selection also passed 28 tests.

## Next execution gate

The user authorized stopping workloads on this host, but `achyad` lacks Docker
socket permission and `sudo -n` requires interactive authentication. No workload
was stopped. An administrator must stop the GPU containers or grant suitable
access before performance qualification. Do not benchmark through inference
contention or silently relax stationarity gates.

2026-10-27 clarification and read-only recheck: authenticated sudo succeeded,
and `sudo docker ps` listed the running containers. The earlier `sudo -n`
failure meant password authentication was required, not that `achyad` lacked
sudo privileges. The user authorized this verification without stopping work;
no containers were stopped, restarted or modified. This access verification
does not authorize benchmarking on the currently occupied GPU. Credentials
are not stored in this repository.

Once the GPU is exclusively available, use the existing comparator for serial,
paired, and paired+async-2/3, then tune complete backward per shape. Preserve
original-BF16 inputs, FP32 scaling/correction, independent stochastic draws,
and the stable layer's MXFP8 backward default. Keep per-shape timing and memory
guards; follow with matched compiled model training before promoting anything.

The 5060 connection was subsequently restored on October 26. Its cache-hit
and timeline artifacts were retrieved and verified; see the final section
of the [5060 study](ms_eden_pipeline_20261025.md). The 5090 performance gate
above remains unchanged.
