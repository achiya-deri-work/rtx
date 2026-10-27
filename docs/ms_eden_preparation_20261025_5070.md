# MS-EDEN preparation diagnostic: RTX 5070 Laptop, 2026-10-25

Current evidence identifies the transposed Hadamard/EDEN producers as the
largest dW cost. An existing coalesced-load schedule is promising, but no stable
default or public backward policy was changed. This is a focused experiment,
not release qualification; historical 5090 timings in
`nvfp4_completion_20261023.md` are not current measurements.

## Scope and validity

- GPU: RTX 5070 Laptop, 36 SM, 8 GiB; Torch 2.13.0+cu132, Python 3.13.16.
- Isolated remote source: `/tmp/rtx-prep-20261025.sQX5dI`; git archive of HEAD,
  then the diagnostic benchmark update. Existing remote projects untouched.
- Shape `(M,N,K)=(2048,768,1536)`, dW, original BF16 sources, MMA128 scales,
  default GEMM. Caller-owned fixed signs/uniforms unless explicitly RNG-inclusive.
- Existing graph8x10, 15-sample, bounded 200-ms warmup timing contract.
- Six direct128/coalesced512 comparisons passed bit-exact packed bytes, scale
  bytes, and tensor factors: sources 256x384, 2048x768, 2048x1536, each logical
  transpose and nontranspose, random and all-zero inputs. This proves schedule
  equivalence for these cases, not a broad training/estimator contract.

## Ordinary CUDA-event measurements

All values below are microseconds. Isolated phases are not additive; complete
gradient timing is authoritative.

| Fresh process | Preparation | GEMM | Complete dW | Complete stationary? |
|---|---:|---:|---:|---|
| direct128 baseline | 147.04 | 26.29 | 216.04 | yes |
| staged amax, direct128 | 146.07 | 25.93 | 214.10 | yes |
| coalesced128 | 88.81 | 25.68 | 144.89 | yes |
| coalesced512 | 83.80 | 26.39 | 137.07 | **no** |
| reverse coalesced512 | 83.16 | 25.83 | 132.92 | **no** |
| reverse direct128 | 143.81 | 25.69 | 208.47 | yes |
| confirm coalesced512 | see JSON | see JSON | 126.44 | **no** |

Coalesced128's single eligible pair is 32.9% faster complete dW. It still needs
reverse-order confirmation and dX/larger-shape checks. The 512-thread numbers
are timing-ineligible, not confirmed wins. Laptop clock/thermal effects were
not controlled; observed drift is sufficient reason to withhold acceptance.

The new opt-in `--preparation-phases` exposes producer/finalizer/product, RNG,
and complete-with-RNG timings. This is diagnostic RNG (`random_`/`mul_`/`sub_`),
not public API RNG (`rand`/`where`) or public compiled-backward timing.
One direct run isolated producers at 43.57/90.54
us and finalizers at 5.45/8.53 us; its fixed complete measurement failed
stationarity. RNG alone was 9.97 us. Do not subtract the separately measured
complete-with-RNG time from fixed complete: they execute different random
rotations, change cache state, and may run under different device clocks.

## Timeline attribution and limitations

One warmed, fixed-randomness dW Nsight Systems capture attributes GPU kernel
time as follows: producer kernels 46.5% + 35.6% = 82.1%; GEMM 10.7%; scale
finalizers 6.9%; factor product 0.3%. This is diagnostic profiler time, not the
ordinary latency table above. There are six kernels. Factor-product fusion is
not a priority on this workload.

Nsight Compute and Nsight Python are absent. No measured DRAM/L2 byte,
occupancy, register, spill, or tensor-pipe claim is made. The source-level
mechanism is strided transpose reads; coalesced mode uses shared-memory staging
but groups only `threads/32` neighboring source columns per CTA.

## Wider source-tile implementation and paired evidence

Implemented opt-in `producer_load="coalesced_wide"`: load twice as many
neighboring source columns into shared memory, then each warp prepares two
logical rows before the next CTA synchronization. This amortizes the two
barriers and source staging over twice the data without doubling CTA threads.
The direct and narrow coalesced paths retain one row per warp. Defaults are
unchanged. The new mode is exposed in the study/tuning CLI, adapter axes,
resource seed prior, and serialized `PosthocPreparationSchedule`; exposure
allows measured exploration, not automatic promotion of these results.

The helper refactor affects all producer modes, so independent validation
covered direct, coalesced, and wide, both transposes, 128/512 threads (12 cases).
Independent FP64 Hadamard plus EDEN reference: packed values exact, corrected
BF16 scales pass the existing midpoint gate, partial maxima agree, and every
partial is written after NaN initialization. Zero-input CUDA graph replay
checks packed bytes/scales/maxima including conservative idle CTA slots.
Sixteen further bit-exact wide/direct checks cover BF16 and saved-packed
sources, two shapes, both traversal orders, and 128/512 threads. CPU resource
and adapter tests: 6 passed. Partial source-column tiles are rejected.
Final current-source CUDA regression: **86 passed** in 40.47 seconds (14 Torch
deprecation warnings): independent original Hadamard tests, all narrow/wide
thread/order/packed geometry tests, source-tile boundary guard, and the complete
experimental API test file including fullgraph/opcheck, changed-randomness
replay, no-live-growth, checkpoint/RNG restoration, mixed policies, missing
gradients, and new wide original/packed compiled-RNG cases. This focused check
is not a repository-wide or release suite. `final-focused-cuda.log` retains it.

Independent timing windows remained nonstationary even with 5 seconds of
same-work preconditioning. Therefore the bounded paired comparator captures
both schedules before timing, preconditions for 3 seconds, then runs 12 rounds
with both execution orders. Identical sources, signs, and uniforms are used
within each round; sources/signs change between rounds. Every round verifies
bit-exact payload, scales, factors, and complete gradient output. Forty graph
replays per sample (12 for the larger case), eight gradients per graph.

| Conditional complete gradient | Narrow/wide median, us | Median paired speedup | 95% round-bootstrap interval |
|---|---:|---:|---:|
| dW, 2048x768x1536 | 126.81 / 115.60 | 1.1004x | 1.0859–1.1122x |
| dX, 2048x768x1536 | 88.70 / 84.02 | 1.0548x | 1.0482–1.0734x |
| dW, 4096x1536x4096 | 865.59 / 857.47 | 1.0220x | 1.0060–1.0257x |

These support a relative benefit under common drift, not individually eligible
absolute latency promises or public compiled-backward performance. The
representative dW JSON initially stored a window-bootstrap interval; the
round-cluster interval above was recomputed from its retained raw round IDs.
Later comparator outputs directly bootstrap complete rounds. Intervals are
within-session diagnostics, not cross-device/run reproducibility bounds.
Both-order medians are retained in JSON (representative dW 1.0991/1.1037x).

The conservative producer CTA cap remains based on warp count; wide mode can
launch idle CTAs on tiny shapes, which still publish zero maxima. Structural
features model this actual cap and the larger shared-memory tile accurately.
No DRAM traffic reduction is claimed without Compute counters. Next: verify
the public compiled backward and compare other devices before selecting a
shape/device-specific schedule; retain the narrow fallback.

In-kernel counter RNG could avoid the FP32 uniform allocation/read/write and
separate RNG launches, but it needs an explicit graph-safe seed/offset contract,
distinct operand substreams, and deterministic replay tests. It is not a safe
drop-in change to the current caller-owned-uniform semantics. At this shape,
the two uniform buffers contain 294,912 FP32 values (1,179,648 bytes total),
while signs occupy 512 bytes; this modeled storage is not measured HBM traffic.

## Artifacts and reproduction

Local JSON, CSV exports, and logs:
`autotune_reports/ms_eden_preparation_20261025_5070/`, including the raw trace
`baseline-timeline.nsys-rep` (also retained in the remote snapshot).
`environment.json` records the diagnostic doctor result. Pytest is absent in
the remote Torch213 environment; bit-exact checks used `torch.testing` directly.

From an isolated source snapshot, using the remote Python above:

```bash
PYTHONPATH=$PWD /home/achiya/vs_code_main/rtx/.venv/bin/python -m benchmarks.study_ms_eden_backward --shape 2048,768,1536 --phase dw --output baseline.json
PYTHONPATH=$PWD /home/achiya/vs_code_main/rtx/.venv/bin/python -m benchmarks.study_ms_eden_backward --shape 2048,768,1536 --phase dw --producer-load coalesced --preparation-phases --output coalesced.json
PYTHONPATH=$PWD /home/achiya/vs_code_main/rtx/.venv/bin/python -m benchmarks.validate_ms_eden_producer --output producer-oracle.json
PYTHONPATH=$PWD /home/achiya/vs_code_main/rtx/.venv/bin/python -m benchmarks.compare_ms_eden_producers --phase dw --output paired-dw.json
```
