# BalancedMoE joint recompute: smaller live shared storage

The joint backward recomputes gate/up and emits four independently scaled
MXFP8 matrices: hidden and derivative, each in row and leading-axis
orientations. It does not retain any forward intermediates. The previous
profile found roughly 10% active warps and 5.5% tensor-pipe activity at
`N=7376,D=384,H=128,E=32,K=4` on the RTX 5070. Its 123 registers/thread and
roughly 68 KB shared storage/CTA made resource liveness a concrete target.
See the [baseline profile](balanced_moe_amortization_scaling_20261118.md).

`MXFP8PreparedMoEConfig.joint_recompute_engine` exposes three train schedules:

| Choice | Dedicated joint scratch | Lifetime change |
| --- | ---: | --- |
| `legacy` (default) | `2*M*N` bytes | Keeps FP32 products live across derivative quantization |
| `operand_reuse` | None | Reuses retired B for the existing exchange; retains the legacy product lifetime |
| `dual_operand` | None | Writes FP32 products to retired A and BF16 derivatives to retired B; then reuses B for decoded hidden |

Here M/N are CTA tile dimensions and N includes both projections. Each
operand array is independently 1024-byte aligned. For a tile with K reduction
values and S stages, B must supply at least `2*M*N` bytes out of `N*K*S`;
the dual schedule also requires that size out of `M*K*S` in A. Decoded
hidden in B needs only `N*(M+1)` bytes. No alias extends across independent
arrays or overlaps live TMA data. Every MMA warp waits for each TMA
transaction and reads the operand before it releases the stage. The alias
engines admit only a single nonpersistent output tile and producer scale
staging, with a CTA-wide rendezvous before reuse and before each producer/
consumer exchange.

Hidden feature-axis quantization still uses FP32 SiLU products; the
leading-axis quantizer consumes the **BF16-rounded decode** of that exact
forward link. Derivatives are rounded to BF16 before each independently
scaled quantizer. Each reduction sees only its own 32-value block. Returned
tensor dtypes, forward storage/recomputation policy, route gradients and long
FP32 reductions are unchanged.

The shared tuner can propose compact 64x64 one-stage and 64x128 one-/two-
stage producer schedules early, then tune recompute geometry, layout and
vector choices alongside the whole expert path. Register caps and role
register budgets are inherited unchanged. No production default or SKU
winner is set. These are alternatives to the legacy joint kernel, not
replacements for separate backward, scale-TMA, or packed active-prefix
training.

## Compiler evidence, not a device timing result

SM120a CuTe 4.7 offline CUBINs at `Mproblem=928,Nproblem=256,Kproblem=384,E=32`
on the local CUDA 13.2 toolchain:

| Tile and staging | Engine | Operand bytes/CTA | Dedicated epilogue bytes/CTA | Registers/thread |
| --- | --- | ---: | ---: | ---: |
| 64x128, 2 stages, default consumer-scale | Legacy | 49,152 | 16,384 | 123 |
| 64x64, 1 stage, producer-scale | Legacy (same-tile control) | 16,384 | 8,192 | 64 |
| 64x64, 1 stage, producer-scale | Operand reuse | 16,384 | 0 | 64 |
| 64x64, 1 stage, producer-scale | Dual operand | 16,384 | 0 | 64 |
| 64x128, 1 stage, producer-scale | Operand reuse | 24,576 | 0 | 127 |
| 64x128, 2 stages, producer-scale | Legacy (same-tile control) | 49,152 | 16,384 | 124 |
| 64x128, 2 stages, producer-scale | Dual operand | 49,152 | 0 | 94 |

Operand/dedicated bytes omit the unchanged scale, barrier and alignment
overhead. Registers are from `cuobjdump --dump-resource-usage`; the same
builds report no stack/local memory. The 64x128 two-stage same-tile control
isolates the live-range change (`124 -> 94` registers/thread), while 64x64
shows that lower shared storage does **not** imply lower registers on every
tile. Small tiles can cost more grid work and operand rereads; single
buffering can lose load/compute overlap. These resource estimates are not
measured occupancy or elapsed-time results.
The [seven CUBINs](../autotune_reports/moe_next_priorities_20261118/recompute_5060/offline/)
are retained as offline compiler artifacts, not 5060 measurements.

## HOME RTX 5090: compact tile wins; wide register change does not

Matched CUDA Graph event medians on physical GPU 0 (170 SMs, Torch
2.13.0+cu132). D=384, E=32, K=4, shared H=256; only the expert recompute
schedule varies. Each phase uses seven calibrated samples in each of
four ABBA visits in one fresh process per comparison. Compilation and
capture are excluded. Frozen router, shared and legacy-expert settings
come from the archive, with newly generated, identical operands/routes
within each pair. All phases below are in microseconds, legacy → candidate:

| Tokens / H | Candidate | Joint recompute | Routed expert step | Whole-layer step |
| --- | --- | ---: | ---: | ---: |
| 128 / 128 | Dual 64×64, one stage | 14.330 → 10.240 | 89.874 → 86.039 | 495.636 → 491.754 |
| 2,000 / 128 | Dual 64×64, one stage | 33.104 → 18.458 | 171.856 → 157.599 | 665.001 → 648.728 |
| 7,376 / 128 | Dual 64×64, one stage | 112.628 → 57.714 | 522.494 → 464.481 | 1,490.958 → 1,435.208 |
| 7,376 / 128 | Dual 64×128, two stages, same-tile control | 114.762 → 117.072 | 529.213 → 523.615 | 1,497.615 → 1,497.802 |
| 29,504 / 512 | Dual 64×128, two stages | 1,939.634 → 2,033.034 | 4,788.851 → 4,880.786 | 7,865.047 → 7,944.438 |

The compact schedule cuts whole-layer latency by **0.8%, 2.4%, and
3.7%** at the three H=128 points. At 7,376 tokens it cuts recompute
latency by 48.8% and expert-step latency by 11.1%.
The wide same-tile comparison holds everything except
epilogue ownership fixed: the standalone stage is 2.0% slower and the
complete step ties. Thus the register savings alone do not explain the
compact schedule's gains; they do not establish better achieved occupancy.
The wide schedule also regresses at H=512 (1.0% whole-layer). This does
not establish how the compact schedule would perform at that width.
The two large H=128 comparisons are independent controls; the different
legacy medians should not be subtracted as a same-policy measurement.
No clock/power settings changed. Across phases, maximum inter-visit drift
was 0.60% and maximum relative sample MAD was 0.29%.

All five comparisons match every emitted recompute byte and all expert and
whole-layer outputs/gradients bitwise. The independent N=128 quantized FP64
oracle produced the same errors for legacy and candidate; worst gradient
NRMS was 0.000605. Every warmed expert/whole replay check passed and
synchronized live-allocation growth was zero. These are not measured
peak-memory or hardware-occupancy reductions.

[Raw JSON, source snapshots, archived configs and JUnit](../autotune_reports/moe_next_priorities_20261118/recompute_home5090/)
include the runtime source digest
`c0a5ef1ab72fed895ff4168c3f9c3cd64879ef2254d5a5f60e2d11f658887054`.
The two older snapshots preserve the benchmark-setup iteration that exposed
Torch 2.13's stale default-stream `AccumulateGrad` capture error. The timed
snapshot (`timing-source-v2.tar.gz`) puts tensor creation, warmup and capture
on the same graph stream; no numerical gate or runtime synchronization
contract was relaxed.

## Validation and reproduction

`tests/test_moe_joint_recompute.py` covers static legality/search, poisoned
output buffers, byte equality to a same-tile legacy control (including partial
row/feature/K tiles), and independent FP64 all-adjoint checks with a
noncontiguous dY. It also exercises fullgraph, CUDA Graph repeatability,
and synchronized live allocations. The focused CPU plan/cache/selection/
compiler suites passed (180 passed, 50 skips). All seven tests passed on
HOME GPU 0 (24.58 s). The earlier laptop transfer was not approved, so there
is no candidate 5060 throughput result. The two resource mechanisms remain
measured alternatives, not an installed winner or new default.

On an authorized, idle SM120/121 system, run qualification before timing:

```bash
PYTHONPATH=. python -m pytest -q -x tests/test_moe_joint_recompute.py
PYTHONPATH=. python benchmarks/benchmark_moe_joint_recompute.py \
  --baseline-result /path/to/frozen/result.json --candidate-index 0 \
  --output /path/to/paired.json
```

Use `--same-tile-control` to isolate storage/ownership and `--oracle` for
a bounded small/narrow shape. The timer retains matched operand/route hashes,
correctness and live allocated bytes, as well as calibrated ordinary CUDA
event samples in ABBA order for standalone recompute, expert training, and
whole-layer training. The three measurements are not additive. Repeat with
`--variant baseline|candidate` in fresh processes and report raw WSL drift.
Use the 5060 archived N=2000 and N=7376 deep-joint configurations as
controls; test large H separately rather than treating a kernel resource
reduction as evidence of a whole-layer win.
