# BalancedMoE: matched whole-layer recomputation

This qualification uses **checkpointed BF16 FFNs as the relevant comparison**.
BF16 that saves intermediates is included only as a diagnostic. The saved-state
router is identical within each precision comparison and is never checkpointed.

## Measurement contract

- RTX 5090, Torch 2.13.0+cu132, CuTe DSL 4.7.0. The existing detector service
  remains active; these are controlled shared-host measurements, not exclusive
  machine throughput guarantees.
- Main model: 2,000 tokens, width 768, 48 routed experts, six experts per token,
  expert hidden width 256, and one shared SwiGLU with hidden width 512.
- Same initialized model, input, upstream gradient, Parameters and frozen
  expert/shared schedules across precision modes. No optimizer or autotuning is
  inside the measured step.
- BF16 public activations and bulk Parameters/gradients; existing FP32 router
  scale/bias Parameters retain FP32 gradients. Activation/gating and routed
  combination use FP32. MXFP8 FFNs recompute their intermediates.
- Every forward uses default-mode `torch.compile(fullgraph=True, dynamic=False)`.
  Inductor's automatic CUDA graphs are disabled. Ordinary synchronized host-call
  latency and one outer CUDA-graph replay are measured separately.
- Each step calls actual `output.backward(dy)` for the input and **all 151
  Parameters**, rather than benchmarking a subset of `autograd.grad` outputs.
- Production-style `set_to_none` gradient clearing is the default. Cold
  compilation, warmup, correctness checks and graph construction are excluded.
- Three alternating rounds, nine samples per mode/phase/round. Tables give the
  median of the three round medians.

The benchmark-only BF16 wrapper checkpoints only routed and shared FFNs with
`use_reentrant=True`. Sinkhorn's small saved Jacobian history, hard assignments
and metadata are preserved normally. It is not a new production backend.

## Frozen-source baseline

This source includes the original quota counter-reset correctness fix, but not
the later native metadata or cached-quota implementations. The original expert,
shared and native-warp router schedules are frozen.

| Mode | Forward GPU | Forward + backward GPU | Ordinary step host latency |
|---|---:|---:|---:|
| MXFP8, recompute | 0.636128 ms | 1.275776 ms | 2.245415 ms |
| BF16, checkpoint FFNs | 0.684832 ms | 1.542560 ms | 1.886784 ms |
| BF16, save FFNs; diagnostic only | 0.697984 ms | 1.363680 ms | 1.713245 ms |

MXFP8 is **1.209×** as fast as matched checkpointed BF16 in the outer GPU graph
on this shape. Its ordinary host-call step remains slower. Graph replay removes
Python launch submission from the GPU comparison; it does not establish an eager
host-latency win or imply that applications must manage internal workspaces.

A separate earlier run used preallocated `.grad` buffers plus one common compiled
zeroing operation. It measured 1.632352 ms MXFP8 versus 1.886784 ms checkpointed
BF16, or 1.156×. These are a different gradient-accumulation contract and must not
be mixed with the `set_to_none` results. The reduction from that experiment is
not a new GPU kernel speedup. That earlier snapshot also predates the quota
counter-reset fix, so it is not a strictly single-variable clearing experiment.

## Qualified selected candidate

The integrated source exposes native metadata construction and cached/early-exit
quota selection as tuning choices. A bounded 16-trial UCB search on the actual
frozen wide-model affinities selected CuTe metadata at 128 threads and Sinkhorn
forward/backward geometries `(256, 4, 2)`. It did **not** select cached quota or
quota early exit. Expert/shared precision and schedules remain unchanged; the
selected router is applied to BF16 and MXFP8 equally.

| Mode | Forward GPU | Forward + backward GPU | Ordinary step host latency |
|---|---:|---:|---:|
| MXFP8, recompute | 0.572192 ms | 1.209344 ms | 2.195324 ms |
| BF16, checkpoint FFNs | 0.622528 ms | 1.470752 ms | 1.803561 ms |
| BF16, save FFNs; diagnostic only | 0.634080 ms | 1.296416 ms | 1.730055 ms |

The selected candidate is **1.216×** as fast as its matching checkpointed BF16
whole layer in the outer GPU graph. Native whole-step time is **5.21% lower**
than the frozen native baseline (1.275776 to 1.209344 ms). The BF16 router also
benefits. A large metadata-component speedup must not be substituted for this
measured whole-layer improvement. Ordinary host-call MXFP8 remains slower than
checkpointed BF16; no eager-host win is claimed.

Input/upstream/Parameter digests and discrete `expert_ids`/`inverse` match the
baseline exactly. All 153 output/gradient tensors pass the source tolerance:
maximum normalized RMS difference is 0.001864 for MXFP8 and 0.000669 for BF16.
These are **not bitwise-equality claims**: the selected Sinkhorn reduction
geometry differs, and small route-weight changes can cross quantization or BF16
rounding thresholds. The largest MXFP8 difference is an expert gate-weight
gradient, with maximum absolute difference 0.0009765625.

The independent expert oracle passes; independent FP64 router-gradient maximum
normalized RMS is 0.00005838. BF16 checkpoint versus saved-BF16 maximum is
0.000276. Every output/gradient is finite; two repeated captured steps per mode
pass gradient/nonaccumulation checks; all three modes show zero live allocation
growth over eight repeated steps.

Candidate common-warmed-process memory diagnostics:

| Mode | Allocated before repeated steps | Additional temporary peak | Live growth |
|---|---:|---:|---:|
| MXFP8, recompute | 254,781,440 B | 224,005,120 B | 0 B |
| BF16, checkpoint FFNs | 254,781,440 B | 104,479,744 B | 0 B |
| BF16, save FFNs; diagnostic only | 254,781,440 B | 102,980,608 B | 0 B |

These peaks include the warmed process's multiple mode caches and graph pools;
they do not establish isolated inference/training VRAM for an application.

## Small-token counterexample

The same wide model is also tested at **128 tokens**, deliberately retaining
the inherited wide expert/shared schedules rather than silently tuning this
second point. The candidate uses the same selected router as the main case.

| Whole-step GPU path | Frozen source | Selected candidate |
|---|---:|---:|
| MXFP8, recompute | 0.724224 ms | 0.696416 ms |
| BF16, checkpoint FFNs | 0.499328 ms | 0.470784 ms |
| BF16, save FFNs; diagnostic only | 0.471360 ms | 0.443968 ms |

The candidate reduces native time by 3.84%, but **MXFP8 remains 47.9% slower than
checkpointed BF16 at this inherited schedule**. This is a real small-token
limitation, not a universal 1.2× claim or a regression introduced by the new
router. The main wide-model gain does not establish good small-expert occupancy,
capacity-padding efficiency or launch amortization for every token count.

Candidate ordinary host-step medians are 2.533265 ms native, 2.042494 ms
checkpointed BF16 and 2.027302 ms saved BF16. All modes again pass finite,
independent-oracle, two-replay/nonaccumulation and zero-live-growth checks. Exact
input/Parameter identities and discrete route metadata match the frozen source.
Maximum source normalized RMS differences are 0.005942 native, 0.001644 BF16
checkpoint and 0.001586 BF16 saved; independent FP64 router-gradient maximum is
0.00003562. These selected-policy floating-point comparisons are tolerance
checks, not bitwise equivalence.

The small-token common-process baseline allocation is 133,015,552 B; temporary
peaks are 151,830,016 B native and 97,229,824 B for either BF16 mode. Live growth
is zero for every mode. As above, these are not isolated-process VRAM figures.

### Bounded small-token expert search

A further **16-trial existing expert autotuner search** took 27.5 seconds and
selected compact capacity handling (`capacity_multiple=1`) and direct token
outputs, retaining TMA weight-gradient operations. These were structural tuner
choices, not handpicked registers or an alternate-precision fallback. Its expert
objective improved from 338.716 to 289.713 microseconds; that objective is not a
whole-layer measurement.

One final matched whole-layer run, with the same 3×9 sampling and strict oracles,
measures:

| Mode | Forward GPU | Whole-step GPU |
|---|---:|---:|
| MXFP8, tuned compact experts | 0.295136 ms | 0.657344 ms |
| BF16, checkpoint FFNs | 0.257632 ms | 0.469056 ms |
| BF16, save FFNs; diagnostic only | 0.259648 ms | 0.444032 ms |

This reduces native whole time by another **5.61%**, or **9.23%** relative to the
original frozen small-token baseline. Nevertheless native remains **40.1% slower
than checkpointed BF16** on this small-token case. This is the qualified result,
not a promise that a larger search eliminates the remaining gap.

The complete final record is locally preserved. Exact identities/route metadata,
independent oracles, all finite gradients and both captured replays pass.
Source normalized RMS maxima are 0.005942 native, 0.001597 checkpointed BF16 and
0.001633 saved BF16; checkpoint versus saved-BF16 maximum is 0.001565. All modes
again have zero live allocation growth. The compact native temporary peak falls
to 121,481,728 B, compared with 151,830,016 B before expert tuning; the common
baseline and both BF16 temporary peaks are unchanged. These remain warmed-process
diagnostics rather than application VRAM estimates. Ordinary host-step medians
are 2.235901 ms native, 1.754799 ms checkpointed BF16 and 1.623953 ms saved BF16.

## Correctness repair and integrated coverage

Quota source packing reused per-expert counters as cursors but failed to reset
them before target counting. The fix resets them inside the existing source-mark
stage, covering `max(M,E)` items, without another kernel launch. A bounded valid
imbalance reproduction executes only safe count/prefix stages: the old target
prefix starts at 513 and totals 969 for a 513-entry workspace; the fixed prefix
starts at zero and totals 456. **The unsafe old target scatter is never run.**
The original and cached/early-exit implementations pass 19 quota tests, including
severe valid imbalance, exact ties, fallback and `M < E` reset coverage.

Native stable metadata is qualified on actual production-router fixtures, with
exact expert ordering/inverse indices and independent weight/gradient references.
Diagnostic metadata-only forward+backward times are:

| Tokens / experts / K | Compiled Torch | Best measured native |
|---|---:|---:|
| 128 / 48 / 6 | 33.024 us | 5.120 us |
| 2,000 / 48 / 6 | 74.400 us | 6.848 us |
| 8,192 / 48 / 6 | 68.320 us | 12.960 us |
| 2,000 / 24 / 6 | 74.336 us | 6.720 us |

These are component experiments across 128/256/512 threads, not whole-layer
speedups or manually installed SKU defaults. Twelve metadata tests pass, and a
strengthened full-forward/backward graph test also passes after changing both
transport inputs and upstream gradients.

Integrated qualification passes five targeted tests, including two CUDA tests
covering compiled cold-to-hot routing and all router gradients. A separate
composed `cache`/CuTe-metadata test verifies **user-side default-mode
`torch.compile` automatic forward and backward graph replay**, changing inputs
and router weights and checking all gradients. It uses no private per-layer
replay wrapper and is a compiler contract test, not a speed/convergence claim.
The focused local CPU suite passes 74 tests with 42 CUDA-dependent skips; the
new checkpoint harness's six CPU contract tests pass independently.

## Correctness and memory

The baseline passes these checks before timing:

- Independent quantized FP64 expert forward/backward oracle, not just a previous
  implementation as reference.
- Independent FP64 Sinkhorn normalization gradients on the actual frozen hard
  assignment, including all router/input gradients.
- Correct token-to-expert mapping, distinct K experts per token, and normalized
  real-token route weights.
- Finite output, input gradient and every Parameter gradient in all three modes.
- BF16 checkpoint versus saved-BF16 comparison: maximum normalized RMS error
  0.000262 across all 153 checked tensors.
- Two captured full-step replays per mode match their gradient references and
  retain stable leaf-gradient addresses. Fresh gradients overwrite rather than
  silently accumulate across replay.
- Eight repeated steps per mode show zero live allocation growth. Peak-memory
  diagnostics share a warmed process and its caches; they are not an isolated
  per-mode VRAM comparison.

The source-comparison reference additionally stores exact `expert_ids` and
`inverse` metadata, plus input/upstream/Parameter digests. Candidate qualification
requires exact discrete metadata and fixture identity separately from tolerant
floating-point gradient comparisons. The enriched reference's native tensors
match the timed baseline exactly; BF16 repeat differences are below 0.000360
normalized RMS.

## Reproduction artifacts

Driver: [`benchmark_moe_checkpoint.py`](../benchmarks/benchmark_moe_checkpoint.py).
CPU contract tests: [`test_moe_checkpoint_benchmark.py`](../tests/test_moe_checkpoint_benchmark.py).

Ignored local evidence is retained under
`autotune_reports/balanced_moe_finish_20261117/checkpoint/`: exact source snapshot,
frozen configs, raw per-round/per-sample measurements, environment information,
driver identities and the complete gradient reference. Preserve these artifacts
when comparing a different source revision; do not substitute a retuned schedule
and label it a same-policy regression test.

The command accepts `--source`, `--seed-studies`, `--router-config`, optional
`--shared-config`, and `--reference`/`--save-reference`. `--grad-mode
preallocated_zero` is an explicit accumulated-gradient diagnostic; the default
is `set_to_none`. `--nonreentrant` selects the alternative BF16 checkpoint API.
There is deliberately no implicit autotuning in this comparison driver.

Related native metadata/quota oracle evidence is retained under the common
report root's `metadata/` and `quota/` directories. `integration/` preserves the
complete integrated source/results and both 16-trial residual stores, including
the final tuned small-token numerical/memory record. The integrated archived
`rtx/` source is verified identical to the final working source. Interrupted
transfer bytes are not accepted as qualified evidence.

## Packed independent sequences, 2026-11-17/18

The packed extension is qualified separately on RTX 5090/Torch 2.13+cu132/CuTe
4.7. Inputs are `[7376,D]` with independent lengths `[512,2000,768,4096]`,
not one global balancing pool. One shared SwiGLU is included. The original
packed baseline is already tuned; expert/shared schedules, input/upstream and
every Parameter are held fixed. New row/cache/grid choices use shared UCB
tuning, not manually selected registers.

| D / E / K / routed H / shared H | Old packed MXFP8 | Final MXFP8 | BF16 checkpoint | BF16 saved, diagnostic | Native latency reduction | Speedup vs checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| 384 / 24 / 6 / 128 / 256 | 2.171 ms | 1.649 ms | 2.107 ms | 1.907 ms | 24.0% | 1.277x |
| 384 / 32 / 4 / 128 / 256 | 2.341 ms | 1.853 ms | 1.742 ms | 1.605 ms | 20.8% | 0.940x |
| 512 / 64 / 8 / 128 / 384 | 3.608 ms | 2.951 ms | 3.021 ms | 2.672 ms | 18.2% | 1.023x |
| 768 / 48 / 6 / 256 / 512 | 4.722 ms | 4.007 ms | 3.643 ms | 3.182 ms | 15.1% | 0.909x |

These are actual all-Parameter `.backward()` with `set_to_none`, excluding
optimizer, compilation, tuning and warmup. Three rotating rounds, nine
calibrated outer-graph event samples each. BF16 uses the same optimized native
router and checkpoints only FFNs with `use_reentrant=True`. No universal BF16
win is claimed: the 32-expert and widest cases remain slower.

Changes include optional register/warp Sinkhorn rows, exact register-cached
column selection with a complete streaming fallback, early column-grid coverage,
tighter shape-only workspace bounds, and finiteness checking fused into the
existing packing launch. No boundary readback or caller workspace is added.

The final packed matrix passes 47 tests; a separate flat automatic-graph
regression passes. The focused CPU suite passes 97 tests. Every large point
passes independent quantized FP64 expert forward/adjoint checks, all-gradient
fresh-replay checks, and zero live-allocation-growth checks. Maximum operational
normalized RMS is 0.000157; this is error **against specified quantized math**,
not against unquantized BF16. The six-step compiled synthetic learning smoke
reduces loss from 0.2023 to 0.01918; it does not establish model convergence.

The widest final trace still has a roughly 588 us expert quantization portfolio,
343 us joint recompute, several 130–214 us GEMMs and a 182 us single-CTA affinity
backward reduction. These are the next optimization targets. No new DRAM/L2,
spill or tensor-utilization counters were available, and no new SM121/Torch
2.12/export/convergence qualification is implied.

The full report, all source snapshots, configs, residuals, XML/logs and raw CUDA
timelines are preserved in
`autotune_reports/balanced_moe_varlen_optimization_20261117/`. Its `README.md`
owns exact commands and scope; `delivery/rtx` matches the tested working source.
Use `benchmarks/benchmark_moe_varlen.py` for fresh bounded tuning or frozen-recipe
reproduction. Saved BF16 remains a diagnostic, not the optimization target.
