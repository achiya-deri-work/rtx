# BalancedMoE: measured dispatch, traffic and search fixes

The profiling-driven changes improve three different bottlenecks. They do not
establish a universal win over compiled BF16, or training-convergence parity.

## What changed

### Compiler-visible storage, launch-only custom operation

The traceable autograd wrapper constructs output allocations and logical
scratch views outside `rtx::moe_execute`. Its CUDA implementation only dispatches
a prebuilt sequence of existing kernels. A bounded metadata/code cache avoids
rebuilding that sequence on every invocation; it does not own runtime tensors.
Scratch uses typed, liveness-planned arenas. Returned outputs and gradients
have separate storage rather than retaining a whole arena.

The original BF16 inputs/parameters and routing metadata are saved for backward;
expert intermediates are still recomputed. Activation/gating arithmetic,
independent quantization along both backward reduction axes and FP32 routed
accumulation retain their existing numerical policies. There is no implicit
fallback to BF16 GEMMs.

E8M0 scale *bits* cross the private Torch boundary in canonical UINT8 buffers.
CuTe reinterprets their pointer type, with no conversion, copy or extra launch.
This avoids two observed Torch 2.13 integration hazards: cross-dtype aliases in
mutating `Tensor[]` storage, and failed functionalization decomposition with
E8M0-typed scratch. It does not change the scale format or arithmetic.

First-hit selection remains a real-input operation. After selecting a config,
a bounded shape-local guard changes once: the next call compiles its exact
storage plan. Hot calls no longer create/read a CPU configuration token. Other
shapes do not invalidate that cell. Pending backwards keep their original
selection even after explicit refinement or cache clearing. Cold selection
still has a self-contained token until that transition finishes.

No user workspace, stream management, nested `torch.compile` or per-layer
internal graph replay is required. Users compile/capture their outer workload.
Finish tuning and its guard transition before manual CUDA graph capture.

### Quantize original tokens once

`config.prepared.input_preparation` now offers:

- `gather_bf16`: retain the existing gather-then-quantize path.
- `quantize_once`: quantize original X once, then gather packed E4M3 data and
  its E8M0 scales into expert slots. Forward no longer writes gathered BF16 X.

The latter avoids repeated quantization for K selected experts. Backward still
constructs the BF16 view needed for its independently quantized leading-axis
reduction; a logical transpose of packed forward values cannot substitute for
that numerical policy. The mixed quantization portfolio uses a flat work grid,
not empty expert planes for the single global-X operand. Dummy/tail slots are
zeroed, including native-scale layouts and half-aligned BF16 source pointers.

Both paths remain autotuner choices. The existing default is unchanged: extra
packed storage/gather traffic can lose at small shapes or low reuse.

### Compact capacity without discarding good dW schedules

When physical expert capacity is at least twice logical capacity, structural
seeds now include compound compact variants retaining the inherited schedules.
Only token-dependent partial output tiles change from TMA to direct stores.
Aligned dW output tiles keep TMA: their output dimensions are weight dimensions,
not token capacity. Both input-preparation choices are included.

An initial all-direct compact variant was slower. Retaining aligned dW TMA
stores made compact schedules competitive within the same eight-trial budget.
This is an autotunable structural seed, not a hardcoded device/shape winner or
manual register setting.

## Measurements

All speedups below compare against our frozen previous implementation, **not
BF16**, unless explicitly labelled. Compilation, tuning, warmup and capture
are excluded. Public host latency uses fullgraph static-shape compilation with
automatic Inductor graphs disabled; GPU latency uses an outer manual CUDA
graph. Fresh sequential workers run in ABBA order with identical operands,
upstream gradients and numerical/config policies, except the named change.
Raw samples, worker spread, digests, configs, source snapshots and commands are
retained alongside the results.

Narrow means D/E/K/H = 384/24/6/128. Wide means 768/48/6/256.
Times are forward plus all four required gradients, in milliseconds.

| Change and context | Before | After | Speedup |
|---|---:|---:|---:|
| 5090 narrow 2K, compiled public host, same schedule | 0.943 | 0.631 | 1.49x |
| 5090 wide 2K, compiled public host, same schedule | 0.902 | 0.741 | 1.22x |
| 5060 wide 8K, compiled public host, quantize-once | 11.288 | 10.667 | 1.06x |
| 5060 wide 8K, outer GPU graph, quantize-once | 9.900 | 9.309 | 1.06x |
| 5070 wide 128 tokens, frozen versus eight-trial selected raw GPU graph | 1.753 | 1.368 | 1.28x |

The 5090 same-schedule GPU graph was essentially unchanged at narrow 2K
(0.2054 to 0.2055 ms); wide changed 0.4788 to 0.4862 ms. This is a **host
dispatch fix**, not a tensor-core throughput improvement. The narrow whole
layer's compiled host time improved 2.472 to 2.033 ms, but the baseline workers
had 12% spread; whole GPU graph changed 0.733 to 0.747 ms.

On the 5060, same-schedule wide 2K was tied in GPU time (3.091 to 3.099 ms).
Narrow 2K's apparent 8.5% host win had 13.4% baseline worker spread and is not
a confident isolated effect. The final quantize-once wide 8K ABBA GPU result
had baseline/candidate worker ranges of 1.6%/0.76%, below its 6.3% speedup.
Earlier raw wide 8K trials showed 23% forward and 8% combined improvement;
the final canonical-UINT8 public-path result above is the more relevant claim.

The eight-trial 5070 search took 24.4 seconds outside timing. Its selected
public GPU result was 1.329 ms versus compiled BF16 0.648 ms: **BF16 remains
about twice as fast on this narrow-per-expert workload**. This improves search
quality; it does not eliminate the small-expert performance gap.

### Hardware-counter confirmation

For the same 5060 wide 8K forward fixture, Nsight Compute measured:

| Traffic | Gather BF16 | Quantize once |
|---|---:|---:|
| DRAM reads + writes | 560.0 MB | 401.4 MB |
| Preparation + gather DRAM | 283.6 MB | 124.7 MB |
| L2 traffic | 1.320 GB | 1.147 GB |
| Kernel launches | 5 | 5 |

Total DRAM traffic fell **28.3%**. GLU/down/combine traffic was nearly
unchanged: the improvement is where intended, in operand preparation.
Counter collection used separate application-replay processes, three passes,
without forced cache flushing or clock control. Instrumented durations are
not used as ordinary performance results.

### Limitations and negative results

- Same-kernel graph timings can slightly regress despite much lower host
  latency. No claim that compiler-visible arenas improve every GPU workload.
- Private eager phase helpers regressed on the 5090 in the measured snapshot
  (0.294 to 0.376 ms narrow; 0.514 to 0.598 ms wide). Metadata is now memoized,
  but that later change has not been rebenchmarked; the eager regression is
  not declared fixed. Public compiled execution is the measured target.
- Global-X reuse needs extra packed storage. Earlier raw tests showed about
  0.5 MB/6.3 MB additional peak allocation at wide 2K/8K. It is not universally
  preferable.
- The 5090 retained its idle detector service throughout this A/B. No service
  was stopped. Timings are not an exclusive-machine qualification.
- Laptop variance is substantial. Small differences are not dispatch rules.
- We have not solved expert-output traffic, proven fully on-chip arbitrary-H
  fusion, or established end-to-end training convergence here.

## Correctness, compiler and memory qualification

The tests independently check the quantized operational computation, not just
agreement between two versions of our kernel. They cover output, input gradient,
both weight gradients and route-weight gradient; aligned/ragged shapes, native
scale layouts, dummy padding, joint/separate backward preparation and global-X
reuse. Structural input reuse was bit-identical to its matched original path.

Compiler tests check fullgraph forward/backward, first-hit cold/hot transitions,
pending forwards across selection changes, noncontiguous upstream gradients,
private-op schema/fake registration, external CUDA graphs with mutable weights,
user-enabled Inductor CUDAgraph capture and multiple streams. Buffer-plan tests
independently verify producer/read sets, live intervals, alias roots, physical
capacity, canonical dtypes and dedicated output ownership. Repeated steady-state
calls show zero live allocation growth in the measured fixtures; this is not
a long-running training or arbitrary-shape leak guarantee.

Final qualification batches (not an additive unique-test count):

- Local CPU/compiler/documentation: **129 passed, 40 CUDA skips**, plus 316
  documentation subtests. No local GPU claim is made from these skips.
- 5070 broad MoE regression: **89 passed, 2 skipped**; latest-source focused
  compiled aligned/ragged gradients, first-hit and private opcheck: **4 passed**.
- 5060 global-input/native-layout/integrated gradients: **27 passed**;
  retained staged/on-chip eager and compiled gradients: **4 passed**;
  final unified offline/runtime raw wrappers: **2 passed**.

The cold-first-hit test disables automatic Inductor graphs; automatic Inductor
graph capture is separately qualified on a fixed prepared policy. Their exact
combined cold-to-hot sequence with automatic graphs enabled was not exercised.
Final late edits to memoization/cache cleanup/refinement are CPU-checked; timed
and broad-GPU source snapshots remain separately identified in the evidence.

## Evidence and reproduction

The frozen profiling baseline has source SHA256
`2d986d94926d9d097961ff6455b8121b33bc0c09b2342e1d57a0596c5b98dbe0`.
Different follow-up snapshots are explicitly archived; do not assume the final
working-tree source is byte-identical to every timed intermediate snapshot.
The final MoE code digest is
`6c9f6d5ce2138e04eb3270418a29ce9deedcc8d5fc1f593a5bece2bed85aa834`.

Local evidence root: `autotune_reports/balanced_moe_fixes_20261117/`:

- `work_5090_execution/{narrow,wide}/comparison.json`: source ABBA and workers.
- `5060/results/public-abba/*/comparison.json`: final public ABBA results.
- `5060/results/input-counters/`: raw reports/CSVs, fixture/configs and summary.
- `5060/results/input-reuse/`: earlier raw structural A/B, including snapshot.
- `laptop_5070/`: compiled contracts, bounded search and worker results.

These generated binary fixtures/reports are local evidence, not package files.
The baseline diagnosis remains in [the profiling report](balanced_moe_profile_20261117.md).
Public behavior and tuning controls are in [the API guide](balanced_moe.md).

Focused checks, without starting another collection campaign:

```bash
python -m pytest -q tests/test_moe_plan.py tests/test_moe_selection.py \
  tests/test_moe_execution_benchmark.py
python -m pytest -q tests/test_moe_compiler_execution.py \
  tests/test_moe_global_input.py tests/test_moe_legacy_execution.py
```

Run the second command on a supported, idle SM120/SM121 CUDA environment.
The execution benchmark driver provides `compare` and `worker` subcommands;
use `--help` for source snapshots, fixed configs, seed studies and timing phases.
