# BalancedMoE whole-layer work on RTX 5090

This follow-up starts from commit `d51b702`, not the older profiling source.
It separates compiler/ownership fixes from GPU speedups, and applies common
router improvements to **both** the MXFP8 and BF16 whole-layer comparisons.
The whole-layer GPU improvement is modest: **about 0.9%** on the measured wide
case. The final focused MLP comparison uses checkpointed BF16 as the relevant
recomputation reference, not a goal of beating BF16 that saves intermediates.

## Scope and contract

- RTX 5090, 170 SMs, Torch 2.13.0+cu132, CuTe DSL 4.7.0.
- Main case: 2,000 tokens, D=768, E=48, K=6, expert H=256, one shared H=512
  SwiGLU. Inherited schedules/actual inputs and parameters are frozen.
- BF16 parameters and public activations, MXFP8 expert/shared GEMMs and packed
  hidden, FP32 activation/gating/routed accumulation. Native MLPs recompute;
  ordinary BF16 autograd saves intermediates. Not a matched-memory baseline.
- FP32 sigmoid affinities and Sinkhorn remain differentiable with saved
  Jacobian state. QB/repair assignments remain nondifferentiable. Exact equal
  scheduled loads, distinct K experts/token and zero-weight dummies are retained.
- Default-mode fullgraph static-shape compilation, with automatic Inductor
  graphs disabled for attribution. Outer CUDA-graph timing and synchronized
  ordinary host latency are separate measurements. Warmup, compilation, tuning
  and graph construction are excluded.
- The work machine's existing detector service stays running. No service,
  driver, power policy or unrelated project is changed. This is shared-host
  qualification, not an exclusive-machine throughput certification.

## What actually dominates

Frozen baseline, two rounds/five samples per round; milliseconds:

| Path | Outer GPU graph | Ordinary compiled host call |
|---|---:|---:|
| Complete MXFP8 MoE | 1.284 | 2.19–2.23 |
| Complete BF16 FFNs, same CuTe router | 1.367 | 1.76–1.85 |
| Router forward | 0.489 | 0.665–0.669 |
| Router forward+backward | 0.591 | 1.024–1.025 |
| Shared MXFP8 MLP forward+backward | 0.154 | 0.904–0.909 |
| Shared BF16 MLP forward+backward | 0.111 | 0.199 |
| Packed routed experts forward+backward | 0.481 | 0.78–0.81 |
| Routed experts with individual Parameter gradients | 0.492 | 1.03–1.04 |

These are separately compiled production components with their real upstream
gradients. They **are not additive phase timings**: compiler boundaries,
autograd work, cache state and captured lifetimes differ. In particular, their
differences are not a clean measurement of pure parameter-gradient plumbing.
They nevertheless identify two important priorities:

1. Router GPU work is substantial; removing a CPU token alone cannot erase it.
2. The inherited shared MXFP8 recipe is about 45 microseconds slower than its
   BF16 counterpart. Shared MLP search coverage matters to the relative win.

Additional frozen-router diagnostic GPU timings:

| Router subpath | ms |
|---|---:|
| Affinities forward | 0.024 |
| Affinities forward+backward | 0.085 |
| Sinkhorn forward / backward | 0.058 / 0.044 |
| QB initialization | 0.091 |
| QB + conditional repair + submission/extraction copies | 0.318 |
| Expert-major metadata forward / forward+backward | 0.070 / 0.074 |

QB is measured as stateless initialization; the full hard path includes repair,
control and copies. Subtracting the two is **not** a standalone repair-kernel
measurement. Metadata includes gathered-weight softmax, stable expert ordering
and the inverse map. Actual assignments and inverse indices are checked against
the whole-model fixture before measuring these subpaths.

## Implementations and bounded studies

### Router execution ownership

Bound Sinkhorn execution allocates outputs/history/scratch outside its opaque
launch-only mutation operation, retains the complete backward history, and uses
immutable forward/backward schedule metadata. Cold real-input selection keeps
its self-contained token; a shape-local guard transitions subsequent calls to
the bound path. Pending backwards do not consult a mutable winner cache.

The hard scheduler's compiler-facing output allocation is external too; the
conditional graph retains explicitly owned private state. The public API does
not gain a workspace/stream management requirement.

The first same-policy candidate comparison showed essentially unchanged GPU
time (whole layer about 1.292 versus 1.284 ms), with no convincing host win in
that sample. This is an ownership/compiler-contract improvement, **not a proven
throughput improvement**.

### Exact warp-owned QB row selection

The optional `native_warp` implementation replaces small-row CTA radix selection
with an exact warp-owned order statistic. Original column selection and routing
semantics remain unchanged. It stays an autotuner choice rather than a forced
default or manually installed SKU winner.

An eight-trial actual-model routing search took 11.0 seconds and selected
`native_warp`, retaining the inherited Sinkhorn geometries and 128 QB threads.
Its tuner objective changed 388.1 to 373.8 microseconds. Independent component
timing then showed QB initialization about 91 to 76 microseconds, hard routing
318 to 302 microseconds, and the complete router about 591 to 580 microseconds.
These are modest gains, not a solution to the remaining repair/metadata cost.

Both precision variants receive the selected router config. Accelerating a
common router cannot be reported as an MXFP8-only advantage. Even a hypothetical
zero-cost router would not by itself establish a universal 1.2x whole-MoE win
for the inherited FFN schedules.

### Shared MLP search coverage

The generic structural proposal order now exposes legal row-gradient
recomputation/quantization fusion early for fully prequantized incumbents.
For N=2,000, hidden/both-orientation fusion is not assumed legal: the token
dimension is only half aligned. Existing independent dW partition candidates
remain available. No kernel or precision policy changes are required for this
search-coverage correction, and no shape/SKU winner is hardcoded.

The 8-trial search took 15.5 seconds and the 16-trial refinement another 19.2
seconds, including four-round qualification outside the steady-state timer.
Both retained the exact incumbent after confirmation. Standalone shared GPU
time stayed about 154–155 microseconds versus BF16's 111 microseconds. A legal
early proposal is not automatically a good proposal: the new search coverage
does **not** eliminate the shared-MLP performance gap on this fixture.

No handpicked configuration is installed in place of the qualified incumbent.

## Final whole-layer ABBA

Four fresh sequential workers, ABBA source order, two rounds/five samples:

| GPU-graph path | Frozen baseline | Selected candidate | Speedup |
|---|---:|---:|---:|
| Complete MXFP8 MoE | 1.287800 ms | 1.275896 ms | 1.0093x |
| Complete BF16 FFNs, matching router within each source | 1.369368 ms | 1.364928 ms | 1.0033x |
| Router forward+backward | 0.592168 ms | 0.579792 ms | 1.0213x |
| Shared MXFP8 MLP forward+backward | 0.154416 ms | 0.154392 ms | tied |
| Packed routed experts forward+backward | 0.481640 ms | 0.483616 ms | 0.9959x |

Candidate MXFP8 versus its **matching** BF16 whole path is 1.070x, compared
with 1.063x for the frozen baseline. This does not meet a universal 1.2x target.
Whole GPU worker ranges were 0.014% baseline and 0.132% candidate. The ordinary
compiled host comparison was 2.358 to 2.434 ms with 12% baseline worker spread;
there is no convincing whole-host latency win.

All 153 whole tensors matched the baseline exactly in this ABBA, all independent
expert-oracle gates passed, and repeated whole steps showed zero live allocation
growth. The existing detector service remained active.

## Focused shared MLP: saved versus checkpointed BF16

Shape `[2000,768] -> H=512 -> [2000,768]`, using the exact same frozen shared
gate/up/down Parameters, input and upstream gradient. All four modes use BF16
Parameters, FP32 SiLU/gating, BF16 public output and BF16 gradient buffers.
BF16 hidden is BF16; MXFP8 uses its existing packed hidden/recomputed backward.

Every forward is default-mode `torch.compile(fullgraph=True, dynamic=False)`
with automatic Inductor graphs disabled for attribution. Backward uses real
`.backward(dy)`, not `autograd.grad`, so reentrant checkpointing is supported.
All modes preallocate the same BF16 `.grad` buffers and include the same compiled
in-place gradient zeroing in the measured step. No optimizer step is included.
There is **no new tuning** in this comparison.

Three alternating rounds/nine samples, warmed outer CUDA-graph medians:

| Mode | Forward | Forward+backward | Newly saved intermediate storage |
|---|---:|---:|---:|
| BF16, save intermediates | 35.456 us | 119.424 us | 6.144 MB |
| BF16 checkpoint, `use_reentrant=True` | 35.520 us | 141.504 us | 0 MB |
| BF16 checkpoint, `use_reentrant=False` | 35.488 us | 141.920 us | 0 MB |
| Current MXFP8 full recomputation | 33.664 us | 163.840 us | 0 MB |

**MXFP8 is about 15.8% slower than reentrant-checkpointed BF16 for the complete
step on this shape**, despite its approximately 5% faster forward. Checkpointed
BF16 costs about 18.5% more than BF16 saving intermediates. The two checkpoint
variants are effectively similar here; no new search or performance target is
inferred from their sub-microsecond difference.

The saved-storage column is measured with autograd hooks, deduplicated by
storage, and **excludes the already-existing input/weights**. All modes retain
5.431 MB of existing input/weight storage. Ordinary BF16 additionally saves
the projected gate/up and hidden tensors. Both checkpoint outputs and every
gradient were bit-identical to saved BF16 in this run. MXFP8 remained finite;
its output/input/three-Parameter gradient NRMS versus unquantized BF16 was
about 6.6%, a different comparison from independent quantized-kernel correctness
or training convergence.

Repeated live allocation growth was zero in all modes. Additional temporary
peaks over the **common warmed process** were 15.065 MB for the BF16 modes and
15.614 MB for MXFP8. These are not isolated-per-process peak-VRAM comparisons:
compiled code, graph pools and runtime caches from all modes were already warm.
Lower retained activation storage does not imply a lower backward temporary peak.
Autograd hooks are collected separately after timing and do not count private
kernel traffic. Under compilation the AOT checkpoint lowering selects actual
rematerialization; no unmeasured exact GEMM/recompute count is asserted.

Generated local evidence (requires the local, ignored evidence bundle):
[checkpoint timing/memory JSON](../autotune_reports/balanced_moe_router_20261117/work_5090/results/shared-checkpoint-00.json),
[whole ABBA JSON](../autotune_reports/balanced_moe_router_20261117/work_5090/results/wide-final-abba/comparison.json).
The focused driver is `benchmarks/benchmark_shared_mlp_checkpoint.py`.

## Correctness and evidence

The source comparison checks **153 tensors**: whole output, input gradient and
every one of the 151 named optimizer-visible Parameter gradients, with identical
inputs, parameters and upstream gradients. It validates shape/dtype/finiteness
and normalized/absolute errors against the frozen source. That parity check is
separate from the independent FP64 quantized expert oracle and independent
router-gradient tests; it is not a training-convergence claim.

Private schema/fake registration, eager/compiled routing, pending selection
metadata, noncontiguous gradients, mutable-input graph replay, multiple streams
and repeated live-memory checks are covered by focused tests. Exact row
selection tests include finite values, ties, signed zero and nonfinite ordering.
An end-to-end default-mode `torch.compile` test with user-enabled
`triton.cudagraphs=True` also passed: N=64, D=128, E=8, K=2, expert/shared H=128.
It verified actual CUDA graph replay in **both** AOT forward and backward,
mutated input/router parameters and compared every Parameter gradient with
eager execution. This is separate from manual outer graph timing.

A subsequent correctness-only batch also qualified cache-mode cold-to-bound
execution for the complete layer, including its shared MLP: **7 checks passed**.
Both fixed and cache-mode tests verified actual forward/backward graph replay.
The MoE-local shared wrapper uses the existing static MLP operation after cold
selection, never a private per-layer replay. This does not alter the general
MLP API or establish balanced-search quality/convergence. This batch exposed
and fixed a Dynamo dictionary-snapshot issue when several cold selectors occur
in one trace; guards now use independent real-source cell attributes. CPU
multi-policy tests verify one cold-to-hot transition without repeated selection.
Exact passing source and the initial failure logs are preserved separately at
`autotune_reports/balanced_moe_router_20261117/final_shared_guard/`; these late
ownership fixes were **not** used to claim another timing improvement.

Evidence root: `autotune_reports/balanced_moe_router_20261117/work_5090/`.
It contains raw worker JSON/logs, saved all-Parameter references, selected
configs and residual search records, environment information and source
archives. `benchmarks/benchmark_moe_execution.py --help` exposes whole-layer
`--decompose`, `--router-details`, source ABBA and bounded router/shared searches.

The final whole-package Python digests are baseline
`0a36b055808a479427af99db04d2524091f88f21f21bfa985f4c381b3cc51eb6` and candidate
`867f9497ec04d3b7c12008ada459c0e865c061629b419a92baca7e62b8a79dab`.
The focused checkpoint comparison uses that same frozen candidate, not later
working-tree helper experiments. The full baseline archive, candidate source
archive, frozen benchmark drivers and both full 153-tensor replay references
are retained locally; transferred reference/source SHA256 checks matched remote.

The environment doctor found CUDA functional but no Nsight Systems, Compute
or Nsight Python in this environment. This follow-up therefore makes no new
hardware-counter claims. Ordinary component timing, not profiler time, is used.

See [the previous traffic diagnosis](balanced_moe_profile_20261117.md) and
[the first measured fixes](balanced_moe_fixes_20261117.md) for their separately
identified source snapshots and hardware scopes.
