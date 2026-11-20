# BalancedMoE: three structural improvements, qualified on HOME GPU 0

All three priorities from the [amortization study](balanced_moe_amortization_scaling_20261118.md)
are implemented as optional autotuner choices. Two produce useful wins in
their measured regimes; active-prefix execution is correct but has not earned
a full-step promotion. Legacy defaults remain unchanged.

## Matched results

HOME RTX 5090, physical GPU 0 only, Torch 2.13.0+cu132, CuTe 4.7. D=384,
E=32, K=4, H=128, shared H=256. Each comparison freezes all other schedules,
inputs, parameters and upstream gradients. Times are complete compiled layer
forward/backward under warmed CUDA Graph replay, excluding compile/tuning.
They are neither optimizer-step timings nor additive improvements.

| Alternative and workload | Baseline → candidate | Complete-layer result |
| --- | ---: | --- |
| Persistent Sinkhorn, one 128-token pool | 0.495676 → 0.485841 ms | 2.0% lower latency |
| Compact dual-operand recompute, one 7376-token pool | 1.490958 → 1.435208 ms | 3.7% lower latency |
| Active prefix, packed `[128,512,360,1000]`, fixed dual recompute | 0.68757 → 0.68919 ms | Effectively tied; no promotion |

These comparisons use different workloads/configurations. They do not imply
their percentage gains combine, or supersede differently packed historical
winners. All new timing results have exact source/config/raw-sample records.

## Combined-path qualification

The gains are now consolidated in the ordinary `rtx.BalancedMoE` path and
tested together, not inferred from those independent comparisons. A matched
factorial benchmark holds X, dY, all parameters, shared-MLP schedule and hard
assignments fixed. Only packed Sinkhorn and joint recompute change. Each
variant uses fullgraph compilation and outer CUDA Graph replay; four balanced
forward/reverse visits each contain seven calibrated event samples. Compilation,
capture and warmup are excluded. This is layer forward/backward, not an
optimizer step or a comparison against BF16.

HOME physical GPU 0, D=384, E=32, K=4, H=128, shared H=256; microseconds:

| Pool lengths | Baseline | Routing only | Recompute only | Both |
| --- | ---: | ---: | ---: | ---: |
| `[128]` | 496.870 | 486.770 | 493.533 | **483.555** |
| `[255,0,129]` | 637.107 | 652.448 | **632.341** | 648.287 |
| `[7376]` | 1496.197 | Not statically eligible | **1436.099** | Not statically eligible |

The tiny case is **2.68% lower latency** with both improvements. Independent
reductions are 10.100 and 3.337 microseconds; the measured combined reduction
is 13.315 microseconds. The interaction term is just +0.122 microseconds, within
the observed timing variation: the gains are effectively additive here.
The long-pool recompute gain is 4.02%. The ragged control must retain staged
routing: compact recompute helps, but forcing persistent routing regresses.
No universal backend/default switch or per-SKU fixed winner is introduced.
Active-prefix trimming remains optional/off by default; it has not earned
a training-step promotion.

Every variant matches hard expert IDs and inverse metadata exactly. All input,
expert, router and shared-parameter gradients are checked. Worst paired NRMS
is 0.00108% for the tiny case and 0.285% for the ragged case; the long pair is
bitwise equal. All replay checks pass, synchronized live memory is flat, and
maximum per-visit relative sample MAD is 0.41%. The ragged benchmark initially
observed a one-time live-memory **decrease** when compiler cycles were collected;
the final test explicitly collects them after warmup and before each sample.
No numerical tolerance or production allocation behavior changed.

The production selection/cache test writes separate source/device-specific
router and expert runtime winners, clears in-process bindings, and calls the
regular compiled module. Both winners load together without invoking either
tuner. One cold specialization transitions to one fixed hot specialization;
repeated forward/backward matches every gradient. The actual hot graph no
longer contains either cold selection operation. No new frontend, private
per-layer graph, caller scratch or stream-management API was added.

An actual eight-trial-per-context GPU run also selected **both** improvements
without being pinned to them: persistent packed routing and compact
dual-operand recompute, together with quantize-once input preparation and fused
route-gradient reduction. All sixteen router/expert trials were valid. Router
and expert searches took about 77.4 and 15.0 seconds respectively, excluded
from training timing. The selected complete layer measured 483.3/484.5
microseconds over two timing rounds. This run uses the benchmark's original
router initialization, so it is a deployment check, not another paired delta
against the factorial baseline. The independent quantized FP64 expert oracle
passes (worst gradient NRMS 0.000715%), captured gradients replay exactly,
and warmed live memory stays flat. Its caches/residuals remain isolated in
the study output, not installed into the user's default runtime cache.

Reproduce with `python -m benchmarks.benchmark_moe_composition`; raw factorial
results and exact source snapshots are in
`autotune_reports/moe_next_priorities_20261118/composition_home0/`.
Persistent Sinkhorn is currently a packed-router implementation (including
one explicit pool via `cu_seqlens`); omitting offsets retains the original
single-pool routing path. Compact recompute is available in both expert paths.

## What worked, and what did not

**Small-pool routing:** one CTA per sequence replaces repeated saved-state
Sinkhorn launches, without altering iterations, hard assignments or repair.
The tiny router phase improves 3.6%. The 384-token ragged control is 1.7%
slower at the full layer, so the staged fallback matters. Persistent forward,
backward or both, plus CTA width, remain independent tuning choices.
[Routing report](balanced_moe_small_pool_sinkhorn_20261118.md).

**Recompute:** compact 64x64 one-stage dual-operand scheduling cuts joint
recompute 48.8% and the complete routed-expert path 11.1% at 7376 tokens.
Retired TMA arrays are reused only after their consumers finish, retaining
FP32 activation/products and independent quantization directions. A wider
same-tile dual-operand comparison does not improve the complete layer despite
fewer registers; the tested H=512 wide variant regresses about 1%. These are
structural schedule options, not manually chosen register caps or a universal
replacement. [Recompute report](balanced_moe_joint_recompute_20261118.md).

**Packed active prefixes:** compiler-visible buffers retain static shapes,
while GPU metadata skips unused rows/reduction suffixes. Partial reduction
tiles are initialized explicitly; sequence-local scale blocks are preserved.
Legacy and dual recompute compose correctly. Packed expert forward improves
3.7%, but forward/backward is tied/slightly slower. The no-skip one-pool
control loses materially, and the laptop tuner selected the inactive path.
The option stays off by default. [Active-prefix report](balanced_moe_active_prefix_20261118.md).

## Autotuner integration

The integration audit found a real coverage bug: eight actual trials could
remain entirely in legacy implementations even though both new expert
families appeared near the front of the structural pool.

The MoE wrapper now protects a legal compact liveness proposal on each opening
traffic basin (inherited, quantize-once and indexed), plus an active-prefix
proposal in eligible packed contexts. Normalized identities deduplicate
equivalent combinations. Testing each win separately was insufficient: a
synthetic independent-cost test exposed a missed combination even at sixteen
trials. The corrected opening explicitly measures the combinations at eight
and sixteen trials. This applies at ordinary budgets of at least
eight trials, subject to the time deadline. Explicit research-prior prefixes
are not overridden. No proposal is presumed to win. The search checkpoint
portfolio revision is 14; runtime/source identities remain verified.

The component-search arm also rebases the first legal compact recompute
recipe on its measured incumbent, preserving input/dY/route-gradient traffic
choices. It does not spend every local turn cycling liveness recipes: other
tiles remain ordinary structural/coordinate choices. These are measured
compound proposals within the existing shared UCB engine, not manual tuning
or arithmetic addition of measured component costs.

CPU tests execute the real shared orchestrator with stub timing: both legacy
and mixed eight-trial searches evaluate the new families, retain old traffic
alternatives and do not repeat configurations. Progressive 8-to-16 extension
preserves the exact prefix and resumes without repeated trials. Composition
is reachable through either coordinate update order, not only manual configs.

## Validation and artifacts

- Recomputation: seven HOME tests; poisoned/partial tiles, independent FP64
  adjoints, bitwise paired bytes/outputs/gradients, replay and flat live memory.
- Routing: ten laptop and two targeted HOME tests; saved-state FP64 checks,
  exact hard metadata, mutable-boundary compiled replay and all gradients.
- Active prefixes: eleven laptop tests and fifteen HOME plan/kernel tests;
  legacy/dual poisoned tails, independent oracle, both mutable-boundary
  replays, matching full-layer gradients and zero live growth.
- Parent integration: 194 CPU tests passed, 101 CUDA cases skipped in the CPU
  process; post-composition focused tests also pass. Three real-orchestrator
  coverage/resume tests passed.
- Four adjacent HOME GPU regression tests, with twelve subtests, passed:
  MXFP8 prequantized fullgraph execution, direct/TMA stores, independent MLP
  forward/backward references, and ragged compiled NVFP4 training. This checks
  the opt-in changes did not break those shared-GEMM baseline paths; it is
  not a new exhaustive library release qualification.

User inputs, allocations and streams require no new API management. Norms,
activation precision, scales, gradients and recomputation policy are unchanged.
No driver, power, service or unrelated workload was modified; GPU 1 was hidden
from every HOME CUDA process. No production winner was installed by these tests.

Raw source snapshots, controls, commands, samples and test logs are under
`autotune_reports/moe_next_priorities_20261118/`, including
`routing_home_5090`, `recompute_home5090`, `active_tail_home0`, and the laptop
directories linked from each focused report. See the
[manual](balanced_moe.md) for the optional constructor/config controls.

The consolidation adds seven real-orchestrator/compound-search CPU checks,
two factorial bookkeeping checks and one GPU paired-runtime-cache test.
The final focused CPU suite passes **21 tests**, with 25 CUDA cases skipped
in that CPU-only invocation. The GPU cache test passes separately; the
factorial runs and actual bounded search provide fullgraph Inductor, replay,
all-gradient and steady-memory evidence without rerunning a large release
matrix after a search-only change. API-reference generation checks and
`git diff --check` pass.
