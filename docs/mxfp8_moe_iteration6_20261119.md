# MoE iteration 6: reject unsafe speedups, fix capture lifetime

Consolidation → [Astra 6.2 High review](mxfp8_moe_iteration6_review_20261119.md)
→ three Sol 6.2 High owners, using **HOME RTX 5090 rank 0 only**.
This iteration started November 19 and its final checks ran November 20.
Rank 1, remote services and other projects were untouched.

## Performance decisions

The [iteration 5 improvements](mxfp8_moe_iteration5_20261119.md) are the
starting point, not gains to count again. **No new performance search policy
is promoted this iteration.**

| Work | Measured result | Decision |
|---|---|---|
| Packed expert recompute | Both wider 64×128 alternatives lost at D384 and D512. Even the narrow 64×64 operand-reuse control added 12.6 µs to D384 expert F+B. | Keep the existing opening. |
| Flat Sinkhorn forward | On one exact E32 fixture, router F+B fell 2.733→1.751 ms; whole native MoE fell 4.112→3.121 ms. | Do not promote: E128 alternatives and the actual eight-trial E56 winner changed hard assignments against the control, including changed-input replay. |
| CUDA graph retirement | The old path reproducibly destroyed graphs or finalized scratch pools during a different capture. | Qualify a lifetime fix separately from performance policy. |

The E32 forward result is a **fixed-fixture experiment**, not a new default
or a portable MXFP8 speedup. With that identical faster router, compact
checkpointed BF16 also improved, to 3.344 ms; saved BF16 was 3.003 ms.
Changing reduction geometry can perturb near-tie routing while preserving
small FP64 transport error and valid equal-load schedules. That still fails
this experiment's exact hard-assignment equivalence gate. Neither the
numerical gate nor the held-out inputs were relaxed to admit the winner.
The discarded proposal and all eight actual journal admissions are preserved.

The expert comparisons retained original-row quantization scale groups,
bounded/compact preparation and all other winning schedules. All 64 timing
visits were stationary, with independent quantized FP64 output/input/weight/
route-gradient checks, poison tails and flat live allocations. The flat
router's production tuner and focused tests were restored byte-for-byte to
the starting snapshot. See the [expert report](mxfp8_moe_iteration6_recompute_20261119.md)
and [router report](mxfp8_moe_iteration6_router_20261119.md).

## Capture lifetime

The controlled reproduction identified two distinct hazards:

- Native graph destruction during another capture can invalidate that
  capture even without RTX pins. Deferring only RTX's synchronization is
  insufficient.
- The recursive scratch-pool tagging helper retained its own closure and
  allocator pool until cyclic GC. A later collection during capture could
  run a native MemPool destructor in an invalid capture context.

The production change is confined to `rtx/_resources.py` and
`rtx/_workspace.py`, relative to the frozen iteration-6 baseline. Existing
tracked graphs receive temporary strong leases across capture; retirement
and completion checks happen at a safe boundary, not during replay. Failed
completion retains pins. Explicit reset during an active capture is rejected
before calling CUDA. The tagging helper breaks its recursive closure in a
`finally` block. There is no permanent GC disable, replay synchronization,
caller cleanup API, cache enlargement or numerical/kernel-policy change.

Focused qualification passed **33 tests in the CUDA-enabled run** and
**16 CPU tests** (17 GPU skips in the CPU-only invocation). Twenty-four default-GC cyclic capture/drop
iterations stayed at 105,472 live bytes, with zero dropped graphs retained
and zero pending owner groups at each completed boundary. Separate checks
cover asynchronous replay on a changed stream, two live graphs, reset and
exception paths. A real fresh B8×1088/D384/E128/K6/H192/S384 packed search
recorded eight stationary successful trials. Its selected expert plan,
router/shared plans, seed, initialization and hard-route identities were
rechecked on the final source without additional search: both all-field
replays were exact and live memory stayed at 329,286,656 bytes.

The parent repeated the original four-native-cell packed composition,
with three matched BF16 controls, **default GC**, exact archived inputs and
changed-dY replay. All seven cells passed, including every one of the
393 output/input/parameter fields, exact hard routing and five flat live
allocation samples per cell. Two alternating visits × three ordinary timing
batches measured the existing combined winner at **3.758230 ms**, versus
iteration 5's 3.760056 ms; compact checkpointed BF16 was 3.736676 ms and
saved BF16 was 4.325201 ms. This is a regression control, not a new speedup
study. The factorial intentionally starts from the old pre-iteration-5
fixture to repeat that capture sequence; its expert/shared gains are old
wins and are not attributed to lifetime handling.

This demonstrates the controlled mechanisms and successful full-sequence
regression, not proof that a particular destructor caused the historical
intermittent failure. See the [resource report](mxfp8_moe_iteration6_resources_20261119.md).

## Evidence and limits

- [Review and experiment contract](mxfp8_moe_iteration6_review_20261119.md).
- [Starting source and qualified controls](mxfp8_moe_iteration6_context_20261119.md).
- Raw staging: `/tmp/rtx-moe-iteration6-20261119/`, including immutable source
  snapshots, exact fixtures, rejected configurations, journals and failures.
- [Preserved Storage4 evidence](../autotune_reports/mxfp8_moe_iteration6_20261119/):
  baseline/final source archives, raw evidence archive, the original factorial
  inputs/configs and these reports. Source digests and archive integrity were
  independently checked; original data were not deleted.
- Parent CPU integration subset: **44 passed, 13 CUDA skips**; this overlaps
  owner suites and is not an additional unique-test count.

The default-GC parent replay is `integration/packed-default-gc.json` in the
evidence archive. To repeat it with the archived source, use
`python -m benchmarks.benchmark_moe_composition`, baseline
`fixtures/factorial-parent.json`, expert `fixtures/factorial-expert.json`,
and shared `fixtures/factorial-shared.json`, with `--match-inputs
--bf16-checkpoint --bf16-compact --bf16-saved --changed-dy --max-autotune
--visits 2 --samples 3`. Do not add `--isolate-capture-gc` for this regression.

Starting runtime SHA256:
`759c030b5867e029ebdd40def2bac8d8ebee82154075d06a2a79c31b324a79b9`.
Final production SHA256:
`1fe29c84cb28bf0f5e654097424d1b6eb030e07aa93126d2b0645e9880d09ab2`.

These are synthetic correctness, lifetime and fixed-shape graph-replay
measurements, not training convergence or host-throughput qualification.
This pass does not establish broad 1.2× over compact checkpointed BF16.
No commit or push was requested for this iteration.
No study or GPU collection was left running at handoff.
