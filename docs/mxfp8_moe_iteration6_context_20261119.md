# Iteration 6: consolidated starting point

Iteration 5 is complete and preserved. Current production source matches its
final SHA256:
`759c030b5867e029ebdd40def2bac8d8ebee82154075d06a2a79c31b324a79b9`.
No old performance gain should be counted again. Hardware authorization is
HOME RTX 5090 **rank 0 only**; rank 1 and all remote workloads remain untouched.

## Starting controls for the next work

| Scope | Improved native F+B baseline | Same-router compact checkpoint BF16 | Existing paired result to reuse |
|---|---:|---:|---|
| Packed B16×1024 D384/E128/K6/H192/S320 | 3.760056 ms | 3.738822 ms | iteration5/integration/packed-ac-gc.json, `combined` |
| Packed B16×1024 D512/E128/K6/H256/S384 | 4.652784 ms | 4.886413 ms | iteration5/experts/d512/whole.json, `candidate` |
| Flat T16384 D384/E32/K4/H320/S192 | 4.111259 ms | 4.325157 ms | iteration5/integration/flat-bc.json, `combined` |

Paths above are relative to `/tmp/rtx-moe-iteration5-20261119/`. The complete
[iteration 5 report](mxfp8_moe_iteration5_20261119.md) defines scope, sampling,
numerical gates and caveats. Its archive is
`autotune_reports/mxfp8_moe_iteration5_20261119/` (baseline/final source,
242-MiB evidence archive, exact fixtures/journals/failures, CPU/GPU cache checks).
The artifacts were integrity checked. New baseline code is frozen separately
at `/tmp/rtx-moe-iteration6-20261119/baseline/` before new implementation edits.

Composition fixture loading supports these qualified cells through
`--baseline-result <report> --baseline-variant combined` (or `candidate`).
It restores true packed/flat layout and the original initialization metadata.
`--match-inputs` must verify X, dY, all parameters and hard IDs/inverse before
claiming a matched A/B. Do not silently restart from the pre-iteration-5
expert/router/shared policies.

## Solved enough to avoid repeating

- Cold padded experts now compare strong static, prefix, bounded original,
  compact and bounded+compact on the same fragment/dual-weight/persistent-dW
  anchor within eight actual trials. Do not retest the earlier mismatched
  legacy/direct controls as the main opportunity.
- Flat routing now gives backward thread/column geometry a fair early trial.
  A 128/1/4-only search is no longer the current baseline.
- Ordinary shared MLP search now probes independent gate/down dW partitions
  on the same stored prepared plan. Its extra whole benefit was small; do
  not forecast a large MoE gain from the remaining few standalone microseconds.
- Real held-out eight-trial searches passed for all three owners; compiled
  MoE cache binding and strict long-MLP deployment/runtime winner reuse passed.
- No kernel math, numerical ABI, leading-32 scale group or BF16 public dtype
  changed. Five production files changed only search/identity behavior.

## Evidence available, and evidence absent

There are 102 completed, audited original/fanout context executions at the
earlier runtime; their consolidated JSON/CSV/Parquet is in
`/tmp/rtx-moe-iteration5-20261119/consolidated/`. These remain useful for
regime selection, but do not describe new residual per-kernel costs after
iteration 5. Its targets have fresh component and whole-layer timings, not
new Systems/Compute captures. Older profiles are historical context only.
Do not sum overlapping router/expert/shared measurements or infer saturation
from modeled CTA occupancy. The original/fanout RNG recipes differ; preserve
their per-record identities when making numerical comparisons.

The packed multi-cell factorial once failed during the third CUDA graph
capture. A C-only comparison passed, and the full factorial passed with
benchmark-only cyclic-GC isolation across capture. No sticky CUDA fault was
seen. Delayed finalization is plausible but not proven. No production resource
lifetime fix was made. A next owner may isolate this if it is the highest-value
stability task, with deterministic proof rather than a global GC workaround.

## Review and work contract

Requested sequence again: consolidation → Astra 6.2 High review → three Sol
6.2 High owners. Reviewer should pick three independent implementation scopes,
not launch another broad profiling campaign. A narrow fresh attribution check
is allowed later if a concrete structural hypothesis needs it.

Keep the user's primary B8/16/32, each sample >=1024, D384/512, K4/6/8,
expert counts multiple8, hidden widths multiple64, and hard ±7.5% active-4D
budget. Tiny forward stability remains a control, not the main backward target.
All new choices must be autotuner-visible templates/heuristics, no manual
register winner or per-SKU measured-config table. Any backward compaction
must preserve original leading-scale membership and dW numerics; silently
regrouping K is not permitted. Avoid eager hot-op arithmetic/allocations;
scratch stays compiler visible. Preserve independent oracles, changed-data
replay, every optimizer-visible gradient and bounded live memory.

All GPU commands use `CUDA_VISIBLE_DEVICES=0` and
`flock /tmp/rtx-moe-backward-20261119-gpu0.lock`, with sandbox escalation for
CUDA access. Use fresh process/source snapshots, bounded GPU batches, and
ordinary timings with actual measured compositions against the stronger
same-router BF16 controls. Root owns shared benchmark/integration code and
archival. Preserve the pre-existing dirty worktree; no commit/push is requested.
