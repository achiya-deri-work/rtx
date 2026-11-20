# Flat-router autotuning: integration and fresh shapes

Sharded FP32 column reductions and register-resident warp rows now work
through normal first-hit selection, durable cache hits and saved backward
plans. They are no longer limited to the Astra research copy. Exposing more
parallel work was more useful than fitting another ranker.

## Whole-MoE results

Four rotating-order graph visits per arm on two HOME RTX 5090s, nine samples
per visit, batched CUDA events. Input/parameter/upstream hashes, expert/shared
plans and affinity layout are fixed within each shape. Only router configs
change. Compilation is caller-owned, fullgraph/static/default mode.

| T / D / E / K | Old 8 | New 8 | Old 32 | New 32 | New 8→32 |
|---|---:|---:|---:|---:|---:|
| 24576 / 384 / 96 / 4 | 5.905 ms | 5.235 ms | 5.887 ms | 5.063 ms | 3.29% less time |
| 30720 / 512 / 120 / 6 | 9.202 ms | 8.194 ms | 9.132 ms | 8.033 ms | 1.97% less time |

**11.34% / 10.95% less whole-layer time at eight trials; 14.01% / 12.04%
at 32**, versus the old same-budget tuner. These are forward-plus-backward
MoEs, not summed phases or speedups over BF16. Shapes were chosen before the
measurements. They use exactly 4D active hidden budgets, all hiddens aligned
to 64: H320/S256 and H256/S512. They are flat pools of eight flattened
3072/3840-token examples, not packed performance evidence. Both boards are
the same SKU.

Making new families available early buys most of the gain. Deeper search adds
another 2.0–3.3%, not another 12–14%. Trial 32 is not proven globally optimal.
On E120 the best search objective was already found by trial 16.
FP32 reductions can change discrete hard decisions: each arm reproduces its
own IDs and passes its oracle, not cross-arm bitwise equivalence. Exact
capacity, routing budgets and precision remain unchanged.

## Search and dispatch

- Independent forward/backward shard counts (0/4/8/16/32/64/128), CTA/warp
  rows. Explicit zero-shard/CTA configs retain the old programs. Packed pools
  canonicalize these flat options away and keep their existing domain.
- Protected first-eight opening contrasts parallel columns, warp rows and
  hard backends for long underfilled pools. Geometry/SM count proposes about
  two CTA waves; measured selection decides. No named-SKU winner table.
- A coupled `flat_transport` proposal arm competes under shared UCB with
  structural, random and local search. It varies coalescing, shard depth,
  directional sharding, row ownership and hard scheduling. Eligible completed
  eight-trial prefixes anchor neighborhoods, not every noisy latest minimum.
  Explicit opening reservations remain authoritative.
- Four journals contain **32 distinct, timing-eligible attempts each**. Both
  new journals reach the parallel family within eight. Seven/eight later
  attempts use the new arm. E96's final winner came from that arm; E120's
  came from random search. Tests cover 8→16→32 resume/prefix equivalence under
  fixed observations and exclusion of unstable timing anchors.
- Cold selection formerly forwarded only six geometry integers. It now
  honors the entire token, like hot execution. Pending backward retains its
  forward-selected policy after refinement. Old tokens migrate with defaults;
  flat revision 3 and source-safe keys isolate incompatible evidence.

Eight router trials took 20.09/19.66 s; 8→32 another 50.30/53.79 s. This study
allowed 300 s to prevent deadline truncation. Default soft deadlines can admit
fewer attempts; eight fresh compilations are not guaranteed within 15 s.

## Numerical regression caught and fixed

One new E120/32 Torch-metadata plan failed strict near-constant-upstream
full-router FP64 checks: scalar errors 3.68e-5 (`logit_scale`) and 4.57e-5
(`score_bias`), above the unchanged 1e-5 absolute scalar tolerance. Tensor and
replay checks passed. The failed record is retained, not averaged away.

The selected-softmax Jacobian annihilates a common upstream offset. Torch
and flat/packed native adjoints now subtract the first selected upstream
before their FP32 weighted reduction, avoiding nearly equal order-one
subtraction. Forward values, precision and policy do not change. Torch work
stays visible to AOTAutograd/Inductor; native kernels add a warp broadcast,
not eager work inside a launch op. The tuning proxy matches production and
checks exact constant-upstream annihilation outside timing.

The same fixture's scalar errors fell to 1.27e-9 and 1.73e-9. **24/24 final
strict records pass**: four saved plans per shape, normal → changed/near-
constant → original replay. All five router gradient groups and weights use
an independent FP64 affinity/Sinkhorn/softmax oracle. Per-plan replay equals
fresh execution. No tolerance was relaxed.

## Qualification and provenance

135 focused CPU/CUDA routing/metadata/packed/compiler/graph tests passed,
one skipped. The finalized evaluator then passed actual compiled first-hit
tuning: eight distinct residuals, no repeat-call retuning. Final focused CPU
search/cache/sampling checks: 86 passed, 10 subtests, 10 CUDA skips. Integration
covers empty shards, odd E, inactive axes, old tokens, refinement before
backward, durable caches and nondefault-stream capture.

Original whole arms passed independent quantized-FP64 expert checks, replay,
exact capacity and zero live allocation growth. Final paired graphs pass
fresh/replay checks and stable live-memory plateaus. One compiler temporary
release lowered an initial sample by 1 MiB; it was not growth.

Original search source `1904c9f...` is retained. Final runtime/evaluator digest:
`b525f29dcc47c22b08863f05c51095c4522ea63f695e5125f5ec28f76664302e`.
The table requalifies/replays selected fixed plans on the final source;
old journals are **not** relabeled as final-source 32-trial searches.

Evidence: `autotune_reports/autotune_cycle3_20261120/`: source snapshots, all
observations/failures, commands/identities, strict gates, raw paired samples,
and CSV/Parquet in `analysis/`. Reproduce analysis:

```bash
python -m benchmarks.analyze_autotune_regions \
  autotune_reports/autotune_cycle3_20261120 --flat-transport-cycle \
  --output /tmp/rtx-flat-cycle-analysis
```

Phase-specialized hard repair remains research-only; generic phase balancing
is disabled. Next: reduce unused cold compilation and study caller versus
synthetic-calibration ranking before fitting more models. Packed performance,
cross-SKU transfer and universally near-optimal 32-trial quality are unproven.
