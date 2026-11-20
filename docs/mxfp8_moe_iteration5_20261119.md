# MoE iteration 5: make the small-budget search try the useful combinations

Completed the requested sequence: consolidate the finished studies →
[Astra 6.2 High review](mxfp8_moe_iteration5_review_20261119.md) → three Sol
6.2 High implementation owners → matched component and whole-layer checks.
Hardware scope is HOME RTX 5090 **rank 0 only**. Rank 1, WORK services and
other projects were untouched. No further collection is running.

## Whole-layer outcome

Complete forward plus backward, BF16 inputs/parameters/output, MXFP8 FFNs,
unchanged FP32-sensitive math. User-side static/fullgraph Inductor with
supported aggressive options, no freezing, warmed outer graph replay.
Four alternating visits × seven ordinary CUDA-event batches; setup/search
excluded. Every native cell uses identical archived X, dY, parameters and
hard routing. These are synthetic performance/correctness results, not
convergence or Python-host throughput claims.

All rows use B16×1024 tokens. S is one shared hidden width; H is routed H.

| Layout / D,E,K,H,S | Previous MX | New MX | Less step time | Matched compact checkpoint BF16 | Matched native-layout saved BF16 |
|---|---:|---:|---:|---:|---:|
| Packed / 384,128,6,192,320; A+C | 4.337 ms | **3.760 ms** | **13.3%** | **3.739 ms** | 4.327 ms |
| Packed / 512,128,6,256,384; A | 5.529 ms | **4.653 ms** | **15.8%** | 4.886 ms | 5.532 ms |
| Flat / 384,32,4,320,192; B+C | 5.224 ms | **4.111 ms** | **21.3%** | 4.325 ms, new router | **3.982 ms**, new router |

We closed much of the many-expert gap, but **did not establish broad 1.2×
over the stronger BF16 reference**. D384 packed remains ~0.6% slower than
compact checkpoint BF16; D512 packed and flat are ~1.05× faster. Saved BF16
is a distinct no-recompute control and is not always slower than native.

The flat router improvement is shared with BF16: its compact reference fell
5.424→4.325 ms when using that same new router. This is not an MX-only
tensor-core gain. Packed and flat routing have different balancing semantics;
we did not interchange them.

### Do the changes actually compose?

- Packed D384: baseline **4.337186**, A **3.763427**, C **4.329771**, A+C
  **3.760056 ms**. All four combined-vs-baseline visit deltas were positive
  (575.5–578.5 µs).
- Flat: baseline **5.223934**, B **4.113855**, C **5.207736**, B+C
  **4.111259 ms**. Combined-vs-baseline saved 1112.5–1118.9 µs on each visit.
- C alone helps, but its extra few microseconds on top of A/B are marginal
  relative to between-visit variability. Do not multiply isolated speedups
  or claim a material extra whole-layer win from that small residual.

## What changed

**A — expert cold opening.** The [expert tuner](../rtx/autotune/moe.py) now
compares six legal controls on the *same* strong indexed/fused preparation:
legacy/persistent, fragment+dual-weight/persistent, prefix, bounds on original
rows, compact forward, and bounded+compact. Cold padded packs with proven
pool count get this coverage; inherited plans, tight/flat controls, compatible
priors, exploration and progressive prefixes remain available. The old
eight-trial comparison changed several unrelated execution choices at once.
Paired expert F+B was **2.380→1.732 ms** at D384 and **3.434→2.502 ms** at
D512. Private compact-map construction cost ~6.15 µs and is included in F+B.

**B — flat backward geometry.** The [flat router tuner](../rtx/autotune/routing.py)
now spends one early trial comparing an existing backward geometry with its
unchanged forward/hard/metadata control. Candidate generation uses scheduled
rows per lane, column CTA count and SM count. No new kernel or workspace was
needed. Standalone router steps fell **3.350→2.721 ms** (E32) and
**4.714→3.881 ms** (E128); IDs/inverse and route weights were exact.

**C — independent shared dW partitions.** The [MLP tuner](../rtx/autotune/mlp.py)
now probes independent gate/down partitions on the same prepared/store plan,
subject to grid fill, K panels and bounded partial workspace. A measured
down-only alternative reduced S192 **206.97→198.00 µs**, S320
**280.58→274.33 µs**, and held-out S256 **182.83→177.11 µs**. Gate-only
alternatives sometimes lost; they are comparisons, not forced winners. A
literal 32K upper cutoff was removed in favor of geometry/resource gates;
~40K coverage this iteration is metadata-only, not a measured speed claim.

Only five production files differ from the qualified baseline:
`rtx/autotune/{moe,routing,mlp}.py`, `rtx/_mlp_autotune.py`, and
`rtx/deployment.py`. The last two share the affected ordinary-MLP winner-key
logic so runtime selection and deployment coverage agree. Existing tuning
cleanup is preserved. Kernel/config numerical ABIs did not change; kernel
revisions remain MoE 15, flat router 2 and MLP autotune 49. Affected search
identities/source hashes/checkpoints are versioned. No manual register winner,
BF16 fallback, new frontend burden or eager hot-launch arithmetic was added.

## First-hit and correctness qualification

Each owner ran a real bounded search on an unseen component workload:

| Owner | Held-out workload | Evidence |
|---|---|---|
| A | B8×1152, D384/E128/K6/H192/S320 packed | 8/8 eligible, expert objective 1.375→1.019 ms |
| B | T12288, D512/E48/K6/H256/S512 flat | 8/8 eligible, exact 4→8 journal prefix; matched full router 2.914→2.454 ms |
| C | B8×1536, D384/E32/K4/H320/S256 | 8/8 eligible; selected down-only partition; paired shared 182.83→177.11 µs |

All use the common bounded orchestration, legality and stationarity gates;
larger finite continuations are not global-optimum claims. CPU admission
tests also check progressive prefix and explicit-prior behavior.

The consolidated original 102 contexts retain their independent quantized
FP64 expert oracle (worst NRMS 0.01295%). New comparisons retain the same
numerical recipes and 32-value scale membership. Expert schedule changes are
bitwise-equal to their frozen reference. The parent factorial checks every
optimizer-visible gradient: **393 output/input/parameter fields packed**
and **105 flat**. Maximum native cross-schedule NRMS in the combined parent
runs was **0.00227% packed / 0.00196% flat**. Changed dY at fixed addresses,
fresh-vs-replayed gradients, finiteness and zero synchronized live-allocation
growth passed. Those percentages measure implementation/schedule differences,
not the model-level quantization error versus unquantized BF16.

Integrated CPU subset: **114 passed, 4 CUDA tests skipped**. Relevant actual
GPU checks ran separately, including poisoned producers/maps, changed packing,
tiny/wide controls, current-source compiled MoE cache binding, and strict
long-shape ordinary MLP cache loading. The latter loaded the measured 12K
winner once, reused it across fullgraph fwd/bwd calls without tuning, returned
bitwise-identical results, and the deployment reader reported the same key.
Owner-specific test counts are documented separately; do not sum overlapping
test suites into an invented unique total.

No new tensor workspace implementation was introduced. C increases logical
partials on one leg, but power-of-two scratch classes and serial lane reuse
mean logical-byte increases are not physical VRAM deltas. Whole benchmark
footprints include retained comparator graphs and are not per-policy VRAM.

## Benchmark fixes and remaining capture caveat

The parent consolidated cold fixture reconstruction across the ordinary and
composition harnesses, including stable expert/input RNG and true flat layout.
Three archived legacy/fanout/flat GPU fixtures reproduced exact X/dY and every
parameter byte. Actual factorials now support expert/shared/router cells and
same-router compact/saved BF16 controls. Benchmark-only specialization limits
allow the deliberately many independent models without disabling fullgraph.

One initial packed four-native-cell capture failed with
`cudaErrorStreamCaptureInvalidated` at the shared cell after baseline/A passed.
The error was non-sticky; a fresh two-cell C comparison passed. The same full
factorial then passed with **benchmark-only cyclic-GC isolation across each
capture** (`--isolate-capture-gc`), with collection after warmup and restoration
after capture. All steady-state timings are graph replay with normal GC state.
Default-capture logs, CUDA diagnostics and cache/memory snapshots are retained.
Delayed cleanup/finalization is plausible but **not proven as the root cause**.
We did not silently enlarge runtime caches, suppress numerical errors, disable
fullgraph, or claim a production lifetime fix. This remains an explicit
multi-cell capture diagnostic; flat qualification required no GC isolation.

## Evidence and current state

- [102-context consolidation](mxfp8_moe_iteration5_profile_20261119.md)
- [Astra review](mxfp8_moe_iteration5_review_20261119.md)
- Owner reports: [experts](mxfp8_moe_iteration5_experts_20261119.md),
  [router](mxfp8_moe_iteration5_router_20261119.md),
  [shared](mxfp8_moe_iteration5_shared_20261119.md)
- Raw staging: `/tmp/rtx-moe-iteration5-20261119/`; parent factorial JSONs are
  `integration/packed-ac-gc.json` and `integration/flat-bc.json`.
- [Preserved HOME/Storage4 evidence](../autotune_reports/mxfp8_moe_iteration5_20261119/)
  includes baseline/final source, raw fixtures, journals, failures, ordinary
  samples, cache tests and audited CSV/Parquet, not only this summary.

Qualified starting runtime was `a5e83c8eb4400c824105bc15a3a710f7c76e1cf13eb3d62a7d84ca80ffd2baf8`;
final production-source digest is
`759c030b5867e029ebdd40def2bac8d8ebee82154075d06a2a79c31b324a79b9`.
Existing unrelated dirty work was preserved. No commit/push was requested in
this iteration. Remaining work is the stronger-BF16 gap, more general first-hit
qualification and the capture-lifetime diagnostic—not another broad repeat of
the same 102 contexts or a claimed universal 1.2×.
