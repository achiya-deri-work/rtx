# MoE iteration 4: bounded producers, compact forward, and shared first-hit coverage

Implemented the three priorities from the [Astra review](mxfp8_moe_iteration4_review_20261119.md)
with three Sol 6.2 high agents. Qualified on HOME GPU0 and WORK's RTX 5090;
HOME GPU1 and unrelated projects/services were untouched. WORK used one
persistent SSH transport. No new broad profiling campaign was run.

## Whole-layer results

Milliseconds, complete forward plus backward, user-side fullgraph/static
Inductor max-autotune with warmed outer CUDA graph replay. Four alternating
visits × seven ordinary CUDA-event batches; compilation/tuning excluded.
Every A/B uses identical parameters, X, dY and hard routes. All cases have
D384/E32/K4, routed H128 and one shared H256 SwiGLU.

| Device / input | Previous MX schedule | Bounded preparation | Compact forward | Both | Original checkpoint BF16 | Compacted checkpoint BF16 |
|---|---:|---:|---:|---:|---:|---:|
| HOME: 32 × 256 tokens | 1.201364 | 1.163462 | 1.157370 | **1.143247** | 1.622905 | 1.189021 |
| HOME: `[63,193,127,641]` × 8 | 1.474810 | 1.460135 | 1.436059 | **1.423204** | 1.816631 | **1.387615** |
| WORK: 128 tokens; only shared forward changes | .469353 | — | — | **.417754** | **.340116** | .340110 |

The new producer/layout choices are additive in measured whole paths, not
just isolated estimates. Packed-32 now beats the stronger BF16 reference by
**1.040x**; the irregular pack is still **2.56% slower**, and tiny MoE remains
**22.83% slower**. No universal 1.2x claim is established. These are synthetic
fixed-buffer replay results, not convergence or Python host-latency claims.

## Implementations

1. **Bound original-layout preparation.** Independent `bounded_gather`,
   `bounded_dy`, and `bounded_xt` choices retain physical descriptors and skip
   suffix work, including both forward/backward packed X gathers. They define
   each consumer's complete final M/K tile. When an unbounded reader remains,
   the index map still initializes its entire physical extent. Static producer
   code remains available. Modeled payload traffic falls, not arena size.
2. **Compact only forward M.** A private native stable row map gathers packed
   X into smaller storage, runs the unchanged fragment GLU/down, and combines
   through translated inverse indices in original rank order. Backward still
   uses the old routes, original BF16 recomputation and 32-value scale groups.
   No transient map/hidden is saved into autograd. Pool-count proofs are derived
   by the module, stripped from opaque functional specifications, and not trusted
   merely because a user copied a configuration field.
3. **Give shared MLP forward real coverage.** Short-reduction search admits
   on-chip, gate-only prepared and gate+down prepared forward on the same
   backward policy, plus an unsplit direct-dW triplet. Prepared forward has
   compiler-visible output/scratch and one launch-only mutation boundary with
   CuTe scale-byte casts. No hidden eager Torch arithmetic or cached dynamic
   weights was introduced. Both the ordinary MLP and MoE's cold/bound shared
   dispatch use this path.

No register caps or SKU-specific winner tables were hand tuned. New producer,
layout, movement and existing tile/pipeline alternatives remain selectable;
defaults retain the original layout/full producers. See [MoE controls](balanced_moe.md)
and [MLP controls](mxfp8_mlp.md).

## Component evidence, not whole-path predictions

- HOME four-producer graph: packed-32 **102.66→63.43 us**, irregular
  **102.82→82.33 us**. Their modeled byte+scale writes fall by 62.85/30.41 MB.
  Static arenas remain unchanged. Frozen v25 packed-32 full control is 102.50 us.
- HOME compact-only expert F+B: packed-32 **398.61→378.21 us**, irregular
  **483.48→445.23 us**; metadata construction is counted. Forward arena falls
  by 35.43 MB, but measured F+B peak is unchanged. The packed-32 gain is not
  evidence of internal-hole removal: its active GEMM extent was already 1024.
- WORK shared MLP at T128, archived winning backward fixed: forward
  **67.60→16.39 us**, F+B **112.61→61.42 us**. Gate-only is 84.00 us F+B.
  The isolated local **nonreentrant** BF16 control is 24.57 us; do not substitute
  it for the whole table's reentrant checkpoint reference. T512 already deployed
  prepared forward and is a control, not a newly improved winner. Ragged
  T173/K576/H320/O576 also passed, but its default backward remains expensive.

## Actual first-hit selection and correctness

On an unseen `[97,159,223,545]` × 8 pack, fresh eight-trial tuning measured:

| Admission | Complete expert-path choice | Objective |
|---|---|---:|
| 1 | Static incumbent | 528.99 us |
| 2 | Prefix, retained fragment and both persistent dWs | 491.18 us |
| 3 | Same + token combine/dX | 478.33 us |
| 4 | Same + bounded producers | 458.14 us |
| 5 | Compact forward, unchanged original producers | 440.26 us |
| 6 | **Bounded + compact** | **425.09 us** |
| 7 | Quantize-once control | 638.11 us |
| 8 | Ordinary component exploration | 427.72 us |

All eight are unique and pass; trial 6 wins. Search is 19.89 s, full frontend
invocation 26.20 s, reported separately from runtime. Expert correctness uses
an independent quantized FP64 oracle: worst NRMS **.003980%**, zero observed
live-allocation growth. WORK also measured the actual shared forward triplet
by admissions 2–4; CPU tests verify progressive continuation and reservations.

Final focused qualification: **173 CPU checks** plus **25 MLP CPU checks** pass;
**24 integrated checks on the CUDA host** pass, including device cases and
shape-only helpers. Agent-specific producer, packed-layout and WORK ABI suites
provide additional retained evidence. Coverage includes poisoned scratch,
zero/partial bounds, malformed-map overflow guards, empty/one-token pools,
changed boundaries/weights/X/dY at fixed addresses, noncontiguous dY, every
optimizer-visible gradient, fake/mutation contracts and partial-gradient masks.
All 105 named whole-layer outputs/gradients have zero measured discrepancy
across MX variants (the expert-only harness has 99 fields).
Backward scale membership and numerical gates were not relaxed.

Two test-harness problems were fixed, not hidden: duplicate import names had
re-registered a CPU mock operator; Torch 2.13's schema mutation comparator used
unsupported E4M3 `allclose` arithmetic. The latter test uses a stricter raw-byte
comparator only during schema checking; production and alias checks are unchanged.

MoE revision **15**, structural portfolio **27**, MLP tuning revision **49**.
Final qualified runtime digest:
`a5e83c8eb4400c824105bc15a3a710f7c76e1cf13eb3d62a7d84ca80ffd2baf8`.
Frozen intermediate source hashes, ordinary samples, failed harness logs,
fixtures, exact journal configurations and source archives are retained in
[the Storage4-backed evidence bundle](../autotune_reports/mxfp8_moe_iteration4_20261119/).

Known remaining deficits: small-token routed overhead and shared backward, plus the
irregular packing gap. Any denser *backward K* regrouping is a separate numerical
policy, not a free follow-on to compact forward. Existing long scalar router
gradient/affinity limitations are unchanged. No claim of training convergence
or full production qualification follows from these kernel checks.

Subsequent workload-priority clarification: the main training target is
B=8/16/32 independent samples with each L>=1024; B>=8/L>=512 is secondary.
Preserve tiny forward and general-shape correctness/stability, but do not treat
tiny backward parity as the next primary optimization target. See the
[current workload policy](balanced_moe.md#workload-priorities).
