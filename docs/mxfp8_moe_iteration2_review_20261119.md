# Iteration 2: what the composed winner still spends

The qualified whole-layer result is **5.27036 ms MX vs 5.69191 ms BF16 (1.080×)**. The target is 4.74326 ms: another **527 µs** native-only improvement. Three new isolated structural experiments are worth doing; their plausible savings do **not** fund a promise of 1.2×. The passing existing-template controls below already establish a modest scheduling opportunity. Distinguish it from a new kernel win.

## Refreshed attribution and controls

HOME GPU0 RTX 5090, 170 SM, Torch 2.13.0+cu132, CuTe 4.7; runtime `f2a63f5465c46d62c24b13e3b0b74b6d0c76e002241bbbd06e6c188b25499f81`. D768/H256/E48/top6/shared H512, lengths 4097+8223, physical C1600. Frozen combined schedules from [the backward ownership study](mxfp8_moe_backward_ownership_20261119.md). Full recomputation; same source inputs, upstream, weights and hard routes. No change to BF16 checkpointing, math or router. No production edits or new NCU injection.

One warmed expert F+B graph trace contains 18 kernels; **diagnostic durations** (µs, in execution order):

| Forward | µs | Backward | µs |
|---|---:|---|---:|
| Mixed quant portfolio | 51.9 | Mixed quant portfolio | 110.7 |
| Gather packed X | 33.8 | Gather packed X + map | 42.5 |
| Gate/up GLU | 220.5 | Routed dY both axes / indexed X transpose | 126.3 / 71.3 |
| Down | 125.1 | Down dX / joint recompute | 87.0 / **340.5** |
| Combine | 81.1 | Route adjoint / final reduction | 202.9 / 1.3 |
| | | Down dW / gate dX / gate dW | 126.8 / 164.7 / 231.2 |
| | | dX movement | 84.0 |

There was also a 32.0 µs copy between forward and backward. Ordinary expert graph visit medians 2.0701/2.1290 ms; five identical live samples, peak 846,710,272 B for this diagnostic process. Do not attribute isolated expert time directly inside the whole model. No new sector, bandwidth, spill or occupancy measurements.

Separate paired **whole** ordinary-event test: four alternating visits × five samples, same combined expert/router/shared baseline, static fullgraph, aggressive dynamic-weight options. Median of visit medians:

| Only change from combined | Whole native F+B | Cut |
|---|---:|---:|
| None | 5.26180 ms | — |
| Joint GLU vector 1→4 | 5.24703 ms | 14.8 µs |
| Joint scratch plain→xor_16b | 5.24158 ms | 20.2 µs |
| Forward gate/up 64×128/S2/consumer scales→64×64/S1/producer scales | 5.24379 ms | 18.0 µs |

Every variant beat baseline in all four visits; all primary, replay and changed-dY error records have zero NRMS; routes exact, no within-cell live growth. These are **separate controls**, not three additive wins. No new BF16 measurement in this four-cell ablation. Forward changed three coupled schedule fields; this does not isolate the cause. Test vec4+XOR and compact forward composition before crediting any new implementation.

## Ranked work for one wave

1. **Joint recompute: independent epilogue warp teams.** Current `DualOperandJointRecompute` already uses retired A for FP32 products and B for rounded dP. All threads run row dP then transposed dP, and only after a CTA barrier can B hold decoded hidden. Give one team both dP orientations and a separate team feature-axis H plus leading-axis H; explicitly reserve a disjoint padded BF16 decoded plane (64×32: 32×65×2 = 4,160 B). An alternative for a third team is separate row/column dP owners. Keep full CTA barriers outside divergent roles and prove operand retirement. No additional global bytes, launches, or retained activations. Use paired FP8 conversions and packed contiguous H stores when vector>1; current vec1 already has adjacent-lane contiguous stores. Do not infer an old scalar-GEMM sector multiplier. Retain per-block E8M0, exact FP32 SiLU division/product, original row H FP8, its BF16-rounded decode before H transpose, BF16 dP before either orientation. Preserve physical pad zeroing, original ABIs and aligned masked fallback. Compare identical vec/swizzle/geometry with and without teams; tune team partition by block counts (not registers); reject if synchronization/occupancy eats the benefit. A **10–25% screening hypothesis** on 340.5 µs means 34–85 µs total, not a demonstrated cut beyond existing controls.

2. **Forward grouped prequant GLU: own scales in MMA fragments, then exchange packed output bytes.** The current prequant class hardwires `warp_quant=False`; at 64×128 register activation it exchanges a full 64×64 FP32 product plane (16 KiB) for row quantization. The other GLU backend has `_prepare_glu`/`_quantize_registers` as a concrete starting point. Derive the actual 32-feature block groups from the grouped prequant accumulator layout; explicitly limit supported geometries/warp layouts until proven. Exchange only FP32 partial maxima (e.g. 1 KiB at 64×128, two N teams), keep FP32 products in registers through scale selection, emit packed four-byte stores via a byte exchange (~4 KiB) or guarded retired-operand alias. Contrast fragment/direct, fragment/packed, and existing N64 compact controls. No change in global output size, number of launches, SiLU math or quantization; weight/gate interleave remains untouched. Fragment-register liveness can negate the scratch cut: inspect actual compiled regs/local use; do not inherit unvalidated lane ownership or assume 64-column equivalence. **10–25% of 220.5 µs = 22–55 µs** conditional, potentially overlapping the measured 18 µs control.

3. **Backward weights: one BF16 source tile for both independent axis quantizers.** In `joint_backward` the mixed portfolio prepares original-token X, GU row, down row, down transpose, GU transpose; the latter four independently read two dynamic parameter arrays. Replace those four logical jobs with a paired-weight CTA per aligned 32×32 block or block group: vector-load a source tile once; emit the existing row and transpose FP8 tensors/scales, with GU row logical gate/up interleaving but transpose **original** order. Preserve unrelated token X job, physical strides/padding, separate scales, no caching between calls or across F+B. Original logical weight source reads 113.25 MB → 56.62 MB (decimal, no claim about DRAM); same 58.39 MB of weight-output pairs. Prefer one multiplexed portfolio launch; measure transparently if splitting token X costs an extra launch. Tune tile group shape and register shuffle vs small padded SMEM exchange; compare against current warp-transpose portfolio, including X. **10–30% of 110.7 µs = 11–33 µs** if successful. Keep separate path for unsupported interleaving/alignment, precision modes and odd shapes.

For all three: require independent quantized-oracle agreement and bitwise outputs when only scheduling changes; check tails including padded holes, changed dY/inputs, all leaves, allocations, and explicit-policy legality without tuning. No donor scopes on protected GPU1. Promote to opening only if paired whole F+B survives against the strongest incumbent controls with no held-out first-hit loss; retain fallback within eight trials and measure real prefix/cache behavior. Keep global buffers compiler-visible, launcher-only operators and original stream API. Do not sum overlapping estimates or treat role assignment as measured occupancy. Even the optimistic 173 µs sum before interaction leaves ~354 µs unfunded. Common transport is still a product opportunity, but an identical reduction C on both paths only reduces this target gap by C/6. Prior rejected indexed-X inside dW, wider dW, or shared upstream staging do not address these mechanisms.

## Raw evidence and reproduction

Review artifacts originally in `/tmp/rtx-moe-iteration2-astra-20261119/`: `winner.json`, all three config sidecars, `winner.expert_step.trace.json`, `paired.json` and `paired.timings.json`. The frozen combined expert config is `combined.expert.json`. For the trace use `benchmarks.benchmark_moe_varlen` with the source D768 shape/lengths, final combined expert config, exact-key router/shared sidecars, `--max-autotune --expert-phases --rounds 2 --samples 5 --phases expert_step --trace-phases expert_step`. For the paired controls use `benchmarks.benchmark_moe_composition --baseline-result <d768-final-paired.json> --baseline-variant combined --max-autotune --match-inputs --changed-dy --visits 4 --samples 5` with the three nested `prepared.recompute` or `prepared.gate_up` overrides listed in the table and an independent output path. GPU commands were serialized with the existing GPU0 lock. No GPU remains reserved by this review.
