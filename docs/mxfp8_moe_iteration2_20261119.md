# MXFP8 MoE iteration 2: consolidation, review, implementation

This cycle starts from the qualified [iteration-1 winner](mxfp8_moe_backward_ownership_20261119.md),
not the older direct-store implementation. [Astra's updated review](mxfp8_moe_iteration2_review_20261119.md)
led to three Sol 6.2 high prototypes. Two add small whole-layer gains; the
split-warp recompute remains a negative result. A fresh half-aligned workload
also exposed and fixed a much larger first-hit search-coverage failure.

**The 1.2x target remains unmet.** These are 5090/GPU0 results, not a new
cross-device or convergence qualification. GPU1 and other workloads were
untouched; no resets, clock changes or new Nsight Compute injection.

## Whole-layer gains against the previous winner

D768, H256, E48, K6, shared H512, independent packed lengths 4097+8223,
physical capacity 1600. Same inputs, upstream, parameters and hard-route
hashes; MXFP8 forward/backward and full recomputation unchanged. Both paths
are fullgraph/static compiled with aggressive dynamic-weight options and
outer warmed CUDA graphs; BF16 uses reentrant checkpointed FFNs.

The complete factorial uses four alternating visits, seven ordinary-event
samples per visit:

| Change from iteration 1 | Whole F+B |
|---|---:|
| Previous combined winner | 5.27336 ms |
| Existing vector-4/XOR recompute + compact forward controls | 5.22465 ms |
| Controls + fragment/packed forward GLU | 5.21046 ms |
| Controls + paired-orientation backward weights | 5.19767 ms |
| Controls + both new implementations | 5.18280 ms |
| Same-router compiled checkpointed BF16 | 5.69351 ms |

The full improvement is **90.56 us**, a **1.75% throughput gain**
(a **1.0175x** speedup). The two new implementations contribute **41.85 us**
beyond the strongest existing-schedule control. Their interaction is -0.68 us;
the gains survive composition. This is measured, not a sum of microbenchmarks.

A final guarded repeat measures **5.27572 -> 5.18968 ms**, with controls at
5.22593 ms and BF16 at 5.70416 ms. BF16/MXFP8 improves from approximately
1.081x to **1.099x**. Another **436 us native-only reduction** would be needed
at that BF16 time to reach 1.2x.

Outputs and every gradient are **bitwise identical** across the new schedules.
Changed-dY replay passes, and each cell has five identical live-allocation
samples after warmup. No extra saved forward activations, eager arithmetic
inside launch operations, caller workspaces or stream-management API.

## What was built and what should be selected

### Forward fragment-owned quantization: optional positive

`forward_glu_engine="fragment_direct"|"fragment_packed"` derives ownership
from CuTe's actual accumulator partition. It retains FP32 SiLU products in
registers and exchanges compact scale maxima; packed mode exchanges FP8 bytes
for coalesced word stores rather than a full FP32 product plane. Row E8M0
scales, infinity rounding and original output bytes are unchanged.

At 64x128, scratch changes from 16,384 B to 1,024 B direct or 5,120 B packed.
The compact forward control measures 194.38 us versus **186.83 us packed**;
the original geometry gains only 1.26 us. The small subset loses or ties,
so this is an autotuner choice, not an unconditional replacement. Offline
codegen confirms packed 32-bit stores, unchanged registers within each
geometry and no stack/local usage. Cold compile cost rises roughly 0.5-0.8 s.
Initial scope is static packed row-major SiLU, qualified M64/M128 layouts,
physical K32 alignment, and direct output. Unsupported configurations fail
legality rather than using unvalidated lane ownership.

### Paired-orientation dynamic weights: optional positive

`weight_preparation="dual_orientation"` reads one BF16 GU/down tile and
emits both independent quantized orientations. GU row output preserves
gate/up interleaving; GU transpose preserves original logical order. It
retains the original singleton token-X grid and adds **no launch or buffer**.
No forward packed state is retained for backward.

Four alternating visits measure the full mixed X+weight portfolio
**98.72 -> 86.64 us**, complete preparation **336.38 -> 317.41 us**, and the
native backward plan **1.60449 -> 1.58435 ms**. Ordinary launches confirm the
same result. All arrays/gradients are bitwise equal after changing weights
and dY; typed arena size and the 12 backward launches remain unchanged.
One modeled BF16 weight pass (56.62 MB) is removed; this is not a measured
DRAM-traffic claim. Tile rows/columns, load values, store bits, warp count,
raster and grid remain tunable. The unfused portfolio remains available.

### Split-warp joint recompute: keep experimental

`joint_recompute_engine="split_warp"` gives derivative and hidden output
independent warp teams, with a dedicated BF16 decoded-hidden plane. It
preserves all four independent quantized orientations and FP32/BF16 rounding
boundaries. `joint_recompute_gradient_warps` controls the division.

It is slower: joint **350.69 us original / 330.97 us existing vec4+XOR /
385.65 us best split**. Compiled expert F+B is **2.1141 / 2.0926 / 2.1580 ms**.
All bytes and gradients match, with no growth. The 4,160-B plane costs
5,120 B after alignment; all variants use 63 registers, with zero stack/local
usage. No claim that spills caused the loss. Static capacity only; no
protected light-tuning slot. The existing vec4+XOR control is the better
candidate on this fixture.

## First-hit gap found by the fresh shape

New D576/H192/E24/K3/shared H320, lengths **7103+9037** (16140 tokens),
physical expert capacity 2080. Half-aligned transposed operands have 576 or
320 rows. The old shared-MLP complete seeds forced 128-row transpose tiles,
so all legal first-eight proposals stayed on-chip. Both eight and sixteen
trials agreed on a poor family: finite-budget agreement alone was misleading.

Provenance correction: the requested expert H160 was rounded to **effective
H192** by this study's 64-wide benchmark alignment. The recorded `case` and
all timings above use H192; shared H320 was unchanged. The subsequent broad
challenge explicitly uses 32-wide alignment to test H160 without that rounding.

The fix derives each transpose tile from the **actually transposed operand
extents**, not row-oriented token M. This shape uses 64-row tiles for all
four backward legs; fully aligned operands retain larger legal tiles. Existing
4/8-way FP32 split/direct controls, arithmetic and quantization policies are
preserved. No validator or numerical gate was weakened.

| Actual progressive search | Old shared F+B | Fixed shared F+B | Fixed whole MX / BF16 |
|---|---:|---:|---:|
| 8 trials per component | 2.35330 ms | .48402 ms | 4.70026 / 4.50873 ms |
| 16 trials per component | 2.35689 ms | .48560 ms | 4.70755 / 4.51302 ms |

Shared execution improves about **4.86x**; whole MX previously took about
6.57 ms. It still trails BF16 on this half-aligned workload (~0.959x ratio),
so this is not a universal low-precision win. Both components now have
**zero eight-versus-sixteen regret**. Best journal scores are 1.068832 ms
expert and .473056 ms shared; standalone deployed timings above are separate
measurements, not substituted for journal statistics.

All 16 expert + 16 shared observations are timing-eligible, with no numerical
or compile failures. Worst independent quantized FP64 expert NRMS is 0.0126%.
Both whole results have zero live allocation growth. Eight-trial cold calls
take 23.6 s expert + 19.9 s shared; deadlines remain soft admission limits.

## Integration, versioning and tests

- MoE **revision 13 / portfolio 23** includes all new sources in its fingerprint.
  Complete positive recipes compose on the incumbent; original controls stay
  reachable. Small-subset and split-team losses are retained, not forced wins.
- Shared/general MLP **revision 48**, structural seeds **16**, prepared opening
  **5** invalidates old first-hit winners that missed legal half-aligned families.
  The fix applies to the ordinary MLP, not a sequence-aware shared kernel.
- Explicit autotuning-off policies validate before allocation. Raw factories
  check physical K, initial/dynamic register roles, static-capacity limits
  and actual new scratch; public legality does not charge the obsolete full
  FP32 plane to fragment kernels. Weight proposals check real padded extents.
- **54 GPU-enabled targeted tests pass** across new and retained kernels:
  independent quantized oracles, exact E8M0 threshold cases, tails, shifted
  pointers, graph replay, changed weights/dY, all gradients and flat live bytes.
  **220 CPU tests pass, one skip**, including half-aligned seed coverage and
  compiler-visible allocation/liveness checks. API reference and whitespace pass.
- `benchmark_moe_composition --baseline-variant` now accepts a qualified cell
  directly and preserves hidden-width initialization provenance and identities;
  it never carries obsolete timing claims into the next baseline.

Final runtime digest:
`ebec8ac46b0ad3b10902194ef56ce698988b2d5e19e3a1b01dd06ce766fec58b`.
The final whole ablation/GPU kernel suite used
`cb440c8c27a4df70506c856765d0e5fe1d1d4905d6963b456b1686b12ff8165e`;
only the later cold MLP seed/version fix differs, not the hot kernels.
The repaired eight/sixteen study and final CPU suite use the final runtime.

## Evidence and reproduction

`autotune_reports/mxfp8_moe_iteration2_20261119/` (existing Storage4 symlink)
retains `integration/`, `review/`, `joint/`, `forward/`, `weights/`, a source
snapshot and SHA-256 manifest. `integration/final-qualification/` is the
pre-fix negative coverage study; **`qualified-shape-prior/` is the repaired
frozen-source study**. Do not merge those contexts or call the negative data
corruption. Failed test-development logs remain separate from final XML.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.qualify_moe_store_prior \
  --output /tmp/rtx-iter2-heldout --cases preparation_heldout \
  --budgets 8 16 --max-autotune

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \
  tests/test_moe_recompute_split.py tests/test_moe_fragment_glu.py \
  tests/test_moe_weight_quant.py
```

Next performance work must address the remaining measured preparation/router/
thin-GEMM costs. Repeating the rejected split team, increasing dW tiles, or
counting common-router savings as native-only savings does not fund 1.2x.
