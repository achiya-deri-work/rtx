# MoE iteration 3: better first-hit coverage and composable padded-row skipping

The cycle is complete: consolidated the stronger BF16 reference, obtained an
[Astra review of the 45-context profiling results](mxfp8_moe_iteration3_review_20261119.md),
and implemented its three priorities with three Sol 6.2 high agents. The changes
are integrated as autotuner choices, not hard-coded shape winners.

## Whole-layer result

HOME RTX 5090 GPU0 only; GPU1 was left untouched. User-side static/fullgraph
Inductor with max-autotune, warmed outer CUDA graphs, actual backward, identical
inputs/parameters/upstreams/hard routes, and four alternating visits of seven
ordinary CUDA-event batch samples. Compilation and selection are excluded.
Both workloads use D384, E32, K4, routed H128, and one shared H256 SwiGLU.
These are fixed-buffer synthetic replay measurements, not training convergence.

| T8192 independent packing | Static MXFP8 incumbent | Active prefix, same fragment/persistent owners | Active + token combine/dX | Checkpoint BF16 | Compacted checkpoint BF16 |
|---|---:|---:|---:|---:|---:|
| 32 sequences of 256 | 1.2946 ms | 1.2173 ms | **1.2034 ms** | 1.6265 ms | **1.1937 ms** |
| New `[63,193,127,641]` repeated eight times | 1.4950 ms | 1.4904 ms | **1.4765 ms** | 1.8227 ms | **1.3898 ms** |

The first composition saves **91.2 us / 7.05% whole-layer time** over the
static incumbent. The held-out pack saves 18.5 us / 1.24%. The native paths
are 1.352x and 1.234x the original BF16 reference, but **0.992x and 0.941x the
stronger compacted reference**. We have not reached 1.2x against that reference.
The first pack's dX-only change saves just 3.6 us without prefix skipping;
adding dX and both token reductions on the active path saves 7.8 and 13.9 us.
These are measured combinations, not sums of independent phase improvements.

The benchmark's `--bf16-compact` reference is now reusable. It differentiably
compacts valid expert rows and retains the original routing-weight gradients;
all compaction, gather and checkpoint recomputation costs are timed. Its safe
capacity bound uses shapes only, not a hot host read of sequence boundaries.
On the first pack it allocates 1152 BF16 rows/expert instead of the native
scheduled bound 2016. This is a benchmark, never a runtime BF16 fallback.

## What changed

1. **Packed router opening:** an eligible staged-sharded family is admitted
   second rather than fifth. At four trials, whole time drops 2.089→1.176 ms
   on the original uneven pack and 2.557→1.431 ms on a held-out skewed pack.
   An independently balanced control stays at .658 ms. Both old and new
   openings converge to the same winners at eight trials. Faster routing helps
   BF16 too; this is not a native-FFN speedup. [Matched router qualification](mxfp8_moe_iteration3_router_20261119.md).
2. **Prefix + fragment + persistent dW:** dynamic-M fragment GLU and both
   unsplit retained-panel dW kernels now honor the same private device prefix.
   Physical strides/arenas stay unchanged. Producers and all consumer groups
   agree on the K-loop bound; zero-prefix dW writes zeros; partial final K
   tiles retain the initialization scrub. Indexed dW, split dW, cooperative
   transport and split-warp recomputation remain excluded from this composition.
3. **Token-owned combine/dX:** adjacent features share routing metadata and
   use alignment-guarded vector loads/stores. Each leg independently exposes
   engine, vector width and lanes/token. No extra launch/workspace, atomics,
   numerical policy change or user allocation/stream responsibility was added.

Frozen-route expert-only packed-32 F+B measured .5334 ms static, .4439 ms
with the older active-only owners, .4346 ms with active+fragment, and .4145 ms
with active+fragment+both persistent dWs. The latter saves another 29.4 us
over the old active implementation. Do not substitute these expert-only
improvements for the whole-layer table.

## Actual eight-trial selection

On the new packing, a fresh final-revision search measured eight unique
candidates, all passing the unchanged correctness gate. The first three
expert-path objectives were **532.08 → 493.25 → 481.75 us**. Admission 2
preserves the incumbent's fragment and both persistent dWs while enabling the
prefix; admission 3 adds token reductions on that *same* recipe. Admission 3
won. Search took 18.53 s; setup and the full frontend invocation are separately
recorded in the result. No manual register or device-name selection was used.

An initial opening offered token reduction only alongside another preparation
policy. Its higher preparation cost hid the movement gain. That failed opening
and its eight-trial journal are retained, not overwritten. The final opening
also preserves a quantize-once challenger and the older store/direct controls.
CPU tests cover actual progressive 8→16 continuation, distinct admissions,
and a second unseen width/expert geometry. Router-specific tests cover 2→4→8→16,
soft deadlines, explicit priors/reservations and unchanged short/single pools.

MoE kernel revision is **14**, structural portfolio revision **25**. The
packed-router semantic revision stays 7 with a separate opening revision 1.
Fingerprints include the new kernel. Final full runtime digest:
`8f1003f6a34e9b06352b47ecd053adebfcaefddbb9c7e6dcc42bb3074eed3b2e`.

## Correctness and negative controls

- Final focused CPU suite: **219 passed, 35 CUDA skips**; final priority subset
  also rerun after the opening adjustment. Generated API reference matches source.
- Prefix CUDA suite: **14 passed**, including poisoned scratch, zero/partial/full
  bounds, both dW reuse axes, fragment geometries and captured changed tensors.
- Integrated movement suite: **30 passed**, including scalar fallback on shifted
  pointers/tails, independent rank-order oracle, opcheck/fake ABI, fullgraph,
  strided upstreams and changed-input replay. No stream mismatch remained.
- Every named whole-layer output/gradient has zero measured difference across the
  MX schedules on both packs. BF16 compaction's worst gradient NRMS is .03253%
  relative to its own uncompacted BF16 reference, not to MXFP8.
- New-pack selected expert result passes the independent quantized FP64 oracle:
  worst NRMS .004263%; fixed-buffer replay has zero observed live-byte growth.
  This is implementation fidelity, not an assertion of BF16 training equivalence.
- Token reduction is **not always faster**. The long/wide paired movement graphs
  save only about 5–6 us, two K6 controls regress slightly, and tiny V8 loads lose.
  Scalar remains available/default; the tuner measures the complete alternatives.

The remaining important work is reducing physical-capacity preparation passes,
especially the additional sequence-local scale-group padding in irregular packs,
and solving the already documented small-token shared-MLP loss. The long scalar
router-gradient and rejected affinity-layout cases in the original profile are
still numerical investigations; this wave did not relax their gates.

## Evidence and reproduction

Durable bundle: [iteration-3 evidence](../autotune_reports/mxfp8_moe_iteration3_20261119/).
It retains isolated source snapshots, original fixture, exact residual journals,
event samples, input/route/parameter hashes, changed-gradient checks and JUnit.
The earlier [45-context study](balanced_moe_challenge_20261119.md) remains intact.

Entry points: `benchmarks/benchmark_moe_composition.py --bf16-compact`,
`benchmarks/benchmark_moe_prefix_composition.py`,
`benchmarks/benchmark_moe_token_reduce.py`, and
`benchmarks/qualify_varlen_router_opening.py`. Public configuration options and
limitations are described in [balanced MoE](balanced_moe.md).
