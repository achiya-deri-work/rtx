# Experimental MS-EDEN backward: consolidated status and autotuning

## Support boundary

Stable `rtx.NVFP4Linear` continues to use MXFP8 backward. Nothing in this
study changes stable precision policies or installs research winners into the
runtime cache. MS-EDEN is opt-in, requires positive 128-aligned rank-two BF16
operands, and does not support higher-order derivatives.

The target is complete compiled dX+dW, including preparation and fresh RNG,
against matched stable MXFP8 backward with identical NVFP4 forward and original
BF16 sources. FP32 scale/correction semantics and numerical, graph-replay,
memory and timing-stationarity gates remain mandatory.

## What the evidence supports

On the RTX 5060, independent 24-round confirmation of a retained TMA/dual
finalizer design measured:

| M,N,K | FP4 backward | Paired MXFP8 / FP4 | Timing eligible |
|---|---:|---:|---|
| 4096,1536,512 | 243.470 us | 1.10610x | No |
| 4096,2048,512 | 321.939 us | 1.17821x | No |
| 4096,512,1024 | 158.720 us | 1.13877x | Yes |
| 4096,512,512 | 102.032 us | 0.98663x | Yes |
| 4096,2048,2048 | 784.981 us | 1.30908x | Yes |

The wide-shape interval was [1.30742, 1.31064]. All numerical and memory gates
passed, but timing-ineligible rows are not qualified wins. The small shape
regresses. **The 1.20x multi-shape objective and new matched-training
qualification remain incomplete.**

A later 64-trial layout-first study had 59 timing-eligible observations and
selected 238.925 us / 1.11002x on the narrow 1536 projection. This screened
result is not an independently confirmed improvement. Larger row-major tiles
and producer-owned scale loading have not demonstrated a speed advantage.

Detailed local evidence, excluded from Git:
`autotune_reports/ms_eden_speed20_20261026_5060/composable-design/`.
The `confirmation-v4/`, `v7/` and `owner-v8/` subdirectories distinguish
independent confirmation, search and layout qualification.

## Composable structural search

`rtx.autotune.ms_eden_design.MSEdenDesign` combines independent dX/dW
schedules with a transpose-loader and finalizer implementation. It uses the
existing orchestrator, UCB1 scheduler, portfolio, random and coordinate-local
search strategies—not a separate tuning engine.

Searchable choices include load/store widths and layouts, TMA XOR4/XOR5
buffering (1/2/3 stages), operand-specific producer/finalizer geometry, GEMM
tiles and stages, scale movement, locality, persistence and split reduction.
Register caps/allocation axes are excluded from this structural study.
SM-count-derived grid candidates supplement fixed grid sizes.

Native `mma128` scales currently require 128x128x128 tiles. Optional
row-major scales enable larger legal tiles, including 256x128 and 128x256.
Coupled seeds cross layout, epilogue, warp-arrangement and scale-ownership
constraints; merely changing one coordinate can produce an illegal candidate.
M64 remains quarantined and NVFP4 N64 unsupported.

```bash
python -m benchmarks.tune_ms_eden_design \
  --selection path/to/selection.json --shape decoder1536 \
  --output path/to/new-study --trials 64 --seconds 1800

# Optional layout/ownership coverage:
# --allow-row-major --coverage layout
```

Output directories must be new. Each candidate runs in an isolated process;
JSONL residuals and exact execution manifests are retained. A selection is a
research artifact, not an installed runtime winner. Semantic identity removes
inactive controls, and timing-ineligible observations cannot anchor this
study's random/local search. Legacy generic-search defaults are unchanged.

All proposals must pass legality, identical-draw checks, exact implementation
dispatch, current-input/upstream/weight replay, fresh RNG, both gradient
gates, zero live allocation growth, and aggregate plus AB/BA timing gates.
Preparation-only refinement cannot mutate GEMM or split-K settings.

Confirm an exact selection separately:

```bash
python -m benchmarks.tune_ms_eden_design \
  --selection path/to/study/selection.json --shape decoder1536 \
  --output path/to/new-confirmation --confirm-only --rounds 24
```

Timing options can strengthen but not weaken the study's minima:
`--warmup-ms 5000`, `--replays 16`, `--min-batch-ms 2`.
Validate transfer shapes and matched training before claiming a general win.

## Exact profiling and training handoff

`benchmarks.profile_backward_subparts --mode public --design-selection FILE`
uses the exact schedule and scoped implementation dispatcher. It records source
hashes and actual TMA dispatch. Nsight durations are diagnostic, not performance
acceptance evidence. Keep profiling separate from ordinary paired timing.

`benchmarks.train_precision20m --ms-eden-designs FILE` accepts a JSON mapping
canonical `M,N,K` keys to full selection `candidate` objects. Exact model-shape
coverage is required. Schedules can differ by shape, but loader/finalizer pairs
must currently agree because research compilation patches are process-scoped.
Mixed implementations and conflicting legacy overrides are rejected.

The full design map and dispatch source hashes enter checkpoint identity.
Warmup verifies TMA dispatch and records it outside measured training blocks.
Scoped patches restore on failure. See [the training recipe](precision20m_recipe.md)
for precision boundaries and matched-step/wall-time interpretation.

## Correctness qualification

The assigned-5060 scale-layout matrix passed 29 tests: five CPU contracts and
24 GPU combinations spanning two shapes, both gradients, three tile shapes and
both scale-loading owners. Checks include opcheck, fullgraph compilation,
independent references and changed-input/randomness graph replay. This proves
integration correctness, not performance superiority.

A consolidation CPU pass completed 1,006 tests with 830 GPU-only cases skipped.
The exact proposed commit was also tested from a clean exported tree:
432 passed, 443 GPU-only skips and four subtests passed. This guards against
accidental dependence on standalone experiments left outside the commit.
Final RTX 5060 qualification on Torch 2.13.0+cu132 / CUDA 13.2 passed:

- 160 focused GPU/contract tests: eager/compiled backward, current-input and
  RNG replay, paired pipelines, row-major ownership, factor pairs and split-K.
- 49/49 quick production-matrix cases across stable MXFP8/NVFP4 frontends,
  including training, inference and packed operands; zero failures.

The first focused run exposed a stale forward reference: the research call
requested exact regional scaling but the stable reference retained power2 in
its dynamic configuration. The reference now explicitly selects exact mode;
bitwise equality and all numerical tolerances are unchanged. The production
runner's module-import failure was also corrected and both script/module CLI
entry points are regression-tested.

These bounded qualification results are not full training convergence or
proof of the outstanding multi-shape speedup target. Logs and machine-readable
matrix output are retained in the ignored local consolidation report directory.

The research adapter additionally exposes `tma_tile_rows=32|64|128` for TMA
loaders, independently of one/two/three-stage buffering. The default remains
32; larger transfers retain the same 32-row arithmetic groups. Non-default
geometry with a native loader is rejected. Design context revision 5 includes
this field in candidate identity; older design manifests load with 32 rows.
On the RTX 5060, all 18 shape/tile/stage combinations passed byte-exact packed
value and scale checks, global-maximum checks, and changed-input CUDA-graph
replay (19 tests including constructor validation). These are correctness
results only: larger tiles have not been promoted as performance winners.

The subsequent 24-trial RTX 5060 complete-backward sweep on
`4096x1536x512` covered both XOR load layouts, all three tile sizes and all
three buffer depths, then random/local proposals. Larger transfers regressed:
128-row two/three-stage candidates took roughly 370–383 us. The selected
eligible candidate retained 32 rows, XOR5 and one stage. Independent 24-round
confirmation measured MXFP8 272.456 us versus MS-EDEN 246.860 us, paired
speedup 1.103867 (95% round-cluster interval 1.100441–1.107148), with both
order-stationarity gates passing. This does not establish a new performance
win over the prior retained composition or meet the 1.20x goal. No defaults
were changed. Full search and confirmation evidence is retained under
`autotune_reports/ms_eden_speed20_20261026_5060/composable-design/tma-v10/`.

A follow-up 40-trial row-dY geometry search varied 64/128/256/512 threads,
26/104/256 CTAs and 4/8/16 values per lane, preserving transposed producers
and GEMMs. It retained the unchanged anchor (240.685 us, paired 1.103327x);
there was no new winner. Evidence is in the adjacent `row-v11/` directory.
An isolated native row-producer Nsight Compute capture (`row-counters-v12/`)
found 57 registers/thread, 51.07% achieved occupancy, zero local/shared spills,
8,611,310 executed warp instructions and 31.30% scheduler cycles without an
eligible warp. The detailed capture reported 1,277,952 fused and 2,248,704
non-fused FP32 instructions. Profiler duration is diagnostic, not full-backward
latency. This favors investigating arithmetic instruction work rather than
manual register caps; changing FP32 association remains prohibited by the
same-draw contract. The profiler's suggested FMA speedup is not a measured win.

The opt-in `row_transform="fma"` design (context revision 6) reuses the
cross-lane ±1 FMA butterfly without changing register-local association,
normalization or EDEN correction. Four GPU cases passed byte-exact packed,
scale and partial-maximum checks, including changed-input graph replay, zero,
outliers and exponent-range inputs. Complete-backward confirmation first
failed timing stability; that attempt is retained. A fresh 24-round run with
15-second warmup passed unchanged gates: MXFP8 269.942 us, candidate 241.963 us,
paired 1.115712x (95% interval 1.113527–1.117724). This remains below 1.20x;
cross-session comparison does not establish the gain over native preparation.
Evidence: adjacent `row-fma-v13/`. Stable defaults remain native.

### Final consolidation (2026-10-27)

The performance pursuit is paused at the user's request. No automatic
promotion or additional tuning/training run is scheduled. The final five-shape
matrix used the retained per-shape schedules, 24 paired rounds against stable
MXFP8 and 15-second warmups. Ratios below are MXFP8/MS-EDEN within each run,
not a direct paired native-versus-FMA comparison:

| M,N,K | Native ratio | Row-FMA ratio |
|---|---:|---:|
| 4096,1536,512 | ineligible | 1.1143 |
| 4096,2048,512 | 1.1648 | 1.1677 |
| 4096,512,512 | 0.9863 | ineligible |
| 4096,512,1024 | ineligible | ineligible |
| 4096,2048,2048 | 1.3071 | 1.3085 |

All ten trials passed the numerical/runtime checks; four failed timing
stationarity and remain excluded. The matrix does not establish a meaningful
row-FMA improvement, nor 1.20x across the representative shapes. Narrow-shape
regression remains explicit. Row-FMA stays an opt-in research axis, not a
runtime default. No new model-training speedup is claimed. The complete
matrix, including failed timing attempts, is retained in
`row-fma-v13/matrix-consolidation.tar.gz` under the report root above.

Consolidated search coverage supports `--coverage tma` and
`--coverage row_producer`; `--row-transform native|fma` explicitly selects the
recorded butterfly implementation. Candidate identities, actual dispatch
checks and training/profile source fingerprints include this distinction.
