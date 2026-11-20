# Iteration 5B: give the flat router a real backward-geometry trial

The existing Sinkhorn backward kernels reduce the standalone flat-router
step by **18.8% (E32)** and **17.7% (E128)** on matched, long global pools.
The change is in the flat search opening only. The forward, hard QB/quota
repair, stable metadata, solver budgets, affinity and packed routing are
unchanged. No sharded column kernel or new workspace was needed.

## Matched results

HOME GPU0 RTX 5090 (170 SM), Torch 2.13.0+cu132, CuTe 4.7.0; both arms
use the same frozen runtime
`a5e83c8eb4400c824105bc15a3a710f7c76e1cf13eb3d62a7d84ca80ffd2baf8`.
For each D384/K4/T16384 flat arm, backward threads and column groups are
the only changed config fields. X, full-layer dY, all archived parameters,
hard IDs and inverse hashes match the fanout fixtures. Route weights are
bitwise equal. The standalone router uses a common, seeded route-weight
upstream for both arms. Affinity layout, hard plan, forward Sinkhorn and metadata
are fixed to the archived selected plan. All timings are ordinary
CUDA-event timings over warmed graph replay: four alternating-order
visits × seven calibrated batches of at least 10 ms. All 96 phase/arm
visits in these two comparisons passed the existing stationarity gate.

Median of visit medians; row groups are always 1. Column time is one
actual final-iteration column-backward kernel, not the whole transport.

| Experts | Backward threads / column groups | One column kernel (µs) | Sinkhorn backward (ms) | Full router F+B (ms) |
|---:|---:|---:|---:|---:|
| 32 | 128 / 4, original | 168.20 | .8754 | 3.3496 |
| 32 | 128 / 1 | 55.28 | .3073 | 2.7892 |
| 32 | **256 / 1** | **40.95** | **.2481** | **2.7205** |
| 32 | 256 / 2 | 51.18 | .3014 | 2.7751 |
| 128 | 128 / 4, original | 169.97 | 1.3312 | 4.7136 |
| 128 | 128 / 1 | 77.89 | .4698 | 3.9361 |
| 128 | 256 / 1 | 75.81 | .4480 | 3.9191 |
| 128 | **256 / 2** | **53.23** | **.4368** | **3.8811** |

The winning full-router savings were positive on every visit: 628.7–629.3 µs
for E32 and 823.6–833.5 µs for E128. The old launch has 8/32
column CTAs at E32/E128 and 512 scheduled rows per lane. The wins launch
32/64 CTAs and shorten those scans to 64/128 rows per lane, respectively.
Changing backward threads also changes the row launch block size; row
groups do not change. These are static launch/work counts, not measured
occupancy or DRAM bytes. The single-kernel, complete Sinkhorn backward
and full-router measurements directly establish the causal direction
without a new broad profiler wave. Their scopes overlap; don't add them
or attribute all standalone router cost to the column adjoint.

## Opening and numerical qualification

[`make_flat_effort_autotuner`](../rtx/autotune/routing.py) adds one real
same-plan contrast after the existing first hard alternative. On the
common native-column cold seed the first four attempts are incumbent,
Torch QB alternative, that alternative with only backward geometry
changed, and the retained native QB alternative. Previously inherited
Torch QB plans use that incumbent as the geometry control; their hard
alternatives remain in the opening. The candidate uses scheduled
rows per reduction lane, experts per column group and device SM count.
It is limited to a training adjoint with at least 256 rows per old
lane and a grid using at most half an SM wave. It selects a narrower
group targeting `min(E, ceil(SM/4))` column CTAs, preferring adjacent
columns when that target permits and never creating a second wave.
Wider blocks shorten the scan while retaining the original row groups.
These are candidate rules, not winner mandates. The existing deadline,
correctness/stationarity admission, fallback and exploration rules
remain in force.

Explicit priors/predictors and custom reservations keep their admission
order, as do legacy/forward-only/short or already-covered openings.
The existing flat source hash refreshes flat winner/residual keys,
and the progressive checkpoint records the amended ordered prefix.
The packed source digest was unchanged in both snapshots
(`ff84b82703fc63e11e3cd2c1baad3e9a68a2d7cb2d342bd133c6e25788580f68`).
The operator revision and config token are unchanged, so prior schema,
saved backward and packed keys do not require migration.

The held-out single pool (T12288/D512/E48/K6/H256/S512) admitted an exact
4→8 progressive continuation in two fresh processes. The first four raw
journal lines were byte-for-byte unchanged. All eight distinct admissions
were correctness- and timing-eligible; no fallback occurred. Cold search
invocation took 15.64 s and the resumed four admissions
took 16.45 s, including setup. The third trial won at both caps.

| Trial | Configuration ID | Admission | Objective (ms) |
|---:|---|---|---:|
| 1 | `044b0167967033012957d9d0` | Cold native-column QB | 3.240 |
| 2 | `4823230bc5eece9bba6c9ac9` | Torch QB control | 2.764 |
| 3 | `6af25cebf5e33033508ef410` | Same Torch QB, backward 256/1/1 | 2.318 |
| 4 | `374c70dbc161be07d91c744a` | Retained native QB | 3.195 |
| 5 | `296d903bb09a814b03851555` | Retained structural | 3.182 |
| 6 | `c8009e6a17a3112ad92cf7fc` | Retained structural | 3.172 |
| 7 | `2981d103f5504c54c589bb9f` | Coordinate-local | 2.789 |
| 8 | `599f89ededda0acfbb6e15b1` | Random exploration | 3.158 |

The tuner objective averages caller and synthetic fixtures and starts
from FP32 detached log-affinities; it is not full-router latency.
With the winning geometry frozen on the same Torch QB control, the
held-out shape's paired full router fell from **2.914 to 2.454 ms**
(15.8%), and its full Sinkhorn backward fell from .6632 to .2152 ms.
All 24 phase/arm visits were stationary. The full-router savings were
457.8–470.3 µs across the four paired visits.

Independent FP64 log-Sinkhorn and score-adjoint checks use the unchanged
`(rtol, atol)` gates `(5e-5, 5e-6)` and `(5e-4, 8e-6)` on caller and
synthetic scores. Across both fixtures and all geometries, worst
absolute transport and score-adjoint error were 1.87e-6 and 9.23e-7
(worst score-adjoint NRMS 9.41e-8). Compiled complete-router checks
include input, projection, expert-token, logit-scale and score-bias
gradients, not only the score adjoint. With fixed graph addresses, replay
changes X and the route-weight upstream, includes a nearly constant
upstream to exercise small scalar adjoints, then restores both.
Worst projection/expert-token/input NRMS across the two shapes was
6.72e-5, and the maximum absolute difference was .0078125 in the
BF16 gradient tensors.
E128's modified-upstream reference bias/scale gradients were about
`-4.35e-5/-1.02e-4`, with worst absolute differences sub-1e-9.
Each arm had zero live allocated byte growth across six synchronized
post-warmup iterations. The retained individual timing graphs add to
the process-wide peak, so those peak figures are not arm-to-arm memory
savings.

The unchanged small packed control used two separate pools of 127 and
257 tokens at D384/E32/K4. From separate baseline and candidate-source
processes, the median full router was .33805 vs .33806 ms, within visit
variation. Inputs, parameters, packed config, expert IDs, inverse and
weights had identical hashes. Both processes passed fixed-address
changed-input/upstream replay and had zero live growth.

This is a shared router improvement: use the identical selected router
in the paired compact checkpoint BF16 and saved BF16 remeasurements.
Do not count avoided router time as an MXFP8-only FFN gain.

## Raw evidence and rerun

All artifacts are under
`/tmp/rtx-moe-iteration5-20261119/router/`:
`e32-geometry.json`, `e128-geometry.json`, and the ready-to-compose
`e32-candidate.router.json` / `e128-candidate.router.json`.
The [geometry/holdout runner](../benchmarks/qualify_flat_router_backward.py)
was copied with the shared cold fixture helper into the immutable
snapshots alongside the original runtime. From `geometry-src`:

```bash
out=/tmp/rtx-moe-iteration5-20261119/router
raw=/tmp/rtx-moe-k4-20261119/study
router_python=/home/achiya/vs_code_main/rtx/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 flock /tmp/rtx-moe-backward-20261119-gpu0.lock \
  "$router_python" -m benchmarks.qualify_flat_router_backward \
  --fixture "$raw/d384_f1of6_k4_e32_b16_l1024_flat/result.json" \
  --geometry 128,1,1 --geometry 256,1,1 --geometry 256,1,2 \
  --output "$out/e32-geometry-repro.json"
```

Replace `e32` by `e128` for the second fixture. The corresponding
[focused admission tests](../tests/test_flat_router_opening.py) exercise
actual unique 2→4→8 prefixes, legacy/forward/short controls, rejection,
SM-dependent geometry, explicit priors/reservations, no promotion of
unstable measurements, and a two-attempt deadline. With the existing
packed opening and routing-autotune tests, the CPU suite passed 35
tests (2 CUDA tests skipped).

For the held-out 4→8 journal, run from `opening-src` with the same
`out`/`raw` variables above; the two commands use one new journal directory:

```bash
for budget in 4 8; do
  CUDA_VISIBLE_DEVICES=0 flock /tmp/rtx-moe-backward-20261119-gpu0.lock \
    "$router_python" -m benchmarks.qualify_flat_router_backward \
    --fixture "$raw/d512_f1of4_k6_e48_b16_l1024_flat/result.json" \
    --fixture-mode heldout --tune-trials "$budget" \
    --tune-initial "$out/cold-initial.router.json" \
    --tuning-dir "$out/heldout-tuning-repro" --output "$out/heldout-${budget}-repro.json"
done
```

The raw recorded prefix and continuation are `heldout-four-journal/`
and `heldout-tuning/`, with summaries in `heldout-four.json` and
`heldout-eight.json`. The frozen geometry comparison is in
`heldout-pair.json`; rerun with `--fixture-mode heldout --geometry 256,1,1`
instead of tuning options. Only flat search code differs from the baseline
in this snapshot (runtime digest
`5907932b0bab61c277881f312f490867dbd8b4c01d872ad7330bb7b7c807e6f2`).
The small packed control is in `small-packed-base.json` and
`small-packed-candidate.json`; the two source directories are
`control-base-src` and `control-candidate-src`, using
`d384_f1of6_k4_e32_b16_l1024_packed/result.json` with
`--fixture-mode small-packed` and no geometry overrides.
