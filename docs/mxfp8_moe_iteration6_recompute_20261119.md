# Iteration 6 packed joint recompute: keep the incumbent

Neither 64×128 alternative won on the actual bounded backward buffers or the
full frozen expert step. The same-width operand-reuse control also lost on
D384. Leave the eight-trial opening, its exploration slot, and all production
code unchanged. No arm advanced to an unseen search or whole-MoE composition.

## Matched measurements

HOME RTX 5090 rank 0, B16×1024 packed, E128/K6; D384/H192/S320 and
D512/H256/S384. Four alternating visits per cell with seven ordinary CUDA-event
batches per visit; median of visit medians. Each challenger was slower in
every visit in both scopes. All 64 visits were stationary, with worst
relative MAD 0.34%.

| Case | Joint engine, M×N / stages (K128) | Joint kernel (µs) | Frozen expert F+B (ms) | F+B added (µs) |
|---|---|---:|---:|---:|
| D384 | dual, 64×64 / 1 (incumbent) | 308.158 | 1.718809 | — |
| D384 | dual, 64×128 / 2 | 775.916 | 2.188113 | +469.304 |
| D384 | reuse, 64×128 / 1 | 370.611 | 1.793634 | +74.825 |
| D512 | dual, 64×64 / 1 (incumbent) | 460.584 | 2.501169 | — |
| D512 | dual, 64×128 / 2 | 1074.182 | 3.074483 | +573.314 |
| D512 | reuse, 64×128 / 1 | 534.898 | 2.591155 | +89.986 |
| D384, separate control | dual, 64×64 / 1 (incumbent) | 315.211 | 1.734141 | — |
| D384, separate control | reuse, 64×64 / 1 | 326.607 | 1.746743 | +12.602 |

The wide tiles halve the number of N tiles for each active M tile. That can
cut repeated panel loads and tile setup, but the observed net effect is a
longer kernel. The same-width comparison removes the width confound for the
engine change on D384. These measurements do not identify the cost as spills,
occupancy, staging, or bandwidth; no new counters were collected. The joint
kernel is included in expert F+B. Neither duration can be added to full-layer
execution or compared directly to the same-router BF16 totals.

## What was held fixed and checked

- Source SHA256 `759c030b5867e029ebdd40def2bac8d8ebee82154075d06a2a79c31b324a79b9`,
  Torch 2.13.0+cu132. The chosen packed D384 `packed-ac-gc.json:combined`
  and D512 `d512/whole.json:candidate` configs were loaded by the two derived
  specs, not by selecting an untuned default. Prior expert fixtures were
  verified against all input, dY, router/affinity, hard IDs/inverse, packed
  expert weight and selected-plan identities available across the two reports.
  Actual route-weight bytes were verified against the archived frozen expert
  result and reused in every cell.
- The component is the production planner's joint launch after real active
  prefix, both weight orientations, indexed gather, bounded routed dY and
  transpose producers, and the actual dHidden GEMM. The backward arenas
  preserve original-row layout and 32-value scale groups; physical C=1312,
  active C=1024 and consumer M alignment=128 in all cells. The component is
  captured before later aliases can overwrite its sources. Forward still uses
  private compact rows, and expert F+B keeps both persistent dWs and the
  original reduction, producer, quantization, and recompute policies.
- Every captured cell was exactly equal elementwise to its fresh execution and
  the matched native anchor on expert output, dX, every one of 384 optimizer
  leaves, and route adjoints. Dummy route adjoints were exactly zero. The
  private expert contribution and route adjoint are FP32; X, upstream at
  the public output, weights and expert parameter gradients are BF16 (dY is
  converted to FP32 entering the private expert backward as in production).
  Independent operational FP64 oracle worst NRMS was `9.0244e-4` at D384
  and `3.7528e-4` at D512 (fractions); maximum absolute errors were
  `9.7656e-4` and `7.3242e-4`. Gates: `.003` output, `.006` gradients,
  or `2e-7` absolute. Raw results name every output/gradient.
- All eight payload/scale byte outputs in the written extent matched the
  incumbent, were insensitive to two distinct output-poison values, and
  left the inactive suffix poison intact. Five synchronized repeated
  live-allocation samples per cell were constant. Peaks across all retained
  cells and correctness buffers were 3.460 GB (D384 wide), 4.873 GB (D512
  wide), and 2.801 GB (D384 narrow), not per-cell memory costs. Capture used
  benchmark-only cyclic-GC isolation; no production lifetime claim follows.
- CPU focused suite: 13 passed, 7 CUDA tests skipped. The device checks above
  are from the measured runner, not those skipped tests. No promoted
  B8×1216 admission, changed-boundary stress test for the rejected arms,
  wider parameter sweep, or fresh BF16 comparison was run.

## Evidence and reproduction

Raw root `/tmp/rtx-moe-iteration6-20261119/recompute/`: `d384.json`,
`d512.json`, `d384-narrow.json`, runner logs, and the corresponding rejected
configuration sidecars. Each JSON contains its fixture SHA256, source and
script hashes, exact configuration and sample arrays. `d384.log` is the
initial pre-timing Python failure (missing a required constructor keyword);
`d384-r2.log` is the successful run. The D512/narrow runs use the generalized
extent check, which gives the same 1024 rows on both frozen fixtures.

In a fresh copy of the frozen baseline source, install only the
[dedicated runner](../benchmarks/qualify_moe_iteration6_recompute.py) under
`benchmarks/`, then run this command. Use a fresh output name; change `d` to
`512` for the second geometry, and add `--narrow-only` for the same-width D384
control.

```bash
root=/tmp/rtx-moe-iteration6-20261119
d=384
old=/tmp/rtx-moe-iteration5-20261119/experts/d$d
py=/home/achiya/vs_code_main/rtx/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 flock /tmp/rtx-moe-backward-20261119-gpu0.lock \
  timeout 420 "$py" -m benchmarks.qualify_moe_iteration6_recompute \
  --spec "$root/fixtures/packed$d.json" --fixture "$old/fixture.pt" \
  --prior-expert-result "$old/frozen.json" --prior-router-config "$old/result.router.json" \
  --output "$root/recompute/d$d-reproduction.json"
```
