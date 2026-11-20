# Shared MLP: compare gate and down dW partitions independently

The best same-store down-dW partition change saved **9.0 µs (4.34%)** at
`T16384/D384/S192` and **6.3 µs (2.23%)** at `T16384/D384/S320`, standalone
shared forward plus backward on HOME GPU0. Increasing the gate partitions
helped the former but hurt the latter. These are shared-component timings,
not savings that can be added to routed MLP or router timings.

## What changed and why

All 48 archived fanout shared searches tried prepared 8/8 gate/down splits
with a TMA down store. None admitted unequal splits or a count above eight.
At S192, the old 128×128 output tiles give nine gate-dW and six down-dW tiles;
eight splits supply 72 and 48 producer CTAs, respectively, on 170 SMs.
Doubling a split count doubles that producer grid on the unchanged tile and
adds one incumbent's worth of logical FP32 partial writes and subsequent
reads. This is a grid/workspace model, not measured occupancy or traffic.

The new bounded cold opening keeps the same legal prepared 4/4 and 8/8 stored
controls and both direct variants. It uses two previously redundant slots on
one-leg challengers, more underfilled first. It inherits their exact gate,
forward, dX, quantizer, split partition type, dW GEMM tile and FP32 reducer.
The extra workspace cap is **8 MiB per candidate**, with at least four K
panels per proposed split on average plus the existing per-partition legality
check. Both incumbent grids must be under one device
wave. There is no hardcoded 16-split winner; legality and eligible timings
still decide. There is no upper token cutoff if the grids remain underfilled;
the inherited-prepared and long-token fused incumbents keep their existing
opening, as do wide and small-forward cases. Explicit priors/portfolios retain
their ordering, progressive increases keep their exact prefix, and the
eighth admission remains ordinary exploration.

The affected training contexts only receive an opening tag and corresponding
runtime winner variant; deployment cache inspection computes the same key.
There is no kernel, serialized schedule, or universal MLP identity bump.
Affected measurements batch the actual execution mode and use the existing
stationarity gate; an unstable candidate is not eligible for selection.

## Frozen, same-operand results

RTX 5090, 170 SMs, Torch 2.13.0+cu132, CuTe 4.7.0. Fullgraph static
outer CUDA graph, four alternating visits with seven ordinary CUDA-event
batches after warmup. Source archive X/dY and **every model parameter** hash
matched. Each native step fills X, gate, up and down gradients through real
optimizer-visible leaves. The baseline is the archived best stored plan, not
the much slower initial on-chip trial.

| Shared shape / exact archive | Baseline 8/8 | Down only | Gate only | Added logical split scratch, down/gate |
|---|---:|---:|---:|---:|
| D384/S192/E32/K4/H320, packed B16×1024 | 206.973 µs | 197.998 µs | 200.754 µs | 2.25 / 4.50 MiB |
| D384/S320/E128/K6/H192, packed B16×1024 | 280.585 µs | 274.327 µs | 288.148 µs | 3.75 / 7.50 MiB |
| D384/S256/E32/K4/H320, unseen B8×1536 | 182.828 µs | 177.106 µs | 188.594 µs | 3.00 / 6.00 MiB |

All visits stationary. Down gains over the corresponding baseline visit
were 8.69–9.35 µs, 4.74–6.43 µs, and 5.57–6.48 µs respectively. Unchanged phases gave
bitwise-equal output, dX and other-leg dW. The affected dW differed in 16
and 27 BF16 elements for the archived down wins, and 10 on the holdout. This change does not
assume bitwise equality across FP32 partition orders. Full independent FP64
quantized GEMM/STE oracle worst NRMS, including five fixed-address changed
input/upstream replays, was 0.0001777 (0.01777%); existing gates were
0.003 output / 0.006 gradient or max abs below `2e-7`. No live allocation
growth in five synchronized replay samples per plan. The raw live allocations
include previously captured variants still resident and are not an estimate
of incremental split workspace. In these three down wins the automatic
workspace class goes from 4 to 8 MiB, the gate leg's existing class. The
same-stream serial legs can share that class, so added logical traffic is
not necessarily added physical residency. No new per-kernel Nsight counters were
collected for these comparisons.

Only the down variants were exported as standalone challengers. The S320 and
unseen gate increases are measured negative results; no fused producer/kernel is
warranted for either target. The separate [S192 flat MoE factorial](</tmp/rtx-moe-iteration5-20261119/integration/flat-bc.json>)
measured **5.224 → 5.208 ms** with the router and routed experts frozen
(14.95–16.76 µs saved in each paired visit). On top of the new router, the
additional shared gain was only 0.97–3.11 µs per visit. Consult that factorial
for the matched BF16 comparisons; component savings are not additive.

## First eight trials on the unseen shape

The tuner used a fresh progressive journal, 180-second cap, five calibrated
samples per candidate and the normal seed-1729 tuner dY. All eight admitted
candidates were correct and stationary. Total invocation: 17.77 seconds,
including six alternating deployment checks against the original initial
plan. The selection was the same-store down-only candidate
`90cd9c00e4e7a504fbe8c8c0`, not a predetermined schedule.

| Admission | Plan | Tuner objective |
|---:|---|---:|
| 1 | On-chip incumbent | 644.232 µs |
| 2 | 4/4 stored | 217.440 µs |
| 3 | 8/8 stored | 182.071 µs |
| 4 | 8/16 stored | 175.907 µs |
| 5 | 16/8 stored | 186.592 µs |
| 6 | 4/4 direct | 254.496 µs |
| 7 | 8/8 direct | 233.717 µs |
| 8 | Random exploration | 212.118 µs |

The final paired ordinary replay uses the real generated holdout dY and
retargets the archived S256 8/8 plan by changing only its token-axis reduction
chunks. Its separately compiled selected config measured 177.637 µs, also
faster in all four visits. The holdout regenerates independently named RNG
streams at the unseen partition, so no equality to the 16K archived X/dY is
claimed.

The unchanged wider D512/S512 4/4 control was stationary at 456.190 µs,
with finite oracle/replay results and no live growth. The M128/D384/S256
forward smoke retained all three on-chip/gate-only/gate-and-down variants;
prepared gate-and-down was 16.392 µs in both visits. Its unchanged checkpoint
BF16 forward was 8.197 µs. All native gradients passed the smoke oracle
(max NRMS 0.00000453), and every mutable replay matched the uncaptured result
bitwise with no live growth. This is a small-forward regression check, not a
claim that the tiny MXFP8 backward beats BF16.

## Raw evidence and reproduction

Root: `/tmp/rtx-moe-iteration5-20261119/shared/`. Source-matched comparisons
used immutable `source-v2.X00t7F`; baseline runtime snapshot SHA256
`a5e83c8eb4400c824105bc15a3a710f7c76e1cf13eb3d62a7d84ca80ffd2baf8`.
Each result includes command, actual experimental source hash, options via
its archived source result, operand hashes, sample batches, numerical
metrics, live bytes and revisioned schedules. Use HOME GPU0 only under the
shared lock. The benchmark will not install/overwrite a runtime winner.

```bash
CUDA_VISIBLE_DEVICES=0 flock /tmp/rtx-moe-backward-20261119-gpu0.lock \
  /home/achiya/vs_code_main/rtx/.venv/bin/python -m benchmarks.benchmark_mlp_dw_opening \
  --baseline-result /tmp/rtx-moe-k4-20261119/study/d384_f1of8_k4_e32_b16_l1024_packed/result.json \
  --output /tmp/rtx-moe-iteration5-20261119/shared/s192-paired-reproduction.json
```

- S192: `s192-paired-b.json`, `s192-shared.json`.
- S320 packed factorial input: `s320-a-target-paired.json`,
  `s320-a-target-shared.json`. Its baseline archive is
  `d384_f1of6_k6_e128_b16_l1024_packed/result.json`.
- Holdout: `s256-t12288-holdout.json` and adjacent
  `s256-t12288-holdout-tuning/425bb6fcda03a45015ee64a4/0c99f34ac4fa0cddbd3daa6a/observations.jsonl`;
  `s256-t12288-shared.json` has the frozen candidate. The holdout used
  `source-v3.mSscOv`, archived S256 `d384_f1of6_k4_e32_b16_l1024_packed`, and
  `--search --timing --lengths 1536 1536 1536 1536 1536 1536 1536 1536`.
- Controls: `s512-wide-control.json` and `tiny-forward-smoke.json`, using
  `source-v4.YA52DH`. The latter runs `benchmark_mlp_small_forward` with
  `--shape 128,384,256,384 --visits 2 --samples 7`.
- Strict cache reuse: `runtime-cache-reuse.log`. The opt-in CUDA test installs
  the verified S256 holdout sidecar in a pytest temporary cache, performs two
  bitwise-equal fullgraph forward/backward calls with no tuning, and verifies
  that the deployment cache reader returns the identical durable winner. It
  isolates cache inspection from the separate preflight-worker workflow.
- `s192-paired.json` was a sandbox CUDA NoDevice failure; `s192-paired-a.json`
  was an early harness failure reusing a nonleaf virtual projection view.
  Neither produced any accepted timings. The qualified harness obtains the
  view inside the compiled forward and checks the original leaf gradients.
