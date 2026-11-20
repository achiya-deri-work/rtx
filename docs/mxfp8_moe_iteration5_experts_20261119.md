# Iteration 5 packed experts: spend the eight trials on matched plans

The cold packed opening now tests prefix, producer bounds and compact forward
rows on one strong plan. No expert kernel, public routing metadata, numerical
contract, or untuned runtime default changed. On two previously losing packed
shapes and one unseen holdout, the bounded+compact family won all three eight-trial expert
searches. That is a scoped search result, not an automatic selection rule or
evidence for a universal 1.2× whole-layer gain.

## What changed

The old opening tested bounded and compact plans on weaker input, GLU, weight
and dW recipes. The matched experiment keeps those owners fixed. It admits
one legal indexed/fused, joint, row-store-matched legacy control with both
persistent dWs, then uses the existing preparation generator to add packed
fragment GLU and dual-orientation weights. The next four candidates change
only active prefix, the three producer bounds, or the private forward rows.
Both compact candidates keep the original backward row order and 32-value
quantization scale groups. The real trial still decides which configuration wins.

The opening applies only to a cold training recipe, a proven multi-pool pack,
the existing padding threshold, enough mean work for two dW panels, and six
unique legal controls. If a complete comparison cannot be built, the former
opening stands. Inherited ownership recipes, compatible opted-in priors, flat,
tight and tiny controls retain their opening. The only explicit revision is
`cold_packed_opening=1` in affected progressive checkpoint keys; the usual
expert source fingerprint invalidates stale expert runtime results. Kernel
revision 15, measurement revision 1, stationary-timing policy and soft deadline
are unchanged.

## Actual eight-trial journals

HOME GPU0 RTX 5090, Torch 2.13.0+cu132, CuTe 4.7.0. All values below are ms for
the CUDA-graph routed-expert F+B objective — not whole-layer times. Every row
was admitted, passed the independent quantized FP64 oracle before and after
replay, and passed the unmodified timing stationarity gate. Both B16 cases
replayed archived inputs and expert weights with the same tuned router; the
holdout has a fresh B8×1152 geometry, not a transferred winner.

| Admission | D384/E128/K6/H192/S320 B16×1024 | D512/E128/K6/H256/S384 B16×1024 | Holdout D384/E128/K6/H192/S320 B8×1152 |
|---|---:|---:|---:|
| 1. Cold incumbent | 4.869 | 6.630 | 2.663 |
| 2. Static legacy, both persistent dWs | 2.358 | 3.399 | 1.375 |
| 3. Strong static | 2.090 | 3.016 | 1.216 |
| 4. Strong + prefix | 1.879 | 2.706 | 1.085 |
| 5. Strong + all bounds, original rows | 1.837 | 2.643 | 1.051 |
| 6. Strong + compact, unbounded backward producers | 1.759 | 2.541 | 1.044 |
| 7. Strong + all bounds + compact | 1.725 | 2.496 | 1.021 |
| 8. Exploration: change the gate dW panel reuse axis | 1.724 | 2.491 | 1.019 |

Trial 2 is the exact archived expert winner on both target cases. Trial 8 is
the measured winner on all three shapes, but trial 7 and 8 are nearly tied;
the evidence is for their shared preparation/layout, not for preferring the
explorer's panel axis as a default. Invocation-to-winner wall times were
26.83, 27.22 and 27.78 s respectively, including context/oracle setup but excluding
the outer fixture/router compilation. The 90 s admission cap was not exhausted,
and no retry or fallback was needed. This does not establish that all eight
compilations fit a short first-hit deadline.

The two target cases schedule 48 rows per expert per L1024 pool, rounded to 64
for the original layout. Observed original live prefix was 1024 of 1312 physical
rows, versus 768 real rows and an 896-row safe compact storage bound. The
holdout had 432 scheduled/real, 512 original group-padded, and 704 physical
rows; compact storage was 512. Compaction bounds only forward M; bounded
backward still uses original-row addresses and the original live K envelope
(1024/512), before producer/consumer tile rounding. These are storage/work
envelopes, not issued FLOPs, bytes transferred, or occupancy.

## Matched frozen-route replay

This checks the production expert forward/backward through the fullgraph-static
wrapper, with the **archived** upstream gradient and the same route tensors
for every cell. The search uses its own fixed deterministic FP32 upstream;
these paired runs are independent confirmation. Each cell has four
alternating-order visits over seven ordinary timing batches. No compilation,
setup or injected profiler is timed.

| Geometry | Archived plan | Strong static | Bounded original | Bounded+compact control | Eight-trial winner |
|---|---:|---:|---:|---:|---:|
| D384 B16×1024 | 2.380 ms | 2.111 ms | 1.849 ms | 1.730 ms | 1.732 ms |
| D512 B16×1024 | 3.434 ms | 3.059 ms | 2.696 ms | 2.510 ms | 2.502 ms |

The winners were faster in every paired visit: D384 by 0.633–0.648 ms
(27.2% lower median), D512 by 0.920–0.933 ms (27.1% lower median). All 40
visits were stationary. Changing only compact forward rows on the bounded
anchor saved another 0.119 / 0.186 ms at the median. The separately timed
896-row private map took 6.144 / 6.153 µs (kernel-only graph, not added to or
subtracted from F+B). The explorer's slight search advantage did not repeat
on D384; the measured panel axis is not a default recommendation. Output,
input and every named expert parameter and route gradient were bitwise equal
to the respective archived plans; synchronized live memory stayed constant
through replay. The D384/D512 peaks for all retained cells and warmup
allocations were 1,388,730,880 / 2,074,204,672 bytes, not per-plan memory
measurements. Raw results: `d384/frozen.json` and `d512/frozen.json`.

## Whole D512: same router and shared MLP

| Cell | Median F+B | Million tokens/s |
|---|---:|---:|
| Archived native expert | 5.529 ms | 2.963 |
| Eight-trial expert winner | 4.653 ms | 3.521 |
| Compact checkpoint BF16, same router | 4.886 ms | 3.353 |
| Original-row checkpoint BF16, same router | 6.385 ms | 2.566 |
| Saved-intermediate BF16, same router | 5.532 ms | 2.962 |

The new expert won all four paired visits by 0.861–0.883 ms: 15.85% lower
time against the archived native plan, 1.188× its throughput and 1.050× the
strongest compact checkpoint comparator. All 20 visits passed the ordinary
timing stationarity check. Exact archived X, dY, all model parameter bytes,
hard IDs and inverse were verified. Output, input and every one of the 391
optimizer-visible parameter gradients agreed bitwise with the original native
plan; both changed-upstream graph checks passed. Each cell had zero growth
across five synchronized live-byte samples. Raw result: `d512/whole.json`.
Expert-only and whole times are separate scopes; don't add them or their
speedups. The paired D384 whole-factorial is handled by the integration study.

## Correctness and provenance

- Each of 24 actual evaluations checked output, input gradient, gate/up and
  down weight gradients, and the route-weight adjoint against the operational
  FP64 quantized oracle, including repeated graph replay. Worst per-shape
  NRMS across these values was 0.0001206, 0.0001008, 0.0000711 (fractions).
  The gates remain 0.003 output, 0.006 gradients, with a `2e-7` absolute-error
  safeguard for near-zero references.
- The exact generated bounded+compact plan passed fullgraph-static replay
  at fixed addresses across equal, ragged-with-empty-pools and all-in-one
  boundaries, a strided and changed upstream gradient, and a zero-weight
  real route with a nonzero adjoint. IDs and inverse stayed exact; padded
  route adjoints stayed zero. Worst oracle NRMS in this test was `3.70e-5`;
  five synchronized live-byte samples stayed at 36,183,040. Six existing
  compact-map and bounded-producer tests also passed their poison/suffix
  checks. The initial new-test failure was a Python stream-context binding,
  corrected before timing; it was not a CUDA fault.
- The new CPU tests use actual persisted admissions with a synthetic objective
  to test order, deduplication, the exploration slot, an uninterrupted vs
  resumed 8→16→32 prefix, compatible prior behavior, legal fallback, and
  unchanged flat, tight-capacity, tiny and healthy packed K4 policy controls.
  The existing inherited-anchor admission test is unchanged and passes.
  The focused CPU suite finished 41 passed, one CUDA-only skip (that test
  ran separately on GPU0). The unaffected controls are policy-equality
  checks, not new measurements of their latency.

Baseline runtime SHA256 `a5e83c8eb4400c824105bc15a3a710f7c76e1cf13eb3d62a7d84ca80ffd2baf8`.
Expert-only patched runtime snapshot SHA256
`774335c8e7791360a540dae99262aea685c38ce0f095937b0bf4cb61593c5a7f`.
Raw root `/tmp/rtx-moe-iteration5-20261119/experts/` contains `checked-source/`,
`search_summary.json`, `checked_correctness.xml`, and each case's
`search.log`, `fixture.pt`, `result.expert.json` and eight-row
`tuning/moe_effort/*/*/observations.jsonl`. Winner IDs:

- D384: `667b18e1d1847042a27f4222`
- D512: `fae797d3d385e73248d88d82`
- Holdout: `43b81af494c74d065a663eaa`

To reproduce the D384 admission and export a fresh frozen-route fixture:

```bash
study=/tmp/rtx-moe-iteration5-20261119/experts
archive=/tmp/rtx-moe-k4-20261119/study/d384_f1of6_k6_e128_b16_l1024_packed
cd "$study/checked-source"
py=/home/achiya/vs_code_main/rtx/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 flock /tmp/rtx-moe-backward-20261119-gpu0.lock \
  timeout 180 "$py" -m benchmarks.benchmark_moe_varlen \
  --source "$PWD" --output "$study/repro-d384/result.json" \
  --case-id d384_e32_k4_h128 \
  --case-spec '{"d_model":384,"expert_hidden_dim":192,"shared_hidden_dim":320,"top_k":6,"num_experts":128}' \
  --lengths 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 \
  --layout packed --stable-inputs --stable-experts --seed 17391 \
  --router-config "$archive/result.router.json" --shared-config "$archive/result.shared.json" \
  --max-autotune --search-schedule progressive --candidate-mix mixed \
  --tune-expert-trials 8 --tune-seconds 90 --prepare-only \
  --save-expert-fixture "$study/repro-d384/fixture.pt"
```

For D512 use its archive sidecars and D/H/S of 512/256/384. For the holdout,
retain the D384 case spec and router, give eight lengths of 1152, use a fresh
output directory, and omit `--shared-config` (the shared leg is not timed in
this expert-objective check). To reproduce the D512 whole and frozen paired
runs from the measured sidecars and recorded fixture:

```bash
archive=/tmp/rtx-moe-k4-20261119/study/d512_f1of6_k6_e128_b16_l1024_packed
case_dir="$study/d512"
CUDA_VISIBLE_DEVICES=0 flock /tmp/rtx-moe-backward-20261119-gpu0.lock \
  timeout 220 "$py" -m benchmarks.benchmark_moe_composition \
  --baseline-result "$archive/result.json" --candidate-config "$case_dir/result.expert.json" \
  --output "$case_dir/repro-whole.json" --bf16-checkpoint --bf16-compact --bf16-saved \
  --match-inputs --changed-dy --max-autotune --visits 4 --samples 7
CUDA_VISIBLE_DEVICES=0 flock /tmp/rtx-moe-backward-20261119-gpu0.lock \
  timeout 180 "$py" -m benchmarks.qualify_moe_packed_opening \
  --source-result "$archive/result.json" --fixture "$case_dir/fixture.pt" \
  --output "$case_dir/repro-frozen.json" \
  --config "static=$case_dir/static-strong.expert.json" \
  --config "bounded_original=$case_dir/bounded-original.expert.json" \
  --config "bounded_compact=$case_dir/bounded-compact.expert.json" \
  --config "winner=$case_dir/result.expert.json"
```

Keep the shape-gated opening: the matched expert wins survived full F+B on
D512, without forcing a winner or changing the public algorithm. No profiler
injection was used for these new timings; historical traces are background
only, not counter evidence for the newly admitted combinations. For another
kernel iteration, first reprofile the improved whole step rather than
extrapolating old bottleneck shares or the standalone map duration.
