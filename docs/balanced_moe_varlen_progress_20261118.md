# Packed BalancedMoE: 18 November progress

Status: experimental. The requested **1.2x full-forward/backward speedup on
all four designs is not qualified**. In particular, two D=384 designs still
lose to aggressively autotuned checkpointed BF16 on the RTX 5090.

## Matched comparison

The measurement is the complete compiled layer: independent per-sequence
routing, one shared MLP, routed experts, and actual backward to the input and
every Parameter. Native experts recompute; the BF16 comparison checkpoints
its FFNs with `use_reentrant=True`. Both retain the same native router.
Saved-intermediate BF16 is a separate diagnostic, not the target baseline.
Inductor `max_autotune=True` is enabled on both sides. Its candidate
benchmarking runs in subprocesses: an illegal-accessing Inductor candidate
previously poisoned an in-process tuning session. Neither the candidate pool
nor the precision policy was weakened to avoid that compiler failure.

Timing excludes compilation, tuning, capture and warmup. Three rounds of
nine CUDA-event samples alternate the measurement order. The reported latency
is the median of round medians; a 1.2x pass additionally requires every paired
round to reach 1.2x. A single lucky median does not qualify a case.

Snapshot `safe-final`, source digest
`32a8de6a90124fbb0da5e06029b247287a2e257a58606492f2f9fac4951ace55`,
work_5090, Torch 2.13/CUDA 13.2, CuTe DSL 4.7:

| Design D/E/K/H, shared H | Tokens per sequence | MXFP8 ms | BF16 checkpoint ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| 384/24/6/128, 256 | 2048,8000,3072,16384 | 5.376 | 5.684 | 1.057x |
| 384/32/4/128, 256 | 2048,8000,3072,16384 | 5.116 | 4.823 | 0.943x |
| 512/64/8/128, 384 | 2048,8000,3072,16384 | 8.249 | 9.771 | 1.184x |
| 768/48/6/256, 512 | 2048,8000,3072,16384 | — | — | Inherited seed rejected |
| 384/24/6/128, 256 | 512,2000,768,4096 | 1.517 | 1.416 | 0.933x |
| 384/32/4/128, 256 | 512,2000,768,4096 | 1.484 | 1.450 | 0.977x |
| 512/64/8/128, 384 | 512,2000,768,4096 | 2.584 | 2.786 | 1.078x |

The remaining contexts and the corrected-search study are separate results,
not implied by this partial table. Earlier 5090 comparisons used a weaker
BF16 compilation baseline; their ratios must not be presented as wins over
this baseline.

## Implemented search alternatives

All remain shared-autotuner choices, not hand-pinned SKU schedules:

- Warp/register-transpose quantization with 2/4/8 token lanes, feature/token
  traversal and independently quantized 32-slot scales. No materialized BF16
  transpose or CTA SMEM transpose is needed in this alternative.
- Independent split-K for each expert weight gradient, FP32 partial storage,
  deterministic FP32 reduction, and one final BF16 conversion.
- Indexed input-transpose preparation: a compact compiler-owned int32 map
  replaces replicated expert BF16 X. Feature-axis scales are not reused as
  leading-axis scales.
- A recomputed-down route-gradient epilogue that retains ordinary BF16
  output rounding but writes FP32 dot-product partials instead of complete
  expert outputs. The separate epilogue remains available.
- Independent cooperative/TMA choices for linear stages, including complete
  single-buffer and alternative tile seeds. One short reduction does not
  force every weight-gradient GEMM to use the same transport.
- H=128 on-chip forward decoupled from prepared backward. Original packed,
  staged and fully on-chip families remain available.

The last alternative passed independent quantized-FP64 forward/all-gradient
checks in two geometries and compiled full-MoE checks on all three H=128
designs. However, a 64-trial 5060 search at 29,504 tokens selected the packed
forward instead. It measured 22.378 ms native versus 23.975 ms checkpointed
BF16 (1.071x median; round ratios about 1.065/1.139/1.109). This is a negative
result for that fused-forward hypothesis, not evidence for a new default.
The fused kernel still requantizes operands per tile; a genuinely prepared
on-chip implementation is different work and has not been implemented here.

## Search/startup fixes

An inherited full-tile TMA epilogue rejected the large H=256 problem before
the first expert trial. Tuning now first checks static legality and, if
necessary, validates a direct-store alternative or a legal default seed.
The reason is recorded. This changes only search initialization, not explicit
untuned configuration semantics, numerical policy or winner selection.

Expanded prepared subfamilies also crowded other engines out of the early
structural proposal pool. Search now round-robins execution engines first,
then the layout/traffic families within each engine. CPU regressions check
that the early pool includes staged scalar/cp.async/TMA pipelines, on-chip
execution and the hybrid, rather than many aliases of prepared execution.
Automatic fusion exploration starts after eight recorded trials, unless
the user explicitly starts from a fused schedule. Disabling on-chip search
also disables the hybrid axis. The search checkpoint identity changed.

## Evidence and remaining bottlenecks

The final focused CPU suite, including startup/fusion-gate regressions, was
194 passed / 100 CUDA-skipped. Earlier candidate suites on both 5090 and 5060 each
passed 179 combined CPU/CUDA tests, plus four compiled packed-design tests.
These counts are not claims of 179 distinct GPU shapes. The additional
hybrid checks are two FP64-oracle cases and three compiled H=128 designs.

A graph-replay Torch/CUPTI trace of the D=384,E=32,7,376-token case with an
inherited schedule recorded roughly 444 us in packed column Sinkhorn
operations, 123 us in an Inductor scalar-router-parameter reduction, and
substantial expert quantization/GEMM time. That trace is diagnostic, not the
latency of the newly selected winner. Nsight hardware counters were not
available on work_5090; no measured L2 hit rate, bandwidth saturation or
tensor-pipe utilization is claimed. CUDA graphs already remove host launch
gaps from the headline comparison, so adding tiny internal graphs is not a
solution to the remaining device work.

Raw snapshots, timings, traces, frozen configurations and residuals belong in
`autotune_reports/balanced_moe_varlen_12x_20261118/` (Storage4-backed). Active
remote runs use immutable source directories and detached, nonoverlapping
jobs. They do not modify unrelated services or require client-managed
allocations/streams. The next large-pack wave tests the corrected engine
coverage and H=256 startup; it must be measured before declaring improvement.

## Routed H x1.5, aligned to 64

The follow-up changes only routed hidden widths: H=128 becomes 192 and H=256
becomes 384. Shared H remains 256/256/384/512. It covers all four designs at
both packed length sets above. The runtime source is unchanged from the
completed `hierarchical` study (digest
`bfb6fa23def59bbef323ad6e79109cb0d731ce5d68f2c02564f43dc226a58b66`).

The benchmark first replays the original-width initialization, then creates
the wider routed weights and copies the unchanged router/shared parameters.
X, dY, router/shared parameter hashes and routing identities are checked
against each original context. Original router/shared schedules stay fixed;
the wider expert forward/backward receives up to 96 fresh shared-autotuner trials
with a 180-second search deadline. Both precision baselines get the wider H.
Compilation, tuning and warmup remain excluded from timing. This is a kernel
performance/correctness study, not a training-convergence comparison.

H=192 is intentionally not rounded to 256. It exercises tails in 128-wide
K tiles, whereas H=384 is 128-aligned. The existing H=128-only on-chip engine
is ineligible; prepared and staged implementations remain selectable.

The original reference bundle, exact wider-study source and CPU benchmark
regressions are preserved in
`autotune_reports/balanced_moe_h15_20261118/`. All eight wider contexts completed
successfully; their complete evidence bundle and summary are preserved locally
there as `rtx-moe-h15-complete-evidence.tar.gz` and
`rtx-moe-h15-complete-summary.json`.

Large-pack results (29,504 real tokens), full layer forward/backward:

| D/E/K | H before -> after | Native ms, wider H | BF16 checkpoint ms, wider H | Speedup before -> after |
| --- | --- | ---: | ---: | ---: |
| 384/24/6 | 128 -> 192 | 6.084 | 6.815 | 1.007x -> 1.120x |
| 384/32/4 | 128 -> 192 | 5.563 | 5.509 | 0.944x -> 0.990x |
| 512/64/8 | 128 -> 192 | 9.515 | 11.769 | 1.142x -> 1.237x |
| 768/48/6 | 256 -> 384 | 12.543 | 15.136 | 1.175x -> 1.207x |

The before column uses the completed same-source `hierarchical` matrix, not
historical best-ever schedules. Every large-pack context completed all 96
expert trials, preserved all unchanged hashes, passed the quantized FP64
oracle and fresh-gradient replay checks, and had zero live-allocation growth.
Worst output/input/weight/route-gradient oracle NRMS across these four was
0.000127 (agreement with the quantized oracle, not error versus unquantized
BF16). Both designs above 1.2x cleared that threshold in every paired round.

Smaller-pack results (7,376 real tokens), full layer forward/backward:

| D/E/K | H before -> after | Native ms, wider H | BF16 checkpoint ms, wider H | Speedup before -> after |
| --- | --- | ---: | ---: | ---: |
| 384/24/6 | 128 -> 192 | 1.680 | 1.668 | 0.954x -> 0.993x |
| 384/32/4 | 128 -> 192 | 1.558 | 1.525 | 0.975x -> 0.979x |
| 512/64/8 | 128 -> 192 | 2.982 | 3.409 | 1.068x -> 1.143x |
| 768/48/6 | 256 -> 384 | 3.584 | 3.971 | 1.129x -> 1.108x |

All eight contexts completed 96 expert trials apiece and passed the same
hash, FP64-oracle, finite-gradient, fresh-gradient-replay and zero-live-growth
checks. Worst oracle NRMS remains 0.000127 across the complete matrix. The
final local archive audit parsed 50 JSON files and 3,064 JSONL records with
no invalid records; the completed compressed bundle also passed gzip integrity.
Ten focused CPU benchmark/CLI regression tests passed.

**Conclusion:** wider H helps particularly at the larger token pack, but is
not a universal fix. Two of eight wider contexts meet 1.2x in every round.
The D=384,E=32 case remains at/below parity at both sizes, and the widest
design's smaller-pack ratio regresses slightly. This experiment does not
establish a hard roofline, an exhausted search space, or convergence quality.
The benchmark includes aggressive Inductor autotuning with isolated failed
candidate fallback; it is not a claim of a globally optimal BF16 backend.
