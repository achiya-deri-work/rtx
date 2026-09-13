# Three-device collection analysis — 2026-09-13

## Inputs and audit

Analyzed the completed local `combined_first_hit_v1_achiyas-pc` directory,
the laptop `combined_parallel_v1_DESKTOP-8GVONEQ_20260912T231548Z.zip`, and
the repaired 5090 `combined_parallel_v2_sphera2-Dell-Pro-Max-Tower-T2-FCT2250_20260912T232403Z.zip`.
The v1/v2 names are launch labels, not kernel versions. Both remote runs record
the same RTX source identity. The local run includes pre/post execution-pipeline
identities; the 32 anchor tasks have no portfolio mismatches.

| Device | Observations | Successful | Explicitly unstable successes |
|---|---:|---:|---:|
| RTX 5070 Ti | 22,047 | 21,240 | 51 |
| RTX 5070 laptop | 12,529 | 11,797 | 1,026 |
| RTX 5090 | 27,870 | 27,111 | 13 |
| Total | 62,446 | 60,148 | 1,090 |

ZIP CRC checks and JSONL parsing passed. No observation identities were duplicated.
There are 59,058 successful observations eligible for the latency view after
removing explicitly unstable timings. This is **not** a claim that all remaining
timings are stationary: 4,226 MLP successes lack the linear sampling stationarity
metadata. Keep all 62,446 observations in the feasibility view.

Generated artifacts live in `autotune_reports/combined_three_device_v1/`:

- `audit.json`, `summary.csv`: source coverage and audit results.
- `latency_selection.parquet`, `feasibility_selection.parquet`: selection
  manifests pointing to original source/context/observation IDs. Raw data is unchanged.
- `all_device_holdout/evaluation.json`, `all_device_summary/summary.csv`.
- `mlp_device_holdout/evaluation.json`, `mlp_shape_holdout/evaluation.json`,
  and their corresponding summary directories.
- `reciprocal_probe.json`, `packed_failure_replay.json`: local GPU diagnostics.

## Two analysis corrections

The first-hit fitter previously included explicitly nonstationary successful
timings even though ingestion reported them. It now excludes them from training
**and evaluation**. Regression tests include an unstable, artificially fast row
and verify that it cannot become the oracle winner.

Cross-device linear fitting previously partitioned on machine ID and sharded
manifest digest, making every device a separate benchmark contract with no
holdout possible. Partitioning now excludes those fields and Git bookkeeping,
while retaining the actual source hash, kernel revision, precision, cache regime,
scaling policy and other measurement tags. Source/numerical policies are not
silently merged to manufacture larger training groups.

## Offline selector results

Gradient-boosted model: existing builtin context-ranking model, 16 estimators,
two ensembles, depth three. No deployment qualification was enabled.

The table is **mean percent regret at four candidates** relative to the best
successful observed candidate in each held-out context. Lower is better.
Each device is held out from training. These are observed-candidate replays,
not end-to-end first-hit wall times.

| Family | Boosted ranker | Winner retrieval | Original ordering | Random |
|---|---:|---:|---:|---:|
| MXFP8 backward | 7.79 | 46.94 | 47.34 | 13.67 |
| MXFP8 fully prequantized | 4.78 | 7.87 | 8.00 | 6.85 |
| MXFP8 fused forward | 26.08 | 132.61 | 133.26 | 45.15 |
| MXFP8 prequant forward | 3.51 | 6.26 | 6.52 | 5.44 |
| MXFP8 recomputed MLP | 1.05 | 0.27 | 79.16 | 66.83 |
| MXFP8 weight-prequantized | 4.68 | 10.49 | 10.45 | 6.81 |
| NVFP4 delayed | 2.98 | 4.34 | 4.43 | 3.85 |
| NVFP4 dynamic block | 3.02 | 2.65 | 2.54 | 3.26 |
| NVFP4 fully prequantized | 7.80 | 6.34 | 5.32 | 4.36 |
| NVFP4 JIT-region | 3.59 | 4.35 | 4.40 | 3.66 |
| NVFP4 weight-prequantized | 4.57 | 4.77 | 4.76 | 3.07 |

MLP **unseen-shape** holdout reverses the retrieval/ranker preference:
four-candidate mean regret is **0.80% ranker versus 4.07% retrieval**.
Device holdout can see the same shapes on other devices; shape holdout cannot.
Thus a good candidate for prospective testing is retrieval on familiar shapes,
with model-guided diversification for unseen shapes—not one universal selector.

Interpretation limits:

- Linear portfolios were adaptive, not a matched exhaustive configuration grid.
  Retrieval can only select configurations present in each observed test pool;
  unavailable retrieved IDs fall back to portfolio ordering.
- Failures and compiler cost do not enter successful-candidate regret. This
  does not estimate time-to-first-valid configuration or first-hit overhead.
- Random values are means over 100 seeded subsets per context.
- There are 116 MLP benchmark contexts, not 116 independent shape replicates:
  the local run contains additional measurement-context identities. Detailed
  summaries separate precision, phase and held-out device.
- Results are exploratory policy-selection evidence, not unbiased certification
  of a policy chosen after inspecting this table. No winners were installed.

## NVFP4 packed-weight numerical localization

The 5090 recorded 35 correctness failures, all weight-prequantized NVFP4.
The local device recorded 22 in that family and the laptop recorded two.

On the local 5070 Ti, replayed five archived failures across
`127×1536×1536`, `1009×2053×257`, and `513×1031×145` (M×N×K), using the
recorded input seeds and candidate configurations. All five failed the existing
gate. Changing **only** `quant_x.scale_reciprocal` to `direct` made all five
match the baseline output exactly, with maximum absolute error zero.

Independent nibble decoding on nine shape/seed cases showed identical E4M3
block scales but different FP4 payloads for alternative reciprocal calculations.
For example, the recorded `127×1536` activation input had 80 changed values
with the LUT reciprocal and 180 with the approximate reciprocal. The decoded
relative L2 differences from direct quantization were 0.577% and 0.843%.

The expressions are algebraically equivalent but not floating-point equivalent:

- direct: `(1 / tensor_scale) / block_scale`;
- LUT: rounded reciprocal of `block_scale`, then division by `tensor_scale`;
- approximate: approximate reciprocal of `tensor_scale * block_scale`.

Tiny reciprocal differences can move values across coarse FP4 rounding boundaries.
This explains the reproduced failures before GEMM, not a general tensor-core
failure. The GEMM comparison uses the existing baseline schedule; it is not an
independent proof of the entire packed-inference implementation. The separate
decode probe isolates the representation change without GEMM.

No tolerance was relaxed and no production kernel changed. Treat these options
as numerical-policy choices unless exact equivalence is established. The safe
next step for the default packed-weight search is a direct-reciprocal contract,
or an explicitly validated alternative policy—not globally disabling useful
block-only optimizations where the outer scale is one.

Other retained failures still need separate investigation: 12 MLP capture errors
(eight local, four laptop), one local MXFP8 fully-prequantized correctness error,
one laptop MXFP8 weight-prequantized error, and four compiler errors across the
local/laptop runs. This study does not mark those paths release-qualified.

## Reproduction

Use the audit and fit commands in [first-hit autotuning](first_hit_autotuning.md).
GPU diagnostics are correctness runs, not performance benchmarks:

```bash
python -m benchmarks.probe_nvfp4_reciprocal --output /tmp/reciprocal_probe.json
python -m benchmarks.replay_nvfp4_failures <5090.zip> --output /tmp/packed_failure_replay.json
python -m benchmarks.summarize_first_hit_evaluation <evaluation.json> --output <summary-dir>
```

Keep the raw failures, freeze these reports, and prospectively compare the
retrieval/ranker hybrid with random and existing ordering before deployment.
