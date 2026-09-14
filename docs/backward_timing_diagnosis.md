# Backward timing and graph scheduling (2026-09-14)

## Collecting mode-qualified evidence

Backward dataset jobs can set `"tags": {"backward_measurement_mode": "cuda_graph"}`.
Omit this tag for historical host-launch behavior. This is backward-only;
it does not switch forward or component diagnostics to graph timing.
Graph mode captures a complete calibrated batch, validates replayed dX/dW,
and reports `graph_capture_ms` separately from compile and event timings.
The mode changes manifest/context identity: use a new output bundle, never
relabel old host timings as graph timings.

For matched research pools, add `--backward-measurement-mode cuda_graph` to
`python -m tools.research.matched_gpu prepare --family mxfp8_bwd ...`.
Search/reference workers use the mode frozen in that pool, not a new CLI
override. Priors trained on host data remain cross-mode hypotheses, not
validated graph priors. Do not promote them without prospective evaluation.

The revision-23 backward study exposed an unstable promotion on
M6144/N512/K896. Promotion revision 2 now conservatively retains the incumbent
on explicitly nonstationary or order-reversing confirmation. This document
examines the timing mechanism; it does not promote a new measurement default.

## Fixed-candidate experiment

RTX 5070 Ti, Torch 2.13.0+cu132. Two fresh processes, identical seeded BF16
inputs per configuration, FP32 dX/dW references cast to BF16, unchanged 7%
normalized-RMS gate. All three execution modes passed both gradients. Forty
samples per mode, 32 calls per sample, forward/reverse mode ordering, compile,
warmup and graph capture outside measured batches.

| Configuration | Host-fed median | Single-call graph replay median | Whole-batch graph median |
|---|---:|---:|---:|
| Incumbent `8f041a7821c9f584c459` | 65.50 us | 64.23 us | 64.13 us |
| Challenger `036a60c43daa5cf63fa0` | 69.31 us | 88.09 us | 86.61 us |

Incumbent relative standard deviation fell from 11.47% host-fed to 0.28% in
whole-batch replay. However, the challenger became substantially slower under
capture. Both use dual-stream backward, so removing Python launch work is not
the only execution change. These are one-shape diagnostic measurements, not
an end-to-end compiled-training speed comparison.

## Narrow timeline evidence

A separate Nsight Systems run captured the challenger with CUDA graph node
tracing after warmup. Each mode contains exactly 96 kernels for 32 calls.
No Compute-counter or occupancy conclusion is drawn.

| Mode | GPU kernel span | Sum of concurrent overlap | Idle time within span |
|---|---:|---:|---:|
| Host-fed | 2248.96 us | 1180.19 us | 226.62 us |
| Single-call graph | 2752.35 us | 156.74 us | 62.75 us |
| Whole-batch graph | 2691.04 us | 152.16 us | 11.71 us |

Overlap is summed kernel durations minus their interval union. Graph replay
removes idle gaps but also loses most concurrent kernel overlap, explaining
why it can be slower. The exact scheduler/resource cause needs a controlled
single-stream/dual-stream and execution-order comparison; a timeline alone
does not prove resource saturation or a graph dependency bug.

Do not replace host-fed timing with replay and relabel old records. Capture
mode is an execution contract, must be explicit in measurement/cache identity,
and must match the intended consumer. The current backward study remains
timing-quarantined. No new model training, hard pruning or winner installation
is justified by this experiment.

## Reproduction and artifacts

### Single-stream controls

Four additional fresh-process runs change only stream schedule/order, keeping
the recorded kernels, shape and inputs. All dX/dW checks pass the unchanged
7% normalized-RMS gate. Artifacts:
`autotune_reports/bwd_timing_controls_20260914_v1/`.

| Source configuration | Single-stream order | Host-fed | Whole-batch graph |
|---|---|---:|---:|
| Incumbent | dX first | 91.24 us | 90.58 us |
| Incumbent | dW first | 91.21 us | 90.53 us |
| Challenger | dX first | 88.55 us | 88.02 us |
| Challenger | dW first | 89.12 us | 88.47 us |

This supports a dual-stream scheduling interaction rather than a general
graph-timing penalty. It does not yet prove which scheduler/resource mechanism
reduces overlap. The production dual-stream branch currently always launches
dX before dW, ignoring `execution_order`; those variants are not independent
implementations. Any future implementation/canonicalization of that axis must
version its evidence and preserve old records. The diagnostic driver records
the original configuration ID separately from its full overridden config.

Runtime search now performs that canonicalization under backward coordinate
revision 1. Only the ignored dual-stream `dw_first` alias is collapsed;
single-stream order still affects execution, and illegal interleaved/dual
configurations remain rejected. The raw offline adapter and old observations
are unchanged. Refinement requires new coordinate-revision evidence, while
plain reuse can keep a valid cached kernel. Focused CPU and CUDA checks passed;
see `autotune_qualification_matrix.md`.

The next autotuner change should make execution mode explicit and reject
cross-mode evidence reuse. A graph benchmark must capture/join auxiliary
streams correctly, validate dX/dW after replay, preserve cache-regime semantics,
and report capture setup separately. Do not silently enable graph measurement
for host-fed consumers or train a prior across these modes without a feature
and identity distinction. No timing default is changed by these controls.

### Commands

Driver: `tools/research/bwd_timing_modes.py`. For each configuration above:

```bash
python -m tools.research.bwd_timing_modes \
  --pool autotune_reports/matched_backward_20260914_v1/case1/pool.json \
  --config-id CONFIG_ID --output FRESH_OUTPUT.json
```

Use `--stream-schedule single --execution-order dx_first` (or `dw_first`)
with a new output path for the controls.

Artifacts live in `autotune_reports/bwd_timing_modes_20260914_v1/`: ordinary
timing/numerics JSON per configuration, `challenger_nodes.nsys-rep`, exported
SQLite, and `timeline_analysis.json` with its reproducible read-only script.
The first `challenger_trace` uses graph-level tracing and cannot resolve node
overlap; retain it but use the node trace for the table above. Profiled-run
latencies are diagnostic, not replacements for the ordinary timing files.
