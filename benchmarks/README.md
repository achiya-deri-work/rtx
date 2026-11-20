# Benchmarks and tuning utilities

These scripts are development tools, not public Python APIs. Run them from the
repository root so imports and output paths are predictable.

NVFP4 backward and matched SiTU decoder studies are indexed in
[the consolidated research guide](../docs/nvfp4_research.md). These opt-in
MS-EDEN/prototype tools do not change stable `NVFP4Linear`'s MXFP8 backward.
Use `tools/qualify_nvfp4_research.sh` for the bounded GPU correctness smoke,
not a performance or convergence claim.

| Script | Purpose |
| --- | --- |
| `qualify_autotune_regions.py` / `run_moe_region_qualification.py` / `analyze_autotune_regions.py` | Prespecified region/retrieval/ranker ablations, actual progressive budgets, whole-layer replay/oracle/memory checks and CSV/Parquet export; [HOME results](../docs/autotune_region_opening_20261120.md) |
| `qualify_flat_transport_cycle.py` / `validate_router_fp64.py` / `qualify_flat_transport_replay.py` | Frozen-source 8/32-trial router study with fixed expert/shared plans, independent full-router FP64 checks and paired saved-plan replay on explicitly requalified sources; [follow-up results](../docs/autotune_cycle3_20261120.md) |
| `train_decoder.py` | Resumable byte-level TinyStories BF16/MXFP8/NVFP4-delayed/NVFP4-block decoder comparison |
| `run_decoder_four_mode.sh` | Exact 460.8M-token four-mode release convergence/runtime launcher |
| `run_decoder_batch64_autotune.sh` | Two bounded 12-minute searches plus verification, winner installation, and batch-24/64 model races |
| `benchmark_mxfp8_frontend.py` | End-to-end `torch.compile` MXFP8 frontend benchmark |
| `benchmark_balanced_moe.py` | Correctness-gated whole routed-expert tuning versus batched compiled BF16; optional same-CuTe-router complete-layer comparison, no NVFP4 |
| `profile_moe.py` / `run_moe_profile_study.py` | Frozen-schedule MoE phases, token scaling, host versus graph timing, real-route profiler fixtures and controlled structural A/Bs |
| `profile_moe_amortization.py` / `analyze_moe_amortization.py` | Bounded one-axis token/width/hidden/expert/fan-out/packing study; matched checkpointed BF16 and frozen-route experts, ordinary timing separated from diagnostic traces |
| `profile_moe_challenge.py` / `report_moe_challenge.py` | Broad current-source 38-context MoE matrix, equal-token packing and input-distribution controls, fresh bounded tuning, compiled checkpointed/saved BF16, independent quantized oracles and phase/trace attribution; resumable CSV/Parquet outputs |
| `profile_moe_active_budget.py` / `run_moe_active_budget_study.sh` | Hard ±7.5% active-4D budget, 64-aligned shared/expert widths, packed/flat and B=8/16/32 scaling; `--study fanout` adds K4, joint least-fraction-error selection and matched K/E controls; matched saved/checkpointed/compact BF16, oracle/replay/memory gates and automatic compressed artifact collection; [study contract](../docs/mxfp8_moe_active_budget_20261119.md) |
| `consolidate_moe_profiles.py` | CPU-only audit and namespaced JSON/CSV/Parquet consolidation of complete studies, including exact shared-width control hashes; does not mutate input data |
| `qualify_moe_packed_opening.py` / `qualify_flat_router_backward.py` / `benchmark_mlp_dw_opening.py` | Iteration-5 matched expert/router/shared comparisons, bounded actual-admission checks and independent numerical/replay/memory gates; [results and limitations](../docs/mxfp8_moe_iteration5_20261119.md) |
| `qualify_moe_iteration6_recompute.py` / `qualify_flat_router_forward.py` / `diagnose_moe_graph_resources.py` | Bounded recompute/forward-geometry comparisons and graph-lifetime reproduction; preserves rejected candidates and distinguishes fixture-only speed from portable qualification; [iteration 6](../docs/mxfp8_moe_iteration6_20261119.md) |
| `analyze_moe_profile.py` | CPU-only correlated Torch/Systems timelines, Compute counters and ordinary timing/variant CSV exports; [diagnosis and reproduction](../docs/balanced_moe_profile_20261117.md) |
| `collect_moe_clean.py` / `finalize_moe_collection.py` | Audit-gated, deadline-bounded collection and immutable qualified-prefix partitions with shared-MLP CSV/Parquet export |
| `benchmark_moe_geometry_frontier.py` | Focused existing-template geometry/pipeline search through the shared autotuner; separate paired promotion required |
| `study_moe_heuristics.py` / `analyze_moe_heuristics.py` / `qualify_moe_heuristics.py` | Immutable two-GPU discovery/held-out 32-trial studies, exact conditional effects, geometry-grouped cost models and prospective eight-trial research-prior checks; [contract and findings](../docs/mxfp8_moe_heuristics_20261120.md) |
| `profile_moe_routing_cost.py` / `analyze_moe_routing_cost.py` | Matched ordinary router/preparation/whole timings, source-labelled expert traces, node-level graph copy attribution and NCU traffic; overlapping scopes are never summed |
| `qualify_moe_store_prior.py` / `analyze_moe_store_prior.py` | Serialized actual eight-trial held-out expert/shared searches, progressive sixteen-trial controls, oracle/replay/memory gates and compound-store-prior ablation |
| `benchmark_mlp_store_frontier.py` | Common dense-MLP finite store/partition/fusion controls with independent numerical checks, optional hash-matched source-study inputs and warmed preparation traces |
| `benchmark_moe_execution.py` / `benchmark_moe_input_reuse.py` | Matched-source ABBA compiled-host versus outer-graph latency, independent gradient checks and quantize-once traffic A/Bs; [measured fixes](../docs/balanced_moe_fixes_20261117.md) |
| `benchmark_moe_composition.py` | Whole-layer fixed-recipe factorial, all-gradient replay checks, and `--bf16-compact` to count stronger BF16 compaction/checkpoint cost |
| `benchmark_moe_prefix_composition.py` | Frozen-route fragment/persistent/prefix expert-only ablation; verifies archived inputs, virtual parameters and hard-route hashes |
| `benchmark_moe_token_reduce.py` | Independent scalar/token combine and dX timings, paired graphs, alignment fallbacks and preserved numerical contracts |
| `benchmark_moe_bounded_preparation.py` | Byte-qualified full versus independently bounded original-layout producer graphs, with actual prefix/read-envelope metadata |
| `benchmark_moe_prefix_composition.py --compact-forward` | Exact archived-route compact-M expert forward/F+B comparison, metadata costs and arena/peak accounting |
| `benchmark_mlp_small_forward.py` | Backward-matched on-chip/gate-only/gate+down shared MLP, independent recipe, changed-input replay and actual eight-trial opening |
| `qualify_varlen_router_opening.py` | Matched-source fresh four-trial and cumulative eight-trial router qualification with exact progressive journal prefixes |
| `benchmark_shared_mlp_checkpoint.py` | Same-parameter saved BF16, both checkpoint variants and recomputed MXFP8; compiled `.backward()`, graph timings and retained activation storage, without tuning |
| `compare_moe_precisions.py` | Broad matched flat/packed whole-MoE MXFP8 versus saved and checkpointed BF16, native/compacted BF16 layouts, fresh bounded component tuning, independent oracles, replay/memory checks and CSV/Parquet export |
| `benchmark_nvfp4_end_to_end.py` | Production backend paired forward-plus-MXFP8-backward layer benchmark |
| `benchmark_nvfp4_frontend.py` | Paired fullgraph dynamic-forward latency and normalized numerical-error comparison against MXFP8 |
| `validate_nvfp4_convergence.py` | Controlled BF16/current/rowwise-JIT/delayed/exact scale-policy convergence study |
| `benchmark_mxfp8_prequant.py` | Dynamic BF16 quantization plus native-scale MXFP8 GEMM, with mainloop/epilogue stage and persistent-locality controls |
| `benchmark_torchao_fp8_rowwise.py` | TorchAO rowwise FP8 comparison baseline |
| `validate_production.py` | Unified MXFP8/NVFP4 eager/compiled, training/inference, packed-state, stream, cache, and long-dW readiness matrix |
| `validate_mxfp8_production.py` | Legacy MXFP8-only readiness matrix |
| `benchmark_mxfp8_bwd.py` | Fused and quantize-once full/workspace/atomic, persistent split-FP32, ordinary and one-BF16-load shared-G quad quantization, dual-stream, wide-CTA, and TMA/cp.async clustered operand-reuse backward families |
| `tune_composable_prequant.py` | Composable learned/global plus local forward tuning |
| `tune_composable_bwd.py` | Composable MXFP8 backward tuning |
| `qualify_mxfp8_long_mlp.py` | Complete long/narrow MLP tuning, recipe checks and paired qualification; optional steady-state tracing |
| `qualify_long_linear_bwd.py` | Numerical-policy-pinned complete linear backward tuning and calibrated racing |
| `compare_long_narrow_revisions.py` | Isolated previous/current source workers, common CUDA-graph warmup and alternating qualification |
| `compare_mlp_numerics_revisions.py` | Identical CPU-seeded inputs across isolated source versions; fixed-policy and selected-policy output/all-gradient errors against independent MXFP8, BF16 and FP32 references |
| `verify_composable_bwd.py` | Independent backward winner verification and racing |
| `tune_mxfp8_native_quant.py` | Standalone native quantizer coordinate sweep |
| `tune_mxfp8_native_gemm.py` | Standalone prequantized GEMM coordinate sweep |
| `run_5070_autotuner_study.sh` | Resumable 3-hour laptop / 6-hour desktop prospective optimizer study |
| `launch_nvfp4_full_power_6h.sh` | Clock-driven six-hour NVFP4 release/topology/deep-JIT/overflow campaign for SM120 SKUs |

Portable multi-shape and cross-device campaigns belong in
`autotune_manifests/` and should be launched with `rtx-autotune`; these scripts
are for focused kernel investigation.

MoE BF16 references using the native packed router inherit its static expert
capacity. They are matched-schedule controls, not necessarily the fastest BF16
implementation. Padding-sensitive speedups need a compaction-aware control;
Torch traces can also omit conditional repair-body kernels. See the
[45-context challenge study](../docs/balanced_moe_challenge_20261119.md) for both
controls, raw Systems traces, numerical caveats and interpretation limits.

MoE study collection now supervises each visit's entire process group and
aborts on fatal CUDA warnings even if an inner autotuner returns success.
Use `python -m benchmarks.audit_collection_health <study> --output health.json`
to inspect old logs without CUDA; add `--quarantine` to preserve but exclude
confirmed fault-contaminated evidence. Audit, offline training/export and
prior loading honor quarantine markers, including in copied ZIP bundles.
The GPU0 incident and zero-copy VirtualLayer correction are documented in
[`docs/balanced_moe_gpu0_incident_20261118.md`](../docs/balanced_moe_gpu0_incident_20261118.md).

After pulling the same commit on each 5070 machine, launch with
`./benchmarks/run_5070_autotuner_study.sh laptop-3h` or
`./benchmarks/run_5070_autotuner_study.sh ti-6h`. The script calibrates once,
uses the pulled checkout even with a non-editable environment install, and
resumes the same residual journals when rerun.

For the current-revision release-candidate campaign, run the same command on
the RTX 5070 Ti and RTX 5070 Laptop:

```bash
./run_5070_4h.sh
```

The root launcher fast-forward pulls, detaches itself, runs the CUDA and unified
production gates, calibrates the exact GPU/compiler checkout, and spends the
remaining four-hour deadline on balanced residual collection across all nine
MXFP8/NVFP4 production families. It then verifies finalists, audits the
journals, exports CSV/Parquet, and installs verified device-local winners.

All generated JSON, JSONL, logs, datasets, and compiled artifacts must go to an
ignored output directory such as `autotune_results/`, `autotune_logs/`, or
`autotune_datasets/`.

For the matched NVFP4 revision-8 campaign on the RTX 5070 Ti, RTX 5070 Laptop,
and RTX 5090, pull the same commit and run this command on each machine:

```bash
./benchmarks/launch_nvfp4_full_power_6h.sh 6h
```

The launcher identifies the machine by hostname; set a stable explicit label
when hostnames are ambiguous:

```bash
RTX_AUTOTUNE_NODE=rtx5070ti ./benchmarks/launch_nvfp4_full_power_6h.sh 6h
RTX_AUTOTUNE_NODE=rtx5070_laptop ./benchmarks/launch_nvfp4_full_power_6h.sh 6h
RTX_AUTOTUNE_NODE=rtx5090 ./benchmarks/launch_nvfp4_full_power_6h.sh 6h
```

The worker reserves the final five minutes for journal audit. Before that it
uses 25% of the clock for matched release contexts, 35% for broad M/N/K and
ragged topology coverage, 30% for deep JIT-region exploration, and all
remaining time for two-replicate overflow contexts. A phase that exhausts its
search space early advances immediately; watchdog/CUDA-context exits resume
from append-only residual journals. Each phase emits independent CSV and
Parquet reports and never installs exploratory winners automatically.

Run the fixed-token decoder convergence comparison with:

```bash
python benchmarks/train_decoder.py \
  --precision bf16 mxfp8 nvfp4_delayed nvfp4_block \
  --optimizer fp32_master \
  --batch-size 24 \
  --steps 37500 \
  --warmup-steps 375 \
  --log-interval 25 \
  --validation-interval 625 \
  --checkpoint-interval 625
```

The equivalent reproducible launcher is
`./benchmarks/run_decoder_four_mode.sh`. Set `RTX_DECODER_OUTPUT` to choose a
new artifact directory; additional CLI arguments are forwarded verbatim. A
completed run writes `summary.json` with per-mode runtimes, speedups, final
losses, complete validation-loss curves, and the honest throughput-gate
status. An unmet performance target is reported without discarding an
otherwise valid convergence campaign.

Production model execution uses BF16 parameters, gradients, and activations.
FP32 is reserved for sensitive reductions and calculations as defined by
[`PRECISION_POLICY.md`](../PRECISION_POLICY.md). Optimizer-state precision is
reported separately: the production convergence/runtime comparison uses FP32
AdamW moments and master parameters uniformly across all four modes. The
`--optimizer bf16` path is a kernel-isolation throughput ablation and must be
reported as such. A bounded study can use
`--stop-after-step 5000` without changing the 37,500-step cosine schedule;
rerunning the same command without that bound resumes from its atomic
checkpoint.

Check the append-only convergence and runtime results with:

```bash
python benchmarks/check_decoder_throughput.py <run-directory>
```

This reports final convergence and steady-state runtime for all four modes. It
enforces the default 1.30x MXFP8 and 1.35x NVFP4-block speedup thresholds
against the otherwise identical BF16 production baseline; delayed scaling is
reported without a default speed gate.

The first invocation downloads one official TinyStories training parquet
shard and creates a byte-token cache. All precisions receive the same initial
state and deterministic step-indexed windows. Metrics are append-only JSONL;
each precision has an atomic checkpoint and resumes by default. A clean
37,500-step run at physical batch 24 presents exactly 460.8 million tokens to
each model, approximately one shard-equivalent token pass. The 375 warmup
steps likewise preserve the original 4.608-million-token warmup. This is a
real 12,288-token optimizer batch, not gradient accumulation.

The RTX 5070 Ti saturation sweep measured these steady-state medians with
FP32 AdamW state/master weights (thousands of tokens/s):

| Physical batch | BF16 | MXFP8 | NVFP4 delayed | NVFP4 block | Peak allocation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 202.1 | 279.8 | 204.3 | 251.6 | 1.94 GiB |
| 16 | 215.3 | 295.1 | 226.0 | 267.0 | 3.00 GiB |
| 24 | 216.9 | 296.7 | 216.8 | 268.1 | 3.99 GiB |
| 32 | 219.3 | 294.3 | 222.6 | 267.3 | 5.02 GiB |
| 48 | 220.9 | 293.7 | 224.5 | 266.8 | 7.07 GiB |
| 64 | 222.0 | 294.1 | 227.2 | 269.1 | 9.19 GiB |

Batch 24 is the selected low-precision throughput knee: increasing to 64
consumes another 5.2 GiB without improving MXFP8 or block-NVFP4 throughput.
Use a new `--output` directory when changing the model geometry or training
schedule.

Run the complete frontend matrix before promoting a release:

```bash
python benchmarks/validate_production.py \
  --output autotune_reports/production_matrix.json
```

`--quick` retains every category but reduces the long-reduction check from
M=8192 to M=1024. Use `--frontend mxfp8` or `--frontend nvfp4` for focused
diagnosis; the release gate uses the default `both`.

Collect the paired all-kernel/all-scaling autotuning dataset in a detached,
resumable process with:

```bash
./benchmarks/launch_all_kernel_autotune.sh 4h
```

The launcher performs `git pull --ff-only`, refreshes the editable install,
checks SM12x and the exact runtime contract, calibrates the GPU when needed,
then starts the campaign with `nohup` and `disown`. It prints the PID, durable
residual dataset directory, report directory, and log path. Follow progress
with the exact `tail -f` command it prints. Set `RTX_AUTOTUNE_NODE` to give
each machine a stable label; rerunning with the same label resumes its residual
contexts. Override `RTX_AUTOTUNE_OUTPUT_DIR`, `RTX_AUTOTUNE_REPORT_DIR`, or
`RTX_AUTOTUNE_LOG_DIR` when collecting multiple independent replicates.

The worker rotates breadth-first across MXFP8 forward/inference, shared MXFP8
backward, NVFP4 current/block/delayed/JIT-region forward, and
NVFP4 packed inference. On exit it audits the journals and emits both CSV and
Parquet without automatically installing exploratory winners.

Benchmark the production materialized delayed forward and its MXFP8 backward
against `MXFP8Linear` with paired AB/BA rounds:

```bash
python benchmarks/benchmark_nvfp4_end_to_end.py \
  --compile --nv-scaling delayed --nv-backend auto \
  --output autotune_reports/nvfp4_delayed_end_to_end.json
```

The end-to-end benchmark supports eager and `--compile` runs. It verifies that
both frontends produce bit-identical MXFP8 dX/dW and uses device-wide fences so
dual-stream backward work cannot escape the measurement. Add `--profile` for a
per-kernel CPU/CUDA breakdown while investigating launch overhead.

Use `benchmark_mxfp8_bwd.py --reuse-sweep` for paired races across the legal
128x256 and 256x128 dW reuse basins, BF16 stage counts, and register budgets.

## Balanced routing

For expert preparation attribution, `benchmark_moe_prepared.py` compares the
previous selected configuration, quantize-once execution, and compiled batched
BF16 with identical routes. `--joint-backward`, `--native-scales`,
`--transpose-mode`, `--glu-quant-vec`, `--linear-transport`,
`--weight-epilogue`, and `--all-tma-outputs` expose implementation A/Bs, not hidden
runtime dispatch overrides. `benchmark_balanced_moe.py --whole-layer` includes
the router/shared path; tune those with `--auxiliary-effort` or freeze exact
matching configs with `--auxiliary-study`. `--config-study` carries forward a
verified expert seed instead of restarting each optimization wave from defaults.
Do not present expert-only forward wins as complete training-step wins.

`benchmark_moe_checkpoint.py` compares recomputed MXFP8 against BF16 with only
the expert/shared FFNs checkpointed (`use_reentrant=True`). The saved-state
router is common to both. It measures actual backward for input and every
Parameter, with ordinary synchronized host calls and outer CUDA-graph replay
reported separately. Saved-BF16 is a diagnostic, not the optimization target.
Use `--source`, `--seed-studies`, `--router-config`, `--save-reference` and
`--reference` to freeze policy, require exact routing/input identities and check
all gradients across source revisions. Compilation/autotuning is not timed.
See [whole-layer qualification](../docs/balanced_moe_finish_20261117.md).

`benchmark_moe_varlen.py` qualifies packed independent sequences, with exact
source/input/route identities, a quantized FP64 oracle, fresh gradients under
graph replay, stable live-allocation checks, and saved versus checkpointed
BF16 reported separately. `--max-autotune` enables Inductor max-autotuning on
both wrappers and BF16 references. The FP64 oracle chunks independent experts
to avoid converting the whole expert batch to FP64 at once.

`tune_moe_varlen_matrix.py` runs all four designs at mixed lengths
`[512,2000,768,4096]` and 4x those lengths, in balanced waves. It uses the shared
router/expert/MLP tuners and isolated processes, records a separate JSON/log per
context, and resumes completed source-matched results. Seeds are initial
schedules, not forced winners. For example:

```bash
python benchmarks/tune_moe_varlen_matrix.py \
  --seeds <directory-with-delivery-config-sidecars> \
  --output autotune_reports/moe_varlen_matrix --max-autotune \
  --expert-trials 96 --router-trials 16 --shared-trials 32
```

The 1.2x target is checked against the complete checkpointed-BF16 training
step, not isolated expert GEMMs. A missed target is reported as a miss.

`benchmark_balanced_router.py` measures warmed CUDA-graph phase timings and
whole forward/backward, optionally running bounded shared tuning with
`--tune quick|balanced|thorough|extensive`. `qualify_balanced_router.py` performs
alternating same-input A/B qualification of the resulting configs, retaining
unstable results rather than promoting them. See
[the router guide](../docs/balanced_router.md). Optional CUDA/CUB reference
sources are isolated in [routing_reference](routing_reference/README.md), never
imported by production RTX.

`compare_balanced_router_torch.py` compares against BiMoE's unchanged standalone
`router.py`, selecting its Torch Sinkhorn5/QB5/fixed-repair64 path. Both sides
use fullgraph compilation and CUDA graph replay; inference and training have
separate bounded autotuning contexts. Exact metadata and every gradient are
checked before alternating measurements. For example:

```bash
PYTHONPATH=. python benchmarks/compare_balanced_router_torch.py \
  --baseline-source /path/to/BiMoE/bidirectional_mixture_of_experts/router.py \
  --output autotune_reports/balanced_router_vs_torch/results.json
```

Add `--cases benchmarks/balanced_moe_cases.json` for the initial 12-case
BalancedMoE suite. This driver measures routing only, not shared/expert FFNs.
## Packed independent BalancedMoE

`benchmark_moe_varlen.py` measures routing subparts and full forward/backward
on packed sequences. The two BF16 comparisons keep the same native router but
separately checkpoint or save FFN intermediates. Compilation, tuning and warmup
are excluded; phase measurements are standalone and must not be summed.

```bash
python benchmarks/benchmark_moe_varlen.py --source . \
  --output autotune_reports/moe_varlen/run.json \
  --case-id d384_e24_k6_h128 --lengths 512 2000 768 4096 \
  --tune-router-trials 24 --tune-expert-trials 16 --tune-shared-trials 16 \
  --tune-seconds 120 --oracle --rounds 3 --samples 9
```

Every run writes frozen `.router.json`, `.expert.json` and `.shared.json`
configurations. Reuse them with `--router-config`, `--expert-config` and
`--shared-config` for controlled comparisons in a fresh process. `--oracle`
checks all routed-expert outputs/gradients against independent quantized FP64
math on the actual assignment. `--profile torch --profile-phase mxfp8_step`
exports a warmed CUDA/CPU timeline in a separate diagnostic process; ordinary
CUDA-graph event timings, not instrumented durations, are the performance data.

`tune_moe_varlen_matrix.py` runs all four designs in balanced length waves,
one fresh process and independent result/residual files per context. It can
resume completed contexts with the same source, baseline and sufficient
budget. Seeds are starting candidates, never fixed winners. Run detached:

```bash
RTX_PYTHON=.venv/bin/python bash benchmarks/launch_moe_varlen_matrix.sh \
  --seeds <directory-with-delivery-config-sidecars> \
  --output autotune_reports/moe_varlen_matrix \
  --length-scales 4 1 --expert-trials 96 --router-trials 16 \
  --shared-trials 32 --tune-seconds 180 --max-autotune
```

The launcher uses `nohup`, disconnects stdin, disowns its study, and prints
the PID/log. `summary.json` records target misses honestly; a successful
process exit is not a speedup qualification. `--max-autotune` enables
Inductor candidate subprocesses to contain faulty compiler-generated
benchmark choices rather than poisoning the training-comparison process.

For a routed-width ablation, add `--expert-hidden-scale 1.5 --hidden-alignment 64`.
This tests H=192/192/192/384 without changing shared-expert widths. Effective
case IDs include the actual H, preventing collisions with original results.
The benchmark replays the original initialization before creating wider routed
weights, preserving X, dY, router parameters and shared parameters across H.
Use `--reference-matrix <completed-original-matrix> --router-trials 0
--shared-trials 0` to keep those unchanged components' selected schedules
fixed while retuning the wider routed experts. The H=128-only fused on-chip
engine is not eligible at these wider H values; staged/prepared choices
remain available. No kernel legality or runtime dispatch rule is changed by
these benchmark options.

### Composed MoE improvements

`benchmark_moe_composition.py` measures baseline, routing-only, recompute-only
and combined schedules in a single matched factorial experiment. It consumes
a frozen `benchmark_moe_varlen.py` result and uses the normal `BalancedMoE`
frontend with user-side fullgraph compilation and outer graph replay:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.benchmark_moe_composition \
  --baseline-result <frozen-result.json> --output <composition.json>
```

Use `--lengths 255 0 129` for a matched ragged control. For statically
ineligible persistent pools it measures only baseline/recompute. It checks
hard assignments, every output/gradient, repeated replay and live allocations;
reports raw balanced-order visits and the measured interaction term; and
never installs a winner. It does not add standalone component times or claim
an optimizer-step/BF16 speedup. Production still selects each legal candidate
through the shared bounded autotuner and source/device-specific cache.
