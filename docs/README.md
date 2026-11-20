# RTX Blackwell documentation

Install **rtx-blackwell**, import **rtx**. Bias-free MXFP8/NVFP4 linear layers
for RTX Blackwell, with a reusable GPU autotuner. Dynamic Torch-side inputs,
parameters, outputs and gradients are BF16; NVFP4 uses MXFP8 backward.

## User manual

Start with [installation](installation.md) and [quickstart](quickstart.md).
The [API map](api.md) owns public names and shared frontend controls;
the [generated reference](api_reference.md) owns exact signatures and defaults.

| Task | Canonical guide |
|---|---|
| Use the linear layers | [MXFP8Linear](mxfp8_linear.md) · [NVFP4Linear](nvfp4_linear.md) |
| Keep QKV/expert optimizer matrices separate but physically packed | [Virtual layers](virtual_layers.md) |
| Convert a model for training or PTQ | [Model conversion](model_conversion.md) |
| Calibrate NVFP4 and select numerical policies | [Quantization policy](quantization_policy.md) |
| Review measured NVFP4 strengths, weaknesses and external recipes | [Broad evaluation, 2026-10-23](nvfp4_broad_evaluation_20261023.md) |
| Experiment with MS-EDEN backward and structural tuning | [Consolidated research](nvfp4_research.md) · [Search architecture](ms_eden_composable_search.md) |
| Choose dynamic, packed-weight or fully packed inference | [Inference](inference.md) |
| Understand precision | [Training](training.md) · [Formats](formats.md) |
| Configure effort and cache reuse/refinement | [First-hit autotuning](first_hit_autotuning.md) |
| Compile and use graphs, streams and caches | [Execution](execution.md) |
| Retire warmup scratch and share reduction workspaces safely | [Execution resources](execution_resources.md) |
| Collect datasets and deploy winners | [Autotuning operations](autotuning.md) |
| Understand measured finite-budget MoE tuning gains | [Execution-region opening](autotune_region_opening_20261120.md) |
| Understand deeper 32-trial search and remaining headroom | [Astra investigation](astra32_autotuning_findings_20261120.md) |
| Use the integrated flat-router opening and coupled refinement | [Autotuner follow-up cycle](autotune_cycle3_20261120.md) |
| Compare whole-MoE MXFP8 against strong saved/checkpointed BF16 | [Broad 5090 precision matrix](mxfp8_bf16_broad_20261120.md) |
| Tune long-token weight gradients | [Balanced backward](balanced_backward.md) |
| Diagnose errors and limitations | [Troubleshooting](troubleshooting.md) · [Runtime contract](runtime_contract.md) |

Linears replace **bias-free** nn.Linear under the documented dtype/device
constraints, not every PyTorch/distributed configuration. Calling eval() does
not pack weights. Operand state, numerical policy, backend and tuning effort
are separate choices.

## Feature APIs

- [MXFP8GLU](mxfp8_glu.md): paired projection and activation, BF16/MXFP8 output,
  forward only.
- [MXFP8MLP](mxfp8_mlp.md): packed hidden connection with recomputed backward.
- [NVFP4MLP](nvfp4_mlp.md): on-chip FP32 SiTU, NVFP4 forward and recomputed
  MXFP8 backward; packed MXFP8 hidden storage by default, BF16 optional.
- [BalancedRouter](balanced_router.md): experimental fixed-K/equal-capacity
  routing; CuTe saved-state Sinkhorn and detached GPU QB/repair, with bounded
  shared first-hit autotuning.
- [BalancedMoE](balanced_moe.md): experimental MXFP8 routed/shared SwiGLU,
  prepared/staged/on-chip expert families and recomputed backward; performance work remains.

The GLU/MLP APIs default to tuning off, with opt-in first-hit selection. Feature APIs remain explicit
imports, outside the stable wildcard surface. The optional packed-down MLP
singleton failure is recorded in [numerical qualification](mxfp8_mlp_numerics.md).

## Development and evidence

- [Two-HOME-GPU MoE heuristics](mxfp8_moe_heuristics_20261120.md): current-source
  32-trial discovery/holdout matrix, controlled schedule panels, routing versus
  operand movement, actual counters and isolated packed-metadata tail fix.
- [MoE iteration 6](mxfp8_moe_iteration6_20261119.md): bounded recompute/router
  experiments, rejected held-out hard-routing changes, and default-GC CUDA
  graph/scratch lifetime qualification on HOME rank 0.
- [MoE iteration 5](mxfp8_moe_iteration5_20261119.md): consolidation of 102 completed
  contexts, Astra review, three small-budget search improvements, and paired
  packed/flat whole-layer qualification on HOME rank 0.
- [Active-budget MoE profile](mxfp8_moe_active_budget_20261119.md): hard 4D budget,
  64-aligned feasible architectures, independent packed versus regular flat
  routing, B=8/16/32 scaling and matched BF16 controls; both studies complete.
- [MoE iteration 4](mxfp8_moe_iteration4_20261119.md): bounded producers and compact
  forward measured separately/combined, small shared forward coverage on WORK,
  actual eight-trial composition and retained BF16 gaps.
- [MoE iteration 3](mxfp8_moe_iteration3_20261119.md): protected first-hit router
  coverage, composable prefix/fragment/persistent backward, optional token-owned
  reductions, actual eight-trial selection and stronger BF16 comparisons.
- [45-context balanced MoE challenge profile](balanced_moe_challenge_20261119.md):
  two 5090 hosts, shape/input/packing sweeps, finite-budget controls, conditional
  graph tracing, scalar-gradient caveats and a stronger compacted BF16 reference.
- [MoE iteration 2](mxfp8_moe_iteration2_20261119.md): additive fragment-owned
  GLU and paired weight preparation, rejected split-warp control, and a fixed
  half-aligned shared-MLP first-hit coverage gap with actual 8/16-trial results.
- [Routed backward ownership](mxfp8_moe_backward_ownership_20261119.md):
  retained dW panels, independent route-adjoint loads/tiles, additive whole-layer
  gains, rejected indexed quantization and actual eight-to-sixteen trial evidence.
- [Hybrid output stores and 8/16/32-trial qualification](mxfp8_hybrid_tails_20261119.md):
  interior-only TMA, masked boundary stores, measured L2 effects, paired training
  correctness, three-SKU checks and the remaining aggressive-BF16 whole-MoE gap.
- [Eight-trial MoE store prior](mxfp8_moe_eight_trial_prior_20261119.md):
  legal compound output-store/geometry/buffering proposals, unchanged numerical
  policies, real eight-trial held-out searches and sixteen-trial controls.
- [Measured MXFP8 MoE causes](mxfp8_moe_measured_bottlenecks_20261119.md):
  matched BF16/FP8 counters, fourfold direct-store sector traffic, causal TMA
  controls and whole-layer gap attribution without production kernel changes.
- [MoE review execution](balanced_moe_priority_execution_20261118.md):
  early geometry/dY and shared preparation coverage, faster CPU planning,
  matched expert/shared factorial and retained negative fusion results.
- [Clean MoE gap and prior analysis](balanced_moe_gap_and_prior_20261118.md):
  finalized 1,024-row prefix, measured geometry starvation, bounded-search
  coverage correction and paired complete-layer qualification.
- [GPU0 fault correction](balanced_moe_gpu0_incident_20261118.md): VirtualLayer
  storage-span fix, fail-closed collection and quarantined study evidence.
- [Clean MoE recollection](balanced_moe_clean_recollection_20261118.md):
  pilot-first, per-visit audit gates and independent CSV/Parquet snapshots.
- [MoE search and kernel follow-up](balanced_moe_search_and_kernels_20261118.md):
  integrated affinity selection, short-reduction dW, per-pool routing controls,
  corrected MoE/device learning identities and resumable balanced collection.
- [Three-priority consolidation](balanced_moe_next_priorities_20261118.md):
  HOME GPU0 results, retained negative controls, composed-path correctness,
  actual bounded-search coverage and adjacent-kernel regression checks.
- [Small-pool Sinkhorn](balanced_moe_small_pool_sinkhorn_20261118.md),
  [joint recompute](balanced_moe_joint_recompute_20261118.md), and
  [packed active prefixes](balanced_moe_active_prefix_20261118.md): optional
  autotunable schedules, matched HOME/5070 checks, and retained negative controls.
- [MoE amortization scaling](balanced_moe_amortization_scaling_20261118.md):
  bounded token/width/hidden/expert/fan-out/packing sweeps, whole-layer versus
  frozen-route experts, hardware counters and remaining structural bottlenecks.
- [Packed MoE overhead optimizations](balanced_moe_amortization_5090_20261118.md):
  frozen-winner profiling, fused backward preparation, sharded routing,
  exact QB layout and controlled full-step comparisons.
- [Packed MoE current progress](balanced_moe_varlen_progress_20261118.md):
  aggressive BF16 comparison, remaining 1.2x misses, indexed/warp/split-dW
  alternatives, hybrid forward and search-coverage/startup fixes.
- [5090 whole-MoE and checkpoint comparison](balanced_moe_router_20261117.md):
  matched router/shared/expert attribution and saved versus checkpointed BF16 MLPs.
- [Matched whole-MoE recomputation qualification](balanced_moe_finish_20261117.md):
  checkpointed BF16 FFNs versus recomputed MXFP8, actual all-Parameter backward,
  exact routing identity, and fresh-gradient CUDA-graph replay; includes the
  independent packed-sequence four-design RTX 5090 matrix and remaining gaps.
- [BalancedMoE measured fixes](balanced_moe_fixes_20261117.md): compiler-visible
  expert storage, quantize-once input reuse and bounded compact-capacity search.
- [BalancedMoE profiling](balanced_moe_profile_20261117.md): per-launch timing,
  measured DRAM/L2/occupancy, token scaling, graph dispatch and pipeline A/Bs.
- [BalancedMoE speed study](balanced_moe_speedup_20261115.md): expert/whole-layer
  results, autotunable preparation/transport/output stores, and remaining BF16 gaps.

- [BalancedMoE design](balanced_moe_design.md): fixed-K/equal-capacity routing,
  packed expert banks, implemented MXFP8 families and remaining execution plan.
- [Consolidated NVFP4 backward/training research](nvfp4_research.md): retained
  experiments, model-context scheduling, training tools and qualification limits.
- [Development](development.md), [autotuner framework](../rtx/autotune/README.md),
  [manifest catalogue](../autotune_manifests/README.md), [benchmarks](../benchmarks/README.md).
- [Research and qualification index](research.md): experiments and historical
  results, separate from current API instructions.
- [Release checklist](release_checklist.md), [precision policy](../PRECISION_POLICY.md).
- [Current main qualification](main_qualification_status.md): completed evidence,
  approved fallback policy, and remaining integration gate.
- [MXFP8/GLU merge qualification](mxfp8_glu_merge_qualification.md): scheduling
  transfer and earlier cross-stack validation history.

Native kernels target SM120/SM121 on Linux with the supported CUDA 13.2 stack.
SM110/Thor and datacenter Blackwell are not supported. Import is CPU-safe;
native execution is CUDA-only. Source: [Apache-2.0](../LICENSE).
[SM120 notes](../sm120_hardware.md) credit Tri Dao's Quack project.
Optional [DINOv3 tests](dinov3_regression.md) need separately obtained assets.
The current primary application target is [Ettin](ettin_qualification.md);
its checkpoint is downloaded separately and is not included in RTX.
