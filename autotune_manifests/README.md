# Autotuning manifests

Manifests are immutable experiment specifications. Never edit a manifest after
collecting data with it: its normalized contents are part of every tuning
context identity. Add a newly versioned file instead.

## Current campaigns

| Manifest | Purpose | Contexts |
| --- | --- | ---: |
| `blackwell_diversity_atlas_v1.json` | Coverage-first matched atlas across 10 retained families, 30 stratified/ragged shapes, and both cache regimes | 600 |
| `blackwell_all_kernels_scaling_v2.json` | Retained-family release study across MXFP8/NVFP4 forward, shared backward, scaling, cache, and packed inference | 120 |
| `decoder_batch64_mxfp8_v1.json` | Batch-64 decoder MXFP8 fused-forward/shared-backward tuning | 8 |
| `decoder_batch64_nvfp4_v2.json` | Batch-64 decoder block-materialized NVFP4 tuning after fused-family retirement | 4 |
| `nvfp4_jit_row_region_v1.json` | Current local FP32 row-region scale geometry plus NVFP4 quantizer/GEMM tuning | 16 |
| `decoder_training_hot_v1.json` | Exact decoder-shape NVFP4 dynamic-forward and MXFP8 backward training study | 8 |
| `decoder_mxfp8_forward_hot_v1.json` | Exact decoder-shape materialized MXFP8 forward study | 4 |
| `blackwell_release_candidate_5070_v2.json` | Retained-family MXFP8/NVFP4 production calibration across every operand state on both 5070 variants | 96 |
| `nvfp4_dynamic_persistent_shapes_v1.json` | Bounded revision-3 anchor/generalization study at M=128/512/8192 | 3 |
| `nvfp4_dynamic_balanced_iteration_v5.json` | Corrected cap-four balanced-grid confirmation at M=N=K=1536 | 1 |
| `nvfp4_dynamic_balanced_iteration_v4.json` | Immutable cap-two anchor control which preserves the 72-CTA tail | 1 |
| `nvfp4_dynamic_balanced_iteration_v3.json` | Immutable unanchored revision-3 search-policy control | 1 |
| `nvfp4_dynamic_block_iteration_v2_128.json` | Bounded 128-trial joint dynamic block-quantizer/native-GEMM study | 2 |
| `nvfp4_dynamic_components_iteration_v1.json` | Bounded packed-GEMM and dynamic-X component study | 4 |
| `nvfp4_inference_states_v1.json` | NVFP4 dynamic-X/AOT-W and fully packed inference tuning | 16 |
| `mxfp8_bwd_revision19_calibration_v1.json` | Frozen-revision MXFP8 backward calibration and runtime-winner campaign | 12 |
| `autotuner_prospective_5070_v1.json` | Interruption-safe random vs. random+local vs. online-bandit study on the two 5070s | 144 residual units |
| `cross_device_dataset_bandit_v1.json` | Hierarchical strategy/context-bandit campaign for new cross-device measurements | 54 |
| `cross_device_dataset_v2.json` | Sequential-strategy control campaign and active resumable v2 dataset | 54 |
| `inference_states_pilot_v1.json` | Dynamic-X/AOT-W and fully prequantized inference-state pilot | 12 |
| `dataset_pilot.json` | Small CPU-contract/GPU-smoke campaign | 7 |

`cross_device_dataset_v1.json` is the first composable 54-context campaign and
is retained for dataset reproducibility. It should not be silently replaced by
v2 or the bandit campaign.

Validate a composable manifest without launching a kernel:

```bash
rtx-autotune validate autotune_manifests/cross_device_dataset_bandit_v1.json
```

After an anytime deadline, independently confirm the best candidates found at
any completed depth with `rtx-autotune verify-winners MANIFEST ...`. This is
required before promoting a winner discovered after the initial coverage wave.

The all-kernel/scaling campaign uses six identical shapes and both cache
regimes across all 12 production families. Its balanced-category rotation and
4/8/16/32/64/128 milestones make every interruption a useful paired dataset.
NVFP4 backward is represented by `mxfp8_bwd`, because that is the actual
backward implementation shared by every NVFP4 scaling policy.

The diversity atlas is the broad portable-model campaign. It covers micro and
under-wave expert shapes, vision and decoder projections, balanced and strongly
rectangular GEMMs, long sequences, tile boundaries, prime/ragged dimensions,
and packed-NVFP4 stride boundaries. Shared `shape_sets` and `job_defaults` keep
the source manifest reviewable; validation expands them into the same immutable
normalized manifest stored with every bundle. The launcher uses
1/2/4/8/16/32/64/128 breadth-first milestones so an interrupted run contains a
balanced screening dataset instead of a few deeply tuned early jobs:

```bash
./benchmarks/launch_diversity_atlas.sh 6h
tail -f autotune_logs/blackwell_diversity_atlas_v1_$(hostname -s).log
```

Run the full, unsharded atlas on different GPU SKUs. Matched contexts are what
allow a portable model to separate GPU effects from shape and kernel effects.
Only use `RTX_AUTOTUNE_SHARD_INDEX` and `RTX_AUTOTUNE_SHARD_COUNT` to divide
work among equivalent GPUs of the same SKU. Results are written as per-context
residual journals and finalized as CSV and Parquet. Environment overrides can
change visit duration and milestones, but a different experiment policy should
normally receive a new manifest version.

Prospective manifests expand every base workload across independent search
treatments. Their order balances every prefix across kernel family, shape
category, hot/rotating inputs, treatment, and replicate. Each context writes
its own residual journal. Transfer learning is allowed between shapes inside
one treatment/replicate, but never across experimental arms.

`blackwell_prospective_v3.json` is the strict held-out evaluation for portable
artifact `3317a6fef969ec5129c56732`. Its 12 shapes do not occur in the artifact's
training corpus. Four arms separate random search, online-only search, the
validated portable cost model, and cost model plus pairwise priors. This makes
the marginal value of each learned component identifiable.

## Legacy experiment-runner manifests

`blackwell_cross_device_v1.json` and `pilot_rigorous.json` use the older
single-family schema consumed by `rtx.prequant_experiments`. They remain in the
repository because existing experiment journals refer to their exact contents;
they are not accepted by `rtx-autotune run`.

## Naming rules

- Use a descriptive campaign name followed by `_vN`.
- Increment the version for any shape, policy, protocol, or tag change.
- Use `schema_version: 2` for new composable campaigns.
- Keep generated datasets, calibration files, logs, and winners outside this
  directory; those paths are ignored by Git.
