# Autotuning manifests

Manifests are immutable experiment specifications. Never edit a manifest after
collecting data with it: its normalized contents are part of every tuning
context identity. Add a newly versioned file instead.

## Current campaigns

| Manifest | Purpose | Contexts |
| --- | --- | ---: |
| `blackwell_all_kernels_scaling_v1.json` | Paired release study across every MXFP8/NVFP4 forward, shared backward, scaling, cache, and packed-inference family | 144 |
| `decoder_batch64_mxfp8_v1.json` | Batch-64 decoder MXFP8 fused-forward/shared-backward tuning | 8 |
| `decoder_batch64_nvfp4_v1.json` | Batch-64 decoder delayed-fused and block-materialized NVFP4 tuning | 8 |
| `nvfp4_jit_row_region_v1.json` | Current local FP32 row-region scale geometry plus NVFP4 quantizer/GEMM tuning | 16 |
| `decoder_training_hot_v1.json` | Exact decoder-shape NVFP4 dynamic-forward and MXFP8 backward training study | 8 |
| `decoder_mxfp8_forward_hot_v1.json` | Exact decoder-shape materialized MXFP8 forward study | 4 |
| `blackwell_release_candidate_5070_v1.json` | Current-revision MXFP8/NVFP4 production calibration across every operand state on both 5070 variants | 108 |
| `nvfp4_dynamic_persistent_shapes_v1.json` | Bounded revision-3 anchor/generalization study at M=128/512/8192 | 3 |
| `nvfp4_dynamic_balanced_iteration_v5.json` | Corrected cap-four balanced-grid confirmation at M=N=K=1536 | 1 |
| `nvfp4_dynamic_balanced_iteration_v4.json` | Immutable cap-two anchor control which preserves the 72-CTA tail | 1 |
| `nvfp4_dynamic_balanced_iteration_v3.json` | Immutable unanchored revision-3 search-policy control | 1 |
| `nvfp4_dynamic_block_iteration_v2_128.json` | Bounded 128-trial joint dynamic block-quantizer/native-GEMM study | 2 |
| `nvfp4_dynamic_components_iteration_v1.json` | Bounded packed-GEMM and dynamic-X component study | 4 |
| `nvfp4_inference_states_v1.json` | NVFP4 dynamic-X/AOT-W and fully packed inference tuning | 16 |
| `nvfp4_training_v1.json` | NVFP4 single-launch training-forward tuning, including per-CTA delayed-scale telemetry | 16 |
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

The prospective 5070 manifest expands every base workload across three search
treatments and two independent replicates. Its order balances every prefix
across kernel family, shape category, hot/rotating inputs, treatment, and
replicate. Each context writes its own residual journal. Transfer learning is
allowed between shapes inside one treatment/replicate, but never across
experimental arms.

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
