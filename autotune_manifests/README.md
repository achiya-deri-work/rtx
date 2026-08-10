# Autotuning manifests

Manifests are immutable experiment specifications. Never edit a manifest after
collecting data with it: its normalized contents are part of every tuning
context identity. Add a newly versioned file instead.

## Current campaigns

| Manifest | Purpose | Contexts |
| --- | --- | ---: |
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
