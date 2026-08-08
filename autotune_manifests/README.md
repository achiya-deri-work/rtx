# Autotuning manifests

Manifests are immutable experiment specifications. Never edit a manifest after
collecting data with it: its normalized contents are part of every tuning
context identity. Add a newly versioned file instead.

## Current campaigns

| Manifest | Purpose | Contexts |
| --- | --- | ---: |
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
