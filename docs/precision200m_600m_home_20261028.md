# Matched 600M-token decoder qualification

## Completed results (verified 2026-11-03)

Both final `result` records report 16,276 updates, **599,998,464 byte tokens**,
finite parameters, and `interrupted=false`.

| Board / backward | Training seconds | Total seconds | Final validation loss | Peak allocated bytes |
|---|---:|---:|---:|---:|
| GPU 0 / NVFP4 MS-EDEN | 3607.02 | 3757.66 | 0.40467 | 26,858,755,584 |
| GPU 1 / MXFP8 | 3413.83 | 3549.90 | 0.37308 | 27,875,715,584 |

Full NVFP4 is 0.03159 worse on this validation loss. This is a single-seed
quality observation, not evidence of general equivalence. The fixed board
assignment confounds a direct runtime comparison; use the earlier within-board
crossover for the scheduling/performance claim. PyTorch allocation counters
are not total device-memory accounting. The launch/monitoring details below
are retained as the historical recipe, not as a claim that these jobs still run.

Fresh runs launched on 2026-10-28 from the frozen, previously qualified source
snapshot `/tmp/rtx-model-refine-20261028.02EPio`. These are convergence checks
of the optimized backward schedule, not a new autotuning search.

| Physical GPU | Forward | Backward | Service |
|---|---|---|---|
| 0 | NVFP4 JIT regional | experimental NVFP4/MS-EDEN, `paired_dw_first` | `rtx-600m-gpu0-20261028` |
| 1 | same NVFP4 JIT regional | MXFP8 | `rtx-600m-gpu1-20261028` |

Artifacts:
`autotune_reports/precision200m_600m_home_20261028.A2AKmI/`.
Each arm has a frozen configuration manifest, training log, append-only JSONL
measurements and resumable checkpoints. The detached user services survive
terminal disconnection; they are not configured to survive a host reboot.

## Matched recipe

- 205,852,672 parameters: 16 layers, width 1024, FFN width 2816, 16 heads.
- SiTU MLP, QK normalization, pre/post sublayer normalization, tied byte-token
  embedding/output head. Existing initialization and precision recipe unchanged.
- Batch 72 × sequence 512 = 36,864 tokens/update; 16,276 updates give
  **599,998,464 training byte tokens per arm**.
- Same first 100M-byte TinyStories corpus selection, with the same fixed 90/10
  training/validation split. Training repeats the training portion: this is not
  600M unique tokens, nor 600M BPE tokens.
- Same initial parameter, corpus and validation-offset hashes, forward configs,
  seed 941, AdamW settings and source hashes (verified at startup).
- BF16 matrix parameters/gradients/optimizer moments and activation boundaries;
  FP32 normalization parameters, sensitive reductions, loss and clipping.
- LR 3e-4, 113 warmup updates, cosine decay over the new full run horizon.
- `torch.compile(fullgraph=True, dynamic=False)` with CUDA graph replay and the
  previously qualified pointwise/combo fusion options.
- Reuse the independently confirmed configs for these exact batch-72 shapes.
  No tuning is included in the training timer. Compile/warmup, validation,
  checkpointing and logging are also excluded from that timer; end-to-end wall
  time is recorded separately.
- Checkpoint every 2048 updates (75,497,472 tokens), plus the final update.
  Resume is supported only with a matching long-run recipe/checkpoint identity.

Both boards retain their existing 525 W limit. GPU 0 was approximately 5.6%
slower in the preceding within-board crossover study, so this fixed assignment
tests convergence but does **not** isolate backward-policy speed from hardware
differences. Use the preceding crossover measurements for that claim.

## Monitoring

```bash
systemctl --user status rtx-600m-gpu0-20261028 rtx-600m-gpu1-20261028
tail -f autotune_reports/precision200m_600m_home_20261028.A2AKmI/gpu0-nvfp4_full/training.jsonl
tail -f autotune_reports/precision200m_600m_home_20261028.A2AKmI/gpu1-nvfp4_mxfp8/training.jsonl
```

The approximate duration inferred from the preceding 100M-token runs is
60–65 minutes per arm, excluding exceptional compile stalls. A running service
is not itself evidence of successful completion: require a final `result`
record with the requested token count, finite parameters, and no interruption.
