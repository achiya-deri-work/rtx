# Matched 21M transformer training: 2026-10-25

All six runs completed: three precision policies on each laptop. Full NVFP4
has slightly lower held-out loss at equal steps in this seed, but takes longer
and loses that advantage at equal training-wall budgets. This is early training,
not a multi-seed convergence claim or an exhaustively tuned throughput result.

## Protocol

- 21,121,024 parameters; 8 layers, width 512, SwiGLU width 1024, 8 heads.
- TinyStories byte-level LM, vocabulary 257, tied embeddings/head; sequence 256,
  batch 16, 2,048 optimizer steps: 8,388,608 byte tokens per run.
- Seed 941; identical CPU-initialized weights, corpus prefix and held-out
  offsets verified by hashes across all six runs. Train/validation byte ranges
  are disjoint. Validation uses the same 32 batches every 128 updates.
- Only the 32 residual-branch projections change precision. BF16 embeddings,
  head and activation boundaries; FP32 norms/normalization parameters, RoPE,
  residual additions before BF16 storage, loss and gradient-norm reduction.
- Native FLASH attention is forced, with BF16 QKV and FP32 softmax arithmetic;
  see the pinned-source audit and GPU smoke in [the recipe](precision20m_recipe.md).
- BF16 AdamW states for BF16 parameters; FP32 states for normalization parameters.
  Learning rate warms for 128 steps to 3e-4, then cosine-decays to 3e-5. QK and
  pre/post norms, 0.02 truncated-normal initialization and depth-scaled residual
  projections are preserved.
- Both NVFP4 policies use explicit exact-FP32 JIT-region forward. Full NVFP4
  uses experimental MS-EDEN backward from original BF16 operands and the wider
  coalesced preparation seed. MXFP8 backward uses the stable paired operation.
- Fullgraph/static `torch.compile`, CUDA graph replay enabled, fused AdamW,
  block-synchronized wall measurements. No optimizer updates during compile
  warmup; warmup and validation RNG state is restored.
- Fixed recorded schedules, no fresh exhaustive autotuning. Runs are fresh
  processes; existing compiler caches are allowed. 5060 order: MXFP8, hybrid,
  full NVFP4. 5070 order: reverse. This reduces order bias but is not repeated
  randomized timing or a controlled thermal study.

## Equal work: 2,048 steps

Loss is fixed-held-out cross entropy in **nats per byte**, not subword-token
perplexity. Training wall includes input preparation/transfers, forward,
backward, FP32 clipping and optimizer updates. It excludes compilation,
validation, checkpoint writes, logging and outside-block finite/dtype audits.

| GPU | Forward / backward | Validation loss | Delta vs MXFP8 | Training wall | Inclusive run wall |
|---|---|---:|---:|---:|---:|
| 5060 | MXFP8 / MXFP8 | 0.761809 | — | 39.60 s | 60.75 s |
| 5060 | NVFP4 / MXFP8 | 0.767338 | +0.005529 | 38.34 s | 55.61 s |
| 5060 | NVFP4 / NVFP4 | 0.751993 | -0.009816 | 50.95 s | 74.15 s |
| 5070 | MXFP8 / MXFP8 | 0.761822 | — | 43.19 s | 68.54 s |
| 5070 | NVFP4 / MXFP8 | 0.762087 | +0.000264 | 41.17 s | 63.98 s |
| 5070 | NVFP4 / NVFP4 | 0.759456 | -0.002367 | 50.44 s | 78.76 s |

Inclusive run wall starts at workload initialization and includes compile,
validation, audits, logging and final checkpoint; it excludes interpreter/import
startup. The two laptops are not pooled as timing replicates. Full NVFP4's
observed training wall is 29%/17% longer than MXFP8 on the 5060/5070. Hybrid is
about 3–5% faster in these executions; that small margin needs repetitions
before claiming a stable hardware-wide speedup.

## Equal training-wall budgets

Use the last **actually evaluated** checkpoint at or before the shared budget,
not interpolated losses. Checkpoint ages are retained in JSON. This conservatively
answers what each completed run had demonstrably achieved within the budget;
it is not a validation measurement at an identical fractional instant.

| GPU / shared budget | MXFP8 | NVFP4 / MXFP8 | Full NVFP4 |
|---|---:|---:|---:|
| 5060 / 38.343 s | 0.763654, step 1920 | 0.767338, step 2048 | 0.778241, step 1536 |
| 5070 / 41.169 s | 0.763192, step 1920 | 0.762087, step 2048 | 0.772934, step 1664 |

The full-NVFP4 wall-adjusted losses are higher by 0.014587 and 0.009742 versus
MXFP8 at those observed checkpoints. Matched inclusive-wall comparisons,
observed time-to-loss crossings and every validation checkpoint are also in
the analysis JSON. No crossing beyond the observed trajectory is extrapolated.

## Qualification and artifacts

Six full-size two-step smoke runs passed before the six longer runs. All longer
runs completed their planned step/token counts with finite losses, gradients
and final parameters; optimizer-state dtypes were audited after every block.
The analyzer rejects mismatched initialization/data/recipe/attention, incomplete
runs, missing terminal evidence and inconsistent budgets by default.

Local ignored artifacts:

- `autotune_reports/precision20m_20261025/{5060,5070}/run2048/`: raw JSONL,
  per-policy logs and launch receipts, including source hashes and configurations.
- `autotune_reports/precision20m_20261025/analysis2048_final.json`: validated
  provenance, all checkpoints, block-throughput samples and both wall clocks.
- `autotune_reports/precision20m_20261025/curves2048.svg`: actual validation
  checkpoints versus steps and training wall. Step zero is excluded from the
  zoom; connecting lines are visual guides, not inferred observations.

Remote snapshots/checkpoints remain at:

- 5060: `/tmp/rtx-precision20m-20261025.HIIlhPGP/run2048/`
- 5070: `/tmp/rtx-precision20m-20261025.x3oUXo4a/run2048/`

No default precision policy was changed. A roughly 100M-byte-token follow-up
was offered separately; it has not been launched without the user's choice.
It should start from matched initialization with a newly fixed longer learning-
rate schedule, not silently stretch these already-decayed runs.
