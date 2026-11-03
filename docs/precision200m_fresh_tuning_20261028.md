# Batch-aware fresh tuning for the 206M SiTU decoder

The earlier batch-16 comparison used fixed kernels. Its timings are not a
comparison of the best available configurations. This study retains its
numerical recipe and replaces that performance protocol.

## Outer batch search

`benchmarks.probe_precision200m_batch` runs each batch in a fresh process with
the actual compiled model, gradients, optimizer state, validation and CUDA
graphs. It searches powers of two, then refines at eight-sequence granularity,
keeping 5% of VRAM free according to peak reserved memory. Non-OOM failures
stop the probe; they are not treated as capacity limits.

On the home RTX 5090s, the seed schedules qualified batch 72 for NVFP4-forward /
MXFP8-backward (80 OOM), and batch 80 for full NVFP4 (88 OOM). These are the
largest **tested safe** batches at the declared granularity, not proof that
every intervening integer batch is impossible. The common batch candidate is
72; batch 48 is the smaller throughput comparison. Tuning can change workspace
requirements, so neither is accepted without a full-model tuned pilot.

## Inner schedule search

`benchmarks.study_precision200m_tuned` runs fresh searches at both batches:

- Four projections: `(M,3072,1024)`, `(M,1024,1024)`, `(M,5632,1024)`,
  `(M,1024,2816)`, where `M = batch * 512`.
- Shared NVFP4 forward, complete MXFP8 backward, and complete experimental
  NVFP4 backward. No shape is silently left at a historical winner.
- 32 trials / 300 seconds per context, whichever budget is exhausted first;
  actual trial counts and all failures remain in residual journals.
- Component searches use the library's composable bandit/local/model-guided
  machinery, with history transfer and pretrained retrieval disabled. New
  output directories are required. Batch, sequence and model identity accompany
  the forward/MXFP8 contexts; NVFP4 design searches are isolated under the batch
  directory with exact GEMM shape and device/revision context.
- FP32 exact outer scaling and regional geometry are fixed numerical policy;
  the shared forward config is identical in both training arms. Search varies
  implementation/scheduling, not the accuracy recipe.
- NVFP4 backward selects a compatible implementation across all projections
  and independently confirms it before model timing. Any failed correctness or
  timing gate stops the workflow; no unstable winner is automatically accepted.

The forward harness measures complete materialized quantization plus GEMM;
MXFP8 backward uses CUDA-graph timing, and experimental NVFP4 backward uses its
full compiled comparison. The **compiled model pilot** is the final check of
how these choices perform together, rather than assuming kernel scores add up.

## Training and reporting

Both batches get a 64-update full-model pilot with selected explicit config
maps. Report memory and tokens/s for both. The largest common batch that passes
the workflow is selected as requested, even if the smaller batch is faster;
that throughput tradeoff must be shown, not hidden. A tuned-model capacity
failure currently stops for review rather than silently reducing the batch.

Final matched training starts from fresh seed-941 weights with approximately
100.7M byte tokens. Both arms use the same batch and updates. Warmup and
validation intervals scale approximately with token count. Different batch
sizes change optimization dynamics, so do not attribute differences against
the old batch-16 study solely to precision or kernel tuning.

Search, compilation and qualification run before training, in separate
processes. `preparation.json` records preparation wall time separately.
Training-block time includes model forward/backward, clipping, input transfer
and AdamW, but excludes tuning, compilation, validation and checkpoints.
Total training-process wall is reported separately. The training launcher
uses explicit config maps with `autotune="off"`; there is no hidden first-hit
search inside the measured steps. Config maps are part of checkpoint identity.

Raw capacity evidence: `autotune_reports/batch200m_*_20261028/`.
Fresh campaign: `autotune_reports/precision200m_fresh_tuned_20261028/`.
Source snapshot: `/tmp/rtx-fresh-study-20261028.aMsLmq`.
The two initial component smoke timings were ineligible due to clock drift;
they remain preserved. Workload warmup and longer samples were added without
relaxing timing gates.
