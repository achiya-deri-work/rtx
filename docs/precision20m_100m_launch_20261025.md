# 100M-byte-token matched training follow-up

Launched 2026-10-25 at 12:28 UTC following user approval. Fresh initialization,
not continuation of the already-decayed 2,048-step runs. Same model, seed,
corpus, precision recipes and fixed schedules as
[the initial study](precision20m_results_20261025.md).

- Each policy: 24,576 updates, batch 16, sequence 256 = 100,663,296 byte tokens.
- LR warmup: 1,536 updates; original peak/minimum LR and cosine schedule.
- Fixed 32-batch validation every 512 updates, logging every 128,
  atomic checkpoint every 2,048. Compiled fullgraph with CUDA graph replay.
- This repeats samples from the same 10M-byte corpus prefix; it does not add
  100M unique training bytes. Train/validation regions remain disjoint.
- Modes: MXFP8/MXFP8, NVFP4/MXFP8, NVFP4/NVFP4 (experimental MS-EDEN).
- Separate detached `nohup`/`disown` launcher on each machine, opposite policy
  order; no concurrent training modes on one GPU.

## Artifacts

5060 (`personal_laptop`), launcher PID 10965:
`/tmp/rtx-precision20m-20261025.HIIlhPGP/run100m/`

5070 (`laptop_5070`), launcher PID 76546:
`/tmp/rtx-precision20m-20261025.x3oUXo4a/run100m/`

Each snapshot has `run100m-launch.log`; each mode has JSONL, stderr/stdout log
and checkpoint under `run100m/`. PIDs are launch receipts, not proof of current
process health. Both GPUs had no reported compute processes before launch.

Analyze completed runs with `benchmarks/analyze_precision20m.py`, preserving
equal-step and equal-training-wall comparisons. Do not present startup or
intermediate finite checks as completed convergence results.
