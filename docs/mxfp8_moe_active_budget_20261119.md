# Active-budget BalancedMoE profiling — 2026-11-19

Status: the original 54-context HOME GPU0 study is complete. All 54 contexts
passed the independent quantized oracle, gradient replay and live-allocation
checks. The 48-context K4/joint-fraction follow-up also completed successfully
on HOME GPU0 and was collected on HOME. See the final section and the
[102-context consolidation](mxfp8_moe_iteration5_profile_20261119.md).
WORK's original main shard stopped at a timing-eligibility
failure and is excluded from controlled performance conclusions. No production
kernel, dispatch heuristic, or API change is made by these studies.

## Architecture contract

The active per-token hidden budget is `S + K*H`, targeting `4*D` with a **hard
±7.5%** bound. Both `S` (one shared SwiGLU) and `H` (each routed expert) must
be multiples of 64. Total expert count `E` is a multiple of eight and does not
change active compute per token.

For each requested shared fraction, independently round nominal shared and
expert widths to nearest 64, with ties upward. Record the effective fraction;
do not silently adjust an omitted cell to a different shared fraction or
relax the expansion bound. Four of twelve requested geometry cells survive:

| D | Target shared fraction | K | Shared S | Expert H | Active hidden | Expansion | Effective shared fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 384 | 1/4 | 6 | 384 | 192 | 1536 | 4.00× | 25.00% |
| 512 | 1/4 | 6 | 512 | 256 | 2048 | 4.00× | 25.00% |
| 512 | 1/4 | 8 | 512 | 192 | 2048 | 4.00× | 25.00% |
| 512 | 1/8 | 6 | 256 | 320 | 2176 | 4.25× | 11.76% |

All nearest-rounded 1/6 cells and all D384/K8 fraction cells are omitted.
The complete proposed/omitted geometry records are in each manifest; this is
an architecture selection rule, not a restriction in the kernel templates.

## Coverage and controls

`benchmarks/profile_moe_active_budget.py` generates a bounded 54-context
stratified matrix, not an exhaustive Cartesian product:

- All four geometries at E=16/32/64/128 and B=16, length=1024.
- Eight B=8/length=1024 pilot contexts spanning both widths/fanouts/layouts.
- Two geometry anchors at B=8/32 and length=1024.
- Equal-total-token B=16/length=2048 and B=8/length=4096 controls.
- Uneven independent lengths `[1024,1536,2048,3584]` repeated four times.

`packed` means independently balanced sequence pools. `flat` is the actual
regular `[T,D]` API with `cu_seqlens=None`, **one global pool**, not a padded
batch and not a loop over individual samples. Its B/length labels describe
the source partition only. These layouts intentionally have different routing
semantics; flat duplicate total-token contexts are removed.

Whole-layer controls use the same parameters, inputs, upstream gradients and
router: aggressively compiled BF16 with saved intermediates, checkpointed
native expert layout, and checkpointed compact expert layout including the
compaction cost. MXFP8 recomputes expert intermediates. Standalone measurements
include router, frozen-route experts and shared MLP, with matched checkpointed
BF16 expert/shared controls. Standalone phase timings overlap; **never add
them to predict whole-step time**.

## Measurement and validation

- Fresh bounded searches per context: 8 expert, 8 shared, 4 router trials.
- Fullgraph/static `torch.compile`, aggressive supported options, no freezing;
  outer CUDA graph replay for steady-state timing.
- Compilation, search, setup, forward preparation for backward-only capture,
  and warmup excluded. Two rotated phase rounds, seven ordinary CUDA-event
  batches per phase; median, minimum and relative MAD retained.
- Backward-only measurement uses a real forward outside capture, pins saved
  input storage, and captures normal `retain_graph=False` backward. It does
  not subtract forward from step latency or globally disable donated buffers.
- Independent FP64 operational quantized expert forward/backward oracle on
  actual routes, every public output/gradient finite, and fresh-versus-replay
  gradient checks. Oracle gates are unchanged: output NRMS 0.003, gradient
  NRMS 0.006, with absolute protection at 2e-7 for near-zero references.
- Post-warmup synchronized live-allocation growth must be zero. Reported
  peak allocation is the warmed benchmark-process footprint, **not** an
  isolated per-policy VRAM comparison or measured memory traffic.
- Resume validates runtime/driver identity, architecture, layout, lengths,
  initialization policy, tuning budgets, required timing samples and gates.
  Atomic per-context results and fsynced progress JSONL preserve partial work.

One initial pilot failed because the old backward-only harness requested
`retain_graph=True`, incompatible with Inductor donated buffers. The failed
artifact is retained; this was a measurement-harness failure, not a CUDA fault.
The corrected packed and flat pilots passed on both hosts.

## Initial HOME pilot, not the full sweep

D384, S384, H192, K6, E16, T8192; 8×1024 independent packed samples versus
one regular flat pool. Medians across the two rounds:

| Measurement | Packed | Flat |
| --- | --- | --- |
| MXFP8 whole step | 1.394 ms | 2.735 ms |
| BF16 compact/checkpoint step | 1.384 ms | 2.739 ms |
| Standalone router step | 0.479 ms | 1.662 ms |
| Frozen-route MXFP8 expert step | 0.716 ms | 0.625 ms |
| Shared MXFP8 MLP step | 0.172 ms | 0.172 ms |

There is no meaningful whole-step speedup here. The flat slowdown accompanies
larger router cost despite slightly faster expert work; it is not evidence of
tensor-core saturation. Packed routing and global routing solve different
balancing problems. The independent quantized oracle's worst NRMS was below
0.0001 in both pilots; backward-only and whole-step MXFP8 replay gradients
matched fresh execution exactly; synchronized live growth was zero.

## Reproduction and evidence

```bash
python -m benchmarks.profile_moe_active_budget list
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.profile_moe_active_budget run \
  --source "$PWD" --output /tmp/moe-active-budget --max-seconds 5400
python -m benchmarks.profile_moe_active_budget summarize \
  --output /tmp/moe-active-budget
```

For two isolated hosts, add `--shards 2 --shard 0` on HOME and `--shard 1`
on WORK. Geometry groups stay on one host so expert-count/layout controls
are not confounded by device clock or platform differences. Each manifest
contains source/driver hashes, exact commands, actual widths and omissions.
Summaries are exported to JSON/CSV/Parquet.

During this run, WORK rejected all four flat router candidates as
`timing_ineligible` and raised `NoEligibleTimingError`. A subsequent read-only
GPU check showed other Python/Triton-server processes holding 6.8 GiB and
nonzero utilization. This does not prove the sole cause of instability, nor
establish when interference began. WORK timings are retained as diagnostic
evidence, not clean paired performance results. No service was stopped. Its
evidence was retrieved and its one persistent SSH session closed. The second
architecture shard is queued on HOME GPU0 and starts only if the first shard
finishes successfully; a fault cannot silently launch the follow-on shard.

Current local staging: `/tmp/rtx-moe-active-budget-20261119/`.
Preserved pilot/source/environment evidence and an explicitly partial main
summary: `autotune_reports/mxfp8_moe_active_budget_20261119/`, using the existing
Storage4-backed artifact location. Active HOME records are in `home-study/`
and `home-second-shard/` under staging, with independent manifests, progress
logs and atomic JSON/CSV/Parquet summaries.
HOME uses GPU0 only; GPU1 is protected. WORK commands share a single
persistent SSH transport (now closed after retrieval). Profilers are not injected during ordinary timing.
Nsight Python is absent locally, and WORK has no Nsight CLI tools available;
any later HOME CLI capture is separate diagnostic evidence, not a replacement
for ordinary timing or a claim that the full counter matrix was collected.

CPU study-tooling/reference regression tests: 24 passed. This is bounded
profiling qualification, not a new release-wide correctness/convergence claim.

## Joint-fraction and K4 follow-up

The user approved joint architecture selection instead of independently
rounding the nominal shared/expert widths. Under the same hard 64-alignment
and ±7.5% active-budget constraints, enumerate legal `(S,H)` pairs, minimize
absolute error in `S/(S+K*H)` against the requested fraction, then break ties
by minimum error against `4D`. This is architecture selection, not manual
kernel schedule tuning. Legacy geometry generation remains unchanged for
reproducibility; select the new wave with `--study fanout`.

The primary new D384/K4 family holds routed H=320 fixed and sweeps shared
S=192/256/320. Active widths are 1472/1536/1600 (−4.17%/0%/+4.17%); actual
shared fractions are 13.04%/16.67%/20.00%. The exact 1/6 member is S256/H320.
S320/H320 is the nearest legal quarter-shared approximation, **not** 25%.

Additional joint selections:

| D | K | Requested fraction | S | H | Actual shared | Active-budget error |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 4 | 1/6 | 384 | 448 | 17.65% | +6.25% |
| 512 | 4 | 1/4 | 512 | 384 | 25.00% | 0% |
| 384 | 6 | 1/6 | 320 | 192 | 21.74% | −4.17% |
| 512 | 6 | 1/6 | 384 | 256 | 20.00% | −6.25% |
| 512 | 8 | 1/6 | 384 | 192 | 20.00% | −6.25% |

The 48 contexts cover the new D384/K4 family at E16/32/64/128, nearest-fraction
controls, selected flat layouts, B8/B16/B32 and long/uneven sample lengths.
The exact-quarter D512 trio K4/E32, K6/E48, K8/E64 holds active hidden width
2048, shared width 512, and mean assignments per expert fixed. These are
architecture comparisons, not a claim that varying K leaves routing identical.

For this follow-up, named per-projection expert RNG streams also hold routed
weights fixed when S changes. Inputs/upstream/router/shared streams retain
their explicit controls; weights remain BF16 with 0.02 truncated-normal
initialization. This change is confined to benchmark setup, not the library
frontend or timed launch path. Raw identities permit verification.

The production runtime digest is unchanged from the original study:
`a5e83c8eb4400c824105bc15a3a710f7c76e1cf13eb3d62a7d84ca80ffd2baf8`.
All controls, search budgets, numerical gates and ordinary graph timing are
retained; 26 tooling/reference tests passed before launch.

WORK was checked through one persistent SSH connection and was serving other
processes (~8.7 GiB allocated, 23% utilization). The user explicitly chose to
leave its services untouched and use HOME rank 0 only. The unused transport
was closed; no new WORK performance results will be mixed into this wave.

Completed run logs: `/tmp/rtx-moe-k4-20261119/launch.log` and `study/progress.jsonl`.
The detached transient user job `rtx-moe-k4-20261119.service` exited successfully
after all 48 contexts and artifact export (both exit codes zero). Rank 1 and
all other services were untouched. It used
the immutable `source/` snapshot, not the evolving worktree. The launcher
`benchmarks/run_moe_active_budget_study.sh` preserves partial results on a
normal benchmark failure and, on exit, collects source/environment/raw results
and JSON/CSV/Parquet summaries into
`autotune_reports/mxfp8_moe_k4_fraction_20261119/` on HOME's Storage4-backed
artifact location. There are no new remote results to retrieve in this
HOME-only wave.

```bash
python -m benchmarks.profile_moe_active_budget list --study fanout
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.profile_moe_active_budget run \
  --study fanout --source "$PWD" --output /tmp/moe-fanout --max-seconds 8400
```

The original complete 54-context evidence is preserved separately in
`autotune_reports/mxfp8_moe_active_budget_20261119/home-complete.tar.gz`.

Initial follow-up qualification: D384/K4/S256/H320/E32, B16×1024 packed
measured 2.342 ms whole step, 1.044× compact-checkpointed BF16 and 0.997×
saved-intermediate BF16. The native-capacity checkpoint control is 1.176×;
do not headline that weaker baseline. Independent quantized oracle worst
NRMS was 0.0000726, replay-gradient checks passed, and live allocation growth
was zero. This is one pilot, not the final fanout/fraction sweep conclusion.
