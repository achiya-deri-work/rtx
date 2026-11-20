# MXFP8 interior-TMA stores and bounded-search qualification

The common MXFP8 GEMM now has an autotunable **hybrid output epilogue**:
complete interior tiles use TMA; only boundary tiles use masked direct stores.
The ordinary shared MLP still processes flattened `[total_tokens, width]`.
It does not need sequence boundaries or a separate ragged shared-expert kernel.

This fixes an all-or-nothing store fallback, not quantization precision. It
adds no eager tensor arithmetic, padding allocation, user workspace, stream
management, or private per-invocation CUDA graph. FP32 accumulation and the
existing activation/quantization policies are unchanged.

## Runtime contract and search

`MXFP8GemmConfig.epilogue` accepts `direct`, `tma`, and `hybrid`.

- `direct`: masked stores for every tile; supports the existing general tails.
- `tma`: full-tile bulk stores; M and N must divide the selected CTA geometry.
- `hybrid`: bulk interior and masked boundary tiles; contiguous BF16 output
  rows must have a 16-byte stride (`N % 8 == 0`), with at least one full tile.
  Output-SMEM budgeting includes the complete buffered tile, not just its tail.

Hybrid boundary tiles drain outstanding output readers before a persistent
CTA can wrap the shared output-stage index. Consecutive interior tiles retain
buffered overlap. Partial final CTA work slots cannot resume active work;
device-side active-prefix execution is separately restricted to one
nonpersistent tile per CTA. Both cases preserve the output-buffer protocol.

FP32 split-dW workspaces remain direct. Hybrid is deliberately not admitted
for NVFP4: its regional FP32 output-scale behavior requires separate support.
This store schedule is unrelated to the hybrid-precision MLP backend.

The shared search module proposes legal store/geometry/buffering combinations
for the common dense MLP and routed experts, using physical expert capacity
for the latter. It compares forward-only stores, all eligible row stores,
existing dW partition alternatives, and independent direct controls. Prepared
MLP openings preserve a one-leg partition control within the first eight
trials. Quantizers and register budgets are inherited, not hand-selected.
Existing recompute/quantization fusion remains optional and searchable.

Relevant revisions: MLP **47**, structural seeds **15**, prepared opening **4**;
MoE kernel **11**, portfolio **21**; common prequant **12**, backward **31**.
The runtime source digest for the final local and laptop studies is
`487c4ce2bb3c4b915e34e3ac14249aab18ebd7137744042d81821b08bc08298d`.
Older [revision-46 results](mxfp8_moe_eight_trial_prior_20261119.md) remain
historical evidence, not measurements of this revision.

## What the store fix actually buys

On HOME RTX 5090/GPU0, the frozen common packed GEMM
`M=17568, N=512, K=384` changes only its output-store policy. Ordinary event
timings are approximately **52.4 us direct versus 21.3 us hybrid**.
Of 552 output tiles, 548 use TMA and four cover the final 32 rows directly.

An isolated Nsight Compute run, without router injection, measured:

| Metric | Direct | Hybrid |
|---|---:|---:|
| LSU global-store sectors | 2,248,704 | 4,096 |
| Total L2 bytes | 128,135,072 | 74,121,344 |
| Actual / allocated registers per thread | 65 / 72 | 65 / 72 |
| Local spill load/store sectors | 0 / 0 | 0 / 0 |
| Average tensor-pipe active percentage | 14.81% | 41.21% |

Direct sector bytes equal four times the logical BF16 output size. The
remaining hybrid LSU sectors correspond to the boundary only; total L2
traffic falls about **42%**. These are measured L2/LSU effects, not a claim
about DRAM bandwidth: the direct export omitted the requested DRAM-write
metric, so the unmatched hybrid DRAM counter is not compared. Counter-run
times are diagnostic; the performance claim above uses ordinary events.

A paired whole-MoE test holds inputs, weights, routes, experts, geometry,
quantizers, and registers fixed, changing only three eligible shared-MLP
epilogues. Four alternating visits with seven samples each measure
**6.4545 ms -> 6.3635 ms**, a **1.0143x** speedup, not the isolated-GEMM ratio.
Outputs and every gradient are bitwise identical. Three actual SGD updates
also produce identical outputs, gradients, and resulting parameters; all
199 parameters change. This is optimization correctness, not convergence.

## Reprofiling preparation and recompute

A separate dense-MLP diagnostic fixture compares the old direct anchor with
the store/fused-preparation control. Its warmed graph traces contain:

| Group | Direct anchor | Store + fused preparation |
|---|---:|---:|
| Standalone quantization | 11 kernels / 113.9 us | 8 kernels / 63.6 us |
| Recompute/derivative kernel | 76.4 us | 120.5 us |
| Five linear GEMMs | 279.4 us | 198.1 us |
| Total launch count | 20 | 17 |

GPU trace coverage exceeds 99.6% in both cases: host launch gaps are not the
explanation. Fusion removes standalone passes but transfers work into
recompute; the 50 us standalone reduction is **not** 50 us of free gain.
The recompute kernel reports 72 registers in both traces. These traces use
an independently seeded dense fixture, not the paired MoE weights above.

A second finite search reconstructs and hash-verifies the original study's
X and shared weights. Its selected eight-trial anchor measures **452.0 us**;
the best tested store + both-orientation preparation control measures
**446.5 us**, about **1.22%** faster. Direct, geometry, and alternate partition
controls are retained. This is a finite diagnostic envelope, not a global
optimum or a newly installed runtime winner.

## Actual 8/16/32-trial searches

All results below use actual residual observation prefixes, not the size of
a seed pool. Each search resumes the preceding prefix without duplicate
evaluations. Both shared and expert components get the stated budget.
The 32-trial reference is finite, not exhaustive.

Final 5090 study: all 256 observations pass numerical validation and timing
eligibility. The independent quantized FP64 expert oracle's worst NRMS across
outputs and gradients is **0.0115%**. This is error against the quantized
numerical recipe, not a BF16 model-accuracy claim.

| Workload: packed lengths; D / E / K / expert H / shared H | Expert best 8 / 16 / 32 (ms) | Shared best 8 / 16 / 32 (ms) |
|---|---:|---:|
| 7000+10568; 512 / 64 / 8 / 128 / 384 | 2.0448 / 2.0448 / 1.9944 | .4572 / .4565 / .4565 |
| 2831+5393; 384 / 32 / 4 / 128 / 256 | .4059 / .4034 / .3891 | .1684 / .1684 / .1679 |
| 4097+8223; 768 / 48 / 6 / 256 / 512 | 2.3630 / 2.3630 / 2.3107 | .5304 / .5301 / .5301 |
| 127+385; 384 / 32 / 4 / 128 / 256 | .1023 / .1019 / .1019 | .04592 / .04525 / .04506 |

Maximum eight-versus-32 regret is **4.33% experts**, **1.92% shared**. This
reduces the risk of a disastrous eight-trial choice; it does not establish
unseen-shape optimality. On the first workload, the full eight-trial cold
calls take **27.2 s expert + 19.6 s shared**, including setup/confirmation.
Across four workloads their sequential sum is about 46.9-54.2 s. The 180 s
study limit is a soft search-admission cap, not an end-to-end deadline.

### Stronger BF16 control

Both paths use `torch.compile(fullgraph=True, dynamic=False)`, the user's
aggressive supported Inductor options, dynamic weights, `freezing=False`,
and an outer warmed benchmark CUDA graph. Reentrant BF16 FFN checkpointing
is enabled. A shared-only trace confirms **seven GEMMs**, including repeated
gate/up execution; this is not a saved-intermediate BF16 comparison.

| Workload above | Whole-MoE BF16/MXFP8 at 8 trials | At 32 trials |
|---|---:|---:|
| D512, 17568 tokens | 1.011x | 1.023x |
| D384, 8224 tokens | .968x | .974x |
| D768, 12320 tokens | 1.030x | 1.038x |
| D384, 512 tokens | .906x | .905x |

**There is no general 1.2x whole-MoE win against this stronger baseline.**
Small/narrow workloads still lose. The store bottleneck is fixed, but
preparation, recomputation and routing amortization remain material.
Standalone phase times are not summed to manufacture whole-layer gains.

### Second-SKU results and rejected timing evidence

The plugged-in RTX 5070 laptop completes the D384/8224-token 8/16/32 study.
Expert regret is **0.33%** and shared regret **5.23%** at eight versus 32.
All 32 shared timings and 31 of 32 expert timings are eligible; the summary
excludes the one unstable observation. Whole-layer BF16/MXFP8 ratios are
1.189x, 1.196x, and 1.081x in the three separate runs. The variation between
runs is not evidence that more trials made the kernels slower: bounded
search regret is computed from eligible same-context journal prefixes.
Treat these whole-layer ratios as descriptive laptop measurements, not a
controlled 10% cross-budget regression or a cross-device speed guarantee.

The RTX 5060 passes the hybrid correctness suite, but both its ordinary
eight-trial study and a diagnostic with 100 ms timing batches have **zero
eligible timings**. Their numerical observations and raw samples are retained
as unqualified evidence. No timing-quality gate was weakened and no winner
or performance claim is promoted from those runs.

## Validation and reproduction

- Hybrid epilogue suite: **13 passed on each of 5090, 5070 and 5060**, including
  variable E8M0 scales, M/N/K tails, interleaved persistent boundary/full tiles,
  output canaries, short-K scale fallback, and changed-input CUDA-graph replay.
- Related CPU/selection suite: **77 passed, 3 skipped**; later benchmark-summary
  and composition checks: **16 passed, 1 skipped**.
- Public packed first-hit/warm-cache/progressive-resume integration: **passed**,
  including fullgraph compilation and noncontiguous upstream gradients.
- All 12 local whole-module results and all three 5070 results have five
  identical live-allocation samples. The paired direct/hybrid test additionally
  passes changed-gradient replay and the three SGD updates described above.
- Generated API reference and whitespace checks pass. These are targeted
  qualifications, not a claim that every repository test or model converges.

Raw evidence is retained under
`autotune_reports/mxfp8_hybrid_tails_20261119/`: `matrix5090/` contains final
residuals and summaries; `diagnostics/` contains paired measurements, raw
Nsight reports/CSVs, traces, tests, matched finite controls, source archives,
and the laptop evidence. The folder is on the existing Storage4 symlink.
Its SHA-256 manifest checks the copied artifacts. Preliminary pre-guard runs
are labeled separately and are not ingested into the final matrix.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.qualify_moe_store_prior \
  --output /tmp/rtx-store-qualification \
  --cases new_ragged hybrid_384 hybrid_768 small_pool \
  --budgets 8 16 32 --max-autotune

.venv/bin/python -m benchmarks.analyze_moe_store_prior \
  /tmp/rtx-store-qualification --output /tmp/rtx-store-summary.json

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.benchmark_mlp_store_frontier \
  --seed-result /tmp/rtx-store-qualification/new_ragged/result8.json \
  --match-inputs --output /tmp/rtx-shared-finite-controls.json
```

GPU selection is caller/environment-owned. The study used only HOME GPU0;
GPU1 and unrelated workloads were not touched. Laptop tests used temporary
source directories and existing interpreters; missing pytest on the 5070 was
installed into a task-owned temporary target, not its project environment.
