# MXFP8 / GLU integration qualification

Status: local application qualification passed. On 2026-10-24 the user selected
`jhu-clsp/ettin-encoder-32m` to replace DINO as the primary model-level target.
See [Ettin qualification](ettin_qualification.md). Earlier DINO findings below
are retained as historical limitations, not silently turned into passes.
The model evidence supports the local GLU integration merge, not a universal
release or convergence claim.

## Scope

RTX 5090, CUDA 13.2, CuTe DSL 4.7.0, supported Torch 2.12.1 and 2.13.0
environments. Preserve BF16 public I/O, E4M3/E8M0 block quantization, FP32
accumulation, high-precision GLU activation/gating, and fully recomputed MLP
backward. NVFP4 backward remains an explicit experimental policy, not the
default. No downloaded models, vendored Quartet/DINO checkouts, datasets, or
private benchmark results are included in the package.

## Scheduling changes

- Packed MXFP8 GEMMs expose `k_tile_unroll=1/2/3/4/6/8/12`; default remains 1.
- Forward, dX, and dW have independent autotune coordinates.
- GLU inherits the control for BF16/packed FP8 output and recomputation.
- MLP gate/down/recompute and both gradient groups inherit the search axes.
- On-chip linear and GLU `k_unroll` now expose the same factor choices.
- Serialization defaults missing fields to 1. No benchmark winner is silently
  installed, and quantization policy is never selected by latency alone.

## Qualification procedure

Run pytest, not only unittest discovery: the suite contains both styles.
Dev dependencies include the optional research-test `joblib`/scikit-learn stack;
the installed runtime remains free of that requirement.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/validate_production.py \
  --output autotune_reports/production.json
.venv/bin/python tools/doc_reference.py --check
.venv/bin/python -m build
```

Repeat on `.venv-torch2121/bin/python` for the other supported stack. Preserve
per-run logs/JUnit XML and run a final qualification after source edits stop.
Failures from an exploratory run during edits do not establish the final
source's status; neither do later fixes retroactively turn those runs green.

## Current evidence

Artifacts: `autotune_reports/mxfp8_glu_merge_20261024/` (local, not distributed).
Initial CPU suites: 909 tests and 447 subtests passed on each supported stack;
816 CUDA/optional cases skipped. The expanded unroll matrix passed 34 CUDA
tests on Torch 2.13: aligned/ragged reductions, multiple buffering depths,
BF16 and packed GLU output, SiLU/GELU, and compiled MLP output plus every
gradient. Arithmetic is bitwise unchanged from factor one; MLP also passes
the existing independently decomposed numerical reference.

Latest CPU suites on both stacks: 916 passed, 827 skipped, 460 subtests passed.
A combined CUDA scheduling/optimizer matrix passed 79 tests on Torch 2.13,
including 20-step compiled AdamW with BF16 parameters and moment states for
MXFP8, NVFP4 current/delayed/JIT-region/block, and SiLU/GELU recomputed MLPs.
These short lifecycle tests are not transformer-convergence qualification.

The final broad regression runs on both Torch 2.13 and 2.12.1 each passed
1731 tests and 1106 subtests, with 8 skips and the two known DINO quality
tests explicitly deselected.
This is regression evidence, not an unconditional green release suite:
the DINO failures below remain release blockers for this merge.

The first broad Torch 2.13 CUDA run produced 1669 passes and 32 failures
(including subtest failures), with 8 skips. It is exploratory, not a green
release result. Targeted reruns are resolving failures before the merge.

One narrow-tile failure was independently shown to be FP32-reference double
rounding. At seed 1963, shape 1041x322x256, output (632,12), the exact dot is
`-0.008331306278705597`; FP32 reduction rounds to `-0.008331298828125` and its
BF16 cast becomes `-0.00830078125`. The kernel's `-0.00836181640625` is the
correct BF16 rounding of the exact dot. That test retains its original FP32
error gate and accepts an otherwise failing value only when it exactly equals
independent FP64 reduction with direct nearest-even BF16 rounding. It does not
demand FP64 accumulation from the FP32-accumulating kernel.

The selected-backward memory test now samples at dX exit: the implementation
already releases an unused placeholder before derivative entry, so measuring
only there cannot detect the extra transient allocation. Exact gradient parity
and the expected `M*H*2` byte allocation saving remain required.

Exploratory unroll-only sweeps show approximately 1–2% forward headroom on
the tested MXFP8 schedules, not the larger NVFP4 gains. Backward and packed
GLU have no clear win in these sweeps. They do not justify changing defaults.
Batch timings include preparation; MLP measurements must include recomputation.

## First-hit reliability finding

Very short host-launched timing windows could mark every numerically valid
candidate unstable and fail the model's first invocation. The prequant
evaluator now calibrates longer batches and records actual repetition counts
without weakening stationarity gates or switching to graph timing. Revision
11 separates this measurement contract from older records.

Runtime MXFP8 tuning may use the current session's numerically checked initial
schedule if no timing winner exists. It warns and keeps the choice process-local;
it never writes a persistent winner. An unchecked seed, compile/numerical error,
or explicit offline search still fails rather than hiding the problem.
Cold fullgraph, reduce-overhead inference, training/packed inference and
8-to-16-trial resume tests pass after these changes.

The same checked-seed runtime fallback covers NVFP4 dynamic, delayed, regional,
weight-prequantized, and fully-prequantized first-hit calls. Direct/offline
tuning remains strict. Historical portable predictors remain revision-gated;
old artifacts correctly fall back to ordinary search. A resumed session may
retry a numerically valid but timing-ineligible candidate; the attempt consumes
budget and cannot duplicate a candidate with previously eligible timing.

## Production and application boundaries

The unified production matrix passed all 49 cases on each supported Torch
stack. A 52-test targeted Torch 2.12.1 rerun (plus 12 subtests) passed after
the runtime/contract fixes. This includes cold compile, interrupted and
concurrent caches, stale-prior fallback, narrow tiles, and MLP retention.

Two optional local DINO quality tests still fail (three assertions). They also
fail on an untouched detached worktree at `31b316de4677b140b70769e371886efa72a9d9ee`
with identical values:

| Policy/output | Observed minimum cosine | Existing gate |
|---|---:|---:|
| MXFP8 PTQ / CLS, seed 4403 | 0.96795851 | >0.97 |
| NVFP4 block dynamic / patches, seed 4410 | 0.91706991 | >0.92 |
| NVFP4 block PTQ / patches, seed 4410 | 0.91706991 | >0.92 |

The thresholds remain unchanged. A final suite excluding these two explicitly
identified baseline tests must not be described as an unconditional full-suite
pass. No model-quality or 1.0 release certification follows from kernel gates.

`benchmarks/diagnose_dinov3_quantized_reference.py` independently decomposes
quantization and uses FP32 PyTorch GEMMs. On identical inputs to all 48 linears:

- NVFP4 block agrees exactly for all 16,671,744 output values, including the
  complete model output. Its quality limit here is the numerical policy, not
  an RTX implementation discrepancy.
- MXFP8 differs at 352 of 16,671,744 values; the worst per-linear relative L2
  is 0.00005321. Through the quantized transformer the final CLS relative L2
  versus this oracle becomes 0.07077, with minimum cosine 0.99670. Small local
  rounding differences can change subsequent quantization decisions; a small
  kernel error therefore does not promise equally small whole-model drift.
- The independent MXFP8 oracle itself has CLS minimum cosine 0.97340 versus
  BF16 on this input. Improving the model-level boundary is separate numerical
  research, not evidence that the new unroll option changed accumulation.

Artifacts include `dino_head_baseline.xml`, `dino_local_oracle.json`, both
production JSON reports, raw broad-suite XML, and targeted rerun XML. No
threshold was lowered and no dataset/checkpoint is packaged.

### Quality investigation following the merge hold

`benchmarks/study_dinov3_quality_policies.py` compares four combinations of
MXFP8 X/W scale rounding and eight NVFP4 combinations (block/current outer
scaling with amax/Four Over Six/MSE5/LSQ1 block selection). Schedules stay fixed,
autotuning is off, all 48 backbone linears are converted, and original bias
and mask semantics are preserved. The default seeds include both regressions
(4403/4410) and two additional inputs (4511/4512). This is synthetic-image
feature fidelity, not downstream task accuracy. Policies remain explicit;
the experiment does not promote a default or change a test threshold.

A CPU checkpoint inspection found 34,836 of 1,327,104 weight blocks (2.625%)
have amax/6 below block-only's normal E4M3 scale floor of 2^-6. The fraction
is strongly layer-dependent: 95.28% in `blocks.0.attn.proj`, compared with
0.033% in `blocks.11.attn.proj`. This identifies a candidate source of
block-only distortion, not proof that changing the scale domain would improve
the whole model. Native scale semantics remain unchanged.

The initial 12-policy/four-seed sweep completed on both supported Torch stacks:
all 48 policy/seed output-metric records matched exactly. A separate 16-seed
evaluation (4600–4615) compared the selected alternatives with their baselines.
No default was changed.

| Policy | Mean CLS relative L2, 16 additional seeds | Worst patch cosine | CLS L2 wins versus its baseline |
|---|---:|---:|---:|
| MXFP8 infinity X / infinity W (baseline) | 0.13485 | 0.99069 | — |
| MXFP8 floor X / infinity W | 0.15167 | 0.98982 | 1/16 |
| NVFP4 block amax (baseline) | 0.45066 | 0.91725 | — |
| NVFP4 block Four Over Six | 0.37707 | 0.93659 | 16/16 |
| NVFP4 block MSE5 | 0.40693 | 0.93005 | 16/16 |

Floor-X improved the originally failing MXFP8 seed, but failed the broader
quality comparison. It is not promoted as a fix. Four Over Six improved all
three measured output groups on every additional seed, making it a supported
candidate for an explicit DINO NVFP4 recipe—not a universal accuracy claim or
a reason to silently change block-only's default numerical policy.

`tests/test_dinov3_quality_policies.py` additionally checks the explicit
Four Over Six recipe on six seeds, retaining patch cosine >0.92 and storage
cosine >0.95, and requires bitwise dynamic/packed output parity. The original
default-policy DINO tests remain unchanged and unresolved. Thus the merge hold
is still in effect.

The final two-GPU operand-device tests passed on both Torch versions, and the
final source-matched production matrices again passed 49/49 on each stack.

### MXFP8 accumulation audit

The diagnostic now supports `--audit-rounding` and
`--oracle-accumulation fp64`. For all 352 cells where native MXFP8 and the
FP32 reference disagree, exact rational dot products verified the FP64 sums:
103 native values were nearest-even BF16, 237 FP32-reference values were
nearest-even, and neither was nearest in 12 cases. This audit is restricted
to differing cells; it does not certify cells where both implementations
agree. The full FP64 model oracle reaches CLS minimum cosine 0.97461 on seed
4403, versus native 0.96796, without changing quantization.

The policy study's optional `--mx-reduction-tiles 1 2 4` reuses existing native
split GEMM/reduction kernels to test shorter FP32 accumulation chains. It is
research-only: workspace zero-padding and allocation are not proposed runtime
launchers, and no performance claim is made. On identical default-model layer
inputs, one-tile splits reduce disagreements against FP64-rounded outputs from
330 to 184. Layer RMS error improves in 25 layers, ties in 8, and worsens in
15; the worst layer relative L2 is not improved. Thus even this arithmetic
comparison is mixed, rather than a blanket precision win.

All three split experiments clear the original seed's CLS threshold, but on
16 additional seeds their mean CLS relative L2 is 0.13675/0.13740/0.13586,
compared with default 0.13485. None is promoted as a universal DINO fix.
This separates local accumulation behavior from whole-model quantization
trajectory sensitivity and preserves the original merge hold.

Ten diagnostic tests passed on each Torch stack: exact-rounding audit cases,
independent quantized-dot checks for split/unsplit and ragged inputs, policy
construction, and the explicit NVFP4 dynamic/packed six-seed quality recipe.
Artifacts: `dino_exact_rounding.json`, `dino_fp64_oracle.json`,
`dino_split_paired.json`, `dino_split_accumulation_v4.jsonl`, and
`dino_split_holdout.jsonl`. Earlier v1–v3 split-driver attempts failed interface
validation and are not numerical evidence. No production kernel changed in
this investigation.

### Consolidated packaging check

After the diagnostic additions, both CPU suites passed 922 tests and 460
subtests, with 831 CUDA/optional skips. The source archive and wheel built
successfully and passed `twine check`. Inspection of the 616 source-archive
entries found no git-ignored source files; generated setuptools egg-info is
expected metadata and was excluded from that privacy check. The 136 wheel
entries contain only the runtime package and distribution metadata.

The historical DINO approval boundary was resolved by the user's explicit
selection of Ettin instead. Original DINO assertions remain available as
optional research tests; their failures have not been relabeled as passes.

### Final Ettin replacement gate

All ten prospective inference policies passed predeclared guards on both Torch
versions, including ragged length 511 and fully recomputed GELU MLP. Eight
dynamic policies completed twelve-step finite-gradient/BF16-AdamW lifecycle
checks on both compiled stacks and eager Torch 2.13. See the linked Ettin
report for actual errors, uncertainty and the important non-convergence caveat.

Final CPU suites: 923 passed, 815 skipped, 22 historical research cases
deselected, and 461 subtests on each stack. Source-matched CUDA suites before
the Ettin harness additions passed 1,731 tests and 1,106 subtests per stack
(eight skips, two DINO failures explicitly excluded). Both current-runtime
production matrices passed 49/49; two-GPU operand checks passed on each stack.
The later changes are benchmark, tests, documentation and packaging only.
