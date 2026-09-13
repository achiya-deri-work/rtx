# Shape-stratified first-hit qualification

The [manifest](../autotune_manifests/shape_stratified_v1.json) contains 16
explicit `(M,N,K)` geometries. For GLU/MLP, `N` is the intermediate width;
MLP output width is `K`. Cases cover single-token/small-M workloads, small
CTA grids, square/expansion/contraction projections, long-K and long-M
reductions, narrow/wide outputs, large grids, M=255/257 tile boundaries,
ragged M, dimensions aligned to 32 but not 128, and odd N/K.

“Underfilled” and “multiwave” are workload-category descriptions. Actual
CTA coverage depends on the selected tile and device and is not assumed
constant across configurations or SKUs.

## Coverage and holdout

- Eight families: MXFP8 prequantized-pipeline forward and backward; NVFP4
  block, tensor-delayed and JIT-region forward; GLU; MLP forward and training.
- Four policies: legacy/mixed pools, each without/with frozen residual+MLP heads.
- Independent budgets of 8, 16, 32 and 64 attempts; one replicate by default.
- Total planned: 2,048 cells. A four-hour launch is a resumable subset, not
  a promise to complete every cell.
- Each 64-cell sweep visits all 16 strata and all eight families. The
  diagonal family rotation covers every family/shape over eight sweeps.
  Each four-treatment comparison stays adjacent, with rotated order.
- All eight-trial cells precede the sixteen-trial wave, and so on.

Plan creation checks every `(M,K)` against the frozen audited inventory of
142,110 observations and excludes the preceding study's evaluation shapes
too. This is stricter than excluding exact `(M,N,K)` triples. It reuses the
previous inventory audit, not an assertion that every untracked dataset on
disk has been scanned. The parent audit hash is preserved. New observations
are evaluation-only and are not admitted into model training.

## Measurement boundaries

This study is hot-cache only. MLP uses the existing BF16-gate precision
policy; GLU uses MXFP8 output. It does not add distribution sweeps, MLP
cache rotation, or new numerical recipes. Existing family correctness gates
run before a candidate is eligible to win. Each winner receives three extra
verification measurements. Compilation/proposal time and failures are recorded.

Ragged shapes are intentional coverage probes. A setup, search or verification
failure remains visible with its phase and traceback, and is excluded from
paired speedup claims. A failed harness is not proof that the public linear
API cannot handle the shape through another path. These cases should inform
subsequent harness/frontend work, not be silently replaced by aligned shapes.

The first pilot exposed an illegal instruction in the prequant forward
reference at `(1,1280,896)`. Its four setup failures and 11 successful cells
are preserved under `shape_stratified_local_v1`. The guarded restart is
`shape_stratified_local_v2`: it imports only that exact setup quarantine,
not the pilot's latency observations. Once a setup fails, subsequent policies
and budgets for that exact family/shape/regime are recorded as `blocked_setup`
without launching it again. Other contexts remain independently testable.

## Launch and resume

Prepare once on the machine containing the parent audit:

```bash
.venv/bin/python -m tools.shape_stratified_study \
  --output autotune_datasets/shape_stratified_local_v2 \
  --quarantine-from autotune_datasets/shape_stratified_local_v1
```

Then launch or resume (nohup/disown; prints PID and log):

```bash
bash run_shape_stratified_study.sh autotune_datasets/shape_stratified_local_v2 4
```

Do not launch a second copy while the local run is active. It holds an exclusive
run lock. The clock is checked between cells; an in-flight cell can exceed the
deadline. Results use independent task journals and atomic result JSON. CSV
and Parquet summaries are exported when the runner exits normally.

For another SKU, synchronize the exact source revision, copy the frozen
`plan.json` and predictor artifact, then use a fresh output directory:

```bash
bash run_shape_stratified_study.sh autotune_datasets/shape_stratified_laptop_v1 4 /path/to/plan.json
```

Artifact relocation is supported with SHA256 verification. Source changes
invalidate resume; preserve completed older studies and create a new plan.
Do not copy one machine's result files into another machine's output directory.

Analyze after stopping/completion:

```bash
.venv/bin/python -m tools.analyze_predictor_study autotune_datasets/shape_stratified_local_v2
```

`summary.csv` / `summary.parquet` retain category, shape group, family,
budget, treatment and failure phase for stratified analysis. No winners or
learned policies are automatically installed into the runtime cache.

## Frozen overnight release

The published `autotune_manifests/shape_stratified_overnight_v1/plan.json`
and frozen predictor are sufficient; raw historical datasets are not needed.
Use `bash run_shape_overnight.sh 5090` for five hours, or
`bash run_shape_overnight.sh laptop` / `5070ti` for seven hours.
The launcher checks the installed runtime, then backgrounds with nohup/disown,
prints the PID/log, and exports CSV/Parquet plus an analysis on exit. Set
`RTX_STUDY_OUTPUT` to an existing compatible output directory to resume.

Known pilot setup/initial-candidate failures and fatal CUDA contexts are
blocked by exact family/shape/regime. New such failures suppress retries in
that context across treatments and budgets; ordinary search errors do not.
Blocked cases remain visible. The pilot found M=1 forward/backward faults and
an odd-width MLP-training alignment failure; this release does not claim to fix
those kernel/harness defects.
