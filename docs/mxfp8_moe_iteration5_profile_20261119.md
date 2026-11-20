# Iteration 5 consolidated evidence: active-budget and K4 studies

All **102 context executions** completed: 54 original and 48 fanout/fraction
follow-up cases. All are HOME GPU0 RTX 5090, Torch 2.13.0+cu132, CuTe 4.7.0.
Production runtime SHA256 is
`a5e83c8eb4400c824105bc15a3a710f7c76e1cf13eb3d62a7d84ca80ffd2baf8`.
No collection remains running. WORK services and HOME GPU1 are out of scope.

The CPU-only consolidator rechecks the original manifest/driver/runtime,
shape/layout/initialization/tuning contracts, samples, finite values, independent
quantized FP64 expert oracle, repeated graph-gradient replay and zero live-byte
growth. Worst operational-oracle NRMS is **0.000129462 (0.01295%)**. This
proves implementation fidelity to its quantized recipe, not equality to BF16,
training convergence, or performance beyond synthetic warmed graph replay.

## Main matrix, not the unequal mix of all auxiliary controls

B16 × L1024, medians over contexts; speedup means reference/native time.

| Wave/layout | Cases | Whole vs compact checkpoint BF16 | Whole vs saved BF16 | Frozen expert vs native checkpoint BF16 | Shared vs checkpoint BF16 |
|---|---:|---:|---:|---:|---:|
| Original packed | 16 | 1.049× | 1.014× | 1.288× | 1.082× |
| Original flat | 16 | 1.083× | 1.001× | 1.258× | 1.082× |
| Fanout packed | 24 | 1.023× | 1.022× | 1.361× | 0.950× |
| Fanout flat | 10 | 1.049× | 0.982× | 1.216× | 0.945× |

These are descriptive context medians, not independent-sample confidence
intervals. Expert-only uses the original BF16 row layout; its gain must not
be presented as a gain against compacted experts. Router/expert/shared scopes
overlap, so **do not add their standalone times**. Both whole precisions use
the same router. Native recomputes FFNs; the saved BF16 control does not.

## Useful controlled findings

- Across the original four packed geometries, E16 median whole speedup against
  compact BF16 was about 1.133× versus 0.882× at E128. The poor E128 K6 path is
  not resolved by greater token count alone.
- New D384/K4/S256/H320/E32/B16/L1024 packed measured 2.342 ms, 1.044× compact
  checkpoint BF16 and 0.997× saved BF16. New architecture is not an automatic
  route to 1.2×.
- Same active budget and mean rows/expert at D512 quarter-shared:

| K/E | Shared/expert H | Packed whole | Compact-BF16 speedup | Flat whole | Compact-BF16 speedup |
|---|---|---:|---:|---:|---:|
| 4/32 | 512/384 | 3.242 ms | 1.131× | 5.944 ms | 1.113× |
| 6/48 | 512/256 | 3.659 ms | 1.103× | 6.297 ms | 1.095× |
| 8/64 | 512/192 | 4.060 ms | 1.083× | 6.878 ms | 1.084× |

- The D384/K4/H320 S192/256/320 sweep has independently verified identical
  X, dY, routed expert weight hashes and hard-route identities in all six
  available three-way controls. Shared fractions really are 13.04%, 16.67%,
  20%, not the requested labels 1/8, 1/6, 1/4. Joint-fraction candidates retain
  the hard ±7.5% active-4D bound and 64-aligned widths.
- Flat/global routing at B16 diagnostic partition T16384 costs ~3.3–4.9 ms
  standalone versus ~0.6–1.8 ms for independent packed L1024 pools. Their
  semantics differ; no routing interchangeability is implied. This points
  toward long-pool routing work but is not new per-kernel counter attribution.

## Search coverage warrants inspection before more kernel families

Winner counts across the 102 executions:

- Forward row layout: **original 102, compact 0**.
- Active prefix and each bounded producer: **enabled 5, disabled 97**.
- Fanout packed: fragment GLU + dual-orientation weights selected in 26/34;
  the remaining 8 use the older GLU/separate-weight path. Legacy packed has
  25/30 on the newer preparation path.
- The two worst fanout packed contexts, D384/K6/E128/H192 and
  D512/K6/E128/H256 (S320/384), still select no prefix, no producer bounds,
  original forward rows, old GLU and separate weight preparation. Physical
  row capacity is 1312 versus 768 mean real rows/expert (1.708× modeled slots).

**Selection counts are not admission counts.** Inspect raw journals and
`rtx/autotune/moe.py` before concluding these were tried and lost. These fields
suggest a first-hit coverage problem, but do not quantify its recoverable gain.
Static capacity/slot ratios are not measured issued FLOPs, DRAM traffic, or
occupancy. The 102-context wave collected ordinary phase timing, not new
Systems/Compute profiles; use existing earlier captures only with that caveat.

No kernel or public API change was made during collection. The follow-up's
independent named expert RNG setup differs from the original wave: they are
not matched numerical A/Bs. Raw source, initialization identities and compiler
options remain available to reproduce matched comparisons now.

## Audit/export and review inputs

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m benchmarks.consolidate_moe_profiles \
  /tmp/rtx-moe-active-budget-20261119/home-study \
  /tmp/rtx-moe-active-budget-20261119/home-second-shard \
  /tmp/rtx-moe-k4-20261119/study \
  --output /tmp/rtx-moe-iteration5-20261119/consolidated
```

The output has namespaced rows, raw hashes, phase samples/ratios, selected
feature counts, control-identity checks, and JSON/CSV/Parquet exports. It does
not train on or mutate the original records.

Raw roots for the reviewer:

- `/tmp/rtx-moe-active-budget-20261119/{home-study,home-second-shard}` (27 each).
- `/tmp/rtx-moe-k4-20261119/study` (48), including exact `.expert/.router/.shared`
  sidecars, ordinary logs and per-component tuning journals.
- `/tmp/rtx-moe-iteration5-20261119/consolidated/summary.json` (102 audited rows).
- Storage4-backed original archive:
  `autotune_reports/mxfp8_moe_active_budget_20261119/home-complete.tar.gz`.
- Storage4-backed fanout archive:
  `autotune_reports/mxfp8_moe_k4_fraction_20261119/`.
- Previous implementation and review:
  [iteration 4](mxfp8_moe_iteration4_20261119.md) and
  [iteration 4 review](mxfp8_moe_iteration4_review_20261119.md).

Next sequence: Astra 6.2 High review using this evidence and current source,
then three Sol 6.2 High implementation owners with disjoint scopes. Their
experiments must use HOME GPU0 under the shared lock, fresh processes, fixed
numerical policy, bounded autotuning rather than hand-picked winners, whole
layer qualification against compact checkpoint BF16, and preserve tiny-forward
correctness plus meaningful unseen-shape eight-trial checks.
