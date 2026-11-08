# Balanced long-reduction backward

Long-token, small-output weight gradients can expose too few CTAs even though
their reduction is large. RTX's decomposed MXFP8 backward now offers balanced
K-tile partitioning and FP32 workspace reductions with 64 or 128 partitions.
These are additional autotuning choices, **not new universal defaults**.
MXFP8 linears/MLPs and NVFP4 frontends using MXFP8 backward share this machinery.

## Contract

`MXFP8BwdMatmulConfig.split_partition` accepts:

- `"uniform"` (default): the existing fixed positive `reduction_tile` chunks,
  with the existing nonempty-partition and covering-range checks.
- `"balanced"`: derive ranges over complete GEMM K tiles. For T tiles and P
  partitions, partition i owns `[floor(i*T/P), floor((i+1)*T/P))`. Every tile
  appears exactly once, ranges are nonempty, and sizes differ by at most one
  tile. The final logical tail remains masked/zero-filled as before.

Balanced mode requires `backend="decomposed"`,
`reduction="split_fp32_workspace"`, `reduction_tile=0` and P no greater than T.
Allowed counts are powers of two through 128. Fused/atomic/cluster split
support is unchanged; large counts are not silently enabled in those paths.
No BF16 operand padding or physical transpose is introduced.

For example, M=208192 and tile-K=128 gives 1627 tiles. Uniformly assigning
26 tiles to each of 64 partitions would create an empty last partition.
Balanced mode instead gives each partition 25 or 26 tiles and covers the
entire reduction. Permitting arbitrary split counts without changing the
power-of-two reduction tree would be incorrect; that shortcut is not used.

All partial GEMMs and their final reduction retain FP32 accumulation, with a
single final BF16 output cast. E4M3/E8M0 grouping, scale rounding, operand
quantization arithmetic and amax policy are unchanged. Different FP32 sum
orders may cause small final rounding differences. Original quantized forward
scales are not reused for a differently grouped backward reduction.

## Autotuning, ownership and compatibility

Normal users keep low-level schedules unset and use existing autotune controls.
No additional streams, output buffers, graphs or scratch management is required.
Explicit compatible configs can select `split_partition="balanced"` through
the normal backward-config argument.

The shared search adds compound `dw_balanced` moves covering legal 64×64,
64×128, 128×64 and 128×128 row-major output geometries and independently
selected split counts. Proposals preserve each operand's numerical policy.
Automatic candidates are bounded to small-output problems and at most 128 MiB
of partial workspace per leg; this is a search/resource bound, not an API shape
restriction. Explicit legal schedules remain available outside that subset.

SM count and output-grid size prioritize one/two/four-wave candidates without
claiming occupancy or tensor-core saturation. Whole-MLP opening proposals can
change both projection backwards together. MXFP8 and NVFP4 MLP searches share
the legal family; existing modes and independently configured legs remain.
Whole-MXFP8-MLP openings also compare fused gradient/hidden preparation early,
so that a small budget can discover preparation savings without more splits.

Uniform default fields are omitted from canonical backward dictionaries, and
old JSON schedules continue to load with uniform semantics. Search/runtime
identities are updated: backward search 30, backward frontend 6, MLP search 44
(including the later both-orientation recompute expansion).
Do not transplant winners from another device, source or application fixture.

## Qualification and deployment boundary

Both laptop 5070 and laptop 5060 passed the initial 19-case suite on Torch
2.13.0+cu132: exact reducer contributions, independent quantized dW oracles,
long/ragged reductions through 208192 rows, changed-buffer graphs and compiled
recomputed-MLP output plus all gradients. Follow-up memory/legacy regression
results are recorded with the implementation report, not inferred from this
initial suite. No 5090 timing or real G2DETR training improvement is asserted.

The focused application-policy driver composes the existing tuner, independent
oracle, residual store and paired deployment qualification:

```bash
python -m benchmarks.qualify_mxfp8_long_mlp \
  --configs /path/to/mlp_selected.json --shape 208192,128,256,128 \
  --trials 16 --rounds 12 --output autotune_reports/new-long-mlp-study
```

`--history` can reuse a compatible residual root while writing a new result
directory. This never installs a runtime winner or changes an application.
Compilation/search/qualification setup is outside the training timer. The
driver uses synthetic BF16 inputs, not a detector fixture.

Performance acceptance measures complete compiled forward/backward and memory,
not only dW GEMM duration. Existing fused recompute gradient/hidden preparation
and overlap remain competing choices; enabling more splits or smaller tiles
alone is not a guaranteed win. See [MLP controls](mxfp8_mlp.md) and
[the retained qualification](balanced_backward_qualification_20261108.md).
