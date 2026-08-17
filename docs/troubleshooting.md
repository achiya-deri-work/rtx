# Troubleshooting

## Runtime validation fails

Run:

```bash
python -c 'import rtx; print(rtx.validate_runtime_environment())'
rtx-autotune probe
```

Check that PyTorch uses CUDA 13.2, CUTLASS DSL is 4.7.0 with the `cu13` extra,
and the target GPU reports compute capability 12.0 or 12.1. `CUDA_TOOLKIT_PATH`
must point to a toolkit whose NVVM backend supports `sm_120a`.

## First call is slow

Native CuTe kernels compile lazily. `autotune="cache"` may load a winner but
does not eliminate initial compilation. The default `autotune="balanced"`
additionally benchmarks up to 24 candidates or 30 seconds when no winner
exists; `"online"` is its compatibility alias. Use `module.explain(x)` to inspect
whether selection is deferred and `rtx-autotune list-winners --summary` to
inspect installed winners.

## An online campaign emits compile errors

Compile failures are valid search observations, not necessarily fatal runtime
failures. Audit the residual bundle, inspect repeated configuration IDs, and
verify the toolkit before restarting:

```bash
rtx-autotune audit <bundle-or-zip>
rtx-autotune probe
```

Repeated failures across otherwise unrelated legal configurations usually
indicate an NVVM/toolkit mismatch. Failures isolated to particular IDs should
remain in the dataset so feasibility models and static legality can learn from
them.

## Packed module cannot train or cast dtype

This is intentional. Packed modules contain no BF16 master Parameter. Keep the
dynamic module/checkpoint for training and format conversion. Move a packed
module with `.to(device=...)`; create a new packed module to change formats.

## JIT-regional scaling rejects a packed NVFP4 weight

JIT regions observe both dynamic BF16 operands. Use the normal conversion API,
which selects current packed-weight scaling automatically:

```python
packed = dynamic_nvfp4.to_quantized_weight()
```

Or choose `to_quantized_weight(scaling="block")` after validating its numeric
range on representative data.

## Performance differs from documented numbers

Published timings are device-, shape-, cache-regime-, and software-specific.
Confirm that BF16 and low-precision runs share compilation, optimizer, batch,
and measurement policies. Install verified local winners rather than copying a
winner from another SKU.
