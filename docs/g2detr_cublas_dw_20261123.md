# Final cuBLAS dW trial — RTX 5090 qualification

The standard MXFP8 backward search tests BF16 cuBLAS dW as its final
challenger. It uses the best native candidate's dX kernels and replaces
only the weight-gradient arithmetic. Quad quantization and cross-leg
operand aliasing are converted to independent preparation for this
candidate, so the full benchmark includes that change. No automatic
replacement occurs until the mixed workload validates and wins.

On the HOME RTX 5090, a fresh eight-trial graph-mode search on
`[138304,384]` selected the final candidate. Complete dX+dW was
**501.4 → 382.9 µs** (23.6% lower). A separate graph-replay comparison
measured **189.1 µs** for the selected dW and **188.4 µs** for BF16
`torch.mm`. dW NRMS against BF16-input FP32 GEMM, cast once to BF16,
was 0.030%. This is not a claim of equivalence to MXFP8 dW bytes.

A separate two-trial MLP probe (M=4096, K=H=O=128, SiLU) measured
128.4 → 81.4 µs for its initial native plan versus its BF16-dW
challenger. Both weight gradients selected cuBLAS and both input
gradients stayed MXFP8. This is a two-plan functional/performance
check, not a claim about comparison with a fully tuned MXFP8 MLP.

The direct cuBLAS launcher reuses PyTorch's current-stream handle
and the caller's output storage. It requests BF16 operands and
FP32 compute, disallows reduced-precision reduction, and restores
handle modes. There is no eager matrix multiply or transpose
materialization in the launcher.

Checks covered fullgraph MXFP8Linear and MXFP8MLP, partial/dW-only
backward, noncontiguous upstream gradients, changed inputs, custom-op
checking, terminal budget accounting, and persistent cache identity.
PyTorch 2.13.0+cu132, driver 595.99.02. No G2DETR deployment or
production cache was modified.

Raw search journals and regression outputs are under
`research_data/g2detr_cublas_dw_20261123/` (ignored locally).
Reproduce the long linear case from the original configuration:

```bash
python -m benchmarks.diagnose_long_dw \
  --winner research_data/g2detr_long_dw_20261123/winner.json \
  --cases bf16,native_tail,tuned --tune-cache fresh_dw_cache \
  --output cublas_dw_confirmation.json
```
