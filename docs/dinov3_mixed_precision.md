# DINOv3 W4A16 and W4A8 study

This study asks whether retaining higher-precision activations fixes the
quality loss observed when all 48 DINOv3 transformer projections use NVFP4.
The short answer is no: uncalibrated four-bit weights dominate the remaining
error.

The reproducible harness is `benchmarks/dinov3_mixed_precision.py`. It is a
numeric format emulator, not a speed benchmark: packed values are dequantized
and consumed by BF16 `F.linear`. This deliberately separates format quality
from native-kernel availability.

```bash
python benchmarks/dinov3_mixed_precision.py \
  --device cuda:0 \
  --batch-size 2 \
  --image-size 224 \
  --output autotune_reports/dinov3_mixed_precision.json
```

## Results

Against the bias-separated BF16 DINOv3 reference:

| Format | Relative L2 | Mean cosine | Minimum cosine |
|---|---:|---:|---:|
| NVFP4-W / BF16-A | 0.3547 | 0.93708 | 0.93429 |
| NVFP4-W / MXFP8-A | 0.3538 | 0.93734 | 0.93420 |
| MXF4-W / BF16-A | 0.6638 | 0.78076 | 0.76379 |
| MXF4-W / MXFP8-A | 0.6484 | 0.79054 | 0.77218 |

For context, full MXFP8 W8A8 produced 0.98218 mean cosine, while native
NVFP4 W4A4 current-scale PTQ produced 0.90752.

Across identical BF16 projection inputs, NVFP4 weight rounding alone averaged
0.08193 relative L2. Adding MXFP8 activation rounding raised that only to
0.08526. The activation choice therefore cannot repair the accumulated weight
error. Calibrated, activation-aware weight packing or NVFP4-aware fine-tuning
is required.

## SM120 instruction reality

CuTe DSL 4.7 exposes two relevant warp MMA families:

- `MmaMXF4NVF4Op` accepts E2M1 for both operands with UE4M3 block-16 scales.
  It is the native NVFP4 W4A4 path and cannot consume MXFP8 activations.
- `MmaMXF8F6F4Op` accepts mixed E2M1/E4M3 operands, but uses UE8M0 block-32
  scales for both. Its four-bit operand is MXF4, not NVFP4.

Consequently, NVFP4-W/MXFP8-A is only a conceptual numeric combination on
SM120. MXF4-W/MXFP8-A is directly representable by the mixed MMA, but its
uncalibrated DINO quality is substantially worse. NVFP4-W/BF16-A requires
unpacking/dequantizing weights before a BF16 MMA; it may reduce weight traffic
for small-M inference, but does not provide FP4 tensor-core arithmetic
throughput.

These findings rule out scaling-policy work as the remedy. The next quality
work should target weight calibration, per-block reconstruction objectives,
AWQ/GPTQ-style optimization, or a short NVFP4 QAT recovery phase.
