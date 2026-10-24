# Quartet II native-kernel evaluation

Follow-up: [stream/RNG repair and deeper training study](quartet2_deeper_study_20261023.md)
includes corrected graph timings and 63 decoder runs. The unmodified-upstream
failures below remain historical evidence; the follow-up does not erase them.

This is an isolated research comparison, not a new RTX backend or release
qualification. No runtime defaults or production backward contracts changed.

## Reproducibility

- Upstream: [IST-DASLab/Quartet-II](https://github.com/IST-DASLab/Quartet-II),
  revision `5f2a47e8f464b17ef4b1cac78db8d14edd1f3f7d`, Apache-2.0.
- RTX 5070 Laptop, Torch 2.13.0+cu132, CUDA toolkit 13.2, FlashInfer 0.6.3,
  cuDNN frontend 1.31.0; native `quartet2` built for SM120a.
- Isolated remote installation: `/home/achiya/rtx-quartet-study-20261023`.
  No shared environment or unrelated workload was modified.
- Raw local artifacts: `autotune_reports/quartet2_20261023/` (not source-controlled).
- Harness: `benchmarks/evaluate_quartet2.py`, one process per mode/shape.
  Shapes are M,N,K. Inputs and parameters are BF16; independent arithmetic
  references use FP32 on those exact BF16 values. TF32 is disabled.
- `--compile` uses fullgraph, static shapes, without automatic CUDA graphs.
  Each case uses 64 gradient samples and 100 steps of matched BF16-SGD linear
  regression. This is not a transformer convergence or multi-seed result.

Example, with the optional upstream installation on `PYTHONPATH`:

```bash
python -m benchmarks.evaluate_quartet2 --mode quartet \
  --shape 128,256,256 --compile --samples 64 --training-steps 100 \
  --output /tmp/quartet-new-result.jsonl
```

Output creation is exclusive to prevent accidental replacement of results.

## What was actually executed

The published native CUDA quantizers and FlashInfer FP4 GEMMs, not simulation.
The current upstream GEMM backend is cuDNN. Its backward rotates operands,
requantizes saved forward FP4 operands, applies EDEN correction, and rounds
microscales stochastically. It does not require stochastic FP4 conversion
hardware on RTX: FP4 conversion is nearest rounding, while scale rounding
uses software-generated randomness.

Compared modes: BF16, RTX MXFP8, default RTX NVFP4, upstream Four-over-Six,
upstream RNE, and an experimental upstream-forward/RTX-MXFP8-backward hybrid.
All 18 compiled cases completed across 128×256×256, 512×768×768, and
2048×768×1536, including 1,800 total training steps. Outputs, gradients, and
final parameters were finite in every case.
The hybrid saves original BF16 operands; it is not a memory-equivalent
replacement for upstream's packed saved state.

## Upstream tests

Unmodified `test_rounding.py` and `test_linear.py`: **14 passed, 3 failed**.
Failures are retained in `upstream-tests.log`:

- 256-sample backward test: W2 6.474 bits versus required 6.5.
- 4096-sample backward test: W1 7.592 versus 7.6, with additional reported
  values W2 8.04 and W3 8.72 below their respective thresholds.
- Forward test: projection coefficient 0.998607 versus required 0.999.

The upstream quantity named `cosine` is actually `<candidate,reference>`
divided by reference squared norm. Do not interpret it as cosine similarity.
Several misses are small; this one stochastic suite run does not establish
their failure probabilities. Fullgraph forward passed.

## Numerical distinction: bias target versus single-step noise

At 128×256×256, normalized RMS errors against original BF16 operands are:

| Mode | Forward | dX | dW |
|---|---:|---:|---:|
| BF16 | 0.17% | 0.17% | 0.17% |
| RTX MXFP8 | 3.75% | 3.77% | 3.78% |
| RTX NVFP4 default | 13.27% | 3.77% | 3.78% |
| Quartet Four-over-Six | 12.17% | 16.18% | 16.83% |
| Quartet RNE | 13.28% | 16.69% | 17.28% |
| Quartet forward / RTX backward | 12.17% | 3.77% | 3.78% |

Averaging 64 Quartet Four-over-Six gradients reduces dX/dW error against
**its forward-quantized STE reference** to 1.74%/1.72%. Against original BF16
operands, mean dX error remains 8.87%. These are different targets: averaging
away backward noise does not undo forward operand quantization.

At 512×768×768, Four-over-Six forward error is 12.28%, versus RNE 13.41%.
The hybrid retains 3.76%/3.77% backward error. This supports testing the
forward technique separately from stochastic FP4 backward.

At 2048×768×1536, Quartet forward error is 12.271% versus RTX default
13.428%; Quartet dX/dW are 16.253%/16.240%, whereas hybrid dX/dW are
3.762%/3.773%. Final regression loss is 0.979723 for Quartet, 0.979751
for RTX NVFP4, and 0.980275 for the hybrid: the hybrid is not a universal
training-loss winner.

Matched regression final five-step median losses at 128×256×256:
BF16 0.241435, MXFP8 0.243045, RTX NVFP4 0.250156, Quartet 0.250181,
Quartet RNE 0.251778, hybrid 0.248262. These are diagnostic, single-seed,
one-layer losses, not evidence of a model-quality ranking.

## Performance boundary

The harness measures ordinary event-timed compiled calls, including Python
launch gaps. Both implementations use untuned schedules; upstream CPU seed
generation and multiple custom-op launches remain visible. These timings
must not be compared with existing CUDA-graph RTX speedup tables or treated
as tensor-core throughput. A fair throughput decision requires stream-safe,
RNG-correct graph replay or an equivalent full-training execution regime.

## Ideas worth carrying forward

1. **Four-over-Six forward:** confirmed reduction in forward error here.
   RTX already exposes its own implementation and headroom policy. Keep it
   optional: existing model-level tests do not show universal superiority.
2. **Post-hoc scale range alignment:** the upstream rotation/quantization
   stage writes FP4 plus extended-range temporary scales, then revisits only
   scales to align their range. This avoids rereading and rerotating the full
   operand. Worth an isolated RTX candidate; no speedup claimed from inspection.
3. **MS-EDEN backward:** demonstrated sample-mean improvement, but considerably
   noisier single-step gradients than MXFP8. Requires multi-seed transformer
   trials, graph-safe RNG, and performance qualification before public adoption.
4. **Hybrid forward/backward:** experimentally works without replacing our
   backward contract. Prefer borrowing a measured forward improvement over
   assuming the entire external recipe must be adopted together.

Remaining integration risks include host-generated seeds and native launches
without an explicit PyTorch stream. A side-stream smoke alone is not proof
of race freedom. Runtime-probe results are recorded separately.

### Reproduced graph-capture blocker

`benchmarks/probe_quartet2_runtime.py` successfully captured upstream forward
and backward, but both gradients were identical on repeated replay. More
importantly, doubling the input in place before replay left the forward
output unchanged; comparison with a fresh eager forward gave **50.98% NRMS**.
The side-stream equal-input smoke passed, illustrating why that smoke alone
is insufficient. Logs: `runtime-probe-changed-input.log`.
The same probe with `--mode rtx` correctly changed its output and matched
eager execution exactly (0% NRMS); see `runtime-probe-rtx-control.log`.
Identical repeated gradients are expected for deterministic RTX MXFP8
backward, unlike a claim of fresh stochastic samples.

This upstream build is not qualified for CUDA-graph execution. Missing
current-stream plumbing is a source-level suspect, not yet a demonstrated
complete root-cause fix. Repair stream integration first, then independently
verify stochastic-state advancement and changed inputs/weights before any
graph-replay performance comparison. CPU-generated stochastic seeds are an
additional replay concern even after stream integration is fixed.
