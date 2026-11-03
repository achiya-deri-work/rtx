# Home dual-5090 SiTU precision comparison

Requested comparison: identical NVFP4 forward with either stable MXFP8
backward or experimental MS-EDEN NVFP4 backward. This is a single-seed
training experiment, not proof of a general speed or convergence advantage.

## Recipe

- Decoder: 16 layers, hidden width 1024, MLP width 2816, 16 attention heads;
  approximately 206M parameters, tied 257-token byte vocabulary.
- Batch 16, sequence 512; 12,288 updates (100,663,296 byte tokens) per policy.
- TinyStories first 100M corpus tokens, fixed disjoint 90/10 train/validation
  ranges; deterministic sampling and seed 941. Token counts are bytes, not BPE.
- BF16 parameters/activations/AdamW moments, FP32 normalization parameters and
  their moments; FP32 normalization, RoPE, residual addition, loss and clipping.
- Flash attention with QK normalization and pre/post sublayer normalization.
- Truncated-normal initialization 0.02, residual projection standard deviation
  `0.02 / sqrt(2 * one_based_depth)`; CPU initialization shared by both modes.
- Exact JIT regional NVFP4 forward; original BF16 operands for backward.
  Experimental full NVFP4 uses the existing paired pipeline, not a silently
  promoted autotuning winner. No new exhaustive tuning; timings describe these
  recorded configurations, not optimal implementations.
- SiTU-GLU: `4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)`, FP32 arithmetic then
  BF16 output. The sigmoid takes the uncapped gate. See the
  [Moonshot reference](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)
  and [K3 cap values](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json).
  This independently implemented formula does not import model remote code.
- AdamW LR 3e-4, 512 warmup steps, cosine decay to 0.1 of peak, betas 0.9/0.95,
  weight decay 0.1 on matrices and zero on norms, gradient norm cap 1.
- `torch.compile(fullgraph=True, dynamic=False)`, CUDA graphs and pointwise
  combo kernels enabled for both policies.

## Execution and evidence

The eight-update smoke pair must finish and pass the matched-provenance
analyzer before the full pair starts. Smoke optimizer state is discarded.
GPU 0 runs NVFP4/MXFP8; GPU 1 runs NVFP4/NVFP4. Hardware assignment is not
crossed over, so small speed differences can include board/thermal differences.
No power limits or unrelated services are changed.

Immutable source snapshot: `/tmp/rtx-precision200m-20261027.oAifI7`, based on
`5b1471b` plus the SiTU decoder/harness and launch script changes. Each run
records source hashes, initial parameter hash, corpus/validation hashes and
projection configurations. Validation every 1024 steps, checkpoint every 4096.
The analyzer reports matched-step loss, observed equal-training-wall and
equal-total-wall comparisons separately; compilation is excluded only from
the training-block clock.

Output: `autotune_reports/precision200m_situ_home_20261027_v2/`.
Launcher log: same path with `.launch.log` appended.
User service: `rtx-precision200m-situ-20261027`.
The original non-v2 launch exited before model startup; its empty logs are
preserved. The user service keeps this run independent of terminal lifetime.

Prelaunch checks: 110 CPU tests passed (one CUDA-only skip, 12 subtests), and
all eight SiTU tests passed on CUDA including compiled forward/backward.
Training outcomes are pending; inspect JSONL result records before claiming
completion or convergence.
