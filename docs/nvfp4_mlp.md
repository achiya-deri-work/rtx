# NVFP4 SiTU MLP with recomputed MXFP8 backward

`rtx.NVFP4MLP` and `rtx.nvfp4_mlp` are experimental feature APIs for a bias-free
GLU feed-forward block. Parameters and input/output are BF16. Both forward
projections use NVFP4; backward GEMMs use MXFP8 with FP32 accumulators. SiTU,
its product and its analytic derivatives are evaluated in FP32.

```python
import torch
import rtx

mlp = rtx.NVFP4MLP(
    in_features=768,
    hidden_features=1536,       # H after gating, not the packed 2H width
    out_features=768,          # defaults to in_features
    device="cuda",
    activation="situ",        # currently the only NVFP4 MLP activation
    autotune="balanced",      # off, cache, balanced; default is off
    tuning_policy=rtx.AutotunePolicy(max_trials=16, max_seconds=60),
    autotune_cache_dir="autotune_results/situ",
)
compiled = torch.compile(mlp, fullgraph=True, dynamic=False)  # default mode
x = torch.randn(4, 512, 768, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
y = compiled(x)               # first input tunes; later calls reuse its plan
y.float().square().mean().backward()
```

Aligned shapes are the primary performance target. Half-128-aligned shapes and
other logical tails remain supported through the existing masked/native kernel
contracts, without materializing BF16 padding. As with the linear APIs, native
quantized storage may contain zero-filled tail elements.

## Numerical contract

The activation follows Moonshot's mathematical
[Kimi K3 SituAndMul definition](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py):

```text
a = beta * tanh(gate / beta) * sigmoid(gate)
b = linear_beta * tanh(up / linear_beta)
hidden = a * b
```

The sigmoid receives the original gate, not the soft-capped gate. The defaults
`beta=4` and `linear_beta=25` preserve this project's training recipe; they are
explicit settings, not a claim that every Kimi configuration uses these caps.
This is an independently implemented kernel, not vendored upstream model code.

`gate_up_weight` has shape `[2H,K]` in ordinary `[gate; up]` order. Its native
tensor-core loads use a logical interleaving layout. FP32 gate/up accumulators
are multiplied by their FP32 outer factors **before** SiTU. There is no BF16
rounding of gate/up before activation. This intentionally differs from composing
two BF16-output linear calls followed by a pointwise activation.

Gate/up and down use current JIT row-region FP32 outer scales, E2M1 data and E4M3
16-value block scales. Caps, region sizes and block-scale selection are numerical
policy, not silent speed-tuning choices. Configure them explicitly:

```python
from dataclasses import replace

config = rtx.NVFP4MLPConfig(beta=4.0, linear_beta=25.0)
config = replace(config,
    gate_up=replace(config.gate_up,
        x_scale_region_rows=8, weight_scale_region_rows=4),
)
mlp = rtx.NVFP4MLP(768, 1536, config=config, device="cuda")
```

The default H-wide connection is packed E4M3/E8M0 MXFP8, matching the hidden
storage policy of `MXFP8MLP`. SiTU's FP32 products are quantized directly in its
epilogue in 32-value blocks. Down preparation decodes them into BF16 registers,
observes current regional amax, and prepares NVFP4 operands; it never writes a
BF16 hidden activation to GMEM in forward. This does not replace NVFP4's FP32
outer factors with E8M0: E8M0 belongs only to the MXFP8 intermediate format.

`replace(config, hidden_dtype="bf16")` explicitly selects an uncompressed H-wide
boundary, avoiding the additional MXFP8 rounding. This is a numerical policy,
not a timing-only autotuning coordinate. Both formats are tested independently.

`gate_up` and `down` are independent `NVFP4DynamicConfig` schedules. Existing
adaptive block policies can be selected in their `quant` configuration. The
current MLP does not accept tensor-delayed, tensor-current, block-only, or
experimental FP4-backward policies; those remain separate linear features.

## Recomputation and intermediate storage

Autograd saves only X and the two BF16 weights. It retains no gate/up projection,
hidden activation, packed quantized operand, outer-scale array, or RNG state.
Runtime selection additionally saves an immutable **CPU metadata token** so
outstanding forwards cannot be retargeted by winner refinement or eviction.

Forward fuses gate/up, outer-scale application, SiTU, multiplication and hidden
MXFP8 quantization on chip, writing only an H-wide packed connection rather than
a 2H-wide projection. Both forward GEMMs use NVFP4. This materialized route has
two quantization/preparation launches and two GEMM launches. It is **not a claim
of strict two-CUDA-launch parity** with the on-chip `MXFP8MLP` family: that would
also require new in-GEMM NVFP4 operand preparation implementations.

Backward computes down dX using MXFP8, recomputes the original NVFP4 gate/up
projection and current scales, differentiates SiTU on chip, and jointly emits
BF16 dGate/dUp and the recreated packed hidden tensor. When down dW is required,
the hidden is decoded to BF16 in backward before independent leading-axis
MXFP8 quantization, matching the selected forward hidden policy. MXFP8 computes the required
weight/input gradients. Recomputing with MXFP8 projections instead would be a
different numerical operation and is not used here. Long dW reductions retain
the existing FP32 split-workspace options. Only first-order gradients are
supported. Selective frozen-parameter/input gradient requests are supported.

No user-managed scratch, streams or cleanup scopes are required. The existing
bounded caches and automatic compiled/graph ownership retain active resources.
CUDA graphs are optional and caller-controlled. A manually captured graph must
be warmed first; cold tuning/capture is rejected explicitly.

## Autotuning and controls

The implementation uses `rtx.autotune`'s discrete adapter, persistent residual
store, budgeted UCB portfolio and coordinate/model-guided search. It measures
complete default-compiled forward/backward workloads, including recomputation,
against an independent quantization-aware FP32 oracle before admitting latency.

Independent forward schedules expose existing GEMM geometry, buffering, TMA and
scale transport, register budgets, persistence, locality and quantizer schedules.
SiTU adds register versus shared-exchange BF16 epilogues, exchange packet width,
packed hidden quantization vectors and scratch swizzle. Packed hidden uses an
FP32 exchange for quantization and direct register derivative stores. Down
preparation exposes warp, CTA and register-cached CTA ownership rather than
forcing one decoding schedule. Backward exposes both MXFP8 search spaces, structural
dX/dW families, split reductions and optional overlap. Inactive coordinates are
deduplicated. Existing quarantined kernel combinations remain rejected.
Small training budgets include complete backward-family candidates for both
projections, rather than changing one backward leg while another dominates.
Half-aligned defaults use a legal materialized backward family when available;
they do not relax native reduction-axis alignment or fall back to BF16 padding.

```python
mlp.autotune_(example_x,
    tuning_policy=rtx.AutotunePolicy(max_trials=32, max_seconds=120),
    output_dir="autotune_results/situ", phase="train")
# Install policies before compiling/recapturing:
mlp.configure_shape_(example_x, config=explicit_config)
starting_or_explicit_config = mlp.policy_for(example_x)
```

Shape policies are bounded, retain metadata only and do not belong in the
state_dict. First-hit identities distinguish input batch/leading dimensions,
hidden/output widths, device/runtime/source, numerical policy, inference versus
training, and optional `MLPExecutionMix` workloads. Runtime winner writes are
atomic and trial evidence is written as independent residual records.

`autotune="cache"` loads compatible winners without searching and otherwise uses
the initial legal config. Call `require_cached_()` before compilation to reject
misses instead. `export_tuning_spec()` exports exact metadata for preparation.
`policy_for()` is introspection of explicit starting/shape policies, not a lookup
or tuning operation; runtime-selected winners are recorded in the cache.
The separate `MLPCandidate` / qualified-plan deployment helpers remain specific
to `MXFP8MLP`; use this module's own cache/export controls for NVFP4.

## Conversion and unsupported features

`NVFP4MLP.from_linears(gate, up, down, config=...)` copies three bias-free BF16
linears, preserves training/gradient flags, and consumes no RNG. Gate/up must
share shape and requires_grad because they become one packed parameter. Source
modules are unchanged; create a new optimizer for the new parameters.

Dropout, residual addition, norms, projection biases, noncontiguous input storage
and higher-order gradients are outside this operator. SiTU caps must be positive
and finite. Parameters initialize with 0.02 truncated-normal weights. The module
does not independently add model-depth residual initialization scaling.

Use `benchmarks/benchmark_nvfp4_mlp.py` for matched timing and quantitative
evidence. Compilation/tuning are excluded from steady-state timing. Performance
results must name their shape, schedule, precision policy and checkpointing
baseline; tensor-level correctness is not a convergence result.
See the [5090 qualification and measured limits](nvfp4_mlp_qualification_20261103.md)
before treating this experimental feature as a production-model qualification.
