"""Numerically conservative decoder-only transformer for training studies.

The model intentionally keeps its token embedding and vocabulary projection in
BF16 while replacing only the attention and MLP projections with RTX linear
frontends.  Every quantized sublayer is bounded by FP32-reduction RMSNorms and
attention applies QK normalization before RoPE.  This makes the model useful as
a convergence test without quietly changing the precision of unrelated paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


LinearPrecision = Literal["bf16", "mxfp8", "nvfp4"]


@dataclass(frozen=True, slots=True)
class LinearSpec:
    """Precision and runtime policy for transformer-internal projections."""

    precision: LinearPrecision = "bf16"
    autotune: str | bool | None = "cache"
    mxfp8_backend: str = "auto"
    nvfp4_scaling: str = "delayed"
    nvfp4_backend: str = "auto"

    def __post_init__(self) -> None:
        if self.precision not in ("bf16", "mxfp8", "nvfp4"):
            raise ValueError("linear precision must be bf16, mxfp8, or nvfp4")
        if self.mxfp8_backend not in ("auto", "fused", "materialized"):
            raise ValueError("MXFP8 backend must be auto, fused, or materialized")
        if self.nvfp4_scaling not in ("delayed", "current", "regional", "block"):
            raise ValueError(
                "NVFP4 scaling must be delayed, current, regional, or block"
            )
        if self.nvfp4_backend not in ("auto", "fused", "materialized"):
            raise ValueError("NVFP4 backend must be auto, fused, or materialized")


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Configuration for :class:`DecoderOnlyTransformer`.

    Defaults describe the short TinyStories convergence model.  With an 8,192
    token vocabulary it has roughly 53 million parameters.  A microbatch of
    three 512-token sequences presents M=1536 to every internal linear.
    """

    vocab_size: int = 8_192
    max_seq_len: int = 512
    num_layers: int = 8
    hidden_size: int = 768
    intermediate_size: int = 1_536
    num_attention_heads: int = 12
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 10_000.0
    attention_dropout: float = 0.0
    embedding_scale: bool = True
    tie_word_embeddings: bool = True
    qk_norm: bool = True
    post_sublayer_norm: bool = True
    gradient_checkpointing: bool = False
    initializer_range: float = 0.02
    residual_init_multiplier: float = 2.0
    dtype: torch.dtype = torch.bfloat16
    linear: LinearSpec = field(default_factory=LinearSpec)

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.rms_norm_eps <= 0.0 or self.rope_theta <= 0.0:
            raise ValueError("normalization epsilon and RoPE theta must be positive")
        if self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be positive")
        if self.residual_init_multiplier <= 0.0:
            raise ValueError("residual_init_multiplier must be positive")
        if self.linear.precision != "bf16" and self.dtype is not torch.bfloat16:
            raise TypeError("MXFP8/NVFP4 transformer projections require BF16 weights")
        if self.linear.precision == "mxfp8":
            if self.hidden_size % 32 or self.intermediate_size % 32:
                raise ValueError("MXFP8 transformer widths must be divisible by 32")
        if self.linear.precision == "nvfp4":
            if self.hidden_size % 64 or self.intermediate_size % 64:
                raise ValueError("NVFP4 transformer widths must be divisible by 64")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def residual_initializer_range(self, depth: int) -> float:
        """Residual-projection standard deviation at one-based layer depth."""

        if not 1 <= depth <= self.num_layers:
            raise ValueError(
                f"layer depth must be in [1, {self.num_layers}], got {depth}"
            )
        return self.initializer_range / math.sqrt(
            self.residual_init_multiplier * depth
        )


class FP32RMSNorm(nn.Module):
    """RMSNorm whose reduction and learned scale are applied in FP32."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1.0e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        normalized = x_fp32 * torch.rsqrt(
            x_fp32.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(dtype=x.dtype)


class QKRMSNorm(nn.Module):
    """Separate FP32-reduction RMSNorms for query and key head channels."""

    def __init__(
        self,
        head_dim: int,
        *,
        eps: float = 1.0e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.q_weight = nn.Parameter(torch.ones(head_dim, device=device, dtype=dtype))
        self.k_weight = nn.Parameter(torch.ones(head_dim, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)

    def _normalize(self, x: Tensor, weight: Tensor) -> Tensor:
        x_fp32 = x.float()
        normalized = x_fp32 * torch.rsqrt(
            x_fp32.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * weight.float()).to(dtype=x.dtype)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._normalize(query, self.q_weight),
            self._normalize(key, self.k_weight),
        )


class RotaryEmbedding(nn.Module):
    """Fixed FP32 RoPE table with FP32 application and activation downcast."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        *,
        theta: float,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        frequency_ids = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        inverse_frequency = theta ** (-frequency_ids / head_dim)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def _apply(self, fn):
        # Module-wide BF16 casts must not silently lower the RoPE table. Apply
        # the requested device transform, then restore its numerical dtype.
        super()._apply(fn)
        self.cos = self.cos.float()
        self.sin = self.sin.float()
        return self

    def values(self, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x_fp32 = x.float()
    even, odd = x_fp32[..., 0::2], x_fp32[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).flatten(-2)
    return rotated.to(dtype=x.dtype)


def _make_linear(
    in_features: int,
    out_features: int,
    *,
    spec: LinearSpec,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> nn.Module:
    if spec.precision == "bf16":
        return nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
    if spec.precision == "mxfp8":
        from ..fp8 import MXFP8Linear

        return MXFP8Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=dtype,
            autotune=spec.autotune,
            backend=spec.mxfp8_backend,
        )
    from ..fp4 import NVFP4Linear

    return NVFP4Linear(
        in_features,
        out_features,
        bias=False,
        device=device,
        dtype=dtype,
        autotune=spec.autotune,
        scaling=spec.nvfp4_scaling,
        backend=spec.nvfp4_backend,
    )


def _initialize_weight(module: nn.Module, std: float) -> None:
    weight = getattr(module, "weight", None)
    if weight is not None:
        nn.init.trunc_normal_(weight, mean=0.0, std=std)


class CausalSelfAttention(nn.Module):
    """Fused-QKV causal attention with QK normalization before FP32 RoPE."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        layer_depth: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.qkv_proj = _make_linear(
            config.hidden_size,
            3 * config.hidden_size,
            spec=config.linear,
            device=device,
            dtype=config.dtype,
        )
        self.out_proj = _make_linear(
            config.hidden_size,
            config.hidden_size,
            spec=config.linear,
            device=device,
            dtype=config.dtype,
        )
        self.qk_norm = (
            QKRMSNorm(
                config.head_dim,
                eps=config.rms_norm_eps,
                device=device,
                dtype=config.dtype,
            )
            if config.qk_norm
            else None
        )
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.max_seq_len,
            theta=config.rope_theta,
            device=device,
        )
        self.initializer_range = config.initializer_range
        self.residual_initializer_range = config.residual_initializer_range(layer_depth)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _initialize_weight(self.qkv_proj, self.initializer_range)
        _initialize_weight(self.out_proj, self.residual_initializer_range)
        if self.qk_norm is not None:
            self.qk_norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        query, key, value = (
            self.qkv_proj(x)
            .reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        if self.qk_norm is not None:
            query, key = self.qk_norm(query, key)
        cos, sin = self.rope.values(seq_len)
        query = _apply_rope(query, cos, sin)
        key = _apply_rope(key, cos, sin)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
        return self.out_proj(attended)


class SwiGLU(nn.Module):
    """SwiGLU with a fused gate/up projection and low-precision linear policy."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        layer_depth: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = _make_linear(
            config.hidden_size,
            2 * config.intermediate_size,
            spec=config.linear,
            device=device,
            dtype=config.dtype,
        )
        self.down_proj = _make_linear(
            config.intermediate_size,
            config.hidden_size,
            spec=config.linear,
            device=device,
            dtype=config.dtype,
        )
        self.initializer_range = config.initializer_range
        self.residual_initializer_range = config.residual_initializer_range(layer_depth)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _initialize_weight(self.gate_up_proj, self.initializer_range)
        _initialize_weight(self.down_proj, self.residual_initializer_range)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    """Pre/post-normalized attention and feed-forward residual block."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        layer_depth: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        norm_kwargs = {
            "eps": config.rms_norm_eps,
            "device": device,
            "dtype": config.dtype,
        }
        self.input_layernorm = FP32RMSNorm(config.hidden_size, **norm_kwargs)
        self.self_attn = CausalSelfAttention(
            config, layer_depth=layer_depth, device=device
        )
        self.post_attention_layernorm = (
            FP32RMSNorm(config.hidden_size, **norm_kwargs)
            if config.post_sublayer_norm
            else nn.Identity()
        )
        self.pre_feedforward_layernorm = FP32RMSNorm(
            config.hidden_size, **norm_kwargs
        )
        self.mlp = SwiGLU(config, layer_depth=layer_depth, device=device)
        self.post_feedforward_layernorm = (
            FP32RMSNorm(config.hidden_size, **norm_kwargs)
            if config.post_sublayer_norm
            else nn.Identity()
        )

    def reset_parameters(self) -> None:
        self.input_layernorm.reset_parameters()
        self.self_attn.reset_parameters()
        if isinstance(self.post_attention_layernorm, FP32RMSNorm):
            self.post_attention_layernorm.reset_parameters()
        self.pre_feedforward_layernorm.reset_parameters()
        self.mlp.reset_parameters()
        if isinstance(self.post_feedforward_layernorm, FP32RMSNorm):
            self.post_feedforward_layernorm.reset_parameters()

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class DecoderOnlyTransformer(nn.Module):
    """Causal language model with BF16 boundaries and configurable RTX linears."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=config.dtype,
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, layer_depth=index + 1, device=device)
            for index in range(config.num_layers)
        )
        self.final_norm = FP32RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=config.dtype,
        )
        # The vocabulary projection deliberately remains BF16 for every mode.
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=config.dtype,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._embedding_multiplier = (
            math.sqrt(config.hidden_size) if config.embedding_scale else 1.0
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(
            self.embed_tokens.weight,
            mean=0.0,
            std=self.config.initializer_range,
        )
        for layer in self.layers:
            layer.reset_parameters()
        self.final_norm.reset_parameters()
        if not self.config.tie_word_embeddings:
            nn.init.trunc_normal_(
                self.lm_head.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds configured maximum "
                f"{self.config.max_seq_len}"
            )
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self._embedding_multiplier
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)

    def num_parameters(self, *, exclude_embeddings: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if exclude_embeddings:
            total -= self.embed_tokens.weight.numel()
        return total


def causal_lm_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Compute token cross entropy with the numerically sensitive path in FP32."""

    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match every non-vocabulary logits dimension")
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


__all__ = [
    "CausalSelfAttention",
    "DecoderConfig",
    "DecoderLayer",
    "DecoderOnlyTransformer",
    "FP32RMSNorm",
    "LinearPrecision",
    "LinearSpec",
    "QKRMSNorm",
    "SwiGLU",
    "causal_lm_loss",
]
