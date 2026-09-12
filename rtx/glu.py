"""Experimental forward-only fused MXFP8 gate/up projection.

The packed weight is [gate rows; up rows], with no bias. Inputs are BF16;
both projections, activation and multiplication happen on chip. Quantization
can be CTA-local or materialized once for reuse, as selected by configuration.
This is not yet a training replacement: backward is deliberately unsupported.
"""
import json

import torch

from .kernels.mxfp8 import MXFP8FwdConfig, fwd_config_to_dict

from .configs.glu import DEFAULT_GLU_CONFIG, MXFP8GLUPrequantConfig


@torch.compiler.assume_constant_result
def _config_key(config):
    # Include frontend lowering revision in the FX cache key: registering a new
    # lowering alone does not invalidate an old cached opaque wrapper.
    if not isinstance(config,(MXFP8FwdConfig,MXFP8GLUPrequantConfig)):
        raise TypeError("config must be MXFP8GLUConfig or MXFP8GLUPrequantConfig")
    return json.dumps({**fwd_config_to_dict(config),"_glu_frontend_revision":3}, sort_keys=True)


def _validate(x, weight, activation, output_format):
    if activation not in ("silu", "gelu", "relu"):
        raise ValueError("activation must be silu, gelu, or relu")
    if output_format not in ("bf16", "mxfp8"):
        raise ValueError("output_format must be bf16 or mxfp8")
    if x.ndim < 2 or weight.ndim != 2:
        raise ValueError("expected X[..., M, K] and W[2H, K]")
    if x.shape[-1] != weight.shape[1] or weight.shape[0] % 2:
        raise ValueError("weight must contain matching [gate; up] rows")
    if x.numel() == 0 or weight.numel() == 0:
        raise ValueError("GLU dimensions must be positive")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("GLU inputs must be BF16")
    if x.device.type != "cuda" or weight.device != x.device:
        raise ValueError("GLU inputs must share one CUDA device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("GLU currently requires contiguous BF16 inputs")


@torch.library.custom_op("rtx::mxfp8_glu", mutates_args=(),device_types="cuda")
def _glu_op(x: torch.Tensor, weight: torch.Tensor, activation: str,
            output_format: str, config_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    _validate(x, weight, activation, output_format)
    from .runtime import validate_runtime_environment
    validate_runtime_environment()
    if torch.cuda.get_device_capability(x.device)[0] != 12:
        raise RuntimeError("MXFP8 GLU requires SM12x")
    m, k = x.numel() // x.shape[-1], x.shape[-1]
    h = weight.shape[0] // 2
    width = h if output_format == "bf16" else (h + 31) // 32 * 32
    out = torch.empty((m, width), device=x.device,
                      dtype=torch.bfloat16 if output_format == "bf16" else torch.float8_e4m3fn)
    scales = torch.empty((m, 0 if output_format == "bf16" else (h+31)//32), device=x.device, dtype=torch.uint8)
    from ._glu_inductor import LAUNCHERS
    LAUNCHERS[(activation,output_format,config_key)](
        x.view(m,k),weight,out=out,scales=scales)
    return out, scales


@_glu_op.register_fake
def _glu_fake(x, weight, activation, output_format, config_key):
    _validate(x, weight, activation, output_format)
    h = weight.shape[0] // 2
    m = x.numel() // x.shape[-1]
    width = h if output_format == "bf16" else (h+31)//32*32
    return (x.new_empty((m,width), dtype=torch.bfloat16 if output_format == "bf16" else torch.float8_e4m3fn),
            x.new_empty((m,0 if output_format == "bf16" else (h+31)//32), dtype=torch.uint8))


def mxfp8_glu(x: torch.Tensor, weight: torch.Tensor, *, activation: str = "silu",
               output_format: str = "bf16", config: MXFP8FwdConfig | MXFP8GLUPrequantConfig = DEFAULT_GLU_CONFIG):
    """Compute ``act(X @ W_gate.T) * (X @ W_up.T)`` without a 2H output.

    ``weight`` is contiguous BF16 [2H,K], gate first, up second. Leading X
    dimensions are preserved. ``gelu`` uses erf, not the tanh approximation.
    Both projections use dynamic E4M3/E8M0 MXFP8 with FP32 accumulation and an
    FP32 activation/product. BF16 output rounds once, after the product.
    ``output_format='mxfp8'`` instead returns a TorchAO MXTensor with row-major
    E8M0 scales per 32 output values and zero-filled tail storage.

    Experimental forward only: use no_grad/inference_mode. Pass a configuration
    selected by ``rtx.autotune.glu.tune_mxfp8_glu``; tuning never runs implicitly
    in a compiled graph. This is a gate/up projection, not a complete FFN.
    MXFP8GLUConfig selects CTA-local quantization; MXFP8GLUPrequantConfig selects
    quantize-once operand reuse. Neither backend caches a quantized weight
    across invocations. Both keep the GLU intermediate off global memory.
    """
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise RuntimeError("MXFP8 GLU backward is not implemented; use torch.no_grad() for forward evaluation")
    key = _config_key(config)
    out, scales = _glu_op(x, weight, activation, output_format, key)
    shape = (*x.shape[:-1], weight.shape[0]//2)
    if output_format == "bf16":
        return out.view(shape)
    from .formats.mxfp8 import make_mxfp8_tensor
    return make_mxfp8_tensor(out, scales.view(torch.float8_e8m0fnu), shape)


def mxfp8_glu_packed(x: torch.Tensor, weight: torch.Tensor, *,
                     activation: str = "silu", config=DEFAULT_GLU_CONFIG):
    """Tensor-only MXFP8 output for compiler/custom-kernel composition.

    Returns ``(data, scales)`` with leading X dimensions preserved. Data is
    E4M3 with width ``ceil(H/32)*32``; scales are E8M0 with width ``ceil(H/32)``.
    H is ``weight.shape[0]//2``. Tail data is zero. No TorchAO wrapper is built,
    no dequantization occurs, and the numerical contract matches mxfp8_glu.
    Forward-only; call under no_grad/inference_mode for trainable parameters.
    """
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise RuntimeError("MXFP8 GLU backward is not implemented; use torch.no_grad() for forward evaluation")
    out, scales = _glu_op(x, weight, activation, "mxfp8", _config_key(config))
    blocks = (weight.shape[0]//2 + 31)//32
    return (out.view(*x.shape[:-1], blocks*32),
            scales.view(torch.float8_e8m0fnu).view(*x.shape[:-1], blocks))


class MXFP8GLU(torch.nn.Module):
    """Experimental bias-free gate/up module; **backward is not implemented**.

    Owns one BF16 parameter ``weight[2*hidden_features, in_features]`` (gate
    first). Supports SiLU, exact GELU, and ReLU. ``output_format`` is ``bf16``
    or ``mxfp8``; ``packed_output=True`` returns the tensor-only MXFP8 pair.
    Both operands are dynamically quantized on every call, including replay.

    Call ``autotune_(example_x, max_trials=..., time_budget_s=...)`` before
    torch.compile/capture. It selects one config for the representative shape;
    it does not launch searches from forward or claim that config is optimal
    for other shapes. Reports remain in the standard residual dataset.
    State dicts contain weights, not hardware-specific tuning decisions.
    Loading weights into a newly constructed module requires retuning it.
    """

    def __init__(self, in_features, hidden_features, *, activation="silu",
                 output_format="bf16", packed_output=False,
                 config=MXFP8GLUPrequantConfig(), device=None, dtype=torch.bfloat16):
        super().__init__()
        if type(in_features) is not int or type(hidden_features) is not int or min(in_features,hidden_features)<=0:
            raise ValueError("in_features and hidden_features must be positive integers")
        if activation not in ("silu","gelu","relu"):
            raise ValueError("activation must be silu, gelu, or relu")
        if output_format not in ("bf16","mxfp8"):
            raise ValueError("output_format must be bf16 or mxfp8")
        if packed_output and output_format!="mxfp8":
            raise ValueError("packed_output requires output_format='mxfp8'")
        if not isinstance(config,(MXFP8FwdConfig,MXFP8GLUPrequantConfig)):
            raise TypeError("config must be MXFP8GLUConfig or MXFP8GLUPrequantConfig")
        if dtype != torch.bfloat16:
            raise TypeError("MXFP8GLU parameters must be BF16")
        self.in_features, self.hidden_features = in_features, hidden_features
        self.activation, self.output_format = activation, output_format
        self.packed_output, self.config = packed_output, config
        self.tuning_result = None
        self.weight = torch.nn.Parameter(torch.empty(2*hidden_features,in_features,device=device,dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.weight,mean=0.,std=.02,a=-.04,b=.04)

    def forward(self, x):
        if self.packed_output:
            return mxfp8_glu_packed(x,self.weight,activation=self.activation,config=self.config)
        return mxfp8_glu(x,self.weight,activation=self.activation,
                        output_format=self.output_format,config=self.config)

    @torch.no_grad()
    def autotune_(self, example_x, **kwargs):
        """Tune explicitly, update config, and return self; inspect tuning_result."""
        if torch.compiler.is_compiling() or (example_x.is_cuda and torch.cuda.is_current_stream_capturing()):
            raise RuntimeError("autotune_ must run before compilation and CUDA graph capture")
        from .autotune.glu import tune_mxfp8_glu
        result = tune_mxfp8_glu(example_x,self.weight,activation=self.activation,
                                output_format=self.output_format,**kwargs)
        self.config, self.tuning_result = result.config, result
        return self

    @classmethod
    def from_linears(cls, gate, up, **kwargs):
        """Copy two matching bias-free BF16 nn.Linear weights; originals unchanged."""
        if not isinstance(gate,torch.nn.Linear) or not isinstance(up,torch.nn.Linear):
            raise TypeError("gate and up must be nn.Linear modules")
        if gate.bias is not None or up.bias is not None:
            raise ValueError("MXFP8GLU does not support bias")
        if gate.weight.shape != up.weight.shape or gate.weight.device != up.weight.device:
            raise ValueError("gate and up must have matching shapes and devices")
        if gate.weight.dtype != torch.bfloat16 or up.weight.dtype != torch.bfloat16:
            raise TypeError("gate and up weights must be BF16")
        result = cls(gate.in_features,gate.out_features,device=gate.weight.device,**kwargs)
        with torch.no_grad():
            result.weight[:gate.out_features].copy_(gate.weight)
            result.weight[gate.out_features:].copy_(up.weight)
        result.train(gate.training)
        result.weight.requires_grad_(gate.weight.requires_grad or up.weight.requires_grad)
        return result

    def extra_repr(self):
        return (f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
                f"activation={self.activation!r}, output_format={self.output_format!r}, "
                f"packed_output={self.packed_output}, bias=False, forward_only=True")


from ._glu_inductor import register as _register_inductor
_register_inductor()


def _clear_runtime_caches():
    from ._glu_inductor import LAUNCHERS
    count=0
    for launcher in LAUNCHERS.values():
        count+=len(launcher.runners)
        launcher.runners.clear()
    return {"runners":count}
