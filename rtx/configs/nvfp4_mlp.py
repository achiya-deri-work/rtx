"""CPU-safe numerical and scheduling contract for the recomputed SiTU MLP."""
from dataclasses import asdict, dataclass, field
import math

from .nvfp4 import NVFP4DynamicConfig, NVFP4GemmConfig, NVFP4QuantConfig, NVFP4Problem
from ..kernels.mxfp8_bwd import MXFP8BwdConfig


NVFP4_MLP_REVISION = 1


def _projection():
    return NVFP4DynamicConfig(x_scale_region_rows=8, weight_scale_region_rows=8)


def _down_projection():
    return NVFP4DynamicConfig(x_scale_region_rows=8,weight_scale_region_rows=8,
                             region_ownership='cta_cached')


@dataclass(frozen=True)
class NVFP4MLPConfig:
    """SiTU numerical policy and independent forward/backward schedules.

    Gate/up and down use current regional FP32 outer scales. ``beta`` and
    ``linear_beta`` are numerical choices, never autotuned. The gate/up GEMM
    applies outer scales, SiTU, and the product in FP32, emitting an H-wide
    packed MXFP8 connection by default (``hidden_dtype='mxfp8'``). Down's
    producer decodes it into registers while preparing NVFP4, never BF16 GMEM.
    ``hidden_dtype='bf16'`` explicitly selects an uncompressed hidden boundary.
    Neither format is retained by autograd. Backward recomputes the same
    NVFP4 projection and hidden policy, then uses MXFP8 GEMMs/FP32 accumulators.
    """
    gate_up: NVFP4DynamicConfig = field(default_factory=_projection)
    down: NVFP4DynamicConfig = field(default_factory=_down_projection)
    gate_backward: MXFP8BwdConfig | None = None
    down_backward: MXFP8BwdConfig | None = None
    beta: float = 4.0
    linear_beta: float = 25.0
    backward_overlap: bool = False
    glu_epilogue: str = "registers"
    glu_store_vec: int = 2
    glu_scratch_swizzle: str = "none"
    hidden_dtype: str = "mxfp8"
    hidden_quant_vec: int = 4

    def __post_init__(self):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in (self.beta, self.linear_beta)):
            raise ValueError("SiTU beta and linear_beta must be finite and positive")
        if any(not isinstance(c, NVFP4DynamicConfig) for c in (self.gate_up, self.down)):
            raise TypeError("gate_up/down must be NVFP4DynamicConfig")
        if any(c is not None and not isinstance(c, MXFP8BwdConfig)
               for c in (self.gate_backward, self.down_backward)):
            raise TypeError("backward configs must be MXFP8BwdConfig or None")
        if type(self.backward_overlap) is not bool:
            raise TypeError("backward_overlap must be bool")
        if self.glu_epilogue not in ("registers","exchange"):
            raise ValueError("glu_epilogue must be registers or exchange")
        if self.glu_store_vec not in (1,2,4,8):
            raise ValueError("glu_store_vec must be 1, 2, 4, or 8")
        if self.glu_scratch_swizzle not in ("none","xor_16b"):
            raise ValueError("unsupported GLU scratch swizzle")
        if self.hidden_dtype not in ('bf16','mxfp8'):
            raise ValueError("hidden_dtype must be bf16 or mxfp8")
        if self.hidden_quant_vec not in (2,4,8):
            raise ValueError("hidden_quant_vec must be 2, 4, or 8")

    def rejection(self, m, k, h, o):
        for name, c, p in (("gate_up", self.gate_up, NVFP4Problem(m,2*h,k)),
                           ("down", self.down, NVFP4Problem(m,o,h))):
            reason = c.rejection(p)
            if reason:
                return f"{name}: {reason}"
            if not c.jit_row_region or c.quant_launches != "dual":
                return f"{name}: SiTU MLP requires JIT regional dual quantization"
            if c.programmatic_dependent_launch:
                return f"{name}: SiTU MLP PDL handoff is not qualified"
        g = self.gate_up.gemm
        if g.scale_layout != "row_major" or g.epilogue != "direct":
            return "gate_up requires row-major scales and a direct GLU epilogue"
        if g.regional_epilogue_schedule != "mma":
            return "gate_up GLU currently owns the MMA-warp epilogue"
        if self.hidden_dtype=='mxfp8':
            if self.glu_epilogue!='registers':
                return "packed hidden requires register GLU with an FP32 quantization exchange"
        if self.glu_epilogue=='exchange' or self.hidden_dtype=='mxfp8':
            from ..kernels.mxfp8 import SM120_SMEM_CAPACITY_BYTES,SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES
            operands=g.stages*(g.tile_m+g.tile_n)*g.tile_k//2
            scales=g.stages*((g.tile_m+127)//128+(g.tile_n+127)//128)*128*(g.tile_k//16)
            extra=g.tile_m*g.tile_n*(2 if self.hidden_dtype=='mxfp8' else 3)
            if operands+scales+extra+SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES>SM120_SMEM_CAPACITY_BYTES:
                return "joint SiTU exchange scratch exceeds SM120 shared memory"
        from ..kernels.mxfp8 import MXFP8Problem
        for name, c, p in (("gate_backward",self.gate_backward,MXFP8Problem(m,2*h,k)),
                           ("down_backward",self.down_backward,MXFP8Problem(m,o,h))):
            if c is not None:
                reason = c.implementation_rejection(p)
                if reason:
                    return f"{name}: {reason}"
        return None

    def to_dict(self):
        return {"revision": NVFP4_MLP_REVISION, **asdict(self)}

    def to_json(self):
        import json
        return json.dumps(self.to_dict(),indent=2,sort_keys=True)

    @classmethod
    def from_json(cls,text):
        import json
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if values.pop("revision", None) != NVFP4_MLP_REVISION:
            raise ValueError("unsupported NVFP4 MLP revision")
        from ..bwd_autotune import bwd_config_from_dict
        for name in ("gate_up", "down"):
            v = dict(values[name])
            v["quant"] = NVFP4QuantConfig(**v["quant"])
            v["gemm"] = NVFP4GemmConfig(**v["gemm"])
            values[name] = NVFP4DynamicConfig(**v)
        for name in ("gate_backward", "down_backward"):
            if values.get(name) is not None:
                values[name] = bwd_config_from_dict(values[name])
        return cls(**values)
