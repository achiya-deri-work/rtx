"""CPU-safe configuration/legality for the experimental paired projection."""
from dataclasses import dataclass, asdict, replace
from ..kernels.mxfp8 import MXFP8Problem, MXFP8FwdConfig, normalize_fwd_config
from .mxfp8 import MXFP8GemmConfig, MXFP8QuantConfig


def cooperative_glu_rejection(problem,config):
    """Conservative rendezvous qualification boundary for cooperative GLU.

    A ragged-M experiment hung on SM120. Keep partial output tiles out of this
    experimental backend until its synchronization has an independent audit;
    ordinary on-chip/materialized paths remain available for these shapes.
    """
    if problem.m%config.tile_m or problem.n%config.tile_n:
        return "cooperative GLU requires full output tiles; ragged synchronization is unqualified"
    return None


@dataclass(frozen=True, slots=True)
class MXFP8GLUGemmConfig(MXFP8GemmConfig):
    glu_hidden_decode: str = "fp32"
    glu_silu_math: str = "division"
    glu_grad_store: str = "direct"
    glu_grad_load: str = "interleaved"
    glu_scratch_swizzle: str = "none"
    stages: int = 1
    epilogue: str = "direct"
    store_vec: int = 1
    glu_epilogue: str = "registers"
    glu_store: str = "direct"
    glu_store_stages: int = 1
    glu_quant_vec: int = 1
    reduction: str = "shuffle"
    scale_rounding: str = "infinity"


@dataclass(frozen=True, slots=True)
class MXFP8GLUPrequantConfig:
    quant: MXFP8QuantConfig = MXFP8QuantConfig()
    gemm: MXFP8GLUGemmConfig = MXFP8GLUGemmConfig()
    backend: str = "prequant"
    quant_launches: str = "dual"
    weight_quant: MXFP8QuantConfig | None = None
    weight_scale_layout: str | None = None

    def resolved_weight_quant(self):
        q = self.weight_quant or self.quant
        return replace(q, scale_layout=self.weight_scale_layout or q.scale_layout)

    def rejection(self, problem, output_format="bf16"):
        from ..fp8 import MXFP8PrequantConfig
        legacy = MXFP8PrequantConfig(quant=self.quant,gemm=self.gemm,
                                    quant_launches=self.quant_launches,weight_quant=self.weight_quant,
                                    weight_scale_layout=self.weight_scale_layout)
        reason = legacy.rejection(problem)
        if reason:
            return reason
        if self.gemm.epilogue != "direct" or self.gemm.glu_epilogue not in ("exchange","registers"):
            return "unsupported prequant GLU epilogue"
        if self.gemm.scale_rounding != "infinity" or self.quant.scale_rounding != "infinity" or self.resolved_weight_quant().scale_rounding != "infinity":
            return "GLU uses round-to-infinity scales"
        g = self.gemm
        if g.glu_hidden_decode not in ("fp32","bf16x2"):
            return "invalid hidden reconstruction arithmetic"
        if g.glu_silu_math not in ("division","shared_reciprocal","approx_reciprocal"):
            return "invalid SiLU arithmetic policy"
        if g.glu_grad_load not in ("interleaved","prefetch","shared") or g.glu_grad_store not in ("direct","exchange"):
            return "invalid GLU derivative transport policy"
        if g.glu_grad_load=="shared" and g.glu_grad_store=="exchange":
            return "gradient store exchange and shared gradient load have overlapping lifetimes"
        if g.glu_scratch_swizzle not in ("none","xor_16b"):
            return "invalid GLU scratch swizzle"
        if g.reduction not in ("shuffle","redux"):
            return "invalid GLU output reduction"
        if g.glu_quant_vec not in (1,2,4,8):
            return "invalid GLU output quantization vector"
        if g.glu_store not in ("direct","tma") or g.glu_store_stages not in (1,2,3,4):
            return "invalid GLU store schedule"
        if g.glu_store == "tma" and (output_format!="bf16" or (problem.n//2)%8):
            return "TMA GLU output requires BF16 and H divisible by 8"
        scratch = g.tile_m*g.tile_n*(0 if output_format=="bf16" and g.glu_epilogue=="registers" else 2 if g.glu_epilogue=="registers" else 4)
        operands = g.stages*(g.tile_m+g.tile_n)*g.tile_k
        scales = g.stages*((g.tile_m+127)//128+(g.tile_n+127)//128)*128*(g.tile_k//32)
        if g.glu_store=="tma":
            scratch += g.tile_m*g.tile_n*g.glu_store_stages
        if operands+scales+scratch+2048>101376:
            return "GLU prequant shared memory exceeds capacity"
        return None


def prequant_glu_config_from_dict(values):
    return MXFP8GLUPrequantConfig(
        quant=MXFP8QuantConfig(**values["quant"]),
        gemm=MXFP8GLUGemmConfig(**values["gemm"]),
        quant_launches=values.get("quant_launches","dual"),
        weight_quant=None if values.get("weight_quant") is None else MXFP8QuantConfig(**values["weight_quant"]),
        weight_scale_layout=values.get("weight_scale_layout"),
    )


@dataclass(frozen=True, slots=True)
class MXFP8GLUConfig(MXFP8FwdConfig):
    glu_hidden_decode: str = "fp32"
    glu_silu_math: str = "division"
    glu_grad_store: str = "direct"
    glu_grad_load: str = "interleaved"
    glu_scratch: str = "dedicated"
    glu_scratch_swizzle: str = "none"
    glu_quant_epilogue: str = "exchange"
    glu_epilogue: str = "exchange"
    glu_store: str = "direct"
    glu_store_stages: int = 1
    glu_quant_vec: int = 1


def normalize_glu_config(base=None, /, **updates):
    if "glu_tile" in updates:
        tile_m,tile_n,atom_m = updates.pop("glu_tile")
        updates.update(tile_m=tile_m,tile_n=tile_n,atom_layout_m=atom_m,
                       atom_layout_n=tile_n//64,mxfp8_stages=1,glu_scratch="reuse_operands",
                       glu_epilogue="registers",glu_quant_epilogue="exchange",
                       cooperative_warps=0,load_engine="scalar",schedule="cooperative",
                       persistent=False,persistent_waves=1,reuse="none",cluster_reuse="none",cluster_size=1)
    values = asdict(base or normalize_fwd_config(tile_m=64))
    hidden_decode = updates.pop("glu_hidden_decode", values.pop("glu_hidden_decode", "fp32"))
    silu_math = updates.pop("glu_silu_math", values.pop("glu_silu_math", "division"))
    grad_store = updates.pop("glu_grad_store", values.pop("glu_grad_store", "direct"))
    grad_load = updates.pop("glu_grad_load", values.pop("glu_grad_load", "interleaved"))
    scratch = updates.pop("glu_scratch", values.pop("glu_scratch", "dedicated"))
    scratch_swizzle = updates.pop("glu_scratch_swizzle", values.pop("glu_scratch_swizzle", "none"))
    quant_epilogue = updates.pop("glu_quant_epilogue", values.pop("glu_quant_epilogue", "exchange"))
    epilogue = updates.pop("glu_epilogue", values.pop("glu_epilogue", "exchange"))
    store = updates.pop("glu_store", values.pop("glu_store", "direct"))
    stages = updates.pop("glu_store_stages", values.pop("glu_store_stages", 1))
    quant_vec = updates.pop("glu_quant_vec", values.pop("glu_quant_vec", 1))
    projection = normalize_fwd_config(MXFP8FwdConfig(**values), **updates)
    return MXFP8GLUConfig(**asdict(projection), glu_hidden_decode=hidden_decode, glu_silu_math=silu_math, glu_grad_store=grad_store, glu_grad_load=grad_load, glu_scratch_swizzle=scratch_swizzle, glu_quant_epilogue=quant_epilogue, glu_scratch=scratch, glu_epilogue=epilogue,glu_store=store,glu_store_stages=stages,glu_quant_vec=quant_vec)


def glu_config_from_dict(values):
    values=dict(values)
    values.pop("_glu_frontend_revision",None)
    return normalize_glu_config(**values)

DEFAULT_GLU_CONFIG = normalize_glu_config()


def glu_rejection(problem: MXFP8Problem, config: MXFP8FwdConfig, output_format="bf16"):
    reason = config.implementation_rejection(problem)
    if reason:
        return reason
    if problem.n % 2:
        return "GLU requires paired gate/up rows"
    epilogue = getattr(config, "glu_epilogue", "exchange")
    if config.glu_hidden_decode not in ("fp32","bf16x2"):
        return "invalid hidden reconstruction arithmetic"
    if config.glu_silu_math not in ("division","shared_reciprocal","approx_reciprocal"):
        return "invalid SiLU arithmetic policy"
    if config.glu_grad_load not in ("interleaved","prefetch","shared"):
        return "invalid GLU gradient load schedule"
    if config.glu_grad_store not in ("direct","exchange"):
        return "invalid GLU gradient store schedule"
    if config.glu_grad_store=="exchange" and config.glu_grad_load=="shared":
        return "gradient store exchange and shared gradient load have overlapping lifetimes"
    if getattr(config,"glu_quant_vec",1) not in (1,2,4,8,16,32):
        return "invalid GLU output quantization vector"
    if config.glu_scratch_swizzle not in ("none","xor_16b"):
        return "invalid GLU scratch swizzle"
    if epilogue not in ("exchange", "registers"):
        return "unknown GLU epilogue"
    quant_epilogue = getattr(config,"glu_quant_epilogue","exchange")
    if quant_epilogue not in ("exchange","fragment_registers"):
        return "unknown GLU quantization epilogue"
    if quant_epilogue == "fragment_registers" and epilogue != "registers":
        return "fragment-register quantization requires the paired register epilogue"
    if quant_epilogue == "fragment_registers" and config.tile_n != 128:
        return "fragment-register quantization currently requires tile_n=128"
    store = getattr(config,"glu_store","direct")
    stages = getattr(config,"glu_store_stages",1)
    if store not in ("direct","tma") or stages not in (1,2,3,4):
        return "invalid GLU store schedule"
    if store=="tma" and (output_format!="bf16" or (problem.n//2)%8):
        return "TMA GLU output requires BF16 and H divisible by 8"
    if config.epilogue != "direct" or config.scale_rounding != "infinity":
        return "GLU requires direct epilogue and infinity scale rounding"
    if config.store_vec != 1 or config.epilogue_stages != 1:
        return "GLU cooperative exchange currently implements scalar output stores"
    operand = config.mxfp8_stages * (config.tile_m + config.tile_n) * config.tile_k
    padded_rows = ((config.tile_m + 127) // 128 + (config.tile_n + 127) // 128) * 128
    scales = config.mxfp8_stages * padded_rows * (config.tile_k // 32)
    staged = 0 if config.load_engine == "scalar" else config.bf16_stages * (config.tile_m + config.tile_n) * config.bf16_tile_k * 2
    exchange = config.tile_m * config.tile_n
    if epilogue == "registers":
        exchange = 0 if output_format == "bf16" else exchange // 2
    if quant_epilogue == "fragment_registers":
        exchange = config.tile_m*config.tile_n*config.atom_layout_n//64 if output_format == "mxfp8" else 0
    output_bytes = config.tile_m*config.tile_n*stages if store=="tma" else 0
    scratch = getattr(config, "glu_scratch", "dedicated")
    if scratch not in ("dedicated", "reuse_operands"):
        return "unknown GLU scratch policy"
    if scratch == "reuse_operands":
        # The cluster mainloop rendezvous after its final MMA proves all remote
        # operand users are finished before each CTA reuses its own storage.
        if config.persistent:
            return "operand scratch reuse requires nonpersistent transport"
        operand = max(operand,4*exchange)
        exchange = 0
    reserve = 1024 if scratch == "reuse_operands" and store == "direct" else 2048
    if operand + scales + staged + 4 * exchange + output_bytes + reserve > 101376:
        return "GLU accumulator exchange exceeds shared-memory capacity"
    return None
