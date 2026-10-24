"""Research adapter for policy-fixed MS-EDEN preparation scheduling."""
from dataclasses import dataclass,asdict,replace
from .core import DiscreteKernelAdapter,KernelContext,stable_id
from rtx.configs.nvfp4 import NVFP4GemmConfig,NVFP4Problem


@dataclass(frozen=True)
class PosthocPreparationSchedule:
    producer_threads: int = 128
    producer_blocks: int = 128
    producer_order: str = "row_major"
    producer_load: str = "direct"
    finalizer_threads: int = 128
    finalizer_blocks: int = 128
    amax_reduction: str = "inline"
    reader_mode: str = "direct"
    decoder_threads: int = 128
    decoder_blocks: int = 128
    gemm: NVFP4GemmConfig = NVFP4GemmConfig(scale_layout="mma128",scale_role="tma")

    def __post_init__(self):
        if isinstance(self.gemm,dict):
            object.__setattr__(self,"gemm",NVFP4GemmConfig(**self.gemm))
        if not isinstance(self.gemm,NVFP4GemmConfig):
            raise ValueError("invalid GEMM configuration")
        if self.reader_mode not in ("direct","staged"):
            raise ValueError("invalid reader mode")
        if self.producer_order not in ("row_major", "reduction_major"):
            raise ValueError("invalid producer order")
        if self.producer_load not in ("direct", "coalesced"):
            raise ValueError("invalid producer load")
        if self.amax_reduction not in ("inline", "staged"):
            raise ValueError("invalid amax reduction")
        for name in ("producer_threads","finalizer_threads","decoder_threads"):
            if type(getattr(self,name)) is not int or getattr(self,name) not in (64,128,256,512):
                raise ValueError(f"invalid {name}")
        for name in ("producer_blocks","finalizer_blocks","decoder_blocks"):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f"invalid {name}")


def preparation_grid_candidates(shape, phase, sm_count, initial=None):
    """Soft grid proposals, not a dispatch rule or an occupancy prediction.

    Vary producer and finalizer independently before combining them. Each
    launch already caps its grid by work; cap proposals too to avoid redundant
    trials on tiny operands. Keep the baseline even when it exceeds useful work.
    """
    if phase not in ("dx", "dw"):
        raise ValueError("unknown backward phase")
    if len(shape) != 3 or any(type(d) is not int or d <= 0 or d % 128 for d in shape):
        raise ValueError("grid proposals require dimensions divisible by 128")
    if type(sm_count) is not int or sm_count <= 0:
        raise ValueError("SM count must be a positive integer")
    base = initial or PosthocPreparationSchedule()
    m, n, k = shape
    elements = max(m * n, n * k if phase == "dx" else m * k)
    warps = base.producer_threads // 32
    producer_cap = (elements // 128 + warps - 1) // warps
    finalizer_cap = (elements // 16 + base.finalizer_threads - 1) // base.finalizer_threads
    result = [base]
    for waves in (1, 2, 4, 8):
        grid = ((sm_count * waves + 31) // 32) * 32
        producer = min(grid, producer_cap)
        finalizer = min(grid, finalizer_cap)
        for changes in ({"producer_blocks": producer},
                        {"finalizer_blocks": finalizer},
                        {"producer_blocks": producer, "finalizer_blocks": finalizer}):
            candidate = replace(base, **changes)
            if candidate not in result:
                result.append(candidate)
    # Preserve the cheap isolated probes above, then explore interactions:
    # more producer CTAs also enlarge every finalizer's reduction input.
    producers = list(dict.fromkeys(c.producer_blocks for c in result))
    finalizers = list(dict.fromkeys(c.finalizer_blocks for c in result))
    for producer in producers:
        for finalizer in finalizers:
            candidate = replace(base, producer_blocks=producer, finalizer_blocks=finalizer)
            if candidate not in result:
                result.append(candidate)
    return result


def preparation_resource_seeds(shape, phase, sm_count, initial=None):
    """Small soft portfolio probing coupled load/thread/grid regimes.

    Never dispatch without measurement. The two coalesced seeds are alternatives,
    not a fitted shape threshold; keep the original and direct-load challenger.
    """
    base=initial or PosthocPreparationSchedule()
    # Reuse the validated geometry/capping contract rather than inventing a
    # separate legality model for the prior.
    preparation_grid_candidates(shape,phase,sm_count,base)
    result=[base]
    order="reduction_major" if phase=="dw" else "row_major"
    for threads,waves,load in ((128,2,"coalesced"),(256,4,"coalesced"),(128,2,"direct")):
        seed=replace(base,producer_threads=threads,producer_order=order,producer_load=load)
        producer_grid=((sm_count*waves+31)//32)*32
        finalizer_grid=((sm_count*8+31)//32)*32
        seed=replace(seed,producer_blocks=producer_grid,finalizer_blocks=finalizer_grid)
        features=preparation_schedule_features(shape,phase,seed)
        seed=replace(seed,producer_blocks=max(features["a_producer_ctas"],features["b_producer_ctas"]),
                     finalizer_blocks=max(features["a_finalizer_ctas"],features["b_finalizer_ctas"]))
        if seed not in result:
            result.append(seed)
    return result


def preparation_schedule_features(shape, phase, config, *, sm_count=0, l2_bytes=0, packed_reader=False):
    """Structural model inputs, not measured traffic or a latency prediction.

    Counts match launch capping and include inactive-axis normalization. Scratch
    excludes output, caller RNG and prepared payload; logical maximum loads do
    not account for cache/broadcast and must not be labeled DRAM bytes.
    """
    if phase not in ("dx", "dw"):
        raise ValueError("unknown backward phase")
    if len(shape) != 3 or any(type(d) is not int or d <= 0 or d % 128 for d in shape):
        raise ValueError("feature geometry requires dimensions divisible by 128")
    if any(type(d) is not int or d < 0 for d in (sm_count,l2_bytes)):
        raise ValueError("device capacities must be nonnegative integers")
    m,n,k=shape
    operands=((m,n,phase=="dw"),(n,k,True) if phase=="dx" else (m,k,True))
    features={"schedule_features_revision":1,"sm_count_known":float(sm_count>0),
              "l2_known":float(l2_bytes>0)}
    scratch,payload,amax_loads=0,0,0
    for name,(rows,cols,transposed) in zip(("a","b"),operands):
        elements=rows*cols
        warps=config.producer_threads//32
        blocks=min(config.producer_blocks,(elements//128+warps-1)//warps)
        partials=blocks*warps
        finalizers=min(config.finalizer_blocks,(elements//16+config.finalizer_threads-1)//config.finalizer_threads)
        cooperative=transposed and config.producer_load=="coalesced"
        features.update({
            f"{name}_elements":elements,
            f"{name}_transposed":float(transposed),
            f"{name}_source_row_stride":cols,
            f"{name}_source_column_group":warps if cooperative else 1,
            f"{name}_producer_ctas":blocks,
            f"{name}_producer_waves":blocks/sm_count if sm_count else 0,
            f"{name}_producer_partial_count":partials,
            f"{name}_finalizer_ctas":finalizers,
            f"{name}_producer_smem_bytes":2*(127*(warps+1)+warps) if cooperative else 0,
        })
        staged=config.amax_reduction=="staged"
        amax_loads += (partials+finalizers*config.finalizer_threads if staged
                       else finalizers*(config.finalizer_threads//32)*partials)
        scratch += elements//8+partials*4+4*staged
        payload += elements//2+elements//16
    decoded=operands[1][0]*operands[1][1]*2 if packed_reader and config.reader_mode=="staged" else 0
    features.update(preparation_scratch_bytes=scratch+decoded,
                    packed_payload_bytes=payload,decoded_workspace_bytes=decoded,
                    amax_logical_load_elements=amax_loads,
                    payload_l2_ratio=payload/l2_bytes if l2_bytes else 0,
                    preparation_launches=5+2*(config.amax_reduction=="staged")+bool(decoded))
    return features


def make_posthoc_preparation_adapter(shape,phase,layout,evaluator,*,device=None,environment=None,
                                    source_policy="original_bf16",packed_reader=False):
    if source_policy not in ("original_bf16","forward_current_roundtrip_bf16"):
        raise ValueError("unknown backward source policy")
    if packed_reader and source_policy != "forward_current_roundtrip_bf16":
        raise ValueError("packed reader requires explicit forward-roundtrip source policy")
    forward_source_config = None
    if source_policy == "forward_current_roundtrip_bf16":
        from rtx.configs.nvfp4 import DEFAULT_NVFP4_QUANT_CONFIG
        forward_source_config = asdict(DEFAULT_NVFP4_QUANT_CONFIG)
    if phase not in ("dx","dw") or layout not in ("row_major","mma128"):
        raise ValueError("unsupported post-hoc phase/layout")
    if len(shape)!=3 or any(type(d) is not int or d<=0 or d%128 for d in shape):
        raise ValueError("research GEMM path requires dimensions divisible by 128")
    axes={
        "producer_threads":(64,128,256,512),
        "producer_blocks":(32,64,128,256,512,1024,2048),
        "producer_order":("row_major", "reduction_major"),
        "producer_load":("direct", "coalesced"),
        "finalizer_threads":(64,128,256,512),
        "finalizer_blocks":(32,64,128,256,512,1024,2048),
        "amax_reduction":("inline", "staged"),
        "reader_mode":("direct","staged") if packed_reader else ("direct",),
        "decoder_threads":(64,128,256,512) if packed_reader else (128,),
        "decoder_blocks":(32,64,128,256,512,1024) if packed_reader else (128,),
    }
    from rtx.nvfp4_inference_autotune import _gemm_axes
    for name,updates in _gemm_axes().items():
        values=tuple(update["gemm"] for update in updates if "gemm" in update)
        if values:
            axes["gemm:"+name]=values
    m,n,k=shape
    problem=NVFP4Problem(m,k,n) if phase=="dx" else NVFP4Problem(n,k,m)
    initial=PosthocPreparationSchedule(gemm=NVFP4GemmConfig(
        scale_layout=layout,scale_role="tma" if layout=="mma128" else "consumers"))
    def update(c,axis,value):
        if axis.startswith("gemm:"):
            return replace(c,gemm=replace(c.gemm,**value))
        return replace(c,**{axis:value})
    def reject(c):
        if c.reader_mode=="staged" and not packed_reader:
            return "policy_rejected","staged reader requires packed source"
        if c.gemm.scale_layout!=layout:
            return "policy_rejected","GEMM scale layout must match the producer"
        reason=c.gemm.rejection(problem)
        return ("implementation_rejected",reason) if reason else None
    def identity(c):
        fields=asdict(c)
        if c.reader_mode == "direct":
            fields.pop("decoder_threads"); fields.pop("decoder_blocks")
        return stable_id(fields)
    return DiscreteKernelAdapter(
        context=KernelContext("ms_eden_preparation_research",7,dict(zip(("m","n","k"),shape)),
            device=device or {},environment=environment or {},tags={
                "phase":phase,"layout":layout,"source":source_policy,"packed_reader":packed_reader,
                "forward_source_config":forward_source_config,
                "schedule_features_revision":1,
                "policy":"shared_sign_hadamard128_e8m4_rtz_eden_scale_sr",
                "rng":"fixed_explicit_inputs_seed807","gemm":"schedule_configurable_nvfp4",
                "measurement":"graph8x10_samples15_warm200ms_revision2"}),
        initial_config=initial,axes=axes,
        config_id_fn=identity,serialize_fn=asdict,
        deserialize_fn=lambda v:PosthocPreparationSchedule(**v),
        update_fn=update,evaluator=evaluator,rejection_fn=reject,
        extra_features_fn=lambda c:preparation_schedule_features(shape,phase,c,
            sm_count=int((device or {}).get("multiprocessor_count",0) or 0),
            l2_bytes=int((device or {}).get("l2_cache_size",0) or 0),packed_reader=packed_reader))
