"""Research adapter for policy-fixed MS-EDEN preparation scheduling."""
from dataclasses import dataclass,asdict,replace
from .core import DiscreteKernelAdapter,KernelContext,stable_id
from rtx.configs.nvfp4 import NVFP4GemmConfig,NVFP4Problem


@dataclass(frozen=True)
class PosthocOperandGeometry:
    """Optional launch geometry only; None inherits the leg's shared value."""
    producer_threads: int | None = None
    producer_blocks: int | None = None
    finalizer_threads: int | None = None
    finalizer_blocks: int | None = None

    def __post_init__(self):
        for name in ("producer_threads", "finalizer_threads"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value not in (64,128,256,512)):
                raise ValueError(f"invalid operand {name}")
        for name in ("producer_blocks", "finalizer_blocks"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"invalid operand {name}")


@dataclass(frozen=True)
class PosthocPreparationSchedule:
    producer_threads: int = 128
    producer_blocks: int = 128
    producer_order: str = "row_major"
    producer_load: str = "direct"
    producer_signs: str = "reload"
    producer_stages: int = 1
    producer_async_copy: bool = False
    producer_copy_bits: int = 32
    rng_mode: str = "explicit"
    philox_packet_size: int = 4
    finalizer_threads: int = 128
    finalizer_blocks: int = 128
    amax_reduction: str = "inline"
    reader_mode: str = "direct"
    decoder_threads: int = 128
    decoder_blocks: int = 128
    gemm: NVFP4GemmConfig = NVFP4GemmConfig(scale_layout="mma128",scale_role="tma")
    a_geometry: PosthocOperandGeometry | None = None
    b_geometry: PosthocOperandGeometry | None = None
    correction_decode: str = "lut"
    partial_max_scope: str = "warp"
    probability_math: str = "divide"
    normalization_math: str = "divide"
    gemm_split_reduction: int = 1
    gemm_reduction_algorithm: str = "tree"
    gemm_reduction_threads: int = 256
    gemm_reduction_vector: int = 4
    gemm_reduction_waves: int = 4
    outer_factor_mode: str = "product_kernel"
    producer_unroll_rows: bool = True
    producer_row_tile_batch: int = 1
    producer_row_load_bits: int = 16
    producer_values_per_lane: int = 4
    producer_transpose_values_per_lane: int = 4
    producer_scale_math: str = "divide"
    # Maximum store width for VPL8/16 producers; VPL4 remains byte stores.
    producer_grouped_store_bits: int = 8

    def __post_init__(self):
        if type(self.producer_grouped_store_bits) is not int or self.producer_grouped_store_bits not in (8,16,32,64):
            raise ValueError("invalid grouped packed store width")
        if self.producer_scale_math not in ("divide", "integer_rtz4"):
            raise ValueError("invalid producer scale math")
        if type(self.producer_transpose_values_per_lane) is not int or self.producer_transpose_values_per_lane not in (4,8,16):
            raise ValueError("producer transpose values per lane must be 4, 8, or 16")
        if type(self.producer_values_per_lane) is not int or self.producer_values_per_lane not in (4,8,16):
            raise ValueError("producer values per lane must be 4, 8, or 16")
        if type(self.producer_row_load_bits) is not int or self.producer_row_load_bits not in (16,32,64):
            raise ValueError("producer row load bits must be 16, 32, or 64")
        if type(self.producer_row_tile_batch) is not int or self.producer_row_tile_batch not in (1,2,4):
            raise ValueError("producer row tile batch must be 1, 2, or 4")
        if type(self.producer_unroll_rows) is not bool:
            raise ValueError("producer unroll rows must be boolean")
        if self.outer_factor_mode not in ("product_kernel","pair"):
            raise ValueError("invalid outer factor mode")
        if type(self.gemm_split_reduction) is not int or self.gemm_split_reduction not in (1,2,4,8,16,32):
            raise ValueError("invalid GEMM split reduction")
        if self.gemm_reduction_algorithm not in ("serial","tree","persistent_tree"):
            raise ValueError("invalid GEMM reduction algorithm")
        for name, values in (("gemm_reduction_threads",(64,128,256,512,1024)),
                             ("gemm_reduction_vector",(1,2,4,8)),
                             ("gemm_reduction_waves",(1,2,3,4,6,8))):
            if type(getattr(self,name)) is not int or getattr(self,name) not in values:
                raise ValueError(f"invalid {name}")
        if type(self.producer_copy_bits) is not int or self.producer_copy_bits not in (32,64,128):
            raise ValueError("producer copy bits must be 32, 64, or 128")
        if self.normalization_math not in ("divide", "refine", "threshold_scale_free", "refine_warp_lut", "guarded_scale_free"):
            raise ValueError("invalid normalization math")
        if self.probability_math not in ("divide", "power2"):
            raise ValueError("invalid probability math")
        if self.partial_max_scope not in ("warp", "cta"):
            raise ValueError("invalid partial maximum scope")
        if self.correction_decode not in ("lut", "native"):
            raise ValueError("invalid correction decode mode")
        for name in ("a_geometry", "b_geometry"):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = PosthocOperandGeometry(**value)
                object.__setattr__(self, name, value)
            if value is not None and not isinstance(value, PosthocOperandGeometry):
                raise ValueError(f"invalid {name}")
        if isinstance(self.gemm,dict):
            object.__setattr__(self,"gemm",NVFP4GemmConfig(**self.gemm))
        if not isinstance(self.gemm,NVFP4GemmConfig):
            raise ValueError("invalid GEMM configuration")
        if self.reader_mode not in ("direct","staged"):
            raise ValueError("invalid reader mode")
        if self.producer_order not in ("row_major", "reduction_major"):
            raise ValueError("invalid producer order")
        if self.producer_load not in ("direct", "coalesced", "coalesced_wide", "coalesced_x4", "coalesced_x8"):
            raise ValueError("invalid producer load")
        if self.producer_signs not in ("reload", "register"):
            raise ValueError("invalid producer signs reuse")
        if type(self.producer_stages) is not int or self.producer_stages not in (1,2,3):
            raise ValueError("invalid producer stages")
        if type(self.producer_async_copy) is not bool:
            raise ValueError("invalid producer async copy flag")
        if self.rng_mode not in ("explicit", "philox"):
            raise ValueError("invalid RNG mode")
        if type(self.philox_packet_size) is not int or self.philox_packet_size not in (1,4):
            raise ValueError("invalid Philox packet size")
        if self.amax_reduction not in ("inline", "staged"):
            raise ValueError("invalid amax reduction")
        for name in ("producer_threads","finalizer_threads","decoder_threads"):
            if type(getattr(self,name)) is not int or getattr(self,name) not in (64,128,256,512):
                raise ValueError(f"invalid {name}")
        for name in ("producer_blocks","finalizer_blocks","decoder_blocks"):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f"invalid {name}")

    def operand_schedule(self, operand):
        """Resolve A=dY and B=W/X without changing numerical/RNG controls."""
        if operand not in ("a", "b"):
            raise ValueError("operand must be a or b")
        geometry = getattr(self, operand + "_geometry")
        overrides = {} if geometry is None else {name: value for name, value in asdict(geometry).items()
                                                 if value is not None}
        return replace(self, a_geometry=None, b_geometry=None, **overrides)

    def gemm_reduction_tile(self, reduction_k):
        """Static aligned split geometry; reject empty slices before compile.

        Raw JSON is not silently rewritten. The tuning axis supplies a coherent
        direct/nonpersistent configuration when entering the split regime.
        """
        if self.gemm_split_reduction == 1:
            return 0
        if type(reduction_k) is not int or reduction_k <= 0:
            raise ValueError("positive integer GEMM reduction K required")
        if (self.gemm.epilogue != "direct" or self.gemm.store_vec != 1
                or self.gemm.tiles_per_cta != 1 or self.gemm.persistent_waves
                or self.gemm.regional_epilogue_schedule != "mma"):
            raise ValueError("split GEMM requires direct epilogue/store_vec=1 and nonpersistent single-tile MMA schedule")
        tile = ((reduction_k + self.gemm_split_reduction*self.gemm.tile_k-1)
                // (self.gemm_split_reduction*self.gemm.tile_k)) * self.gemm.tile_k
        if (self.gemm_split_reduction-1)*tile >= reduction_k:
            raise ValueError("split GEMM has an empty aligned reduction slice")
        return tile


def check_preparation_copy_alignment(source, schedule, transpose):
    """Validate actual storage before trusting vector-copy alignment metadata.

    Called inside runtime launch construction, not a traced/fake/meta contract.
    Row-major operands do not use asynchronous transpose staging.
    """
    if not transpose:
        if schedule.producer_row_load_bits>16:
            alignment=schedule.producer_row_load_bits//8
            if source.ndim!=2 or source.element_size()!=2 or not source.is_contiguous():
                raise ValueError("vector row loads require contiguous rank-two BF16 storage")
            if source.data_ptr()%alignment or source.stride(0)*source.element_size()%alignment:
                raise ValueError("source base and row stride must align to vector row load width")
        return
    if not schedule.producer_async_copy:
        return
    if not schedule.producer_async_copy or schedule.producer_load=="direct":
        raise ValueError("wide copies require asynchronous coalesced transpose loads")
    alignment=schedule.producer_copy_bits//8
    if source.ndim!=2 or source.element_size()!=2 or not source.is_contiguous():
        raise ValueError("wide copies require contiguous rank-two BF16 storage")
    if source.data_ptr()%alignment or (source.stride(0)*source.element_size())%alignment:
        raise ValueError("source base and row stride must align to asynchronous copy width")


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

    Never dispatch without measurement. The coalesced seeds are alternatives,
    not a fitted shape threshold; keep the original and direct-load challenger.
    """
    base=initial or PosthocPreparationSchedule()
    # Reuse the validated geometry/capping contract rather than inventing a
    # separate legality model for the prior.
    preparation_grid_candidates(shape,phase,sm_count,base)
    result=[base]
    order="reduction_major" if phase=="dw" else "row_major"
    for threads,waves,load in ((128,2,"coalesced"),(256,4,"coalesced"),(128,2,"direct"),
                               (128,2,"coalesced_wide"),
                               (128,8,"coalesced_wide"),(128,8,"coalesced_x4"),
                               (128,8,"coalesced_x8")):
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
    features={"schedule_features_revision":22,"sm_count_known":float(sm_count>0),
              "l2_known":float(l2_bytes>0)}
    scratch,payload,amax_loads=0,0,0
    for name,(rows,cols,transposed) in zip(("a","b"),operands):
        operand_config=config.operand_schedule(name)
        elements=rows*cols
        warps=operand_config.producer_threads//32
        blocks=min(operand_config.producer_blocks,(elements//128+warps-1)//warps)
        work_warps=blocks*warps
        row_batch=config.producer_row_tile_batch if not transposed else 1
        values_per_lane=config.producer_values_per_lane if not transposed else config.producer_transpose_values_per_lane
        groups_per_warp=values_per_lane//4
        partials=blocks if config.partial_max_scope=="cta" else work_warps
        packet=config.philox_packet_size if config.rng_mode=="philox" else 1
        finalizer_items=(elements//16+packet-1)//packet
        finalizers=min(operand_config.finalizer_blocks,(finalizer_items+operand_config.finalizer_threads-1)//operand_config.finalizer_threads)
        cooperative=transposed and config.producer_load!="direct"
        source_tile_rows=warps*{"coalesced_wide":2,"coalesced_x4":4,"coalesced_x8":8}.get(config.producer_load,1)
        asynchronous=cooperative and config.producer_async_copy
        stages=config.producer_stages if asynchronous else 1
        copy_elements=config.producer_copy_bits//16
        smem=(2*(stages*128*(source_tile_rows+copy_elements)-copy_elements) if asynchronous
              else 2*(127*(source_tile_rows+1)+source_tile_rows)) if cooperative else 0
        maximum_smem=warps*4 if config.partial_max_scope=="cta" else 0
        # The maximum array precedes the 16B-aligned transpose staging tile.
        smem+=((maximum_smem+15)//16)*16 if cooperative else maximum_smem
        features.update({
            f"{name}_elements":elements,
            f"{name}_transposed":float(transposed),
            f"{name}_source_row_stride":cols,
            f"{name}_source_column_group":source_tile_rows if cooperative else 1,
            f"{name}_producer_ctas":blocks,
            f"{name}_producer_threads":operand_config.producer_threads,
            f"{name}_producer_waves":blocks/sm_count if sm_count else 0,
            f"{name}_producer_partial_count":partials,
            f"{name}_producer_work_warp_count":work_warps,
            f"{name}_producer_cta_maximum":float(config.partial_max_scope=="cta"),
            f"{name}_producer_maximum_smem_bytes":maximum_smem,
            f"{name}_finalizer_ctas":finalizers,
            f"{name}_finalizer_threads":operand_config.finalizer_threads,
            f"{name}_finalizer_values_per_thread":packet,
            f"{name}_logical_sign_reads":work_warps*32*values_per_lane if config.producer_signs=="register" else elements,
            f"{name}_producer_smem_bytes":smem,
            f"{name}_producer_stages":stages,
            f"{name}_producer_unroll_rows":float(config.producer_unroll_rows),
            f"{name}_producer_rows_per_warp":source_tile_rows//warps if transposed else 1,
            f"{name}_producer_transpose_consumer_iterations":source_tile_rows//warps//groups_per_warp if cooperative else 0,
            f"{name}_producer_row_tile_batch":row_batch,
            f"{name}_producer_values_per_lane":values_per_lane,
            f"{name}_producer_hadamards_per_warp":groups_per_warp,
            f"{name}_producer_hadamard_shuffle_instructions_per_group":values_per_lane*(7-(values_per_lane.bit_length()-1))/groups_per_warp,
            f"{name}_producer_row_load_bits":config.producer_row_load_bits if not transposed else 0,
            f"{name}_producer_row_load_instructions":elements//(config.producer_row_load_bits//16) if not transposed else 0,
            f"{name}_producer_row_outer_iterations_max":((elements//128+work_warps*row_batch*groups_per_warp-1)//(work_warps*row_batch*groups_per_warp)
                                                       if not transposed else 0),
            f"{name}_producer_async_copy":float(asynchronous),
            f"{name}_producer_copy_bits":config.producer_copy_bits if asynchronous else 0,
            f"{name}_producer_async_row_stride":source_tile_rows+copy_elements if asynchronous else 0,
            f"{name}_producer_async_copy_instructions":elements//copy_elements if asynchronous else 0,
            f"{name}_correction_decode_native":float(config.correction_decode=="native"),
            f"{name}_probability_power2":float(config.probability_math=="power2"),
            f"{name}_normalization_refine":float(config.normalization_math=="refine"),
            f"{name}_normalization_threshold_scale_free":float(config.normalization_math=="threshold_scale_free"),
            f"{name}_normalization_refine_warp_lut":float(config.normalization_math=="refine_warp_lut"),
            f"{name}_normalization_guarded_scale_free":float(config.normalization_math=="guarded_scale_free"),
            f"{name}_producer_scale_integer_rtz4":float(config.producer_scale_math=="integer_rtz4"),
            f"{name}_producer_grouped_store_bits":min(config.producer_grouped_store_bits,values_per_lane*4) if values_per_lane>4 else 8,
        })
        staged=config.amax_reduction=="staged"
        amax_loads += (partials+finalizers*operand_config.finalizer_threads if staged
                       else finalizers*(operand_config.finalizer_threads//32)*partials)
        scratch += elements//8+partials*4+4*staged
        payload += elements//2+elements//16
    decoded=operands[1][0]*operands[1][1]*2 if packed_reader and config.reader_mode=="staged" else 0
    gemm_m,gemm_n,gemm_k=(m,k,n) if phase=="dx" else (n,k,m)
    split=config.gemm_split_reduction
    reduction_tile=config.gemm_reduction_tile(gemm_k)
    output_tiles=((gemm_m+config.gemm.tile_m-1)//config.gemm.tile_m
                  *((gemm_n+config.gemm.tile_n-1)//config.gemm.tile_n))
    workspace=split*gemm_m*gemm_n*4 if split>1 else 0
    features.update(preparation_scratch_bytes=scratch+decoded,
                    gemm_split_reduction=split,gemm_reduction_tile=reduction_tile,
                    gemm_output_tiles=output_tiles,gemm_split_ctas=output_tiles*split if split>1 else 0,
                    gemm_workspace_bytes=workspace,preparation_and_gemm_scratch_bytes=scratch+decoded+workspace,
                    gemm_launches=1+int(split>1),gemm_reduction_launches=int(split>1),
                    gemm_reduction_threads=config.gemm_reduction_threads if split>1 else 0,
                    gemm_reduction_vector=config.gemm_reduction_vector if split>1 else 0,
                    gemm_reduction_tree=float(split>1 and config.gemm_reduction_algorithm!="serial"),
                    packed_payload_bytes=payload,decoded_workspace_bytes=decoded,
                    amax_logical_load_elements=amax_loads,
                    payload_l2_ratio=payload/l2_bytes if l2_bytes else 0,
                    preparation_launches=5-int(config.outer_factor_mode=="pair")+2*(config.amax_reduction=="staged")+bool(decoded),
                    outer_factor_pair=float(config.outer_factor_mode=="pair"),
                    outer_factor_bytes=8 if config.outer_factor_mode=="pair" else 12,
                    scale_product_launches=int(config.outer_factor_mode=="product_kernel"),
                    rng_uniform_bytes=sum(rows*cols for rows,cols,_ in operands)//4 if config.rng_mode=="explicit" else 0,
                    rng_seed_bytes=16 if config.rng_mode=="philox" else 0,
                    rng_sign_bytes=512)
    return features


def make_posthoc_preparation_adapter(shape,phase,layout,evaluator,*,device=None,environment=None,
                                    source_policy="original_bf16",packed_reader=False,randomness="explicit",
                                    measurement_contract=None):
    """Create an adapter with an explicit randomness measurement contract.

    ``randomness='public'`` requires an evaluator of the public stochastic
    backward including RNG generation. Fixed-uniform evaluators must leave the
    default, otherwise a fused RNG configuration could be credited with timing
    a path that never exercised it.
    """
    if randomness not in ("explicit","public"):
        raise ValueError("unknown randomness measurement contract")
    if measurement_contract is None:
        measurement_contract=("graph8x10_samples15_warm200ms_revision2" if randomness=="explicit"
                              else "public_rng_evaluator_unspecified")
    if not isinstance(measurement_contract,str) or not measurement_contract.strip():
        raise ValueError("nonempty measurement contract required")
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
        "producer_load":("direct", "coalesced", "coalesced_wide", "coalesced_x4", "coalesced_x8"),
        "producer_signs":("reload", "register"),
        "producer_stages":(1,2,3),
        "producer_async_copy":(False,True),
        "producer_copy_bits":(32,64,128),
        "producer_unroll_rows":(True,False),
        "producer_row_tile_batch":(1,2,4),
        "producer_row_load_bits":(16,32,64),
        "producer_values_per_lane":(4,8,16),
        "producer_transpose_values_per_lane":(4,8,16),
        "producer_scale_math":("divide", "integer_rtz4"),
        "producer_grouped_store_bits":(8,16,32,64),
        "correction_decode":("lut","native"),
        # This adapter measures explicit caller-owned randomness. RNG-inclusive
        # tuning needs a separately identified public-backward evaluator.
        "rng_mode":("explicit","philox") if randomness=="public" else ("explicit",),
        "philox_packet_size":(1,4) if randomness=="public" else (4,),
        "finalizer_threads":(64,128,256,512),
        "finalizer_blocks":(32,64,128,256,512,1024,2048),
        "amax_reduction":("inline", "staged"),
        "partial_max_scope":("warp", "cta"),
        "probability_math":("divide", "power2"),
        "normalization_math":("divide", "refine", "threshold_scale_free", "refine_warp_lut", "guarded_scale_free"),
        "outer_factor_mode":("product_kernel","pair"),
        "gemm_reduction_algorithm":("serial","tree","persistent_tree"),
        "gemm_reduction_threads":(64,128,256,512,1024),
        "gemm_reduction_vector":(1,2,4,8),
        "gemm_reduction_waves":(1,2,3,4,6,8),
        "a_geometry":(None, *(asdict(PosthocOperandGeometry(producer_blocks=b,finalizer_blocks=b))
                              for b in (32,64,128,512))),
        "b_geometry":(None, *(asdict(PosthocOperandGeometry(producer_blocks=b,finalizer_blocks=b))
                              for b in (32,64,128,512))),
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
    split_counts=[1]
    for split in (2,4,8,16,32):
        tile=((problem.k+split*initial.gemm.tile_k-1)//(split*initial.gemm.tile_k))*initial.gemm.tile_k
        if (split-1)*tile<problem.k:
            split_counts.append(split)
    axes["gemm_split_reduction"]=tuple(split_counts)
    def update(c,axis,value):
        if axis.startswith("gemm:"):
            return replace(c,gemm=replace(c.gemm,**value))
        changes={axis:value}
        if axis=="producer_grouped_store_bits" and value>8 and max(c.producer_values_per_lane,c.producer_transpose_values_per_lane)==4:
            changes.update(producer_transpose_values_per_lane=16,producer_load="coalesced_x8")
        if axis=="producer_transpose_values_per_lane" and value>4:
            changes["producer_load"]="coalesced_x8"
        if axis=="gemm_split_reduction" and value>1:
            changes["gemm"]=replace(c.gemm,epilogue="direct",store_vec=1,epilogue_stages=1,
                persistent_waves=0,tiles_per_cta=1,regional_epilogue_schedule="mma")
        # Coherent structural proposals keep the buffered regime reachable
        # from the synchronous/direct seed with a small coordinate budget.
        if ((axis=="producer_stages" and value>1) or (axis=="producer_async_copy" and value)
            or (axis=="producer_copy_bits" and value>32)):
            changes["producer_async_copy"]=True
            if c.producer_load=="direct":
                changes["producer_load"]="coalesced_x8" if axis=="producer_copy_bits" else "coalesced"
            if packed_reader: changes["reader_mode"]="staged"
        if (axis=="producer_async_copy" and not value) or (axis=="producer_load" and value=="direct"):
            changes.update(producer_stages=1,producer_async_copy=False,producer_copy_bits=32)
        return replace(c,**changes)
    def reject(c):
        if c.producer_grouped_store_bits>8 and max(c.producer_values_per_lane if phase=="dx" else 4,c.producer_transpose_values_per_lane)==4:
            return "implementation_rejected","wide packed stores require a grouped producer"
        if c.producer_transpose_values_per_lane>4 and (c.producer_load!="coalesced_x8" or c.normalization_math!="divide" or packed_reader):
            return "implementation_rejected","grouped transpose preparation requires original BF16, coalesced_x8 and divide normalization"
        if c.normalization_math in ("refine_warp_lut","guarded_scale_free") and c.producer_values_per_lane!=4:
            return "implementation_rejected","warp reciprocal modes currently require VPL4 preparation"
        if phase=="dx" and c.producer_values_per_lane>4 and (c.normalization_math!="divide" or c.producer_row_tile_batch!=1):
            return "implementation_rejected","grouped row preparation requires divide normalization and row tile batch 1"
        if c.producer_copy_bits!=32 and not c.producer_async_copy:
            return "implementation_rejected","wide copies require asynchronous preparation"
        if c.producer_stages>1 and not c.producer_async_copy:
            return "implementation_rejected","buffered producer requires asynchronous copy"
        if c.producer_async_copy and c.producer_load=="direct":
            return "implementation_rejected","asynchronous producer requires coalesced transpose loads"
        if c.producer_async_copy and packed_reader and c.reader_mode!="staged":
            return "implementation_rejected","asynchronous producer requires original or decoded BF16"
        if c.producer_async_copy:
            for operand in ("a","b"):
                if operand=="a" and phase=="dx":
                    continue
                geometry=c.operand_schedule(operand)
                rows_per_warp={"coalesced_wide":2,"coalesced_x4":4,"coalesced_x8":8}.get(c.producer_load,1)
                if (geometry.producer_threads//32*rows_per_warp)%(c.producer_copy_bits//16):
                    return "implementation_rejected","copy width exceeds aligned source row group"
        if c.rng_mode != "explicit" and randomness=="explicit":
            return "policy_rejected","fixed-uniform preparation does not measure fused RNG"
        if c.reader_mode=="staged" and not packed_reader:
            return "policy_rejected","staged reader requires packed source"
        if c.gemm.scale_layout!=layout:
            return "policy_rejected","GEMM scale layout must match the producer"
        try:
            c.gemm_reduction_tile(problem.k)
        except ValueError as exc:
            return "implementation_rejected",str(exc)
        reason=c.gemm.rejection(problem)
        return ("implementation_rejected",reason) if reason else None
    def identity(c):
        fields=asdict(c)
        for operand in ("a", "b"):
            resolved=c.operand_schedule(operand)
            fields[operand+"_geometry"]={name:getattr(resolved,name) for name in
                ("producer_threads","producer_blocks","finalizer_threads","finalizer_blocks")}
        for name in ("producer_threads","producer_blocks","finalizer_threads","finalizer_blocks"):
            fields.pop(name)
        if c.rng_mode == "explicit":
            fields.pop("philox_packet_size")
        if c.reader_mode == "direct":
            fields.pop("decoder_threads"); fields.pop("decoder_blocks")
        if c.gemm_split_reduction==1:
            for name in ("gemm_reduction_algorithm","gemm_reduction_threads","gemm_reduction_vector","gemm_reduction_waves"):
                fields.pop(name)
        elif c.gemm_reduction_algorithm!="persistent_tree":
            fields.pop("gemm_reduction_waves")
        return stable_id(fields)
    return DiscreteKernelAdapter(
        context=KernelContext("ms_eden_preparation_research",27,dict(zip(("m", "n", "k"),shape)),
            device=device or {},environment=environment or {},tags={
                "phase":phase,"layout":layout,"source":source_policy,"packed_reader":packed_reader,
                "forward_source_config":forward_source_config,
                "schedule_features_revision":22,
                "grouped_store_contract":"byte_exact_vector_stores_vpl8_16_max_width_v1",
                "producer_scale_contract":"rn_div6_trunc4_flush_or_exact_integer_rtz4_v1",
                "guarded_scale_free_contract":"threshold_scale_free_stats_exact_packing_rn_lut_w16_v1",
                "warp_reciprocal_contract":"exact_16_mantissa_rn_startup_fullwarp_lookup_v1",
                "gemm_split_contract":"fp32_raw_partials_scaled_once_before_final_bf16_v1",
                "normalization_contract":"divide_exact_refine_or_reassociated_threshold_scale_free_v2",
                "correction_decode_contract":"lut_or_exact_e2m1_native_pair_fp32_v1",
                "policy":"shared_sign_hadamard128_e8m4_rtz_eden_scale_sr",
                "rng":"fixed_explicit_inputs_seed807" if randomness=="explicit" else "public_graph_safe_rng",
                "gemm":"schedule_configurable_nvfp4",
                "measurement":measurement_contract}),
        initial_config=initial,axes=axes,
        config_id_fn=identity,serialize_fn=asdict,
        deserialize_fn=lambda v:PosthocPreparationSchedule(**v),
        update_fn=update,evaluator=evaluator,rejection_fn=reject,
        extra_features_fn=lambda c:preparation_schedule_features(shape,phase,c,
            sm_count=int((device or {}).get("multiprocessor_count",0) or 0),
            l2_bytes=int((device or {}).get("l2_cache_size",0) or 0),packed_reader=packed_reader))
