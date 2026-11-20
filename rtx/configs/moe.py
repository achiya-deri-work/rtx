"""CPU-safe quantize-once expert schedules, independent for every adjoint."""
from dataclasses import asdict,dataclass,field,replace

from .mxfp8 import MXFP8GemmConfig,MXFP8QuantConfig
from .glu import MXFP8GLUGemmConfig,MXFP8GLUPrequantConfig,prequant_recompute_rejection
from ..kernels.mxfp8 import MXFP8Problem,SM120_SMEM_CAPACITY_BYTES,SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES


def _linear():
    return MXFP8GemmConfig(tile_m=64,tile_n=64,atom_layout_m=2,atom_layout_n=2,stages=2,
                          epilogue='direct',store_vec=1,epilogue_stages=1)


def _glu():
    return MXFP8GLUGemmConfig(tile_m=64,tile_n=128,atom_layout_m=2,atom_layout_n=2,stages=2,
                            glu_epilogue='registers',glu_silu_math='division')


def _quant():
    return MXFP8QuantConfig(quant_vec=8,load_bits=128,quant_store_bits=32,
        quant_math='fp32',quant_amax='bf16_bits',reduction='redux',num_warps=4)


STAGES=('gate_up','down','recompute','down_dx','down_dw','gate_dx','gate_dw')
JOINT_RECOMPUTE_ENGINES=('legacy','operand_reuse','dual_operand','split_warp')


@dataclass(frozen=True,slots=True)
class _ShortDWGemmConfig(MXFP8GemmConfig):
    """Internal legality for the cooperative, scalar-scale K32 MMA path."""
    @property
    def native_mma_k(self):return 32


def short_dw_config(config):
    return _ShortDWGemmConfig(**asdict(config))


def short_dw_rejection(problem,config):
    if config.tile_k not in (32,64) or problem.k%32 or problem.k>128:
        return 'Short dW needs one to four physical 32-value scale blocks and a K32/K64 tile'
    if (config.tile_m,config.tile_n,config.atom_layout_m,config.atom_layout_n)!=(64,64,2,2):
        return 'Short dW uses 64x64 output tiles and four cooperative MMA warps'
    if (config.scale_layout!='row_major' or config.stages!=1 or config.tiles_per_cta!=1
            or config.persistent_waves or config.tile_locality!='raster' or config.epilogue!='direct'
            or config.scale_role!='producer' or config.scale_load_vec!=1
            or config.a_swizzle!='32b' or config.b_swizzle!='32b'):
        return 'Short dW needs direct cooperative nonpersistent K_SW32 transport and row-major scales'
    return short_dw_config(config).rejection(problem)


def joint_recompute_rejection(problem,g,engine,gradient_warps=3):
    """The alias engines use only finished, nonpersistent TMA operand arrays."""
    if engine not in JOINT_RECOMPUTE_ENGINES:return 'Unknown joint recompute engine'
    if problem.m%32 or problem.n%64 or g.tile_m%32 or g.tile_n%64:
        return 'Joint recompute needs complete 32-value blocks'
    if g.scale_layout!='row_major' or g.glu_grad_store!='exchange':
        return 'Joint recompute needs row-major scales and exchange derivatives'
    if engine=='legacy':return None
    if g.scale_role!='producer' or g.tiles_per_cta!=1 or g.persistent_waves:
        return 'Operand reuse needs producer scales and one nonpersistent tile per CTA'
    scratch=2*g.tile_m*g.tile_n
    if g.stages*g.tile_n*g.tile_k<scratch:
        return 'Joint epilogue scratch does not fit in the retired B operand'
    if engine in ('dual_operand','split_warp'):
        if g.stages*g.tile_m*g.tile_k<scratch:
            return 'Joint hidden products do not fit in the retired A operand'
        if g.glu_grad_load!='interleaved':
            return 'Dual-operand recompute loads gradients at point of use'
    if engine=='split_warp':
        from .moe_backward import tma_register_roles_rejection
        if reason:=tma_register_roles_rejection(g):return reason
        if (g.epilogue!='direct' or g.glu_store!='direct' or g.glu_epilogue!='registers'
                or g.glu_quant_vec not in (1,2,4,8) or g.scale_rounding!='infinity'
                or g.glu_silu_math not in ('division','shared_reciprocal')):
            return 'Split recompute needs direct register output, qualified vectors and the original scaling/math'
        if type(gradient_warps) is not int or not 1<=gradient_warps<g.num_threads//32:
            return 'Split recompute requires nonempty full-warp gradient and hidden teams'
        decoded=2*(g.tile_n//2)*(g.tile_m+1)
        decoded=(decoded+1023)//1024*1024
        operands=g.stages*(g.tile_m+g.tile_n)*g.tile_k
        scales=g.stages*(((g.tile_m+127)//128)*128+((g.tile_n+127)//128)*128)*(g.tile_k//32)
        if operands+scales+decoded+SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES>SM120_SMEM_CAPACITY_BYTES:
            return 'Split recompute decoded plane exceeds shared-memory capacity'
    return None


@dataclass(frozen=True)
class MXFP8PreparedMoEConfig:
    quant:MXFP8QuantConfig=field(default_factory=_quant)
    gate_up:MXFP8GLUGemmConfig=field(default_factory=_glu)
    down:MXFP8GemmConfig=field(default_factory=_linear)
    recompute:MXFP8GLUGemmConfig=field(default_factory=_glu)
    down_dx:MXFP8GemmConfig=field(default_factory=_linear)
    down_dw:MXFP8GemmConfig=field(default_factory=_linear)
    gate_dx:MXFP8GemmConfig=field(default_factory=_linear)
    gate_dw:MXFP8GemmConfig=field(default_factory=_linear)
    transpose_mode:str='staged'
    backward_preparation:str='separate'
    transports:tuple[str,...]=('tma',)*7
    capacity_multiple:int=1
    quant_grid:str='operand'
    input_preparation:str='gather_bf16'
    transpose_warp_tokens:int=8
    transpose_warp_raster:str='features'
    down_dw_splits:int=1
    gate_dw_splits:int=1
    dw_reduce_threads:int=128
    dw_reduce_vector:int=4
    route_gradient_epilogue:str='separate'
    forward_engine:str='packed'
    dy_preparation:str='materialized'
    joint_recompute_engine:str='legacy'
    active_prefix:bool=False
    gate_dw_engine:str='packed'
    down_dw_engine:str='packed'
    gate_dw_reuse_axis:str='n'
    down_dw_reuse_axis:str='n'
    gate_dw_reuse_tiles:int=2
    down_dw_reuse_tiles:int=2
    indexed_dw_token_lanes:int=8
    indexed_dw_gather_pairs:int=1
    indexed_dw_producer_warps:int=0
    route_gradient_loader:str='accumulator'
    route_gradient_gemm:MXFP8GemmConfig|None=None
    joint_recompute_gradient_warps:int=3
    forward_glu_engine:str='legacy'
    weight_preparation:str='separate'
    weight_tile_rows:int=32
    weight_tile_cols:int=64
    weight_load_values:int=4
    weight_store_bits:int=32
    weight_warps:int=4
    weight_raster:str='features'
    bounded_gather:bool=False
    bounded_dy:bool=False
    bounded_xt:bool=False
    forward_row_layout:str='original'
    compact_pool_count:int=0

    def __post_init__(self):
        for name in ('bounded_gather','bounded_dy','bounded_xt'):
            if type(getattr(self,name)) is not bool:raise ValueError(f'{name} must be bool')
        if self.forward_row_layout not in ('original','compact'):
            raise ValueError('forward_row_layout must be original/compact')
        if type(self.compact_pool_count) is not int or self.compact_pool_count<0:
            raise ValueError('compact_pool_count must be a nonnegative bound supplied by the packed frontend')
        if self.transpose_mode not in ('direct','staged','warp'):raise ValueError('transpose_mode must be direct/staged/warp')
        if type(self.transpose_warp_tokens) is not int or self.transpose_warp_tokens not in (2,4,8):
            raise ValueError('transpose_warp_tokens must be 2/4/8')
        if self.transpose_warp_raster not in ('features','tokens'):
            raise ValueError('transpose_warp_raster must be features/tokens')
        for name in ('down_dw_splits','gate_dw_splits'):
            if type(getattr(self,name)) is not int or getattr(self,name) not in (1,2,4,8,16):
                raise ValueError(f'{name} must be 1/2/4/8/16')
        if type(self.dw_reduce_threads) is not int or self.dw_reduce_threads not in (64,128,256,512):
            raise ValueError('dw_reduce_threads must be 64/128/256/512')
        if type(self.dw_reduce_vector) is not int or self.dw_reduce_vector not in (1,2,4,8):
            raise ValueError('dw_reduce_vector must be 1/2/4/8')
        if self.route_gradient_epilogue not in ('separate','fused'):
            raise ValueError('route_gradient_epilogue must be separate/fused')
        for stage in ('gate_dw','down_dw'):
            engines=('packed','persistent','indexed') if stage=='gate_dw' else ('packed','persistent')
            if getattr(self,stage+'_engine') not in engines:
                raise ValueError(f'{stage}_engine must be one of {engines}')
            if getattr(self,stage+'_reuse_axis') not in ('m','n'):
                raise ValueError(f'{stage}_reuse_axis must be m/n')
            tiles=getattr(self,stage+'_reuse_tiles')
            if type(tiles) is not int or tiles not in (2,3):
                raise ValueError(f'{stage}_reuse_tiles must be 2/3')
        for name,values in (('indexed_dw_token_lanes',(2,4,8)),
                            ('indexed_dw_gather_pairs',(1,2)),
                            ('indexed_dw_producer_warps',(0,2,4,8))):
            if type(getattr(self,name)) is not int or getattr(self,name) not in values:
                raise ValueError(f'{name} must be one of {values}')
        if self.route_gradient_loader not in ('accumulator','shared_vec4','warp_vec2'):
            raise ValueError('route_gradient_loader must be accumulator/shared_vec4/warp_vec2')
        if self.route_gradient_gemm is not None and not isinstance(self.route_gradient_gemm,MXFP8GemmConfig):
            raise TypeError('route_gradient_gemm must be an MXFP8GemmConfig or None')
        if self.forward_engine not in ('packed','onchip'):
            raise ValueError('forward_engine must be packed/onchip')
        if self.dy_preparation not in ('materialized','fused','fused_all'):
            raise ValueError('dy_preparation must be materialized/fused/fused_all')
        if self.backward_preparation not in ('separate','joint'):raise ValueError('backward_preparation must be separate/joint')
        if self.joint_recompute_engine not in JOINT_RECOMPUTE_ENGINES:
            raise ValueError(f'joint_recompute_engine must be one of {JOINT_RECOMPUTE_ENGINES}')
        if type(self.joint_recompute_gradient_warps) is not int or not 1<=self.joint_recompute_gradient_warps<=31:
            raise ValueError('joint_recompute_gradient_warps must be between 1 and 31')
        if self.forward_glu_engine not in ('legacy','fragment_direct','fragment_packed'):
            raise ValueError('forward_glu_engine must be legacy/fragment_direct/fragment_packed')
        if self.weight_preparation not in ('separate','dual_orientation'):
            raise ValueError('weight_preparation must be separate/dual_orientation')
        for name,values in (('weight_tile_rows',(32,64,128)),('weight_tile_cols',(32,64,128)),
                            ('weight_load_values',(1,2,4,8)),('weight_store_bits',(8,16,32)),('weight_warps',(2,4,8))):
            if type(getattr(self,name)) is not int or getattr(self,name) not in values:
                raise ValueError(f'{name} must be one of {values}')
        if self.weight_raster not in ('features','rows'):
            raise ValueError('weight_raster must be features/rows')
        if type(self.active_prefix) is not bool:raise ValueError('active_prefix must be bool')
        if not isinstance(self.transports,tuple) or len(self.transports)!=len(STAGES) or any(t not in ('tma','cooperative') for t in self.transports):
            raise ValueError('transports must specify tma/cooperative for each of the seven GEMMs')
        if type(self.capacity_multiple) is not int or self.capacity_multiple not in (1,32,64,128,256):
            raise ValueError('capacity_multiple must be 1/32/64/128/256')
        if self.quant_grid not in ('operand','expert','portfolio'):
            raise ValueError('quant_grid must be operand/expert/portfolio')
        if self.input_preparation not in ('gather_bf16','quantize_once','indexed_transpose'):
            raise ValueError('input_preparation must be gather_bf16/quantize_once/indexed_transpose')
        if not isinstance(self.quant,MXFP8QuantConfig):raise TypeError('Expected an MXFP8QuantConfig')
        for name in STAGES:
            cls=MXFP8GLUGemmConfig if name in ('gate_up','recompute') else MXFP8GemmConfig
            if not isinstance(getattr(self,name),cls):raise TypeError(f'{name}: expected {cls.__name__}')

    def schedules(self):
        return {**{name:getattr(self,name) for name in STAGES},
                **{name+'_transport':value for name,value in zip(STAGES,self.transports)}}

    def to_dict(self):return asdict(self)

    def physical_capacity(self,logical):
        multiple=max(self.capacity_multiple,128 if self.quant.scale_layout=='mma128'
                     else 32 if self.backward_preparation=='joint' else 1)
        return (logical+multiple-1)//multiple*multiple

    def with_transport(self,stage,transport):
        """Select one component's transport, including its required schedule.

        Cooperative cp.async is currently single-buffer, linear-only and uses
        all MMA warps for loading. It has no producer/consumer register roles.
        Tile sizes, swizzles, ldmatrix width, raster and compiler register cap
        remain independently selectable. TMA retains the multi-buffer family.
        """
        if stage not in STAGES or transport not in ('tma','cooperative'):
            raise ValueError('Expected a known expert stage and tma/cooperative transport')
        if transport=='cooperative' and stage in ('gate_up','recompute'):
            raise ValueError('Cooperative transport is currently linear-only')
        transports=list(self.transports);transports[STAGES.index(stage)]=transport
        c=getattr(self,stage)
        if transport=='cooperative':
            c=replace(c,stages=1,scale_role='producer',tiles_per_cta=1,persistent_waves=0,tile_locality='raster',
                epilogue='direct',store_vec=1,epilogue_stages=1,
                producer_registers=24,consumer_registers=64,scale_load_vec=1,scale_smem_store='scalar',
                scale_l2_prefetch='none',scale_l1_evict='default',scale_cache='default',
                scale_schedule='before_wait',scale_recycle='barrier',mma_schedule='interleaved')
        elif c.tile_k in (32,64):
            c=replace(c,tile_k=128)
        return replace(self,transports=tuple(transports),**{stage:c})

    def with_short_dw(self,stage,tile_k):
        """Select a complete legal small-K transport/geometry coordinate."""
        if stage not in ('down_dw','gate_dw') or type(tile_k) is not int or tile_k not in (0,32,64):
            raise ValueError('Short dW requires down_dw/gate_dw and tile K=0/32/64')
        if tile_k==0:return self.with_transport(stage,'tma')
        p=self.with_transport(stage,'cooperative')
        c=replace(getattr(p,stage),tile_m=64,tile_n=64,tile_k=tile_k,atom_layout_m=2,atom_layout_n=2,
                  a_swizzle='32b',b_swizzle='32b')
        return replace(p,**{stage:c,stage+'_splits':1})

    def with_scale_layout(self,layout):
        if layout=='row_major':return type(self)()
        if layout!='mma128':raise ValueError('Scale layout must be row_major/mma128')
        stages={name:replace(getattr(self,name),tile_m=128,tile_n=128,tile_k=128,
            atom_layout_m=4,atom_layout_n=2,stages=1,scale_layout='mma128',scale_role='tma',
            scale_schedule='before_wait',scale_load_vec=4,scale_smem_store='scalar',
            scale_l2_prefetch='none',scale_l1_evict='default',scale_cache='default') for name in STAGES}
        return replace(self,quant=replace(self.quant,scale_layout='mma128'),transports=('tma',)*7,**stages)

    @classmethod
    def from_dict(cls,values):
        result=dict(values)
        if 'transports' in result:result['transports']=tuple(result['transports'])
        if 'quant' in result:result['quant']=MXFP8QuantConfig(**result['quant'])
        for name in STAGES:
            if name in result:
                ctor=MXFP8GLUGemmConfig if name in ('gate_up','recompute') else MXFP8GemmConfig
                result[name]=ctor(**result[name])
        if result.get('route_gradient_gemm') is not None:
            result['route_gradient_gemm']=MXFP8GemmConfig(**result['route_gradient_gemm'])
        return cls(**result)

    def active_prefix_rejection(self,phase='train'):
        if self.active_prefix:
            if self.quant.scale_layout!='row_major' or self.forward_engine!='packed':
                return 'Active prefix requires packed forward and row-major scales'
            training=phase!='forward'
            stages=STAGES if training else ('gate_up','down')
            if any(self.transports[STAGES.index(s)]!='tma' for s in stages):
                return 'Active prefix requires TMA operand transport'
            for stage in stages:
                c=getattr(self,stage)
                if c.tiles_per_cta!=1 or c.persistent_waves:
                    return f'{stage}: active prefix requires one nonpersistent tile per CTA'
            if training and self.route_gradient_epilogue=='fused':
                c=self.route_gradient_gemm or self.down
                if c.tiles_per_cta!=1 or c.persistent_waves:
                    return 'Route-gradient active prefix requires one nonpersistent tile per CTA'
            if training and (self.backward_preparation!='joint'
                             or max(self.down_dw_splits,self.gate_dw_splits)>1):
                return 'Active prefix training requires joint recompute and unsplit weight gradients'
        return None

    def bounded_preparation_rejection(self,phase='train'):
        enabled=self.bounded_gather or (phase!='forward' and (self.bounded_dy or self.bounded_xt))
        if not enabled:return None
        if not self.active_prefix or self.quant.scale_layout!='row_major' or self.forward_engine!='packed':
            return 'Bounded preparation requires active prefix, row-major scales and packed forward'
        if self.input_preparation=='gather_bf16':
            return 'Bounded preparation requires original-token input preparation'
        if phase!='forward':
            if self.backward_preparation!='joint':return 'Bounded backward preparation requires joint recompute'
            if self.bounded_gather and self.input_preparation!='indexed_transpose':
                return 'Bounded training gather currently requires indexed input without a BF16 gathered tensor'
            if self.bounded_dy and (self.input_preparation!='indexed_transpose' or self.dy_preparation!='fused'):
                return 'Bounded dY currently requires indexed input and fused dY preparation'
            if self.bounded_xt and (self.input_preparation!='indexed_transpose'
                    or self.dy_preparation=='fused_all' or self.gate_dw_engine=='indexed'):
                return 'Bounded X transpose requires the separate indexed-transpose producer'
        return None

    def compact_forward_rejection(self,shape):
        if self.forward_row_layout=='original':return None
        n,d,e,c,k,h=shape
        if (self.compact_pool_count<1 or self.forward_engine!='packed'
                or self.quant.scale_layout!='row_major' or self.input_preparation=='gather_bf16'):
            return 'Compact forward requires a proven pool bound and prepared row-major packed inputs'
        if d%32 or h%32:return 'Compact forward currently requires complete feature scale blocks'
        if self.forward_glu_engine=='legacy':return 'Compact forward currently requires a fragment GLU'
        if any(self.transports[STAGES.index(s)]!='tma' or getattr(self,s).tiles_per_cta!=1
               or getattr(self,s).persistent_waves for s in ('gate_up','down')):
            return 'Compact forward requires one independent TMA tile per CTA'
        from .._moe_compact_layout import compact_forward_capacity
        import math
        if not compact_forward_capacity(n,self.compact_pool_count,e,k,self.physical_capacity(c),
                math.lcm(32,self.gate_up.tile_m,self.down.tile_m)):
            return 'Compact forward has no shape-bound storage reduction'
        return None

    def rejection(self,problems,phase='train'):
        layout=self.quant.scale_layout
        if reason:=self.active_prefix_rejection(phase):return reason
        if reason:=self.bounded_preparation_rejection(phase):return reason
        reason=self.quant.rejection(128,128)
        if reason:return f'Operand quantizer: {reason}'
        if self.forward_glu_engine!='legacy':
            from .moe_backward import fragment_glu_rejection
            if self.forward_engine!='packed':return 'Fragment GLU needs packed forward'
            logical=problems['gate_up'][0]
            reason=fragment_glu_rejection(MXFP8Problem(self.physical_capacity(logical.m),logical.n,logical.storage_k),
                                          self.gate_up,self.forward_glu_engine)
            if reason:return reason
        if phase=='train' and self.weight_preparation!='separate':
            if self.backward_preparation!='joint' or self.input_preparation=='gather_bf16':
                return 'Paired weight preparation requires joint backward and original-token input quantization'
            from .moe_backward import backward_weight_quant_rejection
            d,h=problems['gate_up'][0].k,problems['gate_up'][0].n//2
            reason=backward_weight_quant_rejection(1,d,1,h,self.quant,self.quant_grid,
                self.weight_tile_rows,self.weight_tile_cols,self.weight_load_values,self.weight_store_bits,
                self.weight_warps,self.weight_raster)
            if reason:return reason
        if self.input_preparation=='indexed_transpose' and (self.transpose_mode!='warp' or self.backward_preparation!='joint'):
            return 'Indexed input transpose requires warp quantization and joint backward preparation'
        if self.dy_preparation!='materialized':
            if self.input_preparation!='indexed_transpose':
                return 'Fused dY preparation requires the indexed joint-backward input map'
            q=self.quant
            shared_bytes=2*q.transposed_tile_k*(q.transposed_tile_rows+q.transposed_smem_padding)
            if self.dy_preparation=='fused_all':shared_bytes+=4*q.transposed_tile_k
            if shared_bytes+SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES>SM120_SMEM_CAPACITY_BYTES:
                return 'Fused dY tile exceeds shared-memory capacity'
        if self.backward_preparation=='joint':
            if layout!='row_major':return 'Joint backward currently requires row-major scales'
            if self.joint_recompute_engine=='split_warp' and self.active_prefix:
                return 'Split recompute is currently qualified only for static capacity'
            g=self.recompute
            if g.glu_grad_store!='exchange' or g.tile_m%32 or g.tile_n%64 or problems['gate_up'][0].n%64:
                return 'Joint backward needs exchange derivatives and complete 32-value blocks'
            if g.glu_hidden_decode not in ('fp32','fp32_packed'):
                return 'Joint backward keeps forward-hidden decoding in FP32'
        elif phase=='train' and self.joint_recompute_engine!='legacy':
            return 'A joint recompute engine requires joint backward preparation'
        if phase=='train':
            from .moe_backward import persistent_dw_rejection,indexed_gate_dw_rejection
            for stage in ('down_dw','gate_dw'):
                engine=getattr(self,stage+'_engine')
                if engine=='packed':continue
                if getattr(self,stage+'_splits')!=1 or (self.active_prefix and engine=='indexed'):
                    return f'{stage}: alternate ownership requires unsplit reduction; indexed requires static capacity'
                if layout!='row_major':return f'{stage}: alternate ownership requires row-major scales'
                if engine=='persistent':
                    if self.transports[STAGES.index(stage)]!='tma':
                        return f'{stage}: persistent operand reuse requires TMA transport'
                    logical=problems[stage][0]
                    capacity=self.physical_capacity(problems['down'][0].m)
                    reason=persistent_dw_rejection(MXFP8Problem(logical.m,logical.n,capacity),
                        getattr(self,stage),getattr(self,stage+'_reuse_axis'),getattr(self,stage+'_reuse_tiles'))
                    if reason:return f'{stage}: {reason}'
                elif self.input_preparation!='indexed_transpose' or self.backward_preparation!='joint' or self.dy_preparation=='fused_all':
                    return 'Indexed gate dW requires joint indexed input and independently prepared dY'
                else:
                    logical=problems[stage][0]
                    capacity=self.physical_capacity(problems['down'][0].m)
                    reason=indexed_gate_dw_rejection(MXFP8Problem(logical.m,logical.n,capacity),
                        self.gate_dw,self.quant,self.indexed_dw_token_lanes,
                        self.indexed_dw_gather_pairs,self.indexed_dw_producer_warps)
                    if reason:return reason
        if phase=='train' and self.route_gradient_epilogue=='fused':
            if self.backward_preparation!='joint' or self.transports[STAGES.index('down')]!='tma':
                return 'Fused route-gradient epilogue needs joint recomputation and TMA down transport'
            from .moe_backward import route_gradient_scratch_bytes
            g=self.route_gradient_gemm or self.down
            logical=problems['down'][0]
            effective=replace(g,epilogue='direct',epilogue_stages=1,store_vec=1)
            if reason:=effective.rejection(MXFP8Problem(self.physical_capacity(logical.m),logical.n,logical.storage_k)):
                return f'Route-gradient GEMM: {reason}'
            if (g.scale_layout!=layout or g.scale_role=='tma'):
                return 'Route-gradient loader requires matching row-major operand scales'
            if (g.num_mma_warps*g.consumer_registers+g.producer_registers)*32>65536:
                return 'Route-gradient GEMM exceeds CTA register-role budget'
            if (g.num_mma_warps*g.consumer_registers+g.producer_registers)*32>g.num_threads*g.maxrregcount:
                return 'Route-gradient register roles exceed the initial CTA ceiling'
            if g.producer_registers>min(g.maxrregcount,65536//g.num_threads):
                return 'Route-gradient producer register decrease exceeds initial allocation'
            qbytes=g.stages*(g.tile_m+g.tile_n)*g.tile_k
            scales=g.stages*(((g.tile_m+127)//128)*128+((g.tile_n+127)//128)*128)*(g.tile_k//32)
            partials=route_gradient_scratch_bytes(g,self.route_gradient_loader)
            if qbytes+scales+partials+SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES>SM120_SMEM_CAPACITY_BYTES:
                return 'Fused route-gradient row partials exceed shared-memory capacity'
        if layout not in ('row_major','mma128') or self.quant.scale_rounding!='infinity':
            return 'Prepared experts require infinity-rounded row-major/native128 scales'
        if self.quant.native_scale_store!='scalar':return 'Prepared portfolio currently uses scalar scale stores'
        if layout=='mma128' and (problems['down'][0].n%128 or problems['down'][0].k%128):
            return 'Native128 expert scales require D and H divisible by 128'
        for name,(logical,_,_) in problems.items():
            if phase=='forward' and name not in ('gate_up','down'):continue
            if name=='gate_dw' and self.gate_dw_engine=='indexed':
                # The on-chip quantizer owns a separate cooperative/dedicated
                # staging schedule, already checked above without allocating.
                continue
            m,k=logical.m,logical.storage_k
            capacity=self.physical_capacity(problems['down'][0].m)
            if name in ('down_dw','gate_dw'):k=(capacity+31)//32*32
            else:m=capacity
            problem=MXFP8Problem(m,logical.n,k)
            config=getattr(self,name)
            transport=self.transports[STAGES.index(name)]
            if config.tile_k in (32,64):
                if name not in ('down_dw','gate_dw') or transport!='cooperative' or getattr(self,name+'_splits')!=1:
                    return f'{name}: sub-128 K tiles require unsplit cooperative weight gradients'
                if layout!='row_major':return f'{name}: short dW requires row-major prepared scales'
                if reason:=short_dw_rejection(problem,config):return f'{name}: {reason}'
                continue
            if name in ('down_dw','gate_dw') and getattr(self,name+'_splits')>1:
                split=getattr(self,name+'_splits')
                if transport!='tma' or config.epilogue!='direct' or config.tiles_per_cta!=1 or config.persistent_waves:
                    return f'{name}: split reduction requires direct TMA GEMM without output persistence'
                if split>(k+config.tile_k-1)//config.tile_k:
                    return f'{name}: split reduction cannot create empty K-tile partitions'
            if transport=='cooperative':
                if name in ('gate_up','recompute'):return 'Cooperative transport is currently linear-only'
                if (layout!='row_major' or config.stages!=1 or config.epilogue!='direct' or config.tiles_per_cta!=1
                        or config.persistent_waves or config.tile_locality!='raster'):
                    return f'{name}: cooperative transport needs one row-scale buffer and ordinary tile ownership'
            if (config.scale_layout!=layout
                    or ((config.scale_role=='tma')!=(layout=='mma128'))):
                return f'{name}: scale transport must match the quantized physical layout'
            if (config.num_mma_warps*config.consumer_registers+config.producer_registers)*32>65536:
                return f'{name}: impossible dynamic CTA register-role budget'
            if (config.num_mma_warps*config.consumer_registers+config.producer_registers)*32>config.num_threads*config.maxrregcount:
                return f'{name}: dynamic register roles exceed the configured initial CTA ceiling'
            if config.producer_registers>min(config.maxrregcount,65536//config.num_threads):
                return f'{name}: producer register decrease exceeds initial allocation'
            reason=config.rejection(problem)
            if name in ('gate_up','recompute'):
                if config.epilogue!='direct':return f'{name}: GLU requires direct output'
                if config.scale_rounding!='infinity' or config.glu_silu_math not in ('division','shared_reciprocal'):
                    return f'{name}: preserve FP32 SiLU and infinity-rounded hidden scales'
                if config.glu_store!='direct':return f'{name}: grouped GLU stores must be direct'
                if config.glu_quant_vec not in (1,2,4,8):return f'{name}: unsupported GLU conversion vector'
                if name=='recompute':
                    if config.glu_epilogue!='registers':return 'Prepared derivative needs paired-register epilogue'
                    reason=reason or prequant_recompute_rejection(config)
                    if self.backward_preparation=='joint':
                        engine=self.joint_recompute_engine
                        reason=reason or joint_recompute_rejection(problem,config,engine,self.joint_recompute_gradient_warps)
                        # Operand-reuse engines retire A/B; split_warp also
                        # reserves a decoded plane checked by the joint guard.
                        reason=reason or MXFP8GLUPrequantConfig(quant=self.quant,gemm=config).rejection(
                            problem,'mxfp8' if engine=='legacy' else 'bf16')
                else:
                    reason=reason or MXFP8GLUPrequantConfig(quant=self.quant,gemm=config).rejection(
                        problem,'mxfp8' if self.forward_glu_engine=='legacy' else 'bf16')
            if reason:return f'{name}: {reason}'
        return None


DEFAULT_PREPARED_MOE_CONFIG=MXFP8PreparedMoEConfig()
