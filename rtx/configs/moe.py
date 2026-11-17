"""CPU-safe quantize-once expert schedules, independent for every adjoint."""
from dataclasses import asdict,dataclass,field,replace

from .mxfp8 import MXFP8GemmConfig,MXFP8QuantConfig
from .glu import MXFP8GLUGemmConfig,MXFP8GLUPrequantConfig,prequant_recompute_rejection
from ..kernels.mxfp8 import MXFP8Problem


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

    def __post_init__(self):
        if self.transpose_mode not in ('direct','staged'):raise ValueError('transpose_mode must be direct/staged')
        if self.backward_preparation not in ('separate','joint'):raise ValueError('backward_preparation must be separate/joint')
        if not isinstance(self.transports,tuple) or len(self.transports)!=len(STAGES) or any(t not in ('tma','cooperative') for t in self.transports):
            raise ValueError('transports must specify tma/cooperative for each of the seven GEMMs')
        if type(self.capacity_multiple) is not int or self.capacity_multiple not in (1,32,64,128,256):
            raise ValueError('capacity_multiple must be 1/32/64/128/256')
        if self.quant_grid not in ('operand','expert','portfolio'):
            raise ValueError('quant_grid must be operand/expert/portfolio')
        if self.input_preparation not in ('gather_bf16','quantize_once'):
            raise ValueError('input_preparation must be gather_bf16/quantize_once')
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
        return replace(self,transports=tuple(transports),**{stage:c})

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
        return cls(**result)

    def rejection(self,problems,phase='train'):
        layout=self.quant.scale_layout
        reason=self.quant.rejection(128,128)
        if reason:return f'Operand quantizer: {reason}'
        if self.backward_preparation=='joint':
            if layout!='row_major':return 'Joint backward currently requires row-major scales'
            g=self.recompute
            if g.glu_grad_store!='exchange' or g.tile_m%32 or g.tile_n%64 or problems['gate_up'][0].n%64:
                return 'Joint backward needs exchange derivatives and complete 32-value blocks'
            if g.glu_hidden_decode not in ('fp32','fp32_packed'):
                return 'Joint backward keeps forward-hidden decoding in FP32'
        if layout not in ('row_major','mma128') or self.quant.scale_rounding!='infinity':
            return 'Prepared experts require infinity-rounded row-major/native128 scales'
        if self.quant.native_scale_store!='scalar':return 'Prepared portfolio currently uses scalar scale stores'
        if layout=='mma128' and (problems['down'][0].n%128 or problems['down'][0].k%128):
            return 'Native128 expert scales require D and H divisible by 128'
        for name,(logical,_,_) in problems.items():
            if phase=='forward' and name not in ('gate_up','down'):continue
            m,k=logical.m,logical.storage_k
            capacity=self.physical_capacity(problems['down'][0].m)
            if name in ('down_dw','gate_dw'):k=(capacity+31)//32*32
            else:m=capacity
            problem=MXFP8Problem(m,logical.n,k)
            config=getattr(self,name)
            transport=self.transports[STAGES.index(name)]
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
                        reason=reason or MXFP8GLUPrequantConfig(quant=self.quant,gemm=config).rejection(problem,'mxfp8')
                else:
                    reason=reason or MXFP8GLUPrequantConfig(quant=self.quant,gemm=config).rejection(problem,'mxfp8')
            if reason:return f'{name}: {reason}'
        return None


DEFAULT_PREPARED_MOE_CONFIG=MXFP8PreparedMoEConfig()
