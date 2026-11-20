"""CPU-only legality for optional routed-backward ownership schedules."""
from dataclasses import replace
from ..kernels.mxfp8 import SM120_SMEM_CAPACITY_BYTES,SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES


def token_reduce_rejection(config,shape,phase='train'):
    """Shape-only legality, shared by explicit plans and tuning proposals."""
    _,width,_,_,top_k,_=shape
    for kind in (('combine','dx') if phase!='forward' else ('combine',)):
        if getattr(config,kind+'_reduce_engine')!='token':continue
        if width%getattr(config,kind+'_vector_width'):
            return f'{kind}: token reduction requires width divisible by vector width'
        if top_k>getattr(config,kind+'_lanes_per_token'):
            return f'{kind}: token reduction needs enough lanes for selected experts'
    return None


def tma_register_roles_rejection(config):
    total=(config.num_mma_warps*config.consumer_registers+config.producer_registers)*32
    if total>min(65536,config.num_threads*config.maxrregcount):
        return 'Dynamic TMA register roles exceed the hardware or initial CTA budget'
    if config.producer_registers>min(config.maxrregcount,65536//config.num_threads):
        return 'TMA producer register decrease exceeds its initial allocation'
    return None


def route_gradient_scratch_bytes(config,loader):
    if loader not in ('accumulator','shared_vec4','warp_vec2'):
        raise ValueError('Unknown route-gradient loader')
    return 4*config.tile_m*(config.atom_layout_n+(config.tile_n+1 if loader=='shared_vec4' else 0))


def persistent_dw_rejection(problem,config,axis='n',tiles=2):
    if min(problem.m,problem.n)<2:
        return 'Persistent dW currently requires nonsingleton output dimensions'
    if reason:=config.rejection(problem):return reason
    if axis not in ('m','n') or type(tiles) is not int or tiles not in (2,3):
        return 'Persistent dW needs m/n reuse and two or three accumulator tiles'
    extent=problem.m if axis=='m' else problem.n
    if (extent+63)//64<tiles:return 'Persistent dW needs enough neighboring output tiles'
    if (config.tile_m,config.tile_n,config.tile_k,config.atom_layout_m,config.atom_layout_n)!=(64,64,128,2,2):
        return 'Persistent dW requires independent 64x64 tiles and K128 transport'
    if (config.scale_layout!='row_major' or config.scale_role!='consumers'
            or config.scale_smem_store!='scalar' or config.scale_load_vec!=1):
        return 'Persistent dW requires scalar consumer row-scale transport'
    if (config.mma_schedule!='interleaved' or config.epilogue!='direct'
            or config.tiles_per_cta!=1 or config.persistent_waves):
        return 'Persistent dW requires direct independent ownership without generic persistence'
    if problem.k%32:return 'Persistent dW reduction must contain complete 32-value scale blocks'
    shared=(tiles+1)*config.stages*(64*128+128*4)+SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES
    if shared>SM120_SMEM_CAPACITY_BYTES:return 'Persistent dW operand panels exceed shared-memory capacity'
    return None


def indexed_gate_dw_schedule(gemm,producer_warps):
    schedule=replace(gemm,stages=gemm.stages if producer_warps else 1,
        scale_role='producer',scale_load_vec=1,scale_schedule='before_wait',
        scale_recycle='barrier',scale_smem_store='scalar',scale_l2_prefetch='none',
        scale_l1_evict='default',scale_cache='default')
    if gemm.tile_k<128:
        from .moe import short_dw_config
        schedule=short_dw_config(replace(schedule,a_swizzle='32b',b_swizzle='32b'))
    return schedule


def indexed_gate_dw_rejection(problem,gemm_config,quant_config,token_lanes=8,gather_pairs=1,producer_warps=0):
    g,q=gemm_config,quant_config
    if (g.scale_layout!='row_major' or q.scale_layout!='row_major' or q.native_scale_store!='scalar'):
        return 'Indexed gate dW needs independent row-major scales'
    if (g.tiles_per_cta!=1 or g.persistent_waves or g.tile_locality!='raster'
            or g.epilogue!='direct' or g.store_vec!=1):
        return 'Indexed gate dW needs one output tile per CTA and direct scalar stores'
    if g.tile_k not in (32,64,128) or problem.k%32:
        return 'Indexed gate dW needs full 32-slot blocks and K32/K64/K128 staging'
    if g.tile_k<128 and (g.tile_m,g.tile_n,g.atom_layout_m,g.atom_layout_n)!=(64,64,2,2):
        return 'Indexed K32/K64 staging uses 64x64 tiles and four MMA warps'
    if type(token_lanes) is not int or token_lanes not in (2,4,8):
        return 'Indexed transpose needs 2/4/8 token lanes'
    if type(gather_pairs) is not int or gather_pairs not in (1,2):
        return 'Indexed transpose gathers one or two BF16 pairs per lane'
    features=(32//token_lanes)*2*gather_pairs
    if g.tile_n%features:return 'Indexed gather strips must partition the output feature tile'
    if type(producer_warps) is not int or producer_warps not in (0,2,4,8):
        return 'Indexed gate dW needs cooperative or 2/4/8 dedicated producer warps'
    if g.num_mma_warps+producer_warps>32:return 'Indexed gate dW exceeds one CTA thread limit'
    if producer_warps and not 2<=g.stages<=3:return 'Dedicated indexed producers need two or three stages'
    if max(g.tile_m,g.tile_n)>128:return 'Indexed gate dW supports at most 128 rows/columns per output tile'
    return indexed_gate_dw_schedule(g,producer_warps).rejection(problem) or q.rejection(problem.n,problem.k)


def fragment_glu_scratch_bytes(config,engine):
    if engine not in ('fragment_direct','fragment_packed'):raise ValueError('Unknown fragment GLU engine')
    return 8*config.tile_m*(config.tile_n//64)+(config.tile_m*(config.tile_n//2) if engine=='fragment_packed' else 0)


def fragment_glu_rejection(problem,config,engine,activation='silu',output_format='mxfp8'):
    if engine not in ('fragment_direct','fragment_packed'):return 'Unknown fragment GLU engine'
    if activation!='silu' or output_format!='mxfp8':return 'Fragment GLU currently supports MXFP8 SiLU output'
    g=config
    if problem.k%32:return 'Fragment GLU requires complete physical K32 scale blocks'
    if problem.m<2 or problem.n%2:return 'Fragment GLU requires nonsingleton rows and paired projections'
    if (g.tile_m,g.atom_layout_m) not in ((64,2),(128,8)) or g.tile_n not in (64,128) or g.atom_layout_n!=2:
        return 'Fragment GLU requires a qualified 64/128-row paired-warp layout'
    if (g.scale_layout!='row_major' or g.epilogue!='direct' or g.glu_store!='direct'
            or g.glu_epilogue!='registers' or g.glu_silu_math!='division' or g.scale_rounding!='infinity'):
        return 'Fragment GLU preserves row-major direct register SiLU and infinity scales'
    if g.tiles_per_cta!=1 or g.persistent_waves:return 'Fragment GLU currently requires one static output tile per CTA'
    if reason:=g.rejection(problem):return reason
    if reason:=tma_register_roles_rejection(g):return reason
    scales=(((g.tile_m+127)//128)*128+((g.tile_n+127)//128)*128)*(g.tile_k//32)
    smem=g.stages*((g.tile_m+g.tile_n)*g.tile_k+scales)+fragment_glu_scratch_bytes(g,engine)+2048
    if smem>SM120_SMEM_CAPACITY_BYTES:return 'Fragment GLU scratch exceeds shared-memory capacity'
    return None


def backward_weight_quant_rejection(n,d,e,h,quant,grid_mode='operand',tile_rows=32,tile_cols=64,
                                   load_values=4,store_bits=32,weight_warps=4,raster='features'):
    if any(type(v) is not int or v<1 for v in (n,d,e,h)):return 'Weight preparation requires positive integer dimensions'
    padded=lambda v:(v+31)//32*32
    if max(n*padded(d),e*2*h*padded(d),e*d*padded(2*h),e*d*padded(h),e*h*padded(d))>=2**31:
        return 'Weight preparation exceeds native signed index range'
    if reason:=quant.rejection(n,d):return reason
    if quant.scale_layout!='row_major' or quant.native_scale_store!='scalar':return 'Paired weight preparation needs scalar row-major scales'
    if grid_mode not in ('operand','expert','portfolio'):return 'Unknown weight preparation grid'
    if tile_rows not in (32,64,128) or tile_cols not in (32,64,128) or tile_rows*tile_cols>8192:
        return 'Weight preparation requires bounded complete 32x32 scale tiles'
    if load_values not in (1,2,4,8) or store_bits not in (8,16,32) or store_bits>8*quant.quant_vec:
        return 'Unsupported weight preparation load/store vector'
    if weight_warps not in (2,4,8) or raster not in ('features','rows'):return 'Unsupported weight preparation warp/raster schedule'
    if 2*tile_rows*tile_cols+SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES>SM120_SMEM_CAPACITY_BYTES:
        return 'Weight preparation tile exceeds shared-memory capacity'
    return None
