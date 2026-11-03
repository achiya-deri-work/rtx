"""Shape-derived, complete backward families shared by linear and MLP tuners."""
from dataclasses import replace
import math

from .kernels.mxfp8 import MXFP8Problem


def reduction_chunk(length, parts, alignment):
    """Smallest aligned covering chunk; legality still rejects empty partitions."""
    return ((length + parts * alignment - 1) // (parts * alignment)) * alignment


def decomposed_leg(leg, problem):
    rows = max(32, math.gcd(128, problem.m, problem.n))
    quant = replace(leg.quant_a, scale_layout='row_major', native_scale_store='scalar',
                    transposed_tile_rows=rows, transposed_tile_k=32,
                    transposed_load_engine='register')
    gemm = replace(leg.gemm, tile_m=128, tile_n=128, tile_k=128,
        atom_layout_m=4, atom_layout_n=2, stages=2,
        scale_layout='row_major', scale_role='producer', epilogue='direct',
        epilogue_stages=1, store_vec=1, tiles_per_cta=1, tile_locality='raster')
    return replace(leg, backend='decomposed', quant_launches='separate',
        quant_a=quant, quant_b=None, gemm=gemm, reduction='full_fp32',
        split_reduction=1, reduction_tile=0, workspace_epilogue='none',
        tile_scheduler='static', persistent_waves=1, tiles_per_cta=1,
        reuse_operand='none', tile_locality='raster')


def native_decomposed_leg(leg, problem):
    """Existing native layouts with orientation-aware quantizer tiles.

    A half-128 row extent can use MMA64 SFA and MMA128 SFB when N/K
    permit it. Keep each operand's numerical policy; never relax the native
    reduction-axis or transpose-layout contracts.
    """
    from .fp8 import _default_dynamic_prequant_config
    seed = _default_dynamic_prequant_config(problem)
    if seed is None or seed.gemm.scale_layout == 'row_major':
        return None
    def quant(config, rows, layout):
        return replace(config, scale_layout=layout, native_scale_store='scalar',
            transposed_tile_rows=max(32, math.gcd(128, rows)))
    candidate = replace(decomposed_leg(leg, problem),
        quant_a=quant(leg.quant_a, problem.m, seed.quant.scale_layout),
        quant_b=quant(leg.resolved_quant_b(), problem.n, seed.resolved_weight_quant().scale_layout),
        gemm=seed.gemm)
    return candidate if candidate.implementation_rejection(problem) is None else None


def split_leg(leg, length, parts):
    if parts == 1:
        return replace(leg, reduction='full_fp32', split_reduction=1,
                       reduction_tile=0, workspace_epilogue='none')
    alignment = leg.gemm.tile_k if leg.backend=='decomposed' else leg.fused.bf16_tile_k
    return replace(leg, reduction='split_fp32_workspace', split_reduction=parts,
        reduction_tile=reduction_chunk(length,parts,alignment),workspace_epilogue='serial',
        fused=replace(leg.fused,epilogue='direct',epilogue_stages=1,store_vec=1),
        gemm=replace(leg.gemm,epilogue='direct',epilogue_stages=1,store_vec=1,tiles_per_cta=1))


def materialized_bwd_seed(problem,initial,parts=1):
    """Complete legal dX/dW family, including native dX and masked reductions.

    A composite operator must be able to enter both legs together: changing
    only one projection's backward can leave another dominant fused fallback.
    This is a starting candidate, not a timing winner or relaxed validator.
    """
    dx_problem=MXFP8Problem(problem.m,problem.k,problem.n)
    dw_problem=MXFP8Problem(problem.n,problem.k,problem.m)
    dx=native_decomposed_leg(initial.dx,dx_problem) or decomposed_leg(initial.dx,dx_problem)
    dw=decomposed_leg(initial.dw,dw_problem)
    dw=split_leg(dw,problem.m,parts)
    candidate=replace(initial,dx=dx,dw=dw,quant_schedule='per_matmul',stream_schedule='single',
                      execution_order='dx_first',operand_storage='separate')
    return candidate if candidate.implementation_rejection(problem) is None else None


def backward_structural_seeds(problem, initial):
    """Reserve useful alternatives to underfilled unsplit dW, never a winner."""
    dw_problem = MXFP8Problem(problem.n,problem.k,problem.m)
    dx_problem = MXFP8Problem(problem.m,problem.k,problem.n)
    base = replace(initial,quant_schedule='per_matmul',stream_schedule='single',
                   execution_order='dx_first',operand_storage='separate')
    seeds = []
    # Interleave implementations: opening trials cover complete families.
    families = (decomposed_leg(base.dw,dw_problem), replace(base.dw,backend='fused'))
    for parts in (32,16,8,4,2,1):
        for leg in families:
            candidate = replace(base,dw=split_leg(leg,problem.m,parts))
            if candidate.implementation_rejection(problem) is None and candidate not in seeds:
                seeds.append(candidate)
    dx = decomposed_leg(base.dx,dx_problem)
    for current in (base, *seeds[:2]):
        candidate=replace(current,dx=dx)
        if candidate.implementation_rejection(problem) is None and candidate not in seeds:
            seeds.append(candidate)
    combined=next((c for c in seeds if c.dx.backend=='decomposed' and c.dw.backend=='decomposed'
                   and c.dw.split_reduction>1),None)
    if combined is not None:
        seeds.remove(combined)
        seeds.insert(1,combined)
    native_dx = native_decomposed_leg(base.dx, dx_problem)
    if native_dx is not None and seeds:
        candidate = replace(seeds[0], dx=native_dx)
        if candidate.implementation_rejection(problem) is None and candidate not in seeds:
            # Reach native dX + split materialized dW inside small budgets,
            # retaining the row-major and fused alternatives as candidates.
            seeds.insert(1,candidate)
    # Storage policy is a real axis, including the partial-call topology used
    # by recomputed MLPs. Never activate aliasing for overlapping/quad schedules.
    for current in tuple(seeds):
        for storage in ('serial_reuse','serial_reuse_partials'):
            candidate=replace(current,operand_storage=storage)
            if candidate.implementation_rejection(problem) is None and candidate not in seeds:
                seeds.append(candidate)
    return seeds


def backward_search_space(problem, axes):
    """Extend declared coordinates with shape-derived chunks and compound moves."""
    from .kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    from .bwd_autotune import bwd_config_to_dict
    result = dict(axes)
    def unique(values):
        result=[]
        for value in values:
            if value not in result:result.append(value)
        return tuple(result)
    for name, length, p in (('dx',problem.n,MXFP8Problem(problem.m,problem.k,problem.n)),
                            ('dw',problem.m,MXFP8Problem(problem.n,problem.k,problem.m))):
        backend_key=f'{name}_backend'
        if backend_key in result and any(v.get(name,{}).get('backend')=='decomposed' for v in result[backend_key]):
            leg=decomposed_leg(getattr(DEFAULT_FUSED_MXFP8_BWD_CONFIG,name),p)
            payload=bwd_config_to_dict(replace(DEFAULT_FUSED_MXFP8_BWD_CONFIG,**{name:leg}))[name]
            payload['quant_b']=dict(payload['quant_a'])
            result[backend_key]=unique((*result[backend_key], {name:payload}))
            native=native_decomposed_leg(getattr(DEFAULT_FUSED_MXFP8_BWD_CONFIG,name),p)
            if native is not None:
                payload=bwd_config_to_dict(replace(DEFAULT_FUSED_MXFP8_BWD_CONFIG,**{name:native}))[name]
                result[backend_key]=unique((*result[backend_key], {name:payload}))
        reduction_key=f'{name}_reduction'
        if reduction_key in result and any(v.get(name,{}).get('reduction')=='split_fp32_workspace' for v in result[reduction_key]):
            moves=[]
            for parts in (2,4,8,16,32):
                for alignment in (128,256,512):
                    tile=reduction_chunk(length,parts,alignment)
                    if (parts-1)*tile>=length:
                        continue
                    moves.append({name:dict(reduction='split_fp32_workspace',split_reduction=parts,
                        reduction_tile=tile,workspace_epilogue='serial',
                        fused=dict(epilogue='direct',epilogue_stages=1,store_vec=1),
                        gemm=dict(epilogue='direct',epilogue_stages=1,store_vec=1,tiles_per_cta=1))})
            result[reduction_key]=unique((*result[reduction_key],*moves))
    return result


def active_backward_dict(config):
    """Deduplicate only fields proven not to execute; keep full serialization."""
    from dataclasses import asdict
    result=asdict(config.normalized())
    for name in ('dx','dw'):
        leg=result[name]
        inactive=('quant_a','quant_b','quant_launches','gemm') if leg['backend']=='fused' else ('fused',)
        for key in inactive:
            leg.pop(key,None)
        if leg['reduction']=='full_fp32':
            for key in ('reduction_threads','reduction_vector','reduction_waves'):
                leg.pop(key,None)
    return result


def backward_coverage(configs, problem):
    """Observed structural geometry, not an occupancy or utilization claim."""
    result={name:[] for name in ('dx','dw')}
    for config in configs:
        for name,m,n in (('dx',problem.m,problem.k),('dw',problem.n,problem.k)):
            leg=getattr(config,name)
            tile=leg.fused if leg.backend=='fused' else leg.gemm
            row=dict(backend=leg.backend,reduction=leg.reduction,parts=leg.split_reduction,
                tile_m=tile.tile_m,tile_n=tile.tile_n,
                grid_ctas=((m+tile.tile_m-1)//tile.tile_m)*((n+tile.tile_n-1)//tile.tile_n)*leg.split_reduction)
            if row not in result[name]:result[name].append(row)
    return result
