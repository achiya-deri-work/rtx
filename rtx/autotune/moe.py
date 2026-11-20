"""Shared-engine whole grouped-expert tuning; policy and numerics stay fixed."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
from threading import RLock
import statistics
import time

from .core import DiscreteKernelAdapter,KernelContext,Proposal,stable_id
from .effort import AutotunePolicy,make_effort_autotuner,effort_residual_root
from .outcomes import TrialOutcome,raise_if_fatal_device_context_error
from .store import ResidualTuningStore
from ..moe import MXFP8MoEConfig,DEFAULT_MOE_CONFIG
from ..kernels.mxfp8 import MXFP8Problem,normalize_fwd_config
from ..configs.glu import normalize_glu_config,glu_rejection
from ..configs.moe import DEFAULT_PREPARED_MOE_CONFIG,JOINT_RECOMPUTE_ENGINES,STAGES
from ..configs.moe_backward import route_gradient_scratch_bytes
from ..runtime import BoundedCache

MOE_KERNEL_REVISION=15
MEASUREMENT_CONTRACT=dict(revision=1,timing='cuda_graph',objective='whole_routed_expert_path',
                          oracle='torch_fp64_quantized',gradient_nrms=.006,output_nrms=.003)
_choices=BoundedCache(256)
_lock=RLock()
_LOG=logging.getLogger('rtx.moe.autotune')


@lru_cache(maxsize=1)
def source_revision():
    root=Path(__file__).resolve().parents[1]
    names=('moe.py','_moe_reference.py','autotune/moe.py','kernels/mxfp8_moe.py',
           'kernels/mxfp8_fwd.py','kernels/mxfp8_glu.py','kernels/mxfp8_mlp.py',
           'kernels/mxfp8.py','kernels/mxfp8_moe_onchip.py','configs/glu.py',
           '_execution_plan_token.py','_moe_prepared.py','_moe_execution.py','_moe_selection.py','_moe_shared.py','configs/moe.py','configs/mxfp8.py',
           'kernels/mxfp8_moe_prepared.py','kernels/mxfp8_gemm.py','kernels/mxfp8_quant.py',
           'kernels/mxfp8_glu_prequant.py','kernels/mxfp8_moe_cooperative.py','kernels/mxfp8_moe_input.py',
           'kernels/mxfp8_moe_buffers.py')
    names=(*names,'autotune/output_transport.py','kernels/mxfp8_moe_warp_quant.py','kernels/mxfp8_moe_route_epilogue.py',
           'kernels/mxfp8_moe_dy.py','kernels/mxfp8_moe_backward_quant.py','kernels/mxfp8_moe_recompute.py',
           'kernels/mxfp8_moe_active.py','kernels/mxfp8_moe_short_dw.py',
           'configs/moe_backward.py','kernels/mxfp8_moe_indexed_dw.py','kernels/mxfp8_moe_persistent_dw.py')
    names=(*names,'kernels/mxfp8_moe_recompute_split.py','kernels/mxfp8_moe_glu_epilogue.py',
           'kernels/mxfp8_moe_weight_quant.py','kernels/mxfp8_moe_token_reduce.py',
           '_moe_compact_layout.py','kernels/mxfp8_moe_compact.py')
    digest=hashlib.sha256()
    for name in names:
        digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def problems(shape):
    n,d,e,c,k,h=shape
    return dict(gate_up=(MXFP8Problem(c,2*h,d),'row','row'),
                down=(MXFP8Problem(c,d,h),'row','row'),
                recompute=(MXFP8Problem(c,2*h,d),'row','row'),
                down_dx=(MXFP8Problem(c,h,d),'row','transpose'),
                down_dw=(MXFP8Problem(d,h,c),'transpose','transpose'),
                gate_dx=(MXFP8Problem(c,d,2*h),'row','transpose'),
                gate_dw=(MXFP8Problem(2*h,d,c),'transpose','transpose'))


def prepared_dw_geometry_features(config,shape,phase,device):
    """Static scheduled dW work/coverage, not measured utilization.

    In active-prefix execution runtime K can be smaller than this bound.
    """
    from .hardware import profile_value
    if phase!='train' or config.family!='grouped_prepared':return {}
    n,d,e,c,top_k,h=shape;p=config.prepared
    physical_k=(p.physical_capacity(c)+31)//32*32
    real_k=max(1.,n*top_k/e)
    sms=int(profile_value(device,'multiprocessor_count',0) or 0)
    features={}
    for stage,m,w in (('down_dw',d,h),('gate_dw',2*h,d)):
        g=getattr(p,stage)
        padded_k=((physical_k+g.tile_k-1)//g.tile_k)*g.tile_k
        mt=(m+g.tile_m-1)//g.tile_m;nt=(w+g.tile_n-1)//g.tile_n
        tile_ctas=mt*nt*getattr(p,stage+'_splits')
        launch_ctas=(tile_ctas+g.tiles_per_cta-1)//g.tiles_per_cta
        if getattr(p,stage+'_engine')=='persistent':
            group=getattr(p,stage+'_reuse_tiles')
            launch_ctas=(mt*((nt+group-1)//group) if getattr(p,stage+'_reuse_axis')=='n'
                         else ((mt+group-1)//group)*nt)
        if g.persistent_waves and sms:
            launch_ctas=max(launch_ctas,min(tile_ctas,sms*g.persistent_waves))
        features.update({
            f'{stage}_padded_k':padded_k,
            f'{stage}_padded_reduction_ratio':padded_k/real_k,
            f'{stage}_tile_work_amplification':(mt*g.tile_m*nt*g.tile_n*padded_k)/(m*w*real_k),
            f'{stage}_grid_ctas':e*launch_ctas,
        })
        if sms:features[f'{stage}_cta_waves_single_residency']=e*launch_ctas/sms
    return features


def prepared_row_store_features(config,shape,phase):
    from .output_transport import row_store_features
    if config.family!='grouped_prepared':return {}
    p=config.prepared
    physical=(*shape[:3],p.physical_capacity(shape[3]),*shape[4:])
    output={}
    for stage in (('down','gate_dx','down_dx') if phase=='train' else ('down',)):
        if stage=='down' and p.forward_engine=='onchip':continue
        output.update({stage+'.'+key:value for key,value in
            row_store_features(getattr(p,stage),problems(physical)[stage][0]).items()})
    return output


def rejection(config,shape,phase='train'):
    from ..configs.moe_backward import token_reduce_rejection
    if why:=token_reduce_rejection(config,shape,phase):return why
    if config.family=='grouped_prepared':
        if why:=config.prepared.compact_forward_rejection(shape):return why
        if phase=='train' and config.prepared.weight_preparation=='dual_orientation':
            from ..configs.moe_backward import backward_weight_quant_rejection
            n,d,e,_,_,h=shape
            p=config.prepared
            why=backward_weight_quant_rejection(n,d,e,h,p.quant,p.quant_grid,p.weight_tile_rows,
                p.weight_tile_cols,p.weight_load_values,p.weight_store_bits,p.weight_warps,p.weight_raster)
            if why:return why
        if config.prepared.forward_engine=='onchip':
            physical=(*shape[:3],config.prepared.physical_capacity(shape[3]),*shape[4:])
            why=rejection(replace(config,family='grouped_onchip'),physical,'forward')
            if why:return why
        return config.prepared.rejection(problems(shape),phase)
    if config.family=='grouped_onchip':
        g,b=config.gate_up,config.down
        if shape[5]!=128:return 'on-chip family currently supports H=128; staged covers other widths'
        if (g.tile_n!=256 or g.tile_k!=128 or g.persistent or g.mxfp8_stages!=1
                or g.glu_scratch!='dedicated'):
            return 'on-chip gate must own full H=128 with dedicated scratch and one native stage'
        if (b.tile_m!=g.tile_m or b.tile_k!=128 or b.tile_n not in (64,128)
                or b.quant_vec<2 or b.mxfp8_stages!=1):return 'invalid on-chip down geometry'
        if config.onchip_down_stages*b.tile_n>256:return 'on-chip down buffers exceed dead gate storage'
        if g.num_mma_warps<=b.num_mma_warps:return 'On-chip handoff needs spare gate MMA warps'
    stages=problems(shape)
    if phase=='forward':stages={key:stages[key] for key in ('gate_up','down')}
    for name,(problem,a,b) in stages.items():
        c=getattr(config,name)
        if c.epilogue!='direct' or c.cluster_reuse!='none' or c.cluster_size!=1:
            return f'{name}: grouped stores/clusters are unqualified'
        reason=(glu_rejection(problem,c,'mxfp8' if name=='gate_up' else 'bf16')
                if name in ('gate_up','recompute') else c.oriented_implementation_rejection(problem,a,b))
        if reason:return f'{name}: {reason}'
        if c.scale_rounding!='infinity':return f'{name}: infinity-rounded scales required'
        if name in ('gate_up','recompute') and c.glu_silu_math!='division':
            return f'{name}: the current precision contract uses FP32 SiLU division'
        if c.load_engine=='tma' and ((a=='transpose' and c.tile_m<64 and c.bf16_swizzle=='128b')
                                  or (b=='transpose' and c.tile_n<64)):
            return f'{name}: TMA transpose staging atom exceeds tile extent'
    return None


def make_moe_adapter(shape,evaluator,*,initial=DEFAULT_MOE_CONFIG,phase='train',device=None,tags=None,
                     allow_onchip=True):
    if phase not in ('forward','train'):raise ValueError('phase must be forward/train')
    n,d,e,c,k,h=shape
    shape_problems=problems(shape)
    stages=('gate_up','down') if phase=='forward' else tuple(shape_problems)
    def changed(value,**fields):
        # Configs are frozen values. Revalidating/reallocating an unchanged
        # seven-stage tree on every portfolio mutation is pure setup cost.
        if all(getattr(value,name) is field or getattr(value,name)==field
               for name,field in fields.items()):return value
        return replace(value,**fields)
    @lru_cache(maxsize=4096)
    def normalize(config):
        movement={}
        for kind in ('combine','dx'):
            if kind=='dx' and phase=='forward':movement[kind+'_reduce_engine']='scalar'
            if getattr(config,kind+'_reduce_engine')=='scalar' or (kind=='dx' and phase=='forward'):
                movement.update({kind+'_vector_width':4,kind+'_lanes_per_token':32})
        config=changed(config,**movement)
        if config.family=='grouped_prepared':
            fields={name:getattr(DEFAULT_MOE_CONFIG,name) for name in shape_problems}
            fields.update(onchip_down_stages=1,onchip_pipeline='serial')
            # Packed FP8 gathering has its own loader, not RoutedMovement's
            # BF16 gather geometry. Otherwise flat/warp time identical launch
            # plans and consume distinct small-budget trials. The on-chip
            # forward still gathers BF16 and must retain that real choice.
            if config.prepared.input_preparation!='gather_bf16' and config.prepared.forward_engine!='onchip':
                fields['gather_mode']='flat'
            if config.prepared.forward_engine=='onchip':
                legacy=normalize(replace(config,family='grouped_onchip'))
                fields.update(gate_up=legacy.gate_up,down=legacy.down,
                    onchip_down_stages=config.onchip_down_stages,onchip_pipeline=config.onchip_pipeline)
            changes={}
            for name in stages:
                c=getattr(config.prepared,name);fields_c={}
                if config.prepared.transports[STAGES.index(name)]=='cooperative':
                    fields_c.update(producer_registers=24,consumer_registers=64,scale_role='producer',
                        scale_load_vec=1,scale_smem_store='scalar',scale_l2_prefetch='none',scale_l1_evict='default',
                        scale_cache='default',scale_schedule='before_wait',scale_recycle='barrier',mma_schedule='interleaved')
                if c.tile_locality!='raster':fields_c.update(raster='n',grid_swizzle=2)
                physical_k=((config.prepared.physical_capacity(shape[3])+31)//32*32
                            if name in ('down_dw','gate_dw') else shape_problems[name][0].storage_k)
                if physical_k%c.tile_k:
                    fields_c.update(scale_load_vec=1,scale_smem_store='scalar',scale_l2_prefetch='none',
                                    scale_l1_evict='default',scale_cache='default')
                if name=='gate_up':fields_c.update(glu_hidden_decode='fp32',glu_grad_store='direct',glu_grad_load='interleaved')
                if name=='gate_up' and config.prepared.forward_glu_engine!='legacy':
                    fields_c.update(glu_quant_vec=1,glu_scratch_swizzle='none')
                if name=='recompute':
                    fields_c.update(glu_hidden_decode='fp32',glu_epilogue='registers')
                    if config.prepared.backward_preparation=='separate':
                        fields_c.update(glu_grad_store='direct',glu_scratch_swizzle='none',glu_quant_vec=1)
                if name in ('gate_dw','down_dw') and getattr(config.prepared,name+'_engine')=='persistent':
                    # These role/cap/cache fields are not used by the retained-
                    # panel kernel. Do not time repeated aliases of one program.
                    fields_c.update(maxrregcount=255,producer_registers=48,consumer_registers=192,
                        scale_recycle='barrier',raster='n',grid_swizzle=2,tile_locality='raster',scale_l2_prefetch='none',
                        scale_l1_evict='default',scale_cache='default')
                if name=='gate_dw' and config.prepared.gate_dw_engine=='indexed':
                    from ..configs.moe_backward import indexed_gate_dw_schedule
                    effective=indexed_gate_dw_schedule(c,config.prepared.indexed_dw_producer_warps)
                    fields_c.update({key:getattr(effective,key) for key in (
                        'stages','scale_role','scale_load_vec','scale_schedule','scale_recycle',
                        'scale_smem_store','scale_l2_prefetch','scale_l1_evict','scale_cache')})
                    fields_c.update(mma_schedule='interleaved',maxrregcount=255,
                                    producer_registers=48,consumer_registers=192)
                changes[name]=changed(c,**fields_c)
            p=config.prepared
            pools=(tags or {}).get('batch_contract',{}).get('sequences',0)
            p=changed(p,compact_pool_count=pools if p.forward_row_layout=='compact' else 0)
            if phase=='forward':
                p=changed(p,bounded_dy=False,bounded_xt=False)
                if p.forward_row_layout=='compact':p=changed(p,bounded_gather=False)
            if p.gate_dw_engine=='indexed':
                transports=list(p.transports);transports[STAGES.index('gate_dw')]='tma'
                p=changed(p,transports=tuple(transports))
            if phase=='forward':p=changed(p,dy_preparation='materialized',joint_recompute_engine='legacy')
            if phase=='forward':
                p=changed(p,gate_dw_engine='packed',down_dw_engine='packed',
                    route_gradient_loader='accumulator',route_gradient_gemm=None,weight_preparation='separate')
            if p.joint_recompute_engine!='split_warp':p=changed(p,joint_recompute_gradient_warps=3)
            if p.forward_engine=='onchip':p=changed(p,forward_glu_engine='legacy')
            if p.weight_preparation=='separate':
                p=changed(p,weight_tile_rows=32,weight_tile_cols=64,weight_load_values=4,
                    weight_store_bits=32,weight_warps=4,weight_raster='features')
            if p.gate_dw_engine!='indexed':
                p=changed(p,indexed_dw_token_lanes=8,indexed_dw_gather_pairs=1,indexed_dw_producer_warps=0)
            for stage in ('gate_dw','down_dw'):
                if getattr(p,stage+'_engine')!='persistent':
                    p=changed(p,**{stage+'_reuse_axis':'n',stage+'_reuse_tiles':2})
            if p.route_gradient_epilogue!='fused':
                p=changed(p,route_gradient_loader='accumulator',route_gradient_gemm=None)
            elif p.route_gradient_gemm==p.down:p=changed(p,route_gradient_gemm=None)
            physical=p.physical_capacity(shape[3])
            # Alignment labels are inactive when they produce identical
            # storage. Do not spend five trials timing the same 1024 rows.
            for multiple in (1,32,64,128,256):
                if multiple==p.capacity_multiple or replace(p,capacity_multiple=multiple).physical_capacity(shape[3])==physical:
                    p=changed(p,capacity_multiple=multiple)
                    break
            if max(p.down_dw_splits,p.gate_dw_splits)==1:
                p=changed(p,dw_reduce_threads=128,dw_reduce_vector=4)
            if p.transpose_mode!='warp':
                p=changed(p,transpose_warp_tokens=8,transpose_warp_raster='features')
            else:
                qfields=dict(transposed_load_engine='register')
                if p.dy_preparation=='materialized':
                    qfields.update(transposed_tile_rows=128,transposed_tile_k=32,transposed_smem_padding=1)
                p=changed(p,quant=changed(p.quant,**qfields))
            fields['prepared']=changed(p,**changes)
            if p.forward_engine=='onchip' and p.backward_preparation=='joint':
                fields['prepared']=changed(fields['prepared'],gate_up=DEFAULT_PREPARED_MOE_CONFIG.gate_up)
            return changed(config,**fields)
        updates={}
        for name in stages:
            c=getattr(config,name)
            changes={}
            if not c.persistent:changes['persistent_waves']=1
            if c.load_engine=='scalar':changes['bf16_swizzle']='128b'
            if name=='gate_up':
                changes.update(glu_grad_load='interleaved',glu_grad_store='direct',glu_hidden_decode='fp32')
            if name=='recompute':
                changes.update(glu_epilogue='registers',glu_scratch='dedicated',glu_quant_epilogue='exchange',
                    glu_scratch_swizzle='none',glu_quant_vec=4,glu_grad_store='direct',glu_hidden_decode='fp32')
                if c.glu_grad_load=='shared':changes['glu_grad_load']='interleaved'
            updates[name]=changed(c,**changes)
        config=changed(config,prepared=DEFAULT_PREPARED_MOE_CONFIG,**updates)
        if config.family=='grouped_staged':
            return replace(config,onchip_down_stages=1,onchip_pipeline='serial')
        if shape[5]==128:
            gate=normalize_glu_config(config.gate_up,tile_n=256,tile_k=128,mxfp8_stages=1,
                glu_scratch='dedicated',glu_quant_epilogue='exchange',persistent=False,reuse='none')
            # The on-chip down engine has its own B-buffer pipeline, no BF16
            # transport ring or separate compile/register budget. Canonicalize
            # those dead shared-core coordinates instead of timing aliases.
            active=('quant_vec','quant_load_bits','quant_store_bits','quant_amax','quant_math',
                    'reduction','scale_rounding','a_swizzle','b_swizzle','a_ldmatrix_matrices',
                    'b_ldmatrix_matrices','sfa_s2r_bits','sfb_s2r_bits','atom_layout_m','atom_layout_n')
            fields={name:getattr(config.down,name) for name in active}
            fields['quant_vec']=max(2,fields['quant_vec'])
            down=normalize_fwd_config(DEFAULT_MOE_CONFIG.down,**fields,tile_m=gate.tile_m,
                tile_k=128,mxfp8_stages=1,tile_n=min(config.down.tile_n,128))
            config=replace(config,gate_up=gate,down=down)
        return config
    initial=normalize(initial)
    common=dict(tile_m=(32,64,128,256),tile_n=(64,128,256),tile_k=(128,256),
                bf16_tile_k=(32,64,128),bf16_stages=(1,2,3,4),mxfp8_stages=(1,2,3,4),
                quant_vec=(1,2,4,8,16),quant_load_bits=(16,32,64,128),quant_store_bits=(8,16,32),
                quant_amax=('fp32','bf16_bits'),quant_math=('fp32','bf16x2'),
                quant_schedule=('combined','split_a_b'),reduction=('shuffle','redux'),
                a_ldmatrix_matrices=(1,2,4),b_ldmatrix_matrices=(1,2,4),
                a_swizzle=('none','64b','128b'),b_swizzle=('none','64b','128b'),
                sfa_s2r_bits=(0,8),sfb_s2r_bits=(0,8),raster=('m','n'),grid_swizzle=(1,2,4,8),
                persistent=(False,True),persistent_waves=(1,2,3,4),
                maxrregcount=(96,128,192,255),producer_registers=(24,32,40,64),
                quantizer_registers=(64,96,128),consumer_registers=(96,128,160,192,232),
                scalar_bounds=('always','auto'))
    families=('grouped_staged','grouped_prepared','grouped_onchip') if allow_onchip else ('grouped_staged','grouped_prepared')
    axes={'gather_mode':('flat','warp'),'movement_threads':(64,128,256,512),'family':families,
          'onchip_down_stages':(1,2,3,4),'onchip_pipeline':('serial','warp_specialized')}
    for kind in (('combine','dx') if phase=='train' else ('combine',)):
        axes[kind+'_reduce_engine']=('scalar','token')
        axes[kind+'_vector_width']=(2,4,8)
        axes[kind+'_lanes_per_token']=(8,16,32)
    axes['prepared.transpose_mode']=('direct','staged','warp')
    axes['prepared.transpose_warp_tokens']=(2,4,8)
    axes['prepared.transpose_warp_raster']=('features','tokens')
    axes['prepared.backward_preparation']=('separate','joint')
    axes['prepared.glu_output']=tuple((v,math) for v in (1,2,4,8) for math in ('division','shared_reciprocal'))
    axes['prepared.scale_transport']=('row_major','mma128')
    axes['prepared.linear_transport']=('tma','cooperative')
    axes['prepared.capacity_multiple']=(1,32,64,128,256)
    axes['prepared.quant_grid']=('operand','expert','portfolio')
    axes['prepared.input_preparation']=('gather_bf16','quantize_once','indexed_transpose')
    axes['prepared.forward_engine']=('packed','onchip') if allow_onchip else ('packed',)
    axes['prepared.forward_glu_engine']=('legacy','fragment_direct','fragment_packed')
    axes['prepared.bounded_gather']=(False,True)
    if phase=='train':
        axes['prepared.bounded_dy']=(False,True)
        axes['prepared.bounded_xt']=(False,True)
    if (tags or {}).get('batch_contract',{}).get('sequences',0)>0:
        axes['prepared.forward_row_layout']=('original','compact')
    if (tags or {}).get('batch_contract',{}).get('layout')=='independent_varlen':
        axes['prepared.active_prefix']=(False,True)
    if phase=='train':
        axes['prepared.dy_preparation']=('materialized','fused','fused_all')
        axes['prepared.joint_recompute_engine']=JOINT_RECOMPUTE_ENGINES
        axes['prepared.joint_recompute_gradient_warps']=(1,2,3,4)
        axes['prepared.weight_preparation']=('separate','dual_orientation')
        axes['prepared.weight_tile']=tuple((m,n) for m in (32,64,128) for n in (32,64,128) if m*n<=8192)
        axes['prepared.weight_load_values']=(1,2,4,8)
        axes['prepared.weight_store_bits']=(8,16,32)
        axes['prepared.weight_warps']=(2,4,8)
        axes['prepared.weight_raster']=('features','rows')
        if c<=128:
            for stage in ('down_dw','gate_dw'):
                axes[f'prepared.{stage}.short_k']=(0,32,64) if c>32 else (0,32)
        for name in ('down_dw_splits','gate_dw_splits'):axes['prepared.'+name]=(1,2,4,8,16)
        axes['prepared.dw_reduce_threads']=(64,128,256,512)
        axes['prepared.dw_reduce_vector']=(1,2,4,8)
        axes['prepared.route_gradient_epilogue']=('separate','fused')
        axes['prepared.route_gradient_loader']=('accumulator','shared_vec4','warp_vec2')
        for stage in ('gate_dw','down_dw'):
            axes[f'prepared.{stage}_engine']=('packed','persistent','indexed') if stage=='gate_dw' else ('packed','persistent')
            axes[f'prepared.{stage}_reuse_axis']=('m','n')
            axes[f'prepared.{stage}_reuse_tiles']=(2,3)
        axes['prepared.indexed_dw_token_lanes']=(2,4,8)
        axes['prepared.indexed_dw_gather_pairs']=(1,2)
        axes['prepared.indexed_dw_producer_warps']=(0,2,4,8)
        axes['prepared.route_gradient_gemm.geometry']=tuple((m,n,128,am,2)
            for m,am in ((64,2),(128,4)) for n in (64,128))
        axes['prepared.route_gradient_gemm.stages']=(1,2,3,4)
    axes['prepared.bf16_outputs']=('direct','tma_dw','tma_all')
    envelopes=((96,32,64),(128,32,96),(160,40,128),(192,48,160),(255,48,192),(255,64,232))
    axes['prepared.resource_envelope']=envelopes
    qaxes=dict(quant_vec=(1,2,4,8),load_bits=(16,32,64,128),quant_store_bits=(8,16,32),
               quant_math=('fp32','bf16x2'),quant_amax=('fp32','bf16_bits'),reduction=('shuffle','redux'),
               num_warps=(4,8,16),persistent_waves=(1,2,3,4,6,8),maxrregcount=(64,96,128,192,255),
               transposed_load_engine=('register','cp_async'),transposed_tile_rows=(32,64,128,256),
               transposed_tile_k=(32,64,128),transposed_smem_padding=(0,1,2,4,8))
    axes.update({f'prepared.quant.{key}':values for key,values in qaxes.items()})
    packed_axes=dict(stages=(1,2,3,4),k_tile_unroll=(1,2,3,4,6,8),
        a_swizzle=('none','32b','64b','128b'),b_swizzle=('none','32b','64b','128b'),
        a_ldmatrix_matrices=(1,2,4),b_ldmatrix_matrices=(1,2,4),mma_schedule=('interleaved','preload'),
        sfa_s2r_bits=(0,8),sfb_s2r_bits=(0,8),scale_schedule=('before_wait','after_wait'),
        scale_recycle=('barrier','staged'),scale_load_vec=(1,2,4,8),scale_role=('consumers','producer'),
        scale_smem_store=('scalar','packed'),scale_l2_prefetch=('none','64b','128b','256b'),
        scale_l1_evict=('default','normal','first','last','noallocate'),scale_cache=('default','ca','cg','cs'),
        producer_registers=(24,32,48,64),consumer_registers=(64,96,128,160,192,232),
        maxrregcount=(96,128,192,255),raster=('m','n'),grid_swizzle=(1,2,4,8),
        tiles_per_cta=(1,2,4,8),tile_locality=('raster','same_a','same_b','serpentine_a','serpentine_b'),
        persistent_waves=(0,1,2,3,4))
    geometries=tuple((m,n,k,am,an) for m,am in ((64,2),(128,2),(128,4),(128,8),(256,8))
                     for n,an in ((64,2),(128,2),(256,4)) for k in (128,256))
    for stage in stages:
        axes.update({f'prepared.{stage}.{key}':values for key,values in packed_axes.items()})
        axes[f'prepared.{stage}.geometry']=geometries
        if stage not in ('gate_up','recompute'):axes[f'prepared.{stage}.transport']=('tma','cooperative')
        if stage not in ('gate_up','recompute'):
            axes[f'prepared.{stage}.epilogue_transport']=(('direct',1,1),('tma',1,1),('tma',2,1),
                ('tma',4,1),('tma',4,2),('tma',4,3),('tma',4,4),
                ('hybrid',1,1),('hybrid',2,1),('hybrid',4,1),('hybrid',4,2),
                ('hybrid',4,3),('hybrid',4,4))
        if stage=='gate_up':
            axes[f'prepared.{stage}.glu_epilogue']=('exchange','registers')
            axes[f'prepared.{stage}.glu_quant_vec']=(1,2,4,8)
            axes[f'prepared.{stage}.glu_scratch_swizzle']=('none','xor_16b')
        if stage=='recompute':
            axes[f'prepared.{stage}.glu_grad_load']=('interleaved','prefetch')
            axes[f'prepared.{stage}.glu_quant_vec']=(1,2,4,8)
            axes[f'prepared.{stage}.glu_scratch_swizzle']=('none','xor_16b')
    for name in stages:
        axes.update({f'{name}.{field}':values for field,values in common.items()})
        if name!='down':
            axes[f'{name}.transport']=tuple((engine,stages) for engine in ('scalar','cpasync','tma') for stages in (1,2,3,4))
        if name=='gate_up':
            axes[f'{name}.glu_epilogue']=('exchange','registers')
            axes[f'{name}.glu_scratch']=('dedicated','reuse_operands')
            axes[f'{name}.glu_scratch_swizzle']=('none','xor_16b')
            axes[f'{name}.glu_quant_vec']=(1,2,4,8,16)
        if name=='recompute':axes[f'{name}.glu_grad_load']=('interleaved','prefetch')
    def update(config,name,value):
        if name.startswith('prepared.'):
            parts=name.split('.')
            if parts[1]=='forward_row_layout':
                candidate=config
                if value=='compact':
                    if candidate.prepared.forward_glu_engine=='legacy':
                        candidate=update(candidate,'prepared.forward_glu_engine','fragment_packed')
                    if candidate.prepared.input_preparation=='gather_bf16':
                        candidate=update(candidate,'prepared.input_preparation','quantize_once')
                return normalize(replace(candidate,prepared=replace(candidate.prepared,forward_row_layout=value)))
            if parts[1] in ('bounded_gather','bounded_dy','bounded_xt') and value:
                candidate=update(config,'prepared.active_prefix',True)
                if parts[1]=='bounded_dy':candidate=update(candidate,'prepared.dy_preparation','fused')
                elif parts[1]=='bounded_xt':candidate=update(candidate,'prepared.input_preparation','indexed_transpose')
                elif candidate.prepared.input_preparation=='gather_bf16':
                    candidate=update(candidate,'prepared.input_preparation','quantize_once')
                return normalize(replace(candidate,prepared=replace(candidate.prepared,**{parts[1]:True})))
            if parts[1]=='forward_glu_engine':
                p=config.prepared
                if value!='legacy':
                    g=replace(p.gate_up,tile_m=64,tile_n=128,atom_layout_m=2,atom_layout_n=2,
                        glu_silu_math='division',glu_epilogue='registers',glu_store='direct',
                        epilogue='direct',store_vec=1,epilogue_stages=1,
                        tiles_per_cta=1,persistent_waves=0,tile_locality='raster')
                    p=replace(p,forward_engine='packed',gate_up=g)
                return normalize(replace(config,prepared=replace(p,forward_glu_engine=value)))
            if parts[1]=='weight_preparation':
                p=config.prepared
                if value!='separate':
                    if p.input_preparation=='gather_bf16':
                        p=update(config,'prepared.input_preparation','quantize_once').prepared
                    p=replace(p,backward_preparation='joint',
                        recompute=replace(p.recompute,glu_grad_store='exchange'))
                return normalize(replace(config,prepared=replace(p,weight_preparation=value)))
            if parts[1]=='weight_tile':
                rows,cols=value
                return normalize(replace(config,prepared=replace(config.prepared,weight_tile_rows=rows,weight_tile_cols=cols)))
            if parts[1] in ('gate_dw_engine','down_dw_engine'):
                stage=parts[1][:-7];p=config.prepared
                if value=='indexed':
                    p=update(config,'prepared.input_preparation','indexed_transpose').prepared
                    p=replace(p,dy_preparation='fused' if p.dy_preparation=='fused_all' else p.dy_preparation,
                              active_prefix=False,gate_dw_splits=1)
                    g=replace(p.gate_dw,epilogue='direct',store_vec=1,epilogue_stages=1,
                        tiles_per_cta=1,persistent_waves=0,tile_locality='raster')
                    p=replace(p,gate_dw=g)
                elif value=='persistent':
                    p=p.with_transport(stage,'tma');g=getattr(p,stage)
                    g=replace(g,tile_m=64,tile_n=64,tile_k=128,atom_layout_m=2,atom_layout_n=2,
                        scale_layout='row_major',scale_role='consumers',scale_smem_store='scalar',
                        scale_load_vec=1,mma_schedule='interleaved',epilogue='direct',store_vec=1,
                        epilogue_stages=1,tiles_per_cta=1,persistent_waves=0,tile_locality='raster')
                    p=replace(p,**{stage:g,stage+'_splits':1})
                return normalize(replace(config,prepared=replace(p,**{parts[1]:value})))
            if parts[1]=='route_gradient_loader':
                p=update(config,'prepared.route_gradient_epilogue','fused').prepared
                return normalize(replace(config,prepared=replace(p,route_gradient_loader=value)))
            if parts[1]=='route_gradient_gemm':
                p=update(config,'prepared.route_gradient_epilogue','fused').prepared
                g=p.route_gradient_gemm or p.down;field=parts[2]
                fields={field:value}
                if field=='geometry':
                    m,n,k,am,an=value
                    fields=dict(tile_m=m,tile_n=n,tile_k=k,atom_layout_m=am,atom_layout_n=an)
                return normalize(replace(config,prepared=replace(p,route_gradient_gemm=replace(g,**fields))))
            if parts[1]=='scale_transport':
                p=config.prepared.with_scale_layout(value)
                if value=='mma128':
                    p=replace(p,backward_preparation='separate',dy_preparation='materialized',
                        joint_recompute_engine='legacy',active_prefix=False,
                        route_gradient_epilogue='separate',
                        input_preparation='quantize_once' if p.input_preparation=='indexed_transpose' else p.input_preparation)
                return normalize(replace(config,prepared=p))
            if parts[1]=='linear_transport':
                result=config
                for stage in stages:
                    if stage not in ('gate_up','recompute'):
                        result=update(result,f'prepared.{stage}.transport',value)
                return result
            if parts[1]=='bf16_outputs':
                result=config
                affected=('down_dw','gate_dw') if value=='tma_dw' else tuple(s for s in stages if s not in ('gate_up','recompute'))
                if value=='tma_all':
                    multiple=max(getattr(config.prepared,s).tile_m for s in affected)
                    result=replace(result,prepared=replace(result.prepared,capacity_multiple=multiple))
                for stage in affected:
                    if value!='direct':result=update(result,f'prepared.{stage}.transport','tma')
                    result=update(result,f'prepared.{stage}.epilogue_transport',
                                  ('direct',1,1) if value=='direct' else ('tma',4,1))
                return result
            if parts[1]=='glu_output':
                vec,math=value
                p=replace(config.prepared,**{stage:replace(getattr(config.prepared,stage),
                    glu_quant_vec=vec,glu_silu_math=math) for stage in ('gate_up','recompute')})
                return normalize(replace(config,prepared=p))
            if parts[1]=='backward_preparation':
                p=replace(config.prepared,backward_preparation=value)
                if value=='joint':p=replace(p,recompute=replace(p.recompute,glu_grad_store='exchange'))
                else:
                    p=replace(p,dy_preparation='materialized',route_gradient_epilogue='separate',joint_recompute_engine='legacy',
                        active_prefix=p.active_prefix and phase=='forward',
                        input_preparation='quantize_once' if p.input_preparation=='indexed_transpose' else p.input_preparation)
                return normalize(replace(config,prepared=p))
            if parts[1]=='joint_recompute_engine':
                p=config.prepared
                if value!='legacy':
                    g=replace(p.recompute,glu_grad_store='exchange',glu_grad_load='interleaved',
                              scale_role='producer',scale_smem_store='scalar',
                              tiles_per_cta=1,persistent_waves=0,tile_locality='raster')
                    p=replace(p,backward_preparation='joint',recompute=g)
                return normalize(replace(config,prepared=replace(p,joint_recompute_engine=value)))
            if parts[1]=='active_prefix' and value:
                p=config.prepared
                if phase=='train':
                    p=replace(p,backward_preparation='joint',
                              recompute=replace(p.recompute,glu_grad_store='exchange'))
                return normalize(replace(config,prepared=replace(p,active_prefix=True)))
            if parts[1]=='input_preparation' and value=='indexed_transpose':
                p=config.prepared
                p=replace(p,input_preparation=value,transpose_mode='warp',backward_preparation='joint',
                    recompute=replace(p.recompute,glu_grad_store='exchange'))
                return normalize(replace(config,prepared=p))
            if parts[1]=='input_preparation':
                return normalize(replace(config,prepared=replace(config.prepared,input_preparation=value,
                    dy_preparation='materialized')))
            if parts[1]=='dy_preparation' and value!='materialized':
                candidate=update(config,'prepared.input_preparation','indexed_transpose')
                return normalize(replace(candidate,prepared=replace(candidate.prepared,dy_preparation=value)))
            if parts[1]=='route_gradient_epilogue' and value=='fused':
                p=config.prepared.with_transport('down','tma')
                p=replace(p,route_gradient_epilogue=value,backward_preparation='joint',
                    recompute=replace(p.recompute,glu_grad_store='exchange'))
                return normalize(replace(config,prepared=p))
            if parts[1] in ('down_dw_splits','gate_dw_splits') and value>1:
                stage=parts[1][:-7]
                p=config.prepared.with_transport(stage,'tma')
                p=replace(p,**{parts[1]:value,stage:replace(getattr(p,stage),epilogue='direct',
                    store_vec=1,epilogue_stages=1,tiles_per_cta=1,persistent_waves=0)})
                return normalize(replace(config,prepared=p))
            if parts[1]=='resource_envelope':
                cap,producer,consumer=value
                p=replace(config.prepared,**{stage:replace(getattr(config.prepared,stage),
                    maxrregcount=cap,producer_registers=producer,consumer_registers=consumer) for stage in stages})
                return normalize(replace(config,prepared=p))
            if len(parts)==2:return normalize(replace(config,prepared=replace(config.prepared,**{parts[1]:value})))
            stage,field=parts[1:]
            if field=='transport':
                return normalize(replace(config,prepared=config.prepared.with_transport(stage,value)))
            if field=='short_k':
                return normalize(replace(config,prepared=config.prepared.with_short_dw(stage,value)))
            changes={field:value}
            if field=='epilogue_transport':
                kind,vec,buffers=value
                changes=dict(epilogue=kind,store_vec=vec,epilogue_stages=buffers)
            if field=='geometry':
                m,n,k,am,an=value
                changes=dict(tile_m=m,tile_n=n,tile_k=k,atom_layout_m=am,atom_layout_n=an)
            prepared=replace(config.prepared,**{stage:replace(getattr(config.prepared,stage),**changes)})
            return normalize(replace(config,prepared=prepared))
        if '.' not in name:return normalize(replace(config,**{name:value}))
        stage,field=name.split('.')
        changes={field:value}
        if field=='transport':
            engine,depth=value
            changes=dict(load_engine=engine,bf16_stages=depth,
                         bf16_tile_k=128 if engine=='scalar' else 32,
                         schedule='cooperative' if engine!='tma' else 'warp_specialized')
        update_stage=normalize_glu_config if stage in ('gate_up','recompute') else normalize_fwd_config
        return normalize(replace(config,**{stage:update_stage(getattr(config,stage),**changes)}))
    context=KernelContext('mxfp8_balanced_experts',MOE_KERNEL_REVISION,
        dict(tokens=n,d_model=d,experts=e,capacity=c,top_k=k,hidden=h,phase=phase),
        device={} if device is None else device,tags={} if tags is None else tags)
    serialize=lambda c:normalize(c).to_dict()
    @lru_cache(maxsize=4096)
    def identifier(config):
        return stable_id(serialize(config))
    return DiscreteKernelAdapter(context,initial,axes,identifier,
        serialize,lambda values:normalize(MXFP8MoEConfig.from_dict(values)),update,evaluator,
        lambda c:None if (why:=rejection(c,shape,phase)) is None else ('implementation_rejected',why),
        lambda c:dict(assignments=e*shape[3],input_bytes=n*d*2,expert_weights_bytes=e*3*d*h*2,
                      bounded_gather=c.prepared.bounded_gather,
                      bounded_dy=phase=='train' and c.prepared.bounded_dy,
                      bounded_xt=phase=='train' and c.prepared.bounded_xt,
                      compact_forward=c.prepared.forward_row_layout=='compact',
                      combine_token_reduction=c.combine_reduce_engine=='token',
                      dx_token_reduction=phase=='train' and c.dx_reduce_engine=='token',
                      combine_metadata_reuse=(c.combine_vector_width if c.combine_reduce_engine=='token' else 1),
                      dx_metadata_reuse=(c.dx_vector_width if phase=='train' and c.dx_reduce_engine=='token' else 1),
                      hidden_payload_bytes=e*shape[3]*((h+31)//32*32),saved_hidden_bytes=0,
                      assignment_reuse=k,per_expert_reduction=c.prepared.physical_capacity(shape[3]),
                      weight_bf16_read_bytes_model=(0 if phase=='forward' else
                          (6 if c.prepared.weight_preparation=='dual_orientation' else 12)*e*d*h),
                      joint_extra_smem_bytes=(c.prepared.recompute.tile_n*(c.prepared.recompute.tile_m+1)
                          if phase=='train' and c.prepared.joint_recompute_engine=='split_warp' else 0),
                      gate_dw_indexed_quantization=c.prepared.gate_dw_engine=='indexed',
                      indexed_x_transpose_payload_bytes=(0 if phase=='forward' or c.prepared.gate_dw_engine=='indexed'
                          else e*d*c.prepared.physical_capacity(shape[3])*33//32),
                      route_gradient_shared_bytes=(0 if phase=='forward' or c.prepared.route_gradient_epilogue!='fused' else
                          route_gradient_scratch_bytes(c.prepared.route_gradient_gemm or c.prepared.down,
                                                       c.prepared.route_gradient_loader)),
                      transpose_warp_features=64//c.prepared.transpose_warp_tokens,
                      down_dw_partials=c.prepared.down_dw_splits,gate_dw_partials=c.prepared.gate_dw_splits,
                      gathered_bf16_bytes=0 if c.prepared.input_preparation=='indexed_transpose' else e*c.prepared.physical_capacity(shape[3])*d*2,
                      slot_dy_bf16_bytes=0 if phase=='forward' or c.prepared.dy_preparation!='materialized' else e*c.prepared.physical_capacity(shape[3])*d*2,
                      dy_global_bytes_saved=6*e*c.prepared.physical_capacity(shape[3])*d if phase=='train' and c.prepared.dy_preparation!='materialized' else 0,
                      dw_workspace_bytes=e*4*d*h*((c.prepared.down_dw_splits if c.prepared.down_dw_splits>1 else 0)
                        +2*(c.prepared.gate_dw_splits if c.prepared.gate_dw_splits>1 else 0)),
                      **prepared_dw_geometry_features(c,shape,phase,device),
                      **prepared_row_store_features(c,shape,phase)))


class MoEEvaluator:
    def __init__(self,operands,phase,policy):
        import torch
        from .._moe_reference import operational_reference
        self.operands=tuple(t.detach() for t in operands)
        self.phase,self.policy=phase,policy
        self.gradient=torch.randn(operands[0].shape,device=operands[0].device,dtype=torch.float32,
                                 generator=torch.Generator(device=operands[0].device).manual_seed(policy.seed))
        self.expected=operational_reference(*self.operands,self.gradient if phase=='train' else None)

    def call(self,config):
        from ..moe import _run_forward,_run_backward
        y=_run_forward(*self.operands,config)
        return (y,*_run_backward(*self.operands,self.gradient,config)) if self.phase=='train' else (y,)

    def evaluate(self,config):
        import torch
        from .measurement import measure_cuda_batches,runtime_timing_metadata
        started=time.monotonic();stage='compile';graph=None
        try:
            actual=self.call(config)
            torch.cuda.synchronize(self.operands[0].device)
            stage='correctness';errors={}
            for name,value,expected in zip(('output','dx','gate_dw','down_dw','route_dw'),actual,self.expected):
                if not bool(value.isfinite().all()):raise AssertionError(f'{name}: nonfinite values')
                delta=value.float()-expected.float()
                nrms=(delta.norm()/expected.float().norm().clamp_min(1e-12)).item()
                maximum=delta.abs().max().item()
                errors[name]=dict(nrms=nrms,max_abs=maximum)
                limit=.003 if name=='output' else .006
                if nrms>limit and maximum>2e-7:raise AssertionError(f'{name}: NRMS {nrms} > {limit}')
            del actual
            stage='capture'
            stream=torch.cuda.Stream(device=self.operands[0].device)
            stream.wait_stream(torch.cuda.current_stream(self.operands[0].device))
            with torch.cuda.stream(stream):
                for _ in range(self.policy.warmup):self.call(config)
            stream.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):result=self.call(config)
            torch.cuda.current_stream(self.operands[0].device).wait_stream(stream)
            from .preparation_runtime import _warm_measurement_graph
            _warm_measurement_graph(graph)
            compile_ms=(time.monotonic()-started)*1000
            stage='timing'
            samples,metadata=measure_cuda_batches(graph.replay,samples=self.policy.samples,
                calls_per_sample=self.policy.calls_per_sample,device=self.operands[0].device,min_batch_ms=3.)
            # Some pipeline hazards appear only after repeated fast replay,
            # not on the initial synchronized correctness call.
            stage='correctness';replay_errors={}
            for name,value,expected in zip(('output','dx','gate_dw','down_dw','route_dw'),result,self.expected):
                if not bool(value.isfinite().all()):raise AssertionError(f'{name}: nonfinite replay output')
                delta=value.float()-expected.float()
                nrms=(delta.norm()/expected.float().norm().clamp_min(1e-12)).item()
                maximum=delta.abs().max().item()
                replay_errors[name]=dict(nrms=nrms,max_abs=maximum)
                if nrms>(.003 if name=='output' else .006) and maximum>2e-7:
                    raise AssertionError(f'{name}: replay NRMS {nrms}')
            del result
            return TrialOutcome('ok',median_ms=statistics.median(samples),timings_ms=samples,
                compile_ms=compile_ms,metadata=dict(validation=errors,validation_after_replay=replay_errors,
                                                   **metadata,**runtime_timing_metadata(samples)))
        except Exception as error:
            raise_if_fatal_device_context_error(error)
            return TrialOutcome('correctness_error' if stage=='correctness' else
                'compile_error' if stage=='compile' else 'runtime_error',
                error=f'{type(error).__name__}: {error}'[:4000],compile_ms=(time.monotonic()-started)*1000)
        finally:
            if graph is not None:graph.reset()


def row_store_seeds(adapter,initial):
    """Compare compound row-output stores without changing saved-state policy."""
    from .output_transport import row_store_candidates
    if initial.family!='grouped_prepared':return
    shape=tuple(adapter.context.workload[k] for k in
                ('tokens','d_model','experts','capacity','top_k','hidden'))
    p=initial.prepared
    physical=(*shape[:3],p.physical_capacity(shape[3]),*shape[4:])
    training=adapter.context.workload['phase']=='train'
    combined=initial;individual=[]
    for stage in (('down','gate_dx','down_dx') if training else ('down',)):
        # On-chip forward does not execute the prepared down GEMM. Fused
        # routing adjoints override its store to direct as well. Do not spend
        # a scarce trial measuring an inactive epilogue alias.
        if stage=='down' and p.forward_engine=='onchip' and (
                not training or p.route_gradient_epilogue=='fused'):continue
        if p.transports[STAGES.index(stage)]!='tma':continue
        problem=problems(physical)[stage][0]
        for g in row_store_candidates(getattr(p,stage),problem):
            candidate=replace(initial,prepared=replace(p,**{stage:g}))
            if candidate!=initial and adapter.rejection(candidate) is None:
                individual.append(candidate)
                joint=replace(combined,prepared=replace(combined.prepared,**{stage:g}))
                if adapter.rejection(joint) is None:combined=joint
                break
    if combined!=initial:yield combined
    for candidate in individual:
        if candidate!=combined:yield candidate


def geometry_growth_seeds(adapter,initial):
    """One measured growth basin per row GEMM, not a winning dispatch rule.

    Derive legal coupled geometry from existing axes, preserving registers,
    transport, reduction, numerical and traffic policies. Grow only dimensions
    with enough real work; prefer preserving the incumbent's row fragment.
    """
    shape=tuple(adapter.context.workload[k] for k in
                ('tokens','d_model','experts','capacity','top_k','hidden'))
    if initial.family!='grouped_prepared':return
    for stage in ('down','gate_dx'):
        key=f'prepared.{stage}.geometry'
        if key not in adapter.axes:continue
        g=getattr(initial.prepared,stage);problem=problems(shape)[stage][0]
        wanted_m=g.tile_m*2 if problem.m>=g.tile_m*2 else g.tile_m
        wanted_n=g.tile_n*2 if problem.n>=g.tile_n*2 else g.tile_n
        order=sorted(adapter.axes[key],key=lambda v:(v[2]!=g.tile_k,
            abs(v[0]-wanted_m)+abs(v[1]-wanted_n),
            abs(v[0]*g.atom_layout_m-g.tile_m*v[3])))
        for geometry in order:
            value=adapter.update_fn(initial,key,geometry)
            if adapter.config_id(value)!=adapter.config_id(initial) and adapter.rejection(value) is None:
                # A single K tile has no next-K DMA to overlap. Offer the
                # shallow/preload basin without pruning deeper pipelines.
                # All register budgets remain inherited and every candidate
                # still needs the normal oracle and measured comparison.
                if problem.storage_k<=geometry[2]:
                    shallow=adapter.update_fn(value,f'prepared.{stage}.stages',1)
                    shallow=adapter.update_fn(shallow,f'prepared.{stage}.mma_schedule','preload')
                    if adapter.rejection(shallow) is None:value=shallow
                yield value
                break


def backward_ownership_seeds(adapter,initial):
    """Complete measured families, ranked by padding/operand panels, not SKU.

    Indexed quantization remains a deeper-search axis: repeated amax work
    lost the producer+consumer comparison and gets no protected opening slot.
    """
    if adapter.context.workload['phase']!='train' or initial.family!='grouped_prepared':return
    w=adapter.context.workload;p=initial.prepared
    capacity=p.physical_capacity(w['capacity'])
    route=adapter.update_fn(initial,'prepared.route_gradient_loader','warp_vec2')
    g=route.prepared.route_gradient_gemm or route.prepared.down
    # Preserve the forward-down schedule; only the adjoint gets its own tile.
    g=replace(g,tile_n=64,atom_layout_n=2)
    route=replace(route,prepared=replace(route.prepared,route_gradient_gemm=g))
    persistent=initial
    alternatives=[]
    for stage,m,n in (('gate_dw',2*w['hidden'],w['d_model']),
                       ('down_dw',w['d_model'],w['hidden'])):
        choices=[]
        for group in (2,3):
            for axis in ('m','n'):
                c=adapter.update_fn(initial,f'prepared.{stage}_engine','persistent')
                c=adapter.update_fn(c,f'prepared.{stage}_reuse_axis',axis)
                c=adapter.update_fn(c,f'prepared.{stage}_reuse_tiles',group)
                if capacity<2*getattr(c.prepared,stage).tile_k or adapter.rejection(c) is not None:continue
                mt=(m+63)//64;nt=(n+63)//64
                ct=mt*((nt+group-1)//group) if axis=='n' else ((mt+group-1)//group)*nt
                # Every CTA loads group+1 rather than 2*group panels. Include
                # grouped-edge waste, instead of assuming any direction wins.
                choices.append((ct*(group+1),ct,group,axis,c))
        if not choices:continue
        choices.sort(key=lambda x:x[:4]);best=choices[0][-1]
        for *_,c in choices:alternatives.append(c)
        persistent=replace(persistent,prepared=replace(persistent.prepared,
            **{stage:getattr(best.prepared,stage),stage+'_engine':'persistent',
               stage+'_reuse_axis':getattr(best.prepared,stage+'_reuse_axis'),
               stage+'_reuse_tiles':getattr(best.prepared,stage+'_reuse_tiles'),
               stage+'_splits':1,'active_prefix':False}))
    combined=replace(persistent,prepared=replace(persistent.prepared,
        backward_preparation='joint',recompute=replace(persistent.prepared.recompute,glu_grad_store='exchange'),
        route_gradient_epilogue='fused',route_gradient_loader='warp_vec2',
        route_gradient_gemm=route.prepared.route_gradient_gemm))
    seen={adapter.config_id(initial)}
    for c in (combined,persistent,route,*alternatives):
        key=adapter.config_id(c)
        if key not in seen and adapter.rejection(c) is None:
            seen.add(key);yield c


def preparation_ownership_seeds(adapter,initial):
    """Complete preparation controls; no negative split-team fast opening."""
    if initial.family!='grouped_prepared' or initial.prepared.quant.scale_layout!='row_major':return
    training=adapter.context.workload['phase']=='train'
    p=initial.prepared
    control=initial
    if training:
        shape=tuple(adapter.context.workload[k] for k in ('tokens','d_model','experts','capacity','top_k','hidden'))
        joint=next(joint_recompute_seeds(shape,control),None)
        if joint is not None:control=joint
        control=replace(control,prepared=replace(control.prepared,recompute=replace(
            control.prepared.recompute,glu_quant_vec=4,glu_scratch_swizzle='xor_16b')))
    g=control.prepared.gate_up
    compact=replace(control,prepared=replace(control.prepared,forward_engine='packed',gate_up=replace(
        g,tile_m=64,tile_n=64,tile_k=128,atom_layout_m=2,atom_layout_n=2,stages=1,
        scale_role='producer',epilogue='direct',store_vec=1,epilogue_stages=1,
        tiles_per_cta=1,persistent_waves=0,tile_locality='raster')))
    fragment=replace(compact,prepared=replace(compact.prepared,forward_glu_engine='fragment_packed',active_prefix=False))
    weight=compact
    if training:
        weight=adapter.update_fn(compact,'prepared.weight_preparation','dual_orientation')
        weight=replace(weight,prepared=replace(weight.prepared,weight_tile_rows=32,weight_tile_cols=128,
            weight_load_values=8,weight_store_bits=min(32,8*weight.prepared.quant.quant_vec),weight_warps=4))
    combined=replace(weight,prepared=replace(weight.prepared,forward_glu_engine='fragment_packed',active_prefix=False))
    seen={adapter.config_id(initial)}
    for c in (combined,compact,fragment,weight,control):
        key=adapter.config_id(c)
        if key not in seen and adapter.rejection(c) is None:
            seen.add(key);yield c


def cold_packed_padding_opening(adapter,stored,legacy,strong):
    """Matched layout/producer controls on one legal cold prepared recipe.

    An inherited ownership recipe already provides a useful incumbent to
    rebase on; keep its established opening. Compact requires a proven pool
    count, and every member still goes through normal legality and timing.
    """
    initial=adapter.initial_config.prepared
    contract=adapter.context.tags.get('batch_contract',{})
    w=adapter.context.workload
    if (adapter.context.workload['phase']!='train' or contract.get('sequences',0)<2
            or 'prepared.active_prefix' not in adapter.axes
            or 'prepared.forward_row_layout' not in adapter.axes
            or initial.forward_glu_engine!='legacy' or initial.weight_preparation!='separate'
            or initial.gate_dw_engine!='packed' or initial.down_dw_engine!='packed'
            or initial.active_prefix or initial.forward_row_layout!='original'):
        return ()
    # The generic ownership slot requires a long physical reduction. A
    # padded pack can justify a comparison earlier, but still needs at
    # least two real panels' worth of average work per expert.
    if legacy is None:legacy=next(backward_ownership_seeds(adapter,stored),None)
    if legacy is None:return ()
    if w['tokens']*w['top_k']<2*w['experts']*max(legacy.prepared.gate_dw.tile_k,legacy.prepared.down_dw.tile_k):
        return ()
    if strong is None:strong=next(preparation_ownership_seeds(adapter,legacy),None)
    if strong is None:return ()
    for config in (legacy,strong):
        p=config.prepared
        if (p.active_prefix or p.forward_row_layout!='original'
                or p.backward_preparation!='joint' or p.input_preparation!='indexed_transpose'
                or p.dy_preparation!='fused' or p.route_gradient_epilogue!='fused'
                or p.gate_dw_engine!='persistent' or p.down_dw_engine!='persistent'):
            return ()
    if (legacy.prepared.forward_glu_engine!='legacy' or legacy.prepared.weight_preparation!='separate'
            or strong.prepared.forward_glu_engine!='fragment_packed'
            or strong.prepared.weight_preparation!='dual_orientation'):
        return ()
    prefix=adapter.update_fn(strong,'prepared.active_prefix',True)
    bounds=dict(bounded_dy=True,bounded_xt=True,bounded_gather=True)
    bounded=replace(prefix,prepared=replace(prefix.prepared,**bounds))
    compact=adapter.update_fn(prefix,'prepared.forward_row_layout','compact')
    both=replace(compact,prepared=replace(compact.prepared,**bounds))
    opening=(legacy,strong,prefix,bounded,compact,both)
    ids={adapter.config_id(c) for c in opening}
    # Don't spend protected slots on a partial comparison when one shape
    # cannot legally realize it (for example no compact storage win).
    if (len(ids)!=len(opening) or adapter.config_id(adapter.initial_config) in ids
            or any(adapter.rejection(c) is not None for c in opening)):
        return ()
    return opening


class PreparedComponentSearch:
    """Stage-aware local arm inside the shared UCB orchestrator.

    Whole-layer random mutation spreads a small budget over hundreds of
    coordinates, many inactive in a given family. This arm changes one active
    component of the best prepared path, but still evaluates the complete
    expert forward/backward with the same correctness/timing contract.
    No per-SKU winners or manually selected register counts are baked in.
    """
    name='prepared_components'

    def propose(self,adapter,history,rng,limit):
        ranked=sorted((r for r in history.successful if r.config.family=='grouped_prepared'
                       and r.outcome.timing_eligible),key=lambda r:r.score)
        if not ranked:return []
        base=ranked[0]
        training=adapter.context.workload['phase']=='train'
        # Rebase a complete liveness recipe onto the measured incumbent.
        # Mutating only the engine keeps the incumbent's old (often wide)
        # tile, while replaying a frozen structural seed discards its input,
        # dy and route-gradient traffic wins. Measure the combination instead.
        # Offer just the first legal compact recipe per incumbent so local
        # coordinates still receive budget; this never installs a winner.
        result=[]
        if training:
            shape=tuple(adapter.context.workload[k] for k in
                        ('tokens','d_model','experts','capacity','top_k','hidden'))
            for candidate in joint_recompute_seeds(shape,base.config):
                key=adapter.config_id(candidate)
                if key not in history.seen_ids and adapter.rejection(candidate) is None:
                    result.append(Proposal(candidate,self.name,parent_config_id=base.config_id,
                                           coordinate='prepared.recompute.recipe',
                                           coordinate_value=candidate.prepared.joint_recompute_engine))
                # Other tile/liveness recipes remain in structural/local
                # search; do not spend every component turn cycling them.
                break
            for candidate in short_dw_seeds(shape,base.config):
                key=adapter.config_id(candidate)
                if key not in history.seen_ids and adapter.rejection(candidate) is None:
                    result.append(Proposal(candidate,self.name,parent_config_id=base.config_id,
                                           coordinate='prepared.dw.short_recipe',
                                           coordinate_value=candidate.prepared.gate_dw.tile_k))
                break
            for candidate in backward_ownership_seeds(adapter,base.config):
                key=adapter.config_id(candidate)
                if key not in history.seen_ids:
                    result.append(Proposal(candidate,self.name,parent_config_id=base.config_id,
                        coordinate='prepared.backward.ownership_recipe'))
                    break
        for candidate in row_store_seeds(adapter,base.config):
            key=adapter.config_id(candidate)
            if key not in history.seen_ids:
                result.append(Proposal(candidate,self.name,parent_config_id=base.config_id,
                                       coordinate='prepared.row_gemm.store_recipe'))
                break
        if len(result)>=limit:return result[:limit]
        # Rotate the row-GEMM stage across component turns. Repeatedly
        # rebasing a winning down tile must not starve gate-dX indefinitely.
        geometry=list(geometry_growth_seeds(adapter,base.config))
        turns=sum(getattr(row,'coordinate',None)=='prepared.row_gemm.geometry_recipe'
                  for row in history.observations)
        if geometry:
            shift=turns%len(geometry)
            geometry=geometry[shift:]+geometry[:shift]
        for candidate in geometry:
            key=adapter.config_id(candidate)
            if key not in history.seen_ids:
                result.append(Proposal(candidate,self.name,parent_config_id=base.config_id,
                    coordinate='prepared.row_gemm.geometry_recipe'))
                break
        if len(result)>=limit:return result[:limit]
        # New families must not starve the existing geometry rotation.
        for candidate in preparation_ownership_seeds(adapter,base.config):
            key=adapter.config_id(candidate)
            if key not in history.seen_ids:
                result.append(Proposal(candidate,self.name,parent_config_id=base.config_id,
                                       coordinate='prepared.preparation.ownership_recipe'))
                break
        if len(result)>=limit:return result[:limit]
        stages=('gate_dw','down','gate_dx','down_dx','down_dw','gate_up','recompute') if training else ('down','gate_up')
        # Round-robin components before deepening any one stage's geometry.
        coordinates=['prepared.forward_engine','prepared.input_preparation','prepared.quant_grid','prepared.bf16_outputs','prepared.linear_transport','prepared.glu_output','prepared.quant.quant_math',
                     'prepared.forward_glu_engine',
                     'prepared.transpose_mode','prepared.transpose_warp_tokens','prepared.transpose_warp_raster',
                     'prepared.quant.transposed_tile_rows','prepared.quant.transposed_tile_k',
                     *[f'prepared.{s}.transport' for s in stages if s not in ('gate_up','recompute')],
                     *[f'prepared.{s}.epilogue_transport' for s in stages if s not in ('gate_up','recompute')],
                     *[f'prepared.{s}.geometry' for s in stages],
                     *[f'prepared.{s}.stages' for s in stages],
                     'prepared.quant.num_warps','prepared.quant.transposed_load_engine',
                     'prepared.quant.persistent_waves',
                     ]
        if training:
            coordinates[2:2]=['prepared.dy_preparation','prepared.route_gradient_epilogue',
                'prepared.joint_recompute_gradient_warps','prepared.weight_preparation',
                'prepared.weight_tile','prepared.weight_load_values','prepared.weight_store_bits',
                'prepared.weight_warps','prepared.weight_raster',
                'prepared.route_gradient_loader','prepared.route_gradient_gemm.geometry',
                'prepared.gate_dw_engine','prepared.down_dw_engine',
                'prepared.gate_dw_reuse_axis','prepared.down_dw_reuse_axis',
                'prepared.gate_dw_reuse_tiles','prepared.down_dw_reuse_tiles',
                'prepared.indexed_dw_token_lanes','prepared.indexed_dw_gather_pairs','prepared.indexed_dw_producer_warps',
                'prepared.down_dw_splits','prepared.gate_dw_splits']
            coordinates.extend(('prepared.joint_recompute_engine','prepared.recompute.glu_quant_vec',
                                'prepared.recompute.glu_scratch_swizzle'))
            coordinates.extend(('prepared.dw_reduce_threads','prepared.dw_reduce_vector'))
            coordinates.extend(f'prepared.{stage}.short_k' for stage in ('down_dw','gate_dw')
                               if f'prepared.{stage}.short_k' in adapter.axes)
        if base.config.prepared.forward_engine=='onchip':
            coordinates[1:1]=['gate_up.tile_m','gate_up.transport','down.tile_n','onchip_down_stages','onchip_pipeline']
        if 'prepared.active_prefix' in adapter.axes:coordinates.append('prepared.active_prefix')
        turn=sum(r.strategy==self.name for r in history.observations)%len(coordinates)
        coordinates=coordinates[turn:]+coordinates[:turn]
        neighbors=[]
        for coordinate in coordinates:
            choices=[]
            for value in adapter.axes[coordinate]:
                c=adapter.update_fn(base.config,coordinate,value)
                key=adapter.config_id(c)
                if key!=base.config_id and key not in history.seen_ids and adapter.rejection(c) is None:
                    choices.append(Proposal(c,self.name,parent_config_id=base.config_id,
                                            coordinate=coordinate,coordinate_value=value))
            rng.shuffle(choices);neighbors.append(choices)
        seen={adapter.config_id(p.config) for p in result}
        while any(neighbors) and len(result)<limit:
            for choices in neighbors:
                if choices:
                    p=choices.pop(0);key=adapter.config_id(p.config)
                    if key not in seen:seen.add(key);result.append(p)
                    if len(result)==limit:break
        return result

    def observe(self,observation):pass


class MoEFamilySearch:
    """A budget-independent progressive prefix: GMEM first, fusion after eight.

    Keep the full candidate pool/checkpoint identical when extending 8 to
    16/32 trials. Filtering the pool by requested budget broke resumability.
    The gate applies to every search arm, not only structural proposals.
    """
    def __init__(self,inner):self.inner,self.name=inner,inner.name

    @property
    def pool(self):return self.inner.pool

    def propose(self,adapter,history,rng,limit):
        candidates=self.inner.propose(adapter,history,rng,max(limit,32))
        initial=adapter.initial_config
        admit_onchip=(len(history.current)>=8 or initial.family=='grouped_onchip'
            or (initial.family=='grouped_prepared' and initial.prepared.forward_engine=='onchip'))
        def fused(config):
            return config.family=='grouped_onchip' or (
                config.family=='grouped_prepared' and config.prepared.forward_engine=='onchip')
        return [p for p in candidates if admit_onchip or not fused(p.config)][:limit]

    def observe(self,observation):return self.inner.observe(observation)


def _contract(operands,spec,training):
    from .legacy import default_cache_dir
    from .winners import runtime_winner_key
    x,gu,down,ids,weights,inverse=operands
    shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
    values=dict(spec['policy']);values['predictor_heads']=tuple(values['predictor_heads'])
    policy=AutotunePolicy(**values).for_pass(training)
    variant=dict(shape=shape,train=training,source=source_revision(),initial=spec['initial'],
                 measurement=MEASUREMENT_CONTRACT,effort=policy.evidence_contract())
    if 'batch_contract' in spec:variant['batch_contract']=spec['batch_contract']
    key=runtime_winner_key('mxfp8_balanced_experts',MXFP8Problem(shape[0],shape[1],shape[5]),
        device=x.device,kernel_revision=MOE_KERNEL_REVISION,
        variant=stable_id(variant))
    return shape,key,Path(spec['cache_dir']) if spec['cache_dir'] else default_cache_dir(),policy


def joint_recompute_seeds(shape,initial=DEFAULT_MOE_CONFIG):
    """Matched recompute alternatives with different liveness and tile costs.

    Only the recompute schedule changes for an existing joint seed. Preserve
    inherited role registers and compiler cap; schedule gains must come from
    lower liveness, less staging, or a more appropriate tile size.
    """
    base=initial.prepared if initial.family=='grouped_prepared' else DEFAULT_PREPARED_MOE_CONFIG
    g=base.recompute
    if base.quant.scale_layout!='row_major':return
    if shape[5]%32:return
    for rows,cols,stages,engine in (
            (64,64,1,'dual_operand'),(64,128,1,'operand_reuse'),
            (64,64,1,'operand_reuse'),(64,128,2,'dual_operand')):
        if cols>2*shape[5]:continue
        kwargs=dict(tile_m=rows,tile_n=cols,tile_k=128,atom_layout_m=2,atom_layout_n=2,
                    stages=stages,scale_role='producer',scale_smem_store='scalar',
                    tiles_per_cta=1,persistent_waves=0,tile_locality='raster',
                    glu_grad_store='exchange',glu_grad_load='interleaved')
        # Joint executables can require 32-row physical padding; keep all
        # other policy of an inherited executable unchanged.
        p=replace(base,backward_preparation='joint',joint_recompute_engine=engine,
                  recompute=replace(g,**kwargs))
        candidate=replace(initial,family='grouped_prepared',prepared=p)
        if rejection(candidate,shape) is None:yield candidate


def short_dw_seeds(shape,initial=DEFAULT_MOE_CONFIG):
    """Cover packed K32/K64 as whole expert schedules before local rebasing."""
    base=initial.prepared if initial.family=='grouped_prepared' else DEFAULT_PREPARED_MOE_CONFIG
    k=((base.physical_capacity(shape[3])+31)//32)*32
    if k>128 or base.quant.scale_layout!='row_major' or base.active_prefix:return
    # Start with the broadest exact staging tile to avoid extra copy/barrier
    # rounds. K32 is still proposed separately for K64, and leads at K96
    # where K64 would introduce an extra empty K32 instruction.
    tiles=(32,) if k==32 else (64,32) if k%64==0 else (32,64)
    for tile_k in tiles:
        for stages in (('down_dw','gate_dw'),('gate_dw',),('down_dw',)):
            p=base
            for stage in stages:p=p.with_short_dw(stage,tile_k)
            candidate=replace(initial,family='grouped_prepared',prepared=p)
            if rejection(candidate,shape) is None:yield candidate


def token_reduce_seeds(shape,initial,*,training=True):
    """Complete movement alternatives on the incumbent, never a dispatched winner."""
    phase='train' if training else 'forward'
    for width,lanes in ((4,32),(8,16),(8,32),(4,16),(2,8)):
        kinds=(('combine','dx'),('combine',),('dx',)) if training else (('combine',),)
        for enabled in kinds:
            changes={}
            for kind in enabled:
                changes.update({kind+'_reduce_engine':'token',kind+'_vector_width':width,
                                kind+'_lanes_per_token':lanes})
            candidate=replace(initial,**changes)
            if rejection(candidate,shape,phase) is None:yield candidate


def preparation_layout_seeds(adapter,initial):
    """Complete producer/layout alternatives; deep search retains each leg."""
    if initial.family!='grouped_prepared':return
    anchor=adapter.update_fn(initial,'prepared.active_prefix',True)
    if adapter.context.workload['phase']=='train':
        anchor=adapter.update_fn(anchor,'prepared.dy_preparation','fused')
    elif anchor.prepared.input_preparation=='gather_bf16':
        anchor=adapter.update_fn(anchor,'prepared.input_preparation','quantize_once')
    if adapter.rejection(anchor) is not None:return
    names=tuple(n for n in ('bounded_gather','bounded_dy','bounded_xt') if 'prepared.'+n in adapter.axes)
    for mask in range(1,1<<len(names)):
        candidate=anchor
        for bit,name in enumerate(names):
            candidate=adapter.update_fn(candidate,'prepared.'+name,bool(mask&(1<<bit)))
        if adapter.rejection(candidate) is None:yield candidate
    if 'prepared.forward_row_layout' in adapter.axes:
        candidate=adapter.update_fn(anchor,'prepared.forward_row_layout','compact')
        if adapter.rejection(candidate) is None:
            yield candidate
            for name in names:candidate=adapter.update_fn(candidate,'prepared.'+name,True)
            if adapter.rejection(candidate) is None:yield candidate


def structural_seeds(shape,initial=DEFAULT_MOE_CONFIG,*,training=True):
    """Complete, legal transport families, including oriented backward loads.

    A vectorized forward plus scalar backward is not a vectorized training
    family. Logical transposes need the core's staged ldmatrix quantizer and
    paired stores; independently changing load width otherwise rejects the
    entire seed before it ever reaches a timing trial.
    """
    yield replace(initial,family='grouped_prepared',gather_mode='warp')
    yield from token_reduce_seeds(shape,initial,training=training)
    base=initial.prepared if initial.family=='grouped_prepared' else DEFAULT_PREPARED_MOE_CONFIG
    if base.physical_capacity(shape[3])>=2*shape[3]:
        # An inherited aligned-output winner can waste most rows at a new
        # small shape. Offer a complete compact family early, not a sequence
        # of individually illegal capacity/epilogue coordinate mutations.
        if base.quant.scale_layout=='row_major':
            compact=replace(base,capacity_multiple=1)
            cs=compact.physical_capacity(shape[3])
            # Only these outputs have token capacity as M. Weight-gradient
            # M/N are weight dimensions, so their aligned TMA stores remain
            # legal and valuable even when the token reduction is short.
            for stage,columns in (('down',shape[1]),('down_dx',shape[5]),('gate_dx',shape[1])):
                value=getattr(compact,stage)
                if value.epilogue=='tma' and (cs%value.tile_m or columns%value.tile_n):
                    compact=replace(compact,**{stage:replace(value,epilogue='direct',store_vec=1,epilogue_stages=1)})
            for mode in ('gather_bf16','quantize_once'):
                candidate=replace(initial,family='grouped_prepared',gather_mode='warp',
                    prepared=replace(compact,input_preparation=mode))
                if rejection(candidate,shape,'train' if training else 'forward') is None:yield candidate
        for mode in ('gather_bf16','quantize_once'):
            candidate=replace(initial,family='grouped_prepared',gather_mode='warp',
                prepared=replace(DEFAULT_PREPARED_MOE_CONFIG,input_preparation=mode))
            if rejection(candidate,shape,'train' if training else 'forward') is None:yield candidate
    if training:
        yield from joint_recompute_seeds(shape,initial)
        yield from short_dw_seeds(shape,initial)
    yield replace(initial,family='grouped_prepared',gather_mode='warp',prepared=replace(base,input_preparation='quantize_once'))
    for token_lanes in (8,4,2):
        for raster in ('features','tokens'):
            for grid in ('operand','expert','portfolio'):
                yield replace(initial,family='grouped_prepared',gather_mode='warp',
                    prepared=replace(base,transpose_mode='warp',transpose_warp_tokens=token_lanes,
                                     transpose_warp_raster=raster,quant_grid=grid))
    for mode in ('expert','portfolio'):
        yield replace(initial,family='grouped_prepared',gather_mode='warp',prepared=replace(base,quant_grid=mode))
    adapter=make_moe_adapter(shape,None,initial=replace(initial,family='grouped_prepared'),
                             phase='train' if training else 'forward')
    cooperative=adapter.update_fn(adapter.initial_config,'prepared.linear_transport','cooperative')
    if rejection(cooperative,shape,'train' if training else 'forward') is None:yield cooperative
    # A single all-stages cooperative candidate couples short forward/dX
    # reductions to long dW loops. Offer independent short-reduction basins:
    # one buffer, cooperative transport, and larger output ownership. The
    # ordinary multi-buffer/TMA candidates remain; these are not dispatch
    # rules or hand-selected SKU schedules.
    for stage,(problem,_,_) in problems(shape).items():
        if stage in ('gate_up','recompute') or (not training and stage!='down'):continue
        direct=adapter.update_fn(adapter.initial_config,f'prepared.{stage}.transport','cooperative')
        if rejection(direct,shape,'train' if training else 'forward') is None:yield direct
        if problem.storage_k>2*getattr(base,stage).tile_k:continue
        single=adapter.update_fn(adapter.initial_config,f'prepared.{stage}.stages',1)
        if rejection(single,shape,'train' if training else 'forward') is None:yield single
        for transport in ('tma','cooperative'):
            tiled=adapter.update_fn(adapter.initial_config,f'prepared.{stage}.geometry',(128,128,128,4,2))
            tiled=adapter.update_fn(tiled,f'prepared.{stage}.transport',transport)
            tiled=adapter.update_fn(tiled,f'prepared.{stage}.stages',1)
            if rejection(tiled,shape,'train' if training else 'forward') is None:yield tiled
    if training:
        indexed=adapter.update_fn(adapter.initial_config,'prepared.input_preparation','indexed_transpose')
        if rejection(indexed,shape,'train') is None:
            yield indexed
            for mode in ('fused','fused_all'):
                fused_dy=adapter.update_fn(indexed,'prepared.dy_preparation',mode)
                if rejection(fused_dy,shape,'train') is None:yield fused_dy
                for rows,slots in ((32,32),(64,32),(128,32),(64,64),(128,64),(128,128),(256,32)):
                    candidate=adapter.update_fn(fused_dy,'prepared.quant.transposed_tile_rows',rows)
                    candidate=adapter.update_fn(candidate,'prepared.quant.transposed_tile_k',slots)
                    if rejection(candidate,shape,'train') is None:yield candidate
            fused_indexed=adapter.update_fn(indexed,'prepared.route_gradient_epilogue','fused')
            if rejection(fused_indexed,shape,'train') is None:yield fused_indexed
        fused_route=adapter.update_fn(adapter.initial_config,'prepared.route_gradient_epilogue','fused')
        if rejection(fused_route,shape,'train') is None:yield fused_route
        # Complete split-K basins, not individually illegal epilogue/split
        # mutations. K-tiles, never sequence boundaries, define the work;
        # packed zero gaps and independently quantized blocks are preserved.
        for splits in (2,4,8,16):
            split_config=adapter.initial_config
            for stage in ('down_dw','gate_dw'):
                split_config=adapter.update_fn(split_config,'prepared.'+stage+'_splits',splits)
            if rejection(split_config,shape,'train') is None:yield split_config
        for vec in (1,4):
            tma_output=adapter.initial_config
            for stage in ('gate_dw','down_dw'):
                tma_output=adapter.update_fn(tma_output,f'prepared.{stage}.epilogue_transport',('tma',vec,1))
            if rejection(tma_output,shape,'train') is None:yield tma_output
        tma_output=adapter.update_fn(adapter.initial_config,'prepared.bf16_outputs','tma_all')
        if rejection(tma_output,shape,'train') is None:yield tma_output
    # Cover the two principal short-reduction tiling basins early, before
    # spending depth on conversion/resource variants of only the 64x64 seed.
    for rows,cols,am in ((128,64,4),(64,128,2)):
        values={stage:replace(getattr(base,stage),tile_m=rows,tile_n=cols,atom_layout_m=am,
                              atom_layout_n=2,stages=2) for stage in problems(shape)}
        p=replace(base,**values)
        candidate=replace(initial,family='grouped_prepared',gather_mode='warp',prepared=p)
        if rejection(candidate,shape,'train' if training else 'forward') is None:yield candidate
    for vec in (4,8):
        p=replace(base,gate_up=replace(base.gate_up,glu_quant_vec=vec,glu_silu_math='shared_reciprocal'),
                  recompute=replace(base.recompute,glu_quant_vec=vec,glu_silu_math='shared_reciprocal'))
        candidate=replace(initial,family='grouped_prepared',gather_mode='warp',prepared=p)
        if rejection(candidate,shape,'train' if training else 'forward') is None:yield candidate
        if training:
            p=replace(p,backward_preparation='joint',recompute=replace(p.recompute,glu_grad_store='exchange'))
            candidate=replace(candidate,prepared=p)
            if rejection(candidate,shape,'train') is None:yield candidate
    native=replace(initial,family='grouped_prepared',gather_mode='warp',
                   prepared=DEFAULT_PREPARED_MOE_CONFIG.with_scale_layout('mma128'))
    if rejection(native,shape,'train' if training else 'forward') is None:yield native
    # Resource/occupancy basins are compound moves, not fixed register tuning.
    # Include narrow single-stage producer-scale tiles and ordinary/native
    # tiles under the same generic low/high register envelopes.
    for layout,rows,cols,am,depth,role in (
        ('row_major',64,64,2,1,'producer'),('row_major',128,128,4,1,'consumers'),
        ('mma128',128,128,4,1,'tma')):
        for cap,producer,consumer in ((128,32,96),(160,40,128),(255,48,192)):
            base=DEFAULT_PREPARED_MOE_CONFIG.with_scale_layout(layout)
            values={stage:replace(getattr(base,stage),tile_m=rows,tile_n=cols,
                atom_layout_m=am,atom_layout_n=2,stages=depth,scale_role=role,
                maxrregcount=cap,producer_registers=producer,consumer_registers=consumer) for stage in problems(shape)}
            candidate=replace(initial,family='grouped_prepared',gather_mode='warp',prepared=replace(base,**values))
            if rejection(candidate,shape,'train' if training else 'forward') is None:yield candidate
    # Whole prepared execution families. Each stage remains independently
    # mutable afterwards; these coupled moves cross geometry/buffering basins
    # without requiring an illegal one-field intermediate configuration.
    for rows,cols,am,an,depth in ((64,128,2,2,2),(128,128,4,2,1),
                                 (64,256,2,4,1),(128,64,4,2,2)):
        schedules={}
        for name in problems(shape):
            c=getattr(DEFAULT_PREPARED_MOE_CONFIG,name)
            schedules[name]=replace(c,tile_m=rows,tile_n=cols,atom_layout_m=am,atom_layout_n=an,
                stages=depth,scale_role='producer' if rows==64 and depth==1 else 'consumers')
        candidate=replace(initial,family='grouped_prepared',gather_mode='warp',
                          prepared=replace(DEFAULT_PREPARED_MOE_CONFIG,**schedules))
        if rejection(candidate,shape,'train' if training else 'forward') is None:yield candidate
    vector=replace(initial,family='grouped_staged')
    for stage,(problem,a,b) in problems(shape).items():
        if not training and stage not in ('gate_up','down'):continue
        normalize=normalize_glu_config if stage in ('gate_up','recompute') else normalize_fwd_config
        fields=dict(quant_vec=8,quant_load_bits=128,quant_store_bits=32,
                    quant_amax='bf16_bits',reduction='redux')
        if 'transpose' in (a,b):
            fields.update(load_engine='cpasync',schedule='cooperative',bf16_tile_k=32,
                          bf16_swizzle='none',bf16_stages=2,quant_store_bits=16)
        candidate=normalize(getattr(vector,stage),**fields)
        if 'transpose' in (a,b) and candidate.oriented_implementation_rejection(problem,a,b):
            # cp.async requires full tiles; TMA can zero-fill logical tails
            # when its physical strides remain aligned. Otherwise retain a
            # scalar-load/paired-conversion family rather than losing every
            # vectorized forward seed because one backward tail is ineligible.
            candidate=normalize(candidate,load_engine='tma',schedule='cooperative')
            if candidate.oriented_implementation_rejection(problem,a,b):
                candidate=normalize(getattr(vector,stage),quant_vec=8,quant_load_bits=16,
                    quant_store_bits=16,quant_amax='bf16_bits',reduction='redux')
        vector=replace(vector,**{stage:candidate})
    yield vector
    adapter=make_moe_adapter(shape,None,initial=initial,phase='train' if training else 'forward')
    for depth in (1,2,3,4):
        fused=adapter.update_fn(vector,'family','grouped_onchip')
        fused=replace(fused,onchip_down_stages=depth,onchip_pipeline='warp_specialized')
        if rejection(fused,shape,'train' if training else 'forward') is None:yield fused
        for pipeline in ('serial','warp_specialized'):
            for rows in (32,64):
                for engine in ('scalar',fused.gate_up.load_engine):
                    gate=normalize_glu_config(fused.gate_up,tile_m=rows,load_engine=engine,
                        schedule='cooperative',bf16_stages=1 if engine=='scalar' else fused.gate_up.bf16_stages)
                    down=normalize_fwd_config(fused.down,tile_m=rows)
                    hybrid=replace(initial,family='grouped_prepared',gate_up=gate,down=down,
                        onchip_down_stages=depth,onchip_pipeline=pipeline,prepared=replace(base,forward_engine='onchip'))
                    if rejection(hybrid,shape,'train' if training else 'forward') is None:yield hybrid
    for rows in (32,64,128):
        c=vector
        for stage in ('gate_up','recompute','down_dx','down_dw','gate_dx','gate_dw'):
            if not training and stage!='gate_up':continue
            normalize=normalize_glu_config if stage in ('gate_up','recompute') else normalize_fwd_config
            c=replace(c,**{stage:normalize(getattr(c,stage),tile_m=rows)})
        yield c
    for schedule in ('cooperative','warp_specialized','three_role'):
        for depth in (1,2,3,4):
            c=vector
            for stage in ('gate_up','recompute','down_dx','down_dw','gate_dx','gate_dw'):
                if not training and stage!='gate_up':continue
                normalize=normalize_glu_config if stage in ('gate_up','recompute') else normalize_fwd_config
                engine='cpasync' if schedule=='cooperative' else 'tma'
                c=replace(c,**{stage:normalize(getattr(c,stage),tile_m=64,load_engine=engine,
                    schedule=schedule,bf16_tile_k=32,bf16_swizzle='none',bf16_stages=depth,
                    quant_load_bits=128,quant_vec=8,quant_store_bits=16,
                    quant_amax='bf16_bits',reduction='redux',
                    consumer_registers=232 if engine=='cpasync' else 160)})
            if rejection(c,shape,'train' if training else 'forward') is None:yield c


def make_moe_tuner(adapter,store,policy,*,progress=None):
    """Keep complete families in the existing UCB exploration arm.

    The shared effort engine gives only two explicit seeds opening coverage.
    Preserve the remaining seeds as proposals too, round-robin by execution
    family, instead of leaving TMA pipelines stranded in a pool audit.
    """
    from collections import defaultdict,deque
    from .sampling import StructuralSearch
    shape=tuple(adapter.context.workload[k] for k in
                ('tokens','d_model','experts','capacity','top_k','hidden'))
    seeds=list(structural_seeds(shape,adapter.initial_config,
                               training=adapter.context.workload['phase']=='train'))
    seeds=[c for c in seeds if c.family in adapter.axes['family']
           and (c.family!='grouped_prepared' or c.prepared.forward_engine in adapter.axes['prepared.forward_engine'])]
    # Protect complete memory-traffic alternatives in the two opening slots.
    # Otherwise a low budget can spend both slots on inactive normalization
    # aliases and never try removing replicated BF16 traffic. These are
    # measured candidates, not preferred winners or SKU-specific dispatch.
    opening=[]
    base=replace(adapter.initial_config,family='grouped_prepared')
    materialized=adapter.update_fn(base,'prepared.input_preparation','quantize_once')
    if adapter.context.workload['phase']=='train':
        materialized=adapter.update_fn(materialized,'prepared.backward_preparation','joint')
    if adapter.rejection(materialized) is None:opening.append(materialized)
    indexed=adapter.update_fn(base,'prepared.input_preparation','indexed_transpose')
    if adapter.context.workload['phase']=='train':
        indexed=adapter.update_fn(indexed,'prepared.route_gradient_epilogue','fused')
    if adapter.rejection(indexed) is None:opening.append(indexed)
    if 'prepared.active_prefix' in adapter.axes:
        for seed in (base,materialized,indexed):
            active=adapter.update_fn(seed,'prepared.active_prefix',True)
            if adapter.rejection(active) is None:opening.append(active)
    preparation=list(preparation_ownership_seeds(adapter,indexed if adapter.context.workload['phase']=='train' else materialized))
    layouts=list(preparation_layout_seeds(adapter,adapter.initial_config))
    seeds=[*opening,*layouts,*preparation,*seeds]
    tuner=make_effort_autotuner(adapter,store,policy,progress=progress,candidates=seeds)
    # A first-hit deadline may expire before a long generic neighborhood.
    # Protect complete structural challengers *before* those neighbors, with
    # one trial left for shared-UCB exploration in an eight-trial prefix.
    # Never override a user's compatible research-prior opening.
    ranked_opening=any(name=='frozen_ranker' or name.startswith('portable_') for name in tuner.strategies)
    packed_opening=()
    if (policy.max_trials>=8 and not ranked_opening and not policy.initial_neighborhood_trials
            and policy.exploration_trials==1 and not policy.baseline_trials):
        from .core import Proposal
        inherited=list(tuner.coverage_proposals)
        protected=[]
        covered={adapter.config_id(adapter.initial_config)}
        alternatives=[]
        ownership=prepared=None
        if adapter.context.workload['phase']=='train':
            compact=next((c for c in joint_recompute_seeds(shape,indexed)
                          if adapter.rejection(c) is None),indexed)
            short_dw=next((c for c in short_dw_seeds(shape,base) if adapter.rejection(c) is None),None)
            if short_dw is not None:
                challenger=short_dw
            else:
                challenger=next(geometry_growth_seeds(adapter,compact),compact) if shape[3]>=128 else compact
            # Compare pass elimination on the same complete recipe rather
            # than proposing an isolated coordinate on the original seed.
            # At a 15s admission cap only the incumbent and one challenger may
            # fit: expose the complete traffic/geometry combination there.
            fused=adapter.update_fn(challenger,'prepared.dy_preparation','fused')
            stores=None
            for anchor in (fused,challenger,materialized,base):
                if adapter.rejection(anchor) is not None:continue
                stores=next(row_store_seeds(adapter,anchor),None)
                if stores is not None:break
            # One complete store basin at trial two, then its same-recipe
            # direct control. Do not spend eight trials on isolated illegal
            # tile/buffer/epilogue intermediate moves.
            if stores is not None:alternatives.extend((stores,anchor))
            # Protect one ownership challenger on long reductions, without
            # displacing the original store and direct controls or incumbent.
            # On short reductions extra consumers cannot amortize their setup.
            if indexed.prepared.physical_capacity(shape[3])>=8*indexed.prepared.gate_dw.tile_k:
                ownership=next(backward_ownership_seeds(adapter,stores or fused),None)
                if ownership is not None:
                    prepared=next(preparation_ownership_seeds(adapter,ownership),None)
                    if prepared is not None:alternatives.append(prepared)
                    alternatives.append(ownership)
            alternatives.extend((challenger,fused) if short_dw is not None else (fused,challenger))
            alternatives.extend((compact,materialized))
        else:
            packed=adapter.update_fn(materialized,'prepared.forward_engine','packed')
            for anchor in (packed,base):
                if adapter.rejection(anchor) is not None:continue
                store=next(row_store_seeds(adapter,anchor),None)
                if store is not None:
                    alternatives.extend((store,anchor));break
            alternatives.extend((materialized,indexed))
        if 'prepared.active_prefix' in adapter.axes:
            first=base if base.prepared.backward_preparation=='joint' else materialized
            for candidate in (adapter.update_fn(seed,'prepared.active_prefix',True)
                              for seed in (first,indexed)):
                if adapter.rejection(candidate) is None:
                    alternatives.append(candidate)
                    break
        # Rebase prefix and movement challengers onto the incumbent instead
        # of throwing away already-qualified fragment/panel ownership wins.
        # A shape-only padding bound gives packed batches early coverage;
        # actual timing still decides. No cu_seqlens readback or SKU table.
        active=None
        if 'prepared.active_prefix' in adapter.axes:
            for seed in (base,materialized,indexed):
                candidate=adapter.update_fn(seed,'prepared.active_prefix',True)
                if adapter.rejection(candidate) is None:
                    active=candidate;break
        mean_rows=max(32,((shape[0]*shape[4]+shape[2]-1)//shape[2]+31)//32*32)
        padded=active is not None and base.prepared.physical_capacity(shape[3])>=1.5*mean_rows
        active_movement=None
        if padded:
            alternatives.insert(0,active)
            # Measure the movement owner on that *same* preserved recipe.
            # Combining it only with a different input preparation can hide
            # a gain under the cost of that unrelated change.
            active_movement=next(token_reduce_seeds(shape,active,
                training=adapter.context.workload['phase']=='train'),None)
            if active_movement is not None:alternatives.insert(1,active_movement)
        movement_anchor=materialized
        if active is not None:
            prefixed=adapter.update_fn(materialized,'prepared.active_prefix',True)
            if adapter.rejection(prefixed) is None:movement_anchor=prefixed
        movement=next(token_reduce_seeds(shape,movement_anchor,
                      training=adapter.context.workload['phase']=='train'),None)
        if movement is not None:
            # Keep the first store/direct controls, but do not let six old
            # structural families hide the new complete movement alternative.
            alternatives.insert(min(3,len(alternatives)),movement)
        if padded:
            anchor=active_movement or active
            early=[active,anchor]
            bounded=anchor
            for name in ('bounded_dy','bounded_xt','bounded_gather'):
                if 'prepared.'+name not in adapter.axes:continue
                candidate=adapter.update_fn(bounded,'prepared.'+name,True)
                if adapter.rejection(candidate) is None:bounded=candidate
            if adapter.config_id(bounded)!=adapter.config_id(anchor):early.append(bounded)
            if 'prepared.forward_row_layout' in adapter.axes:
                candidate=adapter.update_fn(anchor,'prepared.forward_row_layout','compact')
                if adapter.rejection(candidate) is None:
                    early.append(candidate)
                    both=replace(candidate,prepared=replace(candidate.prepared,
                        bounded_gather=bounded.prepared.bounded_gather,
                        bounded_dy=bounded.prepared.bounded_dy,bounded_xt=bounded.prepared.bounded_xt))
                    if adapter.rejection(both) is None:early.append(both)
            early.append(materialized)
            alternatives=[*early,*alternatives]
            if adapter.context.workload['phase']=='train':
                packed_opening=cold_packed_padding_opening(adapter,stores or fused,ownership,prepared)
                if packed_opening:alternatives=[*packed_opening,*alternatives]
        for candidate in alternatives:
            identifier=adapter.config_id(candidate)
            if identifier not in covered and adapter.rejection(candidate) is None:
                protected.append(Proposal(candidate,'structural'))
                covered.add(identifier)
        # Keep useful generic anchors only after structural families. Six
        # protected proposals + incumbent leave one UCB-selected trial; use
        # the same prefix for progressive 8/16/32 continuation.
        for proposal in inherited:
            identifier=adapter.config_id(proposal.config)
            if identifier not in covered:
                protected.append(proposal);covered.add(identifier)
        tuner.coverage_proposals=protected[:6]
    groups=defaultdict(deque)
    seen=set()
    for config in seeds:
        key=adapter.config_id(config)
        if key in seen or adapter.rejection(config) is not None:continue
        seen.add(key)
        movement_identity=(config.combine_reduce_engine,config.dx_reduce_engine,
                           config.combine_vector_width,config.dx_vector_width,
                           config.combine_lanes_per_token,config.dx_lanes_per_token)
        identity=(movement_identity,config.family,config.prepared.quant.scale_layout,config.prepared.backward_preparation,
                  config.prepared.transpose_mode,
                  config.prepared.quant_grid,
                  config.prepared.input_preparation=='indexed_transpose',
                  config.prepared.route_gradient_epilogue,
                  config.prepared.transports,
                  config.prepared.forward_engine,
                  config.prepared.forward_row_layout,
                  config.prepared.bounded_gather,config.prepared.bounded_dy,config.prepared.bounded_xt,
                  config.prepared.dy_preparation,config.prepared.joint_recompute_engine,config.prepared.active_prefix,
                  (config.prepared.down_dw.tile_k<128,config.prepared.gate_dw.tile_k<128),
                  (config.gate_up.tile_m,config.gate_up.load_engine) if config.prepared.forward_engine=='onchip' else None,
                  max(config.prepared.down_dw_splits,config.prepared.gate_dw_splits)>1) if config.family=='grouped_prepared' else (
            movement_identity,config.family,config.gate_up.load_engine,config.gate_up.schedule)
        groups[identity].append(config)
    # Hierarchical coverage: dozens of prepared subfamilies must not crowd
    # all other execution engines out of a small-budget prefix. First cycle
    # engines, then cycle complete layout/traffic families within each engine.
    engines=defaultdict(deque)
    for identity,group in groups.items():
        first=group[0]
        engine=(first.family,first.prepared.forward_engine,first.prepared.dy_preparation!='materialized',
                first.prepared.joint_recompute_engine!='legacy',first.prepared.active_prefix) if first.family=='grouped_prepared' else (
            first.family,first.gate_up.load_engine,first.gate_up.schedule)
        engines[engine].append(group)
    portfolio=[]
    while any(engines.values()):
        for queue in engines.values():
            if queue:
                group=queue.popleft();portfolio.append(group.popleft())
                if group:queue.append(group)
    ordinary=tuner.strategies.get('structural')
    if ordinary is not None:
        portfolio.extend(c for c in ordinary.pool if adapter.config_id(c) not in seen)
    if portfolio:tuner.strategies['structural']=StructuralSearch(portfolio)
    tuner.strategies['prepared_components']=PreparedComponentSearch()
    tuner.strategies={name:MoEFamilySearch(strategy) for name,strategy in tuner.strategies.items()}
    tuner.scheduler.minimum_pulls={**tuner.scheduler.minimum_pulls,'prepared_components':2}
    if tuner.checkpoint_key is not None:
        opening_revision=dict(cold_packed_opening=1) if packed_opening else {}
        tuner.checkpoint_key=stable_id(dict(shared=tuner.checkpoint_key,expert_portfolio_revision=27,
                                           portfolio=[adapter.config_id(c) for c in portfolio],**opening_revision))
    return tuner


def legal_tuning_initial(shape,requested,phase='train'):
    """Recover a legal tuning seed when an inherited shape has changed.

    This is not runtime dispatch or a measured winner: every returned seed
    still passes the ordinary numerical and timing trial. In particular,
    a full-tile TMA store from another packed capacity must not abort a new
    search before it can try tail-safe direct stores. Explicit untuned
    configurations keep their normal rejection behavior.
    """
    def normalized(config):
        return make_moe_adapter(shape,None,initial=config,phase=phase).initial_config
    initial=normalized(requested)
    reason=rejection(initial,shape,phase)
    if reason is None:return initial,None
    candidates=[]
    if initial.family=='grouped_prepared':
        p=initial.prepared
        direct={name:replace(getattr(p,name),epilogue='direct',store_vec=1,epilogue_stages=1)
                for name in STAGES}
        candidates.append(replace(initial,prepared=replace(p,**direct)))
    candidates.extend((replace(DEFAULT_MOE_CONFIG,family='grouped_prepared'),DEFAULT_MOE_CONFIG))
    for candidate in candidates:
        candidate=normalized(candidate)
        if rejection(candidate,shape,phase) is None:return candidate,reason
    raise ValueError(f'No legal expert tuning seed for {shape}: inherited seed: {reason}')


def tune_moe(operands,spec,*,training=True,progress=None):
    from .preparation_runtime import _preparation_device_profile
    from .winners import save_runtime_winner
    cold_started=time.monotonic()
    shape,key,root,policy=_contract(operands,spec,training)
    reference_started=time.monotonic()
    evaluator=MoEEvaluator(operands,'train' if training else 'forward',policy)
    reference_seconds=time.monotonic()-reference_started
    phase='train' if training else 'forward'
    initial,initial_rejection=legal_tuning_initial(shape,MXFP8MoEConfig.from_dict(spec['initial']),phase)
    if initial_rejection and progress:
        progress(f'SEED inherited configuration rejected ({initial_rejection}); validating a legal replacement')
    tags=dict(source=source_revision(),initial=spec['initial'],measurement=MEASUREMENT_CONTRACT,effort=policy.evidence_contract())
    if initial_rejection:tags['initial_rejection']=initial_rejection
    if 'batch_contract' in spec:tags['batch_contract']=spec['batch_contract']
    adapter=make_moe_adapter(shape,evaluator.evaluate,initial=initial,
        allow_onchip=True,
        phase=phase,device=_preparation_device_profile(operands[0].device),
        tags=tags)
    residual_root=root/'moe_effort'
    store=ResidualTuningStore(effort_residual_root(residual_root/adapter.context.identifier,policy),transfer_root=residual_root)
    search_started=time.monotonic()
    if progress:progress(f'SETUP reference={reference_seconds:.3f}s context={search_started-cold_started:.3f}s; search admission cap={policy.max_seconds}s')
    result=make_moe_tuner(adapter,store,policy,progress=progress).tune()
    evidence=result.tuning_evidence or dict(schema=1,contract=policy.evidence_contract(),
        attempted_trials=len(list(store.records(adapter.context))))
    evidence={**evidence,'wall_time':dict(reference_setup_s=reference_seconds,
        context_setup_s=search_started-cold_started,
        search_call_s=time.monotonic()-search_started,
        invocation_to_winner_s=time.monotonic()-cold_started,
        limit_scope='soft search admission; setup and in-flight compilation are additional')}
    result.tuning_evidence=evidence
    save_runtime_winner(key,result.config.to_dict(),root=root,config_id=stable_id(result.config.to_dict()),
                        median_ms=result.median_ms,tuning_evidence=evidence,
                        metadata=dict(objective=MEASUREMENT_CONTRACT,phase='train' if training else 'forward'))
    return result


def _selection_identity(operands,spec,training):
    return str(operands[0].device),tuple(tuple(t.shape) for t in operands),spec,training


def remember_config(operands,spec,training,config):
    """Keep explicit refinement consistent with cold-selection memoization."""
    with _lock:
        _choices[_selection_identity(operands,spec,training)]=config


def resolve_config(operands,spec,training):
    import torch
    from .winners import load_runtime_winner,runtime_tuning_lock
    from .orchestrator import NoEligibleTimingError
    from ..moe import _parse_spec
    identity=_selection_identity(operands,spec,training)
    with _lock:
        cached=_choices.get(identity)
        if cached is not None:return cached
        spec=_parse_spec(spec)
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm expert selection before manual graph capture')
        shape,key,root,policy=_contract(operands,spec,training)
        def load():
            return load_runtime_winner(key,MXFP8MoEConfig.from_dict,root=root,
                tuning_policy=policy if spec['mode']=='balanced' else None)
        selected=load()
        if selected is None and spec['mode']=='balanced':
            with runtime_tuning_lock(key,root=root):
                selected=load()
                if selected is None:
                    stream=torch.cuda.current_stream(operands[0].device)
                    def execute():
                        from .._managed import tuning_resources
                        @tuning_resources
                        def run():
                            with torch.cuda.device(operands[0].device),torch.cuda.stream(stream):
                                return tune_moe(operands,spec,training=training,progress=_LOG.info).config
                        return run()
                    try:
                        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='rtx-moe-first-hit') as pool:
                            selected=pool.submit(execute).result()
                    except NoEligibleTimingError as error:
                        if error.fallback_config is None:raise
                        selected=error.fallback_config
        selected=selected or MXFP8MoEConfig.from_dict(spec['initial'])
        _choices[identity]=selected
        return selected


def _clear_runtime_caches(*,preserve_compiled=False):
    if preserve_compiled:return {}
    from .._moe_selection import clear_selection_cells
    from .._moe_execution import _COMPILED,_COMPILE_LOCK
    count=len(_choices);_choices.clear()
    cells=clear_selection_cells()
    # Drop only the launch-plan LRU, not kernel-library compilation caches or
    # independent resource owners retained by an outstanding CUDA graph.
    with _COMPILE_LOCK:
        compiled=len(_COMPILED);_COMPILED.clear()
    return {'moe_selections':count,'moe_selection_cells':cells,
            'moe_compiled_plans':compiled}
