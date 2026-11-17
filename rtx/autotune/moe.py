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
from ..configs.moe import DEFAULT_PREPARED_MOE_CONFIG,STAGES
from ..runtime import BoundedCache

MOE_KERNEL_REVISION=3
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
           '_execution_plan_token.py','_moe_prepared.py','_moe_execution.py','_moe_selection.py','configs/moe.py','configs/mxfp8.py',
           'kernels/mxfp8_moe_prepared.py','kernels/mxfp8_gemm.py','kernels/mxfp8_quant.py',
           'kernels/mxfp8_glu_prequant.py','kernels/mxfp8_moe_cooperative.py','kernels/mxfp8_moe_input.py',
           'kernels/mxfp8_moe_buffers.py')
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


def rejection(config,shape,phase='train'):
    if config.family=='grouped_prepared':
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
    stages=('gate_up','down') if phase=='forward' else tuple(problems(shape))
    def normalize(config):
        if config.family=='grouped_prepared':
            fields={name:getattr(DEFAULT_MOE_CONFIG,name) for name in problems(shape)}
            fields.update(onchip_down_stages=1,onchip_pipeline='serial')
            changes={}
            for name in stages:
                c=getattr(config.prepared,name);fields_c={}
                if config.prepared.transports[STAGES.index(name)]=='cooperative':
                    fields_c.update(producer_registers=24,consumer_registers=64,scale_role='producer',
                        scale_load_vec=1,scale_smem_store='scalar',scale_l2_prefetch='none',scale_l1_evict='default',
                        scale_cache='default',scale_schedule='before_wait',scale_recycle='barrier',mma_schedule='interleaved')
                if c.tile_locality!='raster':fields_c.update(raster='n',grid_swizzle=2)
                physical_k=((config.prepared.physical_capacity(shape[3])+31)//32*32
                            if name in ('down_dw','gate_dw') else problems(shape)[name][0].storage_k)
                if physical_k%c.tile_k:
                    fields_c.update(scale_load_vec=1,scale_smem_store='scalar',scale_l2_prefetch='none',
                                    scale_l1_evict='default',scale_cache='default')
                if name=='gate_up':fields_c.update(glu_hidden_decode='fp32',glu_grad_store='direct',glu_grad_load='interleaved')
                if name=='recompute':
                    fields_c.update(glu_hidden_decode='fp32',glu_epilogue='registers')
                    if config.prepared.backward_preparation=='separate':
                        fields_c.update(glu_grad_store='direct',glu_scratch_swizzle='none',glu_quant_vec=1)
                changes[name]=replace(c,**fields_c)
            fields['prepared']=replace(config.prepared,**changes)
            return replace(config,**fields)
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
            updates[name]=replace(c,**changes)
        config=replace(config,prepared=DEFAULT_PREPARED_MOE_CONFIG,**updates)
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
    axes['prepared.transpose_mode']=('direct','staged')
    axes['prepared.backward_preparation']=('separate','joint')
    axes['prepared.glu_output']=tuple((v,math) for v in (1,2,4,8) for math in ('division','shared_reciprocal'))
    axes['prepared.scale_transport']=('row_major','mma128')
    axes['prepared.linear_transport']=('tma','cooperative')
    axes['prepared.capacity_multiple']=(1,32,64,128,256)
    axes['prepared.quant_grid']=('operand','expert','portfolio')
    axes['prepared.input_preparation']=('gather_bf16','quantize_once')
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
                ('tma',4,1),('tma',4,2),('tma',4,3),('tma',4,4))
        if stage=='gate_up':
            axes[f'prepared.{stage}.glu_epilogue']=('exchange','registers')
            axes[f'prepared.{stage}.glu_quant_vec']=(1,2,4,8)
            axes[f'prepared.{stage}.glu_scratch_swizzle']=('none','xor_16b')
        if stage=='recompute':axes[f'prepared.{stage}.glu_grad_load']=('interleaved','prefetch')
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
            if parts[1]=='scale_transport':
                return normalize(replace(config,prepared=config.prepared.with_scale_layout(value)))
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
            if parts[1]=='backward_preparation' and value=='joint':
                p=replace(config.prepared,backward_preparation=value,
                    recompute=replace(config.prepared.recompute,glu_grad_store='exchange'))
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
    return DiscreteKernelAdapter(context,initial,axes,lambda c:stable_id(serialize(c)),
        serialize,lambda values:normalize(MXFP8MoEConfig.from_dict(values)),update,evaluator,
        lambda c:None if (why:=rejection(c,shape,phase)) is None else ('implementation_rejected',why),
        lambda c:dict(assignments=e*shape[3],input_bytes=n*d*2,expert_weights_bytes=e*3*d*h*2,
                      hidden_payload_bytes=e*shape[3]*((h+31)//32*32),saved_hidden_bytes=0))


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
        stages=('gate_dw','down','gate_dx','down_dx','down_dw','gate_up','recompute') if training else ('down','gate_up')
        # Round-robin components before deepening any one stage's geometry.
        coordinates=['prepared.input_preparation','prepared.quant_grid','prepared.bf16_outputs','prepared.linear_transport','prepared.glu_output','prepared.quant.quant_math',
                     'prepared.quant.transposed_tile_rows','prepared.quant.transposed_tile_k',
                     *[f'prepared.{s}.epilogue_transport' for s in stages if s not in ('gate_up','recompute')],
                     *[f'prepared.{s}.geometry' for s in stages],
                     *[f'prepared.{s}.stages' for s in stages],
                     'prepared.quant.num_warps','prepared.quant.transposed_load_engine',
                     'prepared.transpose_mode']
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
        result=[];seen=set()
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
        admit_onchip=len(history.current)>=8 or adapter.initial_config.family=='grouped_onchip'
        return [p for p in candidates if admit_onchip or p.config.family!='grouped_onchip'][:limit]

    def observe(self,observation):return self.inner.observe(observation)


def _contract(operands,spec,training):
    from .legacy import default_cache_dir
    from .winners import runtime_winner_key
    x,gu,down,ids,weights,inverse=operands
    shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
    values=dict(spec['policy']);values['predictor_heads']=tuple(values['predictor_heads'])
    policy=AutotunePolicy(**values).for_pass(training)
    key=runtime_winner_key('mxfp8_balanced_experts',MXFP8Problem(shape[0],shape[1],shape[5]),
        device=x.device,kernel_revision=MOE_KERNEL_REVISION,
        variant=stable_id(dict(shape=shape,train=training,source=source_revision(),
            initial=spec['initial'],measurement=MEASUREMENT_CONTRACT,effort=policy.evidence_contract())))
    return shape,key,Path(spec['cache_dir']) if spec['cache_dir'] else default_cache_dir(),policy


def structural_seeds(shape,initial=DEFAULT_MOE_CONFIG,*,training=True):
    """Complete, legal transport families, including oriented backward loads.

    A vectorized forward plus scalar backward is not a vectorized training
    family. Logical transposes need the core's staged ldmatrix quantizer and
    paired stores; independently changing load width otherwise rejects the
    entire seed before it ever reaches a timing trial.
    """
    yield replace(initial,family='grouped_prepared',gather_mode='warp')
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
    yield replace(initial,family='grouped_prepared',gather_mode='warp',prepared=replace(base,input_preparation='quantize_once'))
    for mode in ('expert','portfolio'):
        yield replace(initial,family='grouped_prepared',gather_mode='warp',prepared=replace(base,quant_grid=mode))
    adapter=make_moe_adapter(shape,None,initial=replace(initial,family='grouped_prepared'),
                             phase='train' if training else 'forward')
    cooperative=adapter.update_fn(adapter.initial_config,'prepared.linear_transport','cooperative')
    if rejection(cooperative,shape,'train' if training else 'forward') is None:yield cooperative
    if training:
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
    seeds=[c for c in seeds if c.family in adapter.axes['family']]
    tuner=make_effort_autotuner(adapter,store,policy,progress=progress,candidates=seeds)
    groups=defaultdict(deque)
    seen=set()
    for config in seeds:
        key=adapter.config_id(config)
        if key in seen or adapter.rejection(config) is not None:continue
        seen.add(key)
        identity=(config.family,config.prepared.quant.scale_layout,config.prepared.backward_preparation) if config.family=='grouped_prepared' else (
            config.family,config.gate_up.load_engine,config.gate_up.schedule)
        groups[identity].append(config)
    portfolio=[]
    while any(groups.values()):
        for group in groups.values():
            if group:portfolio.append(group.popleft())
    ordinary=tuner.strategies.get('structural')
    if ordinary is not None:
        portfolio.extend(c for c in ordinary.pool if adapter.config_id(c) not in seen)
    if portfolio:tuner.strategies['structural']=StructuralSearch(portfolio)
    tuner.strategies['prepared_components']=PreparedComponentSearch()
    tuner.strategies={name:MoEFamilySearch(strategy) for name,strategy in tuner.strategies.items()}
    tuner.scheduler.minimum_pulls={**tuner.scheduler.minimum_pulls,'prepared_components':2}
    tuner.checkpoint_key=stable_id(dict(shared=tuner.checkpoint_key,expert_portfolio_revision=3,
                                       portfolio=[adapter.config_id(c) for c in portfolio]))
    return tuner


def tune_moe(operands,spec,*,training=True,progress=None):
    from .preparation_runtime import _preparation_device_profile
    from .winners import save_runtime_winner
    shape,key,root,policy=_contract(operands,spec,training)
    evaluator=MoEEvaluator(operands,'train' if training else 'forward',policy)
    initial=MXFP8MoEConfig.from_dict(spec['initial'])
    adapter=make_moe_adapter(shape,evaluator.evaluate,initial=initial,
        allow_onchip=True,
        phase='train' if training else 'forward',device=_preparation_device_profile(operands[0].device),
        tags=dict(source=source_revision(),initial=spec['initial'],
                  measurement=MEASUREMENT_CONTRACT,effort=policy.evidence_contract()))
    residual_root=root/'moe_effort'
    store=ResidualTuningStore(effort_residual_root(residual_root/adapter.context.identifier,policy),transfer_root=residual_root)
    result=make_moe_tuner(adapter,store,policy,progress=progress).tune()
    evidence=result.tuning_evidence or dict(schema=1,contract=policy.evidence_contract(),
        attempted_trials=len(list(store.records(adapter.context))))
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
