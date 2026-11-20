"""Balanced routing adapter, graph evaluator and durable shared-engine search.

Policy budgets are context, not search axes. The objective includes native
Sinkhorn, hard scheduling, stable expert metadata and, for training, both
metadata and saved-state Sinkhorn adjoints. The differentiable affinity
projection is qualified in its own bounded context before scores detach.
"""
from dataclasses import replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import statistics
import time

from .core import DiscreteKernelAdapter,KernelContext,stable_id,ComposableTuningResult,Proposal
from .effort import AutotunePolicy,make_effort_autotuner,effort_residual_root
from .outcomes import TrialOutcome,raise_if_fatal_device_context_error
from .store import ResidualTuningStore
from ..routing import RoutingKernelConfig,SinkhornKernelConfig

ROUTING_KERNEL_REVISION=3
FLAT_BACKWARD_OPENING_REVISION=3
MEASUREMENT_CONTRACT=dict(revision=2,timing='cuda_graph',calibration='caller_and_sigmoid_normal',
                          oracle='independent_fp64',saved_state='all_normalization_probabilities',
                          objective='native_routing_and_metadata_stages',nrms_floor=1e-12,graph_warmup_ms=200)


@lru_cache(maxsize=1)
def _source_revision():
    root=Path(__file__).resolve().parents[1]
    files=['routing.py','_routing_autotune.py','_routing_hard.py','kernels/routing_sinkhorn.py',
           'kernels/routing_selection.py','_routing_graph.py','autotune/routing.py',
           '_routing_execution.py','_routing_reference.py','_moe_selection.py','_routing_metadata.py',
           'kernels/routing_metadata.py','kernels/routing_quota_cached.py',
           'kernels/routing_varlen.py','kernels/routing_flat.py']
    digest=hashlib.sha256()
    for name in files:digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def rejection(config,n,e,k=1):
    if n*e>=2**31:return 'implementation_rejected','Routing matrix exceeds signed kernel index range'
    packed=config.segmented_quotas and n*k>=config.quota_threshold
    if not packed and n*k>4096:
        return 'implementation_rejected','Direct quota implementation is bounded to 4096 proposals'
    if e>1024:return 'implementation_rejected','Native repair currently supports at most 1024 experts'
    if config.flat_row_backend=='warp' and e>128:
        return 'implementation_rejected','Flat warp rows currently support at most 128 experts'
    if e*max(config.flat_column_shards_forward,config.flat_column_shards_backward)>=2**31:
        return 'implementation_rejected','Flat column partials exceed native index range'
    if config.qb_backend=='native_warp' and e>128:
        return 'implementation_rejected','Warp QB row selection supports at most 128 experts'
    if config.metadata_backend=='cute' and (e>128 or k>32):
        return 'implementation_rejected','CuTe stable metadata supports E<=128 and K<=32'
    return None


def make_routing_adapter(n,e,k,evaluator,*,initial=None,phase='train',device=None,
                         sinkhorn_iterations=5,dual_iterations=5,repair_rounds=64,tags=None):
    if min(n,e,k)<=0 or k>e or n*k%e:raise ValueError('Invalid scheduled pool geometry')
    if phase not in ('forward','train'):raise ValueError('phase must be forward/train')
    initial=initial or RoutingKernelConfig()
    backward_seed=initial.backward
    def normalize(c):
        packed=c.segmented_quotas and n*k>=c.quota_threshold
        warp=c.flat_row_backend=='warp'
        return replace(c,segmented_quotas=packed,quota_threshold=0,
                       qb_threads=128 if c.qb_backend=='torch' else c.qb_threads,
                       quota_cache_items=c.quota_cache_items if packed else 0,
                       quota_early_exit=c.quota_early_exit if packed else False,
                       metadata_threads=c.metadata_threads if c.metadata_backend=='cute' else 128,
                       forward=replace(c.forward,row_groups=1) if warp else c.forward,
                       backward=backward_seed if phase=='forward' else (
                           replace(c.backward,row_groups=1) if warp else c.backward),
                       flat_column_shards_backward=0 if phase=='forward' else c.flat_column_shards_backward,
                       packed_column_cache_items=0,packed_row_backend='cta',
                       packed_column_shards_forward=0,packed_column_shards_backward=0,
                       packed_qb_column_backend='row_major',packed_sinkhorn_backend='staged',packed_persistent_threads=0)
    initial=normalize(initial)
    axes={'qb_backend':('torch','native_column','native','native_warp'),'qb_threads':(64,128,256),
          'segmented_quotas':(False,True),'quota_threshold':(0,256,1024,4096,16384),
          'quota_cache_items':(0,4,8,16),'quota_threads':(64,128,256),'quota_early_exit':(False,True),
          'metadata_backend':('torch','cute'),'metadata_threads':(128,256,512),
          'forward.threads':(64,128,256),'forward.row_groups':(1,2,4,8),
          'forward.column_groups':(1,2,4,8)}
    if phase=='train':
        axes.update({'backward.threads':(64,128,256),'backward.row_groups':(1,2,4,8),
                     'backward.column_groups':(1,2,4,8)})
    axes['flat_row_backend']=('cta','warp')
    axes['flat_column_shards_forward']=(0,4,8,16,32,64,128)
    if phase=='train':axes['flat_column_shards_backward']=(0,4,8,16,32,64,128)
    context=KernelContext('balanced_routing',ROUTING_KERNEL_REVISION,
        dict(m=n,n=e,k=k,phase=phase,sinkhorn_iterations=sinkhorn_iterations,
             dual_iterations=dual_iterations,repair_rounds=repair_rounds),
        device={} if device is None else device,tags={} if tags is None else tags)
    def update(c,name,value):
        if '.' in name:
            group,field=name.split('.')
            return normalize(replace(c,**{group:replace(getattr(c,group),**{field:value})}))
        return normalize(replace(c,**{name:value}))
    def serialize(c):
        values=normalize(c).to_dict()
        # Default schedules retain their existing canonical IDs. Nondefault
        # fields remain in full configs and immutable execution tokens.
        for name,default in (('flat_row_backend','cta'),('flat_column_shards_forward',0),
                             ('flat_column_shards_backward',0)):
            if values[name]==default:values.pop(name)
        return values
    def geometry_features(c):
        values=dict(elements=n*e,saved_state_bytes=(2*sinkhorn_iterations+1)*n*e*4)
        sm=(device or {}).get('multiprocessor_count',0)
        for direction in ('forward','backward'):
            g=getattr(c,direction);shards=getattr(c,'flat_column_shards_'+direction)
            ctas=(shards*math.ceil(e/(4*g.column_groups)) if shards else math.ceil(e/g.column_groups))
            lanes=g.threads//((4 if shards else 1)*g.column_groups)
            warp=c.flat_row_backend=='warp'
            values.update({direction+'_row_ctas':math.ceil(n/(g.threads//32 if warp else g.row_groups)),
                direction+'_column_ctas':ctas,direction+'_column_lanes':lanes,
                direction+'_column_rows_per_lane':math.ceil(n/(lanes*max(1,shards))),
                direction+'_column_ctas_per_sm':ctas/sm if sm else 0,
                direction+'_column_partial_bytes':shards*e*4,
                direction+'_row_lanes':32 if warp else g.threads//g.row_groups})
        values['hardware_sm_count_known']=int(bool(sm))
        return values
    return DiscreteKernelAdapter(context,initial,axes,lambda c:stable_id(serialize(c)),
        serialize,lambda value:normalize(RoutingKernelConfig.from_dict(value)),update,evaluator,
        lambda c:rejection(normalize(c),n,e,k),geometry_features)


def structural_seeds(initial):
    # Complete structural alternatives, not a shape/SKU-specific winner table.
    yield replace(initial,metadata_backend='cute')
    yield replace(initial,quota_cache_items=8,quota_early_exit=True,segmented_quotas=True)
    yield replace(initial,metadata_backend='cute',quota_cache_items=8,quota_early_exit=True,segmented_quotas=True)
    yield replace(initial,quota_cache_items=0,quota_early_exit=True,segmented_quotas=True)
    for threads in (64,256):
        yield replace(initial,quota_cache_items=8,quota_early_exit=True,segmented_quotas=True,quota_threads=threads)
    for backend in ('native_warp','native','native_column','torch'):
        for segmented in (False,True):
            yield replace(initial,qb_backend=backend,segmented_quotas=segmented)
    yield replace(initial,forward=SinkhornKernelConfig(128,4,8),backward=SinkhornKernelConfig(128,4,8))
    yield replace(initial,forward=SinkhornKernelConfig(64,4,4),backward=SinkhornKernelConfig(64,4,4))


def _flat_backward_geometry(adapter,config):
    """Find more column owners for a long, severely underfilled adjoint."""
    w=adapter.context.workload
    sm_count=adapter.context.device.get('multiprocessor_count')
    if w['phase']!='train' or not sm_count:
        return None
    rows,experts=w['m'],w['n']
    current=config.backward
    # Each CTA owns column_groups whole columns. Each lane scans
    # ceil(rows / lanes_per_column) rows twice. Reserve opening pressure
    # for long reductions with substantial headroom in the first CTA wave.
    rows_per_lane=math.ceil(rows/(current.threads//current.column_groups))
    column_ctas=math.ceil(experts/current.column_groups)
    if rows_per_lane<256 or 2*column_ctas>sm_count:
        return None
    groups=[g for g in adapter.axes['backward.column_groups']
            if g<current.column_groups and math.ceil(experts/g)<=sm_count]
    if not groups:
        return None
    # At low E maximize column owners. If there are enough experts to
    # reach even a quarter of a wave, retain adjacent columns per CTA
    # where possible for row-major coalescing. This is a shape heuristic,
    # not a forced winner or an achieved occupancy claim.
    owner_target=min(experts,math.ceil(sm_count/4))
    group=min(groups,key=lambda g:(max(0,owner_target-math.ceil(experts/g)),-g))
    threads=max(t for t in adapter.axes['backward.threads']
                if math.ceil(rows/(t//group))>=16)
    return replace(current,threads=threads,column_groups=group)


def flat_geometry_opening(adapter):
    """Shape/SM-driven contrasts; no named-SKU or measured-winner lookup.

    A long column scan with less than half a first wave has an obvious
    ownership alternative. Row ownership remains a contrasting choice, not
    an imposed winner. Kernel legality and the ordinary oracle still apply.
    """
    w=adapter.context.workload;sm=adapter.context.device.get('multiprocessor_count')
    if not sm:return []
    rows,experts=w['m'],w['n'];initial=adapter.initial_config
    def geometry(direction):
        old=getattr(initial,direction)
        scan=math.ceil(rows/(old.threads//old.column_groups))
        if scan<256 or 2*math.ceil(experts/old.column_groups)>sm:return old
        groups=[g for g in adapter.axes[direction+'.column_groups'] if math.ceil(experts/g)<=sm]
        if not groups:return old
        columns=min(groups);threads=max(adapter.axes[direction+'.threads'])
        row_lanes=32 if direction=='forward' else min(64,1<<(experts-1).bit_length())
        ownership=min(adapter.axes[direction+'.row_groups'],key=lambda g:abs(math.log2(threads/g/row_lanes)))
        return replace(old,threads=threads,row_groups=ownership,column_groups=columns)
    forward=geometry('forward')
    backward=geometry('backward') if w['phase']=='train' else initial.backward
    joint=replace(initial,forward=forward,backward=backward)
    if rows>=4096 and experts<=128 and math.ceil(experts/initial.forward.column_groups)<sm:
        # Supply executable alternatives early, not only after a deep budget.
        # The starting point targets two CTA waves with the sharded primitive's
        # four-adjacent-column ownership, then ordinary measurements decide.
        f=flat_shard_count(adapter,joint.forward.column_groups)
        b=flat_shard_count(adapter,joint.backward.column_groups) if w['phase']=='train' else 0
        parallel=replace(joint,flat_column_shards_forward=f,flat_column_shards_backward=b)
        warp=replace(parallel,flat_row_backend='warp')
        cells=(('parallel',parallel),('warp_parallel',warp),
               ('serial_torch',replace(joint,flat_column_shards_forward=0,
                    flat_column_shards_backward=0,flat_row_backend='cta',qb_backend='torch')),
               ('parallel_torch',replace(warp,qb_backend='torch')),
               ('parallel_native_warp',replace(warp,qb_backend='native_warp')))
        return [Proposal(c,'region_coverage',metadata=dict(opening_role='geometry',
                    opening_family='flat_parallel_transport',contrast=name,
                    revision=FLAT_BACKWARD_OPENING_REVISION)) for name,c in cells]
    if joint==initial:return []
    result=[]
    for name,c in (('joint',joint),('forward',replace(initial,forward=forward)),
                   ('backward',replace(initial,backward=backward))):
        result.append(Proposal(c,'region_coverage',metadata=dict(opening_role='geometry',
            opening_family='flat_column_ownership',contrast=name,revision=FLAT_BACKWARD_OPENING_REVISION)))
    for backend in ('torch','native_warp'):
        result.append(Proposal(replace(joint,qb_backend=backend),'region_coverage',
            metadata=dict(opening_role='fallback',opening_family='flat_hard_family',
                          contrast=backend,revision=FLAT_BACKWARD_OPENING_REVISION)))
    return result


def _flat_region(c):
    return (f'f{c.forward.column_groups}s{c.flat_column_shards_forward}/'
            f'b{c.backward.column_groups}s{c.flat_column_shards_backward}/'
            f'{c.flat_row_backend}/{c.qb_backend}')


def flat_shard_count(adapter,groups):
    sm=adapter.context.device.get('multiprocessor_count') or 1
    ctas=math.ceil(adapter.context.workload['n']/(4*groups))
    return min((4,8,16,32,64,128),key=lambda s:abs(math.log2(s*ctas/(2*sm))))


class FlatTransportSearch:
    """A resumable UCB arm crossing coupled transport/ownership regions.

    Anchors freeze in eight-attempt tranches; a noisy winner cannot endlessly
    restart the first one-field coordinate. No total-budget-dependent plan.
    """
    name='flat_transport'

    def propose(self,adapter,history,rng,limit):
        if len(history.current)<8:return []
        cutoff=max(8,(len(history.current)//8)*8)
        eligible=[r for r in history.current[:cutoff] if r.outcome.timing_eligible]
        if not eligible:return []
        anchor=min(eligible,key=lambda r:r.score);base=anchor.config
        train=adapter.context.workload['phase']=='train'
        f=flat_shard_count(adapter,base.forward.column_groups)
        b=flat_shard_count(adapter,base.backward.column_groups) if train else 0
        cells=[]
        def add(name,**values):cells.append((name,replace(base,**values)))
        add('parallel',flat_column_shards_forward=f,flat_column_shards_backward=b)
        for groups in (1,2,4,8):
            add('coalescing',forward=replace(base.forward,column_groups=groups),
                backward=replace(base.backward,column_groups=groups) if train else base.backward,
                flat_column_shards_forward=flat_shard_count(adapter,groups),
                flat_column_shards_backward=flat_shard_count(adapter,groups) if train else 0)
        for backend in ('native_warp','torch'):
            add('cached_hard',qb_backend=backend,quota_cache_items=8,quota_early_exit=True)
        for scale in (.5,2):
            add('shard_depth',flat_column_shards_forward=max(4,min(128,int(f*scale))),
                flat_column_shards_backward=max(4,min(128,int(b*scale))) if train else 0)
        add('forward_only',flat_column_shards_forward=f,flat_column_shards_backward=0)
        if train:add('backward_only',flat_column_shards_forward=0,flat_column_shards_backward=b)
        for row in ('warp','cta'):add('row_ownership',flat_row_backend=row)
        for backend in ('native_warp','torch','native_column'):add('hard_backend',qb_backend=backend)
        seen=set(history.seen_ids);out=[]
        for contrast,c in cells:
            key=adapter.config_id(c)
            if key in seen or adapter.rejection(c):continue
            seen.add(key);out.append(Proposal(c,self.name,parent_config_id=anchor.config_id,
                metadata=dict(contrast=contrast,anchor_prefix=cutoff)))
            if len(out)>=limit:break
        return out

    def observe(self,observation):pass


def make_flat_effort_autotuner(adapter,store,policy,*,progress=None,started_at=None,region_coverage=True):
    """Reserve a same-plan Sinkhorn backward contrast in the flat opening."""
    if (region_coverage and policy.region_coverage and policy.search_schedule=='progressive'
            and policy.retrieval_trials is None and not policy.initial_neighborhood_trials
            and not policy.baseline_trials and policy.exploration_trials==1
            and not policy.ranked_opening_trials):
        opening=flat_geometry_opening(adapter)
        if opening:
            adapter=replace(adapter,opening_fn=lambda:opening,region_fn=_flat_region)
            tuner=make_effort_autotuner(adapter,store,policy,progress=progress,
                candidates=list(structural_seeds(adapter.initial_config)),started_at=started_at)
            if any(p.metadata.get('opening_family')=='flat_parallel_transport' for p in opening):
                tuner.strategies['flat_transport']=FlatTransportSearch()
                if tuner.checkpoint_key:
                    tuner.checkpoint_key=stable_id(dict(base=tuner.checkpoint_key,flat_transport_revision=1))
            return tuner
    tuner=make_effort_autotuner(adapter,store,policy,progress=progress,
        candidates=list(structural_seeds(adapter.initial_config)),started_at=started_at)
    if (policy.search_schedule!='progressive' or policy.prior_artifact or policy.predictor_artifact
            or policy.retrieval_trials is not None or policy.initial_neighborhood_trials
            or policy.baseline_trials or policy.exploration_trials!=1):
        return tuner
    coverage=tuner.coverage_proposals
    # Keep the first existing hard alternative and compare it against a plan
    # differing only in backward geometry. The other original local and
    # structural attempts remain, as does the bounded exploration arm.
    if not coverage or coverage[0].strategy!='initial_neighborhood' or coverage[0].coordinate!='qb_backend':
        return tuner
    # An inherited Torch QB seed already occupies the basin that the
    # first QB neighbor covers from the common native-column cold seed.
    anchor=(adapter.initial_config if adapter.initial_config.qb_backend=='torch' else coverage[0].config)
    geometry=_flat_backward_geometry(adapter,anchor)
    if geometry is None:
        return tuner
    candidate=replace(anchor,backward=geometry)
    if adapter.rejection(candidate) is not None:
        return tuner
    cid=adapter.config_id(candidate)
    if cid==adapter.config_id(adapter.initial_config):
        return tuner
    contrast=Proposal(candidate,'structural',parent_config_id=adapter.config_id(anchor),
        metadata={'opening_family':'flat_sinkhorn_backward_geometry'})
    tuner.coverage_proposals=(coverage[0],contrast,*(p for p in coverage[1:] if adapter.config_id(p.config)!=cid))
    if tuner.checkpoint_key:
        tuner.checkpoint_key=stable_id(dict(base=tuner.checkpoint_key,revision=FLAT_BACKWARD_OPENING_REVISION,
            coverage=[adapter.config_id(p.config) for p in tuner.coverage_proposals]))
    return tuner


def cache_contract(scores,spec,phase,policy=None):
    from .legacy import default_cache_dir
    from .winners import runtime_winner_key
    from ..kernels.mxfp8 import MXFP8Problem
    if policy is None:
        values=dict(spec['policy']);values['predictor_heads']=tuple(values['predictor_heads'])
        policy=AutotunePolicy(**values)
    policy=policy.for_pass(phase=='train')
    root=Path(spec['cache_dir']) if spec['cache_dir'] else default_cache_dir()
    w=spec['workload']
    key=runtime_winner_key('balanced_routing',MXFP8Problem(scores.shape[0],scores.shape[1],w['top_k']),
        device=scores.device,kernel_revision=ROUTING_KERNEL_REVISION,
        variant=stable_id(dict(source=_source_revision(),workload=w,phase=phase,initial=spec['initial'],
                               measurement=MEASUREMENT_CONTRACT,effort=policy.evidence_contract())))
    return key,root,policy


def _metadata_gradient_reference(gradient,weights,assignments,inverse,n,e):
    """Traceable FP32 softmax adjoint for the Torch metadata candidate."""
    import torch
    selected=weights.flatten()[inverse]
    upstream=gradient.flatten()[inverse]
    upstream=upstream-upstream[:,:1]
    derivative=selected*(upstream-(upstream*selected).sum(1,keepdim=True))
    padded=torch.nn.functional.pad(derivative,(0,0,0,n-inverse.shape[0]))
    return torch.zeros((n,e),device=weights.device,dtype=torch.float32).scatter(1,assignments,padded)


class RoutingEvaluator:
    def __init__(self,scores,spec,phase,policy):
        import torch
        self.torch=torch;self.spec=spec;self.phase=phase;self.policy=policy
        self.owner_id=id(self);self.fixtures=[]
        w=spec['workload'];self.k=w['top_k'];self.steps=w['sinkhorn_iterations']
        self.qb=w['dual_iterations'];self.rounds=w['repair_rounds']
        self.real_tokens=w.get('real_tokens',scores.shape[0])
        if not 1<=self.real_tokens<=scores.shape[0]:raise ValueError('Invalid real routing token count')
        from .. import _routing_reference as reference
        compile_options=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        self.metadata_reference=torch.compile(lambda value,ids:reference.routing_metadata(
            value,ids,num_real_tokens=self.real_tokens),**compile_options)
        self.metadata_gradient=torch.compile(lambda grad,weight,ids,inverse:_metadata_gradient_reference(
            grad,weight,ids,inverse,scores.shape[0],scores.shape[1]),**compile_options)
        generator=torch.Generator(device=scores.device).manual_seed(policy.seed)
        synthetic=torch.nn.functional.logsigmoid(torch.randn(scores.shape,device=scores.device,generator=generator)*1.5)
        for source in (scores.detach(),synthetic):
            gradient=torch.randn(source.shape,device=source.device,generator=generator)
            # Matrix normalization is independent of the native potential form.
            with torch.enable_grad():
                oracle=source.double().requires_grad_()
                value=oracle
                for _ in range(self.steps):
                    value=value-torch.logsumexp(value,1,keepdim=True)+math.log(self.k)
                    value=value-torch.logsumexp(value,0,keepdim=True)+math.log(source.shape[0]*self.k/source.shape[1])
                expected_gradient=torch.autograd.grad(value,oracle,gradient.double())[0]
            p=value.detach().float().exp()
            route_gradient=torch.randn((scores.shape[1],scores.shape[0]*self.k//scores.shape[1]),
                                       device=scores.device,generator=generator)
            self.fixtures.append((source,gradient,value.detach(),expected_gradient,p,route_gradient))

    def hard(self,p,c):
        from .._routing_hard import dispatch_hard
        return dispatch_hard(p,self.k,self.qb,self.rounds,c.segmented_quotas,self.owner_id,
                             c.quota_threshold,c.qb_backend,c.qb_threads,c.quota_cache_items,c.quota_threads,c.quota_early_exit)

    def metadata(self,value,ids,c):
        if c.metadata_backend=='cute':
            from .._routing_metadata import run_forward
            return run_forward(value,ids,self.real_tokens,c.metadata_threads)
        return self.metadata_reference(value,ids)

    def metadata_backward(self,gradient,weights,ids,inverse,c):
        if c.metadata_backend=='cute':
            from .._routing_metadata import run_backward
            return run_backward(gradient,weights,ids,inverse,scheduled_tokens=self.fixtures[0][0].shape[0],
                                 experts=self.fixtures[0][0].shape[1],threads=c.metadata_threads)
        return self.metadata_gradient(gradient,weights,ids,inverse)

    def evaluate(self,c):
        from .._routing_execution import run_forward,run_backward
        from .measurement import measure_cuda_batches,runtime_timing_metadata
        torch=self.torch
        stage='compile';started=time.monotonic();graph=None
        try:
            errors={};max_error=0.
            for index,(source,gradient,expected,expected_gradient,p,route_gradient) in enumerate(self.fixtures):
                token=tuple(c.to_token())
                value,history=run_forward(source,self.k,self.steps,token)
                actual_gradient=run_backward(gradient,history,self.k,self.steps,token)
                stage='correctness'
                torch.testing.assert_close(value.double(),expected,rtol=5e-5,atol=5e-6)
                torch.testing.assert_close(actual_gradient.double(),expected_gradient,rtol=5e-4,atol=8e-6)
                baseline=self.hard(p,RoutingKernelConfig())
                actual=self.hard(p,c)
                torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
                ids,weights,inverse=self.metadata(value,actual,c)
                expected_ids,expected_weights,expected_inverse=self.metadata_reference(value,actual)
                torch.testing.assert_close(ids,expected_ids,rtol=0,atol=0)
                torch.testing.assert_close(inverse,expected_inverse,rtol=0,atol=0)
                torch.testing.assert_close(weights,expected_weights,rtol=5e-5,atol=5e-7)
                if self.phase=='train':
                    meta_grad=self.metadata_backward(route_gradient,weights,actual,inverse,c)
                    expected_meta_grad=self.metadata_gradient(route_gradient,expected_weights,actual,expected_inverse)
                    torch.testing.assert_close(meta_grad,expected_meta_grad,rtol=5e-4,atol=2e-6)
                    # Constant upstream must lie in the softmax nullspace.
                    # This cold-only check catches cancellation that random
                    # score-gradient tolerances can hide after scalar sums.
                    constant=self.metadata_backward(torch.ones_like(weights),weights,actual,inverse,c)
                    torch.testing.assert_close(constant,torch.zeros_like(constant),rtol=0,atol=0)
                loads=torch.zeros(p.shape[1],device=p.device,dtype=torch.int64).scatter_add_(
                    0,actual.flatten(),torch.ones_like(actual.flatten()))
                if not bool((loads==p.shape[0]*self.k//p.shape[1]).all()):
                    return TrialOutcome('correctness_error',error='repair budget exhausted on calibration fixture')
                error=(value.double()-expected).abs().max().item();max_error=max(max_error,error)
                errors[str(index)]=dict(transport_max_abs=error,
                    gradient_max_abs=(actual_gradient.double()-expected_gradient).abs().max().item())
            def call():
                for source,gradient,_,_,_,route_gradient in self.fixtures:
                    token=tuple(c.to_token())
                    value,history=run_forward(source,self.k,self.steps,token)
                    ids=self.hard(value.exp(),c)
                    _,weights,inverse=self.metadata(value,ids,c)
                    metadata_gradient=(self.metadata_backward(route_gradient,weights,ids,inverse,c)
                                       if self.phase=='train' else None)
                    grad=(run_backward(metadata_gradient,history,self.k,self.steps,token)
                          if self.phase=='train' else None)
                return value,history,ids,grad
            stage='capture'
            stream=torch.cuda.Stream(device=self.fixtures[0][0].device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self.policy.warmup):call()
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):output=call()
            torch.cuda.current_stream().wait_stream(stream)
            compile_ms=(time.monotonic()-started)*1000
            from .preparation_runtime import _warm_measurement_graph
            graph_warmup=_warm_measurement_graph(graph)
            stage='timing'
            samples,metadata=measure_cuda_batches(graph.replay,samples=self.policy.samples,
                calls_per_sample=self.policy.calls_per_sample,device=self.fixtures[0][0].device,min_batch_ms=3.)
            samples=[v/len(self.fixtures) for v in samples]
            return TrialOutcome('ok',median_ms=statistics.median(samples),timings_ms=samples,
                compile_ms=compile_ms,max_abs_error=max_error,
                metadata=dict(validation=errors,graph_warmup=graph_warmup,**metadata,**runtime_timing_metadata(samples)))
        except Exception as error:
            raise_if_fatal_device_context_error(error)
            status=('correctness_error' if stage=='correctness' else 'runtime_error' if stage in ('capture','timing') else 'compile_error')
            return TrialOutcome(status,error=f'{type(error).__name__}: {error}'[:4000],
                                compile_ms=(time.monotonic()-started)*1000)
        finally:
            if graph is not None:graph.reset()

    def close(self):
        from .._routing_hard import _plans
        from .._managed import submission_lock
        device=self.fixtures[0][0].device
        with submission_lock(device.index),_plans._lock:
            for key in tuple(_plans._values):
                if key[2]==self.owner_id:del _plans[key]


def tune_routing(scores,spec,*,phase='train',policy=None,progress=None,runtime_fallback=False):
    import torch
    from .preparation_runtime import _preparation_device_profile
    from .winners import save_runtime_winner
    from .orchestrator import NoEligibleTimingError
    if phase not in ('forward','train'):raise ValueError('phase must be forward/train')
    with torch.cuda.device(scores.device):
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Tune before manual CUDA graph capture')
        key,root,policy=cache_contract(scores,spec,phase,policy)
        started=time.monotonic()
        evaluator=RoutingEvaluator(scores,spec,phase,policy)
        w=spec['workload']
        adapter=make_routing_adapter(*scores.shape,w['top_k'],evaluator.evaluate,
            initial=RoutingKernelConfig.from_dict(spec['initial']),phase=phase,
            device=_preparation_device_profile(scores.device),sinkhorn_iterations=w['sinkhorn_iterations'],
            dual_iterations=w['dual_iterations'],repair_rounds=w['repair_rounds'],
            tags=dict(workload=w,source=_source_revision(),measurement=MEASUREMENT_CONTRACT,
                      initial=spec['initial'],effort=policy.evidence_contract()))
        residual_root=root/'routing_effort'
        store=ResidualTuningStore(effort_residual_root(residual_root/adapter.context.identifier,policy),transfer_root=residual_root)
        try:
            try:
                result=make_flat_effort_autotuner(adapter,store,policy,progress=progress,started_at=started).tune()
            except NoEligibleTimingError as error:
                if not runtime_fallback or error.fallback_config is None:raise
                result=ComposableTuningResult(error.fallback_config,math.inf,'timing-fallback',
                    adapter.context.identifier,0,time.monotonic()-started,{},str(store.path),
                    tuning_evidence=dict(timing_fallback=True))
            if result.tuning_evidence is None:
                result.tuning_evidence=dict(schema=1,contract=policy.evidence_contract(),
                    attempted_trials=len(list(store.records(adapter.context))))
            save_runtime_winner(key,result.config.to_dict(),root=root,config_id=stable_id(result.config.to_dict()),
                median_ms=result.median_ms,tuning_evidence=result.tuning_evidence,
                metadata=dict(objective=MEASUREMENT_CONTRACT,phase=phase))
            return result
        finally:evaluator.close()


def main():
    """Portable bounded collection: python -m rtx.autotune.routing --help."""
    import argparse
    from dataclasses import asdict
    import json
    import torch
    from ..routing import BalancedRouter
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',type=int,required=True)
    parser.add_argument('--d-model',type=int,default=384)
    parser.add_argument('--experts',type=int,default=128)
    parser.add_argument('--top-k',type=int,default=8)
    parser.add_argument('--router-dim',type=int,default=64)
    parser.add_argument('--effort',choices=('quick','balanced','thorough','extensive'),default='balanced')
    parser.add_argument('--trials',type=int)
    parser.add_argument('--seconds',type=float)
    parser.add_argument('--phase',choices=('forward','train'),default='train')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    options=dict(search_schedule='progressive',candidate_mix='mixed',cache_behavior='refine')
    if args.trials is not None:options['max_trials']=args.trials
    if args.seconds is not None:options['max_seconds']=args.seconds
    policy=AutotunePolicy.preset(args.effort,**options)
    torch.manual_seed(policy.seed)
    router=BalancedRouter(args.d_model,args.experts,top_k=args.top_k,router_dim=args.router_dim,
                           device=args.device,autotune='off')
    with torch.no_grad():router.projection_weight.normal_(std=.02)
    x=torch.randn(args.tokens,args.d_model,device=args.device,dtype=torch.bfloat16)
    from .orchestrator import NoEligibleTimingError
    try:
        result=router.autotune_(x,tuning_policy=policy,output_dir=args.output,phase=args.phase,progress=print)
    except NoEligibleTimingError as error:
        import uuid
        destination=Path(args.output)/f'timing-ineligible-{uuid.uuid4().hex}.json'
        destination.write_text(json.dumps(dict(status='timing_ineligible',
            correctness_checked_seed=error.fallback_config is not None,
            residuals_retained=True,persistent_winner=False),indent=2)+'\n')
        print(f'NO_ELIGIBLE_TIMING {destination}; residuals retained, no latency winner promoted',flush=True)
        raise SystemExit(2)
    destination=Path(args.output)/f'result-{result.session_id}.json'
    destination.write_text(json.dumps(asdict(result),indent=2,allow_nan=False)+'\n')
    print(f'DONE {destination}',flush=True)


if __name__=='__main__':main()
