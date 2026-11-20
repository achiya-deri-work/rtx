"""Shared-UCB packed routing search, with independent segmented derivatives."""
from dataclasses import replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import statistics
import time

from .core import DiscreteKernelAdapter,KernelContext,stable_id,ComposableTuningResult
from .effort import AutotunePolicy,make_effort_autotuner,effort_residual_root
from .outcomes import TrialOutcome,raise_if_fatal_device_context_error
from .routing import make_routing_adapter,structural_seeds
from .store import ResidualTuningStore
from ..routing import PACKED_SINKHORN_BACKENDS,RoutingKernelConfig

KERNEL_REVISION=7
OPENING_REVISION=1  # Search order only; portable semantic priors still use revision 7.
CONTRACT=dict(revision=7,independent_pools=True,expert_segment_alignment=32,
              objective='packed_router_forward_and_score_adjoint',timing='cuda_graph',oracle='segmented_fp64',
              per_pool_persistent=True,opening_revision=OPENING_REVISION)


@lru_cache(maxsize=1)
def source_revision():
    root=Path(__file__).resolve().parents[1]
    files=('routing.py','_routing_varlen.py','_routing_varlen_autotune.py','_routing_graph.py',
           '_routing_autotune.py','_routing_reference.py','_moe_selection.py',
           'kernels/routing_varlen.py','kernels/routing_selection.py','kernels/routing_quota_cached.py',
           'kernels/routing_sinkhorn.py','kernels/routing_metadata.py','autotune/varlen_routing.py')
    digest=hashlib.sha256()
    for name in files:digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def contract(scores,cu,spec,training,strict,policy=None,lengths=None):
    from .legacy import default_cache_dir
    from .winners import runtime_winner_key
    from ..kernels.mxfp8 import MXFP8Problem
    if policy is None:
        values=dict(spec['policy']);values['predictor_heads']=tuple(values['predictor_heads'])
        policy=AutotunePolicy(**values)
    policy=policy.for_pass(training)
    # Only the cold selector/explicit search reads boundaries. Hot dispatch and
    # graph replay bind T/B geometry and never fetch length values to the host.
    if lengths is None:
        values=cu.detach().cpu().tolist()
        if values[0]!=0 or values[-1]!=scores.shape[0] or any(z<a for a,z in zip(values,values[1:])):
            raise ValueError('Invalid cu_seqlens')
        lengths=[z-a for a,z in zip(values,values[1:])]
    workload={**spec['workload'],'total_tokens':scores.shape[0],'sequences':len(lengths),
              'lengths':lengths,'strict_padding':strict,'phase':'train' if training else 'forward'}
    root=Path(spec['cache_dir']) if spec['cache_dir'] else default_cache_dir()
    key=runtime_winner_key('balanced_varlen_routing',MXFP8Problem(scores.shape[0],scores.shape[1],workload['top_k']),
        device=scores.device,kernel_revision=KERNEL_REVISION,
        variant=stable_id(dict(source=source_revision(),workload=workload,initial=spec['initial'],
                               measurement=CONTRACT,effort=policy.evidence_contract())))
    return key,root,policy,lengths


def make_adapter(t,b,e,k,steps,initial,evaluator,*,training=True,device=None,tags=None,lengths=None,
                 dual_iterations=5,repair_rounds=64,strict=False):
    from .._routing_varlen import layout,persistent_hybrid_rows,persistent_pool_rows
    n=layout(t,b,e,k,steps,tuple(initial.to_token()))[4]
    hybrid=persistent_hybrid_rows(t,e,k,pools=b)
    # Cold lengths only affect which policies are worth measuring, never
    # which device pools are writable by a runtime kernel.
    hybrid_short=(sum(0<length<=hybrid for length in lengths) if lengths is not None else None)
    hybrid_all_short=(all(length<=hybrid for length in lengths) if lengths is not None else t<=hybrid*b)
    small=bool(persistent_pool_rows(t,e,k,pools=b) or (hybrid and (
        hybrid_short if hybrid_short is not None else hybrid_all_short)))
    old=make_routing_adapter(n,e,k,evaluator,initial=initial,phase='train' if training else 'forward')
    def normalize(c):
        packed=c.segmented_quotas and n*k>=c.quota_threshold
        mode=PACKED_SINKHORN_BACKENDS.index(c.packed_sinkhorn_backend) if small else 0
        if not training:mode&=1
        persistent_f,persistent_b=bool(mode&1),bool(mode&2)
        width=c.packed_persistent_threads if mode else 0
        direct_f=persistent_f and not hybrid
        direct_b=persistent_b and not hybrid
        row_backend='cta' if direct_f and (direct_b or not training) else c.packed_row_backend
        warp=row_backend=='warp'
        native_columns=c.qb_backend!='torch' and k<e
        return replace(c,segmented_quotas=packed,quota_threshold=0,
            flat_column_shards_forward=0,flat_column_shards_backward=0,flat_row_backend='cta',
            quota_cache_items=c.quota_cache_items if packed else 0,
            quota_early_exit=c.quota_early_exit if packed else False,
            qb_threads=128 if c.qb_backend=='torch' else c.qb_threads,
            packed_column_cache_items=c.packed_column_cache_items if native_columns else 0,
            packed_qb_column_backend=c.packed_qb_column_backend if native_columns else 'row_major',
            packed_sinkhorn_backend=PACKED_SINKHORN_BACKENDS[mode],packed_persistent_threads=width,
            packed_row_backend=row_backend,
            packed_column_shards_forward=0 if direct_f else c.packed_column_shards_forward,
            packed_column_shards_backward=c.packed_column_shards_backward if training and not direct_b else 0,
            forward=replace(c.forward,threads=128 if direct_f and width else c.forward.threads,
                            row_groups=1 if warp or direct_f else c.forward.row_groups,
                            column_groups=1 if direct_f else c.forward.column_groups),
            backward=replace(c.backward,threads=128 if direct_b and width else c.backward.threads,
                             row_groups=1 if warp or direct_b else c.backward.row_groups,
                             column_groups=1 if direct_b else c.backward.column_groups) if training else initial.backward)
    def update(c,name,value):
        if '.' in name:
            group,field=name.split('.');return normalize(replace(c,**{group:replace(getattr(c,group),**{field:value})}))
        return normalize(replace(c,**{name:value}))
    def reject(c):
        c=normalize(c);reason=old.rejection(c)
        if reason:return reason
        if (c.quota_cache_items or c.quota_early_exit) and b*e>1024:
            return 'implementation_rejected','Cached packed quota groups require B*E<=1024'
        if b*e*max(c.packed_column_shards_forward,c.packed_column_shards_backward)>=2**31:
            return 'implementation_rejected','Packed column scratch exceeds signed native index range'
    context=KernelContext('balanced_varlen_routing',KERNEL_REVISION,
        dict(total_tokens=t,sequences=b,experts=e,top_k=k,scheduled_bound=n,
             sinkhorn_iterations=steps,dual_iterations=dual_iterations,repair_rounds=repair_rounds,
             strict_padding=strict,phase='train' if training else 'forward'),
        device=device or {},tags=tags or {})
    serialize=lambda c:normalize(c).to_dict()
    shards=(0,4,8,16,32)
    axes={**old.axes,'packed_column_cache_items':(0,4,8,16,32,64),'packed_row_backend':('cta','warp'),
          'packed_column_shards_forward':shards,'packed_qb_column_backend':('row_major','expert_keys')}
    for name in ('flat_column_shards_forward','flat_column_shards_backward','flat_row_backend'):
        axes.pop(name,None)
    if training:axes['packed_column_shards_backward']=shards
    if small:
        axes['packed_sinkhorn_backend']=PACKED_SINKHORN_BACKENDS if training else PACKED_SINKHORN_BACKENDS[:2]
        axes['packed_persistent_threads']=(0,512,1024)
    def features(c):
        c=normalize(c);warp=c.packed_row_backend=='warp'
        f,bwd=c.packed_column_shards_forward,c.packed_column_shards_backward
        mode=PACKED_SINKHORN_BACKENDS.index(c.packed_sinkhorn_backend)
        staged_f=not mode&1 or (hybrid and not hybrid_all_short)
        staged_b=not mode&2 or (hybrid and not hybrid_all_short)
        persistent_launches=1+int(bool(hybrid))  # pool kernel plus device predicate
        def column_ctas(groups,shards):
            return b*shards*math.ceil(e/(4*groups)) if shards else math.ceil(b*e/groups)
        return dict(elements=n*e,sequences=b,
            forward_row_ctas=(b if mode&1 else 0)+(math.ceil(n/(c.forward.threads//32 if warp else c.forward.row_groups)) if staged_f else 0),
            forward_column_ctas=column_ctas(c.forward.column_groups,f) if staged_f else b,
            backward_row_ctas=(b if mode&2 else 0)+(math.ceil(n/(c.backward.threads//32 if warp else c.backward.row_groups)) if staged_b else 0),
            backward_column_ctas=column_ctas(c.backward.column_groups,bwd) if staged_b else b,
            forward_column_partial_bytes=b*e*f*4,
            backward_column_partial_bytes=b*e*bwd*4,
            column_launches=steps*((1+int(f>0) if staged_f else 0)+
                                  (1+int(bwd>0) if training and staged_b else 0)),
            transport_launches=(persistent_launches if mode&1 else 0)+(2+steps*(2+int(f>0)) if staged_f else 0)+
                (((persistent_launches if mode&2 else 0)+(1+steps*(2+int(bwd>0)) if staged_b else 0)) if training else 0),
            hybrid_pool_cap=hybrid if mode else 0,
            cold_short_pools=0 if hybrid_short is None else hybrid_short,
            forward_persistent_cta_threads=(c.packed_persistent_threads or c.forward.threads) if mode&1 else 0,
            backward_persistent_cta_threads=(c.packed_persistent_threads or c.backward.threads) if mode&2 else 0,
            saved_history_bytes=2*steps*n*e*4,
            qb_ordered_key_bytes=n*e*4 if c.packed_qb_column_backend=='expert_keys' else 0,
            qb_column_cache_capacity=c.qb_threads*c.packed_column_cache_items)
    return DiscreteKernelAdapter(context,normalize(initial),axes,lambda c:stable_id(serialize(c)),serialize,
        lambda c:normalize(RoutingKernelConfig.from_dict(c)),update,evaluator,reject,features)


def packed_structural_seeds(initial,*,max_scheduled_length=None,persistent_eligible=False,training=True,mixed=False):
    """Cover row ownership AND column-grid parallelism before local search.

    B*E column vectors can otherwise collapse to far fewer CTAs than an RTX
    has SMs. These are measured alternatives, not a forced SKU heuristic;
    grouped columns and both original algorithms remain in the search space.
    """
    if persistent_eligible:
        # Same QB/repair and metadata as the initial candidate. Compare
        # direction mixing as an implementation choice, not a policy change.
        if training:
            yield replace(initial,packed_sinkhorn_backend='persistent',packed_persistent_threads=1024)
            if not mixed:
                yield replace(initial,packed_sinkhorn_backend='persistent_backward',packed_persistent_threads=1024)
                yield replace(initial,packed_sinkhorn_backend='persistent_forward',packed_persistent_threads=512)
                yield replace(initial,packed_sinkhorn_backend='persistent',packed_persistent_threads=512)
        else:
            yield replace(initial,packed_sinkhorn_backend='persistent_forward',packed_persistent_threads=1024)
            if not mixed:yield replace(initial,packed_sinkhorn_backend='persistent_forward',packed_persistent_threads=512)
    native=replace(initial,qb_backend='native_column',metadata_backend='cute',
                   segmented_quotas=True,quota_threshold=0,
                   packed_row_backend='cta',packed_column_cache_items=0,
                   packed_column_shards_forward=0,packed_column_shards_backward=0,
                   packed_qb_column_backend='row_major',packed_sinkhorn_backend='staged',packed_persistent_threads=0)
    warp=replace(native,packed_row_backend='warp')
    parallel=replace(warp,forward=replace(warp.forward,column_groups=1),
                     backward=replace(warp.backward,column_groups=1))
    wide=replace(warp,packed_column_shards_forward=32,packed_column_shards_backward=32,
                  forward=replace(warp.forward,column_groups=8,threads=256),
                  backward=replace(warp.backward,column_groups=8,threads=256))
    # Compare both column families at the same radix capacity. Size it from
    # cold boundary data, not a GPU-specific win.
    threads=256
    capacity=next((i for i in (4,8,16,32,64)
                   if max_scheduled_length is not None and max_scheduled_length<=threads*i),0)
    opening=replace(wide,qb_backend='native_warp',qb_threads=threads,packed_column_cache_items=capacity)
    yield opening
    yield replace(opening,packed_qb_column_backend='expert_keys')
    yield parallel
    yield native
    yield replace(warp,packed_column_shards_forward=16,packed_column_shards_backward=16)
    yield wide
    yield replace(wide,packed_qb_column_backend='expert_keys',qb_backend='native_warp',qb_threads=256)
    yield replace(wide,packed_qb_column_backend='expert_keys',qb_backend='native_warp',qb_threads=256,
                  packed_column_cache_items=16)
    yield replace(wide,packed_qb_column_backend='expert_keys',qb_backend='native_warp',qb_threads=256,
                  packed_column_cache_items=64)
    yield replace(warp,packed_column_shards_forward=8,packed_column_shards_backward=8,
                  forward=replace(warp.forward,column_groups=2),backward=replace(warp.backward,column_groups=2))
    yield replace(warp,packed_column_shards_backward=16)
    yield replace(parallel,qb_backend='native_warp')
    yield replace(parallel,packed_column_cache_items=16)
    yield replace(parallel,packed_column_cache_items=32)
    yield replace(native,qb_backend='native_warp')
    yield warp
    yield replace(warp,packed_column_cache_items=32)
    yield replace(native,packed_column_cache_items=32)
    yield replace(native,quota_cache_items=8,quota_early_exit=True)
    yield from structural_seeds(initial)


def _staged_sharded(config, training):
    return (config.packed_sinkhorn_backend=='staged' and config.packed_row_backend=='warp'
            and config.packed_column_shards_forward>0
            and (not training or config.packed_column_shards_backward>0)
            and config.qb_backend=='native_warp' and config.packed_qb_column_backend=='row_major')


def make_packed_effort_autotuner(adapter, store, policy, *, lengths, training, progress=None, started_at=None):
    """Put a contrasting, already-budgeted packed plan before implicit local trials."""
    from .._routing_varlen import persistent_hybrid_rows,persistent_pool_rows

    w=adapter.context.workload
    t,b,e,k=(w[name] for name in ('total_tokens','sequences','experts','top_k'))
    multiple=e//math.gcd(e,k)
    longest=((max(lengths)+multiple-1)//multiple)*multiple
    hybrid=persistent_hybrid_rows(t,e,k,pools=b)
    candidates=tuple(packed_structural_seeds(adapter.initial_config,max_scheduled_length=longest,
        persistent_eligible='packed_sinkhorn_backend' in adapter.axes,training=training,
        mixed=bool(hybrid and longest>hybrid)))
    tuner=make_effort_autotuner(adapter,store,policy,progress=progress,candidates=candidates,started_at=started_at)

    # A one-pool problem or a pack whose every sequence fits the persistent
    # backend keeps its existing order. Leave explicit admission policies
    # alone, even when a requested prior happens not to match this context.
    if (sum(length>0 for length in lengths)<2
            or longest<=max(hybrid,persistent_pool_rows(t,e,k,pools=b))
            or policy.prior_artifact or policy.predictor_artifact or policy.retrieval_trials is not None
            or policy.initial_neighborhood_trials or policy.baseline_trials or policy.exploration_trials!=1
            or _staged_sharded(adapter.initial_config,training)):
        return tuner

    # The shared progressive constructor places its default local neighbors
    # ahead of the structural list. Move a legal *actual coverage proposal*,
    # keeping the rest in order, so no trial or exploration reserve is added.
    coverage=tuner.coverage_proposals
    for index,proposal in enumerate(coverage):
        if (proposal.strategy=='structural' and _staged_sharded(proposal.config,training)
                and adapter.rejection(proposal.config) is None):
            if index:
                tuner.coverage_proposals=(proposal,*coverage[:index],*coverage[index+1:])
                if tuner.checkpoint_key:
                    tuner.checkpoint_key=stable_id(dict(base=tuner.checkpoint_key,revision=OPENING_REVISION,
                        coverage=[adapter.config_id(p.config) for p in tuner.coverage_proposals]))
            break
    return tuner


class Evaluator:
    def __init__(self,scores,cu,spec,policy,lengths,training,strict):
        import torch
        self.torch=torch;self.cu=cu;self.spec=spec;self.policy=policy
        self.lengths=lengths;self.training=training;self.strict=strict;self.owner=id(self)
        self.source=scores.detach().requires_grad_();self.k=spec['workload']['top_k']
        self.steps=spec['workload']['sinkhorn_iterations']
        self.generator=torch.Generator(device=scores.device).manual_seed(policy.seed)
        synthetic=torch.nn.functional.logsigmoid(torch.randn(scores.shape,device=scores.device,generator=self.generator)*1.5)
        self.sources=(self.source,synthetic.requires_grad_())
        self.upstream=None

    def route(self,c):
        from .._routing_varlen import routing
        w=self.spec['workload']
        return routing(self.source,self.cu,top_k=self.k,steps=self.steps,qb=w['dual_iterations'],
                       rounds=w['repair_rounds'],config=c,strict=self.strict,owner=self.owner)
    def call(self,c):
        result=self.route(c)
        gradient=None
        if self.training:
            gradient=self.torch.autograd.grad(result.route_weights,self.source,self.upstream)[0]
        return result,gradient

    def evaluate(self,c):
        records=[]
        for source in self.sources:
            self.source=source
            result=self._evaluate_fixture(c)
            if result.status!='ok':return result
            records.append(result)
        samples=[sum(values)/len(values) for values in zip(*(r.timings_ms for r in records))]
        return TrialOutcome('ok',median_ms=statistics.median(samples),timings_ms=samples,
            compile_ms=sum(r.compile_ms for r in records),metadata=dict(lengths=self.lengths,
                calibration='caller_and_sigmoid_normal',fixtures=[r.metadata for r in records]))

    def _evaluate_fixture(self,c):
        torch=self.torch;started=time.monotonic();graph=None;stage='compile'
        try:
            from .._routing_varlen import layout
            key=layout(self.source.shape[0],len(self.lengths),self.source.shape[1],self.k,self.steps,tuple(c.to_token()),self.strict)
            if self.upstream is None:
                self.upstream=torch.randn((self.source.shape[1],key[5]),device=self.source.device,generator=self.generator)
            result,gradient=self.call(c);stage='correctness'
            if key[17]&1:
                # Holding the detached hard policy fixed is stronger than
                # just checking the FP64 Jacobian on whichever route won.
                # Warp rows + 32 reduction lanes/column use the persistent
                # forward's FP32 reduction order without persistent state.
                staged=replace(c,packed_sinkhorn_backend='staged',packed_persistent_threads=0)
                match=replace(staged,packed_row_backend='warp',packed_column_shards_forward=0,
                              forward=replace(c.forward,column_groups=c.forward.threads//32))
                multiple=self.source.shape[1]//math.gcd(self.source.shape[1],self.k)
                hybrid=key[18]
                with torch.no_grad():
                    reference_route=self.route(match)
                    long_route=self.route(staged) if hybrid and any(v>hybrid for v in self.lengths) else None
                if long_route is None:
                    torch.testing.assert_close(result.expert_ids,reference_route.expert_ids,rtol=0,atol=0)
                    torch.testing.assert_close(result.inverse,reference_route.inverse,rtol=0,atol=0)
                else:
                    # Preserve the configured staged geometry for long pools.
                    # Their exact hard metadata must not be compared against a
                    # numerically different short-pool reduction tree.
                    begin=offset=0
                    for length in self.lengths:
                        scheduled=(length+multiple-1)//multiple*multiple
                        cap=(scheduled*self.k//self.source.shape[1]+31)//32*32
                        ref=long_route if scheduled>hybrid else reference_route
                        torch.testing.assert_close(result.expert_ids[:,offset:offset+cap],ref.expert_ids[:,offset:offset+cap],rtol=0,atol=0)
                        torch.testing.assert_close(result.inverse[begin:begin+length],ref.inverse[begin:begin+length],rtol=0,atol=0)
                        begin+=length;offset+=cap
                    torch.testing.assert_close(result.expert_ids[:,offset:],long_route.expert_ids[:,offset:],rtol=0,atol=0)
                del reference_route,long_route
            # Independent direct normalizations, with the actual detached hard
            # choices. Also test K distinct experts and token/slot isolation.
            oracle=self.source.detach().double().requires_grad_();pieces=[];start=0
            # Reject malformed native metadata before indexing with it. An
            # oracle gather must report a bad candidate, not poison the CUDA
            # context and abort every remaining tuning trial.
            if not bool(((result.inverse>=0)&(result.inverse<result.expert_ids.numel())).all()):
                raise AssertionError('Packed inverse outside expert storage')
            if not bool(((result.expert_ids>=0)&(result.expert_ids<self.source.shape[0])).all()):
                raise AssertionError('Packed token ID outside input storage')
            chosen=(result.inverse//key[5]).detach()
            torch.testing.assert_close(result.expert_ids.flatten()[result.inverse],
                torch.arange(self.source.shape[0],device=self.source.device)[:,None].expand_as(result.inverse),rtol=0,atol=0)
            if bool((chosen.sort(1).values.diff(dim=1)==0).any()):raise AssertionError('Repeated expert in one token')
            physical=0
            for length in self.lengths:
                if not length:continue
                m=self.source.shape[1]//math.gcd(self.source.shape[1],self.k)
                ns=(length+m-1)//m*m;capacity=ns*self.k//self.source.shape[1]
                local=result.inverse[start:start+length]%key[5]
                if not bool(((local>=physical)&(local<physical+capacity)).all()):raise AssertionError('Cross-sequence slot')
                value=torch.nn.functional.pad(oracle[start:start+length],(0,0,0,ns-length),value=-math.log(2))
                for _ in range(self.steps):
                    value=value-torch.logsumexp(value,1,keepdim=True)+math.log(self.k)
                    value=value-torch.logsumexp(value,0,keepdim=True)+math.log(capacity)
                pieces.append(value[:length].gather(1,chosen[start:start+length]).softmax(1))
                physical+=(capacity+31)//32*32;start+=length
            expected=torch.cat(pieces)
            actual=result.route_weights.flatten()[result.inverse]
            torch.testing.assert_close(actual.double(),expected,rtol=5e-5,atol=1e-6)
            if self.training:
                expected_grad=torch.autograd.grad(expected,oracle,self.upstream.flatten()[result.inverse].double())[0]
                torch.testing.assert_close(gradient.double(),expected_grad,rtol=5e-4,atol=8e-6)
            # Release every view retaining the qualification graph before
            # warming this leaf on a different capture stream. Keeping actual
            # or result alive pins its default-stream AccumulateGrad edge.
            del actual,result,gradient
            stage='capture';stream=torch.cuda.Stream(device=self.source.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(max(1,self.policy.warmup)):self.call(c)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):outputs=self.call(c)
            torch.cuda.current_stream().wait_stream(stream)
            compile_ms=(time.monotonic()-started)*1000
            from .preparation_runtime import _warm_measurement_graph
            from .measurement import measure_cuda_batches
            _warm_measurement_graph(graph);stage='timing'
            samples,metadata=measure_cuda_batches(graph.replay,samples=self.policy.samples,
                calls_per_sample=self.policy.calls_per_sample,device=self.source.device,min_batch_ms=3.)
            torch.cuda.synchronize(self.source.device)
            if not bool(outputs[0].route_weights.isfinite().all()):raise AssertionError('Nonfinite replay weights')
            if self.training:torch.testing.assert_close(outputs[1].double(),expected_grad,rtol=5e-4,atol=8e-6)
            return TrialOutcome('ok',median_ms=statistics.median(samples),timings_ms=samples,
                                compile_ms=compile_ms,metadata=dict(lengths=self.lengths,**metadata))
        except Exception as error:
            raise_if_fatal_device_context_error(error)
            status='correctness_error' if stage=='correctness' else 'runtime_error' if stage in ('capture','timing') else 'compile_error'
            return TrialOutcome(status,error=f'{type(error).__name__}: {error}'[:4000],compile_ms=(time.monotonic()-started)*1000)
        finally:
            if graph is not None:graph.reset()

    def close(self):
        from .._routing_varlen import _PLANS
        with _PLANS._lock:
            for key in tuple(_PLANS._values):
                if key[2]==self.owner:del _PLANS[key]


def tune(scores,cu,spec,*,training=True,strict=False,policy=None,lengths=None,progress=None,runtime_fallback=False):
    from .preparation_runtime import _preparation_device_profile
    from .winners import save_runtime_winner
    from .orchestrator import NoEligibleTimingError
    key,root,policy,lengths=contract(scores,cu,spec,training,strict,policy,lengths)
    evaluator=Evaluator(scores,cu,spec,policy,lengths,training,strict);w=spec['workload']
    adapter=make_adapter(scores.shape[0],len(lengths),scores.shape[1],w['top_k'],w['sinkhorn_iterations'],
        RoutingKernelConfig.from_dict(spec['initial']),evaluator.evaluate,training=training,
        device=_preparation_device_profile(scores.device),lengths=lengths,dual_iterations=w['dual_iterations'],
        repair_rounds=w['repair_rounds'],strict=strict,
        tags=dict(lengths=lengths,source=source_revision(),measurement=CONTRACT,initial=spec['initial'],
                  workload=w,effort=policy.evidence_contract()))
    residual=root/'varlen_routing_effort'
    store=ResidualTuningStore(effort_residual_root(residual/adapter.context.identifier,policy),transfer_root=residual)
    started=time.monotonic()
    try:
        try:
            result=make_packed_effort_autotuner(adapter,store,policy,lengths=lengths,
                training=training,progress=progress,started_at=started).tune()
        except NoEligibleTimingError as error:
            if not runtime_fallback or error.fallback_config is None:raise
            result=ComposableTuningResult(error.fallback_config,math.inf,'timing-fallback',adapter.context.identifier,
                0,time.monotonic()-started,{},str(store.path),tuning_evidence=dict(timing_fallback=True))
        save_runtime_winner(key,result.config.to_dict(),root=root,config_id=adapter.config_id(result.config),
            median_ms=result.median_ms,tuning_evidence=result.tuning_evidence,metadata=dict(lengths=lengths,measurement=CONTRACT))
        return result
    finally:evaluator.close()
