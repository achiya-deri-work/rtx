"""Balanced routing adapter, graph evaluator and durable shared-engine search.

Policy budgets are context, not search axes. The objective includes native
Sinkhorn, hard scheduling and (for training) the saved-state Sinkhorn adjoint.
Unchanged affinity/metadata math is checked in end-to-end deployment benchmarks.
"""
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
from .store import ResidualTuningStore
from ..routing import RoutingKernelConfig,SinkhornKernelConfig

ROUTING_KERNEL_REVISION=1
MEASUREMENT_CONTRACT=dict(revision=1,timing='cuda_graph',calibration='caller_and_sigmoid_normal',
                          oracle='independent_fp64',saved_state='all_normalization_probabilities',
                          objective='native_routing_stages',nrms_floor=1e-12,graph_warmup_ms=200)


@lru_cache(maxsize=1)
def _source_revision():
    root=Path(__file__).resolve().parents[1]
    files=['routing.py','_routing_autotune.py','_routing_hard.py','kernels/routing_sinkhorn.py',
           'kernels/routing_selection.py','_routing_graph.py','autotune/routing.py']
    digest=hashlib.sha256()
    for name in files:digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def rejection(config,n,e,k=1):
    if n*e>=2**31:return 'implementation_rejected','Routing matrix exceeds signed kernel index range'
    packed=config.segmented_quotas and n*k>=config.quota_threshold
    if not packed and n*k>4096:
        return 'implementation_rejected','Direct quota implementation is bounded to 4096 proposals'
    if e>1024:return 'implementation_rejected','Native repair currently supports at most 1024 experts'
    return None


def make_routing_adapter(n,e,k,evaluator,*,initial=None,phase='train',device=None,
                         sinkhorn_iterations=5,dual_iterations=5,repair_rounds=64,tags=None):
    if min(n,e,k)<=0 or k>e or n*k%e:raise ValueError('Invalid scheduled pool geometry')
    if phase not in ('forward','train'):raise ValueError('phase must be forward/train')
    initial=initial or RoutingKernelConfig()
    backward_seed=initial.backward
    def normalize(c):
        packed=c.segmented_quotas and n*k>=c.quota_threshold
        return replace(c,segmented_quotas=packed,quota_threshold=0,
                       qb_threads=128 if c.qb_backend=='torch' else c.qb_threads,
                       backward=backward_seed if phase=='forward' else c.backward)
    initial=normalize(initial)
    axes={'qb_backend':('torch','native_column','native'),'qb_threads':(64,128,256),
          'segmented_quotas':(False,True),'quota_threshold':(0,256,1024,4096,16384),
          'forward.threads':(64,128,256),'forward.row_groups':(1,2,4,8),
          'forward.column_groups':(1,2,4,8)}
    if phase=='train':
        axes.update({'backward.threads':(64,128,256),'backward.row_groups':(1,2,4,8),
                     'backward.column_groups':(1,2,4,8)})
    context=KernelContext('balanced_routing',ROUTING_KERNEL_REVISION,
        dict(m=n,n=e,k=k,phase=phase,sinkhorn_iterations=sinkhorn_iterations,
             dual_iterations=dual_iterations,repair_rounds=repair_rounds),
        device={} if device is None else device,tags={} if tags is None else tags)
    def update(c,name,value):
        if '.' in name:
            group,field=name.split('.')
            return normalize(replace(c,**{group:replace(getattr(c,group),**{field:value})}))
        return normalize(replace(c,**{name:value}))
    serialize=lambda c:normalize(c).to_dict()
    return DiscreteKernelAdapter(context,initial,axes,lambda c:stable_id(serialize(c)),
        serialize,lambda value:normalize(RoutingKernelConfig.from_dict(value)),update,evaluator,
        lambda c:rejection(normalize(c),n,e,k),
        lambda c:dict(elements=n*e,forward_row_ctas=math.ceil(n/c.forward.row_groups),
                      forward_column_ctas=math.ceil(e/c.forward.column_groups),
                      backward_row_ctas=math.ceil(n/c.backward.row_groups),
                      backward_column_ctas=math.ceil(e/c.backward.column_groups),
                      saved_state_bytes=(2*sinkhorn_iterations+1)*n*e*4))


def structural_seeds(initial):
    # Complete structural alternatives, not a shape/SKU-specific winner table.
    for backend in ('native','native_column','torch'):
        for segmented in (False,True):
            yield replace(initial,qb_backend=backend,segmented_quotas=segmented)
    yield replace(initial,forward=SinkhornKernelConfig(128,4,8),backward=SinkhornKernelConfig(128,4,8))
    yield replace(initial,forward=SinkhornKernelConfig(64,4,4),backward=SinkhornKernelConfig(64,4,4))


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


class RoutingEvaluator:
    def __init__(self,scores,spec,phase,policy):
        import torch
        self.torch=torch;self.spec=spec;self.phase=phase;self.policy=policy
        self.owner_id=id(self);self.fixtures=[]
        w=spec['workload'];self.k=w['top_k'];self.steps=w['sinkhorn_iterations']
        self.qb=w['dual_iterations'];self.rounds=w['repair_rounds']
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
            self.fixtures.append((source,gradient,value.detach(),expected_gradient,p))

    def hard(self,p,c):
        from .._routing_hard import hard_schedule
        return hard_schedule(p,self.k,self.qb,self.rounds,c.segmented_quotas,self.owner_id,
                             c.quota_threshold,c.qb_backend,c.qb_threads)

    def evaluate(self,c):
        from ..routing import _run_sinkhorn_forward,_run_sinkhorn_backward
        from .measurement import measure_cuda_batches,runtime_timing_metadata
        torch=self.torch
        stage='compile';started=time.monotonic();graph=None
        try:
            errors={};max_error=0.
            for index,(source,gradient,expected,expected_gradient,p) in enumerate(self.fixtures):
                f,b=c.forward,c.backward
                value,history=_run_sinkhorn_forward(source,self.k,self.steps,f.threads,f.row_groups,f.column_groups)
                actual_gradient=_run_sinkhorn_backward(gradient,history,self.k,self.steps,b.threads,b.row_groups,b.column_groups)
                stage='correctness'
                torch.testing.assert_close(value.double(),expected,rtol=5e-5,atol=5e-6)
                torch.testing.assert_close(actual_gradient.double(),expected_gradient,rtol=5e-4,atol=8e-6)
                baseline=self.hard(p,RoutingKernelConfig())
                actual=self.hard(p,c)
                torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
                loads=torch.zeros(p.shape[1],device=p.device,dtype=torch.int64).scatter_add_(
                    0,actual.flatten(),torch.ones_like(actual.flatten()))
                if not bool((loads==p.shape[0]*self.k//p.shape[1]).all()):
                    return TrialOutcome('correctness_error',error='repair budget exhausted on calibration fixture')
                error=(value.double()-expected).abs().max().item();max_error=max(max_error,error)
                errors[str(index)]=dict(transport_max_abs=error,
                    gradient_max_abs=(actual_gradient.double()-expected_gradient).abs().max().item())
            def call():
                for source,gradient,_,_,_ in self.fixtures:
                    f,b=c.forward,c.backward
                    value,history=_run_sinkhorn_forward(source,self.k,self.steps,f.threads,f.row_groups,f.column_groups)
                    ids=self.hard(value.exp(),c)
                    grad=(_run_sinkhorn_backward(gradient,history,self.k,self.steps,b.threads,b.row_groups,b.column_groups)
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
                result=make_effort_autotuner(adapter,store,policy,progress=progress,
                    candidates=list(structural_seeds(adapter.initial_config)),started_at=started).tune()
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
