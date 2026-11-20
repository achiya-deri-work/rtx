"""Bounded compiler-affinity qualification before detached routing search."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import logging
import math
from pathlib import Path
import statistics
import time

import torch

from .._routing_affinity import LAYOUTS, ROW, SCALAR, affinity, signature
from .core import DiscreteKernelAdapter, KernelContext, Proposal, TuningBudget, stable_id
from .effort import AutotunePolicy
from .measurement import measure_cuda_batches, runtime_timing_metadata
from .orchestrator import AutotuneOrchestrator, ConfirmationPolicy, NoEligibleTimingError, SequentialScheduler
from .outcomes import TrialOutcome, raise_if_fatal_device_context_error
from .sampling import StructuralSearch
from .store import ResidualTuningStore


FAMILY='balanced_router_affinity'
REVISION=1
MEASUREMENT=dict(revision=1, objective='affinity_projection_and_all_requested_gradients',
    compiler='inductor_aot_min_cut', timing='cuda_graph',
    correctness='bitwise_scalar_affinity_and_fp32_adjoint_agreement_nonzero_probe')
_LOG=logging.getLogger('rtx.routing.affinity')


@lru_cache(maxsize=1)
def source_revision():
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for name in ('routing.py','_routing_affinity.py','_routing_reference.py','autotune/routing_affinity.py'):
        digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def _policy(spec):
    values=dict(spec['policy']);values['predictor_heads']=tuple(values['predictor_heads'])
    user=AutotunePolicy(**values).for_pass(True)
    # This is a separate two-point context, not an affinity coordinate added
    # to the detached-score transport study. No extra search arms or priors.
    return AutotunePolicy(max_trials=min(2,user.max_trials),max_seconds=user.max_seconds,
        seed=user.seed,warmup=user.warmup,samples=user.samples,calls_per_sample=user.calls_per_sample,
        exploration_trials=0,confirmation_repeats=0,cache_behavior=user.cache_behavior,
        finalist_rounds=max(2,user.finalist_rounds),
        finalist_margin=max(.02,user.finalist_margin if user.finalist_rounds else 0.),
        finalist_reserve_fraction=user.finalist_reserve_fraction if user.finalist_rounds else .10)


def _decode(payload):
    value=payload['layout']
    if value not in LAYOUTS:raise ValueError('Unknown affinity layout')
    return value


def _encode(layout):return {'layout':layout}


def _adapter(context,evaluate):
    return DiscreteKernelAdapter(context,SCALAR,{'layout':LAYOUTS},lambda c:stable_id(_encode(c)),
        _encode,_decode,lambda c,name,value:_decode({'layout':value}),evaluate,lambda c:None)


def _contract(inputs,spec,mask,precision,policy):
    from .legacy import DeviceFingerprint, default_cache_dir
    from .preparation_runtime import _preparation_device_profile
    from .winners import runtime_winner_key
    from ..kernels.mxfp8 import MXFP8Problem
    x,p,e,_,_=inputs
    fingerprint=DeviceFingerprint.current(x.device)
    workload=dict(shapes_strides_dtypes=signature(inputs),grad_mask=mask,phase='train',
        tokens=x.shape[0],d_model=x.shape[1],experts=e.shape[0],router_dim=p.shape[0],
        autocast=False,matmul_precision=precision,rms_epsilon=spec['workload']['rms_epsilon'])
    tags=dict(source=source_revision(),measurement=MEASUREMENT,effort=policy.evidence_contract(),
              confirmation=dict(rounds=policy.finalist_rounds,margin=policy.finalist_margin))
    context=KernelContext(FAMILY,REVISION,workload,
        device={**fingerprint.as_dict(),**_preparation_device_profile(x.device)},tags=tags)
    key=runtime_winner_key(FAMILY,MXFP8Problem(x.shape[0],e.shape[0],p.shape[0]),
        fingerprint=fingerprint,variant=context.identifier,kernel_revision=REVISION)
    root=Path(spec['cache_dir']) if spec['cache_dir'] is not None else default_cache_dir()
    return key,root,context


def _compile(layout,epsilon):
    # Scoped AOT/Inductor without a long-lived Dynamo recompile budget or
    # compiled router/operator recursion. Only the pure affinity is traced.
    from functorch.compile import make_boxed_func, min_cut_rematerialization_partition
    from torch._functorch.aot_autograd import aot_function
    from torch._inductor.decomposition import select_decomp_table

    def program(x,p,e,s,b):return affinity((x,p,e,s,b),epsilon,layout)
    def lower(graph,args):
        return make_boxed_func(torch._inductor.compile(graph,list(args),options={'triton.cudagraphs':False}))
    return aot_function(program,fw_compiler=lower,bw_compiler=lower,
        partition_fn=min_cut_rematerialization_partition,decompositions=select_decomp_table(),enable_log=False)


class _Evaluator:
    def __init__(self,inputs,mask,epsilon,policy,deadline):
        self.inputs=tuple(t.detach().requires_grad_(bool(mask&(1<<i))) for i,t in enumerate(inputs))
        self.epsilon,self.policy,self.deadline=epsilon,policy,deadline
        self.device=inputs[0].device
        rng=torch.Generator(device=self.device).manual_seed(policy.seed)
        # Protect against a zero-initialized projection making two different
        # compiled forwards accidentally appear exact on the caller alone.
        x,p,*other=self.inputs
        with torch.no_grad():
            px=torch.empty_like(x).normal_(generator=rng).requires_grad_(x.requires_grad)
            pp=torch.empty_like(p).normal_(std=1./math.sqrt(p.shape[1]),generator=rng).requires_grad_(p.requires_grad)
        self.probe=(px,pp,*other)
        self.probe_same_layout=signature(self.inputs)==signature(self.probe)
        self.dy=torch.randn((x.shape[0],inputs[2].shape[0]),device=self.device,generator=rng)*.1
        self.graphs={}
        self.reference=None

    def _measure_values(self,program,inputs):
        value=program(*inputs)
        grads=torch.autograd.grad(value,tuple(t for t in inputs if t.requires_grad),self.dy)
        return value,grads

    def _qualify(self,program,layout):
        results=[];worst=0.
        for index,fixture in enumerate((self.inputs,self.probe)):
            if time.monotonic()>=self.deadline:raise TimeoutError('Affinity qualification exhausted the search deadline')
            # AOT callables have no Dynamo stride guards. empty_like can
            # legally densify a gapped/overlapping caller view.
            check=program if index==0 or self.probe_same_layout else _compile(layout,self.epsilon)
            value,grads=self._measure_values(check,fixture)
            if not all(bool(t.isfinite().all()) for t in (value,*grads)):
                raise AssertionError('Nonfinite affinity or gradient')
            if layout==ROW:
                assert self.reference is not None
                expected,reference_grads=self.reference[index]
                if value.dtype!=expected.dtype or not torch.equal(value,expected):
                    raise AssertionError('Affinity layout changed scores used by hard routing')
                for actual,expected_grad in zip(grads,reference_grads):
                    torch.testing.assert_close(actual,expected_grad,rtol=2e-4,atol=2e-5)
                    worst=max(worst,float((actual.float()-expected_grad.float()).abs().max()))
            results.append((value.detach(),tuple(t.detach() for t in grads)))
        if layout==SCALAR:self.reference=tuple(results)
        return worst

    def _capture(self,layout):
        program=_compile(layout,self.epsilon)
        max_error=self._qualify(program,layout)
        if time.monotonic()>=self.deadline:raise TimeoutError('Affinity compilation exhausted the search deadline')
        stream=torch.cuda.Stream(device=self.device);stream.wait_stream(torch.cuda.current_stream(self.device))
        graph=torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(stream):
                for _ in range(self.policy.warmup):self._measure_values(program,self.inputs)
                with torch.cuda.graph(graph,stream=stream):output=self._measure_values(program,self.inputs)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph.replay()
            expected,grads=self.reference[0]
            torch.testing.assert_close(output[0],expected,rtol=0,atol=0)
            for actual,target in zip(output[1],grads):torch.testing.assert_close(actual,target,rtol=2e-4,atol=2e-5)
        except BaseException:
            graph.reset();raise
        # Retain the graph's explicit I/O only for this selection transaction.
        self.graphs[layout]=(graph,output,max_error)
        return self.graphs[layout]

    def evaluate(self,layout):
        t0=time.monotonic();stage='compile'
        try:
            # On resumed residual evidence, a confirmation may ask for the
            # challenger first. Rebuild the current-input scalar reference.
            if layout==ROW and SCALAR not in self.graphs:self._capture(SCALAR)
            graph,_,error=self.graphs.get(layout) or self._capture(layout)
            compile_ms=(time.monotonic()-t0)*1000
            stage='timing'
            times,metadata=measure_cuda_batches(graph.replay, samples=self.policy.samples,
                calls_per_sample=self.policy.calls_per_sample,device=self.device)
            return TrialOutcome('ok',median_ms=statistics.median(times),timings_ms=times,
                compile_ms=compile_ms,max_abs_error=error,
                metadata={**metadata,**runtime_timing_metadata(times),'exact_scores':True})
        except Exception as error:
            raise_if_fatal_device_context_error(error)
            status='correctness_error' if isinstance(error,AssertionError) else (
                'runtime_error' if stage=='timing' or isinstance(error,TimeoutError) else 'compile_error')
            return TrialOutcome(status,error=f'{type(error).__name__}: {error}'[:4000])

    def close(self):
        for graph,_,_ in self.graphs.values():graph.reset()
        self.graphs.clear();self.reference=None


def _confirmed(result):
    decision=result.winner_decision or {}
    confirmation=decision.get('confirmation') or {}
    # Never promote a lone/unstably-measured challenger or persist a
    # deadline/correctness fallback as paired performance evidence.
    if confirmation.get('status')!='complete' or confirmation.get('timing_unreliable'):
        return False
    return not confirmation.get('invalid') and (result.config==SCALAR or decision.get('reason')=='paired_improvement')


def _search(inputs,mask,context,key,root,policy,started,progress):
    from .winners import save_runtime_winner
    try:evaluator=_Evaluator(inputs,mask,context.workload['rms_epsilon'],policy,started+policy.max_seconds)
    except RuntimeError as error:
        raise_if_fatal_device_context_error(error)
        _LOG.warning('Affinity setup kept scalar: %s',error)
        return SCALAR
    try:
        adapter=_adapter(context,evaluator.evaluate)
        store=ResidualTuningStore(root/'routing_affinity_effort'/context.identifier)
        tuner=AutotuneOrchestrator(adapter,store,[StructuralSearch((ROW,))],
            SequentialScheduler((('structural',None),)),TuningBudget(2,policy.max_seconds),
            seed=policy.seed,max_trials_includes_resumed=True,transfer_history=False,
            confirmation=ConfirmationPolicy(0),coverage_proposals=(Proposal(ROW,'structural'),),
            progress=progress,started_at=started)
        tuner.finalist_rounds=policy.finalist_rounds
        tuner.finalist_margin=policy.finalist_margin
        tuner.finalist_reserve_s=policy.max_seconds*policy.finalist_reserve_fraction
        tuner.effort_contract=policy.evidence_contract()
        try:result=tuner.tune()
        except (RuntimeError,NoEligibleTimingError) as error:
            raise_if_fatal_device_context_error(error)
            _LOG.warning('Affinity search kept scalar: %s',error)
            return SCALAR
        if not _confirmed(result):return SCALAR
        save_runtime_winner(key,_encode(result.config),config_id=adapter.config_id(result.config),
            root=root,median_ms=result.median_ms,tuning_evidence=result.tuning_evidence,metadata=MEASUREMENT)
        return result.config
    finally:evaluator.close()


def resolve(inputs,spec_key,mask,precision,*,force=False,progress=None):
    from .._routing_autotune import parse_spec
    from .._managed import tuning_resources
    from .winners import load_runtime_winner,runtime_tuning_lock

    started=time.monotonic();spec=parse_spec(spec_key);policy=_policy(spec);device=inputs[0].device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Warm router affinity first-hit selection before manual graph capture')
        key,root,context=_contract(inputs,spec,mask,precision,policy)
        def load():return load_runtime_winner(key,_decode,root=root,tuning_policy=policy if force or spec['mode']=='balanced' else None)
        chosen=load()
        if chosen is not None:return chosen
        if not (force or spec['mode']=='balanced') or policy.max_trials<2:return SCALAR
        with runtime_tuning_lock(key,root=root):
            chosen=load()
            if chosen is not None:return chosen
            if time.monotonic()-started>=policy.max_seconds:return SCALAR
            stream=torch.cuda.current_stream(device)

            @tuning_resources
            def work():
                with torch.cuda.device(device),torch.cuda.stream(stream),torch.enable_grad(),torch.autocast('cuda',enabled=False):
                    return _search(inputs,mask,context,key,root,policy,started,progress or _LOG.info)

            with ThreadPoolExecutor(max_workers=1,thread_name_prefix='rtx-affinity-first-hit') as workers:
                return workers.submit(work).result()
