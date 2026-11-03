"""Whole-SiTU-MLP adapter for the shared durable, UCB effort portfolio."""
from dataclasses import asdict,replace
from functools import lru_cache
from pathlib import Path
import hashlib
import statistics
import time

import torch

from .core import DiscreteKernelAdapter, KernelContext, stable_id
from .effort import AutotunePolicy, make_effort_autotuner, effort_residual_root
from .legacy import DeviceFingerprint
from .outcomes import TrialOutcome, raise_if_fatal_device_context_error
from .store import ResidualTuningStore
from .._managed import tuning_resources
from ..configs.nvfp4_mlp import NVFP4MLPConfig, NVFP4_MLP_REVISION


@lru_cache(maxsize=1)
def source_identity():
    root=Path(__file__).resolve().parents[1]
    return stable_id({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(root.rglob('*.py'))})


def resolved(config,m,k,h,o):
    from ..fp4_mlp import _default_mlp_backward
    from ..kernels.mxfp8 import MXFP8Problem
    return replace(config,
        gate_backward=config.gate_backward or _default_mlp_backward(MXFP8Problem(m,2*h,k)),
        down_backward=config.down_backward or _default_mlp_backward(MXFP8Problem(m,o,h)))


def backward_openings(config,m,k,h,o):
    from .._bwd_search import materialized_bwd_seed
    from ..kernels.mxfp8 import MXFP8Problem
    result=[]
    for gp,dp in ((4,4),(1,1),(2,2),(8,8),(16,16),(32,32),(4,8),(8,4)):
        gate=materialized_bwd_seed(MXFP8Problem(m,2*h,k),config.gate_backward,gp)
        down=materialized_bwd_seed(MXFP8Problem(m,o,h),config.down_backward,dp)
        if gate is not None and down is not None:
            result.append(replace(config,gate_backward=gate,down_backward=down))
    return result


def search_space(m,k,h,o,training=True,hidden_dtype='mxfp8'):
    from ..nvfp4_inference_autotune import NVFP4_JIT_ROW_REGION_SEARCH_SPACE
    # Geometry/caps/outer-scale arithmetic are the user's numerical policy.
    excluded={"implementation_anchor","x_region_rows","weight_region_rows","outer_scale_math",
              "dependent_launch"}
    axes={'glu_epilogue':('registers',) if hidden_dtype=='mxfp8' else ('registers','exchange'),'glu_store_vec':(1,2,4,8),
          'glu_scratch_swizzle':('none','xor_16b')}
    if hidden_dtype=='mxfp8':
        axes['hidden_quant_vec']=(2,4,8)
    for role in ("gate_up","down"):
        for name,values in NVFP4_JIT_ROW_REGION_SEARCH_SPACE.items():
            if name in excluded or name.startswith("fine_region_geometry"):
                continue
            if role=="gate_up" and (name.startswith("regional_") or name=="scale_transport"):
                continue
            axes[f"{role}.{name}"]=values
    if training:
        from .._bwd_search import backward_search_space
        from ..bwd_autotune import BWD_SEARCH_SPACE
        from ..kernels.mxfp8 import MXFP8Problem
        for role,problem in (("gate_backward",MXFP8Problem(m,2*h,k)),("down_backward",MXFP8Problem(m,o,h))):
            axes.update({f"{role}.{name}":values for name,values in backward_search_space(problem,BWD_SEARCH_SPACE).items()})
        axes["backward_overlap"]=(False,True)
    return axes


def update(config,coordinate,value):
    if coordinate=='backward_families':
        from ..bwd_autotune import bwd_config_from_dict
        return replace(config,**{name:bwd_config_from_dict(value[name]) for name in ('gate_backward','down_backward')})
    if coordinate in ("backward_overlap","glu_epilogue","glu_store_vec","glu_scratch_swizzle","hidden_quant_vec"):
        return replace(config,**{coordinate:value})
    role,_=coordinate.split(".",1)
    if role.endswith("backward"):
        from ..bwd_autotune import update_bwd_config
        updated=update_bwd_config(getattr(config,role),value)
    else:
        from ..nvfp4_inference_autotune import update_dynamic_config
        updated=update_dynamic_config(getattr(config,role),value)
    return replace(config,**{role:updated})


def active_schedule(config,training=True):
    """Do not spend user budgets measuring provably inactive coordinates."""
    data=config.to_dict()
    if config.glu_epilogue=='registers':
        data.pop('glu_store_vec',None)
        if config.hidden_dtype=='bf16':
            data.pop('glu_scratch_swizzle',None)
    if config.hidden_dtype=='bf16':
        data.pop('hidden_quant_vec',None)
    gemm=data['gate_up']['gemm']
    for name in tuple(gemm):
        if name.startswith('regional_') or name in ('store_vec','epilogue_stages'):
            gemm.pop(name)
    for role in ('gate_up','down'):
        p=data[role]
        if p['region_ownership']=='cta_cached':
            p.pop('region_amax_load_bits',None)
            p.pop('region_amax_unroll',None)
        if p['quant']['block_scale_selection']=='amax':
            p['quant'].pop('adaptive_reuse_packed',None)
            p['quant'].pop('adaptive_decode',None)
    if training:
        from .._bwd_search import active_backward_dict
        for name in ('gate_backward','down_backward'):
            data[name]=active_backward_dict(getattr(config,name))
    else:
        for name in ('gate_backward','down_backward','backward_overlap'):
            data.pop(name,None)
    return data


def make_adapter(x,gu,down,*,initial,phase="train",policy=None,execution_mix=None):
    from ..fp4_mlp import _key,_validate,nvfp4_mlp
    from .._nvfp4_mlp_reference import reference
    from .execution_mix import MLPExecutionMix
    policy=policy or AutotunePolicy()
    if phase not in ("train","forward"):
        raise ValueError("phase must be train or forward")
    training=phase=="train"
    m,k=x.numel()//x.shape[-1],x.shape[-1]
    h,o=gu.shape[0]//2,down.shape[0]
    initial=resolved(initial,m,k,h,o)
    _validate(x,gu,down,initial)
    mix=execution_mix or MLPExecutionMix(training_pairs=1 if training else 0,inference_forwards=0 if training else 1)
    mix.validate_phase(phase)
    values=tuple(t.detach().clone().requires_grad_(training) for t in (x,gu,down))
    generator=torch.Generator(device=x.device).manual_seed(1729)
    dy=torch.randn((*x.shape[:-1],o),device=x.device,dtype=x.dtype,generator=generator)
    oracle=reference(*values,dy,initial)
    held={}
    def evaluate(c):
        stage="compile"
        started=time.monotonic()
        try:
            held.clear()
            def forward(a,b,d):
                return nvfp4_mlp(a,b,d,config=c)
            # Reset only this disposable benchmark closure, never the user's
            # compiled model or global Dynamo state as the budget increases.
            torch._dynamo.reset_code(forward.__code__)
            function=torch.compile(forward,fullgraph=True,dynamic=False)
            @torch.enable_grad()
            def run():
                result=None
                for _ in range(mix.inference_forwards):
                    with torch.no_grad():
                        result=(function(*values),)
                for _ in range(mix.grad_forwards):
                    result=(function(*values),)
                for _ in range(mix.training_pairs):
                    y=function(*values)
                    result=(y,*torch.autograd.grad(y,values,dy))
                return result
            actual=run()
            metrics=[]
            for a,r in zip(actual,oracle):
                delta=a.detach().float()-r.float()
                rms=float(delta.square().mean().sqrt())
                rr=float(r.float().square().mean().sqrt())
                metrics.append(dict(finite=bool(a.isfinite().all()),rms=rms,reference_rms=rr,
                    relative_l2=rms/max(rr,1e-12),max_abs=float(delta.abs().max())))
            if any(not v['finite'] or v['relative_l2']>.03 for v in metrics):
                return TrialOutcome("correctness_error",error=str(metrics),metadata={"numerics":metrics})
            # Do not keep a default-stream grad_fn / AccumulateGrad alive
            # while warming and capturing this candidate on its own stream.
            del actual
            for _ in range(policy.warmup):
                run()
            caller=torch.cuda.current_stream(x.device)
            stream=torch.cuda.Stream(device=x.device)
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                run()
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                outputs=run()
            caller.wait_stream(stream)
            compile_ms=(time.monotonic()-started)*1000
            stage="runtime"
            timings=[]
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for _ in range(policy.samples):
                begin.record()
                for _ in range(policy.calls_per_sample):
                    graph.replay()
                end.record();end.synchronize()
                timings.append(begin.elapsed_time(end)/policy.calls_per_sample)
            held.update(function=function,graph=graph,outputs=outputs)
            from .measurement import runtime_timing_metadata
            return TrialOutcome("ok",median_ms=statistics.median(timings),timings_ms=timings,
                compile_ms=compile_ms,metadata={"numerics":metrics,"measurement":"complete_default_compile_graph_replay",
                                              **runtime_timing_metadata(timings)})
        except Exception as exc:
            raise_if_fatal_device_context_error(exc)
            return TrialOutcome("compile_error" if stage=="compile" else "runtime_error",error=str(exc),
                                compile_ms=(time.monotonic()-started)*1000)
    from .hardware import static_device_profile
    fingerprint=DeviceFingerprint.current(x.device)
    axes=search_space(m,k,h,o,training,initial.hidden_dtype)
    if training:
        openings=backward_openings(initial,m,k,h,o)
        if openings:
            axes['backward_families']=tuple({name:asdict(getattr(c,name)) for name in ('gate_backward','down_backward')} for c in openings)
    return DiscreteKernelAdapter(
        context=KernelContext("nvfp4_recompute_mlp",NVFP4_MLP_REVISION,
            {"input_shape":list(x.shape),"m":m,"k":k,"h":h,"o":o,"activation":"situ","phase":phase,
             "beta":initial.beta,"linear_beta":initial.linear_beta,
             "hidden_dtype":initial.hidden_dtype,
             "numerical_policy":{name:{"x_rows":getattr(initial,name).x_scale_region_rows,
                "w_rows":getattr(initial,name).weight_scale_region_rows,
                "outer":getattr(initial,name).tensor_scale_mode,
                "block":getattr(initial,name).quant.block_scale_selection} for name in ("gate_up","down")}},
            device={**fingerprint.as_dict(),**static_device_profile(x.device)},
            tags={"source":source_identity(),"compile":"default_fullgraph_static",
                  "execution_mix":mix.contract(),"warmup":policy.warmup,"samples":policy.samples,
                  "calls_per_sample":policy.calls_per_sample}),
        initial_config=initial,axes=axes,
        config_id_fn=lambda c:stable_id(active_schedule(c,training)),serialize_fn=lambda c:c.to_dict(),
        deserialize_fn=NVFP4MLPConfig.from_dict,update_fn=update,
        evaluator=evaluate,rejection_fn=lambda c:(("implementation_rejected",reason)
            if (reason:=c.rejection(m,k,h,o)) else None))


@tuning_resources
def tune_nvfp4_mlp(x,gu,down,*,initial=None,phase="train",tuning_policy=None,
                   output_dir=None,execution_mix=None,progress=None,max_trials=None,
                   time_budget_s=None,seed=None):
    """Tune the complete default-compiled workload using the shared UCB portfolio."""
    from .legacy import default_cache_dir
    if torch.compiler.is_compiling():
        raise RuntimeError("explicit tuning must run outside compilation")
    if tuning_policy is not None and not isinstance(tuning_policy,AutotunePolicy):
        raise TypeError("tuning_policy must be AutotunePolicy")
    policy=(tuning_policy or AutotunePolicy()).for_pass(phase=="train")
    overrides={}
    if max_trials is not None:
        overrides['max_trials']=max_trials
    if time_budget_s is not None:
        overrides['max_seconds']=time_budget_s
    if seed is not None:
        overrides['seed']=seed
    if overrides:
        policy=replace(policy,**overrides)
    root=Path(output_dir) if output_dir is not None else default_cache_dir()/"nvfp4_mlp"
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm/tune NVFP4MLP before CUDA graph capture")
        adapter=make_adapter(x,gu,down,initial=initial or NVFP4MLPConfig(),phase=phase,
                             policy=policy,execution_mix=execution_mix)
        store=ResidualTuningStore(effort_residual_root(root/adapter.context.identifier,policy),transfer_root=root)
        c=adapter.initial_config
        seeds=[]
        # Small budgets must visit both forward projections, not only the
        # backward family openings. These are candidates, never forced wins.
        for role in ('gate_up','down'):
            p=getattr(c,role)
            for stages in (1,3):
                choices={'gemm':replace(p.gemm,stages=stages)}
                if role=='down' and c.hidden_dtype=='mxfp8':
                    choices['region_ownership']='warp'
                candidate=replace(c,**{role:replace(p,**choices)})
                if adapter.rejection(candidate) is None:
                    seeds.append(candidate)
                    break
        exchange=replace(c,glu_epilogue='exchange')
        if adapter.rejection(exchange) is None:
            seeds.append(exchange)
        if phase=="train":
            from .._bwd_search import backward_structural_seeds
            from ..kernels.mxfp8 import MXFP8Problem
            for role,p in (("gate_backward",MXFP8Problem(x.numel()//x.shape[-1],gu.shape[0],x.shape[-1])),
                           ("down_backward",MXFP8Problem(x.numel()//x.shape[-1],down.shape[0],down.shape[1]))):
                seeds.extend(replace(c,**{role:leg}) for leg in backward_structural_seeds(p,getattr(c,role))[:2])
            # The shared effort layer admits only a bounded structural prefix.
            # Put complete paired families before cosmetic forward neighbors.
            seeds=backward_openings(c,x.numel()//x.shape[-1],x.shape[-1],down.shape[1],down.shape[0])+seeds
        return make_effort_autotuner(adapter,store,policy,progress=progress,candidates=seeds).tune()
