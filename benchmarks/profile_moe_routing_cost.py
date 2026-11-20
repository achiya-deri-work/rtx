"""Matched current-winner router costs and source-labelled expert movement.

Fresh process per profiler phase. Source annotations are diagnostic only;
ordinary event timings use uninstrumented launchers. No production changes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import json
import math
from pathlib import Path
import statistics


class PlanRanges:
    def __init__(self):self.enabled=False

    @contextmanager
    def install(self):
        import torch
        from rtx import runtime,_moe_execution as execution
        load=runtime.load_kernel_symbol;compile_plan=execution.compiled_plan
        wrappers={}

        def labelled_load(module,symbol):
            factory=load(module,symbol)
            if not module.startswith('mxfp8_moe'):return factory
            @functools.wraps(factory)
            def build(*parameters,**kwargs):
                function=factory(*parameters,**kwargs)
                label=f'moe/{module}/{symbol}'
                if symbol in ('compile_movement','compile_token_reduce'):label+='_'+str(parameters[6])
                @functools.wraps(function)
                def launch(*args,**kw):
                    if not self.enabled:return function(*args,**kw)
                    with torch.profiler.record_function(label),torch.cuda.nvtx.range(label):
                        return function(*args,**kw)
                launch.__rtx_resources__=(function,)
                return launch
            return build

        def labelled_plan(key,device):
            identity=key,device
            if identity not in wrappers:
                original=compile_plan(key,device);phase=json.loads(key)[2]
                @functools.wraps(original)
                def call(inputs,buffers):
                    if not self.enabled:return original(inputs,buffers)
                    with torch.profiler.record_function('moe/'+phase),torch.cuda.nvtx.range('moe/'+phase):
                        return original(inputs,buffers)
                call.__rtx_resources__=original.__rtx_resources__
                wrappers[identity]=call
            return wrappers[identity]
        runtime.load_kernel_symbol=labelled_load;execution.compiled_plan=labelled_plan
        try:yield self
        finally:runtime.load_kernel_symbol=load;execution.compiled_plan=compile_plan


def construct(saved, *, include_components=True):
    import torch
    from benchmarks.benchmark_moe_varlen import initialize_fixture,compile_options
    from benchmarks.benchmark_moe_execution import _tensor_digest,_python_source_digest
    from rtx.moe import BalancedMoE,MXFP8MoEConfig,_selection_spec,_packed_expert_spec,_regular_expert_spec,_dispatch_experts
    from rtx.mlp import MXFP8MLPConfig
    from rtx.routing import RoutingKernelConfig,BalancedRouting
    case=saved['case'];config=MXFP8MoEConfig.from_dict(saved['expert_config'])
    shared=MXFP8MLPConfig.from_dict(saved['shared_config']);router_config=RoutingKernelConfig.from_dict(saved['router_config'])
    if _python_source_digest(Path.cwd()) != saved['source_digest']:raise ValueError('Source drift')
    def factory(c):
        return BalancedMoE(c['d_model'],c['num_experts'],c['expert_hidden_dim'],top_k=c['top_k'],
            shared_hidden_features=c['shared_hidden_dim'],config=config,shared_config=shared,
            device='cuda',autotune='off',router_options=dict(backend='cute',autotune='off',
                config=router_config,affinity_layout=saved['router_affinity_layout']))
    model,x,cu,dy=initialize_fixture(factory,case,saved['initialization_case'],saved['lengths'],
        layout=saved['layout'],seed=saved['seed'],stable_inputs=saved['stable_inputs'],
        stable_experts=saved['stable_experts'],input_profile=saved['input_profile'])
    assert _tensor_digest(x)==saved['identities']['input']
    assert _tensor_digest(dy)==saved['identities']['upstream']
    assert all(_tensor_digest(p)==saved['identities']['parameters'][name] for name,p in model.named_parameters())
    options=compile_options(saved['inductor_max_autotune'])
    route_fn=torch.compile(model.router,**options);route=route_fn(x,cu_seqlens=cu)
    assert all(_tensor_digest(getattr(route,name))==saved['identities']['routing'][name]
               for name in ('expert_ids','inverse'))
    frozen=BalancedRouting(route.expert_ids.detach(),route.route_weights.detach().requires_grad_(),route.inverse.detach())
    spec=_selection_spec(config,'off',None,None)
    spec=(_packed_expert_spec(spec,len(saved['lengths'])) if cu is not None else _regular_expert_spec(spec))
    def expert_forward(value,weights):
        return _dispatch_experts(value,model.gate_up.weight,model.down.weight,
                                 frozen.expert_ids,weights,frozen.inverse,spec,True)
    expert=torch.compile(expert_forward,**options);native=torch.compile(model,**options)
    expert_leaves=(x,*model.gate_up.parameters(),*model.down.parameters(),frozen.route_weights)
    leaves=(x,*model.parameters())
    dw=torch.autograd.grad(expert(x,frozen.route_weights),frozen.route_weights,dy.float())[0].detach()
    def backward(call,parameters,upstream):
        for leaf in parameters:leaf.grad=None
        value=call();value.backward(upstream)
        return value,*(p.grad for p in parameters)
    route_leaves=(x,*model.router.parameters())
    def route_step():
        value=route_fn(x,cu_seqlens=cu)
        return value.route_weights,*torch.autograd.grad(value.route_weights,route_leaves,dw)
    calls=dict(router_forward=lambda:route_fn(x,cu_seqlens=cu),router_step=route_step,
        expert_step=lambda:backward(lambda:expert(x,frozen.route_weights),expert_leaves,dy.float()),
        whole_step=lambda:backward(lambda:native(x,cu_seqlens=cu),leaves,dy))
    if include_components and cu is not None:
        from rtx import _routing_varlen as execution
        scores=torch.compile(model.router.log_affinities,**options)(x).detach()
        key=execution.layout(x.shape[0],len(saved['lengths']),case['num_experts'],case['top_k'],
            model.router.sinkhorn_iterations,tuple(router_config.to_token()))
        transport=execution.transport_forward(scores,cu,key)
        logp,history,scheduled,physical,sequence,local,rowmap,probabilities=transport
        assignments=torch.empty((key[4],key[3]),device=x.device,dtype=torch.int64)
        status=torch.empty(2*len(saved['lengths'])*case['num_experts'],device=x.device,dtype=torch.int64)
        def hard():
            execution.hard_execute([probabilities,scheduled,sequence,local],assignments,status,execution._flat(key),
                model.router.dual_iterations,model.router.repair_rounds,router_config.to_token(),saved['seed'])
        hard();p=logp.detach().requires_grad_()
        metadata=torch.compile(lambda v:execution.SelectedMetadata.apply(v,assignments,scheduled,physical,sequence,rowmap,key),**options)
        reconstructed=metadata(p)
        assert torch.equal(reconstructed[0],route.expert_ids) and torch.equal(reconstructed[2],route.inverse)
        torch.testing.assert_close(reconstructed[1],route.route_weights,rtol=5e-5,atol=5e-7)
        log_gradient=torch.autograd.grad(reconstructed[1],p,dw)[0].detach()
        calls.update(transport_forward=lambda:execution.transport_forward(scores,cu,key),
            transport_backward=lambda:execution.transport_backward(log_gradient,history,scheduled,sequence,rowmap,key),
            hard_forward=hard,metadata_forward=lambda:metadata(p),
            metadata_step=lambda:torch.autograd.grad(metadata(p)[1],p,dw))
    elif include_components:
        from benchmarks.benchmark_moe_execution import _router_component_calls
        extra,_=_router_component_calls(model,x,route,dw,options);calls.update(extra)
    shape=tuple(saved['expert_plan']['shape'])
    from rtx._moe_execution import Builder,layout
    from rtx.moe import _config_key
    buffers={}
    for phase in ('forward','backward'):
        builder=Builder(shape,config,phase).build()
        _,dtypes,sizes,views,outputs=layout(shape,_config_key(config),phase)
        buffers[phase]=dict(arena_bytes=sum(dtype.itemsize*size for dtype,size in zip(dtypes,sizes)),
            logical_buffers=[dict(shape=list(b.shape),dtype=str(b.dtype),bytes=b.dtype.itemsize*math.prod(b.shape),
                                  birth=b.birth,output=b.output) for b in builder.buffers],
            launches=[dict(module=m,symbol=s,parameters=str(p)) for m,s,p,a in builder.steps],
            caveat='Buffer sizes and static launch plan; not measured HBM traffic or simultaneous live storage')
    data=dict(case=case,tokens=x.shape[0],**{name:value.detach().cpu() for name,value in
        dict(x=x,gu=model.gate_up.weight,down=model.down.weight,ids=frozen.expert_ids,
             weights=frozen.route_weights,inverse=frozen.inverse,dy=dy.float()).items()})
    return calls,buffers,data


def tensor_results(function):
    """Uniform capture ABI: never iterate a Tensor into thousands of views."""
    import torch
    from torch.utils._pytree import tree_leaves
    def call():return tuple(v for v in tree_leaves(function()) if isinstance(v,torch.Tensor))
    return call


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('times','trace','nsys'),default='times')
    parser.add_argument('--phase',default='expert_step');parser.add_argument('--fixture',type=Path)
    parser.add_argument('--graph-replay',action='store_true',help='Profile an outer warmed graph, not annotated host launches')
    args=parser.parse_args()
    import torch
    from benchmarks.profile_moe_active_budget import write
    from benchmarks.benchmark_moe_varlen import statistics_ms
    from benchmarks.benchmark_moe_composition import capture_step,compare
    from rtx.autotune.measurement import measure_cuda_batches,runtime_timing_metadata
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    saved=json.loads(args.result.read_text());stream=torch.cuda.Stream();ranges=PlanRanges()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.cuda.stream(stream),ranges.install():
        calls,buffers,data=construct(saved)
        calls={name:tensor_results(call) for name,call in calls.items()}
        if args.fixture:
            args.fixture.parent.mkdir(parents=True,exist_ok=True);torch.save(data,args.fixture)
        selected=list(calls) if args.mode=='times' else [args.phase]
        record=dict(source_result=str(args.result.resolve()),source_digest=saved['source_digest'],
            case=saved['case'],lengths=saved['lengths'],layout=saved['layout'],buffers=buffers,
            matched_input_parameter_route_hashes=True,mode=args.mode,timings={},checks={},
            caveat='Scopes overlap. Source-labelled host traces are diagnostic; ordinary timings use graph replay without labels.')
        if args.mode=='times':
            for phase in selected:
                call=calls[phase];expected=call();stream.synchronize()
                graph,value=capture_step(call,stream)
                _warm_measurement_graph(graph);graph.replay();stream.synchronize()
                if phase.endswith('_step'):
                    record['checks'][phase]=compare(value,expected)
                visits=[]
                for visit in range(3):
                    samples,metadata=measure_cuda_batches(graph.replay,samples=7,calls_per_sample=1,min_batch_ms=10,device='cuda')
                    visits.append(dict(**statistics_ms(samples),measurement=metadata,**runtime_timing_metadata(samples)))
                record['timings'][phase]=visits
                print('TIME',phase,statistics.median(v['median_ms'] for v in visits),flush=True)
                graph.reset();del value,expected
                write(args.output,record)
        else:
            call=calls[args.phase]
            for _ in range(3):value=call()
            stream.synchronize();graph=None
            if args.graph_replay:
                graph,value=capture_step(call,stream);_warm_measurement_graph(graph);call=graph.replay
            ranges.enabled=args.phase=='expert_step' and not args.graph_replay
            if args.mode=='trace':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                    with torch.profiler.record_function('moe/measured_'+args.phase):value=call()
                    stream.synchronize()
                profile.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
            else:
                torch.cuda.cudart().cudaProfilerStart()
                with torch.cuda.nvtx.range('moe/measured_'+args.phase):value=call()
                stream.synchronize();torch.cuda.cudart().cudaProfilerStop()
            ranges.enabled=False
            if graph is not None:graph.reset()
            record['graph_replay']=args.graph_replay
        if saved['layout']=='packed':
            from rtx._routing_varlen import _PLANS
            # Cold diagnostic read after all timing/profiler ranges ended.
            # Never add a host boundary/phase read to the production launcher.
            record['repair_state']=[dict(rounds=int(plan.phase.detach().cpu()),
                total_abs_load_error=int((plan.loads-plan.caps).abs().sum().detach().cpu()))
                for plan in _PLANS.values() if plan.key[0]==sum(saved['lengths'])]
        record['status']='complete';write(args.output,record)


if __name__=='__main__':main()
