"""Matched component, expert and whole-layer timing for prepared MoE schedules."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import statistics


def _component(operands,dy,config,stage='recompute'):
    from rtx._moe_execution import allocate,compiled_plan,layout,plan
    from rtx.moe import _config_key
    x,gu,down,ids,weights,inverse=operands
    shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
    key,dtypes,sizes,views,_=layout(shape,_config_key(config),'backward')
    builder,*_=plan(key)
    buffers=allocate(x,dtypes,sizes,views)
    inputs=[x,gu.flatten(0,1),down.flatten(0,1),ids,weights,inverse,dy.contiguous()]
    functions=compiled_plan(key,x.device.index).__rtx_resources__
    def resolve(ref):
        return (inputs[-ref-1] if ref<0 else buffers[ref]) if isinstance(ref,int) else tuple(map(resolve,ref))
    target=None if stage=='recompute' else builder.outputs[1 if stage=='gate_dw' else 2]
    for step,fn in zip(builder.steps,functions):
        args=tuple(map(resolve,step[-1]))
        if stage=='recompute' and step[1]=='compile_expert_joint_recompute':
            # The prep plan initially reuses memory based on the whole
            # backward liveness schedule. Never execute its later launches:
            # they can overwrite component inputs before replay.
            return (lambda:fn(*args)),args[-1][1:]
        if target is not None and step[1]=='compile_expert_packed_gemm' and step[-1][4]==target:
            return (lambda:fn(*args)),(args[4],)
        fn(*args)
    raise ValueError(f'Expected a prepared unsplit {stage} step')


def _capture(call,stream):
    import torch
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result=call()
            del result
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        result=call()
        if result is not None:result=tuple(v.detach() for v in result)
    torch.cuda.current_stream().wait_stream(stream)
    return graph,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-result',type=Path,required=True,
        help='Frozen benchmark_moe_varlen.py result; consumes seed, case, lengths and all three schedules')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-index',type=int,default=0,help='Select from legal structural seeds')
    p.add_argument('--short-dw',type=int,choices=(32,64),help='Compare unsplit K32/K64 dW instead of joint recompute')
    p.add_argument('--cooperative128-control',action='store_true',help='Retain the cooperative transport but use baseline K128')
    p.add_argument('--dw-stages',nargs='+',choices=('gate_dw','down_dw'),default=('gate_dw','down_dw'))
    p.add_argument('--tokens',type=int,help='Override token count for a frozen single-pool schedule')
    p.add_argument('--d-model',type=int,help='Override D for the frozen schedules')
    p.add_argument('--hidden',type=int,help='Override routed H for the frozen schedules')
    p.add_argument('--same-tile-control',action='store_true',help='Hold every other coordinate fixed, including the new GEMM tile')
    p.add_argument('--variant',choices=('baseline','candidate'),help='Time only this side in a fresh process after paired validation')
    p.add_argument('--samples',type=int,default=7)
    p.add_argument('--skip-whole',action='store_true',help='Run component and frozen-route experts only')
    p.add_argument('--oracle',action='store_true',help='Independent full FP64 expert oracle (use bounded shapes)')
    args=p.parse_args()
    import torch
    # Create even benchmark-only views/leaves on the same stream as warmup
    # and capture. A view prepared on the default stream can create a cached
    # AccumulateGrad before the first backward and invalidate capture.
    with torch.cuda.stream(torch.cuda.Stream()):
        run(args)


def run(args):
    import torch
    from rtx._moe_reference import operational_reference
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.moe import joint_recompute_seeds,rejection,source_revision
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.mlp import MXFP8MLPConfig
    from rtx.moe import BalancedMoE,MXFP8MoEConfig,mxfp8_routed_experts
    from rtx.routing import BalancedRouting,RoutingKernelConfig
    from benchmarks.benchmark_moe_execution import _tensor_digest,_python_source_digest,write
    from benchmarks.profile_moe import hardware_attributes,statistics_ms

    saved=json.loads(args.baseline_result.read_text())
    case=dict(saved['case']); lengths=saved['lengths']
    if args.tokens is not None:
        if len(lengths)!=1:raise ValueError('Can only override single-pool token count')
        lengths=[args.tokens]
    if args.d_model is not None:case['d_model']=args.d_model
    if args.hidden is not None:case['expert_hidden_dim']=args.hidden
    seed,n=saved['seed'],sum(lengths)
    d,e,h,k=(case[field] for field in ('d_model','num_experts','expert_hidden_dim','top_k'))
    base=MXFP8MoEConfig.from_dict(saved['expert_config'])
    if base.family!='grouped_prepared' or base.prepared.backward_preparation!='joint':
        raise ValueError('Expected frozen row-major grouped_prepared/joint benchmark result')
    router=RoutingKernelConfig.from_dict(saved['router_config'])
    shared=MXFP8MLPConfig.from_dict(saved['shared_config'])
    def build(config):
        return BalancedMoE(d,e,h,top_k=k,shared_hidden_features=case['shared_hidden_dim'],
            num_shared_experts=case['num_shared_experts'],device='cuda',dtype=torch.bfloat16,
            autotune='off',config=config,shared_config=shared,router_options=dict(backend='cute',config=router))
    torch.manual_seed(seed)
    model=build(base)
    with torch.no_grad():model.router.projection_weight.normal_(std=.02)
    x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    dy=torch.randn(x.shape,device='cuda',dtype=torch.float32)
    cu=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
    with torch.no_grad():route=model.router(x,cu_seqlens=cu)
    ids,weights,inverse=route
    weights=weights.detach().requires_grad_()
    gu=model.gate_up.weight.detach().requires_grad_()
    down=model.down.weight.detach().requires_grad_()
    operands=(x,gu,down,ids,weights,inverse)
    shape=(n,d,e,ids.shape[1],k,h)
    if args.short_dw:
        if args.same_tile_control:raise ValueError('Ordinary GEMM has no sub-128 same-tile control')
        p=base.prepared
        for stage in args.dw_stages:p=p.with_short_dw(stage,args.short_dw)
        candidate=replace(base,prepared=p)
        components=tuple(dict.fromkeys(args.dw_stages))
        if args.cooperative128_control:
            baseline_p=replace(p,**{stage:replace(getattr(p,stage),tile_k=128) for stage in components})
            base=replace(base,prepared=baseline_p)
    else:
        if args.cooperative128_control:raise ValueError('Cooperative K128 control requires --short-dw')
        choices=list(joint_recompute_seeds(shape,base))
        if not choices:raise ValueError(f'No legal joint recompute alternatives for frozen config at {shape}')
        candidate=choices[args.candidate_index]
        if args.same_tile_control:
            base=replace(candidate,prepared=replace(candidate.prepared,joint_recompute_engine='legacy'))
        components=('recompute',)
    configs={'baseline':base,'candidate':candidate}
    for config in configs.values():
        if why:=rejection(config,shape):raise ValueError(why)
    operations={};replay_tensors={}
    options=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    for name,config in configs.items():
        operations[name]={}
        replay_tensors[name]={}
        for phase in components:
            with torch.no_grad():
                component,outputs=_component(operands,dy,config,phase)
            operations[name][phase]=component
            replay_tensors[name][phase]=outputs
        def expert(a,b,c,w,config=config):
            return mxfp8_routed_experts(a,b,c,BalancedRouting(ids,w,inverse),config=config)
        compiled=torch.compile(expert,**options)
        def expert_step(fn=compiled):
            y=fn(x,gu,down,weights)
            return y,*torch.autograd.grad(y,(x,gu,down,weights),dy)
        operations[name]['expert']=expert_step
        if not args.skip_whole:
            layer=build(config);layer.load_state_dict(model.state_dict())
            parameters=tuple(layer.parameters())
            whole=torch.compile(layer,**options)
            whole_dy=dy.bfloat16()
            def whole_step(fn=whole,params=parameters):
                y=fn(x,cu_seqlens=cu)
                return y,*torch.autograd.grad(y,(x,*params),whole_dy)
            operations[name]['whole']=whole_step

    expected=operational_reference(*operands,dy) if args.oracle else None
    def errors(actual,reference,first=.003):
        reports=[]
        assert len(actual)==len(reference)
        for i,(a,b) in enumerate(zip(actual,reference)):
            assert a.shape==b.shape and bool(a.isfinite().all())
            delta=a.detach().float()-b.detach().float()
            nrms=float(delta.norm()/b.detach().float().norm().clamp_min(1e-12))
            absolute=float(delta.abs().max())
            assert nrms<(first if i==0 else .006) or absolute<2e-7,(i,nrms,absolute)
            reports.append(dict(nrms=nrms,max_abs=absolute))
        return reports

    graphs={name:{} for name in configs}
    live={}
    stream=torch.cuda.current_stream()
    for name in configs:
        for phase,call in operations[name].items():
            print(f'VALIDATE {name} {phase}',flush=True)
            graphs[name][phase]=_capture(call,stream)
            graph,out=graphs[name][phase]
            for _ in range(3):graph.replay()
            if out is not None:
                for value in out:assert bool(value.isfinite().all())
                # Keep all backward calls on the capture stream. Multiple
                # graph variants can share leaf AccumulateGrad objects, which
                # retain the stream of their previous backward on Torch 2.13.
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    errors(call(),out)
                    stream.synchronize()
                    samples=[torch.cuda.memory_allocated()]
                    for _ in range(5):
                        eager=call();del eager
                        stream.synchronize()
                        samples.append(torch.cuda.memory_allocated())
                live[f'{name}_{phase}']=samples
                assert samples[-1]==samples[0]

    checks={}
    for phase in operations['baseline']:
        if phase in components:
            a,b=(replay_tensors[name][phase] for name in configs)
            checks[phase]=dict(unequal_bytes=[
                int(torch.count_nonzero(x.view(torch.uint8)!=y.view(torch.uint8))) for x,y in zip(a,b)])
            if args.same_tile_control:assert not any(checks[phase]['unequal_bytes'])
            if args.short_dw:checks[phase]['numeric']=errors(b,a,first=.006)
        else:
            a=graphs['baseline'][phase][1];b=graphs['candidate'][phase][1]
            checks[phase]=errors(b,a)
            if phase=='expert' and expected is not None:
                checks['fp64_oracle']={'baseline':errors(a,expected),'candidate':errors(b,expected)}

    rounds={phase:[] for phase in operations['baseline']}
    order=(args.variant,) if args.variant else ('baseline','candidate','candidate','baseline')
    for phase in rounds:
        for name in order:
            graph=graphs[name][phase][0];_warm_measurement_graph(graph)
            values,metadata=measure_cuda_batches(graph.replay,samples=args.samples,calls_per_sample=16,
                device=x.device,min_batch_ms=3.)
            rounds[phase].append(dict(label=name,**statistics_ms(values),**metadata))
    summary={}
    for phase,visits in rounds.items():
        medians={name:[v['median_ms'] for v in visits if v['label']==name] for name in configs if name in order}
        times={name:statistics.median(ms) for name,ms in medians.items()}
        summary[phase]=dict(median_ms=times,speedup=times['baseline']/times['candidate'] if len(times)>1 else None,
            max_round_drift=None if args.variant else max(abs(v[1]-v[0])/statistics.median(v) for v in medians.values()),
            max_relative_mad=max(v['relative_mad'] for v in visits))
    write(args.output,dict(shape=shape,lengths=lengths,seed=seed,same_tile_control=args.same_tile_control,
        baseline_result=str(args.baseline_result),short_dw=args.short_dw,components=components,
        cooperative128_control=args.cooperative128_control,
        source_revision=source_revision(),runtime_digest=_python_source_digest(Path(__file__).resolve().parents[1]),
        gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
        hardware=hardware_attributes(),configs={name:c.to_dict() for name,c in configs.items()},
        router_config=router.to_dict(),shared_config=asdict(shared),
        hashes={key:_tensor_digest(value) for key,value in zip(
            ('x','gu','down','ids','weights','inverse','dy'),(*operands,dy))},
        checks=checks,live_allocated_bytes=live,rounds=rounds,summary=summary))


if __name__=='__main__':
    main()
