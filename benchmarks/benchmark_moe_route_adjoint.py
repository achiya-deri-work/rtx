"""Paired route-adjoint phase timings on a frozen compiled-router MoE fixture."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
from statistics import median
import sys


def prepared_operands(torch,data,config):
    from rtx._moe_execution import allocate,layout,plan
    from rtx.moe import _config_key
    from rtx.runtime import load_kernel_symbol

    x,gu,down,ids,weights,inverse,dy=(data[k] for k in ('x','gu','down','ids','weights','inverse','dy'))
    n,d=x.shape
    e,c=ids.shape
    h,k=down.shape[-1],inverse.shape[-1]
    key,dtypes,sizes,views,_=layout((n,d,e,c,k,h),_config_key(config),'backward')
    builder,*_=plan(key)
    buffers=allocate(x,dtypes,sizes,views)
    inputs=(x,gu.flatten(0,1),down.flatten(0,1),ids,weights,inverse,dy)
    def value(ref):
        if not isinstance(ref,int):return tuple(value(v) for v in ref)
        return inputs[-ref-1] if ref<0 else buffers[ref]
    for module,symbol,args,refs in builder.steps:
        if symbol=='compile_expert_route_gradient':
            qx,qw,sx,sw,_,mapping,*_=value(refs)
            # Stop before any arena reuse by later backward stages.
            return (qx,qw,sx,sw),mapping,args[4]
        load_kernel_symbol(module,symbol)(*args)(*value(refs))
    raise ValueError('Frozen plan must have fused route gradients')


def compare(torch,candidate,control):
    a,b=candidate.double(),control.double()
    if not (a.isfinite().all() and b.isfinite().all()):raise AssertionError('Nonfinite adjoint')
    delta=(a-b).abs()
    return dict(nrms=float(delta.norm()/b.norm().clamp_min(1e-12)),
                max_abs=float(delta.max()),equal=bool(torch.equal(a,b)))


def measure(torch,call,graphs,rounds,samples,iters):
    from rtx.autotune.preparation_runtime import _warm_measurement_graph

    torch.cuda.synchronize()
    live=[torch.cuda.memory_allocated()]
    records={label:[] for label in graphs}
    labels=tuple(call)
    for r in range(rounds):
        order=labels[r%len(labels):]+labels[:r%len(labels)]
        if r%2:order=order[::-1]
        for label in order:
            g=graphs[label]
            _warm_measurement_graph(g)
            visit=[]
            for _ in range(samples):
                start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(iters):g.replay()
                end.record();end.synchronize()
                visit.append(start.elapsed_time(end)*1000/iters)
            records[label].append(visit)
        live.append(torch.cuda.memory_allocated())
    assert len(set(live))==1,live
    return dict(median_us={label:median(map(median,visits)) for label,visits in records.items()},
                rounds_us=records,live_bytes=live,peak_allocated_bytes=torch.cuda.max_memory_allocated())


def expert_steps(torch,data,config,schedules,args):
    from rtx._moe_reference import operational_reference
    from rtx.moe import mxfp8_routed_experts
    from rtx.routing import BalancedRouting

    x,gu,down,weights=(data[name].detach().requires_grad_() for name in ('x','gu','down','weights'))
    ids,inverse,dy=(data[name] for name in ('ids','inverse','dy'))
    operands=x,gu,down,weights
    configs={label:replace(config,prepared=replace(config.prepared,
        route_gradient_gemm=g,route_gradient_loader=loader))
        for label,(g,loader) in schedules.items() if loader!='shared_vec4'}
    configs['separate']=replace(config,prepared=replace(config.prepared,route_gradient_epilogue='separate',
        route_gradient_loader='accumulator',route_gradient_gemm=None))
    def step_for(c):
        def f(x,gu,down,weights):
            return mxfp8_routed_experts(x,gu,down,BalancedRouting(ids,weights,inverse),config=c,autotune='off')
        compiled=torch.compile(f,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        def step(upstream):
            y=compiled(*operands)
            return y,*torch.autograd.grad(y,operands,upstream)
        return step
    steps={label:step_for(c) for label,c in configs.items()}
    names=('output','input','gate_dw','down_dw','route_dw')
    validation={}
    for label,upstream in (('fixture',dy),('changed',dy*-.37+.21)):
        oracle=operational_reference(x,gu,down,ids,weights,inverse,upstream)
        validation[label]={}
        for variant,step in steps.items():
            actual=step(upstream)
            check={name:compare(torch,a.detach(),b) for name,a,b in zip(names,actual,oracle)}
            for name,err in check.items():
                assert err['nrms']<(.003 if name=='output' else .006) or err['max_abs']<2e-7,(variant,name,err)
            validation[label][variant]=check
            del actual
        del oracle
    calls={label:(lambda step=step:step(dy)) for label,step in steps.items()}
    live=[]
    for _ in range(4):
        for call in calls.values():call()
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    assert len(set(live))==1,live

    stream=torch.cuda.Stream()
    graphs={}
    for label,call in calls.items():
        reference=tuple(t.detach() for t in call())
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):captured=tuple(t.detach() for t in call())
        g.replay();torch.cuda.synchronize()
        for a,b in zip(captured,reference):torch.testing.assert_close(a,b,rtol=0,atol=0)
        graphs[label]=g
    return dict(scope='Frozen expert public fullgraph F+B; 4 packed leaves, no router/shared/optimizer',
                validation=validation,eager_call_live_bytes=live,
                **measure(torch,calls,graphs,args.rounds,args.samples,max(1,args.sample_iters//4)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--source-result',type=Path,required=True,
                        help='Archived exact-key benchmark JSON; verifies original tensors/routes')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--samples',type=int,default=7)
    parser.add_argument('--sample-iters',type=int,default=24)
    parser.add_argument('--expert-steps',action='store_true',
                        help='Also time compiled frozen expert F+B and oracle-check two upstreams')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)

    import torch
    from benchmarks.benchmark_moe_execution import _python_source_digest,_tensor_digest
    from rtx.kernels.mxfp8_moe import compile_movement
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_packed_gemm
    from rtx.kernels.mxfp8_moe_route_epilogue import (
        compile_expert_route_gradient,compile_route_gradient_reduce,
    )
    from rtx.moe import MXFP8MoEConfig

    archived=json.loads(args.source_result.read_text())
    config=MXFP8MoEConfig.from_dict(archived['expert_config'])
    p=config.prepared
    if p.route_gradient_epilogue!='fused' or p.active_prefix or p.input_preparation!='indexed_transpose':
        raise ValueError('Expected a fused, indexed, non-prefix anchor')
    fixture=torch.load(args.fixture,map_location='cpu',weights_only=True)
    digest={name:_tensor_digest(fixture[name]) for name in ('x','gu','down','ids','weights','inverse','dy')}
    identity=archived['identities']
    assert digest['x']==identity['input']
    assert _tensor_digest(fixture['dy'].bfloat16())==identity['upstream']
    assert digest['ids']==identity['routing']['expert_ids']
    assert digest['inverse']==identity['routing']['inverse']
    for expert,(gu,down) in enumerate(zip(fixture['gu'],fixture['down'])):
        gate,up=gu.chunk(2)
        leaves=identity['parameters']
        assert _tensor_digest(gate)==leaves[f'gate_up.projections.e{expert}_gate.weight']
        assert _tensor_digest(up)==leaves[f'gate_up.projections.e{expert}_up.weight']
        assert _tensor_digest(down)==leaves[f'down.projections.e{expert}.weight']
    data={name:value.cuda() for name,value in fixture.items() if isinstance(value,torch.Tensor)}
    del fixture
    ids,weights,inverse,dy=(data[name] for name in ('ids','weights','inverse','dy'))
    n,d=dy.shape
    e,c=ids.shape
    k=inverse.shape[1]
    packed,mapping,problem=prepared_operands(torch,data,config)
    torch.cuda.synchronize()
    cs=problem.m
    slots=torch.arange(e*c,device='cuda').view(e,c)
    valid=inverse[ids].eq(slots[...,None]).any(-1)
    assert torch.equal(mapping.view(e,cs)[:,:c],torch.where(valid,ids,-1).int())
    assert (mapping.view(e,cs)[:,c:]==-1).all()
    fused={}
    anchor=p.route_gradient_gemm or p.down
    for schedule in (anchor,replace(anchor,tile_n=64 if anchor.tile_n==128 else 128)):
        for loader in ('accumulator','shared_vec4','warp_vec2'):
            label=f'{loader}_n{schedule.tile_n}'
            fused[label]=(schedule,loader)
    output,call={},{}
    def fused_call(schedule,loader):
        columns=(d+schedule.tile_n-1)//schedule.tile_n
        partial=torch.empty((e*cs,columns),device='cuda',dtype=torch.float32)
        result=torch.empty((e,c),device='cuda',dtype=torch.float32)
        op=compile_expert_route_gradient(n,e,c,k,problem,schedule,True,False,loader)
        reduce=compile_route_gradient_reduce(e,c,cs,columns,config.movement_threads)
        def run():
            op(*packed,partial,mapping,inverse,dy)
            reduce(partial,result)
        return result,run
    for label,(schedule,loader) in fused.items():
        output[label],call[label]=fused_call(schedule,loader)
    slot=torch.empty((e*cs,d),device='cuda',dtype=torch.bfloat16)
    down=compile_expert_packed_gemm(e,problem,p.down)
    dot=compile_movement(n,d,e,c,k,config.movement_threads,'route_grad','flat',cs)
    separate=torch.empty((e,c),device='cuda',dtype=torch.float32)
    def down_only():down(*packed,slot,slot)
    def dot_only():dot(dy,ids,inverse,weights,slot,slot,separate)
    def separate_run():
        down_only()
        dot_only()
    output['separate'],call['separate']=separate,separate_run
    call.update(down_only=down_only,dot_only=dot_only)

    for run in call.values():run()
    torch.cuda.synchronize()
    base_name=f'accumulator_n{anchor.tile_n}'
    checks={label:compare(torch,t,output[base_name]) for label,t in output.items()}
    for label,check in checks.items():
        assert (check['nrms']<.006 or check['max_abs']<2e-7),(label,check)
        assert (output[label][~valid]==0).all(),label
    for label,(schedule,_) in fused.items():
        assert torch.equal(output[label],output[f'accumulator_n{schedule.tile_n}']),label

    stream=torch.cuda.Stream()
    graphs={}
    for label,run in call.items():
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):run()
        graphs[label]=g
    record=dict(command=sys.argv,source_digest=_python_source_digest(Path.cwd()),
                device=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,cuda=torch.version.cuda,
                shape=dict(tokens=n,features=d,experts=e,logical_capacity=c,physical_capacity=cs,top_k=k,hidden=problem.k),
                identities=digest,fixture=str(args.fixture),exact_archive_match=True,
                schedules={label:dict(config=asdict(schedule),loader=loader) for label,(schedule,loader) in fused.items()},
                validation=checks,**measure(torch,call,graphs,args.rounds,args.samples,args.sample_iters))
    print(json.dumps(record['median_us']),flush=True)
    if args.expert_steps:
        # Keep the expert cache and graph memory objective separate from the
        # prior phase's preallocated operand/partial buffers.
        del graphs,call,output,packed,mapping,slot
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        record['expert_steps']=expert_steps(torch,data,config,fused,args)
        print(json.dumps(record['expert_steps']['median_us']),flush=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
