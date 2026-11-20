"""Paired frozen-route expert F+B on a fixture exported by benchmark_moe_varlen.

Use a fresh process, fixed compiler policy, and the HOME GPU0 study lock.
Compare only configuration sidecars from actual measured expert admissions.
Private route-map timing is diagnostic, not a term to add to F+B timings.
"""

import argparse
import json
from math import ceil, gcd, lcm
from pathlib import Path
import statistics


def work_envelope(config, shape, lengths, inverse):
    from rtx._moe_compact_layout import compact_forward_capacity

    n, d, e, c, k, _ = shape
    p = config.prepared
    physical = p.physical_capacity(c)
    real = inverse[inverse >= 0]
    last_original = ceil((int(real.remainder(c).max()) + 1) / 32) * 32
    max_real = int((real // c).bincount(minlength=e).max())
    period = e // gcd(e, k)
    scheduled = [ceil(length / period) * (k // gcd(e, k)) for length in lengths]
    compact = compact_forward_capacity(n, len(lengths), e, k, physical,
        lcm(32, p.gate_up.tile_m, p.down.tile_m)) if p.forward_row_layout == 'compact' else None
    return dict(real_mean=n*k/e, max_real=max_real, original_runtime_prefix=last_original,
        original_scheduled=sum(scheduled), original_group_padded=sum(ceil(s/32)*32 for s in scheduled),
        physical=physical, forward_storage=compact or physical,
        forward_math_m_bound=(max_real if compact else last_original) if p.active_prefix else physical,
        backward_math_k_bound=last_original if p.active_prefix else physical,
        scope='Rows before CTA rounding; not measured issued work')


def run(args):
    import torch
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest
    from benchmarks.benchmark_moe_varlen import compile_options, statistics_ms
    from benchmarks.study_moe_priors import atomic_json
    from rtx.autotune.measurement import measure_cuda_batches, runtime_timing_metadata
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.moe import (BalancedMoE, MXFP8MoEConfig, _dispatch_experts, _packed_expert_spec, _selection_spec)
    from rtx.runtime import load_kernel_symbol, retain_runtime_resources

    if args.output.exists():raise FileExistsError(args.output)
    saved=json.loads(args.source_result.read_text());case=saved['case']
    n,d,e,c,k,h=(sum(saved['lengths']),case['d_model'],case['num_experts'],saved['expert_capacity_bound'],
                 case['top_k'],case['expert_hidden_dim'])
    shape=(n,d,e,c,k,h);pools=len(saved['lengths'])
    configs={'baseline':MXFP8MoEConfig.from_dict(saved['expert_config'])}
    for item in args.config:
        label,path=item.split('=',1)
        if label in configs:raise ValueError(f'Duplicate cell {label}')
        configs[label]=MXFP8MoEConfig.from_dict(json.loads(Path(path).read_text())['config'])
    data=torch.load(args.fixture,map_location='cpu',weights_only=True)
    ids=data['ids'];inv=data['inverse']
    hashes={name:_tensor_digest(data[name]) for name in ('x','dy','gu','down','ids','weights','inverse')}
    hashes['dy_bf16']=_tensor_digest(data['dy'].to(data['x'].dtype))
    expected=saved['identities']; assert hashes['x']==expected['input'] and hashes['dy_bf16']==expected['upstream']
    assert hashes['ids']==expected['routing']['expert_ids'] and hashes['inverse']==expected['routing']['inverse']
    assert all(_tensor_digest(data['gu'][expert,offset:offset+h])==
        expected['parameters'][f'gate_up.projections.e{expert}_{name}.weight']
        for expert in range(e) for name,offset in (('gate',0),('up',h)))
    assert all(_tensor_digest(data['down'][expert])==expected['parameters'][f'down.projections.e{expert}.weight']
               for expert in range(e))
    assert tuple(ids.shape)==(e,c) and tuple(inv.shape)==(n,k)
    work={label:work_envelope(config,shape,saved['lengths'],inv) for label,config in configs.items()}
    options=compile_options(saved['inductor_max_autotune'])
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        data={name:data[name].cuda() for name in ('x','dy','gu','down','ids','weights','inverse')}
        x=data['x'].requires_grad_();dy=data['dy'];ids=data['ids'];inv=data['inverse']
        w=data['weights'].requires_grad_()
        model=BalancedMoE(d,e,h,top_k=k,shared_hidden_features=case['shared_hidden_dim'],
                          device='cuda',autotune='off')
        with torch.no_grad():
            model.gate_up.weight.copy_(data['gu']);model.down.weight.copy_(data['down'])
        assert _tensor_digest(model.gate_up.weight)==hashes['gu']
        assert _tensor_digest(model.down.weight)==hashes['down']
        params=tuple((name,p) for name,p in model.named_parameters() if name.startswith(('gate_up.','down.')))
        names=('output','input',*(name for name,_ in params),'route_weights')
        leaves=(x,*(p for _,p in params),w)

        def step_for(config):
            spec=_packed_expert_spec(_selection_spec(config,'off',None,None),pools)

            def forward(value,weights):
                return _dispatch_experts(value,model.gate_up.weight,model.down.weight,ids,weights,inv,spec,True)
            call=torch.compile(forward,**options)

            def step():
                value=call(x,w)
                return value,*torch.autograd.grad(value,leaves,dy)
            return step

        def compare(values,expected):
            errors={}
            for name,value,ref in zip(names,values,expected,strict=True):
                assert value.shape==ref.shape and value.dtype==ref.dtype
                assert bool(value.isfinite().all()) and bool(ref.isfinite().all()),name
                target=ref.detach().float();delta=value.detach().float()-target
                nrms=float(delta.norm()/target.norm().clamp_min(1e-12));maximum=float(delta.abs().max())
                assert nrms<=(.003 if name=='output' else .006) or maximum<=2e-7,(name,nrms,maximum)
                errors[name]=dict(nrms=nrms,max_abs=maximum)
            return errors

        graphs={};checks={};live={}
        try:
            for label,config in configs.items():
                step=step_for(config)
                reference=step();stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with retain_runtime_resources(),torch.cuda.graph(graph,stream=stream):values=step()
                graphs[label]=(graph,values,step)
                _warm_measurement_graph(graph);graph.replay();stream.synchronize()
                checks[label]=dict(replay=compare(values,reference),matched_native=compare(values,graphs['baseline'][1]))
                del reference
            for label,(graph,_,_) in graphs.items():
                live[label]=[]
                for _ in range(5):
                    graph.replay();stream.synchronize();live[label].append(torch.cuda.memory_allocated())
                assert len(set(live[label]))==1
            visits={label:[] for label in graphs}
            order=list(graphs)
            for visit in range(4):
                for label in (order if visit%2==0 else order[::-1]):
                    graph=graphs[label][0];_warm_measurement_graph(graph)
                    values,meta=measure_cuda_batches(graph.replay,samples=7,calls_per_sample=8,device=x.device)
                    visits[label].append(dict(**statistics_ms(values),**meta,**runtime_timing_metadata(values)))
                    print('TIME',visit,label,visits[label][-1]['median_ms'],'ms',flush=True)
            map_cost={}
            for stride in sorted({v['forward_storage'] for label,v in work.items()
                                  if configs[label].prepared.forward_row_layout=='compact'}):
                call=load_kernel_symbol('mxfp8_moe_compact','compile_compact_route')(n,e,c,k,stride)
                operands=(ids,inv,torch.empty((e,c),device='cuda',dtype=torch.int32),
                          torch.empty((e,stride),device='cuda',dtype=torch.int32),
                          torch.empty(e,device='cuda',dtype=torch.int32))
                call(*operands);stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                try:
                    with retain_runtime_resources(),torch.cuda.graph(graph,stream=stream):call(*operands)
                    _warm_measurement_graph(graph)
                    values,meta=measure_cuda_batches(graph.replay,samples=7,calls_per_sample=32,device=x.device)
                    map_cost[stride]=dict(**statistics_ms(values),**meta,**runtime_timing_metadata(values))
                finally:graph.reset()
            atomic_json(args.output,dict(status='complete',shape=shape,hashes=hashes,work=work,
                fixture=str(args.fixture.resolve()),source_digest=_python_source_digest(Path.cwd()),
                gpu=torch.cuda.get_device_name(x.device),torch=str(torch.__version__),compile_options=options,
                checks=checks,live_bytes=live,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                configs={label:config.to_dict() for label,config in configs.items()},map_cost=map_cost,visits=visits,
                medians_ms={label:statistics.median(v['median_ms'] for v in rows) for label,rows in visits.items()}))
        finally:
            for graph,_,_ in graphs.values():graph.reset()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-result',required=True,type=Path)
    p.add_argument('--fixture',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--config',action='append',required=True,help='Name=admitted expert sidecar; repeat for each cell')
    run(p.parse_args())


if __name__=='__main__':main()
