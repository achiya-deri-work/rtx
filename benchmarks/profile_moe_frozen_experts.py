"""Counter-safe actual expert branch: fixed real routing, no router execution.

Create the fixture outside profiler injection. This retains the public packed
virtual parameters, true BF16 checkpointing and all optimizer-visible leaves.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-result',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',choices=('bf16','mxfp8'),required=True)
    p.add_argument('--capture',choices=('counters','torch'),default='counters')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    from torch.utils.checkpoint import checkpoint
    from benchmarks.benchmark_moe_execution import _tensor_digest,_python_source_digest
    from benchmarks.benchmark_moe_varlen import compile_options
    from benchmarks.study_moe_priors import atomic_json
    from rtx.moe import (BalancedMoE,MXFP8MoEConfig,torch_routed_experts,
                        _selection_spec,_packed_expert_spec,_regular_expert_spec,_dispatch_experts)
    from rtx.routing import BalancedRouting
    saved=json.loads(args.source_result.read_text());case=saved['case']
    data=torch.load(args.fixture,map_location='cuda',weights_only=True)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        model=BalancedMoE(case['d_model'],case['num_experts'],case['expert_hidden_dim'],
            top_k=case['top_k'],shared_hidden_features=case['shared_hidden_dim'],
            device='cuda',autotune='off',config=MXFP8MoEConfig.from_dict(saved['expert_config']))
        with torch.no_grad():
            model.gate_up.weight.copy_(data['gu']);model.down.weight.copy_(data['down'])
        x=data['x'].detach().requires_grad_();dy=data['dy']
        route=BalancedRouting(data['ids'],data['weights'].detach().requires_grad_(),data['inverse'])
        assert _tensor_digest(x)==saved['identities']['input']
        for name in ('expert_ids','inverse'):
            assert _tensor_digest(getattr(route,name))==saved['identities']['routing'][name]
        leaves=(x,*model.gate_up.parameters(),*model.down.parameters(),route.route_weights)
        spec=_selection_spec(model.config,'off',None,None)
        spec=(_packed_expert_spec(spec,len(saved['lengths']))
              if saved.get('layout')=='packed' else _regular_expert_spec(spec))
        def native(value,weights):
            return _dispatch_experts(value,model.gate_up.weight,model.down.weight,
                route.expert_ids,weights,route.inverse,spec,True)
        def reference(value,weights):
            def core(v,g,o,w):return torch_routed_experts(v,g,o,BalancedRouting(route.expert_ids,w,route.inverse))
            return checkpoint(core,value,model.gate_up.weight,model.down.weight,weights,use_reentrant=True)
        compiled=torch.compile(native if args.precision=='mxfp8' else reference,**compile_options(True))
        def step():
            for leaf in leaves:leaf.grad=None
            y=compiled(x,route.route_weights);y.backward(dy)
            values=(y,*(v.grad for v in leaves))
            if any(v is None for v in values):raise AssertionError('Missing gradient')
            return values
        for _ in range(3):values=step()
        stream.synchronize()
        assert all(bool(v.isfinite().all()) for v in values)
        reference_values=tuple(v.detach().clone() for v in values)
        values=step();stream.synchronize()
        from benchmarks.benchmark_moe_composition import compare
        check=compare(values,reference_values)
        until=time.monotonic()+.3
        while time.monotonic()<until:values=step()
        stream.synchronize()
        if args.capture=='torch':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                values=step();stream.synchronize()
            prof.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
        else:
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range('fixed_experts/'+args.precision):values=step()
            stream.synchronize();torch.cuda.cudart().cudaProfilerStop()
        assert all(bool(v.isfinite().all()) for v in values)
        atomic_json(args.output,dict(status='complete',precision=args.precision,checks=check,
            fixture=str(args.fixture.resolve()),fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            runtime_digest=_python_source_digest(Path.cwd()),scope='Fixed real routing; same packed virtual leaves and true BF16 checkpointing; no router execution under injection',
            source=str(args.source_result.resolve()),config=model.config.to_dict()))


if __name__=='__main__':main()
