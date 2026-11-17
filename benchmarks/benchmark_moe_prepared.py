"""Matched expert-only A/B: fused quantization, quantize-once, batched BF16."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch

from rtx.moe import BalancedMoE,MXFP8MoEConfig,DEFAULT_MOE_CONFIG,_run_forward,_run_backward,torch_routed_experts
from rtx._moe_prepared import forward as prepared_forward,backward as prepared_backward
from rtx._moe_reference import operational_reference
from rtx.routing import BalancedRouting
from benchmarks.benchmark_balanced_router import measure
from benchmarks.benchmark_balanced_moe import metric


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id',default='d384_e24_k6_h128')
    parser.add_argument('--tokens',type=int,default=512)
    parser.add_argument('--baseline-study',type=Path)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--gather-mode',choices=('flat','warp'),default='flat')
    parser.add_argument('--transpose-mode',choices=('direct','staged'),default='staged')
    parser.add_argument('--native-scales',action='store_true')
    parser.add_argument('--joint-backward',action='store_true')
    parser.add_argument('--glu-quant-vec',type=int,choices=(1,2,4,8))
    parser.add_argument('--silu-math',choices=('division','shared_reciprocal'))
    parser.add_argument('--linear-transport',choices=('tma','cooperative'))
    parser.add_argument('--weight-epilogue',choices=('direct','tma'))
    parser.add_argument('--all-tma-outputs',action='store_true')
    parser.add_argument('--quant-grid',choices=('operand','expert','portfolio'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    case=next(c for c in json.loads(Path('benchmarks/balanced_moe_cases.json').read_text())['cases'] if c['id']==args.case_id)
    torch.manual_seed(819)
    n,d,e,k,h=args.tokens,case['d_model'],case['num_experts'],case['top_k'],case['expert_hidden_dim']
    model=BalancedMoE(d,e,h,top_k=k,shared_hidden_features=case['shared_hidden_dim'],
                      device='cuda',autotune='off',router_options={'backend':'cute'})
    with torch.no_grad():model.router.projection_weight.normal_(std=.02)
    x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    with torch.no_grad():r=model.router(x)
    route=BalancedRouting(r.expert_ids,r.route_weights.detach().requires_grad_(),r.inverse)
    gu=model.gate_up.weight.detach().requires_grad_();down=model.down.weight.detach().requires_grad_()
    operands=(x,gu,down,*route)
    dy=torch.randn(x.shape,device='cuda',dtype=torch.float32)
    config=DEFAULT_MOE_CONFIG
    if args.baseline_study:
        row=next(r for r in json.loads(args.baseline_study.read_text())['results']
                 if r['case']['id']==args.case_id and r['tokens']==n)
        config=MXFP8MoEConfig.from_dict(row['config'])
    config=replace(config,gather_mode=args.gather_mode)
    prepared_config=replace(config,family='grouped_prepared',prepared=replace(config.prepared,
                            transpose_mode=args.transpose_mode))
    if args.native_scales:
        prepared_config=replace(prepared_config,prepared=prepared_config.prepared.with_scale_layout('mma128'))
    p=prepared_config.prepared
    if args.quant_grid:p=replace(p,quant_grid=args.quant_grid)
    if args.joint_backward:
        p=replace(p,backward_preparation='joint',recompute=replace(p.recompute,glu_grad_store='exchange'))
    glu_changes={key:value for key,value in (('glu_quant_vec',args.glu_quant_vec),('glu_silu_math',args.silu_math)) if value is not None}
    if glu_changes:p=replace(p,**{stage:replace(getattr(p,stage),**glu_changes) for stage in ('gate_up','recompute')})
    prepared_config=replace(prepared_config,prepared=p)
    if args.linear_transport:
        from rtx.autotune.moe import make_moe_adapter
        a=make_moe_adapter((n,d,e,route.expert_ids.shape[1],k,h),None,initial=prepared_config)
        prepared_config=a.update_fn(prepared_config,'prepared.linear_transport',args.linear_transport)
    if args.weight_epilogue:
        p=prepared_config.prepared
        p=replace(p,**{s:replace(getattr(p,s),epilogue=args.weight_epilogue,
                               store_vec=4 if args.weight_epilogue=='tma' else 1,
                               epilogue_stages=1) for s in ('gate_dw','down_dw')})
        prepared_config=replace(prepared_config,prepared=p)
    if args.all_tma_outputs:
        from rtx.autotune.moe import make_moe_adapter
        a=make_moe_adapter((n,d,e,route.expert_ids.shape[1],k,h),None,initial=prepared_config)
        prepared_config=a.update_fn(prepared_config,'prepared.bf16_outputs','tma_all')
    def prepared_forward_call():return _run_forward(*operands,prepared_config)
    def prepared_backward_call():return _run_backward(*operands,dy,prepared_config)
    reference=operational_reference(*operands,dy)
    started=time.monotonic()
    actual=(prepared_forward_call(),*prepared_backward_call())
    errors={key:metric(a,b) for key,a,b in zip(('output','dx','gate_dw','down_dw','route_dw'),actual,reference)}
    assert all(v['nrms']<.006 or v['max_abs']<2e-7 for v in errors.values()),errors
    del reference,actual
    print(f'CORRECT cold={time.monotonic()-started:.2f}s {errors}',flush=True)
    bf16=torch.compile(lambda x,gu,down,w:torch_routed_experts(x,gu,down,
        BalancedRouting(route.expert_ids,w,route.inverse)),fullgraph=True,dynamic=False,
        options={'triton.cudagraphs':False})
    y=bf16(x,gu,down,route.route_weights)
    torch.autograd.grad(y,(x,gu,down,route.route_weights),dy);del y
    stream=torch.cuda.Stream();phases={}
    for phase in ('forward','forward_backward'):
        def baseline():
            y=bf16(x,gu,down,route.route_weights)
            return torch.autograd.grad(y,(x,gu,down,route.route_weights),dy) if phase=='forward_backward' else y
        def fused():
            y=_run_forward(*operands,config)
            return (y,*_run_backward(*operands,dy,config)) if phase=='forward_backward' else y
        def prepared():
            y=prepared_forward_call()
            return (y,*prepared_backward_call()) if phase=='forward_backward' else y
        functions={'bf16':baseline,'fused':fused,'prepared':prepared};observations={name:[] for name in functions}
        for iteration in range(args.rounds):
            order=list(functions) if iteration%2==0 else list(reversed(functions))
            for name in order:
                result=measure(functions[name],samples=9,stream=stream)
                observations[name].append(result)
                print(f'MEASURE {phase} {name} {result["median_ms"]*1000:.3f}us',flush=True)
        phases[phase]={name:dict(median_ms=statistics.median(r['median_ms'] for r in values),observations=values)
                       for name,values in observations.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(case=case,tokens=n,errors=errors,phases=phases,
        baseline_config=config.to_dict(),prepared_config=prepared_config.to_dict(),transpose_mode=args.transpose_mode,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__),indent=2))
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function('prepared_forward'):prepared_forward_call()
            with torch.profiler.record_function('prepared_backward'):prepared_backward_call()
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
    print(f'DONE {args.output}',flush=True)


if __name__=='__main__':main()
