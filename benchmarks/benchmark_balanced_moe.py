"""Qualify/tune grouped MXFP8 experts against compiled batched BF16 experts.

Routes are matched and frozen for expert attribution. This is NOT a whole-MoE
or fixed64-router speedup benchmark. Shared FFNs/router are outside its timer.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from rtx.moe import BalancedMoE,DEFAULT_MOE_CONFIG,_selection_spec,_parse_spec,_run_forward,_run_backward,torch_routed_experts
from rtx.routing import BalancedRouting
from rtx._moe_reference import operational_reference
from rtx.autotune.effort import AutotunePolicy
from rtx.autotune.moe import tune_moe,source_revision
from benchmarks.benchmark_balanced_router import measure


def metric(actual,expected):
    assert bool(actual.isfinite().all()) and bool(expected.isfinite().all())
    delta=actual.float()-expected.float()
    return dict(max_abs=delta.abs().max().item(),
                nrms=(delta.norm()/expected.float().norm().clamp_min(1e-12)).item())


def whole_layer(model,x,config,args):
    """Same CuTe router/shared count; routed and shared FFN precision changes."""
    assert model.router.backend in ('auto','cute')
    model.config=config
    model._expert_spec=_selection_spec(config,'off',None,None)
    auxiliary={}
    if args.auxiliary_study:
        from rtx.routing import RoutingKernelConfig
        from rtx.mlp import MXFP8MLPConfig
        row=next(r for r in json.loads(args.auxiliary_study.read_text())['results']
            if r['tokens']==x.shape[0] and r['case']['d_model']==model.d_model
            and r['case']['num_experts']==model.num_experts and r['case']['top_k']==model.top_k
            and r['case']['expert_hidden_dim']==model.expert_hidden_features)
        auxiliary=row['whole_layer']['auxiliary']
        model.router.config=RoutingKernelConfig.from_dict(auxiliary['router']['config'])
        model.router.sinkhorn_config=model.router.config.forward
        model.router.autotune='off';model.router._refresh_selection_spec()
        model.shared_config=MXFP8MLPConfig.from_dict(auxiliary['shared']['config']);model._shared_spec=None
    elif args.auxiliary_effort!='off':
        policy=AutotunePolicy.preset(args.auxiliary_effort,search_schedule='progressive',candidate_mix='mixed')
        root=Path(args.output).parent/'auxiliary'
        started=time.monotonic()
        routing=model.router.autotune_(x,phase='train',tuning_policy=policy,output_dir=root/'routing',progress=print)
        model.router.config=routing.config;model.router.sinkhorn_config=routing.config.forward
        model.router.autotune='off';model.router._refresh_selection_spec()
        auxiliary['router']=dict(config=routing.config.to_dict(),median_ms=routing.median_ms,
                                 cold_seconds=time.monotonic()-started)
        if len(model.shared_down):
            from rtx.autotune.mlp import tune_mxfp8_mlp
            started=time.monotonic()
            shared=tune_mxfp8_mlp(x.detach().requires_grad_(),
                model.shared_gate_up[0].weight.detach().requires_grad_(),
                model.shared_down[0].weight.detach().requires_grad_(),activation='silu',phase='train',
                initial=model.shared_config,tuning_policy=policy,output_dir=root/'shared',
                qualification_rounds=4,progress=print)
            model.shared_config=shared.config;model._shared_spec=None
            auxiliary['shared']=dict(config=shared.config.to_dict(),median_ms=shared.median_ms,
                                     cold_seconds=time.monotonic()-started)
    baseline=BalancedMoE(model.d_model,model.num_experts,model.expert_hidden_features,
        top_k=model.top_k,shared_hidden_features=model.shared_down[0].in_features,
        num_shared_experts=len(model.shared_down),router_dim=model.router.router_dim,
        device=x.device,backend='torch',autotune='off',
        router_options={'backend':'cute','config':model.router.config})
    baseline.load_state_dict(model.state_dict())
    options=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    native_fn,baseline_fn=torch.compile(model,**options),torch.compile(baseline,**options)
    dy=torch.randn_like(x)
    params=[tuple(m.parameters()) for m in (baseline,model)]
    functions=(baseline_fn,native_fn)
    # Compile forward and adjoints before creating the capture stream.
    for fn,p in zip(functions,params):
        torch.autograd.grad(fn(x),(x,*p),dy)
    stream=torch.cuda.Stream()
    phases={}
    for phase in ('forward','forward_backward'):
        observations=[[],[]]
        for round_index in range(args.rounds):
            for index in ((0,1) if round_index%2==0 else (1,0)):
                def call():
                    y=functions[index](x)
                    return torch.autograd.grad(y,(x,*params[index]),dy) if phase=='forward_backward' else y
                measurement=measure(call,samples=args.samples,stream=stream,warmup_ms=args.warmup_ms)
                observations[index].append(measurement)
                print(f"WHOLE n={x.shape[0]} d={x.shape[1]} {phase} "
                      f"{'bf16' if index==0 else 'mxfp8'} {measurement['median_ms']*1000:.3f}us",flush=True)
        times=[statistics.median(o['median_ms'] for o in rows) for rows in observations]
        phases[phase]=dict(bf16_ms=times[0],mxfp8_ms=times[1],speedup=times[0]/times[1],
                           observations=observations)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function('whole_native_forward_backward'):
                y=native_fn(x);torch.autograd.grad(y,(x,*params[1]),dy);torch.cuda.synchronize()
        profile.export_chrome_trace(str(Path(args.output).with_suffix('.whole.trace.json')))
    return dict(scope='complete layer; same CuTe router configuration; one shared SwiGLU',
                router_and_shared_autotune='frozen_reuse' if args.auxiliary_study else args.auxiliary_effort,
                auxiliary=auxiliary,phases=phases)


def run(case,n,args):
    # Every context is a separate baseline compilation. Do not exhaust the
    # per-code-object guard limit by specializing one benchmark closure to all
    # 12 shapes. This isolated-process reset is never used in the library.
    torch.compiler.reset()
    torch.manual_seed(args.seed)
    d,e,k,h=(case[key] for key in ('d_model','num_experts','top_k','expert_hidden_dim'))
    model=BalancedMoE(d,e,h,top_k=k,shared_hidden_features=case['shared_hidden_dim'],
                      num_shared_experts=case['num_shared_experts'],device='cuda',autotune='off')
    with torch.no_grad():model.router.projection_weight.normal_(std=.02)
    x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    with torch.no_grad():route=model.router(x)
    route=BalancedRouting(route.expert_ids,route.route_weights.detach().requires_grad_(),route.inverse)
    # Expert attribution differentiates packed operands, not VirtualLayer's
    # optimizer bridge. Never retain a default-stream nonleaf autograd node
    # across side-stream warmup/capture.
    gu=model.gate_up.weight.detach().requires_grad_()
    down=model.down.weight.detach().requires_grad_()
    operands=(x,gu,down,*route)
    dy=torch.randn(x.shape,device='cuda',dtype=torch.float32)
    config=DEFAULT_MOE_CONFIG
    if args.config_study:
        study=json.loads(args.config_study.read_text())
        previous=next(r for r in study.get('results',[study])
                      if r['case']['id']==case['id'] and r['tokens']==n)
        from rtx.moe import MXFP8MoEConfig
        config=MXFP8MoEConfig.from_dict(previous.get('prepared_config',previous.get('config')))
    if args.initial_family:
        config=replace(config,family=args.initial_family,gather_mode='warp')
    tuning=None
    if args.effort!='off':
        policy=AutotunePolicy.preset(args.effort,search_schedule='progressive',candidate_mix='mixed')
        if args.trials:policy=AutotunePolicy(**{**asdict(policy),'max_trials':args.trials})
        spec=_parse_spec(_selection_spec(config,'balanced',policy,Path(args.output).parent/'tuning'))
        started=time.monotonic()
        result=tune_moe(operands,spec,training=True,progress=print)
        config=result.config
        tuning=dict(cold_seconds=time.monotonic()-started,config=config.to_dict(),
                    median_ms=result.median_ms,evidence=result.tuning_evidence)
    expected=operational_reference(*operands,dy)
    output=_run_forward(*operands,config)
    gradients=_run_backward(*operands,dy,config)
    errors={name:metric(a,b) for name,a,b in zip(('output','dx','gate_dw','down_dw','route_dw'),
                                                (output,*gradients),expected)}
    for name,value in errors.items():
        limit=.003 if name=='output' else .006
        assert value['nrms']<limit or value['max_abs']<2e-7,(case['id'],n,name,value)
    baseline=torch.compile(lambda x,gu,down,w:torch_routed_experts(
        x,gu,down,BalancedRouting(route.expert_ids,w,route.inverse)),
        fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    by=baseline(x,gu,down,route.route_weights)
    bg=torch.autograd.grad(by,(x,gu,down,route.route_weights),dy)
    bf16_errors={name:metric(a,b) for name,a,b in zip(('output','dx','gate_dw','down_dw','route_dw'),
                                                     (output,*gradients),(by,*bg))}
    del output,gradients,expected,by,bg
    print(f"CORRECTNESS {case['id']} n={n} capacity={route.expert_ids.shape[1]} passed",flush=True)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    phases={}
    for phase in ('forward','forward_backward'):
        observations=[[],[]]
        def native():
            y=_run_forward(*operands,config)
            return (y,*_run_backward(*operands,dy,config)) if phase=='forward_backward' else y
        def torch_call():
            y=baseline(x,gu,down,route.route_weights)
            return torch.autograd.grad(y,(x,gu,down,route.route_weights),dy) if phase=='forward_backward' else y
        for round_index in range(args.rounds):
            for index in ((0,1) if round_index%2==0 else (1,0)):
                result=measure((torch_call,native)[index],samples=args.samples,stream=stream,warmup_ms=args.warmup_ms)
                observations[index].append(result)
                with Path(args.output).with_suffix('.measurements.jsonl').open('a') as journal:
                    journal.write(json.dumps(dict(case=case,tokens=n,phase=phase,
                        implementation='bf16' if index==0 else 'mxfp8',round=round_index,
                        config=config.to_dict() if index else None,measurement=result))+'\n')
                print(f"MEASURE {case['id']} n={n} {phase} "
                      f"{'bf16' if index==0 else 'mxfp8'} {result['median_ms']*1000:.3f}us "
                      f"stable={result['sampling']['collection']['stationary']}",flush=True)
        medians=[statistics.median(row['median_ms'] for row in rows) for rows in observations]
        phases[phase]=dict(bf16_ms=medians[0],mxfp8_ms=medians[1],speedup=medians[0]/medians[1],
            stationary=all(row['sampling']['collection']['stationary'] for rows in observations for row in rows),
            observations=observations)
    if args.profile:
        # Dedicated diagnostic iteration, fully warmed and outside timings.
        with torch.cuda.stream(stream):
            for _ in range(2):_run_forward(*operands,config);_run_backward(*operands,dy,config)
        stream.synchronize()
        destination=Path(args.output).parent/f"{case['id']}-n{n}-trace.json"
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function('routed_forward'):_run_forward(*operands,config)
            with torch.profiler.record_function('routed_backward'):_run_backward(*operands,dy,config)
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(destination))
    complete=whole_layer(model,x,config,args) if args.whole_layer else None
    return dict(case=case,tokens=n,capacity=route.expert_ids.shape[1],whole_layer=complete,
                operational_errors=errors,bf16_errors=bf16_errors,config=config.to_dict(),
                tuning=tuning,phases=phases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',default='benchmarks/balanced_moe_cases.json',type=Path)
    p.add_argument('--case-ids',nargs='+')
    p.add_argument('--tokens',nargs='+',type=int)
    p.add_argument('--effort',choices=('off','quick','balanced','thorough','extensive'),default='off')
    p.add_argument('--trials',type=int)
    p.add_argument('--initial-family',choices=('grouped_staged','grouped_onchip','grouped_prepared'))
    p.add_argument('--config-study',type=Path,help='Freeze a previously verified expert schedule for A/B attribution')
    p.add_argument('--auxiliary-effort',choices=('off','quick','balanced','thorough','extensive'),default='off')
    p.add_argument('--auxiliary-study',type=Path,help='Freeze a matched previous router/shared winner for expert-only A/B changes')
    p.add_argument('--samples',type=int,default=11)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--warmup-ms',type=float,default=200)
    p.add_argument('--seed',type=int,default=819)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--whole-layer',action='store_true',help='Also compare full compiled modules with the same CuTe router')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    suite=json.loads(args.cases.read_text())
    cases=[case for case in suite['cases'] if not args.case_ids or case['id'] in args.case_ids]
    if not cases:p.error('No matching cases')
    result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        source_revision=source_revision(),benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='routed experts only; matched frozen routes; no shared/router time',
        configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},results=[])
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    for case in cases:
        for n in (args.tokens or suite['token_counts']):
            result['results'].append(run(case,n,args))
            destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'DONE {destination}',flush=True)


if __name__=='__main__':main()
