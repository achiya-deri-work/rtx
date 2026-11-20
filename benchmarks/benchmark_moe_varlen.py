"""Packed independent MoE qualification and profiling, never timing compilation.

Run each source/configuration in a fresh process. BF16 checkpointed and saved
FFNs are reported separately; both keep the same native packed router. Phase
timings are standalone measurements and must NOT be added to whole-layer time.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


def scaled_case(case,scale=1.,alignment=64):
    """Change only routed H; keep a truthful effective workload identity."""
    if not math.isfinite(scale) or scale<=0:raise ValueError('Positive finite expert hidden scale required')
    if type(alignment) is not int or alignment<1:raise ValueError('Positive integer hidden alignment required')
    result=dict(case)
    h=math.ceil(case['expert_hidden_dim']*scale/alignment)*alignment
    result['expert_hidden_dim']=h
    if h!=case['expert_hidden_dim']:
        result['id']=f"d{case['d_model']}_e{case['num_experts']}_k{case['top_k']}_h{h}"
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case-id',default='d384_e24_k6_h128')
    p.add_argument('--case-spec',help='JSON workload overrides; benchmark-only, not kernel dispatch rules')
    p.add_argument('--expert-phases',action='store_true',help='Also measure routed experts with frozen differentiable route weights')
    p.add_argument('--shared-phases',action='store_true',help='Also measure the actual shared MLP versus independently checkpointed BF16')
    p.add_argument('--trace-phases',nargs='*',default=[],help='After ordinary timing, export one warmed graph replay trace per named phase')
    p.add_argument('--save-expert-fixture',type=Path,help='Save real selected routes/operands for profiler processes that must avoid router graph injection')
    p.add_argument('--prepare-only',action='store_true',help='Write --save-expert-fixture outside profiling, then exit without timing')
    p.add_argument('--expert-hidden-scale',type=float,default=1.)
    p.add_argument('--hidden-alignment',type=int,default=64)
    p.add_argument('--seed',type=int,default=17391)
    p.add_argument('--input-profile',choices=('normal','uniform','outliers','correlated'),default='normal',
        help='Benchmark-only input distribution; never changes kernel precision or policy')
    p.add_argument('--bf16-numerics',action='store_true',
        help='Report output/all-gradient discrepancy versus matched checkpointed BF16; not an exact-kernel oracle')
    p.add_argument('--lengths',nargs='+',type=int,default=[512,2000,768,4096])
    p.add_argument('--layout',choices=('packed','flat'),default='packed',
        help='Independent sample pools, or true regular API with cu_seqlens=None')
    p.add_argument('--bf16-compact',action='store_true',help='Include shape-bound compacted BF16 checkpoint control')
    p.add_argument('--stable-inputs',action='store_true',help='Independent input/router/shared RNG streams for controlled geometry sweeps')
    p.add_argument('--stable-experts',action='store_true',help='Independent named expert RNG streams; shared-width changes cannot perturb routed weights')
    p.add_argument('--router-config',type=Path)
    p.add_argument('--affinity-layout',choices=('auto','scalar','row_broadcast'),
        help='Affinity execution override; otherwise restore frozen sidecar metadata or use auto')
    p.add_argument('--router-overrides',default='{}',help='JSON execution-field overrides for controlled ablations')
    p.add_argument('--expert-config',type=Path)
    p.add_argument('--expert-overrides',default='{}',help='Nested JSON execution-field overrides for controlled ablations')
    p.add_argument('--shared-config',type=Path)
    p.add_argument('--tune-router-trials',type=int,default=0)
    p.add_argument('--tune-expert-trials',type=int,default=0)
    p.add_argument('--tune-shared-trials',type=int,default=0)
    p.add_argument('--tune-seconds',type=float,default=120.)
    p.add_argument('--tuning-dir',type=Path,help='Shared residual root for progressive effort waves')
    p.add_argument('--search-schedule',choices=('legacy','progressive'),default='legacy')
    p.add_argument('--candidate-mix',choices=('legacy','mixed'),default='legacy')
    p.add_argument('--prior-artifact',type=Path,help='Portable-prior JSON(.gz), not a pretrain manifest directory; never installed')
    p.add_argument('--prior-search',choices=('hybrid','retrieval','ranker'),default='hybrid')
    p.add_argument('--region-coverage',action=argparse.BooleanOptionalAction,default=True,
                   help='Adapter-owned early structural coverage; --no-region-coverage is the ordinary-search control')
    p.add_argument('--allow-research-prior',action='store_true',help='Explicitly allow unqualified prior for a prospective comparison')
    p.add_argument('--samples',type=int,default=9)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--phases',nargs='+',default=['router_forward','router_step','transport_forward',
        'transport_backward','hard_forward','metadata_forward','metadata_step','mxfp8_forward',
        'mxfp8_step','bf16_checkpoint_step','bf16_saved_step'])
    p.add_argument('--profile',choices=['none','torch','nsys'],default='none')
    p.add_argument('--profile-phase',default='mxfp8_step')
    p.add_argument('--profile-replay',action='store_true',help='Profile a warmed outer graph instead of synchronized host launches')
    p.add_argument('--reference',type=Path)
    p.add_argument('--save-reference',type=Path)
    p.add_argument('--oracle',action='store_true',help='Independent FP64 quantized expert forward/backward on actual routes')
    p.add_argument('--max-autotune',action='store_true',help='Also enable Inductor GEMM/pointwise autotuning in both baselines and native wrappers')
    return p


def statistics_ms(values):
    median=statistics.median(values)
    return dict(median_ms=median,min_ms=min(values),max_ms=max(values),samples_ms=values,
        relative_mad=statistics.median(abs(v-median) for v in values)/max(median,1e-12))


def profile_input(value,profile):
    """Cold synthetic controls, applied after identical RNG consumption."""
    if profile=='normal':return value
    if profile=='uniform':
        # Gaussian CDF makes an approximately RMS-matched bounded input without
        # consuming another RNG stream or changing the model initialization.
        return ((value.float()/.3/math.sqrt(2)).erf()*(math.sqrt(3)*.3)).to(value.dtype)
    if profile=='outliers':
        result=value.clone()
        result.view(-1)[::1024]*=16
        return result
    if profile=='correlated':
        return ((value[:1].float()+.1*value.float())/math.sqrt(1.01)).to(value.dtype)
    raise ValueError('Unknown input profile')


def discrepancy(actual,reference):
    """Observational precision difference, not an implementation-correctness gate."""
    import torch
    a=actual.detach().float();r=reference.detach().float()
    if a.shape!=r.shape or not bool(a.isfinite().all()) or not bool(r.isfinite().all()):
        raise AssertionError('Nonfinite or incompatible numerical comparison')
    delta=a-r
    rn=r.norm();an=a.norm()
    return dict(nrms=float(delta.norm()/rn.clamp_min(1e-12)),
        rms_error=float(delta.square().mean().sqrt()),reference_rms=float(r.square().mean().sqrt()),
        max_abs=float(delta.abs().max()),
        cosine=float((a.flatten()@r.flatten())/(an*rn).clamp_min(1e-24)),all_finite=True)


def initialize_fixture(construct,case,initialization_case,lengths,*,layout,seed,
                       input_profile='normal',stable_inputs=False,stable_experts=False,
                       router_std=.05,device='cuda'):
    """Cold fixture replay shared by timing and matched factorial harnesses."""
    import torch
    torch.manual_seed(seed)
    model=construct(initialization_case)
    with torch.no_grad():
        generator=torch.Generator(device=device).manual_seed(seed+8191) if stable_inputs else None
        model.router.projection_weight.normal_(std=router_std,generator=generator)
        if stable_inputs:
            for index,parameter in enumerate((*model.shared_gate_up.parameters(),*model.shared_down.parameters())):
                torch.nn.init.trunc_normal_(parameter,std=.02,a=-.04,b=.04,
                    generator=torch.Generator(device=device).manual_seed(seed+16381+index))
    generator=torch.Generator(device=device).manual_seed(seed+104729) if stable_inputs else None
    x=(torch.randn(sum(lengths),case['d_model'],device=device,dtype=torch.bfloat16,generator=generator)*.3).requires_grad_()
    x=profile_input(x.detach(),input_profile).requires_grad_()
    cu=(torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],device=device,dtype=torch.int32)
        if layout=='packed' else None)
    dy=(torch.randn(x.shape,device=x.device,dtype=x.dtype,
        generator=torch.Generator(device=device).manual_seed(seed+130363)) if stable_inputs else torch.randn_like(x))
    if case['expert_hidden_dim']!=initialization_case['expert_hidden_dim']:
        wider=construct(case)
        wider.router.load_state_dict(model.router.state_dict())
        wider.shared_gate_up.load_state_dict(model.shared_gate_up.state_dict())
        wider.shared_down.load_state_dict(model.shared_down.state_dict())
        model=wider
    if stable_experts:
        with torch.no_grad():
            for prefix,module in (('gate_up',model.gate_up),('down',model.down)):
                for name,parameter in module.named_parameters():
                    offset=int.from_bytes(hashlib.sha256(f'{prefix}.{name}'.encode()).digest()[:4],'little')
                    torch.nn.init.trunc_normal_(parameter,std=.02,a=-.04,b=.04,
                        generator=torch.Generator(device=device).manual_seed(seed+200000+offset))
    return model,x,cu,dy


def compile_options(max_autotune):
    # Keep the complete candidate set and nested process isolation. VirtualLayer
    # now exposes its full-span input so nested benchmark allocations are safe.
    # The collection supervisor separately fails closed on any device fault.
    options={'triton.cudagraphs':False,'max_autotune':max_autotune,
             'autotune_in_subproc':max_autotune}
    if max_autotune:
        import torch._inductor
        supported=set(torch._inductor.list_options())
        requested=dict(epilogue_fusion=True,prologue_fusion=True,max_autotune_gemm=True,
            aggressive_fusion=True,max_autotune_pointwise=True,online_softmax=True,
            layout_optimization=True,b2b_gemm_pass=True,joint_graph_constant_folding=True,
            benchmark_epilogue_fusion=True,coordinate_descent_tuning=False,
            memory_planning=True,inplace_buffers=True,allow_buffer_reuse=True,
            reorder_for_locality=True,reorder_for_compute_comm_overlap=True,
            group_fusion=True,use_fast_math=True,freezing=False)
        requested.update({'triton.autotune_cublasLt':False,'auto_chunker.enable':True,
                          'triton.persistent_reductions':True})
        options.update({k:v for k,v in requested.items() if k in supported})
        omitted=sorted(set(requested)-supported)
        if omitted:print('COMPILE unsupported options omitted:',','.join(omitted),flush=True)
    return dict(fullgraph=True,dynamic=False,options=options)


def main():
    args=parser().parse_args()
    if not args.lengths or min(args.lengths)<0 or sum(args.lengths)<1:
        raise ValueError('Nonnegative sequence lengths with positive total required')
    if min(args.samples,args.rounds)<1:raise ValueError('Positive samples/rounds required')
    source=args.source.resolve();sys.path.insert(0,str(source))
    import torch
    from rtx.moe import BalancedMoE,DEFAULT_MOE_CONFIG,MXFP8MoEConfig,_selection_spec
    from rtx.mlp import MXFP8MLPConfig
    from rtx.routing import RoutingKernelConfig
    from rtx import _routing_varlen as execution
    from rtx.autotune import AutotunePolicy
    from benchmarks.benchmark_moe_checkpoint import BF16FFNReference
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    import rtx
    if not Path(rtx.__file__).resolve().is_relative_to(source):raise AssertionError(rtx.__file__)
    # Create autograd nodes, warm the compiled programs and capture on the
    # same non-default stream. Torch 2.13 correctly rejects stale default-
    # stream AccumulateGrad nodes kept alive by compiled checkpoint graphs.
    # This is private to the standalone benchmark, never a library/user flag.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    args.output=args.output.resolve();args.output.parent.mkdir(parents=True,exist_ok=True)
    def write(record):
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temporary.replace(args.output)
    cases=json.loads((source/'benchmarks/balanced_moe_cases.json').read_text())['cases']
    base_case=next(c for c in cases if c['id']==args.case_id)
    if args.case_spec:
        patch=json.loads(args.case_spec)
        allowed={'d_model','num_experts','top_k','expert_hidden_dim','shared_hidden_dim'}
        if not isinstance(patch,dict) or set(patch)-allowed:
            raise ValueError('Unknown workload field in --case-spec')
        if any(type(v) is not int or v<1 for v in patch.values()):
            raise ValueError('Positive integer workload dimensions required')
        base_case={**base_case,**patch}
        if base_case['top_k']>base_case['num_experts']:raise ValueError('K exceeds E')
        base_case['id']='d{d_model}_e{num_experts}_k{top_k}_h{expert_hidden_dim}_s{shared_hidden_dim}'.format(**base_case)
    case=scaled_case(base_case,args.expert_hidden_scale,args.hidden_alignment)
    # Hidden-width sweeps must not perturb X, dY, router or shared weights
    # merely because creating a wider expert consumed more RNG values.
    initialization_case={**base_case,'expert_hidden_dim':128} if args.case_spec else base_case
    router=RoutingKernelConfig(qb_backend='native_column',metadata_backend='cute',quota_threshold=0)
    router_document=json.loads(args.router_config.read_text()) if args.router_config else {}
    if args.router_config:router=RoutingKernelConfig.from_dict(router_document.get('config',router_document))
    affinity_layout=args.affinity_layout or router_document.get('affinity_layout','auto')
    router=RoutingKernelConfig.from_dict({**router.to_dict(),**json.loads(args.router_overrides)})
    expert=DEFAULT_MOE_CONFIG
    if args.expert_config:expert=MXFP8MoEConfig.from_dict(json.loads(args.expert_config.read_text()).get('config',json.loads(args.expert_config.read_text())))
    def merge(base,patch):
        return {key:merge(base[key],value) if isinstance(value,dict) and isinstance(base.get(key),dict) else value
                for key,value in {**base,**patch}.items()}
    expert=MXFP8MoEConfig.from_dict(merge(expert.to_dict(),json.loads(args.expert_overrides)))
    shared=None
    if args.shared_config:shared=MXFP8MLPConfig.from_dict(json.loads(args.shared_config.read_text()).get('config',json.loads(args.shared_config.read_text())))
    shared_changes=[]
    if shared is not None:
        from benchmarks.profile_moe import retarget_auxiliary
        retargeted,shared_changes=retarget_auxiliary({'shared':{'config':shared.to_dict()}},sum(args.lengths))
        shared=MXFP8MLPConfig.from_dict(retargeted['shared']['config'])
    def construct(c):
        return BalancedMoE(c['d_model'],c['num_experts'],c['expert_hidden_dim'],top_k=c['top_k'],
            shared_hidden_features=c['shared_hidden_dim'],device='cuda',autotune='off',config=expert,shared_config=shared,
            router_options={'backend':'cute','autotune':'off','config':router,'affinity_layout':affinity_layout})
    # Replay the original initialization before changing H. Thus input, dY,
    # router and shared parameters match the original-width study exactly;
    # extra routed weights cannot perturb its random-number stream.
    model,x,cu,dy=initialize_fixture(construct,case,initialization_case,args.lengths,
        layout=args.layout,seed=args.seed,input_profile=args.input_profile,
        stable_inputs=args.stable_inputs,stable_experts=args.stable_experts)
    parameters=tuple(model.parameters());leaves=(x,*parameters)
    from rtx.kernels.mxfp8 import MXFP8Problem
    for name,problem in (('gate_backward',MXFP8Problem(x.shape[0],2*case['shared_hidden_dim'],case['d_model'])),
                         ('down_backward',MXFP8Problem(x.shape[0],case['d_model'],case['shared_hidden_dim']))):
        configured=getattr(model.shared_config,name)
        if configured is not None and (reason:=configured.implementation_rejection(problem)):
            raise ValueError(f'Inherited shared {name} needs a legal retarget before tuning: {reason}')
    policy=lambda trials:AutotunePolicy.preset('balanced',max_trials=trials,max_seconds=args.tune_seconds,
        samples=5,calls_per_sample=1,confirmation_repeats=0,
        search_schedule=args.search_schedule,candidate_mix=args.candidate_mix,
        prior_artifact=None if args.prior_artifact is None else str(args.prior_artifact.resolve()),
        prior_search=args.prior_search,region_coverage=args.region_coverage,
        allow_research_prior=args.allow_research_prior)
    tuning_dir=args.tuning_dir or args.output.parent/'tuning'
    tuning_counts={}
    if args.tune_router_trials:
        invocation_started=time.monotonic()
        tuned=model.router.autotune_(x,cu_seqlens=cu,tuning_policy=policy(args.tune_router_trials),
                              output_dir=tuning_dir,progress=lambda s:print(s,flush=True))
        # Freeze the independently selected affinity plan before disabling
        # benchmark tuning. Merely setting mode=off would restore scalar and
        # silently discard an accepted row-broadcast decision.
        if model.router.affinity_layout=='auto':
            from rtx._routing_affinity import identity,grad_mask,_math_context
            from rtx._moe_selection import bound_config
            inputs=model.router._affinity_inputs(x)
            chosen=bound_config(*identity(inputs,model.router._selection_spec,grad_mask(inputs),_math_context()))
            if chosen is not None:model.router.affinity_layout=chosen
        model.router.config=tuned.config;model.router.autotune='off';model.router._refresh_selection_spec()
        tuning_counts['router']=dict(evaluated_trials=tuned.evaluated_trials,elapsed_s=tuned.elapsed_s,
                                    invocation_s=time.monotonic()-invocation_started)
    if args.tune_expert_trials:
        invocation_started=time.monotonic()
        tuned=model.autotune_experts_(x,cu_seqlens=cu,tuning_policy=policy(args.tune_expert_trials),
                               output_dir=tuning_dir,progress=lambda s:print(s,flush=True))
        invocation_seconds=time.monotonic()-invocation_started
        model.config=tuned.config;model.autotune='off';model._expert_spec=_selection_spec(tuned.config,'off',None,None)
        tuning_counts['expert']=dict(evaluated_trials=tuned.evaluated_trials,elapsed_s=tuned.elapsed_s,
            invocation_s=invocation_seconds,tuning_evidence=tuned.tuning_evidence)
    if args.tune_shared_trials:
        from rtx.autotune.mlp import tune_mxfp8_mlp
        invocation_started=time.monotonic()
        tuned=tune_mxfp8_mlp(x.detach().requires_grad_(),model.shared_gate_up[0].weight.detach().requires_grad_(),
            model.shared_down[0].weight.detach().requires_grad_(),initial=model.shared_config,phase='train',
            tuning_policy=policy(args.tune_shared_trials),output_dir=args.output.parent/'shared-tuning',
            qualification_rounds=4,progress=lambda s:print(s,flush=True))
        model.shared_config=tuned.config
        tuning_counts['shared']=dict(evaluated_trials=tuned.evaluated_trials,elapsed_s=tuned.elapsed_s,
            invocation_s=time.monotonic()-invocation_started,tuning_evidence=tuned.tuning_evidence)
    model._shared_spec=None
    for name,c in (('router',model.router.config),('expert',model.config),('shared',model.shared_config)):
        metadata={'affinity_layout':model.router.affinity_layout} if name=='router' else {}
        args.output.with_suffix('.'+name+'.json').write_text(json.dumps(dict(config=c.to_dict(),**metadata),indent=2)+'\n')
    router=model.router.config
    options=compile_options(args.max_autotune)
    native=torch.compile(model,**options);route_fn=torch.compile(model.router,**options)
    route=route_fn(x,cu_seqlens=cu)
    upstream=torch.randn_like(route.route_weights)
    route_identity=dict(expert_ids=route.expert_ids.detach().cpu(),inverse=route.inverse.detach().cpu())
    if args.save_expert_fixture:
        args.save_expert_fixture.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(case={**case,'id':args.case_id},tokens=x.shape[0],
            **{name:value.detach().cpu() for name,value in dict(x=x,gu=model.gate_up.weight,
                down=model.down.weight,ids=route.expert_ids,weights=route.route_weights,
                inverse=route.inverse,dy=dy.float()).items()}),args.save_expert_fixture)
    if args.prepare_only:
        if not args.save_expert_fixture:raise ValueError('--prepare-only requires --save-expert-fixture')
        write(dict(status='fixture_prepared',fixture=str(args.save_expert_fixture),case=case))
        return
    del route
    route_leaves=(x,*model.router.parameters())
    def route_forward():return route_fn(x,cu_seqlens=cu)
    def route_step():
        result=route_forward()
        return result,torch.autograd.grad(result.route_weights,route_leaves,upstream)
    calls=dict(router_forward=route_forward,router_step=route_step,
        mxfp8_forward=lambda:native(x,cu_seqlens=cu))
    if cu is not None:
        scores=model.router.log_affinities(x).detach()
        key=execution.layout(x.shape[0],len(args.lengths),case['num_experts'],case['top_k'],
                             model.router.sinkhorn_iterations,tuple(router.to_token()))
        transport=execution.transport_forward(scores,cu,key)
        logp,history,scheduled,physical,sequence,local,rowmap,probabilities=transport
        logp=logp.detach().requires_grad_()
        score_gradient=torch.randn_like(logp)
        assignments=torch.empty((key[4],key[3]),device=x.device,dtype=torch.int64)
        status=torch.empty(2*len(args.lengths)*case['num_experts'],device=x.device,dtype=torch.int64)
        def hard():
            execution.hard_execute([probabilities,scheduled,sequence,local],assignments,status,execution._flat(key),
                model.router.dual_iterations,model.router.repair_rounds,router.to_token(),args.seed)
        hard()
        def metadata():return execution.SelectedMetadata.apply(logp,assignments,scheduled,physical,sequence,rowmap,key)
        def metadata_step():
            ids,w,inv=metadata()
            return ids,w,inv,torch.autograd.grad(w,logp,upstream)[0]
        calls.update(transport_forward=lambda:execution.transport_forward(scores,cu,key),
            transport_backward=lambda:execution.transport_backward(score_gradient,history,scheduled,sequence,rowmap,key),
            hard_forward=hard,metadata_forward=metadata,metadata_step=metadata_step)
        scheduled_bound=key[4]
    else:
        period=case['num_experts']//math.gcd(case['num_experts'],case['top_k'])
        scheduled_bound=(x.shape[0]+period-1)//period*period
    def step(fn):
        for leaf in leaves:leaf.grad=None
        result=fn(x,cu_seqlens=cu)
        result.backward(dy)
        gradients=tuple(leaf.grad for leaf in leaves)
        if any(g is None for g in gradients):raise AssertionError('Missing optimizer-visible gradient')
        return result,gradients
    calls['mxfp8_step']=lambda:step(native)
    references=[('bf16_checkpoint',True,'native'),('bf16_saved',False,'native')]
    if args.bf16_compact:
        references.extend((('bf16_compact_checkpoint',True,'compact'),
                           ('bf16_compact_saved',False,'compact')))
    for mode,ckp,expert_layout in references:
        reference=torch.compile(BF16FFNReference(model,checkpoint_ffns=ckp,expert_layout=expert_layout),**options)
        calls[mode+'_step']=lambda fn=reference:step(fn)
        calls[mode+'_forward']=lambda fn=reference:fn(x,cu_seqlens=cu)
    backward_states={};backward_saved=[]
    def prepare_backward():
        # Keep input saved-tensor storage alive for a backward-only CUDA
        # graph, without changing AOT's normal donated-buffer policy.
        backward_saved.clear()
        def pin(tensor):backward_saved.append(tensor);return tensor
        with torch.autograd.graph.saved_tensors_hooks(pin,lambda tensor:tensor):
            backward_states['mxfp8']=native(x,cu_seqlens=cu)
    def backward_only(fn,name):
        for leaf in leaves:leaf.grad=None
        value=backward_states.pop(name)
        value.backward(dy,retain_graph=False)
        gradients=tuple(leaf.grad for leaf in leaves)
        if any(g is None for g in gradients):raise AssertionError('Missing backward-only gradient')
        if not torch.cuda.is_current_stream_capturing():backward_saved.clear()
        return value,gradients
    calls['mxfp8_backward']=lambda:backward_only(native,'mxfp8')
    def prepare_phase(phase):
        if phase=='mxfp8_backward':prepare_backward()
    expert_plan=None
    if args.expert_phases:
        from rtx.moe import (_dispatch_experts,_packed_expert_spec,_regular_expert_spec,
                             _initial_config,torch_routed_experts)
        from rtx.routing import BalancedRouting
        from torch.utils.checkpoint import checkpoint
        frozen=route_fn(x,cu_seqlens=cu)
        frozen=BalancedRouting(frozen.expert_ids.detach(),frozen.route_weights.detach().requires_grad_(),frozen.inverse.detach())
        expert_spec=_selection_spec(model.config,'off',None,None)
        expert_spec=(_packed_expert_spec(expert_spec,len(args.lengths)) if cu is not None
                     else _regular_expert_spec(expert_spec))
        resolved_expert=_initial_config(expert_spec)
        expert_leaves=(x,*model.gate_up.parameters(),*model.down.parameters(),frozen.route_weights)
        def expert_forward(value,weights):
            return _dispatch_experts(value,model.gate_up.weight,model.down.weight,
                frozen.expert_ids,weights,frozen.inverse,expert_spec,True)
        def expert_reference(value,weights):
            def core(v,g,o,w):
                return torch_routed_experts(v,g,o,BalancedRouting(frozen.expert_ids,w,frozen.inverse))
            return checkpoint(core,value,model.gate_up.weight,model.down.weight,weights,use_reentrant=True)
        expert_native=torch.compile(expert_forward,**options)
        expert_bf16=torch.compile(expert_reference,**options)
        def expert_step(fn):
            for leaf in expert_leaves:leaf.grad=None
            value=fn(x,frozen.route_weights)
            value.backward(dy.float())
            grads=tuple(v.grad for v in expert_leaves)
            if any(v is None for v in grads):raise AssertionError('Missing expert gradient')
            return value,grads
        calls['expert_forward']=lambda:expert_native(x,frozen.route_weights)
        calls['expert_step']=lambda:expert_step(expert_native)
        calls['expert_bf16_checkpoint_step']=lambda:expert_step(expert_bf16)
        # Static launch inventory is not measured traffic or occupancy.
        from rtx._moe_execution import Builder
        shape=(x.shape[0],x.shape[1],case['num_experts'],frozen.expert_ids.shape[1],case['top_k'],case['expert_hidden_dim'])
        expert_plan=dict(shape=list(shape),physical_capacity=(model.config.prepared.physical_capacity(shape[3])
            if model.config.family=='grouped_prepared' else shape[3]),
            stages={phase:[dict(module=m,symbol=s,stage_parameters=str(p))
                for m,s,p,a in Builder(shape,resolved_expert,phase).build().steps]
                for phase in ('forward','backward')})
    if args.shared_phases:
        from rtx.mlp import mxfp8_mlp
        from torch.utils.checkpoint import checkpoint
        shared_leaves=(x,*model.shared_gate_up.parameters(),*model.shared_down.parameters())
        def shared_forward(value):
            outputs=[mxfp8_mlp(value,g.weight,o.weight,config=model.shared_config)
                     for g,o in zip(model.shared_gate_up,model.shared_down)]
            return sum(v.float() for v in outputs).bfloat16()
        def shared_reference(value):
            outputs=[checkpoint(BF16FFNReference.shared,value,g.weight,o.weight,use_reentrant=True)
                     for g,o in zip(model.shared_gate_up,model.shared_down)]
            return sum(v.float() for v in outputs).bfloat16()
        shared_native=torch.compile(shared_forward,**options)
        shared_bf16=torch.compile(shared_reference,**options)
        def shared_step(fn):
            for leaf in shared_leaves:leaf.grad=None
            value=fn(x);value.backward(dy)
            gradients=tuple(v.grad for v in shared_leaves)
            if any(v is None for v in gradients):raise AssertionError('Missing shared gradient')
            return value,gradients
        calls['shared_step']=lambda:shared_step(shared_native)
        calls['shared_bf16_checkpoint_step']=lambda:shared_step(shared_bf16)
    selected=list(dict.fromkeys(args.phases))
    if set(selected)-set(calls):raise ValueError(('Unknown phases',set(selected)-set(calls)))
    digest=hashlib.sha256()
    for path in sorted((source/'rtx').rglob('*.py')):digest.update(str(path.relative_to(source)).encode());digest.update(path.read_bytes())
    def tensor_digest(value):
        return hashlib.sha256(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    record=dict(status='running',case=case,base_case=base_case,initialization_case=initialization_case,
        expert_hidden_scale=args.expert_hidden_scale,hidden_alignment=args.hidden_alignment,
        lengths=args.lengths,layout=args.layout,balancing_pools=len(args.lengths) if cu is not None else 1,
        total_tokens=x.shape[0],scheduled_bound=scheduled_bound,
        expert_capacity_bound=route_identity['expert_ids'].shape[1],source_digest=digest.hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source=str(source),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        gpu_sms=torch.cuda.get_device_properties(x.device).multi_processor_count,
        seed=args.seed,input_profile=args.input_profile,stable_inputs=args.stable_inputs,stable_experts=args.stable_experts,
        router_config=router.to_dict(),expert_config=model.config.to_dict(),shared_config=model.shared_config.to_dict(),
        router_affinity_layout=model.router.affinity_layout,
        inductor_max_autotune=args.max_autotune,compile_options=options,
        inductor_autotune_in_subproc=options['options']['autotune_in_subproc'],
        tuning_budgets=dict(expert=args.tune_expert_trials,router=args.tune_router_trials,
            shared=args.tune_shared_trials,seconds=args.tune_seconds),
        search_schedule=args.search_schedule,candidate_mix=args.candidate_mix,
        prior_artifact=None if args.prior_artifact is None else str(args.prior_artifact.resolve()),
        allow_research_prior=args.allow_research_prior,
        tuning_counts=tuning_counts,
        expert_plan=expert_plan,
        contract='same router; BF16 saved/checkpointed FFNs separately; actual backward to all leaves with set_to_none; ordinary graph replay timing; phases not additive; excludes optimizer',
        shared_partition_adaptations=shared_changes,
        timings={},validation={},identities=dict(input=tensor_digest(x),upstream=tensor_digest(dy),
            parameters={name:tensor_digest(value) for name,value in model.named_parameters()},
            routing={name:tensor_digest(value) for name,value in route_identity.items()}))
    write(record)
    if args.oracle:
        from rtx.moe import _run_forward,_run_backward
        from rtx._moe_reference import operational_reference
        with torch.no_grad():
            route=route_fn(x,cu_seqlens=cu)
            operands=(x,model.gate_up.weight,model.down.weight,*route)
            upstream_fp32=dy.float()  # Private expert adjoint ABI; public Y remains BF16.
            actual=(_run_forward(*operands,model.config),*_run_backward(*operands,upstream_fp32,model.config))
            expected=operational_reference(*operands,upstream_fp32)
            errors={}
            for name,a,b in zip(('output','input','gate_dw','down_dw','route_dw'),actual,expected):
                if not bool(a.isfinite().all()) or not bool(b.isfinite().all()):raise AssertionError(('Nonfinite oracle',name))
                delta=a.float()-b.float();nrms=float(delta.norm()/b.float().norm().clamp_min(1e-12))
                maximum=float(delta.abs().max());errors[name]=dict(nrms=nrms,max_abs=maximum)
                if nrms>(.003 if name=='output' else .006) and maximum>2e-7:raise AssertionError((name,nrms,maximum))
            record['validation']['independent_quantized_fp64_experts']=errors
            del operands,actual,expected,route
            write(record)
    # All values/grads, not just an aggregate loss, must remain finite.
    if 'mxfp8_step' in selected or args.reference or args.save_reference:
        output,gradients=calls['mxfp8_step']()
        names=('output','input',*(n for n,_ in model.named_parameters()))
        values=(output,*gradients)
        if not all(bool(v.isfinite().all()) for v in values):raise AssertionError('Nonfinite model output/gradient')
        record['validation']['all_finite']=True
        if args.reference:
            expected=torch.load(args.reference,map_location='cpu',weights_only=True)
            if list(names)!=expected['names']:raise AssertionError('Parameter identity differs')
            for name,actual in route_identity.items():
                if not torch.equal(actual,expected['routing'][name]):raise AssertionError(('Hard routing identity differs',name))
            errors={}
            for name,value,ref in zip(names,values,expected['values']):
                actual=value.detach().float().cpu();ref=ref.float();delta=actual-ref
                nrms=float(delta.norm()/ref.norm().clamp_min(1e-12));maximum=float(delta.abs().max())
                errors[name]=dict(nrms=nrms,max_abs=maximum)
                if nrms>.01 and maximum>2e-7:raise AssertionError((name,nrms,maximum))
            record['validation']['source_parity']=errors
        if args.save_reference:
            args.save_reference.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(names=list(names),values=[v.detach().cpu() for v in values],routing=route_identity),args.save_reference)
        del output,gradients,values
        write(record)
    if args.bf16_numerics:
        output,gradients=calls['mxfp8_step']()
        quantized=tuple(v.detach().clone() for v in (output,*gradients))
        del output,gradients
        output,gradients=calls['bf16_checkpoint_step']()
        names=('output','input',*(name for name,_ in model.named_parameters()))
        record['validation']['matched_bf16_discrepancy']={
            name:discrepancy(a,r) for name,a,r in zip(names,quantized,(output,*gradients),strict=True)}
        record['validation']['matched_bf16_contract']='Same model/input/upstream/router; quantization policy differs; descriptive error, not an oracle acceptance threshold'
        del quantized,output,gradients
        write(record)
    for phase in selected:
        print('WARM',phase,flush=True);prepare_phase(phase);value=calls[phase]();del value
    if args.profile!='none':
        phase=args.profile_phase;call=calls[phase]
        for _ in range(3):prepare_phase(phase);value=call();del value
        torch.cuda.synchronize()
        profile_graph=None
        if args.profile_replay:
            profile_stream=stream
            with torch.cuda.stream(profile_stream):prepare_phase(phase);value=call()
            profile_stream.synchronize();profile_graph=torch.cuda.CUDAGraph()
            prepare_phase(phase);profile_stream.synchronize()
            with torch.cuda.graph(profile_graph,stream=profile_stream):value=call()
            torch.cuda.current_stream().wait_stream(profile_stream);_warm_measurement_graph(profile_graph)
            call=profile_graph.replay
        torch.cuda.synchronize()
        if not args.profile_replay:prepare_phase(phase);torch.cuda.synchronize()
        record['profile_replay']=args.profile_replay
        if args.profile=='torch':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('packed/'+phase):value=call()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
            record['profile_table']=prof.key_averages().table(sort_by='cuda_time_total',row_limit=40)
            print(record['profile_table'],flush=True)
        else:
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range('packed/'+phase):value=call()
            torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
        record['status']='profiled';write(record)
        if profile_graph is not None:profile_graph.reset()
        return
    for round_index in range(args.rounds):
        # Round-robin order reduces ramp/temperature ordering bias.
        phases=selected[round_index%len(selected):]+selected[:round_index%len(selected)]
        for phase in phases:
            call=calls[phase];references=None
            if round_index==0 and phase in ('mxfp8_step','mxfp8_backward','bf16_checkpoint_step',
                    'bf16_compact_checkpoint_step','bf16_compact_saved_step','bf16_saved_step',
                    'expert_step','expert_bf16_checkpoint_step'):
                prepare_phase(phase);checked=call()
                references=[v.detach().cpu().clone() for v in (checked[0],*checked[1])]
                del checked
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):prepare_phase(phase);value=call();del value
            stream.synchronize();graph=torch.cuda.CUDAGraph()
            prepare_phase(phase);stream.synchronize()
            with torch.cuda.graph(graph,stream=stream):value=call()
            torch.cuda.current_stream().wait_stream(stream);_warm_measurement_graph(graph)
            if references is not None:
                # Captured set_to_none must overwrite every gradient, not
                # accidentally accumulate through dozens of timed replays.
                errors=[]
                for _ in range(2):
                    graph.replay();torch.cuda.synchronize()
                    worst_nrms=0.;worst_abs=0.
                    for tensor,ref in zip((value[0],*value[1]),references):
                        actual=tensor.detach().float().cpu();ref=ref.float();delta=actual-ref
                        nrms=float(delta.norm()/ref.norm().clamp_min(1e-12));maximum=float(delta.abs().max())
                        if not bool(actual.isfinite().all()) or (nrms>.006 and maximum>2e-7):
                            raise AssertionError(('Captured replay gradients differ',phase,nrms,maximum))
                        worst_nrms=max(worst_nrms,nrms);worst_abs=max(worst_abs,maximum)
                    errors.append(dict(max_nrms=worst_nrms,max_abs=worst_abs))
                record['validation'].setdefault('fresh_gradient_replay',{})[phase]=errors
                del references
            samples,measurement=measure_cuda_batches(graph.replay,samples=args.samples,calls_per_sample=1,
                min_batch_ms=5.,device=x.device)
            graph.reset();del value
            if phase=='mxfp8_backward':backward_saved.clear()
            gc.collect();torch.cuda.synchronize()
            record['timings'].setdefault(phase,[]).append(dict(**statistics_ms(samples),measurement=measurement))
            print('TIME',round_index,phase,statistics.median(samples),'ms',flush=True);write(record)
    # Warmed identical iterations, reporting live allocations rather than reserve.
    call=calls['mxfp8_step'] if 'mxfp8_step' in selected else calls[selected[0]]
    for _ in range(2):value=call();del value
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();live=[]
    for _ in range(5):
        value=call();del value;gc.collect();torch.cuda.synchronize();live.append(torch.cuda.memory_allocated())
    record['memory']=dict(live_bytes=live,live_growth_bytes=max(live)-min(live),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    if max(live)!=min(live):raise AssertionError(('Live allocation growth',live))
    # Publish valid ordinary measurements before optional profiler injection.
    # A failed trace must not erase already checked timing/replay evidence.
    record['status']='complete';record['trace_status']='pending' if args.trace_phases else 'not_requested';write(record)
    for phase in args.trace_phases:
        if phase not in calls:raise ValueError(('Unknown trace phase',phase))
        call=calls[phase]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):prepare_phase(phase);value=call()
        stream.synchronize();graph=torch.cuda.CUDAGraph()
        prepare_phase(phase);stream.synchronize()
        with torch.cuda.graph(graph,stream=stream):value=call()
        torch.cuda.current_stream().wait_stream(stream);_warm_measurement_graph(graph)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function('packed/'+phase):graph.replay()
            torch.cuda.synchronize()
        path=args.output.with_name(args.output.stem+'.'+phase+'.trace.json')
        prof.export_chrome_trace(str(path))
        record.setdefault('traces',{})[phase]=str(path)
        write(record)
        graph.reset();del value
    record['trace_status']='complete' if args.trace_phases else 'not_requested';write(record)


if __name__=='__main__':main()
