"""Frozen-source MoE diagnosis: phases, launches, token scaling and controlled A/Bs.

Never change production kernels. Profiler processes warm before cudaProfilerStart;
ordinary event timings, not instrumented times, are performance measurements.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import functools
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-id',default='d768_e48_k6_h256')
    p.add_argument('--tokens',type=int,default=2000)
    p.add_argument('--seed-studies',nargs='+',type=Path,default=[])
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--samples',type=int,default=9)
    p.add_argument('--whole-layer',action='store_true')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--trace-only',action='store_true',help='Fresh profiler process: no preceding manual graph benchmark')
    p.add_argument('--variants',choices=('none','pipelines','reuse','preparation','all'),default='none')
    p.add_argument('--variant-rounds',type=int,default=1)
    p.add_argument('--tune-trials',type=int,default=0)
    p.add_argument('--capture',choices=('none','timeline','counters'),default='none')
    p.add_argument('--capture-phase',default='expert_step')
    p.add_argument('--minimal-ranges',action='store_true',help='Only one outer NVTX range; qualify annotation-induced CPU gaps')
    p.add_argument('--cpu-profile',choices=('expert_step','public_expert_step','whole_step'))
    p.add_argument('--evict',action='store_true')
    p.add_argument('--skip-baseline',action='store_true')
    p.add_argument('--fixture',type=Path,help='Exact production inputs/routes recorded outside profiler injection')
    p.add_argument('--save-fixture',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--calibrate-memory',action='store_true')
    return p


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class LaunchRanges:
    """Annotate existing helper boundaries, without retaining intermediates.

    Each prepared helper is one GPU launch; kernel correlation in the exported
    trace is authoritative. Instrumentation is disabled for ordinary timings.
    """
    def __init__(self):self.phase='';self.counts={};self.manifest=[];self.enabled=False

    @contextmanager
    def phase_range(self,name):
        import torch
        old=self.phase;self.phase=name;self.counts={}
        with torch.profiler.record_function('moe/'+name),torch.cuda.nvtx.range('moe/'+name):
            yield
        self.phase=old

    def decorate(self,name,fn):
        @functools.wraps(fn)
        def run(*args,**kwargs):
            if not self.enabled:return fn(*args,**kwargs)
            import torch
            suffix=name
            if name=='movement':suffix='movement_'+args[6]
            if name=='gemm':suffix='gemm_'+(args[4] if len(args)>4 else kwargs.get('kind','linear'))
            index=self.counts.get(suffix,0);self.counts[suffix]=index+1
            label=f'moe/{self.phase}/{suffix}_{index}'
            tensors=[]
            def metadata(value):
                if isinstance(value,torch.Tensor):tensors.append(dict(shape=list(value.shape),dtype=str(value.dtype)))
                elif isinstance(value,(tuple,list)):
                    for item in value:metadata(item)
            metadata(args)
            self.manifest.append(dict(label=label,tensors=tensors))
            with torch.profiler.record_function(label),torch.cuda.nvtx.range(label):return fn(*args,**kwargs)
        return run

    @contextmanager
    def install(self):
        from rtx import moe,_moe_prepared as prep
        originals=[(moe,'_run_forward',moe._run_forward),(moe,'_run_backward',moe._run_backward),
                   (moe,'_movement',moe._movement),(prep,'prepare',prep.prepare),(prep,'packed_gemm',prep.packed_gemm),
                   (prep,'load_kernel_symbol',prep.load_kernel_symbol)]
        def phase_wrapper(name,fn):
            def run(*a,**kw):
                with self.phase_range(name):return fn(*a,**kw)
            return run
        moe._run_forward=phase_wrapper('expert_forward',moe._run_forward)
        moe._run_backward=phase_wrapper('expert_backward',moe._run_backward)
        moe._movement=self.decorate('movement',moe._movement)
        prep.prepare=self.decorate('quant_portfolio',prep.prepare)
        prep.packed_gemm=self.decorate('gemm',prep.packed_gemm)
        load=prep.load_kernel_symbol
        def loader(module,symbol):
            factory=load(module,symbol)
            if symbol not in ('compile_dequant_hidden','compile_native_hidden_scales','compile_expert_joint_recompute'):
                return factory
            def build(*a,**kw):return self.decorate(symbol.removeprefix('compile_'),factory(*a,**kw))
            return build
        prep.load_kernel_symbol=loader
        self.enabled=True
        try:yield self
        finally:
            self.enabled=False
            for module,name,original in originals:setattr(module,name,original)


def selected(args,case):
    from rtx.moe import DEFAULT_MOE_CONFIG,MXFP8MoEConfig
    rows=[]
    for path in args.seed_studies:
        payload=json.loads(path.read_text())
        rows.extend((row,path) for row in payload.get('results',[payload]) if row['case']['id']==case['id'])
    if rows:
        row,path=min(rows,key=lambda item:abs(item[0]['tokens']-args.tokens))
        if args.config:
            explicit=json.loads(args.config.read_text())
            return MXFP8MoEConfig.from_dict(explicit.get('config',explicit)),row.get('whole_layer',{}).get('auxiliary',{}),dict(config=str(args.config),auxiliary=str(path))
        return MXFP8MoEConfig.from_dict(row['config']),row.get('whole_layer',{}).get('auxiliary',{}),dict(path=str(path),tokens=row['tokens'])
    if args.config:
        row=json.loads(args.config.read_text())
        return MXFP8MoEConfig.from_dict(row.get('config',row)),{},str(args.config)
    return DEFAULT_MOE_CONFIG,{},'default'


def retarget_auxiliary(aux,tokens):
    """Retarget only shape-dependent shared-dW partitions, never silently.

    A frozen N=2000 split count is illegal at N=128 or N=32768. Keep its
    reduction tile where possible; record any changed partition explicitly.
    This is benchmark schedule adaptation, not a production-policy change.
    """
    aux=json.loads(json.dumps(aux));changes=[]
    for name in ('gate_backward','down_backward'):
        bwd=aux.get('shared',{}).get('config',{}).get(name)
        if not bwd:continue
        dw=bwd['dw']
        if dw['reduction'] not in ('split_fp32_workspace','split_fp32_atomic'):continue
        before=dict(dw)
        if dw.get('split_partition','fixed')=='balanced':
            tile=dw['gemm']['tile_k'];dw['split_reduction']=min(dw['split_reduction'],(tokens+tile-1)//tile)
        else:
            tile=dw['reduction_tile'];dw['split_reduction']=(tokens+tile-1)//tile
        if dw['split_reduction']==1:
            dw.update(reduction='full_fp32',reduction_tile=0,workspace_epilogue='none')
        if dw!=before:
            changes.append(dict(stage=name,old={k:before.get(k) for k in ('reduction','split_reduction','reduction_tile')},
                                new={k:dw.get(k) for k in ('reduction','split_reduction','reduction_tile')}))
    return aux,changes


def build(args):
    import torch
    from rtx.moe import BalancedMoE,_run_forward,_run_backward,mxfp8_routed_experts,torch_routed_experts
    from rtx.routing import BalancedRouting,RoutingKernelConfig
    from rtx.mlp import MXFP8MLPConfig
    cases=json.loads(Path('benchmarks/balanced_moe_cases.json').read_text())['cases']
    case=next(c for c in cases if c['id']==args.case_id)
    config,aux,origin=selected(args,case)
    aux,shared_changes=retarget_auxiliary(aux,args.tokens)
    origin=dict(seed=origin,shared_partition_adaptations=shared_changes,auxiliary=aux)
    torch.manual_seed(819)
    options=dict(backend='cute')
    if aux:options['config']=RoutingKernelConfig.from_dict(aux['router']['config'])
    model=BalancedMoE(case['d_model'],case['num_experts'],case['expert_hidden_dim'],top_k=case['top_k'],
        shared_hidden_features=case['shared_hidden_dim'],device='cuda',autotune='off',config=config,
        shared_config=MXFP8MLPConfig.from_dict(aux['shared']['config']) if aux else None,router_options=options)
    with torch.no_grad():model.router.projection_weight.normal_(std=.02)
    if args.fixture:
        if args.whole_layer:raise ValueError('Recorded fixtures isolate experts; do not substitute a whole-model routing path')
        data=torch.load(args.fixture,map_location='cuda',weights_only=True)
        assert data['case']==case and data['tokens']==args.tokens
        x,gu,down=[data[key].detach().requires_grad_() for key in ('x','gu','down')]
        route=BalancedRouting(data['ids'],data['weights'].requires_grad_(),data['inverse'])
    else:
        x=(torch.randn(args.tokens,case['d_model'],device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
        with torch.no_grad():route=model.router(x)
        route=BalancedRouting(route.expert_ids,route.route_weights.detach().requires_grad_(),route.inverse)
        gu=model.gate_up.weight.detach().requires_grad_();down=model.down.weight.detach().requires_grad_()
    operands=(x,gu,down,*route)
    dy=data['dy'] if args.fixture else torch.randn(x.shape,device='cuda',dtype=torch.float32)
    if args.save_fixture:
        args.save_fixture.parent.mkdir(parents=True,exist_ok=True)
        values=dict(zip(('x','gu','down','ids','weights','inverse','dy'),(*operands,dy)))
        torch.save(dict(case=case,tokens=args.tokens,**{k:v.detach().cpu() for k,v in values.items()}),args.save_fixture)
    from rtx.autotune.moe import make_moe_adapter
    shape=(args.tokens,case['d_model'],case['num_experts'],route.expert_ids.shape[1],case['top_k'],case['expert_hidden_dim'])
    adapter=make_moe_adapter(shape,None,initial=config)
    config=adapter.initial_config
    why=adapter.rejection(config)
    if why and 'TMA epilogue requires full M/N tiles' in str(why):
        stages=('down','down_dx','down_dw','gate_dx','gate_dw')
        required=max(getattr(config.prepared,s).tile_m for s in stages if getattr(config.prepared,s).epilogue=='tma')
        config=replace(config,prepared=replace(config.prepared,capacity_multiple=max(required,config.prepared.capacity_multiple)))
        origin=dict(seed=origin,shape_adaptation='zero_pad_expert_capacity_for_frozen_TMA_output_tile',multiple=required)
        adapter=make_moe_adapter(shape,None,initial=config);config=adapter.initial_config;why=adapter.rejection(config)
    if why:raise ValueError(f'Frozen seed is not legal at this shape: {why}')
    if args.tune_trials:
        if args.capture!='none':raise ValueError('Never autotune inside a profiler process; supply a frozen --config')
        from rtx.autotune.moe import tune_moe
        from rtx.autotune import AutotunePolicy
        from rtx.moe import _selection_spec,_parse_spec
        policy=AutotunePolicy.preset('thorough',max_trials=args.tune_trials,search_schedule='progressive',candidate_mix='mixed')
        config=tune_moe(operands,_parse_spec(_selection_spec(config,'balanced',policy,args.output.parent/'tuning')),
                        training=True,progress=lambda s:print(s,flush=True)).config
    model.config=config
    from rtx.moe import _selection_spec
    model._expert_spec=_selection_spec(config,'off',None,None)
    from rtx import moe as moe_impl
    def fwd():return moe_impl._run_forward(*operands,config)
    def bwd():return moe_impl._run_backward(*operands,dy,config)
    def step():return (fwd(),*bwd())
    calls=dict(expert_forward=fwd,expert_backward=bwd,expert_step=step)
    if not args.skip_baseline:
        baseline=torch.compile(lambda x,gu,d,w:torch_routed_experts(x,gu,d,
            BalancedRouting(route.expert_ids,w,route.inverse)),fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        calls['bf16_expert_forward']=lambda:baseline(x,gu,down,route.route_weights)
        calls['bf16_expert_step']=lambda:torch.autograd.grad(baseline(x,gu,down,route.route_weights),
                                                          (x,gu,down,route.route_weights),dy)
    public=torch.compile(lambda x,gu,d,w:mxfp8_routed_experts(x,gu,d,
        BalancedRouting(route.expert_ids,w,route.inverse),config=config),fullgraph=True,dynamic=False,
        options={'triton.cudagraphs':False})
    calls['public_expert_step']=lambda:torch.autograd.grad(public(x,gu,down,route.route_weights),
                                                         (x,gu,down,route.route_weights),dy)
    if args.whole_layer:
        native=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        params=tuple(model.parameters())
        calls['whole_forward']=lambda:native(x)
        # dy.bfloat16 is captured in a separate immutable input, not timed.
        whole_dy=dy.bfloat16()
        calls['whole_step']=lambda:torch.autograd.grad(native(x),(x,*params),whole_dy)
        if not args.skip_baseline:
            bf=BalancedMoE(case['d_model'],case['num_experts'],case['expert_hidden_dim'],top_k=case['top_k'],
                shared_hidden_features=case['shared_hidden_dim'],device='cuda',autotune='off',backend='torch',router_options=options)
            bf.load_state_dict(model.state_dict());bp=tuple(bf.parameters())
            bfn=torch.compile(bf,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
            calls['bf16_whole_step']=lambda:torch.autograd.grad(bfn(x),(x,*bp),whole_dy)
    return case,shape,config,origin,operands,dy,calls


def correctness(operands,dy,calls,n):
    import torch
    from rtx._moe_reference import operational_reference
    actual=calls['expert_step']()
    if n<=2000:
        expected=operational_reference(*operands,dy)
        errors={}
        for name,a,b in zip(('output','dx','gate_dw','down_dw','route_dw'),actual,expected):
            delta=a.float()-b.float();nrms=float(delta.norm()/b.float().norm().clamp_min(1e-12));maximum=float(delta.abs().max())
            assert bool(a.isfinite().all()) and (nrms<(.003 if name=='output' else .006) or maximum<2e-7),(name,nrms,maximum)
            errors[name]=dict(nrms=nrms,max_abs=maximum)
        return dict(oracle='independent_fp64_quantized',errors=errors)
    # Bounded-memory large-C dW oracle: one expert at a time, full reduction.
    from rtx._moe_reference import quantize_decode
    x,gu,down,ids,weights,inverse=operands
    e,c=ids.shape;h=down.shape[-1];checks={}
    for expert in sorted({0,e-1}):
        token=ids[expert];slots=expert*c+torch.arange(c,device=x.device)
        valid=inverse[token].eq(slots[:,None]).any(-1)
        xx=(x[token]*valid[:,None]).unsqueeze(0)
        p=(quantize_decode(xx).double()@quantize_decode(gu[expert:expert+1]).double().transpose(1,2)).float()
        gate,up=p[...,:h],p[...,h:];sig=1/(1+torch.exp(-gate));act=gate/(1+torch.exp(-gate))
        hidden=quantize_decode(act*up)
        sd=(dy[token]*weights[expert,:,None]*valid[:,None]).bfloat16().unsqueeze(0)
        dh=(quantize_decode(sd).double()@quantize_decode(down[expert:expert+1].transpose(1,2)).double().transpose(1,2)).float().bfloat16()
        dp=torch.cat((dh.float()*up*sig*(1+gate*(1-sig)),dh.float()*act),-1).bfloat16()
        refs=(quantize_decode(dp.transpose(1,2)).double()@quantize_decode(xx.transpose(1,2)).double().transpose(1,2),
              quantize_decode(sd.transpose(1,2)).double()@quantize_decode(hidden.bfloat16().transpose(1,2)).double().transpose(1,2))
        for name,value,ref in zip(('gate_dw','down_dw'),(actual[2][expert],actual[3][expert]),refs):
            ref=ref[0].float().bfloat16();nrms=float((value.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))
            assert nrms<.006,(expert,name,nrms);checks[f'{expert}/{name}']=nrms
    assert all(bool(t.isfinite().all()) for t in actual)
    return dict(oracle='two_experts_full_C_independent_fp64_dW_plus_all_finite',checks=checks,
                limitation='Large-shape output/dX/route adjoints are finite-checked, not independently compared at every element')


def statistics_ms(samples):
    med=statistics.median(samples)
    return dict(median_ms=med,min_ms=min(samples),p90_ms=sorted(samples)[int(.9*(len(samples)-1))],
                relative_mad=statistics.median(abs(x-med) for x in samples)/max(med,1e-12),samples_ms=samples)


def hardware_attributes():
    from cuda.bindings import driver as cuda
    result={}
    for name in ('MULTIPROCESSOR_COUNT','GLOBAL_MEMORY_BUS_WIDTH','MEMORY_CLOCK_RATE','L2_CACHE_SIZE',
                 'MAX_REGISTERS_PER_MULTIPROCESSOR','MAX_SHARED_MEMORY_PER_MULTIPROCESSOR',
                 'MAX_THREADS_PER_MULTIPROCESSOR','MAX_BLOCKS_PER_MULTIPROCESSOR',
                 'MAX_SHARED_MEMORY_PER_BLOCK_OPTIN','CLOCK_RATE','WARP_SIZE'):
        key='CU_DEVICE_ATTRIBUTE_'+name
        attribute=getattr(cuda.CUdevice_attribute,key,None)
        if attribute is None:result[name]=None;continue
        answer=cuda.cuDeviceGetAttribute(attribute,0)
        result[name]=int(answer[1]) if answer[0]==cuda.CUresult.CUDA_SUCCESS else str(answer[0])
    return result


def measure(call,stream,samples,eviction=None):
    import torch
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    for _ in range(2):call()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):call()
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):result=call()
    torch.cuda.current_stream().wait_stream(stream);_warm_measurement_graph(graph)
    timings=[];started=time.time()
    for _ in range(samples):
        if eviction is not None:eviction.zero_()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize();timings.append(a.elapsed_time(b))
    graph.reset();del result
    return dict(**statistics_ms(timings),wall_started=started,wall_finished=time.time())


def host_timing(call,samples):
    import torch
    values=[]
    for _ in range(samples):
        torch.cuda.synchronize();start=time.perf_counter();result=call();torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000);del result
    return statistics_ms(values)


def variants(config,shape,choice):
    from rtx.autotune.moe import make_moe_adapter
    a=make_moe_adapter(shape,None,initial=config)
    yield 'frozen',config
    if choice in ('pipelines','all'):
        for stage in ('gate_up','down','gate_dw'):
            for depth in (1,2,3,4):
                yield f'{stage}_buffers{depth}',a.update_fn(config,f'prepared.{stage}.stages',depth)
                if depth==1:
                    c=getattr(config.prepared,stage)
                    yield f'{stage}_buffers1_producer',replace(config,prepared=replace(config.prepared,
                        **{stage:replace(c,stages=1,scale_role='producer')}))
        yield 'cooperative_linear',a.update_fn(config,'prepared.linear_transport','cooperative')
        yield 'joint_backward',a.update_fn(config,'prepared.backward_preparation','joint')
    if choice in ('reuse','all'):
        for raster in ('m','n'):
            for swizzle in (1,2,4,8):
                p=replace(config.prepared,down=replace(config.prepared.down,raster=raster,grid_swizzle=swizzle))
                yield f'down_{raster}_swizzle{swizzle}',replace(config,prepared=p)
        for locality in ('same_a','same_b','serpentine_a','serpentine_b'):
            yield 'down_'+locality,a.update_fn(a.update_fn(config,'prepared.down.tiles_per_cta',4),
                                             'prepared.down.tile_locality',locality)
        for mode in ('operand','expert','portfolio'):
            yield 'quant_grid_'+mode,replace(config,prepared=replace(config.prepared,quant_grid=mode))
    if choice=='preparation':
        yield 'joint_backward',a.update_fn(config,'prepared.backward_preparation','joint')
        for mode in ('expert','portfolio'):
            yield 'quant_grid_'+mode,replace(config,prepared=replace(config.prepared,quant_grid=mode))


def main():
    args=parser().parse_args()
    import torch
    from rtx.autotune.moe import source_revision,MoEEvaluator,make_moe_adapter
    from rtx.autotune import AutotunePolicy
    if args.calibrate_memory:
        src=torch.randn(128*1024*1024//4,device='cuda');dst=torch.empty_like(src)
        timed=measure(lambda:torch.add(src,1,out=dst),torch.cuda.Stream(),args.samples)
        write(args.output,dict(status='complete',hardware=hardware_attributes(),timing=timed,
            read_write_bytes=2*src.numel()*src.element_size(),
            measured_streaming_add_GBps=2*src.numel()*src.element_size()/timed['median_ms']/1e6,
            caveat='Measured reference, not guaranteed physical peak; no profiler injection'))
        return
    case,shape,config,origin,operands,dy,calls=build(args)
    record=dict(case=case,shape=shape,source_revision=source_revision(),config=config.to_dict(),seed_origin=origin,
                device=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,cuda=torch.version.cuda,
                scope='Fused prepared MXFP8 experts; same CuTe routing policy for whole-layer BF16; BF16 saves intermediates',
                commands=__import__('sys').argv,seed=819,status='running',hardware=hardware_attributes(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(args.output.with_suffix('.config.json'),dict(config=config.to_dict(),case=case,shape=shape))
    record['validation']=correctness(operands,dy,calls,args.tokens)
    if args.prepare_only:
        record['status']='fixture_and_oracle_complete';write(args.output,record);return
    # Compile and initialize EVERY selected public path before instrumentation.
    for name,call in calls.items():
        print('WARM',name,flush=True);call()
    torch.cuda.synchronize()
    if args.cpu_profile:
        import cProfile,pstats
        selected_call=calls[args.cpu_profile]
        warm_until=time.monotonic()+.3
        while time.monotonic()<warm_until:
            warmed=selected_call();del warmed
        torch.cuda.synchronize();prof=cProfile.Profile();prof.enable()
        for _ in range(20):
            result=selected_call();del result
        torch.cuda.synchronize();prof.disable()
        prof.dump_stats(str(args.output.with_suffix('.pstats')))
        with args.output.with_suffix('.cpu.txt').open('w') as handle:
            pstats.Stats(prof,stream=handle).strip_dirs().sort_stats('cumulative').print_stats(70)
        record.update(status='complete',cpu_profile_phase=args.cpu_profile,
                      caveat='Instrumented Python call attribution, not ordinary latency; synchronize at end only')
        write(args.output,record);return
    if args.capture!='none':
        selected_call=calls[args.capture_phase];ranges=LaunchRanges()
        # Fresh application-replay processes otherwise capture while clocks
        # are still ramping after CPU compilation. No driver clock changes.
        warm_until=time.monotonic()+.3
        while time.monotonic()<warm_until:
            warmed=selected_call();del warmed
        torch.cuda.synchronize()
        record['capture_warmup_host_seconds']=.3
        torch.cuda.cudart().cudaProfilerStart()
        if args.minimal_ranges:
            with torch.cuda.nvtx.range('moe/'+args.capture_phase):
                for _ in range(3 if args.capture=='timeline' else 1):selected_call()
        else:
            with ranges.install(),ranges.phase_range(args.capture_phase):
                for _ in range(3 if args.capture=='timeline' else 1):selected_call()
        torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
        record['capture_annotations']='outer_nvtx_only' if args.minimal_ranges else 'per_helper_scopes'
        record['launch_manifest']=ranges.manifest;record['status']='complete'
        write(args.output,record);return
    if args.trace_only:
        ranges=LaunchRanges()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            with ranges.install(),ranges.phase_range(args.capture_phase):result=calls[args.capture_phase]()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
        record['launch_manifest']=ranges.manifest;record['status']='complete'
        write(args.output,record);return
    stream=torch.cuda.Stream()
    memory=[]
    for _ in range(8):
        result=calls['expert_step']();del result;torch.cuda.synchronize();memory.append(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
    result=calls['expert_step']();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();del result
    record['memory']=dict(live_samples=memory,live_growth=max(memory)-min(memory),before=before,peak=peak,
                          temporary_peak=peak-before,reserved=torch.cuda.memory_reserved())
    record['timings']={};eviction=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8) if args.evict else None
    for round_id in range(args.rounds):
        ordered=list(calls) if round_id%2==0 else list(reversed(calls))
        for name in ordered:
            hot=measure(calls[name],stream,args.samples)
            print('TIME',round_id,name,hot['median_ms'],flush=True)
            entry=record['timings'].setdefault(name,dict(hot=[],evict=[]));entry['hot'].append(hot)
            if eviction is not None:entry['evict'].append(measure(calls[name],stream,args.samples,eviction))
            write(args.output,record)
    record['host_timings']={name:host_timing(calls[name],args.samples) for name in
                            ('expert_step','public_expert_step','whole_step') if name in calls}
    if args.trace:
        ranges=LaunchRanges()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            with ranges.install():
                for name in ('expert_forward','expert_backward','public_expert_step','whole_step','bf16_whole_step'):
                    if name in calls:
                        with ranges.phase_range(name):result=calls[name]()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
        record['launch_manifest']=ranges.manifest
    if args.variants!='none':
        evaluator=object.__new__(MoEEvaluator)
        evaluator.operands=tuple(t.detach() for t in operands);evaluator.phase='train'
        evaluator.policy=AutotunePolicy(max_trials=1,samples=5,calls_per_sample=2,warmup=2)
        evaluator.gradient=dy;evaluator.expected=tuple(t.detach() for t in calls['expert_step']())
        record['variant_oracle']='frozen result; baseline independently qualified as reported in validation'
        adapter=make_moe_adapter(shape,None,initial=config);seen=set();record['variants']=[]
        candidates=[]
        for label,candidate in variants(config,shape,args.variants):
            cid=adapter.config_id(candidate)
            if cid in seen:continue
            seen.add(cid);candidates.append((label,candidate))
        for round_id in range(args.variant_rounds):
            for label,candidate in candidates if round_id%2==0 else reversed(candidates):
                reason=adapter.rejection(candidate)
                print('VARIANT',round_id,label,reason,flush=True)
                outcome=dict(status='static_rejected',error=reason) if reason else evaluator.evaluate(candidate).as_dict()
                record['variants'].append(dict(label=label,round=round_id,config=candidate.to_dict(),outcome=outcome));write(args.output,record)
    record['status']='complete';write(args.output,record)
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()
