"""Fresh-process BF16/recomputed-BF16/MXFP8 deployment comparison."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .execution_mix import MLPExecutionMix
from .replacement import (ReplacementPolicy, recommend_replacement,
                          comparison_orders, summarize_repeated_comparison)


def source_manifest(root=None):
    """Pin all RTX Python sources, including kernels and runtime helpers."""
    root=Path(__file__).resolve().parents[1] if root is None else Path(root)
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py'))}


def error_metrics(candidate, reference):
    """Independent tensor metrics; both operands must contribute to finiteness."""
    if candidate.shape != reference.shape:
        raise ValueError('candidate/reference shapes differ')
    a, b = candidate.detach().float(), reference.detach().float()
    delta = a-b
    rms = delta.square().mean().sqrt().item()
    scale = b.square().mean().sqrt().item()
    return dict(max_abs=delta.abs().max().item(), rms=rms, reference_rms=scale,
                normalized_rms=rms/max(scale,1e-12),
                finite=bool(a.isfinite().all() & b.isfinite().all()))


def require_recipe_metrics(metrics, names):
    """Fail closed on incomplete/nonfinite/inconsistent numerical evidence."""
    try:
        if set(metrics) != set(names):
            raise ValueError('missing output/gradient metrics')
        for name, value in metrics.items():
            if value['finite'] is not True:
                raise ValueError(f'{name}: nonfinite candidate/reference')
            for key in ('max_abs','rms','reference_rms','normalized_rms'):
                number = value[key]
                if type(number) not in (int,float) or not math.isfinite(number) or number < 0:
                    raise ValueError(f'{name}: invalid {key}')
            expected = value['rms']/max(value['reference_rms'],1e-12)
            if not math.isclose(expected,value['normalized_rms'],rel_tol=1e-6,abs_tol=1e-12):
                raise ValueError(f'{name}: inconsistent normalized RMS')
            if (value['max_abs'] > 1e-7 if value['reference_rms'] <= 1e-12
                    else value['normalized_rms'] > .025):
                raise ValueError(f'{name}: recipe tolerance exceeded')
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'recipe correctness failure: {exc}') from exc


def worker(args):
    sources_before=source_manifest()
    import torch
    from ..mlp import DEFAULT_MLP_CONFIG, MXFP8MLPConfig, mxfp8_mlp
    from .._mlp_autotune import selected_mlp
    from .mlp import reference
    torch.manual_seed(args.seed)
    m,k,h,o=map(int,args.shape.split(','))
    mix=MLPExecutionMix(*map(int,args.mix.split(',')))
    mix.validate_phase('train')
    config=DEFAULT_MLP_CONFIG
    if args.config:
        data=json.loads(Path(args.config).read_text())
        if args.config_field != 'config' and args.config_field not in data:
            raise ValueError(f'missing config field {args.config_field!r}')
        config=MXFP8MLPConfig.from_dict(data.get(args.config_field,data))
    inputs=tuple((torch.randn(*s,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
                 for s,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    def bf16(x,w,d):
        g,u=torch.nn.functional.linear(x,w).chunk(2,-1)
        hidden=(torch.nn.functional.silu(g.float())*u.float()).bfloat16()
        return torch.nn.functional.linear(hidden,d)
    spec=json.dumps(dict(mode='fixed',initial=config.to_dict()),sort_keys=True)
    def target(x,w,d):
        if args.backend=='bf16':
            return bf16(x,w,d)
        if args.backend=='bf16_recompute':
            return mxfp8_mlp(x,w,d,precision='bf16',config=config)
        return selected_mlp(x,w,d,'silu',spec,torch.is_grad_enabled())[0]
    options={'max_autotune':True,'triton.cudagraphs':False}
    compiled=torch.compile(target,fullgraph=True,dynamic=False,options=options)
    def clear():
        for value in inputs:
            value.grad=None
    def backward(y):
        clear()
        y.backward(dy)
        return tuple(v.grad for v in inputs)
    def pair():
        clear()
        y=compiled(*inputs)
        return (y,*backward(y))
    def inference():
        with torch.no_grad():
            return compiled(*inputs)
    def workload():
        return mix.execute(compiled,inputs,backward,torch.set_grad_enabled)
    smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    input_bytes=torch.cuda.memory_allocated()
    started=time.monotonic()
    actual=pair()
    torch.cuda.synchronize()
    cold_s=time.monotonic()-started
    cold_peak=torch.cuda.max_memory_allocated()
    actual=tuple(t.detach() for t in actual)
    clear()
    y=bf16(*inputs)
    bf16_expected=(y.detach(),*(v.detach() for v in backward(y)))
    del y
    clear()
    with torch.no_grad():
        expected=reference(*inputs,dy,'silu') if args.backend=='mxfp8' else bf16_expected
    def metrics(ref,values=None):
        result={}
        for name,a,b in zip(('output','dx','dgate_up','ddown'),actual if values is None else values,ref):
            result[name]=error_metrics(a,b)
        return result
    numerics=metrics(expected)
    bf16_difference=metrics(bf16_expected)
    inference_actual=(inference().detach(),)
    inference_numerics=metrics(expected[:1],inference_actual)
    require_recipe_metrics(numerics, ('output','dx','dgate_up','ddown'))
    require_recipe_metrics(inference_numerics, ('output',))
    if any(not value['finite'] for value in bf16_difference.values()):
        raise RuntimeError('nonfinite BF16 comparison candidate/reference')
    del actual,expected,bf16_expected,inference_actual
    gc.collect()
    clear()
    # The independent FP32 oracle can leave cuBLAS workspace allocations that
    # are not used by the CuTe candidate. Reset library workspaces only here,
    # before target warmup/capture; each backend then reacquires what it needs.
    # Never apply this process-global diagnostic cleanup in an application.
    torch.cuda.synchronize()
    oracle_cleanup_before=torch.cuda.memory_allocated()
    torch._C._cuda_clearCublasWorkspaces()
    oracle_cleanup_after=torch.cuda.memory_allocated()
    reference_workspace_cleanup=dict(
        mechanism='torch._C._cuda_clearCublasWorkspaces',
        allocated_before=oracle_cleanup_before,allocated_after=oracle_cleanup_after,
        released_bytes=oracle_cleanup_before-oracle_cleanup_after,
        scope='after correctness, before target warmup and graph capture')
    # Warm both grad/no-grad specializations and the exact workload.
    for _ in range(3):
        pair(); inference(); workload()
    clear()
    torch.cuda.synchronize()
    resident=torch.cuda.memory_allocated()
    def measure(fn,prepare=None):
        samples=[]
        for _ in range(args.samples):
            clear()
            prepared=prepare() if prepare else None
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            value=fn(prepared) if prepare else fn()
            end.record();end.synchronize()
            samples.append(start.elapsed_time(end))
            del value,prepared
        clear()
        return dict(median_ms=statistics.median(samples),min_ms=min(samples),samples_ms=samples)
    timings=dict(inference_forward=measure(inference),training_forward=measure(lambda:compiled(*inputs)),
                 backward=measure(backward,lambda:compiled(*inputs)),training_pair=measure(pair),
                 workload=measure(workload))
    timing_modes={name:'compiled_direct' for name in timings}
    replay_validation=[]
    graph_resources=None
    measured_workload=workload
    if args.mode=='outer_capture':
        from ..runtime import retain_runtime_resources
        def capture(fn):
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with retain_runtime_resources() as owners:
                with torch.cuda.graph(graph,stream=stream):
                    outputs=fn()
            def replay():
                graph.replay()
                return outputs
            backups=tuple(v.detach().clone() for v in inputs)
            for delta in (0.,.001):
                with torch.no_grad():
                    for value,original in zip(inputs,backups):
                        value.copy_(original+delta)
                expected=tuple(t.detach().clone() for t in fn())
                actual=replay()
                torch.cuda.synchronize()
                for a,b in zip(actual,expected):
                    torch.testing.assert_close(a,b,rtol=.001,atol=1e-6)
            with torch.no_grad():
                for value,original in zip(inputs,backups):
                    value.copy_(original)
            clear()
            return replay,(graph,owners,outputs,stream)
        # Do not retain two separate graph pools in the workload memory peak.
        replay,resources=capture(pair)
        timings['training_pair']=measure(replay)
        replay_validation.append('training_pair: original and changed X/weights')
        del replay,resources
        gc.collect();torch.cuda.synchronize()
        measured_workload,graph_resources=capture(workload)
        timings['workload']=measure(measured_workload)
        replay_validation.append('workload: original and changed X/weights')
        timing_modes.update(training_pair='outer_capture',workload='outer_capture')
        resident=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    live=[]
    for _ in range(5):
        value=measured_workload(); del value
        clear();torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    peak=torch.cuda.max_memory_allocated()
    if len(set(live))!=1:
        raise RuntimeError(f'post-warmup live memory growth: {live}')
    source_hashes=source_manifest()
    if source_hashes != sources_before:
        raise RuntimeError('RTX sources changed during the benchmark worker; discard this measurement')
    return dict(status='ok',backend=args.backend,environment_qualified=False,
        source_hashes=source_hashes,
        config_source=None if not args.config else dict(path=str(Path(args.config).resolve()),
                            sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                            field=args.config_field),
        qualification_note='diagnostic: no repeated alternating fresh-process comparison or contention qualification yet',
        monitoring=dict(ok=smi.returncode==0,stdout=smi.stdout,stderr=smi.stderr),
        contract=dict(shape=[m,k,h,o],seed=args.seed,mix=mix.contract(),torch=torch.__version__,
                      measurement_revision=2,
                      numerical_gate_revision=2,
                      oracle_workspace_policy='clear_cublas_before_target_rewarm',
                      cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),options=options,
                      activation='silu_fp32_product_then_bf16',mode=args.mode,
                      timing_modes=timing_modes,samples=args.samples),
        config=config.to_dict() if args.backend=='mxfp8' else None,
        numerics=numerics,inference_numerics=inference_numerics,bf16_difference=bf16_difference,timings=timings,
        measurement_revision=2,reference_workspace_cleanup=reference_workspace_cleanup,
        replay_validation=replay_validation,
        memory=dict(input_allocated=input_bytes,cold_peak_allocated=cold_peak,
            warm_peak_allocated=peak,warm_resident_allocated=resident,live_samples=live,
            incremental_warm_peak=max(0,peak-resident),reserved=torch.cuda.memory_reserved(),
            reserved_includes_reference_freed_blocks=True,
            unaccounted='external allocations, allocator-private breakdown, and scratch ownership not attributed'),
        cold_first_training_pair_s=cold_s)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('all','bf16','bf16_recompute','mxfp8'),default='all')
    p.add_argument('--shape',default='128,128,256,128')
    p.add_argument('--mix',default='1,0,0')
    p.add_argument('--config',help='Pinned MXFP8 config or result JSON; default is not a historical winner')
    p.add_argument('--config-field',default='config',help='Explicit object field, e.g. challenger in a paired report')
    p.add_argument('--seed',type=int,default=317)
    p.add_argument('--samples',type=int,default=12)
    p.add_argument('--mode',choices=('compiled_direct','outer_capture'),default='compiled_direct')
    p.add_argument('--rounds',type=int,default=1,help='1 for smoke; a multiple of 6 for order-balanced fresh workers')
    p.add_argument('--prefer-memory',action='store_true',
                   help='Explicitly prefer lower peak allocation; does not authorize a model change')
    p.add_argument('--max-slowdown',type=float,default=0.,
                   help='Allowed fractional slowdown for memory savings; requires --prefer-memory')
    p.add_argument('--minimum-speedup',type=float,default=.01,
                   help='Required fractional time reduction for speed-based recommendation')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        policy=ReplacementPolicy(args.prefer_memory,args.max_slowdown,args.minimum_speedup)
    except (TypeError,ValueError) as exc:
        p.error(str(exc))
    if args.samples<3: p.error('at least three timing samples required')
    if args.rounds != 1:
        try: comparison_orders(args.rounds)
        except ValueError as exc: p.error(str(exc))
    if args.output.exists(): raise FileExistsError(args.output)
    if args.backend=='all':
        args.output.mkdir(parents=True)
        if args.rounds != 1:
            batches=[]
            for index, order in enumerate(comparison_orders(args.rounds)):
                directory=args.output/f'round-{index:03d}'
                directory.mkdir()
                batch=[]
                for backend in order:
                    destination=directory/f'{backend}.json'
                    command=[sys.executable,'-m',__spec__.name,'--backend',backend,
                             '--shape',args.shape,'--mix',args.mix,'--seed',str(args.seed),
                             '--samples',str(args.samples),'--mode',args.mode,'--output',str(destination)]
                    if args.config: command+=['--config',args.config,'--config-field',args.config_field]
                    print(f'round={index+1}/{args.rounds} backend={backend}',flush=True)
                    with (directory/f'{backend}.log').open('x') as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                    batch.append(json.loads(destination.read_text()))
                batches.append(batch)
            result=summarize_repeated_comparison(batches,policy=policy)
            result['orders']=comparison_orders(args.rounds)
            with (args.output/'comparison.json').open('x') as stream:
                json.dump(result,stream,indent=2)
            print(json.dumps(result['decision']),flush=True)
            return
        reports=[]
        for backend in ('bf16','bf16_recompute','mxfp8'):
            command=[sys.executable,'-m',__spec__.name,'--backend',backend,'--shape',args.shape,
                     '--mix',args.mix,'--seed',str(args.seed),'--samples',str(args.samples),
                     '--mode',args.mode,'--output',str(args.output/f'{backend}.json')]
            if args.config: command+=['--config',args.config,'--config-field',args.config_field]
            with (args.output/f'{backend}.log').open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            reports.append(json.loads((args.output/f'{backend}.json').read_text()))
        result=dict(reports=reports,decision=recommend_replacement(reports,policy))
        with (args.output/'comparison.json').open('x') as stream: json.dump(result,stream,indent=2)
    else:
        result=worker(args)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result.get('decision',dict(backend=args.backend,status='ok'))),flush=True)


if __name__=='__main__': main()
