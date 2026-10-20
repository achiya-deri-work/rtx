"""Disposable research-only down-gradient arena reuse, not runtime policy."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import rtx
from rtx.autotune.replacement_benchmark import source_manifest, error_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-field', default='config')
    p.add_argument('--shape', default='20480,256,768,256')
    p.add_argument('--mode', choices=('separate','paired','integrated'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--graph-check', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = source_manifest()
    m,k,h,o = map(int,args.shape.split(','))
    data = json.loads(args.config.read_text())
    config = rtx.MXFP8MLPConfig.from_dict(data[args.config_field])
    import rtx.fp8_bwd as bwd
    from rtx.kernels.mxfp8 import MXFP8Problem
    dc = config.down_backward or bwd._default_bwd_config(MXFP8Problem(m,o,h))
    # Both arms have identical config metadata, schedules and numerical policy.
    dc = replace(dc, operand_storage='serial_reuse_partials' if args.mode=='integrated' else 'serial_reuse')
    reason = dc.implementation_rejection(MXFP8Problem(m,o,h))
    if reason:
        raise ValueError(reason)
    config = replace(config,down_backward=dc)
    down_key = bwd._intern_bwd_config(dc)
    original = bwd._partial_bwd_runner_from_context
    intercepted = {'dx':0,'dw':0}
    if args.mode == 'paired':
        def lookup(context, gradient):
            problem,key,cfg,dy,x,w = context
            if key != down_key or (problem.m,problem.n,problem.k) != (m,o,h):
                return original(context,gradient)
            intercepted[gradient] += 1
            _,runner,dy,x,w = bwd._resolve_bwd_runner(dy,x,w,key)
            return getattr(runner,gradient),dy,x,w
        bwd._partial_bwd_runner_from_context = lookup
    torch.manual_seed(990)
    layer = rtx.MXFP8MLP(k,h,o,device='cuda',config=config)
    layer._first_hit_spec = json.dumps(dict(mode='fixed',initial=config.to_dict()),sort_keys=True)
    model = torch.compile(layer,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    x = (torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.2).requires_grad_()
    dy = torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    def pair():
        y = model(x)
        return y,*torch.autograd.grad(y,(x,*layer.parameters()),dy)
    from rtx.autotune.mlp import reference
    gates = []
    for seed in (990,991):
        torch.manual_seed(seed)
        with torch.no_grad():
            x.normal_(0,.2)
            for w in layer.parameters():
                w.normal_(0,.02)
            expected = reference(x,*layer.parameters(),dy,'silu')
        actual = pair()
        metrics = [error_metrics(a,b) for a,b in zip(actual,expected)]
        for metric in metrics:
            if metric['normalized_rms'] >= .025:
                raise AssertionError(metrics)
        if not all(bool(t.isfinite().all()) for t in (*actual,*expected)):
            raise AssertionError('nonfinite output or gradient')
        gates.append(dict(seed=seed,metrics=metrics))
        del actual,expected
    for _ in range(5):
        pair()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    live=[]; samples=[]
    for _ in range(9):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            pair()
        end.record();end.synchronize()
        samples.append(start.elapsed_time(end)/20)
        live.append(torch.cuda.memory_allocated())
    report = dict(mode=args.mode,shape=[m,k,h,o],config=config.to_dict(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        source_hashes=sources,source_hashes_after=source_manifest(),
        numerical_gates=gates,pair_samples_ms=samples,pair_median_ms=statistics.median(samples),
        live_bytes=live,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        workspace=rtx.workspace_report(),intercepted=intercepted,
        research_only=True,application_qualified=False)
    if args.graph_check:
        # Diagnostic only, after ordinary timing and memory measurements.
        from rtx.runtime import retain_runtime_resources,clear_runtime_caches
        captures=[]
        for _ in range(2):
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    pair()
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with retain_runtime_resources() as owners:
                with torch.cuda.graph(graph,stream=stream):
                    outputs=pair()
            captures.append((graph,outputs,owners,stream))
        torch.cuda.synchronize()
        clear_runtime_caches(preserve_compiled=True)
        checks=[]
        for generation in range(3):
            with torch.no_grad():
                x.add_(.001)
                for w in layer.parameters():
                    w.add_(.0001)
                expected=reference(x,*layer.parameters(),dy,'silu')
            for graph,outputs,owners,stream in captures:
                assert owners
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph.replay()
            torch.cuda.synchronize()
            for graph,outputs,owners,stream in captures:
                metrics=[error_metrics(a,b) for a,b in zip(outputs,expected)]
                assert all(bool(t.isfinite().all()) for t in (*outputs,*expected))
                assert all(v['normalized_rms'] < .025 for v in metrics),metrics
                checks.append(dict(generation=generation,metrics=metrics))
        report['graph_checks_after_cache_clear']=checks
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({key:report[key] for key in ('mode','pair_median_ms','live_bytes','peak_allocated_bytes','intercepted')}))


if __name__ == '__main__':
    main()
