"""Fixed-candidate host versus CUDA-graph timing diagnosis, not autotuning."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from rtx.bwd_autotune import bwd_config_from_dict
from rtx.bwd_experiments import BwdBenchmarkHarness
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec
from tools.research.matched_gpu import independent_metrics, source_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',required=True)
    p.add_argument('--config-id',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--profile',action='store_true',help='Capture a short post-warmup CUDA-profiler/NVTX range')
    p.add_argument('--stream-schedule',choices=('single','dual_stream'))
    p.add_argument('--execution-order',choices=('dx_first','dw_first'))
    args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    raw=Path(args.pool).read_bytes(); pool=json.loads(raw)
    config=bwd_config_from_dict(next(row['config'] for row in pool['configs'] if row['config_id']==args.config_id))
    overrides={k:getattr(args,k) for k in ('stream_schedule','execution_order') if getattr(args,k) is not None}
    config=replace(config,**overrides)
    shape=ShapeSpec(**pool['task']['shape'])
    harness=BwdBenchmarkHarness(shape,BenchmarkProtocol(),device='cuda:0',seed=pool['task']['seed'])
    candidate=harness.prepare(config)
    def run():
        candidate.runner(harness.grad_output,harness.x,harness.weight,candidate.grad_x,candidate.grad_weight)
    for _ in range(10): run()
    torch.cuda.synchronize()
    calls=32
    graphs={}
    capture_start=time.perf_counter()
    for mode,count in [('single_graph',1),('batch_graph',calls)]:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(count): run()
        graph.replay(); torch.cuda.synchronize()
        graphs[mode]=graph
    capture_s=time.perf_counter()-capture_start
    def batch(mode):
        if mode=='batch_graph': graphs[mode].replay()
        else:
            for _ in range(calls):
                if mode=='host': run()
                else: graphs[mode].replay()
    metrics={}
    for mode in ('host','single_graph','batch_graph'):
        batch(mode);torch.cuda.synchronize()
        metrics[mode]={label:independent_metrics(actual,ref,.07) for label,actual,ref in
            zip(('dX','dW'),(candidate.grad_x,candidate.grad_weight),harness._reference)}
        for _ in range(5): batch(mode)
        torch.cuda.synchronize()
    live_before=torch.cuda.memory_allocated()
    rows=[]
    for round_ in range(40):
        order=['host','single_graph','batch_graph']
        if round_%2: order.reverse()
        for mode in order:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();start.record()
            batch(mode)
            end.record();end.synchronize()
            rows.append(dict(round=round_,mode=mode,event_us=start.elapsed_time(end)*1000/calls,
                             wall_us=(time.perf_counter()-wall)*1e6/calls))
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for mode in ('host','single_graph','batch_graph'):
            with torch.cuda.nvtx.range(mode): batch(mode)
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    summary={}
    for mode in order:
        values=[row['event_us'] for row in rows if row['mode']==mode]
        summary[mode]=dict(median_us=statistics.median(values),min_us=min(values),max_us=max(values),
            relative_stdev=statistics.stdev(values)/statistics.mean(values))
    report=dict(source_config_id=args.config_id,config=asdict(config),overrides=overrides,
        shape=pool['task']['shape'],seed=pool['task']['seed'],
        calls=calls,samples=40,profiled=args.profile,summary=summary,measurements=rows,numerics=metrics,
        compile_ms=candidate.compile_ms,capture_s=capture_s,
        live_before=live_before,live_after=torch.cuda.memory_allocated(),
        torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        pool_sha256=hashlib.sha256(raw).hexdigest(),source_digest=source_digest(),
        scope='Fixed hot inputs and candidates; modes change launch batching, not arithmetic. No runtime promotion.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
