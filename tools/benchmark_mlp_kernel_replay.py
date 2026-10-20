"""Compare host launches, private CUDA replay and warmed outer CUDA capture.

No tuning, no internal/outer torch.compile: identical explicit MXFP8 policy.
Run each mode in a fresh process. Timings include functional output ownership.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import rtx


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('direct','replay','outer'), required=True)
    p.add_argument('--shape', default='512,768,1536,768')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile', action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    from rtx._mlp_autotune import _selected_forward,_selected_backward
    m,k,h,o=map(int,args.shape.split(','))
    torch.manual_seed(1962)
    layer=rtx.MXFP8MLP(k,h,o,device='cuda',autotune='cache',
        autotune_cache_dir=args.output.with_suffix('.unused-cache'))
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    replay=args.mode=='replay'
    def forward():
        return _selected_forward(x,layer.gate_up_weight,layer.down_weight,
            'silu',layer._first_hit_spec,True,replay=replay)
    with torch.no_grad():
        _,token=forward()
        def backward():
            return _selected_backward(x,layer.gate_up_weight,layer.down_weight,dy,
                token,'silu',True,True,True,replay=replay)
        def step():return forward(),backward()
        start=time.perf_counter()
        for _ in range(4):step()
        torch.cuda.synchronize()
        warmup_seconds=time.perf_counter()-start
        routines={'forward':forward,'backward':backward,'fwd_bwd':step}
        if args.mode=='outer':
            for name,fn in tuple(routines.items()):
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):fn()
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):outputs=fn()
                def captured(g=graph,owned=outputs):
                    g.replay()
                    return owned
                routines[name]=captured
        timings={}
        for name,fn in routines.items():
            for _ in range(5):fn()
            torch.cuda.synchronize()
            samples=[]
            for _ in range(20):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(10):result=fn()
                b.record();b.synchronize()
                samples.append(a.elapsed_time(b)/10*1000)
            timings[name]=dict(median_us=statistics.median(samples),min_us=min(samples),
                              stdev_us=statistics.stdev(samples),samples_us=samples)
        if args.profile:
            torch.cuda.profiler.start()
            with torch.cuda.nvtx.range('rtx_mlp_kernel_replay_steady'):
                for _ in range(10):routines['fwd_bwd']()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        report=dict(mode=args.mode,shape=[m,k,h,o],torch=str(torch.__version__),
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),seed=1962,
            timings=timings,warmup_seconds=warmup_seconds,
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            config=layer.config.to_dict() if hasattr(layer,'config') else 'default',
            scope='Identical default config; CUDA events include host feeding and boundary copies. Outer capture uses fixed caller buffers; no optimizer.')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps(report))


if __name__=='__main__':main()
