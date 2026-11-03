"""Isolated shared-dY ownership diagnostic; not a backward speedup claim."""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=('shared128','roles128','shared256','roles256'))
    p.add_argument('--register-cap',type=int,choices=(80,96,255),default=255)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    from rtx.kernels.ms_eden_shared_dy import compile_shared_dy
    torch.manual_seed(984)
    source=torch.randn(4096,1536,device='cuda',dtype=torch.bfloat16)*.25
    signs=[(torch.randint(2,(128,),device='cuda')*2-1).float() for _ in range(2)]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    graphs,outputs,runners,arguments={},{},{},{}
    for threads in (128,256):
        for roles in (False,True):
            name=('roles' if roles else 'shared')+str(threads)
            run,count=compile_shared_dy(4096,1536,threads,128,16,128,0,warp_roles=roles,
                                       register_cap=args.register_cap if roles else 255)
            out=[(torch.empty(source.numel()//2,device='cuda',dtype=torch.uint8),
                  torch.empty(source.numel()//16,device='cuda',dtype=torch.bfloat16),
                  torch.empty(count,device='cuda')) for _ in range(2)]
            call=(source.flatten(),signs[0],*out[0],signs[1],*out[1])
            with torch.cuda.stream(stream):
                for _ in range(3):run(*call)
            stream.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                for _ in range(8):run(*call)
            graphs[name],outputs[name],runners[name],arguments[name]=graph,out,run,call
    torch.cuda.synchronize()
    for out in outputs.values():
        for actual,expected in zip(out,outputs['shared128']):
            for a,b in zip(actual[:2],expected[:2]):assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
            assert torch.isfinite(actual[2]).all()
            torch.testing.assert_close(actual[2].max(),expected[2].max(),rtol=0,atol=0)
    samples={name:[] for name in graphs}
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
        runners[args.profile](*arguments[args.profile])
        torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
    else:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            for graph in graphs.values():graph.replay()
            torch.cuda.synchronize()
        rng=random.Random(984)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        before=torch.cuda.memory_allocated()
        for _ in range(24):
            order=list(graphs);rng.shuffle(order)
            for name in order:
                start.record();graphs[name].replay();end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/8)
        assert torch.cuda.memory_allocated()==before
    root=Path(__file__).resolve().parents[1]
    record=dict(shape=[4096,1536],device=torch.cuda.get_device_name(),torch=torch.__version__,
        profile=args.profile,register_cap=args.register_cap,exact_outputs=True,samples_us=samples,
        median_us={k:statistics.median(v) for k,v in samples.items() if v},
        scope='isolated two-orientation shared dY preparation; excludes finalization and GEMMs',
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'rtx/kernels/ms_eden_shared_dy.py','benchmarks/compare_ms_eden_shared_roles.py')})
    with args.output.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record['median_us']),flush=True)


if __name__=='__main__':main()
