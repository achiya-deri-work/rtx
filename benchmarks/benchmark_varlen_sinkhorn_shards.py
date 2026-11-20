"""CUDA-graph packed Sinkhorn component measurement (not expert execution).

Run from the source snapshot with `python -m benchmarks.benchmark_varlen_sinkhorn_shards`.
All candidate tensors and independent FP64 oracle inputs are identical. The
graph includes packing, all saved normalization history, and the score
adjoint; hard routing is outside this diagnostic. The production packed
routing tuner still selects using the whole router, not this benchmark.
"""
import argparse
from dataclasses import replace
import itertools
import json
import math
import random
import statistics

import torch

from rtx._routing_varlen import layout, transport_backward, transport_forward
from rtx.autotune.varlen_routing import source_revision
from rtx.routing import RoutingKernelConfig, SinkhornKernelConfig


CASES=((24,6),(32,4),(64,8),(48,6))
LENGTHS=(512,2000,768,4096)
SCHEDULES=(
    ('legacy-g4',4,128,0,0),
    ('legacy-g1',1,128,0,0),
    ('legacy-g8-t256',8,256,0,0),
    ('shard-4-g1',1,128,4,4),
    ('shard-8-g2',2,128,8,8),
    ('shard-16-g4',4,128,16,16),
    ('shard-16-g4-backward',4,128,0,16),
    ('shard-32-g8',8,128,32,32),
    ('shard-32-g8-t256',8,256,32,32),
)


def oracle(source,lengths,e,k,steps,dy):
    value=source.double().requires_grad_();outputs=[];losses=[];start=0;packed=0
    for length in lengths:
        if not length:continue
        multiple=e//math.gcd(e,k);n=math.ceil(length/multiple)*multiple
        part=torch.nn.functional.pad(value[start:start+length],(0,0,0,n-length),value=-math.log(2))
        for _ in range(steps):
            part=part-torch.logsumexp(part,dim=1,keepdim=True)+math.log(k)
            part=part-torch.logsumexp(part,dim=0,keepdim=True)+math.log(n*k//e)
        outputs.append(part);losses.append((part*dy[packed:packed+n]).sum());start+=length;packed+=n
    return torch.cat(outputs).detach(),torch.autograd.grad(sum(losses),value)[0]


def errors(actual,expected):
    actual=actual[:expected.shape[0]].double();delta=actual-expected
    result=dict(finite=bool(torch.isfinite(actual).all()),
                max_abs=float(delta.abs().max()),nrms=float(delta.norm()/expected.norm().clamp_min(1e-12)))
    if not result['finite'] or result['nrms']>=5e-6 or result['max_abs']>=5e-5:
        raise AssertionError(result)
    return result


def capture(step):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):result=step()
    torch.cuda.current_stream().wait_stream(stream)
    return graph,result


def sample(graph,samples,replays):
    # Multiple replays amortize event granularity. Each sample is a separate
    # event pair on the same current stream as graph replay.
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    out=[]
    for _ in range(samples):
        start.record()
        for _ in range(replays):graph.replay()
        end.record();end.synchronize()
        out.append(start.elapsed_time(end)*1000/replays)
    return out


def run(e,k,scale,args):
    torch.manual_seed(args.seed);lengths=tuple(x*scale for x in LENGTHS);t=sum(lengths);b=len(lengths)
    source=torch.nn.functional.logsigmoid(torch.randn(t,e,device='cuda')*1.5)
    cu=torch.tensor([0,*itertools.accumulate(lengths)],dtype=torch.int32,device='cuda')
    default=RoutingKernelConfig(packed_row_backend='warp')
    bound=layout(t,b,e,k,args.steps,tuple(default.to_token()))[4]
    dy=torch.randn(bound,e,device='cuda');expected=oracle(source,lengths,e,k,args.steps,dy)
    graphs={};metrics={};retained=[]
    for name,g,threads,sf,sb in SCHEDULES:
        if args.schedule and name not in args.schedule:continue
        geometry=SinkhornKernelConfig(threads=threads,column_groups=g)
        c=replace(default,forward=geometry,backward=geometry,
                  packed_column_shards_forward=sf,packed_column_shards_backward=sb)
        key=layout(t,b,e,k,args.steps,tuple(c.to_token()))
        def forward():return transport_forward(source,cu,key)
        def backward(f):return transport_backward(dy,f[1],f[2],f[4],f[6],key)
        def training():
            f=forward();return f[0],backward(f)
        graph,result=capture(training);graph.replay();torch.cuda.synchronize()
        metrics[name]=dict(output=errors(result[0],expected[0]),score_gradient=errors(result[1],expected[1]),
                           config=c.to_dict(),column_forward_scratch_bytes=b*e*sf*4,
                           column_backward_scratch_bytes=b*e*sb*4)
        fixed=forward();fg,y=capture(forward);bg,dx=capture(lambda:backward(fixed))
        retained.append((result,fixed,y,dx));graphs[name]=dict(training=graph,forward=fg,backward=bg)
    timings={name:{phase:[] for phase in phases} for name,phases in graphs.items()}
    order=list(itertools.product(graphs,('training','forward','backward')));rng=random.Random(args.seed)
    for _ in range(args.rounds):
        rng.shuffle(order)
        for name,phase in order:timings[name][phase].append(sample(graphs[name][phase],args.samples,args.replays))
    for name,phases in timings.items():
        metrics[name]['timings_us']={}
        for phase,rounds in phases.items():
            medians=[statistics.median(values) for values in rounds]
            metrics[name]['timings_us'][phase]=dict(samples=rounds,round_medians=medians,
                median=statistics.median(medians),minimum=min(itertools.chain.from_iterable(rounds)))
    if args.trace:
        graph=next(iter(graphs.values()))['training']
        torch.cuda.synchronize();torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range('packed Sinkhorn F+B graph replay'):
                graph.replay();torch.cuda.synchronize()
        finally:torch.cuda.profiler.stop()
    for values in graphs.values():
        for graph in values.values():graph.reset()
    return dict(e=e,k=k,lengths=lengths,steps=args.steps,scheduled_bound=bound,results=metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experts',type=int,action='append',choices=[e for e,k in CASES])
    parser.add_argument('--scale',type=int,action='append',choices=[1,4])
    parser.add_argument('--schedule',action='append',choices=[s[0] for s in SCHEDULES])
    parser.add_argument('--steps',type=int,default=5);parser.add_argument('--seed',type=int,default=7120)
    parser.add_argument('--rounds',type=int,default=3);parser.add_argument('--samples',type=int,default=9)
    parser.add_argument('--replays',type=int,default=12)
    parser.add_argument('--trace',action='store_true',help='Bracket one warmed graph replay with cudaProfilerApi/NVTX')
    args=parser.parse_args()
    if min(args.steps,args.rounds,args.samples,args.replays)<1:parser.error('Iteration counts must be positive')
    if args.trace and not all(len(values or ())==1 for values in (args.experts,args.scale,args.schedule)):
        parser.error('--trace requires one --experts, one --scale and one --schedule')
    print(json.dumps(dict(device=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,
        cuda=torch.version.cuda,source=source_revision(),args=vars(args))),flush=True)
    for e,k in CASES:
        if not args.experts or e in args.experts:
            for scale in args.scale or (1,4):print(json.dumps(run(e,k,scale,args)),flush=True)


if __name__=='__main__':main()
