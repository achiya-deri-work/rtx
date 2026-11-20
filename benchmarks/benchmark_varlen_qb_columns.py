"""Measure exact packed QB radix columns, including optional key transpose."""
import argparse
import itertools
import json
import random
import statistics

import torch

from benchmarks.benchmark_varlen_sinkhorn_shards import CASES, LENGTHS, capture, sample
from rtx._routing_varlen import layout, transport_forward
from rtx.autotune.varlen_routing import source_revision
from rtx.kernels.routing_varlen import compile_varlen_quantile
from rtx.routing import RoutingKernelConfig


def rank(scores,lengths,k):
    values=[];offset=0;e=scores.shape[1]
    for n in lengths:
        values.append(scores[offset:offset+n].sort(0,descending=True).values[n*k//e])
        offset+=n
    value=torch.stack(values)
    return torch.where(value==0,0.,value)


def run(e,k,scale,args):
    torch.manual_seed(99371)
    lengths=tuple(x*scale for x in LENGTHS);t=sum(lengths);b=len(lengths)
    cu=torch.tensor([0,*itertools.accumulate(lengths)],device='cuda',dtype=torch.int32)
    source=torch.nn.functional.logsigmoid(torch.randn(t,e,device='cuda')*1.5)
    cfg=RoutingKernelConfig(packed_row_backend='warp')
    key=layout(t,b,e,k,5,tuple(cfg.to_token()))
    value,_,_,_,seq,*_=transport_forward(source,cu,key)
    # Model a real late QB round: recompute the row cutoff after each new
    # sequence-local price. Don't reuse a key order across rounds.
    value=value[:t];seq=seq[:t];prices=torch.zeros(b,e,device='cuda')
    for _ in range(5):
        column_values=value-(value-prices[seq]).topk(k+1,dim=1).values[:,k:k+1]
        prices=rank(column_values,lengths,k)
    graphs={};buffers=[];stats={}
    for threads,items,expert_keys in itertools.product(args.threads,args.items,(False,True)):
        name=f"{'expert-keys' if expert_keys else 'row-major'}-t{threads}-i{items}"
        fn=compile_varlen_quantile(t,b,e,k,threads,items,expert_keys)
        out=torch.empty(b,e,device='cuda')
        workspace=[torch.empty(t*e,dtype=torch.uint32,device='cuda')] if expert_keys else []
        def step():fn(column_values.flatten(),cu,out.flatten(),*workspace)
        graph,_=capture(step);graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(out.view(torch.int32),prices.view(torch.int32),rtol=0,atol=0)
        graphs[name]=graph;stats[name]=[];buffers.append((out,workspace))
    rng=random.Random(6153);order=list(graphs)
    for _ in range(args.rounds):
        rng.shuffle(order)
        for name in order:stats[name].append(sample(graphs[name],args.samples,args.replays))
    for graph in graphs.values():graph.reset()
    return dict(e=e,k=k,lengths=lengths,exact=True,result={
        name:dict(samples=values,median=statistics.median(map(statistics.median,values)),
                  min=min(itertools.chain.from_iterable(values))) for name,values in stats.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experts',type=int,action='append',choices=[e for e,k in CASES])
    parser.add_argument('--scale',type=int,action='append',choices=[1,4])
    parser.add_argument('--threads',type=int,action='append',choices=[64,128,256])
    parser.add_argument('--items',type=int,action='append',choices=[0,4,8,16,32,64])
    parser.add_argument('--rounds',type=int,default=3);parser.add_argument('--samples',type=int,default=9)
    parser.add_argument('--replays',type=int,default=64)
    args=parser.parse_args()
    if min(args.rounds,args.samples,args.replays)<1:parser.error('Iteration counts must be positive')
    args.threads=args.threads or [128,256];args.items=args.items or [0,16,32,64]
    print(json.dumps(dict(device=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,
        cuda=torch.version.cuda,source=source_revision(),args=vars(args))),flush=True)
    for e,k in CASES:
        if not args.experts or e in args.experts:
            for scale in args.scale or (1,4):print(json.dumps(run(e,k,scale,args)),flush=True)


if __name__=='__main__':main()
