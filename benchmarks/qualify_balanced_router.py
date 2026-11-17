"""Alternating, same-input end-to-end A/B of shared-tuner chosen schedules."""
import argparse
import json
from pathlib import Path
import torch

from rtx.routing import BalancedRouter,RoutingKernelConfig
from rtx.autotune.deployment_qualification import qualify_pair
from rtx.autotune.outcomes import TrialOutcome
from benchmarks.benchmark_balanced_router import measure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--samples',type=int,default=13)
    p.add_argument('--warmup-ms',type=float,default=200)
    args=p.parse_args()
    study=json.loads(Path(args.study).read_text())
    output=dict(gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                source_study=args.study,results=[])
    for row in study['results']:
        s=row['shape'];n,d,e,k,rd=(s[key] for key in ('tokens','d_model','experts','top_k','router_dim'))
        torch.manual_seed(study['seed'])
        base=BalancedRouter(d,e,top_k=k,router_dim=rd,device='cuda',autotune='off')
        with torch.no_grad():base.projection_weight.normal_(std=.02)
        challenger=BalancedRouter(d,e,top_k=k,router_dim=rd,device='cuda',autotune='off',
                                   config=RoutingKernelConfig.from_dict(row['tuning']['config']))
        challenger.load_state_dict(base.state_dict())
        x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        ns=base.top_k*n//e
        coefficient=torch.randn(e,ns,device='cuda')
        functions=[torch.compile(r,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
                   for r in (base,challenger)]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        results={}
        for phase in ('forward','forward_backward'):
            def evaluate(index):
                fn=functions[index];router=(base,challenger)[index]
                def call():
                    route=fn(x)
                    if phase=='forward':return route
                    return torch.autograd.grad((route.route_weights*coefficient).sum(),(x,*router.parameters()))
                measured=measure(call,samples=args.samples,stream=stream,warmup_ms=args.warmup_ms)
                eligible=measured['sampling']['collection']['stationary']
                print(f'{n} {phase} candidate={index} {measured["median_ms"]*1000:.3f}us eligible={eligible}',flush=True)
                return TrialOutcome('ok' if eligible else 'runtime_error',
                    median_ms=measured['median_ms'] if eligible else None,
                    timings_ms=measured['timings_ms'],metadata=measured,
                    error=None if eligible else 'unstable mobile clock/timing samples')
            try:
                selected,decision=qualify_pair(0,1,evaluate,rounds=args.rounds,margin=.01)
                results[phase]=dict(selected=selected,**decision)
            except RuntimeError as error:
                results[phase]=dict(status='inconclusive',error=str(error))
        output['results'].append(dict(shape=s,phases=results))
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
