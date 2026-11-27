"""Compare the captured FP32 accumulator handoff with the current G2 optimizer.

Run in the detector environment with the study's batched-addmm detector on
PYTHONPATH. Large mode uses the actual update-20 parameter/group/stride
geometry with synthetic gradients; its timings are optimizer-only.
"""
import argparse
import json
import statistics

import torch

from benchmarks.g2detr_bound_update import bind, bound_class


def groups(path):
    if not path:
        return dict(adamw=[({},[((64,),torch.float32,(1,)),((32,64),torch.bfloat16,(64,1))])],
                    muon=[({},[((128,64),torch.bfloat16,(64,1))]*3+[((64,32),torch.float32,(32,1))])])
    import numpy as np
    with torch.serialization.safe_globals([np.ndarray,np.dtype,np.dtypes.UInt32DType,np._core.multiarray._reconstruct]):
        data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    result={}
    for name in ('adamw','muon'):
        w=data['optimizer'][name]
        result[name]=[]
        for g in w['optimizer']['param_groups']:
            options={k:v for k,v in g.items() if k not in ('params','param_names','lr','initial_lr')}
            result[name].append((options,[(tuple(t.shape),t.dtype,t.stride())
                for n in g['param_names'] for t in (data['model'][n],)]))
    return result


class Schedule:
    def lr_at(self,step):return .0007/(1+step*.01)


def make(spec,bound,clip,large):
    from g2_detr.training.optimizer import GraphedOptimizerUpdate,GradientAccumulator
    from g2_detr.optim import DualOptimizer,FP32MasterOptimizer
    rng=torch.Generator(device='cuda').manual_seed(739)
    wrappers=[]
    for name,cls in (('adamw',torch.optim.AdamW),('muon',torch.optim.Muon)):
        gs=[]
        for opts,items in spec[name]:
            ps=[torch.nn.Parameter(torch.empty_strided(shape,strides,device='cuda',dtype=dtype).normal_(
                std=.02,generator=rng)) for shape,dtype,strides in items]
            gs.append(dict(opts,params=ps))
        wrappers.append(FP32MasterOptimizer(gs,cls,lr=.0007,**({'fused':True} if name=='adamw' else {})))
    dual=DualOptimizer(*wrappers,manifest={})
    opts={'triton.cudagraphs':False,'max_autotune':large,'emulate_precision_casts':True}
    cls=bound_class(GraphedOptimizerUpdate) if bound else GraphedOptimizerUpdate
    update=cls(dual,Schedule(),compile_options=opts,max_grad_norm=clip)
    acc=GradientAccumulator(update.parameters,2,opts,loss_predivided=True)
    if bound:bind(update,acc)
    return update,acc


def check(base,challenger):
    a,b=base[0],challenger[0]
    assert a.update_index==b.update_index
    for x,y in zip((*a.parameters,*a.masters),(*b.parameters,*b.masters),strict=True):
        torch.testing.assert_close(x,y,atol=0,rtol=0)
    for k in ('opt_adamw','opt_muon'):
        o,p=(getattr(m.optimizer,k) for m in (a,b))
        for x,y in zip(o.master_parameters,p.master_parameters,strict=True):
            state=o.inner.state[x];assert state.keys()==p.inner.state[y].keys()
            for name,v in state.items():torch.testing.assert_close(v,p.inner.state[y][name],atol=0,rtol=0)


def run(path,clip):
    large=bool(path)
    spec=groups(path)
    pairs=[make(spec,b,clip,large) for b in (False,True)]
    device='cuda'
    loss=torch.zeros((),device=device)
    ts=[[],[]]
    prev=None
    for step in range(14 if large else 5):
        loss.fill_(step+.25)
        scale=(0.,.00001,.3)[step%3]
        rng=torch.Generator(device=device).manual_seed(4322+step)
        for j,(u,acc) in enumerate(pairs):
            if not large:
                # Exercise the real two-microbatch compiled accumulator.
                rng.manual_seed(4322+step)
                for micro in range(2):
                    u.optimizer.zero_grad(set_to_none=True)
                    for p in u.parameters:
                        p.grad=torch.empty_like(p).normal_(std=scale,generator=rng)
                    acc.accumulate(loss,first=micro==0)
            else:
                # Identical but changing gradients. No compilation or fills
                # inside the event pair and no data-dependent CPU waits.
                torch._foreach_zero_(acc.buffers)
                torch._foreach_add_(acc.buffers,scale)
        metrics=[None,None]
        for i in ((0,1) if step%2==0 else (1,0)):
            u,acc=pairs[i]
            start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record();metrics[i]=u.step(loss,grads=acc.finish());end.record();end.synchronize()
            if step>=4:ts[i].append(start.elapsed_time(end))
        torch.testing.assert_close(*metrics,atol=0,rtol=0)
        if prev is not None:
            for m,copy in prev:torch.testing.assert_close(m,copy,atol=0,rtol=0)
        prev=[(m,m.clone()) for m in metrics]
        if not large or step in (0,13):check(*pairs)
        assert pairs[1][0].master_grads is pairs[1][1].buffers
    print(json.dumps(dict(parameters=len(pairs[0][0].parameters),clip=clip,exact=True,
        pairs_ms=ts,stock_median_ms=statistics.median(ts[0]),bound_median_ms=statistics.median(ts[1]))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    args=parser.parse_args()
    run(args.checkpoint,1.)
    if not args.checkpoint:run(None,None)
