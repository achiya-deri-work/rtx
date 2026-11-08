"""Bounded whole-MLP search and paired qualification for an application policy.

Uses the shared tuner, oracle, residual store and deployment qualification;
does not publish runtime winners or modify a caller's model. Synthetic BF16
inputs are a kernel study, not a replacement for real-fixture qualification.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path

import torch


def initial_schedule(document,shape):
    """Retarget a schedule, not tensor state or numerical policy.

    Only uniform reduction ranges depend on token count. Their split count,
    geometry and final-reducer policy remain unchanged; normal legality still
    rejects a template that cannot execute the requested shape.
    """
    from rtx.mlp import MXFP8MLPConfig
    from rtx._bwd_search import reduction_chunk
    if 'selected_config' in document:
        document=document['selected_config']
    elif 'result' in document:
        # Dataclasses.asdict on older result objects omits to_dict's schema
        # revision. Recover it only from the explicitly recorded initial ABI.
        document={'revision':document['initial']['revision'],**document['result']['config']}
    elif 'config' in document:
        document=document['config']
    c=MXFP8MLPConfig.from_dict(document)
    for role in ('gate_backward','down_backward'):
        bwd=getattr(c,role)
        if bwd is not None:
            for leg in ('dx','dw'):
                schedule=getattr(bwd,leg)
                if schedule.split_reduction>1 and schedule.split_partition=='uniform':
                    length=shape[0] if leg=='dw' else (2*shape[2] if role=='gate_backward' else shape[3])
                    alignment=(schedule.gemm.tile_k if schedule.backend=='decomposed'
                               else schedule.fused.bf16_tile_k)
                    bwd=replace(bwd,**{leg:replace(schedule,reduction_tile=
                        reduction_chunk(length,schedule.split_reduction,alignment))})
            c=replace(c,**{role:bwd})
    return c


def trace_schedule(inputs,c,output):
    """Steady compiled kernels only; use nsys --capture-range=cudaProfilerApi."""
    from rtx.mlp import mxfp8_mlp
    values=tuple(t.detach().requires_grad_() for t in inputs)
    dy=torch.randn(values[0].shape[0],values[2].shape[0],device='cuda',dtype=torch.bfloat16)*.1
    fn=torch.compile(lambda *a:mxfp8_mlp(*a,config=c),fullgraph=True,dynamic=False,
                     options={'triton.cudagraphs':False})
    for _ in range(3):
        y=fn(*values);grads=torch.autograd.grad(y,values,dy)
    del y,grads
    torch.cuda.synchronize()
    before=torch.cuda.memory_allocated()
    reserved=torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(3):
        with torch.cuda.nvtx.range('long_narrow/forward'):
            y=fn(*values)
        with torch.cuda.nvtx.range('long_narrow/backward'):
            grads=torch.autograd.grad(y,values,dy)
        del y,grads
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    (output/'profile-config.json').write_text(c.to_json())
    (output/'profile-memory.json').write_text(json.dumps(dict(
        allocated_before=before,allocated_after=torch.cuda.memory_allocated(),
        peak_allocated=torch.cuda.max_memory_allocated(),reserved_before=reserved,
        reserved_after=torch.cuda.memory_reserved(),
        caveat='Warm compiled direct calls, no outer graph; PyTorch counters omit driver and graph-pool allocations'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--configs',type=Path,help='Captured MLP selection list with key/config entries')
    source.add_argument('--initial',type=Path,help='Raw config or prior result; adapt uniform chunks to the new shape')
    p.add_argument('--shape',required=True,help='M,K,H,O')
    p.add_argument('--output',type=Path,required=True,help='New directory; no overwrite')
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--history',type=Path,help='Compatible residual root to resume; reports still go to a new output')
    p.add_argument('--diagnostic-portfolio',action='store_true',
                   help='Add explicit preparation/overlap challengers; default uses normal shared proposals')
    p.add_argument('--trace-only',action='store_true',help='Profile the initial schedule without search or timing claims')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    m,k,h,o=map(int,args.shape.split(','))
    from rtx.mlp import MXFP8MLPConfig
    from rtx.autotune.mlp import tune_mxfp8_mlp,MLP_AUTOTUNE_REVISION
    from rtx.autotune.effort import AutotunePolicy
    from rtx._bwd_search import balanced_dw_candidates
    from rtx.kernels.mxfp8 import MXFP8Problem
    if args.configs:
        rows=json.loads(args.configs.read_text())
        row=next(r for r in rows if r['key']==[[m,k],[2*h,k],[o,h],'silu'])
        initial=MXFP8MLPConfig.from_dict(row['config'])
    else:
        initial=initial_schedule(json.loads(args.initial.read_text()),(m,k,h,o))
    torch.manual_seed(81110);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    inputs=tuple(torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*scale
        for shape,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    if args.trace_only:
        trace_schedule(inputs,initial,args.output)
        return
    fused=replace(initial,joint_output_storage='separate',gradient_quantization='recompute',
        backward_overlap=False,recompute=replace(initial.recompute,
            gemm=replace(initial.recompute.gemm,glu_grad_store='exchange')))
    balanced=initial
    for role,problem in (('gate_backward',MXFP8Problem(m,2*h,k)),
                         ('down_backward',MXFP8Problem(m,o,h))):
        choices=balanced_dw_candidates(problem,getattr(initial,role),
            torch.cuda.get_device_properties(0).multi_processor_count)
        if choices:balanced=replace(balanced,**{role:choices[0]})
    portfolio=((balanced,replace(fused,hidden_quantization='recompute'),fused,
               replace(initial,backward_overlap=True)) if args.diagnostic_portfolio else ())
    policy=AutotunePolicy(max_trials=args.trials,max_seconds=args.seconds,
        search_schedule='progressive',warmup=3,samples=9,calls_per_sample=3,
        confirmation_repeats=0,exploration_trials=1)
    result=tune_mxfp8_mlp(*inputs,initial=initial,phase='train',precision='mxfp8',
        tuning_policy=policy,portfolio=portfolio,output_dir=args.history or args.output/'residuals',
        qualification_rounds=args.rounds,qualification_margin=.01,
        progress=lambda message:print(message,flush=True))
    root=Path(__file__).resolve().parents[1]
    payload=dict(shape=[m,k,h,o],seed=81110,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=str(torch.cuda.get_device_properties(0)),revision=MLP_AUTOTUNE_REVISION,
        source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in sorted((root/'rtx').rglob('*.py'))},
        initial=initial.to_dict(),selected_config=result.config.to_dict(),result=asdict(result),
        arguments=vars(args),benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        history=None if args.history is None else str(args.history),
        caveat='Synthetic-input complete compiled MLP qualification; not a 5090 or detector-step speedup')
    (args.output/'result.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
    decision=result.winner_decision or {}
    qualification=decision.get('deployment_qualification',{})
    print('COMPLETE',json.dumps(dict(median_ms=result.median_ms,
        reason=decision.get('reason'),promoted=qualification.get('promoted'),
        aggregate_speedup=qualification.get('aggregate_speedup'))),flush=True)


if __name__=='__main__':main()
