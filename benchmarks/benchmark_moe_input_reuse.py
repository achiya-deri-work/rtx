"""Paired raw-core global-token input reuse A/B; no public-API speed claim.

Both policies use identical frozen GEMM/quantization schedules and actual
router outputs. Only prepared.input_preparation differs. Ordinary warmed
external CUDAgraph events exclude Python dispatch, compilation and warmup.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def errors(actual,expected):
    import torch
    result={}
    for name,a,b in zip(('output','dx','gate_dw','down_dw','route_dw'),actual,expected):
        delta=a.detach().float()-b.detach().float()
        nrms=float(delta.norm()/b.float().norm().clamp_min(1e-12))
        maximum=float(delta.abs().max())
        finite=bool(a.isfinite().all())
        assert finite and (nrms<(.003 if name=='output' else .006) or maximum<2e-7),(name,nrms,maximum)
        result[name]=dict(nrms=nrms,max_abs=maximum,finite=finite,
                          bit_identical=bool(torch.equal(a,b)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-id',default='d768_e48_k6_h256')
    p.add_argument('--tokens',type=int,default=2000)
    p.add_argument('--seed-studies',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=9)
    p.add_argument('--rounds',type=int,default=3)
    args=p.parse_args()
    # Existing build helper records/retargets frozen auxiliary policy only;
    # its lazily constructed compiled public functions are never invoked here.
    args.config=None;args.tune_trials=0;args.skip_baseline=True
    args.whole_layer=False;args.fixture=None;args.save_fixture=None
    import torch
    from benchmarks.profile_moe import build,correctness,measure
    from rtx.moe import _run_forward,_run_backward
    from rtx.autotune.moe import source_revision
    case,shape,initial,origin,operands,dy,_unused_calls=build(args)
    policies={label:replace(initial,prepared=replace(initial.prepared,input_preparation=choice))
              for label,choice in (('baseline','gather_bf16'),('candidate','quantize_once'))}
    calls={}
    for label,config in policies.items():
        def forward(config=config):return _run_forward(*operands,config)
        def backward(config=config):return _run_backward(*operands,dy,config)
        def step(forward=forward,backward=backward):return (forward(),*backward())
        calls[label]=dict(forward=forward,backward=backward,step=step)
    record=dict(status='running',case=case,shape=shape,source_revision=source_revision(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Raw expert core only; frozen schedules; external CUDAgraph; NOT public API/whole MoE',
        timing_contract='AB/BA; three rounds by default; nine event samples; 200ms graph warmup',
        commands=__import__('sys').argv,device=str(torch.cuda.get_device_properties(0)),
        torch=torch.__version__,cuda=torch.version.cuda,seed_origin=origin,
        configs={label:c.to_dict() for label,c in policies.items()})
    started=time.monotonic()
    record['independent_oracle']=correctness(operands,dy,{'expert_step':calls['baseline']['step']},args.tokens)
    expected=calls['baseline']['step']()
    actual=calls['candidate']['step']()
    record['candidate_all_gradients_vs_qualified_baseline']=errors(actual,expected)
    del actual
    for label in calls:
        for call in calls[label].values():
            output=call();del output
    torch.cuda.synchronize()
    record['construction_compile_validation_seconds']=time.monotonic()-started
    record['timings']={label:{phase:[] for phase in calls[label]} for label in calls}
    stream=torch.cuda.Stream()
    for round_id in range(args.rounds):
        order=('baseline','candidate') if round_id%2==0 else ('candidate','baseline')
        for label in order:
            phases=list(calls[label]) if round_id%2==0 else list(reversed(calls[label]))
            for phase in phases:
                result=measure(calls[label][phase],stream,args.samples)
                record['timings'][label][phase].append(result)
                print('TIME',round_id,label,phase,result['median_ms'],flush=True)
                write(args.output,record)
    record['memory']={}
    for label in calls:
        live=[]
        for _ in range(8):
            result=calls[label]['step']();del result;torch.cuda.synchronize()
            live.append(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
        result=calls[label]['step']();torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated();del result
        record['memory'][label]=dict(live_samples=live,live_growth=max(live)-min(live),
            allocated_before=before,peak_allocated=peak,temporary_peak=peak-before,
            reserved=torch.cuda.memory_reserved())
    record['summary']={}
    for phase in calls['baseline']:
        medians={label:statistics.median(r['median_ms'] for r in record['timings'][label][phase]) for label in calls}
        record['summary'][phase]=dict(**{label+'_ms':v for label,v in medians.items()},
            speedup=medians['baseline']/medians['candidate'],
            round_relative_range={label:max(r['median_ms'] for r in record['timings'][label][phase])/
                min(r['median_ms'] for r in record['timings'][label][phase])-1 for label in calls})
    record['status']='complete';write(args.output,record)
    print(json.dumps(record['summary'],indent=2),flush=True)


if __name__=='__main__':main()
