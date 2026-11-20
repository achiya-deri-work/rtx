"""Frozen, supervised prospective coverage/retrieval ablations on fresh MoEs."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def contexts():
    def row(name,d,e,k,h,s,b,length,layout):
        return dict(id=name,case=dict(d_model=d,num_experts=e,top_k=k,expert_hidden_dim=h,shared_hidden_dim=s),
            lengths=[length]*b,layout=layout,samples=b,total_tokens=b*length,seed=384720,stable_experts=True)
    return [row('flat-heldout-d384-e72',384,72,4,320,256,8,2304,'flat'),
            row('flat-heldout-d512-e104',512,104,6,256,512,8,3328,'flat'),
            row('packed-sentinel-d384-e40',384,40,4,320,256,8,1920,'packed'),
            row('short-flat-sentinel-d384-e64',384,64,4,320,256,8,128,'flat')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,choices=(0,1),required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--depth',action='store_true',help='Also resume ordinary/regions to 16 and 32 actual trials per component')
    args=p.parse_args();source=args.source.resolve();args.output.mkdir(parents=True,exist_ok=True)
    from benchmarks.profile_moe_active_budget import identity,write,worker_command
    from rtx.autotune.supervisor import supervise_command,DEVICE_FAULT_EXIT_CODE
    rows=contexts()[args.shard::2]
    manifest=dict(identity=identity(source),contexts=rows,depth=args.depth,prior=str(args.prior.resolve()),
        question='Equal actual effort: ordinary vs geometry regions vs retrieval vs both vs both+ranker',
        gates='Same input/parameters/upstream; independent quantized FP64 experts, all-leaf graph replay, zero live growth')
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Resume contract changed')
    write(path,manifest);events=[]
    for index,row in enumerate(rows):
        arms=['ordinary','regions'] if index else ['ordinary','regions','retrieval','regions_retrieval','regions_hybrid']
        if args.shard:arms=list(reversed(arms))
        tasks=[(arm,8) for arm in arms]
        if args.depth and not index:tasks += [(arm,budget) for budget in (16,32) for arm in ('ordinary','regions')]
        for arm,budget in tasks:
            if identity(source)!=manifest['identity']:raise ValueError('Frozen runtime/driver drift')
            folder=args.output/row['id']/arm;target=folder/f'budget-{budget}.json'
            if target.exists() and json.loads(target.read_text()).get('status')=='complete':continue
            settings=SimpleNamespace(expert_trials=budget,shared_trials=budget,router_trials=budget,
                tune_seconds=300.,rounds=3,samples=7)
            command=worker_command(source,target,row,settings)
            command+=['--region-coverage' if arm.startswith('regions') else '--no-region-coverage',
                      '--tuning-dir',str(folder/'tuning')]
            if 'retrieval' in arm or 'hybrid' in arm:
                command+=['--prior-artifact',str(args.prior.resolve()),'--allow-research-prior',
                          '--prior-search','hybrid' if 'hybrid' in arm else 'retrieval']
            write(folder/f'command-{budget}.json',command)
            print('START',row['id'],arm,budget,flush=True)
            with (folder/f'run-{budget}.log').open('a') as log:
                code=supervise_command(command,cwd=source,environment={**os.environ,'OMP_NUM_THREADS':'4',
                    'MKL_NUM_THREADS':'4','TORCHINDUCTOR_COMPILE_THREADS':'4'},output=log,
                    wall_timeout_s=1200,stall_timeout_s=240)
            events.append(dict(context=row['id'],arm=arm,budget=budget,exit_code=code))
            write(args.output/'progress.json',events);print('DONE',row['id'],arm,budget,code,flush=True)
            if code==DEVICE_FAULT_EXIT_CODE:raise RuntimeError('Device fault; no more GPU work')
            if code:continue
            result=json.loads(target.read_text())
            if result.get('status')!='complete' or result['memory']['live_growth_bytes']:
                raise ValueError('Qualification incomplete or live memory grows')
    write(args.output/'finished.json',dict(events=events))


if __name__=='__main__':main()
