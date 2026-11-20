"""Frozen baseline/new-space 8->32 router comparison with fixed expert plans."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,choices=(0,1),required=True)
    args=p.parse_args()
    from benchmarks.profile_moe_active_budget import identity,write,worker_command
    from rtx.autotune.supervisor import supervise_command,DEVICE_FAULT_EXIT_CODE
    sources={name:path.resolve() for name,path in [('baseline',args.baseline),('candidate',args.candidate)]}
    cases=[dict(d_model=384,num_experts=96,top_k=4,expert_hidden_dim=320,shared_hidden_dim=256),
           dict(d_model=512,num_experts=120,top_k=6,expert_hidden_dim=256,shared_hidden_dim=512)]
    row=dict(id=f'new-flat-{args.shard}',case=cases[args.shard],lengths=[(3072,3840)[args.shard]]*8,
        layout='flat',seed=86323+args.shard,stable_experts=True)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=dict(context=row,identities={n:identity(s) for n,s in sources.items()},
        contract='baseline8 chooses expert/shared plans, then frozen for all arms; only router8->32 differs',
        settings=dict(seconds=300.,rounds=3,samples=7),fresh_shapes=True)
    write(output/'manifest.json',manifest);events=[]
    deep=('candidate','baseline') if args.shard==0 else ('baseline','candidate')
    for arm,budget in [('baseline',8),('candidate',8),*((a,32) for a in deep)]:
        source=sources[arm]
        if identity(source)!=manifest['identities'][arm]:raise ValueError('Frozen source drift')
        folder=output/arm;target=folder/f'budget-{budget}.json'
        if target.exists() and json.loads(target.read_text()).get('status')=='complete':continue
        initial=arm=='baseline' and budget==8
        settings=SimpleNamespace(expert_trials=8 if initial else 0,shared_trials=8 if initial else 0,
            router_trials=budget,tune_seconds=300.,rounds=3,samples=7)
        command=worker_command(source,target,row,settings)
        command+=['--tuning-dir',str(folder/'tuning')]
        if not initial:
            for name in ('expert','shared'):
                command+=['--'+name+'-config',str(output/'baseline'/f'budget-8.{name}.json')]
            old=json.loads((output/'baseline'/'budget-8.json').read_text())
            command+=['--affinity-layout',old['router_affinity_layout']]
        write(folder/f'command-{budget}.json',command)
        print('START',arm,budget,flush=True)
        with (folder/f'run-{budget}.log').open('a') as log:
            code=supervise_command(command,cwd=source,stall_timeout_s=240,wall_timeout_s=1200,output=log)
        events.append(dict(arm=arm,budget=budget,exit_code=code));write(output/'progress.json',events)
        print('DONE',arm,budget,code,flush=True)
        if code:raise RuntimeError('Device fault' if code==DEVICE_FAULT_EXIT_CODE else 'Worker failed')
        result=json.loads(target.read_text())
        if result.get('status')!='complete' or result['memory']['live_growth_bytes']:
            raise ValueError('Incomplete oracle/replay/memory qualification')
    write(output/'finished.json',dict(events=events))


if __name__=='__main__':main()
