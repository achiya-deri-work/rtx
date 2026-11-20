"""One fixed-router whole-MoE control for active-ID vs phase-balanced search."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--router-result',type=Path,required=True)
    p.add_argument('--arm',choices=('active_ids','phase_balanced'),help='Run one arm only in a separately identified study')
    args=p.parse_args();source=args.source.resolve();output=args.output.resolve()
    from benchmarks.profile_moe_active_budget import identity,write,worker_command
    from rtx.autotune.supervisor import supervise_command,DEVICE_FAULT_EXIT_CODE
    saved=json.loads(args.router_result.read_text())
    output.mkdir(parents=True,exist_ok=True)
    write(output/'router-fixed.json',dict(config=saved['router_config']))
    row=dict(id='phase-heldout-d384-e80-t20480',
        case=dict(d_model=384,num_experts=80,top_k=4,expert_hidden_dim=320,shared_hidden_dim=256),
        lengths=[2560]*8,layout='flat',seed=806432,stable_experts=True)
    manifest=dict(identity=identity(source),context=row,
        router_config=saved['router_config'],affinity_layout=saved['router_affinity_layout'],
        question='Same active-program IDs and fixed router; measure value of phase-balanced later trials',
        protocol='actual8->32 per expert/shared component; no router search; default-mode compiled whole F+B',
        arms=[args.arm] if args.arm else ['active_ids','phase_balanced'])
    write(output/'manifest.json',manifest);events=[]
    for budget,arm in [(8,'active_ids'),(8,'phase_balanced'),(32,'phase_balanced'),(32,'active_ids')]:
        if args.arm and arm!=args.arm:continue
        if identity(source)!=manifest['identity']:raise ValueError('Frozen source/driver drift')
        folder=output/arm;target=folder/f'budget-{budget}.json'
        if target.exists() and json.loads(target.read_text()).get('status')=='complete':continue
        settings=SimpleNamespace(expert_trials=budget,shared_trials=budget,router_trials=0,
            tune_seconds=300.,rounds=3,samples=7)
        command=worker_command(source,target,row,settings)
        command+=['--router-config',str(output/'router-fixed.json'),
                  '--affinity-layout',saved['router_affinity_layout'],
                  '--tuning-dir',str(folder/'tuning')]
        if arm=='phase_balanced':
            command=[command[0],'-u','-m','benchmarks.astra32_phase_search',
                     '--module','benchmarks.benchmark_moe_varlen','--',*command[3:]]
        write(folder/f'command-{budget}.json',command)
        print('START',arm,budget,flush=True)
        with (folder/f'run-{budget}.log').open('x') as stream:
            code=supervise_command(command,cwd=source,stall_timeout_s=240,
                wall_timeout_s=1200,output=stream)
        events.append(dict(arm=arm,budget=budget,exit_code=code))
        write(output/'progress.json',events);print('DONE',arm,budget,code,flush=True)
        if code:raise RuntimeError('Device fault' if code==DEVICE_FAULT_EXIT_CODE else 'Worker failed; inspect log')
        result=json.loads(target.read_text())
        if result.get('status')!='complete' or result['memory']['live_growth_bytes']:
            raise ValueError('Incomplete qualification or live-memory growth')
    write(output/'finished.json',dict(events=events))


if __name__=='__main__':main()
