"""Compare source snapshots with isolated workers and a common graph protocol.

Autotune each snapshot separately first. This driver reuses the shared MLP
evaluator and paired decision rule; it neither tunes nor installs a winner.
Workers take turns on the GPU. A benchmark-only warmup shim makes old/new
evaluators use the same measured-GPU-time warmup without changing RTX files.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys


PREFIX='RTX_COMPARE '


def emit(value):
    print(PREFIX+json.dumps(value,default=str),flush=True)


def worker(args):
    import torch
    from benchmarks.qualify_mxfp8_long_mlp import initial_schedule
    from rtx.autotune.mlp import make_mlp_adapter
    import rtx.autotune.preparation_runtime as timing
    torch.manual_seed(args.seed);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    m,k,h,o=map(int,args.shape.split(','))
    config=initial_schedule(json.loads(args.config.read_text()),(m,k,h,o))
    values=tuple(torch.randn(*s,device='cuda',dtype=torch.bfloat16)*scale
        for s,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    # Old evaluators replay once before timing; make new ones identical. The
    # common outer loop below supplies measured GPU warmup to BOTH revisions.
    def one_replay(graph):
        graph.replay()
        return {'external_revision_comparison_warmup':True}
    timing._warm_measurement_graph=one_replay
    adapter=make_mlp_adapter(*values,initial=config,phase='train',warmup=3,
        samples=9,calls_per_sample=3,measurement_mode='outer_capture')
    outcome=adapter.evaluate(config)
    if outcome.status!='ok':
        emit({'error':asdict(outcome)});return
    root=Path.cwd()
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((root/'rtx').rglob('*.py'))}
    emit(dict(ready=True,config=config.to_dict(),source_sha256=sources,
        torch=str(torch.__version__),device=str(torch.cuda.get_device_properties(0)),
        numerics=outcome.metadata['numerics'],shape=[m,k,h,o],seed=args.seed))
    for command in sys.stdin:
        if command.strip()=='quit':break
        if command.strip()!='measure':raise ValueError('invalid comparison command')
        warm_ms=0.
        for _ in range(128):
            warm=adapter.evaluate(config)
            if warm.status!='ok':raise RuntimeError(warm.error)
            warm_ms+=sum(warm.timings_ms)*3
            if warm_ms>=200.:break
        else:raise RuntimeError('common GPU warmup did not reach 200 ms')
        before=torch.cuda.memory_allocated()
        result=adapter.evaluate(config)
        torch.cuda.synchronize()
        result.metadata.update(common_gpu_warmup_ms=warm_ms,allocated_before=before,
            allocated_after=torch.cuda.memory_allocated())
        emit({'outcome':asdict(result)})


def receive(process,timeout=300):
    while True:
        if not select.select([process.stdout],[],[],timeout)[0]:
            raise TimeoutError('comparison worker did not respond')
        line=process.stdout.readline()
        if not line:raise RuntimeError(f'comparison worker exited {process.poll()}')
        if line.startswith(PREFIX):return json.loads(line[len(PREFIX):])


def compare(args):
    from rtx.autotune.deployment_qualification import qualify_pair
    from rtx.autotune.outcomes import TrialOutcome
    args.output.mkdir(parents=True,exist_ok=False)
    workers={};logs={};ready={};observations=[]
    try:
        for name,root,config in (('previous',args.previous,args.previous_config),
                                 ('current',args.current,args.current_config)):
            logs[name]=(args.output/f'{name}-worker.log').open('w')
            env={**os.environ,'PYTHONPATH':str(root),
                 'TORCHINDUCTOR_CACHE_DIR':str(args.output/f'{name}-inductor-cache')}
            workers[name]=subprocess.Popen([sys.executable,'-u','-m',
                'benchmarks.compare_long_narrow_revisions','--worker','--shape',args.shape,
                '--config',str(config),'--seed',str(args.seed)],cwd=root,env=env,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs[name],text=True,bufsize=1)
        for name,process in workers.items():
            ready[name]=receive(process)
            if not ready[name].get('ready'):raise RuntimeError(ready[name])
        def evaluate(name):
            process=workers[name];process.stdin.write('measure\n');process.stdin.flush()
            result=receive(process)['outcome']
            observations.append(dict(source=name,**result))
            with (args.output/'observations.jsonl').open('a') as stream:
                stream.write(json.dumps(observations[-1])+'\n')
            return TrialOutcome(**result)
        selected,decision=qualify_pair('previous','current',evaluate,rounds=args.rounds,margin=.01)
        report=dict(arguments=vars(args),ready=ready,decision=decision,selected=selected,
            target_5pct_met=decision['promoted'] and decision['aggregate_speedup']>=.05,
            protocol='same GPU; alternating source workers; >=200 measured GPU ms warmup; 9 samples x 3 complete graph workloads',
            caveat='Synthetic complete MLP forward/backward; not BF16-relative or model-step speedup')
        (args.output/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        print('COMPLETE',json.dumps(decision),flush=True)
    finally:
        for process in workers.values():
            try:
                if process.poll() is None:
                    process.stdin.write('quit\n');process.stdin.flush();process.wait(timeout=15)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                process.terminate();process.wait(timeout=15)
        for stream in logs.values():stream.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--shape',required=True,help='M,K,H,O')
    p.add_argument('--config',type=Path)
    p.add_argument('--previous',type=Path)
    p.add_argument('--current',type=Path)
    p.add_argument('--previous-config',type=Path)
    p.add_argument('--current-config',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--rounds',type=int,default=12)
    p.add_argument('--seed',type=int,default=81110)
    args=p.parse_args()
    worker(args) if args.worker else compare(args)


if __name__=='__main__':main()
