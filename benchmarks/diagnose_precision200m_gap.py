"""Replay recorded tuned model pilots, with same-board timing or narrow Nsys capture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--profile-from-start',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    commands={}
    for path in args.study.glob('commands-gpu*.jsonl'):
        for line in path.read_text().splitlines():
            command=json.loads(line)['command']
            if 'benchmarks.train_precision20m' in command:
                destination=command[command.index('--output')+1]
                if '/batch-72/pilot-' in destination:
                    commands[command[command.index('--mode')+1]]=command
    assert set(commands)=={'nvfp4_full','nvfp4_mxfp8'}
    order=['nvfp4_mxfp8','nvfp4_full'] if args.profile else [
        'nvfp4_mxfp8','nvfp4_full','nvfp4_full','nvfp4_mxfp8']
    for i,mode in enumerate(order):
        label=f'{i}-{mode}'
        command=list(commands[mode])
        command[command.index('--output')+1]=str(args.output/f'{label}.jsonl')
        if args.profile:
            capture=[]
            if not args.profile_from_start:
                command+=['--profile-step','32']
                capture=['--capture-range=cudaProfilerApi','--capture-range-end=stop']
            command=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none',
                '--cpuctxsw=none',*capture,
                '--cuda-graph-trace=node','-o',str(args.output/label),*command]
        print('START',label,flush=True)
        started=time.monotonic()
        with (args.output/f'{label}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu)},timeout=900)
        with (args.output/'commands.jsonl').open('a') as log:
            log.write(json.dumps(dict(command=command,gpu=args.gpu,returncode=result.returncode,
                                     wall_s=time.monotonic()-started))+'\n')
        if result.returncode:raise RuntimeError(f'{label} failed')
        print('DONE',label,flush=True)


if __name__=='__main__':main()
