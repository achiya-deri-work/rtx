"""Matched post-fix training; opposite policy order on the two home GPUs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--refinement',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+1800
    while not (args.refinement/'selection.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('no independently confirmed model selection; not training')
        time.sleep(15)
    confirmation=json.loads((args.refinement/'confirmation.json').read_text())
    from rtx.autotune.outcomes import TrialOutcome
    if not TrialOutcome.from_dict(confirmation).timing_eligible:
        raise RuntimeError('model selection is not eligible')
    # The lease serializes our launchers; the process check also protects jobs
    # started outside that lease. Never kill another GPU user to make room.
    uuid=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}',
        '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if not uuid.startswith('GPU-') or any(c not in 'GPU-0123456789abcdefABCDEF' for c in uuid):
        raise RuntimeError('invalid GPU identity for lease')
    lease=open(f'/tmp/rtx-study-lease-{uuid}.lock','a')
    print('WAIT_LEASE',uuid,flush=True)
    fcntl.flock(lease,fcntl.LOCK_EX)
    deadline=time.monotonic()+900
    while True:
        users=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}',
            '--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
        busy=[value.strip() for value in users if value.strip().isdigit() and int(value)!=os.getpid()]
        if not busy:break
        if time.monotonic()>deadline:raise RuntimeError(f'GPU still busy: {busy}; not launching')
        print('WAIT_GPU',args.gpu,busy,flush=True)
        time.sleep(5)
    commands={}
    for path in args.study.glob('commands-gpu*.jsonl'):
        for line in path.read_text().splitlines():
            c=json.loads(line)['command']
            if 'benchmarks.train_precision20m' in c and '/batch-72/train-' in c[c.index('--output')+1]:
                commands[c[c.index('--mode')+1]]=c
    assert set(commands)=={'nvfp4_full','nvfp4_mxfp8'}
    modes=['nvfp4_mxfp8','nvfp4_full'] if args.gpu==0 else ['nvfp4_full','nvfp4_mxfp8']
    for mode in modes:
        command=list(commands[mode]);command[0]=sys.executable
        command[command.index('--output')+1]=str(args.output/f'{mode}.jsonl')
        if mode=='nvfp4_full':command[command.index('--ms-eden-designs')+1]=str(args.refinement/'designs.json')
        started=time.monotonic()
        print('TRAIN',args.gpu,mode,flush=True)
        with (args.output/f'{mode}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu)})
        with (args.output/'commands.jsonl').open('a') as log:
            log.write(json.dumps(dict(command=command,gpu=args.gpu,returncode=result.returncode,
                                     wall_s=time.monotonic()-started))+'\n')
        if result.returncode:raise RuntimeError(f'failed {mode}')
    subprocess.run([sys.executable,'-m','benchmarks.analyze_precision20m','--baseline-mode','nvfp4_mxfp8',
        str(args.output/'nvfp4_mxfp8.jsonl'),str(args.output/'nvfp4_full.jsonl'),
        '--output',str(args.output/'comparison.json')],check=True)
    print('FINISHED',args.gpu,flush=True)


if __name__=='__main__':main()
