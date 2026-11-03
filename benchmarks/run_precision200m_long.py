"""Launch a matched long decoder run using independently confirmed configs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def budget_command(command, target_tokens, output, *, checkpoint_interval=2048):
    """Preserve the recipe; extend only budget, cosine horizon and checkpoint cadence."""
    result=list(command)
    batch=int(result[result.index('--batch')+1])
    sequence=int(result[result.index('--sequence')+1])
    if target_tokens<batch*sequence:
        raise ValueError('target must contain at least one complete update')
    steps=target_tokens//(batch*sequence)
    if steps<=int(result[result.index('--warmup-steps')+1]):
        raise ValueError('budget must exceed existing warmup')
    result[result.index('--steps')+1]=str(steps)
    result[result.index('--output')+1]=str(output)
    result[result.index('--checkpoint-interval')+1]=str(checkpoint_interval)
    return result,steps*batch*sequence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--refinement',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),required=True)
    p.add_argument('--mode',choices=('nvfp4_full','nvfp4_mxfp8'),required=True)
    p.add_argument('--target-tokens',type=int,default=600000000)
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if (args.gpu,args.mode) not in ((0,'nvfp4_full'),(1,'nvfp4_mxfp8')):
        p.error('this matched launch assigns full NVFP4 to GPU 0 and hybrid to GPU 1')
    args.output.mkdir(parents=True,exist_ok=False)
    from rtx.autotune.outcomes import TrialOutcome
    confirmation=json.loads((args.refinement/'confirmation.json').read_text())
    if not TrialOutcome.from_dict(confirmation).timing_eligible:
        raise RuntimeError('refinement lacks independent eligible confirmation')
    candidates=[]
    for path in args.study.glob('commands-gpu*.jsonl'):
        for line in path.read_text().splitlines():
            command=json.loads(line)['command']
            if ('benchmarks.train_precision20m' in command and
                command[command.index('--mode')+1]==args.mode and
                '/batch-72/train-' in command[command.index('--output')+1]):candidates.append(command)
    if len(candidates)!=1:raise RuntimeError('expected one exact previous training command')
    command,actual_tokens=budget_command(candidates[0],args.target_tokens,args.output/'training.jsonl')
    command[0]=sys.executable
    if args.mode=='nvfp4_full':
        command[command.index('--ms-eden-designs')+1]=str(args.refinement/'designs.json')
    # Freeze the exact configs independently of the earlier output directories.
    configs={}
    for option in ('--forward-configs','--mxfp8-backward-configs','--ms-eden-designs'):
        if option in command:
            source=Path(command[command.index(option)+1])
            destination=args.output/(option.removeprefix('--')+'.json')
            content=source.read_bytes();destination.write_bytes(content)
            command[command.index(option)+1]=str(destination)
            configs[option]=dict(source=str(source),sha256=hashlib.sha256(content).hexdigest())
    if args.resume:command+=['--resume',str(args.resume)]
    uuid=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}',
        '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if not uuid.startswith('GPU-') or any(c not in 'GPU-0123456789abcdefABCDEF' for c in uuid):
        raise RuntimeError('invalid GPU UUID')
    lease=open(f'/tmp/rtx-study-lease-{uuid}.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX)
    deadline=time.monotonic()+900
    while True:
        users=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}',
            '--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
        busy=[v.strip() for v in users if v.strip().isdigit() and int(v)!=os.getpid()]
        if not busy:break
        if time.monotonic()>deadline:raise RuntimeError(f'GPU occupied: {busy}; not launching')
        print('WAIT_GPU',args.gpu,busy,flush=True);time.sleep(5)
    manifest=dict(command=command,gpu=args.gpu,uuid=uuid,mode=args.mode,
        requested_tokens=args.target_tokens,actual_tokens=actual_tokens,
        configs=configs,source_snapshot=str(Path.cwd()),
        tuning='reuse confirmed configs for identical shapes; no search in training',
        fresh_initialization=args.resume is None)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('TRAIN',args.mode,'gpu',args.gpu,'tokens',actual_tokens,flush=True)
    started=time.monotonic()
    with (args.output/'training.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu)},timeout=10800)
    manifest.update(returncode=result.returncode,process_wall_s=time.monotonic()-started)
    (args.output/'process-result.json').write_text(json.dumps(manifest,indent=2))
    if result.returncode:raise RuntimeError('training failed; inspect training.log')
    rows=[json.loads(s) for s in (args.output/'training.jsonl').read_text().splitlines()]
    terminal=rows[-1]
    if terminal.get('event')!='result' or terminal['interrupted'] or not terminal['finite_parameters'] or terminal['tokens']!=actual_tokens:
        raise RuntimeError('training did not complete the finite requested budget')
    print('FINISHED',args.mode,flush=True)


if __name__=='__main__':main()
