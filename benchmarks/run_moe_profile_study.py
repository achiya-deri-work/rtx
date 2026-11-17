"""Resumable fresh-process profiling matrix; one GPU worker at a time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cases',nargs='+',default=['d384_e24_k6_h128','d384_e32_k4_h128','d512_e64_k8_h128','d768_e48_k6_h256'])
    p.add_argument('--tokens',nargs='+',type=int,default=[128,512,2000,8192,32768])
    p.add_argument('--whole-layer',action='store_true')
    p.add_argument('--seed-studies',nargs='+',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--samples',type=int,default=9)
    p.add_argument('--variants',choices=('none','pipelines','reuse','all'),default='none')
    p.add_argument('--tune-trials',type=int,default=0)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    jobs=[(c,n) for n in args.tokens for c in args.cases]
    journal=args.output/'events.jsonl'
    for index,(case,n) in enumerate(jobs):
        destination=args.output/f'{case}-n{n}.json';log=destination.with_suffix('.log')
        if destination.exists():
            try:
                if json.loads(destination.read_text()).get('status')=='complete':
                    print('RESUME',case,n,flush=True);continue
            except ValueError:pass
        command=[sys.executable,'-m','benchmarks.profile_moe','--case-id',case,'--tokens',str(n),
                 '--output',str(destination),'--rounds',str(args.rounds),'--samples',str(args.samples),
                 '--variants',args.variants,'--tune-trials',str(args.tune_trials),'--evict',
                 '--seed-studies',*[str(x) for x in args.seed_studies]]
        if args.whole_layer:command.append('--whole-layer')
        print(f'START {index+1}/{len(jobs)} {case} tokens={n}',flush=True)
        start=time.time()
        with log.open('w') as handle:result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
        record=dict(case=case,tokens=n,exit_code=result.returncode,seconds=time.time()-start,command=command,log=str(log))
        with journal.open('a') as handle:
            handle.write(json.dumps(record)+'\n');handle.flush();os.fsync(handle.fileno())
        print('END',case,n,result.returncode,round(record['seconds'],2),flush=True)
        if result.returncode:
            text=log.read_text(errors='replace')
            if 'OutOfMemoryError' in text or 'out of memory' in text.lower():
                print('CAPACITY_LIMIT retained; next context',flush=True);continue
            raise RuntimeError(f'Worker failed; stop and inspect {log}')
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()
