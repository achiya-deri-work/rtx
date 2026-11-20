"""Bounded one-axis MoE scaling study, one fresh GPU process per context.

No production policies are changed. Every context gets the same small search
budget; complete traces are collected after, never during, ordinary timing.
Independent JSON results and an append-only progress stream survive interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def contexts():
    base=dict(d_model=384,num_experts=32,top_k=4,expert_hidden_dim=128,shared_hidden_dim=256)
    rows=[]
    def add(name,n=2000,patch=None,lengths=None):
        rows.append(dict(id=name,case={**base,**(patch or {})},lengths=lengths or [n]))
    for n in (2000,128,512,7376,29504):add(f'tokens_{n}',n)
    for d in (256,512,768):add(f'width_{d}',patch=dict(d_model=d))
    for h in (64,256,512):add(f'hidden_{h}',patch=dict(expert_hidden_dim=h))
    for k in (2,8):add(f'fanout_{k}',patch=dict(top_k=k))
    for e in (16,64):add(f'experts_{e}',patch=dict(num_experts=e))
    add('packed_2000',lengths=[128,512,360,1000])
    add('long_hidden_512',29504,dict(expert_hidden_dim=512))
    add('long_width_768',29504,dict(d_model=768))
    add('thin_width_128',patch=dict(d_model=128))
    add('long_thin_width_128',29504,dict(d_model=128))
    return rows


def compatible_result(record,row,source_digest,max_autotune,budgets):
    return (record.get('status')=='complete' and record.get('source_digest')==source_digest
        and record.get('lengths')==row['lengths']
        and all(record.get('case',{}).get(k)==v for k,v in row['case'].items())
        and record.get('inductor_max_autotune',False)==max_autotune
        and all(record.get('tuning_budgets',{}).get(k,-1)>=v for k,v in budgets.items()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contexts',nargs='+')
    p.add_argument('--max-seconds',type=float,default=2700.)
    p.add_argument('--expert-trials',type=int,default=12)
    p.add_argument('--router-trials',type=int,default=8)
    p.add_argument('--shared-trials',type=int,default=8)
    p.add_argument('--tune-seconds',type=float,default=30.)
    p.add_argument('--max-autotune',action='store_true')
    p.add_argument('--seed-directory',type=Path,help='Prior per-context result.{expert,router,shared}.json files for a bounded tuning-headroom control')
    args=p.parse_args()
    root=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    for path in sorted((root/'rtx').rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    source_digest=digest.hexdigest()
    rows=contexts()
    if args.contexts:
        unknown=set(args.contexts)-{r['id'] for r in rows}
        if unknown:raise ValueError(('Unknown contexts',sorted(unknown)))
        rows=[r for name in args.contexts for r in rows if r['id']==name]
    manifest=dict(contexts=rows,source_digest=source_digest,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        command=sys.argv,contract='Independent per-sequence routing; BF16 operands; actual fullgraph backward to all leaves; router shared by precision paths; no optimizer; bounded tuning is not exhaustive')
    budgets=dict(expert=args.expert_trials,router=args.router_trials,shared=args.shared_trials,seconds=args.tune_seconds)
    for row in rows:
        target=out/row['id']/'result.json'
        if target.exists():
            previous=json.loads(target.read_text())
            if previous.get('status')=='complete' and not compatible_result(previous,row,source_digest,args.max_autotune,budgets):
                raise ValueError(('Completed evidence has a different contract; use a new output directory',str(target)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.monotonic()
    env={**os.environ,'PYTHONPATH':str(root),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
         'TORCHINDUCTOR_COMPILE_THREADS':'4'}
    for row in rows:
        if time.monotonic()-start>args.max_seconds:break
        directory=out/row['id'];directory.mkdir(exist_ok=True)
        target=directory/'result.json'
        if target.exists():
            previous=json.loads(target.read_text())
            if previous.get('status')=='complete':
                if not compatible_result(previous,row,source_digest,args.max_autotune,budgets):
                    raise ValueError(('Completed evidence has a different contract; use a new output directory',str(target)))
                print('CACHED',row['id'],flush=True);continue
        command=[sys.executable,str(root/'benchmarks/benchmark_moe_varlen.py'),'--source',str(root),
            '--output',str(target),'--case-id','d384_e32_k4_h128','--case-spec',json.dumps(row['case']),
            '--lengths',*map(str,row['lengths']),'--expert-phases',
            '--phases','mxfp8_forward','mxfp8_step','bf16_checkpoint_step','router_step',
            'expert_forward','expert_step','expert_bf16_checkpoint_step',
            '--trace-phases','mxfp8_step','expert_step',
            '--rounds','2','--samples','7',
            '--tune-router-trials',str(args.router_trials),'--tune-expert-trials',str(args.expert_trials),
            '--tune-shared-trials',str(args.shared_trials),'--tune-seconds',str(args.tune_seconds)]
        if args.max_autotune:command.append('--max-autotune')
        if args.seed_directory:
            seed=args.seed_directory.resolve()/row['id']
            for name in ('expert','router','shared'):
                path=seed/f'result.{name}.json'
                if not path.exists():raise ValueError(('Missing seed',str(path)))
                command.extend(('--'+name+'-config',str(path)))
        (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        print('START',row['id'],f'elapsed={time.monotonic()-start:.1f}s',flush=True)
        with (directory/'run.log').open('a') as log:
            try:
                result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                status=result.returncode
            except subprocess.TimeoutExpired:status='timeout'
        print('DONE',row['id'],status,f'elapsed={time.monotonic()-start:.1f}s',flush=True)
        with (out/'progress.jsonl').open('a') as f:
            f.write(json.dumps(dict(context=row['id'],status=status,elapsed_s=time.monotonic()-start))+'\n')


if __name__=='__main__':main()
