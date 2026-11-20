"""Resumable, balanced whole-MoE speed qualification; no hard-coded winners.

One fresh process per context prevents compilation/graph pools from obscuring
memory behavior. The existing shared autotuner owns every schedule decision.
Results are independent JSON files, published by the benchmark after each
round; interruption never corrupts a single contiguous study dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

if __package__:
    from .benchmark_moe_varlen import scaled_case
else:
    from benchmark_moe_varlen import scaled_case


def summarize(record,target):
    times={name:statistics.median(r['median_ms'] for r in rounds) for name,rounds in record['timings'].items()}
    native=record['timings']['mxfp8_step'];baseline=record['timings']['bf16_checkpoint_step']
    if len(native)!=len(baseline) or not native:raise ValueError('Matched nonempty timing rounds required')
    paired=[b['median_ms']/a['median_ms'] for a,b in zip(native,baseline)]
    ratio=times['bf16_checkpoint_step']/times['mxfp8_step']
    return dict(timings=times,speedup_checkpoint=ratio,paired_round_speedups=paired,
        meets_target=ratio>=target and min(paired)>=target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--seeds',type=Path,required=True,help='delivery-<case>.{expert,shared,router}.json; seeds only, never fixed winners')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length-scales',nargs='+',type=int,default=[1,4])
    p.add_argument('--expert-trials',type=int,default=96)
    p.add_argument('--router-trials',type=int,default=16)
    p.add_argument('--shared-trials',type=int,default=32)
    p.add_argument('--tune-seconds',type=float,default=240.)
    p.add_argument('--target-speedup',type=float,default=1.2)
    p.add_argument('--max-autotune',action='store_true')
    p.add_argument('--case-ids',nargs='+')
    p.add_argument('--expert-hidden-scale',type=float,default=1.)
    p.add_argument('--hidden-alignment',type=int,default=64)
    p.add_argument('--reference-matrix',type=Path,help='Seed unchanged router/shared and expert search from matching original contexts')
    args=p.parse_args()
    if not args.length_scales or min(args.length_scales)<1:raise ValueError('Positive length scales required')
    root=args.source.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    for path in sorted((root/'rtx').rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    source_digest=digest.hexdigest()
    cases=json.loads((root/'benchmarks/balanced_moe_cases.json').read_text())['cases']
    if args.case_ids:cases=[c for c in cases if c['id'] in args.case_ids]
    if not cases:raise ValueError('No matching cases')
    base_lengths=[512,2000,768,4096]
    summary=[]
    def publish():
        temporary=args.output/'summary.tmp'
        temporary.write_text(json.dumps(dict(target=args.target_speedup,results=summary),indent=2)+'\n')
        temporary.replace(args.output/'summary.json')
    # Length waves balance the four architectures before deepening lengths.
    for scale in args.length_scales:
        for case in cases:
            base_id=case['id'];effective=scaled_case(case,args.expert_hidden_scale,args.hidden_alignment)
            identity=effective['id'];seed_id='d384_e24' if base_id=='d384_e24_k6_h128' else base_id
            context=args.output/f'{identity}-lengths{scale}'
            context.mkdir(exist_ok=True)
            output=context/'qualification.json'
            record=None
            if output.exists():
                candidate=json.loads(output.read_text())
                if (candidate.get('status')=='complete' and candidate.get('source_digest')==source_digest
                        and candidate.get('case')==effective
                        and candidate.get('lengths')==[scale*v for v in base_lengths]
                        and candidate.get('inductor_max_autotune',False)==args.max_autotune
                        and all(candidate.get('tuning_budgets',{}).get(key,-1)>=value for key,value in
                            dict(expert=args.expert_trials,router=args.router_trials,shared=args.shared_trials,
                                 seconds=args.tune_seconds).items())):record=candidate
            if record is None:
                command=[sys.executable,str(root/'benchmarks/benchmark_moe_varlen.py'),
                    '--source',str(root),'--output',str(output),'--case-id',base_id,
                    '--expert-hidden-scale',str(args.expert_hidden_scale),'--hidden-alignment',str(args.hidden_alignment),
                    '--lengths',*[str(scale*v) for v in base_lengths],
                    '--phases','mxfp8_step','bf16_checkpoint_step','bf16_saved_step',
                    '--tune-expert-trials',str(args.expert_trials),'--tune-router-trials',str(args.router_trials),
                    '--tune-shared-trials',str(args.shared_trials),'--tune-seconds',str(args.tune_seconds),
                    '--rounds','3','--samples','9','--oracle']
                for kind in ('expert','router','shared'):
                    path=args.seeds.resolve()/f'delivery-{seed_id}.{kind}.json'
                    if args.reference_matrix:
                        reference=args.reference_matrix.resolve()/f'{base_id}-lengths{scale}'/f'qualification.{kind}.json'
                        if reference.exists():path=reference
                    command.extend(('--'+kind+'-config',str(path)))
                if args.max_autotune:command.append('--max-autotune')
                (context/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                print('START',identity,'lengths',scale,flush=True)
                with (context/'qualification.log').open('w') as log:
                    result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    row=dict(case=identity,scale=scale,status='failed',returncode=result.returncode,log=str(context/'qualification.log'))
                    summary.append(row);publish();print('FAILED',row,flush=True);continue
                record=json.loads(output.read_text())
            row=dict(case=identity,base_case=base_id,expert_hidden_dim=effective['expert_hidden_dim'],
                lengths=record['lengths'],**summarize(record,args.target_speedup),
                source_digest=source_digest,result=str(output))
            summary.append(row)
            # Honest qualification: a miss stays a miss, not an exit-code win.
            print('RESULT',json.dumps(row),flush=True)
            publish()
    if any(r.get('status')=='failed' or not r.get('meets_target',False) for r in summary):
        print('NOT ALL CONTEXTS MET TARGET; inspect summary and per-context logs',flush=True)


if __name__=='__main__':main()
