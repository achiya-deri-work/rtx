"""Balanced, resumable MoE evidence collection through the shared autotuner.

Wave order covers every architecture before deeper effort. Residuals live
outside the per-budget result directory so progressive 8/16/32 searches reuse
the exact initial context rather than restarting from their previous winner.
Run from an immutable source snapshot; live source drift is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from rtx.autotune.collection_health import (
    COLLECTION_GUARD_REVISION, HEALTH_FILENAME, assert_healthy_path, first_device_fault,
)
from rtx.autotune.supervisor import DEVICE_FAULT_EXIT_CODE, supervise_command


class StudyVisitError(RuntimeError):
    """A failed visit with a durable status record, not a successful result."""

    def __init__(self,message,record):
        super().__init__(message)
        self.record=record


def workloads(cases):
    profiles = (
        ('tiny', [128]), ('medium', [2000]),
        ('mixed', [512,2000,768,4096]), ('short_packed', [128]*16),
        ('packed_control', [2048]), ('boundary', [127,0,129,257]),
        ('boundary_control', [513]), ('large', [29504]),
    )
    return [dict(id=f'{case["id"]}-{name}', case=dict(case), lengths=lengths,
                 profile=name) for name,lengths in profiles for case in cases]


def source_digest(root):
    digest=hashlib.sha256()
    for path in sorted((root/'rtx').rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def build_command(root,out,row,budget,args):
    directory=out/row['id']
    command=[sys.executable,str(root/'benchmarks/benchmark_moe_varlen.py'),
        '--source',str(root),'--output',str(directory/f'effort-{budget}'/'result.json'),
        '--case-id',row['case']['id'],'--lengths',*map(str,row['lengths']),
        '--tune-expert-trials',str(budget),'--tune-router-trials',str(budget),
        '--tune-shared-trials',str(args.shared_trials),'--tune-seconds',str(args.tune_seconds),
        '--tuning-dir',str(directory/'residuals'),
        '--search-schedule','progressive','--candidate-mix','mixed',
        '--rounds','2','--samples','7','--expert-phases',
        '--phases','mxfp8_step','bf16_checkpoint_step','router_step','expert_step',
        'expert_bf16_checkpoint_step','--oracle']
    if args.max_autotune:command.append('--max-autotune')
    if (budget==8 and row['case']['id']=='d384_e32_k4_h128'
            and row['profile'] in ('tiny','medium','mixed')):
        command.extend(('--trace-phases','mxfp8_step','expert_step'))
    # Same initial schedules in every effort wave. Never seed 16 from the
    # measured 8-trial winner: that would change the context and defeat resume.
    if args.seeds:
        for kind in ('expert','router','shared'):
            candidate=args.seeds/row['case']['id']/f'result.{kind}.json'
            if not candidate.exists():raise ValueError(f'Missing immutable seed {candidate}')
            command.extend(('--'+kind+'-config',str(candidate.resolve())))
    return command


def run(args):
    root=args.source.resolve();out=args.output.resolve()
    assert_healthy_path(out)
    cases=json.loads((root/'benchmarks/balanced_moe_cases.json').read_text())['cases']
    rows=workloads(cases)
    if args.contexts:
        unknown=set(args.contexts)-{row['id'] for row in rows}
        if unknown:raise ValueError(f'Unknown contexts: {sorted(unknown)}')
        rows=[row for row in rows if row['id'] in args.contexts]
    drivers={name:hashlib.sha256((root/'benchmarks'/name).read_bytes()).hexdigest()
             for name in ('study_moe_priors.py','benchmark_moe_varlen.py')}
    identity=dict(schema=2,guard_revision=COLLECTION_GUARD_REVISION,
        drivers=drivers,source=source_digest(root),contexts=rows,
        budgets=args.budgets,shared_trials=args.shared_trials,tune_seconds=args.tune_seconds,
        max_autotune=args.max_autotune,seeds=None if args.seeds is None else str(args.seeds.resolve()),
        contract='progressive mixed shared UCB; compiler-visible allocations; actual fullgraph all-leaf backward; graph timing; no optimizer; independent pools')
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=identity:
        raise ValueError('Study identity changed: use another output directory; old evidence preserved')
    atomic_json(path,identity)
    if args.plan_only:return identity
    machine=out/'machine.json'
    if not machine.exists():
        # Same bundle ABI as the existing audit/ingestion tools. Import only
        # for a real GPU run; planning remains CPU-only. No timings here.
        sys.path.insert(0,str(root))
        from rtx.autotune.dataset import machine_snapshot
        atomic_json(machine,machine_snapshot('cuda:0'))
    started=time.monotonic();visits=0;completed=[]
    for budget in args.budgets:
        for row in rows:
            if time.monotonic()-started>=args.max_seconds or visits>=args.max_visits:
                return completed
            if source_digest(root)!=identity['source']:
                raise RuntimeError('Source changed during collection: use an immutable snapshot')
            if any(hashlib.sha256((root/'benchmarks'/name).read_bytes()).hexdigest()!=digest
                   for name,digest in drivers.items()):
                raise RuntimeError('Benchmark driver changed during collection; use an immutable snapshot')
            directory=out/row['id']/f'effort-{budget}';target=directory/'result.json'
            if target.exists():
                old=json.loads(target.read_text())
                if old.get('status')=='complete' and old.get('source_digest')==identity['source']:
                    fault=first_device_fault(directory/'run.log')
                    if fault is not None:
                        atomic_json(out/HEALTH_FILENAME,dict(evidence_eligible=False,
                            reason='fatal CUDA fault in cached visit',fault=fault))
                        raise RuntimeError('Cached visit contains a fatal CUDA fault; bundle quarantined')
                    health=directory/'health.json'
                    if not health.exists() or json.loads(health.read_text()).get('evidence_eligible') is not True:
                        raise RuntimeError('Cached visit lacks guarded completion; use a new output directory')
                    print('CACHED',row['id'],budget,flush=True);continue
            command=build_command(root,out,row,budget,args)
            atomic_json(directory/'command.json',command)
            print('START',row['id'],f'effort={budget}',flush=True)
            visit_start=time.monotonic()
            with (directory/'run.log').open('a') as log:
                # The supervisor owns a fresh process group and observes CUDA
                # faults from nested compiler workers, not just the exit code.
                status=supervise_command(command,stall_timeout_s=min(120.,args.visit_timeout),
                    wall_timeout_s=args.visit_timeout,output=log,cwd=root)
            fault=first_device_fault(directory/'run.log')
            if fault is not None:status=DEVICE_FAULT_EXIT_CODE
            visits+=1
            valid_result=False
            if target.exists():
                result=json.loads(target.read_text())
                valid_result=(result.get('status')=='complete' and
                              result.get('source_digest')==identity['source'])
            eligible=status==0 and valid_result
            atomic_json(directory/'health.json',dict(guard_revision=COLLECTION_GUARD_REVISION,
                evidence_eligible=eligible,exit_code=status,fault=fault))
            record=dict(context=row['id'],budget=budget,status=status,
                wall_seconds=time.monotonic()-visit_start,result=str(target),
                remaining_seconds=max(0,args.max_seconds-(time.monotonic()-started)))
            record['evidence_eligible']=eligible
            atomic_json(directory/'visit.json',record)
            # Each result/residual is independently durable. This progress
            # stream is convenience metadata, not the only copy of evidence.
            with (out/'progress.jsonl').open('a') as log:
                log.write(json.dumps(record)+'\n');log.flush()
            completed.append(record);print('DONE',json.dumps(record),flush=True)
            if status==DEVICE_FAULT_EXIT_CODE:
                atomic_json(out/HEALTH_FILENAME,dict(evidence_eligible=False,
                    reason='fatal CUDA device fault; collection stopped',fault=fault,visit=record))
                raise StudyVisitError('Fatal CUDA fault; collection stopped and evidence quarantined',record)
            if not eligible:
                raise StudyVisitError('Visit failed or did not publish a complete result; collection stopped',record)
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=Path)
    p.add_argument('--budgets',type=int,nargs='+',default=[8,16,32])
    p.add_argument('--contexts',nargs='+')
    p.add_argument('--shared-trials',type=int,default=0)
    p.add_argument('--tune-seconds',type=float,default=120.)
    p.add_argument('--max-seconds',type=float,default=3600.)
    p.add_argument('--max-visits',type=int,default=1000000)
    p.add_argument('--visit-timeout',type=float,default=900.)
    p.add_argument('--max-autotune',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if (args.budgets!=sorted(set(args.budgets)) or min(args.budgets)<8
            or args.max_seconds<=0 or args.max_visits<1 or args.tune_seconds<=0
            or args.visit_timeout<=0 or args.shared_trials<0):
        p.error('Increasing unique budgets >=8 and positive time/visit limits required')
    run(args)


if __name__=='__main__':main()
