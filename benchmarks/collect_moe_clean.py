"""Guarded, audit-gated MoE collection from an immutable source snapshot.

Each visit is isolated by the existing study supervisor. After it finishes,
check device/log health, journal integrity and published numerical/replay/
memory gates before admitting another visit. Exports are immutable per-visit
CSV/Parquet snapshots; raw residuals remain authoritative. No prior or runtime
winner is installed, and no quarantined input is reused.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from benchmarks.study_moe_priors import StudyVisitError, atomic_json, run as study_run
from benchmarks.audit_collection_health import audit_logs
from benchmarks.analyze_moe_priors import analyze
from rtx.autotune.audit import audit_bundles
from rtx.autotune.collection_health import HEALTH_FILENAME, assert_healthy_path
from rtx.autotune.supervisor import TIMEOUT_EXIT_CODE


def validate_result(path):
    result=json.loads(path.read_text())
    if result.get('status')!='complete':raise ValueError(f'Incomplete result: {path}')
    validation=result.get('validation',{})
    if validation.get('all_finite') is not True:raise ValueError('Missing all-leaf finiteness gate')
    oracle=validation.get('independent_quantized_fp64_experts',{})
    if set(oracle)!= {'output','input','gate_dw','down_dw','route_dw'}:
        raise ValueError('Missing independent expert oracle components')
    for name,row in oracle.items():
        if not all(math.isfinite(row[key]) and row[key]>=0 for key in ('nrms','max_abs')):
            raise ValueError(f'Nonfinite/invalid expert metric: {name}')
        if row['nrms']>(.003 if name=='output' else .006) and row['max_abs']>2e-7:
            raise ValueError(f'Expert oracle gate failed: {name}')
    replay=validation.get('fresh_gradient_replay',{})
    required={'mxfp8_step','bf16_checkpoint_step','expert_step','expert_bf16_checkpoint_step'}
    if not required.issubset(replay):raise ValueError('Missing whole/expert replay gates')
    for phase in required:
        if len(replay[phase])<2:raise ValueError(f'Insufficient replay checks: {phase}')
        for row in replay[phase]:
            if not all(math.isfinite(row[key]) and row[key]>=0 for key in ('max_nrms','max_abs')):
                raise ValueError(f'Nonfinite/invalid replay metric: {phase}')
            if row['max_nrms']>.006 and row['max_abs']>2e-7:
                raise ValueError(f'Replay gate failed: {phase}')
    memory=result.get('memory',{})
    live=memory.get('live_bytes',[])
    if len(live)<5 or memory.get('live_growth_bytes')!=0 or len(set(live))!=1:
        raise ValueError('Missing or failed steady live-memory gate')
    if not result.get('inductor_max_autotune') or not result.get('inductor_autotune_in_subproc'):
        raise ValueError('Expected max-autotune and subprocess isolation')
    return dict(result=str(path),source_digest=result['source_digest'],
                live_bytes=live[0],oracle_max_nrms=max(r['nrms'] for r in oracle.values()),
                tuning_counts=result.get('tuning_counts',{}))


def kernel_faults(since,pci):
    # Linux HOME qualification: inability to inspect the journal is unknown,
    # not evidence of zero faults. Never reset or otherwise operate on a GPU.
    proc=subprocess.run(['journalctl','-k','--since',f'@{since:.6f}',
        '--no-pager','-o','short-iso','-g','NVRM: Xid'],capture_output=True,text=True,timeout=15)
    no_entries=(proc.returncode==1 and proc.stdout.strip()=='-- No entries --' and not proc.stderr.strip())
    if (proc.returncode and not no_entries) or 'permission' in proc.stderr.lower() or 'not seeing' in proc.stderr.lower():
        raise RuntimeError(f'Cannot qualify kernel fault history: {proc.stderr[:500]}')
    return [line for line in proc.stdout.splitlines() if f'PCI:{pci}' in line]


def collect(args):
    source=args.source.resolve();root=args.output.resolve();study=root/'study'
    assert_healthy_path(study)
    start=time.monotonic();kernel_since=time.time()
    existing=sorted((root/'checks').glob('visit-*'))
    visit_number=len(existing)
    completed=[]
    stop_reason='visit_limit'
    unfinished=None
    for _ in range(args.max_visits):
        remaining=args.max_seconds-(time.monotonic()-start)
        if remaining<=0:
            stop_reason='deadline'
            break
        common=SimpleNamespace(source=source,output=study,seeds=None,budgets=[8,16,32],
            contexts=None,shared_trials=args.shared_trials,tune_seconds=120.,
            max_autotune=True,plan_only=False,max_seconds=remaining,max_visits=1,
            visit_timeout=min(900.,remaining))
        try:
            records=study_run(common)
        except StudyVisitError as error:
            # Only a visit whose hard limit was clipped to this campaign's
            # remaining budget is an expected deadline stop. A 900s per-visit
            # timeout with campaign time left, or any CUDA fault, still fails.
            if (error.record.get('status')==TIMEOUT_EXIT_CODE and
                    common.visit_timeout==common.max_seconds):
                stop_reason='deadline'
                unfinished=error.record
                print('STOP deadline; unfinished visit excluded',flush=True)
                break
            raise
        if not records:
            stop_reason='deadline' if time.monotonic()-start>=args.max_seconds else 'all_contexts_complete'
            break
        visit_number+=1
        check=root/'checks'/f'visit-{visit_number:03d}'
        record=records[0]
        try:
            # Validate before exporting or launching the next CUDA context.
            numeric=validate_result(Path(record['result']))
            log_health=audit_logs(study)
            journal=audit_bundles([study])
            xids=kernel_faults(kernel_since,args.pci)
            atomic_json(check/'log-health.json',log_health)
            atomic_json(check/'journal-audit.json',journal)
            atomic_json(check/'qualification.json',dict(numeric=numeric,kernel_xids=xids))
            if not log_health['evidence_eligible'] or not journal['ok'] or xids:
                raise RuntimeError('Post-visit log, journal or GPU fault gate failed')
        except Exception as error:
            atomic_json(study/HEALTH_FILENAME,dict(evidence_eligible=False,
                reason=f'Post-visit qualification failed: {type(error).__name__}: {error}'))
            raise
        export=analyze([study],root/'exports'/f'visit-{visit_number:03d}')
        summary=dict(visit=visit_number,context=record['context'],budget=record['budget'],
            observations=export['observations'],contexts=len(export['contexts']),
            audit_errors=journal['summary']['errors'],kernel_xids=0,
            remaining_seconds=max(0,args.max_seconds-(time.monotonic()-start)))
        atomic_json(check/'summary.json',summary)
        atomic_json(root/'latest.json',summary)
        completed.append(summary)
        print('QUALIFIED',json.dumps(summary),flush=True)
    completion=dict(status='stopped',reason=stop_reason,qualified_this_launch=len(completed),
        unfinished_visit=unfinished,elapsed_seconds=time.monotonic()-start,
        last_qualified=None if not (root/'latest.json').exists() else
            json.loads((root/'latest.json').read_text()))
    atomic_json(root/'completion.json',completion)
    print('STOP',json.dumps(completion),flush=True)
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-seconds',type=float,default=5400.)
    parser.add_argument('--max-visits',type=int,default=96)
    parser.add_argument('--shared-trials',type=int,default=8)
    parser.add_argument('--pci',default='0000:01:00',help='Authorized physical GPU PCI address for journal checks')
    args=parser.parse_args()
    if args.max_seconds<=0 or args.max_visits<1 or args.shared_trials<1:
        parser.error('Positive time/visit/shared trial limits required')
    collect(args)


if __name__=='__main__':main()
