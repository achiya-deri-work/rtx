"""Paired winner replays and pool-membership diagnosis; never retunes or promotes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import zipfile

from tools.predictor_study import atomic, build_adapter, digest


def documents(source):
    source=Path(source)
    if source.suffix=='.zip':
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                if name.endswith(('/result.json','/sessions.jsonl','/observations.jsonl')) and '/tasks/' in name:
                    yield name, archive.read(name).decode()
    else:
        for path in source.glob('tasks/**/*'):
            if path.name in ('result.json','sessions.jsonl','observations.jsonl'):
                yield str(path),path.read_text()


def prepare(args):
    if args.rounds < 1 or args.cases_per_source < 1:
        raise ValueError('rounds and cases-per-source must be positive')
    out=Path(args.output)
    if (out/'plan.json').exists():raise ValueError('plan exists')
    cases=[]
    for source in args.sources:
        grouped=defaultdict(dict); journals=defaultdict(list); sessions=defaultdict(list)
        for name,text in documents(source):
            tid=name.split('/tasks/')[1].split('/')[0]
            if name.endswith('/result.json'):
                r=json.loads(text);t=r['task']
                if r['status']=='ok' and r.get('verified_ms') and r.get('attempts')==t['budget']:
                    grouped[(t['family'],t['category'],t['budget'])][t['treatment']]=r
            elif name.endswith('/observations.jsonl'):journals[tid].extend(json.loads(x) for x in text.splitlines())
            else:sessions[tid].extend(json.loads(x) for x in text.splitlines())
        eligible=[]
        for key,g in grouped.items():
            if 'existing' not in g or 'mixed_residual_mlp' not in g:continue
            a,b=g['existing'],g['mixed_residual_mlp']
            if not b.get('contract_compatible'):continue
            ratio=a['verified_ms']/b['verified_ms']
            eligible.append((ratio,key,a,b))
        # One case per family/category so multiple budgets do not consume the
        # entire targeted audit. Worst regressions are explicit development data.
        used=set()
        for ratio,key,a,b in sorted(eligible,key=lambda x:x[0]):
            if key[:2] in used:continue
            used.add(key[:2])
            baseline_records=journals[a['task']['id']]
            winner_id=next((r['config_id'] for r in baseline_records if r['config']==a['winner_config']),None)
            learned_records=journals[b['task']['id']]
            origins={}
            for s in sessions[b['task']['id']]:
                metadata=s.get('metadata',{})
                origins.update((metadata.get('candidate_pool') or {}).get('origins',{}))
            case=dict(source=str(source),task=a['task'],baseline=a['winner_config'],candidate=b['winner_config'],
                source_gpu=a['hardware']['name'],original_ratio=ratio,
                original_ms={'baseline':a['verified_ms'],'candidate':b['verified_ms']},
                pool_audit=dict(baseline_config_id=winner_id,
                    pool_recorded=bool(origins),baseline_in_pool=winner_id in origins if origins else None,
                    pool_origin=origins.get(winner_id),
                    baseline_measured=any(r['config_id']==winner_id for r in learned_records),
                    measured_strategies=[r['strategy'] for r in learned_records]))
            case['id']=digest(case)[:20];cases.append(case)
            if len(used)>=args.cases_per_source:break
    atomic(out/'plan.json',dict(cases=cases,rounds=args.rounds,
        timing=dict(warmup=3,samples=7,calls_per_sample=5),purpose='development_regression_diagnosis'))
    print(json.dumps([dict(id=c['id'],family=c['task']['family'],category=c['task']['category'],
                          ratio=c['original_ratio'],audit=c['pool_audit']) for c in cases],indent=2))


def replay(args):
    import torch
    root=Path(args.output);plan=json.loads((root/'plan.json').read_text())
    case=next(c for c in plan['cases'] if c['id']==args.case)
    result=dict(case=case,device=torch.cuda.get_device_name(args.device),rounds=[],status='error')
    try:
        adapters={}
        for label in ('baseline','candidate'):
            adapter,profile=build_adapter(case['task'],plan['timing'],args.device)
            config=adapter.deserialize(case[label])
            rejection=adapter.rejection(config)
            if rejection is not None:raise ValueError(f'{label}: {rejection}')
            outcome=adapter.evaluate(config)
            if not outcome.successful:raise RuntimeError(f'{label}: {outcome.as_dict()}')
            adapters[label]=(adapter,config)
        if args.balanced:
            from rtx.autotune.confirmation import confirm_finalists
            event_index = 0
            def record(event):
                nonlocal event_index
                atomic(root/'paired_events'/args.case/f'{event_index:04d}.json', event)
                event_index += 1
            result['paired_confirmation'] = confirm_finalists(
                lambda pair: pair[0].evaluate(pair[1]), adapters,
                rounds=plan['rounds'], seed=20260919, record=record)
            result['status'] = result['paired_confirmation']['status']
            print(json.dumps({k:v for k,v in result['paired_confirmation'].items() if k!='rounds'},indent=2))
            return
        rng=random.Random(20260919)
        for i in range(plan['rounds']):
            order=['baseline','candidate'];rng.shuffle(order);row={'order':order}
            for label in order:
                adapter,config=adapters[label];outcome=adapter.evaluate(config)
                if not outcome.successful:raise RuntimeError(f'{label}: {outcome.as_dict()}')
                row[label]=outcome.as_dict()
            row['ratio']=row['baseline']['median_ms']/row['candidate']['median_ms']
            result['rounds'].append(row)
        result.update(status='ok',median_paired_ratio=statistics.median(r['ratio'] for r in result['rounds']),
            minimum_paired_ratio=min(r['ratio'] for r in result['rounds']),
            maximum_paired_ratio=max(r['ratio'] for r in result['rounds']))
    except Exception:
        import traceback
        result['error']=traceback.format_exc()
        raise
    finally:
        atomic(root/(args.case+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('case','rounds')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--sources',nargs='+',required=True)
    a.add_argument('--output',required=True);a.add_argument('--rounds',type=int,default=12)
    a.add_argument('--cases-per-source',type=int,default=2)
    a=sub.add_parser('replay');a.add_argument('--output',required=True);a.add_argument('--case',required=True)
    a.add_argument('--device',default='cuda:0')
    a.add_argument('--balanced', action='store_true', help='Use the shared balanced finalist-confirmation primitive')
    args=p.parse_args();prepare(args) if args.command=='prepare' else replay(args)
