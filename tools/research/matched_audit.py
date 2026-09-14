"""Read-only integrity and budget accounting for a completed matched context."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

TREATMENTS=('ordinary','portfolio','rules','portfolio_ml')
BUDGETS=(8,16,32,64)


def valid_confirmation(decision, winner_id):
    confirmation=decision.get('confirmation') or {}
    rounds=confirmation.get('rounds',[])
    if (decision.get('config_id')!=winner_id or confirmation.get('status')!='complete' or
            winner_id in confirmation.get('invalid',[]) or len(rounds)<2 or len(rounds)%2):
        return False
    for first,second in zip(rounds[::2],rounds[1::2]):
        if first.get('order')!=list(reversed(second.get('order',[]))):return False
    for row in rounds:
        outcome=row.get('outcomes',{}).get(winner_id,{}).get('outcome',{})
        ms=outcome.get('median_ms')
        if outcome.get('status')!='ok' or type(ms) not in (int,float) or not math.isfinite(ms) or ms<=0:
            return False
    return True


def audit(root):
    root=Path(root)
    pool=json.loads((root/'pool.json').read_text())
    ranks=json.loads((root/'rankings.json').read_text())
    configs={c['config_id']:c['config'] for c in pool['configs']}
    context_id=hashlib.sha256(json.dumps(pool['context'],sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
    issues=[]
    if len(configs)!=len(pool['configs']):issues.append('duplicate pool IDs')
    if ranks['pool_id']!=pool['pool_id']:issues.append('ranking pool mismatch')
    if ranks['provenance']['pool']!=hashlib.sha256((root/'pool.json').read_bytes()).hexdigest():
        issues.append('ranking pool file digest mismatch')
    if set(ranks['rankings'])!=set(TREATMENTS):issues.append('incomplete ranking treatments')
    for name, order in ranks['rankings'].items():
        if len(order)!=len(configs) or set(order)!=set(configs):
            issues.append(f'{name}: ranking not a pool permutation')
    summaries=[]
    openings=[]
    for name in TREATMENTS:
        path=root/'results'/name/'observations.jsonl'
        if not path.exists():
            issues.append(f'{name}: missing journal')
            continue
        text=path.read_text()
        if text and not text.endswith('\n'):issues.append(f'{name}: unterminated JSONL tail')
        try:rows=[json.loads(line) for line in text.splitlines()]
        except json.JSONDecodeError:
            issues.append(f'{name}: corrupt JSONL')
            continue
        ids=[r['config_id'] for r in rows]
        if len(set(ids))!=len(ids):issues.append(f'{name}: repeated candidate')
        if len(rows)!=64:issues.append(f'{name}: expected 64 completed trials, found {len(rows)}')
        if len({r['context_id'] for r in rows})!=1:issues.append(f'{name}: mixed context IDs')
        for row in rows:
            if row['context_id']!=context_id:issues.append(f'{name}: context differs from frozen pool')
            if row['config_id'] not in configs or row['config']!=configs.get(row['config_id']):
                issues.append(f'{name}: out-of-pool or changed config')
            if (row['family'],row['kernel_revision'])!=(pool['context']['family'],pool['context']['kernel_revision']):
                issues.append(f'{name}: family/revision mismatch')
            outcome=row['outcome']
            if outcome['status']=='ok' and (not isinstance(outcome.get('median_ms'),(int,float)) or
                    not math.isfinite(outcome['median_ms']) or outcome['median_ms']<=0):
                issues.append(f'{name}: invalid successful timing')
        opening=ids[:1+len(pool['protected_prefix'])]
        openings.append(opening)
        if opening[1:]!=pool['protected_prefix']:issues.append(f'{name}: protected opening differs')
        event_path=path.with_name('events.jsonl')
        decisions={}
        if not event_path.exists():issues.append(f'{name}: missing confirmation events')
        else:
            try:
                for line in event_path.read_text().splitlines():
                    event=json.loads(line)
                    if event.get('kind')=='winner_decision':
                        decision=event['payload']
                        if decision.get('context_id')!=context_id:
                            issues.append(f'{name}: decision context mismatch')
                        decisions[decision.get('completed_trials')]=decision
            except json.JSONDecodeError:issues.append(f'{name}: corrupt event journal')
        for budget,increment in zip(BUDGETS,(8,8,16,32)):
            result_path=root/'results'/f'{name}-{budget}.json'
            if not result_path.exists():
                issues.append(f'{name}: missing budget {budget}')
                continue
            result=json.loads(result_path.read_text())
            if result['pool_id']!=pool['pool_id'] or result['budget']!=budget or result['treatment']!=name:
                issues.append(f'{name}: result identity mismatch at {budget}')
            if result['evaluated_trials']!=increment:
                issues.append(f'{name}: unexpected resumed trial increment at {budget}')
            if result['verification']['status']!='ok':issues.append(f'{name}: failed winner verification')
            prefix=rows[:budget]
            if not any(r['config']==result['winner'] and r['outcome']['status']=='ok' for r in prefix):
                issues.append(f'{name}: winner not a successful budget-prefix candidate')
            winner_id=next((cid for cid,config in configs.items() if config==result['winner']),None)
            if not valid_confirmation(decisions.get(budget,{}),winner_id):
                issues.append(f'{name}: missing or invalid paired winner decision at {budget}')
        compile_values=[r['outcome']['compile_ms'] for r in rows if r['outcome'].get('compile_ms') is not None]
        summaries.append(dict(treatment=name,trials=len(rows),outcomes=dict(Counter(r['outcome']['status'] for r in rows)),
            arms=dict(Counter(r['strategy'] for r in rows)),
            evaluation_s=sum(r['elapsed_s'] for r in rows),
            recorded_compile_ms=sum(compile_values),missing_compile_cost=len(rows)-len(compile_values)))
    if openings and any(value!=openings[0] for value in openings):issues.append('treatment openings differ')
    return dict(schema=1,pool_id=pool['pool_id'],complete=not issues,issues=sorted(set(issues)),
        treatments=summaries,ranking_setup_s=ranks.get('total_setup_s'),
        cost_scope='Search observations only; confirmation, process startup and model setup are separate')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=audit(args.root)
    with Path(args.output).open('x') as stream:json.dump(result,stream,allow_nan=False,indent=2)
    print(json.dumps(result,indent=2))
    if not result['complete']:raise SystemExit(1)


if __name__=='__main__':main()
