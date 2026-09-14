"""Read-only, session-scoped cost accounting for public first-hit probes."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def duration(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('invalid duration')
    return float(value)


def summarize(document, observations, events):
    evidence=document['winner']['metadata']['tuning_evidence']
    session=evidence['session_id'];context=evidence['context_id']
    rows=[r for r in observations if r['session_id']==session]
    if not rows or any(r['context_id']!=context for r in rows):
        raise ValueError('missing or mixed session observations')
    if len({r['observation_id'] for r in rows})!=len(rows):
        raise ValueError('duplicate observations')
    confirmations=[r['payload'] for r in events if r['session_id']==session and r['kind']=='winner_confirmation']
    if any(r['context_id']!=context for r in confirmations):
        raise ValueError('mixed confirmation context')
    # Decisions embed the same confirmation rows. Count only actual events.
    search_s=sum(duration(r['elapsed_s']) for r in rows)
    finalist_s=sum(duration(r['elapsed_s']) for r in confirmations)
    proposal_s=sum(duration(r.get('metadata',{}).get('proposal_elapsed_s',0)) for r in rows)
    errors=[r['payload'] for r in events if r['session_id']==session and r['kind']=='strategy_error']
    proposal_s+=sum(duration(r['proposal_elapsed_s']) for r in errors)
    first_s=duration(document['first_call_s'])
    unexplained=first_s-search_s-finalist_s-proposal_s
    if unexplained < -1e-6:
        raise ValueError('component times exceed first-call wall time')
    def compiles(outcomes):
        values=[o.get('compile_ms') for o in outcomes]
        return dict(recorded_s=sum(duration(v)/1000 for v in values if v is not None),
                    missing_count=sum(v is None for v in values),
                    scope='nested in evaluator wall time; not an additive component')
    statuses=sorted({r['outcome'].get('status','unknown') for r in rows})
    by_status={}
    for status in statuses:
        group=[r for r in rows if r['outcome'].get('status','unknown')==status]
        by_status[status]=dict(count=len(group),evaluator_s=sum(duration(r['elapsed_s']) for r in group),
                              compile=compiles([r['outcome'] for r in group]))
    return dict(mode=document['mode'],budget=document['budget'],session_id=session,
        context_id=context,new_search_trials=len(rows),first_call_s=first_s,
        search_evaluator_s=search_s,search_compile=compiles([r['outcome'] for r in rows]),
        search_by_status=by_status,
        finalist_evaluations=len(confirmations),finalist_evaluator_s=finalist_s,
        finalist_compile=compiles([r['outcome'] for r in confirmations]),
        proposal_observed_lower_bound_s=proposal_s,
        unclassified_upper_bound_s=max(0.,unexplained),
        model_load_s=None,model_inference_s=None,frontend_compile_s=None,
        limitations=['Search and finalist compile times are subsets of evaluator times.',
            'Proposal time includes ranking when invoked, not just model inference.',
            'Empty/deadline-ended proposal time and setup/model loading are not fully isolated.',
            'Unclassified is an upper bound, not an identified compiler or Python overhead.',
            'First-call time excludes process startup and may use compiler disk caches.'])


def read_jsonl(path):
    data=path.read_bytes()
    if data and not data.endswith(b'\n'):
        raise ValueError(f'incomplete JSONL tail: {path}')
    return [json.loads(line) for line in data.splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.root);output=Path(args.output)
    if output.exists():raise ValueError('do not overwrite cost evidence')
    costs=[];sources={}
    for mode in ('ordinary','retrieval','ml'):
        observations=[];events=[]
        for name,destination in (('observations.jsonl',observations),('events.jsonl',events)):
            for path in sorted((root/mode/'cache').rglob(name)):
                destination.extend(read_jsonl(path))
                sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for budget in (8,16,32,64):
            path=root/mode/f'{budget}.json'
            document=json.loads(path.read_text())
            if document['mode']!=mode or document['budget']!=budget:
                raise ValueError('probe path/identity mismatch')
            costs.append(summarize(document,observations,events))
            sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    from .matched_gpu import write_new
    write_new(output,dict(schema=1,costs=costs,source_sha256=sources,
                         analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for item in costs:
        print(item['mode'],item['budget'], 'first=%.3f search=%.3f finalists=%.3f proposal>=%.3f unclassified<=%.3f' %
              tuple(item[key] for key in ('first_call_s','search_evaluator_s','finalist_evaluator_s',
                                          'proposal_observed_lower_bound_s','unclassified_upper_bound_s')))


if __name__=='__main__':main()
