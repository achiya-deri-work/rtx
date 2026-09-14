"""Balanced cross-treatment winner measurements; never update search evidence."""
import argparse
import json
import os
from pathlib import Path
import statistics

import numpy as np
from .matched_gpu import source_digest


def summarize_pairs(confirmation, choices, reference_id=None):
    if confirmation['status'] != 'complete' or confirmation['invalid']:
        raise ValueError('incomplete or invalid paired evidence')
    if confirmation.get('timing_unreliable') or any(
        item['outcome'].get('metadata', {}).get('sampling', {}).get('collection', {}).get('stationary') is False
        for row in confirmation['rounds'] for item in row['outcomes'].values()
    ):
        raise ValueError('nonstationary paired evidence; preserve and quarantine before policy comparison')
    rounds = confirmation['rounds']
    if len(rounds) < 4 or len(rounds) % 2:
        raise ValueError('at least two balanced pairs required')
    result = []
    for budget, treatments in sorted(choices.items(), key=lambda item:int(item[0])):
        baseline = reference_id or treatments['ordinary']
        for treatment, cid in sorted(treatments.items()):
            ratios = [row['outcomes'][cid]['outcome']['median_ms'] /
                      row['outcomes'][baseline]['outcome']['median_ms'] for row in rounds]
            if not np.isfinite(ratios).all() or min(ratios) <= 0:
                raise ValueError('invalid timing ratios')
            # Resample balanced forward/reverse blocks, not individual calls.
            pairs = np.asarray(ratios).reshape(-1,2)
            rng = np.random.default_rng(1729)
            boots = [float(np.median(pairs[rng.integers(len(pairs),size=len(pairs))]))
                     for _ in range(2000)]
            result.append(dict(budget=int(budget), treatment=treatment, config_id=cid,
                median_us=1000*statistics.median(row['outcomes'][cid]['outcome']['median_ms'] for row in rounds),
                **{('ratio_to_reference' if reference_id else 'ratio_to_ordinary'):statistics.median(ratios)},
                paired_block_bootstrap_95=np.quantile(boots,[.025,.975]).tolist(),
                **{('same_config_as_reference' if reference_id else 'same_config_as_ordinary'):cid==baseline},
                interval_scope='timing uncertainty in this context, not policy generalization'))
    return result


def main():
    from .matched_gpu import build_adapter, independent_initial, write_new, verify_sources
    from .matched_pool import MatchedPoolAdapter
    from rtx.autotune.confirmation import confirm_finalists
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','results','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--rounds',type=int,default=10)
    parser.add_argument('--reference',help='Completed full-pool reference directory; confirm its top five')
    args=parser.parse_args()
    manifest=json.loads(Path(args.pool).read_text())
    verify_sources(manifest,args.pool)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    adapter, hardware=build_adapter(manifest['task'],manifest['timing'],args.device)
    pool=MatchedPoolAdapter(adapter,[adapter.deserialize(c['config']) for c in manifest['configs']])
    if pool.pool_id != manifest['pool_id']:
        raise ValueError('device or semantic context changed')
    metrics=independent_initial(adapter,manifest['task'],args.device)
    choices, finalists={},{}
    for budget in (8,16,32,64):
        choices[budget]={}
        for treatment in ('ordinary','portfolio','rules','portfolio_ml'):
            path=Path(args.results)/f'{treatment}-{budget}.json'
            result=json.loads(path.read_text())
            if result['pool_id'] != pool.pool_id or result['budget'] != budget or result['treatment'] != treatment:
                raise ValueError('mismatched result')
            config=pool.deserialize(result['winner'])
            if pool.rejection(config) is not None:
                raise ValueError('winner is outside frozen legal pool')
            cid=pool.config_id(config)
            choices[budget][treatment]=cid
            finalists[cid]=config
    reference_ids=[]
    if args.reference:
        reference_root=Path(args.reference)
        reference_result=json.loads((reference_root/'result.json').read_text())
        if reference_result['pool_id'] != pool.pool_id or not reference_result['complete']:
            raise ValueError('reference pool mismatch or incomplete')
        rows=[json.loads(line) for line in (reference_root/'journal'/'observations.jsonl').read_text().splitlines()]
        if {r['config_id'] for r in rows} != set(pool._configs) or len(rows) != len(pool.pool):
            raise ValueError('reference does not cover the unique common pool')
        valid=sorted((r for r in rows if r['outcome']['status']=='ok'),
                     key=lambda r:(r['outcome']['median_ms'],r['config_id']))[:5]
        for row in valid:
            config=pool.deserialize(row['config'])
            cid=pool.config_id(config)
            if cid != row['config_id'] or pool.rejection(config) is not None:
                raise ValueError('invalid reference candidate')
            reference_ids.append(cid)
            finalists[cid]=config
        if not reference_ids:
            raise ValueError('reference has no valid candidates')
    if len(finalists) < 2:
        raise ValueError('all policies selected the same config; record an exact selection tie instead')
    write_new(output/'plan.json',dict(choices=choices,rounds=args.rounds,
        pool_id=pool.pool_id,hardware=hardware,initial_numerics=metrics,
        source_sha256=source_digest(),reference_ids=reference_ids,runtime_promotion=False))
    with (output/'measurements.jsonl').open('x') as stream:
        def record(item):
            stream.write(json.dumps(item,allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        confirmation=confirm_finalists(pool.evaluate,finalists,rounds=args.rounds,seed=1729,record=record)
    write_new(output/'confirmation.json',confirmation)
    reference_id=None
    if reference_ids and not confirmation['invalid']:
        reference_id=min(reference_ids,key=lambda cid:statistics.median(
            row['outcomes'][cid]['outcome']['median_ms'] for row in confirmation['rounds']))
    summary=summarize_pairs(confirmation,choices,reference_id=reference_id)
    write_new(output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
