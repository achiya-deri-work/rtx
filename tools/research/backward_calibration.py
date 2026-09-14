"""Prespecified revision-23 calibration, with separate future evaluation shapes."""
import argparse
import json
from pathlib import Path
import random

from .matched_gpu import build_adapter,independent_initial,verify_sources,write_new,source_digest
from .matched_pool import MatchedPoolAdapter,make_reference_tuner

TRAIN=((640,512,384),(1920,512,896),(5120,768,640),(10240,384,1280))
HELDOUT=((3072,896,640),(6144,512,896))


def candidates(adapter,seed):
    from rtx.autotune.coverage import coverage_plan
    proposals,_=coverage_plan(adapter)
    chosen=[adapter.initial_config,*[p.config for p in proposals[:31]]]
    seen={adapter.config_id(c) for c in chosen}
    for config in adapter.sample(random.Random(seed),512,()):
        cid=adapter.config_id(config)
        if cid not in seen and adapter.rejection(config) is None:
            chosen.append(config);seen.add(cid)
        if len(chosen)==128:break
    if len(chosen)!=128:raise ValueError('insufficient distinct calibration candidates')
    return chosen


def prepare(root):
    from .source_snapshot import write_snapshot
    root.mkdir(parents=True,exist_ok=False)
    assert not ({(m,k) for m,n,k in TRAIN}&{(m,k) for m,n,k in HELDOUT})
    metadata=write_snapshot(Path(__file__).resolve().parents[2],root/'sources.zip')
    plan=dict(schema=1,train=TRAIN,heldout=HELDOUT,budgets=[32,64,128],seed=81818,
        revision=23,runtime_promotion=False,training_admission='train shapes only, after stability and numerical audit',
        allocation='initial + first 31 interleaved structural neighbors + 96 unique legal random candidates',
        model_plan='fixed ExtraTrees interaction model and depth-4 rules; no selection on heldout outcomes',
        source_snapshot=metadata,source_sha256=source_digest())
    write_new(root/'plan.json',plan)
    timing=dict(warmup=3,samples=7,calls_per_sample=5)
    for index,(m,n,k) in enumerate(TRAIN):
        task=dict(family='mxfp8_bwd',shape=dict(m=m,n=n,k=k),regime='hot',seed=81818+index)
        adapter,hardware=build_adapter(task,timing,'cuda:0')
        assert adapter.context.kernel_revision==23
        metrics=independent_initial(adapter,task,'cuda:0')
        pool=MatchedPoolAdapter(adapter,candidates(adapter,task['seed']))
        doc=pool.manifest()
        doc.update(task=task,timing=timing,hardware=hardware,initial_numerics=metrics,
                   source_snapshot=metadata,source_sha256=source_digest(),split='train')
        write_new(root/f'pool-{index}.json',doc)
        print('FROZEN',index,len(pool.pool),task['shape'],flush=True)


def run(root,index,budget):
    from rtx.autotune.store import ResidualTuningStore
    from rtx.autotune.core import TuningBudget
    path=root/f'pool-{index}.json'
    manifest=json.loads(path.read_text())
    verify_sources(manifest,path)
    adapter,_=build_adapter(manifest['task'],manifest['timing'],'cuda:0')
    assert adapter.context.kernel_revision==23
    independent_initial(adapter,manifest['task'],'cuda:0')
    pool=MatchedPoolAdapter(adapter,[adapter.deserialize(c['config']) for c in manifest['configs']])
    assert pool.pool_id==manifest['pool_id']
    output=root/f'train-{index}'
    if (output/f'result-{budget}.json').exists():raise ValueError('completed budget already exists')
    store=ResidualTuningStore(output/'journal')
    tuner=make_reference_tuner(pool,store,max_seconds=300.,seed=manifest['task']['seed'])
    tuner.budget=TuningBudget(budget,300.)
    tuner.progress=lambda message:print(message,flush=True)
    result=tuner.tune()
    rows=list(store.records(pool.context))
    assert len(rows)==budget and len({r['config_id'] for r in rows})==budget
    write_new(output/f'result-{budget}.json',dict(pool_id=pool.pool_id,completed_trials=len(rows),
        new_trials=result.evaluated_trials,split='train',runtime_promotion=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=int,choices=range(4))
    parser.add_argument('--budget',type=int,choices=(32,64,128))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.output)
    else:run(args.output,args.index,args.budget)


if __name__=='__main__':main()
