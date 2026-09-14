"""Freeze a new-shape policy study, then run isolated local CUDA cells.

Historical model selection is development only. Fresh cells must never be fed
back into this artifact. No runtime cache winners are installed here.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.predictor_study import atomic, build_adapter, observed_geometry, source_digest, digest


def prepare(args):
    from .export_portfolio import export
    study=Path(args.study)
    rows=json.loads((study/'summary.json').read_text())
    plan=json.loads((study/'plan.json').read_text())
    expected=len(plan['protocols'])*len(plan['models'])*len(plan['views'])
    if len(list(study.glob('metrics-*.json')))!=expected:
        raise ValueError('model study is incomplete; do not select from partial results')
    candidates={}
    for row in rows:
        if row['budget']==8 and row['policy']=='portfolio_then_ranker':
            candidates.setdefault(row['model'],[]).append(row['mean_regret'])
    chosen=min(candidates,key=lambda name:(max(candidates[name]),sum(candidates[name])/len(candidates[name]),name))
    name,view=chosen.split(':')
    key=hashlib.sha256('chronological_development'.encode()).hexdigest()[:12]
    model=study/f'model-{key}-{name}-{view}.joblib'
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    artifact=root/'portfolio_predictor.json.gz'
    export(args.dataset,model,artifact)
    sources=json.loads(Path(args.sources).read_text())
    seen,count=observed_geometry([p for paths in sources.values() for p in paths])
    for previous in args.exclude_plan:
        for shape in json.loads(Path(previous).read_text())['shapes']:
            seen.add((shape['m'], shape['k']))
    shapes=[]
    # Whole-(M,K) novelty, not just a different output width.
    for category,m,k,n in (
        ('small_m',384,1152,768),('contraction',1792,1408,512),
        ('balanced',3584,1152,1408),('expansion',1280,1792,4608)):
        while (m,k) in seen:
            m+=128
        shapes.append(dict(category=category,m=m,k=k,n=n))
        seen.add((m,k))
    tasks=[]
    for budget in (8,16,32,64):
        for replicate in range(2):
            for si,shape in enumerate(shapes):
                for family in ('mxfp8_prequant_fwd','nvfp4_delayed_fwd','nvfp4_jit_row_region_fwd','nvfp4_dynamic_fwd'):
                    treatments=['existing','mixed_guarded','portfolio_only','learned_portfolio']
                    shift=(si+replicate)%len(treatments)
                    for treatment in treatments[shift:]+treatments[:shift]:
                        task=dict(family=family,shape={k:shape[k] for k in ('m','n','k')},category=shape['category'],
                            regime='hot',budget=budget,replicate=replicate,treatment=treatment,seed=20260921+replicate)
                        task['id']=digest(task)[:20]
                        tasks.append(task)
    atomic(root/'plan.json',dict(schema=1,source_sha256=source_digest(),artifact=artifact.name,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),selected_model=chosen,
        selection_rule='minimum worst-protocol mean observed-pool regret at budget 8, portfolio_then_ranker',
        historical_selection_scores=candidates,geometry_audit=dict(records=count,prior_m_k_pairs=len(seen)-len(shapes)),
        shapes=shapes,tasks=tasks,training_admission=False,runtime_promotion=False,
        guard='initial_neighborhood_v1', excluded_plans=args.exclude_plan,
        timing=dict(warmup=3,samples=7,calls_per_sample=5),task_seconds=90,
        withheld_families={'mxfp8_bwd':'revision changed; revalidation required',
                           'mxfp8_recompute_mlp':'revision changed and intermittent numerical failure'}))
    print('FROZEN',chosen,len(tasks),'cells',flush=True)


def worker(root,task_id,device):
    from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
    from rtx.autotune.store import ResidualTuningStore
    from rtx.autotune.predictors import load_predictor
    import torch
    root=Path(root)
    plan=json.loads((root/'plan.json').read_text())
    if source_digest()!=plan['source_sha256']:
        raise ValueError('frozen runtime source changed')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['driver_sha256']:
        raise ValueError('frozen study driver changed')
    artifact=root/plan['artifact']
    if hashlib.sha256(artifact.read_bytes()).hexdigest()!=plan['artifact_sha256']:
        raise ValueError('frozen artifact changed')
    task=next(t for t in plan['tasks'] if t['id']==task_id)
    output=root/'tasks'/task_id
    started=time.monotonic()
    result=dict(task=task,status='error',device=torch.cuda.get_device_name(device))
    try:
        adapter,hardware=build_adapter(task,plan['timing'],device)
        if task['family']=='mxfp8_prequant_fwd':
            from rtx.fp8 import _build_prequant_runner,_intern_prequant_config
            torch.manual_seed(task['seed'])
            m,n,k=(task['shape'][key] for key in ('m','n','k'))
            x=torch.randn(m,k,device=device,dtype=torch.bfloat16)*.1
            w=torch.randn(n,k,device=device,dtype=torch.bfloat16)*.02
            actual=torch.empty(m,n,device=device,dtype=torch.bfloat16)
            runner=_build_prequant_runner(x,w,_intern_prequant_config(adapter.initial_config))
            runner(x,w,actual)
            reference=(x.float()@w.float().T).bfloat16()
            difference=actual.float()-reference.float()
            error=float(difference.norm()/reference.float().norm().clamp_min(1e-12))
            result['independent_initial_numerics']=dict(relative_l2=error,
                max_abs=float(difference.abs().max()),finite=bool(actual.isfinite().all()))
            if not result['independent_initial_numerics']['finite'] or error>.07:
                raise RuntimeError('independent FP32 GEMM reference gate failed')
            del x,w,actual,reference,difference,runner
        result['setup_s']=time.monotonic()-started
        learned=task['treatment']=='learned_portfolio'
        with_prior=task['treatment'] in ('learned_portfolio','portfolio_only')
        if with_prior and load_predictor(artifact,research=True).bundle(adapter) is None:
            raise ValueError('historical predictor contract does not match this implementation')
        policy=AutotunePolicy(max_trials=task['budget'],max_seconds=plan['task_seconds'],
            candidate_mix='legacy' if task['treatment']=='existing' else 'mixed',
            baseline_trials=0,
            initial_neighborhood_trials=0 if task['treatment']=='existing' else 2,
            local_anchor_tolerance=0 if task['treatment']=='existing' else .01,
            exploration_trials=0 if task['treatment']=='existing' else 1,
            predictor_artifact=str(artifact) if with_prior else None,
            predictor_heads=('trees',) if learned else (),allow_research_prior=with_prior,seed=task['seed'],**plan['timing'])
        store=ResidualTuningStore(output/'journal')
        tuned=make_effort_autotuner(adapter,store,policy,progress=lambda s:print(s,flush=True)).tune()
        verification=adapter.evaluate(tuned.config)
        if not verification.successful:
            raise RuntimeError(str(verification.as_dict()))
        result.update(status='ok',hardware=hardware,policy=asdict(policy),
            winner_config=adapter.serialize(tuned.config),winner_ms=tuned.median_ms,
            verification=verification.as_dict(),evaluated_trials=tuned.evaluated_trials,
            tuning_wall_s=tuned.elapsed_s)
    except Exception:
        import traceback
        result['error']=traceback.format_exc()
        raise
    finally:
        result['total_wall_s']=time.monotonic()-started
        atomic(output/'result.json',result)


def run(args):
    root=Path(args.output)
    plan=json.loads((root/'plan.json').read_text())
    tasks=plan['tasks']
    if args.pilot:
        tasks=[t for t in tasks if t['budget']==8 and t['replicate']==0 and
               t['family']=='mxfp8_prequant_fwd' and t['category'] in ('small_m','contraction')]
    for task in tasks[:args.max_tasks]:
        output=root/'tasks'/task['id']
        if (output/'result.json').exists():
            continue
        output.mkdir(parents=True,exist_ok=True)
        print('START',task['id'],task['family'],task['category'],task['treatment'],flush=True)
        with (output/'worker.log').open('w') as log:
            completed=subprocess.run([sys.executable,'-m','tools.research.prospective_campaign','worker','--output',str(root),
                '--task',task['id'],'--device',args.device],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        print('DONE',task['id'],completed.returncode,flush=True)
        if completed.returncode:
            raise RuntimeError(f'worker failed; inspect {output}/worker.log')


def compare(root,device):
    from collections import defaultdict
    from rtx.autotune.confirmation import confirm_finalists
    root=Path(root)
    plan=json.loads((root/'plan.json').read_text())
    if source_digest()!=plan['source_sha256']:
        raise ValueError('frozen runtime source changed')
    groups=defaultdict(dict)
    for path in (root/'tasks').glob('*/result.json'):
        result=json.loads(path.read_text())
        task=result['task']
        if result['status']=='ok':
            groups[(task['family'],task['category'],task['budget'],task['replicate'])][task['treatment']]=result
    for key,results in groups.items():
        if set(results)!={'existing','mixed_guarded','portfolio_only','learned_portfolio'}:
            continue
        pairs={}
        for treatment,result in results.items():
            adapter,_=build_adapter(result['task'],plan['timing'],device)
            config=adapter.deserialize(result['winner_config'])
            outcome=adapter.evaluate(config)
            if not outcome.successful:
                raise RuntimeError(str(outcome.as_dict()))
            pairs[treatment]=(adapter,config)
        output=root/'confirmation'/digest(key)[:20]
        index=0
        def record(event):
            nonlocal index
            atomic(output/f'evaluation-{index:04d}.json',event)
            index+=1
        result=confirm_finalists(lambda pair:pair[0].evaluate(pair[1]),pairs,
                                rounds=6,seed=20260921,record=record)
        atomic(output/'result.json',dict(context=key,confirmation=result))
        print('CONFIRMED',key,result['winner'],result['normalized_scores'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--exclude-plan', action='append', default=[])
    for name in ('dataset','study','sources','output'):
        p.add_argument('--'+name,required=True)
    p=sub.add_parser('worker');p.add_argument('--output',required=True);p.add_argument('--task',required=True);p.add_argument('--device',default='cuda:0')
    p=sub.add_parser('run');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--pilot',action='store_true');p.add_argument('--max-tasks',type=int,default=8)
    p=sub.add_parser('compare');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.command=='run' and args.max_tasks<1:
        parser.error('--max-tasks must be positive')
    if args.command=='prepare':prepare(args)
    elif args.command=='worker':worker(args.output,args.task,args.device)
    elif args.command=='compare':compare(args.output,args.device)
    else:run(args)
