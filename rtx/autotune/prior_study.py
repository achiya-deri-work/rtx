"""Balanced, resumable prospective prior trials using isolated GPU workers."""
import argparse
from dataclasses import replace
from collections import defaultdict
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

from .core import stable_id
from .first_hit_study import atomic_json,source_id
from .prior import FrozenPrior


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(prior,output):
    prior=Path(prior).resolve();FrozenPrior.load(prior,research=True)
    dataset=json.loads(Path('autotune_manifests/blackwell_diversity_atlas_v1.json').read_text())
    buckets=defaultdict(list)
    # Half anchor-shaped, half new shapes. Labels support separate reporting;
    # actual training membership must still be checked against source data.
    shapes=[(128,1536,1536),(512,1536,1536),(1536,1536,1536),(777,2051,1025),
            (256,2048,1024),(1025,2057,513),(3072,1024,2048),(6144,3072,1024)]
    for job in dataset['jobs']:
        family=job['family']
        for index,(m,n,k) in enumerate(shapes):
            for regime in ('hot','rotate'):
                buckets[family].append(dict(family=family,problem=[m,n,k],shape=[m,k,n,n],
                    regime=regime,cohort='anchor_like' if index<4 else 'new_shape_candidate'))
    for index,shape in enumerate(([512,768,1536,768],[4096,768,768,768],
                                  [2048,512,1536,512],[3072,1024,2048,1024])):
        for precision in ('mxfp8','bf16_gate'):
            for phase in ('forward','train'):
                buckets['mxfp8_recompute_mlp'].append(dict(family='mxfp8_recompute_mlp',shape=shape,
                    precision=precision,phase=phase,regime='hot',
                    cohort='anchor_like' if index<2 else 'new_shape_candidate'))
    tasks=[];policies=['ordered','random','retrieval','ranker','hybrid']
    for index in range(max(map(len,buckets.values()))):
        for family in sorted(buckets):
            if index>=len(buckets[family]):continue
            base={**buckets[family][index],'seed':20260914}
            rotation=int(stable_id(base)[:8],16)%len(policies)
            for policy in policies[rotation:]+policies[:rotation]:
                task={**base,'policy':policy};tasks.append({**task,'id':stable_id(task)})
    plan=dict(schema_version=1,prior=str(prior),prior_sha256=file_hash(prior),tasks=tasks,policies=policies,
        pool_size=128,milestones=[4,8,16,32,64,128],slice_seconds=90,worker_timeout_seconds=180,
        selector=dict(mix=[2,4,2],feasibility_penalty=.25,preparation_penalty=.05),
        linear_manifest='autotune_manifests/blackwell_diversity_atlas_v1.json',
        cache_contract='shared compiler disk caches; counterbalanced policy order, not cold-start proof')
    atomic_json(output,plan);print(f'PLAN tasks={len(tasks)} output={output}',flush=True)


def make_adapter(task,root,device):
    import torch
    if task['family']=='mxfp8_recompute_mlp':
        from .mlp import make_mlp_adapter
        generator=torch.Generator(device=device).manual_seed(task['seed'])
        m,k,h,o=task['shape']
        tensors=[torch.randn(s,device=device,dtype=torch.bfloat16,generator=generator)*scale
                 for s,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02))]
        return make_mlp_adapter(*tensors,precision=task['precision'],phase=task['phase']),None
    from .dataset import DatasetCampaign,DatasetManifest
    from ..prequant_experiments import ShapeSpec
    manifest=DatasetManifest.from_dict(json.loads(Path('autotune_manifests/blackwell_diversity_atlas_v1.json').read_text()))
    manifest=replace(manifest,seed=task['seed'])
    campaign=DatasetCampaign(manifest,root/'harness',device=device,progress=None)
    job=next(job for job in manifest.jobs if job.family==task['family'])
    shape=ShapeSpec(*task['problem'])
    harness=campaign._make_harness(job,shape,task['regime'])
    return campaign._make_adapter(job,shape,task['regime'],harness),harness


def worker(manifest,output,index,milestone,device):
    from .compile_pipeline import compile_pipeline
    from .first_hit import PortfolioSearch,bounded_candidates
    from .orchestrator import AutotuneOrchestrator,ConfirmationPolicy,SequentialScheduler
    from .store import ResidualTuningStore
    from .core import TuningBudget
    from .first_hit_study import anchor_candidates
    task=manifest['tasks'][index];root=Path(output)/task['id'];root.mkdir(parents=True,exist_ok=True)
    if file_hash(manifest['prior'])!=manifest['prior_sha256']:raise ValueError('frozen prior changed')
    prior=FrozenPrior.load(manifest['prior'],research=True)
    with compile_pipeline(Path(output)/'gpu_execution.lock',device):
        started=time.monotonic();adapter,harness=make_adapter(task,root,device)
        adapter.context=replace(adapter.context,tags={**adapter.context.tags,
            'prior_policy':task['policy'],'prior_sha256':manifest['prior_sha256'],
            'prospective_seed':task['seed'],'prospective_cohort':task['cohort']})
        adapter.__post_init__()
        if task['family']=='mxfp8_recompute_mlp':
            candidates=anchor_candidates(adapter,task['precision'],task['phase'],manifest['pool_size'],task['seed'])
        else:
            candidates=[adapter.initial_config,*adapter.sample(random.Random(task['seed']),
                manifest['pool_size'],[adapter.initial_config])]
        # Every policy receives exactly this pool: retrieval enriches ALL arms,
        # not just the learned treatment. Timing never transfers from the prior.
        natural_order={adapter.config_id(c):i for i,c in enumerate(candidates)}
        candidates=bounded_candidates(adapter,[adapter.initial_config,*prior.retrieval(adapter),*candidates],manifest['pool_size'])
        # Keep the unranked control in generator order; prior-only additions
        # belong at its tail, not silently at the front of the control arm.
        candidates.sort(key=lambda c:natural_order.get(adapter.config_id(c),len(natural_order)))
        order=prior.order(adapter,candidates,policy=task['policy'],seed=task['seed'],**manifest.get('selector',{}))
        identity=dict(task=task,context=adapter.context.as_dict(),prior_sha256=manifest['prior_sha256'],
            policies=manifest.get('policies',['ordered','random','retrieval','hybrid']),
            pool_id=stable_id([dict(adapter.serialize(c)) for c in candidates]),
            candidate_ids=[adapter.config_id(c) for c in candidates],
            order_ids=[adapter.config_id(c) for c in order],prior_compatible=prior.bundle(adapter) is not None)
        identity_path=root/'trial.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise ValueError('prospective task identity or candidate pool changed')
        atomic_json(identity_path,identity)
        store=ResidualTuningStore(root/'residuals')
        strategy=PortfolioSearch(order)
        if not order:raise ValueError('no legal candidates in prospective pool')
        # The orchestrator evaluates initial_config before asking its strategy.
        # Make this the treatment's first choice, not an unaccounted default.
        adapter.initial_config=order[0]
        print(f'PRIOR_TRIAL {task["family"]} {task["policy"]} pool={len(candidates)} milestone={milestone}',flush=True)
        result=AutotuneOrchestrator(adapter,store,[strategy],SequentialScheduler(((strategy.name,None),)),
            TuningBudget(min(milestone,len(candidates)),manifest['slice_seconds']),transfer_history=False,
            max_trials_includes_resumed=True,confirmation=ConfirmationPolicy(1,.02,True),progress=print).tune()
        journal=root/'residuals/observations.jsonl'
        observed=len({json.loads(line)['config_id'] for line in journal.read_text().splitlines()}) if journal.exists() else 0
        atomic_json(root/f'milestone-{milestone}.json',dict(elapsed_s=time.monotonic()-started,
            prior_sha256=manifest['prior_sha256'],pool_id=identity['pool_id'],
            note='elapsed includes harness and lease waits; compiler caches shared',
            evaluated_trials=result.evaluated_trials,observed_configs=observed,
            target=min(milestone,len(candidates)),reached_target=observed>=min(milestone,len(candidates))))


def run(manifest_path,output,hours,device):
    from .compile_pipeline import run_workers
    if not math.isfinite(hours) or hours<=0:raise ValueError('hours must be positive and finite')
    manifest=json.loads(Path(manifest_path).read_text())
    if file_hash(manifest['prior'])!=manifest['prior_sha256']:raise ValueError('frozen prior changed')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+hours*3600
    workers=max(1,int(os.getenv('RTX_AUTOTUNE_CPU_WORKERS','4')))
    # Same device lease as the combined collector prevents overlapping campaigns.
    lock_id=stable_id([device,os.getenv('CUDA_VISIBLE_DEVICES','')])
    lock_path=Path('autotune_logs')/f'combined-{lock_id}.lock';lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(manifest=manifest,source_id=source_id())
        path=root/'study.json'
        if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('study identity changed; use a fresh output')
        atomic_json(path,identity)
        replicate=0
        while time.monotonic()<deadline:
            rr=root/f'replicate-{replicate:03d}';rr.mkdir(exist_ok=True)
            tasks=[{**task,'seed':task['seed']+replicate} for task in manifest['tasks']]
            rm={**manifest,'tasks':tasks};mp=rr/'manifest.json';atomic_json(mp,rm)
            for milestone in manifest['milestones']:
                pending=[(i,t) for i,t in enumerate(tasks) if not (rr/t['id']/f'milestone-{milestone}.json').exists()]
                if not pending:continue
                print(f'WAVE replicate={replicate} milestone={milestone} pending={len(pending)} remaining={deadline-time.monotonic():.1f}s',flush=True)
                if not run_workers(pending,rm,mp,rr,milestone,device,deadline,workers,
                                   worker_module='rtx.autotune.prior_study'):
                    print(f'DONE deadline output={root}; journals are resumable',flush=True)
                    return
            replicate+=1
        print(f'DONE output={root}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    plan=s.add_parser('plan');plan.add_argument('--prior',required=True);plan.add_argument('--output',required=True)
    for name in ('run','worker'):
        q=s.add_parser(name);q.add_argument('manifest');q.add_argument('--output',required=True);q.add_argument('--device',default='cuda:0')
        if name=='run':q.add_argument('--hours',type=float,default=4)
        else:q.add_argument('--index',type=int,required=True);q.add_argument('--milestone',type=int,required=True)
    a=p.parse_args()
    def stop(signum,frame):raise KeyboardInterrupt('prior study stopped')
    signal.signal(signal.SIGTERM,stop)
    if a.command=='plan':make_plan(a.prior,a.output)
    elif a.command=='run':run(a.manifest,a.output,a.hours,a.device)
    else:worker(json.loads(Path(a.manifest).read_text()),a.output,a.index,a.milestone,a.device)


if __name__=='__main__':main()
