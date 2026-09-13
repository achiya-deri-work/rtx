"""Isolated GPU revalidation and structural-panel measurements; no promotion."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

from tools.predictor_study import atomic, source_digest
from tools.research.coverage_audit import classify, effective_config, identity, policy
from tools.research.structural_sampling import flatten, distance


def rows(paths):
    for path in map(Path,paths):
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as archive:
                for name in sorted(archive.namelist()):
                    if name.endswith('/observations.jsonl'):
                        with archive.open(name) as stream:
                            for line in stream:yield json.loads(line)
        else:
            for file in sorted(path.rglob('observations.jsonl')):
                with file.open() as stream:
                    for line in stream:yield json.loads(line)


def prepare(args):
    root=Path(args.output)
    if (root/'plan.json').exists():raise ValueError('plan exists; run to resume')
    source=json.loads(Path(args.sources).read_text())
    groups=defaultdict(dict)
    for r in rows(source['train_paths']):
        old,active,known=classify(r)
        if r['family']!='mxfp8_recompute_mlp' or not old or active or not known or r['outcome']['status']!='ok':continue
        c=r['config']
        key=(policy(r,'precision'),policy(r,'phase'),c.get('gate_backend','onchip'),
             c.get('down_backend','onchip'),c.get('recompute_backend','onchip'))
        eid=identity(effective_config(r))
        groups[key].setdefault(eid,r)
    chosen=[]
    for key,bank in sorted(groups.items()):
        candidates=sorted(bank.values(),key=lambda r:r['config_id'])
        chosen.append(candidates[0])
        if len(candidates)>1:
            base=flatten(effective_config(candidates[0]))
            chosen.append(max(candidates[1:],key=lambda r:distance(base,flatten(effective_config(r)))))
    tasks=[]
    for r in chosen:
        for shape in ((128,256,512,256),(512,768,1536,768)):
            item=dict(kind='recovery',config=r['config'],source_observation_id=r['observation_id'],
                original_config_id=r['config_id'],precision=policy(r,'precision'),phase=policy(r,'phase'),
                shape=shape,seed=1729)
            item['id']=identity(item);tasks.append(item)
    panel=json.loads(Path(args.panel).read_text())
    selected=set(panel['selected_ids'][:16])
    # Reconstruct the original adapter on saved static metadata only. No GPU
    # work is performed by prepare, including sampling the local-walk control.
    from rtx.autotune.adapters import make_mxfp8_prequant_adapter
    from rtx.kernels.mxfp8 import MXFP8Problem
    import random
    c=panel['source_context'];w=c['workload']
    adapter=make_mxfp8_prequant_adapter(MXFP8Problem(w['m'],w['n'],w['k']),lambda c:None,device=c['device'])
    payloads={}
    for r in panel['raw_panel']['proposals']:
        config=adapter.deserialize(r['config']);cid=adapter.config_id(config)
        if cid in selected:payloads[cid]=dict(config=r['config'],sources=['structural'])
    walk={adapter.config_id(adapter.initial_config):adapter.initial_config}
    for candidate in adapter.sample(random.Random(1729),64,[adapter.initial_config]):
        walk.setdefault(adapter.config_id(candidate),candidate)
        if len(walk)>=16:break
    for config in walk.values():
        cid=adapter.config_id(config)
        payloads.setdefault(cid,dict(config=dict(adapter.serialize(config)),sources=[]))['sources'].append('local_walk')
    for cid,item in sorted(payloads.items()):
        task=dict(kind='panel',**item,config_id=cid,shape=[w['m'],w['n'],w['k']],seed=1729)
        task['id']=identity(task);tasks.append(task)
    document=dict(schema=1,source_sha256=source_digest(),worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tasks=tasks,source_manifest=args.sources,panel=args.panel,automatically_reinstated=False)
    atomic(root/'plan.json',document)
    print(f'PLAN recovery={sum(t["kind"]=="recovery" for t in tasks)} panel={sum(t["kind"]=="panel" for t in tasks)} output={root}',flush=True)


def metrics(actual,reference):
    import torch
    a=actual.detach().float();r=reference.detach().float();d=a-r
    if not bool(a.isfinite().all()) or not bool(r.isfinite().all()):
        return dict(finite=False,max_abs=None,rms_error=None,reference_rms=None,normalized_rms=None,normalized_max=None)
    rr=r.square().mean().sqrt();rm=d.square().mean().sqrt()
    return dict(finite=bool(a.isfinite().all()),max_abs=float(d.abs().max()),rms_error=float(rm),
                reference_rms=float(rr),normalized_rms=float(rm/rr.clamp_min(1e-12)),
                normalized_max=float(d.abs().max()/r.abs().max().clamp_min(1e-12)))


def worker(root,task_id):
    import torch
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    task=next(t for t in plan['tasks'] if t['id']==task_id)
    report=dict(task=task,status='error',torch_version=torch.__version__,cuda_version=torch.version.cuda)
    started=time.monotonic()
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        report['gpu']=torch.cuda.get_device_name()
        if task['kind']=='recovery':
            from rtx.mlp import MXFP8MLPConfig, mxfp8_mlp
            from rtx.autotune.mlp import make_mlp_adapter, reference
            m,k,h,o=task['shape'];torch.manual_seed(task['seed'])
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.2
            gu=torch.randn(2*h,k,device='cuda',dtype=torch.bfloat16)*.02
            down=torch.randn(o,h,device='cuda',dtype=torch.bfloat16)*.02
            adapter=make_mlp_adapter(x,gu,down,precision=task['precision'],phase=task['phase'],
                                     warmup=2,samples=7,calls_per_sample=3)
            config=adapter.deserialize(task['config'])
            rejection=adapter.rejection(config)
            if rejection is not None:
                report.update(status='static_rejection',rejection=str(rejection));return
            # First use an independently constructed recipe oracle to record
            # per-output and per-gradient absolute/normalized metrics.
            tensors=tuple(t.detach().requires_grad_() for t in (x,gu,down))
            dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
            expected=reference(*tensors,dy,'silu',task['precision'])
            def forward():return mxfp8_mlp(*tensors,activation='silu',config=config,precision=task['precision'])
            fn=torch.compile(forward,fullgraph=True,dynamic=False)
            if task['phase']=='train':
                y=fn();actual=(y,*torch.autograd.grad(y,tensors,dy))
            else:
                with torch.no_grad():actual=(fn(),)
            names=('output','dX','dGateUp','dDown')
            report['numerics']={n:metrics(a,r) for n,a,r in zip(names,actual,expected)}
            report['normalized_rms_gate']=.003 if task['precision']=='bf16_gate' else .025
            if any(not v['finite'] or v['normalized_rms']>report['normalized_rms_gate'] for v in report['numerics'].values()):
                report['status']='correctness_error';return
            # The established evaluator checks again before graph timing, using
            # fresh leaves/stream. It does not relax the existing numerical gate.
            result=adapter.evaluate(config)
        else:
            from tools.predictor_study import build_adapter
            m,n,k=task['shape']
            adapter,hardware=build_adapter(dict(family='mxfp8_prequant_fwd',shape=dict(m=m,n=n,k=k),
                regime='hot',seed=task['seed']),dict(warmup=3,samples=7,calls_per_sample=5),'cuda:0')
            report['hardware']=hardware
            config=adapter.deserialize(task['config'])
            rejection=adapter.rejection(config)
            if rejection is not None:
                report.update(status='static_rejection',rejection=str(rejection));return
            result=adapter.evaluate(config)
        report['outcome']=result.as_dict();report['status']=result.status
        if result.successful:
            report['verification']=[adapter.evaluate(config).as_dict() for _ in range(2)]
            if any(r['status']!='ok' for r in report['verification']):report['status']='verification_failed'
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    except Exception:
        report['error']=traceback.format_exc();traceback.print_exc()
    finally:
        report['elapsed_s']=time.monotonic()-started
        atomic(root/'tasks'/task_id/'result.json',report)


def run(root,limit):
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text())
    if source_digest()!=plan['source_sha256'] or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['worker_sha256']:
        raise ValueError('source changed; prepare a new run')
    import fcntl
    with (root/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=0
        for task in plan['tasks']:
            dest=root/'tasks'/task['id'];dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():continue
            if limit and done>=limit:break
            print('START',task['kind'],task['id'],task['shape'],flush=True)
            with (dest/'worker.log').open('a') as stream:
                try:
                    p=subprocess.run([sys.executable,'-u','-m','tools.coverage_gpu_study','worker',str(root),'--task',task['id']],
                        stdout=stream,stderr=subprocess.STDOUT,timeout=240,
                        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
                    code=p.returncode
                except subprocess.TimeoutExpired:code=-1
            if not (dest/'result.json').exists():atomic(dest/'result.json',dict(task=task,status='worker_failed',returncode=code))
            print('DONE',task['id'],json.loads((dest/'result.json').read_text())['status'],flush=True)
            done+=1
        summarize(root)


def summarize(root):
    from collections import Counter
    import statistics
    root=Path(root)
    records=[json.loads(p.read_text()) for p in sorted((root/'tasks').glob('*/result.json'))]
    groups=defaultdict(list)
    for r in records:
        if r['task']['kind']=='panel' and r['status']=='ok':
            ms=statistics.median([r['outcome']['median_ms'],*[v['median_ms'] for v in r['verification']]])
            for source in r['task']['sources']:groups[source].append(ms)
    report=dict(completed=len(records),statuses=dict(Counter((r['task']['kind']+':'+r['status']) for r in records)),
        panel={k:dict(successful=len(v),best_ms=min(v),median_ms=statistics.median(v)) for k,v in groups.items()},
        automatically_reinstated=False)
    atomic(root/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('prepare');a.add_argument('--output',required=True)
    a.add_argument('--sources',default='autotune_reports/effort_ml_v2/audit.json')
    a.add_argument('--panel',default='autotune_reports/coverage_audit_v2/mxfp8_panel.json')
    a=s.add_parser('run');a.add_argument('output');a.add_argument('--limit',type=int,default=0)
    a=s.add_parser('worker');a.add_argument('output');a.add_argument('--task',required=True)
    a=s.add_parser('summarize');a.add_argument('output')
    args=p.parse_args()
    if args.cmd=='prepare':prepare(args)
    elif args.cmd=='worker':worker(args.output,args.task)
    elif args.cmd=='run':run(args.output,args.limit)
    else:summarize(args.output)


if __name__=='__main__':main()
