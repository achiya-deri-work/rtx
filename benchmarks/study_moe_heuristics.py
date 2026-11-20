"""Controlled current-source MoE discovery/holdout study, not runtime dispatch.

One supervised GPU worker at a time per shard. Independent residual journals,
atomic results and explicit source contracts make partial work reusable.
CUDA_VISIBLE_DEVICES and external flock select/isolate the GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from benchmarks.profile_moe_active_budget import identity, write, worker_command


def study_contexts():
    rows = []

    def add(d, k, h, s, e, b=16, length=1024, layout='packed', *,
            partition='discovery', lengths=None):
        lengths = [length] * b if lengths is None else list(lengths)
        active = s + k*h
        if d <= 0 or min(k,h,s,e) <= 0 or h % 64 or s % 64 or e % 8:
            raise ValueError('Invalid aligned architecture')
        if abs(active-4*d)*40 > 3*4*d:
            raise ValueError('Active expansion outside hard ±7.5%')
        if k > e or not lengths or min(lengths) <= 0:
            raise ValueError('Invalid routing/sequence shape')
        suffix = '' if len(set(lengths)) == 1 else '_' + hashlib.sha256(
            json.dumps(lengths).encode()).hexdigest()[:6]
        rows.append(dict(id=f'd{d}_e{e}_k{k}_h{h}_s{s}_b{len(lengths)}_t{sum(lengths)}{suffix}_{layout}',
            partition=partition, layout=layout, lengths=lengths, samples=len(lengths),
            total_tokens=sum(lengths), seed=17391, stable_experts=True,
            case=dict(d_model=d,top_k=k,expert_hidden_dim=h,shared_hidden_dim=s,num_experts=e),
            geometry=dict(active_hidden=active,effective_expansion=active/d,
                          effective_shared_fraction=s/active),axis='heuristic_discovery'))

    for e in (16,32,64,128): add(384,4,320,256,e)
    for e in (32,128): add(384,4,320,256,e,layout='flat')
    for d,k,h,s,flat_e in ((384,6,192,320,64),(512,4,384,512,64),
                           (512,6,256,384,128),(512,8,192,512,64)):
        for e in (16,64,128): add(d,k,h,s,e)
        add(d,k,h,s,flat_e,layout='flat')
    for b in (8,32): add(384,4,320,256,64,b)
    add(384,4,320,256,24,8,1536,partition='holdout')
    add(384,6,192,384,96,8,1792,partition='holdout')
    add(512,4,448,384,48,8,1216,partition='holdout')
    add(512,6,256,512,48,8,2048,partition='holdout')
    add(512,8,192,384,96,8,1536,partition='holdout')
    add(384,4,320,256,48,8,partition='holdout',lengths=[1024,1536,2048,3584]*2)
    add(384,4,320,256,56,8,2304,layout='flat',partition='holdout')
    add(384,4,320,256,32,2,128,partition='holdout')
    assert len(rows) == 32 and len({r['id'] for r in rows}) == len(rows)
    return rows


def matches(payload,row):
    return (payload.get('status') == 'complete'
            and all(payload.get('case',{}).get(k) == v for k,v in row['case'].items())
            and payload.get('lengths') == row['lengths'] and payload.get('layout') == row['layout']
            and payload.get('seed') == row['seed'] and payload.get('stable_inputs') is True)


def result_validation(payload,row,stamp,settings):
    if not matches(payload,row): raise ValueError('Result shape/input contract mismatch')
    if payload['source_digest'] != stamp['runtime']: raise ValueError('Source mismatch')
    if payload['driver_sha256'] != stamp['drivers']['benchmark_moe_varlen.py']:
        raise ValueError('Driver mismatch')
    if payload.get('hidden_alignment') != 64: raise ValueError('Alignment mismatch')
    if payload.get('search_schedule') != 'progressive' or payload.get('candidate_mix') != 'mixed':
        raise ValueError('Search contract mismatch')
    for name in ('expert','shared','router'):
        if payload['tuning_budgets'][name] != settings[name+'_trials']:
            raise ValueError('Requested tuning budget mismatch')
        # Actual search counts may be lower: export them, never assume the cap was reached.
        if name not in payload['tuning_counts']: raise ValueError('Missing actual tuning evidence')
    if payload['memory']['live_growth_bytes'] != 0: raise ValueError('Live allocation growth')
    if not payload['validation'].get('independent_quantized_fp64_experts'):
        raise ValueError('Missing independent numerical oracle')
    if not payload['validation'].get('all_finite'): raise ValueError('Missing gradient finiteness')
    for phase in ('mxfp8_step','bf16_compact_checkpoint_step','router_step','expert_step','shared_step'):
        rounds = payload['timings'][phase]
        if len(rounds) != settings['rounds']: raise ValueError('Timing round mismatch')
        if any(len(r['samples_ms']) != settings['samples'] or
               any(not math.isfinite(v) or v <= 0 for v in r['samples_ms']) for r in rounds):
            raise ValueError('Invalid timing samples')


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text());rows=[];missing=[]
    for context in manifest['contexts']:
        path=root/context['id']/'result.json'
        if not path.exists(): missing.append(context['id']);continue
        payload=json.loads(path.read_text())
        if payload.get('status') != 'complete': missing.append(context['id']);continue
        result_validation(payload,context,manifest['identity'],manifest['settings'])
        row={k:context[k] for k in ('id','partition','layout','samples','total_tokens')}
        row.update(context['case']);row.update(context['geometry'])
        for phase,rounds in payload['timings'].items():
            row[phase+'_ms']=statistics.median(r['median_ms'] for r in rounds)
        for component in ('expert','shared','router'):
            row[component+'_trials']=payload['tuning_counts'][component]['evaluated_trials']
        row['compact_bf16_speedup']=row['bf16_compact_checkpoint_step_ms']/row['mxfp8_step_ms']
        row['saved_bf16_speedup']=row['bf16_saved_step_ms']/row['mxfp8_step_ms']
        row['router_to_whole_diagnostic_ratio']=row['router_step_ms']/row['mxfp8_step_ms']
        row['physical_capacity']=payload['expert_plan']['physical_capacity']
        row['mean_rows_per_expert']=row['total_tokens']*row['top_k']/row['num_experts']
        row['stable_experts']=payload['stable_experts']
        row['live_growth_bytes']=payload['memory']['live_growth_bytes']
        row['record_path']=str(path.resolve());row['source_digest']=payload['source_digest']
        rows.append(row)
    write(root/'summary.json',dict(rows=rows,missing=missing,manifest=manifest,
        caveats=['router/expert/shared timings overlap and must not be summed',
                 'both boards are RTX 5090, not cross-SKU qualification',
                 'cap != actual unique trials; use residual journal prefix eligibility',
                 'holdout rows must not select or fit a discovery heuristic']))
    if rows:
        names=list(dict.fromkeys(k for row in rows for k in row))
        with (root/'summary.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=names);writer.writeheader();writer.writerows(rows)
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows),root/'summary.parquet',compression='zstd')
    print(json.dumps(dict(complete=len(rows),remaining=len(missing))),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('list','run','summarize'))
    p.add_argument('--output',type=Path)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--partition',choices=('all','discovery','holdout'),default='all')
    p.add_argument('--expert-trials',type=int,default=32)
    p.add_argument('--shared-trials',type=int,default=32)
    p.add_argument('--router-trials',type=int,default=32)
    p.add_argument('--tune-seconds',type=float,default=300.)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--samples',type=int,default=7)
    p.add_argument('--max-seconds',type=float,default=7200.)
    p.add_argument('--job-seconds',type=float,default=1200.)
    p.add_argument('--adopt',nargs='*',type=Path,default=[],help='Exact-source completed pilot results; paths remain authoritative')
    args=p.parse_args()
    if args.action == 'list': print(json.dumps(study_contexts(),indent=2));return
    if args.output is None: p.error('--output required')
    root=args.output.resolve()
    if args.action == 'summarize': summarize(root);return
    if not 0 <= args.shard < args.shards: p.error('Invalid shard')
    source=args.source.resolve();stamp=identity(source)
    rows=[r for i,r in enumerate(study_contexts()) if i % args.shards == args.shard
          and (args.partition == 'all' or r['partition'] == args.partition)]
    settings={k:getattr(args,k) for k in ('expert_trials','shared_trials','router_trials','tune_seconds','rounds','samples')}
    manifest=dict(schema=1,identity=stamp,settings=settings,contexts=rows,
        study_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command=sys.argv,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        study='current-source discovery and untrained holdout',
        constraints=dict(active_expansion=4,relative_deviation=.075,hidden_alignment=64))
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        old=json.loads((root/'manifest.json').read_text())
        if any(old[k] != manifest[k] for k in ('identity','settings','contexts','constraints')):
            raise ValueError('Resume contract differs; use a separate directory')
    else: write(root/'manifest.json',manifest)
    # Adopt only an exact matched completed result, retaining the original residual root.
    # Copying the small result/sidecars is evidence export, not new measurement.
    import shutil
    for path in args.adopt:
        payload=json.loads(path.read_text())
        for row in rows:
            if matches(payload,row):
                result_validation(payload,row,stamp,settings)
                folder=root/row['id'];folder.mkdir(exist_ok=True)
                target=folder/'result.json'
                if target.exists():continue
                shutil.copy2(path,target)
                for kind in ('expert','shared','router'):
                    shutil.copy2(path.with_suffix('.'+kind+'.json'),target.with_suffix('.'+kind+'.json'))
                write(folder/'adoption.json',dict(source=str(path.resolve()),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),residual_root=str(path.parent.resolve())))
    from rtx.autotune.supervisor import supervise_command,DEVICE_FAULT_EXIT_CODE
    started=time.monotonic()
    env={**os.environ,'PYTHONPATH':str(source),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
         'TORCHINDUCTOR_COMPILE_THREADS':'4'}
    for index,row in enumerate(rows):
        path=root/row['id']/'result.json'
        if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
            result_validation(json.loads(path.read_text()),row,stamp,settings);continue
        if time.monotonic()-started >= args.max_seconds:break
        if identity(source) != stamp:raise RuntimeError('Frozen source changed')
        path.parent.mkdir(exist_ok=True)
        command=worker_command(source,path,row,args);write(path.parent/'command.json',command)
        print(f'START {index+1}/{len(rows)} {row["partition"]} {row["id"]}',flush=True)
        with (path.parent/'run.log').open('a') as log:
            code=supervise_command(command,cwd=source,environment=env,output=log,
                stall_timeout_s=240,wall_timeout_s=args.job_seconds)
        event=dict(context=row['id'],exit_code=code,elapsed_s=time.monotonic()-started)
        with (root/'progress.jsonl').open('a') as out:
            out.write(json.dumps(event)+'\n');out.flush();os.fsync(out.fileno())
        print('DONE',json.dumps(event),flush=True);summarize(root)
        if code == DEVICE_FAULT_EXIT_CODE:raise RuntimeError('CUDA fault; no follow-on GPU work')
        if code:print(f'REJECT context; preserved failure at {path.parent/"run.log"}',flush=True)
    summarize(root)


if __name__ == '__main__':main()
