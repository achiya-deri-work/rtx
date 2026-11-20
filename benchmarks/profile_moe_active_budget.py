"""Bounded, resumable packed/flat MoE study with a truthful active-width budget.

Architecture targets are not kernel schedules. Infeasible rounded geometries
are recorded, never silently substituted or timed. One GPU child at a time.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


def nearest64(value):
    value=Fraction(value)
    return max(64,64*((value.numerator+32*value.denominator)//(64*value.denominator)))


def geometry(d,fraction,k):
    f=Fraction(fraction)
    target=4*d
    shared=nearest64(target*f)
    hidden=nearest64(target*(1-f)/k)
    active=shared+k*hidden
    return dict(id=f'd{d}_f{f.numerator}of{f.denominator}_k{k}',d_model=d,top_k=k,
        requested_fraction=str(f),requested_shared_fraction=float(f),
        shared_hidden_dim=shared,expert_hidden_dim=hidden,active_hidden=active,
        target_active_hidden=target,effective_expansion=active/d,
        effective_shared_fraction=shared/active,expansion_error=(active-target)/target,
        feasible=abs(active-target)*40<=3*target,
        rounding='independently nearest-64 nominal shared/expert widths, ties upward')


def geometries():
    return [geometry(d,f,k) for d in (384,512)
            for f in (Fraction(1,4),Fraction(1,6),Fraction(1,8)) for k in (6,8)]


def joint_geometry(d,fraction,k):
    """Architecture selection, not kernel tuning: fraction first, budget second."""
    f=Fraction(fraction)
    if type(d) is not int or type(k) is not int or min(d,k)<1 or not 0<f<1:
        raise ValueError('Positive integer D/K and shared fraction strictly inside (0,1) required')
    target=4*d;upper=43*target//40;choices=[]
    for shared in range(64,upper+1,64):
        for hidden in range(64,(upper-shared)//k+1,64):
            active=shared+k*hidden
            if abs(active-target)*40<=3*target:
                choices.append((abs(Fraction(shared,active)-f),abs(active-target),shared,hidden))
    if not choices:raise ValueError('No 64-aligned geometry inside the hard active-budget interval')
    error,_,shared,hidden=min(choices);active=shared+k*hidden
    return dict(id=f'd{d}_f{f.numerator}of{f.denominator}_k{k}',d_model=d,top_k=k,
        requested_fraction=str(f),requested_shared_fraction=float(f),
        shared_hidden_dim=shared,expert_hidden_dim=hidden,active_hidden=active,
        target_active_hidden=target,effective_expansion=active/d,
        effective_shared_fraction=shared/active,expansion_error=(active-target)/target,
        shared_fraction_error_pp=float(error)*100,feasible=True,
        rounding='joint minimum absolute shared-fraction error; tie: minimum active-budget error')


def fanout_geometries():
    return [joint_geometry(d,f,k) for d,f,k in (
        (384,'1/6',4),(384,'1/8',4),(384,'1/4',4),
        (512,'1/6',4),(512,'1/4',4),(384,'1/6',6),
        (512,'1/6',6),(512,'1/6',8),(512,'1/4',6),(512,'1/4',8))]


def fanout_contexts():
    models={g['id']:g for g in fanout_geometries()};rows=[];seen=set()
    def add(model,e,b=16,length=1024,axis='architecture',layouts=('packed',),lengths=None):
        g=models[model];lengths=[length]*b if lengths is None else list(lengths)
        for layout in layouts:
            signature=(model,e,layout,tuple(lengths) if layout=='packed' else sum(lengths))
            if signature in seen:continue
            seen.add(signature)
            suffix=f'b{b}_l{length}' if len(set(lengths))==1 else f'b{b}_uneven_t{sum(lengths)}'
            rows.append(dict(id=f'{model}_e{e}_{suffix}_{layout}',axis=axis,
                case={name:g[name] for name in ('d_model','expert_hidden_dim','shared_hidden_dim','top_k')}
                    |dict(num_experts=e),geometry=g,lengths=lengths,layout=layout,
                samples=b,total_tokens=sum(lengths),seed=17391,stable_experts=True,
                study_group=0 if g['d_model']==384 and g['top_k']==4 else 1))
    # Qualify new K4 on both widths/layouts before broader coverage.
    for g in ('d384_f1of6_k4','d512_f1of4_k4'):add(g,32,axis='pilot',layouts=('packed','flat'))
    near=('d512_f1of6_k4','d384_f1of6_k6','d512_f1of6_k6','d512_f1of6_k8')
    for g in near:add(g,32,axis='nearest_fraction')
    # Exact quarter-shared budget and fixed K/E: equal mean rows per expert.
    for k,e in ((4,32),(6,48),(8,64)):
        add(f'd512_f1of4_k{k}',e,axis='matched_load',layouts=('packed','flat'))
    for e in (16,128,32,64):
        for f in ('1of6','1of8','1of4'):add(f'd384_f{f}_k4',e)
    for g in (*near,'d512_f1of4_k4'):add(g,128,axis='many_experts')
    for f in ('1of6','1of8','1of4'):
        add(f'd384_f{f}_k4',128,axis='flat_control',layouts=('flat',))
    for g in ('d384_f1of8_k4','d384_f1of4_k4','d512_f1of6_k4'):
        add(g,32,axis='flat_control',layouts=('flat',))
    for g,e in (('d384_f1of6_k4',64),('d512_f1of6_k4',32)):
        for b in (8,32):add(g,e,b,axis='batch',layouts=('packed','flat'))
        for b,length in ((16,2048),(8,4096)):add(g,e,b,length,axis='equal_total')
        add(g,e,16,axis='uneven',lengths=[1024,1536,2048,3584]*4)
    return rows


def contexts():
    models=[g for g in geometries() if g['feasible']]
    rows=[];seen=set()
    def add(g,e,b=16,length=1024,lengths=None,axis='architecture'):
        lengths=[length]*b if lengths is None else list(lengths)
        for layout in ('packed','flat'):
            # Flat semantic work depends only on T, not the diagnostic split.
            signature=(g['id'],e,layout,tuple(lengths) if layout=='packed' else sum(lengths))
            if signature in seen:continue
            seen.add(signature)
            suffix=f'b{b}_l{length}' if len(set(lengths))==1 else f'b{b}_uneven_t{sum(lengths)}'
            rows.append(dict(id=f'{g["id"]}_e{e}_{suffix}_{layout}',axis=axis,
                case={name:g[name] for name in ('d_model','expert_hidden_dim','shared_hidden_dim','top_k')}
                     |dict(num_experts=e),geometry=g,lengths=lengths,layout=layout,
                samples=b,total_tokens=sum(lengths),seed=17391))
    # Pilot wave covers both fanouts, both widths, many/few experts and layouts.
    for i,g in enumerate(models):add(g,16 if i%2==0 else 128,8,axis='pilot')
    for e in (16,128,32,64):
        for g in models:add(g,e)
    anchors=(models[0],models[-1])
    for b in (8,32):
        for g in anchors:add(g,64,b,axis='batch')
    for b,length in ((16,2048),(8,4096)):
        for g in anchors:add(g,64,b,length,axis='equal_total')
    for g in anchors:add(g,64,16,lengths=[1024,1536,2048,3584]*4,axis='uneven')
    return rows


def identity(source):
    digest=hashlib.sha256()
    for path in sorted((source/'rtx').rglob('*.py')):
        digest.update(str(path.relative_to(source)).encode());digest.update(path.read_bytes())
    return dict(runtime=digest.hexdigest(),drivers={name:hashlib.sha256(
        (source/'benchmarks'/name).read_bytes()).hexdigest() for name in
        ('profile_moe_active_budget.py','benchmark_moe_varlen.py','benchmark_moe_checkpoint.py')})


def worker_command(source,target,row,args):
    return [sys.executable,'-m','benchmarks.benchmark_moe_varlen','--source',str(source),
        '--output',str(target),'--case-id','d384_e32_k4_h128','--case-spec',json.dumps(row['case']),
        '--lengths',*map(str,row['lengths']),'--layout',row['layout'],'--hidden-alignment','64',
        '--stable-inputs','--seed',str(row['seed']),'--bf16-compact','--expert-phases','--shared-phases',
        '--oracle','--max-autotune','--search-schedule','progressive','--candidate-mix','mixed',
        '--tune-expert-trials',str(args.expert_trials),'--tune-shared-trials',str(args.shared_trials),
        '--tune-router-trials',str(args.router_trials),'--tune-seconds',str(args.tune_seconds),
        '--rounds',str(args.rounds),'--samples',str(args.samples),'--phases',
        'mxfp8_forward','mxfp8_backward','mxfp8_step','bf16_checkpoint_step','bf16_compact_checkpoint_step',
        'bf16_saved_step','router_step','expert_step','expert_bf16_checkpoint_step',
        'shared_step','shared_bf16_checkpoint_step']+(['--stable-experts'] if row.get('stable_experts') else [])


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_result(payload,context,manifest):
    """Fail closed on stale or semantically mismatched resumable records."""
    if payload.get('status')!='complete':raise ValueError('Incomplete result')
    if payload['source_digest']!=manifest['identity']['runtime']:raise ValueError('Source drift')
    if payload['driver_sha256']!=manifest['identity']['drivers']['benchmark_moe_varlen.py']:
        raise ValueError('Benchmark driver drift')
    for key in ('layout','lengths','total_tokens','seed'):
        if payload[key]!=context[key]:raise ValueError(f'Context mismatch: {key}')
    if any(payload['case'][k]!=v for k,v in context['case'].items()):
        raise ValueError('Architecture mismatch')
    if not payload['stable_inputs'] or payload['hidden_alignment']!=64:
        raise ValueError('Input/alignment contract mismatch')
    if context.get('stable_experts') and not payload.get('stable_experts'):
        raise ValueError('Expert initialization contract mismatch')
    settings=manifest['settings']
    for name in ('expert','shared','router'):
        if payload['tuning_budgets'][name]!=settings[name+'_trials']:
            raise ValueError('Tuning budget mismatch')
    required=('mxfp8_forward','mxfp8_backward','mxfp8_step','bf16_checkpoint_step',
              'bf16_compact_checkpoint_step','bf16_saved_step','router_step','expert_step',
              'expert_bf16_checkpoint_step','shared_step','shared_bf16_checkpoint_step')
    for name in required:
        rounds=payload['timings'][name]
        if len(rounds)!=settings['rounds']:raise ValueError('Timing round mismatch')
        if any(len(r['samples_ms'])!=settings['samples'] or
               any(not math.isfinite(v) or v<=0 for v in r['samples_ms']) for r in rounds):
            raise ValueError('Invalid timing samples')
    if not payload['validation'].get('all_finite'):raise ValueError('Missing finite validation')
    if not payload['validation'].get('independent_quantized_fp64_experts'):
        raise ValueError('Missing expert oracle')
    if payload['memory']['live_growth_bytes']!=0:raise ValueError('Live allocation growth')


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text());rows=[];incomplete=[]
    for context in manifest['contexts']:
        path=root/context['id']/'result.json'
        if not path.exists():incomplete.append(context['id']);continue
        p=json.loads(path.read_text())
        if p.get('status')!='complete':incomplete.append(context['id']);continue
        validate_result(p,context,manifest)
        row={key:context[key] for key in ('id','axis','layout','samples','total_tokens')}
        row.update(context['case']);row.update({k:v for k,v in context['geometry'].items()
            if k not in ('id','d_model','top_k','expert_hidden_dim','shared_hidden_dim')})
        for phase,rounds in p['timings'].items():
            values=[v for r in rounds for v in r['samples_ms']]
            row[phase+'_ms']=statistics.median(values)
            row[phase+'_min_ms']=min(values)
            row[phase+'_relative_mad']=statistics.median(abs(v-statistics.median(values)) for v in values)/statistics.median(values)
        native=row['mxfp8_step_ms']
        row['speedup_checkpoint']=row['bf16_checkpoint_step_ms']/native
        row['speedup_compact_checkpoint']=row['bf16_compact_checkpoint_step_ms']/native
        row['speedup_saved']=row['bf16_saved_step_ms']/native
        row['speedup_frozen_experts']=row['expert_bf16_checkpoint_step_ms']/row['expert_step_ms']
        row['speedup_shared']=row['shared_bf16_checkpoint_step_ms']/row['shared_step_ms']
        row['tokens_per_second']=row['total_tokens']*1000/native
        row['mean_real_rows_per_expert']=row['total_tokens']*row['top_k']/row['num_experts']
        row['physical_capacity']=p['expert_plan']['physical_capacity']
        row['peak_allocated_bytes']=p['memory']['peak_allocated_bytes']
        row['live_growth_bytes']=p['memory']['live_growth_bytes']
        row['oracle_max_nrms']=max(v['nrms'] for v in p['validation']['independent_quantized_fp64_experts'].values())
        row['expert_trials']=p['tuning_counts'].get('expert',{}).get('evaluated_trials',0)
        row['shared_trials']=p['tuning_counts'].get('shared',{}).get('evaluated_trials',0)
        rows.append(row)
    write(root/'summary.json',dict(rows=rows,incomplete=incomplete,identity=manifest['identity'],
        omitted_geometries=manifest['omitted_geometries'],
        caveats=['Standalone phases overlap: do not add them',
                 'Backward-only pins warmed forward saved tensors; normal donated-buffer/retain_graph=False; not subtraction',
                 'Packed and flat have different balancing semantics',
                 'Effective 64-aligned dimensions identical in each MX/BF16 comparison',
                 'Ordinary graph replay excludes compilation/search/warmup; no optimizer']))
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    if rows:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq.write_table(pa.Table.from_pylist(rows),root/'summary.parquet')
        except ImportError:pass
    print(json.dumps(dict(completed=len(rows),remaining=len(incomplete))),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('list','run','summarize'))
    p.add_argument('--output',type=Path)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--study',choices=('legacy','fanout'),default='legacy')
    p.add_argument('--contexts',nargs='+')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--expert-trials',type=int,default=8);p.add_argument('--shared-trials',type=int,default=8)
    p.add_argument('--router-trials',type=int,default=4);p.add_argument('--tune-seconds',type=float,default=90.)
    p.add_argument('--rounds',type=int,default=2);p.add_argument('--samples',type=int,default=7)
    p.add_argument('--max-seconds',type=float,default=5400.);p.add_argument('--job-seconds',type=float,default=600.)
    args=p.parse_args();source=args.source.resolve()
    proposed=geometries() if args.study=='legacy' else fanout_geometries()
    proposed_contexts=contexts() if args.study=='legacy' else fanout_contexts()
    if args.action=='list':print(json.dumps(dict(geometries=proposed,contexts=proposed_contexts),indent=2));return
    if args.output is None:p.error('--output required')
    root=args.output.resolve()
    if args.action=='summarize':summarize(root);return
    if not 0<=args.shard<args.shards:p.error('Invalid shard')
    models=[g['id'] for g in proposed if g['feasible']]
    rows=[r for r in proposed_contexts if
        (r.get('study_group',models.index(r['geometry']['id']))%args.shards)==args.shard]
    if args.contexts:
        requested=set(args.contexts)
        rows=[r for r in rows if r['id'] in requested]
        if requested!={r['id'] for r in rows}:p.error('Unknown or wrong-shard context')
    stamp=identity(source)
    settings={key:getattr(args,key) for key in ('expert_trials','shared_trials','router_trials','tune_seconds','rounds','samples')}
    manifest=dict(schema=1,study=args.study,identity=stamp,settings=settings,contexts=rows,
        omitted_geometries=[g for g in proposed if not g['feasible']],
        constraints=dict(active_expansion=4,maximum_relative_deviation=.075,all_hiddens_multiple=64),
        environment=dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),command=sys.argv)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        old=json.loads((root/'manifest.json').read_text())
        if any(old[k]!=manifest[k] for k in ('identity','settings','contexts','constraints')):
            raise ValueError('Resume contract differs; use another directory')
    else:write(root/'manifest.json',manifest)
    from rtx.autotune.supervisor import supervise_command,DEVICE_FAULT_EXIT_CODE
    start=time.monotonic()
    env={**os.environ,'PYTHONPATH':str(source),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TORCHINDUCTOR_COMPILE_THREADS':'4'}
    for index,row in enumerate(rows):
        path=root/row['id']/'result.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='complete':
            validate_result(json.loads(path.read_text()),row,manifest)
            continue
        if time.monotonic()-start>=args.max_seconds:break
        if identity(source)!=stamp:raise RuntimeError('Source changed')
        path.parent.mkdir(exist_ok=True)
        cmd=worker_command(source,path,row,args);write(path.parent/'command.json',cmd)
        print(f'START {index+1}/{len(rows)} {row["id"]}',flush=True)
        with path.with_name('run.log').open('a') as log:
            code=supervise_command(cmd,cwd=source,environment=env,output=log,
                stall_timeout_s=180,wall_timeout_s=args.job_seconds)
        event=dict(context=row['id'],exit_code=code,elapsed_s=time.monotonic()-start)
        with (root/'progress.jsonl').open('a') as log:
            log.write(json.dumps(event)+'\n');log.flush();os.fsync(log.fileno())
        print('DONE',json.dumps(event),flush=True);summarize(root)
        if code==DEVICE_FAULT_EXIT_CODE:raise RuntimeError('Device fault; stopped without reset')
        if code and not ('out of memory' in path.with_name('run.log').read_text().lower()):
            raise RuntimeError(f'Worker failed: {path.with_name("run.log")}')
    summarize(root)


if __name__=='__main__':main()
