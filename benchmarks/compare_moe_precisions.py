"""Broad matched MXFP8 versus saved/checkpointed BF16, using the existing timer.

Flat and packed are different balancing semantics; compare precision within
each context. One disposable child at a time per GPU, durable per-case results.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

from benchmarks.profile_moe_active_budget import identity,write,worker_command


PHASES=('mxfp8_forward','bf16_saved_forward','bf16_compact_saved_forward',
        'mxfp8_backward','mxfp8_step','bf16_checkpoint_step',
        'bf16_compact_checkpoint_step','bf16_saved_step','bf16_compact_saved_step',
        'router_step','expert_step','expert_bf16_checkpoint_step',
        'shared_step','shared_bf16_checkpoint_step')


def contexts():
    # D,E,K,H,S and per-sample lengths. Architecture choices, not GPU schedules.
    designs=[(384,32,4,320,256,[1024]*8),
             (512,48,6,256,384,[2048]*8),
             (384,96,4,320,256,[1024,1536,2560,3072]*4),
             (512,64,8,192,384,[1024,2048,3072,2048]*4),
             (384,24,6,192,320,[1024]*16),
             (512,128,4,384,512,[1024]*32),
             (768,48,6,384,768,[1024]*8),
             (384,32,4,320,256,[128]*4)]
    rows=[]
    for index,(d,e,k,h,s,lengths) in enumerate(designs):
        for layout in ('flat','packed'):
            rows.append(dict(id=f'd{d}_e{e}_k{k}_h{h}_s{s}_t{sum(lengths)}_{layout}',
                case=dict(d_model=d,num_experts=e,top_k=k,expert_hidden_dim=h,shared_hidden_dim=s),
                lengths=lengths,layout=layout,seed=99137+index,stable_experts=True,
                group=index,axis='small_guard' if max(lengths)<1024 else 'primary',
                samples=len(lengths),total_tokens=sum(lengths)))
    return rows


def command(source,target,row,args):
    budget=min(8,args.trials) if row['axis']=='small_guard' else args.trials
    settings=SimpleNamespace(expert_trials=budget,shared_trials=budget,router_trials=budget,
        tune_seconds=args.seconds,rounds=args.rounds,samples=args.samples)
    cmd=worker_command(source,target,row,settings)
    cmd=cmd[:cmd.index('--phases')]+['--phases',*PHASES,'--stable-experts','--bf16-numerics']
    return cmd


def validate(payload,row,stamp):
    if payload.get('status')!='complete':raise ValueError('Incomplete worker')
    if payload['source_digest']!=stamp['runtime']:raise ValueError('Runtime source drift')
    if payload['driver_sha256']!=stamp['drivers']['benchmark_moe_varlen.py']:raise ValueError('Driver drift')
    for k in ('layout','lengths','seed','total_tokens'):
        if payload[k]!=row[k]:raise ValueError(('Context drift',k))
    if any(payload['case'][k]!=v for k,v in row['case'].items()):raise ValueError('Architecture drift')
    if payload['memory']['live_growth_bytes']:raise ValueError('Live allocation growth')
    if not payload['validation'].get('all_finite'):raise ValueError('Missing all-gradient finite check')
    if not payload['validation'].get('independent_quantized_fp64_experts'):raise ValueError('Missing independent oracle')
    if not payload['validation'].get('matched_bf16_discrepancy'):raise ValueError('Missing BF16 discrepancy')
    if set(payload['timings'])!=set(PHASES):raise ValueError('Missing timed phase')
    round_counts=set();sample_counts=set()
    for rounds in payload['timings'].values():
        round_counts.add(len(rounds))
        for sample in rounds:
            values=sample['samples_ms'];sample_counts.add(len(values))
            if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Invalid latency sample')
    if len(round_counts)!=1 or 0 in round_counts or len(sample_counts)!=1 or 0 in sample_counts:
        raise ValueError('Incomplete timing rounds/samples')


def summarize(root):
    from benchmarks.analyze_autotune_regions import export_table
    rows=[];missing=[];failures=[];numerics=[];tuning=[];timing_warnings=[]
    for folder in sorted(root.glob('gpu?')):
        manifest=json.loads((folder/'manifest.json').read_text())
        for context in manifest['contexts']:
            path=folder/context['id']/'result.json'
            if not path.exists() or json.loads(path.read_text()).get('status')!='complete':
                missing.append(dict(gpu=folder.name,context=context['id']));continue
            p=json.loads(path.read_text());validate(p,context,manifest['identity'])
            row=dict(gpu=folder.name,id=context['id'],layout=context['layout'],axis=context['axis'],
                total_tokens=context['total_tokens'],batch=len(context['lengths']),**context['case'])
            for phase,rounds in p['timings'].items():
                samples=[v for r in rounds for v in r['samples_ms']]
                row[phase+'_ms']=statistics.median(samples)
                row[phase+'_relative_mad']=statistics.median(abs(v-statistics.median(samples)) for v in samples)/statistics.median(samples)
                medians=[v['median_ms'] for v in rounds]
                drift=max(medians)/min(medians)-1
                if row[phase+'_relative_mad']>.02 or drift>.05:
                    timing_warnings.append(dict(context=context['id'],phase=phase,
                        relative_mad=row[phase+'_relative_mad'],round_drift=drift))
            mx=row['mxfp8_step_ms'];forward=row['mxfp8_forward_ms']
            saved=min(row[m+'_ms'] for m in ('bf16_saved_step','bf16_compact_saved_step'))
            checkpoint=min(row[m+'_ms'] for m in ('bf16_checkpoint_step','bf16_compact_checkpoint_step'))
            bf_forward=min(row[m+'_ms'] for m in ('bf16_saved_forward','bf16_compact_saved_forward'))
            row.update(speedup_saved=saved/mx,speedup_checkpoint=checkpoint/mx,
                speedup_training_forward=bf_forward/forward,tokens_per_second=context['total_tokens']*1000/mx,
                mean_real_rows_per_expert=context['total_tokens']*context['case']['top_k']/context['case']['num_experts'],
                physical_capacity=p['expert_plan']['physical_capacity'],
                oracle_max_nrms=max(v['nrms'] for v in p['validation']['independent_quantized_fp64_experts'].values()),
                output_bf16_nrms=p['validation']['matched_bf16_discrepancy']['output']['nrms'],
                live_growth_bytes=p['memory']['live_growth_bytes'],peak_allocated_bytes=p['memory']['peak_allocated_bytes'])
            for label,choices in (
                ('saved',('bf16_saved_step','bf16_compact_saved_step')),
                ('checkpoint',('bf16_checkpoint_step','bf16_compact_checkpoint_step')),
                ('forward',('bf16_saved_forward','bf16_compact_saved_forward'))):
                row['bf16_'+label+'_winner']=min(choices,key=lambda phase:row[phase+'_ms'])
            row['physical_capacity']=p['expert_plan']['physical_capacity']
            for name,values in p['validation']['matched_bf16_discrepancy'].items():
                numerics.append(dict(context=context['id'],gpu=folder.name,tensor=name,**values))
            from rtx.autotune.outcomes import TrialOutcome
            for journal in path.parent.rglob('observations.jsonl'):
                records=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
                if not records:continue
                outcomes=[TrialOutcome(**r['outcome']) for r in records]
                tuning.append(dict(context=context['id'],gpu=folder.name,family=records[0]['family'],
                    attempted=len(records),unique=len({r['config_id'] for r in records}),
                    timing_eligible=sum(o.timing_eligible for o in outcomes),
                    compile_errors=sum(o.status=='compile_error' for o in outcomes),
                    correctness_errors=sum(o.status=='correctness_error' for o in outcomes),
                    elapsed_s=sum(r['elapsed_s'] for r in records),journal=str(journal)))
            for part,counts in p['tuning_counts'].items():
                row[part+'_trials']=counts['evaluated_trials'];row[part+'_cold_s']=counts.get('invocation_s',counts['elapsed_s'])
            rows.append(row)
        progress=folder/'progress.jsonl'
        if progress.exists():failures.extend(json.loads(l) for l in progress.read_text().splitlines() if json.loads(l)['exit_code'])
    output=root/'analysis';output.mkdir(exist_ok=True)
    refinement=None
    refinement_dir=root/'refine_e96_router32'
    if (refinement_dir/'qualification.json').exists():
        qualification=json.loads((refinement_dir/'qualification.json').read_text())
        if not qualification['complete']:raise ValueError('Incomplete refinement')
        refined=json.loads((refinement_dir/'result.json').read_text())
        def median_phase(payload,name):
            return statistics.median(v for r in payload['timings'][name] for v in r['samples_ms'])
        baseline=next(r for r in rows if r['id']=='d384_e96_k4_h320_s256_t32768_packed')
        latency=median_phase(refined,'mxfp8_step')
        refinement=dict(original_whole_ms=baseline['mxfp8_step_ms'],refined_whole_ms=latency,
            original_trials=sum(map(len,qualification['before'].values())),
            final_trials=sum(map(len,qualification['after'].values())),
            newly_evaluated=qualification['newly_evaluated'],
            speedup_checkpoint=min(median_phase(refined,n) for n in (
                'bf16_checkpoint_step','bf16_compact_checkpoint_step'))/latency,
            speedup_saved=min(median_phase(refined,n) for n in (
                'bf16_saved_step','bf16_compact_saved_step'))/latency,
            qualification=str(refinement_dir/'qualification.json'),
            scope='Extra matched fixed-expert/shared diagnostic; not substituted into the primary matrix')
    aggregates={}
    for layout in ('flat','packed'):
        group=[r for r in rows if r['layout']==layout and r['axis']=='primary']
        aggregates[layout]={}
        for metric in ('speedup_training_forward','speedup_checkpoint','speedup_saved'):
            values=[r[metric] for r in group]
            if values:aggregates[layout][metric]=dict(geomean=math.exp(statistics.mean(math.log(v) for v in values)),
                min=min(values),max=max(values),wins=sum(v>1 for v in values),cases=len(values))
    write(output/'summary.json',dict(rows=rows,numerics=numerics,tuning=tuning,
        incomplete=missing,failures=failures,timing_warnings=timing_warnings,
        refinement=refinement,aggregates=aggregates,
        contract='Complete MoE; common router per precision; saved and checkpoint BF16 separately; best of native/compact BF16; no optimizer',
        caveats=['Graph replay excludes compilation/tuning/warmup','Forward is training forward, not no-grad inference',
                 '32 trials requested per component, not 32 globally; deadlines can truncate',
                 'Router/expert/shared phase scopes overlap; do not add',
                 'Fixed-input kernel numerics are not training convergence','Peak allocation is shared-process high water, not per-mode VRAM']))
    export_table(rows,output/'whole_layer')
    export_table(numerics,output/'numerics');export_table(tuning,output/'tuning')
    print(json.dumps(dict(completed=len(rows),incomplete=len(missing),failures=len(failures),aggregates=aggregates)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('list','run','summarize'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--shard',type=int,choices=(0,1),default=0)
    p.add_argument('--trials',type=int,default=32);p.add_argument('--seconds',type=float,default=90)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--samples',type=int,default=9)
    args=p.parse_args()
    if args.action=='list':print(json.dumps(contexts(),indent=2));return
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.action=='summarize':return summarize(root)
    source=args.source.resolve();stamp=identity(source)
    # Each board gets both layouts and covers widths, loads and token counts.
    assignments=(0,1,1,0,0,1,0,1)
    rows=[r for r in contexts() if assignments[r['group']]==args.shard]
    folder=root/f'gpu{args.shard}';folder.mkdir(exist_ok=True)
    manifest=dict(identity=stamp,contexts=rows,settings=vars(args)|dict(output=str(root),source=str(source)),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),command=sys.argv)
    path=folder/'manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[key]!=manifest[key] for key in ('identity','contexts','settings')):
            raise ValueError('Resume identity/settings differ')
    else:write(path,manifest)
    from rtx.autotune.supervisor import supervise_command,DEVICE_FAULT_EXIT_CODE
    env={**os.environ,'PYTHONPATH':str(source),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
         'TORCHINDUCTOR_COMPILE_THREADS':'4'}
    for index,row in enumerate(rows):
        destination=folder/row['id'];destination.mkdir(exist_ok=True);result=destination/'result.json'
        if result.exists() and json.loads(result.read_text()).get('status')=='complete':
            validate(json.loads(result.read_text()),row,stamp);continue
        if identity(source)!=stamp:raise ValueError('Frozen source drift')
        cmd=command(source,result,row,args);write(destination/'command.json',cmd)
        print('START',index+1,len(rows),row['id'],flush=True);start=time.monotonic()
        with (destination/'run.log').open('a') as log:
            code=supervise_command(cmd,cwd=source,environment=env,output=log,
                                  stall_timeout_s=240,wall_timeout_s=900)
        event=dict(context=row['id'],exit_code=code,elapsed_s=time.monotonic()-start)
        with (folder/'progress.jsonl').open('a') as log:
            log.write(json.dumps(event)+'\n');log.flush();os.fsync(log.fileno())
        print('DONE',json.dumps(event),flush=True)
        if code==DEVICE_FAULT_EXIT_CODE:raise RuntimeError('Device fault; stop this GPU')
        if code:raise RuntimeError('Worker failed; retain evidence and inspect before further timing')
        validate(json.loads(result.read_text()),row,stamp)


if __name__=='__main__':main()
