"""Current-source, bounded-effort MoE strengths/weaknesses study.

One supervised GPU child at a time, atomic per-context records and fsynced
residual progress. No kernel implementations or runtime defaults are changed.
GPU choice/locking belongs to the caller. Resume rejects stale contracts.
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


def contexts():
    base=dict(d_model=384,num_experts=32,top_k=4,expert_hidden_dim=128,shared_hidden_dim=256)
    rows=[]
    def add(name,axis,n=8192,patch=None,lengths=None,profile='normal',seed=17391):
        rows.append(dict(id=name,axis=axis,case={**base,**(patch or {})},
            lengths=[n] if lengths is None else lengths,input_profile=profile,seed=seed))
    # Interleave categories so interruption does not yield only one shape family.
    add('anchor_8192','tokens')
    add('tiny_128','tokens',128)
    add('original_768','original_design',patch=dict(d_model=768,num_experts=48,top_k=6,
        expert_hidden_dim=256,shared_hidden_dim=512))
    add('packed_8','packing',lengths=[1024]*8)
    add('thin_long','width_tokens',32768,dict(d_model=128))
    add('hidden_512','expert_hidden',patch=dict(expert_hidden_dim=512))
    add('experts_128','expert_count',patch=dict(num_experts=128))
    add('half_aligned','alignment',lengths=[7103,9037],patch=dict(d_model=576,num_experts=24,
        top_k=3,expert_hidden_dim=160,shared_hidden_dim=320))
    add('fanout_8','top_k',patch=dict(top_k=8))
    add('outliers','input_distribution',profile='outliers')
    add('long_32768','tokens',32768)
    add('original_512','original_design',patch=dict(d_model=512,num_experts=64,top_k=8,
        shared_hidden_dim=384))
    add('small_512','tokens',512)
    add('medium_2048','tokens',2048)
    add('original_384_top6','original_design',patch=dict(num_experts=24,top_k=6))
    add('thin_128','width',patch=dict(d_model=128))
    add('width_256','width',patch=dict(d_model=256))
    add('width_1024','width',patch=dict(d_model=1024))
    add('width_1536','width',patch=dict(d_model=1536))
    for h in (64,192,256):add(f'hidden_{h}','expert_hidden',patch=dict(expert_hidden_dim=h))
    for e in (16,64):add(f'experts_{e}','expert_count',patch=dict(num_experts=e))
    add('fanout_2','top_k',patch=dict(top_k=2))
    add('equal_active_hidden','equal_kh',patch=dict(num_experts=64,top_k=8,expert_hidden_dim=64))
    for h in (128,512):add(f'shared_{h}','shared_hidden',patch=dict(shared_hidden_dim=h))
    add('packed_2','packing',lengths=[4096,4096])
    add('packed_32','packing',lengths=[256]*32)
    add('packed_uneven','packing',lengths=[127,513,2001,767,4095,689])
    add('single_12320','packing_boundary',12320)
    add('ragged_12320','packing_boundary',lengths=[4097,8223])
    add('long_65536','tokens',65536)
    add('wide_long_hidden','interaction',32768,dict(d_model=768,expert_hidden_dim=512,shared_hidden_dim=512))
    add('uniform','input_distribution',profile='uniform')
    add('correlated','input_distribution',profile='correlated')
    add('anchor_seed2','seed',seed=28812)
    return rows


def source_identity(source):
    runtime=hashlib.sha256()
    for path in sorted((source/'rtx').rglob('*.py')):
        runtime.update(str(path.relative_to(source)).encode());runtime.update(path.read_bytes())
    drivers={name:hashlib.sha256((source/'benchmarks'/name).read_bytes()).hexdigest()
        for name in ('profile_moe_challenge.py','benchmark_moe_varlen.py','analyze_moe_amortization.py')}
    return dict(runtime_digest=runtime.hexdigest(),driver_digests=drivers)


def command(source,target,row,args):
    phases=['mxfp8_forward','bf16_saved_forward','mxfp8_step','bf16_checkpoint_step',
        'bf16_saved_step','router_step','expert_step','expert_bf16_checkpoint_step',
        'shared_step','shared_bf16_checkpoint_step']
    cmd=[sys.executable,'-m','benchmarks.benchmark_moe_varlen','--source',str(source),
        '--output',str(target),'--case-id','d384_e32_k4_h128','--case-spec',json.dumps(row['case']),
        '--lengths',*map(str,row['lengths']),'--hidden-alignment','32',
        '--seed',str(row['seed']),'--input-profile',row['input_profile'],
        '--expert-phases','--shared-phases','--bf16-numerics','--oracle',
        '--max-autotune','--search-schedule','progressive','--candidate-mix','mixed',
        '--tune-expert-trials',str(args.expert_trials),'--tune-shared-trials',str(args.shared_trials),
        '--tune-router-trials',str(args.router_trials),'--tune-seconds',str(args.tune_seconds),
        '--rounds',str(args.rounds),'--samples',str(args.samples),'--phases',*phases]
    if not args.no_traces:cmd+=['--trace-phases','mxfp8_step','expert_step','bf16_checkpoint_step']
    return cmd


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def summarize(root):
    from benchmarks.analyze_moe_amortization import result_row,trace_summary
    manifest=json.loads((root/'manifest.json').read_text())
    rows=[];details=[];incomplete=[]
    for context in manifest['contexts']:
        path=root/context['id']/'result.json'
        if not path.exists():incomplete.append(dict(context=context['id'],status='missing'));continue
        record=json.loads(path.read_text())
        if record.get('status')!='complete':
            incomplete.append(dict(context=context['id'],status=record.get('status')));continue
        if record['source_digest']!=manifest['identity']['runtime_digest']:
            raise ValueError(('Runtime drift',str(path)))
        row=result_row(record)
        row.update(context=context['id'],axis=context['axis'],input_profile=context['input_profile'],seed=context['seed'])
        row['speedup_saved']=row['bf16_saved_step_ms']/row['mxfp8_step_ms']
        row['forward_speedup']=row['bf16_saved_forward_ms']/row['mxfp8_forward_ms']
        row['shared_speedup']=row['shared_bf16_checkpoint_step_ms']/row['shared_step_ms']
        row['tokens_per_second']=record['total_tokens']*1000/row['mxfp8_step_ms']
        row['useful_routed_training_flops_model']=24*record['total_tokens']*record['case']['top_k']*record['case']['d_model']*record['case']['expert_hidden_dim']
        row['routed_tflops_model']=row['useful_routed_training_flops_model']/row['expert_step_ms']/1e9
        errors=record['validation'].get('matched_bf16_discrepancy',{})
        for name in ('output','input'):
            if name in errors:row[f'bf16_{name}_nrms']=errors[name]['nrms']
        if errors:
            row['bf16_max_gradient_nrms']=max(v['nrms'] for k,v in errors.items() if k!='output')
        row['oracle_max_nrms']=max(v['nrms'] for v in record['validation']['independent_quantized_fp64_experts'].values())
        detail=dict(context=context['id'],traces={},bf16_discrepancy=errors,
            expert_config=record['expert_config'],shared_config=record['shared_config'])
        for phase in ('mxfp8_step','expert_step','bf16_checkpoint_step'):
            trace=path.with_name('result.'+phase+'.trace.json')
            if not trace.exists():continue
            summary=trace_summary(json.loads(trace.read_text()));detail['traces'][phase]=summary
            row[phase+'_kernel_count']=summary['count']
            row[phase+'_diagnostic_coverage']=summary['coverage']
            for group,values in summary['groups'].items():
                row[phase+'_trace_'+group+'_share']=values['share']
        rows.append(row);details.append(detail)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    output=dict(rows=rows,details=details,incomplete=incomplete,identity=manifest['identity'],
        caveats=['Same router in both precision paths; BF16 checkpoint and saved references separate',
            'Standalone phase ratios are not additive whole-layer attribution',
            'Trace shares are diagnostic durations, not ordinary runtime or hardware saturation counters',
            'Useful FLOPs exclude padding, routing, quantization and activation math; modeled, not measured instructions',
            'Matched BF16 discrepancies measure quantization too, not kernel correctness'])
    atomic_json(root/'summary.json',output)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows),root/'summary.parquet')
    except ImportError:pass
    print(json.dumps(dict(completed=len(rows),incomplete=len(incomplete),output=str(root/'summary.csv'))),flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('run','summarize','list'))
    p.add_argument('--output',type=Path)
    p.add_argument('--contexts',nargs='+')
    p.add_argument('--expert-trials',type=int,default=8)
    p.add_argument('--shared-trials',type=int,default=8)
    p.add_argument('--router-trials',type=int,default=4)
    p.add_argument('--tune-seconds',type=float,default=90.)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--samples',type=int,default=7)
    p.add_argument('--max-seconds',type=float,default=7200.)
    p.add_argument('--job-seconds',type=float,default=900.)
    p.add_argument('--no-traces',action='store_true')
    args=p.parse_args()
    if args.action=='list':print(json.dumps(contexts(),indent=2));return
    if args.output is None:p.error('--output required')
    root=args.output.resolve()
    if args.action=='summarize':summarize(root);return
    if min(args.expert_trials,args.shared_trials,args.router_trials,args.rounds,args.samples)<1:
        p.error('Positive budgets and measurement counts required')
    if min(args.tune_seconds,args.max_seconds,args.job_seconds)<=0:p.error('Positive time limits required')
    rows=contexts()
    if args.contexts:
        unknown=set(args.contexts)-{r['id'] for r in rows}
        if unknown:p.error('Unknown contexts: '+str(sorted(unknown)))
        rows=[r for name in args.contexts for r in rows if r['id']==name]
    source=Path(__file__).resolve().parents[1]
    identity=source_identity(source)
    settings={k:v for k,v in vars(args).items() if k not in ('action','output','contexts','max_seconds')}
    manifest=dict(schema=1,contexts=rows,identity=identity,settings=settings,
        gpu_selection=os.environ.get('CUDA_VISIBLE_DEVICES'),command=sys.argv,
        contract='Bounded first-hit selection, not exhaustive winners; BF16 tensors, FP32-sensitive math, MXFP8 expert/shared GEMMs; compiled fullgraph/static, outer graph replay; all required gradients; same native router; no optimizer')
    root.mkdir(parents=True,exist_ok=True)
    path=root/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous.get(k)!=manifest[k] for k in ('contexts','identity','settings','gpu_selection')):
            raise ValueError('Resume contract differs: choose a new output directory')
    else:atomic_json(path,manifest)
    from rtx.autotune.supervisor import supervise_command,DEVICE_FAULT_EXIT_CODE
    start=time.monotonic()
    env={**os.environ,'PYTHONPATH':str(source),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'TORCHINDUCTOR_COMPILE_THREADS':'4'}
    for index,row in enumerate(rows):
        target=root/row['id']/'result.json'
        if target.exists() and json.loads(target.read_text()).get('status')=='complete':
            print('CACHED',row['id'],flush=True);continue
        if time.monotonic()-start>=args.max_seconds:break
        if source_identity(source)!=identity:raise RuntimeError('Source changed during study')
        target.parent.mkdir(exist_ok=True)
        cmd=command(source,target,row,args)
        atomic_json(target.parent/'command.json',cmd)
        print(f'START {index+1}/{len(rows)} {row["id"]} elapsed={time.monotonic()-start:.1f}s',flush=True)
        job=time.monotonic()
        with (target.parent/'run.log').open('a') as log:
            code=supervise_command(cmd,stall_timeout_s=180,wall_timeout_s=args.job_seconds,
                output=log,cwd=str(source),environment=env)
        event=dict(context=row['id'],exit_code=code,seconds=time.monotonic()-job,
            elapsed_s=time.monotonic()-start)
        with (root/'progress.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
        print('DONE',json.dumps(event),flush=True)
        if code==DEVICE_FAULT_EXIT_CODE:raise RuntimeError('Sticky CUDA fault; study stopped, no GPU reset')
        summarize(root)
    summarize(root)


if __name__=='__main__':main()
