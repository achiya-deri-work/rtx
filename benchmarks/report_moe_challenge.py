"""Offline challenge attribution and controlled contrasts; no CUDA access.

Frozen timing/profiling inputs are never rewritten. This postprocessor records
its own digest and keeps diagnostic trace shares separate from event timing.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from benchmarks.analyze_moe_amortization import kernel_group,result_row
from benchmarks.analyze_moe_profile import interval_union


def group(name):
    if 'persistentexpertdw' in name.lower():return 'gemm'
    if 'direct_copy_kernel' in name.lower():return 'framework_copy'
    return kernel_group(name)


def trace_detail(path):
    payload=json.loads(path.read_text())
    events=payload['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel']
    if not kernels:return dict(kernels=0,groups={})
    total=sum(e['dur'] for e in kernels)
    times=defaultdict(float);counts=Counter()
    for e in kernels:times[group(e['name'])]+=e['dur'];counts[group(e['name'])]+=1
    intervals=[(e['ts'],e['ts']+e['dur']) for e in kernels]
    span=max(b for a,b in intervals)-min(a for a,b in intervals)
    ordered=sorted(kernels,key=lambda e:e['ts'])
    gaps=sorted([dict(us=b['ts']-a['ts']-a['dur'],before=a['name'],after=b['name'])
        for a,b in zip(ordered,ordered[1:]) if b['ts']>a['ts']+a['dur']],key=lambda e:-e['us'])
    return dict(kernels=len(kernels),kernel_sum_us=total,span_us=span,
        coverage=interval_union(intervals)/span if span else None,
        coverage_contract='Visible kernel intervals only; gaps do not prove GPU idleness or CPU/eager overhead; conditional graph activity may be unattributed',
        largest_unattributed_intervals=gaps[:5],
        conditional_predicate_launches=sum('VarlenCondition' in e['name'] for e in kernels),
        groups={k:dict(us=v,share=v/total,count=counts[k]) for k,v in times.items()},
        cuda_api_counts=dict(Counter(e['name'] for e in events if e.get('cat')=='cuda_runtime')),
        top=[dict(name=e['name'],us=e['dur'],group=group(e['name']))
             for e in sorted(kernels,key=lambda e:-e['dur'])[:12]])


def contrasts(rows):
    by={r['context']:r for r in rows};anchor=by.get('anchor_8192')
    if anchor is None:return []
    result=[]
    # Same model and token total for packing/data; remaining axes change only
    # the named dimension, except the explicitly named equal-KH control.
    axes={'packing','input_distribution','seed','width','expert_hidden',
          'expert_count','top_k','shared_hidden','equal_kh'}
    for row in rows:
        if row['axis'] not in axes:continue
        result.append(dict(context=row['context'],axis=row['axis'],
            whole_latency_ratio=row['mxfp8_step_ms']/anchor['mxfp8_step_ms'],
            expert_latency_ratio=row['expert_step_ms']/anchor['expert_step_ms'],
            router_latency_ratio=row['router_step_ms']/anchor['router_step_ms'],
            bf16_whole_latency_ratio=row['bf16_checkpoint_step_ms']/anchor['bf16_checkpoint_step_ms'],
            speedup_change=row['speedup_checkpoint']-anchor['speedup_checkpoint'],
            slot_amplification=row['slot_amplification']))
    if all(k in by for k in ('single_12320','ragged_12320')):
        a,b=by['single_12320'],by['ragged_12320']
        result.append(dict(context='ragged_vs_single_12320',axis='packing_boundary',
            whole_latency_ratio=b['mxfp8_step_ms']/a['mxfp8_step_ms'],
            expert_latency_ratio=b['expert_step_ms']/a['expert_step_ms'],
            router_latency_ratio=b['router_step_ms']/a['router_step_ms'],
            bf16_whole_latency_ratio=b['bf16_checkpoint_step_ms']/a['bf16_checkpoint_step_ms'],
            speedup_change=b['speedup_checkpoint']-a['speedup_checkpoint'],
            slot_amplification=b['slot_amplification']))
    return result


def workspace_model(record):
    """Pure shape planning: typed expert arenas, not process VRAM or traffic."""
    from rtx._moe_execution import plan
    shape=record['expert_plan']['shape']
    config=json.dumps(record['expert_config'],sort_keys=True)
    result={}
    for phase in ('forward','backward'):
        key=json.dumps((shape,config,phase),sort_keys=True)
        builder,dtypes,sizes,views,outputs=plan(key)
        arena=sum(dtype.itemsize*size for dtype,size in zip(dtypes,sizes))
        output=sum(math.prod(builder.buffers[i].shape)*builder.buffers[i].dtype.itemsize for i in outputs)
        result[f'expert_{phase}_typed_arena_bytes_model']=arena
        result[f'expert_{phase}_output_bytes_model']=output
    return result


def systems_detail(folder):
    path=folder/'conditional.sqlite'
    if not path.exists():return None
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
        strings=dict(db.execute('select id,value from StringIds'))
        columns=[r[1] for r in db.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')]
        kernels=[dict(zip(columns,r)) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL')]
        copies=list(db.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY'))
    intervals=[(r['start'],r['end']) for r in kernels]+copies
    span=max(b for a,b in intervals)-min(a for a,b in intervals)
    predicates=sum('VarlenCondition' in strings[r['demangledName']] for r in kernels)
    api_path=folder/'stats_cuda_api_sum.csv'
    api=list(csv.DictReader(api_path.open())) if api_path.exists() else []
    return dict(kernel_count=len(kernels),memcpy_count=len(copies),span_us=span/1000,
        kernel_sum_us=sum(r['end']-r['start'] for r in kernels)/1000,
        kernel_and_copy_coverage=interval_union(intervals)/span,
        predicate_launches=predicates,repair_rounds=predicates-1,
        graph_launch_calls=sum(int(r['Num Calls']) for r in api if 'GraphLaunch' in r['Name']),
        per_kernel_launch_api_calls=sum(int(r['Num Calls']) for r in api if 'LaunchKernel' in r['Name']),
        scope='Diagnostic Nsight Systems node tracing; includes conditional body kernels omitted by the Torch trace; not ordinary performance or tensor-pipe counters')


def analyze(root):
    saved=json.loads((root/'summary.json').read_text());rows=saved['rows']
    detail={}
    for row in rows:
        context=row['context'];record=json.loads((root/context/'result.json').read_text())
        row.update(workspace_model(record))
        row.update(active_prefix=record['expert_config']['prepared']['active_prefix'],
            input_preparation=record['expert_config']['prepared']['input_preparation'],
            weight_preparation=record['expert_config']['prepared']['weight_preparation'],
            forward_glu_engine=record['expert_config']['prepared']['forward_glu_engine'])
        prepared=record['expert_config']['prepared']
        row['weight_elements_per_real_assignment_model']=(3*row['num_experts']*row['d_model']*
            row['expert_hidden_dim']/(row['tokens']*row['top_k']))
        row['active_hidden_width']=row['top_k']*row['expert_hidden_dim']
        c,d,h=row['physical_capacity'],row['d_model'],row['expert_hidden_dim']
        problems=dict(gate_up=(c,2*h,d),down=(c,d,h),recompute=(c,2*h,d),
            down_dx=(c,h,d),down_dw=(d,h,c),gate_dx=(c,d,2*h),gate_dw=(2*h,d,c))
        for index,stage in enumerate(('gate_up','down','recompute','down_dx','down_dw','gate_dx','gate_dw')):
            for field in ('tile_m','tile_n','tile_k','stages'):
                row[f'selected_{stage}_{field}']=prepared[stage][field]
            row[f'selected_{stage}_transport']=prepared['transports'][index]
            shape=problems[stage];g=prepared[stage]
            physical=math.prod(math.ceil(v/g[k])*g[k] for v,k in zip(shape,('tile_m','tile_n','tile_k')))
            logical=math.prod(shape)/c*(row['tokens']*row['top_k']/row['num_experts'])
            row[f'{stage}_macro_tile_work_amplification_model']=physical/logical
        for stage in ('gate_dw','down_dw'):
            row[f'selected_{stage}_engine']=prepared[stage+'_engine']
            row[f'selected_{stage}_splits']=prepared[stage+'_splits']
        traces={}
        for phase in ('mxfp8_step','expert_step','bf16_checkpoint_step'):
            path=root/context/f'result.{phase}.trace.json'
            if not path.exists():continue
            trace=trace_detail(path);traces[phase]=trace
            for g,value in trace['groups'].items():row[f'{phase}_diagnostic_{g}_share']=value['share']
        detail[context]=traces
    stable=[r for r in rows if r['descriptive_full_timing_stable']]
    work_rows=[]
    work_root=root/'work/results'
    work_manifest=work_root/'work-manifest.json'
    if work_manifest.exists():
        work=json.loads(work_manifest.read_text())
        if work['identity']['runtime_digest']!=saved['identity']['runtime_digest']:
            raise ValueError('WORK runtime differs')
        for context in work['contexts']:
            path=work_root/context['id']/'result.json'
            record=json.loads(path.read_text())
            if record['status']!='complete':raise ValueError('Incomplete WORK context')
            row=result_row(record)
            row.update(context=context['id'],axis=context['axis'],host='WORK',input_profile=context['input_profile'],seed=context['seed'])
            row.update(workspace_model(record))
            row['shared_speedup']=row['shared_bf16_checkpoint_step_ms']/row['shared_step_ms']
            row['speedup_saved']=row['bf16_saved_step_ms']/row['mxfp8_step_ms']
            errors=record['validation']['matched_bf16_discrepancy']
            row['bf16_output_nrms']=errors['output']['nrms']
            row['bf16_input_nrms']=errors['input']['nrms']
            row['oracle_max_nrms']=max(v['nrms'] for v in record['validation']['independent_quantized_fp64_experts'].values())
            work_rows.append(row)
    for row in rows:row['host']='HOME'
    controls={}
    for key,path in (('ffn_refinement',root/'refinement.json'),('router_refinement',root/'refinement_router.json'),
        ('home_compaction',root/'packed_32/compact_bf16.json'),('work_compaction',root/'work_compact_bf16.json')):
        if path.exists():controls[key]=json.loads(path.read_text())
    output=dict(rows=rows,work_rows=work_rows,contrasts=contrasts(rows),traces=detail,controls=controls,
        systems={name:systems_detail(root/name/'systems') for name in ('anchor_8192','outliers')},
        counts=dict(completed=len(rows),planned=len(rows)+len(saved['incomplete']),stable_full=len(stable),
            beats_checkpoint=sum(r['speedup_checkpoint']>1 for r in stable),
            above_1p05=sum(r['speedup_checkpoint']>1.05 for r in stable),
            near_parity_5pct=sum(.95<=r['speedup_checkpoint']<=1.05 for r in stable),
            below_p95=sum(r['speedup_checkpoint']<.95 for r in stable),
            exceeds_1p2=sum(r['speedup_checkpoint']>=1.2 for r in stable),
            beats_saved=sum(r['speedup_saved']>1 for r in stable)),
        identity=saved['identity'],postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination=root/'analysis';destination.mkdir(exist_ok=True)
    (destination/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    for name,values in (('contexts',rows),('contrasts',output['contrasts']),('all_contexts',rows+work_rows)):
        keys=list(dict.fromkeys(k for row in values for k in row))
        with (destination/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(values)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows),destination/'contexts.parquet')
        pq.write_table(pa.Table.from_pylist(rows+work_rows),destination/'all_contexts.parquet')
    except ImportError:pass
    print(json.dumps(output['counts']))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    args=p.parse_args();analyze(args.root.resolve())


if __name__=='__main__':main()
