"""Summarize ordinary timing and separately labeled diagnostic kernel shares."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from benchmarks.analyze_moe_profile import interval_union


def kernel_group(name):
    text=name.lower()
    if 'glurecomputederivative' in text:return 'shared_recompute_activation'
    if 'mxfp8linearfwdkernel' in text:return 'dynamic_linear_gemm_quant'
    if 'jointrecompute' in text:return 'fused_recompute_activation_quant'
    if 'expertfragmentglu' in text:return 'gate_up_gemm_activation_quant'
    if 'routegradientreduce' in text or 'expertdwreduce' in text:return 'expert_reductions'
    if 'expertroutegradient' in text:return 'gemm_route_gradient'
    if 'routedmovement' in text or 'packedinputgather' in text:return 'expert_gather_scatter'
    if any(s in text for s in ('routeddyquant','routedbackwardquant','backwardweightquant','expertquant','indexedexperttranspose',
        'mxfp8quantkernel','mxfp8transposedquant','mxfp8dualquant','scale','dequanthidden')):
        return 'quantization_preparation'
    if 'glukernel' in text:return 'gate_up_gemm_activation_quant'
    if text.startswith('triton_') and '_mm_' in text:return 'gemm'
    if any(s in text for s in ('expertgemm','shortreductionexpertdw','mxfp8gemmkernel','cutlass::','cublas','gemm','bmm')):
        return 'gemm'
    if any(s in text for s in ('sinkhorn','columnquantile','expertmajorkey','cachedquota','quotakernel',
        'routing_varlen','routing_selection','routing_metadata','routing_quota')):
        return 'routing'
    return 'compiler_framework_other'


def trace_summary(payload):
    kernels=[e for e in payload['traceEvents'] if e.get('cat')=='kernel']
    groups=defaultdict(lambda:dict(us=0.,count=0))
    for e in kernels:
        row=groups[kernel_group(e['name'])];row['us']+=e['dur'];row['count']+=1
    intervals=[(e['ts'],e['ts']+e['dur']) for e in kernels]
    span=max((b for a,b in intervals),default=0)-min((a for a,b in intervals),default=0)
    active=interval_union(intervals)
    total=sum(e['dur'] for e in kernels)
    return dict(count=len(kernels),kernel_sum_us=total,span_us=span,gap_us=span-active,
        coverage=active/span if span else None,
        short_kernel_count=sum(e['dur']<3 for e in kernels),
        groups={k:{**v,'share':v['us']/total if total else 0.} for k,v in groups.items()},
        top=sorted([dict(name=e['name'],us=e['dur'],group=kernel_group(e['name']),
            grid=e.get('args',{}).get('grid'),block=e.get('args',{}).get('block'),
            registers_per_thread=e.get('args',{}).get('registers per thread'),
            shared_memory=e.get('args',{}).get('shared memory')) for e in kernels],
            key=lambda r:-r['us'])[:12])


def expert_stages(record,payload):
    """Attribute only an exact-count isolated expert plan; never a whole trace."""
    config=record['expert_config'];p=config['prepared']
    if config['family']!='grouped_prepared' or p['forward_engine']!='packed':
        return dict(status='unmapped_family')
    expected=['fwd_gate_up','fwd_down']
    if p['backward_preparation']=='joint':
        expected+=['bwd_down_dx','bwd_joint_recompute','bwd_route_output','bwd_down_dw','bwd_gate_dx','bwd_gate_dw']
    else:
        expected+=['bwd_gate_recompute','bwd_down_recompute','bwd_down_dx','bwd_down_dw',
                   'bwd_derivative_recompute','bwd_gate_dx','bwd_gate_dw']
    groups={'gemm','gate_up_gemm_activation_quant','gemm_route_gradient','fused_recompute_activation_quant'}
    kernels=sorted((e for e in payload['traceEvents'] if e.get('cat')=='kernel'
        and kernel_group(e['name']) in groups),key=lambda e:e['ts'])
    if len(kernels)!=len(expected):
        return dict(status='unmapped_count',expected=len(expected),observed=len(kernels))
    return dict(status='exact_count_plan_order',stages=[dict(stage=stage,us=e['dur'],kernel=e['name'])
        for stage,e in zip(expected,kernels)])


def result_row(record):
    case=record['case'];n=record['total_tokens'];e=case['num_experts'];k=case['top_k']
    medians={p:statistics.median(r['median_ms'] for r in rounds) for p,rounds in record['timings'].items()}
    row=dict(gpu=record['gpu'],tokens=n,sequences=len(record['lengths']),**case)
    row.update({p+'_ms':v for p,v in medians.items()})
    row['runtime_source_digest']=record['source_digest']
    for name,counts in record.get('tuning_counts',{}).items():
        row[name+'_evaluated_trials']=counts['evaluated_trials']
        row[name+'_tuning_seconds']=counts['elapsed_s']
    row['speedup_checkpoint']=medians['bf16_checkpoint_step']/medians['mxfp8_step']
    row['expert_speedup_checkpoint']=medians['expert_bf16_checkpoint_step']/medians['expert_step']
    row['router_to_full_ratio']=(medians['router_step']/medians['mxfp8_step']
                                 if 'router_step' in medians else None)
    row['expert_to_full_ratio']=medians['expert_step']/medians['mxfp8_step']
    # Standalone phase ratios are indicative, not additive attribution.
    row['us_per_token']=medians['mxfp8_step']*1000/n
    shape=record['expert_plan'];cs=shape['physical_capacity']
    multiple=e//math.gcd(e,k)
    active_capacity=sum(math.ceil((math.ceil(length/multiple)*multiple*k//e)/32)*32
        for length in record['lengths'])
    row.update(logical_capacity=shape['shape'][3],physical_capacity=cs,mean_real_rows_per_expert=n*k/e,
        physical_slots=e*cs,slot_amplification=e*cs/(n*k),gpu_sms=record.get('gpu_sms'),
        active_capacity_from_lengths=active_capacity,active_slot_amplification=e*active_capacity/(n*k),
        static_bound_to_active_capacity=cs/active_capacity,
        peak_allocated_bytes=record['memory']['peak_allocated_bytes'],live_growth_bytes=record['memory']['live_growth_bytes'],
        max_relative_mad=max(r['relative_mad'] for rounds in record['timings'].values() for r in rounds))
    row['full_round_relative_range']=(max(r['median_ms'] for r in record['timings']['mxfp8_step'])-
        min(r['median_ms'] for r in record['timings']['mxfp8_step']))/medians['mxfp8_step']
    row['expert_round_relative_range']=(max(r['median_ms'] for r in record['timings']['expert_step'])-
        min(r['median_ms'] for r in record['timings']['expert_step']))/medians['expert_step']
    for phase in ('bf16_checkpoint_step','expert_bf16_checkpoint_step'):
        values=[r['median_ms'] for r in record['timings'][phase]]
        row[phase+'_round_relative_range']=(max(values)-min(values))/medians[phase]
    # Reporting flag only: retain raw nonstationary samples, never promote
    # a flaky timing into a claim about a small policy advantage.
    row['descriptive_full_timing_stable']=(row['max_relative_mad']<=.05 and row['full_round_relative_range']<=.10
        and row['bf16_checkpoint_step_round_relative_range']<=.10)
    row['descriptive_expert_timing_stable']=(row['max_relative_mad']<=.05 and row['expert_round_relative_range']<=.10
        and row['expert_bf16_checkpoint_step_round_relative_range']<=.10)
    p=record['expert_config']['prepared'];g=p['gate_up']
    row['gate_ctas_model']=e*math.ceil(cs/g['tile_m'])*math.ceil(2*case['expert_hidden_dim']/g['tile_n'])
    if row['gpu_sms']:row['gate_cta_waves_model']=row['gate_ctas_model']/row['gpu_sms']
    # CTA rounding is additional to physical-buffer padding. This is modeled
    # GEMM tile work, not measured tensor-core throughput or executed op count.
    d,h=case['d_model'],case['expert_hidden_dim']
    rounded=lambda size,tile:math.ceil(size/tile)*tile
    row['gate_tile_work_amplification_model']=(e*rounded(cs,g['tile_m'])*
        rounded(2*h,g['tile_n'])*rounded(d,g['tile_k'])/(n*k*2*h*d))
    w=p['gate_dw']
    row['gate_dw_tile_work_amplification_model']=(e*rounded(2*h,w['tile_m'])*
        rounded(d,w['tile_n'])*rounded(cs,w['tile_k'])/(n*k*2*h*d))
    if record['expert_config']['family']!='grouped_prepared' or p['forward_engine']!='packed':
        row['gate_ctas_model']=row['gate_cta_waves_model']=row['gate_tile_work_amplification_model']=None
    row.update(family=record['expert_config']['family'],dy_preparation=p['dy_preparation'],
        backward_preparation=p['backward_preparation'],input_preparation=p['input_preparation'])
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[];details=[];failures=[]
    for root in args.roots:
        for path in sorted(root.rglob('result.json')):
            record=json.loads(path.read_text())
            if record.get('status')!='complete':failures.append(str(path));continue
            row=result_row(record);row['context']=(path.parent.parent.name if path.parent.name.startswith('effort-') else path.parent.name);row['source']=str(path)
            detail=dict(context=row['context'],gpu=row['gpu'],source=str(path),traces={})
            for phase in ('mxfp8_step','expert_step','bf16_checkpoint_step','expert_bf16_checkpoint_step'):
                trace=path.with_name('result.'+phase+'.trace.json')
                if trace.exists():
                    payload=json.loads(trace.read_text())
                    summary=trace_summary(payload);detail['traces'][phase]=summary
                    row[phase+'_kernel_count']=summary['count']
                    row[phase+'_trace_coverage']=summary['coverage']
                    if phase=='expert_step':
                        for name,values in summary['groups'].items():row['expert_trace_'+name+'_share']=values['share']
                        attributed=expert_stages(record,payload);detail['expert_stages']=attributed
                        if attributed['status']=='exact_count_plan_order':
                            for stage in attributed['stages']:row['expert_trace_'+stage['stage']+'_us']=stage['us']
            rows.append(row);details.append(detail)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(dict(rows=rows,traces=details,incomplete=failures),indent=2)+'\n')
    print(json.dumps(dict(contexts=len(rows),incomplete=len(failures))))


if __name__=='__main__':main()
