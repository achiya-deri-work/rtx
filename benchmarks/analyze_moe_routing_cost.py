"""Offline attribution of ordinary timings, diagnostic traces and NCU traffic.

Kernel-class durations are diagnostic sums, never additive estimates of the
separately compiled router/expert/shared scopes or steady-state latency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics


def kernel_class(name):
    groups=(
        ('dispatch_combine',('CompactGather','CompactCombine','PackedInputGather','RoutedMovement')),
        ('quantization_fused_preparation',('RoutedDYQuant','IndexedExpertTranspose','ExpertQuantPortfolio','BackwardWeightQuant')),
        ('gemm',('MXFP8GemmKernel','PersistentExpertDW')),
        ('fused_recompute_activation_route_grad',('DualOperandJointRecompute','ExpertFragmentGLU','ExpertRouteGradient')),
        ('metadata_reduction',('CompactRoute','ActivePrefix','RouteGradientReduce')),
        ('framework_copy',('direct_copy_kernel','Memcpy')),
    )
    return next((label for label,terms in groups if any(term in name for term in terms)),'other')


def expert_trace(path):
    events=json.loads(path.read_text())['traceEvents'];durations=defaultdict(float);counts=defaultdict(int)
    for event in events:
        if event.get('cat')!='kernel':continue
        label=kernel_class(event['name']);durations[label]+=event.get('dur',0);counts[label]+=1
    total=sum(durations.values())
    return dict(kernel_sum_us=total,classes=[dict(category=k,duration_us=v,kernels=counts[k],
                fraction_of_kernel_sum=v/total) for k,v in sorted(durations.items(),key=lambda z:-z[1])],
                caveat='One diagnostic host-launched trace; not an uninstrumented latency benchmark')


def router_sqlite(path):
    with sqlite3.connect(path) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
            raise ValueError('Aggregate graph trace hides nodes; collect --cuda-graph-trace=node')
        kernels=list(db.execute('SELECT start,end,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL'))
        names=dict(db.execute('SELECT id,value FROM StringIds'))
        groups=defaultdict(lambda:[0,0.])
        for start,end,name in kernels:
            label=names.get(name,str(name));groups[label][0]+=1;groups[label][1]+=(end-start)/1000
        copies=[]
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            kinds=dict(db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER'))
            for kind,count,byte_count,ns in db.execute('SELECT copyKind,COUNT(*),SUM(bytes),SUM(end-start) '
                    'FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind'):
                copies.append(dict(kind=kinds.get(kind,str(kind)),count=count,bytes=byte_count,duration_us=ns/1000))
        return dict(kernel_count=len(kernels),kernel_sum_us=sum(v[1] for v in groups.values()),copies=copies,
            kernels=[dict(name=k,count=v[0],duration_us=v[1]) for k,v in sorted(groups.items(),key=lambda z:-z[1][1])],
            caveat='Node tracing perturbs execution; memcpy excludes movement performed by kernels')


def counters(path):
    rows=list(csv.DictReader(path.open()));units=rows[0];result=[]
    metrics=('gpu__time_duration.sum','dram__bytes_op_read.sum','dram__bytes_op_write.sum',
             'lts__t_bytes.sum','launch__registers_per_thread','sm__warps_active.avg.pct_of_peak_sustained_active',
             'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')
    for row in rows[1:]:
        if not row.get('ID'):continue
        result.append(dict(name=row['Kernel Name'],category=kernel_class(row['Kernel Name']),
                           metrics={key:dict(value=float(row[key].replace(',','')),unit=units[key])
                                    for key in metrics if row.get(key)}))
    return dict(kernels=result,caveat='Selected kernels only; NCU replay/cache policy differs from natural outer-graph execution')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--times',type=Path,required=True)
    p.add_argument('--expert-trace',type=Path);p.add_argument('--router-sqlite',type=Path)
    p.add_argument('--counters',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    saved=json.loads(args.times.read_text())
    result=dict(case=saved['case'],lengths=saved['lengths'],layout=saved['layout'],
        ordinary_graph_ms={k:statistics.median(v['median_ms'] for v in values) for k,values in saved['timings'].items()},
        caveats=['Scopes overlap; do not sum independently compiled phases',
                 'Logical arena bytes are allocation sizes, not traffic counters'])
    if args.expert_trace:result['expert_trace']=expert_trace(args.expert_trace)
    if args.router_sqlite:result['router_trace']=router_sqlite(args.router_sqlite)
    if args.counters:result['movement_counters']=counters(args.counters)
    from benchmarks.profile_moe_active_budget import write
    write(args.output,result)


if __name__=='__main__':main()
