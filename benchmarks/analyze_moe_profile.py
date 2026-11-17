"""CPU-only MoE profile tables; never substitute profiler duration for timing."""
import argparse
from collections import defaultdict
import csv
import json
import io
import sqlite3
from pathlib import Path
import statistics


def interval_union(intervals):
    ordered=sorted((a,b) for a,b in intervals if b>=a)
    if not ordered:return 0.
    total=0.;start,end=ordered[0]
    for a,b in ordered[1:]:
        if a>end:total+=end-start;start,end=a,b
        else:end=max(end,b)
    return total+end-start


def component(label,kernel):
    if '/quant_portfolio_' in label:return 'quantization/preparation'
    if '/gemm_glu_' in label:return 'gate_up_and_hidden_quantization'
    if '/gemm_derivative_' in label or '/expert_joint_recompute_' in label:return 'gate_recompute_derivative'
    if '/gemm_linear_' in label:return 'packed_linear_gemm'
    if '/movement_' in label:return 'routing_gather_combine_grad_movement'
    if '/dequant_hidden_' in label:return 'hidden_dequantization'
    if '/native_hidden_scales_' in label:return 'hidden_scale_layout'
    text=(label+' '+kernel).lower()
    if any(x in text for x in ('routing','sinkhorn','balanced_hard')):return 'router'
    if 'mlp' in text or 'mxfp8_gemm' in text or 'mxfp8_glu' in text:return 'shared_mlp_or_unattributed_gemm'
    return 'framework_or_unattributed'


def trace_rows(payload):
    events=payload['traceEvents']
    ranges=[e for e in events if e.get('ph')=='X' and 'dur' in e
            and (e.get('name','').startswith('moe/') or e.get('name','').startswith('rtx::'))]
    launches={e.get('args',{}).get('correlation'):e for e in events
              if e.get('ph')=='X' and 'launch' in e.get('name','').lower()
              and e.get('cat') in ('cuda_runtime','cuda_driver')}
    rows=[]
    for e in events:
        if e.get('cat')!='kernel':continue
        args=e.get('args',{});launch=launches.get(args.get('correlation'))
        parents=[]
        if launch:
            parents=[r for r in ranges if r.get('pid')==launch.get('pid') and r.get('tid')==launch.get('tid')
                     and r['ts']<=launch['ts']<=r['ts']+r['dur']]
        leaves=[r for r in parents if r['name'].startswith('moe/')]
        owner=min(leaves or parents,key=lambda r:r['dur'])['name'] if parents else ''
        top=max(leaves,key=lambda r:r['dur'])['name'] if leaves else ''
        rows.append(dict(index=len(rows),kernel=e['name'],label=owner,top_range=top,category=component(owner,e['name']),
                         gpu_us=e['dur'],gpu_start_us=e['ts'],stream=args.get('stream'),
                         launch_cpu_us=launch.get('dur') if launch else None,
                         grid=args.get('grid'),block=args.get('block'),registers=args.get('registers per thread'),
                         shared_memory=args.get('shared memory'),correlation=args.get('correlation')))
    return rows


def summarize_trace(payload):
    rows=trace_rows(payload);groups=defaultdict(list)
    for r in rows:groups[r['category']].append(r)
    intervals=[(r['gpu_start_us'],r['gpu_start_us']+r['gpu_us']) for r in rows]
    span=max((b for a,b in intervals),default=0)-min((a for a,b in intervals),default=0)
    union=interval_union(intervals)
    return dict(launches=rows,kernel_count=len(rows),kernel_sum_us=sum(r['gpu_us'] for r in rows),
                gpu_span_us=span,gpu_active_union_us=union,gpu_gap_us=span-union,
                gpu_coverage=union/span if span else None,
                unmapped_launches=sum(not r['label'] for r in rows),
                components={name:dict(kernel_us=sum(r['gpu_us'] for r in rs),count=len(rs)) for name,rs in groups.items()},
                caveat='Instrumented kernel/span times; coverage is inside first-to-last GPU kernel, not host wall time')


def nsys_trace(path):
    """Normalize the small Systems SQLite capture to our correlated schema."""
    with sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings=dict(db.execute('SELECT id,value FROM StringIds'));events=[]
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
            if table not in tables:continue
            for r in db.execute('SELECT * FROM '+table):
                events.append(dict(ph='X',cat='cuda_runtime',name=strings.get(r['nameId'],''),
                    ts=r['start']/1000,dur=(r['end']-r['start'])/1000,pid=0,tid=r['globalTid'],
                    args=dict(correlation=r['correlationId'])))
        if 'NVTX_EVENTS' in tables:
            for r in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
                label=r['text'] or strings.get(r['textId'],'')
                events.append(dict(ph='X',cat='user_annotation',name=label,
                    ts=r['start']/1000,dur=(r['end']-r['start'])/1000,pid=0,tid=r['globalTid']))
        for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
            events.append(dict(ph='X',cat='kernel',name=strings.get(r['demangledName'],''),
                ts=r['start']/1000,dur=(r['end']-r['start'])/1000,
                args=dict(correlation=r['correlationId'],stream=r['streamId'],
                    grid=[r['gridX'],r['gridY'],r['gridZ']],block=[r['blockX'],r['blockY'],r['blockZ']],
                    **{'registers per thread':r['registersPerThread'],
                       'shared memory':r['staticSharedMemory']+r['dynamicSharedMemory']})))
    return dict(traceEvents=events)


def flatten_timing(payload,path):
    rows=[]
    n,d,e,c,k,h=payload['shape']
    config=payload['config'];p=config['prepared']
    multiple=max(p.get('capacity_multiple',1),128 if p['quant']['scale_layout']=='mma128'
                 else 32 if p.get('backward_preparation')=='joint' else 1)
    physical=(c+multiple-1)//multiple*multiple
    for phase,regimes in payload.get('timings',{}).items():
        for regime,rounds in regimes.items():
            if not rounds:continue
            times=[r['median_ms'] for r in rounds];median=statistics.median(times)
            rows.append(dict(path=str(path),tokens=n,d_model=d,experts=e,top_k=k,hidden=h,logical_capacity=c,
                             physical_capacity=physical,capacity_padding_factor=physical/c,
                             phase=phase,regime=regime,median_ms=median,min_ms=min(r['min_ms'] for r in rounds),
                             round_range=max(times)/min(times)-1,tokens_per_second=n/(median/1000),
                             max_sample_relative_mad=max(r['relative_mad'] for r in rounds),
                             kernel_revision=payload['source_revision'],temporary_peak=payload.get('memory',{}).get('temporary_peak'),
                             live_growth=payload.get('memory',{}).get('live_growth')))
    return rows


def flatten_variants(payload,path):
    baselines={r.get('round',0):r['outcome'].get('median_ms') for r in payload.get('variants',[])
               if r['label']=='frozen' and r['outcome']['status']=='ok'}
    rows=[]
    for r in payload.get('variants',[]):
        outcome=r['outcome'];latency=outcome.get('median_ms');base=baselines.get(r.get('round',0))
        rows.append(dict(path=str(path),shape=json.dumps(payload['shape']),label=r['label'],
                         round=r.get('round',0),status=outcome['status'],median_ms=latency,
                         frozen_ms=base,speedup=base/latency if base and latency else None,
                         error=json.dumps(outcome.get('error')),config=json.dumps(r['config'],sort_keys=True)))
    return rows


def csv_write(path,rows):
    if not rows:
        path.write_text('');return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def counter_rows(text,manifest=None):
    """Read NCU wide raw CSV. Missing counters are unknown, never zero.

    NCU includes an initial units row and can prefix diagnostics. Attribute
    launches by recorded order only when the entire launch count matches.
    """
    lines=text.splitlines()
    start=next((i for i,line in enumerate(lines) if line.startswith('"ID","Process ID"')),None)
    if start is None:return []
    reader=csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    units=next(reader,{})
    records=[r for r in reader if r.get('ID','').isdigit()]
    labels=[r['label'] for r in (manifest or [])]
    matched=len(labels)==len(records)
    result=[]
    def number(row,key,target=None):
        value=row.get(key,'').replace(',','')
        if not value or value in ('n/a','N/A'):return None
        try:value=float(value)
        except ValueError:return None
        if target=='us':
            factor={'ns':.001,'us':1.,'ms':1000.,'s':1e6}.get(units.get(key))
        elif target=='byte':
            factor={'byte':1.,'byte/block':1.,'Kbyte':1000.,'Kbyte/block':1000.,'Mbyte':1e6,'Gbyte':1e9}.get(units.get(key))
        else:factor=1.
        return value*factor if factor is not None else None
    for i,row in enumerate(records):
        duration=number(row,'gpu__time_duration.sum','us')
        read=number(row,'dram__bytes_op_read.sum','byte');write=number(row,'dram__bytes_op_write.sum','byte')
        l2=number(row,'lts__t_bytes.sum','byte')
        label=labels[i] if matched else ''
        r=dict(index=i,kernel=row['Kernel Name'],label=label,category=component(label,row['Kernel Name']),
               attribution='exact_count_order' if matched else 'unmapped',counter_us=duration,
               dram_read_bytes=read,dram_write_bytes=write,l2_bytes=l2,
               l2_hit_pct=number(row,'lts__t_sector_hit_rate.pct'),
               registers=number(row,'launch__registers_per_thread'),
               shared_bytes=number(row,'launch__shared_mem_per_block_allocated','byte'),
               grid_size=number(row,'launch__grid_size'),block_size=number(row,'launch__block_size'),
               waves=number(row,'launch__waves_per_multiprocessor'),
               occupancy_pct=number(row,'sm__warps_active.avg.pct_of_peak_sustained_active'),
               tensor_elapsed_pct=number(row,'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),
               dram_throughput_pct=number(row,'dram__throughput.avg.pct_of_peak_sustained_elapsed'),
               local_load_sectors=number(row,'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'),
               local_store_sectors=number(row,'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum'))
        for stall in ('long_scoreboard','short_scoreboard','barrier'):
            r[stall+'_pct']=number(row,f'smsp__warp_issue_stalled_{stall}_per_warp_active.pct')
        r['stall_ratio_warning']=any(r[stall+'_pct'] is not None and not 0<=r[stall+'_pct']<=100
                                     for stall in ('long_scoreboard','short_scoreboard','barrier'))
        r['dram_GBps']=(read+write)/(duration*1000) if duration and read is not None and write is not None else None
        r['l2_GBps']=l2/(duration*1000) if duration and l2 is not None else None
        result.append(r)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    timings=[];kernels=[];traces=[];failures=[];counters=[];variant_rows=[]
    for path in sorted(args.root.rglob('*.json')):
        try:
            value=json.loads(path.read_text())
            if 'traceEvents' in value:
                result=summarize_trace(value);rows=result.pop('launches')
                kernels.extend(dict(path=str(path),**r) for r in rows);traces.append(dict(path=str(path),**result))
            elif 'timings' in value and 'shape' in value and 'source_revision' in value:
                timings.extend(flatten_timing(value,path))
                variant_rows.extend(flatten_variants(value,path))
        except (ValueError,KeyError,TypeError) as error:failures.append(dict(path=str(path),error=str(error)))
    for path in sorted(args.root.rglob('*.csv')):
        if not path.with_suffix('.ncu-rep').exists():continue
        meta=path.with_suffix('.json');manifest=[]
        if meta.exists():manifest=json.loads(meta.read_text()).get('launch_manifest',[])
        counters.extend(dict(path=str(path),**r) for r in counter_rows(path.read_text(),manifest))
    for path in sorted(args.root.rglob('*timeline.sqlite')):
        try:
            result=summarize_trace(nsys_trace(path));rows=result.pop('launches')
            kernels.extend(dict(path=str(path),**r) for r in rows);traces.append(dict(path=str(path),**result))
        except (sqlite3.Error,KeyError) as error:failures.append(dict(path=str(path),error=str(error)))
    csv_write(args.output/'timings.csv',timings);csv_write(args.output/'kernels.csv',kernels)
    csv_write(args.output/'counters.csv',counters)
    csv_write(args.output/'variants.csv',variant_rows)
    (args.output/'trace-summary.json').write_text(json.dumps(dict(traces=traces,parse_failures=failures),indent=2)+'\n')
    print(json.dumps(dict(timing_rows=len(timings),kernel_rows=len(kernels),traces=len(traces),parse_failures=len(failures))))


if __name__=='__main__':main()
