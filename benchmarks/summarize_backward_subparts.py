"""Summarize independent timing and Nsight attribution without mixing them."""
import argparse
import csv
import io
import json
import sqlite3
from pathlib import Path


def classify(name):
    for token,label in (('HadamardPreparationKernel','rotation_quantization_correction'),
                        ('TmaPreparationKernel','rotation_quantization_correction'),
                        ('PosthocScaleKernel','scale_finalization'),
                        ('DualFinalizer','scale_finalization'),
                        ('ScaleProductKernel','factor_product'),
                        ('NVFP4GemmKernel','fp4_gemm'),
                        ('Quad','fp8_preparation'),
                        ('MXFP8GemmKernel','fp8_gemm'),
                        ('MXFP8Prequant','fp8_gemm')):
        if token in name: return label
    if 'triton_' in name: return 'compiled_rng_or_plumbing'
    if ('random_from_to_kernel' in name or 'FillFunctor' in name
            or 'distribution_elementwise_grid_stride_kernel' in name):
        return 'rng_state_management'
    return 'other'


def timeline(path):
    lines=path.read_text().splitlines()
    start=next(i for i,s in enumerate(lines) if s.startswith('Time (%),Total Time (ns),Instances,'))
    end=next((i for i in range(start+1,len(lines)) if not lines[i] or lines[i].startswith('Processing')),len(lines))
    rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start:end]))))
    groups={}
    for row in rows:
        label=classify(row['Name'])
        item=groups.setdefault(label,{'ns':0,'launches':0})
        item['ns']+=int(row['Total Time (ns)'])
        item['launches']+=int(row['Instances'])
    result={'groups':groups,'kernels':rows,'scope':'10 public backward replays; sum of kernel durations, not critical path'}
    database=path.with_name(path.name.replace('-stats.csv','.sqlite'))
    if database.exists():
        with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as connection:
            intervals=connection.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
            merged=[]
            for start,end in intervals:
                if merged and start<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],end)
                else: merged.append([start,end])
            active=sum(end-start for start,end in merged)
            span=max(end for _,end in intervals)-intervals[0][0]
            result['timeline']=dict(active_union_ns=active,kernel_hull_ns=span,
                coverage=active/span,kernel_count=len(intervals))
            resource_rows=connection.execute('''select distinct s.value,k.registersPerThread,
                k.dynamicSharedMemory,k.localMemoryPerThread,k.gridX,k.blockX,
                k.graphNodeId is not null from CUPTI_ACTIVITY_KIND_KERNEL k
                join StringIds s on k.demangledName=s.id''').fetchall()
            result['launch_resources']=[dict(name=r[0],category=classify(r[0]),
                registers=r[1],dynamic_smem=r[2],local_memory_per_thread=r[3],
                ctas=r[4],threads=r[5],graph_node=bool(r[6])) for r in resource_rows]
    return result


def summarize(root):
    result={'shapes':{},'timelines':{},'paired':{},'paired_graph8':{}}
    for path in sorted(root.glob('results/*.json')):
        row=json.loads(path.read_text())
        shape='x'.join(map(str,row['shape']))
        dst=result['shapes'].setdefault(shape,{})
        if 'backward_timing' in row:
            timing=row['backward_timing']
            dst[row['policy']]={'median_us':timing['median_ms']*1000,
                'min_us':timing['min_ms']*1000,
                'stationary':timing['sampling']['collection']['stationary'],
                'memory':row['memory'],'guards':row['guards']}
        elif 'phase' in row:
            dst['fp4_'+row['phase']]={
                'timings_us':{k:v['median_ms']*1000 for k,v in row['timings'].items()},
                'parts_us':{k:v['median_ms']*1000 for k,v in row['preparation_phases'].items()},
                'quality':row['quality'],'metrics':row['metrics']}
        else:
            dst['fp8_phases']={k:v['median_ms']*1000 for k,v in row['timings'].items()}
    for path in sorted((root/'timelines').glob('*-stats.csv')):
        result['timelines'][path.stem]=timeline(path)
    for group in ('paired','paired_graph8'):
        for path in sorted((root/group).glob('*.json')):
            row=json.loads(path.read_text())
            result[group]['x'.join(map(str,row['shape']))]={
                key:row[key] for key in ('median_ms','paired','replays')}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.root)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2)
    with args.output.with_suffix('.csv').open('x',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(('shape','family','phase','component','median_us','stationary'))
        for path in sorted((args.root/'results').glob('*.json')):
            row=json.loads(path.read_text())
            shape='x'.join(map(str,row['shape']))
            if 'backward_timing' in row:
                timing=row['backward_timing']
                writer.writerow((shape,row['policy'],'both','public_backward',
                    timing['median_ms']*1000,timing['sampling']['collection']['stationary']))
                continue
            for timing_key,quality_key in (('timings','quality'),('preparation_phases','preparation_phase_quality')):
                for name,timing in row.get(timing_key,{}).items():
                    stationary=row.get(quality_key,{}).get(name,{}).get('sampling',{}).get('collection',{}).get('stationary')
                    writer.writerow((shape,'fp4' if 'phase' in row else 'fp8',row.get('phase','both'),
                        name,timing['median_ms']*1000,stationary))
    for shape,data in result['shapes'].items():
        print(shape, {k:round(v['median_us'],2) for k,v in data.items() if 'median_us' in v})
        for phase in ('fp4_dx','fp4_dw','fp8_phases'):
            value=data.get(phase,{})
            print(' ',phase,value.get('timings_us',value))


if __name__=='__main__': main()
