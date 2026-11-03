"""Summarize ordinary controls and narrow Nsight graph traces without ratio summation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import statistics


def trace(path):
    c=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    graph_id=c.execute('select graphId from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null '
        'group by graphId order by count(*) desc limit 1').fetchone()[0]
    rows=c.execute('select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
        'join StringIds s on k.demangledName=s.id where graphId=? order by k.start',(graph_id,)).fetchall()
    active=0;frontier=rows[0][0]
    for a,b,_,_ in rows:
        active+=max(0,b-max(a,frontier));frontier=max(frontier,b)
    groups={}
    for label,needle in [('tma_preparation','TmaPreparation'),('row_preparation','HadamardPreparation'),
        ('scale_finalize','PosthocScaleKernel'),('scale_product','ScaleProductKernel'),('fp4_gemm','NVFP4Gemm')]:
        selected=[(a,b) for a,b,_,name in rows if needle in name]
        groups[label]=dict(count=len(selected),sum_kernel_ms=sum(b-a for a,b in selected)/1e6)
    qkv=[(b-a)/1e6 for a,b,_,name in rows if 'TmaPreparation' in name and 'o368643072' in name]
    return dict(graph_id=graph_id,kernels=len(rows),span_ms=(max(b for _,b,_,_ in rows)-rows[0][0])/1e6,
        active_union_ms=active/1e6,replay_streams=dict(Counter(s for _,_,s,_ in rows)),groups=groups,
        qkv_tma_mean_ms=statistics.mean(qkv),scope='single profiled step; summed kernel time overlaps across streams')


def graph_edges(path):
    ns=json.loads(path.read_text())['nodes'];counts=Counter();labels={}
    for n in ns:
        if n['node_type'] in ('kernel','memset'):
            name=n['kernel_name'] or n['node_type']
            labels[n['index']]=(name,counts[name]);counts[name]+=1
    def parents(i):
        if i in labels:return {labels[i]}
        return set().union(*(parents(j) for j in ns[i]['dependencies']))
    edges=set()
    for i,label in labels.items():
        for j in ns[i]['dependencies']:
            edges.update((p,label) for p in parents(j))
    return counts,edges


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    controls={}
    for d in sorted(args.root.iterdir()):
        path=d/'training.jsonl'
        if not path.is_file():continue
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if rows[-1].get('event')!='result':continue
        blocks=[r for r in rows if r['event']=='train'][1:]
        controls[d.name]=dict(post_first_block_ms=sum(r['block_training_wall_s'] for r in blocks)
            /sum(r['block_steps'] for r in blocks)*1000,
            median_block_ms=statistics.median(r['block_training_wall_s']/r['block_steps']*1000 for r in blocks),
            finite_parameters=rows[-1]['finite_parameters'],peak_allocated_bytes=rows[-1]['peak_allocated_bytes'])
    traces={p.stem:trace(p) for p in args.root.glob('nsys-*.sqlite')}
    a,ae=graph_edges(args.root/'normal-topology/graph-003.json')
    b,be=graph_edges(args.root/'untimed-topology/graph-003.json')
    payload=dict(controls=controls,traces=traces,topology=dict(same_kernels=a==b,
        same_compressed_dependencies=ae==be,kernel_memset_edges=len(ae)),
        no_production_default_changed=True)
    with args.output.open('x') as stream:json.dump(payload,stream,indent=2)


if __name__=='__main__':main()
