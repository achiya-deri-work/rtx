import json
import sqlite3

from benchmarks.analyze_precision200m_graph import graph_edges, trace


def test_event_nodes_preserve_only_existing_dependencies(tmp_path):
    def write(name,nodes):
        path=tmp_path/name
        path.write_text(json.dumps({'nodes':nodes}))
        return path
    a=dict(index=0,node_type='kernel',kernel_name='a',dependencies=[])
    b=dict(index=1,node_type='kernel',kernel_name='b',dependencies=[0])
    base=write('base.json',[a,b])
    expanded=write('expanded.json',[a,dict(index=1,node_type='event_record',kernel_name=None,
        dependencies=[0]),dict(b,index=2,dependencies=[1])])
    assert graph_edges(base)==graph_edges(expanded)
    changed=write('changed.json',[a,dict(b,dependencies=[])])
    assert graph_edges(base)!=graph_edges(changed)


def test_trace_distinguishes_overlap_from_summed_kernel_time(tmp_path):
    path=tmp_path/'trace.sqlite'
    with sqlite3.connect(path) as c:
        c.execute('create table StringIds(id integer,value text)')
        c.execute('create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,streamId integer,demangledName integer,graphId integer)')
        c.executemany('insert into StringIds values(?,?)',[(1,'TmaPreparation_o368643072'),(2,'NVFP4Gemm')])
        c.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?)',[
            (0,4000000,1,1,7),(1000000,3000000,2,2,7)])
    result=trace(path)
    assert result['span_ms']==4
    assert result['active_union_ms']==4
    assert sum(v['sum_kernel_ms'] for v in result['groups'].values())==6
    assert result['qkv_tma_mean_ms']==4
