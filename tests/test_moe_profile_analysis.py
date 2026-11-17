from benchmarks.analyze_moe_profile import interval_union,summarize_trace,flatten_timing


def test_union_does_not_double_count_concurrent_kernels():
    assert interval_union([(0,4),(2,7),(9,10)])==8
    assert interval_union([])==0


def test_driver_correlation_and_deepest_scope():
    events=[dict(ph='X',cat='user_annotation',name='moe/expert_step',ts=0,dur=100,pid=1,tid=2),
            dict(ph='X',cat='user_annotation',name='moe/expert_forward/quant_portfolio_0',ts=10,dur=20,pid=1,tid=2),
            dict(ph='X',cat='cuda_driver',name='cuLaunchKernel',ts=12,dur=3,pid=1,tid=2,args={'correlation':7}),
            dict(ph='X',cat='kernel',name='quant_kernel',ts=15,dur=10,args={'correlation':7,'stream':4}),
            dict(ph='X',cat='kernel',name='unknown',ts=30,dur=5,args={'correlation':8})]
    result=summarize_trace(dict(traceEvents=events))
    assert result['launches'][0]['label']=='moe/expert_forward/quant_portfolio_0'
    assert result['launches'][0]['launch_cpu_us']==3
    assert result['components']['quantization/preparation']['kernel_us']==10
    assert result['gpu_gap_us']==5 and result['unmapped_launches']==1


def test_padding_and_across_round_timing_not_profiler_values():
    p=dict(shape=[128,384,24,32,6,128],source_revision='frozen',
           config=dict(prepared=dict(capacity_multiple=128,quant=dict(scale_layout='row_major'))),
           timings=dict(expert_step=dict(hot=[dict(median_ms=2,min_ms=1.9,relative_mad=.01),
                                             dict(median_ms=4,min_ms=3.9,relative_mad=.02)],evict=[])))
    row=flatten_timing(p,'x')[0]
    assert row['physical_capacity']==128 and row['capacity_padding_factor']==4
    assert row['median_ms']==3 and row['max_sample_relative_mad']==.02


def test_retarget_shared_split_partitions_is_explicit_and_nonmutating():
    from benchmarks.profile_moe import retarget_auxiliary
    dw=dict(reduction='split_fp32_workspace',reduction_tile=256,split_reduction=8,workspace_epilogue='tree')
    aux=dict(shared=dict(config=dict(gate_backward=dict(dw=dw))))
    small,changes=retarget_auxiliary(aux,128)
    assert dw['split_reduction']==8
    assert small['shared']['config']['gate_backward']['dw']['reduction']=='full_fp32'
    assert len(changes)==1
    large,changes=retarget_auxiliary(aux,32768)
    assert large['shared']['config']['gate_backward']['dw']['split_reduction']==128
    assert len(changes)==1
    assert retarget_auxiliary({},128)==({},[])


def test_ncu_units_and_missing_are_not_zero():
    from benchmarks.analyze_moe_profile import counter_rows
    text='''diagnostic prefix
"ID","Process ID","Kernel Name","gpu__time_duration.sum","dram__bytes_op_read.sum","dram__bytes_op_write.sum","launch__shared_mem_per_block_allocated","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
"","","","ns","byte","byte","Kbyte/block","%"
"0","4","quant","2000","8000","4000","1.024","125"
'''
    rows=counter_rows(text,[dict(label='moe/expert_forward/quant_portfolio_0')])
    assert len(rows)==1 and rows[0]['counter_us']==2
    assert rows[0]['dram_GBps']==6 and rows[0]['shared_bytes']==1024
    assert rows[0]['tensor_elapsed_pct'] is None
    assert rows[0]['stall_ratio_warning'] and rows[0]['long_scoreboard_pct']==125
    assert rows[0]['category']=='quantization/preparation'
    assert counter_rows(text,[])[0]['label']==''


def test_nsys_sqlite_correlation(tmp_path):
    import sqlite3
    from benchmarks.analyze_moe_profile import nsys_trace
    path=tmp_path/'timeline.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('''
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        INSERT INTO StringIds VALUES(1,'cudaLaunchKernelExC'),(2,'quant');
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER,correlationId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(2000,3000,1,42,7);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
        INSERT INTO NVTX_EVENTS VALUES(1000,4000,'moe/expert_forward/quant_portfolio_0',NULL,42);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,correlationId INTEGER,streamId INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,registersPerThread INTEGER,staticSharedMemory INTEGER,dynamicSharedMemory INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(5000,8000,2,7,1,2,1,1,128,1,1,32,1024,1024);
        ''')
    row=summarize_trace(nsys_trace(path))['launches'][0]
    assert row['label']=='moe/expert_forward/quant_portfolio_0'
    assert row['gpu_us']==3 and row['shared_memory']==2048
