import sqlite3

from benchmarks.summarize_backward_subparts import classify, timeline


def test_kernel_categories():
    assert classify('kernel_HadamardPreparationKernel') == 'rotation_quantization_correction'
    assert classify('kernel_TmaPreparationKernel') == 'rotation_quantization_correction'
    assert classify('kernel_finalize_inline_DualFinalizer') == 'scale_finalization'
    assert classify('kernel_MXFP8GemmKernel') == 'fp8_gemm'
    assert classify('kernel_NVFP4GemmKernel') == 'fp4_gemm'
    assert classify('triton_rand') == 'compiled_rng_or_plumbing'
    assert classify('random_from_to_kernel') == 'rng_state_management'
    assert classify('distribution_elementwise_grid_stride_kernel') == 'rng_state_management'


def test_timeline_overlap_is_not_added(tmp_path):
    path=tmp_path/'sample-stats.csv'
    path.write_text('Generating SQLite file\n'
        'Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n'
        '100,20,2,10,10,10,10,0,kernel_MXFP8GemmKernel\n\n'
        'Processing APIs\n')
    with sqlite3.connect(tmp_path/'sample.sqlite') as db:
        db.execute('create table StringIds (id integer,value text)')
        db.execute("insert into StringIds values (1,'kernel_MXFP8GemmKernel')")
        db.execute('''create table CUPTI_ACTIVITY_KIND_KERNEL
            (start integer,end integer,demangledName integer,registersPerThread integer,
             dynamicSharedMemory integer,localMemoryPerThread integer,gridX integer,
             blockX integer,graphNodeId integer)''')
        db.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,1,32,0,0,1,128,1)',
            [(0,10),(5,15)])
    result=timeline(path)
    assert result['groups']['fp8_gemm']['ns']==20
    assert result['timeline']['active_union_ns']==15
    assert result['timeline']['coverage']==1
    assert result['launch_resources'][0]['registers']==32
