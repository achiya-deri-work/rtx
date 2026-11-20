from benchmarks.profile_moe_amortization import contexts
from benchmarks.analyze_moe_amortization import kernel_group,trace_summary


def test_bounded_one_axis_matrix_is_unique_and_not_a_cartesian_product():
    rows=contexts()
    assert len(rows)==20 and len({r['id'] for r in rows})==20
    baseline=next(r for r in rows if r['id']=='tokens_2000')
    for row in rows:
        changes=[k for k in baseline['case'] if row['case'][k]!=baseline['case'][k]]
        assert len(changes)<=1
        assert row['case']['top_k']<=row['case']['num_experts']
    packed=next(r for r in rows if r['id']=='packed_2000')
    assert sum(packed['lengths'])==2000 and packed['case']==baseline['case']


def test_diagnostic_fused_math_is_not_mislabeled_as_pure_quantization():
    assert kernel_group('ExpertJointRecompute')=='fused_recompute_activation_quant'
    assert kernel_group('DualOperandJointRecompute')=='fused_recompute_activation_quant'
    assert kernel_group('OperandReuseJointRecompute')=='fused_recompute_activation_quant'
    assert kernel_group('ShortReductionExpertDW')=='gemm'
    assert kernel_group('MXFP8PrequantGLUKernel')=='gate_up_gemm_activation_quant'
    assert kernel_group('RoutedDYQuant')=='quantization_preparation'
    assert kernel_group('ExpertRouteGradient')=='gemm_route_gradient'
    assert kernel_group('GLURecomputeDerivative')=='shared_recompute_activation'
    assert kernel_group('MXFP8LinearFwdKernel')=='dynamic_linear_gemm_quant'
    assert kernel_group('triton_tem_fused_mm_t_4')=='gemm'
    assert kernel_group('triton_tem_fused__to_copy_add_mm_mul_silu_slice_t_2')=='gemm'


def test_trace_duration_is_separate_and_concurrent_coverage_is_not_added():
    value=trace_summary(dict(traceEvents=[dict(cat='kernel',name='RoutedDYQuant',ts=0,dur=2),
        dict(cat='kernel',name='MXFP8GemmKernel',ts=1,dur=4)]))
    assert value['kernel_sum_us']==6 and value['span_us']==5 and value['coverage']==1
    assert value['short_kernel_count']==1
    assert value['groups']['quantization_preparation']['share']==1/3


def test_packed_capacity_bound_is_not_the_actual_active_prefix():
    from rtx._routing_varlen import workspace_bounds
    from math import ceil
    lengths=[128,512,360,1000]
    _,capacity=workspace_bounds(sum(lengths),len(lengths),32,4)
    actual=sum(32*ceil((ceil(n/8)*8*4//32)/32) for n in lengths)
    assert capacity==352 and actual==288 and sum(lengths)*4//32==250


def test_completed_evidence_cannot_silently_resume_a_different_workload():
    from benchmarks.profile_moe_amortization import compatible_result
    row=contexts()[0];budgets=dict(expert=12,router=8,shared=8,seconds=30.)
    record=dict(status='complete',source_digest='frozen',case=row['case'],lengths=row['lengths'],
        inductor_max_autotune=True,tuning_budgets=budgets)
    assert compatible_result(record,row,'frozen',True,budgets)
    assert not compatible_result(record,row,'changed',True,budgets)
    assert not compatible_result(record,row,'frozen',False,budgets)
    assert not compatible_result(record,{**row,'lengths':[128]},'frozen',True,budgets)
    assert not compatible_result(record,row,'frozen',True,{**budgets,'expert':48})


def test_compute_stage_attribution_requires_exact_isolated_plan_count():
    from benchmarks.analyze_moe_amortization import expert_stages
    record=dict(expert_config=dict(family='grouped_prepared',prepared=dict(forward_engine='packed',backward_preparation='joint')))
    names=['MXFP8PrequantGLUKernel','MXFP8GemmKernel','MXFP8GemmKernel','ExpertJointRecompute',
           'ExpertRouteGradient','MXFP8GemmKernel','MXFP8GemmKernel','MXFP8GemmKernel']
    events=[dict(cat='kernel',name=n,ts=i,dur=i+1) for i,n in enumerate(names)]
    result=expert_stages(record,dict(traceEvents=events))
    assert result['status']=='exact_count_plan_order'
    assert result['stages'][-1]['stage']=='bwd_gate_dw'
    assert expert_stages(record,dict(traceEvents=events[:-1]))['status']=='unmapped_count'
