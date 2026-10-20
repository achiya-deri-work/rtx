import pytest
from rtx.autotune.replacement import ReplacementPolicy, recommend_replacement
from rtx.autotune.replacement import comparison_orders, summarize_repeated_comparison


def report(name,ms,peak,qualified=True):
    return dict(backend=name,status='ok',environment_qualified=qualified,contract={'mix':[1,2,3]},
                timings={'workload':{'median_ms':ms}},memory={'warm_peak_allocated':peak})


def test_speed_default_rejects_slower_memory_saving():
    result=recommend_replacement([report('bf16',1,100),report('mxfp8',1.4,50)])
    assert result['selected']=='bf16'


def test_explicit_bounded_memory_tradeoff():
    rows=[report('bf16',1,100),report('mxfp8',1.4,50)]
    assert recommend_replacement(rows,ReplacementPolicy(True,.2))['selected']=='bf16'
    result=recommend_replacement(rows,ReplacementPolicy(True,.5))
    assert result['selected']=='mxfp8' and result['slower_memory_tradeoff']
    assert result['requires_explicit_application_acceptance']


def test_faster_recomputed_bf16_can_win():
    result=recommend_replacement([report('bf16',1,100),report('bf16_recompute',.8,70),report('mxfp8',.9,50)])
    assert result['selected']=='bf16_recompute'


def test_unqualified_reports_cannot_promote():
    result=recommend_replacement([report('bf16',1,100),report('mxfp8',.1,50,False)])
    assert result['status']=='unqualified' and result['selected'] is None
    assert result['comparisons']['mxfp8']['time_ratio']==pytest.approx(.1)
    assert result['comparisons']['mxfp8']['peak_saved_bytes']==50
    assert result['requires_explicit_application_acceptance']
    assert not result['policy']['prefer_memory']


def test_recommendation_records_explicit_policy_and_negative_savings():
    result=recommend_replacement([report('bf16',1,100),report('mxfp8',1.4,150)],
                                  ReplacementPolicy(True,.5))
    assert result['selected']=='bf16'
    assert result['policy']['max_slowdown']==.5
    assert result['comparisons']['mxfp8']['peak_saved_bytes']==-50


def test_contract_mismatch():
    a,b=report('bf16',1,100),report('mxfp8',.5,50)
    b['contract']={}
    with pytest.raises(ValueError,match='contracts'): recommend_replacement([a,b])


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_timings(value):
    with pytest.raises(ValueError): recommend_replacement([report('bf16',value,100)])


def test_slower_requires_opt_in():
    with pytest.raises(ValueError): ReplacementPolicy(max_slowdown=.2)


def test_duplicate_and_missing_baselines():
    with pytest.raises(ValueError): recommend_replacement([report('mxfp8',1,100)])
    with pytest.raises(ValueError): recommend_replacement([report('bf16',1,100)]*2)


def test_balanced_fresh_process_summary_is_not_qualification():
    rounds=[[report(name, {'bf16':1,'bf16_recompute':1.2,'mxfp8':.8}[name]*(i+1),100)
             for name in order] for i,order in enumerate(comparison_orders(6))]
    result=summarize_repeated_comparison(rounds)
    assert result['paired_workload_ratios']['mxfp8']['median']==pytest.approx(.8)
    assert result['decision']['selected'] is None
    assert result['decision']['status']=='unqualified'
    # Drift affects absolute medians; same-round ratios preserve pairing.
    assert result['reports'][0]['timings']['workload']['median_ms']==3.5
    opted=summarize_repeated_comparison(rounds,policy=ReplacementPolicy(True,.25,.02))
    assert opted['decision']['selected'] is None
    assert opted['decision']['policy']==dict(prefer_memory=True,max_slowdown=.25,minimum_speedup=.02)


@pytest.mark.parametrize('flags',[
    ['--max-slowdown','.2'],['--prefer-memory','--max-slowdown','nan'],
    ['--minimum-speedup','1'],['--minimum-speedup','-1']])
def test_cli_invalid_policy_fails_before_creating_output(tmp_path,monkeypatch,flags):
    from rtx.autotune.replacement_benchmark import main
    out=tmp_path/'comparison'
    monkeypatch.setattr('sys.argv',['benchmark','--output',str(out),*flags])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    assert not out.exists()


def test_repeat_summary_rejects_unbalanced_or_changed_sources():
    with pytest.raises(ValueError): comparison_orders(4)
    rounds=[[report(name,1,100) for name in order] for order in comparison_orders(6)]
    rounds[1]=rounds[0]
    with pytest.raises(ValueError,match='order'): summarize_repeated_comparison(rounds)
    rounds=[[report(name,1,100) for name in order] for order in comparison_orders(6)]
    rounds[3][0]['source_hashes']={'kernel':'changed'}
    with pytest.raises(ValueError,match='source/config'): summarize_repeated_comparison(rounds)


def test_source_manifest_includes_kernel_and_ignores_generated_files(tmp_path):
    from rtx.autotune.replacement_benchmark import source_manifest
    (tmp_path/'kernels').mkdir()
    kernel=tmp_path/'kernels'/'matmul.py'
    kernel.write_text('value = 1\n')
    (tmp_path/'runtime.py').write_text('cache = {}\n')
    (tmp_path/'result.json').write_text('{}')
    before=source_manifest(tmp_path)
    assert set(before)=={'kernels/matmul.py','runtime.py'}
    kernel.write_text('value = 2\n')
    after=source_manifest(tmp_path)
    assert before['kernels/matmul.py']!=after['kernels/matmul.py']
    assert before['runtime.py']==after['runtime.py']


def test_summary_rejects_consistently_different_sources_between_backends():
    rounds=[[report(name,1,100) for name in order] for order in comparison_orders(6)]
    for batch in rounds:
        for r in batch:
            r['source_hashes']={'kernel.py':r['backend']}
    with pytest.raises(ValueError,match='source/config'):
        summarize_repeated_comparison(rounds)
