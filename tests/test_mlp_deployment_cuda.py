"""Real execution surfaces: correctness, prepared reuse, and live memory."""
import pytest
import torch

from rtx.autotune.mlp import make_mlp_adapter


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('mode',['outer_capture','compiled_direct','selected_direct','private_replay'])
@pytest.mark.parametrize('phase',['forward','train'])
@pytest.mark.parametrize('shape',[(128,128,256,128),(17,35,97,41)])
def test_deployment_mode(mode,phase,shape,execution_mix=None,recompute='joint'):
    from dataclasses import replace
    import rtx
    torch.manual_seed(1729)
    m,k,h,o=shape
    values=tuple(torch.randn(*s,device='cuda',dtype=torch.bfloat16)*.1
                 for s in ((m,k),(2*h,k),(o,h)))
    adapter=make_mlp_adapter(*values,phase=phase,measurement_mode=mode,
                             warmup=2,samples=3,calls_per_sample=2,execution_mix=execution_mix)
    config=replace(adapter.initial_config,backward_recompute=recompute)
    assert adapter.rejection(config) is None
    first=adapter.evaluate(config)
    assert first.status=='ok',first.error
    allocations=[]
    for _ in range(4):
        result=adapter.evaluate(config)
        assert result.status=='ok',result.error
        assert result.compile_ms==0
        assert result.metadata['prepared_benchmark_reused']
        assert result.metadata['measurement_mode']==mode
        assert len(result.metadata['numerics'])==(4 if phase=='train' else 1)
        for metrics in result.metadata['numerics'].values():
            assert metrics['finite']
            assert metrics['relative_l2']<.025
        torch.cuda.synchronize()
        allocations.append(torch.cuda.memory_allocated())
    assert len(set(allocations))==1,allocations
    del adapter
    rtx.clear_runtime_caches()


@pytest.mark.parametrize('mode',['outer_capture','compiled_direct','selected_direct','private_replay'])
def test_repeated_forward_mix(mode):
    from rtx.autotune.execution_mix import MLPExecutionMix
    test_deployment_mode(mode,'train',(17,35,97,41),
                        MLPExecutionMix(training_pairs=1,grad_forwards=1,inference_forwards=2))


@pytest.mark.parametrize('mode',['outer_capture','compiled_direct','selected_direct','private_replay'])
def test_two_pass_lifetimes(mode):
    test_deployment_mode(mode,'train',(128,128,256,128),recompute='two_pass')


def test_selected_backward_avoids_nested_dispatch(monkeypatch):
    import weakref
    import rtx.mlp as mlp
    import rtx._mlp_autotune as selected
    calls = {'derivative': 0, 'paired': 0}
    dh_refs = []

    def forbidden(*args, **kwargs):
        raise AssertionError('nested custom-op dispatch inside selected backward')

    derivative = mlp._derivative_launch
    paired = selected._independent_bwd_launch

    def derivative_launch(*args, **kwargs):
        calls['derivative'] += 1
        dh_refs.append(weakref.ref(args[2]))
        return derivative(*args, **kwargs)

    def paired_launch(*args, **kwargs):
        calls['paired'] += 1
        assert dh_refs and dh_refs[-1]() is None, 'dH retained after derivative'
        return paired(*args, **kwargs)

    monkeypatch.setattr(mlp, '_derivative_op', forbidden)
    monkeypatch.setattr(selected, 'independent_bwd', forbidden)
    monkeypatch.setattr(mlp, '_derivative_launch', derivative_launch)
    monkeypatch.setattr(selected, '_independent_bwd_launch', paired_launch)
    test_deployment_mode('selected_direct', 'train', (128,128,256,128))
    assert all(count > 0 for count in calls.values()), calls


def test_separate_hidden_released_before_gate_backward(monkeypatch):
    from dataclasses import replace
    import weakref
    import rtx.mlp as mlp
    import rtx._mlp_autotune as selected
    refs=[]
    checked=[]
    derivative=mlp._separate_derivative_launch
    paired=selected._independent_bwd_launch
    def launch(*args,**kwargs):
        outputs=derivative(*args,**kwargs)
        refs.append(weakref.ref(outputs[1]))
        return outputs
    def gate(*args,**kwargs):
        assert refs and refs[-1]() is None, 'separate hidden retained through gate backward'
        checked.append(True)
        return paired(*args,**kwargs)
    monkeypatch.setattr(mlp,'_separate_derivative_launch',launch)
    monkeypatch.setattr(selected,'_independent_bwd_launch',gate)
    values=tuple(torch.randn(*s,device='cuda',dtype=torch.bfloat16)*.02
                 for s in ((128,128),(512,128),(128,256)))
    adapter=make_mlp_adapter(*values,phase='train',measurement_mode='selected_direct',
                             warmup=2,samples=3,calls_per_sample=2)
    cfg=replace(adapter.initial_config,recompute_backend='prequant',joint_output_storage='separate',
                backward_overlap=False)
    result=adapter.evaluate(cfg)
    assert result.status=='ok',result.error
    assert checked


@pytest.mark.parametrize('mode',['outer_capture','compiled_direct','selected_direct','private_replay'])
def test_candidates_reuse_evaluator_stream(mode):
    from dataclasses import replace
    import rtx
    torch.manual_seed(1729)
    values=tuple(torch.randn(*s,device='cuda',dtype=torch.bfloat16)*.02
                 for s in ((128,128),(256,128),(128,128)))
    adapter=make_mlp_adapter(*values,phase='train',measurement_mode=mode,
                             warmup=1,samples=3,calls_per_sample=1)
    initial=adapter.initial_config
    other=replace(initial,gate_up=replace(initial.gate_up,
                  glu_silu_math='shared_reciprocal'))
    assert adapter.rejection(other) is None
    for candidate in (initial,other,initial):
        result=adapter.evaluate(candidate)
        assert result.status=='ok',result.error
        assert not result.metadata['prepared_benchmark_reused']
        assert all(v['finite'] and v['relative_l2']<.025
                   for v in result.metadata['numerics'].values())
    del adapter
    rtx.clear_runtime_caches()
