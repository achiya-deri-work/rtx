from contextlib import contextmanager

import pytest

from rtx.autotune.execution_mix import MLPExecutionMix
from rtx.autotune.core import KernelContext


@pytest.mark.parametrize('values',[
    dict(training_pairs=-1),dict(training_pairs=True),dict(training_pairs=1.5),
    dict(training_pairs=0),dict(inference_forwards=-1),dict(grad_forwards='2'),
])
def test_invalid_counts(values):
    with pytest.raises(ValueError): MLPExecutionMix(**values)


def test_phase_contract():
    MLPExecutionMix().validate_phase('train')
    MLPExecutionMix(training_pairs=0,inference_forwards=2).validate_phase('forward')
    with pytest.raises(ValueError): MLPExecutionMix().validate_phase('forward')
    with pytest.raises(ValueError): MLPExecutionMix(training_pairs=0,grad_forwards=1).validate_phase('train')


def test_exact_call_mix_and_returned_gradients():
    events=[]
    @contextmanager
    def grad(enabled):
        events.append(('grad',enabled))
        yield
    def forward(x):
        events.append(('forward',x))
        return x*2
    def backward(y):
        events.append(('backward',y))
        return (3,4,5)
    mix=MLPExecutionMix(training_pairs=2,grad_forwards=3,inference_forwards=4)
    assert mix.execute(forward,(7,),backward,grad)==(14,3,4,5)
    assert events.count(('forward',7))==9
    assert events.count(('backward',14))==2
    assert events.count(('grad',False))==4
    assert events.count(('grad',True))==5


def test_inference_result():
    from contextlib import nullcontext
    mix=MLPExecutionMix(training_pairs=0,inference_forwards=3)
    assert mix.execute(lambda x:x+1,(2,),None,lambda _:nullcontext())==(3,)


def test_mix_changes_context_identity():
    def context(mix):
        return KernelContext('mlp',1,{},tags={'execution_mix':mix.contract()}).identifier
    assert context(MLPExecutionMix())!=context(MLPExecutionMix(inference_forwards=2))
    assert context(MLPExecutionMix(grad_forwards=2))!=context(MLPExecutionMix(inference_forwards=2))


@pytest.mark.parametrize('bad',[{},False,0,''])
def test_bad_mix_rejected_before_cuda(bad):
    from rtx.autotune.mlp import make_mlp_adapter
    with pytest.raises(TypeError,match='execution_mix'):
        make_mlp_adapter(None,None,None,execution_mix=bad)


def test_runtime_mix_serialization_and_legacy_keys():
    import json
    import rtx
    from rtx._mlp_autotune import deferred_spec,_runtime_variant
    from rtx.autotune.core import stable_id
    c=rtx.MXFP8MLPConfig()
    legacy=json.loads(deferred_spec(c,'mxfp8','cache',None,None))
    assert 'execution_mixes' not in legacy
    mix=rtx.MLPExecutionMix(grad_forwards=2,inference_forwards=3)
    layer=rtx.MXFP8MLP(32,64,autotune='cache',training_mix=mix,
                      inference_mix=rtx.MLPExecutionMix(training_pairs=0,inference_forwards=4))
    spec=json.loads(layer._first_hit_spec)
    assert spec['execution_mixes']['train']['grad_forwards']==2
    assert spec['execution_mixes']['forward']['inference_forwards']==4
    args=(32,'silu',True,c.to_dict(),False)
    expected=stable_id(dict(out_features=32,activation='silu',training=True,
        initial=c.to_dict(),frontend_revision=2,measurement_mode='selected_direct'))
    assert _runtime_variant(*args)==expected
    assert _runtime_variant(*args,mix)!=expected
    assert _runtime_variant(*args,mix)!=_runtime_variant(*args,rtx.MLPExecutionMix())


def test_runtime_mixes_do_not_silently_change_phase_or_off_mode():
    import rtx
    with pytest.raises(ValueError,match='require autotune'):
        rtx.MXFP8MLP(32,64,training_mix=rtx.MLPExecutionMix())
    with pytest.raises(ValueError,match='inference-only'):
        rtx.MXFP8MLP(32,64,autotune='cache',inference_mix=rtx.MLPExecutionMix())
    with pytest.raises(TypeError,match='MLPExecutionMix'):
        rtx.MXFP8MLP(32,64,autotune='cache',training_mix={})
