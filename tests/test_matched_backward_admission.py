from types import SimpleNamespace

import pytest
import torch

from tools.research.matched_gpu import independent_initial,independent_metrics


def test_prepare_rejects_incompatible_prior_before_kernel_admission(monkeypatch):
    from tools.research import matched_gpu
    from rtx import _portable_predictor
    monkeypatch.setattr(matched_gpu,'build_adapter',lambda *args:(object(),{}))
    monkeypatch.setattr(_portable_predictor,'load_predictor',
        lambda *args,**kwargs:SimpleNamespace(bundle=lambda adapter:None))
    def forbidden(*args):raise AssertionError('incompatible prior compiled kernels')
    monkeypatch.setattr(matched_gpu,'independent_initial',forbidden)
    args=SimpleNamespace(family='mxfp8_bwd',m=128,n=128,k=128,seed=1,
                         device='cuda:0',predictor='unused')
    with pytest.raises(ValueError,match='unsupported prior contract'):
        matched_gpu.prepare(args)


def test_independent_metrics_reject_nonfinite_and_shape():
    zero=torch.zeros(4)
    assert independent_metrics(zero,zero,.07)['normalized_rms']==0
    for actual,reference in ((torch.full((4,),float('nan')),zero),
                             (zero,torch.full((4,),float('inf')))):
        with pytest.raises(RuntimeError):independent_metrics(actual,reference,.07)
    with pytest.raises(RuntimeError,match='shape'):
        independent_metrics(zero,torch.zeros(1),.07)


@pytest.mark.parametrize('broken',[None,'dX','dW'])
def test_backward_admission_checks_each_gradient(monkeypatch,broken):
    from rtx import fp8_bwd
    def build(*args):
        def run(dy,x,w,dx,dw):
            dx.copy_((dy.float()@w.float()).bfloat16())
            dw.copy_((dy.float().T@x.float()).bfloat16())
            if broken=='dX':dx.zero_()
            if broken=='dW':dw.zero_()
        return run
    monkeypatch.setattr(fp8_bwd,'_build_bwd_runner',build)
    task=dict(family='mxfp8_bwd',shape=dict(m=8,n=16,k=12),seed=952)
    adapter=SimpleNamespace(initial_config=None)
    if broken:
        with pytest.raises(RuntimeError,match=broken):independent_initial(adapter,task,'cpu')
    else:
        metrics=independent_initial(adapter,task,'cpu')
        assert set(metrics['outputs'])=={'dX','dW'}
        assert metrics['normalized_rms']==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape',[(128,128,128),(17,97,35),(4096,256,128)])
def test_backward_admission_cuda(shape):
    from rtx.fp8_bwd import _default_bwd_config
    from rtx.kernels.mxfp8 import MXFP8Problem
    m,n,k=shape
    task=dict(family='mxfp8_bwd',shape=dict(m=m,n=n,k=k),seed=952)
    adapter=SimpleNamespace(initial_config=_default_bwd_config(MXFP8Problem(m,n,k)))
    metrics=independent_initial(adapter,task,'cuda:0')
    assert metrics['finite'] and metrics['normalized_rms']<.07
    assert all(v['reference_finite'] for v in metrics['outputs'].values())
    print(shape,metrics,flush=True)
