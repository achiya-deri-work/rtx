"""Selected backward must not retain recompute storage through output attributes."""
import weakref
from unittest.mock import patch
import pytest
import torch
import rtx

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_shape_only_dx_reuses_only_explicit_placeholder():
    from rtx._mlp_autotune import independent_dx
    import rtx.fp8_bwd as b
    from rtx.kernels.mxfp8_bwd import DEFAULT_MXFP8_BWD_CONFIG
    key=b._intern_bwd_config(DEFAULT_MXFP8_BWD_CONFIG)
    dy=torch.randn(128,256,device='cuda',dtype=torch.bfloat16)*.1
    weight=torch.randn(256,128,device='cuda',dtype=torch.bfloat16)*.02
    placeholder=torch.empty(128,128,device='cuda',dtype=torch.bfloat16)
    expected=independent_dx(dy,placeholder,weight,key)
    assert expected.data_ptr()!=placeholder.data_ptr()
    actual=independent_dx(dy,placeholder,weight,key,shape_only=True)
    assert actual is placeholder
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    with pytest.raises(RuntimeError,match='fixed configuration'):
        independent_dx(dy,placeholder,weight,'unknown',shape_only=True)


@pytest.mark.parametrize('m', [2048, 20480])
def test_joint_placeholder_live_memory_delta(tmp_path, record_property, m):
    from dataclasses import replace
    import rtx._mlp_autotune as selected
    import rtx.mlp as mlp
    k,h,o=256,768,256
    config=replace(rtx.MXFP8MLPConfig(),backward_recompute='joint',backward_overlap=False)
    layer=rtx.MXFP8MLP(k,h,o,config=config,device='cuda',autotune='cache',autotune_cache_dir=tmp_path)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    values=x,layer.gate_up_weight,layer.down_weight
    with torch.no_grad():
        _,token=selected._selected_forward(*values,'silu',layer._first_hit_spec,True)
        original_dx=selected.independent_dx
        original_derivative=mlp._run_derivative
        def run(reuse):
            entry=[]
            dx_exit=[]
            def dx(*args,shape_only=False):
                result=original_dx(*args,shape_only=shape_only and reuse)
                # Measure while the caller still owns the placeholder. The
                # production backward now releases it before the derivative,
                # so derivative-entry memory is no longer a reuse oracle.
                dx_exit.append(torch.cuda.memory_allocated())
                return result
            def derivative(*args,**kwargs):
                entry.append(torch.cuda.memory_allocated())
                return original_derivative(*args,**kwargs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with patch.object(selected,'independent_dx',dx), patch.object(mlp,'_run_derivative',derivative):
                grads=selected._selected_backward(*values,dy,token,'silu',True,True,True)
            torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated()
            return entry,dx_exit,peak,tuple(g.cpu() for g in grads)
        run(False);run(True)
        before,dx_before,peak_before,expected=run(False)
        after,dx_after,peak_after,actual=run(True)
    assert len(before)==len(after)==1
    assert len(dx_before)==len(dx_after)==1
    assert dx_before[0]-dx_after[0]==m*h*2
    assert before[0]==after[0]  # non-reused placeholder is already released
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    record_property('dx_exit_bytes_saved',dx_before[0]-dx_after[0])
    record_property('backward_peak_before',peak_before)
    record_property('backward_peak_after',peak_after)

@pytest.mark.parametrize('mask',[(True,True,True),(True,False,True),(False,True,True),(False,False,True)])
def test_selected_backward_releases_temporaries(tmp_path,mask):
    from rtx._mlp_autotune import _selected_forward,_selected_backward
    import rtx.mlp as mlp
    torch.manual_seed(72)
    layer=rtx.MXFP8MLP(128,256,128,device='cuda',autotune='cache',autotune_cache_dir=tmp_path)
    x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1
    dy=torch.randn_like(x).t()
    values=x,layer.gate_up_weight,layer.down_weight
    refs=[];original=mlp._run_derivative
    def track(x,w,dh,activation,policy,out):
        refs.append(weakref.ref(out))
        return original(x,w,dh,activation,policy,out)
    with torch.no_grad():
        _,token=_selected_forward(*values,'silu',layer._first_hit_spec,True)
        with patch.object(mlp,'_run_derivative',track):
            grads=_selected_backward(*values,dy,token,'silu',*mask)
        assert all(ref() is None for ref in refs)
        assert all(g is None or not hasattr(g,'_base_inputs') for g in grads)
        for need,g in zip(mask,grads):
            assert (g is not None)==need
            if need: assert torch.isfinite(g).all()
