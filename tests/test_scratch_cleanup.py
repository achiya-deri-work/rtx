from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def test_capture_scope_retains_autograd_worker_resources():
    from concurrent.futures import ThreadPoolExecutor
    from rtx.runtime import BoundedCache,retain_runtime_resources
    cache=BoundedCache(1)
    value=object()
    cache['entry']=value
    with retain_runtime_resources() as owners:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lambda:cache['entry']).result() is value
    cache.clear()
    assert owners[id(value)] is value


def test_global_cleanup_synchronizes_all_devices():
    from rtx.runtime import clear_runtime_caches
    calls=[]
    module=SimpleNamespace(_clear_runtime_caches=lambda **kwargs:kwargs)
    with patch('torch.cuda.is_available',return_value=True), \
         patch('torch.cuda.device_count',return_value=2), \
         patch('torch.cuda.synchronize',side_effect=calls.append), \
         patch('rtx.runtime.import_module',return_value=module), \
         patch('rtx._workspace.clear_scratch_pools'):
        result=clear_runtime_caches(device='cuda:0',preserve_compiled=True)
    assert calls==[0,1]
    assert all(v['preserve_compiled'] for v in result.values())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('preserve',[True,False])
def test_scratch_cleanup_preserves_executables_and_live_graph(tmp_path,preserve):
    import rtx
    from rtx.runtime import clear_runtime_caches,retain_runtime_resources
    from rtx.kernels.mxfp8_mlp import compile_mixed_down
    from rtx._mlp_autotune import SELECTIONS,BACKWARDS
    torch.manual_seed(435)
    layer=rtx.MXFP8MLP(128,256,128,device='cuda',autotune='cache',
                      autotune_cache_dir=tmp_path)
    x=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
    dy=torch.randn_like(x)*.1
    fn=torch.compile(layer,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def step():
        y=fn(x)
        return y,*torch.autograd.grad(y,(x,*layer.parameters()),dy)
    for _ in range(3): step()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): step()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with retain_runtime_resources() as owners:
        with torch.cuda.graph(graph,stream=stream):
            captured=step()
    before=compile_mixed_down.cache_info()
    selections=dict(SELECTIONS)
    direct_keys={k for k in BACKWARDS if not k[-2]}
    clear_runtime_caches(device=x.device,preserve_compiled=preserve)
    if preserve:
        assert compile_mixed_down.cache_info()==before
        assert dict(SELECTIONS)==selections
        assert direct_keys.issubset(set(BACKWARDS))
    else:
        assert compile_mixed_down.cache_info().currsize==0
        assert not SELECTIONS and not BACKWARDS
    for _ in range(3):
        with torch.no_grad():
            x.add_(.001)
            layer.gate_up_weight.add_(.0001)
        expected=step()
        graph.replay()
        torch.cuda.synchronize()
        for actual,ref in zip(captured,expected):
            torch.testing.assert_close(actual,ref,rtol=.001,atol=1e-6)
    if preserve:
        assert compile_mixed_down.cache_info().misses==before.misses
    assert owners
