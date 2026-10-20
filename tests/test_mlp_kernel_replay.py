"""Kernel-only CUDA replay: no nested compiler, stable ownership and inputs."""
from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch
import rtx

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape,activation,overlap", [
    ((17,35,97,41), "gelu", False),
    ((128,128,256,128), "silu", False),
    ((128,128,256,128), "silu", True),
])
def test_kernel_replay_without_internal_compile(tmp_path, shape, activation, overlap):
    from rtx._mlp_autotune import _selected_forward, _selected_backward, FORWARDS, BACKWARDS
    from rtx.autotune.mlp import reference
    torch.manual_seed(1962)
    m,k,h,o = shape
    layer = rtx.MXFP8MLP(k,h,o,device="cuda",activation=activation,
        config=replace(rtx.MXFP8MLPConfig(), backward_overlap=overlap),
        autotune="cache", autotune_cache_dir=tmp_path)
    x = torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.1
    # Deliberately noncontiguous upstream gradient.
    dy = (torch.randn(o,m,device="cuda",dtype=torch.bfloat16)*.1).t()
    values = x, layer.gate_up_weight, layer.down_weight
    metrics = []
    with patch("torch.compile", side_effect=AssertionError("internal compilation")), torch.no_grad():
        y1, token1 = _selected_forward(*values, activation, layer._first_hit_spec, True, replay=True)
        snapshot = y1.clone()
        newer = tuple(v.clone().add_(.001).requires_grad_(v.requires_grad) for v in values)
        y2, token2 = _selected_forward(*newer, activation, layer._first_hit_spec, True, replay=True)
        grads2 = _selected_backward(*newer, dy, token2, activation, True, True, True, replay=True)
        grads1 = _selected_backward(*values, dy, token1, activation, True, True, True, replay=True)
        torch.testing.assert_close(y1, snapshot, rtol=0, atol=0)
        for inputs, actual in ((values, (y1,*grads1)), (newer, (y2,*grads2))):
            expected = reference(*inputs, dy, activation)
            for name,a,b in zip(("y","dx","dgu","ddown"),actual,expected):
                assert torch.isfinite(a).all()
                delta = a.float()-b.float()
                rms = delta.square().mean().sqrt().item()
                ref = b.float().square().mean().sqrt().item()
                metrics.append(dict(output=name,max_abs=delta.abs().max().item(),
                    rms=rms,reference_rms=ref,normalized_rms=rms/max(ref,1e-12)))
                assert rms/max(ref,1e-12)<.025
        # Warm calls must not visit any Python kernel launcher.
        with patch("rtx.mlp._run_forward", side_effect=AssertionError("forward relaunched")), \
             patch("rtx.mlp._backward", side_effect=AssertionError("backward relaunched")), \
             patch("torch.cuda.CUDAGraph", side_effect=AssertionError("warm call recaptured")), \
             patch("rtx.fp8_bwd._launch_bwd_out", side_effect=AssertionError("backward relaunched")):
            for _ in range(3):
                y, token = _selected_forward(*values, activation, layer._first_hit_spec, True, replay=True)
                grads = _selected_backward(*values, dy, token, activation, True, True, True, replay=True)
                torch.testing.assert_close(y, y1, rtol=0, atol=0)
                for a,b in zip(grads,grads1):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
    print("KERNEL_REPLAY", json.dumps(dict(shape=shape,activation=activation,overlap=overlap,metrics=metrics)))


def test_graph_owns_evicted_scratch_and_releases_after_clear(tmp_path):
    from rtx._mlp_autotune import _selected_forward, _selected_backward, FORWARDS, BACKWARDS
    from rtx import fp8_bwd
    layer=rtx.MXFP8MLP(128,256,128,device="cuda",autotune="cache",autotune_cache_dir=tmp_path)
    x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16)*.1
    dy=torch.randn_like(x)*.1
    values=x,layer.gate_up_weight,layer.down_weight
    with torch.no_grad():
        _,token=_selected_forward(*values,"silu",layer._first_hit_spec,True,replay=True)
        expected=_selected_backward(*values,dy,token,"silu",True,True,True,replay=True)
        # Remove lower-level owners without removing the replay itself.
        torch.cuda.synchronize()
        fp8_bwd._clear_runtime_caches()
        for _ in range(3):
            actual=_selected_backward(*values,dy,token,"silu",True,True,True,replay=True)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        live=[]
        del actual
        for _ in range(4):
            for _ in range(5):
                result=_selected_backward(*values,dy,token,"silu",True,True,True,replay=True)
                del result
            torch.cuda.synchronize()
            live.append(torch.cuda.memory_allocated())
        assert len(set(live))==1,live
    rtx.clear_runtime_caches()
    assert len(FORWARDS)==len(BACKWARDS)==0


def test_cuda_capture_forbids_cold_private_graph():
    from rtx._graph_replay import KernelReplay
    x=torch.ones(8,device="cuda")
    with patch("torch.cuda.is_current_stream_capturing",return_value=True):
        with pytest.raises(RuntimeError,match="warm kernel replay"):
            KernelReplay(lambda v:(v.clone(),),(x,))
