"""Host ABI evidence; GPU MLP integration is qualified separately."""

import gc
import io
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rtx._execution_plan_token import PLAN_TOKEN_BYTES, read_plan_token, write_plan_token


def token(key):
    result = torch.empty(PLAN_TOKEN_BYTES, dtype=torch.uint8, device="cpu")
    write_plan_token(result, key)
    return result


def test_actual_mlp_config_alias_clone_and_serialization():
    from rtx.mlp import DEFAULT_MLP_CONFIG, MXFP8MLPConfig
    key = DEFAULT_MLP_CONFIG.to_json()
    original = token(key)
    alias = original.view(-1)
    clone = original.clone()
    del original
    gc.collect()
    stream = io.BytesIO()
    torch.save(alias, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    for saved in (alias, clone, restored):
        assert read_plan_token(saved) == key
        assert MXFP8MLPConfig.from_json(read_plan_token(saved)) == DEFAULT_MLP_CONFIG
    assert torch.equal(alias, token(key))


def test_validation_and_corruption():
    for bad in (torch.empty(1), torch.empty(PLAN_TOKEN_BYTES, dtype=torch.int64),
                torch.empty(PLAN_TOKEN_BYTES * 2, dtype=torch.uint8)[::2]):
        with pytest.raises(ValueError):
            write_plan_token(bad, "config")
    saved = token("config")
    saved[16] ^= 1
    with pytest.raises(ValueError, match="corrupted"):
        read_plan_token(saved)
    with pytest.raises(ValueError, match="header"):
        read_plan_token(torch.zeros(PLAN_TOKEN_BYTES, dtype=torch.uint8))
    with pytest.raises(ValueError, match="capacity"):
        token("x" * PLAN_TOKEN_BYTES)
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        with pytest.raises(TypeError, match="real"):
            read_plan_token(torch.empty(PLAN_TOKEN_BYTES, dtype=torch.uint8))


def test_independent_concurrent_plans():
    keys = [f'{{"revision": 1, "selected": {i}}}' for i in range(64)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        saved = list(pool.map(token, keys))
    assert [read_plan_token(value) for value in saved] == keys


_selected_factor = 2


@torch.library.custom_op("rtx_plan_contract_test::forward_v2", mutates_args=(), device_types=("cpu", "cuda"))
def _forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x * _selected_factor, token(str(_selected_factor))


@_forward.register_fake
def _fake(x):
    return torch.empty_like(x), torch.empty(PLAN_TOKEN_BYTES, dtype=torch.uint8, device="cpu")


@torch.library.custom_op("rtx_plan_contract_test::backward", mutates_args=(), device_types=("cpu", "cuda"))
def _grad(dy: torch.Tensor, saved: torch.Tensor) -> torch.Tensor:
    return dy * int(read_plan_token(saved))


@_grad.register_fake
def _grad_fake(dy, saved):
    return torch.empty_like(dy)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(output[1])
    ctx.mark_non_differentiable(output[1])


def _backward(ctx, dy, unused):
    return _grad(dy, ctx.saved_tensors[0])


torch.library.register_autograd(_forward, _backward, setup_context=_setup)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compiled_outstanding_forwards_pin_distinct_plans(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    global _selected_factor
    compilations = []
    from torch._inductor.compile_fx import compile_fx
    def backend(gm, inputs):
        compilations.append(gm)
        return compile_fx(gm, inputs)
    fn = torch.compile(lambda x: _forward(x)[0], backend=backend, fullgraph=True, dynamic=False)
    try:
        _selected_factor = 2
        x = torch.ones(16, device=device, requires_grad=True)
        y = fn(x)
        _selected_factor = 3
        other = torch.ones(16, device=device, requires_grad=True)
        z = fn(other)
        # A later runtime winner must not change an outstanding backward.
        _selected_factor = 7
        z.sum().backward()
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))
        torch.testing.assert_close(other.grad, torch.full_like(other, 3))
        assert len(compilations) == 1
        torch.library.opcheck(_forward, (torch.ones(16, device=device, requires_grad=True),))
        if device == "cuda":
            _selected_factor = 2
            stream = torch.cuda.Stream()
            static = torch.ones(16, device=device, requires_grad=True)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    fn(static).sum().backward()
                    static.grad.zero_()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fn(static).sum().backward()
                # Replay uses the captured plan, not subsequent Python state.
                _selected_factor = 11
                for _ in range(3):
                    static.grad.zero_()
                    graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(static.grad, torch.full_like(static, 2))
            assert len(compilations) == 1
    finally:
        _selected_factor = 2
