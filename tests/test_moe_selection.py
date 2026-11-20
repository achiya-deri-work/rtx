"""CPU proof of real-input cold selection to compiler-visible fixed dispatch."""
import pytest
import torch

from rtx import _moe_selection as selection


_COLD_CALLS = []


@torch.library.custom_op('rtx_selection_test::cold', mutates_args=(), device_types='cpu')
def _cold(x: torch.Tensor, policy: str) -> torch.Tensor:
    shape = tuple(x.shape)
    assert selection.bound_config(shape, x.device, policy, False) is None
    selection.bind_selected(shape, x.device, policy, False, 'fixed:' + str(shape))
    _COLD_CALLS.append(shape)
    return x * 2


@_cold.register_fake
def _cold_fake(x, policy):
    return torch.empty_like(x)


def _dispatch(x, policy):
    key = selection.bound_config(tuple(x.shape), x.device, policy, False)
    if key is None:
        return _cold(x, policy)
    assert key == 'fixed:' + str(tuple(x.shape))
    return x * 2


@torch.library.custom_op('rtx_selection_test::cold_training', mutates_args=(), device_types='cpu')
def _cold_training(x: torch.Tensor, policy: str) -> tuple[torch.Tensor, torch.Tensor]:
    selection.bind_selected(tuple(x.shape), x.device, policy, True, 'scale:2')
    return x * 2, torch.tensor([2], dtype=torch.int64)


@_cold_training.register_fake
def _cold_training_fake(x, policy):
    return torch.empty_like(x), torch.empty(1, dtype=torch.int64, device='cpu')


@torch.library.custom_op('rtx_selection_test::cold_backward', mutates_args=(), device_types='cpu')
def _cold_backward(dy: torch.Tensor, saved: torch.Tensor) -> torch.Tensor:
    return dy * int(saved[0])


@_cold_backward.register_fake
def _cold_backward_fake(dy, saved):
    return torch.empty_like(dy)


def _training_setup(ctx, inputs, output):
    ctx.save_for_backward(output[1])
    ctx.mark_non_differentiable(output[1])


def _training_backward(ctx, dy, _unused):
    return _cold_backward(dy, ctx.saved_tensors[0]), None


_cold_training.register_autograd(_training_backward, setup_context=_training_setup)


class _Fixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, key):
        ctx.scale = int(key.rsplit(':', 1)[1])
        return x * ctx.scale

    @staticmethod
    def backward(ctx, dy):
        return dy * ctx.scale, None


def _training_dispatch(x):
    key = selection.bound_config(tuple(x.shape), x.device, 'training', True)
    if key is None:
        return _cold_training(x, 'training')[0]
    return _Fixed.apply(x, key)


@pytest.fixture(autouse=True)
def _reset():
    selection.clear_selection_cells()
    _COLD_CALLS.clear()
    torch.compiler.reset()
    yield
    selection.clear_selection_cells()
    _COLD_CALLS.clear()
    torch.compiler.reset()


def test_namespace_clear_preserves_unrelated_bindings():
    selection.bind_selected((4,8),'cuda:0','routing:policy',True,'router')
    selection.bind_selected((4,8),'cuda:0','expert-policy',True,'expert')
    assert selection.clear_selection_cells(namespace='routing:')==1
    assert selection.bound_config((4,8),'cuda:0','expert-policy',True)=='expert'
    assert selection.bound_config((4,8),'cuda:0','routing:policy',True) is None
    with pytest.raises(TypeError):selection.clear_selection_cells(namespace=1)


def test_compiled_cell_transitions_once_and_other_shapes_do_not_invalidate():
    compiled = []

    def backend(graph, _inputs):
        compiled.append(graph)
        return graph.forward

    fn = torch.compile(lambda x: _dispatch(x, 'two-shapes'), fullgraph=True,
                       dynamic=False, backend=backend)
    counts = []
    for size in (4, 4, 4, 8, 8, 8, 4, 8):
        x = torch.randn(size)
        torch.testing.assert_close(fn(x), x * 2)
        counts.append(len(compiled))
    assert counts == [1, 2, 2, 3, 4, 4, 4, 4]
    assert _COLD_CALLS == [(4,), (8,)]
    assert 'rtx_selection_test.cold' in compiled[0].code
    assert 'rtx_selection_test.cold' not in compiled[1].code


def test_multiple_policy_bindings_in_one_trace_do_not_read_a_stale_dictionary_snapshot():
    frames=[]
    def backend(graph,_inputs):frames.append(graph);return graph.forward
    compiled=torch.compile(lambda x:_dispatch(x,'first')+_dispatch(x,'second'),
                           fullgraph=True,dynamic=False,backend=backend)
    x=torch.randn(4)
    torch.testing.assert_close(compiled(x),x*4)
    torch.testing.assert_close(compiled(x),x*4)
    torch.testing.assert_close(compiled(x),x*4)
    assert len(frames)==2
    assert _COLD_CALLS==[(4,),(4,)]
    assert 'rtx_selection_test.cold' in frames[0].code
    assert 'rtx_selection_test.cold' not in frames[1].code


def test_bound_plan_is_immutable_and_saved_metadata_survives_eviction(monkeypatch):
    monkeypatch.setattr(selection, '_MAX_CELLS', 2)
    context = ((4,), 'cpu', 'policy', True)
    selection.bind_selected(*context, 'original')
    saved_by_forward = selection.bound_config(*context)
    selection.bind_selected(*context, 'refined')
    assert selection.bound_config(*context) == 'original'
    selection.bind_selected((8,), 'cpu', 'policy', True, 'eight')
    selection.bind_selected((16,), 'cpu', 'policy', True, 'sixteen')
    assert len(selection._CELLS) == 2
    assert saved_by_forward == 'original'
    assert selection.bound_config(*context) is None
    selection.bind_selected(*context, 'after-eviction')
    assert selection.bound_config(*context) == 'after-eviction'
    assert saved_by_forward == 'original'
    assert len(selection._CELLS) == len(selection._RECENCY) == 2


def test_compiled_eviction_is_a_safe_new_cold_transition(monkeypatch):
    monkeypatch.setattr(selection, '_MAX_CELLS', 2)
    fn = torch.compile(lambda x: _dispatch(x, 'eviction'), fullgraph=True,
                       dynamic=False, backend='eager')
    for size in (4, 4, 8, 8, 16, 16, 4, 4):
        x = torch.randn(size)
        torch.testing.assert_close(fn(x), x * 2)
        assert len(selection._CELLS) <= 2
    assert _COLD_CALLS == [(4,), (8,), (16,), (4,)]


def test_policy_training_device_and_shape_are_distinct_and_values_are_metadata_only():
    selection.bind_selected((4,), 'cuda:0', 'policy', True, 'chosen')
    assert selection.bound_config((4,), 'cuda:0', 'policy', True) == 'chosen'
    for context in (((8,), 'cuda:0', 'policy', True),
                    ((4,), 'cuda:1', 'policy', True),
                    ((4,), 'cuda:0', 'other', True),
                    ((4,), 'cuda:0', 'policy', False)):
        assert selection.bound_config(*context) is None
    with pytest.raises(TypeError):
        selection.bind_selected((4,), 'cpu', 'policy', True, torch.empty(1))
    with pytest.raises(ValueError):
        selection.bound_config((0,), 'cpu', 'policy', True)
    assert all(cell.config_key is None or isinstance(cell.config_key, str)
               for cell in selection._CELLS.values())
    assert selection.clear_selection_cells() == 5
    assert not selection._CELLS and not selection._RECENCY


def test_aot_cold_and_hot_outstanding_backwards_keep_their_original_policy():
    fn = torch.compile(_training_dispatch, backend='aot_eager', fullgraph=True, dynamic=False)
    cold_x = torch.randn(4, requires_grad=True)
    cold_y = fn(cold_x)
    hot_x = torch.randn(4, requires_grad=True)
    hot_y = fn(hot_x)
    selection.clear_selection_cells()
    selection.bind_selected((4,), 'cpu', 'training', True, 'scale:3')
    refined_x = torch.randn(4, requires_grad=True)
    refined_y = fn(refined_x)
    for x, y, scale in ((cold_x, cold_y, 2), (hot_x, hot_y, 2), (refined_x, refined_y, 3)):
        torch.testing.assert_close(y, x * scale)
        gradient, = torch.autograd.grad(y.sum(), (x,))
        torch.testing.assert_close(gradient, torch.full_like(x, scale))


def test_explicit_publication_recompiles_only_that_hot_shape_without_cold_selection():
    from torch._dynamo.testing import CompileCounterWithBackend
    counter = CompileCounterWithBackend('aot_eager')
    fn = torch.compile(_training_dispatch, backend=counter, fullgraph=True, dynamic=False)
    cold_x = torch.randn(4, requires_grad=True);cold_y = fn(cold_x)
    hot_x = torch.randn(4, requires_grad=True);hot_y = fn(hot_x)
    assert counter.frame_count == 2
    context = ((4,), 'cpu', 'training', True)
    old_cell = selection._CELLS[selection._identity(*context)]
    selection.publish_selected(*context, 'scale:3')
    assert old_cell.config_key == 'scale:2'
    assert selection._CELLS[selection._identity(*context)] is not old_cell
    refined_x = torch.randn(4, requires_grad=True);refined_y = fn(refined_x)
    assert counter.frame_count == 3
    # Publishing another context must not invalidate this shape's hot graph.
    selection.publish_selected((8,), 'cpu', 'training', True, 'scale:4')
    same_metadata = torch.ones(4, requires_grad=True)
    torch.testing.assert_close(fn(same_metadata), same_metadata * 3)
    assert counter.frame_count == 3
    for x, y, scale in ((cold_x, cold_y, 2), (hot_x, hot_y, 2), (refined_x, refined_y, 3)):
        gradient, = torch.autograd.grad(y.sum(), (x,))
        torch.testing.assert_close(gradient, torch.full_like(x, scale))
    assert 'cold_training' in counter.graphs[0].code
    assert all('cold_training' not in graph.code for graph in counter.graphs[1:])
