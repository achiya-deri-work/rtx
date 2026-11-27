from types import SimpleNamespace
from inspect import unwrap
from unittest.mock import patch

import pytest

from rtx._managed import _guard_backward_generation


@pytest.mark.parametrize(
    "source,device,is_backward,bypass",
    [
        ("torch.ops.rtx.mxfp8_routed_experts_backward.default", None, True, True),
        ("torch.ops.rtx_routing.metadata_execute.default", None, True, True),
        ("torch.ops.rtx.mxfp8_routed_experts_backward.default", 0, True, False),
        ("torch.ops.rtx.mxfp8_routed_experts_backward.default", None, False, False),
        ("torch.ops.other.backward.default", None, True, False),
    ],
)
def test_tree_handoff_guard(source, device, is_backward, bypass):
    target = object()
    graph = SimpleNamespace(source_code=source, current_callable=target,
                            fx_kwargs={"is_backward": is_backward})
    box = SimpleNamespace(value=device)
    seen = []
    notify = _guard_backward_generation(lambda *args: seen.append(args))
    notify(graph, box)
    assert seen == ([] if bypass else [(graph, box)])
    assert graph.current_callable is target


def test_inductor_skipped_partition_calls_guard():
    from torch._inductor import config, output_code

    original = output_code.maybe_handle_backward_generation
    # Invoke the actual Inductor skip path with the exact state observed in
    # the G2DETR traceback; do not depend on building a particular FX split.
    graph = SimpleNamespace(
        source_code="torch.ops.rtx.mxfp8_routed_experts_backward.default",
        current_callable=lambda inputs: inputs, fx_kwargs={"is_backward": True},
        cudagraph_info=SimpleNamespace(cudagraph_fail_reasons=[]),
        partition_maps=[],
    )
    with config.patch({"triton.cudagraph_trees": True}):
        with patch.object(output_code, "maybe_handle_backward_generation", unwrap(original)):
            with pytest.raises(AssertionError):
                output_code.cudagraph_partition_post_compile(
                    [], graph, output_code.BoxedBool(True), {},
                    output_code.BoxedDeviceIndex(None))
        with patch.object(output_code, "maybe_handle_backward_generation",
                          _guard_backward_generation(unwrap(original))):
            flag = output_code.BoxedBool(True)
            output_code.cudagraph_partition_post_compile(
                [], graph, flag, {}, output_code.BoxedDeviceIndex(None))
    assert not flag


@pytest.mark.parametrize("filename,partition_maps,kernel_free,expected_skip", [
    ("/tmp/src/rtx/virtual.py", [], True, True),
    ("/tmp/src/not_rtx/virtual.py", [], True, False),
    ("/tmp/src/rtx/virtual.py", [object()], True, False),
    ("/tmp/src/rtx/virtual.py", [], False, False),
])
def test_virtual_alias_only_backward_without_any_rtx_op(filename,partition_maps,kernel_free,expected_skip):
    graph=SimpleNamespace(source_code="return (None, reinterpret_tensor(tangents_1, ...))",
        current_callable=lambda x: x,fx_kwargs={"is_backward":True},partition_maps=partition_maps,
        kernel_free_cudagraph=kernel_free,inductor_post_grad_graph_str=(
            f"        # Backward of forward node: File: {filename}:43 in backward, code: "
            "result.append(flat.narrow(0, offset, size).view(shape))"))
    called=[]
    _guard_backward_generation(lambda *args:called.append(args))(graph,SimpleNamespace(value=None))
    assert bool(called) is not expected_skip


@pytest.mark.parametrize("missing", ["box", "callable", "manager"])
def test_unrelated_inductor_assertions_remain_visible(missing):
    from torch._inductor import config, cudagraph_trees, output_code

    graph = SimpleNamespace(source_code="torch.ops.rtx_routing.varlen_hard.default",
                            current_callable=None if missing == "callable" else lambda x: x,
                            fx_kwargs={"is_backward": True})
    box = (None if missing == "box" else
           output_code.BoxedDeviceIndex(None if missing == "callable" else 0))
    notify = _guard_backward_generation(unwrap(output_code.maybe_handle_backward_generation))
    with config.patch({"triton.cudagraph_trees": True}), \
            patch.object(cudagraph_trees, "get_manager", return_value=None) as lookup, \
            pytest.raises(AssertionError):
        notify(graph, box)
    if missing == "manager":
        lookup.assert_called_once_with(0, create_if_none_exists=False)
    else:
        lookup.assert_not_called()


def test_existing_manager_notified_before_each_backward():
    from torch._inductor import config, cudagraph_trees, output_code

    events = []
    manager = SimpleNamespace(set_to_running_backward=lambda: events.append("notify"))

    def target(inputs):
        events.append("execute")
        return inputs

    graph = SimpleNamespace(source_code="torch.ops.rtx_routing.varlen_hard.default",
                            current_callable=target, fx_kwargs={"is_backward": True})
    notify = _guard_backward_generation(unwrap(output_code.maybe_handle_backward_generation))
    with config.patch({"triton.cudagraph_trees": True}), \
            patch.object(cudagraph_trees, "get_manager", return_value=manager) as lookup:
        notify(graph, output_code.BoxedDeviceIndex(1))
        for batch in ([1], [2]):
            assert graph.current_callable(batch) is batch
    lookup.assert_called_once_with(1, create_if_none_exists=False)
    assert events == ["notify", "execute", "notify", "execute"]


def test_fullgraph_checkpoint_saves_only_assignment_and_preserves_gradient():
    from functools import partial
    import torch
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    from rtx import balanced_moe_checkpoint_policy
    import rtx._routing_varlen  # registers the production schema and fake kernel
    op=torch.ops.rtx_routing.varlen_hard.default
    real_calls=[]

    def cpu_assignment(inputs,out,status,config,qb,rounds,policy,owner):
        # Contract-only stand-in: do not run any CUDA or claim routing accuracy.
        real_calls.append(1)
        out.copy_(inputs[0].argmax(1,keepdim=True))
        status.fill_(1)

    def body(x):
        ids=torch.empty((x.shape[0],1),device=x.device,dtype=torch.int64)
        status=torch.empty(2,device=x.device,dtype=torch.int64)
        op([x.detach()],ids,status,[1],4,64,[1],0)
        return x.gather(1,ids).sin()

    def stock(ctx,op,*a,**kw):
        return CheckpointPolicy.PREFER_RECOMPUTE

    policy=balanced_moe_checkpoint_policy(stock)
    assert policy(None,op)==CheckpointPolicy.MUST_SAVE
    assert policy(None,torch.ops.rtx_routing.varlen_prepare.default)==CheckpointPolicy.PREFER_RECOMPUTE
    assert policy(None,torch.ops.rtx_routing.varlen_sinkhorn_execute.default)==CheckpointPolicy.PREFER_RECOMPUTE
    def make_step(choice):
        context=partial(create_selective_checkpoint_contexts,choice)
        def step(x):
            return checkpoint(body,x,use_reentrant=False,context_fn=context).sum()
        return torch.compile(step,backend="inductor",fullgraph=True,dynamic=False)

    with torch.library._scoped_library("rtx_routing","IMPL","CPU") as lib:
        lib.impl("varlen_hard",cpu_assignment)
        data=torch.arange(48,dtype=torch.float32).reshape(8,6)/32
        for choice,calls in ((stock,2),(policy,1)):
            compiled=make_step(choice)
            for value in (data,-data):
                x=value.clone().requires_grad_()
                expected=value.gather(1,value.argmax(1,keepdim=True))
                before=len(real_calls)
                y=compiled(x)
                y.backward()
                assert len(real_calls)-before==calls
                torch.testing.assert_close(y,expected.sin().sum())
                torch.testing.assert_close(x.grad,torch.zeros_like(x).scatter_(1,value.argmax(1,keepdim=True),
                                                                               expected.cos()))
