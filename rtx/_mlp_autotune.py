"""Opt-in runtime MLP selection with a forward-owned recomputation plan."""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import RLock

import torch

from ._execution_plan_token import PLAN_TOKEN_BYTES, read_plan_token, write_plan_token
from .autotune.effort import AutotunePolicy
from .runtime import BoundedCache


SELECTIONS = BoundedCache(64)
BACKWARDS = BoundedCache(32)
FORWARDS = BoundedCache(32)
_LOCK = RLock()
_LOWERINGS_REGISTERED = False


def deferred_spec(config, precision, mode, policy, cache_dir):
    if mode not in ("off", "cache", "balanced"):
        raise ValueError("MLP autotune must be off, cache, or balanced")
    if policy is not None and not isinstance(policy, AutotunePolicy):
        raise TypeError("tuning_policy must be AutotunePolicy")
    if mode == "off":
        return None
    if precision != "mxfp8":
        raise ValueError("automatic MLP tuning currently requires precision='mxfp8'; use autotune_ for bf16_gate")
    return json.dumps(dict(revision=7, initial=config.to_dict(), mode=mode,
        policy=asdict(policy or AutotunePolicy.preset("balanced")),
        cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser())), sort_keys=True)


def _select(x, gu, down, activation, spec_key, training):
    from .mlp import MXFP8MLPConfig
    from .kernels.mxfp8 import MXFP8Problem
    from .autotune.mlp import MLP_AUTOTUNE_REVISION, tune_mxfp8_mlp
    from .autotune.core import stable_id
    from .autotune.legacy import default_cache_dir
    from .autotune.winners import (runtime_winner_key, load_runtime_winner,
                                  save_runtime_winner, runtime_tuning_lock)
    identity = (tuple(x.shape), tuple(gu.shape), tuple(down.shape), str(x.device),
                activation, spec_key, training)
    with _LOCK, torch.cuda.device(x.device):
        found = SELECTIONS.get(identity)
        if found is not None:
            return found
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm MLP first-hit selection before CUDA graph capture")
        spec = json.loads(spec_key)
        values = spec["policy"]
        values["predictor_heads"] = tuple(values["predictor_heads"])
        policy = AutotunePolicy(**values).for_pass(training)
        initial = MXFP8MLPConfig.from_dict(spec["initial"])
        root = default_cache_dir() if spec["cache_dir"] is None else Path(spec["cache_dir"])
        problem = MXFP8Problem(x.numel() // x.shape[-1], gu.shape[0], x.shape[-1])
        key = runtime_winner_key("mxfp8_recompute_mlp",
            problem, device=x.device, kernel_revision=MLP_AUTOTUNE_REVISION,
            variant=stable_id(dict(out_features=down.shape[0], activation=activation,
                                   training=training, initial=spec["initial"], frontend_revision=1)))
        def cached():
            return load_runtime_winner(key, MXFP8MLPConfig.from_dict, root=root,
                tuning_policy=policy if spec["mode"] == "balanced" else None)
        selected = cached()
        if selected is None and spec["mode"] == "balanced":
            with runtime_tuning_lock(key, root=root):
                selected = cached()
                if selected is None:
                    # Custom-op dispatch runs below autograd; the harness owns
                    # independent differentiable inputs and its compiled graph.
                    stream = torch.cuda.current_stream(x.device)
                    def tune():
                        # A fresh thread avoids the custom-op's below-autograd
                        # dispatch TLS. Reuse the caller stream for ordering;
                        # the caller waits until the whole cold search finishes.
                        with torch.cuda.device(x.device), torch.cuda.stream(stream), torch.enable_grad():
                            return tune_mxfp8_mlp(x, gu, down, activation=activation,
                                phase="train" if training else "forward", initial=initial,
                                tuning_policy=policy, output_dir=root / "mlp_effort", progress=None)
                    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtx-mlp-first-hit") as pool:
                        result = pool.submit(tune).result()
                    selected = result.config
                    save_runtime_winner(key, selected.to_dict(), root=root,
                        config_id=stable_id(selected.to_dict()), median_ms=result.median_ms,
                        tuning_evidence=result.tuning_evidence)
        selected = initial if selected is None else selected
        selected_key = json.dumps(selected.to_dict(), sort_keys=True)
        SELECTIONS[identity] = selected_key
        return selected_key


class _BackwardContext:
    def __init__(self, tensors, activation, key, keys, joint, overlap, mask):
        self.saved_tensors = tensors
        self.activation, self.key, self.keys = activation, key, keys
        self.joint, self.overlap, self.needs_input_grad = joint, overlap, mask


def _backward_runner(key, activation, mask, shape, signature, *, replay=False):
    from .mlp import _parse, _backward, _backward_keys
    identity = key, activation, mask, shape, signature, replay
    with _LOCK:
        runner = BACKWARDS.get(identity)
        if runner is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm MLP backward before CUDA graph capture")
            config = _parse(key)
            keys = _backward_keys(key, *shape)
            joint, overlap = config.backward_recompute == "joint", config.backward_overlap
            def execute(x, gu, down, dy):
                ctx = _BackwardContext((x, gu, down), activation, key, keys, joint, overlap, mask)
                return _backward(ctx, dy, independent_outputs=True)[:3]
            # Give each bounded-cache plan its own frame cache. Otherwise
            # unrelated plans/gradient masks exhaust Dynamo's shared 8-entry
            # limit for this factory's single code object.
            execute.__code__ = execute.__code__.replace()
            runner = torch.compile(execute, fullgraph=True, dynamic=False,
                                   options={"triton.cudagraphs": replay})
            if replay:
                # cudaGraphLaunch is not permitted during an enclosing stream
                # capture. Prime a graph-free compiled body on the first call;
                # the outer capture can then record the same kernels directly.
                replay_runner = runner
                def uncaptured(x, gu, down, dy):
                    ctx = _BackwardContext((x, gu, down), activation, key, keys, joint, overlap, mask)
                    return _backward(ctx, dy, independent_outputs=True)[:3]
                uncaptured.__code__ = uncaptured.__code__.replace()
                capture_runner = torch.compile(uncaptured, fullgraph=True, dynamic=False,
                    options={"triton.cudagraphs": False})
                primed = False
                def runner(*values):
                    nonlocal primed
                    if torch.cuda.is_current_stream_capturing():
                        if not primed:
                            raise RuntimeError("warm MLP backward before external CUDA graph capture")
                        return capture_runner(*values)
                    if not primed:
                        with _LOCK:
                            if not primed:
                                capture_runner(*values)
                                primed = True
                    return replay_runner(*values)
            BACKWARDS[identity] = runner
        return runner


@torch.library.custom_op("rtx::mxfp8_mlp_independent_bwd", mutates_args=(), device_types="cuda")
def independent_bwd(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                    config_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """One paired runner, distinct output storages, no gradient staging copy."""
    from .fp8_bwd import _launch_bwd_out
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    _launch_bwd_out(dy, x, weight, config_key, dx, dw)
    return dx, dw


@independent_bwd.register_fake
def _independent_bwd_fake(dy, x, weight, config_key):
    return torch.empty_like(x), torch.empty_like(weight)


# Inductor's automatic partitioner cannot capture the mixed CPU/CUDA output
# container or the host plan read in backward. Manual capture after warmup is
# valid, but automatic capture must leave these host boundaries outside graphs.
# Version private op names: v1 had unsafe partitions; v2 predates replay routing.
@torch.library.custom_op("rtx::mxfp8_mlp_selected_v3", mutates_args=(), device_types="cuda",
                         tags=(torch.Tag.cudagraph_unsafe,))
def selected_mlp(x: torch.Tensor, gu: torch.Tensor, down: torch.Tensor,
                 activation: str, spec: str, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _selected_forward(x, gu, down, activation, spec, training)


def _signature(tensors):
    return tuple((tuple(t.shape), tuple(t.stride()), str(t.device), str(t.dtype), t.requires_grad)
                 for t in tensors)


def _replay_forward(x, gu, down, activation, key):
    from .mlp import _mlp_op
    identity = key, activation, _signature((x, gu, down))
    with _LOCK:
        runner = FORWARDS.get(identity)
        if runner is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm selected MLP replay before CUDA graph capture")
            def execute(x, gu, down):
                return _mlp_op(x, gu, down, activation, key)
            execute.__code__ = execute.__code__.replace()
            runner = torch.compile(execute, fullgraph=True, dynamic=False,
                                   options={"triton.cudagraphs": True})
            FORWARDS[identity] = runner
    with torch.no_grad():
        return runner(x, gu, down)


def _selected_forward(x, gu, down, activation, spec, training, *, replay=False):
    from .mlp import _validate, _parse, _run_forward
    _validate(x, gu, down, activation)
    key = _select(x, gu, down, activation, spec, training)
    saved = torch.empty(PLAN_TOKEN_BYTES, dtype=torch.uint8, device="cpu")
    write_plan_token(saved, key)
    if replay and not torch.cuda.is_current_stream_capturing():
        output = _replay_forward(x, gu, down, activation, key)
    else:
        output = torch.empty((*x.shape[:-1], down.shape[0]), dtype=x.dtype, device=x.device)
        _run_forward(x, gu, down, activation, _parse(key), output)
    return output, saved


@selected_mlp.register_fake
def _fake(x, gu, down, activation, spec, training):
    return (torch.empty((*x.shape[:-1], down.shape[0]), dtype=x.dtype, device=x.device),
            torch.empty(PLAN_TOKEN_BYTES, dtype=torch.uint8, device="cpu"))


@torch.library.custom_op("rtx::mxfp8_mlp_selected_backward_v3", mutates_args=(), device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
    schema="(Tensor x, Tensor gu, Tensor down, Tensor dy, Tensor saved, str activation, bool need_x, bool need_gu, bool need_down) -> (Tensor?, Tensor?, Tensor?)")
def selected_backward(x: torch.Tensor, gu: torch.Tensor, down: torch.Tensor,
                      dy: torch.Tensor, saved: torch.Tensor, activation: str,
                      need_x: bool, need_gu: bool, need_down: bool
                      ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    return _selected_backward(x, gu, down, dy, saved, activation, need_x, need_gu, need_down)


def _selected_backward(x, gu, down, dy, saved, activation, need_x, need_gu, need_down, *, replay=False):
    shape = (x.numel() // x.shape[-1], x.shape[-1], gu.shape[0] // 2, down.shape[0])
    with torch.cuda.device(x.device), torch.no_grad():
        signature = _signature((x, gu, down, dy))
        kwargs = {"replay": True} if replay else {}
        runner = _backward_runner(read_plan_token(saved), activation, (need_x, need_gu, need_down), shape, signature, **kwargs)
        return runner(x, gu, down, dy)


@selected_backward.register_fake
def _backward_fake(x, gu, down, dy, saved, activation, need_x, need_gu, need_down):
    return tuple(torch.empty_like(t) if need else None
                 for t, need in zip((x, gu, down), (need_x, need_gu, need_down)))


def _setup(ctx, inputs, output):
    x, gu, down, activation, spec, training = inputs
    ctx.save_for_backward(x, gu, down, output[1])
    ctx.activation = activation
    ctx.mark_non_differentiable(output[1])


def _backward(ctx, dy, unused):
    x, gu, down, saved = ctx.saved_tensors
    grads = selected_backward(x, gu, down, dy, saved, ctx.activation, *ctx.needs_input_grad[:3])
    return (*grads, None, None, None)


torch.library.register_autograd(selected_mlp, _backward, setup_context=_setup)


# Lowering-only boundaries inherit the caller's Inductor cudagraph option.
# Their nested compiled functions contain CUDA tensors only; the CPU token is
# still allocated/read outside those replay regions. Eager behavior is unchanged.
@torch.library.custom_op("rtx::mxfp8_mlp_selected_replay_v2", mutates_args=(), device_types="cuda",
                         tags=(torch.Tag.cudagraph_unsafe,))
def _replay_op(x: torch.Tensor, gu: torch.Tensor, down: torch.Tensor,
               activation: str, spec: str, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _selected_forward(x, gu, down, activation, spec, training, replay=True)


_replay_op.register_fake(_fake)


@torch.library.custom_op("rtx::mxfp8_mlp_selected_backward_replay_v2", mutates_args=(), device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
    schema="(Tensor x, Tensor gu, Tensor down, Tensor dy, Tensor saved, str activation, bool need_x, bool need_gu, bool need_down) -> (Tensor?, Tensor?, Tensor?)")
def _replay_backward_op(x, gu, down, dy, saved, activation, need_x, need_gu, need_down):
    return _selected_backward(x, gu, down, dy, saved, activation, need_x, need_gu, need_down, replay=True)


_replay_backward_op.register_fake(_backward_fake)


def _register_replay_lowerings():
    """Isolated internal Inductor adapter, qualified on Torch 2.13.0+cu132.

    CUDA graph mode remains the caller's choice; no autotuning policy changes.
    Private op revisions prevent reuse of pre-routing Inductor artifacts.
    """
    global _LOWERINGS_REGISTERED
    if _LOWERINGS_REGISTERED:
        return
    from torch._inductor import config
    from torch._inductor.lowering import register_lowering, fallback_handler
    pairs = ((torch.ops.rtx.mxfp8_mlp_selected_v3.default, torch.ops.rtx.mxfp8_mlp_selected_replay_v2.default),
             (torch.ops.rtx.mxfp8_mlp_selected_backward_v3.default, torch.ops.rtx.mxfp8_mlp_selected_backward_replay_v2.default))
    for ordinary, replay in pairs:
        def lower(*args, _ordinary=ordinary, _replay=replay):
            target = _replay if config.triton.cudagraphs else _ordinary
            return fallback_handler(target, add_to_fallback_set=False)(*args)
        register_lowering(ordinary, type_promotion_kind=None)(lower)
    _LOWERINGS_REGISTERED = True


_register_replay_lowerings()
