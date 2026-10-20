"""Opt-in runtime MLP selection with a forward-owned recomputation plan."""
from dataclasses import asdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import json
import logging
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
_SELECTION_LOCK = RLock()
_LOWERINGS_REGISTERED = False
_LOG = logging.getLogger('rtx.autotune.mlp')


@lru_cache(maxsize=64)
def _spec_mode_and_fixed_key(spec_key):
    """Bounded host-only immutable metadata; never owns tensors or winners."""
    spec = json.loads(spec_key)
    mode = spec['mode']
    return mode, json.dumps(spec['initial'],sort_keys=True) if mode == 'fixed' else None


def deferred_spec(config, precision, mode, policy, cache_dir, *, training_mix=None, inference_mix=None):
    from .autotune.execution_mix import MLPExecutionMix
    mixes={}
    for phase,mix in (('train',training_mix),('forward',inference_mix)):
        if mix is not None:
            if not isinstance(mix,MLPExecutionMix):
                raise TypeError('execution mixes must be MLPExecutionMix')
            mix.validate_phase(phase)
            mixes[phase]=asdict(mix)
    if mode not in ("off", "cache", "balanced"):
        raise ValueError("MLP autotune must be off, cache, or balanced")
    if policy is not None and not isinstance(policy, AutotunePolicy):
        raise TypeError("tuning_policy must be AutotunePolicy")
    if mode == "off":
        if mixes:
            raise ValueError('execution mixes require autotune=cache or balanced')
        return None
    if precision != "mxfp8":
        raise ValueError("automatic MLP tuning currently requires precision='mxfp8'; use autotune_ for bf16_gate")
    return json.dumps(dict(revision=8, initial=config.to_dict(), mode=mode,
        policy=asdict(policy or AutotunePolicy.preset("balanced")),
        **({'execution_mixes':mixes} if mixes else {}),
        cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser())), sort_keys=True)


def _runtime_variant(out_features,activation,training,initial,replay,mix=None):
    """Preserve legacy identities only when the legacy objective is requested."""
    from .autotune.core import stable_id
    return stable_id(dict(out_features=out_features,activation=activation,
        training=training,initial=initial,frontend_revision=2,
        measurement_mode='private_replay' if replay else 'selected_direct',
        **({'execution_mix':mix.contract()} if mix is not None else {})))


@lru_cache(maxsize=128)
def _bound_plan(spec_key, shape, gate_shape, down_shape, device, dtype, activation, training, replay):
    """Immutable startup binding lookup: no disk, tuning or tensor ownership."""
    spec = json.loads(spec_key)
    if (gate_shape != tuple(spec['gate_shape']) or down_shape != tuple(spec['down_shape'])):
        raise RuntimeError('bound MLP parameter shapes changed; requalify and rebind')
    if replay:
        raise RuntimeError('bound MLP plans require selected_direct; set triton.cudagraphs=False')
    if (device != spec['device'] or dtype != 'torch.bfloat16'
            or activation != spec['activation']):
        raise RuntimeError('bound MLP device/dtype/activation changed; requalify and rebind')
    for entry in spec['plans']:
        if tuple(entry['input_shape']) == shape and entry['training'] == training:
            return json.dumps(entry['config'], sort_keys=True)
    raise RuntimeError(f'undeclared bound MLP context: shape={shape}, training={training}; tuning is disabled')


def _select(x, gu, down, activation, spec_key, training, *, replay=False):
    from .mlp import MXFP8MLPConfig
    from .kernels.mxfp8 import MXFP8Problem
    from .autotune.mlp import MLP_AUTOTUNE_REVISION, tune_mxfp8_mlp
    from .autotune.core import stable_id
    from .autotune.legacy import default_cache_dir
    from .autotune.winners import (runtime_winner_key, load_runtime_winner,
                                  save_runtime_winner, runtime_tuning_lock)
    identity = (tuple(x.shape), tuple(gu.shape), tuple(down.shape), str(x.device),
                activation, spec_key, training, replay)
    mode, fixed_key = _spec_mode_and_fixed_key(spec_key)
    if mode == 'bound':
        if not x.is_contiguous():
            raise RuntimeError('bound MLP plans require contiguous inputs')
        return _bound_plan(spec_key, tuple(x.shape), tuple(gu.shape), tuple(down.shape), str(x.device), str(x.dtype),
                           activation, training, replay)
    if mode=='fixed':
        # Qualification must not recurse into selection or wait on the cold
        # caller's lock while it is waiting for the benchmark worker.
        return fixed_key
    with _SELECTION_LOCK, torch.cuda.device(x.device):
        found = SELECTIONS.get(identity)
        if found is not None:
            return found
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm MLP first-hit selection before CUDA graph capture")
        # Parsing/mutating the policy is cold-path work. Warm cache hits above
        # need only immutable spec identity, not a new nested JSON dictionary.
        spec = json.loads(spec_key)
        values = spec["policy"]
        values["predictor_heads"] = tuple(values["predictor_heads"])
        policy = AutotunePolicy(**values).for_pass(training)
        initial = MXFP8MLPConfig.from_dict(spec["initial"])
        from .autotune.execution_mix import MLPExecutionMix
        phase='train' if training else 'forward'
        mix_values=spec.get('execution_mixes',{}).get(phase)
        mix=None if mix_values is None else MLPExecutionMix(**mix_values)
        if mix is not None:
            mix.validate_phase(phase)
        root = default_cache_dir() if spec["cache_dir"] is None else Path(spec["cache_dir"])
        problem = MXFP8Problem(x.numel() // x.shape[-1], gu.shape[0], x.shape[-1])
        key = runtime_winner_key("mxfp8_recompute_mlp",
            problem, device=x.device, kernel_revision=MLP_AUTOTUNE_REVISION,
            variant=_runtime_variant(down.shape[0],activation,training,spec['initial'],replay,mix))
        def cached():
            return load_runtime_winner(key, MXFP8MLPConfig.from_dict, root=root,
                tuning_policy=policy if spec["mode"] == "balanced" else None)
        selected = cached()
        _LOG.info('cache_decision shape=%s training=%s replay=%s hit=%s mode=%s',
                  tuple(x.shape),training,replay,selected is not None,spec['mode'])
        if selected is None and spec["mode"] == "balanced":
            with runtime_tuning_lock(key, root=root):
                selected = cached()
                if selected is None:
                    incumbent=load_runtime_winner(key,MXFP8MLPConfig.from_dict,root=root) or initial
                    # Keep the user's initial search anchor stable across
                    # effort increases. A new winner is a candidate and the
                    # qualification baseline, not a new journal namespace.
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
                                portfolio=() if incumbent == initial else (incumbent,),
                                qualification_incumbent=None if incumbent == initial else incumbent,
                                tuning_policy=policy, output_dir=root / "mlp_effort", progress=_LOG.info,
                                measurement_mode='private_replay' if replay else 'selected_direct',
                                execution_mix=mix)
                    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtx-mlp-first-hit") as pool:
                        try:
                            result = pool.submit(tune).result()
                        finally:
                            # Search candidates own stream/config-specific scratch.
                            # Evicted tensors can leave GiBs reserved in the shared
                            # MemPool even with no live blocks. Release both runner
                            # caches and pool ownership before resuming the model.
                            # Disk winners and compiled model graphs are preserved;
                            # selected runners rebuild lazily on their runtime stream.
                            from .runtime import clear_runtime_caches
                            clear_runtime_caches(device=x.device,preserve_compiled=True)
                    selected = result.config
                    _LOG.info('selection_complete elapsed_s=%s decision=%s',result.elapsed_s,result.winner_decision)
                    save_runtime_winner(key, selected.to_dict(), root=root,
                        config_id=stable_id(selected.to_dict()), median_ms=result.median_ms,
                        metadata={'deployment_decision':result.winner_decision},
                        tuning_evidence=result.tuning_evidence)
        if selected is None:
            if spec.get('cache_miss') == 'error':
                raise RuntimeError(
                    'RTX MLP requires a cached winner; missing contract: '
                    f'shape={tuple(x.shape)}, out_features={down.shape[0]}, '
                    f'activation={activation}, training={training}, replay={replay}, '
                    f'device={x.device}, variant={key.variant if hasattr(key, "variant") else key}. '
                    'Prepare this exact context before full-model allocation; live tuning is disabled.')
            _LOG.info('cache_miss_fallback shape=%s using=initial',tuple(x.shape))
            selected = initial
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
    device = torch.device(signature[0][2])
    identity = key, activation, mask, shape, signature, replay, int(torch.cuda.current_stream(device).cuda_stream)
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
            runner = execute
            if replay:
                from ._graph_replay import KernelReplay
                captured = None
                def runner(*values):
                    nonlocal captured
                    with _LOCK:
                        if captured is None:
                            captured = KernelReplay(execute, values)
                    return captured(*values)
            BACKWARDS[identity] = runner
        return runner


@torch.library.custom_op("rtx::mxfp8_mlp_independent_bwd", mutates_args=(), device_types="cuda")
def independent_bwd(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                    config_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """One paired runner, distinct output storages, no gradient staging copy."""
    return _independent_bwd_launch(dy, x, weight, config_key)


def _independent_bwd_launch(dy, x, weight, config_key):
    """Launch helper for callers already inside an opaque CUDA boundary."""
    from .fp8_bwd import _launch_bwd_out
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    _launch_bwd_out(dy, x, weight, config_key, dx, dw)
    return dx, dw


def independent_dx(dy, x, weight, config_key, *, shape_only=False):
    """Partial gradient without Python ownership of temporary operands.

    Called inside the opaque selected-backward boundary. Same-stream launch
    ordering protects operands until the GPU consumes them; attaching them to
    returned gradients would retain every MLP workspace until backward ends.
    """
    from .fp8_bwd import _launch_dx_out
    # Only joint MLP recomputation passes an unread, fresh shape placeholder.
    # Fixed config keys are required: tuning would inspect its input values.
    if shape_only:
        from .fp8_bwd import _CONFIGS
        if config_key not in _CONFIGS:
            raise RuntimeError('shape-only dX requires an interned fixed configuration')
    dx = x if shape_only else torch.empty_like(x)
    _launch_dx_out(dy, x, weight, config_key, dx)
    return dx


def independent_dw(dy, x, weight, config_key):
    """Return only dW storage, without retaining the recomputed hidden view."""
    from .fp8_bwd import _launch_dw_out
    dw = torch.empty_like(weight)
    _launch_dw_out(dy, x, weight, config_key, dw)
    return dw


@independent_bwd.register_fake
def _independent_bwd_fake(dy, x, weight, config_key):
    return torch.empty_like(x), torch.empty_like(weight)


# Inductor's automatic partitioner cannot capture the mixed CPU/CUDA output
# container or the host plan read in backward. Manual capture after warmup is
# valid, but automatic capture must leave these host boundaries outside graphs.
# Version private names to invalidate wrappers using nested torch.compile.
@torch.library.custom_op("rtx::mxfp8_mlp_selected_v4", mutates_args=(), device_types="cuda",
                         tags=(torch.Tag.cudagraph_unsafe,))
def selected_mlp(x: torch.Tensor, gu: torch.Tensor, down: torch.Tensor,
                 activation: str, spec: str, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _selected_forward(x, gu, down, activation, spec, training)


def _signature(tensors):
    return tuple((tuple(t.shape), tuple(t.stride()), str(t.device), str(t.dtype), t.requires_grad)
                 for t in tensors)


def _replay_forward(x, gu, down, activation, key):
    from .mlp import _run_forward, _parse
    from ._graph_replay import KernelReplay
    identity = key, activation, _signature((x, gu, down)), int(torch.cuda.current_stream(x.device).cuda_stream)
    with _LOCK:
        runner = FORWARDS.get(identity)
        if runner is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm selected MLP replay before CUDA graph capture")
            def execute(x, gu, down):
                y = torch.empty((*x.shape[:-1], down.shape[0]), dtype=x.dtype, device=x.device)
                _run_forward(x, gu, down, activation, _parse(key), y)
                return (y,)
            runner = KernelReplay(execute, (x, gu, down))
            FORWARDS[identity] = runner
    with torch.no_grad():
        return runner(x, gu, down)[0]


def _selected_forward(x, gu, down, activation, spec, training, *, replay=False):
    from .mlp import _validate, _parse, _run_forward
    _validate(x, gu, down, activation)
    key = _select(x, gu, down, activation, spec, training, replay=replay)
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


@torch.library.custom_op("rtx::mxfp8_mlp_selected_backward_v4", mutates_args=(), device_types="cuda",
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
# Their captured kernel sequences contain CUDA tensors only; the CPU token is
# still allocated/read outside those replay regions. Eager behavior is unchanged.
@torch.library.custom_op("rtx::mxfp8_mlp_selected_replay_v3", mutates_args=(), device_types="cuda",
                         tags=(torch.Tag.cudagraph_unsafe,))
def _replay_op(x: torch.Tensor, gu: torch.Tensor, down: torch.Tensor,
               activation: str, spec: str, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _selected_forward(x, gu, down, activation, spec, training, replay=True)


_replay_op.register_fake(_fake)


@torch.library.custom_op("rtx::mxfp8_mlp_selected_backward_replay_v3", mutates_args=(), device_types="cuda",
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
    pairs = ((torch.ops.rtx.mxfp8_mlp_selected_v4.default, torch.ops.rtx.mxfp8_mlp_selected_replay_v3.default),
             (torch.ops.rtx.mxfp8_mlp_selected_backward_v4.default, torch.ops.rtx.mxfp8_mlp_selected_backward_replay_v3.default))
    for ordinary, replay in pairs:
        def lower(*args, _ordinary=ordinary, _replay=replay):
            target = _replay if config.triton.cudagraphs else _ordinary
            return fallback_handler(target, add_to_fallback_set=False)(*args)
        register_lowering(ordinary, type_promotion_kind=None)(lower)
    _LOWERINGS_REGISTERED = True


_register_replay_lowerings()
