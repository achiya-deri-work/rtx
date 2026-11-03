"""First-hit SiTU selection; forward-owned immutable plans and bounded replay."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from threading import RLock

import torch

from ._execution_plan_token import PLAN_TOKEN_BYTES,read_plan_token,write_plan_token
from .configs.nvfp4_mlp import NVFP4MLPConfig,NVFP4_MLP_REVISION
from .runtime import BoundedCache,tensor_storage_bytes


SELECTIONS=BoundedCache(64)
REPLAYS=BoundedCache(16,max_bytes=512*1024**2,sizeof=tensor_storage_bytes)
_LOCK=RLock()


def deferred_spec(config,mode,policy,cache_dir,training_mix=None,inference_mix=None):
    from .autotune.effort import AutotunePolicy
    from .autotune.execution_mix import MLPExecutionMix
    if mode not in ("off","cache","balanced"):
        raise ValueError("autotune must be off, cache, or balanced")
    if policy is not None and not isinstance(policy,AutotunePolicy):
        raise TypeError("tuning_policy must be AutotunePolicy")
    mixes={}
    for phase,mix in (("train",training_mix),("forward",inference_mix)):
        if mix is not None:
            if not isinstance(mix,MLPExecutionMix):
                raise TypeError("execution mixes must be MLPExecutionMix")
            mix.validate_phase(phase)
            mixes[phase]=asdict(mix)
    if mode=="off":
        if mixes:
            raise ValueError("execution mixes require autotune=cache or balanced")
        return None
    return json.dumps(dict(revision=NVFP4_MLP_REVISION,initial=config.to_dict(),mode=mode,
        policy=asdict(policy or AutotunePolicy.preset("balanced")),execution_mixes=mixes,
        cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser())),sort_keys=True)


def _select(x,gu,down,spec_key,training):
    from .autotune.nvfp4_mlp import tune_nvfp4_mlp,source_identity
    from .autotune.core import stable_id
    from .autotune.effort import AutotunePolicy
    from .autotune.execution_mix import MLPExecutionMix
    from .autotune.legacy import default_cache_dir
    from .autotune.winners import runtime_winner_key,load_runtime_winner,save_runtime_winner,runtime_tuning_lock
    from .kernels.mxfp8 import MXFP8Problem
    identity=(x.device.index,tuple(x.shape),tuple(gu.shape),tuple(down.shape),spec_key,training)
    with _LOCK,torch.cuda.device(x.device):
        found=SELECTIONS.get(identity)
        if found is not None:
            return found
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm NVFP4MLP first-hit selection before CUDA graph capture")
        spec=json.loads(spec_key)
        if spec['revision']!=NVFP4_MLP_REVISION:
            raise ValueError("unsupported first-hit NVFP4MLP revision")
        initial=NVFP4MLPConfig.from_dict(spec['initial'])
        values=spec['policy'];values['predictor_heads']=tuple(values['predictor_heads'])
        policy=AutotunePolicy(**values).for_pass(training)
        phase="train" if training else "forward"
        mix=MLPExecutionMix(**spec['execution_mixes'][phase]) if phase in spec['execution_mixes'] else None
        root=default_cache_dir() if spec['cache_dir'] is None else Path(spec['cache_dir'])
        m,k=x.numel()//x.shape[-1],x.shape[-1]
        h,o=gu.shape[0]//2,down.shape[0]
        key=runtime_winner_key("nvfp4_recompute_mlp",MXFP8Problem(m,2*h,k),device=x.device,
            kernel_revision=NVFP4_MLP_REVISION,variant=stable_id(dict(input_shape=list(x.shape),o=o,
                training=training,initial=spec['initial'],mix=None if mix is None else mix.contract(),source=source_identity())))
        def cached():
            return load_runtime_winner(key,NVFP4MLPConfig.from_dict,root=root,
                tuning_policy=policy if spec['mode']=='balanced' else None,backward=training,
                rejection=lambda c:c.rejection(m,k,h,o))
        selected=cached()
        if selected is None and spec['mode']=='balanced':
            with runtime_tuning_lock(key,root=root):
                selected=cached()
                if selected is None:
                    stream=torch.cuda.current_stream(x.device)
                    def tune():
                        with torch.cuda.device(x.device),torch.cuda.stream(stream):
                            return tune_nvfp4_mlp(x,gu,down,initial=initial,phase=phase,tuning_policy=policy,
                                output_dir=root/"nvfp4_mlp",execution_mix=mix)
                    try:
                        # Keep nested Dynamo compilation out of the custom-op
                        # dispatcher's no-grad/trace state, just as MXFP8MLP.
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            result=pool.submit(tune).result()
                    finally:
                        from .runtime import clear_runtime_caches
                        clear_runtime_caches(device=x.device,preserve_compiled=True)
                    selected=result.config
                    save_runtime_winner(key,selected.to_dict(),config_id=stable_id(selected.to_dict()),root=root,
                        median_ms=result.median_ms,tuning_evidence=result.tuning_evidence)
        if selected is None:
            if spec.get('cache_miss')=='error':
                raise RuntimeError(f"NVFP4MLP requires a prepared winner for shape={tuple(x.shape)}, training={training}")
            selected=initial
        selected_key=json.dumps(selected.to_dict(),sort_keys=True)
        SELECTIONS[identity]=selected_key
        return selected_key


def _execute_forward(x,gu,down,key):
    from .fp4_mlp import _run_forward,_parse
    output=x.new_empty((*x.shape[:-1],down.shape[0]))
    _run_forward(x,gu,down,_parse(key),output)
    return (output,)


def _execute_backward(x,gu,down,dy,key,mask):
    from .fp4_mlp import _run_backward
    return _run_backward(x,gu,down,dy,key,*mask)


def _replay(kind,values,key,mask=()):
    from ._graph_replay import KernelReplay
    signature=tuple((tuple(t.shape),tuple(t.stride()),str(t.device),str(t.dtype)) for t in values)
    identity=(values[0].device.index,torch.cuda.current_stream(values[0].device).cuda_stream,kind,key,signature,mask)
    with _LOCK:
        runner=REPLAYS.get(identity)
        if runner is None:
            execute=(lambda *t:_execute_forward(*t,key)) if kind=='forward' else (lambda *t:_execute_backward(*t,key,mask))
            runner=KernelReplay(execute,values)
            REPLAYS[identity]=runner
    return runner(*values)


def _forward(x,gu,down,spec,training,replay=False):
    from .fp4_mlp import _validate,_parse
    key=_select(x,gu,down,spec,training)
    _validate(x,gu,down,_parse(key))
    token=torch.empty(PLAN_TOKEN_BYTES,dtype=torch.uint8,device='cpu')
    write_plan_token(token,key)
    out=(_replay('forward',(x,gu,down),key) if replay else _execute_forward(x,gu,down,key))[0]
    return out,token


@torch.library.custom_op("rtx::nvfp4_mlp_selected",mutates_args=(),device_types="cuda",tags=(torch.Tag.cudagraph_unsafe,))
def selected_mlp(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,spec:str,training:bool)->tuple[torch.Tensor,torch.Tensor]:
    return _forward(x,gu,down,spec,training)


@selected_mlp.register_fake
def _fake(x,gu,down,spec,training):
    from .fp4_mlp import _validate
    _validate(x,gu,down,NVFP4MLPConfig.from_dict(json.loads(spec)['initial']))
    return x.new_empty((*x.shape[:-1],down.shape[0])),torch.empty(PLAN_TOKEN_BYTES,dtype=torch.uint8,device='cpu')


@torch.library.custom_op("rtx::nvfp4_mlp_selected_backward",mutates_args=(),device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
    schema="(Tensor x, Tensor gu, Tensor down, Tensor dy, Tensor token, bool need_x, bool need_gu, bool need_down) -> (Tensor?, Tensor?, Tensor?)")
def selected_backward(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,dy:torch.Tensor,token:torch.Tensor,
                      need_x:bool,need_gu:bool,need_down:bool
                      )->tuple[torch.Tensor|None,torch.Tensor|None,torch.Tensor|None]:
    return _execute_backward(x,gu,down,dy,read_plan_token(token),(need_x,need_gu,need_down))


@selected_backward.register_fake
def _backward_fake(x,gu,down,dy,token,need_x,need_gu,need_down):
    return tuple(t.new_empty(t.shape) if n else None for t,n in zip((x,gu,down),(need_x,need_gu,need_down)))


def _setup(ctx,inputs,output):
    ctx.save_for_backward(*inputs[:3],output[1])
    ctx.mark_non_differentiable(output[1])


def _backward(ctx,dy,unused):
    x,gu,down,token=ctx.saved_tensors
    return (*selected_backward(x,gu,down,dy.contiguous(),token,*ctx.needs_input_grad[:3]),None,None)


torch.library.register_autograd(selected_mlp,_backward,setup_context=_setup)


@torch.library.custom_op("rtx::nvfp4_mlp_selected_replay",mutates_args=(),device_types="cuda",tags=(torch.Tag.cudagraph_unsafe,))
def _forward_replay(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,spec:str,training:bool)->tuple[torch.Tensor,torch.Tensor]:
    return _forward(x,gu,down,spec,training,True)


_forward_replay.register_fake(_fake)


@torch.library.custom_op("rtx::nvfp4_mlp_selected_backward_replay",mutates_args=(),device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
    schema="(Tensor x, Tensor gu, Tensor down, Tensor dy, Tensor token, bool need_x, bool need_gu, bool need_down) -> (Tensor?, Tensor?, Tensor?)")
def _backward_replay(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,dy:torch.Tensor,token:torch.Tensor,
                     need_x:bool,need_gu:bool,need_down:bool
                     )->tuple[torch.Tensor|None,torch.Tensor|None,torch.Tensor|None]:
    return _replay('backward',(x,gu,down,dy),read_plan_token(token),(need_x,need_gu,need_down))


_backward_replay.register_fake(_backward_fake)


def _register():
    from torch._inductor import config
    from torch._inductor.lowering import register_lowering,fallback_handler
    for ordinary,replay in ((selected_mlp,_forward_replay),(selected_backward,_backward_replay)):
        def lower(*args,_ordinary=ordinary,_replay=replay):
            op=_replay if config.triton.cudagraphs else _ordinary
            return fallback_handler(op._opoverload,add_to_fallback_set=False)(*args)
        register_lowering(ordinary._opoverload,type_promotion_kind=None)(lower)


_register()
