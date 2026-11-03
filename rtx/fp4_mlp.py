"""Bias-free, recomputed NVFP4 SiTU MLP with MXFP8 backward."""
from dataclasses import dataclass
from functools import lru_cache
import json

import torch

from .configs.nvfp4_mlp import NVFP4MLPConfig, NVFP4_MLP_REVISION
from .runtime import BoundedCache, tensor_storage_bytes
from ._managed import guard_submission
from ._workspace import persistent_scratch


DEFAULT_NVFP4_MLP_CONFIG = NVFP4MLPConfig()
_RUNNERS = BoundedCache(32, max_bytes=512*1024**2, sizeof=tensor_storage_bytes)


@lru_cache(maxsize=128)
def _parse(key):
    return NVFP4MLPConfig.from_dict(json.loads(key))


@torch.compiler.assume_constant_result
def _key(config):
    if not isinstance(config, NVFP4MLPConfig):
        raise TypeError("config must be NVFP4MLPConfig")
    return _cached_key(config)


@lru_cache(maxsize=128)
def _cached_key(config):
    return json.dumps(config.to_dict(), sort_keys=True)


def _validate(x, gu, down, config):
    if x.ndim < 2 or gu.ndim != 2 or down.ndim != 2:
        raise ValueError("expected input [...,K], gate_up [2H,K], down [O,H]")
    if any(t.dtype != torch.bfloat16 for t in (x,gu,down)):
        raise TypeError("NVFP4MLP input and parameters must be BF16")
    if any(t.device != x.device for t in (gu,down)):
        raise ValueError("NVFP4MLP tensors must share a device")
    if x.device.type not in ("cuda", "meta"):
        raise ValueError("NVFP4MLP requires CUDA")
    if any(not t.is_contiguous() for t in (x,gu,down)):
        raise ValueError("NVFP4MLP input and parameters must be contiguous")
    if (gu.shape[0]%2 or gu.shape[1]!=x.shape[-1]
            or down.shape[1]!=gu.shape[0]//2 or any(v<=0 for v in (*x.shape,*gu.shape,*down.shape))):
        raise ValueError("incompatible or empty NVFP4MLP dimensions")
    reason=config.rejection(x.numel()//x.shape[-1],x.shape[-1],gu.shape[0]//2,down.shape[0])
    if reason:
        raise ValueError(reason)


@dataclass
class _Projection:
    quant: object
    gemm: object
    qx: torch.Tensor
    qw: torch.Tensor
    sx: torch.Tensor
    sw: torch.Tensor
    xp: object
    wp: object
    outer: torch.Tensor
    l2_fetch_granularity: int | None = None
    packed_hidden: bool = False

    @guard_submission
    def __call__(self, x, weight, out, dh=None, hidden=None, hidden_scales=None):
        from .fp8 import _ensure_l2_fetch_granularity
        _ensure_l2_fetch_granularity(self.l2_fetch_granularity)
        self.quant(x,weight,self.qx,self.qw,self.sx,self.sw,self.outer)
        aux=self.outer if dh is None else (self.outer,dh,hidden)
        if self.packed_hidden:
            aux=(self.outer,hidden_scales) if dh is None else (self.outer,dh,hidden,hidden_scales)
        self.gemm(self.xp,self.wp,self.sx,self.sw,out,aux)


def _projection(x, gu, config, derivative=False):
    from .runtime import validate_runtime_environment
    validate_runtime_environment()
    if torch.cuda.get_device_capability(x.device)[0] != 12:
        raise RuntimeError("NVFP4MLP requires RTX Blackwell SM12x")
    m,k=x.shape
    n=gu.shape[0]
    stream=torch.cuda.current_stream(x.device)
    identity=(x.device.index,stream.cuda_stream,m,n,k,config.gate_up,config.beta,config.linear_beta,
              config.glu_epilogue,config.glu_store_vec,config.glu_scratch_swizzle,
              config.hidden_dtype,config.hidden_quant_vec,derivative)
    found=_RUNNERS.get(identity)
    if found is not None:
        return found
    runner=_build_projection(x,gu,config,derivative)
    _RUNNERS[identity]=runner
    return runner


@persistent_scratch
def _build_projection(x,gu,config,derivative=False):
    from .fp4 import _empty_nvfp4_scales,_packed_fp4_view
    from .configs.nvfp4 import NVFP4Problem,nvfp4_storage_k
    from .kernels.nvfp4_quant import compile_nvfp4_jit_region_dual_quant
    from .kernels.nvfp4_mlp import compile_nvfp4_situ
    c=config.gate_up
    m,k=x.shape
    n=gu.shape[0]
    width=nvfp4_storage_k(k)
    qx=torch.empty((m,width//2),device=x.device,dtype=torch.uint8)
    qw=torch.empty((n,width//2),device=x.device,dtype=torch.uint8)
    sx=_empty_nvfp4_scales(m,k,c.quant,x.device)
    sw=_empty_nvfp4_scales(n,k,c.quant,x.device)
    count=(m+c.x_scale_region_rows-1)//c.x_scale_region_rows+(n+c.weight_scale_region_rows-1)//c.weight_scale_region_rows
    return _Projection(
        compile_nvfp4_jit_region_dual_quant(m,n,k,c.quant,c.x_scale_region_rows,c.weight_scale_region_rows,
            c.tensor_scale_mode,c.region_amax_load_bits,c.region_amax_unroll,c.region_waves,
            c.region_order,c.region_ownership),
        compile_nvfp4_situ(NVFP4Problem(m,n,width),c.gemm,c.x_scale_region_rows,c.weight_scale_region_rows,
            config.beta,config.linear_beta,derivative,config.glu_epilogue,config.glu_store_vec,config.glu_scratch_swizzle,
            config.hidden_dtype,config.hidden_quant_vec),
        qx,qw,sx,sw,_packed_fp4_view(qx),_packed_fp4_view(qw),
        torch.empty(count,device=x.device,dtype=torch.float32),c.l2_fetch_granularity,config.hidden_dtype=='mxfp8')


@persistent_scratch
def _build_mixed_down(x,weight,config):
    from .fp4 import NVFP4ForwardConfig,_make_jit_region_dynamic_runner
    from .configs.nvfp4 import NVFP4Problem
    from .kernels.nvfp4_mlp import compile_mxfp8_to_nvfp4_prepare
    m=x.shape[0];n,k=weight.shape
    c=config.down
    # Reuse the native linear's exact scale-transport/epilogue families and
    # automatic ownership. Only replace its source preparation implementation.
    runner=_make_jit_region_dynamic_runner(NVFP4Problem(m,n,k),NVFP4ForwardConfig.from_materialized(c),x.device)
    runner.quant=compile_mxfp8_to_nvfp4_prepare(m,n,k,c)
    return runner


@guard_submission
def _run_forward(x, gu, down, config, out):
    from .fp4 import NVFP4ForwardConfig, _make_jit_region_dynamic_runner
    from .configs.nvfp4 import NVFP4Problem
    m,k=x.numel()//x.shape[-1],x.shape[-1]
    h=gu.shape[0]//2
    x2=x.view(m,k)
    if config.hidden_dtype=='mxfp8':
        width=(h+31)//32*32
        hidden=torch.empty((m,width),device=x.device,dtype=torch.float8_e4m3fn)
        scales=torch.empty((m,width//32),device=x.device,dtype=torch.uint8)
        _projection(x2,gu,config)(x2,gu,hidden,hidden_scales=scales)
        identity=(x.device.index,torch.cuda.current_stream(x.device).cuda_stream,m,down.shape[0],h,config.down,'mixed_down')
        runner=_RUNNERS.get(identity)
        if runner is None:
            runner=_build_mixed_down(hidden,down,config)
            _RUNNERS[identity]=runner
        runner((hidden,scales.view(torch.float8_e8m0fnu)),down,out.view(m,down.shape[0]))
        return
    hidden=x.new_empty((m,h))
    _projection(x2,gu,config)(x2,gu,hidden)
    c=NVFP4ForwardConfig.from_materialized(config.down)
    identity=(x.device.index,torch.cuda.current_stream(x.device).cuda_stream,m,down.shape[0],h,c,"down")
    runner=_RUNNERS.get(identity)
    if runner is None:
        runner=_make_jit_region_dynamic_runner(NVFP4Problem(m,down.shape[0],h),c,x.device)
        _RUNNERS[identity]=runner
    runner(hidden,down,out.view(m,down.shape[0]))


@lru_cache(maxsize=128)
def _backward_keys(key,m,k,h,o):
    from .fp8_bwd import _intern_bwd_config
    from .kernels.mxfp8 import MXFP8Problem
    c=_parse(key)
    return (_intern_bwd_config(c.gate_backward or _default_mlp_backward(MXFP8Problem(m,2*h,k))),
            _intern_bwd_config(c.down_backward or _default_mlp_backward(MXFP8Problem(m,o,h))))


def _default_mlp_backward(problem):
    from .fp8_bwd import _default_bwd_config
    from ._bwd_search import materialized_bwd_seed
    base=_default_bwd_config(problem)
    if base.dx.backend==base.dw.backend=='fused':
        return materialized_bwd_seed(problem,base) or base
    return base


@guard_submission
def _run_backward(x, gu, down, dy, key, need_x=True, need_gu=True, need_down=True):
    from .fp8_bwd import _launch_dx_out, _launch_dw_out, _launch_bwd_out
    m,k=x.numel()//x.shape[-1],x.shape[-1]
    h,o=gu.shape[0]//2,down.shape[0]
    c=_parse(key)
    gate_key,down_key=_backward_keys(key,m,k,h,o)
    x2,dy2=x.view(m,k),dy.reshape(m,o).contiguous()
    hidden=x.new_empty((m,h))
    if c.hidden_dtype=='mxfp8':
        width=(h+31)//32*32
        packed=torch.empty((m,width),device=x.device,dtype=torch.float8_e4m3fn)
        scales=torch.empty((m,width//32),device=x.device,dtype=torch.uint8)
    dp=x.new_empty((m,2*h)) if need_x or need_gu else None
    if need_x or need_gu:
        dh=x.new_empty((m,h))
        _launch_dx_out(dy2,hidden,down,down_key,dh)
        _projection(x2,gu,c,True)(x2,gu,dp,dh,packed if c.hidden_dtype=='mxfp8' else hidden,
                                 scales if c.hidden_dtype=='mxfp8' else None)
        del dh
    elif need_down:
        if c.hidden_dtype=='mxfp8':
            _projection(x2,gu,c)(x2,gu,packed,hidden_scales=scales)
        else:
            _projection(x2,gu,c)(x2,gu,hidden)
    if need_down and c.hidden_dtype=='mxfp8':
        from .kernels.mxfp8_mlp import compile_dequant_hidden
        compile_dequant_hidden(m,h)(packed,scales.view(torch.float8_e8m0fnu),hidden)
    if c.hidden_dtype=='mxfp8':
        del packed,scales
    dx=x.new_empty((m,k)) if need_x else None
    dg=torch.empty_like(gu) if need_gu else None
    dd=torch.empty_like(down) if need_down else None
    parallel=None
    if c.backward_overlap and need_down and (need_x or need_gu):
        from .mlp import _overlap_stream
        caller=torch.cuda.current_stream(x.device)
        parallel=_overlap_stream(x.device.index,int(caller.cuda_stream))
        parallel.wait_stream(caller)
        with torch.cuda.stream(parallel):
            _launch_dw_out(dy2,hidden,down,down_key,dd)
    if need_down:
        if parallel is None:
            _launch_dw_out(dy2,hidden,down,down_key,dd)
    if need_x and need_gu:
        _launch_bwd_out(dp,x2,gu,gate_key,dx,dg)
    elif need_x:
        _launch_dx_out(dp,x2,gu,gate_key,dx)
    elif need_gu:
        _launch_dw_out(dp,x2,gu,gate_key,dg)
    if parallel is not None:
        caller.wait_stream(parallel)
    return None if dx is None else dx.view_as(x),dg,dd


@torch.library.custom_op("rtx::nvfp4_mlp", mutates_args=(), device_types="cuda")
def _mlp_op(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,key:str)->torch.Tensor:
    c=_parse(key)
    _validate(x,gu,down,c)
    out=x.new_empty((*x.shape[:-1],down.shape[0]))
    _run_forward(x,gu,down,c,out)
    return out


@_mlp_op.register_fake
def _fake(x,gu,down,key):
    _validate(x,gu,down,_parse(key))
    return x.new_empty((*x.shape[:-1],down.shape[0]))


@torch.library.custom_op("rtx::nvfp4_mlp_backward",mutates_args=(),device_types="cuda",
    schema="(Tensor x, Tensor gu, Tensor down, Tensor dy, str key, bool need_x, bool need_gu, bool need_down) -> (Tensor?, Tensor?, Tensor?)")
def _backward_op(x:torch.Tensor,gu:torch.Tensor,down:torch.Tensor,dy:torch.Tensor,key:str,
                 need_x:bool,need_gu:bool,need_down:bool
                 )->tuple[torch.Tensor|None,torch.Tensor|None,torch.Tensor|None]:
    return _run_backward(x,gu,down,dy,key,need_x,need_gu,need_down)


@_backward_op.register_fake
def _backward_fake(x,gu,down,dy,key,need_x,need_gu,need_down):
    return tuple(t.new_empty(t.shape) if needed else None for t,needed in zip((x,gu,down),(need_x,need_gu,need_down)))


def _setup(ctx,inputs,output):
    x,gu,down,key=inputs
    ctx.save_for_backward(x,gu,down)
    ctx.key=key


def _backward(ctx,dy):
    # Keep a required noncontiguous/expanded gradient copy compiler-visible.
    return (*_backward_op(*ctx.saved_tensors,dy.contiguous(),ctx.key,*ctx.needs_input_grad[:3]),None)


torch.library.register_autograd(_mlp_op,_backward,setup_context=_setup)


def nvfp4_mlp(x,gate_up_weight,down_weight,*,activation="situ",config=DEFAULT_NVFP4_MLP_CONFIG):
    """NVFP4 SiTU forward, recomputed first-order MXFP8 backward; BF16 I/O.

    Parameters are packed [gate; up] and down BF16 weights. No hidden activation
    or packed forward operand is retained by autograd. No bias/dropout/residual
    or higher-order differentiation is provided by this operation.
    The default forward connection is packed MXFP8, consumed by an on-chip
    decode/NVFP4 preparation kernel. ``config.hidden_dtype='bf16'`` is explicit.
    """
    if activation!='situ':
        raise ValueError("NVFP4MLP currently supports activation='situ' only")
    return _mlp_op(x,gate_up_weight,down_weight,_key(config))


class NVFP4MLP(torch.nn.Module):
    """Recomputed SiTU GLU MLP, NVFP4 forward / MXFP8 backward.

    ``hidden_features`` is the post-GLU H width, not 2H. SiTU uses the original
    gate for its sigmoid and evaluates both softcaps and product in FP32.
    ``config.beta``/``linear_beta`` default to 4/25. Parameters and I/O are
    BF16; only X and the two weights are saved. Biases are unsupported.
    The default H connection is MXFP8/E8M0, not BF16; down's NVFP4 preparation
    decodes it in registers. Both GEMMs use NVFP4. ``hidden_dtype='bf16'`` is
    an explicit higher-precision intermediate policy, not an autotuned choice.
    """
    def __init__(self,in_features,hidden_features,out_features=None,*,activation="situ",config=DEFAULT_NVFP4_MLP_CONFIG,
                 device=None,dtype=torch.bfloat16,autotune="off",tuning_policy=None,autotune_cache_dir=None,
                 training_mix=None,inference_mix=None):
        super().__init__()
        out_features=in_features if out_features is None else out_features
        if any(type(v) is not int or v<=0 for v in (in_features,hidden_features,out_features)):
            raise ValueError("feature dimensions must be positive integers")
        if dtype!=torch.bfloat16:
            raise TypeError("NVFP4MLP parameters must be BF16")
        if activation!='situ':
            raise ValueError("NVFP4MLP currently supports activation='situ' only")
        self.activation=activation
        self.in_features,self.hidden_features,self.out_features=in_features,hidden_features,out_features
        self.config=config
        self._config_key=_key(config)
        from ._nvfp4_mlp_autotune import deferred_spec
        self._first_hit_spec=deferred_spec(config,autotune,tuning_policy,autotune_cache_dir,training_mix,inference_mix)
        self._shape_configs={}
        self.gate_up_weight=torch.nn.Parameter(torch.empty((2*hidden_features,in_features),device=device,dtype=dtype))
        self.down_weight=torch.nn.Parameter(torch.empty((out_features,hidden_features),device=device,dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        for w in (self.gate_up_weight,self.down_weight):
            torch.nn.init.trunc_normal_(w,std=.02,a=-.04,b=.04)

    def forward(self,x):
        selected=self._shape_configs.get(self._shape_key(x))
        if selected is not None:
            return _mlp_op(x,self.gate_up_weight,self.down_weight,_key(selected))
        if self._first_hit_spec is not None:
            from ._nvfp4_mlp_autotune import selected_mlp
            training=torch.is_grad_enabled() and (x.requires_grad or self.gate_up_weight.requires_grad or self.down_weight.requires_grad)
            return selected_mlp(x,self.gate_up_weight,self.down_weight,self._first_hit_spec,training)[0]
        return _mlp_op(x,self.gate_up_weight,self.down_weight,_key(self.config))

    @staticmethod
    def _shape_key(x):
        return tuple(x.shape),str(x.device),str(x.dtype)

    def configure_shape_(self,example_x,*,config=None):
        """Install an explicit shape policy before compile/capture; retains no tensors."""
        if torch.compiler.is_compiling():
            raise RuntimeError("configure_shape_ must run outside compilation")
        if example_x.is_cuda:
            with torch.cuda.device(example_x.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("configure_shape_ must run before CUDA graph capture")
        config=self.config if config is None else config
        _validate(example_x,self.gate_up_weight,self.down_weight,config)
        identity=self._shape_key(example_x)
        self._shape_configs.pop(identity,None)
        self._shape_configs[identity]=config
        if len(self._shape_configs)>128:
            self._shape_configs.pop(next(iter(self._shape_configs)))
        return self

    def policy_for(self,example_x):
        """Return the explicit starting policy; never benchmark from introspection."""
        return self._shape_configs.get(self._shape_key(example_x),self.config)

    def autotune_(self,example_x,**kwargs):
        """Tune complete forward/backward (or phase='forward'), then install by shape.

        ``tuning_policy=AutotunePolicy(max_trials=8/16/32, ...)`` controls the
        shared durable UCB search. Numerical caps/scale regions are preserved.
        Compilation/tuning must finish before capture or timed training.
        """
        from .autotune.nvfp4_mlp import tune_nvfp4_mlp
        if example_x.device.type != "cuda":
            raise ValueError("autotune_ requires a real CUDA example")
        _validate(example_x,self.gate_up_weight,self.down_weight,self.policy_for(example_x))
        try:
            self.tuning_result=tune_nvfp4_mlp(example_x,self.gate_up_weight,self.down_weight,
                initial=self.policy_for(example_x),**kwargs)
        finally:
            from .runtime import clear_runtime_caches
            clear_runtime_caches(device=example_x.device,preserve_compiled=True)
        return self.configure_shape_(example_x,config=self.tuning_result.config)

    def export_tuning_spec(self,path=None):
        """Export exact first-hit metadata to a new file for preflight tooling."""
        if torch.compiler.is_compiling():
            raise RuntimeError("export_tuning_spec must run outside compilation")
        if self._first_hit_spec is None or self._shape_configs:
            raise ValueError("export requires automatic tuning without manual overrides")
        spec=json.loads(self._first_hit_spec)
        if not spec['cache_dir']:
            raise ValueError("export requires an explicit autotune_cache_dir")
        if path is not None:
            from pathlib import Path
            with Path(path).open('x') as f:
                json.dump(spec,f,indent=2,sort_keys=True)
        return spec

    def require_cached_(self):
        """Require a compatible disk winner; fail closed rather than live-tune."""
        spec=self.export_tuning_spec()
        spec.update(mode='cache',cache_miss='error')
        self._first_hit_spec=json.dumps(spec,sort_keys=True)
        return self

    @classmethod
    def from_linears(cls,gate,up,down,*,config=DEFAULT_NVFP4_MLP_CONFIG,**kwargs):
        """Copy three bias-free BF16 linears without mutating sources or RNG.

        Gate/up must share shape and requires_grad; all linears must share
        device and training mode. Build a new optimizer after conversion.
        """
        layers=(gate,up,down)
        if any(not isinstance(t,torch.nn.Linear) for t in layers):
            raise TypeError("source projections must be nn.Linear")
        if any(t.bias is not None for t in layers):
            raise ValueError("NVFP4MLP does not support projection biases")
        if gate.weight.shape!=up.weight.shape or down.in_features!=gate.out_features:
            raise ValueError("incompatible source projection shapes")
        if any(t.weight.dtype!=torch.bfloat16 for t in layers):
            raise TypeError("source weights must be BF16")
        if gate.weight.is_meta or any(t.weight.device!=gate.weight.device for t in layers):
            raise ValueError("source projections must share a materialized device")
        if any(t.training!=gate.training for t in layers):
            raise ValueError("source projections must share training mode")
        if gate.weight.requires_grad!=up.weight.requires_grad:
            raise ValueError("packed gate/up requires_grad must match")
        result=cls(gate.in_features,gate.out_features,down.out_features,config=config,device='meta',**kwargs)
        result.gate_up_weight=torch.nn.Parameter(torch.cat((gate.weight.detach(),up.weight.detach())),requires_grad=gate.weight.requires_grad)
        result.down_weight=torch.nn.Parameter(down.weight.detach().clone().contiguous(),requires_grad=down.weight.requires_grad)
        return result.train(gate.training)

    def extra_repr(self):
        return (f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
                f"out_features={self.out_features}, activation='situ', forward=NVFP4, backward=MXFP8, "
                f"hidden_dtype={self.config.hidden_dtype!r}, beta={self.config.beta}, linear_beta={self.config.linear_beta}")


def _clear_runtime_caches(*,preserve_compiled=False):
    import sys
    stats=_RUNNERS.stats()
    _RUNNERS.clear()
    selected=sys.modules.get('rtx._nvfp4_mlp_autotune')
    if selected is not None:
        selected.REPLAYS.clear()
        if not preserve_compiled:
            selected.SELECTIONS.clear()
    if not preserve_compiled:
        _parse.cache_clear()
        _cached_key.cache_clear()
        _backward_keys.cache_clear()
        registry=getattr(torch,'_rtx_nvfp4_mlp_launchers',None)
        if registry is not None:
            registry.clear()
    return stats


def _register_inductor():
    from ._managed import install
    install()
    from torch._inductor import ir
    from torch._inductor.lowering import register_lowering,view
    class Launcher:
        def __init__(self,key):
            self.config=_parse(key)
        def __call__(self,x,gu,down,*,out):
            _run_forward(x,gu,down,self.config,out)
    class Registry(dict):
        def __missing__(self,key):
            self[key]=value=Launcher(key)
            if len(self)>128:
                self.pop(next(iter(self)))
            return value
    torch._rtx_nvfp4_mlp_launchers=Registry()
    @register_lowering(torch.ops.rtx.nvfp4_mlp.default,type_promotion_kind=None)
    def lower(x,gu,down,key):
        import math
        leading=x.get_size()[:-1]
        m,k=math.prod(leading),x.get_size()[-1]
        inputs=[view(x,[m,k]),gu,down]
        inputs=[ir.ExternKernel.require_contiguous(ir.ExternKernel.realize_input(t)) for t in inputs]
        result=ir.TensorBox.create(ir.ExternKernelOut(
            ir.FixedLayout(inputs[0].get_device(),torch.bfloat16,[m,down.get_size()[0]],[down.get_size()[0],1]),inputs,
            python_kernel_name=f"torch._rtx_nvfp4_mlp_launchers[{key!r}]"))
        return view(result,[*leading,down.get_size()[0]])


_register_inductor()
