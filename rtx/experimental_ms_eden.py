"""Experimental original-BF16 MS-EDEN gradient operation.

Not exported by ``rtx`` and not used by NVFP4Linear. Random signs and uniforms
are explicit tensor inputs; the caller owns generation/checkpoint restoration.
Only contiguous BF16 operands with dimensions divisible by 128 are supported.
No higher-order derivative or claim of unbiasedness is provided.
"""
import json
import threading
from dataclasses import replace
import torch

from .autotune.posthoc import PosthocPreparationSchedule,check_preparation_copy_alignment
from .configs.nvfp4 import NVFP4ScaleConfig

_FORWARD_POLICIES={mode:NVFP4ScaleConfig(tensor_scale_mode=mode) for mode in ("power2","exact","exact_four_over_six")}


def _contract(dy, operand, signs, uniform_a, uniform_b, phase, schedule_json, packed=False,seeded=False):
    if phase not in ("dx", "dw"):
        raise ValueError("phase must be dx or dw")
    schedule = PosthocPreparationSchedule(**json.loads(schedule_json))
    if schedule.rng_mode=="philox" and not seeded:
        raise ValueError("philox requires ms_eden_seeded_gradient, not caller-owned uniform vectors")
    if schedule.gemm.scale_layout not in ("mma128", "row_major"):
        raise ValueError("experimental gradient operator requires mma128 or row_major scales")
    if schedule.reader_mode != "direct" and not packed:
        raise ValueError("this operator accepts original BF16 operands only")
    for index,value in enumerate((dy, operand)):
        expected_dtype=torch.uint8 if packed and index==1 else torch.bfloat16
        if value.ndim != 2 or value.dtype != expected_dtype or not value.is_contiguous():
            raise ValueError(f"contiguous rank-two {expected_dtype} operand required")
        shape=(value.shape[0],value.shape[1]*(2 if packed and index==1 else 1))
        for size in shape:
            torch._check(size > 0)
            torch._check(size % 128 == 0)
    for value in (operand, signs, uniform_a, uniform_b):
        if value.device != dy.device:
            raise ValueError("all inputs must share a device")
    if signs.dtype != torch.float32 or signs.shape != (128,) or not signs.is_contiguous():
        raise ValueError("signs must be contiguous FP32 [128], containing +/-1")
    if seeded:
        if uniform_a.dtype!=torch.int64 or uniform_a.shape!=(2,) or not uniform_a.is_contiguous():
            raise ValueError("Philox seed must be contiguous int64 [2], containing uint32 key words")
        torch._check(dy.numel()//16<2**31)
        torch._check(operand.numel()//16<2**31)
    else:
        for uniform, count in ((uniform_a,dy.numel()),(uniform_b,operand.numel()*(2 if packed else 1))):
            if uniform.dtype != torch.float32 or uniform.ndim != 1 or not uniform.is_contiguous():
                raise ValueError("uniforms must be contiguous FP32 vectors in [0,1)")
            torch._check(uniform.numel() == count // 16)
    output_cols=operand.shape[1]*(2 if packed else 1)
    if phase == "dx":
        torch._check(dy.shape[1] == operand.shape[0])
        shape = (dy.shape[0], output_cols, dy.shape[1])
    else:
        torch._check(dy.shape[0] == operand.shape[0])
        shape = (dy.shape[1], output_cols, dy.shape[0])
    if schedule.gemm_split_reduction>1:
        schedule.gemm_reduction_tile(shape[2])
    return schedule, shape


@torch.library.custom_op("rtx_experimental::ms_eden_gradient", mutates_args=(), device_types="cuda")
def _gradient(dy: torch.Tensor, operand: torch.Tensor, signs: torch.Tensor,
              uniform_a: torch.Tensor, uniform_b: torch.Tensor,
              phase: str, schedule_json: str) -> torch.Tensor:
    schedule, (m,n,k) = _contract(dy,operand,signs,uniform_a,uniform_b,phase,schedule_json)
    return _execute_gradient(dy,operand,signs,uniform_a,uniform_b,phase,schedule,(m,n,k))


def _execute_gradient(dy,operand,signs,uniform_a,uniform_b,phase,schedule,shape,packed_scales=None,packed_outer=None,seed=None):
    launch,out,_ = _build_gradient(dy,operand,signs,uniform_a,uniform_b,phase,schedule,shape,
                                  packed_scales,packed_outer,seed)
    launch()
    return out


class _GradientLaunchPlan:
    """Invocation-owned kernel-only stages; calling preserves serial ordering."""
    __slots__=("prepare_a","prepare_b","finish","output")

    def __init__(self,prepare_a,prepare_b,finish,*,output=None):
        self.prepare_a,self.prepare_b,self.finish=prepare_a,prepare_b,finish
        self.output=output

    def __call__(self):
        self.prepare_a();self.prepare_b();self.finish()


class _PreparationLaunchPlan:
    """Invocation-owned producer/finalizer split; never cached tensor state.

    Ordinary execution preserves the existing sequence. Explicit descriptors
    allow research fusion to replace production without closure introspection,
    allocation or eager Torch arithmetic inside the launch sequence.
    """
    __slots__=("produce","finalize","source","signs","packed","temporary",
               "partials","source_shape","transpose","config","schedule","original_bf16",
               "finalizer_spec","finalizer_args")

    def __init__(self,produce,finalize,*,source,signs,packed,temporary,partials,
                 source_shape,transpose,config,schedule,original_bf16,
                 finalizer_spec=None,finalizer_args=None):
        self.produce,self.finalize=produce,finalize
        self.source,self.signs=source,signs
        self.packed,self.temporary,self.partials=packed,temporary,partials
        self.source_shape,self.transpose=source_shape,transpose
        self.config,self.schedule=config,schedule
        self.original_bf16=original_bf16
        self.finalizer_spec,self.finalizer_args=finalizer_spec,finalizer_args

    def __call__(self):
        self.produce();self.finalize()


def _build_gradient(dy,operand,signs,uniform_a,uniform_b,phase,schedule,shape,packed_scales=None,packed_outer=None,seed=None):
    """Allocate/compile on the caller stream; return a kernel-only launch chain."""
    m,n,k=shape
    reduction_tile=schedule.gemm_reduction_tile(k)
    live=[dy,operand,signs,uniform_a,uniform_b]
    from .fp4 import compile_nvfp4_gemm, _packed_fp4_view, _check_sm12x
    from .configs.nvfp4 import NVFP4Problem, NVFP4GemmConfig
    from .kernels.nvfp4_posthoc import (compile_hadamard_preparation,
        compile_posthoc_scales, compile_posthoc_scale_product, PosthocScaleConfig)
    with torch.cuda.device(dy.device):
        _check_sm12x(dy.device)
        factors=torch.empty(2,device=dy.device,dtype=torch.float32) if schedule.outer_factor_mode=="pair" else None
        factor_options={"outer_factor_mode":"pair"} if factors is not None else {}
        if factors is not None:
            live.append(factors)
        def allocate(source, transpose, uniforms, source_scales=None, source_outer=None,operand_stream=0):
            geometry=schedule.operand_schedule("a" if operand_stream==0 else "b")
            if source_scales is None:
                check_preparation_copy_alignment(source,geometry,transpose)
            r,c = source.shape
            is_packed=source_scales is not None
            if is_packed:
                c*=2
            rows,cols = (c,r) if transpose else (r,c)
            config=PosthocScaleConfig(geometry.producer_threads,geometry.producer_blocks)
            decode_runner=None
            if is_packed and schedule.reader_mode=="direct":
                from .kernels.nvfp4_posthoc import compile_packed_hadamard_preparation
                producer,count=compile_packed_hadamard_preparation(r,c,config,4,transpose,schedule.producer_order,schedule.producer_load,schedule.producer_signs,schedule.producer_stages,schedule.producer_async_copy,correction_decode=schedule.correction_decode,partial_max_scope=schedule.partial_max_scope,normalization_math=schedule.normalization_math,producer_copy_bits=schedule.producer_copy_bits,producer_unroll_rows=schedule.producer_unroll_rows,producer_row_tile_batch=schedule.producer_row_tile_batch,producer_row_load_bits=schedule.producer_row_load_bits,producer_values_per_lane=schedule.producer_values_per_lane,producer_transpose_values_per_lane=schedule.producer_transpose_values_per_lane,producer_scale_math=schedule.producer_scale_math,producer_grouped_store_bits=schedule.producer_grouped_store_bits)
            else:
                producer,count = compile_hadamard_preparation(r,c,config,4,transpose,schedule.producer_order,schedule.producer_load,schedule.producer_signs,schedule.producer_stages,schedule.producer_async_copy,correction_decode=schedule.correction_decode,partial_max_scope=schedule.partial_max_scope,normalization_math=schedule.normalization_math,producer_copy_bits=schedule.producer_copy_bits,producer_unroll_rows=schedule.producer_unroll_rows,producer_row_tile_batch=schedule.producer_row_tile_batch,producer_row_load_bits=schedule.producer_row_load_bits,producer_values_per_lane=schedule.producer_values_per_lane,producer_transpose_values_per_lane=schedule.producer_transpose_values_per_lane,producer_scale_math=schedule.producer_scale_math,producer_grouped_store_bits=schedule.producer_grouped_store_bits)
                if is_packed:
                    from .kernels.nvfp4_posthoc import compile_packed_bf16_decode
                    decode_runner=compile_packed_bf16_decode(r,c,
                        PosthocScaleConfig(schedule.decoder_threads,schedule.decoder_blocks))
            reducer=None
            if schedule.amax_reduction == "staged":
                from .kernels.nvfp4_posthoc import compile_partial_maximum
                reducer=compile_partial_maximum(count)
            finalizer_config=PosthocScaleConfig(geometry.finalizer_threads,geometry.finalizer_blocks,schedule.gemm.scale_layout,probability_math=schedule.probability_math)
            if seed is None:
                finalizer=compile_posthoc_scales(rows,cols//16,finalizer_config,1 if reducer else count)
            else:
                from .kernels.nvfp4_rng import compile_philox_posthoc_scales
                finalizer=compile_philox_posthoc_scales(rows,cols//16,finalizer_config,
                                                       1 if reducer else count,operand_stream,
                                                       schedule.philox_packet_size)
            packed = torch.empty(rows*cols//2,device=dy.device,dtype=torch.uint8)
            temporary = torch.empty(rows*cols//16,device=dy.device,dtype=torch.bfloat16)
            partials = torch.empty(count,device=dy.device,dtype=torch.float32)
            maximum = torch.empty(1,device=dy.device,dtype=torch.float32) if reducer else partials
            scales = torch.empty(rows*cols//16,device=dy.device,dtype=torch.float8_e4m3fn)
            outer = (factors[operand_stream:operand_stream+1] if factors is not None
                     else torch.empty(1,device=dy.device,dtype=torch.float32))
            src = source.view(-1)
            if is_packed:
                sf=source_scales.view(-1)
                outer_source=source_outer.view(1)
                decoded=torch.empty(r*c,device=dy.device,dtype=torch.bfloat16) if decode_runner else None
                if decoded is not None:
                    check_preparation_copy_alignment(decoded.view(r,c),geometry,transpose)
            packed_view = _packed_fp4_view(packed.view(rows,cols//2))
            scale_view = (scales.view(rows//128,cols//128,1024)
                          if schedule.gemm.scale_layout == "mma128"
                          else scales.view(rows,cols//16))
            live.extend((packed,temporary,partials,maximum,scales,outer))
            if is_packed:
                live.extend((source_scales,source_outer))
                if decoded is not None:
                    live.append(decoded)
            def produce():
                if is_packed:
                    if decode_runner:
                        decode_runner(src,sf,outer_source,decoded)
                        producer(decoded,signs,packed,temporary,partials)
                    else:
                        producer(src,sf,outer_source,signs,packed,temporary,partials)
                else:
                    producer(src,signs,packed,temporary,partials)
            def finalize():
                if reducer:
                    reducer(partials,maximum)
                finalizer(temporary,maximum,uniforms if seed is None else seed,scales,outer)
            run=_PreparationLaunchPlan(produce,finalize,source=src,signs=signs,
                packed=packed,temporary=temporary,partials=partials,source_shape=(r,c),
                transpose=transpose,config=config,schedule=schedule,original_bf16=not is_packed,
                finalizer_spec=(rows,cols//16,finalizer_config),
                finalizer_args=(temporary,partials,maximum,uniforms,scales,outer) if seed is None else None)
            return run,packed_view,scale_view,outer
        a, b = (allocate(dy,phase=="dw",uniform_a,operand_stream=0),
                allocate(operand,True,uniform_b,packed_scales,packed_outer,operand_stream=1))
        alpha = factors if factors is not None else torch.empty(1,device=dy.device,dtype=torch.float32)
        out = torch.empty(m,n,device=dy.device,dtype=torch.bfloat16)
        product = compile_posthoc_scale_product() if factors is None else None
        workspace=None
        gemm_reducer=None
        if schedule.gemm_split_reduction>1:
            from .kernels.nvfp4_gemm import compile_nvfp4_split_gemm,compile_nvfp4_split_workspace_reduce
            workspace=torch.empty(schedule.gemm_split_reduction*m*n,device=dy.device,dtype=torch.float32)
            gemm=compile_nvfp4_split_gemm(NVFP4Problem(m,n,k),schedule.gemm,
                split_reduction=schedule.gemm_split_reduction,reduction_tile=reduction_tile)
            gemm_reducer=compile_nvfp4_split_workspace_reduce(m,n,schedule.gemm_split_reduction,
                algorithm=schedule.gemm_reduction_algorithm,threads=schedule.gemm_reduction_threads,
                vector=schedule.gemm_reduction_vector,persistent_waves=schedule.gemm_reduction_waves,**factor_options)
            live.append(workspace)
        else:
            gemm = compile_nvfp4_gemm(NVFP4Problem(m,n,k),schedule.gemm,**factor_options)
        # All metadata, allocations and compilation precede this launch chain.
        live.extend((alpha,out))
        def finish():
            if product is not None:
                product(a[3],b[3],alpha)
            if workspace is not None:
                gemm(a[1],b[1],a[2],b[2],workspace)
                gemm_reducer(workspace,out,alpha)
            else:
                gemm(a[1],b[1],a[2],b[2],out,alpha)
        return _GradientLaunchPlan(a[0],b[0],finish,output=out),out,live


@_gradient.register_fake
def _gradient_fake(dy,operand,signs,uniform_a,uniform_b,phase,schedule_json):
    _,(m,n,_) = _contract(dy,operand,signs,uniform_a,uniform_b,phase,schedule_json)
    return dy.new_empty((m,n))


def _no_higher_order(ctx, *grads):
    raise RuntimeError("experimental MS-EDEN gradients do not support higher-order autograd")


_gradient.register_autograd(_no_higher_order)


@torch.library.custom_op("rtx_experimental::ms_eden_seeded_gradient",mutates_args=(),device_types="cuda")
def _seeded_gradient(dy:torch.Tensor,operand:torch.Tensor,signs:torch.Tensor,seed:torch.Tensor,
                     phase:str,schedule_json:str)->torch.Tensor:
    schedule,shape=_contract(dy,operand,signs,seed,seed,phase,schedule_json,seeded=True)
    return _execute_gradient(dy,operand,signs,seed,seed,phase,schedule,shape,seed=seed)


@_seeded_gradient.register_fake
def _seeded_gradient_fake(dy,operand,signs,seed,phase,schedule_json):
    _,(m,n,_)=_contract(dy,operand,signs,seed,seed,phase,schedule_json,seeded=True)
    return dy.new_empty((m,n))


_seeded_gradient.register_autograd(_no_higher_order)


_PAIRED_STREAMS = {}
_PAIRED_STREAMS_LOCK = threading.Lock()
_PAIRED_STREAMS_PER_DEVICE = 16
_OPERAND_STREAMS = {}
_OPERAND_STREAMS_LOCK = threading.Lock()
_OPERAND_STREAMS_PER_DEVICE = 16


def _paired_stream(device,caller):
    # No tensors, RNG, outputs or events are cached. Each caller-stream identity
    # owns a side stream, so independent graph captures never share a fork.
    key=(device.index,caller.cuda_stream)
    with _PAIRED_STREAMS_LOCK:
        entry=_PAIRED_STREAMS.get(key)
        if entry is None:
            if sum(k[0]==device.index for k in _PAIRED_STREAMS)>=_PAIRED_STREAMS_PER_DEVICE:
                return None  # Bounded cache: safely use the serial launch path.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm paired MS-EDEN on this caller stream before CUDA graph capture")
            entry=(torch.cuda.Stream(device=device),threading.Lock())
            _PAIRED_STREAMS[key]=entry
        return entry


def _operand_streams(device,caller):
    """Only stream identities/lock persist; graph storage and events never do."""
    key=(device.index,caller.cuda_stream)
    with _OPERAND_STREAMS_LOCK:
        entry=_OPERAND_STREAMS.get(key)
        if entry is None:
            if sum(k[0]==device.index for k in _OPERAND_STREAMS)>=_OPERAND_STREAMS_PER_DEVICE:
                return None
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm operand_paired MS-EDEN on this caller stream before CUDA graph capture")
            entry=(tuple(torch.cuda.Stream(device=device) for _ in range(3)),threading.Lock())
            _OPERAND_STREAMS[key]=entry
        return entry


def _paired_contract(dy,x,weight,dx_signs,dx_a,dx_b,dw_signs,dw_a,dw_b,
                     dx_schedule_json,dw_schedule_json,dx_seeded,dw_seeded,pipeline):
    if pipeline not in ("serial","paired","operand_paired"):
        raise ValueError("pipeline must be serial, paired, or operand_paired")
    dx=_contract(dy,weight,dx_signs,dx_a,dx_b,"dx",dx_schedule_json,seeded=dx_seeded)
    dw=_contract(dy,x,dw_signs,dw_a,dw_b,"dw",dw_schedule_json,seeded=dw_seeded)
    torch._check(x.shape[1]==weight.shape[1])
    return dx,dw


def _launch_paired(run_dx,run_dw,live_dw,device):
    caller=torch.cuda.current_stream(device)
    entry=_paired_stream(device,caller)
    if entry is None:
        run_dx(); run_dw()
        return
    side,lock=entry
    with lock:
        # wait_stream uses invocation-local pooled events; never reuse
        # a mutable event across callers or graph capture sessions.
        side.wait_stream(caller)
        try:
            for value in live_dw:
                value.record_stream(side)
            with torch.cuda.stream(side):
                run_dw()
            run_dx()
        finally:
            # Also join on launch failure before any storage can die.
            caller.wait_stream(side)


def _launch_operand_paired(dx,dw,live_dx,live_dw,device):
    caller=torch.cuda.current_stream(device)
    entry=_operand_streams(device,caller)
    if entry is None:
        dx();dw()
        return
    sides,lock=entry
    dx_b,dw_a,dw_b=sides
    with lock:
        try:
            # Each wait records an invocation-local pooled event at the input/
            # RNG frontier, before any caller-stream joins can create cycles.
            for side in sides:side.wait_stream(caller)
            for value in live_dx:value.record_stream(dx_b)
            for side in (dw_a,dw_b):
                for value in live_dw:value.record_stream(side)
            with torch.cuda.stream(dx_b):dx.prepare_b()
            with torch.cuda.stream(dw_a):dw.prepare_a()
            with torch.cuda.stream(dw_b):dw.prepare_b()
            dx.prepare_a()
            caller.wait_stream(dx_b)
            dx.finish()
            with torch.cuda.stream(dw_a):
                dw_a.wait_stream(dw_b)
                dw.finish()
        except BaseException:
            # Attempt every join even if one fails, and retain the launch error.
            for side in sides:
                try:caller.wait_stream(side)
                except BaseException:pass
            raise
        try:
            # Transitively joins DW-B as well; DX-B was joined before DX finish.
            caller.wait_stream(dw_a)
        except BaseException:
            for side in (dx_b,dw_b):
                try:caller.wait_stream(side)
                except BaseException:pass
            raise


@torch.library.custom_op("rtx_experimental::ms_eden_paired_gradient",mutates_args=(),device_types="cuda")
def _paired_gradient(dy:torch.Tensor,x:torch.Tensor,weight:torch.Tensor,
                     dx_signs:torch.Tensor,dx_a:torch.Tensor,dx_b:torch.Tensor,
                     dw_signs:torch.Tensor,dw_a:torch.Tensor,dw_b:torch.Tensor,
                     dx_schedule_json:str,dw_schedule_json:str,dx_seeded:bool,dw_seeded:bool,
                     pipeline:str)->tuple[torch.Tensor,torch.Tensor]:
    dx,dw=_paired_contract(dy,x,weight,dx_signs,dx_a,dx_b,dw_signs,dw_a,dw_b,
                          dx_schedule_json,dw_schedule_json,dx_seeded,dw_seeded,pipeline)
    with torch.cuda.device(dy.device):
        # Build both legs before forking: no eager Torch arithmetic or allocation
        # is hidden in a launch closure, and all storage belongs to caller.
        run_dx,out_dx,live_dx=_build_gradient(dy,weight,dx_signs,dx_a,dx_b,"dx",*dx,
                                       seed=dx_a if dx_seeded else None)
        run_dw,out_dw,live_dw=_build_gradient(dy,x,dw_signs,dw_a,dw_b,"dw",*dw,
                                             seed=dw_a if dw_seeded else None)
        if pipeline=="serial":
            run_dx(); run_dw()
        elif pipeline=="operand_paired":
            _launch_operand_paired(run_dx,run_dw,live_dx,live_dw,dy.device)
        else:
            _launch_paired(run_dx,run_dw,live_dw,dy.device)
        return out_dx,out_dw


@_paired_gradient.register_fake
def _paired_gradient_fake(dy,x,weight,dx_signs,dx_a,dx_b,dw_signs,dw_a,dw_b,
                          dx_schedule_json,dw_schedule_json,dx_seeded,dw_seeded,pipeline):
    dx,dw=_paired_contract(dy,x,weight,dx_signs,dx_a,dx_b,dw_signs,dw_a,dw_b,
                          dx_schedule_json,dw_schedule_json,dx_seeded,dw_seeded,pipeline)
    return dy.new_empty(dx[1][:2]),dy.new_empty(dw[1][:2])


_paired_gradient.register_autograd(_no_higher_order)


def ms_eden_paired_gradient(dy,x,weight,dx_signs,dx_a,dx_b,dw_signs,dw_a,dw_b,*,
                            dx_schedule_json="{}",dw_schedule_json="{}",
                            dx_seeded=False,dw_seeded=False,pipeline="serial"):
    """Explicit original-BF16 dX/dW, with independent signs and rounding RNG.

    Each leg accepts the same uniforms as :func:`ms_eden_gradient`. For a
    seeded leg, pass its int64[2] key as both a/b and set that leg's seeded
    flag, matching :func:`ms_eden_seeded_gradient`. No RNG is generated here.
    ``paired`` forks dW onto a reusable side stream and joins before return;
    ``operand_paired`` forks both operands of both legs across caller/three
    side streams, joining each GEMM to its two prepared inputs before return.
    ``serial`` is the numerical/control baseline. Both keep BF16 outputs and
    the unchanged FP32 kernel arithmetic. Warm on the actual caller stream
    before graph capture. The cache is bounded to 16 caller streams/device;
    further streams use serial execution. No default performance claim.
    """
    return _paired_gradient(dy,x,weight,dx_signs,dx_a,dx_b,dw_signs,dw_a,dw_b,
                            dx_schedule_json,dw_schedule_json,dx_seeded,dw_seeded,pipeline)


def _packed_contract(dy,data,scales,outer,signs,ua,ub,phase,schedule_json):
    schedule,shape=_contract(dy,data,signs,ua,ub,phase,schedule_json,packed=True)
    if (scales.dtype != torch.uint8 or not scales.is_contiguous()
            or outer.dtype != torch.float32 or outer.numel()!=1 or not outer.is_contiguous()):
        raise ValueError("packed operand requires contiguous E4M3 scales and one FP32 outer scale")
    if scales.device != dy.device or outer.device != dy.device:
        raise ValueError("packed scales must share the operand device")
    torch._check(scales.numel() == data.numel()//8)
    return schedule,shape


@torch.library.custom_op("rtx_experimental::ms_eden_packed_gradient",mutates_args=(),device_types="cuda")
def _packed_gradient(dy:torch.Tensor,data:torch.Tensor,scales:torch.Tensor,outer:torch.Tensor,
                     signs:torch.Tensor,ua:torch.Tensor,ub:torch.Tensor,
                     phase:str,schedule_json:str)->torch.Tensor:
    schedule,shape=_packed_contract(dy,data,scales,outer,signs,ua,ub,phase,schedule_json)
    return _execute_gradient(dy,data,signs,ua,ub,phase,schedule,shape,scales.view(torch.float8_e4m3fn),outer)


@_packed_gradient.register_fake
def _packed_gradient_fake(dy,data,scales,outer,signs,ua,ub,phase,schedule_json):
    _,(m,n,_)=_packed_contract(dy,data,scales,outer,signs,ua,ub,phase,schedule_json)
    return dy.new_empty((m,n))


_packed_gradient.register_autograd(_no_higher_order)


def _pack_forward_contract(x,w,xs,ws):
    for value in (x,w):
        if value.ndim!=2 or value.dtype!=torch.bfloat16 or not value.is_contiguous():
            raise ValueError("contiguous rank-two BF16 inputs required")
        for size in value.shape:
            torch._check(size>0)
            torch._check(size%128==0)
    torch._check(x.shape[1]==w.shape[1])
    for value in (w,xs,ws):
        if value.device!=x.device:
            raise ValueError("packed forward inputs must share a device")
    for value in (xs,ws):
        if value.dtype!=torch.float32 or value.shape!=(1,) or not value.is_contiguous():
            raise ValueError("outer scales must be contiguous FP32 [1]")


@torch.library.custom_op("rtx_experimental::ms_eden_pack_forward",mutates_args=(),device_types="cuda")
def _pack_forward(x:torch.Tensor,w:torch.Tensor,xs:torch.Tensor,ws:torch.Tensor
                  )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    _pack_forward_contract(x,w,xs,ws)
    from .fp4 import compile_nvfp4_quant,compile_nvfp4_gemm,_packed_fp4_view,_check_sm12x
    from .configs.nvfp4 import DEFAULT_NVFP4_QUANT_CONFIG,NVFP4GemmConfig,NVFP4Problem
    from .kernels.nvfp4_posthoc import compile_posthoc_scale_product
    m,k=x.shape; n=w.shape[0]
    with torch.cuda.device(x.device):
        _check_sm12x(x.device)
        qx=torch.empty(m,k//2,device=x.device,dtype=torch.uint8)
        qw=torch.empty(n,k//2,device=x.device,dtype=torch.uint8)
        sx=torch.empty(m,k//16,device=x.device,dtype=torch.float8_e4m3fn)
        sw=torch.empty(n,k//16,device=x.device,dtype=torch.float8_e4m3fn)
        alpha=torch.empty(1,device=x.device,dtype=torch.float32)
        out=torch.empty(m,n,device=x.device,dtype=torch.bfloat16)
        quant_x=compile_nvfp4_quant(m,k,DEFAULT_NVFP4_QUANT_CONFIG)
        quant_w=compile_nvfp4_quant(n,k,DEFAULT_NVFP4_QUANT_CONFIG)
        product=compile_posthoc_scale_product()
        gemm=compile_nvfp4_gemm(NVFP4Problem(m,n,k),NVFP4GemmConfig())
        px,pw=_packed_fp4_view(qx),_packed_fp4_view(qw)
        quant_x(x,qx,sx,xs); quant_w(w,qw,sw,ws)
        product(xs,ws,alpha)
        gemm(px,pw,sx,sw,out,alpha)
        return out,qx,sx.view(torch.uint8),qw,sw.view(torch.uint8)


@_pack_forward.register_fake
def _pack_forward_fake(x,w,xs,ws):
    _pack_forward_contract(x,w,xs,ws)
    m,k=x.shape; n=w.shape[0]
    return (x.new_empty((m,n)),x.new_empty((m,k//2),dtype=torch.uint8),
            x.new_empty((m,k//16),dtype=torch.uint8),x.new_empty((n,k//2),dtype=torch.uint8),
            x.new_empty((n,k//16),dtype=torch.uint8))


_pack_forward.register_autograd(_no_higher_order)


def ms_eden_gradient(dy, operand, signs, uniform_a, uniform_b, *, phase, schedule_json="{}"):
    """Explicit experimental dX or dW; output is BF16, accumulation FP32.

    ``phase='dx'`` computes dy @ W; ``phase='dw'`` computes dy.T @ X.
    Original BF16 operands are quantized using shared-sign Hadamard-128,
    E8M4 RTZ temporary scales, EDEN correction and stochastic E4M3 scales.
    Provide one independent uniform per 16 values in each *logical* operand,
    and the same 128 +/-1 signs for both. Contents are caller preconditions;
    no synchronizing host validation or random generation occurs here.

    ``schedule_json`` is a static serialized PosthocPreparationSchedule.
    Warm kernels before CUDA graph capture. This opaque operator permits user
    torch.compile orchestration; it does not capture an internal CUDA graph.
    """
    return _gradient(dy,operand,signs,uniform_a,uniform_b,phase,schedule_json)


def ms_eden_seeded_gradient(dy,operand,signs,seed,*,phase,schedule_json="{}"):
    """Experimental original-BF16 gradient with explicit Philox scale RNG.

    Seed is contiguous int64[2], each word in [0,2**32); contents are a caller
    precondition (no device-to-host validation). Philox4x32-10 uses counters
    (logical_scale_index//4, operand_stream, 0, 0), streams 0/1 for operands.
    Its top24-bit uniforms use an exact 24-bit [0,1) grid. The intended uniform
    stochastic-rounding rule is unchanged, but neither the sample sequence nor
    backend-specific torch.rand discretization is promised identical. No
    persistent or hidden RNG state is used.
    The explicit-seed API determines RNG behavior regardless of rng_mode in
    schedule_json; the field selects behavior only for ms_eden_linear.
    """
    return _seeded_gradient(dy,operand,signs,seed,phase,schedule_json)


@torch._dynamo.assume_constant_result
def _schedule_rng_mode(schedule_json):
    return PosthocPreparationSchedule(**json.loads(schedule_json)).rng_mode


def _random_gradient(dy, operand, phase, schedule_json, packed=None):
    # These Torch operations live in the traceable autograd formula, outside
    # the opaque kernel operator. AOTAutograd/Inductor can lower/fuse them.
    rng_mode=_schedule_rng_mode(schedule_json)
    if rng_mode=="philox" and packed is not None:
        raise ValueError("philox scale RNG currently requires original_bf16 sources")
    signs = torch.where(torch.rand(128,device=dy.device) < .5, -1., 1.)
    if rng_mode=="philox":
        seed=torch.randint(0,2**32,(2,),device=dy.device,dtype=torch.int64)
        return ms_eden_seeded_gradient(dy,operand,signs,seed,phase=phase,schedule_json=schedule_json)
    ua = torch.rand(dy.numel()//16,device=dy.device,dtype=torch.float32)
    ub = torch.rand(operand.numel()*(2 if packed is not None else 1)//16,device=dy.device,dtype=torch.float32)
    if packed is not None:
        return _packed_gradient(dy,operand,packed[0],packed[1],signs,ua,ub,phase,schedule_json)
    return ms_eden_gradient(dy,operand,signs,ua,ub,phase=phase,schedule_json=schedule_json)


def _random_paired_gradient(dy,x,weight,dx_schedule_json,dw_schedule_json,pipeline):
    # Preserve serial dX-then-dW draw order and independent per-leg policies.
    def draw(operand,schedule_json):
        signs=torch.where(torch.rand(128,device=dy.device)<.5,-1.,1.)
        seeded=_schedule_rng_mode(schedule_json)=="philox"
        if seeded:
            a=torch.randint(0,2**32,(2,),device=dy.device,dtype=torch.int64)
            b=a
        else:
            a=torch.rand(dy.numel()//16,device=dy.device,dtype=torch.float32)
            b=torch.rand(operand.numel()//16,device=dy.device,dtype=torch.float32)
        return signs,a,b,seeded
    ds,da,db,dseed=draw(weight,dx_schedule_json)
    ws,wa,wb,wseed=draw(x,dw_schedule_json)
    return ms_eden_paired_gradient(dy,x,weight,ds,da,db,ws,wa,wb,
        dx_schedule_json=dx_schedule_json,dw_schedule_json=dw_schedule_json,
        dx_seeded=dseed,dw_seeded=wseed,pipeline=pipeline)


class _ExperimentalLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, schedule_json, source_policy,forward_outer_mode,
                forward_scaling,forward_config,dx_schedule_json,dw_schedule_json,
                dx_policy,dw_policy,mxfp8_config_key,backward_pipeline):
        from .fp4 import nvfp4_linear,_outer_tensor_scale
        ctx.schedule_json = schedule_json
        ctx.dx_schedule_json = dx_schedule_json
        ctx.dw_schedule_json = dw_schedule_json
        ctx.source_policy = source_policy
        ctx.dx_policy,ctx.dw_policy = dx_policy,dw_policy
        ctx.mxfp8_config_key = mxfp8_config_key
        ctx.backward_pipeline = backward_pipeline
        if source_policy == "forward_current_roundtrip_bf16":
            xs,ws=_outer_tensor_scale(x,forward_outer_mode),_outer_tensor_scale(weight,forward_outer_mode)
            out,qx,sx,qw,sw=_pack_forward(x,weight,xs,ws)
            # Detached plain tensors sever Python _base_inputs annotations on
            # quantizer outputs; no original-BF16 operand is saved here.
            ctx.save_for_backward(qx.detach(),sx.detach(),xs.detach(),qw.detach(),sw.detach(),ws.detach())
            return out
        ctx.save_for_backward(x,weight)
        if forward_scaling == "jit_row_region":
            from .fp4 import _nvfp4_dynamic_linear_with_keys
            return _nvfp4_dynamic_linear_with_keys(x,weight,forward_key="",backward_key="",
                tensor_scale_mode=forward_outer_mode,region_geometry_explicit=False,
                jit_row_region=True,materialized_request_key=forward_config)
        return nvfp4_linear(x,weight,scaling=forward_scaling,autotune="off",
            scale_config=_FORWARD_POLICIES[forward_outer_mode],dynamic_config=forward_config)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        if ctx.source_policy == "forward_current_roundtrip_bf16":
            x,xs,xo,weight,ws,wo=ctx.saved_tensors
            dx=(_random_gradient(dy,weight,"dx",ctx.dx_schedule_json,(ws,wo))
                if ctx.needs_input_grad[0] else None)
            dw=(_random_gradient(dy,x,"dw",ctx.dw_schedule_json,(xs,xo))
                if ctx.needs_input_grad[1] else None)
            return (dx,dw)+(None,)*11
        x,weight = ctx.saved_tensors
        if (ctx.backward_pipeline in ("paired","operand_paired") and all(ctx.needs_input_grad[:2])
                and ctx.dx_policy==ctx.dw_policy=="ms_eden"):
            dx,dw=_random_paired_gradient(dy,x,weight,ctx.dx_schedule_json,
                                          ctx.dw_schedule_json,ctx.backward_pipeline)
            return (dx,dw)+(None,)*11
        dx = (_linear_gradient(dy,x,weight,"dx",ctx.dx_policy,ctx.dx_schedule_json,ctx.mxfp8_config_key)
              if ctx.needs_input_grad[0] else None)
        dw = (_linear_gradient(dy,x,weight,"dw",ctx.dw_policy,ctx.dw_schedule_json,ctx.mxfp8_config_key)
              if ctx.needs_input_grad[1] else None)
        return (dx,dw)+(None,)*11


def _linear_gradient(dy,x,weight,phase,policy,schedule_json,mxfp8_config_key):
    if policy == "ms_eden":
        return _random_gradient(dy,weight if phase=="dx" else x,phase,schedule_json)
    if policy != "mxfp8":
        raise ValueError("unknown experimental backward policy")
    from .fp8_bwd import _mxfp8_linear_dx_op,_mxfp8_linear_dw_op
    op=_mxfp8_linear_dx_op if phase=="dx" else _mxfp8_linear_dw_op
    return op(dy,x,weight,mxfp8_config_key)


def _jit_forward_config(m,n,k,forward_config,mode):
    if forward_config is None:
        from .configs.nvfp4 import NVFP4Problem
        from .nvfp4_inference_autotune import preferred_jit_row_region_config
        forward_config=preferred_jit_row_region_config(NVFP4Problem(m,n,k))
    return replace(forward_config,tensor_scale_mode=mode)


@torch._dynamo.assume_constant_result
def _jit_forward_request(m,n,k,forward_config,mode):
    from .fp4 import _intern_autotune_request
    forward_config=_jit_forward_config(m,n,k,forward_config,mode)
    return _intern_autotune_request("off",None,None,dynamic=forward_config)


def _resolve_outer_mode(source_policy,forward_outer_mode,forward_scaling,forward_config):
    explicit=forward_outer_mode is not None
    if forward_outer_mode is None:
        forward_outer_mode=(forward_config.tensor_scale_mode
            if forward_scaling=="jit_row_region" and forward_config is not None else
            "exact" if source_policy=="forward_current_roundtrip_bf16" else "power2")
    supported=("exact","power2") if explicit or forward_scaling!="jit_row_region" else ("exact","power2","exact_four_over_six")
    if forward_outer_mode not in supported:
        raise ValueError("forward_outer_mode must be exact or power2")
    return forward_outer_mode


def ms_eden_linear(x, weight, *, schedule_json="{}",source_policy="original_bf16",forward_outer_mode=None,
                   forward_scaling="current",forward_config=None,
                   dx_schedule_json=None,dw_schedule_json=None,
                   dx_policy="ms_eden",dw_policy="ms_eden",mxfp8_backward_config=None,
                   backward_pipeline="serial"):
    """Research-only NVFP4 forward with MS-EDEN FP4 backward.

    By default original contiguous rank-two BF16 X/W are saved. The explicit
    forward_current_roundtrip_bf16 policy instead saves FP4 bytes, E4M3 scale
    bytes and FP32 outer scales, and computes gradients from BF16-rounded
    decoded forward operands. Explicit forward_outer_mode='exact' or 'power2'
    permits matched forward-policy comparisons. None preserves earlier defaults:
    power2 for original sources and exact for packed sources.
    For JIT-region forward, an explicit mode overrides the supplied config mode;
    with no explicit mode, a supplied config's mode is retained. The resolved
    mode is pinned in the dynamic config consumed by the regional quantizer.
    Original-source mode supports current, jit_row_region, or block forward
    with an explicit NVFP4DynamicConfig. Packed-source mode retains only the
    fixed tensor-current forward: regional packed scale reuse is not implemented.
    Independent dx_schedule_json/dw_schedule_json override the shared schedule;
    these alter execution, not the gradient numerical policy.
    Each schedule may set a_geometry (dY) or b_geometry (W for dX, X for dW)
    with producer_threads/blocks and finalizer_threads/blocks overrides only;
    omitted geometry fields inherit the corresponding shared leg setting.
    Stateful delayed forward is not supported by this stateless function.
    Independently select dx_policy/dw_policy='ms_eden' (FP4) or 'mxfp8'.
    MXFP8 uses the existing deterministic default partial-backward kernels,
    consumes no RNG, and currently requires original_bf16 source policy.
    mxfp8_backward_config optionally supplies an explicit MXFP8BwdConfig shared
    by MXFP8 phases (including its independent dx/dw matmul configurations).
    No online tuning or implicit environment selection occurs in this API.
    Schedules apply only to MS-EDEN phases. Unrequested gradients do no work
    and consume no RNG, regardless of their selected policy.
    Opt-in backward_pipeline='paired' overlaps the two original-BF16 MS-EDEN
    legs on caller/side streams without changing their independent RNG draws.
    'operand_paired' additionally overlaps each leg's two operand preparations
    on three bounded, caller-owned side streams; it changes no kernel math.
    Mixed-policy, packed-source and partial-gradient calls retain serial paths.
    Warm on the capture stream before CUDA graph capture. Serial is the default;
    paired execution is experimental and not a qualified performance win.
    All dimensions must be
    positive multiples of 128. Fresh shared signs and independent scale uniforms
    are generated for each requested gradient using PyTorch's RNG, outside the
    opaque gradient operator. RNG generation cost is part of training execution.
    Opt-in schedule rng_mode='philox' instead draws a tiny int64[2] key tensor
    outside the opaque operation and generates scale uniforms in the finalizer.
    Signs still use PyTorch RNG. The same rounding probability expression uses
    a documented 24-bit uniform grid; random sample sequence and consumption
    change relative to explicit
    uniforms. It currently supports original_bf16 sources only; checkpoint RNG
    preservation applies to the caller's key/sign draws, not mutable kernel state.
    This is not an unbiasedness guarantee: sparse-input conditional error is
    documented in operand_preparation_plan.md. Not a default or stable API.
    """
    if backward_pipeline not in ("serial","paired","operand_paired"):
        raise ValueError("backward_pipeline must be serial, paired, or operand_paired")
    if source_policy not in ("original_bf16","forward_current_roundtrip_bf16"):
        raise ValueError("unknown experimental source policy")
    if dx_policy not in ("ms_eden","mxfp8") or dw_policy not in ("ms_eden","mxfp8"):
        raise ValueError("dx_policy and dw_policy must be ms_eden or mxfp8")
    if source_policy != "original_bf16" and (dx_policy != "ms_eden" or dw_policy != "ms_eden"):
        raise ValueError("MXFP8 research backward requires original_bf16 source policy")
    if source_policy != "original_bf16" and any(_schedule_rng_mode(value)=="philox" for value in
            (schedule_json if dx_schedule_json is None else dx_schedule_json,
             schedule_json if dw_schedule_json is None else dw_schedule_json)):
        raise ValueError("philox scale RNG currently requires original_bf16 sources")
    if forward_scaling not in ("current","jit_row_region","block"):
        raise ValueError("experimental forward_scaling must be current, jit_row_region, or block")
    if source_policy == "forward_current_roundtrip_bf16" and (forward_scaling != "current" or forward_config is not None):
        raise ValueError("packed-source backward requires fixed tensor-current forward")
    for value in (x,weight):
        if value.ndim != 2 or value.dtype != torch.bfloat16 or not value.is_contiguous():
            raise ValueError("experimental linear requires contiguous rank-two BF16 inputs")
        for size in value.shape:
            torch._check(size > 0)
            torch._check(size % 128 == 0)
    torch._check(x.shape[1] == weight.shape[1])
    forward_outer_mode=_resolve_outer_mode(source_policy,forward_outer_mode,forward_scaling,forward_config)
    if forward_scaling=="jit_row_region":
        forward_config=_jit_forward_request(x.shape[0],weight.shape[0],x.shape[1],forward_config,forward_outer_mode)
    mxfp8_key=None
    if "mxfp8" in (dx_policy,dw_policy):
        from .fp8_bwd import _default_bwd_config,_intern_bwd_config
        from .kernels.mxfp8 import MXFP8Problem
        config=mxfp8_backward_config
        if config is None:
            config=_default_bwd_config(MXFP8Problem(x.shape[0],weight.shape[0],x.shape[1]))
        mxfp8_key=_intern_bwd_config(config)
    elif mxfp8_backward_config is not None:
        raise ValueError("mxfp8_backward_config requires an MXFP8 backward phase")
    return _ExperimentalLinear.apply(x,weight,schedule_json,source_policy,forward_outer_mode,
        forward_scaling,forward_config,
        schedule_json if dx_schedule_json is None else dx_schedule_json,
        schedule_json if dw_schedule_json is None else dw_schedule_json,
        dx_policy,dw_policy,mxfp8_key,backward_pipeline)
