"""GPU-controlled fixed-K routing with CuTe selection and cuda-python graphs.

Compiler-visible proposal arithmetic retains the qualified BiMoE policy. No
production path builds CUDA C++ or depends on CUB/NVCC.
"""
from dataclasses import dataclass
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
import torch
from . import _routing_reference as reference
from .runtime import BoundedCache,tensor_storage_bytes,runner_cache_limit,runner_cache_byte_limit
from ._managed import guard_submission

_quota_stages=BoundedCache(runner_cache_limit('quota',128,namespace='ROUTING'))
_quantile_stages=BoundedCache(runner_cache_limit('quantile',128,namespace='ROUTING'))
_plans=BoundedCache(runner_cache_limit('hard',namespace='ROUTING'),
                    max_bytes=runner_cache_byte_limit('hard',namespace='ROUTING'),sizeof=tensor_storage_bytes)


def _compile_body(function,examples):
    """Static nondifferentiable FX -> Inductor, without Dynamo guard caches.

    Shape/stream plan construction must not consume the application's per-code
    object recompile allowance. All captures happen below this private boundary.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    graph=make_fx(function,tracing_mode='fake')(*examples)
    return torch._inductor.compile(graph,list(examples),options={'triton.cudagraphs':False})


@torch.library.custom_op('rtx_routing::quantile',mutates_args=(),device_types='cuda')
def quantile(matrix:torch.Tensor,rank:int,threads:int,column:bool)->torch.Tensor:
    if matrix.ndim!=2 or matrix.dtype!=torch.float32 or not matrix.is_contiguous():
        raise ValueError('Quantile requires contiguous CUDA FP32 [N,E]')
    n,e=matrix.shape
    length=n if column else e
    if threads not in (64,128,256) or not 0<=rank<length:
        raise ValueError('Unsupported native quantile geometry/rank')
    with torch.cuda.device(matrix.device):
        from .routing import _validate_native_device
        from .runtime import load_kernel_symbol
        _validate_native_device(matrix.device.index)
        compile_quantile=load_kernel_symbol('routing_selection','compile_quantile')
        result=torch.empty(e if column else n,device=matrix.device,dtype=torch.float32)
        key=('quantile',matrix.device.index,n,e,rank,threads,column)
        function=_quantile_stages.get(key)
        if function is None:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm routing quantiles before capture')
            function=compile_quantile(n,e,rank,threads,column)
            _quantile_stages[key]=function
        function(matrix.view(-1),result)
    return result


@quantile.register_fake
def _quantile_fake(matrix,rank,threads,column):
    return torch.empty(matrix.shape[1] if column else matrix.shape[0],device=matrix.device,dtype=matrix.dtype)


@torch.library.custom_op('rtx_routing::warp_quantile_into',mutates_args=('out',),device_types='cuda')
def warp_quantile_into(source:torch.Tensor,out:torch.Tensor,n:int,e:int,rank:int,threads:int)->None:
    with torch.cuda.device(source.device):
        from .routing import _validate_native_device
        from .runtime import load_kernel_symbol
        _validate_native_device(source.device.index)
        key=('warp_quantile',source.device.index,n,e,rank,threads)
        function=_quantile_stages.get(key)
        if function is None:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm warp quantiles before capture')
            compile_warp_quantile=load_kernel_symbol('routing_selection','compile_warp_quantile')
            function=compile_warp_quantile(n,e,rank,threads)
            _quantile_stages[key]=function
        function(source,out)


@warp_quantile_into.register_fake
def _warp_quantile_fake(source,out,n,e,rank,threads):
    return None


def warp_quantile(matrix,rank,threads):
    if matrix.ndim!=2 or matrix.dtype!=torch.float32 or not matrix.is_contiguous():
        raise ValueError('Warp quantile requires contiguous CUDA FP32 [N,E]')
    n,e=matrix.shape
    if n<1 or not 1<=e<=128 or not 0<=rank<e or threads not in (64,128,256):
        raise ValueError('Unsupported warp quantile geometry/rank')
    result=torch.empty(n,device=matrix.device,dtype=torch.float32)
    warp_quantile_into(matrix.view(-1),result,n,e,rank,threads)
    return result


def _initialize_native(scores,k,iterations,backend,threads):
    n,e=scores.shape
    if k==e:
        ids=torch.arange(e,device=scores.device).expand(n,-1)
        return ids,reference.assignment_loads(ids,e)
    values=scores+reference._deterministic_tie_break(n,e,device=scores.device,magnitude=1e-7)
    prices=torch.zeros(e,device=scores.device,dtype=torch.float32)
    capacity=n*k//e
    for _ in range(iterations):
        margins=values-prices[None,:]
        if backend=='native_warp':cutoff=warp_quantile(margins,k,threads)[:,None]
        elif backend=='native':cutoff=quantile(margins,k,threads,False)[:,None]
        else:cutoff=torch.topk(margins,k+1,dim=1).values[:,k:k+1]
        prices=quantile(values-cutoff,capacity,threads,True)
    ids=torch.topk(values-prices[None,:],k,dim=1).indices
    return ids,reference.assignment_loads(ids,e)


def repair_step(scores, tie_break, current, loads, phase, token):
    """One functional round, matching repair_balanced_assignment verbatim."""
    n, e = scores.shape
    k = current.shape[1]
    capacity = n*k//e
    pool_size = min(e, max(2*k, 16))
    surplus = (loads-capacity).clamp_min(0)
    deficit = (capacity-loads).clamp_min(0)
    held = torch.zeros_like(scores, dtype=torch.bool).scatter(1, current, True)
    eligible = (deficit[None, :] > 0) & ~held
    pool_values, pool_targets = torch.topk(
        (scores+tie_break).masked_fill(~eligible, float('-inf')), pool_size, dim=1)
    proposal_hash = (token*1_103_515_245 + pool_targets*12_345
                     + token*pool_targets*1_013_904_223
                     + phase*(pool_targets*2_654_435_761+2_246_822_519)) % 104_729
    diverse = torch.topk(proposal_hash.float().masked_fill(
        ~torch.isfinite(pool_values), float('-inf')), k, dim=1).indices
    targets = torch.where(phase >= 16, pool_targets.gather(1, diverse), pool_targets[:, :k])
    target_values = torch.where(phase >= 16, pool_values.gather(1, diverse), pool_values[:, :k])
    source_values, source_slots = torch.sort(
        scores.gather(1, current).masked_fill(surplus[current] <= 0, float('inf')),
        dim=1, stable=True)
    sources = current.gather(1, source_slots)
    gains = (target_values-source_values).flatten()
    source_accepted = reference._quota_accept(sources.flatten(), gains, surplus)
    target_accepted = reference._quota_accept(
        targets.flatten(), gains.masked_fill(~source_accepted, float('-inf')), deficit)
    accepted = source_accepted & target_accepted & torch.isfinite(gains)
    result = current.scatter(1, source_slots,
        torch.where(accepted.reshape(n,k), targets, sources))
    moved = accepted.to(torch.int64)
    new_loads = loads.scatter_add(0, sources.flatten(), -moved).scatter_add(0, targets.flatten(), moved)
    return result, new_loads, phase+1


def prepare_quota(m,e,device,k=1,segmented=True,cache_items=0,threads=128,early_exit=False):
    if not 1<=e<=1024 or not 0<=m<2**31:raise ValueError('Quota kernel supports E<=1024, M<2**31')
    dev=torch.device(device)
    if dev.type!='cuda':raise ValueError('CUDA device required')
    if not segmented and m>4096:raise ValueError('Direct quota implementation supports at most 4096 proposals')
    if cache_items not in (0,4,8,16) or threads not in (64,128,256) or type(early_exit)is not bool:
        raise ValueError('Invalid quota cache/thread/early-exit geometry')
    cache_items=cache_items if segmented else 0
    early_exit=early_exit if segmented else False
    key=('quota',dev.index if dev.index is not None else torch.cuda.current_device(),m,e,k,segmented,cache_items,threads,early_exit)
    stages=_quota_stages.get(key)
    if stages is None:
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Prepare quota workspace before capture')
        from .runtime import load_kernel_symbol
        if segmented and (cache_items or early_exit):
            compile_quota=load_kernel_symbol('routing_quota_cached','compile_quota_cached')
            stages=compile_quota(m,e,k,threads=threads,items=cache_items,early_exit=early_exit)
        else:
            compile_quota=load_kernel_symbol('routing_selection','compile_quota')
            stages=compile_quota(m,e,k,threads=threads,segmented=segmented)
        _quota_stages[key]=stages
    return stages


@torch.library.custom_op('rtx_routing::quota_apply',mutates_args=(),device_types='cuda')
def quota_apply(sources:torch.Tensor,targets:torch.Tensor,gains:torch.Tensor,
                surplus:torch.Tensor,deficit:torch.Tensor,slots:torch.Tensor,
                loads:torch.Tensor,phase:torch.Tensor,segmented:bool=True,cache_items:int=0,
                threads:int=128,early_exit:bool=False)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    if slots.ndim!=2 or loads.ndim!=1 or phase.ndim!=0:raise ValueError('Invalid state shapes')
    n,k=slots.shape;m=n*k;e=loads.numel()
    integer_inputs=(sources,targets,surplus,deficit,slots,loads,phase)
    if any(v.dtype!=torch.int64 for v in integer_inputs) or gains.dtype!=torch.float32:
        raise ValueError('Expected int64 metadata and FP32 gains')
    if any(v.device!=sources.device or not v.is_contiguous() for v in (*integer_inputs,gains)):
        raise ValueError('Contiguous inputs on one CUDA device required')
    if any(v.shape!=(m,) for v in (sources,targets,gains)) or surplus.shape!=(e,) or deficit.shape!=(e,):
        raise ValueError('Proposal/state size mismatch')
    if m==0:raise ValueError('Positive routing work required')
    stages=prepare_quota(m,e,sources.device,k,segmented,cache_items,threads,early_exit);device=sources.device
    counts=torch.empty(e,device=device,dtype=torch.int32)
    offsets=torch.empty(e+1,device=device,dtype=torch.int32)
    keys=torch.empty(m,device=device,dtype=torch.uint64)
    thresholds=torch.empty(e,device=device,dtype=torch.uint64)
    source_mask=torch.empty(m,device=device,dtype=torch.bool);target_mask=torch.empty_like(source_mask)
    result=torch.empty((n,k),device=device,dtype=torch.int64)
    new_loads=torch.empty_like(loads);new_phase=torch.empty_like(phase)
    arguments=(sources,targets,gains,surplus,deficit,slots.view(-1),loads,phase.view(-1),
               counts,offsets,keys,thresholds,source_mask,target_mask,result.view(-1),new_loads,new_phase.view(-1))
    for stage in stages:stage(*arguments)
    return result,new_loads,new_phase


@quota_apply.register_fake
def fake_apply(sources,targets,gains,surplus,deficit,slots,loads,phase,segmented=True,cache_items=0,threads=128,early_exit=False):
    return torch.empty(slots.shape,device=slots.device,dtype=torch.int64),torch.empty_like(loads),torch.empty_like(phase)


def _repair_step_fused(scores, tie_break, current, loads, phase, token,segmented=True,cache_items=0,threads=128,early_exit=False):
    """One functional round, matching repair_balanced_assignment verbatim."""
    n, e = scores.shape
    k = current.shape[1]
    capacity = n * k // e
    pool_size = min(e, max(2 * k, 16))
    surplus = (loads - capacity).clamp_min(0)
    deficit = (capacity - loads).clamp_min(0)
    held = torch.zeros_like(scores, dtype=torch.bool).scatter(1, current, True)
    eligible = (deficit[None, :] > 0) & ~held
    pool_values, pool_targets = torch.topk((scores + tie_break).masked_fill(~eligible, float('-inf')), pool_size, dim=1)
    proposal_hash = (token * 1103515245 + pool_targets * 12345 + token * pool_targets * 1013904223 + phase * (pool_targets * 2654435761 + 2246822519)) % 104729
    diverse = torch.topk(proposal_hash.float().masked_fill(~torch.isfinite(pool_values), float('-inf')), k, dim=1).indices
    targets = torch.where(phase >= 16, pool_targets.gather(1, diverse), pool_targets[:, :k])
    target_values = torch.where(phase >= 16, pool_values.gather(1, diverse), pool_values[:, :k])
    source_values, source_slots = torch.sort(scores.gather(1, current).masked_fill(surplus[current] <= 0, float('inf')), dim=1, stable=True)
    sources = current.gather(1, source_slots)
    gains = (target_values - source_values).flatten()
    return quota_apply(sources.flatten(),targets.flatten(),gains,surplus,deficit,source_slots,loads,phase,segmented,cache_items,threads,early_exit)


def _repair_step_direct(scores,tie_break,current,loads,phase,token):
    return _repair_step_fused(scores,tie_break,current,loads,phase,token,False)


def _repair_step_packed(scores,tie_break,current,loads,phase,token):
    return _repair_step_fused(scores,tie_break,current,loads,phase,token,True)


def _repair_step_segmented(scores,tie_break,current,loads,phase,token):
    # Tiny proposal sets favor the reference's specialized short sorts.
    if current.numel()<1024:
        return repair_step(scores,tie_break,current,loads,phase,token)
    return _repair_step_fused(scores,tie_break,current,loads,phase,token)


@dataclass(init=False)
class HardRoutingPlan:
    scores:torch.Tensor
    current:torch.Tensor
    loads:torch.Tensor
    phase:torch.Tensor
    tie:torch.Tensor
    token:torch.Tensor

    def __init__(self,scores,k,qb_rounds,rounds,segmented=True,quota_threshold=1024,qb_backend='torch',qb_threads=128,
                 quota_cache_items=0,quota_threads=128,quota_early_exit=False):
        from ._resources import install_graph_resource_tracking
        install_graph_resource_tracking()
        self.n,self.e=scores.shape;self.k=k
        self.stream=torch.cuda.current_stream(scores.device)
        self.graph=None
        self.lock=RLock()
        self.scores=torch.empty_like(scores)
        self.current=torch.empty((self.n,k),device=scores.device,dtype=torch.int64)
        self.loads=torch.empty(self.e,device=scores.device,dtype=torch.int64)
        self.phase=torch.zeros((),device=scores.device,dtype=torch.int64)
        self.tie=reference._deterministic_tie_break(self.n,self.e,device=scores.device,magnitude=1e-6)
        self.token=torch.arange(self.n,device=scores.device,dtype=torch.int64)[:,None]
        self.scores.copy_(scores)
        if qb_backend not in ('torch','native_column','native','native_warp'):raise ValueError('Invalid QB backend')
        if qb_backend=='native_warp' and self.e>128:raise ValueError('Warp QB row selection supports at most 128 experts')
        packed=segmented and self.n*k>=quota_threshold
        def body(*arguments):
            return _repair_step_fused(*arguments,segmented=packed,cache_items=quota_cache_items,
                                      threads=quota_threads,early_exit=quota_early_exit)
        prepare_quota(self.n*k,self.e,scores.device,k,packed,quota_cache_items,quota_threads,quota_early_exit)
        e=self.e
        def initialize(value):
            if qb_backend!='torch':return _initialize_native(value,k,qb_rounds,qb_backend,qb_threads)
            assignments,_=reference.dual_price_assignment(value,top_k=k,iterations=qb_rounds)
            return assignments,reference.assignment_loads(assignments,e)
        self.initialize_op=_compile_body(initialize,(self.scores,))
        self.body_op=_compile_body(body,(self.scores,self.tie,self.current,self.loads,self.phase,self.token))
        capture=torch.cuda.Stream(device=scores.device);capture.wait_stream(self.stream)
        with torch.no_grad(),torch.cuda.stream(capture):
            for _ in range(3):self.initialize();self.body()
            self.initializer_graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.initializer_graph,stream=capture):self.initialize()
            self.body_graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.body_graph,stream=capture):self.body()
        self.stream.wait_stream(capture);capture.synchronize()
        from ._routing_graph import ConditionalRoutingGraph
        self.graph=ConditionalRoutingGraph(self.initializer_graph,self.body_graph,self.loads,self.phase,
                                           self.e,self.n*k//self.e,rounds,scores.device.index)

    def initialize(self):
        assignments,loads=self.initialize_op(self.scores)
        self.current.copy_(assignments);self.loads.copy_(loads);self.phase.zero_()

    def body(self):
        current,loads,phase=self.body_op(self.scores,self.tie,self.current,self.loads,self.phase,self.token)
        self.current.copy_(current);self.loads.copy_(loads);self.phase.copy_(phase)

    def __call__(self,scores):
        out=torch.empty_like(self.current)
        self.launch_into(scores,out)
        return out

    def launch_into(self,scores,out):
        """Launch into caller-owned IDs; no hot Torch allocation or arithmetic.

        Conditional graph nodes require fixed-address internal state. That
        state belongs to this bounded, stream-local cold plan; it never escapes
        as a returned alias. Only the immutable graph is submitted here.
        """
        stream=torch.cuda.current_stream(scores.device).cuda_stream
        if stream!=self.stream.cuda_stream:raise RuntimeError('Internal routing plan stream mismatch')
        if (out.shape!=(self.n,self.k) or out.dtype!=torch.int64
                or out.device!=scores.device or not out.is_contiguous()):
            raise ValueError('Hard routing output must be contiguous CUDA int64 [N,K]')
        # Native copies, graph submission and extraction are one host-critical
        # section. Hidden mutable state never escapes as an aliased output.
        with self.lock:
            from ._routing_graph import copy_async
            copy_async(self.scores.data_ptr(),scores.data_ptr(),scores.numel()*4,stream)
            self.graph.launch(stream)
            copy_async(out.data_ptr(),self.current.data_ptr(),out.numel()*8,stream)

    def __del__(self):
        if getattr(self,'graph',None) is not None:
            try:self.stream.synchronize()
            except Exception:pass
            try:self.graph.close()
            except Exception:pass
            self.graph=None


def _get_plan(scores,k,qb_rounds,rounds,segmented,owner_id,quota_threshold=1024,qb_backend='torch',qb_threads=128,
              quota_cache_items=0,quota_threads=128,quota_early_exit=False):
    if scores.dtype!=torch.float32 or scores.ndim!=2 or not scores.is_contiguous():
        raise ValueError('Hard routing requires contiguous CUDA FP32 scores')
    reference.uniform_capacity(*scores.shape,k)
    if min(qb_rounds,rounds)<1:raise ValueError('Invalid schedule budgets')
    from .routing import _validate_native_device
    _validate_native_device(scores.device.index)
    from ._managed import install
    install()
    stream=torch.cuda.current_stream(scores.device).cuda_stream
    packed=segmented and scores.shape[0]*k>=quota_threshold
    if not packed:quota_cache_items=0;quota_early_exit=False
    key=(scores.device.index,stream,owner_id,*scores.shape,k,qb_rounds,rounds,segmented,quota_threshold,qb_backend,qb_threads,
         quota_cache_items,quota_threads,quota_early_exit)
    with _plans._lock:
        plan=_plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Warm BalancedRouter before manual CUDA graph capture')
            caller_stream=torch.cuda.current_stream(scores.device)
            def prepare():
                # Cold extension preparation must not inherit below-autograd
                # or SchemaCheck TLS from the custom-op dispatcher.
                with torch.cuda.device(scores.device),torch.cuda.stream(caller_stream),torch.no_grad():
                    return HardRoutingPlan(scores,k,qb_rounds,rounds,segmented,quota_threshold,qb_backend,qb_threads,
                                           quota_cache_items,quota_threads,quota_early_exit)
            with ThreadPoolExecutor(max_workers=1,thread_name_prefix='rtx-routing-prepare') as worker:
                plan=worker.submit(prepare).result()
            _plans[key]=plan
    return plan


def _schedule(scores,k,qb_rounds,rounds,segmented,owner_id,quota_threshold=1024,qb_backend='torch',qb_threads=128,
              quota_cache_items=0,quota_threads=128,quota_early_exit=False):
    return _get_plan(scores,k,qb_rounds,rounds,segmented,owner_id,quota_threshold,qb_backend,qb_threads,
                     quota_cache_items,quota_threads,quota_early_exit)(scores)


@torch.library.custom_op('rtx_routing::hard_schedule_into',mutates_args=('out',),device_types='cuda')
@guard_submission
def hard_schedule_into(scores:torch.Tensor,out:torch.Tensor,k:int,qb_rounds:int,rounds:int,
                       segmented:bool=True,owner_id:int=0,quota_threshold:int=1024,
                       qb_backend:str='torch',qb_threads:int=128,quota_cache_items:int=0,
                       quota_threads:int=128,quota_early_exit:bool=False)->None:
    with torch.cuda.device(scores.device):
        plan=_get_plan(scores,k,qb_rounds,rounds,segmented,owner_id,quota_threshold,qb_backend,qb_threads,
                       quota_cache_items,quota_threads,quota_early_exit)
        plan.launch_into(scores,out)


@hard_schedule_into.register_fake
def _hard_into_fake(scores,out,k,qb_rounds,rounds,segmented=True,owner_id=0,
                    quota_threshold=1024,qb_backend='torch',qb_threads=128,quota_cache_items=0,quota_threads=128,quota_early_exit=False):
    pass


def dispatch_hard(scores,k,qb_rounds,rounds,segmented=True,owner_id=0,
                  quota_threshold=1024,qb_backend='torch',qb_threads=128,quota_cache_items=0,quota_threads=128,quota_early_exit=False):
    """Traceable output ownership around GPU-controlled fixed-policy routing."""
    out=torch.empty((scores.shape[0],k),device=scores.device,dtype=torch.int64)
    hard_schedule_into(scores,out,k,qb_rounds,rounds,segmented,owner_id,
                       quota_threshold,qb_backend,qb_threads,quota_cache_items,quota_threads,quota_early_exit)
    return out


@torch.library.custom_op('rtx_routing::hard_schedule',mutates_args=(),device_types='cuda')
@guard_submission
def hard_schedule(scores:torch.Tensor,k:int,qb_rounds:int,rounds:int,segmented:bool=True,owner_id:int=0,
                  quota_threshold:int=1024,qb_backend:str='torch',qb_threads:int=128,quota_cache_items:int=0,
                  quota_threads:int=128,quota_early_exit:bool=False)->torch.Tensor:
    with torch.cuda.device(scores.device):
        return _schedule(scores,k,qb_rounds,rounds,segmented,owner_id,quota_threshold,qb_backend,qb_threads,
                          quota_cache_items,quota_threads,quota_early_exit)


@hard_schedule.register_fake
def _hard_fake(scores,k,qb_rounds,rounds,segmented=True,owner_id=0,quota_threshold=1024,qb_backend='torch',qb_threads=128,
               quota_cache_items=0,quota_threads=128,quota_early_exit=False):
    return torch.empty((scores.shape[0],k),device=scores.device,dtype=torch.int64)


def _clear_runtime_caches(*,preserve_compiled=False):
    count=len(_plans);_plans.clear()
    if not preserve_compiled:_quota_stages.clear();_quantile_stages.clear()
    return {'hard_plans':count}
