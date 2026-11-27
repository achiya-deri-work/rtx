"""Packed independent routing: compiler-visible buffers, GPU-only boundaries.

The portable CPU reference intentionally iterates examples. CUDA never does:
normalizations and quantiles use segmented grids and one GPU-controlled repair
loop serves all pools. Shared expert weights are not replicated per sequence.
"""
from functools import lru_cache
from dataclasses import dataclass
import math
from threading import RLock

import torch

from .runtime import BoundedCache,tensor_storage_bytes,runner_cache_limit,runner_cache_byte_limit
from ._managed import guard_submission


_COMPILED=BoundedCache(128)
_PLANS=BoundedCache(runner_cache_limit('varlen_hard',namespace='ROUTING'),
    max_bytes=runner_cache_byte_limit('varlen_hard',namespace='ROUTING'),sizeof=tensor_storage_bytes)
_LOCK=RLock()
_CONDITION_STREAMS={}


def validate_cu(x,cu):
    if (cu.ndim!=1 or cu.numel()<2 or cu.dtype!=torch.int32
            or not cu.is_contiguous() or cu.device!=x.device):
        raise ValueError('cu_seqlens must be contiguous int32[B+1] on the input device')
    if x.ndim!=2 or x.shape[0]<1:raise ValueError('Packed inputs must be nonempty [total_tokens,D]')


def workspace_bounds(t,b,e,k):
    """Static shape-only bounds, without reading a boundary value.

    At most min(T,B) pools can be nonempty. For q=K/gcd(E,K) dividing 32,
    nested logical/physical rounding collapses exactly to 32*ceil(L/P),
    P=(E/gcd(E,K))*32/q. Enumerating positive-pool counts gives the tight
    worst-case slot bound below. Other q retain a conservative bound.
    This pure symbolic arithmetic is also used by the fake operator.
    """
    multiple=e//math.gcd(e,k);q=k//math.gcd(e,k)
    nonempty=torch.sym_min(t,b)
    n=(t+nonempty*(multiple-1))//multiple*multiple
    if 32%q==0:
        period=multiple*(32//q)
        capacity=32*(nonempty+(t-nonempty)//period)
    else:capacity=((n*k//e+nonempty*31+31)//32)*32
    return n,capacity


def persistent_pool_rows(t,e,k,*,pools=1):
    """Static maximum for any runtime partition; zero selects staged kernels."""
    multiple=e//math.gcd(e,k)
    rows=(t+multiple-1)//multiple*multiple
    # Bound one CTA's shared matrix (odd pitch), shifts and normalizers.
    # Restrict very long, very narrow pools even when they fit in shared memory.
    shared_bytes=4*(rows*(e|1)+rows+e+max(rows,e))
    # Tail initialization stripes across pool CTAs with signed 32-bit indices.
    # Leave room for packed indices even with millions of empty pools.
    return rows if rows<=512 and shared_bytes<=80*1024 and pools*1024<2**30 else 0


def persistent_hybrid_rows(t,e,k,*,pools):
    """Short-pool CTA bound when the packed total does not fit one CTA.

    128 scheduled rows is the measured short-pool regime. A larger pool
    does not fall inside the persistent block's shared allocation, even
    after a shape-stable boundary mutation.
    """
    if pools<2 or persistent_pool_rows(t,e,k,pools=pools):return 0
    multiple=e//math.gcd(e,k)
    rows=128//multiple*multiple
    return persistent_pool_rows(rows,e,k,pools=pools) if rows else 0


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def layout(t,b,e,k,steps,token,strict=False):
    from .routing import PACKED_SINKHORN_BACKENDS,RoutingKernelConfig
    c=RoutingKernelConfig.from_token(token)
    if min(t,b,e,k,steps)<=0 or k>e or e>128 or k>32:
        raise ValueError('Packed routing needs T,B>0, E<=128 and 1<=K<=min(E,32)')
    n,capacity=workspace_bounds(t,b,e,k)
    shards=(c.packed_column_shards_forward,c.packed_column_shards_backward)
    if max(2*steps*n*e,n*k,e*capacity,b*e*max(1,*shards))>=2**31:
        raise ValueError('Packed routing workspace exceeds signed native index range')
    # All choices in this implementation are explicit geometry, not
    # changes to balancing, iteration budgets or quantization block grouping.
    direct=persistent_pool_rows(t,e,k,pools=b)
    hybrid=persistent_hybrid_rows(t,e,k,pools=b)
    mode=PACKED_SINKHORN_BACKENDS.index(c.packed_sinkhorn_backend) if direct or hybrid else 0
    # Hybrid needs both the original staged geometry and the persistent CTA
    # width. For the old direct path the staged directions remain inactive.
    override=c.packed_persistent_threads if direct else 0
    threads=(override if mode&1 and override else c.forward.threads,
             override if mode&2 and override else c.backward.threads)
    return (t,b,e,k,n,capacity,steps,
            (threads[0],c.forward.row_groups,c.forward.column_groups),
            (threads[1],c.backward.row_groups,c.backward.column_groups),
            c.qb_threads,c.metadata_threads,strict,c.packed_column_cache_items,int(c.packed_row_backend=='warp'),
            *shards,int(c.packed_qb_column_backend=='expert_keys'),mode,
            hybrid if mode else 0,c.packed_persistent_threads if mode and hybrid else 0)


def compiled(config,device):
    key=(config,device)
    result=_COMPILED.get(key)
    if result is not None:return result
    with _LOCK:
        result=_COMPILED.get(key)
        if result is None:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm packed routing before manual capture')
                from .routing import _validate_native_device
                _validate_native_device(device)
                from .runtime import load_kernel_symbol
                factory=load_kernel_symbol('routing_varlen','compile_varlen')
                if config[18] and device not in _CONDITION_STREAMS:
                    _CONDITION_STREAMS[device]=torch.cuda.Stream(device=device)
                result=factory(*config);_COMPILED[key]=result
    return result


@torch.library.custom_op('rtx_routing::varlen_prepare',mutates_args=('buffers',),device_types='cuda')
def prepare_execute(scores:torch.Tensor,cu:torch.Tensor,buffers:list[torch.Tensor],config:list[int])->None:
    key=_decode_config(tuple(config))
    prefix,pack,*_=compiled(key,scores.device.index)
    prefix(scores,cu,*buffers);pack(scores,cu,*buffers)


@prepare_execute.register_fake
def _prepare_fake(scores,cu,buffers,config):pass


@lru_cache(maxsize=256)
def _decode_config(config):
    # Flat int ABI keeps fake/AOT registration independent of Python objects.
    a=tuple(config)
    return (*a[:7],tuple(a[7:10]),tuple(a[10:13]),a[13],a[14],bool(a[15]),*a[16:],*((0,)*(24-len(a))))


def _flat(config):return [*config[:7],*config[7],*config[8],config[9],config[10],int(config[11]),*config[12:]]


def _column_workspace(key,device,backward):
    shards=key[15 if backward else 14]
    return [torch.empty(key[1]*key[2]*shards,device=device,dtype=torch.float32)] if shards else []


def _launch_staged(inputs,buffers,kernels,steps,backward):
    if not backward:
        packed,scheduled,sequence,rowmap=inputs
        out,history,row,column,probabilities,*workspace=buffers
        kernels['init',0](packed,column,row,history,out,probabilities,scheduled,sequence,rowmap,0)
        for iteration in range(steps):
            kernels['forward',0](packed,column,row,history,out,probabilities,scheduled,sequence,rowmap,iteration)
            if workspace:
                kernels['forward',1](packed,row,column,history,out,scheduled,workspace[0],iteration)
            else:
                kernels['forward',1](packed,row,column,history,out,probabilities,scheduled,sequence,rowmap,iteration)
        kernels['finish',0](packed,column,row,history,out,probabilities,scheduled,sequence,rowmap,0)
    else:
        gradient,history,scheduled,sequence,rowmap=inputs
        a,z,out,unused,*workspace=buffers
        source=gradient
        for iteration in reversed(range(steps)):
            if workspace:
                kernels['backward',1](source,unused,unused,history,a,scheduled,workspace[0],iteration)
            else:
                kernels['backward',1](source,unused,unused,history,a,gradient,scheduled,sequence,rowmap,iteration)
            kernels['backward',0](a,unused,unused,history,z,gradient,scheduled,sequence,rowmap,iteration)
            source=z
        kernels['unpack',0](source,unused,unused,history,out,gradient,scheduled,sequence,rowmap,0)


def _capture_long_pool_branch(condition,scheduled,launch):
    """Capture a launch-only branch with the actual compiler-owned addresses.

    The child capture makes no allocations and never executes its predicates
    eagerly. There are no fixed-address transport tensors to copy in/out.
    """
    from ._routing_graph import cuda,checked
    import cutlass
    with torch.cuda.device(scheduled.device),_LOCK:
        parent=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _,_,graph,_,_,_=checked(cuda.cuStreamGetCaptureInfo(parent))
        context=checked(cuda.cuCtxGetCurrent())
        handle=checked(cuda.cuGraphConditionalHandleCreate(graph,context,0,1))
        condition(scheduled,cutlass.Uint64(int(handle)))
        _,_,_,deps,edges,count=checked(cuda.cuStreamGetCaptureInfo(parent))
        params=cuda.CUgraphNodeParams()
        params.type=cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        params.conditional.handle=handle
        params.conditional.type=cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
        params.conditional.size=1
        params.conditional.ctx=context
        branch=checked(cuda.cuGraphAddNode(graph,deps or None,edges or None,count,params))
        child=params.conditional.phGraph_out[0]
        capture=_CONDITION_STREAMS[scheduled.device.index]
        with torch.cuda.stream(capture):
            stream=cuda.CUstream(capture.cuda_stream)
            checked(cuda.cuStreamBeginCaptureToGraph(stream,child,None,None,0,
                cuda.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED))
            try:launch()
            finally:checked(cuda.cuStreamEndCapture(stream))
        replace=cuda.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES
        checked(cuda.cuStreamUpdateCaptureDependencies(parent,[branch],None,1,replace))


@torch.library.custom_op('rtx_routing::varlen_sinkhorn_execute',mutates_args=('buffers',),device_types='cuda')
def sinkhorn_execute(inputs:list[torch.Tensor],buffers:list[torch.Tensor],config:list[int],backward:bool)->None:
    key=_decode_config(tuple(config));steps=key[6]
    kernels=compiled(key,inputs[0].device.index)[2]
    if not key[17]&(2 if backward else 1):
        _launch_staged(inputs,buffers,kernels,steps,backward)
        return
    hybrid=bool(key[18])
    if backward:
        gradient,history,scheduled,_,rowmap=inputs
        kernels['persistent',1](gradient,history,buffers[2 if hybrid else 0],scheduled,rowmap,gradient)
    else:
        packed,scheduled,_,rowmap=inputs
        out,history=buffers[:2];probabilities=buffers[4 if hybrid else 2]
        kernels['persistent',0](packed,history,out,scheduled,rowmap,probabilities)
    if hybrid:
        if torch.cuda.is_current_stream_capturing():
            _capture_long_pool_branch(kernels['long_pool_condition',0],scheduled,
                lambda:_launch_staged(inputs,buffers,kernels,steps,backward))
        else:
            # Eager keeps tensor addresses direct and has no host readback.
            # The staged launches themselves predicate out all short pools.
            _launch_staged(inputs,buffers,kernels,steps,backward)


@sinkhorn_execute.register_fake
def _sinkhorn_fake(inputs,buffers,config,backward):pass


class SavedTransport(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,cu,token,steps,strict):
        key=layout(scores.shape[0],cu.numel()-1,scores.shape[1],token[0],steps,token[1:],strict)
        out,history,scheduled,physical,sequence,local,rowmap,probabilities=transport_forward(scores,cu,key)
        ctx.save_for_backward(history,scheduled,sequence,rowmap);ctx.key=key
        ctx.mark_non_differentiable(scheduled,physical,sequence,local,rowmap,probabilities)
        return out,scheduled,physical,sequence,local,rowmap,probabilities

    @staticmethod
    def backward(ctx,gradient,*_):
        if gradient is None:return (None,)*5
        history,scheduled,sequence,rowmap=ctx.saved_tensors
        return transport_backward(gradient,history,scheduled,sequence,rowmap,ctx.key),None,None,None,None


def transport_forward(scores,cu,key):
    t,b,e,k,n,c,steps,*_=key;device=scores.device;flat=_flat(key)
    scheduled=torch.empty(b+1,device=device,dtype=torch.int32)
    physical=torch.empty_like(scheduled)
    sequence=torch.empty(n,device=device,dtype=torch.int64);local=torch.empty_like(sequence)
    rowmap=torch.empty(n,device=device,dtype=torch.int32)
    packed=torch.empty((n,e),device=device,dtype=torch.float32)
    status=torch.empty(1,device=device,dtype=torch.int32)
    prepare_execute(scores.flatten(),cu,[scheduled,physical,sequence,local,rowmap,packed.flatten(),status],flat)
    torch._assert_async(status[0]==1,'Invalid cu_seqlens, nonfinite affinities, or nondivisible sequence with padding=error')
    out=torch.empty_like(packed)
    probabilities=torch.empty_like(packed)
    history=torch.empty(2*steps*n*e,device=device,dtype=torch.float32)
    if key[17]&1 and not key[18]:
        buffers=[out.flatten(),history,probabilities.flatten()]
    else:
        row=torch.empty(max(n,b*e),device=device,dtype=torch.float32);column=torch.empty_like(row)
        buffers=[out.flatten(),history,row,column,probabilities.flatten(),*_column_workspace(key,device,False)]
    sinkhorn_execute([packed.flatten(),scheduled,sequence,rowmap],buffers,flat,False)
    return out,history,scheduled,physical,sequence,local,rowmap,probabilities


def transport_backward(gradient,history,scheduled,sequence,rowmap,key):
    t,b,e,k,n,*_=key;device=gradient.device
    out=torch.empty((t,e),device=device,dtype=torch.float32)
    if key[17]&2 and not key[18]:
        buffers=[out.flatten()]
    else:
        a=torch.empty(n*e,device=device,dtype=torch.float32);z=torch.empty_like(a)
        unused=torch.empty(max(n,b*e),device=device,dtype=torch.float32)
        buffers=[a,z,out.flatten(),unused,*_column_workspace(key,device,True)]
    sinkhorn_execute([gradient.contiguous().flatten(),history,scheduled,sequence,rowmap],buffers,_flat(key),True)
    return out


@torch.library.custom_op('rtx_routing::varlen_quantile',mutates_args=('out',),device_types='cuda')
def quantile_execute(matrix:torch.Tensor,scheduled:torch.Tensor,out:torch.Tensor,config:list[int])->None:
    compiled(_decode_config(tuple(config)),matrix.device.index)[3](matrix,scheduled,out)


@quantile_execute.register_fake
def _quantile_fake(matrix,scheduled,out,config):pass


@torch.library.custom_op('rtx_routing::varlen_quantile_expert_keys',mutates_args=('out','keys'),device_types='cuda')
def quantile_expert_keys_execute(matrix:torch.Tensor,scheduled:torch.Tensor,out:torch.Tensor,
                                 keys:torch.Tensor,config:list[int])->None:
    compiled(_decode_config(tuple(config)),matrix.device.index)[3](matrix,scheduled,out,keys)


@quantile_expert_keys_execute.register_fake
def _quantile_expert_keys_fake(matrix,scheduled,out,keys,config):pass


def _quantile(matrix,scheduled,config):
    result=torch.empty((config[1],config[2]),device=matrix.device,dtype=torch.float32)
    if config[16]:
        keys=torch.empty(config[2]*config[4],device=matrix.device,dtype=torch.uint32)
        quantile_expert_keys_execute(matrix.flatten(),scheduled,result.flatten(),keys,_flat(config))
    else:
        quantile_execute(matrix.flatten(),scheduled,result.flatten(),_flat(config))
    return result


@torch.library.custom_op('rtx_routing::varlen_quota',mutates_args=('buffers',),device_types='cuda')
def quota_execute(buffers:list[torch.Tensor],m:int,e:int,k:int,threads:int,segmented:bool,items:int,early:bool)->None:
    from .runtime import load_kernel_symbol
    identity=('quota',m,e,k,threads,segmented,items,early,buffers[0].device.index)
    stages=_COMPILED.get(identity)
    if stages is None:
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm packed quota selection before capture')
        if segmented and (items or early):
            stages=load_kernel_symbol('routing_quota_cached','compile_quota_cached')(m,e,k,threads,items,early)
        else:stages=load_kernel_symbol('routing_selection','compile_quota')(m,e,k,threads,segmented)
        _COMPILED[identity]=stages
    for stage in stages:stage(*buffers)


@quota_execute.register_fake
def _quota_fake(buffers,m,e,k,threads,segmented,items,early):pass


def _quota(sources,targets,gains,surplus,deficit,slots,loads,phase,threads,segmented,items,early):
    m=sources.numel();groups=loads.numel();device=sources.device
    counts=torch.empty(groups,device=device,dtype=torch.int32)
    offsets=torch.empty(groups+1,device=device,dtype=torch.int32)
    keys=torch.empty(m,device=device,dtype=torch.uint64)
    thresholds=torch.empty(groups,device=device,dtype=torch.uint64)
    sm=torch.empty(m,device=device,dtype=torch.bool);tm=torch.empty_like(sm)
    result=torch.empty_like(slots);new_loads=torch.empty_like(loads);new_phase=torch.empty_like(phase)
    quota_execute([sources,targets,gains,surplus,deficit,slots.flatten(),loads,phase.view(-1),
                   counts,offsets,keys,thresholds,sm,tm,result.flatten(),new_loads,new_phase.view(-1)],
                  m,groups,slots.shape[1],threads,segmented,items,early)
    return result,new_loads,new_phase


def _tie(local,e,magnitude):
    expert=torch.arange(e,device=local.device,dtype=torch.int64)[None,:]
    token=local[:,None];mod=104729
    value=((token*(1103515245%mod))%mod+expert*12345+
           ((token*(1013904223%mod))%mod*expert)%mod)%mod
    return value.float()*(magnitude/mod)


@torch.library.custom_op('rtx_routing::packed_row_topk',mutates_args=('values','ids'),device_types='cuda')
def _packed_row_topk(src:torch.Tensor,values:torch.Tensor,ids:torch.Tensor,n:int,e:int,k:int)->None:
    from .runtime import load_kernel_symbol
    if (e not in (16,24,32) or k not in (4,16) or not 0<n*e<2**31
            or src.shape!=(n*e,) or values.shape!=(n*k,) or ids.shape!=values.shape
            or src.dtype!=torch.float32 or values.dtype!=src.dtype or ids.dtype!=torch.int64
            or any(t.device!=src.device or not t.is_contiguous() for t in (src,values,ids))):
        raise ValueError('Expected contiguous packed FP32 rows and FP32/int64 top-k outputs')
    with torch.cuda.device(src.device):
        fn=load_kernel_symbol('routing_topk','compile_select')(n,e,k)
        fn(src,values,ids)


@_packed_row_topk.register_fake
def _packed_row_topk_fake(src,values,ids,n,e,k):pass


def _select_row(matrix,k,warp=False):
    # The fast path is qualified only for these specific ATen CUDA shapes;
    # preserve stock routing for unsupported widths and ranks.
    n,e=matrix.shape
    if not warp or e not in (16,24,32) or k not in (4,16):
        return matrix.topk(k,dim=1)
    values=matrix.new_empty((n,k))
    ids=torch.empty((n,k),dtype=torch.int64,device=matrix.device)
    _packed_row_topk(matrix.contiguous().view(-1),values.view(-1),ids.view(-1),n,e,k)
    return values,ids


def packed_warp_repair_supported(n,e,k):
    # Only large batches of short rows: smaller launches can dispatch a
    # different ATen algorithm with different equal-key ordering.
    return 4096<=n<=65535 and e in (24,32) and k==4


def _repair_targets(pv,pt,pos,phase,k,stage,warp=False):
    # Keep the original top-k and its ordering even at the phase boundary:
    # recomputing a narrower top-k is not equivalent when scores tie.
    if stage=='early':return pt[:,:k],pv[:,:k]
    ph=(pos[:,None]*1103515245+pt*12345+pos[:,None]*pt*1013904223+
        phase*(pt*2654435761+2246822519))%104729
    _,diverse=_select_row(ph.float().masked_fill(~pv.isfinite(),float('-inf')),k,warp)
    ids,values=pt.gather(1,diverse),pv.gather(1,diverse)
    if stage=='late':return ids,values
    return torch.where(phase>=16,ids,pt[:,:k]),torch.where(phase>=16,values,pv[:,:k])


@dataclass(init=False)
class HardPlan:
    """One conditional GPU repair loop, not a loop of per-sequence launches."""
    scores:torch.Tensor
    scheduled:torch.Tensor
    sequence:torch.Tensor
    local:torch.Tensor
    current:torch.Tensor
    load_caps:torch.Tensor
    phase:torch.Tensor
    def __init__(self,scores,scheduled,sequence,local,key,qb_rounds,rounds,policy,*,split_repair=None,warp_topk=None):
        from ._routing_hard import _compile_body
        from ._routing_graph import ConditionalRoutingGraph
        from .runtime import load_kernel_symbol
        from .routing import RoutingKernelConfig
        execution=RoutingKernelConfig.from_token(policy)
        t,b,e,k,n,c,*_=key;self.key=key;self.lock=RLock();self.graph=None
        if split_repair is None:split_repair=execution.packed_repair_backend!='mixed'
        if warp_topk is None:
            warp_topk=execution.packed_repair_backend=='split_warp' and packed_warp_repair_supported(n,e,k)
        segmented=execution.segmented_quotas and n*k>=execution.quota_threshold
        items=execution.quota_cache_items if segmented else 0
        early=execution.quota_early_exit if segmented else False
        if not segmented and n*k>4096:raise ValueError('Direct packed quota ranking is bounded to 4096 proposals')
        if (items or early) and b*e>1024:raise ValueError('Cached packed quotas support B*E<=1024; choose streaming quotas')
        self.stream=torch.cuda.current_stream(scores.device)
        self.scores=torch.empty_like(scores);self.scheduled=torch.empty_like(scheduled)
        self.sequence=torch.empty_like(sequence);self.local=torch.empty_like(local)
        self.current=torch.empty((n,k),device=scores.device,dtype=torch.int64)
        self.load_caps=torch.empty(2*b*e,device=scores.device,dtype=torch.int64)
        self.loads=self.load_caps[:b*e];self.caps=self.load_caps[b*e:]
        self.phase=torch.empty((),device=scores.device,dtype=torch.int64)
        self.scores.copy_(scores);self.scheduled.copy_(scheduled);self.sequence.copy_(sequence);self.local.copy_(local)
        def initialize(value,cu,seq,pos):
            active=torch.arange(n,device=value.device)<cu[-1]
            caps=((cu[1:]-cu[:-1]).to(torch.int64)*k//e)[:,None].expand(b,e).contiguous().flatten()
            values=value+_tie(pos,e,1e-7)
            prices=torch.zeros((b,e),device=value.device,dtype=torch.float32)
            if k==e:
                chosen=torch.arange(e,device=value.device).expand(n,-1)
            else:
                for _ in range(qb_rounds):
                    margins=values-prices[seq]
                    if execution.qb_backend=='native_warp':
                        from ._routing_hard import warp_quantile
                        cutoff=warp_quantile(margins,k,execution.qb_threads)[:,None]
                    elif execution.qb_backend=='native':
                        from ._routing_hard import quantile
                        cutoff=quantile(margins,k,execution.qb_threads,False)[:,None]
                    else:cutoff=margins.topk(k+1,dim=1).values[:,k:k+1]
                    column_values=values-cutoff
                    if execution.qb_backend=='torch':
                        order=column_values.T.argsort(dim=1,descending=True,stable=True)
                        group=torch.where(active,seq,b)[order]
                        order=order.gather(1,group.argsort(dim=1,stable=True))
                        sorted_values=column_values.T.gather(1,order)
                        length=cu[1:]-cu[:-1]
                        index=(cu[:-1]+length*k//e).to(torch.int64).clamp_max(n-1)
                        prices=sorted_values[:,index].T
                        prices=torch.where(length[:,None]>0,prices,0.)
                    else:prices=_quantile(column_values,cu,key)
                _,chosen=_select_row(values-prices[seq],k,warp_topk)
            ids=chosen+seq[:,None]*e
            mask=active[:,None].expand(n,k).to(torch.int64)
            loads=torch.zeros(b*e,device=value.device,dtype=torch.int64).scatter_add(0,ids.flatten(),mask.flatten())
            return ids,loads,caps
        def body(value,cu,seq,pos,current,loads,caps,phase,stage='mixed'):
            active=torch.arange(n,device=value.device)<cu[-1]
            surplus=(loads-caps).clamp_min(0);deficit=(caps-loads).clamp_min(0)
            held=torch.zeros_like(value,dtype=torch.bool).scatter(1,current%e,True)
            eligible=(deficit.reshape(b,e)[seq]>0)&~held&active[:,None]
            pv,pt=_select_row((value+_tie(pos,e,1e-6)).masked_fill(~eligible,float('-inf')),min(e,max(2*k,16)),warp_topk)
            targets,target_values=_repair_targets(pv,pt,pos,phase,k,stage,warp_topk)
            targets=targets+seq[:,None]*e
            source_values,slots=value.gather(1,current%e).masked_fill(surplus[current]<=0,float('inf')).sort(dim=1,stable=True)
            sources=current.gather(1,slots)
            gains=(target_values-source_values).flatten()
            return _quota(sources.flatten(),targets.flatten(),gains,surplus,deficit,slots,loads,phase,
                          execution.quota_threads,segmented,items,early)
        self.init_op=_compile_body(initialize,(self.scores,self.scheduled,self.sequence,self.local))
        examples=(self.scores,self.scheduled,self.sequence,self.local,self.current,self.loads,self.caps,self.phase)
        def compile_stage(stage):
            def selected(value,cu,seq,pos,current,loads,caps,phase):
                return body(value,cu,seq,pos,current,loads,caps,phase,stage)
            return _compile_body(selected,examples)
        needs_late=split_repair and rounds>16
        self.body_op=compile_stage('late' if needs_late else 'early' if split_repair else 'mixed')
        early_op=compile_stage('early') if needs_late else None
        capture=torch.cuda.Stream(device=scores.device);capture.wait_stream(self.stream)
        with torch.no_grad(),torch.cuda.stream(capture):
            for _ in range(2):
                self.initialize();self.body()
                if early_op is not None:self.body(early_op)
            self.init_graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.init_graph,stream=capture):self.initialize()
            self.body_graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.body_graph,stream=capture):self.body()
            if early_op is not None:
                early_graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(early_graph,stream=capture):self.body(early_op)
        capture.synchronize();self.stream.wait_stream(capture)
        make_condition=load_kernel_symbol('routing_varlen','compile_varlen_condition')
        early=(early_graph,make_condition(b*e,16)) if needs_late else None
        # The captured body references temporary storage owned by its compiled callable.
        self.early_op=early_op
        self.graph=ConditionalRoutingGraph(self.init_graph,self.body_graph,self.load_caps,self.phase,
                                          b*e,0,rounds,scores.device.index,condition=make_condition(b*e,rounds),early=early)

    def initialize(self):
        ids,loads,caps=self.init_op(self.scores,self.scheduled,self.sequence,self.local)
        self.current.copy_(ids);self.loads.copy_(loads);self.caps.copy_(caps);self.phase.zero_()

    def body(self,op=None):
        ids,loads,phase=(self.body_op if op is None else op)(self.scores,self.scheduled,self.sequence,self.local,self.current,self.loads,self.caps,self.phase)
        self.current.copy_(ids);self.loads.copy_(loads);self.phase.copy_(phase)

    def launch(self,inputs,out,status):
        from ._routing_graph import copy_async
        stream=torch.cuda.current_stream(out.device).cuda_stream
        if stream!=self.stream.cuda_stream:raise RuntimeError('Packed routing plan stream mismatch')
        with torch.cuda.device(out.device),self.lock:
            for dst,src in zip((self.scores,self.scheduled,self.sequence,self.local),inputs):
                copy_async(dst.data_ptr(),src.data_ptr(),src.numel()*src.element_size(),stream)
            self.graph.launch(stream)
            compiled(self.key,out.device.index)[4](self.current.flatten(),self.scheduled,out.flatten())
            # Independent per-sequence capacities are checked by the caller's
            # graph-visible assertion, not a device-to-host load read.
            copy_async(status.data_ptr(),self.load_caps.data_ptr(),self.load_caps.numel()*8,stream)

    def __del__(self):
        try:
            if self.graph is not None:self.graph.close()
        except Exception:pass


@torch.library.custom_op('rtx_routing::varlen_hard',mutates_args=('out','status'),device_types='cuda')
@guard_submission
def hard_execute(inputs:list[torch.Tensor],out:torch.Tensor,status:torch.Tensor,config:list[int],qb:int,rounds:int,policy:list[int],owner:int)->None:
    from concurrent.futures import ThreadPoolExecutor
    from ._managed import install
    install()
    key=_decode_config(tuple(config));scores=inputs[0]
    identity=(scores.device.index,torch.cuda.current_stream(scores.device).cuda_stream,owner,key,qb,rounds,tuple(policy))
    with _LOCK:
        plan=_PLANS.get(identity)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm packed routing on capture stream first')
            stream=torch.cuda.current_stream(scores.device)
            def prepare():
                with torch.cuda.device(scores.device),torch.cuda.stream(stream),torch.no_grad():
                    return HardPlan(*inputs,key,qb,rounds,tuple(policy))
            with ThreadPoolExecutor(max_workers=1,thread_name_prefix='rtx-varlen-routing') as worker:
                plan=worker.submit(prepare).result()
            _PLANS[identity]=plan
    plan.launch(inputs,out,status)


@hard_execute.register_fake
def _hard_fake(inputs,out,status,config,qb,rounds,policy,owner):pass


@torch.library.custom_op('rtx_routing::varlen_metadata',mutates_args=('buffers',),device_types='cuda')
def metadata_execute(inputs:list[torch.Tensor],buffers:list[torch.Tensor],config:list[int],backward:bool)->None:
    key=_decode_config(tuple(config))
    count,prefix,emit,adjoint,reset=compiled(key,inputs[0].device.index)[5]
    if not backward:
        logp,assignments,scheduled,physical,sequence,rowmap=inputs
        counts,ranks,offsets,ids,weights,inverse=buffers
        reset(ids,weights)
        count(logp,assignments,counts,ranks,offsets,ids,weights,inverse)
        prefix(logp,assignments,counts,ranks,offsets,ids,weights,inverse)
        emit(logp,assignments,scheduled,physical,sequence,rowmap,ranks,offsets,ids,weights,inverse)
    else:
        weights,inverse,rowmap,gradient=inputs;out=buffers[0]
        adjoint(weights,inverse,rowmap,gradient,out)


@metadata_execute.register_fake
def _metadata_fake(inputs,buffers,config,backward):pass


class SelectedMetadata(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logp,assignments,scheduled,physical,sequence,rowmap,key):
        t,b,e,k,n,c,*_=key;threads=key[10];device=logp.device
        chunks=(n*k+threads-1)//threads
        counts=torch.empty(e*chunks,device=device,dtype=torch.int32)
        ranks=torch.empty(n*k,device=device,dtype=torch.int32);offsets=torch.empty_like(counts)
        ids=torch.empty((e,c),device=device,dtype=torch.int64);weights=torch.empty((e,c),device=device,dtype=torch.float32)
        inverse=torch.empty((t,k),device=device,dtype=torch.int64)
        metadata_execute([logp.flatten(),assignments.flatten(),scheduled,physical,sequence,rowmap],
            [counts,ranks,offsets,ids.flatten(),weights.flatten(),inverse.flatten()],_flat(key),False)
        ctx.save_for_backward(weights,inverse,rowmap);ctx.key=key
        ctx.mark_non_differentiable(ids,inverse)
        return ids,weights,inverse

    @staticmethod
    def backward(ctx,_ids,gradient,_inverse):
        if gradient is None:return (None,)*7
        weights,inverse,rowmap=ctx.saved_tensors;t,b,e,k,n,*_=ctx.key
        result=torch.empty((n,e),device=weights.device,dtype=torch.float32)
        metadata_execute([weights.flatten(),inverse.flatten(),rowmap,gradient.contiguous().flatten()],
                         [result.flatten()],_flat(ctx.key),True)
        return result,None,None,None,None,None,None


def routing(scores,cu,*,top_k,steps,qb,rounds,config,strict=False,owner=0):
    """CUDA packed route; score projection and all allocations are traceable."""
    from .routing import BalancedRouting
    validate_cu(scores,cu)
    token=config if isinstance(config,tuple) else tuple(config.to_token())
    key=layout(scores.shape[0],cu.numel()-1,scores.shape[1],top_k,steps,token,strict)
    logp,scheduled,physical,sequence,local,rowmap,probabilities=SavedTransport.apply(scores,cu,(top_k,*token),steps,strict)
    return route_from_transport(logp,scheduled,physical,sequence,local,rowmap,cu,key,token,qb,rounds,owner,probabilities)


def route_from_transport(logp,scheduled,physical,sequence,local,rowmap,cu,key,config,qb,rounds,owner,probabilities):
    from .routing import BalancedRouting
    token=config if isinstance(config,tuple) else tuple(config.to_token())
    assignments=torch.empty((key[4],key[3]),device=logp.device,dtype=torch.int64)
    status=torch.empty(2*key[1]*key[2],device=logp.device,dtype=torch.int64)
    hard_execute([probabilities,scheduled,sequence,local],assignments,status,
                 _flat(key),qb,rounds,list(token),owner)
    groups=key[1]*key[2]
    torch._assert_async((status[:groups]==status[groups:]).all(),'Packed QB repair exhausted its per-sequence budget')
    if token[10]==1:
        result=SelectedMetadata.apply(logp,assignments,scheduled,physical,sequence,rowmap,key)
    else:result=_torch_metadata(logp,assignments,scheduled,physical,sequence,rowmap,cu,key)
    return BalancedRouting(*result)


def _torch_metadata(logp,assignments,scheduled,physical,sequence,rowmap,cu,key):
    t,b,e,k,n,c,*_=key;device=logp.device
    active=torch.arange(n,device=device)<scheduled[-1]
    # HardPlan extracts local expert IDs before metadata. Conservative
    # workspace-tail rows are not balanced assignments in any sequence.
    # Sort those rows last, and use actual scheduled counts for real ranks.
    keys=torch.where(active[:,None],assignments,e)
    order=keys.flatten().argsort(stable=True)
    rank=torch.empty_like(order).scatter(0,order,torch.arange(n*k,device=device)).reshape(n,k)
    rank=rank-assignments*(scheduled[-1]*k//e)
    slot=assignments*c+physical[sequence,None]+rank-scheduled[sequence,None]*k//e
    slot=torch.where(active[:,None],slot,0).to(torch.int64)
    safe=torch.where(rowmap>=0,rowmap,-rowmap-1).to(torch.int64)[:,None].expand(n,k)
    # Inactive rows contribute zero, including at slot zero. scatter (rather
    # than scatter_add) would let a dummy row overwrite that real assignment.
    ids=torch.zeros(e*c,device=device,dtype=torch.int64).scatter_add(
        0,slot.flatten(),(safe*active[:,None]).flatten()).reshape(e,c)
    from ._routing_reference import selected_softmax
    values=selected_softmax(logp.gather(1,assignments))*(rowmap>=0)[:,None]*active[:,None]
    weights=torch.zeros(e*c,device=device,dtype=torch.float32).scatter_add(0,slot.flatten(),values.flatten()).reshape(e,c)
    real=torch.arange(t,device=device,dtype=torch.int32)
    seq=torch.searchsorted(cu[1:],real,right=True)
    virtual=scheduled[seq]+real-cu[seq]
    inverse=slot[virtual.to(torch.int64)]
    return ids,weights,inverse


def cpu_routing(router,x,cu):
    """Independent portable reference only; never called by the CUDA path."""
    from .routing import BalancedRouting
    validate_cu(x,cu)
    boundaries=cu.tolist()
    if boundaries[0]!=0 or boundaries[-1]!=x.shape[0] or any(z<a for a,z in zip(boundaries,boundaries[1:])):
        raise ValueError('Invalid cu_seqlens boundaries')
    results=[];offset=0;e=router.num_experts;k=router.top_k
    for start,end in zip(boundaries,boundaries[1:]):
        if start==end:continue
        route=router(x[start:end]);c=route.expert_ids.shape[1];physical=(c+31)//32*32
        ids=torch.nn.functional.pad(route.expert_ids+start,(0,physical-c))
        weights=torch.nn.functional.pad(route.route_weights,(0,physical-c))
        results.append((ids,weights,route.inverse,c,offset));offset+=physical
    key=layout(x.shape[0],cu.numel()-1,e,k,router.sinkhorn_iterations,tuple(router.config.to_token()),router.padding=='error')
    capacity=key[5]
    ids=torch.cat([v[0] for v in results],dim=1);weights=torch.cat([v[1] for v in results],dim=1)
    inverse=torch.cat([(v[2]//v[3])*capacity+v[2]%v[3]+v[4] for v in results],dim=0)
    ids=torch.nn.functional.pad(ids,(0,capacity-offset));weights=torch.nn.functional.pad(weights,(0,capacity-offset))
    return BalancedRouting(ids,weights,inverse)


def _clear_runtime_caches(*,preserve_compiled=False):
    with _LOCK:
        plans=len(_PLANS);_PLANS.clear()
        code=0 if preserve_compiled else len(_COMPILED)
        if not preserve_compiled:_COMPILED.clear()
    return {'varlen_routing_plans':plans,'varlen_routing_compiled':code}
