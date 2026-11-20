"""Whole MoE with matched recomputation: MXFP8 versus checkpointed BF16 FFNs.

The router is identical and retains its saved Sinkhorn state in every mode.
Only routed/shared FFNs are checkpointed. Saved BF16 is a diagnostic, not the
performance target. No optimizer step or autotuning runs in this driver.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


MODES=('mxfp8_recompute','bf16_checkpoint','bf16_saved')


def bf16_compact_capacity(tokens,pools,experts,top_k,alignment=128):
    """Shape-only upper bound on real rows/expert, without scale-group padding.

    Each nonempty pool contributes q*ceil(L/p), p=E/gcd(E,K), q=K/gcd(E,K).
    At most min(T,B) pools are nonempty. This bound covers arbitrary changing
    cu_seqlens with these shapes; no boundary is read on the host.
    """
    if min(tokens,pools,experts,top_k,alignment)<1 or top_k>experts:
        raise ValueError('Positive shape extents and K <= E required')
    divisor=math.gcd(experts,top_k);period=experts//divisor
    logical=(tokens+min(tokens,pools)*(period-1))//period*(top_k//divisor)
    return (logical+alignment-1)//alignment*alignment


def torch_compacted_routed_experts(x,gate_up,down,route,capacity):
    """BF16 diagnostic: compact valid rows, retaining original route gradients.

    Integer compaction and its checkpoint recomputation are timed. Unlike the
    MXFP8 recipe, BF16 does not require sequence-local leading scale groups.
    This is a benchmark reference, never an implicit low-precision fallback.
    """
    from rtx.moe import torch_routed_experts
    ids,weights,inverse=route
    experts,physical=ids.shape
    if capacity>=physical:return torch_routed_experts(x,gate_up,down,route)
    slots=torch.arange(experts*physical,device=x.device).reshape(experts,physical)
    valid=(inverse[ids].eq(slots[...,None])).any(-1)
    ranks=valid.to(torch.int64).cumsum(1)-1
    positions=(torch.arange(experts,device=x.device)[:,None]*capacity+ranks.clamp_min(0)).flatten()
    # Zero is reserved for padding. Invalid zero contributions cannot overwrite
    # the valid token zero; true ranks are unique within an expert.
    compact_ids=torch.zeros(experts*capacity,device=x.device,dtype=torch.int64).scatter_reduce(
        0,positions,torch.where(valid,ids+1,0).flatten(),reduce='amax',include_self=True).reshape(experts,capacity)-1
    gathered=x[compact_ids.clamp_min(0)]*(compact_ids>=0)[...,None]
    projection=torch.bmm(gathered,gate_up.transpose(1,2)).float()
    h=down.shape[2]
    hidden=(torch.nn.functional.silu(projection[...,:h])*projection[...,h:]).bfloat16()
    output=torch.bmm(hidden,down.transpose(1,2))
    compact_inverse=inverse//physical*capacity+ranks.flatten()[inverse]
    return (output.flatten(0,1)[compact_inverse].float()*weights.flatten()[inverse][...,None]).sum(1)


class BF16FFNReference(nn.Module):
    """Benchmark-only precision reference; never checkpoint the router."""
    def __init__(self,model,*,checkpoint_ffns:bool,use_reentrant:bool=True,expert_layout='native'):
        super().__init__()
        if expert_layout not in ('native','compact'):raise ValueError('Unknown BF16 expert layout')
        self.model=model
        self.checkpoint_ffns=checkpoint_ffns
        self.use_reentrant=use_reentrant
        self.expert_layout=expert_layout

    @staticmethod
    def shared(value,gate_up,down):
        projection=torch.nn.functional.linear(value,gate_up).float()
        hidden_width=down.shape[1]
        hidden=(torch.nn.functional.silu(projection[:,:hidden_width])*projection[:,hidden_width:]).bfloat16()
        return torch.nn.functional.linear(hidden,down)

    def forward(self,x,*,cu_seqlens=None):
        from rtx.moe import torch_routed_experts
        from rtx.routing import BalancedRouting
        model=self.model
        route=model.router(x) if cu_seqlens is None else model.router(x,cu_seqlens=cu_seqlens)
        gu,down=model.gate_up.weight,model.down.weight
        capacity=bf16_compact_capacity(x.shape[0],1 if cu_seqlens is None else cu_seqlens.numel()-1,
            gu.shape[0],route.inverse.shape[1]) if self.expert_layout=='compact' else None
        def routed(value,gate,output,weights):
            selected=BalancedRouting(route.expert_ids,weights,route.inverse)
            if capacity is not None:return torch_compacted_routed_experts(value,gate,output,selected,capacity)
            return torch_routed_experts(value,gate,output,selected)
        if self.checkpoint_ffns:
            result=checkpoint(routed,x,gu,down,route.route_weights,use_reentrant=self.use_reentrant)
        else:
            result=routed(x,gu,down,route.route_weights)
        for gate,output in zip(model.shared_gate_up,model.shared_down):
            if self.checkpoint_ffns:
                shared=checkpoint(self.shared,x,gate.weight,output.weight,use_reentrant=self.use_reentrant)
            else:
                shared=self.shared(x,gate.weight,output.weight)
            result=result+shared.float()
        return result.bfloat16()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--case-id',default='d768_e48_k6_h256')
    p.add_argument('--tokens',type=int,default=2000)
    p.add_argument('--seed-studies',nargs='*',type=Path,default=[])
    p.add_argument('--config',type=Path)
    p.add_argument('--router-config',type=Path)
    p.add_argument('--shared-config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--samples',type=int,default=9)
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--nonreentrant',action='store_true')
    p.add_argument('--reference',type=Path)
    p.add_argument('--save-reference',type=Path)
    p.add_argument('--grad-mode',choices=('set_to_none','preallocated_zero'),default='set_to_none')
    return p


def metrics(values,references,*,gate=None):
    if len(values)!=len(references):raise ValueError('Reference gradient arity differs')
    result=[]
    for index,(value,reference) in enumerate(zip(values,references)):
        value=value.detach().float().cpu();reference=reference.detach().float().cpu()
        if value.shape!=reference.shape:raise ValueError(('Reference shape differs',index))
        if not bool(value.isfinite().all()) or not bool(reference.isfinite().all()):
            raise AssertionError(('Nonfinite output or gradient',index))
        delta=value-reference
        maximum=float(delta.abs().max())
        nrms=float(delta.norm()/reference.norm().clamp_min(1e-12))
        if gate is not None and nrms>gate and maximum>2e-7:
            raise AssertionError((index,nrms,maximum))
        result.append(dict(nrms=nrms,max_abs=maximum))
    return result


def routing_identity(route):
    """Keep exact discrete assignments separate from tolerant arithmetic gates."""
    return dict(expert_ids=route.expert_ids.detach().cpu(),inverse=route.inverse.detach().cpu())


def compare_routing_identity(actual,reference):
    if set(actual)!=set(reference):raise ValueError('Routing metadata fields differ')
    for name in actual:
        if not torch.equal(actual[name],reference[name]):
            raise AssertionError(f'Routing metadata differs: {name}')
    return dict(expert_ids_exact=True,inverse_exact=True)


def _router_oracle(model,x,coefficients):
    """Differentiate FP64 normalizations on the actual detached assignment."""
    import math
    from rtx import _routing_reference as reference
    router=model.router
    route=router(x)
    k=router.top_k;capacity=route.route_weights.shape[1]
    actual=torch.autograd.grad(route.route_weights,(x,*router.parameters()),coefficients)
    scores=router.log_affinities(x)
    scheduled=reference.padded_token_count(x.shape[0],router.num_experts,k)
    scores=torch.nn.functional.pad(scores,(0,0,0,scheduled-x.shape[0]),value=-math.log(2))
    value=scores.double()
    for _ in range(router.sinkhorn_iterations):
        value=value-torch.logsumexp(value,1,keepdim=True)+math.log(k)
        value=value-torch.logsumexp(value,0,keepdim=True)+math.log(scheduled*k/router.num_experts)
    selected=route.inverse//capacity
    probabilities=value[:x.shape[0]].gather(1,selected).softmax(1)
    upstream=coefficients.flatten()[route.inverse]
    expected=torch.autograd.grad(probabilities,(x,*router.parameters()),upstream.double())
    rows=metrics(actual,expected,gate=.015)
    token=torch.arange(x.shape[0],device=x.device)[:,None].expand_as(route.inverse)
    torch.testing.assert_close(route.expert_ids.flatten()[route.inverse],token,rtol=0,atol=0)
    assert not bool((selected.sort(1).values.diff(dim=1)==0).any())
    torch.testing.assert_close(route.route_weights.flatten()[route.inverse].sum(1),
                               torch.ones(x.shape[0],device=x.device))
    return dict(scope='independent FP64 Sinkhorn on actual frozen hard assignment; all router/input gradients',errors=rows)


def gradient_clearer(leaves,mode,options):
    """Normal fresh-gradient training versus explicitly accumulated buffers."""
    if mode=='set_to_none':
        def clear():
            for value in leaves:value.grad=None
        return clear
    if mode!='preallocated_zero':raise ValueError('Unknown gradient mode')
    for value in leaves:value.grad=torch.zeros_like(value)
    gradients=tuple(value.grad for value in leaves)
    def clear():
        for value in gradients:value.zero_()
    return torch.compile(clear,**options)


def replay_gradient_check(step,leaves,reference,stream):
    """Graph replay must overwrite fresh gradients, not silently accumulate."""
    for _ in range(2):
        output=step();del output
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output=step();del output
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):output=step()
    torch.cuda.current_stream().wait_stream(stream)
    addresses=[value.grad.data_ptr() for value in leaves]
    rows=[]
    for _ in range(2):
        graph.replay();torch.cuda.synchronize()
        if [value.grad.data_ptr() for value in leaves]!=addresses:
            raise AssertionError('Graph replay retargeted leaf gradients')
        actual=[output,*[value.grad for value in leaves]]
        rows.append(metrics(actual,reference,gate=.006))
    graph.reset();del output
    return dict(replays=2,leaf_grad_addresses_stable=True,errors=rows)


def main():
    args=parser().parse_args()
    if args.tokens<1 or args.samples<1 or args.rounds<1:raise ValueError('Positive shape and timing counts required')
    for name in ('source','output','config','router_config','shared_config','reference','save_reference'):
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.resolve())
    args.seed_studies=[path.resolve() for path in args.seed_studies]
    spec=importlib.util.spec_from_file_location('moe_execution_benchmark',Path(__file__).with_name('benchmark_moe_execution.py'))
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    helper=bench._use_source(args.source)
    from rtx.mlp import MXFP8MLPConfig
    from rtx.routing import RoutingKernelConfig
    construction=['--case-id',args.case_id,'--tokens',str(args.tokens),'--whole-layer','--skip-baseline','--output',str(args.output)]
    if args.seed_studies:construction+=['--seed-studies',*[str(path) for path in args.seed_studies]]
    if args.config:construction+=['--config',str(args.config)]
    case,shape,config,origin,operands,dy,calls=helper.build(helper.parser().parse_args(construction))
    x=operands[0];dy=dy.bfloat16()
    model=inspect.getclosurevars(calls['whole_step']).nonlocals['native']._orig_mod
    if args.router_config:
        payload=json.loads(args.router_config.read_text())
        router=RoutingKernelConfig.from_dict(payload.get('config',payload))
        model.router.config=router;model.router.sinkhorn_config=router.forward
        model.router.autotune='off';model.router._refresh_selection_spec()
    if args.shared_config:
        payload=json.loads(args.shared_config.read_text())
        model.shared_config=MXFP8MLPConfig.from_dict(payload.get('config',payload));model._shared_spec=None
    parameters=tuple(model.parameters());leaves=(x,*parameters)
    names=('output','input',*(name for name,_ in model.named_parameters()))
    options=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    zero=gradient_clearer(leaves,args.grad_mode,options)
    eager={'mxfp8_recompute':model,
        'bf16_checkpoint':BF16FFNReference(model,checkpoint_ffns=True,use_reentrant=not args.nonreentrant),
        'bf16_saved':BF16FFNReference(model,checkpoint_ffns=False)}
    forwards={};steps={};tensors={};validation={}
    began=time.monotonic()
    for mode in MODES:
        forward=torch.compile(eager[mode],**options)
        forwards[mode]=lambda forward=forward:forward(x)
        def step(forward=forward):
            zero();output=forward(x);output.backward(dy);return output
        steps[mode]=step
        output=step();torch.cuda.synchronize()
        tensors[mode]=[output.detach().cpu(),*[value.grad.detach().cpu().clone() for value in leaves]]
        metrics(tensors[mode],tensors[mode])
        del output
    validation['bf16_checkpoint_vs_saved']=metrics(tensors['bf16_checkpoint'],tensors['bf16_saved'],gate=.006)
    validation['mxfp8_vs_unquantized_bf16']=metrics(tensors['mxfp8_recompute'],tensors['bf16_saved'])
    # The frozen expert fixture still runs an independent quantized FP64 oracle.
    from rtx.moe import _run_forward,_run_backward
    def expert_step():
        return (_run_forward(*operands,config),*_run_backward(*operands,dy.float(),config))
    validation['expert_oracle']=helper.correctness(operands,dy.float(),{'expert_step':expert_step},args.tokens)
    coefficient=_run_backward(*operands,dy.float(),config)[-1].detach()
    validation['router_oracle']=_router_oracle(model,x,coefficient)
    with torch.no_grad():route_identity=routing_identity(model.router(x))
    input_digests={'x':bench._tensor_digest(x),'dy':bench._tensor_digest(dy),
                   **{name:bench._tensor_digest(value) for name,value in model.named_parameters()}}
    replay_stream=torch.cuda.Stream()
    validation['repeated_graph_gradients']={mode:replay_gradient_check(steps[mode],leaves,tensors[mode],replay_stream)
                                           for mode in args.modes}
    if args.reference:
        reference=torch.load(args.reference,map_location='cpu',weights_only=True)
        if list(reference['names'])!=list(names):raise ValueError('Reference Parameter names/order differ')
        if reference['input_digests']!=input_digests:raise ValueError('Frozen inputs/Parameters differ')
        validation['source_routing_identity']=compare_routing_identity(route_identity,reference['routing_identity'])
        validation['source_parity']={mode:metrics(tensors[mode],reference['modes'][mode],gate=.006) for mode in MODES}
    if args.save_reference:
        args.save_reference.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(names=names,modes=tensors,routing_identity=route_identity,input_digests=input_digests),args.save_reference)
    record=dict(status='running',case=case,shape=shape,source=str(args.source),
        source_digest=bench._python_source_digest(args.source),driver_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
        commands=sys.argv,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        seed_origin=origin,expert_config=config.to_dict(),router_config=model.router.config.to_dict(),shared_config=model.shared_config.to_dict(),
        compile_options=options,checkpoint_use_reentrant=not args.nonreentrant,
        contract='same native saved-state router in every mode; checkpoint ONLY BF16 FFNs; actual backward and identical gradient clearing policy; saved BF16 diagnostic only; no optimizer/tuning',
        grad_mode=args.grad_mode,
        parameter_count=len(parameters),gradient_dtypes={name:str(value.dtype) for name,value in zip(names[1:],leaves)},
        input_digests=input_digests,
        validation=validation,compile_warmup_seconds=time.monotonic()-began,
        timings={mode:{phase:dict(host=[],graph=[]) for phase in ('forward','step')} for mode in args.modes})
    bench.write(args.output,record)
    stream=torch.cuda.Stream()
    for round_id in range(args.rounds):
        order=args.modes if round_id%2==0 else list(reversed(args.modes))
        for mode in order:
            for phase,call in (('forward',forwards[mode]),('step',steps[mode])):
                host=bench._host(call,args.samples);graph=helper.measure(call,stream,args.samples)
                record['timings'][mode][phase]['host'].append(host);record['timings'][mode][phase]['graph'].append(graph)
                print('TIME',round_id,mode,phase,f"host={host['median_ms']:.6f}ms graph={graph['median_ms']:.6f}ms",flush=True)
                bench.write(args.output,record)
    memory={}
    for mode in args.modes:
        zero();torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();live=[]
        for _ in range(8):
            output=steps[mode]();del output;torch.cuda.synchronize();live.append(torch.cuda.memory_allocated())
        memory[mode]=dict(baseline_allocated_bytes=before,temporary_peak_bytes=torch.cuda.max_memory_allocated()-before,
                          live_growth_bytes=max(live)-min(live))
    record.update(status='complete',memory=memory,
        memory_contract='same warmed process, including all mode caches/graph pools; not isolated-process VRAM')
    bench.write(args.output,record)
    for mode in args.modes:
        print('RESULT',mode,{phase:statistics.median(row['median_ms'] for row in record['timings'][mode][phase]['graph'])
                            for phase in ('forward','step')},flush=True)


if __name__=='__main__':main()
