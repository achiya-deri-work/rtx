"""Independent dense-MLP store/partition controls and warmed phase traces.

This is a finite diagnostic envelope, not an exhaustive optimum or a runtime
winner installation. Quantization/activation arithmetic stays with the anchor.
"""
from dataclasses import replace
import argparse
import gc
import json
from pathlib import Path


def controls(adapter):
    from rtx.autotune.mlp import mlp_row_store_seeds,mlp_structural_seeds
    base=adapter.initial_config;w=adapter.context.workload
    yield 'anchor',base
    def direct(c):
        c=replace(c,down_gemm=replace(c.down_gemm,epilogue='direct',store_vec=1,epilogue_stages=1))
        for group in ('gate_backward','down_backward'):
            b=getattr(c,group)
            if b.dx.backend=='decomposed':
                c=replace(c,**{group:replace(b,dx=replace(b.dx,gemm=replace(b.dx.gemm,
                    epilogue='direct',store_vec=1,epilogue_stages=1)))})
        return c
    a=direct(base);yield 'all_direct',a
    geometries=[a]
    for rows in (64,128):
        c=a
        for role in ('down_gemm','gate_backward','down_backward'):
            g=c.down_gemm if role=='down_gemm' else getattr(c,role).dx.gemm
            g=replace(g,tile_m=rows,atom_layout_m=2 if rows==64 else 4,stages=max(2,g.stages))
            if role=='down_gemm':c=replace(c,down_gemm=g)
            else:
                b=getattr(c,role);c=replace(c,**{role:replace(b,dx=replace(b.dx,gemm=g))})
        geometries.append(c)
    for index,c in enumerate(geometries):
        stores=list(mlp_row_store_seeds(adapter,c,'train'))
        if stores:yield f'stores_geometry_{index}',stores[-1]
    # Existing structural and SM-aware balanced dW recipes form independent
    # partition controls. Only their partition fields transfer to the anchor.
    donors=list(mlp_structural_seeds(adapter,'mxfp8','train'))
    from rtx._bwd_search import balanced_dw_candidates
    from rtx.kernels.mxfp8 import MXFP8Problem
    balanced=a
    for group,n,k in (('gate_backward',2*w['h'],w['k']),('down_backward',w['o'],w['h'])):
        candidates=balanced_dw_candidates(MXFP8Problem(w['m'],n,k),getattr(a,group),
                                         adapter.context.device.get('multiprocessor_count'))
        if candidates:balanced=replace(balanced,**{group:candidates[0]})
    donors.insert(0,balanced)
    seen=set()
    for donor in donors:
        plan=tuple((getattr(donor,g).dw.split_reduction,getattr(donor,g).dw.split_partition,
                    getattr(donor,g).dw.reduction_tile) for g in ('gate_backward','down_backward'))
        if plan in seen:continue
        seen.add(plan)
        c=a
        for group in ('gate_backward','down_backward'):
            b=getattr(c,group);d=getattr(donor,group).dw
            c=replace(c,**{group:replace(b,dw=replace(b.dw,split_reduction=d.split_reduction,
                split_partition=d.split_partition,reduction_tile=d.reduction_tile,reduction=d.reduction))})
        if adapter.rejection(c) is None:
            yield f'direct_partition_{len(seen)}',c
            stores=list(mlp_row_store_seeds(adapter,c,'train'))
            if stores:yield f'stores_partition_{len(seen)}',stores[-1]
        if len(seen)>=3:break
    for path in ('recompute','recompute_both'):
        c=adapter.update_fn(a,'gradient_quantization_path',path)
        stores=list(mlp_row_store_seeds(adapter,c,'train'))
        if stores:yield 'stores_'+path,stores[-1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trace',action='store_true')
    p.add_argument('--match-inputs',action='store_true',help='Reconstruct and verify the source study X/shared weights for a matched finite search reference')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    from rtx._managed import tuning_resources
    from rtx.autotune.mlp import make_mlp_adapter
    from rtx.mlp import MXFP8MLPConfig,mxfp8_mlp
    from rtx.autotune.outcomes import raise_if_fatal_device_context_error
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    d=json.loads(args.seed_result.read_text());m=d['total_tokens'];k=d['case']['d_model'];h=d['case']['shared_hidden_dim']
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream);torch.manual_seed(87141)
    if args.match_inputs:
        from rtx.moe import BalancedMoE
        from benchmarks.benchmark_moe_execution import _tensor_digest
        init=d['initialization_case'];torch.manual_seed(d['seed'])
        model=BalancedMoE(init['d_model'],init['num_experts'],init['expert_hidden_dim'],
            top_k=init['top_k'],shared_hidden_features=init['shared_hidden_dim'],
            num_shared_experts=init['num_shared_experts'],device='cuda',autotune='off',
            router_options={'backend':'cute','autotune':'off'})
        with torch.no_grad():model.router.projection_weight.normal_(std=.05)
        x=(torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
        assert _tensor_digest(x)==d['identities']['input']
        for name,value in model.named_parameters():
            if name.startswith('shared_'):
                assert _tensor_digest(value)==d['identities']['parameters'][name],name
        leaves=(x,model.shared_gate_up[0].weight,model.shared_down[0].weight)
    else:
        leaves=tuple((torch.randn(*s,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
            for s,scale in (((m,k),.3),((2*h,k),.02),((k,h),.02)))
    anchor=MXFP8MLPConfig.from_dict(d['shared_config'])
    report=dict(shape=[m,k,h,k],seed=d['seed'] if args.match_inputs else 87141,
        source_result=str(args.seed_result),inputs_matched=args.match_inputs,
        input_contract='Hash-verified source study X/shared weights' if args.match_inputs else
                       'Independent seeded BF16 dense fixture, not the MoE model weights',records={},status='running')
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
    @tuning_resources
    def evaluate():
        a=make_mlp_adapter(*leaves,initial=anchor,warmup=3,samples=7,calls_per_sample=3)
        seen=set();eligible=[]
        for label,c in controls(a):
            key=a.config_id(c)
            if key in seen:continue
            seen.add(key)
            why=a.rejection(c)
            if why:
                report['records'][label]=dict(config=c.to_dict(),rejection=why);save();continue
            print('EVAL',label,flush=True)
            outcome=a.evaluate(c);raise_if_fatal_device_context_error(outcome.error or '')
            report['records'][label]=dict(config_id=key,config=c.to_dict(),outcome=outcome.as_dict());save()
            print('TIME',label,outcome.status,outcome.median_ms,flush=True)
            if outcome.status not in ('ok','compile_error'):raise AssertionError((label,outcome.error))
            if outcome.timing_eligible:eligible.append((outcome.median_ms,label,c))
        if not eligible:raise RuntimeError('No eligible finite control')
        return min(eligible,key=lambda r:r[0])
    score,label,chosen=evaluate();report.update(status='complete',best=dict(label=label,median_ms=score,config=chosen.to_dict()));save()
    if args.trace:
        dy=torch.randn_like(leaves[0])*.1
        for tag,c in (('anchor',anchor),('frontier_best',chosen)):
            fn=torch.compile(lambda x,gu,down:mxfp8_mlp(x,gu,down,config=c),
                fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
            def step():
                out=fn(*leaves)
                return out,*torch.autograd.grad(out,leaves,dy)
            for _ in range(3):value=step();del value
            stream.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):value=step()
            _warm_measurement_graph(graph);stream.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                graph.replay();stream.synchronize()
            trace=args.output.with_name(args.output.stem+'.'+tag+'.trace.json')
            prof.export_chrome_trace(str(trace));graph.reset();del value;gc.collect()
            report.setdefault('traces',{})[tag]=str(trace);save()


if __name__=='__main__':main()
