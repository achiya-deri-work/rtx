"""One frozen shared MLP: saved BF16, checkpoint BF16, recomputed MXFP8.

No tuning. All modes use the same BF16 VirtualLayer Parameters and upstream
gradient. Full backward uses .backward(), supporting reentrant checkpointing.
"""
import argparse
import gc
import importlib.util
import inspect
import json
from pathlib import Path
import statistics
import time


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--seed-study",type=Path,required=True)
    p.add_argument("--shared-config",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--samples",type=int,default=9)
    return p.parse_args()


def errors(values,expected):
    result=[]
    for value,reference in zip(values,expected):
        delta=value.detach().float().cpu()-reference.float()
        result.append(dict(nrms=float(delta.norm()/reference.float().norm().clamp_min(1e-12)),
                           max_abs=float(delta.abs().max())))
    return result


def main():
    args=arguments()
    for name in ("source","seed_study","output","shared_config"):
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.resolve())
    location=Path(__file__).with_name("benchmark_moe_execution.py")
    spec=importlib.util.spec_from_file_location("rtx_shared_benchmark_helpers",location)
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    helper=bench._use_source(args.source)
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    from rtx.mlp import mxfp8_mlp,MXFP8MLPConfig
    from rtx.runtime import clear_runtime_caches
    construction=helper.parser().parse_args(["--case-id","d768_e48_k6_h256","--tokens","2000",
        "--seed-studies",str(args.seed_study),"--whole-layer","--skip-baseline","--output",str(args.output)])
    case,shape,config,origin,operands,upstream,calls=helper.build(construction)
    x=operands[0];dy=upstream.bfloat16()
    model=inspect.getclosurevars(calls["whole_step"]).nonlocals["native"]._orig_mod
    shared_config=model.shared_config
    if args.shared_config:
        payload=json.loads(args.shared_config.read_text())
        shared_config=MXFP8MLPConfig.from_dict(payload.get("config",payload))

    class Core(nn.Module):
        def __init__(self,gate,down):
            super().__init__();self.gate,self.down=gate,down
        def saved(self,value):
            projected=torch.nn.functional.linear(value,self.gate.weight).float()
            h=self.down.weight.shape[1]
            hidden=(torch.nn.functional.silu(projected[:,:h])*projected[:,h:]).bfloat16()
            return torch.nn.functional.linear(hidden,self.down.weight)
        def forward(self,value,mode):
            if mode=="bf16_saved":return self.saved(value)
            if mode=="bf16_checkpoint_reentrant":return checkpoint(self.saved,value,use_reentrant=True)
            if mode=="bf16_checkpoint_nonreentrant":return checkpoint(self.saved,value,use_reentrant=False)
            return mxfp8_mlp(value,self.gate.weight,self.down.weight,activation="silu",config=shared_config)

    core=Core(model.shared_gate_up[0],model.shared_down[0])
    del calls,operands,model,upstream,config
    gc.collect();clear_runtime_caches(device=x.device,preserve_compiled=True);torch.cuda.empty_cache()
    parameters=tuple(core.parameters());names=("input",*(name for name,_ in core.named_parameters()))
    leaves=(x,*parameters)
    assert len(parameters)==3 and all(value.dtype==torch.bfloat16 for value in leaves)
    for value in leaves:value.grad=torch.zeros_like(value)
    gradients=tuple(value.grad for value in leaves)
    options=dict(fullgraph=True,dynamic=False,options={"triton.cudagraphs":False})
    def zero():
        for value in gradients:value.zero_()
    zero=torch.compile(zero,**options)
    modes=("bf16_saved","bf16_checkpoint_reentrant","bf16_checkpoint_nonreentrant","mxfp8_recompute")
    forwards={};steps={};validation={};reference=None
    started=time.monotonic()
    for mode in modes:
        def make_forward(selected):return torch.compile(lambda value:core(value,selected),**options)
        fn=make_forward(mode);forwards[mode]=lambda fn=fn:fn(x)
        def step(fn=fn):
            zero();output=fn(x);output.backward(dy);return output
        steps[mode]=step
        output=step();torch.cuda.synchronize()
        values=[output.detach().cpu(),*[leaf.grad.detach().cpu().clone() for leaf in leaves]]
        assert all(bool(value.isfinite().all()) for value in values)
        if reference is None:reference=values
        metric=errors(values,reference)
        if mode.startswith("bf16_checkpoint"):
            assert all(row["nrms"]<.006 or row["max_abs"]<2e-7 for row in metric),(mode,metric)
        validation[mode]=dict(all_finite=True,against_saved_bf16=metric,
                             names=("output",*names),grad_addresses=[leaf.grad.data_ptr() for leaf in leaves])
        del output,values
    cold=time.monotonic()-started
    record=dict(status="running",shape=dict(tokens=2000,input=768,hidden=512,output=768),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        source=str(args.source),source_digest=bench._python_source_digest(args.source),
        driver_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        seed_origin=origin,shared_config=shared_config.to_dict(),validation=validation,
        input_digests={"x":bench._tensor_digest(x),"dy":bench._tensor_digest(dy),
            **{name:bench._tensor_digest(value) for name,value in core.named_parameters()}},
        contract="same weights/input/upstream, FP32 activation/gating, BF16 hidden for BF16 modes; preallocated BF16 grads; compiled zeroing included in every backward",
        compile_options=options,compile_warmup_seconds=cold,
        timings={mode:{phase:dict(host=[],graph=[]) for phase in ("forward","step")} for mode in modes})
    bench.write(args.output,record)
    stream=torch.cuda.Stream()
    for round_id in range(args.rounds):
        order=modes if round_id%2==0 else tuple(reversed(modes))
        for mode in order:
            for phase,call in (("forward",forwards[mode]),("step",steps[mode])):
                host=bench._host(call,args.samples);graph=helper.measure(call,stream,args.samples)
                record["timings"][mode][phase]["host"].append(host)
                record["timings"][mode][phase]["graph"].append(graph)
                print("TIME",round_id,mode,phase,f"host={host['median_ms']:.6f}ms graph={graph['median_ms']:.6f}ms",flush=True)
                bench.write(args.output,record)
    memory={}
    persistent={value.untyped_storage().data_ptr() for value in leaves}
    for mode in modes:
        saved={}
        def pack(tensor):
            storage=tensor.untyped_storage();pointer=storage.data_ptr()
            saved[pointer]=dict(bytes=storage.nbytes(),persistent=pointer in persistent,
                                dtype=str(tensor.dtype),shape=list(tensor.shape))
            return tensor
        zero()
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value):
            output=forwards[mode]()
        output.backward(dy);del output;torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();live=[]
        for _ in range(6):
            output=steps[mode]();del output;torch.cuda.synchronize();live.append(torch.cuda.memory_allocated())
        memory[mode]=dict(baseline_allocated_bytes=before,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            temporary_peak_bytes=torch.cuda.max_memory_allocated()-before,live_growth_bytes=max(live)-min(live),
            saved_unique_bytes=sum(row["bytes"] for row in saved.values()),
            saved_nonpersistent_bytes=sum(row["bytes"] for row in saved.values() if not row["persistent"]),
            saved_storages=list(saved.values()))
    record.update(status="complete",memory=memory,
        saved_tensor_contract="autograd saved-storage hooks, separately instrumented after timing; not kernel-private traffic or a GEMM-count trace",
        checkpoint_contract="public use_reentrant True/False under Torch compilation; actual AOT rematerialization is compiler-selected, no extra GEMM count asserted")
    bench.write(args.output,record)
    for mode in modes:
        print("RESULT",mode,{phase:statistics.median(row["median_ms"] for row in record["timings"][mode][phase]["graph"])
                            for phase in ("forward","step")},memory[mode],flush=True)


if __name__=="__main__":main()
