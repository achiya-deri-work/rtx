"""Whole-recomputed-MLP tuning using the shared persistent search portfolio."""
from dataclasses import asdict,replace
import json
from pathlib import Path
from functools import lru_cache
import statistics
import time

import torch

from .core import DiscreteKernelAdapter,KernelContext,stable_id
from .legacy import DeviceFingerprint
from .outcomes import TrialOutcome,raise_if_fatal_device_context_error
from .recipes import HybridTuningPolicy,make_hybrid_autotuner
from .store import ResidualTuningStore
from .glu import GLU_SEARCH_SPACE
from ..mlp import DEFAULT_MLP_CONFIG,MLP_CONFIG_REVISION,_parse,_validate,mxfp8_mlp
from ..configs.glu import normalize_glu_config,glu_rejection
from ..kernels.mxfp8 import FWD_SEARCH_SPACE,MXFP8Problem,normalize_fwd_config


@lru_cache(maxsize=16)
def _mlp_hardware(device):
    from .hardware import static_device_profile
    return {**DeviceFingerprint.current(device).as_dict(),**static_device_profile(device)}


def mlp_geometry_features(config,m,k,h,o,profile,precision="mxfp8",phase="train"):
    """Pre-compile geometry estimates; never claim measured occupancy."""
    from .hardware import geometry_features
    gate=config.gate_up if precision=="mxfp8" and config.gate_backend=="onchip" else config.gate_prequant.gemm
    down=config.down if precision=="mxfp8" and config.down_backend=="onchip" else config.down_gemm
    roles={"gate_up":(m,2*h if precision=="mxfp8" else h,k,gate,1),
           "down":(m,o,h,down,1)}
    for name,rows,cols,reduction,bwd in (
        ("gate_dx",m,k,2*h,config.gate_backward.dx),
        ("gate_dw",2*h,k,m,config.gate_backward.dw),
        ("down_dx",m,h,o,config.down_backward.dx),
        ("down_dw",o,h,m,config.down_backward.dw)):
        if phase=="train":
            roles[name]=(rows,cols,reduction,bwd.fused if bwd.backend=="fused" else bwd.gemm,bwd.split_reduction)
    if phase=="train":
        recompute=config.gate_up if config.recompute_backend=="onchip" else config.recompute.gemm
        roles["recompute"]=(m,2*h if precision=="mxfp8" else h,k,recompute if precision=="mxfp8" else gate,1)
    result={}
    for name,(rows,cols,reduction,schedule,split) in roles.items():
        ctas=((rows+schedule.tile_m-1)//schedule.tile_m)*((cols+schedule.tile_n-1)//schedule.tile_n)*split
        values=geometry_features(m=rows,n=cols,k=reduction,tile_m=schedule.tile_m,
            tile_n=schedule.tile_n,tile_k=schedule.tile_k,profile=profile,grid_ctas=ctas)
        result.update({name+"."+key:value for key,value in values.items()})
    return result


def active_schedule(config,precision,phase):
    """Deduplicate schedules differing only in provably inactive fields.

    Full configurations remain serialized for reproducibility and switching
    backends. This identity must change whenever execution ownership changes.
    """
    c=asdict(config)
    if precision=="bf16_gate":
        from .._mlp_precision import prequant_config
        result={"up":asdict(prequant_config(config)),"down":asdict(prequant_config(config,True))}
    else:
        result={"gate_backend":c["gate_backend"],"gate_up":c["gate_up"],"down_backend":c["down_backend"]}
        if config.gate_backend=="cooperative_prequant":
            result.update(gate_prequant=c["gate_prequant"],cooperative_ctas_per_sm=c["cooperative_ctas_per_sm"])
        if config.down_backend=="onchip":
            result["down"]=c["down"]
        else:
            result.update(down_quant=c["down_quant"],down_gemm=c["down_gemm"])
        if phase=="train":
            result.update(backward_recompute=c["backward_recompute"],backward_overlap=c["backward_overlap"],
                          recompute_backend=c["recompute_backend"])
            if config.recompute_backend=="prequant":
                result["recompute"]=c["recompute"]
    if phase=="train":
        for group in ("gate_backward","down_backward"):
            bwd=asdict(getattr(config,group).normalized())
            for name in ("dx","dw"):
                leg=bwd[name]
                inactive=("quant_a","quant_b","gemm","quant_launches") if leg["backend"]=="fused" else ("fused",)
                for field in inactive:
                    leg.pop(field,None)
                if leg["reduction"]=="full_fp32":
                    for field in ("reduction_threads","reduction_vector","reduction_waves"):
                        leg.pop(field,None)
            result[group]=bwd
    def prune(node):
        if not isinstance(node,dict):
            return
        if node.get("glu_store")=="direct":
            node.pop("glu_store_stages",None)
        if phase=="forward":
            for field in ("glu_grad_load","glu_grad_store","glu_hidden_decode"):
                node.pop(field,None)
        for value in node.values():
            prune(value)
    prune(result)
    return result


def reference(x,gu,down,dy,activation,precision="mxfp8"):
    """Independent FP32 arithmetic, with explicit per-GEMM MXFP8 rounding."""
    def q(t):
        width=t.shape[1]
        blocks=torch.nn.functional.pad(t.float(),(0,(-width)%32)).reshape(t.shape[0],-1,32)
        amax=blocks.abs().amax(-1,keepdim=True)
        exponent=torch.ceil(torch.log2((amax/448).clamp_min(2.**-126)))
        scale=torch.exp2(exponent)
        return ((blocks/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale).flatten(1)[:,:width]
    g,u=(q(x)@q(gu).T).chunk(2,-1)
    if precision=="bf16_gate":
        g=torch.nn.functional.linear(x,gu[:gu.shape[0]//2]).float()
        u=u.bfloat16().float()
    a=getattr(torch.nn.functional,activation)(g)
    h=(a*u).bfloat16() if precision=="bf16_gate" else q(a*u).bfloat16()
    y=((q(h) if precision=="bf16_gate" else h.float())@q(down).T).bfloat16()
    dh=(q(dy)@q(down.T).T).bfloat16().float()
    dd=(q(dy.T)@q(h.T).T).bfloat16()
    if activation=="silu":
        s=torch.sigmoid(g);d=s*(1+g*(1-s))
    elif activation=="gelu":
        d=.5*(1+torch.erf(g/2**.5))+g*torch.exp(-g*g/2)/(2*torch.pi)**.5
    else:
        d=(g>0).float()
    dp=torch.cat((dh*u*d,dh*a),-1).bfloat16()
    dx=(q(dp)@q(gu.T).T).bfloat16()
    dg=(q(dp.T)@q(x.T).T).bfloat16()
    return y,dx,dg,dd


def make_mlp_adapter(x,gu,down,*,activation="silu",phase="train",initial=DEFAULT_MLP_CONFIG,
                     precision="mxfp8",warmup=3,samples=7,calls_per_sample=3):
    from ..fp8_bwd import _default_bwd_config
    from ..bwd_autotune import BWD_SEARCH_SPACE,update_bwd_config
    from ..kernels.mxfp8_mlp import mixed_down_rejection
    _validate(x,gu,down,activation)
    if precision not in ("mxfp8","bf16_gate"):
        raise ValueError("tunable precision must be mxfp8 or bf16_gate; BF16 is compiler-managed")
    if phase not in ("forward","train") or min(warmup,samples,calls_per_sample)<1:
        raise ValueError("phase must be forward/train and timing counts positive")
    x=x.detach().reshape(-1,x.shape[-1]).requires_grad_()
    gu=gu.detach().requires_grad_();down=down.detach().requires_grad_()
    m,k=x.shape;h,o=gu.shape[0]//2,down.shape[0]
    gp,dp=MXFP8Problem(m,2*h,k),MXFP8Problem(m,o,h)
    initial=replace(initial,gate_backward=initial.gate_backward or _default_bwd_config(gp),
                    down_backward=initial.down_backward or _default_bwd_config(dp))
    generator=torch.Generator(device=x.device).manual_seed(1729)
    dy=torch.randn((m,o),device=x.device,dtype=torch.bfloat16,generator=generator)*.1
    with torch.no_grad():
        expected=reference(x,gu,down,dy,activation,precision)
    axes={f"gate_up.{name}":values for name,values in GLU_SEARCH_SPACE.items()}
    # Activation arithmetic is a user numerical policy, not a timing choice.
    axes.pop("gate_up.glu_silu_math",None)
    axes.update({f"down.{name}":values for name,values in FWD_SEARCH_SPACE.items()
                 if name != "load_engine"})
    from ..prequant_autotune import PREQUANT_SEARCH_SPACE
    axes["down_backend"]=("onchip","producer_quantized")
    axes["gate_backend"]=("onchip","cooperative_prequant")
    axes["cooperative_ctas_per_sm"]=(.25,.5,1.,2.)
    axes.update({f"cooperative_gemm.{name}":values for name,values in PREQUANT_SEARCH_SPACE.items()
                 if all(set(value)=={"gemm"} for value in values)})
    axes["cooperative_gemm.glu_quant_vec"]=(1,2,4,8)
    axes["cooperative_gemm.glu_scratch_swizzle"]=("none","xor_16b")
    axes["cooperative_quant.vector"]=tuple(v["quant"] for v in PREQUANT_SEARCH_SPACE["x_vector_load"])
    axes["cooperative_quant.arithmetic"]=tuple(v["quant"] for v in PREQUANT_SEARCH_SPACE["x_arithmetic"])
    axes["cooperative_quant.warps"]=(4,8,16)
    axes.update({f"down_gemm.{name}":values for name,values in PREQUANT_SEARCH_SPACE.items()
                 if all(set(value)=={"gemm"} for value in values)})
    axes["down_quant.vector"]=tuple(v["quant"] for v in PREQUANT_SEARCH_SPACE["x_vector_load"])
    axes["down_quant.arithmetic"]=tuple(v["quant"] for v in PREQUANT_SEARCH_SPACE["x_arithmetic"])
    axes["down_quant.warps"]=(4,8,16)
    if phase=="train":
        axes["backward_recompute"]=("joint","two_pass")
        axes["backward_overlap"]=(False,True)
        axes["recompute_backend"]=("onchip","prequant")
        axes.update({f"recompute_gemm.{name}":values for name,values in PREQUANT_SEARCH_SPACE.items()
                     if all(set(value)=={"gemm"} for value in values)})
        axes["recompute_gemm.glu_quant_vec"]=(1,2,4,8)
        axes["recompute_gemm.glu_scratch_swizzle"]=("none","xor_16b")
        axes["recompute_gemm.glu_grad_load"]=("interleaved","prefetch","shared")
        axes["gate_up.glu_grad_load"]=("interleaved","prefetch","shared")
        axes["gate_up.glu_grad_store"]=("direct","exchange")
        axes["recompute_gemm.glu_grad_store"]=("direct","exchange")
        axes["gate_up.glu_hidden_decode"]=("fp32","bf16x2")
        axes["recompute_gemm.glu_hidden_decode"]=("fp32","bf16x2")
        axes["recompute_quant.vector"]=axes["down_quant.vector"]
        axes["recompute_quant.arithmetic"]=axes["down_quant.arithmetic"]
        axes["recompute_quant.warps"]=(4,8,16)
        for group in ("gate_backward","down_backward"):
            axes.update({f"{group}.{name}":values for name,values in BWD_SEARCH_SPACE.items()})
            axes[f"{group}.dw_split_count"]=(1,2,4,8,16,32)
            # A coupled proposal for an independently tunable leg, not a new
            # default. Valuable when M is ragged but dX's reduction is aligned.
            axes[f"{group}.dx_materialized_rows"]=(32,64,128)
    if precision=="bf16_gate":
        active=("cooperative_gemm.","cooperative_quant.","down_gemm.","down_quant.",
                "gate_backward.","down_backward.")
        axes={k:v for k,v in axes.items() if k.startswith(active) and
              k not in ("cooperative_gemm.glu_quant_vec","cooperative_gemm.glu_scratch_swizzle")}
        axes["up_quant_launches"]=("dual","separate")
        axes["up_weight_quant.vector"]=axes["cooperative_quant.vector"]
        axes["up_weight_quant.arithmetic"]=axes["cooperative_quant.arithmetic"]
        axes["up_weight_quant.warps"]=(4,8,16)

    # Reuse the complete quantizer launch/register/store spaces, not only the
    # old vector/arithmetic/warp subset.
    for group in ("cooperative_quant","down_quant","recompute_quant","up_weight_quant"):
        if not any(name.startswith(group+".") for name in axes):
            continue
        for name in ("launch","registers","scale_store"):
            axes[f"{group}.{name}"]=tuple(v["quant"] for v in PREQUANT_SEARCH_SPACE[f"x_{name}"])
    for group in ("cooperative_gemm","down_gemm","recompute_gemm"):
        if any(name.startswith(group+".") for name in axes):
            axes[f"{group}.layout_transport"]=PREQUANT_SEARCH_SPACE["layout_transport"]

    def update(c,axis,value):
        if axis=="up_quant_launches":
            return replace(c,gate_prequant=replace(c.gate_prequant,quant_launches=value))
        if axis in ("backward_recompute","backward_overlap","down_backend","recompute_backend","gate_backend","cooperative_ctas_per_sm"):
            return replace(c,**{axis:value})
        group,name=axis.split(".",1)
        if name=="layout_transport":
            if group=="down_gemm":
                return replace(c,down_quant=replace(c.down_quant,**value.get("quant",{})),
                               down_gemm=replace(c.down_gemm,**value["gemm"]))
            field="gate_prequant" if group=="cooperative_gemm" else "recompute"
            current=getattr(c,field)
            changed=replace(current,quant=replace(current.quant,**value.get("quant",{})),
                            weight_quant=replace(current.resolved_weight_quant(),**value.get("weight_quant",{})),
                            weight_scale_layout=value.get("weight_scale_layout",current.weight_scale_layout),
                            gemm=replace(current.gemm,**value["gemm"]))
            return replace(c,**{field:changed})
        if group=="up_weight_quant":
            weight=replace(c.gate_prequant.resolved_weight_quant(),
                           **({"num_warps":value} if name=="warps" else value))
            return replace(c,gate_prequant=replace(c.gate_prequant,weight_quant=weight))
        if name=="dx_materialized_rows":
            from ..configs.mxfp8 import MXFP8QuantConfig,MXFP8GemmConfig
            current=getattr(c,group)
            quant=MXFP8QuantConfig(quant_vec=4,load_bits=32,quant_store_bits=32,
                                   scale_layout="row_major",transposed_tile_rows=value)
            dx=replace(current.dx,backend="decomposed",quant_a=quant,quant_b=quant,
                       gemm=MXFP8GemmConfig(tile_m=64,atom_layout_m=2,stages=2,
                                           scale_layout="row_major",scale_role="consumers",
                                           epilogue="direct",store_vec=1))
            return replace(c,**{group:replace(current,dx=dx)})
        if name=="dw_split_count":
            current=getattr(c,group)
            needed=(m+value-1)//value
            tile=next((t for t in (128,256,512,1024,2048,4096) if t>=needed),8192)
            dw=replace(current.dw,
                       reduction="full_fp32" if value==1 else "split_fp32_workspace",
                       split_reduction=value,reduction_tile=0 if value==1 else tile,
                       workspace_epilogue="none" if value==1 else "tree",
                       gemm=replace(current.dw.gemm,epilogue="direct",store_vec=1,
                                    epilogue_stages=1,tiles_per_cta=1,tile_locality="raster"))
            return replace(c,**{group:replace(current,dw=dw)})
        if group=="cooperative_gemm":
            updates={name:value} if name in ("glu_quant_vec","glu_scratch_swizzle") else value["gemm"]
            return replace(c,gate_prequant=replace(c.gate_prequant,gemm=replace(c.gate_prequant.gemm,**updates)))
        if group=="cooperative_quant":
            return replace(c,gate_prequant=replace(c.gate_prequant,quant=replace(c.gate_prequant.quant,
                           **({"num_warps":value} if name=="warps" else value))))
        if group=="recompute_gemm":
            updates={name:value} if name in ("glu_quant_vec","glu_grad_load","glu_grad_store","glu_hidden_decode","glu_scratch_swizzle") else value["gemm"]
            return replace(c,recompute=replace(c.recompute,gemm=replace(c.recompute.gemm,**updates)))
        if group=="recompute_quant":
            return replace(c,recompute=replace(c.recompute,quant=replace(c.recompute.quant,
                           **({"num_warps":value} if name=="warps" else value))))
        current=getattr(c,group)
        if group=="gate_up":
            changed=normalize_glu_config(current,**{name:value})
        elif group=="down":
            changed=normalize_fwd_config(current,**{name:value})
        elif group=="down_gemm":
            changed=replace(current,**value["gemm"])
        elif group=="down_quant":
            changed=replace(current,**({"num_warps":value} if name=="warps" else value))
        else:
            changed=update_bwd_config(current,value)
        return replace(c,**{group:changed})

    def reject(c):
        if precision=="bf16_gate":
            from .._mlp_precision import prequant_config
            reason=prequant_config(c).rejection(MXFP8Problem(m,h,k)) or prequant_config(c,True).rejection(dp)
            if phase=="train":
                reason=reason or c.gate_backward.implementation_rejection(gp) or c.down_backward.implementation_rejection(dp)
            return None if reason is None else ("implementation_rejected",reason)
        reason=glu_rejection(gp,c.gate_up,"mxfp8")
        if c.gate_backend=="cooperative_prequant":
            from ..configs.glu import cooperative_glu_rejection
            reason=reason or cooperative_glu_rejection(gp,c.gate_prequant.gemm)
            reason=reason or c.gate_prequant.rejection(gp,"mxfp8")
            if c.gate_prequant.resolved_weight_quant()!=c.gate_prequant.quant or c.gate_prequant.quant_launches!="dual":
                reason=reason or "cooperative gate requires one shared X/weight quantizer policy"
            if k%32 or c.gate_prequant.gemm.scale_layout!="row_major" or not c.gate_prequant.gemm.persistent_waves or c.down_backend!="producer_quantized":
                reason=reason or "cooperative forward requires K%32=0, persistent row-major GEMM and producer-quantized down"
        if c.down_backend == "producer_quantized":
            if c.down_quant.scale_layout != "row_major" or c.down_gemm.scale_layout != "row_major" or c.down_quant.scale_rounding != "infinity":
                reason=reason or "producer quantization requires row-major infinity-rounded scales"
            reason=reason or c.down_quant.rejection(o,h) or c.down_gemm.rejection(MXFP8Problem(m,o,(h+31)//32*32))
        else:
            reason=reason or mixed_down_rejection(dp,c.down)
        if phase=="train":
            reason=reason or c.gate_backward.implementation_rejection(gp) or c.down_backward.implementation_rejection(dp)
            if c.recompute_backend=="prequant":
                reason=reason or c.recompute.rejection(gp,"mxfp8")
                if c.recompute.gemm.scale_layout!="row_major" or c.recompute.gemm.glu_epilogue!="registers" or c.recompute.gemm.glu_store!="direct":
                    reason=reason or "prequant recompute requires row-major paired-register direct epilogue"
        return None if reason is None else ("implementation_rejected",reason)

    prepared={}
    def measure(graph,errors,compile_ms,reused):
        graph.replay()
        timings=[]
        for _ in range(samples):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graph.replay();b.record();b.synchronize()
            timings.append(a.elapsed_time(b)/calls_per_sample)
        return TrialOutcome("ok",median_ms=statistics.median(timings),timings_ms=timings,
            compile_ms=compile_ms,metadata={"relative_l2":errors,"phase":phase,
                                          "prepared_benchmark_reused":reused})

    def evaluate(c):
        started=time.monotonic()
        stage="compile"
        try:
            if prepared.get("config")==c:
                stage="runtime"
                return measure(prepared["graph"],prepared["errors"],0.0,True)
            prepared.clear()
            # AccumulateGrad belongs to the leaf, not just to the temporary
            # loss. Never reuse leaves first exercised on the default stream
            # (or a previous candidate's stream) in a new capture.
            stream=torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                capture_inputs=tuple(t.detach().requires_grad_() for t in (x,gu,down))
            def forward():
                return mxfp8_mlp(*capture_inputs,activation=activation,config=c,precision=precision)
            # Both policies measure the same compiler/capture contract. Retire
            # only this candidate closure, not unrelated model compilations.
            torch._dynamo.reset_code(forward.__code__)
            forward_fn=torch.compile(forward,fullgraph=True,dynamic=False)
            def run():
                if phase=="forward":
                    with torch.no_grad():
                        return (forward_fn(),)
                y=forward_fn()
                return (y,*torch.autograd.grad(y,capture_inputs,dy))
            # Compilation, correctness, warmup, and capture must all exercise
            # these candidate-local leaves on the same stream.
            with torch.cuda.stream(stream):
                actual=run()
            stream.synchronize()
            errors=[]
            for a,b in zip(actual,expected):
                error=((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
                errors.append(error)
                if not a.isfinite().all() or error>(.003 if precision=="bf16_gate" else .025):
                    return TrialOutcome("correctness_error",error=f"relative L2 {errors}")
            compile_ms=(time.monotonic()-started)*1000
            stage="runtime"
            del actual
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    run()
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                for _ in range(calls_per_sample):
                    result=run()
            prepared.update(config=c,graph=graph,outputs=result,errors=errors,
                            inputs=capture_inputs,stream=stream)
            return measure(graph,errors,compile_ms,False)
        except Exception as exc:
            raise_if_fatal_device_context_error(exc)
            return TrialOutcome("compile_error" if stage=="compile" else "runtime_error",error=str(exc),compile_ms=(time.monotonic()-started)*1000)

    hardware=_mlp_hardware(str(x.device))
    return DiscreteKernelAdapter(
        context=KernelContext("mxfp8_recompute_mlp",33,{"m":m,"k":k,"h":h,"o":o,"activation":activation,"phase":phase,
                                                      "precision":precision,"silu_math":initial.gate_up.glu_silu_math},
                              device=hardware,
                              tags={"warmup":warmup,"samples":samples,"calls_per_sample":calls_per_sample,
                                    "capture_warmup":"candidate_leaves_same_stream_v2","compile":"fullgraph_static",
                                    "gradient_probe":"normal_seed1729"}),
        initial_config=initial,axes=axes,config_id_fn=lambda c:stable_id(active_schedule(c,precision,phase)),serialize_fn=asdict,
        deserialize_fn=lambda v:_parse(json.dumps({"revision":MLP_CONFIG_REVISION,**v},sort_keys=True)),update_fn=update,
        rejection_fn=reject,evaluator=evaluate,
        extra_features_fn=lambda c:mlp_geometry_features(c,m,k,h,o,hardware,precision,phase))


def mlp_structural_seeds(adapter,precision,phase):
    """Cheap complete schedules shared by offline and first-hit portfolios."""
    seeds=[]
    base=adapter.initial_config
    if precision=="mxfp8":
        producer=replace(base,down_backend="producer_quantized")
        seeds.extend([producer,replace(base,recompute_backend="prequant"),
                      replace(producer,gate_backend="cooperative_prequant")])
    if phase=="train":
        for group in ("gate_backward","down_backward"):
            for leg in ("dx","dw"):
                current=getattr(base,group)
                for backend in ("fused","decomposed"):
                    seeds.append(replace(base,**{group:replace(current,quant_schedule="per_matmul",**{
                        leg:replace(getattr(current,leg),backend=backend)})}))
            row=adapter.update_fn(base,f"{group}.dx_materialized_rows",64)
            seeds.append(replace(row,**{group:replace(getattr(row,group),quant_schedule="per_matmul")}))
            seeds.append(adapter.update_fn(base,f"{group}.dw_split_count",2))
    return seeds


def tune_mxfp8_mlp(x,gu,down,*,activation="silu",phase="train",initial=DEFAULT_MLP_CONFIG,
                   precision="mxfp8",max_trials=32,time_budget_s=120,output_dir="autotune_results/mlp",seed=0,progress=print,
                   coverage_fraction=2/3,warmup=3,samples=7,calls_per_sample=3,confirmation_repeats=2,
                   search="discovery",ranker_path=None,portfolio=(),tuning_policy=None):
    """Tune total forward or forward+recomputed-backward replay time; persist every trial."""
    tuning_started=time.monotonic()
    from .effort import AutotunePolicy,make_effort_autotuner
    if tuning_policy is not None:
        if not isinstance(tuning_policy,AutotunePolicy):raise TypeError('tuning_policy must be AutotunePolicy')
        tuning_policy=tuning_policy.for_pass(phase=='train')
        max_trials,time_budget_s=tuning_policy.max_trials,tuning_policy.max_seconds
        warmup,samples,calls_per_sample=tuning_policy.warmup,tuning_policy.samples,tuning_policy.calls_per_sample
    if torch.compiler.is_compiling():
        raise RuntimeError("tune before compile/capture")
    if not 0<=coverage_fraction<=1:
        raise ValueError("coverage_fraction must be between zero and one")
    if search not in ("discovery","balanced"):
        raise ValueError("search must be discovery or balanced")
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("tune before CUDA graph capture")
        adapter=make_mlp_adapter(x,gu,down,activation=activation,phase=phase,initial=initial,precision=precision,
                                 warmup=warmup,samples=samples,calls_per_sample=calls_per_sample)
        store=ResidualTuningStore(Path(output_dir)/adapter.context.identifier,transfer_root=output_dir)
        seeds=mlp_structural_seeds(adapter,precision,phase)
        if tuning_policy is not None:
            return make_effort_autotuner(adapter,store,tuning_policy,progress=progress,candidates=[*portfolio,*seeds],started_at=tuning_started).tune()
        if search=="balanced":
            from .first_hit import FirstHitPolicy,FrozenRanker,tune_first_hit
            ranker=None if ranker_path is None else FrozenRanker.load(ranker_path)
            candidates=[adapter.initial_config,*portfolio,*seeds]
            return tune_first_hit(adapter,store,candidates,
                policy=FirstHitPolicy(max_trials=min(max_trials,8),seconds=min(time_budget_s,30),
                                      confirmation_repeats=confirmation_repeats),
                ranker=ranker,progress=progress)
        policy=HybridTuningPolicy(orchestration="bandit",max_trials=max_trials,time_budget_s=time_budget_s,
                                 seed=seed,confirmation_repeats=confirmation_repeats,confirm_initial=True,use_pretrained=False,use_pairwise=False)
        from .coverage import coverage_plan
        planning_started=time.monotonic()
        proposals,audit=(coverage_plan(adapter,seeds) if coverage_fraction and max_trials>1 else ([],[]))
        # Leave an explicit refinement budget; incomplete coverage is reported,
        # never represented as exhaustive tuning.
        coverage_budget=max(0,int((max_trials-1)*coverage_fraction))
        selected=proposals[:coverage_budget]
        path=Path(output_dir)/adapter.context.identifier
        path.mkdir(parents=True,exist_ok=True)
        report={"context_id":adapter.context.identifier,"axes":list(adapter.axes),
                "planning_s":time.monotonic()-planning_started,
                "planned_candidates":len(proposals),"selected_candidates":len(selected),
                "plan":[{"config_id":adapter.config_id(p.config),"coordinate":p.coordinate}
                        for p in proposals],"axis_audit":audit}
        (path/"coverage_plan.json").write_text(json.dumps(report,indent=2,default=str))
        if progress:
            progress(f"COVERAGE {len(selected)}/{len(proposals)} candidates before refinement; "
                     f"{len(adapter.axes)} declared axes (see coverage_plan.json)")
        tuner=make_hybrid_autotuner(adapter,store,policy,progress=progress)
        tuner.coverage_proposals=tuple(selected)
        try:
            return tuner.tune()
        finally:
            records=list(store.records(adapter.context))
            outcomes={r["config_id"]:r["outcome"]["status"] for r in records}
            values={}
            def collect(node,prefix=""):
                for key,value in node.items():
                    field=f"{prefix}.{key}" if prefix else key
                    if isinstance(value,dict):
                        collect(value,field)
                    else:
                        values.setdefault(field,set()).add(json.dumps(value,sort_keys=True))
            for record in records:
                if record["outcome"]["status"]=="ok":
                    collect(active_schedule(adapter.deserialize(record["config"]),precision,phase))
            summary={"context_id":adapter.context.identifier,
                     "measured_active_values":{k:sorted(v) for k,v in values.items()},
                     "coverage":[{**p,"outcome":outcomes.get(p["config_id"],"unmeasured")}
                                 for p in report["plan"]]}
            (path/"coverage_summary.json").write_text(json.dumps(summary,indent=2))


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",default="128,128,256,128",help="M,K,H,O")
    p.add_argument("--activation",choices=("silu","gelu","relu"),default="silu")
    p.add_argument("--phase",choices=("forward","train"),default="train")
    p.add_argument("--precision",choices=("mxfp8","bf16_gate"),default="mxfp8")
    p.add_argument("--initial",help="Revisioned MLP config JSON or benchmark result")
    p.add_argument("--result-json",help="Write a revisioned winning schedule and context metadata")
    p.add_argument("--trials",type=int,default=32)
    p.add_argument("--seconds",type=float,default=120)
    p.add_argument("--search",choices=("discovery","balanced"),default="discovery")
    p.add_argument("--ranker",help="Prospectively qualified frozen first-hit ranker JSON")
    p.add_argument("--coverage-fraction",type=float,default=2/3)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--samples",type=int,default=7)
    p.add_argument("--calls-per-sample",type=int,default=3)
    p.add_argument("--confirmation-repeats",type=int,default=2)
    p.add_argument("--output-dir",default="autotune_results/mlp")
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args();m,k,h,o=map(int,a.shape.split(","));torch.manual_seed(0)
    if a.result_json and Path(a.result_json).exists():
        raise FileExistsError(a.result_json)
    x=torch.randn(m,k,device=a.device,dtype=torch.bfloat16)*.2
    gu=torch.randn(2*h,k,device=a.device,dtype=torch.bfloat16)*.02
    down=torch.randn(o,h,device=a.device,dtype=torch.bfloat16)*.02
    initial=DEFAULT_MLP_CONFIG
    if a.initial:
        from ..mlp import MXFP8MLPConfig
        values=json.loads(Path(a.initial).read_text())
        initial=MXFP8MLPConfig.from_dict(values.get("mlp_preset",values.get("config",values)))
    result=tune_mxfp8_mlp(x,gu,down,activation=a.activation,phase=a.phase,precision=a.precision,
                           initial=initial,max_trials=a.trials,
                           time_budget_s=a.seconds,output_dir=a.output_dir,progress=lambda s:print(s,flush=True),
                           coverage_fraction=a.coverage_fraction,warmup=a.warmup,samples=a.samples,
                           calls_per_sample=a.calls_per_sample,confirmation_repeats=a.confirmation_repeats,
                           search=a.search,ranker_path=a.ranker)
    print(json.dumps(asdict(result),indent=2,default=str))
    if a.result_json:
        path=Path(a.result_json);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"config":result.config.to_dict(),"precision":a.precision,
                                   "shape":[m,k,h,o],"activation":a.activation,"phase":a.phase,
                                   "device":DeviceFingerprint.current(x.device).as_dict(),
                                   "tuning_result":asdict(result)},indent=2,default=str))


if __name__=="__main__":
    main()
