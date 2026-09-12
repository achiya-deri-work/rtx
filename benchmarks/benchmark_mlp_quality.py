"""Actual compiled mixed-precision MLP candidates; isolated worker per config."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from benchmarks.benchmark_mxfp8_mlp import measure, saved_state
from benchmarks.benchmark_torchao_fp8_rowwise import COMPILE_OPTIONS
from benchmarks.study_mlp_precision import evaluate
from benchmarks.stress_mxfp8_mlp_numerics import metrics
from rtx import mxfp8_linear, mxfp8_mlp, MXFP8MLPConfig
from rtx.kernels.mxfp8 import normalize_fwd_config


class BF16GateProjection(torch.autograd.Function):
    """Experimental BF16 gate forward, MXFP8 up and combined backward."""
    @staticmethod
    def forward(ctx,x,gu,fwd,prequant,bwd,backend):
        ctx.save_for_backward(x,gu)
        ctx.bwd=bwd
        g,u=gu.chunk(2,0)
        return (torch.nn.functional.linear(x,g),
                mxfp8_linear(x,u,autotune=False,backend=backend,
                             forward_config=fwd if backend=="fused" else None,
                             prequant_config=prequant,backward_config=bwd))

    @staticmethod
    def backward(ctx,dg,du):
        from rtx.fp8_bwd import mxfp8_linear_backward
        x,gu=ctx.saved_tensors
        dx,dw=mxfp8_linear_backward(torch.cat((dg,du),-1),x,gu,config=ctx.bwd,autotune=False)
        return dx,dw,None,None,None,None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="512,768,1536,768")
    p.add_argument("--activation", default="silu", choices=("silu","gelu","relu"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--distribution", default="init_002", choices=("init_002","uniform_half"))
    p.add_argument("--policy", default="bf16_gate", choices=("bf16","fused","bf16_gate","bf16_gate_down","mx_separate","bf16_gate_fwd"))
    p.add_argument("--tile-m", type=int, default=64)
    p.add_argument("--tile-n", type=int, default=128)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--down-tile-m", type=int)
    p.add_argument("--down-tile-n", type=int)
    p.add_argument("--down-stages", type=int)
    p.add_argument("--load-engine", default="scalar", choices=("scalar","tma"))
    p.add_argument("--quant-vec", type=int, default=8)
    p.add_argument("--materialized-layout", choices=("default","row_major","mma64","mma128"), default="default")
    p.add_argument("--persistent-waves", type=int, default=0)
    p.add_argument("--backward-mode", choices=("preset","default"), default="preset")
    p.add_argument("--compiler-mode", choices=("aggressive","aten","no_epilogue"), default="aggressive")
    p.add_argument("--bwd-stages", type=int)
    p.add_argument("--bwd-waves", type=int)
    p.add_argument("--bwd-quant-math", choices=("fp32","bf16x2"))
    p.add_argument("--bwd-stream",choices=("single","dual_stream"))
    p.add_argument("--bwd-quant-schedule",choices=("per_matmul","quad"))
    p.add_argument("--dw-splits",type=int,choices=(1,2,4,8,16,32),default=1)
    p.add_argument("--dw-split-side",choices=("both","gate","down"),default="both")
    p.add_argument("--fused-bwd-fast",action="store_true")
    p.add_argument("--decomposed-dx",action="store_true",help="Materialize dX operands while retaining the selected dW backend")
    p.add_argument("--gemm-epilogue",choices=("tma","direct"),default="tma")
    p.add_argument("--native-gate-backend",choices=("onchip","cooperative_prequant"))
    p.add_argument("--native-down-backend",choices=("onchip","producer_quantized"))
    p.add_argument("--cooperative-ctas-per-sm",type=float,choices=(.25,.5,1.,2.),default=1.)
    p.add_argument("--backend", default="fused", choices=("fused","materialized"))
    p.add_argument("--preset", default="benchmarks/configs/mxfp8_mlp_5070ti_m512_k768_h1536_o768.json")
    p.add_argument("--output", required=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--public-api", action="store_true",help="Exercise the rtx.mxfp8_mlp precision API")
    a = p.parse_args()
    if a.backward_mode=="default" and (a.bwd_stages is not None or a.bwd_waves is not None
            or a.bwd_quant_math or a.bwd_stream or a.bwd_quant_schedule
            or a.dw_splits!=1 or a.fused_bwd_fast or a.decomposed_dx):
        p.error("explicit backward overrides require --backward-mode preset")
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    m,k,h,o = map(int,a.shape.split(","))
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    x = (torch.rand(m,k,device="cuda")-.5).bfloat16().requires_grad_()
    gu = torch.empty(2*h,k,device="cuda",dtype=torch.bfloat16)
    down = torch.empty(o,h,device="cuda",dtype=torch.bfloat16)
    for t in (gu,down):
        if a.distribution=="init_002":
            torch.nn.init.trunc_normal_(t,std=.02,a=-.04,b=.04)
        else:
            torch.nn.init.uniform_(t,-.5,.5)
        t.requires_grad_()
    inputs = (x,gu,down)
    dy = (torch.rand(m,o,device="cuda")-.5).bfloat16()
    preset = (MXFP8MLPConfig() if a.preset=="default" else
              MXFP8MLPConfig.from_dict(json.loads(Path(a.preset).read_text())["config"]))
    if a.native_gate_backend is not None:
        if a.policy!="fused":
            raise ValueError("Native gate backend only applies to the fused MLP policy")
        preset=replace(preset,gate_backend=a.native_gate_backend,
                       cooperative_ctas_per_sm=a.cooperative_ctas_per_sm)
    if a.native_down_backend is not None:
        if a.policy!="fused":
            raise ValueError("Native down backend only applies to the fused MLP policy")
        preset=replace(preset,down_backend=a.native_down_backend)
    if (a.bwd_stages is not None or a.bwd_waves is not None or a.bwd_quant_math is not None
            or a.dw_splits!=1 or a.fused_bwd_fast or a.bwd_stream or a.bwd_quant_schedule or a.decomposed_dx):
        from rtx.fp8_bwd import _default_bwd_config
        from rtx.kernels.mxfp8 import MXFP8Problem
        def update_bwd(config,side):
            def update_leg(leg):
                changes = {}
                if a.bwd_stages is not None:
                    changes["stages"] = a.bwd_stages
                if a.bwd_waves is not None:
                    changes["persistent_waves"] = a.bwd_waves
                qa,qb = leg.quant_a,leg.resolved_quant_b()
                if a.bwd_quant_math is not None:
                    qa = replace(qa,quant_math=a.bwd_quant_math)
                    qb = replace(qb,quant_math=a.bwd_quant_math)
                fused=leg.fused
                if a.fused_bwd_fast and leg.backend=="fused":
                    fused=normalize_fwd_config(tile_m=64,quant_vec=4,quant_load_bits=16,
                        quant_store_bits=32,quant_math="bf16x2",quant_amax="bf16_bits")
                return replace(leg,gemm=replace(leg.gemm,**changes),quant_a=qa,quant_b=qb,fused=fused)
            dx,dw=update_leg(config.dx),update_leg(config.dw)
            if a.decomposed_dx:
                from rtx.configs.mxfp8 import MXFP8QuantConfig,MXFP8GemmConfig
                quant=MXFP8QuantConfig(quant_vec=4,load_bits=32,quant_store_bits=32,
                                       scale_layout="row_major",transposed_tile_rows=32)
                dx=replace(dx,backend="decomposed",quant_a=quant,quant_b=quant,
                           gemm=MXFP8GemmConfig(tile_m=64,atom_layout_m=2,stages=1,
                                               scale_layout="row_major",scale_role="consumers",
                                               epilogue="direct",store_vec=1))
            if a.dw_splits!=1 and a.dw_split_side in ("both",side):
                dw=replace(dw,reduction="split_fp32_workspace",split_reduction=a.dw_splits,
                    reduction_tile=((m+a.dw_splits-1)//a.dw_splits+127)//128*128,
                    workspace_epilogue="tree",gemm=replace(dw.gemm,epilogue="direct",store_vec=1))
            return replace(config,dx=dx,dw=dw,stream_schedule=a.bwd_stream or config.stream_schedule,
                           quant_schedule=a.bwd_quant_schedule or config.quant_schedule)
        preset = replace(preset,gate_backward=update_bwd(preset.gate_backward or _default_bwd_config(MXFP8Problem(m,2*h,k)),"gate"),
                         down_backward=update_bwd(preset.down_backward or _default_bwd_config(MXFP8Problem(m,o,h)),"down"))
    fwd = normalize_fwd_config(tile_m=a.tile_m,tile_n=a.tile_n,bf16_stages=a.stages,load_engine=a.load_engine,
                               bf16_tile_k=64 if a.load_engine=="tma" else 128,
                               bf16_swizzle="64b" if a.load_engine=="tma" else "128b",
                               quant_math="bf16x2",quant_amax="bf16_bits",
                               quant_vec=a.quant_vec,quant_load_bits=128,quant_store_bits=32)
    prequant = None
    if a.materialized_layout != "default":
        from rtx.fp8 import MXFP8PrequantConfig
        from rtx.configs.mxfp8 import MXFP8QuantConfig, MXFP8GemmConfig
        prequant = MXFP8PrequantConfig(
            quant=MXFP8QuantConfig(quant_vec=a.quant_vec,load_bits=128,quant_store_bits=32,
                                  scale_layout=a.materialized_layout),
            gemm=MXFP8GemmConfig(tile_m=a.tile_m,tile_n=a.tile_n,stages=a.stages,
                                atom_layout_m=2 if a.tile_m==64 else 4,
                                atom_layout_n=a.tile_n//64,
                                scale_layout="mma64x128" if a.materialized_layout=="mma64" else a.materialized_layout,
                                scale_role="consumers" if a.materialized_layout=="row_major" else "tma",
                                epilogue=a.gemm_epilogue,store_vec=1 if a.gemm_epilogue=="direct" else 4,
                                persistent_waves=a.persistent_waves),
            weight_scale_layout="mma128" if a.materialized_layout=="mma64" else None)
    down_prequant = prequant
    if any(v is not None for v in (a.down_tile_m,a.down_tile_n,a.down_stages)):
        if prequant is None or a.backend!="materialized":
            raise ValueError("Independent down tiles require an explicit materialized layout")
        dm,dn = a.down_tile_m or prequant.gemm.tile_m,a.down_tile_n or prequant.gemm.tile_n
        down_prequant=replace(prequant,gemm=replace(prequant.gemm,tile_m=dm,tile_n=dn,
            atom_layout_m=2 if dm==64 else 4,atom_layout_n=dn//64,
            stages=a.down_stages or prequant.gemm.stages))
    if a.policy not in ("bf16","fused"):
        from rtx.kernels.mxfp8 import MXFP8Problem
        problems = [MXFP8Problem(m,h,k)]
        if a.policy != "bf16_gate_down":
            problems.append(MXFP8Problem(m,o,h))
        for index,problem in enumerate(problems):
            selected_prequant=prequant if index==0 else down_prequant
            reason = (selected_prequant.rejection(problem) if selected_prequant else None) if a.backend=="materialized" else fwd.implementation_rejection(problem)
            if reason:
                raise ValueError(f"Static rejection: {reason}")

    def low(v,w,bwd,is_down=False):
        return mxfp8_linear(v,w,autotune=False,backend=a.backend,
                            forward_config=fwd if a.backend=="fused" else None,
                            prequant_config=down_prequant if is_down else prequant,
                            backward_config=bwd if a.backward_mode=="preset" else None)

    def body(x,gu,down):
        g,u = gu.chunk(2,0)
        if a.policy == "bf16":
            gate,up = torch.nn.functional.linear(x,gu).chunk(2,-1)
        elif a.policy == "bf16_gate_fwd":
            gate,up = BF16GateProjection.apply(x,gu,fwd,prequant,
                preset.gate_backward if a.backward_mode=="preset" else None,a.backend)
        else:
            gate = low(x,g,preset.gate_backward) if a.policy=="mx_separate" else torch.nn.functional.linear(x,g)
            up = low(x,u,preset.gate_backward)
        hidden = (getattr(torch.nn.functional,a.activation)(gate.float())*up.float()).bfloat16()
        return (torch.nn.functional.linear(hidden,down) if a.policy in ("bf16","bf16_gate_down")
                else low(hidden,down,preset.down_backward,True))

    def recompute(x,gu,down):
        if a.public_api:
            return mxfp8_mlp(x,gu,down,activation=a.activation,config=public_config,
                             precision={"fused":"mxfp8","bf16":"bf16","bf16_gate_fwd":"bf16_gate"}[a.policy])
        if a.policy == "fused":
            return mxfp8_mlp(x,gu,down,activation=a.activation,config=preset)
        return checkpoint(body,x,gu,down,use_reentrant=True,preserve_rng_state=False)

    public_config=preset
    if a.backward_mode=="default":
        public_config=replace(public_config,gate_backward=None,down_backward=None)
    if a.public_api and a.policy not in ("fused","bf16","bf16_gate_fwd"):
        p.error("public API supports fused, bf16, and bf16_gate_fwd policies")
    if a.public_api and a.policy=="bf16_gate_fwd":
        if prequant is None or down_prequant is None:
            p.error("public hybrid requires explicit materialized configs")
        from rtx.configs.glu import MXFP8GLUGemmConfig
        public_config=replace(public_config,gate_prequant=replace(preset.gate_prequant,quant=prequant.quant,
                                  gemm=MXFP8GLUGemmConfig(**asdict(prequant.gemm))),
                              down_quant=down_prequant.quant,down_gemm=down_prequant.gemm)

    options = dict(COMPILE_OPTIONS)
    if a.compiler_mode == "aten":
        options["max_autotune_gemm_backends"] = "ATEN"
    if a.compiler_mode == "no_epilogue":
        options["epilogue_fusion"] = False
    fn = torch.compile(recompute,fullgraph=True,dynamic=False,options=options)
    y = fn(*inputs)
    actual = (y,*torch.autograd.grad(y,inputs,dy))
    refs = evaluate((x,gu[:h],gu[h:],down),dy,a.activation,"bf16","bf16","bf16")
    bf = (refs[0],refs[1],torch.cat(refs[2:4],0),refs[4])
    if a.policy == "fused":
        from rtx.autotune.mlp import reference
        with torch.no_grad():
            expected = reference(*inputs,dy,a.activation)
    else:
        gm = "bf16_fwd" if a.policy=="bf16_gate_fwd" else ("mx" if a.policy=="mx_separate" else "bf16")
        dm = "bf16" if a.policy in ("bf16","bf16_gate_down") else "mx"
        um = "bf16" if a.policy=="bf16" else "mx"
        ref = evaluate((x,gu[:h],gu[h:],down),dy,a.activation,gm,dm,um)
        expected = (ref[0],ref[1],torch.cat(ref[2:4],0),ref[4])
    names = ("y","dx","dgu","ddown")
    errors = {n:metrics(v,r) for n,v,r in zip(names,actual,expected)}
    # The combined gate reference now matches the actual reduction contract.
    # Do not admit the observed Torch 2.12 epilogue-fusion discrepancy.
    tolerance = .003 if a.policy=="bf16_gate_fwd" else .025
    if not all(e["finite"] and e["relative_l2"]<tolerance for e in errors.values()):
        raise RuntimeError(f"Implementation gate failed: {errors}")
    report = {"args":vars(a),"compile_options":options,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
              "fwd_config":asdict(fwd),"mlp_preset":(public_config if a.public_api else preset).to_dict(),
              "prequant_config":asdict(prequant) if prequant else None,
              "implementation_error":errors,"implementation_tolerance":tolerance,
              "bf16_error":{n:metrics(v,r) for n,v,r in zip(names,actual,bf)}}
    from rtx.fp8_bwd import _default_bwd_config
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG
    gate_problem=MXFP8Problem(m,2*h if a.policy in ("fused","bf16_gate_fwd") else h,k)
    report["resolved_backward"]={
        "gate_up":asdict((preset.gate_backward if a.backward_mode=="preset" else None) or _default_bwd_config(gate_problem)),
        "down":asdict((preset.down_backward if a.backward_mode=="preset" else None) or _default_bwd_config(MXFP8Problem(m,o,h)))}
    if a.backend=="materialized":
        report["prequant_config"]=asdict(prequant or DEFAULT_MXFP8_PREQUANT_CONFIG)
        report["down_prequant_config"]=asdict(down_prequant or DEFAULT_MXFP8_PREQUANT_CONFIG)
    del y,actual,expected,refs,bf
    report["saved_state"] = saved_state(fn,inputs)
    if report["saved_state"]["non_input_saved_storage_bytes"]:
        raise RuntimeError("Recomputed candidate saved intermediate state")
    if a.profile:
        for _ in range(5):
            y=fn(*inputs); gradients=torch.autograd.grad(y,inputs,dy)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("quality/forward"):
            y=fn(*inputs)
        with torch.cuda.nvtx.range("quality/backward"):
            gradients=torch.autograd.grad(y,inputs,dy)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return
    report["forward"] = measure(fn,inputs,dy,False)
    # Independent leaves avoid retaining correctness-probe AccumulateGrad
    # nodes on the default stream when replay warms on its capture stream.
    timing_inputs = tuple(t.detach().requires_grad_() for t in inputs)
    report["train"] = measure(fn,timing_inputs,dy,True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(output),"forward_us":report["forward"]["median_us"],
                      "train_us":report["train"]["median_us"],
                      "errors":{n:v["relative_l2"] for n,v in report["bf16_error"].items()}}),flush=True)


if __name__ == "__main__":
    main()
