"""Whole-recomputed-MLP tuning using the shared persistent search portfolio."""
from dataclasses import asdict,replace
from .._managed import tuning_resources
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
from .execution_mix import MLPExecutionMix
from .glu import GLU_SEARCH_SPACE
from ..mlp import DEFAULT_MLP_CONFIG,MLP_CONFIG_REVISION,_parse,_validate,mxfp8_mlp

# Explicit candidate arguments preserve training autograd during nested runtime
# selection. Do not combine these observations with closure-input revision 34.
# Revision 39 admits policy-preserving BF16x2/bits fused backward quantization.
# Older searches rejected these candidates; do not reuse their coverage or
# runtime winner namespace as evidence for this expanded implementation set.
# Revision 49 opens legal, backward-matched prepared forwards when dW has too
# few reduction panels for the existing 4/8-way family. Arithmetic is unchanged.
# Revision 50 deduplicates unused on-chip gate schedules in prepared execution.
MLP_AUTOTUNE_REVISION = 50
MLP_INDEPENDENT_DW_OPENING_REVISION = 1
_PREPARED_DW_PARTITIONS = (4,8)
from ..configs.glu import normalize_glu_config,glu_rejection,prequant_recompute_rejection
from ..kernels.mxfp8 import FWD_SEARCH_SPACE,MXFP8Problem,normalize_fwd_config


@lru_cache(maxsize=16)
def _mlp_hardware(device):
    from .hardware import static_device_profile
    return {**DeviceFingerprint.current(device).as_dict(),**static_device_profile(device)}


def mlp_geometry_features(config,m,k,h,o,profile,precision="mxfp8",phase="train"):
    """Pre-compile geometry estimates; never claim measured occupancy."""
    from .hardware import geometry_features
    from .output_transport import row_store_features
    gate=config.gate_up if precision=="mxfp8" and config.gate_backend=="onchip" else config.gate_prequant.gemm
    down=config.down if precision=="mxfp8" and config.down_backend=="onchip" else config.down_gemm
    roles={"gate_up":(m,2*h if precision=="mxfp8" else h,k,gate,1),
           "down":(m,o,h,down,1)}
    result={}
    for name,rows,cols,reduction,bwd in (
        ("gate_dx",m,k,2*h,config.gate_backward.dx),
        ("gate_dw",2*h,k,m,config.gate_backward.dw),
        ("down_dx",m,h,o,config.down_backward.dx),
        ("down_dw",o,h,m,config.down_backward.dw)):
        if phase=="train":
            if bwd.backend=="cublas_bf16":
                result[name+".backend_cublas_bf16"]=1.0
                continue
            roles[name]=(rows,cols,reduction,bwd.fused if bwd.backend=="fused" else bwd.gemm,bwd.split_reduction)
    if phase=="train":
        recompute=config.gate_up if config.recompute_backend=="onchip" else config.recompute.gemm
        roles["recompute"]=(m,2*h if precision=="mxfp8" else h,k,recompute if precision=="mxfp8" else gate,1)
    for name,(rows,cols,reduction,schedule,split) in roles.items():
        ctas=((rows+schedule.tile_m-1)//schedule.tile_m)*((cols+schedule.tile_n-1)//schedule.tile_n)*split
        values=geometry_features(m=rows,n=cols,k=reduction,tile_m=schedule.tile_m,
            tile_n=schedule.tile_n,tile_k=schedule.tile_k,profile=profile,grid_ctas=ctas)
        values.update(row_store_features(schedule,MXFP8Problem(rows,cols,reduction)))
        result.update({name+"."+key:value for key,value in values.items()})
    return result


def active_schedule(config,precision,phase):
    """Deduplicate schedules differing only in provably inactive fields.

    Full configurations remain serialized for reproducibility and switching
    backends. This identity must change whenever execution ownership changes.
    """
    c=config.to_dict()
    if precision=="bf16_gate":
        from .._mlp_precision import prequant_config
        result={"up":asdict(prequant_config(config)),"down":asdict(prequant_config(config,True))}
    else:
        onchip_gate=(config.gate_backend=="onchip" or
                     (phase=="train" and config.recompute_backend=="onchip"))
        # Prepared forward/recompute read only this common numerical policy
        # from gate_up. Its on-chip tile/load/quantizer coordinates otherwise
        # do not execute: treating them as new programs starves active stages.
        gate=(c["gate_up"] if onchip_gate else
              {"glu_silu_math":config.gate_up.glu_silu_math})
        result={"gate_backend":c["gate_backend"],"gate_up":gate,"down_backend":c["down_backend"]}
        if config.gate_backend=="cooperative_prequant":
            result.update(gate_prequant=c["gate_prequant"],cooperative_ctas_per_sm=c["cooperative_ctas_per_sm"])
        elif config.gate_backend=="prequant":
            result.update(gate_prequant=c['gate_prequant'])
        if config.down_backend=="onchip":
            result["down"]=c["down"]
        else:
            result.update(down_quant=c["down_quant"],down_gemm=c["down_gemm"])
        if phase=="train":
            result.update(backward_recompute=c["backward_recompute"],backward_overlap=c["backward_overlap"],
                          recompute_backend=c["recompute_backend"])
            if config.recompute_backend=="prequant":
                result["recompute"]=c["recompute"]
                if config.backward_recompute=='joint' and config.joint_output_storage!='packed':
                    result['joint_output_storage']=config.joint_output_storage
                if config.gradient_quantization!='separate':
                    result['gradient_quantization']=config.gradient_quantization
                if config.hidden_quantization!='separate':
                    result['hidden_quantization']=config.hidden_quantization
    if phase=="train":
        for group in ("gate_backward","down_backward"):
            bwd=asdict(getattr(config,group).normalized())
            if bwd['operand_storage']=='separate' or (
                bwd['operand_storage']=='serial_reuse' and precision=='mxfp8'
                and group=='down_backward' and config.backward_recompute=='joint'
            ):
                # Default identity stays compatible. Joint recomputation calls
                # down dX/dW independently. Only serial_reuse_partials opts
                # these partial calls into shared paired-cache ownership.
                bwd.pop('operand_storage',None)
            for name in ("dx","dw"):
                leg=bwd[name]
                if leg["backend"] == "cublas_bf16":
                    bwd[name]={field:leg[field] for field in ("backend","a_orientation","b_orientation")}
                    continue
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


def reference(x,gu,down,dy,activation,precision="mxfp8",*,bf16_dw=(False,False)):
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
    dd=((dy.float().T@h.float()) if bf16_dw[1] else (q(dy.T)@q(h.T).T)).bfloat16()
    if activation=="silu":
        s=torch.sigmoid(g);d=s*(1+g*(1-s))
    elif activation=="gelu":
        d=.5*(1+torch.erf(g/2**.5))+g*torch.exp(-g*g/2)/(2*torch.pi)**.5
    else:
        d=(g>0).float()
    dp=torch.cat((dh*u*d,dh*a),-1).bfloat16()
    dx=(q(dp)@q(gu.T).T).bfloat16()
    dg=((dp.float().T@x.float()) if bf16_dw[0] else (q(dp.T)@q(x.T).T)).bfloat16()
    return y,dx,dg,dd


def make_mlp_adapter(x,gu,down,*,activation="silu",phase="train",initial=DEFAULT_MLP_CONFIG,
                     precision="mxfp8",warmup=3,samples=7,calls_per_sample=3,
                     measurement_mode="outer_capture",_reference_cache=None,execution_mix=None,
                     _calibrated_timing=None,_bf16_dw_final=False):
    mix=execution_mix if execution_mix is not None else (MLPExecutionMix() if phase=='train' else MLPExecutionMix(training_pairs=0,inference_forwards=1))
    if not isinstance(mix,MLPExecutionMix):
        raise TypeError('execution_mix must be MLPExecutionMix')
    mix.validate_phase(phase)
    if measurement_mode not in ('outer_capture','compiled_direct','selected_direct','private_replay'):
        raise ValueError('unknown MLP measurement_mode')
    if precision != 'mxfp8' and measurement_mode in ('selected_direct','private_replay'):
        raise ValueError('selected execution requires mxfp8 precision')
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
    hardware=_mlp_hardware(str(x.device))
    dw_opening=(precision=='mxfp8' and phase=='train' and
        mlp_independent_dw_opening(initial, m, k, h, o, hardware.get('multiprocessor_count')))
    calibrated_timing=dw_opening if _calibrated_timing is None else _calibrated_timing
    if _reference_cache:
        dy,expected=_reference_cache['values']
    else:
        generator=torch.Generator(device=x.device).manual_seed(1729)
        dy=torch.randn((m,o),device=x.device,dtype=torch.bfloat16,generator=generator)*.1
        with torch.no_grad():
            expected=reference(x,gu,down,dy,activation,precision)
        if _reference_cache is not None:
            _reference_cache['values']=(dy,expected)
    mixed_references = {} if _reference_cache is None else _reference_cache.setdefault('mixed_dw', {})
    axes={f"gate_up.{name}":values for name,values in GLU_SEARCH_SPACE.items()}
    # Activation arithmetic is a user numerical policy, not a timing choice.
    axes.pop("gate_up.glu_silu_math",None)
    axes.update({f"down.{name}":values for name,values in FWD_SEARCH_SPACE.items()
                 if name != "load_engine"})
    from ..prequant_autotune import PREQUANT_SEARCH_SPACE
    axes["down_backend"]=("onchip","producer_quantized","prequant")
    axes["gate_backend"]=("onchip","cooperative_prequant","prequant")
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
        axes['joint_output_storage']=('packed','separate')
        axes['gradient_quantization']=('separate','recompute','recompute_both')
        if precision=='mxfp8':
            axes['gradient_quantization_path']=('separate','recompute','recompute_both')
            axes['hidden_quantization']=('separate','recompute')
            axes['hidden_quantization_path']=('separate','recompute')
        axes["backward_overlap"]=(False,True)
        axes["recompute_backend"]=("onchip","prequant")
        axes.update({f"recompute_gemm.{name}":values for name,values in PREQUANT_SEARCH_SPACE.items()
                     if all(set(value)=={"gemm"} for value in values)})
        # Cross legality valleys without accepting an intermediate one-stage
        # consumer schedule. Retain the individual axes and both scale owners;
        # this is a coupled search move, not a preferred numeric/hardware policy.
        axes['recompute_gemm.pipeline'] = tuple(
            {'gemm': {'stages': stages, 'scale_role': role}}
            for role in ('consumers', 'producer') for stages in (1,2,3,4))
        axes["recompute_gemm.glu_quant_vec"]=(1,2,4,8)
        axes["recompute_gemm.glu_scratch_swizzle"]=("none","xor_16b")
        axes["recompute_gemm.glu_grad_load"]=("interleaved","prefetch","shared")
        axes["gate_up.glu_grad_load"]=("interleaved","prefetch","shared")
        axes["gate_up.glu_grad_store"]=("direct","exchange")
        axes["recompute_gemm.glu_grad_store"]=("direct","exchange")
        axes["gate_up.glu_hidden_decode"]=("fp32","bf16x2")
        axes["recompute_gemm.glu_hidden_decode"]=("fp32","fp32_packed","bf16x2")
        axes["recompute_quant.vector"]=axes["down_quant.vector"]
        axes["recompute_quant.arithmetic"]=axes["down_quant.arithmetic"]
        axes["recompute_quant.warps"]=(4,8,16)
        for group in ("gate_backward","down_backward"):
            from .._bwd_search import backward_search_space
            axes.update({f"{group}.{name}":values for name,values in
                         backward_search_space(gp if group=='gate_backward' else dp,BWD_SEARCH_SPACE).items()})
            axes[f"{group}.dw_split_count"]=(1,2,4,8,16,32,64,128)
            axes[f"{group}.dw_partition"]=('uniform','balanced')
            # A coupled proposal for an independently tunable leg, not a new
            # default. Valuable when M is ragged but dX's reduction is aligned.
            axes[f"{group}.dx_materialized_rows"]=(32,64,128)
            axes[f"{group}.dw_materialized_rows"]=(32,64,128)
            axes[f"{group}.dx_materialized_inherited_rows"]=(32,64,128)
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
    if precision=='mxfp8' and phase=='train':
        # Joint structural moves cross the legality prerequisites in one step.
        # For long-token contexts, compare both SMEM layouts inside the fixed
        # two-trial anchor neighborhood; neither is an installed default.
        transpose={'gradient_transpose_path':('none','xor_16b')}
        axes=({**transpose,**axes} if m>=32768 else {**axes,**transpose})

    def update(c,axis,value):
        if axis=='gradient_transpose_path':
            c=update(c,'gradient_quantization_path','recompute_both')
            return replace(c,recompute=replace(c.recompute,
                gemm=replace(c.recompute.gemm,glu_scratch_swizzle=value)))
        if axis=='gradient_quantization_path' and value=='recompute_both':
            from ..mlp import _gradient_quant_rejection,_hidden_quant_rejection
            ready=replace(c,gradient_quantization='recompute_both',hidden_quantization='recompute')
            if (_gradient_quant_rejection(ready,m,k,h) is None
                    and _hidden_quant_rejection(ready,m,k,h,o) is None):
                return ready
            c=update(c,'hidden_quantization_path','recompute')
            gate=materialized_dw_config(c.gate_backward)
            gate=replace(gate,operand_storage='separate',
                dw=replace(gate.dw,quant_launches='separate'))
            return replace(c,gradient_quantization='recompute_both',gate_backward=gate)
        if axis=='hidden_quantization_path':
            if value=='separate':return replace(c,hidden_quantization='separate')
            c=update(c,'gradient_quantization_path','recompute')
            down=materialized_dw_config(c.down_backward)
            down=replace(down,dw=replace(down.dw,quant_launches='separate'))
            return replace(c,hidden_quantization='recompute',down_backward=down)
        if axis=='gradient_quantization_path':
            if value=='separate':
                return replace(c,gradient_quantization='separate')
            # Couple structural prerequisites without changing quantizer math
            # or scale rounding. The normal legality gate still checks those.
            gate=materialized_dx_config(c.gate_backward)
            gate=replace(gate,execution_order='dx_first',stream_schedule='single',
                         dx=replace(gate.dx,quant_launches='separate'))
            return replace(c,gradient_quantization='recompute',gate_backward=gate,
                           backward_recompute='joint',joint_output_storage='separate',
                           backward_overlap=False,recompute_backend='prequant',
                           recompute=replace(c.recompute,gemm=replace(c.recompute.gemm,
                                                                   glu_grad_store='exchange')))
        if axis=="up_quant_launches":
            return replace(c,gate_prequant=replace(c.gate_prequant,quant_launches=value))
        if axis in ("backward_recompute","joint_output_storage","gradient_quantization","hidden_quantization","backward_overlap","down_backend","recompute_backend","gate_backend","cooperative_ctas_per_sm"):
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
        if name=="dx_materialized_inherited_rows":
            return replace(c,**{group:materialized_dx_config(getattr(c,group),value)})
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
        if name=="dw_materialized_rows":
            current=getattr(c,group)
            return replace(c,**{group:materialized_dw_config(current,value)})
        if name=='dw_partition':
            current=getattr(c,group)
            if value=='balanced':
                dw=replace(current.dw,split_partition=value,reduction_tile=0)
            else:
                from .._bwd_search import reduction_chunk
                dw=replace(current.dw,split_partition=value,
                    reduction_tile=0 if current.dw.split_reduction==1 else
                        reduction_chunk(m,current.dw.split_reduction,current.dw.gemm.tile_k))
            return replace(c,**{group:replace(current,dw=dw)})
        if name=="dw_split_count":
            current=getattr(c,group)
            from .._bwd_search import reduction_chunk
            tile=reduction_chunk(m,value,current.dw.gemm.tile_k if current.dw.backend=='decomposed'
                                 else current.dw.fused.bf16_tile_k)
            dw=replace(current.dw,
                       reduction="full_fp32" if value==1 else "split_fp32_workspace",
                       split_reduction=value,reduction_tile=0 if value==1 or current.dw.split_partition=='balanced' else tile,
                       split_partition='uniform' if value==1 else current.dw.split_partition,
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
        reason=(glu_rejection(gp,c.gate_up,"mxfp8")
                if c.gate_backend=='onchip' or (phase=='train' and c.recompute_backend=='onchip') else None)
        if c.gate_backend=='prequant':
            reason=reason or c.gate_prequant.rejection(gp,'mxfp8')
            if c.down_backend not in ('onchip','prequant'):
                reason=reason or 'prequant gate requires onchip or prequant down backend'
        if c.gate_backend=="cooperative_prequant":
            from ..configs.glu import cooperative_glu_rejection
            reason=reason or cooperative_glu_rejection(gp,c.gate_prequant.gemm)
            reason=reason or c.gate_prequant.rejection(gp,"mxfp8")
            if c.gate_prequant.resolved_weight_quant()!=c.gate_prequant.quant or c.gate_prequant.quant_launches!="dual":
                reason=reason or "cooperative gate requires one shared X/weight quantizer policy"
            if k%32 or c.gate_prequant.gemm.scale_layout!="row_major" or not c.gate_prequant.gemm.persistent_waves or c.down_backend!="producer_quantized":
                reason=reason or "cooperative forward requires K%32=0, persistent row-major GEMM and producer-quantized down"
        if c.down_backend in ("producer_quantized","prequant"):
            if c.down_quant.scale_layout != "row_major" or c.down_gemm.scale_layout != "row_major" or c.down_quant.scale_rounding != "infinity":
                reason=reason or "packed down requires row-major infinity-rounded scales"
            reason=reason or c.down_quant.rejection(o,h) or c.down_gemm.rejection(MXFP8Problem(m,o,(h+31)//32*32))
        else:
            reason=reason or mixed_down_rejection(dp,c.down)
        if phase=="train":
            from ..mlp import _gradient_quant_rejection,_hidden_quant_rejection
            reason=reason or _gradient_quant_rejection(c,m,k,h)
            reason=reason or _hidden_quant_rejection(c,m,k,h,o)
            reason=reason or c.gate_backward.implementation_rejection(gp) or c.down_backward.implementation_rejection(dp)
            if c.recompute_backend=="prequant":
                reason=reason or prequant_recompute_rejection(c.recompute.gemm)
                reason=reason or c.recompute.rejection(gp,"mxfp8")
                if c.recompute.gemm.scale_layout!="row_major" or c.recompute.gemm.glu_epilogue!="registers" or c.recompute.gemm.glu_store!="direct":
                    reason=reason or "prequant recompute requires row-major paired-register direct epilogue"
        return None if reason is None else ("implementation_rejected",reason)

    prepared={}
    # Candidates are evaluated serially. One evaluator-owned stream permits
    # stream-keyed runner scratch and allocator blocks to be reused across
    # candidates instead of accumulating one working set per proposal.
    evaluation_stream=torch.cuda.Stream(device=x.device)
    def measure(graph,errors,compile_ms,reused):
        graph_warmup=None
        with torch.cuda.stream(prepared['stream']):
            if measurement_mode == 'outer_capture':
                # Count-only warmup left short/narrow laptop workloads at
                # different boost states after compilation. Reuse the bounded
                # time-calibrated graph warmup, outside every timing sample.
                from .preparation_runtime import _warm_measurement_graph
                graph_warmup=_warm_measurement_graph(graph)
            else:
                graph()
        timings=[]
        timing_metadata={}
        if calibrated_timing:
            from .measurement import measure_cuda_batches,runtime_timing_metadata
            with torch.cuda.stream(prepared['stream']):
                captured=measurement_mode=='outer_capture'
                timings,batching=measure_cuda_batches(graph.replay if captured else graph,
                    device=x.device,samples=samples,calls_per_sample=1 if captured else calls_per_sample)
            if captured:
                # A single replay contains the requested number of steps.
                timings=[v/calls_per_sample for v in timings]
                batching['captured_steps_per_replay']=calls_per_sample
            timing_metadata={**batching,**runtime_timing_metadata(timings)}
        else:
            for _ in range(samples):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(prepared['stream']):
                    a.record()
                    if measurement_mode == 'outer_capture':
                        graph.replay()
                    else:
                        for _ in range(calls_per_sample):
                            result=graph()
                    b.record()
                b.synchronize()
                timings.append(a.elapsed_time(b)/calls_per_sample)
        return TrialOutcome("ok",median_ms=statistics.median(timings),timings_ms=timings,
            compile_ms=compile_ms,metadata={"relative_l2":errors,"phase":phase,
                                          "numerics":prepared['numerics'],
                                          "measurement_mode":measurement_mode,
                                          "execution_mix":mix.contract(),
                                          "timing_unit":"complete_workload_ms",
                                          "graph_warmup":graph_warmup,
                                          **timing_metadata,
                                          "prepared_benchmark_reused":reused})

    def evaluate(c):
        started=time.monotonic()
        stage="compile"
        try:
            from .._mlp_forward import uses_prepared_forward
            if prepared.get("config")==c:
                stage="runtime"
                return measure(prepared["graph"],prepared["errors"],0.0,True)
            prepared.clear()
            # AccumulateGrad belongs to the leaf, not just to the temporary
            # loss. Use fresh leaves even though candidates share this
            # evaluator's stream; never inherit a default-stream accumulator.
            stream=evaluation_stream
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                capture_inputs=tuple(t.detach().requires_grad_() for t in (x,gu,down))
            def forward(*values):
                return mxfp8_mlp(*values,activation=activation,config=c,precision=precision)
            if measurement_mode=='private_replay' or (measurement_mode=='selected_direct' and not uses_prepared_forward(c)):
                from .._mlp_autotune import selected_mlp
                spec=json.dumps(dict(mode='fixed',initial=c.to_dict()),sort_keys=True)
                def forward(*values):
                    return selected_mlp(*values,activation,spec,phase=='train')[0]
            # Both policies measure the same compiler/capture contract. Retire
            # only this candidate closure, not unrelated model compilations.
            torch._dynamo.reset_code(forward.__code__)
            forward_fn=torch.compile(forward,fullgraph=True,dynamic=False,
                options={'triton.cudagraphs':measurement_mode=='private_replay'})
            def run():
                return mix.execute(forward_fn,capture_inputs,
                    lambda y:torch.autograd.grad(y,capture_inputs,dy),torch.set_grad_enabled)
            # Compilation, correctness, warmup, and capture must all exercise
            # these candidate-local leaves on the same stream.
            with torch.cuda.stream(stream):
                actual=run()
            stream.synchronize()
            errors=[]
            numerics={}
            bf16_dw = tuple(getattr(c, group).dw.backend == "cublas_bf16"
                            for group in ("gate_backward", "down_backward"))
            candidate_expected = expected
            if phase == "train" and any(bf16_dw):
                if bf16_dw not in mixed_references:
                    with torch.no_grad():
                        mixed_references[bf16_dw] = reference(
                            x,gu,down,dy,activation,precision,bf16_dw=bf16_dw)
                candidate_expected = mixed_references[bf16_dw]
            for name,a,b in zip(('output','dx','dgate_up','ddown'),actual,candidate_expected):
                difference=a.detach().float()-b.float()
                error=(difference.norm()/b.float().norm().clamp_min(1e-20)).item()
                errors.append(error)
                numerics[name]=dict(relative_l2=error,
                    max_abs=float(difference.abs().max()),
                    rms_error=float(difference.square().mean().sqrt()),
                    reference_rms=float(b.float().square().mean().sqrt()),
                    finite=bool(a.isfinite().all()))
            if any(not v['finite'] or v['relative_l2']>(.003 if precision=="bf16_gate" else .025)
                   for v in numerics.values()):
                return TrialOutcome("correctness_error",error=f"relative L2 {errors}",
                                    metadata={'numerics':numerics,'relative_l2':errors})
            compile_ms=(time.monotonic()-started)*1000
            stage="runtime"
            del actual
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    run()
            stream.synchronize()
            if measurement_mode == 'outer_capture':
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    for _ in range(calls_per_sample):
                        result=run()
            else:
                graph=run
                result=None
            prepared.update(config=c,graph=graph,outputs=result,errors=errors,
                            inputs=capture_inputs,stream=stream,numerics=numerics)
            return measure(graph,errors,compile_ms,False)
        except Exception as exc:
            raise_if_fatal_device_context_error(exc)
            return TrialOutcome("compile_error" if stage=="compile" else "runtime_error",error=str(exc),compile_ms=(time.monotonic()-started)*1000)

    return DiscreteKernelAdapter(
        context=KernelContext("mxfp8_recompute_mlp",MLP_AUTOTUNE_REVISION,{"m":m,"k":k,"h":h,"o":o,"activation":activation,"phase":phase,
                                                      "precision":precision,"silu_math":initial.gate_up.glu_silu_math},
                              device=hardware,
                              tags={"warmup":warmup,"samples":samples,"calls_per_sample":calls_per_sample,
                                    "capture_warmup":"candidate_leaves_evaluator_stream_v3","compile":"fullgraph_static",
                                    "measurement_mode":measurement_mode,"execution_contract_revision":2,
                                    "graph_warmup_ms":200. if measurement_mode=='outer_capture' else None,
                                    "structural_seed_revision":17,
                                    "skip_discarded_pool_generation":True,
                                    "recompute_pipeline_search_revision":1,
                                    **({'bf16_dw_final_trial':1} if _bf16_dw_final else {}),
                                    **({'independent_dw_opening_revision':MLP_INDEPENDENT_DW_OPENING_REVISION}
                                       if dw_opening else {}),
                                    **({'execution_mix':mix.contract()} if execution_mix is not None else {}),
                                    "gradient_probe":"normal_seed1729"}),
        initial_config=initial,axes=axes,config_id_fn=lambda c:stable_id(active_schedule(c,precision,phase)),serialize_fn=asdict,
        deserialize_fn=lambda v:_parse(json.dumps({"revision":MLP_CONFIG_REVISION,**v},sort_keys=True)),update_fn=update,
        rejection_fn=reject,evaluator=evaluate,
        extra_features_fn=lambda c:mlp_geometry_features(c,m,k,h,o,hardware,precision,phase))


def _materialized_backward_leg(current, leg, rows):
    """Preserve arithmetic/tiles while coupling compatible scale transport."""
    from math import gcd
    original=getattr(current,leg)
    quant=replace(original.quant_a,scale_layout="row_major",
                  transposed_tile_rows=rows)
    quant_b=(None if original.quant_b is None else
             replace(original.quant_b,scale_layout="row_major",
                     transposed_tile_rows=rows))
    scale_vec=gcd(original.gemm.scale_load_vec,original.gemm.tile_k//32)
    gemm=replace(original.gemm,scale_layout="row_major",
                 scale_role="consumers",scale_load_vec=scale_vec)
    if gemm.tile_m==64 and gemm.stages==1:
        # A native one-stage/TMA schedule cannot switch to the quarantined
        # row-major consumer variant. Preserve its tile/stage count and use
        # the supported producer owner; both owners/stages remain tuneable.
        gemm=replace(gemm,scale_role='producer')
    if leg=="dx":
        # Token rows may not fill an M tile; TMA stores require full tiles.
        gemm=replace(gemm,epilogue="direct",store_vec=1)
    changed=replace(original,backend="decomposed",quant_a=quant,
                    quant_b=quant_b,gemm=gemm)
    return replace(current,quant_schedule="per_matmul",**{leg:changed})


def materialized_dw_config(current, rows=128):
    """Reach dW materialization for 32-aligned ragged reductions.

    A bare backend switch can inherit mma128/TMA scale transport, rejecting
    useful reductions such as M=11296 before they are ever benchmarked.
    """
    return _materialized_backward_leg(current,"dw",rows)


def materialized_dx_config(current, rows=128):
    """Materialize dX operands while retaining the current GEMM tile choice."""
    return _materialized_backward_leg(current,"dx",rows)


def _materialized_pair(current, n, k):
    from math import gcd
    return materialized_dx_config(materialized_dw_config(current, max(32, gcd(128, n, k))),
                                  max(32, gcd(128, k)))


def _small_forward_opening(adapter):
    """Pair forward backends before spending short reductions on split dW.

    Use the actual token-axis tile of the complete materialized candidate.
    This needs neither a device-specific cutoff nor the router's capacity.
    """
    w = adapter.context.workload
    if w.get('precision') != 'mxfp8' or w.get('phase') != 'train':
        return ()
    m, k, h, o = (w[name] for name in ('m', 'k', 'h', 'o'))
    base = adapter.initial_config
    if base.gate_backend == base.down_backend == 'prequant':
        # This path has no prepared-forward admission hole. Retain its
        # bounded gradient-fusion and output-transport controls.
        return ()
    roles = (('gate_backward', 2*h, k), ('down_backward', o, h))
    materialized = {role: _materialized_pair(getattr(base, role), n, width)
                    for role, n, width in roles}
    if all((m + b.dw.gemm.tile_k - 1)//b.dw.gemm.tile_k >= 4
           for b in materialized.values()):
        return ()

    # Match a known/prepared backward verbatim; for a default on-chip anchor,
    # try the existing joint recompute family with both avoidable passes gone.
    # Fall back through row-only and the actual initial policy on ragged rows.
    back = base
    if base.recompute_backend == 'onchip':
        for axis in ('hidden_quantization_path', 'gradient_quantization_path'):
            candidate = adapter.update_fn(base, axis, 'recompute')
            if adapter.rejection(candidate) is None:
                back = candidate
                break

    # Keep a distinct unsplit direct-store dW schedule. No large partition
    # sweep: final legality checks both logical transpose shapes and K panels.
    direct = replace(base, **materialized, backward_overlap=False)
    for role, _, _ in roles:
        direct = adapter.update_fn(direct, f'{role}.dw_split_count', 1)

    families = []
    for anchor in (back, direct):
        triplet = tuple(replace(anchor, gate_backend=gate, down_backend=down)
                        for gate, down in (('onchip','onchip'), ('prequant','onchip'), ('prequant','prequant')))
        if all(adapter.rejection(candidate) is None for candidate in triplet):
            families.extend(triplet)
    # An ineligible whole triplet does not prevent comparing the base with
    # gate-only when down preparation is illegal (e.g. a narrow edge tile).
    if not families:
        control = replace(base, gate_backend='onchip', down_backend='onchip')
        gate_only = replace(control, gate_backend='prequant')
        if all(adapter.rejection(candidate) is None for candidate in (control, gate_only)):
            families.extend((control, gate_only))
    return tuple(families)


def mlp_row_store_seeds(adapter,base,phase):
    """Forward-only and combined DX recipes; never mutate FP32 dW stores."""
    from .output_transport import row_store_candidates
    w=adapter.context.workload
    if base.down_backend!='prequant':return
    result=base
    roles=[('down_gemm',MXFP8Problem(w['m'],w['o'],w['h']))]
    if phase=='train':
        roles.extend((group,MXFP8Problem(w['m'],n,k)) for group,n,k in (
            ('down_backward',w['h'],w['o']),('gate_backward',w['k'],2*w['h']))
            if getattr(base,group).dx.backend=='decomposed')
    forward=None
    for role,problem in roles:
        g=base.down_gemm if role=='down_gemm' else getattr(base,role).dx.gemm
        for schedule in row_store_candidates(g,problem):
            def change(config):
                if role=='down_gemm':return replace(config,down_gemm=schedule)
                b=getattr(config,role)
                return replace(config,**{role:replace(b,dx=replace(b.dx,gemm=schedule))})
            candidate=change(result)
            if candidate!=result and adapter.rejection(candidate) is None:
                result=candidate
                if role=='down_gemm':forward=result
                break
    # Forward alone has whole-path evidence. Additional DX stores compete as
    # a separate proposal rather than being silently treated as additive wins.
    if forward is not None:yield forward
    if result!=base and result!=forward:yield result


def _dw_split_geometry(m, n, k, gemm, parts):
    ctas = ((n + gemm.tile_m - 1)//gemm.tile_m)*((k + gemm.tile_n - 1)//gemm.tile_n)*parts
    panels = (m + gemm.tile_k - 1)//gemm.tile_k
    # Doubling the count adds precisely one incumbent's worth of partials.
    fits = panels >= 8*parts and parts*n*k*4 <= 8*1024**2
    return ctas, fits


def mlp_independent_dw_opening(initial, m, k, h, o, sm_count):
    """Cold aligned, underfilled prepared-family opportunity, no CUDA work."""
    from ..fp8_bwd import _default_bwd_config

    if (not sm_count or any(d%64 for d in (m,k,h,o))
            or not initial.gate_backend==initial.down_backend==initial.recompute_backend=='onchip'):
        return False
    geometry=[]
    for group,n,width in (('gate_backward',2*h,k),('down_backward',o,h)):
        bwd=getattr(initial,group) or _default_bwd_config(MXFP8Problem(m,n,width))
        # The stronger donor here is the existing second complete setup;
        # materialization and split selection retain its dW GEMM tiles.
        geometry.append(_dw_split_geometry(m,n,width,bwd.dw.gemm,_PREPARED_DW_PARTITIONS[-1]))
    return all(fill<sm_count for fill,_ in geometry) and any(fits for _,fits in geometry)


def mlp_independent_dw_seeds(adapter, anchor):
    """One higher partition level per underfilled dW, on the same stored plan.

    Split producers write a logical FP32 partial per output and partition.
    Budget the added partials, not padded tile allocation, and require at
    least four K panels per partition on average. CTAs estimate grid fill,
    not occupancy; legality separately checks the shortest partition.
    """
    from .._bwd_search import split_leg

    w = adapter.context.workload
    sm_count = adapter.context.device.get('multiprocessor_count')
    if not sm_count or anchor.down_gemm.epilogue != 'tma':
        return ()
    moves = []
    for group, n, k in (('gate_backward', 2*w['h'], w['k']),
                        ('down_backward', w['o'], w['h'])):
        backward = getattr(anchor, group)
        dw = backward.dw
        if dw.backend != 'decomposed' or dw.reduction != 'split_fp32_workspace':
            return ()
        g = dw.gemm
        parts = dw.split_reduction
        ctas,fits = _dw_split_geometry(w['m'],n,k,g,parts)
        if ctas >= sm_count:
            # Keep the opening unchanged in the already-wide regime.
            return ()
        next_parts = 2*parts
        if not fits:
            continue
        changed = split_leg(dw, w['m'], next_parts, partition=dw.split_partition)
        # These are partition-only comparisons, not another epilogue move.
        changed = replace(dw, split_reduction=changed.split_reduction,
                          reduction_tile=changed.reduction_tile)
        c = replace(anchor, **{group:replace(backward,dw=changed)})
        if adapter.rejection(c) is None:
            moves.append((ctas, c))
    return tuple(c for _, c in sorted(moves, key=lambda x:x[0]))


def mlp_structural_seeds(adapter,precision,phase):
    """Cheap complete schedules shared by offline and first-hit portfolios."""
    seeds=[]
    opening=[]
    long_reduction=False
    base=adapter.initial_config
    if phase=='train' and hasattr(adapter,'context') and all(
            k in adapter.context.workload for k in ('m','k','h','o')):
        from .._bwd_search import backward_structural_seeds,balanced_dw_candidates
        workload=adapter.context.workload
        long_reduction=workload['m']>=32768
        # The context uses m/k/h/o; build both actual projection geometries.
        joint=base
        balanced=[]
        for group,p in (('gate_backward',MXFP8Problem(workload['m'],2*workload['h'],workload['k'])),
                        ('down_backward',MXFP8Problem(workload['m'],workload['o'],workload['h']))):
            families=backward_structural_seeds(p,getattr(base,group))
            balanced.append((group,balanced_dw_candidates(p,getattr(base,group),
                adapter.context.device.get('multiprocessor_count'))))
            if families:
                joint=replace(joint,**{group:families[0]})
            seeds.extend(replace(base,**{group:c}) for c in families)
        opening.append(replace(joint,backward_overlap=False))
        if long_reduction and all(values for _,values in balanced):
            for index in (0,1):
                candidate=base
                for group,values in balanced:
                    candidate=replace(candidate,**{group:values[min(index,len(values)-1)]})
                opening.insert(index,replace(candidate,backward_overlap=False))
    if phase=="train":
        # Try complete legal execution families before refining tile geometry.
        for group in ("gate_backward","down_backward"):
            for materialize in (materialized_dw_config,materialized_dx_config):
                seeds.append(replace(base,**{group:materialize(getattr(base,group))}))
        # An opening budget must reach combinations too: improvements in
        # independent legs do not require accepting intermediate tile moves.
        joint=replace(base,gate_backward=materialized_dw_config(base.gate_backward),
                      down_backward=materialized_dx_config(materialized_dw_config(base.down_backward)))
        seeds.extend((joint,replace(joint,gate_backward=materialized_dx_config(joint.gate_backward))))
    if precision=="mxfp8":
        prequant=replace(base,gate_backend='prequant',down_backend='onchip')
        seeds.append(prequant)
        seeds.append(replace(base,down_backend='prequant'))
        seeds.append(replace(prequant,down_backend='prequant'))
        if phase=="train":
            recomputed=replace(prequant,recompute_backend='prequant')
            seeds.append(recomputed)
            seeds.append(replace(recomputed,recompute=replace(recomputed.recompute,
                gemm=replace(recomputed.recompute.gemm,glu_grad_load='prefetch'))))
            # Complete execution families must be reachable without accepting
            # each intermediate materialization/quantization move separately.
            # These are candidates, never shape-independent kernel defaults.
            # Transposed quantizers require complete row tiles, but token M
            # is row-oriented in dX. Derive each tile from the operands that
            # are actually transposed rather than silently rejecting half-
            # aligned feature widths such as 576/320 with a fixed 128 tile.
            w=adapter.context.workload
            complete=replace(recomputed,down_backend='prequant',
                gate_backward=_materialized_pair(base.gate_backward,2*w['h'],w['k']),
                down_backward=_materialized_pair(base.down_backward,w['o'],w['h']),
                recompute=replace(recomputed.recompute,gemm=replace(recomputed.recompute.gemm,
                    tile_m=128,tile_n=64,atom_layout_m=4,atom_layout_n=2,
                    glu_grad_load='prefetch')))
            for index,parts in enumerate(_PREPARED_DW_PARTITIONS):
                candidate=complete
                for group in ('gate_backward','down_backward'):
                    candidate=adapter.update_fn(candidate,f'{group}.dw_split_count',parts)
                seeds.append(candidate)
                # Keep the bounded opening conservative; restored overlap and
                # one-stage producer pipelines remain explicit search moves.
                candidate=replace(candidate,backward_overlap=False)
                if long_reduction:
                    opening.append(candidate)
                else:
                    # Retain the established short/ragged 4/8-way complete
                    # opening before generic backward-only proposals.
                    opening.insert(index,candidate)
            # Compete preparation fusion against more dW parallelism inside
            # the opening budget, not only after individual local-search moves.
            # The coupled update retains arithmetic/scale policies and supplies
            # the required layouts. Normal legality filtering still applies.
            row_fused=adapter.update_fn(base,'gradient_quantization_path','recompute')
            prepared=adapter.update_fn(base,'hidden_quantization_path','recompute')
            both=adapter.update_fn(base,'gradient_quantization_path','recompute_both')
            if long_reduction:
                opening.insert(1,prepared)
                opening.insert(1,both)
                if base.hidden_quantization=='recompute' and base.gradient_quantization in ('recompute','recompute_both'):
                    # A baseline already fusing two preparations should spend
                    # its bounded structural prefix comparing both layouts of
                    # the third. The unchanged baseline is measured first.
                    opening=[adapter.update_fn(base,'gradient_transpose_path',swizzle)
                             for swizzle in ('none','xor_16b')]+opening
            else:
                # A fully prepared incumbent already paid for independent
                # backward quantizers. Compare its smallest pass-elimination
                # move within the two protected structural proposals, even
                # when ragged M makes hidden/both-axis emission unavailable.
                # Retain the established 4/8-family opening for on-chip bases.
                if base.gate_backend==base.down_backend==base.recompute_backend=='prequant':
                    # Already-prepared shared FFNs should compare eliminated
                    # passes before compiling near-identical split grids.
                    # Alignment/legality below retain row-only fusion for
                    # ragged M; this never changes a numerical policy.
                    opening[:0]=[both,row_fused]
                else:
                    opening.append(row_fused)
                opening.append(prepared)
                opening.append(both)
            seeds.append(row_fused)
        # Coupled narrow schedules must be reachable without accepting a
        # sequence of illegal tile/warp/buffering intermediate configurations.
        for move in ((128,"tma",2,2),(256,"scalar",1,2),(256,"tma",2,4)):
            candidate=adapter.update_fn(base,"gate_up.weight_reuse_tile",move)
            candidate=adapter.update_fn(candidate,"down.weight_reuse_tile",move)
            seeds.append(candidate)
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
    if precision=='mxfp8' and phase=='train':
        # Do not couple the two weight-gradient grids by default. Adjacent
        # split counts are existing search coordinates, not inferred winners;
        # reserve nearby one-leg proposals after the protected family opening.
        # A 4/4 incumbent can therefore reach 4/8 without first accepting 8/8.
        for group in ('gate_backward','down_backward'):
            parts=getattr(base,group).dw.split_reduction
            for count in sorted({max(1,parts//2),min(128,parts*2)}-{parts}):
                if count in (1,2,4,8,16,32,64,128):
                    opening.append(adapter.update_fn(base,f'{group}.dw_split_count',count))
    if precision=='mxfp8':
        # Cold public users start on-chip. Rebase stores on the first complete
        # legal prepared family, not only on a previously selected winner.
        prepared=base if base.down_backend=='prequant' else next((c for c in opening+seeds
            if c.gate_backend==c.down_backend=='prequant'
            and (phase!='train' or c.recompute_backend=='prequant')
            and adapter.rejection(c) is None),None)
        if prepared is not None:
            stores=list(mlp_row_store_seeds(adapter,prepared,phase))
            prefix=list(stores)
            if phase=='train' and stores:
                # Cross output transport with another existing complete dW
                # plan. Comparing only A/direct, B/direct and A/TMA can miss
                # B/TMA even when both 8 and 16 trials agree on A/TMA.
                # Donor plans come from the established structural families,
                # not a fixed split count or a per-device winner table.
                for donor in opening+seeds:
                    if not donor.gate_backend==donor.down_backend==donor.recompute_backend=='prequant':continue
                    partition=lambda c:tuple((getattr(c,g).dw.split_reduction,
                        getattr(c,g).dw.split_partition,getattr(c,g).dw.reduction_tile)
                        for g in ('gate_backward','down_backward'))
                    if partition(donor)==partition(prepared):continue
                    candidate=prepared
                    for group in ('gate_backward','down_backward'):
                        before=getattr(prepared,group);dw=getattr(donor,group).dw
                        # Preserve the anchor's quantizers and every non-dW
                        # leg while transferring partition/geometry choices.
                        if dw.backend!=before.dw.backend:break
                        dw=replace(before.dw,split_reduction=dw.split_reduction,
                            split_partition=dw.split_partition,reduction_tile=dw.reduction_tile,
                            reduction=dw.reduction)
                        candidate=replace(candidate,**{group:replace(before,dw=dw)})
                    else:
                        if candidate==prepared or adapter.rejection(candidate) is not None:continue
                        alternatives=list(mlp_row_store_seeds(adapter,candidate,phase))
                        if alternatives:
                            prefix.append(alternatives[-1])
                            if prepared==base:
                                # The direct A control is already trial one.
                                # Use its spare slot for an independent-leg
                                # partition control rather than coupling both
                                # weight-gradient grids in every proposal.
                                for group in ('down_backward','gate_backward'):
                                    one=replace(prepared,**{group:getattr(candidate,group)})
                                    if one!=prepared and adapter.rejection(one) is None:
                                        prefix.append(one);break
                            elif mlp_independent_dw_opening(base,
                                    *(adapter.context.workload[d] for d in ('m','k','h','o')),
                                    adapter.context.device.get('multiprocessor_count')):
                                independent=mlp_independent_dw_seeds(adapter,alternatives[-1])
                                if independent:
                                    # Two stored partition controls, one-leg
                                    # challengers, the two direct fallbacks.
                                    # Keep the old forward-only store in the
                                    # tail and leave trial eight to UCB.
                                    prefix=[stores[-1],alternatives[-1],*independent,
                                            prepared,candidate,*stores[:-1]]
                                    break
                            prefix.extend((prepared,candidate))
                            break
                row=adapter.update_fn(prepared,'gradient_quantization_path','recompute')
                if adapter.rejection(row) is None:prefix.append(row)
            opening[:0]=prefix
    unique=[]
    # Budgeted search uses only two structural opening proposals. Spend them
    # on complete, distinct execution schedules, not partial materialization
    # moves that cannot reveal the combined benefit. Invalid shapes are filtered.
    for candidate in [*_small_forward_opening(adapter),*opening,*seeds]:
        if candidate == base or candidate in unique:
            continue
        if hasattr(adapter,'rejection') and adapter.rejection(candidate) is not None:
            continue
        unique.append(candidate)
    return unique


def _mlp_bf16_dw_challenger(config):
    from .._bwd_search import bf16_dw_challenger

    # Both BF16 dWs need original BF16 dP and hidden, not the two packed
    # transposed intermediates emitted by the quantized-only variant.
    return replace(
        config,
        gate_backward=bf16_dw_challenger(config.gate_backward),
        down_backward=bf16_dw_challenger(config.down_backward),
        hidden_quantization="separate",
        gradient_quantization=("recompute" if config.gradient_quantization == "recompute_both"
                               else config.gradient_quantization),
    )


def make_mlp_tuner(adapter,store,policy,*,progress=None,candidates=(),started_at=None):
    """Shared engine, with complete output-transport recipes before local moves."""
    from .effort import make_effort_autotuner
    from .core import Proposal
    candidates=tuple(candidates)
    tuner=make_effort_autotuner(adapter,store,policy,progress=progress,
                              candidates=candidates,started_at=started_at)
    if policy.allow_bf16_dw and adapter.context.tags.get('bf16_dw_final_trial'):
        tuner.final_proposal = lambda c: Proposal(_mlp_bf16_dw_challenger(c), 'final_challenger')
    ranked=any(name=='frozen_ranker' or name.startswith('portable_') for name in tuner.strategies)
    if (adapter.context.workload.get('precision')=='mxfp8'
            and not ranked and not policy.initial_neighborhood_trials
            and policy.exploration_trials==1 and not policy.baseline_trials):
        coverage=list(tuner.coverage_proposals)
        if policy.max_trials>=8:
            # A cold on-chip schedule can be an order of magnitude slower.
            # Do not waste its small budget on almost-identical on-chip
            # neighbors before comparing complete packed families and their
            # direct controls. Leave one of eight trials to ordinary UCB.
            protected=[];seen={adapter.config_id(adapter.initial_config)}
            for c in (*_small_forward_opening(adapter), *candidates):
                key=adapter.config_id(c)
                if key not in seen and adapter.rejection(c) is None:
                    protected.append(Proposal(c,'structural'));seen.add(key)
                if len(protected)==(5 if tuner.final_proposal is not None else 6):break
            for proposal in coverage:
                if len(protected)>=(5 if tuner.final_proposal is not None else 6):break
                key=adapter.config_id(proposal.config)
                if key not in seen:
                    protected.append(proposal);seen.add(key)
            tuner.coverage_proposals=tuple(protected)
        else:
            tuner.coverage_proposals=tuple([p for p in coverage if p.strategy=='structural']+
                                           [p for p in coverage if p.strategy!='structural'])
        if tuner.checkpoint_key is not None:
            tuner.checkpoint_key=stable_id(dict(shared=tuner.checkpoint_key,prepared_opening_revision=6,
                coverage=[adapter.config_id(p.config) for p in tuner.coverage_proposals]))
    return tuner


@tuning_resources
def tune_mxfp8_mlp(x,gu,down,*,activation="silu",phase="train",initial=DEFAULT_MLP_CONFIG,
                   precision="mxfp8",max_trials=32,time_budget_s=120,output_dir="autotune_results/mlp",seed=0,progress=print,
                   coverage_fraction=2/3,warmup=3,samples=7,calls_per_sample=3,confirmation_repeats=2,
                   search="discovery",ranker_path=None,portfolio=(),tuning_policy=None,
                   measurement_mode="outer_capture",qualification_rounds=6,qualification_margin=.01,
                   execution_mix=None,qualification_incumbent=None):
    """Tune the selected execution surface and persist search and qualification.

    Qualification adds two finalist setups and ``2 * qualification_rounds``
    evaluations beyond the soft search budget. Set rounds=0 for collection
    only; that disables the extra deployment promotion gate.
    ``qualification_incumbent`` can retain a previously deployed winner as the
    paired performance baseline without changing ``initial`` and therefore
    the progressive search journal identity. Pass a fully resolved config.
    Keep ``portfolio`` unchanged when resuming progressive search: the prior
    winner is already in history and must not change the opening sequence.
    """
    if qualification_rounds != 0 and (type(qualification_rounds) is not int or
                                      qualification_rounds < 4 or qualification_rounds % 2):
        raise ValueError('qualification rounds must be zero or even and >=4')
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
        reference_cache={}
        bf16_dw_final = (phase == "train" and precision == "mxfp8" and tuning_policy is not None
                         and tuning_policy.allow_bf16_dw)
        adapter=make_mlp_adapter(x,gu,down,activation=activation,phase=phase,initial=initial,precision=precision,
                                 warmup=warmup,samples=samples,calls_per_sample=calls_per_sample,
                                 measurement_mode=measurement_mode,_reference_cache=reference_cache,
                                 execution_mix=execution_mix,_bf16_dw_final=bf16_dw_final)
        if tuning_policy is not None and tuning_policy.search_schedule == 'progressive':
            from .runtime_coordinates import runtime_schedule_adapter
            adapter=runtime_schedule_adapter(adapter)
        store_root=Path(output_dir)/adapter.context.identifier
        if tuning_policy is not None:
            from .effort import effort_residual_root
            store_root=effort_residual_root(store_root,tuning_policy)
        store=ResidualTuningStore(store_root,transfer_root=output_dir)
        seeds=mlp_structural_seeds(adapter,precision,phase)
        def finish(result):
            from .deployment_qualification import DeploymentQualificationError, qualify_pair
            if not qualification_rounds:
                return result
            baseline=adapter.initial_config if qualification_incumbent is None else qualification_incumbent
            if result.config == baseline:
                return result
            if progress:
                progress(f'QUALIFY mode={measurement_mode} rounds={qualification_rounds}')
            # Keep each finalist's prepared execution alive across alternating
            # rounds; no recompilation between A and B measurements.
            qualification_started=time.monotonic()
            challenger=result.config
            contenders={}
            calibrated=bool(adapter.context.tags.get('independent_dw_opening_revision'))
            for c in (baseline,result.config):
                contenders[c]=make_mlp_adapter(x,gu,down,activation=activation,phase=phase,
                    initial=c,precision=precision,warmup=warmup,samples=samples,
                    calls_per_sample=calls_per_sample,measurement_mode=measurement_mode,
                    _reference_cache=reference_cache,execution_mix=execution_mix,_calibrated_timing=calibrated)
            def evaluate(c):
                outcome=contenders[c].evaluate(c)
                if calibrated and outcome.successful and not outcome.timing_eligible:
                    return replace(outcome,status='runtime_error',error='nonstationary deployment timing')
                return outcome
            try:
                selected,decision=qualify_pair(baseline,result.config,evaluate,
                    rounds=qualification_rounds,margin=qualification_margin)
            except DeploymentQualificationError as error:
                import uuid
                path=store_root/f'deployment-qualification-failed-{uuid.uuid4().hex}.json'
                with path.open('x') as stream:
                    json.dump(dict(error.decision, incumbent=baseline.to_dict(),
                        challenger=challenger.to_dict(),
                        elapsed_s=time.monotonic()-qualification_started),stream,indent=2)
                raise
            result.config=selected
            result.median_ms=decision['median_ms']
            decision.update(incumbent=baseline.to_dict(),challenger=challenger.to_dict(),
                            selected=selected.to_dict(),elapsed_s=time.monotonic()-qualification_started)
            result.elapsed_s+=decision['elapsed_s']
            result.winner_decision={**(result.winner_decision or {}),
                                   'config_id':adapter.config_id(selected),'median_ms':result.median_ms,
                                   'reason':'deployment_improvement' if decision['promoted'] else 'deployment_incumbent_retained',
                                   'deployment_qualification':decision,
                                   'measurement_mode':measurement_mode}
            if result.tuning_evidence is not None:
                result.tuning_evidence={**result.tuning_evidence,'winner_decision':result.winner_decision}
            # Immutable reports preserve previous qualification/rollback evidence.
            import uuid
            path=store_root/f'deployment-qualification-{uuid.uuid4().hex}.json'
            with path.open('x') as stream:
                json.dump(result.winner_decision,stream,indent=2)
            return result
        if tuning_policy is not None:
            return finish(make_mlp_tuner(adapter,store,tuning_policy,progress=progress,candidates=[*portfolio,*seeds],started_at=tuning_started).tune())
        if search=="balanced":
            from .first_hit import FirstHitPolicy,FrozenRanker,tune_first_hit
            ranker=None if ranker_path is None else FrozenRanker.load(ranker_path)
            candidates=[adapter.initial_config,*portfolio,*seeds]
            return finish(tune_first_hit(adapter,store,candidates,
                policy=FirstHitPolicy(max_trials=min(max_trials,8),seconds=min(time_budget_s,30),
                                      confirmation_repeats=confirmation_repeats),
                ranker=ranker,progress=progress))
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
            return finish(tuner.tune())
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
    p.add_argument("--measurement-mode",choices=("outer_capture","compiled_direct","selected_direct","private_replay"),default="outer_capture")
    p.add_argument("--qualification-rounds",type=int,default=6,help="Additional paired deployment rounds; 0 disables")
    p.add_argument("--execution-mix",help="Counts per workload: training pairs,grad forwards,inference forwards (e.g. 1,2,3)")
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
                           search=a.search,ranker_path=a.ranker,measurement_mode=a.measurement_mode,
                           qualification_rounds=a.qualification_rounds,
                           execution_mix=None if a.execution_mix is None else MLPExecutionMix(*map(int,a.execution_mix.split(','))))
    print(json.dumps(asdict(result),indent=2,default=str))
    if a.result_json:
        path=Path(a.result_json);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"config":result.config.to_dict(),"precision":a.precision,
                                   "shape":[m,k,h,o],"activation":a.activation,"phase":a.phase,
                                   "device":DeviceFingerprint.current(x.device).as_dict(),
                                   "tuning_result":asdict(result)},indent=2,default=str))


if __name__=="__main__":
    main()
