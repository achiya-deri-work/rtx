"""Experimental original-BF16 FP4 backward: prepared operands plus RTX GEMM.

Explicit signs/uniforms are fixed inputs. No public backward policy or claim
of unbiasedness. Preparation launchers contain no eager Torch arithmetic.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time

import torch
from benchmarks.compare_nvfp4_packed_gemm import decode
from benchmarks.profile_preparation_phases import measure


def prepare(source, transpose, layout, schedule, packed_source=None, stage_packed=False, *, operand=None, factor_output=None):
    if operand is not None:
        schedule=schedule.operand_schedule(operand)
    from rtx.autotune.posthoc import check_preparation_copy_alignment
    if packed_source is None:
        check_preparation_copy_alignment(source,schedule,transpose)
    from rtx.kernels.nvfp4_posthoc import (compile_hadamard_preparation,
        compile_posthoc_scales,PosthocScaleConfig)
    from rtx.fp4 import _packed_fp4_view
    r,c=source.shape
    rows,k=(c,r) if transpose else (r,c)
    groups=k//16
    if packed_source is None or stage_packed:
        producer,count=compile_hadamard_preparation(r,c,PosthocScaleConfig(schedule.producer_threads,schedule.producer_blocks),4,transpose,schedule.producer_order,schedule.producer_load,schedule.producer_signs,schedule.producer_stages,schedule.producer_async_copy,correction_decode=schedule.correction_decode,partial_max_scope=schedule.partial_max_scope,normalization_math=schedule.normalization_math,producer_copy_bits=schedule.producer_copy_bits,producer_unroll_rows=schedule.producer_unroll_rows,producer_row_tile_batch=schedule.producer_row_tile_batch,producer_row_load_bits=schedule.producer_row_load_bits,producer_values_per_lane=schedule.producer_values_per_lane,producer_transpose_values_per_lane=schedule.producer_transpose_values_per_lane,producer_scale_math=schedule.producer_scale_math,producer_grouped_store_bits=schedule.producer_grouped_store_bits)
    else:
        from rtx.kernels.nvfp4_posthoc import compile_packed_hadamard_preparation
        producer,count=compile_packed_hadamard_preparation(r,c,PosthocScaleConfig(schedule.producer_threads,schedule.producer_blocks),4,transpose,schedule.producer_order,schedule.producer_load,schedule.producer_signs,schedule.producer_stages,schedule.producer_async_copy,correction_decode=schedule.correction_decode,partial_max_scope=schedule.partial_max_scope,normalization_math=schedule.normalization_math,producer_copy_bits=schedule.producer_copy_bits,producer_unroll_rows=schedule.producer_unroll_rows,producer_row_tile_batch=schedule.producer_row_tile_batch,producer_row_load_bits=schedule.producer_row_load_bits,producer_values_per_lane=schedule.producer_values_per_lane,producer_transpose_values_per_lane=schedule.producer_transpose_values_per_lane,producer_scale_math=schedule.producer_scale_math,producer_grouped_store_bits=schedule.producer_grouped_store_bits)
    if packed_source is not None:
        source_bytes=packed_source.qdata.view(torch.uint8).flatten()
        source_scales=packed_source.scale.flatten()
        source_outer=packed_source.per_tensor_scale.reshape(1)
    if stage_packed:
        from rtx.kernels.nvfp4_posthoc import compile_packed_bf16_decode
        decode_runner=compile_packed_bf16_decode(r,c,PosthocScaleConfig(schedule.decoder_threads,schedule.decoder_blocks))
        source=torch.empty_like(source)
        check_preparation_copy_alignment(source,schedule,transpose)
    reducer=None
    if schedule.amax_reduction == "staged":
        from rtx.kernels.nvfp4_posthoc import compile_partial_maximum
        reducer=compile_partial_maximum(count)
    finalizer=compile_posthoc_scales(rows,groups,PosthocScaleConfig(schedule.finalizer_threads,schedule.finalizer_blocks,layout,probability_math=schedule.probability_math),1 if reducer else count)
    packed=torch.empty(rows,k//2,device=source.device,dtype=torch.uint8)
    temporary=torch.empty(rows*groups,device=source.device,dtype=torch.bfloat16)
    partials=torch.empty(count,device=source.device)
    maximum=torch.empty(1,device=source.device) if reducer else partials
    scales=torch.empty(rows*groups,device=source.device,dtype=torch.float8_e4m3fn)
    factor=torch.empty(1,device=source.device,dtype=torch.float32) if factor_output is None else factor_output
    if factor.shape!=(1,) or factor.dtype!=torch.float32 or factor.device!=source.device or not factor.is_contiguous():
        raise ValueError("factor output must be contiguous FP32[1] on the source device")
    uniforms=torch.rand(rows*groups,device=source.device)
    src=source.flatten() if packed_source is None or stage_packed else None
    packed_flat=packed.flatten()
    gemm_scales=scales.reshape(rows,groups) if layout=="row_major" else scales.reshape(rows//128,groups//8,1024)
    def produce(signs):
        if stage_packed:
            decode_runner(source_bytes,source_scales,source_outer,src)
        if packed_source is None or stage_packed:
            producer(src,signs,packed_flat,temporary,partials)
        else:
            producer(source_bytes,source_scales,source_outer,signs,packed_flat,temporary,partials)
    def finalize():
        if reducer:
            reducer(partials,maximum)
        finalizer(temporary,maximum,uniforms,scales,factor)
    def run(signs):
        produce(signs)
        finalize()
    def decode_operand():
        raw=gemm_scales
        if layout=="mma128":
            raw=(raw.view(torch.uint8).reshape(rows//128,groups//8,2,32,4,4)
                 .permute(0,4,3,1,2,5).contiguous().reshape(rows,groups).view(raw.dtype))
        return decode(packed,raw,factor)
    return dict(run=run,packed=_packed_fp4_view(packed),scales=gemm_scales,
                factor=factor,decode=decode_operand,uniforms=uniforms,
                produce=produce,finalize=finalize,partials=partials,
                temporary_scales=temporary)


def study_operand(rows, cols, scale, distribution):
    value = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * scale
    if distribution == "student3":
        chi = torch.randn(3, rows, cols, device="cuda").square().sum(0)
        value = (value.float() / chi.clamp_min(1e-20).sqrt()).to(torch.bfloat16)
    elif distribution == "sparse":
        value = (value.float() * (torch.rand(rows, cols, device="cuda") < .1) * (10 ** .5)).to(torch.bfloat16)
    elif distribution == "outliers":
        factor = torch.where(torch.rand(rows, cols, device="cuda") < .01, 16., 1.)
        value = (value.float() * factor).to(torch.bfloat16)
    elif distribution != "gaussian":
        raise ValueError(f"unknown distribution: {distribution}")
    return value


def run_study(shape, phase, layout, benchmark=True, schedule=None, stochastic_samples=0,
              distribution="gaussian", seed=807, decoded_samples=False, oracle_samples=False,
              source_policy="original_bf16", packed_reader=False, staged_packed_reader=False, profile=False,
              preparation_phases=False, precondition_seconds=0.):
    if not 0 <= precondition_seconds <= 10:
        raise ValueError("precondition_seconds must be between zero and ten")
    if stochastic_samples not in (0,) and stochastic_samples < 2:
        raise ValueError("stochastic_samples must be zero or at least two")
    if decoded_samples and not stochastic_samples:
        raise ValueError("decoded_samples requires stochastic_samples")
    if oracle_samples and not decoded_samples:
        raise ValueError("oracle_samples requires decoded_samples")
    if source_policy not in ("original_bf16", "forward_current_roundtrip_bf16"):
        raise ValueError("unknown backward source policy")
    if packed_reader and staged_packed_reader:
        raise ValueError("choose direct or staged packed reader, not both")
    if (packed_reader or staged_packed_reader) and source_policy != "forward_current_roundtrip_bf16":
        raise ValueError("packed reader requires explicit forward-roundtrip source policy")
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    schedule=schedule or PosthocPreparationSchedule()
    if packed_reader and schedule.reader_mode == "staged":
        packed_reader,staged_packed_reader=False,True
    elif schedule.reader_mode == "staged" and not staged_packed_reader:
        raise ValueError("staged schedule requires packed source execution")
    from rtx.fp4 import compile_nvfp4_gemm
    from rtx.configs.nvfp4 import NVFP4Problem,NVFP4GemmConfig
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scale_product
    from rtx.autotune.measurement import runtime_timing_metadata
    torch.manual_seed(seed)
    m,n,k=shape
    x=study_operand(m,k,.25,distribution)
    w=study_operand(n,k,.02,distribution)
    dy=study_operand(m,n,.25,distribution)
    original_reference = dy.float()@w.float() if phase == "dx" else dy.float().T@x.float()
    forward_source_config = None
    packed_x, packed_w = None, None
    if source_policy == "forward_current_roundtrip_bf16":
        # Numerical ablation only: this materializes decoded BF16, not a
        # memory-saving packed-autograd implementation. dy is never reused
        # from forward. Forward quantization is deterministic tensor-current.
        from rtx.fp4 import quantize_nvfp4, DEFAULT_NVFP4_QUANT_CONFIG
        forward_source_config = asdict(DEFAULT_NVFP4_QUANT_CONFIG)
        packed_x, packed_w = quantize_nvfp4(x), quantize_nvfp4(w)
        x = packed_x.dequantize(torch.bfloat16)
        w = packed_w.dequantize(torch.bfloat16)
    factors=torch.empty(2,device="cuda",dtype=torch.float32) if schedule.outer_factor_mode=="pair" else None
    factor_options={"outer_factor_mode":"pair"} if factors is not None else {}
    a_factor=None if factors is None else factors[:1]
    b_factor=None if factors is None else factors[1:]
    if phase=="dx":
        a,b=prepare(dy,False,layout,schedule,operand="a",factor_output=a_factor),prepare(w,True,layout,schedule,packed_w if packed_reader or staged_packed_reader else None,staged_packed_reader,operand="b",factor_output=b_factor)
        problem=NVFP4Problem(m,k,n)
        reference=dy.float()@w.float()
    elif phase=="dw":
        a,b=prepare(dy,True,layout,schedule,operand="a",factor_output=a_factor),prepare(x,True,layout,schedule,packed_x if packed_reader or staged_packed_reader else None,staged_packed_reader,operand="b",factor_output=b_factor)
        problem=NVFP4Problem(n,k,m)
        reference=dy.float().T@x.float()
    else:
        raise ValueError("phase must be dx or dw")
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    alpha=torch.empty(1,device="cuda") if factors is None else factors
    out=torch.empty(problem.m,problem.n,device="cuda",dtype=torch.bfloat16)
    config=schedule.gemm
    if config.scale_layout != layout:
        config=replace(config,scale_layout=layout,scale_role="tma" if layout=="mma128" else "consumers")
    reduction_tile=schedule.gemm_reduction_tile(problem.k)
    workspace=None
    gemm_reducer=None
    if schedule.gemm_split_reduction>1:
        from rtx.kernels.nvfp4_gemm import compile_nvfp4_split_gemm,compile_nvfp4_split_workspace_reduce
        workspace=torch.empty(schedule.gemm_split_reduction*problem.m*problem.n,device="cuda",dtype=torch.float32)
        gemm=compile_nvfp4_split_gemm(problem,config,split_reduction=schedule.gemm_split_reduction,
                                    reduction_tile=reduction_tile)
        gemm_reducer=compile_nvfp4_split_workspace_reduce(problem.m,problem.n,schedule.gemm_split_reduction,
            algorithm=schedule.gemm_reduction_algorithm,threads=schedule.gemm_reduction_threads,
            vector=schedule.gemm_reduction_vector,persistent_waves=schedule.gemm_reduction_waves,**factor_options)
    else:
        gemm=compile_nvfp4_gemm(problem,config,**factor_options)
    product=compile_posthoc_scale_product() if factors is None else None
    def prep():
        a["run"](signs); b["run"](signs)
        if product is not None:
            product(a["factor"],b["factor"],alpha)
    def multiply():
        if workspace is not None:
            gemm(a["packed"],b["packed"],a["scales"],b["scales"],workspace)
            gemm_reducer(workspace,out,alpha)
        else:
            gemm(a["packed"],b["packed"],a["scales"],b["scales"],out,alpha)
    def full():
        prep(); multiply()
    full()
    packed_reference=a["decode"]()@b["decode"]().T
    metrics={}
    source_reference = reference
    reference = original_reference
    for name,ref in (("packed",packed_reference),("original_bf16",reference),
                     ("backward_source", source_reference)):
        error=out.float()-ref
        metrics[name]=dict(nrms=(error.norm()/ref.norm().clamp_min(1e-12)).item(),
                           max_abs=error.abs().max().item())
    if not out.isfinite().all() or metrics["packed"]["nrms"]>.005:
        raise RuntimeError(f"packed GEMM contract failed: {metrics}")
    result=dict(shape=list(shape),phase=phase,layout=layout,metrics=metrics,gemm_config=asdict(config),
                gemm_split_reduction=schedule.gemm_split_reduction,gemm_reduction_tile=reduction_tile,
                gemm_workspace_bytes=workspace.numel()*workspace.element_size() if workspace is not None else 0,
                preparation_schedule=asdict(schedule),
                policy=source_policy+"_shared_rademacher_hadamard128_e8m4_rtz_eden_scale_sr",
                source_policy=source_policy,
                packed_reader=packed_reader,
                staged_packed_reader=staged_packed_reader,
                forward_source_config=forward_source_config,
                source_reference_nrms=((source_reference-reference).norm()/reference.norm().clamp_min(1e-12)).item(),
                seed=seed,distribution=distribution,torch=torch.__version__,cuda=torch.version.cuda,
                device=str(torch.cuda.get_device_properties(0)))
    if stochastic_samples:
        # Research-only eager statistics, deliberately outside measured launchers.
        # Independent operand uniforms; shared rotation preserves the dot product.
        result["stochastic"] = {}
        ref = reference.double()
        denominator = ref.square().sum().clamp_min(1e-24)
        original_signs = signs.clone()
        if oracle_samples:
            from benchmarks.reference_ms_eden import prepare_extended, finalize_scales
            h = torch.ones(1,1,device="cuda",dtype=torch.float64)
            for _ in range(7):
                h = torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
            norm = torch.tensor(128 ** -.5,device="cuda",dtype=torch.float32).double()
            sources = (dy, w.T) if phase == "dx" else (dy.T, x.T)
            def oracle_operand(source, uniforms):
                rotated = ((source.double().reshape(-1,128)*signs.double())@h*norm).float().reshape(source.shape)
                blocks = prepare_extended(rotated, mantissa_bits=4)
                scales, factor = finalize_scales(blocks.corrected_scales, uniforms.reshape(blocks.corrected_scales.shape))
                return (blocks.values * scales.float().unsqueeze(-1) * factor).reshape(source.shape)
        for rotation in ("fixed", "resampled"):
            generator = torch.Generator(device="cuda").manual_seed(seed+1000)
            signs.copy_(original_signs)
            mean = torch.zeros_like(ref)
            decoded_mean = torch.zeros_like(ref) if decoded_samples else None
            oracle_mean = torch.zeros_like(ref) if oracle_samples else None
            worst_packed_nrms = 0.
            m2 = torch.zeros_like(ref)
            single_squared_error = torch.zeros((), device="cuda", dtype=torch.float64)
            checkpoints = []
            for index in range(1, stochastic_samples + 1):
                if rotation == "resampled":
                    signs.copy_(torch.randint(2, (128,), device="cuda", generator=generator) * 2 - 1)
                a["uniforms"].uniform_(generator=generator)
                b["uniforms"].uniform_(generator=generator)
                full()
                sample = out.double()
                if not sample.isfinite().all():
                    raise RuntimeError("nonfinite stochastic gradient")
                if decoded_samples:
                    decoded = (a["decode"]() @ b["decode"]().T).double()
                    if not decoded.isfinite().all():
                        raise RuntimeError("nonfinite decoded stochastic gradient")
                    packed_nrms = ((sample-decoded).norm()/decoded.norm().clamp_min(1e-12)).item()
                    if packed_nrms > .005:
                        raise RuntimeError(f"stochastic packed GEMM contract failed: {packed_nrms}")
                    worst_packed_nrms = max(worst_packed_nrms, packed_nrms)
                    decoded_mean.add_((decoded-decoded_mean)/index)
                    if oracle_samples:
                        oa = oracle_operand(sources[0], a["uniforms"])
                        ob = oracle_operand(sources[1], b["uniforms"])
                        oracle = (oa @ ob.T).double()
                        if not oracle.isfinite().all():
                            raise RuntimeError("nonfinite independent oracle gradient")
                        oracle_mean.add_((oracle-oracle_mean)/index)
                delta = sample - mean
                mean.add_(delta / index)
                m2.add_(delta * (sample - mean))
                single_squared_error += (sample-ref).square().sum()
                if index & (index - 1) == 0 or index == stochastic_samples:
                    error = mean - ref
                    checkpoints.append(dict(samples=index,
                        mean_nrms=(error.square().sum()/denominator).sqrt().item(),
                        mean_max_abs=error.abs().max().item(),
                        mean_rms=error.square().mean().sqrt().item(),
                        reference_rms=ref.square().mean().sqrt().item(),
                        mean_to_source_nrms=((mean-source_reference.double()).norm()/source_reference.double().norm().clamp_min(1e-12)).item(),
                        signed_gain_error=((error*ref).sum()/denominator).item(),
                        sample_nrms_rms=(single_squared_error/index/denominator).sqrt().item(),
                        mean_noise_nrms=((m2.sum()/max(index-1,1)/index/denominator).clamp_min(0).sqrt()).item()))
                    if decoded_samples:
                        checkpoints[-1].update(
                            decoded_mean_nrms=((decoded_mean-ref).square().sum()/denominator).sqrt().item(),
                            kernel_vs_decoded_mean_nrms=((mean-decoded_mean).square().sum()/denominator).sqrt().item(),
                            worst_sample_packed_nrms=worst_packed_nrms)
                    if oracle_samples:
                        checkpoints[-1].update(
                            oracle_mean_nrms=((oracle_mean-ref).square().sum()/denominator).sqrt().item(),
                            decoded_vs_oracle_mean_nrms=((decoded_mean-oracle_mean).square().sum()/denominator).sqrt().item())
            result["stochastic"][rotation] = dict(seed=seed+1000, checkpoints=checkpoints,
                interpretation="Finite-sample diagnostic, not proof of unbiasedness; conditional on fixed BF16 operands.")
        signs.copy_(original_signs)
    if benchmark:
        if precondition_seconds:
            # Same complete work, outside all timings; no power/clock changes.
            deadline=time.monotonic()+precondition_seconds
            while time.monotonic()<deadline:
                for _ in range(32):
                    full()
                torch.cuda.synchronize()
            result["precondition_seconds"]=precondition_seconds
        result["measurement_contract"]="graph8x10_samples15_warm200ms_revision2"
        result["timings"]={name:measure(fn,time_based_warmup=True) for name,fn in (("preparation",prep),("gemm",multiply),("full",full))}
        result["quality"]={name:runtime_timing_metadata(t["samples_ms"]) for name,t in result["timings"].items()}
        if preparation_phases:
            # Fixed inputs above isolate arithmetic. Charge caller-owned RNG
            # separately and in a complete gradient; graph replay advances it.
            def randomize():
                signs.random_(0,2).mul_(2).sub_(1)
                a["uniforms"].uniform_(); b["uniforms"].uniform_()
            def full_with_rng():
                randomize(); full()
            phases={"a_producer":lambda:a["produce"](signs),
                    "b_producer":lambda:b["produce"](signs),
                    "a_finalizer":a["finalize"],"b_finalizer":b["finalize"]}
            if product is not None:
                phases["scale_product"]=lambda:product(a["factor"],b["factor"],alpha)
            phases.update(rng=randomize,full_with_rng=full_with_rng)
            result["preparation_phases"]={name:measure(fn,time_based_warmup=True)
                                           for name,fn in phases.items()}
            result["preparation_phase_quality"]={name:runtime_timing_metadata(t["samples_ms"])
                for name,t in result["preparation_phases"].items()}
            randomized_reference=a["decode"]()@b["decode"]().T
            randomized_error=out.float()-randomized_reference
            randomized_nrms=(randomized_error.norm()/randomized_reference.norm().clamp_min(1e-12)).item()
            randomized_finite=bool(out.isfinite().all())
            if not randomized_finite or randomized_nrms > .005:
                raise RuntimeError("RNG-inclusive packed GEMM contract failed")
            result["rng_inclusive_packed_contract"]={"finite":randomized_finite,"nrms":randomized_nrms}
            result["preparation_phase_caveat"]=("Isolated phases change cache reuse; full and full_with_rng are authoritative. "
                "Diagnostic RNG uses caller-owned Torch state, materialized FP32 signs/uniforms, no implicit kernel RNG. "
                "Its in-place RNG sequence differs from public API RNG and does not measure public compiled backward.")
    if profile:
        # Keep allocations, compilation, oracle work and warmup out of capture.
        for _ in range(25):
            full()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("ms_eden/"+phase):
            full()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        result["profile_scope"]="one warmed complete gradient; caller-provided fixed randomness"
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",default="2048,768,1536")
    p.add_argument("--phase",choices=("dx","dw"),required=True)
    p.add_argument("--layout",choices=("row_major","mma128"),default="mma128")
    p.add_argument("--output",required=True)
    p.add_argument("--schedule-json")
    p.add_argument("--producer-order", choices=("row_major", "reduction_major"))
    p.add_argument("--producer-load", choices=("direct", "coalesced", "coalesced_wide", "coalesced_x4", "coalesced_x8"))
    p.add_argument("--producer-threads", type=int, choices=(64,128,256,512))
    p.add_argument("--amax-reduction", choices=("inline", "staged"))
    p.add_argument("--stochastic-samples", type=int, default=0)
    p.add_argument("--no-benchmark", action="store_true")
    p.add_argument("--distribution", choices=("gaussian","student3","sparse","outliers"), default="gaussian")
    p.add_argument("--seed", type=int, default=807)
    p.add_argument("--decoded-samples", action="store_true")
    p.add_argument("--oracle-samples", action="store_true")
    p.add_argument("--source-policy", choices=("original_bf16","forward_current_roundtrip_bf16"), default="original_bf16")
    p.add_argument("--packed-reader", action="store_true")
    p.add_argument("--staged-packed-reader", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--preparation-phases", action="store_true")
    p.add_argument("--precondition-seconds", type=float, default=0.)
    args=p.parse_args()
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    schedule=PosthocPreparationSchedule(**json.loads(Path(args.schedule_json).read_text())) if args.schedule_json else None
    if args.producer_order is not None:
        schedule=replace(schedule or PosthocPreparationSchedule(),producer_order=args.producer_order)
    if args.amax_reduction is not None:
        schedule=replace(schedule or PosthocPreparationSchedule(),amax_reduction=args.amax_reduction)
    if args.producer_load is not None:
        schedule=replace(schedule or PosthocPreparationSchedule(),producer_load=args.producer_load)
    if args.producer_threads is not None:
        schedule=replace(schedule or PosthocPreparationSchedule(),producer_threads=args.producer_threads)
    result=run_study(tuple(map(int,args.shape.split(","))),args.phase,args.layout,
                     schedule=schedule, benchmark=not args.no_benchmark,
                     stochastic_samples=args.stochastic_samples, distribution=args.distribution, seed=args.seed,
                     decoded_samples=args.decoded_samples, oracle_samples=args.oracle_samples,
                     source_policy=args.source_policy, packed_reader=args.packed_reader,
                     staged_packed_reader=args.staged_packed_reader,profile=args.profile,
                     preparation_phases=args.preparation_phases,precondition_seconds=args.precondition_seconds)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result))


if __name__=="__main__":
    main()
