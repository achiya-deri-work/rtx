"""Policy-pinned NVFP4 forward search using the composable tuner and residuals.

Default scope fixes GEMM; --scope full searches it too. Region geometry and
numerical policy stay fixed. Preparation/GEMM/whole-forward are recorded
separately; this never promotes into the default runtime cache. Delayed uses
fixed input-amax history, not stateful changing-distribution qualification.
"""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import rtx
from rtx.autotune import DeviceFingerprint, TrialOutcome
from rtx.autotune.adapters import make_nvfp4_jit_row_region_adapter
from rtx.autotune.recipes import HybridTuningPolicy, make_hybrid_autotuner
from rtx.autotune.store import JsonlTuningStore
from rtx.autotune.outcomes import raise_if_fatal_device_context_error
from rtx.fp4 import NVFP4ForwardConfig, _make_jit_region_dynamic_runner
from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config, dynamic_config_from_dict
from benchmarks.study_nvfp4_precision import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape',default='8192,768,1536')
    p.add_argument('--selection',choices=('amax','mse5','lsq1','four_over_six'),default='lsq1')
    p.add_argument('--trials',type=int,default=32)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile-only',action='store_true')
    p.add_argument('--confirm-only',action='store_true',help='Measure an explicit finalist without searching')
    p.add_argument('--initial-config',type=Path)
    p.add_argument('--pack-reuse',choices=('search','on','off'),default='search')
    p.add_argument('--packed-store-bits',type=int,choices=(8,32))
    p.add_argument('--adaptive-decode',choices=('lut','native'))
    p.add_argument('--gemm-overrides',type=json.loads,default={},help='Explicit JSON GEMM schedule overrides')
    p.add_argument('--quant-overrides',type=json.loads,default={},help='Explicit JSON quantizer schedule overrides')
    p.add_argument('--outer-mode', choices=('exact','power2','exact_four_over_six'), default='exact')
    p.add_argument('--scope', choices=('quantizer','geometry','locality','gemm','full'), default='quantizer')
    p.add_argument('--scaling', choices=('jit_row_region','delayed','current'), default='jit_row_region')
    args=p.parse_args()
    if args.confirm_only and (not args.initial_config or args.profile_only):
        p.error('--confirm-only needs --initial-config and cannot profile')
    m,n,k=map(int,args.shape.split(','))
    problem=rtx.NVFP4Problem(m,n,k)
    rtx.validate_runtime_environment()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(17)
    device=torch.device('cuda:0')
    x=torch.randn(m,k,device=device,dtype=torch.bfloat16)
    w=torch.randn(n,k,device=device,dtype=torch.bfloat16)*.02
    ref=x.float()@w.float().T
    base=preferred_jit_row_region_config(problem)
    if args.initial_config:
        document=json.loads(args.initial_config.read_text())
        if document['shape'] != [m,n,k]: p.error('initial config shape mismatch')
        base=dynamic_config_from_dict(document['config'])
    if args.gemm_overrides:
        base=replace(base,gemm=replace(base.gemm,**args.gemm_overrides))
    if args.quant_overrides:
        base=replace(base,quant=replace(base.quant,**args.quant_overrides))
    base=replace(base,x_scale_region_rows=8,weight_scale_region_rows=8,
        tensor_scale_mode=args.outer_mode,quant=replace(base.quant,block_scale_selection=args.selection))
    scale_policy=rtx.NVFP4ScaleConfig(tensor_scale_mode=args.outer_mode)
    if args.scaling in ('delayed','current'):
        from rtx.nvfp4_inference_autotune import preferred_dynamic_config
        from rtx.fp4 import _delayed_amax_state
        base=preferred_dynamic_config(problem) if not args.initial_config else base
        base=replace(base,x_scale_region_rows=0,weight_scale_region_rows=0,
            tensor_scale_mode=args.outer_mode,quant=replace(base.quant,block_scale_selection=args.selection))
        if args.scaling == 'delayed':
            hx,hw=(_delayed_amax_state(t,scale_policy.amax_history_len) for t in (x,w))
    if args.scaling == 'current':
        from rtx.fp4 import _outer_tensor_scale
        def current_scales(a,b):
            sa=_outer_tensor_scale(a,args.outer_mode).reshape(1)
            sb=_outer_tensor_scale(b,args.outer_mode).reshape(1)
            return sa,sb,sa*sb
        scale_runner=torch.compile(current_scales,fullgraph=True,dynamic=False,
                                   options={'triton.cudagraphs':False})
    if args.pack_reuse != 'search':
        base=replace(base,quant=replace(base.quant,adaptive_reuse_packed=args.pack_reuse=='on'))
    if args.packed_store_bits is not None:
        base=replace(base,quant=replace(base.quant,packed_store_bits=args.packed_store_bits))
    if args.adaptive_decode is not None:
        base=replace(base,quant=replace(base.quant,adaptive_decode=args.adaptive_decode))
    if base.rejection(problem): base=replace(base,region_ownership='cta')
    if base.rejection(problem): raise ValueError(base.rejection(problem))
    args.output.mkdir(parents=True,exist_ok=True)
    error_limit=None

    def evaluate(cfg):
        nonlocal error_limit
        started=time.perf_counter()
        try:
            out=torch.empty(m,n,device=device,dtype=torch.bfloat16)
            forward_config=NVFP4ForwardConfig.from_materialized(cfg)
            if args.scaling == 'delayed':
                from rtx.fp4 import _make_delayed_dynamic_runner
                from rtx.fp8_bwd import _zero_tensor_async
                runner=_make_delayed_dynamic_runner(problem,forward_config,scale_policy,device)
                nx,nw=torch.empty_like(hx),torch.empty_like(hw)
                def quant():
                    _zero_tensor_async(nx); _zero_tensor_async(nw)
                    runner.quant(x,w,runner.qx,runner.qw,runner.sx,runner.sw,hx,hw,nx,nw,runner.output_scale)
                def forward():
                    runner(x,w,hx,hw,nx,nw,out)
                factors=runner.output_scale
            elif args.scaling == 'current':
                from rtx.fp4 import _make_dynamic_runner
                runner=_make_dynamic_runner(problem,forward_config,device)
                def quant():
                    nonlocal factors
                    xs,ws,factors=scale_runner(x,w)
                    if runner.quant_launches == 'dual':
                        runner.quant_x(x,w,runner.qx,runner.qw,runner.sx,runner.sw,xs,ws)
                    elif runner.quant_launches == 'separate':
                        runner.quant_x(x,runner.qx,runner.sx,xs)
                        runner.quant_w(w,runner.qw,runner.sw,ws)
                    else:
                        raise ValueError('current phase harness requires dual/separate launches')
                def forward():
                    quant()
                    runner.gemm(runner.qx_packed,runner.qw_packed,runner.sx,runner.sw,out,factors)
            else:
                runner=_make_jit_region_dynamic_runner(problem,forward_config,device)
                def quant():
                    runner.quant(x,w,runner.qx,runner.qw,runner.sx,runner.sw,runner.region_scales)
                def forward():
                    runner(x,w,out)
                factors=runner.region_scales
            forward()
            torch.cuda.synchronize()
            from benchmarks.compare_nvfp4_packed_gemm import decode, row_major_scales
            if args.scaling in ('delayed','current'):
                xs,ws=factors,1.
            else:
                x_regions=(m+cfg.x_scale_region_rows-1)//cfg.x_scale_region_rows
                xs=runner.region_scales[:x_regions].repeat_interleave(cfg.x_scale_region_rows)[:m,None]
                ws=runner.region_scales[x_regions:].repeat_interleave(cfg.weight_scale_region_rows)[:n,None]
            packed_ref=decode(runner.qx,row_major_scales(runner.sx,m,k,cfg.quant.scale_layout),xs) @ decode(
                runner.qw,row_major_scales(runner.sw,n,k,cfg.quant.scale_layout),ws).T
            packed_metrics=metrics(out,packed_ref)
            if not packed_metrics['finite'] or packed_metrics['normalized_rms'] > .005:
                return TrialOutcome('correctness_error',error='independent packed GEMM oracle failed',metadata=packed_metrics)
            correctness=metrics(out,ref)
            if error_limit is None: error_limit=correctness['normalized_rms']*1.001
            if not correctness['finite'] or correctness['normalized_rms'] > error_limit:
                return TrialOutcome('correctness_error',error='output error exceeds fixed-policy baseline',metadata=correctness)
            compile_ms=(time.perf_counter()-started)*1000
            if args.profile_only:
                for _ in range(5): forward()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                with torch.cuda.nvtx.range('nvfp4-'+args.selection):
                    for _ in range(5): forward()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                return TrialOutcome('ok',median_ms=0.,metadata=correctness)
            from benchmarks.profile_preparation_phases import measure as measure_phase
            from rtx.autotune.measurement import runtime_timing_metadata
            def gemm():
                runner.gemm(runner.qx_packed,runner.qw_packed,runner.sx,runner.sw,out,factors)
            timings={name:measure_phase(fn,time_based_warmup=True) for name,fn in
                     (('preparation',quant),('gemm',gemm),('full',forward))}
            whole=timings['full']['samples_ms']
            return TrialOutcome('ok',median_ms=statistics.median(whole),timings_ms=whole,
                compile_ms=compile_ms,metadata=dict(numerics=correctness,packed_numerics=packed_metrics,
                    timings=timings,phase_quality={name:runtime_timing_metadata(value['samples_ms'])
                    for name,value in timings.items()},**runtime_timing_metadata(whole)))
        except Exception as exc:
            raise_if_fatal_device_context_error(exc)
            return TrialOutcome('compile_error',error=f'{type(exc).__name__}: {exc}')

    if args.profile_only:
        outcome=evaluate(base)
        print(json.dumps(outcome.as_dict()),flush=True)
        if not outcome.successful: raise SystemExit(1)
        return
    if args.confirm_only:
        outcome=evaluate(base)
        document=dict(outcome=outcome.as_dict(),config=asdict(base),shape=[m,n,k],
            scaling=args.scaling,seed=17,scale_policy=asdict(scale_policy),
            measurement_contract='graph8x10_samples15_warm200ms_revision2',
            device=DeviceFingerprint.current().as_dict(),source_config=str(args.initial_config))
        with (args.output/'confirmation.json').open('x') as handle:
            json.dump(document,handle,indent=2)
        print(json.dumps(document),flush=True)
        if not outcome.timing_eligible:
            raise SystemExit('finalist failed numerical or timing gate')
        return
    axes={
        'quant_adaptive_decode':tuple({'quant':{'adaptive_decode':v}} for v in ('lut','native')),
        'quant_packed_store':tuple({'quant':{'packed_store_bits':v}} for v in (8,32)),
        'adaptive_pack_reuse':tuple({'quant':{'adaptive_reuse_packed':v}} for v in (False,True)),
        'quant_vector_load':tuple({'quant':{'values_per_lane':v,'load_bits':bits}}
            for v in (2,4,8,16) for bits in (16,32,64,128) if bits<=16*v),
        'quant_launch':tuple({'quant':{'num_warps':w}} for w in (4,8,16)),
        'quant_registers':tuple({'quant':{'maxrregcount':r}} for r in (64,96,128,192)),
        'quant_scale_compute':tuple({'quant':{'scale_compute':s}} for s in ('redundant','leader_broadcast')),
        'region_ownership':tuple({'region_ownership':o} for o in ('cta','cta_cached','warp')),
        'region_grid':tuple({'region_waves':v} for v in (2,4,8)),
    }
    if args.scope in ('geometry','locality','gemm'):
        axes = {}
    if args.scope in ('geometry', 'locality', 'gemm', 'full'):
        from rtx.nvfp4_inference_autotune import _gemm_axes, _jit_gemm_axes
        axes.update(_jit_gemm_axes if args.scaling=='jit_row_region' else _gemm_axes())
        # Runtime ignores per-candidate L2 limits; only a process-wide explicit
        # policy is effective. Do not spend trials on an inactive knob.
        axes.pop('global_l2_fetch',None)
        axes['regional_scale_epilogue']=tuple({'gemm':{'regional_scale_epilogue':v}}
                                        for v in ('direct','fragment_registers'))
        if args.scope == 'geometry':
            axes={'gemm_geometry':axes['gemm_geometry']}
        if args.scope == 'locality':
            axes={key:axes[key] for key in ('raster_group','gemm_persistence')}
    factory=make_nvfp4_jit_row_region_adapter
    if args.pack_reuse != 'search': axes.pop('adaptive_pack_reuse',None)
    if args.packed_store_bits is not None: axes.pop('quant_packed_store',None)
    if args.adaptive_decode is not None: axes.pop('quant_adaptive_decode',None)
    extra={}
    if args.scaling == 'delayed':
        from rtx.autotune.adapters import make_nvfp4_delayed_adapter
        from rtx.nvfp4_inference_autotune import NVFP4_DELAYED_SEARCH_SPACE
        factory=make_nvfp4_delayed_adapter
        axes={key:value for key,value in axes.items() if key in NVFP4_DELAYED_SEARCH_SPACE}
        extra['scale_policy']=scale_policy
    elif args.scaling == 'current':
        from rtx.autotune.adapters import make_nvfp4_dynamic_adapter
        from rtx.nvfp4_inference_autotune import NVFP4_DYNAMIC_SEARCH_SPACE
        factory=make_nvfp4_dynamic_adapter
        axes={key:value for key,value in axes.items() if key in NVFP4_DYNAMIC_SEARCH_SPACE}
    adapter=factory(problem,evaluate,initial=base,axes=axes,
        device=DeviceFingerprint.current(),tags={'study':'nvfp4-phase-v2','scope':args.scope,
            'selection':args.selection,'outer_mode':args.outer_mode,
            'scaling':args.scaling,'history_input':'fixed_at_input_amax' if args.scaling=='delayed' else 'none',
            'measurement':'graph8x10_samples15_warm200ms_revision2'},**extra)
    policy=HybridTuningPolicy(portfolio='random_local',orchestration='bandit',
        max_trials=args.trials,time_budget_s=900,model_warmup=8,
        use_pretrained=False,use_pairwise=False,transfer_history=False,
        confirmation_repeats=2,confirm_initial=True,seed=17)
    result=make_hybrid_autotuner(adapter,JsonlTuningStore(args.output/'residuals'),policy,
        progress=lambda s: print(s,flush=True)).tune()
    document=dict(config=asdict(result.config),median_ms=result.median_ms,
        trials=result.evaluated_trials,context_id=result.context_id,
        shape=[m,n,k],gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        scaling=args.scaling,scope=args.scope,seed=17,scale_policy=asdict(scale_policy),
        measurement_contract='graph8x10_samples15_warm200ms_revision2',
        note='Explicit experimental config; not installed as a runtime winner')
    (args.output/'winner.json').write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps(document),flush=True)


if __name__=='__main__': main()
