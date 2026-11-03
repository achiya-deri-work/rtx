"""Fresh, batch-contextual tuning using the library's composable autotuner."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('forward','mxfp8_backward'),required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--sequence',type=int,default=512)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--k',type=int,required=True)
    p.add_argument('--trials',type=int,default=32)
    p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    from rtx.autotune import (DeviceFingerprint,HybridTuningPolicy,JsonlTuningStore,
        make_hybrid_autotuner,CalibratedPrequantEvaluator,CalibratedBwdEvaluator,
        make_nvfp4_jit_row_region_adapter,make_mxfp8_bwd_adapter)
    from rtx.autotune.runtime_coordinates import runtime_schedule_adapter
    from rtx.prequant_experiments import BenchmarkProtocol,ShapeSpec
    from rtx.configs.nvfp4 import NVFP4Problem
    shape=ShapeSpec(args.batch*args.sequence,args.n,args.k)
    tags=dict(batch=args.batch,sequence=args.sequence,model='decoder205852672_situ',
              fresh=True,training_scope='full_model_confirm_required')
    protocol=BenchmarkProtocol(warmup_calls=8,samples=15,confirm_samples=21,
        race_rounds=21,target_batch_ms=50.,correctness_rtol=.07,correctness_atol=1.,telemetry=False)
    if args.phase=='forward':
        from rtx.nvfp4_inference_experiments import NVFP4JITRowRegionBenchmarkHarness
        from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
        problem=NVFP4Problem(shape.m,shape.n,shape.k)
        initial=replace(preferred_jit_row_region_config(problem),tensor_scale_mode='exact')
        harness=NVFP4JITRowRegionBenchmarkHarness(shape,'hot',protocol,seed=941)
        evaluator=CalibratedPrequantEvaluator(harness,samples=15,seed=941)
        adapter=make_nvfp4_jit_row_region_adapter(problem,evaluator,initial=initial,
            device=DeviceFingerprint.current(),tags=tags)
        # Fix numerical policy, search schedules only. Both training arms use
        # the same selected forward config; geometry changes require a new study.
        original_update=adapter.update_fn
        def update(c,k,v):
            return replace(original_update(c,k,v),tensor_scale_mode='exact',
                x_scale_region_rows=initial.x_scale_region_rows,
                weight_scale_region_rows=initial.weight_scale_region_rows)
        adapter=replace(adapter,update_fn=update)
    else:
        from rtx.bwd_experiments import BwdBenchmarkHarness
        from rtx.kernels.mxfp8_bwd import DEFAULT_MXFP8_BWD_CONFIG
        harness=BwdBenchmarkHarness(shape,protocol,seed=941,measurement_mode='cuda_graph')
        evaluator=CalibratedBwdEvaluator(harness,samples=15,seed=941)
        adapter=make_mxfp8_bwd_adapter(shape.problem,evaluator,initial=DEFAULT_MXFP8_BWD_CONFIG,
            device=DeviceFingerprint.current(),tags=tags)
    adapter=runtime_schedule_adapter(adapter)
    # A cold desktop changes clocks across the first sub-second collection.
    # Warm the actual workload before search; retain unchanged stability gates.
    prepared=harness.prepare(adapter.initial_config)
    warm_until=time.monotonic()+5.
    while time.monotonic()<warm_until:
        harness._time_batch(prepared,32,0)
    del prepared
    policy=HybridTuningPolicy(orchestration='bandit',max_trials=args.trials,
        time_budget_s=args.seconds,cost_model_trials=args.trials,model_warmup=8,
        model_pool_size=512,confirmation_repeats=2,confirmation_ratio=0.,
        confirm_initial=True,resume=False,transfer_history=False,
        use_pretrained=False,use_pairwise=False,seed=941)
    result=make_hybrid_autotuner(adapter,JsonlTuningStore(args.output/'residuals'),policy,
        progress=lambda message:print(message,flush=True)).tune()
    payload=dict(config=adapter.serialize(result.config),context_id=result.context_id,
        config_id=adapter.config_id(result.config),median_ms=result.median_ms,
        evaluated_trials=result.evaluated_trials,tuning_wall_s=time.monotonic()-started,
        batch=args.batch,sequence=args.sequence,shape=[shape.m,shape.n,shape.k],
        phase=args.phase,requires_full_model_confirmation=True)
    (args.output/'selection.json').write_text(json.dumps(payload,indent=2))


if __name__=='__main__':main()
