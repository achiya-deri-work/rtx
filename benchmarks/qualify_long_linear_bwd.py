"""Long-token linear backward study using the shared calibrated UCB tuner.

Compares complete preparation+dX+dW+reduction, not packed GEMM alone. Numerical
operand policies are pinned. Results are not installed in any runtime cache.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path


def numeric_policy(c):
    return tuple((q.quant_math,q.quant_amax,q.scale_rounding)
        for leg in (c.dx,c.dw) for q in (leg.quant_a,leg.resolved_quant_b()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=int,default=40000)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--initial-mlp-result',type=Path,required=True)
    p.add_argument('--trials',type=int,default=32)
    p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from benchmarks.qualify_mxfp8_long_mlp import initial_schedule
    from rtx.autotune import CalibratedBwdEvaluator,DeviceFingerprint,make_mxfp8_bwd_adapter
    from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
    from rtx.autotune.store import ResidualTuningStore
    from rtx.autotune.runtime_coordinates import runtime_schedule_adapter
    from rtx.bwd_experiments import BwdBenchmarkHarness
    from rtx.bwd_autotune import BWD_SEARCH_SPACE,update_bwd_config,bwd_config_to_dict
    from rtx.prequant_experiments import ShapeSpec,BenchmarkProtocol
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    m,k=args.rows,args.width
    initial=initial_schedule(json.loads(args.initial_mlp_result.read_text()),(m,k,2*k,k)).down_backward
    pinned=numeric_policy(initial)
    protocol=BenchmarkProtocol(warmup_calls=3,samples=7,confirm_samples=9,race_rounds=12,
        race_min_rounds=12,target_batch_ms=8,correctness_rtol=.06,telemetry=False)
    harness=BwdBenchmarkHarness(ShapeSpec(m,k,k),protocol,seed=81114,measurement_mode='cuda_graph')
    evaluator=CalibratedBwdEvaluator(harness,samples=7,seed=81114)
    axes=dict(BWD_SEARCH_SPACE)
    # Preserve both operands' arithmetic when entering one-load/two-axis G.
    # Incompatible distinct numerical policies are rejected, never overwritten.
    moves=[]
    if len(set(pinned))==1:
        for original in axes['shared_g_row_major']:
            value=json.loads(json.dumps(original))
            for role in ('dx','dw'):
                for q in ('quant_a','quant_b'):
                    value[role][q].update(dict(zip(('quant_math','quant_amax','scale_rounding'),pinned[0])))
            moves.append(value)
    axes={**({'shared_g_row_major':tuple(moves)} if moves else {}),**{
        name:values for name,values in axes.items() if name!='shared_g_row_major'}}
    adapter=make_mxfp8_bwd_adapter(harness.problem,evaluator,initial=initial,axes=axes,
        device=DeviceFingerprint.current('cuda:0'),tags={'study':'long_narrow_v1',
            'protocol':asdict(protocol),'numeric_policy':pinned,'measurement_mode':'cuda_graph'})
    reject=adapter.rejection_fn
    adapter=replace(adapter,rejection_fn=lambda c:
        ('policy_rejected','operand numerical policy changed') if numeric_policy(c)!=pinned else reject(c))
    adapter=runtime_schedule_adapter(adapter)
    store=ResidualTuningStore(args.output/'residuals')
    policy=AutotunePolicy(max_trials=args.trials,max_seconds=args.seconds,search_schedule='progressive',
        pool_size=128,confirmation_repeats=0,backward_measurement_mode='cuda_graph')
    result=make_effort_autotuner(adapter,store,policy,progress=lambda s:print(s,flush=True)).tune()
    race=harness.race(initial,result.config,seed=81114)
    selected=result.config if race['decision']=='challenger' else initial
    phases={name:harness.measure(c,samples=7,seed=81114,components=True)
            for name,c in (('initial',initial),('selected',selected))}
    root=Path(__file__).resolve().parents[1]
    payload=dict(shape=[m,k,k],seed=81114,arguments=vars(args),torch=str(torch.__version__),
        device=str(torch.cuda.get_device_properties(0)),protocol=asdict(protocol),
        initial=bwd_config_to_dict(initial),selected=bwd_config_to_dict(selected),
        result=asdict(result),race=race,phases=phases,
        source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in sorted((root/'rtx').rglob('*.py'))},
        caveat='Complete linear backward only; not a complete forward/backward or model-step speedup')
    (args.output/'result.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
    print('COMPLETE',race['decision'],race['paired_speedup'],flush=True)


if __name__=='__main__':main()
