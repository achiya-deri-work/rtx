"""Complete-path backward structural regression on long/narrow/ragged shapes."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics

import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tune',action='store_true')
    args=p.parse_args()
    from rtx import fp8_bwd as b
    from rtx._bwd_search import backward_structural_seeds
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG as initial
    torch.manual_seed(2201);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    root=Path(__file__).resolve().parents[1]
    environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(0)),seed=2201,
        oracle='independent FP32 GEMMs',relative_l2_limit=.08,original_path_l2_limit=.01,
        protocol=dict(warmup=2,samples=5,replays_per_sample=5,capture='complete paired backward'),
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('rtx/runtime.py','rtx/_workspace.py','rtx/execution.py','rtx/fp8_bwd.py',
             'rtx/_bwd_search.py','rtx/kernels/mxfp8_quant.py','rtx/kernels/mxfp8_bwd.py')})
    rows=[]
    for m,n,k in ((8500,256,256),(8512,256,256),(8192,64,256),(136000,256,256)):
        problem=MXFP8Problem(m,n,k)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.1
        g=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)*.1
        refs=(g.float()@w.float(),g.float().T@x.float())
        candidates=[initial,*backward_structural_seeds(problem,initial)[:2]]
        reference=None
        for config in candidates:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):b.mxfp8_linear_backward(g,x,w,config=config,autotune=False)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    out=b.mxfp8_linear_backward(g,x,w,config=config,autotune=False)
                graph.replay()
            torch.cuda.synchronize()
            if reference is None:reference=tuple(t.clone() for t in out)
            error=[];equivalence=[];metrics=[]
            for actual,ref,original in zip(out,refs,reference):
                assert torch.isfinite(actual).all()
                error.append(float((actual.float()-ref).norm()/ref.norm().clamp_min(1e-12)))
                equivalence.append(float((actual.float()-original.float()).norm()/original.float().norm().clamp_min(1e-12)))
                delta=actual.float()-ref
                metrics.append(dict(max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),
                    reference_rms=float(ref.square().mean().sqrt()),finite=bool(actual.isfinite().all())))
            assert max(error)<.08,(problem,config,error)
            assert max(equivalence)<.01,(problem,config,equivalence)
            timings=[]
            with torch.cuda.stream(stream):
                for _ in range(5):
                    a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(5):graph.replay()
                    z.record();z.synchronize();timings.append(a.elapsed_time(z)/5)
            row=dict(shape=asdict(problem),config=asdict(config),median_ms=statistics.median(timings),
                     timings_ms=timings,relative_l2_fp32=error,relative_l2_original=equivalence,metrics=metrics)
            rows.append(row);print(m,n,k,config.dw.backend,config.dw.split_reduction,row['median_ms'],flush=True)
            graph.reset()
            args.output.write_text(json.dumps(dict(environment=environment,results=rows),indent=2))
        if args.tune and m in (8500,8512):
            from rtx.bwd_autotune import tune_mxfp8_backward
            from rtx.autotune import AutotunePolicy
            policy=AutotunePolicy(max_trials=8,max_seconds=120,search_schedule='progressive',
                backward_measurement_mode='cuda_graph',confirmation_repeats=0,
                warmup=2,samples=5,calls_per_sample=3,pool_size=16)
            result=tune_mxfp8_backward(g,x,w,initial=initial,policy=policy,
                cache_dir=args.output.parent/f'tune-{m}',progress=print)
            (args.output.parent/f'winner-{m}.json').write_text(json.dumps(dict(config=asdict(result.config),
                median_ms=result.median_ms,evidence=result.tuning_evidence),indent=2))
    print('PASS',flush=True)


if __name__=='__main__':main()
