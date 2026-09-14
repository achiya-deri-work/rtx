"""Paired graph timing of history publication; omission is an ablation, not a valid policy.

Inputs are stationary so both graphs have identical numerical outputs. The
no-publication graph is deliberately invalid for changing-data delayed state.
Never use its timing as a deployable winner or ingest it as tuning evidence.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from rtx.configs.nvfp4 import DEFAULT_NVFP4_DYNAMIC_CONFIG
from rtx.nvfp4_inference_experiments import NVFP4DelayedBenchmarkHarness
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec


def run_shape(m,n,k):
    torch.manual_seed(1729)
    harness=NVFP4DelayedBenchmarkHarness(ShapeSpec(m,n,k),'hot',
        BenchmarkProtocol(telemetry=False),device='cuda',seed=1729)
    runner=harness._build_runner(DEFAULT_NVFP4_DYNAMIC_CONFIG)
    x,w=harness.x,harness.weight
    out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    def full(): runner(x,w,out)
    def ablation():
        runner.runner(x,w,runner.x_history,runner.weight_history,
                      runner.next_x_history,runner.next_weight_history,out)
    with torch.inference_mode():
        for _ in range(5): full()
        torch.cuda.synchronize()
        expected=(x.float()@w.float().T).bfloat16()
        actual=out.clone()
        assert bool(actual.isfinite().all())
        delta=actual.float()-expected.float()
        rms=float(delta.square().mean().sqrt())
        ref_rms=float(expected.float().square().mean().sqrt())
        error=rms/max(ref_rms,1e-12)
        assert error<.20,(m,n,k,error)
        ablation()
        torch.testing.assert_close(out,actual,rtol=0,atol=0)
        graphs={}
        for name,fn in (('full',full),('no_publication_ablation',ablation)):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8): fn()
            graphs[name]=graph
        for graph in graphs.values():
            for _ in range(5): graph.replay()
        torch.cuda.synchronize()
        samples={name:[] for name in graphs}
        memory_before=torch.cuda.memory_allocated()
        for round_index in range(10):
            order=list(graphs) if round_index%2==0 else list(reversed(graphs))
            for name in order:
                start=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(25): graphs[name].replay()
                end.record(); end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/(25*8))
        torch.cuda.synchronize()
        memory_after=torch.cuda.memory_allocated()
    medians={name:statistics.median(values) for name,values in samples.items()}
    return dict(shape=[m,n,k],config=repr(DEFAULT_NVFP4_DYNAMIC_CONFIG),
        relative_rms=error,rms_error=rms,reference_rms=ref_rms,max_abs=float(delta.abs().max()),
        samples_us=samples,median_us=medians,
        min_us={name:min(values) for name,values in samples.items()},
        paired_delta_us=statistics.median([a-b for a,b in zip(samples['full'],samples['no_publication_ablation'])]),
        live_allocated_before=memory_before,live_allocated_after=memory_after)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result=dict(schema=1,command=sys.argv,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),seed=1729,time=time.time(),
        warning='Fixed default config, stationary-input ablation, not deployable-policy comparison',rows=[])
    for shape in ((128,128,256),(512,1536,1536),(32768,768,1536)):
        row=run_shape(*shape); result['rows'].append(row)
        print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as sink: json.dump(result,sink,indent=2)


if __name__=='__main__':main()
