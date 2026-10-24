"""Fixed-runner graph warmup study; no tuning or winner promotion."""
import argparse
import json
from pathlib import Path
import statistics

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-replays", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.warmup_replays < 0:
        parser.error("warmup replays must be nonnegative")
    from rtx.fp8 import _build_prequant_runner, _intern_prequant_config, DEFAULT_MXFP8_PREQUANT_CONFIG
    from rtx.autotune.measurement import runtime_timing_metadata
    torch.manual_seed(20261023)
    x = torch.randn(32768,1536,device="cuda",dtype=torch.bfloat16)*.25
    w = torch.randn(768,1536,device="cuda",dtype=torch.bfloat16)*.02
    out = torch.empty(32768,768,device="cuda",dtype=torch.bfloat16)
    runner = _build_prequant_runner(x,w,_intern_prequant_config(DEFAULT_MXFP8_PREQUANT_CONFIG))
    reference = x.float() @ w.float().T
    for _ in range(3):
        runner(x,w,out)
    torch.cuda.synchronize()
    nrms = ((out.float()-reference).square().mean()/reference.square().mean()).sqrt().item()
    if not out.isfinite().all() or nrms > .06:
        raise RuntimeError(f"FP32 gate failed: {nrms}")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            runner(x,w,out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        for _ in range(8):
            runner(x,w,out)
    for _ in range(args.warmup_replays):
        graph.replay()
    samples=[]
    for _ in range(45):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end)/80)
    result=dict(warmup_replays=args.warmup_replays,nrms=nrms,samples_ms=samples,
        windows=[dict(median_ms=statistics.median(samples[i:i+9]),
                      quality=runtime_timing_metadata(samples[i:i+9])) for i in range(0,45,9)],
        device=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,cuda=torch.version.cuda)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
