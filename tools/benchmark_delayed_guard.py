"""Single-stream paired submission-guard ablation; never ingest as tuning data."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch
import rtx
from rtx._delayed_state import DELAYED_HISTORY_EXECUTION


@torch.inference_mode()
def run_shape(m, n, k, compiled):
    torch.compiler.reset()
    torch.manual_seed(7231)
    layer = rtx.NVFP4Linear(k, n, device='cuda', scaling='delayed', autotune='off').eval()
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16) * .1
    fn = torch.compile(layer, fullgraph=True, dynamic=False) if compiled else layer
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def context(enabled):
        return nullcontext() if enabled else patch.object(
            DELAYED_HISTORY_EXECUTION, 'submission', new=lambda *a, **kw: nullcontext())
    with torch.cuda.stream(stream):
        for _ in range(5): y = fn(x)
        stream.synchronize()
        reference = (x.float() @ layer.weight.float().T).bfloat16()
        assert bool(y.isfinite().all())
        diff = y.float() - reference.float()
        rms = float(diff.square().mean().sqrt())
        rrms = float(reference.float().square().mean().sqrt())
        assert rms / max(rrms, 1e-12) < .20
        original = y.clone()
        with context(False): ablated = fn(x)
        torch.testing.assert_close(original, ablated, rtol=0, atol=0)
        graphs = {}
        for enabled in (True, False):
            graph = torch.cuda.CUDAGraph()
            with context(enabled), torch.cuda.graph(graph, stream=stream):
                captured = fn(x)
            for _ in range(5): graph.replay()
            stream.synchronize()
            torch.testing.assert_close(captured, original, rtol=0, atol=0)
            graphs[enabled] = (graph, captured)
        rows = []
        for replay in (False, True):
            for round_index in range(12):
                for enabled in ((True, False) if round_index % 2 == 0 else (False, True)):
                    # Patch only around a complete sample, outside its timers.
                    # Fixed stationary inputs make the two paths comparable.
                    with context(enabled):
                        call = graphs[enabled][0].replay if replay else lambda: fn(x)
                        for _ in range(3): call()
                        stream.synchronize()
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record(stream)
                        t0 = time.perf_counter_ns()
                        for _ in range(100): result = call()
                        host_us = (time.perf_counter_ns() - t0) / 100_000
                        end.record(stream)
                        end.synchronize()
                        rows.append(dict(replay=replay, guarded=enabled, round=round_index,
                            host_us=host_us, cuda_us=start.elapsed_time(end)*10))
        summary = []
        for replay in (False, True):
            for metric in ('host_us', 'cuda_us'):
                a = [r[metric] for r in rows if r['replay']==replay and r['guarded']]
                b = [r[metric] for r in rows if r['replay']==replay and not r['guarded']]
                summary.append(dict(replay=replay, metric=metric, guarded=statistics.median(a),
                    ablation=statistics.median(b), paired_delta=statistics.median([u-v for u,v in zip(a,b)])))
    return dict(shape=[m,n,k], compiled=compiled, rms=rms, reference_rms=rrms,
        normalized_rms=rms/max(rrms,1e-12), max_abs=float(diff.abs().max()),
        exact_ablation=True, rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before GPU work; preserve partial evidence if execution fails.
    with args.output.open('x') as sink:
        result = dict(command=sys.argv, torch=str(torch.__version__), cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Fixed defaults, single stream, stationary inference, paired same process; no tuning promotion',
            warning='CUDA event submission timing includes host starvation; replay bypasses Python guard', rows=[])
        for shape in ((128,128,256),(512,1536,1536),(32768,768,1536)):
            for compiled in (False,True):
                row=run_shape(*shape,compiled)
                result['rows'].append(row)
                print(json.dumps(dict(shape=shape,compiled=compiled,summary=row['summary'])),flush=True)
        json.dump(result,sink,indent=2)


if __name__ == '__main__': main()
