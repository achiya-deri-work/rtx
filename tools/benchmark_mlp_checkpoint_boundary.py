"""Isolated checkpoint/live-set experiment; not an AFRL performance proxy.

Run each boundary/checkpoint combination in a fresh process. No runtime defaults,
winner caches, or caller compilation settings are changed by this tool.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
from torch.utils.checkpoint import checkpoint
import rtx
from rtx.autotune.replacement_benchmark import error_metrics, source_manifest


class Stack(torch.nn.Module):
    def __init__(self, config, width, hidden, depth, boundary, checkpoints):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            rtx.MXFP8MLP(width, hidden, width, config=config, device='cuda')
            for _ in range(depth)])
        self.config, self.boundary, self.checkpoints = config, boundary, checkpoints
        for layer in self.layers:
            layer._first_hit_spec = json.dumps({'mode': 'fixed', 'initial': config.to_dict()}, sort_keys=True)

    def block(self, x, index):
        layer = self.layers[index]
        z = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        if self.boundary == 'selected':
            y = layer(z)
        else:
            y = rtx.mxfp8_mlp(z, *layer.parameters(), config=self.config)
        return x + y

    def segment(self, x):
        for index in range(len(self.layers)):
            if self.checkpoints == 'nested':
                x = checkpoint(self.block, x, index, use_reentrant=False)
            else:
                x = self.block(x, index)
        return x

    def forward(self, x):
        if self.checkpoints == 'none':
            return self.segment(x)
        return checkpoint(self.segment, x, use_reentrant=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--shape', default='32,640,256', help='actual rank-3 input; config must be legal for flattened M')
    p.add_argument('--hidden', type=int, default=768)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--boundary', choices=('selected', 'fixed'), required=True)
    p.add_argument('--checkpoints', choices=('none', 'outer', 'nested'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--emulate-precision-casts', action='store_true',
                   help='diagnostic caller-side Inductor option, never a runtime default')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    shape = tuple(map(int, args.shape.split(',')))
    if len(shape) != 3 or min(shape) < 1 or args.depth < 1:
        raise ValueError('positive rank-3 shape and depth required')
    config = rtx.MXFP8MLPConfig.from_dict(json.loads(args.config.read_text())['config'])
    sources = source_manifest()
    torch.manual_seed(317)
    model = Stack(config, shape[-1], args.hidden, args.depth, args.boundary, args.checkpoints)
    parameters = tuple(model.parameters())
    identities = tuple(map(id, parameters))
    options = {'triton.cudagraphs': False}
    if args.emulate_precision_casts:
        options['emulate_precision_casts'] = True
    compiled = torch.compile(model, fullgraph=True, dynamic=False, options=options)
    x = torch.empty(shape, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    dy = torch.empty_like(x)

    def pair(fn):
        y = fn(x)
        return y, *torch.autograd.grad(y, (x, *parameters), dy)

    def reference(z):
        for index in range(args.depth):
            z = model.block(z, index)
        return z
    compiled_reference = torch.compile(reference, fullgraph=True, dynamic=False,
                                       options=options)
    checks = []
    for seed in (317, 318):
        torch.manual_seed(seed)
        with torch.no_grad():
            x.normal_(0, .2)
            dy.normal_(0, .1)
            for weight in parameters:
                weight.normal_(0, .02)
        # Match compilation of surrounding LayerNorm/residual operations too.
        # Eager-vs-compiled rounding is a separate diagnostic, not checkpoint error.
        expected = pair(compiled_reference)
        eager = pair(reference)
        eager_metrics = [error_metrics(a, b) for a, b in zip(expected, eager)]
        del eager
        actual = pair(compiled)
        metrics = [error_metrics(a, b) for a, b in zip(actual, expected)]
        finite = all(bool(t.isfinite().all()) for t in (*actual, *expected))
        if not finite or any(v['normalized_rms'] > .025 for v in metrics):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                json.dump(dict(status='numerical_failure', seed=seed, finite=finite,
                    metrics=metrics, compiled_reference_vs_eager=eager_metrics,
                    source_hashes=sources, config=config.to_dict(), command=sys.argv,
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), application_qualified=False), stream, indent=2)
            raise AssertionError({'seed': seed, 'finite': finite, 'metrics': metrics})
        checks.append({'seed': seed, 'finite': finite, 'metrics': metrics,
                       'compiled_reference_vs_eager': eager_metrics})
        del actual, expected
    for _ in range(5):
        pair(compiled)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    timings, live = [], []
    for _ in range(9):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            pair(compiled)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / 5)
        live.append(torch.cuda.memory_allocated())
    assert tuple(map(id, model.parameters())) == identities
    assert sources == source_manifest(), 'source changed during experiment'
    assert max(live) == min(live), live
    report = dict(status='ok', arguments=vars(args) | {'config': str(args.config), 'output': str(args.output)},
        command=sys.argv, torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        config=config.to_dict(), source_hashes=sources,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checks=checks,
        oracle='same-recipe compiled uncheckpointed stack; not independent quantization oracle',
        parameters_preserved=True, resident_bytes=resident, live_bytes=live,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        samples_ms=timings, median_ms=statistics.median(timings), workspace=rtx.workspace_report(),
        application_qualified=False, scope='isolated residual MLP stack, no attention/teacher/optimizer')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: report[k] for k in ('status', 'median_ms', 'resident_bytes', 'peak_allocated_bytes')}), flush=True)


if __name__ == '__main__':
    main()
