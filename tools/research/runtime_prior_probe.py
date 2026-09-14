"""Public compiled linear first-hit probe; no automatic policy promotion."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
from contextlib import nullcontext


def family_spec(family):
    specs={
        'mxfp8_prequant_fwd': ('MXFP8Linear','rtx.prequant_autotune','tune_mxfp8_prequant',{},.07),
        'nvfp4_dynamic_fwd': ('NVFP4Linear','rtx.nvfp4_inference_autotune',
                            'tune_nvfp4_inference_state',{'scaling':'block'},.20),
        'nvfp4_jit_row_region_fwd': ('NVFP4Linear','rtx.nvfp4_inference_autotune',
                                 'tune_nvfp4_inference_state',{'scaling':'jit_row_region'},.20),
    }
    if family not in specs:
        raise ValueError('unsupported probe family')
    return specs[family]


def policy_for(mode, budget, artifact):
    from rtx.autotune.effort import AutotunePolicy
    if mode not in ('ordinary', 'retrieval', 'ml'):
        raise ValueError('unsupported probe mode')
    if budget not in (8, 16, 32, 64):
        raise ValueError('unsupported probe budget')
    if mode != 'ordinary' and not artifact:
        raise ValueError('prior-backed mode requires an artifact')
    return AutotunePolicy(max_trials=budget, max_seconds=180,
        search_schedule='progressive', cache_behavior='refine',
        predictor_artifact=None if mode == 'ordinary' else str(artifact),
        predictor_heads=('trees',) if mode == 'ml' else (), allow_research_prior=True,
        confirmation_repeats=0, finalist_rounds=2, warmup=3, samples=7,
        calls_per_sample=5, seed=2718)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('ordinary', 'retrieval', 'ml'), required=True)
    parser.add_argument('--budget', type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument('--artifact')
    parser.add_argument('--family',default='mxfp8_prequant_fwd')
    parser.add_argument('--startup-trace',action='store_true',help='Diagnostic hooks; imports are prewarmed')
    for name, default in (('m', 4096), ('n', 768), ('k', 1792)):
        parser.add_argument('--' + name, type=int, default=default)
    args = parser.parse_args()
    class_name,module_name,tuner_name,kwargs,gate=family_spec(args.family)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'{args.budget}.json'
    if target.exists():
        raise ValueError('completed probe exists; do not overwrite evidence')
    policy = policy_for(args.mode, args.budget, args.artifact)
    from .source_snapshot import write_snapshot, verify_snapshot
    from .matched_gpu import write_new
    source_root = Path(__file__).resolve().parents[2]
    snapshot = root / 'sources.zip'
    sources = (verify_snapshot(snapshot, root=source_root) if snapshot.exists()
               else write_snapshot(source_root, snapshot))
    import torch
    import rtx
    import importlib
    autotune=importlib.import_module(module_name)
    from unittest.mock import patch
    from torch._inductor.compile_fx import compile_fx
    from rtx.runtime import validate_runtime_environment
    environment = validate_runtime_environment()
    torch.manual_seed(2718)
    layer = getattr(rtx,class_name)(args.k, args.n, device='cuda', backend='materialized',
        tuning_policy=policy, autotune_cache_dir=root/'cache',**kwargs).eval()
    x = torch.randn(args.m, args.k, device='cuda', dtype=torch.bfloat16)*.1
    graphs = []
    from .startup_trace import StartupTrace, install
    trace=StartupTrace() if args.startup_trace else None
    def backend(graph, inputs):
        graphs.append(graph)
        with trace.span('frontend_compile') if trace else nullcontext():
            return compile_fx(graph, inputs)
    compiled = torch.compile(layer, backend=backend, fullgraph=True, dynamic=False)
    def records():
        return [json.loads(line) for path in (root/'cache').rglob('observations.jsonl')
                for line in path.read_text().splitlines()]
    with install(trace) if trace else nullcontext(), torch.inference_mode(), patch.object(autotune, tuner_name,
            wraps=getattr(autotune,tuner_name)) as tuner:
        torch.cuda.synchronize()
        start = time.monotonic()
        with trace.span('first_call') if trace else nullcontext():
            y = compiled(x)
            torch.cuda.synchronize()
        first_call_s = time.monotonic()-start
        assert tuner.call_count == 1, 'expected one first-hit/refinement call'
        reference = (x.float() @ layer.weight.float().T).bfloat16().float()
        difference = y.float()-reference
        rms = float(difference.square().mean().sqrt())
        ref_rms = float(reference.square().mean().sqrt())
        metrics = dict(finite=bool(y.isfinite().all()), rms=rms, reference_rms=ref_rms,
            normalized_rms=rms/max(ref_rms, 1e-12), max_abs=float(difference.abs().max()),
            floor=1e-12, gate=gate)
        assert metrics['finite'] and metrics['normalized_rms'] < gate, metrics
        history = records()
        assert len(history) == args.budget, 'incomplete effort budget'
        if args.mode != 'ordinary':
            assert sum(r['strategy']=='retrieval' for r in history[:8]) == 2, 'prior opening missing'
        graph_count = len(graphs)
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                warmed = compiled(x)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.testing.assert_close(warmed, y, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            captured = compiled(x)
        samples = []
        for _ in range(15):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(20):
                graph.replay()
            end.record(); end.synchronize()
            samples.append(begin.elapsed_time(end)/20)
        torch.testing.assert_close(captured, y, rtol=0, atol=0)
        assert tuner.call_count == 1 and len(graphs) == graph_count
        assert records() == history
    winners = [json.loads(p.read_text()) for p in (root/'cache/runtime_winners').rglob('*.json')]
    assert len(winners) == 1
    assert winners[0]['family']==args.family
    document = dict(family=args.family,mode=args.mode, budget=args.budget, shape=dict(m=args.m,n=args.n,k=args.k),
        policy=asdict(policy), environment=environment, device=torch.cuda.get_device_name(),
        source_snapshot=sources, artifact_sha256=(hashlib.sha256(Path(args.artifact).read_bytes()).hexdigest()
            if args.artifact else None), first_call_s=first_call_s, graphs=graph_count,
        numerics=metrics,startup_trace=None if trace is None else trace.report(),
        graph_samples_ms=samples, median_us=statistics.median(samples)*1000,
        opening=[dict(strategy=r['strategy'], config_id=r['config_id']) for r in history[:8]],
        winner=winners[0], runtime_promotion=False, training_admission=False,
        limitation='Development inference probe; unrestricted runtime search, not matched finite pool; '
                   'process startup excluded and compiler disk caches not cleared; cross-mode timings unpaired')
    write_new(target, document)
    print(json.dumps(dict(mode=args.mode,budget=args.budget,first_call_s=first_call_s,
                         median_us=document['median_us'],numerics=metrics)), flush=True)


if __name__ == '__main__':
    main()
