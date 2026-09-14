"""Paired public-frontend confirmation of already selected linear configurations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import zipfile

from rtx.autotune.core import stable_id
from .source_snapshot import inventory, MANIFEST, verify_snapshot, write_snapshot
from .matched_gpu import write_new
from .matched_compare import summarize_pairs
from .runtime_prior_probe import family_spec


def selections(documents):
    choices, configs = {}, {}
    contracts = set()
    for doc in documents:
        budget, mode = doc['budget'], doc['mode']
        if budget not in (8, 16, 32, 64) or mode not in ('ordinary', 'retrieval', 'ml'):
            raise ValueError('unexpected budget or mode')
        if mode in choices.setdefault(str(budget), {}):
            raise ValueError('duplicate selection cell')
        winner = doc['winner']
        family_spec(winner['family'])
        contracts.add(stable_id(dict(shape=doc['shape'], environment=doc['environment'],
            family=winner['family'],variant=winner.get('variant','default'),
            device=winner['device_id'], revision=winner['kernel_revision'])))
        cid = stable_id(winner['config'])
        configs[cid] = winner['config']
        choices[str(budget)][mode] = cid
    if len(contracts) != 1 or len(choices) != 4 or any(len(v) != 3 for v in choices.values()):
        raise ValueError('incomplete or incompatible selection matrix')
    return choices, configs


def verify_execution_sources(path, metadata, root):
    if verify_snapshot(path) != metadata:
        raise ValueError('selection snapshot identity mismatch')
    with zipfile.ZipFile(path) as archive:
        files = json.loads(archive.read(MANIFEST))['files']
    # New analysis scripts may be added, but execution code and the original
    # input-generation probe must remain identical. Archive verification above
    # still covers all historical files, not just this execution subset.
    selected = lambda name: name.startswith('rtx/') or name == 'tools/research/runtime_prior_probe.py'
    current = {name: data for name, data in inventory(root).items() if selected(name)}
    expected = {name: info for name, info in files.items() if selected(name)}
    if not current or set(current) != set(expected):
        raise ValueError('execution source inventory changed')
    if any(hashlib.sha256(data).hexdigest() != expected[name]['sha256'] for name, data in current.items()):
        raise ValueError('execution source changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    source_root = Path(__file__).resolve().parents[2]
    paths = [root/mode/f'{budget}.json' for mode in ('ordinary', 'retrieval', 'ml')
             for budget in (8, 16, 32, 64)]
    documents = [json.loads(p.read_text()) for p in paths]
    choices, configs = selections(documents)
    for mode in ('ordinary', 'retrieval', 'ml'):
        for doc in (d for d in documents if d['mode'] == mode):
            verify_execution_sources(root/mode/'sources.zip', doc['source_snapshot'], source_root)
    if len(configs) < 2:
        raise ValueError('all selections identical; record a selection tie without a paired contest')
    output.mkdir(parents=True, exist_ok=False)
    sources = write_snapshot(source_root, output/'sources.zip')
    import torch
    import rtx
    from rtx.runtime import validate_runtime_environment
    from rtx.autotune.winners import runtime_winner_key
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.prequant_autotune import prequant_config_from_dict
    from rtx.autotune.confirmation import confirm_finalists
    from rtx.autotune.outcomes import TrialOutcome
    shape = documents[0]['shape']
    family=documents[0]['winner']['family']
    class_name,_,_,kwargs,gate=family_spec(family)
    cls=getattr(rtx,class_name)
    m, n, k = (shape[key] for key in ('m', 'n', 'k'))
    environment = validate_runtime_environment()
    assert environment == documents[0]['environment'], 'software environment changed'
    key = runtime_winner_key(family, MXFP8Problem(m,n,k), device=torch.device('cuda:0'))
    assert key.device_id == documents[0]['winner']['device_id'], 'device identity changed'
    torch.manual_seed(2718)
    base = cls(k,n,device='cuda',backend='materialized',autotune='off',**kwargs).eval()
    x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    prepared, metrics = {}, {}
    with torch.inference_mode():
        reference = (x.float() @ base.weight.float().T).bfloat16().float()
        for cid, config in configs.items():
            if family=='mxfp8_prequant_fwd':
                fixed=dict(prequant_config=prequant_config_from_dict(config))
            else:
                from rtx.nvfp4_inference_autotune import dynamic_config_from_dict
                fixed=dict(dynamic_config=dynamic_config_from_dict(config))
            layer = cls(k,n,device='cuda',backend='materialized',autotune='off',
                        **kwargs,**fixed).eval()
            layer.weight.copy_(base.weight)
            compiled = torch.compile(layer,fullgraph=True,dynamic=False)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    y = compiled(x)
            torch.cuda.current_stream().wait_stream(stream)
            difference = y.float()-reference
            rms, ref_rms = float(difference.square().mean().sqrt()), float(reference.square().mean().sqrt())
            metrics[cid] = dict(finite=bool(y.isfinite().all()), rms=rms, reference_rms=ref_rms,
                normalized_rms=rms/max(ref_rms,1e-12),max_abs=float(difference.abs().max()),gate=gate)
            assert metrics[cid]['finite'] and metrics[cid]['normalized_rms'] < gate, metrics[cid]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                captured = compiled(x)
            graph.replay(); torch.cuda.synchronize()
            torch.testing.assert_close(captured,y,rtol=0,atol=0)
            prepared[cid] = (graph,captured,y,compiled,layer,stream)
        write_new(output/'plan.json',dict(family=family,choices=choices,configs=configs,shape=shape,rounds=10,
            numerics=metrics,source_snapshot=sources,environment=environment,
            inputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            runtime_promotion=False,training_admission=False))
        def evaluate(cid):
            graph,captured,y,*_ = prepared[cid]
            samples=[]
            for _ in range(7):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20): graph.replay()
                end.record();end.synchronize()
                samples.append(begin.elapsed_time(end)/20)
            torch.testing.assert_close(captured,y,rtol=0,atol=0)
            return TrialOutcome('ok',median_ms=statistics.median(samples),timings_ms=samples)
        with (output/'measurements.jsonl').open('x') as sink:
            def record(item):
                sink.write(json.dumps(item,allow_nan=False)+'\n');sink.flush();os.fsync(sink.fileno())
            confirmation=confirm_finalists(evaluate,{cid:cid for cid in configs},rounds=10,seed=2718,record=record)
        write_new(output/'confirmation.json',confirmation)
        summary=summarize_pairs(confirmation,choices)
        write_new(output/'summary.json',summary)
        print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
