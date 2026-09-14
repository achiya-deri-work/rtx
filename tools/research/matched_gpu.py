"""Isolated GPU preparation/execution for immutable matched-pool experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from tools.predictor_study import build_adapter as _build_adapter, source_digest
from .matched_pool import MatchedPoolAdapter, make_matched_tuner


def build_adapter(task, timing, device):
    from rtx.autotune.runtime_coordinates import runtime_schedule_adapter
    adapter, hardware = _build_adapter(task, timing, device)
    return runtime_schedule_adapter(adapter), hardware


def write_new(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(document, stream, allow_nan=False, indent=2)


def verify_sources(manifest, pool_path):
    if manifest['source_sha256'] != source_digest():
        raise ValueError('runtime source changed since pool freeze')
    if 'source_snapshot' in manifest:
        from .source_snapshot import verify_snapshot
        metadata=manifest['source_snapshot']
        name=metadata['path']
        if Path(name).name!=name:
            raise ValueError('snapshot path must be a sibling filename')
        actual=verify_snapshot(Path(pool_path).parent/name,root=Path(__file__).resolve().parents[2])
        if actual!=metadata:
            raise ValueError('source snapshot identity mismatch')


def independent_metrics(output, reference, gate):
    if output.shape != reference.shape:
        raise RuntimeError('independent oracle shape mismatch')
    difference = output.float()-reference.float()
    floor = 1e-12
    rms = float(difference.square().mean().sqrt())
    reference_rms = float(reference.float().square().mean().sqrt())
    maximum = float(difference.abs().max())
    metrics = dict(finite=bool(output.isfinite().all()),
        reference_finite=bool(reference.isfinite().all()), rms_error=rms,
        reference_rms=reference_rms, normalized_rms=rms/max(reference_rms,floor),
        max_abs=maximum, normalized_max=maximum/max(float(reference.abs().max()),floor),
        floor=floor, gate=gate, oracle='FP32 GEMM cast to BF16')
    if not metrics['finite'] or not metrics['reference_finite'] or metrics['normalized_rms'] > gate:
        raise RuntimeError(str(metrics))
    return metrics


def independent_initial(adapter, task, device):
    # Other family-specific independent oracles must be added explicitly;
    # a harness self-reference is not a substitute for this admission gate.
    if task['family'] not in ('mxfp8_prequant_fwd','mxfp8_bwd','nvfp4_dynamic_fwd','nvfp4_jit_row_region_fwd'):
        raise ValueError('matched pilot independent oracle not yet implemented for this family')
    import torch
    from rtx.fp8 import _build_prequant_runner, _intern_prequant_config
    torch.manual_seed(task['seed'])
    m, n, k = (task['shape'][key] for key in ('m','n','k'))
    x = torch.randn(m,k, device=device, dtype=torch.bfloat16)*.1
    w = torch.randn(n,k, device=device, dtype=torch.bfloat16)*.02
    if task['family']=='mxfp8_bwd':
        from rtx.fp8_bwd import _build_bwd_runner
        from rtx.kernels.mxfp8 import MXFP8Problem
        dy = torch.randn(m,n,device=device,dtype=torch.bfloat16)*.1
        dx,dw = torch.empty_like(x),torch.empty_like(w)
        _build_bwd_runner(MXFP8Problem(m,n,k),adapter.initial_config,x.device)(dy,x,w,dx,dw)
        references = ((dy.float() @ w.float()).bfloat16(),
                      (dy.float().T @ x.float()).bfloat16())
        outputs={}
        for name,actual,reference in zip(('dX','dW'),(dx,dw),references):
            try:
                outputs[name]=independent_metrics(actual,reference,.07)
            except RuntimeError as exc:
                raise RuntimeError(f'{name}: {exc}') from exc
        return dict(finite=True,gate=.07,outputs=outputs,
                    normalized_rms=max(v['normalized_rms'] for v in outputs.values()),
                    oracle='independent FP32 dY@W and dY.T@X, each cast to BF16')
    output = torch.empty(m,n, device=device, dtype=torch.bfloat16)
    if task['family']=='mxfp8_prequant_fwd':
        _build_prequant_runner(x,w,_intern_prequant_config(adapter.initial_config))(x,w,output)
        gate=.07
    else:
        from rtx.configs.nvfp4 import NVFP4Problem
        from rtx.fp4 import NVFP4ForwardConfig, _make_block_dynamic_runner, _make_jit_region_dynamic_runner
        builder=(_make_jit_region_dynamic_runner if task['family']=='nvfp4_jit_row_region_fwd'
                 else _make_block_dynamic_runner)
        config=NVFP4ForwardConfig.from_materialized(adapter.initial_config)
        builder(NVFP4Problem(m,n,k),config,x.device)(x,w,output)
        gate=.20
    reference = (x.float() @ w.float().T).bfloat16().float()
    return independent_metrics(output,reference,gate)


def prepare(args):
    from rtx._portable_predictor import load_predictor
    task = dict(family=args.family, shape=dict(m=args.m,n=args.n,k=args.k),
                regime='hot', seed=args.seed)
    if getattr(args, 'backward_measurement_mode', 'host') != 'host':
        task['backward_measurement_mode'] = args.backward_measurement_mode
    timing = dict(warmup=3, samples=7, calls_per_sample=5)
    adapter, hardware = build_adapter(task, timing, args.device)
    predictor = load_predictor(args.predictor, research=True)
    if predictor.bundle(adapter) is None:
        raise ValueError('unsupported prior contract')
    metrics = independent_initial(adapter, task, args.device)
    candidates = [adapter.initial_config]
    candidates.extend(c for _,_,c in adapter.neighbors(adapter.initial_config)
                      if adapter.rejection(c) is None)
    retrieved = predictor.retrieval(adapter, 32)
    candidates.extend(retrieved)
    candidates.extend(adapter.sample(random.Random(args.seed),256,()))
    # Freeze all legal candidates from the prespecified draws. No timing rank
    # or fixed top-k cap selects which proposals survive this preparation.
    pool = MatchedPoolAdapter(adapter, [c for c in candidates if adapter.rejection(c) is None])
    document = pool.manifest(prefix_order=args.prefix_order)
    document.update(task=task, timing=timing, hardware=hardware, ranked_opening=args.ranked_opening,
        runtime_retrieval_ids=[adapter.config_id(c) for c in retrieved],
        source_sha256=source_digest(), initial_numerics=metrics,
        predictor_sha256=hashlib.sha256(Path(args.predictor).read_bytes()).hexdigest())
    from .source_snapshot import write_snapshot
    document['source_snapshot']=write_snapshot(Path(__file__).resolve().parents[2],
                                               Path(args.output).with_suffix('.sources.zip'))
    verify_sources(document,args.output)
    write_new(args.output, document)
    print('POOL', len(pool.pool), args.output, flush=True)


def run(args):
    from rtx.autotune.store import ResidualTuningStore
    manifest = json.loads(Path(args.pool).read_text())
    ranks = json.loads(Path(args.rankings).read_text())
    verify_sources(manifest,args.pool)
    if ranks['provenance']['pool'] != hashlib.sha256(Path(args.pool).read_bytes()).hexdigest():
        raise ValueError('rankings belong to a different pool document')
    if ranks['pool_id'] != manifest['pool_id'] or ranks['seed'] != manifest['task']['seed']:
        raise ValueError('ranking pool or seed mismatch')
    started = time.monotonic()
    adapter, hardware = build_adapter(manifest['task'],manifest['timing'],args.device)
    pool = MatchedPoolAdapter(adapter,[adapter.deserialize(c['config']) for c in manifest['configs']])
    if pool.pool_id != manifest['pool_id']:
        raise ValueError('GPU/context changed since pool freeze')
    metrics = independent_initial(adapter,manifest['task'],args.device)
    root = Path(args.output)
    result_path = root/f'{args.treatment}-{args.budget}.json'
    if result_path.exists():
        raise ValueError('result already exists; do not overwrite completed evidence')
    tuner = make_matched_tuner(pool, ResidualTuningStore(root/args.treatment),
        treatment=args.treatment, ranking=None if args.treatment=='ordinary' else ranks['rankings'][args.treatment],
        max_trials=args.budget, max_seconds=180., seed=ranks['seed'],
        prefix_order=manifest.get('prefix_order', 'axis_major'),
        ranked_opening=manifest.get('ranked_opening', 0))
    tuner.progress = lambda message: print(message,flush=True)
    result = tuner.tune()
    verification = pool.evaluate(result.config)
    if not verification.successful:
        raise RuntimeError(str(verification.as_dict()))
    write_new(result_path,dict(treatment=args.treatment, budget=args.budget,
        pool_id=pool.pool_id, hardware=hardware, initial_numerics=metrics,
        winner=pool.serialize(result.config), winner_ms=result.median_ms,
        verification=verification.as_dict(), elapsed_s=time.monotonic()-started,
        evaluated_trials=result.evaluated_trials, runtime_promotion=False))


def reference(args):
    from collections import Counter
    from rtx.autotune.store import ResidualTuningStore
    from .matched_pool import make_reference_tuner
    manifest=json.loads(Path(args.pool).read_text())
    verify_sources(manifest,args.pool)
    root=Path(args.output)
    if (root/'result.json').exists():
        raise ValueError('completed reference exists')
    adapter, hardware=build_adapter(manifest['task'],manifest['timing'],args.device)
    pool=MatchedPoolAdapter(adapter,[adapter.deserialize(c['config']) for c in manifest['configs']])
    if pool.pool_id != manifest['pool_id']:
        raise ValueError('device or semantic context changed')
    metrics=independent_initial(adapter,manifest['task'],args.device)
    store=ResidualTuningStore(root/'journal')
    tuner=make_reference_tuner(pool,store,seed=manifest['task']['seed'])
    tuner.progress=lambda message:print(message,flush=True)
    result=tuner.tune()
    rows=list(store.records(pool.context))
    complete={r['config_id'] for r in rows} == set(pool._configs)
    if not complete:
        raise RuntimeError('reference pool incomplete; preserve journal and resume')
    write_new(root/'result.json',dict(pool_id=pool.pool_id, hardware=hardware,
        initial_numerics=metrics, candidate_count=len(rows),
        outcomes=dict(Counter(r['outcome']['status'] for r in rows)),
        best_screened_config=pool.serialize(result.config),best_screened_ms=result.median_ms,
        complete=True, runtime_promotion=False,training_admission=False,
        limitation='Best screened observation needs paired confirmation; finite pool, not global optimum'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','reference'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--family',default='mxfp8_prequant_fwd')
    parser.add_argument('--backward-measurement-mode', choices=('host','cuda_graph'), default='host',
                        help='Frozen backward execution mode; applies only when preparing a pool')
    for key in ('m','n','k'):
        parser.add_argument('--'+key,type=int)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--predictor')
    parser.add_argument('--pool')
    parser.add_argument('--rankings')
    parser.add_argument('--treatment',choices=('ordinary','portfolio','rules','portfolio_ml'))
    parser.add_argument('--budget',type=int,choices=(8,16,32,64))
    parser.add_argument('--ranked-opening',type=int,choices=range(5),default=0,
                        help='Preparation only: freeze up to four novel ranked opening candidates')
    parser.add_argument('--prefix-order', default='axis_major',
                        choices=('axis_major', 'structural_interleaved'),
                        help='Preparation only; run uses the frozen manifest policy')
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'reference':reference}[args.action](args)


if __name__=='__main__':
    main()
