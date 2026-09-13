"""Fixed-budget prospective GPU comparison; frozen models, isolated task journals.

No training or winner installation. Each cell runs in a fresh subprocess.
The baseline is the pre-study effort policy (no reserved random pull).
"""
import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TREATMENTS = ('existing', 'retrieval', 'residual', 'residual_mlp')
FAMILIES = ('mxfp8_prequant_fwd', 'mxfp8_bwd', 'nvfp4_dynamic_fwd',
            'nvfp4_delayed_fwd', 'nvfp4_jit_row_region_fwd',
            'mlp_forward', 'mlp_train', 'glu')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def source_digest():
    paths = [*sorted((ROOT/'rtx').rglob('*.py')), Path(__file__)]
    shape_driver = ROOT/'tools/shape_stratified_study.py'
    if shape_driver.is_file():
        paths.append(shape_driver)
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def study_document(output):
    """Relocate the artifact without changing its frozen content identity."""
    document = json.loads((Path(output)/'plan.json').read_text())
    artifact = Path(os.environ.get('RTX_STUDY_ARTIFACT', document['artifact']))
    if not artifact.is_file():
        artifact = ROOT/'autotune_priors'/Path(document['artifact']).name
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != document['artifact_sha256']:
        raise ValueError('frozen artifact changed')
    document['artifact'] = str(artifact.resolve())
    return document


def observed_geometry(paths):
    """Strictly audit every record, including failures, without ML dependencies."""
    seen, count = set(), 0
    def ingest(lines):
        nonlocal count
        for line in lines:
            row = json.loads(line)
            f = row['features']
            # Exclude any previously attempted common (M,K) pair, even across
            # different N/H/O/families. This is stricter than exact-shape novelty.
            m, k = (f.get('context.workload.'+s) for s in ('m', 'k'))
            if m is not None and k is not None:
                seen.add((int(m), int(k)))
            count += 1
    for path in map(Path, paths):
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                for name in sorted(archive.namelist()):
                    if name.endswith('/observations.jsonl'):
                        with archive.open(name) as stream:
                            ingest(stream)
        else:
            files = sorted(path.rglob('observations.jsonl'))
            if not files:
                raise ValueError(f'no observation journals: {path}')
            for file in files:
                with file.open() as stream:
                    ingest(stream)
    return seen, count


def make_tasks(shapes, budgets, replicates, families=FAMILIES, treatments=TREATMENTS):
    tasks = []
    # Complete a low-budget coverage wave first; alternate treatment order
    # across contexts/replicates to reduce cache/thermal/order bias.
    for budget in budgets:
        for rep in range(replicates):
            for si, shape in enumerate(shapes):
                for fi, family in enumerate(families):
                    order = list(treatments)
                    shift = (fi+si+rep) % len(order)
                    order = order[shift:]+order[:shift]
                    for treatment in order:
                        item = dict(family=family, shape=shape, regime='hot',
                                    budget=budget, replicate=rep, treatment=treatment,
                                    seed=20260915+rep)
                        item['id'] = digest(item)[:20]
                        tasks.append(item)
    return tasks


def plan(args):
    if args.replicates < 1 or not math.isfinite(args.task_seconds) or args.task_seconds <= 0:
        raise ValueError('replicates and task seconds must be positive')
    output = Path(args.output)
    if (output/'plan.json').exists():
        raise ValueError('plan already exists; run it to resume, or choose a new output')
    source = json.loads(Path(args.sources).read_text())
    paths = [*source['train_paths'], *source['test_paths']]
    seen, count = observed_geometry(paths)
    shapes = [dict(m=384, k=640, n=1280), dict(m=3072, k=1280, n=2560)]
    if any((s['m'], s['k']) in seen for s in shapes):
        raise ValueError('proposed shape was already observed; select a new held-out geometry')
    artifact = Path(args.artifact).resolve()
    treatments = ('existing', 'mixed', 'residual_mlp', 'mixed_residual_mlp') if getattr(args, 'mixed', False) else TREATMENTS
    document = dict(schema=1, purpose='candidate_pool_followup' if getattr(args, 'mixed', False) else 'prospective_evaluation_only',
        treatments=treatments,
        artifact=str(artifact), artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        source_sha256=source_digest(), geometry_audit=dict(paths=paths, records=count,
            excluded_m_k_pairs=sorted(seen)),
        timing=dict(warmup=3, samples=7, calls_per_sample=5),
        task_seconds=args.task_seconds, cache_scope='fresh process; shared persistent compiler caches',
        tasks=make_tasks(shapes[:args.shapes], tuple(getattr(args, 'budgets', (8,16,32))), args.replicates,
                         treatments=treatments))
    atomic(output/'plan.json', document)
    print(f'PLAN tasks={len(document["tasks"])} audited_records={count} output={output}', flush=True)


def build_adapter(task, timing, device):
    import torch
    from rtx.autotune.hardware import static_device_profile
    from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec
    family, shape = task['family'], task['shape']
    profile = static_device_profile(device)
    if family in ('glu', 'mlp_forward', 'mlp_train'):
        torch.manual_seed(task['seed'])
        m, k, h = shape['m'], shape['k'], shape['n']
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)*.1
        gu = torch.randn(2*h, k, device=device, dtype=torch.bfloat16)*.02
        if family == 'glu':
            from rtx.autotune.glu import make_glu_adapter
            adapter = make_glu_adapter(x, gu, output_format='mxfp8', **timing)
        else:
            from rtx.autotune.mlp import make_mlp_adapter
            down = torch.randn(k, h, device=device, dtype=torch.bfloat16)*.02
            adapter = make_mlp_adapter(x, gu, down, precision='bf16_gate',
                phase='train' if family == 'mlp_train' else 'forward', **timing)
        return adapter, profile
    from rtx.autotune.dataset import _BACKENDS, DatasetJob
    spec = ShapeSpec(**shape)
    protocol = BenchmarkProtocol(warmup_calls=timing['warmup'], samples=timing['samples'],
        adaptive_sampling=False, target_batch_ms=2., stabilization_target_ms=5.,
        min_calls_per_sample=5, max_calls_per_sample=128,
        correctness_rtol=.07 if family=='mxfp8_bwd' else .05,
        correctness_atol=1. if family=='mxfp8_bwd' else .5)
    job = DatasetJob(family, (spec,), regimes=('hot',), protocol=protocol)
    campaign = SimpleNamespace(device=torch.device(device), hardware_profile=profile,
                               manifest=SimpleNamespace(seed=task['seed']))
    backend = _BACKENDS[family]
    harness = backend.make_harness(campaign, job, spec, task['regime'])
    adapter = backend.make_adapter(campaign, job, spec, task['regime'], harness, {})
    return adapter, profile


def task_policy(task, document):
    from rtx.autotune.effort import AutotunePolicy
    treatment = task['treatment']
    heads = {'retrieval': (), 'residual': ('residual',), 'residual_mlp': ('residual','mlp')}
    plain = treatment in ('existing', 'mixed')
    return AutotunePolicy(max_trials=task['budget'], max_seconds=document['task_seconds'],
        candidate_mix='mixed' if treatment.startswith('mixed') else 'legacy',
        pool_size=128, seed=task['seed'], **document['timing'],
        exploration_trials=0 if treatment=='existing' else 1,
        predictor_artifact=None if plain else document['artifact'],
        predictor_heads=heads.get(treatment, ('residual','mlp')), allow_research_prior=True)


def setup_context_key(task):
    """Setup is independent of policy/budget, but not shape/family/regime."""
    return digest({k:task[k] for k in ('family', 'shape', 'regime')})


def blocking_failure(result):
    """Stop repeating broken references, initial candidates or poisoned CUDA contexts."""
    if result['status'] == 'setup_error':
        return True
    error = result.get('error', '') or ''
    return result['status'] == 'search_error' and any(token in error for token in (
        'FatalDeviceContextError', 'initial configuration failed:',
        'cudaErrorIllegalInstruction', 'illegal memory access'))


def setup_quarantines(output, document):
    blocked = dict(document.get('blocked_contexts', {}))
    if document.get('quarantine_setup_failures'):
        for path in sorted((Path(output)/'tasks').glob('*/result.json')):
            result = json.loads(path.read_text())
            if blocking_failure(result):
                blocked.setdefault(setup_context_key(result['task']), dict(
                    task_id=result['task']['id'], error=result.get('error'), source=str(path)))
    return blocked


def worker(output, task_id, device):
    import torch
    from rtx._portable_predictor import load_predictor
    from rtx.autotune.effort import make_effort_autotuner
    from rtx.autotune.store import ResidualTuningStore
    output = Path(output)
    document = study_document(output)
    task = next(t for t in document['tasks'] if t['id']==task_id)
    root = output/'tasks'/task_id
    started = time.monotonic()
    report = dict(task=task, status='error', phase='setup', source_sha256=document['source_sha256'])
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(device)
        adapter, profile = build_adapter(task, document['timing'], device)
        torch.cuda.synchronize()
        machine = dict(host=platform.node(), device=str(device), name=profile['name'],
            capability=profile['capability'], multiprocessor_count=profile['multiprocessor_count'],
            total_memory=profile['total_memory'])
        machine_path = output/'machine.json'
        if machine_path.exists() and json.loads(machine_path.read_text()) != machine:
            raise ValueError('study belongs to another machine/device; use a separate output')
        if not machine_path.exists():
            atomic(machine_path, machine)
        report.update(setup_s=time.monotonic()-started, hardware=profile,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      context=adapter.context.as_dict())
        report['phase'] = 'search'
        policy = task_policy(task, document)
        before = time.monotonic()
        predictor = load_predictor(document['artifact'], research=True)
        report['contract_compatible'] = predictor.bundle(adapter) is not None
        report['contract_probe_s'] = time.monotonic()-before
        # The probe is diagnostic. Clear its cache so artifact loading is still
        # included in each learned treatment's measured construction time.
        from rtx._portable_predictor import _load
        _load.cache_clear()
        del predictor
        store = ResidualTuningStore(root/'journal')
        tune_start = time.monotonic()
        original_evaluator = adapter.evaluator
        def evaluate(config):
            outcome = original_evaluator(config)
            if outcome.successful and 'time_to_first_valid_s' not in report:
                report['time_to_first_valid_s'] = time.monotonic()-tune_start
            return outcome
        adapter.evaluator = evaluate
        tuner = make_effort_autotuner(adapter, store, policy, progress=lambda s:print(s,flush=True))
        report['policy_setup_s'] = time.monotonic()-tune_start
        result = tuner.tune()
        torch.cuda.synchronize()
        report.update(tuning_wall_s=time.monotonic()-tune_start, status='ok',
            attempts=len(list(store.records(adapter.context))), winner_config=adapter.serialize(result.config),
            winner_ms=result.median_ms, context_id=result.context_id,
            strategy_trials=result.strategy_trials)
        # Independent post-search repeats of the selected config. Do not use
        # these to change the search result; preserve failure outcomes too.
        verification = []
        report['phase'] = 'verification'
        for _ in range(3):
            verification.append(adapter.evaluate(result.config).as_dict())
        report['verification'] = verification
        valid = [v['median_ms'] for v in verification if v['status']=='ok' and v['median_ms'] is not None]
        report['verified_ms'] = statistics.median(valid) if len(valid)==3 else None
        if len(valid)!=3:
            report['status'] = 'verification_failed'
        report['verification_cv'] = statistics.pstdev(valid)/statistics.mean(valid) if len(valid)==3 else None
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    except Exception:
        report['status'] = report['phase'] + '_error'
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report['worker_wall_s'] = time.monotonic()-started
        atomic(root/'result.json', report)
    return 0 if report['status']=='ok' else 1


def run(args):
    if not math.isfinite(args.hours) or args.hours <= 0 or args.max_tasks < 0:
        raise ValueError('hours must be positive and max-tasks nonnegative')
    output = Path(args.output).resolve()
    document = study_document(output)
    if document['source_sha256'] != source_digest():
        raise ValueError('source changed since plan creation; use a new study directory')
    if document['artifact_sha256'] != hashlib.sha256(Path(document['artifact']).read_bytes()).hexdigest():
        raise ValueError('frozen artifact changed')
    with (output/'.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic()+args.hours*3600
        done = 0
        blocked = setup_quarantines(output, document)
        for task in document['tasks']:
            if document['source_sha256'] != source_digest():
                raise ValueError('source changed during study; refusing mixed-revision observations')
            root = output/'tasks'/task['id']
            if (root/'result.json').exists():
                continue
            if time.monotonic()>=deadline or args.max_tasks and done>=args.max_tasks:
                break
            root.mkdir(parents=True, exist_ok=True)
            if setup_context_key(task) in blocked:
                reason = blocked[setup_context_key(task)]
                atomic(root/'result.json', dict(task=task, status='blocked_setup', phase='setup',
                    error=reason.get('error'), blocked_by=reason, attempts=0))
                print(f'BLOCK {task["id"]} setup failed for this family/shape; no GPU work', flush=True)
                continue
            print('START', task, flush=True)
            env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
            start = time.monotonic()
            with (root/'worker.log').open('a') as stream:
                command = [sys.executable, '-m', 'tools.predictor_study', 'worker',
                           '--output', str(output), '--task', task['id'], '--device', args.device]
                try:
                    result = subprocess.run(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=document['task_seconds']+180)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = -1
            if not (root/'result.json').exists():
                atomic(root/'result.json', dict(task=task, status='worker_failed', returncode=code))
            atomic(root/'process.json', dict(returncode=code, wall_s=time.monotonic()-start))
            record = json.loads((root/'result.json').read_text())
            if document.get('quarantine_setup_failures') and blocking_failure(record):
                blocked[setup_context_key(task)] = dict(task_id=task['id'], error=record.get('error'),
                                                       source=str(root/'result.json'))
            print(f'DONE {task["id"]} status={code} elapsed={time.monotonic()-start:.1f}s', flush=True)
            done += 1
        summarize(output)


def summarize(output):
    output = Path(output)
    document = json.loads((output/'plan.json').read_text())
    results = [json.loads(p.read_text()) for p in sorted((output/'tasks').glob('*/result.json'))]
    flat = []
    for r in results:
        task = r['task']
        row = dict(task_id=task['id'], family=task['family'], shape=json.dumps(task['shape'], sort_keys=True),
            category=task.get('category'), shape_group=task.get('shape_group'), regime=task['regime'],
            replicate=task['replicate'], budget=task['budget'], treatment=task['treatment'], status=r['status'])
        row['failure_phase'] = r.get('phase') if r['status'] != 'ok' else None
        row['error'] = r.get('error')
        for k in ('attempts','contract_compatible','winner_ms','verified_ms','verification_cv',
                  'setup_s','policy_setup_s','tuning_wall_s','worker_wall_s','peak_allocated_bytes','time_to_first_valid_s'):
            row[k] = r.get(k)
        records = []
        for path in (output/'tasks'/task['id']/'journal').rglob('observations.jsonl'):
            for line in path.read_text().splitlines():
                records.append(json.loads(line))
        row['failed_trials'] = sum(x['outcome']['status']!='ok' for x in records)
        row['compile_s'] = sum((x['outcome'].get('compile_ms') or 0)/1000 for x in records)
        row['wasted_compile_s'] = sum((x['outcome'].get('compile_ms') or 0)/1000 for x in records if x['outcome']['status']!='ok')
        row['proposal_s'] = sum(x.get('metadata',{}).get('proposal_elapsed_s',0) for x in records)
        row['evaluation_s'] = sum(x.get('elapsed_s',0) for x in records)
        row['confirmation_s'] = sum(x.get('metadata',{}).get('confirmation',{}).get('elapsed_s',0) for x in records)
        row['confirmation_compile_s'] = sum((o.get('compile_ms') or 0)/1000 for x in records
            for o in x.get('metadata',{}).get('confirmation',{}).get('outcomes',[]))
        flat.append(row)
    groups = defaultdict(list)
    for row in flat:
        groups[(row['family'], row['shape'], row['replicate'], row['budget'])].append(row)
    for group in groups.values():
        # Matched cells only; best observed is a within-study reference, not
        # an exhaustive oracle. Failed/partial cells remain in the raw export.
        complete = len(group)==len(document.get('treatments', TREATMENTS)) and all(r['status']=='ok' and r['verified_ms'] and r['attempts']==r['budget'] for r in group)
        best = min((r['verified_ms'] for r in group if r['verified_ms']), default=None)
        baseline = next((r['verified_ms'] for r in group if r['treatment']=='existing'), None)
        for row in group:
            row['matched_complete'] = complete
            row['observed_regret_pct'] = 100*(row['verified_ms']/best-1) if complete else None
            row['speedup_vs_existing'] = baseline/row['verified_ms'] if complete else None
    output_rows = output/'summary.csv'
    if flat:
        with output_rows.with_suffix('.csv.tmp').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
        os.replace(output_rows.with_suffix('.csv.tmp'), output_rows)
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(flat), output/'summary.parquet.tmp')
        os.replace(output/'summary.parquet.tmp', output/'summary.parquet')
    atomic(output/'summary.json', dict(planned=len(document['tasks']), completed=len(results), rows=flat))
    print(f'SUMMARY completed={len(results)}/{len(document["tasks"])} output={output_rows}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--output', required=True)
    p.add_argument('--artifact', default='autotune_priors/tournament_predictors_v1.json.gz')
    p.add_argument('--sources', default='autotune_reports/effort_ml_v2/audit.json')
    p.add_argument('--shapes', type=int, choices=(1,2), default=2)
    p.add_argument('--replicates', type=int, default=2)
    p.add_argument('--task-seconds', type=float, default=120)
    p.add_argument('--mixed', action='store_true', help='compare legacy/mixed pools with and without frozen predictors')
    p.add_argument('--budgets', type=int, nargs='+', choices=(8,16,32,64), default=(8,16,32))
    p = sub.add_parser('run')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--hours', type=float, default=2)
    p.add_argument('--max-tasks', type=int, default=0)
    p = sub.add_parser('worker')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--task', required=True)
    p = sub.add_parser('summarize')
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command=='plan': plan(args)
    elif args.command=='run': run(args)
    elif args.command=='worker': sys.exit(worker(args.output, args.task, args.device))
    else: summarize(args.output)


if __name__=='__main__':
    main()
