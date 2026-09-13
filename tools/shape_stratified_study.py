"""Freeze category-balanced, whole-shape-held-out first-hit policy comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

from tools.predictor_study import atomic, digest, source_digest, FAMILIES, setup_quarantines


def tasks_for_manifest(manifest, replicates=1):
    if type(replicates) is not int or replicates < 1:
        raise ValueError('replicates must be positive')
    shapes, families = manifest['shapes'], manifest['families']
    if not shapes or not families or len(set(families)) != len(families):
        raise ValueError('need shapes and unique families')
    if set(families)-set(FAMILIES) or manifest.get('regime') != 'hot':
        raise ValueError('unsupported family or cache regime (MLP/GLU currently hot only)')
    if len({s['category'] for s in shapes}) != len(shapes):
        raise ValueError('one representative per named stratum required')
    for s in shapes:
        if any(type(s[k]) is not int or s[k] <= 0 for k in ('m','n','k')):
            raise ValueError('shape dimensions must be positive integers')
    treatments = manifest['treatments']
    if len(set(treatments)) != len(treatments) or set(treatments) != {'existing','mixed','residual_mlp','mixed_residual_mlp'}:
        raise ValueError('require the four matched legacy/mixed treatments')
    budgets = manifest['budgets']
    if not budgets or budgets != sorted(set(budgets)) or any(b not in (8,16,32,64) for b in budgets):
        raise ValueError('budgets must be increasing unique 8/16/32/64 values')
    tasks = []
    for budget in budgets:
        for rep in range(replicates):
            # Each sweep visits every stratum. The diagonal family rotation
            # covers every (shape,family) exactly once over all sweeps.
            for sweep in range(len(families)):
                for si, entry in enumerate(shapes):
                    family = families[(si+sweep+rep) % len(families)]
                    shape = {k:entry[k] for k in ('m','n','k')}
                    offset = (si+sweep+rep) % len(treatments)
                    order = treatments[offset:]+treatments[:offset]
                    for treatment in order:
                        task = dict(family=family, shape=shape, category=entry['category'],
                            shape_group=digest(shape)[:20], regime='hot', budget=budget,
                            replicate=rep, treatment=treatment, seed=20260918+rep)
                        task['id'] = digest(task)[:20]
                        tasks.append(task)
    return tasks


def prepare(args):
    output = Path(args.output)
    if (output/'plan.json').exists():
        raise ValueError('plan already exists; resume it instead')
    manifest = json.loads(Path(args.manifest).read_text())
    tasks = tasks_for_manifest(manifest, args.replicates)
    audit_path = Path(args.audit_plan)
    audited = json.loads(audit_path.read_text())
    artifact = Path(args.artifact).resolve()
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if sha != audited['artifact_sha256']:
        raise ValueError('geometry audit must correspond to the same frozen predictor')
    # Reuse the frozen, exhaustive previous inventory. Additionally exclude
    # both previous evaluation shapes, not just the model-training shapes.
    seen = {tuple(x) for x in audited['geometry_audit']['excluded_m_k_pairs']}
    seen.update((t['shape']['m'],t['shape']['k']) for t in audited['tasks'])
    overlap = [s for s in manifest['shapes'] if (s['m'],s['k']) in seen]
    if overlap:
        raise ValueError(f'whole (M,K) group overlaps prior inventory: {overlap}')
    blocked = {}
    if args.quarantine_from:
        parent = json.loads((Path(args.quarantine_from)/'plan.json').read_text())
        blocked = setup_quarantines(args.quarantine_from,
                                    {**parent, 'quarantine_setup_failures':True})
    atomic(output/'plan.json', dict(schema=1, purpose='shape_stratified_evaluation_only',
        quarantine_setup_failures=True, blocked_contexts=blocked,
        source_sha256=source_digest(), artifact=str(artifact), artifact_sha256=sha,
        manifest=manifest, treatments=manifest['treatments'], tasks=tasks,
        timing=dict(warmup=3,samples=7,calls_per_sample=5), task_seconds=120,
        geometry_audit=dict(parent_plan=str(audit_path),
            parent_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            excluded_m_k_pairs=sorted(seen), records=audited['geometry_audit']['records']),
        cache_scope='fresh process; shared persistent compiler caches',
        training_admission=False,
        limitations=['Hot-cache only; MLP/GLU rotation is not implemented.',
            'Category names describe workloads, not measured occupancy.',
            'Unsupported harness shapes remain failed cells, not speedup measurements.',
            'Independent budget cells; this is not cumulative refinement timing.']))
    print(f'PLAN {len(manifest["shapes"])} strata, {len(tasks)} cells, output={output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', default='autotune_manifests/shape_stratified_v1.json')
    p.add_argument('--audit-plan', default='autotune_datasets/mixed_pool_local_v1/plan.json')
    p.add_argument('--artifact', default='autotune_priors/tournament_predictors_v1.json.gz')
    p.add_argument('--replicates', type=int, default=1)
    p.add_argument('--quarantine-from', help='retain exact family/shape setup failures from an earlier pilot; no latency import')
    prepare(p.parse_args())
