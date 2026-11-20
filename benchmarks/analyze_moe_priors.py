"""CPU-only candidate-budget diagnostics and portable MoE residual export.

Curves are retrospective against the best *observed* eligible configuration,
not claims about global optima. Candidate elapsed time excludes orchestrator
proposal/setup overhead; full cold-start wall time stays in study result JSON.
No runtime winner or pruning rule is installed by this analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from rtx.autotune.pretrained import load_offline_observations, _device_key, _shape_group_key


def curve(rows,budgets=(8,16,32,64)):
    ordered=sorted(rows,key=lambda r:(r.finished_at,r.sequence,r.observation_id))
    unique=[];seen=set()
    for row in ordered:
        if row.config_id not in seen:
            seen.add(row.config_id);unique.append(row)
    eligible=[r for r in unique if r.outcome.timing_eligible]
    best=min((r.score for r in eligible),default=None)
    elapsed=0.;first_valid=None;failed_seconds=0.;points=[]
    incumbent=None
    for index,row in enumerate(unique,1):
        elapsed+=max(0,row.elapsed_s)
        if row.outcome.timing_eligible:
            if first_valid is None:first_valid=elapsed
            incumbent=row.score if incumbent is None else min(incumbent,row.score)
        elif row.outcome.status in ('compile_error','runtime_error','correctness_error'):
            failed_seconds+=max(0,row.elapsed_s)
        if index in budgets:
            points.append(dict(trials=index,best_ms=incumbent,
                regret=None if incumbent is None or best is None else incumbent/best-1.,
                measured_candidate_seconds=elapsed))
    return dict(unique_trials=len(unique),eligible_trials=len(eligible),best_observed_ms=best,
        first_timing_eligible_candidate_seconds=first_valid,
        failed_candidate_seconds=failed_seconds,measured_candidate_seconds=elapsed,
        statuses=dict(Counter(row.outcome.status for row in unique)),curve=points)


def analyze(paths,output,*,parquet=True,include_shared=False):
    rows,loading=load_offline_observations(paths)
    # This exporter is intentionally scoped: passing a large shared archive
    # must not turn an MoE study into an unrelated all-kernels collection.
    rows=[r for r in rows if 'moe' in r.family or r.family in
          ('mxfp8_balanced_experts','balanced_routing','balanced_varlen_routing','balanced_router_affinity')
          or (include_shared and r.family=='mxfp8_recompute_mlp')]
    by_context=defaultdict(list)
    for row in rows:by_context[row.context_id].append(row)
    contexts=[]
    for key,items in sorted(by_context.items()):
        contexts.append(dict(context=key,family=items[0].family,device=_device_key(items[0]),
            shape_group=list(_shape_group_key(items[0])),**curve(items)))
    output.mkdir(parents=True,exist_ok=True)
    exported=[]
    for row in rows:
        exported.append(dict(observation_id=row.observation_id,context_id=row.context_id,
            family=row.family,kernel_revision=row.kernel_revision,device=_device_key(row),
            sequence=row.sequence,config_id=row.config_id,strategy=row.strategy,
            status=row.outcome.status,timing_eligible=row.outcome.timing_eligible,
            median_ms=row.outcome.median_ms,setup_ms=row.outcome.compile_ms,
            elapsed_s=row.elapsed_s,error=row.outcome.error,
            features_json=json.dumps(row.features,sort_keys=True),
            config_json=json.dumps(row.serialized_config,sort_keys=True)))
    if exported:
        with (output/'observations.csv').open('w',newline='') as sink:
            writer=csv.DictWriter(sink,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
        if parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq.write_table(pa.Table.from_pylist(exported),output/'observations.parquet',compression='zstd')
    report=dict(schema=1,deployment_enabled=False,include_shared=include_shared,loading=loading,
        observations=len(rows),contexts=contexts,
        devices=sorted({_device_key(row) for row in rows}),
        caveats=['retrospective regret is against best observed, not global optimum',
            'missing budget points mean insufficient actual trials, not zero regret',
            'candidate elapsed excludes orchestrator and whole-layer compilation overhead',
            'shared loading can skip unreadable records; run rtx-autotune audit first; residuals remain authoritative',
            'no heuristic, learned prior or runtime winner is automatically promoted'])
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# MoE tuning evidence','',f'{len(rows)} observations, {len(contexts)} contexts, {len(report["devices"])} device profiles.',
        '', 'Regret is retrospective against each context\'s best observed eligible configuration.',
        'Candidate wall cost excludes proposal/whole-layer compilation overhead. No deployment promotion.',
        '', '| Family | Device | Trials | First valid candidate seconds | Failed candidate seconds |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in contexts:
        first=row['first_timing_eligible_candidate_seconds']
        lines.append(f'| {row["family"]} | {row["device"]} | {row["unique_trials"]} | {first if first is not None else "none"} | {row["failed_candidate_seconds"]:.3f} |')
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-parquet',action='store_true')
    p.add_argument('--include-shared',action='store_true')
    args=p.parse_args();report=analyze(args.paths,args.output,parquet=not args.no_parquet,include_shared=args.include_shared)
    print(json.dumps(dict(observations=report['observations'],contexts=len(report['contexts']),
        devices=report['devices'],output=str(args.output))))


if __name__=='__main__':main()
