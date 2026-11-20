"""Audit and consolidate completed active-budget studies without CUDA access.

Preserves per-wave identities and initialization differences. Standalone phase
ratios are diagnostic and are never added to reconstruct whole-layer time.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.profile_moe_active_budget import validate_result,write


FEATURES=('active_prefix','bounded_gather','bounded_dy','bounded_xt',
          'forward_row_layout','forward_glu_engine','weight_preparation',
          'backward_preparation','input_preparation','gate_dw_engine','down_dw_engine')


def grouped(rows):
    result=[]
    for wave in sorted({r['wave'] for r in rows}):
        for layout in ('packed','flat'):
            for scope in ('all','main_b16_l1024'):
                selected=[r for r in rows if r['wave']==wave and r['layout']==layout
                    and (scope=='all' or r['samples']==16 and r['total_tokens']==16384)]
                if not selected:continue
                item=dict(wave=wave,layout=layout,scope=scope,count=len(selected))
                for key in ('speedup_checkpoint','speedup_compact_checkpoint','speedup_saved',
                            'speedup_frozen_experts','speedup_shared'):
                    values=[r[key] for r in selected]
                    item[key]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values),
                        above_1p2=sum(v>=1.2 for v in values))
                result.append(item)
    return result


def consolidate(roots,output):
    rows=[];identities=[];records={};expected_runtime=None
    for root in roots:
        manifest=json.loads((root/'manifest.json').read_text())
        summary=json.loads((root/'summary.json').read_text())
        if summary['incomplete']:raise ValueError(f'Incomplete study: {root}')
        if summary['identity']!=manifest['identity']:raise ValueError('Summary identity mismatch')
        runtime=manifest['identity']['runtime']
        if expected_runtime is None:expected_runtime=runtime
        if runtime!=expected_runtime:raise ValueError('Different production runtimes cannot be pooled')
        identities.append(dict(root=str(root),identity=manifest['identity'],settings=manifest['settings']))
        saved={r['id']:r for r in summary['rows']}
        if len(saved)!=len(manifest['contexts']):raise ValueError('Context coverage mismatch')
        for context in manifest['contexts']:
            path=root/context['id']/'result.json';raw=path.read_bytes();record=json.loads(raw)
            validate_result(record,context,manifest)
            replay=record['validation']['fresh_gradient_replay']
            for phase in ('mxfp8_step','mxfp8_backward','bf16_checkpoint_step','bf16_compact_checkpoint_step'):
                if len(replay[phase])<2:raise ValueError('Missing repeated-gradient validation')
                if any(v['max_nrms']>.006 and v['max_abs']>2e-7 for v in replay[phase]):
                    raise ValueError('Gradient replay mismatch')
            row=dict(saved[context['id']],wave=manifest.get('study','legacy'),
                root=str(root),record_path=str(path),record_sha256=hashlib.sha256(raw).hexdigest(),
                stable_experts=record.get('stable_experts',False),runtime=runtime)
            row['context_key']=f'{root.name}:{context["id"]}'
            if row['context_key'] in records:raise ValueError('Duplicate namespaced context')
            row['slot_amplification_model']=row['physical_capacity']/row['mean_real_rows_per_expert']
            row['router_to_whole_diagnostic_ratio']=row['router_step_ms']/row['mxfp8_step_ms']
            row['remaining_ms_for_1p2_compact']=row['mxfp8_step_ms']-row['bf16_compact_checkpoint_step_ms']/1.2
            prepared=record['expert_config'].get('prepared',{})
            for name in FEATURES:row['selected_'+name]=prepared.get(name)
            rows.append(row);records[row['context_key']]=record
    selected_features={wave:{layout:{name:dict(Counter(str(r['selected_'+name]) for r in rows
        if r['wave']==wave and r['layout']==layout)) for name in FEATURES}
        for layout in ('packed','flat')} for wave in sorted({r['wave'] for r in rows})}
    # Verify the intended shared-width-only controls instead of assuming their
    # routed weights remained fixed merely because all runs used one seed.
    shared_controls=[]
    for e in (16,32,64,128):
        for layout in ('packed','flat'):
            group=[r for r in rows if r['wave']=='fanout' and r['d_model']==384 and r['top_k']==4
                and r['num_experts']==e and r['samples']==16 and r['total_tokens']==16384
                and r['layout']==layout]
            if len(group)!=3:continue
            values=[records[r['context_key']] for r in group]
            def expert_hashes(p):return {k:v for k,v in p['identities']['parameters'].items()
                if k.startswith(('gate_up.','down.'))}
            same_experts=all(expert_hashes(p)==expert_hashes(values[0]) for p in values)
            same_inputs=all(all(p['identities'][k]==values[0]['identities'][k]
                for k in ('input','upstream')) for p in values)
            same_routes=all(p['identities']['routing']==values[0]['identities']['routing'] for p in values)
            shared_controls.append(dict(experts=e,layout=layout,same_experts=same_experts,
                same_inputs=same_inputs,same_routes=same_routes,contexts=[r['context_key'] for r in group]))
            if not same_experts or not same_inputs:raise ValueError('Shared-width control changed expert weights/input')
    payload=dict(context_count=len(rows),runtime=expected_runtime,identities=identities,groups=grouped(rows),
        selected_features=selected_features,shared_width_controls=shared_controls,rows=rows,
        worst_oracle_nrms=max(r['oracle_max_nrms'] for r in rows),
        maximum_live_growth_bytes=max(r['live_growth_bytes'] for r in rows),
        caveats=['102 context executions are not 102 independent random samples',
                 'Legacy/fanout expert initialization differs; do not treat them as matched numerical A/B',
                 'Standalone expert BF16 control is native layout, not compact layout',
                 'Standalone router/expert/shared scopes overlap and must not be summed',
                 'Selection counts show winners, not configurations admitted or tried',
                 'Slot amplification is shape-based, not measured issued FLOPs or DRAM traffic'])
    output.mkdir(parents=True,exist_ok=True);write(output/'summary.json',payload)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'contexts.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows),output/'contexts.parquet')
    print(json.dumps(dict(contexts=len(rows),runtime=expected_runtime,
        worst_oracle_nrms=payload['worst_oracle_nrms'],shared_width_controls=shared_controls)),flush=True)
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();consolidate([p.resolve() for p in args.roots],args.output.resolve())


if __name__=='__main__':main()
