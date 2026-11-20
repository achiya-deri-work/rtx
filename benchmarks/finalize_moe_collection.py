"""Finalize only the qualified prefix of a stopped, audit-gated MoE study.

Preserve raw records in independently checksummed JSONL partitions and export
CSV/Parquet including shared MLP observations. Interrupted later visits remain
in the source study for resume, but cannot leak into this immutable dataset.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

from benchmarks.collect_moe_clean import validate_result
from benchmarks.study_moe_priors import atomic_json
from benchmarks.analyze_moe_priors import analyze
from rtx.autotune.audit import audit_bundles
from rtx.autotune.collection_health import assert_healthy_path
from rtx.autotune.pretrained import _observation_documents


def finished_before(record,cutoff):
    timestamp=datetime.fromisoformat(record['finished_at'].replace('Z','+00:00'))
    if timestamp.tzinfo is None:raise ValueError('Naive observation timestamp cannot define a qualified fence')
    return timestamp.timestamp()<=cutoff


def finalize(root,output,partition_rows=256):
    if partition_rows<=0:raise ValueError('Partition rows must be positive')
    root=root.resolve();output=output.resolve();study=root/'study'
    if output.exists():raise ValueError('Finalized output must be a fresh directory')
    assert_healthy_path(study)
    source_audit=audit_bundles([study])
    if not source_audit['ok'] or source_audit['summary']['malformed_tails']:
        raise ValueError('Source journals must pass structural audit before finalization')
    checkpoints=sorted((root/'checks').glob('visit-*/summary.json'))
    if not checkpoints:raise ValueError('No qualified visit checkpoints')
    latest=checkpoints[-1];summary=json.loads(latest.read_text())
    if json.loads((root/'latest.json').read_text())!=summary:raise ValueError('Latest checkpoint identity differs')
    cutoff=latest.stat().st_mtime
    for path in checkpoints:
        point=json.loads(path.read_text())
        validate_result(study/point['context']/f'effort-{point["budget"]}'/'result.json')
        check=json.loads(path.with_name('journal-audit.json').read_text())
        health=json.loads(path.with_name('qualification.json').read_text())
        if not check['ok'] or health['kernel_xids']:raise ValueError('Qualified checkpoint has failed gates')
    # The narrower MoE-only export is an independent identity fence. Shared
    # observations use the same trustworthy completion-time cutoff.
    with (root/'exports'/latest.parent.name/'observations.csv').open() as source:
        expected={(r['context_id'],r['observation_id']) for r in csv.DictReader(source)}
    rows={};lineage=[];excluded=0
    for source,record in _observation_documents([study]):
        if not finished_before(record,cutoff):excluded+=1;continue
        key=(record['context_id'],record['observation_id'])
        if key in rows:continue
        rows[key]=record;lineage.append(dict(context_id=key[0],observation_id=key[1],source=source))
    primary={key for key,row in rows.items() if row['family']!='mxfp8_recompute_mlp'}
    if primary!=expected:raise ValueError('Qualified prefix does not exactly match the audited MoE export')
    output.mkdir(parents=True)
    atomic_json(output/'collection_health.json',dict(evidence_eligible=False,reason='Finalization in progress'))
    shutil.copy2(study/'machine.json',output/'machine.json')
    files={}
    ordered=sorted(rows.values(),key=lambda r:(r['finished_at'],r['context_id'],r['observation_id']))
    for offset in range(0,len(ordered),partition_rows):
        path=output/'residuals'/f'part-{offset//partition_rows:03d}'/'observations.jsonl'
        path.parent.mkdir(parents=True)
        with path.open('x') as sink:
            for row in ordered[offset:offset+partition_rows]:sink.write(json.dumps(row,allow_nan=False)+'\n')
        files[str(path.relative_to(output))]=hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(output/'lineage.json',dict(observations=lineage))
    manifest=dict(schema=1,type='qualified_moe_prefix',source=str(root),
        source_manifest_sha256=hashlib.sha256((study/'manifest.json').read_bytes()).hexdigest(),
        checkpoint=str(latest),checkpoint_sha256=hashlib.sha256(latest.read_bytes()).hexdigest(),
        cutoff_utc=datetime.fromtimestamp(cutoff,timezone.utc).isoformat(),
        qualified_visits=len(checkpoints),observations=len(rows),primary_observations=len(primary),
        shared_observations=len(rows)-len(primary),excluded_after_checkpoint=excluded,
        partition_rows=partition_rows,files_sha256=files,
        caveat='Qualified collection prefix, not proof every candidate is correct or stationary; failure labels retained')
    atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'collection_health.json',dict(evidence_eligible=True,
        reason='All qualified-prefix partitions and identity checks complete'))
    audit=audit_bundles([output]);atomic_json(output/'audit.json',audit)
    if not audit['ok']:
        atomic_json(output/'collection_health.json',dict(evidence_eligible=False,reason='Finalized structural audit failed'))
        raise ValueError('Finalized dataset failed structural audit')
    analyze([output],output/'exports',include_shared=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(finalize(args.root,args.output)),flush=True)


if __name__=='__main__':main()
