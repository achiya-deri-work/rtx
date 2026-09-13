"""Build auditable selection manifests without rewriting raw residuals.

Parquet views reference original observations by source/context/observation ID.
Failures stay in feasibility data; explicitly unstable timings are excluded
from the latency view. Missing stationarity metadata is reported, not certified.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
from rtx.autotune.pretrained import _observation_documents
from rtx.autotune.first_hit_study import atomic_json, audit_anchors


def analyze(paths,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    selected=[];summary=[];seen=set()
    for path in paths:
        path=Path(path)
        jsonl_records=0
        def check_jsonl(stream,label):
            nonlocal jsonl_records
            for number,line in enumerate(stream,1):
                if not line.strip():continue
                try:json.loads(line)
                except (ValueError,UnicodeDecodeError) as exc:
                    raise ValueError(f'Malformed JSONL: {label}:{number}') from exc
                jsonl_records+=1
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as archive:
                corrupt=archive.testzip()
                if corrupt:raise ValueError(f'CRC failure: {path}:{corrupt}')
                for name in archive.namelist():
                    if name.endswith('.jsonl'):
                        with archive.open(name) as stream:check_jsonl(stream,f'{path}!{name}')
        else:
            for file in path.rglob('*.jsonl'):
                with file.open('rb') as stream:check_jsonl(stream,str(file))
        contexts=defaultdict(Counter);status=Counter();quality=Counter();duplicates=0
        for source,r in _observation_documents([path]):
            identity=(r['context_id'],r['observation_id'])
            if identity in seen:
                duplicates+=1;continue
            seen.add(identity)
            o=r['outcome'];state=o['status'];f=r['features']
            stationary=o.get('metadata',{}).get('sampling',{}).get('collection',{}).get('stationary')
            eligible=state=='ok' and stationary is not False
            status[state]+=1;contexts[r['context_id']][state]+=1
            if state=='ok':quality[str(stationary)]+=1
            sku=next((k.split('=',1)[1] for k in f if k.startswith('context.device.sku.sku_family=')),'unknown')
            selected.append(dict(source=source,observation_id=r['observation_id'],
                context_id=r['context_id'],family=r['family'],config_id=r['config_id'],
                kernel_revision=r['kernel_revision'],sku=sku,status=state,
                median_ms=o.get('median_ms'),latency_eligible=eligible,
                stationarity='unreported' if stationary is None else str(stationary),
                error=o.get('error'),quant_x_json=json.dumps(r['config'].get('quant_x'),sort_keys=True)))
        row=dict(source=str(path),contexts=len(contexts),observations=sum(status.values()),
            parsed_jsonl_records=jsonl_records,
            successful=status['ok'],unstable_successes=quality['False'],
            unreported_stationarity=quality['None'],duplicates_skipped=duplicates,
            minimum_context_observations=min(map(lambda c:sum(c.values()),contexts.values()),default=0),
            statuses=json.dumps(dict(status),sort_keys=True))
        summary.append(row);print(row,flush=True)
    table=pa.Table.from_pylist(selected)
    pq.write_table(table,root/'feasibility_selection.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist([r for r in selected if r['latency_eligible']]),
                  root/'latency_selection.parquet',compression='zstd')
    with (root/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    atomic_json(root/'audit.json',dict(sources=summary,anchors=audit_anchors(paths),
        policy='exclude explicitly unstable successes from latency; retain all outcomes for feasibility',
        limitations=['Selection manifests reference immutable raw observations; they are not runtime winners',
                     'Unreported stationarity is retained, not certified stable']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.paths,a.output)


if __name__=='__main__':main()
