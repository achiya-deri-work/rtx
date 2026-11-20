"""CPU-only log health scan; optionally quarantine without rewriting evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtx.autotune.collection_health import HEALTH_FILENAME, first_device_fault


def audit_logs(root):
    logs=sorted(root.rglob('*.log'))
    faults=[fault for path in logs if (fault:=first_device_fault(path)) is not None]
    return dict(schema=1,root=str(root.resolve()),evidence_eligible=not faults,
                logs_scanned=len(logs),faulted_logs=len(faults),faults=faults,
                scope='fatal device markers in logs; not a numerical/coverage qualification')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',nargs='+',type=Path)
    parser.add_argument('--quarantine',action='store_true',
                        help='Write negative health markers only; never bless a previously quarantined bundle')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results=[]
    for root in args.paths:
        if not root.is_dir():raise ValueError(f'Not a collection directory: {root}')
        report=audit_logs(root)
        if args.quarantine and not report['evidence_eligible']:
            target=root/HEALTH_FILENAME
            if target.exists():
                # Preserve an earlier incident marker instead of replacing its
                # reason/provenance with a later diagnostic scan.
                old=json.loads(target.read_text())
                if old.get('evidence_eligible') is False:
                    report['existing_quarantine_preserved']=True
                    results.append(report)
                    continue
            temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps({**report,'reason':'fatal CUDA faults in collection logs'},indent=2)+'\n')
            temporary.replace(target)
        results.append(report)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([dict(root=r['root'],faulted_logs=r['faulted_logs'],
                           evidence_eligible=r['evidence_eligible']) for r in results]),flush=True)


if __name__=='__main__':main()
