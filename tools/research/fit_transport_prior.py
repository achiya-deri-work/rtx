"""Fit the historically selected MXFP8 interaction prior, research-only."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    import joblib
    from threadpoolctl import threadpool_limits
    from .evidence_dataset import load
    from .models import FittedCandidate,REGISTRY
    from .export_portfolio import export
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[r for r in load(args.dataset) if r['split']=='train' and r['family']=='mxfp8_prequant_fwd']
    plan=dict(dataset=str(args.dataset),training_rows=len(rows),family='mxfp8_prequant_fwd',
        model='extra_trees',view='interactions',variant=0,balanced=False,threads=4,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
        runtime_promotion=False)
    with (args.output/'plan.json').open('x') as f:json.dump(plan,f,indent=2)
    with threadpool_limits(limits=4):
        model=FittedCandidate(REGISTRY['extra_trees'],'interactions',0,4)
        started=time.monotonic();model.fit(rows,balance_contexts=False)
        fit_s=time.monotonic()-started
        joblib.dump(model,args.output/'model.joblib',compress=3)
        export(args.dataset,args.output/'model.joblib',args.output/'predictor.json.gz')
    with (args.output/'fit.json').open('x') as f:json.dump(dict(fit_s=fit_s,features=model.feature_count),f)
    print('EXPORTED',args.output,'fit_s',fit_s,flush=True)


if __name__=='__main__':main()
