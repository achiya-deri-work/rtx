"""Verify local sklearn/portable prediction equivalence, without timing labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    import joblib
    import numpy as np
    from rtx._portable_predictor import load_predictor
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--predictor',required=True,type=Path)
    parser.add_argument('--pools',nargs='+',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    model=joblib.load(args.model)  # Explicitly our locally produced fit only.
    predictor=load_predictor(args.predictor,research=True)
    reports=[]
    for path in args.pools:
        pool=json.loads(path.read_text());context=pool['context']
        rows=[dict(family=context['family'],revision=context['kernel_revision'],features=c['features'])
              for c in pool['configs']]
        expected=model.predict(rows)
        started=time.monotonic();actual=predictor.predict(rows,'trees');elapsed=time.monotonic()-started
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
        reports.append(dict(pool_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(rows),
            max_abs=float(np.max(np.abs(actual-expected))),portable_predict_s=elapsed))
    with args.output.open('x') as f:json.dump(dict(reports=reports,
        predictor_sha256=hashlib.sha256(args.predictor.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        scope='Export equivalence only; no timing labels loaded and no quality promotion'),f,indent=2)
    print('VERIFIED',sum(r['rows'] for r in reports),'rows',flush=True)


if __name__=='__main__':main()
