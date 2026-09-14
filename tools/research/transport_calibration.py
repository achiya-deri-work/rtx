"""Historical training-support audit; never held-out model-quality evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(rows, scores):
    if not rows or len(rows)!=len(scores): raise ValueError('missing rows or scores')
    seen=set(); centers=defaultdict(list)
    for row,score in zip(rows,scores):
        key=(row['context'],row['config_id'])
        if key in seen: raise ValueError('duplicate context/configuration')
        seen.add(key)
        if (type(row['ms']) not in (int,float) or not math.isfinite(row['ms']) or
                row['ms']<=0 or not math.isfinite(float(score))):
            raise ValueError('invalid timing or prediction')
        centers[row['context']].append(math.log(row['ms']))
    centers={c:statistics.median(v) for c,v in centers.items()}
    groups=defaultdict(list)
    for row,score in zip(rows,scores):
        gemm=row['config']['gemm']
        regime=tuple(sorted(k for k,v in row['features'].items() if k.startswith('context.regime=') and v))
        key=(row['device'],row['shape_class'],regime,gemm['scale_layout'],gemm['scale_role'],gemm['epilogue'])
        groups[key].append((row,math.log(row['ms'])-centers[row['context']]-float(score)))
    result=[]
    for key,values in sorted(groups.items()):
        residuals=defaultdict(list)
        for row,error in values:residuals[row['context']].append(error)
        context_errors=[statistics.median(v) for v in residuals.values()]
        result.append(dict(device=key[0],shape_class=key[1],regime=key[2],transport=key[3:],
            rows=len(values),contexts=len(residuals),shapes=len({r['shape'] for r,_ in values}),
            context_median_log_residual=statistics.median(context_errors),
            context_median_abs_log_residual=statistics.median(map(abs,context_errors))))
    return result


def main():
    from .evidence_dataset import load
    from rtx._portable_predictor import load_predictor
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--predictor',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    predictor=load_predictor(args.predictor,research=True)
    digest=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest()
    if digest!=predictor.document['provenance']['dataset_manifest_sha256']:
        raise ValueError('dataset differs from predictor training provenance')
    rows=[r for r in load(args.dataset) if r['split']=='train' and r['family']=='mxfp8_prequant_fwd']
    scores=predictor.predict(rows,'trees')
    result=dict(schema=1,dataset_manifest_sha256=digest,
        predictor_sha256=hashlib.sha256(args.predictor.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_rows=len(rows),revisions=sorted({r['revision'] for r in rows}),
        runtime_contract_tag_rows=sum(any('runtime_search_space_revision' in k for k in r['features']) for r in rows),
        groups=summarize(rows,scores),runtime_promotion=False,
        limitation='In-sample training residuals, not generalization; context-centered log latency. '
                    'Positive residual means underpredicted cost. Each context has equal mass within a group.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as sink:json.dump(result,sink,indent=2,allow_nan=False)
    print('SAVED',args.output,'training rows',len(rows))


if __name__=='__main__':main()
