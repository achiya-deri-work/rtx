"""Research CLI for the portable structural sampler."""
from collections import Counter
import json
import random
from rtx._sampling import (flatten, distance, StructuralSamplingPolicy, raw_portfolio,
                           select_portfolio, pool_diagnostics)


def main():
    """Prepare a raw cross-SKU linear panel using a saved hardware context."""
    import argparse
    from pathlib import Path
    from tools.predictor_study import atomic
    from rtx.autotune import adapters
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.configs.nvfp4 import NVFP4Problem
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-result',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--pool-size',type=int,default=128)
    parser.add_argument('--proposals',type=int,default=2048)
    args=parser.parse_args()
    saved=json.loads(Path(args.context_result).read_text())
    context=saved['context'];family=context['family'];w=context['workload']
    names={'mxfp8_prequant_fwd':'make_mxfp8_prequant_adapter','mxfp8_bwd':'make_mxfp8_bwd_adapter',
        'nvfp4_dynamic_fwd':'make_nvfp4_dynamic_adapter','nvfp4_delayed_fwd':'make_nvfp4_delayed_adapter',
        'nvfp4_jit_row_region_fwd':'make_nvfp4_jit_row_region_adapter'}
    if family not in names:
        raise ValueError('CLI supports dynamic linear families; use raw_portfolio(adapter, ...) for MLP/GLU')
    problem=(NVFP4Problem if family.startswith('nvfp4') else MXFP8Problem)(w['m'],w['n'],w['k'])
    adapter=getattr(adapters,names[family])(problem,lambda c:None,device=saved['hardware'],regime=context['regime'])
    raw=raw_portfolio(adapter,StructuralSamplingPolicy(args.pool_size,args.proposals))
    selected,audit=select_portfolio(adapter,raw,args.pool_size)
    base=flatten(dict(adapter.serialize(adapter.initial_config)))
    walk=adapter.sample(random.Random(1729),args.pool_size,[adapter.initial_config])
    def stats(configs):
        ds=[distance(base,flatten(dict(adapter.serialize(c)))) for c in configs]
        return dict(count=len(ds),mean_distance_from_initial=sum(ds)/max(1,len(ds)),max_distance=max(ds,default=0))
    atomic(args.output,dict(schema=1,source_context=context,raw_panel=raw,local_static_audit=audit,
        selected_ids=[adapter.config_id(c) for c in selected],structural=stats(selected),local_walk=stats(walk),
        note='Raw panel is the cross-device comparison unit. Local accepted subset is not a replacement for the raw panel. No GPU benchmarks performed.'))
    print(json.dumps(dict(structural=stats(selected),local_walk=stats(walk),statuses=dict(Counter(r['status'] for r in audit))),indent=2))


if __name__=='__main__':main()
