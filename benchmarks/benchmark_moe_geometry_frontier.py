"""Focused whole-expert search over existing GEMM geometry/transport templates.

All coordinates come from the shared adapter. No per-SKU register values or
numerical policy changes. Use the ordinary shared engine, oracle and residual
store, retaining incumbent input/dY/recompute/reduction policies.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from itertools import product


def candidates(adapter,stages=('down','gate_dx'),seed=0):
    groups={}
    for stage in stages:
        for transport in adapter.axes[f'prepared.{stage}.transport']:
            bucket=[]
            for geometry in adapter.axes[f'prepared.{stage}.geometry']:
                for depth in adapter.axes[f'prepared.{stage}.stages'] if transport=='tma' else (1,):
                    value=adapter.update_fn(adapter.initial_config,f'prepared.{stage}.geometry',geometry)
                    value=adapter.update_fn(value,f'prepared.{stage}.stages',depth)
                    value=adapter.update_fn(value,f'prepared.{stage}.transport',transport)
                    if adapter.rejection(value) is None:bucket.append(value)
            random.Random(seed).shuffle(bucket)
            groups[(stage,transport)]=bucket
    seen={adapter.config_id(adapter.initial_config)};result=[]
    while any(groups.values()):
        for bucket in groups.values():
            if bucket:
                value=bucket.pop();key=adapter.config_id(value)
                if key not in seen:seen.add(key);result.append(value)
    return result


def pipeline_candidates(adapter,stages):
    result=[];seen={adapter.config_id(adapter.initial_config)}
    for stage in stages:
        keys=[f'prepared.{stage}.{name}' for name in
              ('stages','scale_role','scale_recycle','scale_schedule','mma_schedule')]
        for values in product(*(adapter.axes[key] for key in keys)):
            value=adapter.initial_config
            for key,selection in zip(keys,values):value=adapter.update_fn(value,key,selection)
            identifier=adapter.config_id(value)
            if identifier not in seen and adapter.rejection(value) is None:
                seen.add(identifier);result.append(value)
        for waves,tiles,locality in product(adapter.axes[f'prepared.{stage}.persistent_waves'],
                adapter.axes[f'prepared.{stage}.tiles_per_cta'],adapter.axes[f'prepared.{stage}.tile_locality']):
            value=adapter.initial_config
            for name,selection in (('persistent_waves',waves),('tiles_per_cta',tiles),('tile_locality',locality)):
                value=adapter.update_fn(value,f'prepared.{stage}.{name}',selection)
            identifier=adapter.config_id(value)
            if identifier not in seen and adapter.rejection(value) is None:
                seen.add(identifier);result.append(value)
    return result


def schedule_candidates(adapter,stages,seed=0):
    """Coupled geometry/load/buffering/store panel, preserving quantization.

    This is a bounded research panel of existing templates, not a runtime
    restriction. Balance candidate order across stages and transport families.
    """
    groups={};seen={adapter.config_id(adapter.initial_config)}
    for stage in stages:
        transport_key=f'prepared.{stage}.transport'
        if transport_key not in adapter.axes:continue
        # Cover the principal tall/thin and wide tile regions without
        # multiplying every warp-layout alias in this initial panel.
        geometries=[g for g in adapter.axes[f'prepared.{stage}.geometry']
                    if g[3]==(8 if g[0]==256 else 2)]
        for transport in adapter.axes[transport_key]:
            for depth in adapter.axes[f'prepared.{stage}.stages'] if transport=='tma' else (1,):
                for geometry in geometries:
                    bucket=groups.setdefault((stage,transport,depth,geometry[2]),[])
                    for output in (('direct',1,1),('hybrid',4,1),('hybrid',4,2)):
                        value=adapter.initial_config
                        for key,selection in ((f'prepared.{stage}.geometry',geometry),
                            (f'prepared.{stage}.stages',depth),(transport_key,transport),
                            (f'prepared.{stage}.epilogue_transport',output)):
                            value=adapter.update_fn(value,key,selection)
                        identifier=adapter.config_id(value)
                        if identifier not in seen and adapter.rejection(value) is None:
                            seen.add(identifier);bucket.append(value)
    for bucket in groups.values():random.Random(seed).shuffle(bucket)
    result=[]
    while any(groups.values()):
        for bucket in groups.values():
            if bucket:result.append(bucket.pop())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True,type=Path)
    parser.add_argument('--fixture',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--trials',type=int,default=24)
    parser.add_argument('--max-seconds',type=float,default=300)
    parser.add_argument('--stages',nargs='+',default=['down','gate_dx'])
    parser.add_argument('--mode',choices=('geometry','pipeline','schedule'),default='geometry')
    args=parser.parse_args()
    import torch
    from rtx.autotune import AutotunePolicy
    from rtx.autotune.effort import make_effort_autotuner
    from rtx.autotune.moe import MoEEvaluator,make_moe_adapter,source_revision
    from rtx.autotune.store import ResidualTuningStore
    from rtx.moe import MXFP8MoEConfig
    from rtx.autotune.hardware import static_device_profile
    from benchmarks.study_moe_priors import atomic_json

    saved=json.loads(args.result.read_text())
    data=torch.load(args.fixture,map_location='cuda',weights_only=True)
    values=tuple(data[k] for k in ('x','gu','down','ids','weights','inverse'))
    x,gu,down,ids,weights,inverse=values
    shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
    policy=AutotunePolicy.preset('balanced',max_trials=args.trials,max_seconds=args.max_seconds,
        pool_size=128,structural_proposals=256,candidate_mix='mixed',confirmation_repeats=0,
        samples=7,calls_per_sample=1,finalist_rounds=0)
    # CUDA nodes are initialized/warmed/captured in one non-default stream.
    with torch.cuda.stream(torch.cuda.Stream()):
        evaluator=MoEEvaluator(values,'train',policy)
        tags={'experiment':'geometry_transport_frontier',
              'mode':args.mode,'initial':saved['expert_config'],
              'source':source_revision(),'fixture':str(args.fixture.resolve())}
        if saved.get('layout')=='packed':
            tags['batch_contract']={'layout':'independent_varlen','sequences':len(saved['lengths']),
                                    'segment_alignment':32}
        adapter=make_moe_adapter(shape,evaluator.evaluate,initial=MXFP8MoEConfig.from_dict(saved['expert_config']),
            device=static_device_profile(x.device),tags=tags)
        generator={'geometry':candidates,'pipeline':pipeline_candidates,'schedule':schedule_candidates}[args.mode]
        pool=generator(adapter,args.stages)
        if not pool:raise ValueError('No legal geometry/transport alternatives')
        # One coupled template coordinate avoids illegal intermediate states;
        # selection remains the common UCB/global/local search machinery.
        indexed={adapter.config_id(value):value for value in pool}
        focused=replace(adapter,axes={'gemm_template':tuple(indexed)},
                        update_fn=lambda current,name,value:indexed[value])
        store=ResidualTuningStore(args.output/'residuals')
        result=make_effort_autotuner(focused,store,policy,progress=lambda s:print(s,flush=True)).tune()
        atomic_json(args.output/'result.json',dict(shape=shape,source=source_revision(),
            initial=adapter.serialize(adapter.initial_config),config=adapter.serialize(result.config),
            median_ms=result.median_ms,evaluated_this_launch=result.evaluated_trials,
            candidate_count=len(pool),candidate_stages=args.stages,
            numerical_policy='unchanged; shared FP64 expert oracle; evaluator seed 0 upstream',
            qualification='search result only; paired complete-layer promotion still required'))


if __name__=='__main__':main()
