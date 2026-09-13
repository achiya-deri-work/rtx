"""Offline structural portfolios; no model labels, CUDA, or runtime defaults.

Freeze raw proposals once, then replay that identical payload panel on each SKU.
Unlike the runtime random walk, intermediate states need not be hardware-legal.
Only complete configurations are validated. Coupled adapter update operators
remain available; normalize-time exceptions are recorded, never silently fixed.
"""
from collections import Counter
from dataclasses import dataclass
import json
import math
import random


def flatten(value,prefix=''):
    result={}
    for key,item in value.items():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(item,dict):result.update(flatten(item,name))
        else:result[name]=json.dumps(item,sort_keys=True)
    return result


def distance(a,b):
    keys=a.keys()|b.keys()
    return sum(a.get(k)!=b.get(k) for k in keys)/max(1,len(keys))


@dataclass(frozen=True)
class StructuralSamplingPolicy:
    pool_size:int=128
    proposals:int=1024
    seed:int=1729
    def __post_init__(self):
        if type(self.pool_size) is not int or self.pool_size<1 or type(self.proposals) is not int or self.proposals<self.pool_size:
            raise ValueError('require proposals >= pool_size >= 1')


def raw_portfolio(adapter,policy=StructuralSamplingPolicy(),*,anchors=()):
    """Mix full, deep and 2–4-coordinate assignments; balanced axis values.

    This is not uniform sampling of the legal Cartesian space. It is a bounded
    structural generator independent of incumbent timings. The returned raw
    payloads are the cross-device panel; devices must not regenerate them.
    """
    rng=random.Random(policy.seed)
    bases=[adapter.initial_config,*anchors]
    axes=list(adapter.axes)
    rows=[];seen=set();failures=Counter()
    values={k:list(adapter.axes[k]) for k in axes}
    for vals in values.values():rng.shuffle(vals)
    def add(c,kind,changed):
        payload=dict(adapter.serialize(c));key=json.dumps(payload,sort_keys=True)
        if key not in seen:
            seen.add(key);rows.append(dict(config=payload,source=kind,assigned_coordinates=changed))
    for c in bases:add(c,'anchor',[])
    for i in range(policy.proposals):
        c=bases[i%len(bases)]
        order=list(axes);rng.shuffle(order)
        if i%4 and order:
            maximum=len(order) if i%4==1 else min(4,len(order))
            order=order[:rng.randint(min(2,len(order)),maximum)]
        failed=False
        for j,key in enumerate(order):
            if not values[key]:continue
            value=values[key][(i+j)%len(values[key])]
            try:c=adapter.update_fn(c,key,value)
            except (ValueError,TypeError,KeyError) as exc:
                failures[type(exc).__name__]+=1;failed=True;break
        if not failed:add(c,'full_assignment' if i%4==0 else 'coupled_assignment',order)
    return dict(schema=1,seed=policy.seed,proposals=rows,construction_failures=dict(failures),
                uses_measured_labels=False)


def select_portfolio(adapter,raw,limit,*,effective=lambda x:x):
    """Static target validation then max-min diversity of effective configs.

    All rejections are returned with their payload. The first accepted anchor
    is retained; subsequent points maximize minimum categorical distance.
    """
    if limit<1:raise ValueError('limit must be positive')
    candidates=[];audit=[];seen=set()
    for row in raw['proposals']:
        try:
            c=adapter.deserialize(row['config'])
            rejection=adapter.rejection(c)
            if rejection is not None:
                audit.append(dict(row,status='rejected',reason=str(rejection)));continue
            view=effective(dict(adapter.serialize(c)))
            key=json.dumps(view,sort_keys=True)
            if key in seen:
                audit.append(dict(row,status='effective_duplicate'));continue
            seen.add(key);candidates.append((c,flatten(view),row))
        except (ValueError,TypeError,KeyError) as exc:
            audit.append(dict(row,status='construction_error',reason=str(exc)))
    if not candidates:return [],audit
    selected=[candidates.pop(0)]
    minimum=[distance(x[1],selected[0][1]) for x in candidates]
    while candidates and len(selected)<limit:
        index=max(range(len(candidates)),key=lambda i:minimum[i])
        item=candidates.pop(index);minimum.pop(index);selected.append(item)
        minimum=[min(d,distance(x[1],item[1])) for d,x in zip(minimum,candidates)]
    for _,_,row in selected:audit.append(dict(row,status='selected'))
    for _,_,row in candidates:audit.append(dict(row,status='diversity_not_selected'))
    return [x[0] for x in selected],audit


def pool_diagnostics(latencies,pool_ids,selected_ids):
    """Decompose observed-reference regret into pool and selector penalties.

    Missing labels are reported. Never impute their performance or call this
    a global optimum. Predictions must be selected before providing labels.
    """
    valid={k:v for k,v in latencies.items() if isinstance(v,(int,float)) and math.isfinite(v) and v>0}
    pool=set(pool_ids);selected=set(selected_ids)
    if not selected<=pool:raise ValueError('selected IDs must belong to pool')
    best=min(valid.values(),default=None)
    pool_best=min((valid[k] for k in pool&valid.keys()),default=None)
    chosen=min((valid[k] for k in selected&valid.keys()),default=None)
    return dict(reference_best_ms=best,pool_best_ms=pool_best,selected_best_ms=chosen,
        pool_label_coverage=len(pool&valid.keys())/max(1,len(pool)),
        selected_label_coverage=len(selected&valid.keys())/max(1,len(selected)),
        pool_penalty=None if best is None or pool_best is None else pool_best/best,
        selector_penalty=None if chosen is None or pool_best is None else chosen/pool_best)

