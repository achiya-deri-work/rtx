"""Versioned, CPU-only active-path audit. Never reinstates data automatically."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def policy(record, name):
    prefix='context.workload.'+name+'='
    values=[k[len(prefix):] for k,v in record['features'].items() if k.startswith(prefix) and v]
    return values[0] if len(values)==1 else None


def gemm_fields():
    # Match the dataclass-field projection in _mlp_precision.prequant_config,
    # without importing Torch/CuTe or initializing CUDA in this offline tool.
    source=Path(__file__).resolve().parents[2]/'rtx/configs/mxfp8.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MXFP8GemmConfig')
    return {n.target.id for n in cls.body if isinstance(n,ast.AnnAssign)}


GEMM_FIELDS=gemm_fields()


def effective_config(record):
    """Conservative equivalence: strip only proven inactive MLP fields.

    Unknown revisions/policies retain the entire payload (no guessed merging).
    Nested backward policies remain opaque; this deliberately under-deduplicates.
    """
    c=record['config']
    if record['family']!='mxfp8_recompute_mlp' or record['kernel_revision']!=33:
        return c
    precision,phase=policy(record,'precision'),policy(record,'phase')
    if phase not in ('train','forward') or precision not in ('bf16_gate','mxfp8'):
        return c
    if c.get('gate_backend','onchip') not in ('onchip','cooperative_prequant') or c.get('down_backend','onchip') not in ('onchip','producer_quantized'):
        return c
    if precision=='bf16_gate':
        up=c.get('gate_prequant',{})
        result=dict(up=dict(quant=up.get('quant'),weight_quant=up.get('weight_quant') or up.get('quant'),
            quant_launches=up.get('quant_launches','dual'),
            gemm={k:v for k,v in up.get('gemm',{}).items() if k in GEMM_FIELDS}),
            down_quant=c.get('down_quant'),down_gemm=c.get('down_gemm'))
    else:
        gate=c.get('gate_backend','onchip');down=c.get('down_backend','onchip')
        result=dict(gate_backend=gate,down_backend=down,gate_up=c.get('gate_up'))
        if gate=='cooperative_prequant':
            result.update(gate_prequant=c.get('gate_prequant'),cooperative_ctas_per_sm=c.get('cooperative_ctas_per_sm',1.))
        if down=='producer_quantized':
            result.update(down_quant=c.get('down_quant'),down_gemm=c.get('down_gemm'))
        else:
            result['down']=c.get('down')
        if phase=='train':
            result.update({k:c.get(k) for k in ('backward_recompute','backward_overlap','recompute_backend')})
            if c.get('recompute_backend','onchip')=='prequant':
                result['recompute']=c.get('recompute')
    if phase=='train':
        result.update({k:c.get(k) for k in ('gate_backward','down_backward')})
    return result


def unsafe_gemm(g):
    return isinstance(g,dict) and g.get('tile_m')==64 and g.get('stages')==1 and \
        g.get('scale_role','consumers')=='consumers' and g.get('scale_layout','row_major')=='row_major'


def classify(record):
    """Return legacy reject, active-path reject, and whether routing is known."""
    c=record['config'];family=record['family']
    legacy=unsafe_gemm(c.get('gemm',{}))
    if family=='nvfp4_weight_prequant_fwd' and c.get('quant_x',{}).get('scale_reciprocal','direct')!='direct':
        return True,True,True
    if family!='mxfp8_recompute_mlp':
        return legacy,legacy,True
    legacy |= unsafe_gemm(c.get('gate_prequant',{}).get('gemm',{})) or unsafe_gemm(c.get('down_gemm',{}))
    precision,phase=policy(record,'precision'),policy(record,'phase')
    if record['kernel_revision']!=33 or precision not in ('mxfp8','bf16_gate') or phase not in ('train','forward'):
        return legacy,legacy,False
    if c.get('gate_backend','onchip') not in ('onchip','cooperative_prequant') or c.get('down_backend','onchip') not in ('onchip','producer_quantized') or c.get('recompute_backend','onchip') not in ('onchip','prequant'):
        return legacy,legacy,False
    gemms=[]
    if precision=='bf16_gate' or c.get('gate_backend','onchip')=='cooperative_prequant':
        gemms.append(c.get('gate_prequant',{}).get('gemm',{}))
    if precision=='bf16_gate' or c.get('down_backend','onchip')=='producer_quantized':
        gemms.append(c.get('down_gemm',{}))
    if precision=='mxfp8' and phase=='train' and c.get('recompute_backend','onchip')=='prequant':
        gemms.append(c.get('recompute',{}).get('gemm',{}))
    return legacy,any(unsafe_gemm(g) for g in gemms),True


def context_weights(rows):
    """Equal total weight per family and per context within each family."""
    counts=Counter((r['family'],r['context']) for r in rows)
    contexts=Counter(f for f,_ in counts)
    weights=[1/(contexts[r['family']]*counts[r['family'],r['context']]) for r in rows]
    scale=len(weights)/sum(weights) if weights else 1.
    return [w*scale for w in weights]


def audit(paths):
    from tools.research_effort import rows
    counters=defaultdict(Counter);raw=defaultdict(set);effective=defaultdict(set)
    rescued=[];seen=set()
    for r in rows(paths):
        oid=r.get('observation_id') or identity(r)
        if oid in seen:continue
        seen.add(oid)
        family=r['family'];counts=counters[family];counts['observations']+=1
        legacy,active,known=classify(r)
        counts['legacy_quarantined']+=int(legacy)
        counts['active_quarantined']+=int(active)
        counts['unknown_routing']+=int(not known)
        if r['outcome']['status']!='ok':continue
        counts['successful']+=1
        if legacy and not active:
            counts['inactive_only_successes_pending_revalidation']+=1
            rescued.append(dict(observation_id=oid,family=family,config_id=r['config_id'],
                effective_id=identity(effective_config(r)),revision=r['kernel_revision']))
        if not active:
            raw[family].add(r['config_id']);effective[family].add(identity(effective_config(r)))
    return dict(schema=2,automatically_reinstated=False,
        families={f:dict(c,raw_configs_after_active_filter=len(raw[f]),
                         effective_configs_after_active_filter=len(effective[f])) for f,c in counters.items()},
        pending_revalidation=rescued)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',default='autotune_reports/effort_ml_v2/audit.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source=json.loads(Path(args.sources).read_text())
    result=audit(source['train_paths'])
    result['sources']=source['train_paths']
    from tools.predictor_study import atomic
    atomic(args.output,result)
    print(json.dumps(result['families'],indent=2))


if __name__=='__main__':main()
