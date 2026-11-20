"""Generate legal preparation ablations from a qualified fixture (CPU only).

Meta tensors only build the existing shared adapter's configuration domain.
No candidate is declared correct or fast by this metadata-only step.
"""
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
from unittest.mock import patch


def configurations(saved):
    import torch
    import rtx.autotune.mlp as mlp
    from rtx.autotune.moe import make_moe_adapter
    from rtx.mlp import MXFP8MLPConfig
    from rtx.moe import MXFP8MoEConfig
    case=saved['case'];m=sum(saved['lengths']);d=case['d_model'];h=case['shared_hidden_dim']
    expert=MXFP8MoEConfig.from_dict(saved['expert_config'])
    capacity=saved['expert_plan']['capacity'] if 'capacity' in saved.get('expert_plan',{}) else saved['expert_capacity_bound']
    adapter=make_moe_adapter((m,d,case['num_experts'],capacity,case['top_k'],case['expert_hidden_dim']),
                             None,initial=expert)
    expert=adapter.update_fn(expert,'prepared.dy_preparation','fused')
    if adapter.rejection(expert):raise ValueError(adapter.rejection(expert))
    shared=MXFP8MLPConfig.from_dict(saved['shared_config'])
    values=[torch.empty(*size,device='meta',dtype=torch.bfloat16)
            for size in ((m,d),(2*h,d),(d,h))]
    with ExitStack() as stack:
        stack.enter_context(patch.object(mlp,'_validate',lambda *args:None))
        stack.enter_context(patch.object(mlp,'_mlp_hardware',lambda _:dict(multiprocessor_count=saved['gpu_sms'])))
        stack.enter_context(patch.object(torch.cuda,'Stream',lambda **kwargs:object()))
        a=mlp.make_mlp_adapter(*values,initial=shared,
            _reference_cache={'values':(torch.empty(m,d,device='meta',dtype=torch.bfloat16),None)})
        mode='recompute_both' if m%32==0 else 'recompute'
        fused=a.update_fn(shared,'gradient_quantization_path',mode)
        if a.rejection(fused):raise ValueError(a.rejection(fused))
    return expert,fused


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from benchmarks.study_moe_priors import atomic_json
    saved=json.loads(args.baseline_result.read_text())
    expert,shared=configurations(saved)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    atomic_json(args.output/'expert.json',dict(config=expert.to_dict()))
    atomic_json(args.output/'shared.json',dict(config=shared.to_dict()))
    atomic_json(args.output/'provenance.json',dict(source=str(args.baseline_result.resolve()),
        qualification='metadata legality only; independent GPU oracle and complete-layer qualification required'))


if __name__=='__main__':main()
