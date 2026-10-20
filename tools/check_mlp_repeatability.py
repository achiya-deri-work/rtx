"""Repeated selected-direct MLP recipe checks, including the first invocation."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from rtx._mlp_autotune import selected_mlp
from rtx.autotune.mlp import reference
from rtx.mlp import MXFP8MLPConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--field', default='challenger')
    parser.add_argument('--shape', default='11712,256,768,256')
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--generations', type=int, default=1,
                        help='Repeat with changed inputs, both weights and upstream gradients, preserving tensor identities')
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-bypass-quarantine', action='store_true',
                        help='Unsafe reproducer only: disable the targeted derivative guard in this process')
    parser.add_argument('--scale-schedule', choices=('before_wait','after_wait'))
    parser.add_argument('--scale-role', choices=('consumers','producer'))
    parser.add_argument('--scale-store', choices=('scalar','packed'))
    parser.add_argument('--gate-dw-control', choices=('register_quant', 'after_wait',
                                                     'preload', 'explicit_scale_load', 'producer_scales'))
    parser.add_argument('--diagnostic-overlap-control',
                        choices=('concurrent', 'clone_inputs', 'sync_before', 'prequant_before'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.rounds < 2:
        parser.error('at least two rounds required')
    if args.generations < 1:
        parser.error('at least one generation required')
    if args.diagnostic_bypass_quarantine:
        import rtx.configs.glu as glu_config
        glu_config.prequant_recompute_rejection = lambda config: None
    config = MXFP8MLPConfig.from_dict(json.loads(args.config.read_text())[args.field])
    if args.gate_dw_control:
        dw = config.gate_backward.dw
        if args.gate_dw_control == 'register_quant':
            dw = replace(dw, quant_b=replace(dw.resolved_quant_b(),
                         transposed_load_engine='register'))
        else:
            change = {'after_wait': {'scale_schedule': 'after_wait'},
                      'preload': {'mma_schedule': 'preload'},
                      'producer_scales': {'scale_role': 'producer'},
                      'explicit_scale_load': {'sfb_s2r_bits': 8}}[args.gate_dw_control]
            dw = replace(dw, gemm=replace(dw.gemm, **change))
        config = replace(config, gate_backward=replace(config.gate_backward, dw=dw))
    if args.scale_schedule:
        config = replace(config,recompute=replace(config.recompute,
            gemm=replace(config.recompute.gemm,scale_schedule=args.scale_schedule)))
    if args.scale_role:
        config = replace(config,recompute=replace(config.recompute,
            gemm=replace(config.recompute.gemm,scale_role=args.scale_role)))
    if args.scale_store:
        config = replace(config,recompute=replace(config.recompute,
            gemm=replace(config.recompute.gemm,scale_smem_store=args.scale_store)))
    spec = json.dumps(dict(mode='fixed', initial=config.to_dict()), sort_keys=True)
    m, k, h, o = map(int, args.shape.split(','))
    torch.manual_seed(args.seed)
    values = tuple((torch.randn(*s, device='cuda', dtype=torch.bfloat16)*scale).requires_grad_()
                   for s, scale in (((m,k),.2), ((2*h,k),.02), ((o,h),.02)))
    dy = torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    with torch.no_grad():
        expected = reference(*values,dy,'silu')
    fn = torch.compile(lambda *v: selected_mlp(*v,'silu',spec,True)[0],
                       fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    first = None
    rows = []
    identities=tuple(id(v) for v in values)
    for index in range(args.rounds*args.generations):
        generation=index//args.rounds
        if index and index%args.rounds==0:
            torch.manual_seed(args.seed+generation)
            with torch.no_grad():
                for value,scale in zip(values,(.2,.02,.02)):
                    value.copy_(torch.randn_like(value)*scale)
                dy.normal_(std=.1)
                expected=reference(*values,dy,'silu')
            first=None
        y = fn(*values)
        actual = (y, *torch.autograd.grad(y,values,dy))
        torch.cuda.synchronize()
        row = {'iteration': index, 'generation':generation, 'numerics': {}}
        for name, a, b in zip(('output','dx','dgate_up','ddown'),actual,expected):
            delta = a.detach().float()-b.detach().float()
            row['numerics'][name] = dict(finite=bool(a.isfinite().all()),
                max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm()/b.float().norm().clamp_min(1e-12)))
        if first is None:
            first = tuple(a.detach().clone() for a in actual)
        row['equal_first'] = [torch.equal(a,b) for a,b in zip(actual,first)]
        rows.append(row)
        print(json.dumps(row),flush=True)
        del actual,y
    assert identities==tuple(id(v) for v in values),'input/weight tensor identity changed'
    report = dict(config=config.to_dict(),shape=[m,k,h,o],seed=args.seed,
                  generations=args.generations,rounds_per_generation=args.rounds,
                  identities_unchanged=True,
                  torch=str(torch.__version__),cuda=torch.version.cuda,rows=rows,
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  diagnostic_bypass_quarantine=args.diagnostic_bypass_quarantine,
                  gate_dw_control=args.gate_dw_control,
                  diagnostic_overlap_control=args.diagnostic_overlap_control,
                  scope='identical inputs within each generation; changed inputs/weights/gradients between generations; selected-direct; no timing claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    # Repeatability is distinct from the broad low-precision acceptance gate.
    assert all(all(row['equal_first']) for row in rows), 'same-input results changed'
    assert all(v['finite'] and v['relative_l2'] < .025
               for row in rows for v in row['numerics'].values())


if __name__ == '__main__':
    main()
