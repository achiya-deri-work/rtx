"""Serialized held-out eight-trial transport-prior qualification.

GPU selection belongs to CUDA_VISIBLE_DEVICES. This driver never chooses or
resets a device. Each point runs in a fresh supervised process, preserving
partial results. Eight trials are an actual prefix, not a seed-list audit.
The 16-trial continuation is a finite stronger control, not a global optimum.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


CASES={
    'two_pool':dict(lengths=[5000,7480],case_id='d384_e32_k4_h128'),
    'single_pool':dict(lengths=[12480],case_id='d384_e32_k4_h128'),
    'ragged':dict(lengths=[12384],case_id='d384_e32_k4_h128'),
    'different_width':dict(lengths=[5888,7424],case_id='d384_e32_k4_h128',
        case_spec=dict(d_model=640,num_experts=40,top_k=5,
                       expert_hidden_dim=256,shared_hidden_dim=384)),
    'new_aligned':dict(lengths=[7136,10528],case_id='d512_e64_k8_h128'),
    'new_ragged':dict(lengths=[7000,10568],case_id='d512_e64_k8_h128'),
    'hybrid_384':dict(lengths=[2831,5393],case_id='d384_e32_k4_h128'),
    'hybrid_768':dict(lengths=[4097,8223],case_id='d768_e48_k6_h256'),
    'small_pool':dict(lengths=[127,385],case_id='d384_e32_k4_h128'),
    'ownership_heldout':dict(lengths=[6137,9591],case_id='d384_e32_k4_h128',
        case_spec=dict(d_model=640,num_experts=40,top_k=5,
                       expert_hidden_dim=192,shared_hidden_dim=384)),
    'preparation_heldout':dict(lengths=[7103,9037],case_id='d384_e32_k4_h128',
        case_spec=dict(d_model=576,num_experts=24,top_k=3,
                       expert_hidden_dim=160,shared_hidden_dim=320)),
}


def command(source,output,case,budget,opening_policy='transport',max_autotune=False):
    args=['--source',str(source),'--output',str(output),'--case-id',case['case_id'],
          '--lengths',*[str(n) for n in case['lengths']],
          '--tune-expert-trials',str(budget),'--tune-shared-trials',str(budget),
          '--tune-seconds','180','--search-schedule','progressive',
          '--candidate-mix','mixed','--expert-phases','--shared-phases',
          '--phases','mxfp8_step','bf16_checkpoint_step','expert_step',
          'expert_bf16_checkpoint_step','shared_step','shared_bf16_checkpoint_step',
          '--oracle','--rounds','2','--samples','7']
    if 'case_spec' in case:args+=['--case-spec',json.dumps(case['case_spec'])]
    if max_autotune:args+=['--max-autotune']
    if opening_policy=='without_transport':
        # Research-only ablation. Kernels, validation, search axes, UCB arms,
        # precision and timing are unchanged; only compound proposals vanish.
        snippet=('import runpy;import rtx.autotune.moe as e;import rtx.autotune.mlp as m;'
                 'e.row_store_seeds=lambda *a:iter(());'
                 'm.mlp_row_store_seeds=lambda *a:iter(());'
                 'runpy.run_module("benchmarks.benchmark_moe_varlen",run_name="__main__")')
        return [sys.executable,'-c',snippet,*args]
    return [sys.executable,'-m','benchmarks.benchmark_moe_varlen',*args]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',choices=tuple(CASES),default=list(CASES))
    p.add_argument('--budgets',nargs='+',type=int,default=[8,16,32])
    p.add_argument('--max-autotune',action='store_true',help='Use supported aggressive dynamic-weight Inductor settings for all native/BF16 wrappers')
    p.add_argument('--opening-policy',choices=('transport','without_transport'),default='transport')
    p.add_argument('--skip-existing',action='store_true')
    args=p.parse_args()
    if sorted(set(args.budgets))!=args.budgets or min(args.budgets)<8:
        p.error('Use increasing unique trial budgets, starting at eight or above')
    source=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    from rtx.autotune.supervisor import supervise_command
    from rtx.autotune.moe import source_revision
    manifest=dict(cases={key:CASES[key] for key in args.cases},budgets=args.budgets,
        opening_policy=args.opening_policy,source_revision=source_revision(),max_autotune=args.max_autotune,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        holdout='Not used in the measured 29504-token profiling study; no held-out timing used to rank these proposals',
        comparison='Same-context progressive continuation; neither 16 nor 32 trials is an exhaustive optimum')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for key in args.cases:
        folder=args.output/key;folder.mkdir(parents=True,exist_ok=True)
        for budget in args.budgets:
            output=folder/f'result{budget}.json'
            if args.skip_existing and output.exists():
                saved=json.loads(output.read_text())
                if saved.get('status')!='complete':raise RuntimeError(f'Incomplete result: {output}')
                print('SKIP',key,budget,flush=True);continue
            if output.exists():raise FileExistsError(output)
            print('START',key,budget,flush=True)
            with (folder/f'run{budget}.log').open('w') as log:
                code=supervise_command(command(source,output,CASES[key],budget,args.opening_policy,args.max_autotune),
                    stall_timeout_s=180,wall_timeout_s=1200,output=log,
                    cwd=str(source),environment=dict(os.environ))
            print('DONE',key,budget,'status',code,flush=True)
            if code:raise RuntimeError(f'{key} budget {budget} failed; see {folder}')


if __name__=='__main__':main()
