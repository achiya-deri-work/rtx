"""The broad precision matrix must compare strong, matched complete baselines."""
from pathlib import Path
from types import SimpleNamespace

from benchmarks.compare_moe_precisions import contexts,command,PHASES
from benchmarks.benchmark_moe_varlen import parser


def test_primary_matrix_alignment_active_budget_and_layout_pairs():
    rows=contexts()
    assert len(rows)==16==len({r['id'] for r in rows})
    assert {r['case']['d_model'] for r in rows}=={384,512,768}
    assert {r['case']['num_experts'] for r in rows}>={24,32,48,64,96,128}
    assert {r['case']['top_k'] for r in rows}=={4,6,8}
    for row in rows:
        c=row['case'];active=c['shared_hidden_dim']+c['top_k']*c['expert_hidden_dim']
        assert abs(active/(4*c['d_model'])-1)<=.075
        assert c['shared_hidden_dim']%64==c['expert_hidden_dim']%64==c['num_experts']%8==0
        if row['axis']=='primary':assert min(row['lengths'])>=1024
    for group in range(8):
        pair=[r for r in rows if r['group']==group]
        assert {r['layout'] for r in pair}=={'flat','packed'}
        assert pair[0]['case']==pair[1]['case'] and pair[0]['lengths']==pair[1]['lengths']
        assert pair[0]['seed']==pair[1]['seed']


def test_commands_have_fresh_component_budgets_strong_baselines_and_oracles():
    options=SimpleNamespace(trials=32,seconds=90.,rounds=3,samples=9)
    for row in (contexts()[0],contexts()[-1]):
        args=parser().parse_args(command(Path('/source'),Path('/out.json'),row,options)[3:])
        budget=8 if row['axis']=='small_guard' else 32
        assert args.tune_expert_trials==args.tune_shared_trials==args.tune_router_trials==budget
        assert set(args.phases)==set(PHASES)
        assert args.bf16_compact and args.bf16_numerics and args.oracle
        assert args.stable_inputs and args.stable_experts and args.max_autotune
        assert args.search_schedule=='progressive' and args.candidate_mix=='mixed'
