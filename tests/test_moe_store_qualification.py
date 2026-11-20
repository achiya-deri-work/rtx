import json
from pathlib import Path

from benchmarks.qualify_moe_store_prior import CASES,command
from benchmarks.analyze_moe_store_prior import search_summary


def test_heldout_workloads_separate_pool_alignment_and_architecture():
    assert sum(CASES['two_pool']['lengths'])==sum(CASES['single_pool']['lengths'])==12480
    assert sum(CASES['ragged']['lengths'])%64!=0
    assert CASES['different_width']['case_spec']['d_model']!=384
    args=command(Path('/project'),Path('/result.json'),CASES['different_width'],8)
    assert args[args.index('--tune-expert-trials')+1]=='8'
    assert args[args.index('--tune-shared-trials')+1]=='8'
    assert '--oracle' in args and '--case-spec' in args


def test_summary_uses_actual_observations_and_stronger_budget_not_seed_pool(tmp_path):
    records=[]
    for i in range(16):
        records.append(dict(config_id=str(i),config=dict(down_backend='prequant',
            down_gemm=dict(epilogue='tma' if i==1 else 'direct')),
            strategy='initial' if i==0 else 'structural',
            outcome=dict(status='ok',median_ms=2. if i==0 else 1. if i<8 else .99)))
    (tmp_path/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    d=search_summary(tmp_path,'observations.jsonl')
    assert d['observations']==16
    assert d['prefixes']['8']['first_store_trial']==2
    assert d['prefixes']['8']['best_ms']==1.
    assert abs(d['eight_trial_regret_vs_sixteen']-(1/.99-1))<1e-9


def test_summary_excludes_explicitly_unstable_latencies(tmp_path):
    records=[dict(config_id=str(i),config={},strategy='structural',
        outcome=dict(status='ok',median_ms=.01 if i==1 else 1.,
            metadata=dict(sampling=dict(collection=dict(stationary=i!=1)))))
        for i in range(8)]
    (tmp_path/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary=search_summary(tmp_path,'observations.jsonl')
    assert summary['prefixes']['8']['best_ms']==1.
    assert summary['timing_eligible_observations']==7
    assert summary['prefixes']['8']['timing_eligible_observations']==7
