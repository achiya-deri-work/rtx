from pathlib import Path
from types import SimpleNamespace

from benchmarks.profile_moe_active_budget import (geometries,contexts,worker_command,validate_result,
    joint_geometry,fanout_geometries,fanout_contexts)
from benchmarks.benchmark_moe_varlen import parser


def test_budget_and_alignment_are_hard_and_omissions_truthful():
    rows=geometries()
    assert len(rows)==12 and sum(r['feasible'] for r in rows)==4
    for row in rows:
        assert row['shared_hidden_dim']%64==row['expert_hidden_dim']%64==0
        assert row['active_hidden']==row['shared_hidden_dim']+row['top_k']*row['expert_hidden_dim']
        assert row['feasible']==(abs(row['expansion_error'])<=.075)
    assert all(not r['feasible'] for r in rows if r['requested_fraction']=='1/6')


def test_matrix_has_requested_counts_layouts_and_primary_lengths():
    rows=contexts()
    assert len(rows)==len({r['id'] for r in rows})
    assert {r['layout'] for r in rows}=={'packed','flat'}
    assert {r['case']['num_experts'] for r in rows}=={16,32,64,128}
    assert {8,16,32}<={r['samples'] for r in rows}
    for row in rows:
        assert min(row['lengths'])>=1024
        assert row['samples']==len(row['lengths'])
        assert row['geometry']['feasible'] and row['case']['num_experts']%8==0
    flat=[(r['case']['d_model'],r['case']['num_experts'],r['case']['top_k'],
           r['case']['expert_hidden_dim'],r['case']['shared_hidden_dim'],r['total_tokens'])
          for r in rows if r['layout']=='flat']
    assert len(flat)==len(set(flat))


def test_command_requests_real_flat_mode_and_all_qualified_phases():
    row=next(r for r in contexts() if r['layout']=='flat')
    args=SimpleNamespace(expert_trials=8,shared_trials=8,router_trials=4,
        tune_seconds=90,rounds=2,samples=7)
    cmd=worker_command(Path('/source'),Path('/out.json'),row,args)
    parsed=parser().parse_args(cmd[3:])
    assert parsed.layout=='flat' and parsed.bf16_compact and parsed.stable_inputs
    assert parsed.hidden_alignment==64 and '--oracle' in cmd
    assert {'mxfp8_backward','mxfp8_step','router_step','expert_step','shared_step',
            'bf16_saved_step','expert_bf16_checkpoint_step','shared_bf16_checkpoint_step'}<=set(parsed.phases)


def test_resume_rejects_stale_driver_shape_layout_and_measurement_contract():
    import copy
    import pytest
    row=contexts()[0]
    manifest={'identity':{'runtime':'runtime','drivers':{'benchmark_moe_varlen.py':'driver'}},
              'settings':dict(expert_trials=8,shared_trials=8,router_trials=4,rounds=2,samples=7)}
    payload={key:row[key] for key in ('layout','lengths','total_tokens','seed','case')}
    phases=('mxfp8_forward','mxfp8_backward','mxfp8_step','bf16_checkpoint_step',
        'bf16_compact_checkpoint_step','bf16_saved_step','router_step','expert_step',
        'expert_bf16_checkpoint_step','shared_step','shared_bf16_checkpoint_step')
    payload.update(status='complete',source_digest='runtime',driver_sha256='driver',stable_inputs=True,
        hidden_alignment=64,tuning_budgets=dict(expert=8,shared=8,router=4),
        timings={phase:[{'samples_ms':[1.]*7},{'samples_ms':[1.]*7}] for phase in phases},
        validation={'all_finite':True,'independent_quantized_fp64_experts':{'output':{}}},
        memory={'live_growth_bytes':0})
    validate_result(payload,row,manifest)
    for key,value in [('source_digest','old'),('driver_sha256','old'),('layout','flat'),
                      ('stable_inputs',False),('hidden_alignment',32)]:
        bad=copy.deepcopy(payload);bad[key]=value
        with pytest.raises(ValueError):validate_result(bad,row,manifest)
    bad=copy.deepcopy(payload);bad['timings']['mxfp8_backward'][0]['samples_ms'][0]=float('nan')
    with pytest.raises(ValueError):validate_result(bad,row,manifest)
    bad=copy.deepcopy(payload);bad['memory']['live_growth_bytes']=1024
    with pytest.raises(ValueError):validate_result(bad,row,manifest)


def test_joint_geometry_keeps_hard_budget_and_minimizes_fraction_before_budget():
    from fractions import Fraction
    for d,k,shared,hidden in ((384,4,256,320),(512,4,384,448),(512,6,384,256),
                              (512,8,384,192),(384,6,320,192)):
        row=joint_geometry(d,'1/6',k)
        assert (row['shared_hidden_dim'],row['expert_hidden_dim'])==(shared,hidden)
        assert abs(row['expansion_error'])<=.075
        selected=(abs(Fraction(shared,shared+k*hidden)-Fraction(1,6)),abs(shared+k*hidden-4*d))
        for s in range(64,4*d+192,64):
            for h in range(64,4*d//k+192,64):
                active=s+k*h
                if abs(active-4*d)*40<=3*4*d:
                    assert selected<=(abs(Fraction(s,active)-Fraction(1,6)),abs(active-4*d))
    row=joint_geometry(384,'1/6',4)
    assert row['effective_shared_fraction']==1/6 and row['expansion_error']==0


def test_fanout_study_is_controlled_unique_and_preserves_matched_load_trio():
    rows=fanout_contexts()
    assert len(rows)==48==len({r['id'] for r in rows})
    models=fanout_geometries()
    k4=[g for g in models if g['d_model']==384 and g['top_k']==4]
    assert {g['shared_hidden_dim'] for g in k4}=={192,256,320}
    assert {g['expert_hidden_dim'] for g in k4}=={320}
    for row in rows:
        assert row['stable_experts'] and min(row['lengths'])>=1024
        assert row['case']['num_experts']%8==0
        assert all(row['case'][n]%64==0 for n in ('shared_hidden_dim','expert_hidden_dim'))
    for layout in ('packed','flat'):
        trio=[next(r for r in rows if r['geometry']['id']==f'd512_f1of4_k{k}'
            and r['case']['num_experts']==e and r['layout']==layout
            and r['samples']==16 and r['total_tokens']==16384) for k,e in ((4,32),(6,48),(8,64))]
        assert {r['case']['top_k']/r['case']['num_experts'] for r in trio}=={1/8}
        assert {r['geometry']['active_hidden'] for r in trio}=={2048}
        assert len({r['study_group'] for r in trio})==1
    args=SimpleNamespace(expert_trials=8,shared_trials=8,router_trials=4,
        tune_seconds=90,rounds=2,samples=7)
    parsed=parser().parse_args(worker_command(Path('/source'),Path('/out'),rows[0],args)[3:])
    assert parsed.stable_experts
