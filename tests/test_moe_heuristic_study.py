"""CPU-only contracts for the current-source discovery/holdout campaign."""
from collections import Counter
from benchmarks.study_moe_heuristics import study_contexts, matches


def test_matrix_has_disjoint_discovery_and_unseen_shapes():
    rows=study_contexts()
    assert Counter(r['partition'] for r in rows) == {'discovery':24,'holdout':8}
    signature=lambda r:(tuple(sorted(r['case'].items())),r['layout'],tuple(r['lengths']))
    assert len({signature(r) for r in rows}) == 32
    assert {r['case']['top_k'] for r in rows} == {4,6,8}
    assert {r['layout'] for r in rows} == {'packed','flat'}
    assert {r['samples'] for r in rows if r['partition']=='discovery'} == {8,16,32}


def test_every_architecture_respects_hard_budget_and_alignment():
    for row in study_contexts():
        c=row['case'];active=c['shared_hidden_dim']+c['top_k']*c['expert_hidden_dim']
        assert abs(active-4*c['d_model'])*40 <= 3*4*c['d_model']
        assert c['shared_hidden_dim']%64 == c['expert_hidden_dim']%64 == c['num_experts']%8 == 0
        if row['partition']=='discovery':assert min(row['lengths'])>=1024


def test_two_shards_keep_all_families_in_both():
    for shard in range(2):
        rows=study_contexts()[shard::2]
        assert Counter(r['partition'] for r in rows) == {'discovery':12,'holdout':4}
        assert {r['case']['d_model'] for r in rows} == {384,512}
        assert len({r['id'] for r in rows}) == 16


def test_pilot_adoption_requires_exact_public_shape_and_input_contract():
    row=study_contexts()[0]
    payload=dict(status='complete',case=row['case'],lengths=row['lengths'],
                 layout=row['layout'],seed=row['seed'],stable_inputs=True)
    assert matches(payload,row)
    assert matches({**payload,'case':{**row['case'],'id':'fixture','num_shared_experts':1}},row)
    assert not matches({**payload,'layout':'flat'},row)
    assert not matches({**payload,'stable_inputs':False},row)
    assert not matches({**payload,'seed':row['seed']+1},row)


def test_exact_effects_keep_each_pair_identical_in_other_fields():
    from types import SimpleNamespace
    from benchmarks.analyze_moe_heuristics import exact_effects
    row=lambda value,score,other=4:SimpleNamespace(context_id='shape',family='experts',
        config_id=str(value)+'_'+str(other),score=score,
        outcome=SimpleNamespace(timing_eligible=True),serialized_config={'tile':value,'other':other})
    pairs,summary=exact_effects([row(2,20),row(1,10),row(3,30),row(4,40,8)])
    assert len(pairs)==3
    assert {(p['a'],p['b']):p['b_over_a'] for p in pairs} == {(1,2):2.,(2,3):1.5,(1,3):3.}
    assert len(summary)==3


def test_failed_and_unstable_rows_consume_budget_but_cannot_win():
    from types import SimpleNamespace
    from benchmarks.analyze_moe_heuristics import budget_score
    row=lambda score,eligible:SimpleNamespace(score=score,outcome=SimpleNamespace(timing_eligible=eligible))
    rows=[row(1,True),row(.1,False),row(.5,True)]
    assert budget_score(rows,[0,1,2],budget=2)==1.
    assert budget_score(rows,[0,2,1],budget=2)==0.
    assert budget_score(rows,[1,0,2],budget=1) is None


def test_profile_capture_does_not_expand_tensor_rows():
    import torch
    from benchmarks.profile_moe_routing_cost import tensor_results
    x=torch.zeros((1024,8))
    assert tensor_results(lambda:x)()==(x,)
    result=tensor_results(lambda:([x],{'nested':x},None))()
    assert len(result)==2 and result[0] is x and result[1] is x
    assert tensor_results(lambda:None)()==()


def test_model_folds_cannot_leak_same_shape_across_boards():
    from types import SimpleNamespace
    from benchmarks.analyze_moe_heuristics import semantic_geometry
    row=lambda features:SimpleNamespace(family='mlp',kernel_revision=1,features=features)
    first={'context.workload.m':16384,'context.workload.k':384,'context.device.device_index':0}
    assert semantic_geometry(row(first))==semantic_geometry(row({**first,'context.device.device_index':1}))
    assert semantic_geometry(row(first))!=semantic_geometry(row({**first,'context.workload.m':8192}))


def test_training_features_match_the_proved_inactive_axis_normalization():
    from types import SimpleNamespace
    from benchmarks.analyze_moe_heuristics import canonical_features
    row=SimpleNamespace(features={'config.gather_mode=warp':1.,'context.workload.m':1024},
        serialized_config={'family':'grouped_prepared','prepared':{'input_preparation':'indexed_transpose','forward_engine':'packed'}})
    values=canonical_features(row)
    assert values['config.gather_mode=flat']==1. and 'config.gather_mode=warp' not in values
    row.serialized_config['prepared']['forward_engine']='onchip'
    assert canonical_features(row)['config.gather_mode=warp']==1.


def test_movement_attribution_does_not_call_fused_quantization_a_view():
    from benchmarks.analyze_moe_routing_cost import kernel_class
    assert kernel_class('RoutedDYQuant')=='quantization_fused_preparation'
    assert kernel_class('CompactGather')=='dispatch_combine'
    assert kernel_class('PersistentExpertDW')=='gemm'


def test_router_attribution_refuses_aggregate_only_graph_traces(tmp_path):
    import sqlite3
    import pytest
    from benchmarks.analyze_moe_routing_cost import router_sqlite
    path=tmp_path/'aggregate.sqlite'
    with sqlite3.connect(path) as db:db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start,end)')
    with pytest.raises(ValueError,match='cuda-graph-trace=node'):router_sqlite(path)


def test_prospective_report_never_calls_an_unstarted_study_complete(tmp_path):
    from benchmarks.analyze_moe_heuristics import prospective_results
    value=prospective_results([tmp_path/'not-started'])
    assert not value['complete'] and not value['deployment_enabled'] and not value['rows']


def test_prospective_report_checks_matched_inputs_and_records_hard_changes(tmp_path,monkeypatch):
    import copy
    import json
    import pytest
    from benchmarks.analyze_moe_heuristics import prospective_results
    monkeypatch.setattr('benchmarks.study_moe_heuristics.result_validation',lambda *args:None)
    context={'id':'unseen','layout':'flat'}
    (tmp_path/'manifest.json').write_text(json.dumps(dict(contexts=[context],identity={},settings={},prior_sha256='frozen')))
    value=dict(status='complete',source_digest='fixed',identities=dict(input='x',upstream='dy',parameters={'w':'w'},routing='a'),
        timings={key:[{'median_ms':2.}] for key in ('mxfp8_step','router_step','expert_step','shared_step')},
        tuning_counts={key:{'evaluated_trials':8} for key in ('expert','shared','router')})
    for mode in ('control','prior'):
        (tmp_path/'unseen'/mode).mkdir(parents=True)
        (tmp_path/'unseen'/mode/'result.json').write_text(json.dumps(value))
    prior=copy.deepcopy(value);prior['identities']['routing']='b'
    prior['timings']['mxfp8_step']=[{'median_ms':1.5}]
    path=tmp_path/'unseen/prior/result.json';path.write_text(json.dumps(prior))
    report=prospective_results([tmp_path]);assert report['complete']
    assert not report['rows'][0]['hard_ids_identical']
    assert report['rows'][0]['prior_over_control']['mxfp8_step']==.75
    prior['identities']['input']='different';path.write_text(json.dumps(prior))
    with pytest.raises(ValueError,match='Unmatched prospective'):prospective_results([tmp_path])


def test_pilot_adoption_survives_bundle_relocation_without_changing_provenance(tmp_path):
    import hashlib
    import pytest
    from benchmarks.analyze_moe_heuristics import adopted_residual_root
    pilot=tmp_path/'pilot-gpu0'/'shape';pilot.mkdir(parents=True)
    result=pilot/'result.json';result.write_text('{"status":"complete"}')
    document=dict(residual_root='/missing-original/pilot-gpu0/shape',
                  sha256=hashlib.sha256(result.read_bytes()).hexdigest())
    assert adopted_residual_root(tmp_path/'study-gpu0',document)==pilot
    result.write_text('{}')
    with pytest.raises(ValueError,match='changed adopted pilot'):adopted_residual_root(tmp_path/'study-gpu0',document)
