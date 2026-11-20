from types import SimpleNamespace
from benchmarks.study_moe_priors import workloads, build_command
from rtx.autotune.pretrained import portable_features, _shape_group_key, _device_key


def _cases():
    return [dict(id=f'case-{i}',d_model=384,num_experts=32,top_k=4,
        expert_hidden_dim=128,shared_hidden_dim=256,num_shared_experts=1) for i in range(4)]


def test_study_balances_architectures_and_has_paired_packing_controls():
    rows=workloads(_cases())
    assert len(rows)==32 and len({r['id'] for r in rows})==32
    assert [r['case']['id'] for r in rows[:4]]==[r['id'] for r in _cases()]
    assert {r['profile'] for r in rows[:4]}=={'tiny'}
    selected={r['profile']:r['lengths'] for r in rows if r['case']['id']=='case-0'}
    assert sum(selected['short_packed'])==sum(selected['packed_control'])
    assert sum(selected['boundary'])==sum(selected['boundary_control'])
    assert 0 in selected['boundary']


def test_effort_extensions_share_residuals_and_never_seed_from_winner(tmp_path):
    args=SimpleNamespace(shared_trials=0,tune_seconds=120,max_autotune=False,seeds=None)
    row=workloads(_cases())[0]
    a=build_command(tmp_path,tmp_path/'results',row,8,args)
    b=build_command(tmp_path,tmp_path/'results',row,16,args)
    assert a[a.index('--tuning-dir')+1]==b[b.index('--tuning-dir')+1]
    assert a[a.index('--search-schedule')+1]=='progressive'
    assert '--expert-config' not in b
    assert a[a.index('--output')+1]!=b[b.index('--output')+1]


def test_named_moe_geometry_features_and_packing_are_not_provenance():
    fields=dict(zip(('tokens','d_model','experts','capacity','top_k','hidden'),
                    (128,384,32,32,4,128)))
    raw={'context.workload.'+name:value for name,value in fields.items()}
    raw.update({'context.device.multiprocessor_count':170,
        'context.device.l2_cache_size':96*1024**2,
        'context.tags.batch_contract.sequences':4,
        'context.tags.source=secret-source-hash':1})
    features=portable_features(raw)
    assert features['offline.moe.rows_per_expert']==16
    assert features['offline.moe.slot_amplification']==2
    assert features['offline.moe.sequences']==4
    assert features['offline.moe.weight_to_l2_ratio']>0
    assert not any(key.startswith('context.tags.') for key in features)
    # Feature construction must be idempotent for a loaded offline record.
    assert portable_features(features)==features
    item=SimpleNamespace(family='mxfp8_balanced_experts',kernel_revision=10,features=features)
    other=SimpleNamespace(family=item.family,kernel_revision=10,features={**features,
        'context.workload.hidden':256})
    assert _shape_group_key(item)!=_shape_group_key(other)
    assert _shape_group_key(item)==_shape_group_key(SimpleNamespace(**vars(item)))


def test_packed_group_and_flat_bus_width_fallback():
    fields={'context.workload.total_tokens':2000,'context.workload.n':32,
        'context.workload.k':4,'context.workload.sequences':4,
        'context.device.multiprocessor_count':170,'context.device.memory_bus_width_bits':512}
    a=SimpleNamespace(family='balanced_varlen_routing',kernel_revision=6,features=fields)
    b=SimpleNamespace(**{**vars(a),'features':{**fields,'context.workload.n':64}})
    assert _shape_group_key(a)!=_shape_group_key(b)
    assert _device_key(a)=='sms-170-bus-512'
    named={key:value for key,value in fields.items() if key not in ('context.workload.n','context.workload.k')}
    named.update({'context.workload.experts':32,'context.workload.top_k':4})
    a.features=named;b.features={**named,'context.workload.experts':64}
    assert _shape_group_key(a)!=_shape_group_key(b)


def test_contextual_bandit_uses_named_moe_dimensions():
    from rtx.autotune.bandit import _observation_similarity
    fields={'context.workload.tokens':128.,'context.workload.d_model':384.,
        'context.workload.experts':32.,'context.workload.capacity':32.,
        'context.workload.top_k':4.,'context.workload.hidden':128.}
    same=SimpleNamespace(features=fields)
    far=SimpleNamespace(features={**fields,'context.workload.tokens':32768.,
        'context.workload.capacity':4096.})
    assert _observation_similarity(same,fields)==1.
    assert _observation_similarity(far,fields)<.25
    from rtx.autotune.prior import shape_distance
    assert shape_distance(fields,fields)==0
    assert shape_distance(far.features,fields)>10


def test_retrospective_curve_counts_trials_without_inventing_missing_budgets():
    from benchmarks.analyze_moe_priors import curve
    from rtx.autotune.outcomes import TrialOutcome
    def row(index,ms=None,status='ok',config=None,stationary=True):
        return SimpleNamespace(finished_at=f'{index:03d}',sequence=index,observation_id=str(index),
            config_id=str(index) if config is None else config,elapsed_s=2.,score=ms,
            outcome=TrialOutcome(status,median_ms=ms,metadata={'sampling':{'collection':{'stationary':stationary}}}))
    records=[row(1,status='compile_error'),row(2,1.),row(3,.1,stationary=False),row(4,.8),
             row(5,.7),row(6,.6),row(7,.5),row(8,.4),row(9,.4,config='8')]
    report=curve(records)
    assert report['unique_trials']==8 and report['eligible_trials']==6
    assert report['best_observed_ms']==.4
    assert report['first_timing_eligible_candidate_seconds']==4
    assert report['failed_candidate_seconds']==2
    assert report['curve']==[dict(trials=8,best_ms=.4,regret=0.,measured_candidate_seconds=16.)]


def test_device_fold_reconciles_unique_alias_but_not_ambiguous_named_skus():
    from rtx.autotune.pretrained import _reconcile_device_aliases
    base={'context.device.multiprocessor_count':36.,
          'context.device.memory_bus_width_bits':128.,'context.device.l2_cache_size':32*1024**2}
    named=lambda name:SimpleNamespace(features={**base,'context.device.sku.sku_family='+name:1.})
    fallback=SimpleNamespace(features=dict(base))
    known=named('laptop-5070')
    assert _reconcile_device_aliases([fallback,known])['reconciled_rows']==1
    assert _device_key(fallback)==_device_key(known)
    unknown=SimpleNamespace(features=dict(base))
    report=_reconcile_device_aliases([unknown,named('sku-a'),named('sku-b')])
    assert report['ambiguous_rows']==1 and report['reconciled_rows']==0
    assert _device_key(unknown)=='sms-36-bus-128'


def test_early_affinity_signatures_recover_shape_groups():
    raw={'context.family=balanced_router_affinity':1.,
        'context.workload.shapes_strides_dtypes[0][0][0]':128.,
        'context.workload.shapes_strides_dtypes[0][0][1]':384.,
        'context.workload.shapes_strides_dtypes[1][0][0]':64.,
        'context.workload.shapes_strides_dtypes[2][0][0]':32.}
    f=portable_features(raw)
    a=SimpleNamespace(family='balanced_router_affinity',kernel_revision=1,features=f)
    b=SimpleNamespace(**{**vars(a),'features':{**f,'context.workload.tokens':2000.}})
    assert _shape_group_key(a)!=_shape_group_key(b)
    assert f['context.workload.router_dim']==64
