"""Composition must survive the ordinary module's cold/hot/cache path."""
import pytest
import torch

from benchmarks.benchmark_moe_composition import compare, composition_configs,preparation_configs,composition_baseline
from benchmarks.benchmark_moe_composition import component_configs,reference_configs
from rtx.moe import DEFAULT_MOE_CONFIG
from rtx.routing import RoutingKernelConfig


def test_factorial_changes_only_its_selected_component():
    router = RoutingKernelConfig(qb_backend='native_column', metadata_backend='cute')
    variants = composition_configs((128,384,32,32,4,128),router,DEFAULT_MOE_CONFIG)
    assert set(variants)=={'baseline','router','recompute','combined'}
    assert variants['combined']==(variants['router'][0],variants['recompute'][1])
    assert variants['router'][1]==DEFAULT_MOE_CONFIG
    assert variants['recompute'][0]==router
    assert not variants['combined'][1].prepared.active_prefix
    assert set(composition_configs((2000,384,32,256,4,128),router,DEFAULT_MOE_CONFIG))=={'baseline','recompute'}


def test_factorial_parity_checks_gradients_and_dtypes():
    ref=(torch.ones(2),torch.ones(3,dtype=torch.bfloat16))
    assert all(row['unequal']==0 for row in compare(ref,ref))
    with pytest.raises(AssertionError):compare((ref[0],ref[1]+1),ref)
    with pytest.raises(AssertionError):compare((ref[0],ref[1].float()),ref)


def test_next_iteration_baseline_preserves_identity_not_old_timings():
    source=dict(case={'expert_hidden_dim':192},initialization_case={'expert_hidden_dim':128},
        lengths=[113,157],seed=17,router_affinity_layout='row_broadcast',input_profile='outliers',
        layout='flat',stable_inputs=True,stable_experts=True,
        configs={'combined':dict(router={'x':1},expert={'x':2},shared={'x':3})},
        hashes=dict(x='input',dy='gradient',parameters={'weight':'weights'},
                    expert_ids='ids',inverse='inverse'),summary={'old_ms':10.})
    with pytest.raises(ValueError):composition_baseline(source)
    selected=composition_baseline(source,'combined')
    assert selected['expert_config']=={'x':2}
    assert selected['initialization_case']=={'expert_hidden_dim':128}
    assert selected['input_profile']=='outliers'
    assert selected['layout']=='flat' and selected['stable_inputs'] and selected['stable_experts']
    assert selected['identities']['routing']==dict(expert_ids='ids',inverse='inverse')
    assert 'summary' not in selected and 'timings' not in selected
    assert composition_baseline(selected) is selected


def test_preparation_factorial_keeps_router_and_crosses_independent_policies():
    router,expert,shared,new_expert,new_shared=[object() for _ in range(5)]
    variants=preparation_configs(router,expert,shared,new_expert,new_shared)
    assert variants==dict(baseline=(router,expert,shared),experts=(router,new_expert,shared),
                         shared=(router,expert,new_shared),combined=(router,new_expert,new_shared))


def test_router_factorial_and_bf16_comparators_use_the_same_router():
    router,expert,shared,new_router,new_shared=[object() for _ in range(5)]
    variants=component_configs(router,expert,shared,candidate_router=new_router,candidate_shared=new_shared)
    assert variants==dict(baseline=(router,expert,shared),shared=(router,expert,new_shared),
                         router=(new_router,expert,shared),combined=(new_router,expert,new_shared))
    controls,specs=reference_configs(router,expert,shared,candidate_router=new_router,saved=True)
    assert len(controls)==len(specs)==6
    for label in controls:
        assert controls[label][0] is (new_router if label.startswith('bf16_new_router') else router)
    assert specs['bf16_new_router_saved']==(False,'native')
    assert specs['bf16_compact_checkpoint']==(True,'compact')


def test_component_factorial_retains_single_candidate_and_all_three_axes():
    values=[object() for _ in range(6)];r,e,s,nr,ne,ns=values
    assert component_configs(r,e,s,candidate_expert=ne)==dict(baseline=(r,e,s),candidate=(r,ne,s))
    assert component_configs(r,e,s,candidate_router=nr)==dict(baseline=(r,e,s),router=(nr,e,s))
    variants=component_configs(r,e,s,candidate_expert=ne,candidate_shared=ns,candidate_router=nr)
    assert len(variants)==8 and variants['combined']==(nr,ne,ns)
    assert variants['experts_router']==(nr,ne,s) and variants['shared']==(r,e,ns)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Assigned RTX GPU required')
def test_combined_runtime_winners_load_then_bind_without_retuning(tmp_path,monkeypatch):
    from torch._dynamo.testing import CompileCounterWithBackend
    from rtx._moe_selection import clear_selection_cells
    from rtx.autotune import moe as experts, varlen_routing as routing
    from rtx.autotune.winners import save_runtime_winner
    from rtx.autotune.core import stable_id
    from rtx.moe import BalancedMoE, _packed_expert_spec, _parse_spec
    from rtx._routing_autotune import parse_spec
    from rtx._routing_varlen_autotune import _clear_runtime_caches

    torch.manual_seed(15329)
    router=RoutingKernelConfig(qb_backend='native_column',metadata_backend='cute',quota_threshold=0)
    model=BalancedMoE(128,8,128,top_k=2,shared_hidden_features=128,router_dim=16,
        device='cuda',config=DEFAULT_MOE_CONFIG,autotune='cache',autotune_cache_dir=tmp_path,
        router_options={'config':router})
    # This test isolates router/expert cache composition, not shared-MLP tuning.
    model._shared_spec=None
    with torch.no_grad():model.router.projection_weight.normal_(std=.03)
    x=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    cu=torch.tensor([0,128],device='cuda',dtype=torch.int32)
    with torch.no_grad():
        scores=model.router.log_affinities(x)
        route=model.router(x,cu_seqlens=cu)
    shape=(128,128,8,route.expert_ids.shape[1],2,128)
    selected_router,selected_expert=composition_configs(shape,router,DEFAULT_MOE_CONFIG)['combined']
    key,root,_,_=routing.contract(scores,cu,parse_spec(model.router._selection_spec),True,False)
    save_runtime_winner(key,selected_router.to_dict(),root=root,config_id=stable_id(selected_router.to_dict()))
    spec=_packed_expert_spec(model._expert_spec,1)
    operands=(x,model.gate_up.weight,model.down.weight,*route)
    _,key,root,_=experts._contract(operands,_parse_spec(spec),True)
    save_runtime_winner(key,selected_expert.to_dict(),root=root,config_id=stable_id(selected_expert.to_dict()))
    _clear_runtime_caches();experts._clear_runtime_caches();clear_selection_cells()
    def forbidden(*args,**kwargs):raise AssertionError('Cache composition unexpectedly tuned')
    monkeypatch.setattr(routing,'tune',forbidden)
    monkeypatch.setattr(experts,'tune_moe',forbidden)
    counter=CompileCounterWithBackend('aot_eager')
    compiled=torch.compile(model,backend=counter,fullgraph=True,dynamic=False)
    dy=torch.randn_like(x);leaves=(x,*model.parameters())
    values=[]
    for _ in range(3):
        y=compiled(x,cu_seqlens=cu)
        values.append((y.detach(),*[g.detach() for g in torch.autograd.grad(y,leaves,dy)]))
    compare(values[1],values[0]);compare(values[2],values[0])
    assert len(counter.graphs)==2
    assert 'selected_varlen_route' in counter.graphs[0].code
    assert 'selected_varlen_route' not in counter.graphs[1].code
    assert 'moe_select' not in counter.graphs[1].code
    assert any(c.prepared.joint_recompute_engine=='dual_operand' for c in experts._choices.values())
    assert any(c.packed_sinkhorn_backend=='persistent' for c in __import__(
        'rtx._routing_varlen_autotune',fromlist=['_SELECTIONS'])._SELECTIONS.values())
    _clear_runtime_caches();experts._clear_runtime_caches()
