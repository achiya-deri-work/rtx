"""The recomputation baseline checkpoints FFNs, never the learned router."""
import pytest
import torch

from benchmarks.benchmark_moe_checkpoint import BF16FFNReference,compare_routing_identity,gradient_clearer,metrics,parser


def _model():
    from rtx.moe import BalancedMoE
    torch.manual_seed(1921)
    model=BalancedMoE(8,4,8,top_k=2,shared_hidden_features=8,router_dim=4,backend='torch',autotune='off')
    with torch.no_grad():model.router.projection_weight.normal_(std=.1)
    return model


def test_parser_exposes_matched_checkpoint_baseline_without_implicit_tuning():
    args=parser().parse_args(['--source','snapshot','--output','out.json'])
    assert args.modes==['mxfp8_recompute','bf16_checkpoint','bf16_saved']
    assert not args.nonreentrant and not hasattr(args,'tune_trials')
    assert args.grad_mode=='set_to_none'


@pytest.mark.parametrize('reentrant',[True,False])
def test_checkpoint_only_ffns_and_preserve_all_parameter_gradients(reentrant):
    model=_model();params=tuple(model.parameters())
    saved=BF16FFNReference(model,checkpoint_ffns=False)
    checkpointed=BF16FFNReference(model,checkpoint_ffns=True,use_reentrant=reentrant)
    x=torch.randn(7,8,dtype=torch.bfloat16,requires_grad=True);dy=torch.randn_like(x)
    out=saved(x);out.backward(dy)
    expected=[out.detach(),x.grad.detach().clone(),*[p.grad.detach().clone() for p in params]]
    model.zero_grad(set_to_none=True);x.grad=None
    calls=[]
    handle=model.router.register_forward_hook(lambda *args:calls.append(1))
    try:
        out=checkpointed(x);out.backward(dy)
    finally:handle.remove()
    actual=[out.detach(),x.grad,*[p.grad for p in params]]
    assert len(calls)==1,'Checkpoint backward must not rerun Sinkhorn or repair'
    assert all(value is not None for value in actual)
    for value,reference in zip(actual,expected):
        torch.testing.assert_close(value,reference,rtol=.01,atol=2e-7)
    assert sum(p.dtype==torch.float32 for p in params)==2


def test_metrics_rejects_nonfinite_missing_or_wrong_gradients():
    a=[torch.ones(3),torch.ones(())]
    assert all(row['nrms']==0 for row in metrics(a,a,gate=.001))
    with pytest.raises(ValueError):metrics(a,a[:1])
    with pytest.raises(AssertionError):metrics([torch.zeros(3),torch.ones(())],a,gate=.001)
    with pytest.raises(AssertionError):metrics([torch.full((3,),float('nan')),torch.ones(())],a)


def test_discrete_routing_identity_is_exact_not_a_gradient_tolerance():
    identity=dict(expert_ids=torch.tensor([0,1]),inverse=torch.tensor([[0],[1]]))
    assert compare_routing_identity(identity,identity)==dict(expert_ids_exact=True,inverse_exact=True)
    with pytest.raises(AssertionError):
        compare_routing_identity(identity,{**identity,'inverse':torch.tensor([[1],[0]])})
    with pytest.raises(ValueError):compare_routing_identity(identity,{'inverse':identity['inverse']})


def test_gradient_clearer_distinguishes_fresh_and_accumulated_training():
    values=[torch.randn(2,requires_grad=True),torch.randn((),requires_grad=True)]
    for value in values:value.grad=torch.ones_like(value)
    gradient_clearer(values,'set_to_none',{})()
    assert all(value.grad is None for value in values)
    clear=gradient_clearer(values,'preallocated_zero',dict(backend='eager',fullgraph=True))
    addresses=[value.grad.data_ptr() for value in values]
    for value in values:value.grad.fill_(2)
    clear()
    assert addresses==[value.grad.data_ptr() for value in values]
    assert all(bool((value.grad==0).all()) for value in values)
    with pytest.raises(ValueError):gradient_clearer(values,'unknown',{})


def test_checkpoint_reference_preserves_independent_packed_pools():
    model=_model();x=torch.randn(7,8,dtype=torch.bfloat16,requires_grad=True)
    cu=torch.tensor([0,3,3,7],dtype=torch.int32);dy=torch.randn_like(x)
    wrapper=BF16FFNReference(model,checkpoint_ffns=True)
    output=wrapper(x,cu_seqlens=cu);output.backward(dy)
    actual=[output.detach(),x.grad.clone(),*[p.grad.clone() for p in model.parameters()]]
    model.zero_grad(set_to_none=True);x.grad=None
    reference=BF16FFNReference(model,checkpoint_ffns=False)(x,cu_seqlens=cu);reference.backward(dy)
    for a,b in zip(actual,[reference,x.grad,*[p.grad for p in model.parameters()]]):
        torch.testing.assert_close(a,b,rtol=.01,atol=2e-7)


def test_varlen_benchmark_declares_solver_baselines_and_bounded_search():
    from benchmarks.benchmark_moe_varlen import parser as packed_parser
    args=packed_parser().parse_args(['--source','snapshot','--output','out.json','--oracle'])
    assert args.lengths==[512,2000,768,4096]
    assert {'bf16_checkpoint_step','bf16_saved_step','router_step','mxfp8_step'}<=set(args.phases)
    assert not any((args.tune_router_trials,args.tune_expert_trials,args.tune_shared_trials))
    assert args.oracle and args.profile=='none'


def test_matrix_target_requires_every_matched_round_not_a_lucky_median():
    from benchmarks.tune_moe_varlen_matrix import summarize
    record=dict(timings=dict(mxfp8_step=[dict(median_ms=v) for v in (1.,1.,1.2)],
        bf16_checkpoint_step=[dict(median_ms=1.3) for _ in range(3)]))
    result=summarize(record,1.2)
    assert result['speedup_checkpoint']==1.3 and not result['meets_target']
    assert summarize(record,1.)['meets_target']
    record['timings']['bf16_checkpoint_step'].pop()
    with pytest.raises(ValueError):summarize(record,1.2)


def test_hidden_scale_changes_only_routed_width_with_truthful_identity():
    import json
    from pathlib import Path
    from benchmarks.benchmark_moe_varlen import scaled_case,parser as packed_parser
    cases=json.loads((Path(__file__).parents[1]/'benchmarks/balanced_moe_cases.json').read_text())['cases']
    assert [scaled_case(c,1.5,64)['expert_hidden_dim'] for c in cases]==[192,192,192,384]
    for c in cases:
        before=dict(c);wide=scaled_case(c,1.5,64)
        assert c==before and scaled_case(c)==c
        assert wide['id'].endswith('_h'+str(wide['expert_hidden_dim']))
        assert {k:v for k,v in wide.items() if k not in ('id','expert_hidden_dim')}=={
            k:v for k,v in c.items() if k not in ('id','expert_hidden_dim')}
    assert scaled_case(cases[0],1.51,64)['expert_hidden_dim']==256
    for scale in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):scaled_case(cases[0],scale,64)
    for alignment in (0,-1,1.5,True):
        with pytest.raises(ValueError):scaled_case(cases[0],1.5,alignment)
    args=packed_parser().parse_args(['--source','snapshot','--output','out.json',
        '--expert-hidden-scale','1.5','--hidden-alignment','64'])
    assert args.expert_hidden_scale==1.5 and args.hidden_alignment==64
