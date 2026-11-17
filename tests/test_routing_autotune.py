"""Shared-engine integration; numerical CUDA tests live in test_routing.py."""
from dataclasses import replace
import json

import pytest
import torch

from rtx.routing import BalancedRouter,RoutingKernelConfig,SinkhornKernelConfig
from rtx.autotune.routing import make_routing_adapter,structural_seeds
from rtx.autotune import AutotunePolicy
from rtx.autotune.effort import make_effort_autotuner
from rtx.autotune.store import InMemoryTuningStore
from rtx.autotune.outcomes import TrialOutcome


def test_config_serialization_and_independent_backward_geometry():
    c=RoutingKernelConfig(forward=SinkhornKernelConfig(64,2,4),
        backward=SinkhornKernelConfig(256,8,1),qb_backend='native',qb_threads=256,quota_threshold=4096)
    assert RoutingKernelConfig.from_dict(c.to_dict())==c
    assert RoutingKernelConfig.from_token(c.to_token())==c
    r=BalancedRouter(32,16,top_k=4,config=c)
    assert r.autotune=='off' and r.config==c
    with pytest.raises(ValueError):BalancedRouter(32,16,config=c,sinkhorn_config=SinkhornKernelConfig())


def test_shared_adapter_axes_do_not_change_routing_policy():
    adapter=make_routing_adapter(40000,128,8,lambda c:TrialOutcome('ok',median_ms=1.),
                                 device=dict(multiprocessor_count=36,memory_bus_width_bits=192))
    assert all('iterations' not in name and 'rounds' not in name for name in adapter.axes)
    assert 'backward.threads' in adapter.axes and 'qb_backend' in adapter.axes
    assert adapter.rejection(replace(adapter.initial_config,qb_backend='native')) is None
    for _,_,c in adapter.neighbors(adapter.initial_config):
        assert adapter.deserialize(adapter.serialize(c))==c
    inference=make_routing_adapter(68,16,4,lambda c:TrialOutcome('ok',median_ms=1.),phase='forward')
    assert not any(name.startswith('backward.') for name in inference.axes)
    # Threshold choices and inactive QB thread counts must not burn repeated
    # trials on the same effective execution path.
    same=replace(adapter.initial_config,quota_threshold=1,qb_threads=64)
    assert adapter.config_id(same)==adapter.config_id(adapter.initial_config)


def test_shared_ucb_search_resumes_and_keeps_prior_best():
    def evaluate(c):
        return TrialOutcome('ok',median_ms=(1 if c.qb_backend=='native' else 2)+c.forward.row_groups*.01)
    adapter=make_routing_adapter(68,16,4,evaluate)
    store=InMemoryTuningStore()
    policy=AutotunePolicy(max_trials=4,max_seconds=20,confirmation_repeats=0,
                           search_schedule='progressive',candidate_mix='mixed')
    first=make_effort_autotuner(adapter,store,policy,candidates=list(structural_seeds(adapter.initial_config))).tune()
    second=make_effort_autotuner(adapter,store,replace(policy,max_trials=8),
                                candidates=list(structural_seeds(adapter.initial_config))).tune()
    assert len(store.observations)==8
    assert second.median_ms<=first.median_ms
    assert second.tuning_evidence['attempted_trials']==8


def test_collection_cli_reports_ineligible_timing_without_fake_winner(tmp_path,monkeypatch):
    import rtx.routing as frontend
    import rtx.autotune.routing as tuning
    from rtx.autotune.orchestrator import NoEligibleTimingError
    class Model:
        def __init__(self,*a,**kw):self.projection_weight=torch.empty(1)
        def autotune_(self,*a,**kw):
            from pathlib import Path
            Path(kw['output_dir']).mkdir(parents=True,exist_ok=True)
            raise NoEligibleTimingError(RoutingKernelConfig())
    real_randn=torch.randn
    def cpu_randn(*a,**kw):
        kw['device']='cpu'
        return real_randn(*a,**kw)
    monkeypatch.setattr(frontend,'BalancedRouter',Model)
    monkeypatch.setattr(torch,'randn',cpu_randn)
    monkeypatch.setattr('sys.argv',['routing','--tokens','64','--output',str(tmp_path)])
    with pytest.raises(SystemExit) as result:tuning.main()
    assert result.value.code==2
    reports=list(tmp_path.glob('timing-ineligible-*.json'))
    assert len(reports)==1
    assert json.loads(reports[0].read_text())==dict(status='timing_ineligible',
        correctness_checked_seed=True,residuals_retained=True,persistent_winner=False)


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='assigned RTX CUDA required')


@cuda
def test_first_hit_compile_disk_reuse_and_immutable_backward(tmp_path,monkeypatch):
    import rtx.autotune.routing as tuning
    import rtx._routing_autotune as startup
    import subprocess
    real_run=subprocess.run
    def no_nvcc(command,*a,**kw):
        # Inductor can query NVCC's version while recording diagnostic metadata.
        # That is not a production extension build and must not mask compilation
        # diagnostics. Reject every actual NVCC compilation invocation instead.
        if (isinstance(command,(list,tuple)) and command and 'nvcc' in str(command[0])
                and list(command[1:]) not in (['--version'],['-V'])):
            raise AssertionError('Production routing attempted an NVCC build')
        return real_run(command,*a,**kw)
    monkeypatch.setattr(subprocess,'run',no_nvcc)
    monkeypatch.setenv('CUDA_TOOLKIT_PATH','/no-routing-nvcc-toolkit')
    def check_routes(route,n,e,k):
        assert route.expert_ids.shape==route.route_weights.shape
        assert route.inverse.shape==(n,k)
        torch.testing.assert_close(route.route_weights.flatten()[route.inverse].sum(1),
                                   torch.ones(n,device=route.route_weights.device))
    torch.manual_seed(753)
    policy=AutotunePolicy(max_trials=4,max_seconds=60,samples=3,calls_per_sample=1,
                           warmup=1,confirmation_repeats=0,search_schedule='progressive',candidate_mix='mixed')
    r=BalancedRouter(16,16,top_k=4,router_dim=8,device='cuda',autotune='balanced',
                      tuning_policy=policy,autotune_cache_dir=tmp_path)
    with torch.no_grad():r.projection_weight.normal_(std=.05)
    x=torch.randn(65,16,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    fn=torch.compile(r,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    route=fn(x);check_routes(route,65,16,4)
    winners=list(tmp_path.glob('runtime_winners/balanced_routing/*/*.json'))
    assert len(winners)==1
    winner=json.loads(winners[0].read_text())
    assert winner['schema_version']==2 and winner['metadata']['tuning_evidence']['attempted_trials']<=4
    # A completed forward owns its execution token even after all mutable
    # selections are cleared; backward never retunes or reads CUDA scalars.
    startup._clear_runtime_caches()
    def forbidden(*a,**kw):raise AssertionError('A cache hit/backward unexpectedly started tuning')
    monkeypatch.setattr(tuning,'tune_routing',forbidden)
    (route.route_weights*torch.randn_like(route.route_weights)).sum().backward()
    assert x.grad is not None and bool(x.grad.isfinite().all())
    again=fn(x);check_routes(again,65,16,4)


@cuda
def test_selected_operator_contracts_off_mode():
    from rtx._routing_autotune import selected_sinkhorn,selected_hard
    r=BalancedRouter(16,16,top_k=4,router_dim=8,autotune='off')
    scores=torch.randn(68,16,device='cuda',requires_grad=True)
    assert set(torch.library.opcheck(selected_sinkhorn,(scores,4,5,r._selection_spec,True,65)).values())=={'SUCCESS'}
    _,_,token=selected_sinkhorn(scores,4,5,r._selection_spec,True,65)
    assert token.device.type=='cpu'
    assert set(torch.library.opcheck(selected_hard,(scores.detach().exp(),token,4,5,64,0)).values())=={'SUCCESS'}
    with pytest.raises(ValueError,match='CPU int64'):
        selected_hard(scores.detach().exp(),token.cuda(),4,5,64,0)
