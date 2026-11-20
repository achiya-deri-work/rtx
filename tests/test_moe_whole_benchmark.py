"""Whole-layer benchmark attribution and source-parity bookkeeping."""
import pytest
import torch

from benchmarks.benchmark_moe_execution import _whole_validation,_python_source_digest,_apply_router_config,parser


def test_decomposition_requires_explicit_whole_layer_parser_flag(tmp_path):
    args=parser().parse_args(["compare","--baseline","old","--candidate","new",
        "--output",str(tmp_path),"--whole-layer","--decompose","--include-bf16",
        "--candidate-router-config","selected-router.json"])
    assert args.whole_layer and args.decompose and args.include_bf16
    assert args.candidate_router_config.name=="selected-router.json"


def test_whole_source_identity_includes_router_files(tmp_path):
    source=tmp_path/"rtx";source.mkdir()
    (source/"moe.py").write_text("fixed = 1\n")
    (source/"routing.py").write_text("router = 1\n")
    before=_python_source_digest(tmp_path)
    (source/"routing.py").write_text("router = 2\n")
    assert _python_source_digest(tmp_path)!=before


def test_router_override_is_applied_equally_to_both_precision_baselines():
    from types import SimpleNamespace
    class Router:
        def __init__(self):self.refreshed=0
        def _refresh_selection_spec(self):self.refreshed+=1
    native=SimpleNamespace(_orig_mod=SimpleNamespace(router=Router()))
    bfn=SimpleNamespace(_orig_mod=SimpleNamespace(router=Router()))
    calls={"whole_step":lambda:native,"bf16_whole_step":lambda:bfn}
    config=SimpleNamespace(forward=object())
    assert _apply_router_config(calls,config) is native._orig_mod
    for model in (native._orig_mod,bfn._orig_mod):
        assert model.router.config is config and model.router.sinkhorn_config is config.forward
        assert model.router.autotune=="off" and model.router.refreshed==1


def test_all_parameter_reference_checks_every_gradient(tmp_path):
    path=tmp_path/"reference.pt"
    output=torch.randn(4,3,dtype=torch.bfloat16)
    gradients=(torch.randn_like(output),torch.randn(3,3),torch.randn(()))
    names=("projection","temperature")
    record=_whole_validation(output,gradients,names,save_reference=path)
    assert record["count"]==4 and record["all_finite"]
    same=_whole_validation(output,gradients,names,reference=path)
    assert all(error["max_abs"]==0 for error in same["errors"].values())
    changed=(*gradients[:-1],gradients[-1]+1.)
    with pytest.raises(AssertionError):_whole_validation(output,changed,names,reference=path)
    with pytest.raises(ValueError,match="names/order"):
        _whole_validation(output,gradients,tuple(reversed(names)),reference=path)


def test_whole_reference_rejects_missing_nonfinite_or_dtype_changed_gradient(tmp_path):
    output=torch.ones(2,2);gradients=(torch.ones(2,2),torch.ones(()))
    with pytest.raises(ValueError,match="arity"):_whole_validation(output,gradients,())
    with pytest.raises(AssertionError,match="nonfinite"):
        _whole_validation(output,(gradients[0],torch.tensor(float("nan"))),("weight",))
    path=tmp_path/"ref.pt"
    _whole_validation(output,gradients,("weight",),save_reference=path)
    with pytest.raises(ValueError,match="contract"):
        _whole_validation(output,(gradients[0],gradients[1].double()),("weight",),reference=path)
