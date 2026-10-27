"""CPU contracts for the fresh-process graph comparator; no GPU timing claims."""
import json
from pathlib import Path
import sys

import pytest

from benchmarks import compare_mixed_nvfp4_backward as bench


def arguments(*extra):
    return bench.parser().parse_args(["--output","unused.json",*extra])


def test_shared_data_has_independent_autograd_leaf_identity():
    import torch
    inputs = (torch.randn(4,6).T.requires_grad_(),torch.randn(3,4).requires_grad_(),torch.randn(6,3))
    aliases = bench.independent_shared_leaves(inputs)
    for original,alias in zip(inputs,aliases):
        assert alias is not original and alias.is_leaf
        assert alias.data_ptr() == original.data_ptr()
        assert alias.stride() == original.stride()
        assert alias.requires_grad == original.requires_grad
        torch.testing.assert_close(alias,original,rtol=0,atol=0)
    (aliases[0].square().sum()+aliases[1].square().sum()).backward()
    assert inputs[0].grad is None and inputs[1].grad is None
    torch.testing.assert_close(aliases[0].grad,2*inputs[0])
    with torch.no_grad():
        inputs[0].add_(1)
    torch.testing.assert_close(inputs[0],aliases[0],rtol=0,atol=0)
    assert inputs[0]._version == aliases[0]._version


def test_four_policies_are_full_independent_cross_product():
    assert set(bench.POLICIES.values())=={(a,b) for a in ("ms_eden","mxfp8") for b in ("ms_eden","mxfp8")}


@pytest.mark.parametrize("shape",[(128,128,128),(2048,768,1536),(8192,768,1536)])
def test_actual_dynamic_forward_config_is_exact_not_only_scale_config(shape):
    assert bench.matched_forward_config(*shape).tensor_scale_mode=="exact"


@pytest.mark.parametrize("extra",[("--m","0"),("--n","129"),("--k","-128"),
                                   ("--samples","2"),("--warmup","1"),("--calls-per-sample","0"),
                                   ("--schedule-json",'{"reader_mode":"invalid"}')])
def test_invalid_protocol_fails_on_cpu(extra):
    with pytest.raises(ValueError):
        bench.validate(arguments(*extra))


def test_worker_command_preserves_protocol_and_configs():
    args=arguments("--m","256","--dx-schedule-json",'{"producer_blocks":64}',
                   "--dw-schedule-json",'{"producer_blocks":96}',"--mxfp8-config-json","{}")
    command=bench.worker_command(args,"fp4_mxfp8",Path("child.json"))
    assert command[:3]==[sys.executable,"-m","benchmarks.compare_mixed_nvfp4_backward"]
    child=bench.parser().parse_args(command[3:])
    assert child.worker and child.policy=="fp4_mxfp8"
    for name in ("m","n","k","seed","schedule_json","dx_schedule_json","dw_schedule_json","mxfp8_config_json"):
        assert getattr(child,name)==getattr(args,name)


def test_stationarity_uses_repository_gate():
    good=bench.timing_summary([1.]*9,{"actual_calls_per_sample":4})
    assert good["sampling"]["collection"]["stationary"] is True
    bad=bench.timing_summary([1.,2.],{})
    assert bad["sampling"]["collection"]["stationary"] is False
    assert good["median_ms"]==good["min_ms"]==1.


def test_paired_statistics_cluster_rounds_and_ratio_direction():
    rounds=[dict(reference_ms=2.,candidate_ms=1.) for _ in range(12)]
    result=bench.paired_statistics(rounds,bootstrap_samples=100)
    assert result["geomean_reference_over_candidate"]==pytest.approx(2.)
    assert result["round_cluster_bootstrap95"]==pytest.approx([2.,2.])
    assert result["rounds"]==12


def test_paired_statistics_reproducible():
    rounds=[dict(reference_ms=v,candidate_ms=1.) for v in (1.,2.,3.,4.)]
    assert bench.paired_statistics(rounds,seed=1,bootstrap_samples=100)==bench.paired_statistics(rounds,seed=1,bootstrap_samples=100)


@pytest.mark.parametrize("value",[0.,-1.,float("inf"),float("nan")])
def test_paired_statistics_reject_invalid_timings(value):
    with pytest.raises(ValueError,match="finite and positive"):
        bench.paired_statistics([dict(reference_ms=value,candidate_ms=1.)])


def test_paired_cluster_uses_round_aggregate_not_individual_window():
    rounds=[dict(reference_ms=2.,candidate_ms=1.,windows=[dict(ratio=1.),dict(ratio=4.)]) for _ in range(12)]
    summary=bench.paired_statistics(rounds,bootstrap_samples=100)
    assert summary["rounds"]==12
    assert summary["round_cluster_bootstrap95"]==pytest.approx([2.,2.])


def test_all_policies_launch_fresh_workers(monkeypatch,tmp_path):
    commands=[]
    def run(command,check):
        assert check
        commands.append(command)
        args=bench.parser().parse_args(command[3:])
        Path(args.output).write_text(json.dumps({"policy":args.policy}))
    monkeypatch.setattr(bench.subprocess,"run",run)
    output=tmp_path/"result.json"
    bench.main(["--output",str(output)])
    result=json.loads(output.read_text())
    assert result["fresh_process_per_policy"]
    assert set(result["results"])==set(bench.POLICIES)
    assert len(commands)==len(bench.POLICIES)==5
    assert len({bench.parser().parse_args(command[3:]).output for command in commands})==5


def test_refuses_overwrite_before_launch(monkeypatch,tmp_path):
    output=tmp_path/"result.json"
    output.write_text("retained")
    monkeypatch.setattr(bench.subprocess,"run",lambda *a,**kw:pytest.fail("must not launch"))
    with pytest.raises(FileExistsError):
        bench.main(["--output",str(output)])
    assert output.read_text()=="retained"


def test_source_hashes_cover_dirty_runtime():
    hashes=bench.source_hashes()
    assert "rtx/experimental_ms_eden.py" in hashes
    assert "rtx/kernels/nvfp4_posthoc.py" in hashes
    assert all(len(value)==64 for value in hashes.values())
