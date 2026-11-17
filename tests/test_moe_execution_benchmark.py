"""CPU-only measurement bookkeeping; no CUDA work during these tests."""
import copy

import pytest

from benchmarks.benchmark_moe_execution import aggregate, parser, summary, _tensor_digest


def record(label, milliseconds):
    observation = dict(median_ms=milliseconds)
    return dict(status="complete", label=label, input_digests=dict(x="fixed"),
                timings=dict(public_expert_step=dict(host=[observation], graph=[observation])))


def test_source_abba_summary_and_mad():
    rows = aggregate([record("baseline", 2), record("candidate", 1),
                      record("candidate", 1.2), record("baseline", 2.2)])
    assert {r["execution"] for r in rows} == {"host", "graph"}
    assert all(r["baseline_ms"] == 2.1 and r["candidate_ms"] == 1.1 for r in rows)
    assert all(r["speedup"] == pytest.approx(2.1 / 1.1) for r in rows)
    assert summary([1, 2, 3])["relative_mad"] == .5


def test_comparison_rejects_different_inputs_parameters_or_incomplete_worker():
    a = record("baseline", 2)
    b = record("candidate", 1)
    for field, value in (("input_digests", dict(x="different")),
                         ("whole_parameter_digests", ["different"]),
                         ("status", "running")):
        changed = copy.deepcopy(b)
        changed[field] = value
        with pytest.raises(ValueError):
            aggregate([a, changed])
    with pytest.raises(ValueError, match="Both source"):
        aggregate([a])


def test_benchmark_parser_retains_explicit_source_and_per_source_config(tmp_path):
    args = parser().parse_args(["compare", "--baseline", str(tmp_path/"old"),
        "--candidate", str(tmp_path/"new"), "--output", str(tmp_path/"results"),
        "--baseline-config", "old.json", "--candidate-config", "new.json",
        "--whole-layer", "--tokens", "8192"])
    assert args.whole_layer and args.tokens == 8192
    assert args.baseline_config.name == "old.json" and args.candidate_config.name == "new.json"


def test_input_digest_supports_scalar_router_parameters():
    import torch
    value = torch.tensor(1., dtype=torch.float32)
    assert _tensor_digest(value) == _tensor_digest(value.clone())
    assert _tensor_digest(value) != _tensor_digest(torch.tensor(2.))


def test_controller_preserves_venv_python_symlink(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import benchmarks.benchmark_moe_execution as benchmark

    real_python = tmp_path / "system-python"
    real_python.touch()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        label = command[command.index("--label")+1]
        output = Path(command[command.index("--output")+1])
        output.write_text(json.dumps(record(label, 2 if label == "baseline" else 1)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    args = parser().parse_args(["compare", "--baseline", str(tmp_path/"old"),
        "--candidate", str(tmp_path/"new"), "--output", str(tmp_path/"results"),
        "--python", str(venv_python)])
    benchmark.compare(args)
    assert len(commands) == 4
    assert all(command[0] == str(venv_python.absolute()) for command in commands)
    assert commands[0][0] != str(venv_python.resolve())
