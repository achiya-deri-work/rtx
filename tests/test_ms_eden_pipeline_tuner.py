from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from benchmarks import tune_ms_eden_pipeline as tuner
from rtx.autotune.ms_eden_pipeline import make_ms_eden_pipeline_adapter


def test_presets_and_bounded_priority_portfolio():
    adapter = make_ms_eden_pipeline_adapter((512,512,512), lambda _: {})
    pool = tuner.candidates(adapter, 64)
    assert pool[0] == adapter.initial_config
    assert pool[1].backward_pipeline == "paired"
    assert pool[2].dx.amax_reduction == pool[2].dw.amax_reduction == "staged"
    assert pool[3].dx.producer_stages == pool[3].dw.producer_stages == 2
    assert pool[3].dx.producer_async_copy and pool[3].dw.producer_async_copy
    assert len(pool) <= 64 and len({adapter.config_id(c) for c in pool}) == len(pool)
    assert all(adapter.rejection(c) is None for c in pool)
    assert len(tuner.candidates(adapter, 2)) == 2


def test_worker_command_preserves_full_config_and_complete_scope(tmp_path):
    args = tuner.parser().parse_args(["--output", str(tmp_path)])
    config = tuner.preset("paired")
    command = tuner.evaluation_command(args, config, tmp_path/"result.json")
    assert command[command.index("--scope")+1] == "backward"
    assert command[command.index("--candidate-json")+1] == "{}"
    assert command[command.index("--candidate-pipeline")+1] == "paired"
    assert json.loads(command[command.index("--candidate-dw-json")+1]) == asdict(config.dw)
    assert "--rounds" in command and "--min-batch-ms" in command
    assert float(command[command.index("--warmup-ms")+1]) == 3000


def test_small_paired_portfolio_covers_buffering_resources_and_both_gemms():
    initial = tuner.preset("paired")
    adapter = make_ms_eden_pipeline_adapter((4096,2048,512), lambda _: {}, initial=initial)
    pool = tuner.candidates(adapter, 24)
    assert pool[0] == initial
    for stages in (2, 3):
        assert any(c.backward_pipeline == "paired" and
                   c.dx.producer_stages == c.dw.producer_stages == stages and
                   c.dx.producer_async_copy and c.dw.producer_async_copy for c in pool[:8])
    for blocks in (64, 128, 512):
        assert any(c.backward_pipeline == "paired" and
                   c.dx.producer_blocks == c.dw.producer_blocks == blocks for c in pool[:16])
    for phase, other in (("dx", "dw"), ("dw", "dx")):
        assert any(c.backward_pipeline == "paired" and
                   getattr(c, phase).gemm != getattr(initial, phase).gemm and
                   getattr(c, other) == getattr(initial, other) for c in pool[:16])
        assert any(c.backward_pipeline == "paired" and
                   getattr(c, phase).producer_blocks == 64 and
                   getattr(c, other) == getattr(initial, other) for c in pool[:16])
    assert len(pool) == 24
    assert len({adapter.config_id(c) for c in pool}) == len(pool)
    assert all(adapter.rejection(c) is None for c in pool)


def test_coordinate_tail_interleaves_gradient_legs():
    adapter = make_ms_eden_pipeline_adapter((4096,2048,512), lambda _: {},
        initial=tuner.preset("paired"))
    pool = tuner.candidates(adapter, 64)
    initial = adapter.initial_config
    for phase in ("dx", "dw"):
        assert any(getattr(c, phase).producer_threads == 64 for c in pool[:24])
    assert pool[0] == initial


def test_dry_run_never_probes_cuda_or_creates_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tuner, "runtime_environment", lambda: (_ for _ in ()).throw(AssertionError("CUDA probe")))
    output = tmp_path/"absent"
    tuner.main(["--dry-run", "--trials", "2", "--pool-size", "8", "--output", str(output)])
    result = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert result["automatic_promotion"] is False
    assert "dw.producer_stages" in result["axes"]
    assert result["candidates"][1]["backward_pipeline"] == "paired"


def test_serial_worker_failure_artifact(monkeypatch, tmp_path):
    args = tuner.parser().parse_args(["--output", str(tmp_path)])
    seen=[]
    def fail(command, **kw):
        seen.append((command, kw))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(tuner.subprocess, "run", fail)
    result = tuner.evaluate_subprocess(args, tuner.preset("baseline"), tmp_path)
    assert result["status"] == "runtime_error" and "worker_failure" in result
    assert len(seen) == 1 and seen[0][1]["timeout"] == 600
    assert list((tmp_path/"trials").glob("*/manifest.json"))


def test_first_hit_frontend_uses_residual_store_without_promotion(monkeypatch, tmp_path):
    monkeypatch.setattr(tuner, "runtime_environment", lambda: ({"name":"fake"}, {"torch":"fake"}))
    seen=[]
    def fake(adapter, store, portfolio, **kwargs):
        seen.append((adapter,store,portfolio,kwargs))
        return SimpleNamespace(config=portfolio[1],median_ms=.1)
    monkeypatch.setattr(tuner,"tune_first_hit",fake)
    tuner.main(["--trials","2","--pool-size","8","--output",str(tmp_path)])
    adapter,store,pool,kwargs=seen[0]
    assert kwargs["policy"].max_trials==2 and kwargs["policy"].confirmation_repeats==0
    assert store.path==tmp_path/"residuals"
    selection=json.loads(next(tmp_path.glob("selection-*.json")).read_text())
    assert selection["automatic_promotion"] is False
    assert selection["config"]["backward_pipeline"]=="paired"


def test_real_first_hit_jsonl_transaction_completes_without_nested_flock(tmp_path):
    """Exercise the real orchestration/locking stack under a hard CPU timeout."""
    code = '''
import sys
from benchmarks import tune_ms_eden_pipeline as tuner
from tests.test_ms_eden_pipeline_autotune import evidence
tuner.runtime_environment = lambda: ({"name": "cpu-lock-test"}, {"torch": "fake"})
def evaluate(args, config, root):
    if sys.argv[2] == "cache":
        raise AssertionError("compatible persistent hit must not launch a worker")
    result = evidence(config)
    result["artifact"] = "cpu-synthetic-integration-only"
    return result
tuner.evaluate_subprocess = evaluate
tuner.main(["--trials", "2", "--rounds", "4", "--pool-size", "8",
            "--seconds", "10", "--output", sys.argv[1]])
'''
    for phase in ("cold", "cache"):
        result = subprocess.run([sys.executable, "-c", code, str(tmp_path), phase],
            cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            text=True, capture_output=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in (tmp_path/"residuals"/"observations.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert {record["config"]["backward_pipeline"] for record in records} == {"serial", "paired"}
    assert all(record["outcome"]["status"] == "ok" for record in records)
    selections = [json.loads(path.read_text()) for path in tmp_path.glob("selection-*.json")]
    assert len(selections) == 2
    assert selections[0]["config"] == selections[1]["config"]
    assert selections[0]["median_ms"] == selections[1]["median_ms"]
