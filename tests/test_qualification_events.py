import json
import os
from pathlib import Path
import subprocess
import sys


def test_failure_events_are_durable_and_not_overwritten(tmp_path):
    root = Path(__file__).resolve().parents[1]
    probe = tmp_path / "test_probe.py"
    probe.write_text("def test_pass(): pass\ndef test_fail(): assert False, 'diagnostic sentinel'\n")
    events = tmp_path / "events.jsonl"
    pythonpath = os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH"))))
    env = dict(os.environ, PYTHONPATH=pythonpath, RTX_QUALIFICATION_EVENTS=str(events))
    command = [sys.executable, "-m", "pytest", "-q", "-p", "tools.qualification_events", str(probe)]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert rows[0]["event"] == "session_start"
    failed = [r for r in rows if r.get("outcome") == "failed"]
    assert len(failed) == 1 and "diagnostic sentinel" in failed[0]["traceback"]
    assert rows[-1]["event"] == "session_finish" and rows[-1]["exitstatus"] == 1
    original = events.read_bytes()
    repeated = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
    assert events.read_bytes() == original
