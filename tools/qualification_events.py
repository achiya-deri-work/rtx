"""Optional pytest plugin recording failures immediately, before suite teardown.

Enable with ``-p tools.qualification_events`` and RTX_QUALIFICATION_EVENTS.
The JSONL file is exclusive per run; an interrupted tail is not a passing result.
"""
import json
import os
from pathlib import Path
import time

import pytest

_path = None


def _emit(row):
    if _path is None:
        return
    row = dict(timestamp_ns=time.time_ns(), **row)
    with _path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        if row.get("outcome") == "failed" or row.get("event") == "session_finish":
            os.fsync(stream.fileno())


def pytest_sessionstart(session):
    global _path
    value = os.environ.get("RTX_QUALIFICATION_EVENTS")
    _path = None
    if value:
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.touch(exist_ok=False)
        except FileExistsError as error:
            raise pytest.UsageError(f"refusing to overwrite qualification events: {path}") from error
        _path = path
        _emit(dict(event="session_start", pid=os.getpid()))


def pytest_runtest_logreport(report):
    if report.when == "call" or report.failed or report.skipped:
        _emit(dict(event="test", nodeid=report.nodeid, phase=report.when,
                   outcome=report.outcome, duration_s=report.duration,
                   traceback=str(report.longrepr) if report.failed else None))


def pytest_collectreport(report):
    if report.failed:
        _emit(dict(event="collection", nodeid=report.nodeid, outcome="failed",
                   traceback=str(report.longrepr)))


def pytest_sessionfinish(session, exitstatus):
    _emit(dict(event="session_finish", exitstatus=int(exitstatus),
               failed=session.testsfailed, collected=session.testscollected))
