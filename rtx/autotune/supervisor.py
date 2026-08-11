"""Parent-process watchdog for CUDA autotuning commands.

Threads and signals cannot recover from a GPU kernel which never returns.  The
only reliable boundary is a fresh process and CUDA context, so this supervisor
owns the child process group and treats prolonged output silence as a stall.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
import time
from typing import Mapping, Sequence, TextIO


WATCHDOG_CHILD_ENV = "RTX_AUTOTUNE_WATCHDOG_CHILD"
STALL_EXIT_CODE = 75


def supervise_command(
    command: Sequence[str],
    *,
    stall_timeout_s: float,
    terminate_grace_s: float = 10.0,
    environment: Mapping[str, str] | None = None,
    output: TextIO | None = None,
) -> int:
    """Run ``command`` and kill its process group after prolonged silence.

    Output is streamed unchanged, making ordinary progress lines the heartbeat.
    The returned exit code is 75 after a watchdog kill so callers can resume
    from append-only residuals in a fresh process.
    """

    if stall_timeout_s <= 0 or terminate_grace_s <= 0:
        raise ValueError("watchdog durations must be positive")
    sink = sys.stdout if output is None else output
    env = dict(os.environ if environment is None else environment)
    env[WATCHDOG_CHILD_ENV] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        env=env,
        start_new_session=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    last_progress = time.monotonic()
    pending = bytearray()

    def emit(data: bytes) -> None:
        if not data:
            return
        sink.write(data.decode("utf-8", errors="replace"))
        sink.flush()

    try:
        while process.poll() is None:
            ready = selector.select(timeout=min(1.0, stall_timeout_s))
            if ready:
                chunk = os.read(process.stdout.fileno(), 65536)
                if chunk:
                    pending.extend(chunk)
                    last_progress = time.monotonic()
                    while b"\n" in pending:
                        line, _, pending = pending.partition(b"\n")
                        emit(line + b"\n")
                    continue
            silent_s = time.monotonic() - last_progress
            if silent_s < stall_timeout_s:
                continue
            emit(
                (
                    f"WATCHDOG no child progress for {silent_s:.1f}s; "
                    f"terminating process group pid={process.pid}\n"
                ).encode()
            )
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=terminate_grace_s)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if pending:
                emit(bytes(pending))
            return STALL_EXIT_CODE
        remainder = process.stdout.read()
        if remainder:
            pending.extend(remainder)
        if pending:
            emit(bytes(pending))
        return int(process.returncode or 0)
    finally:
        selector.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=terminate_grace_s)
        process.stdout.close()


__all__ = [
    "STALL_EXIT_CODE",
    "WATCHDOG_CHILD_ENV",
    "supervise_command",
]
