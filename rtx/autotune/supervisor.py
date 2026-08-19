"""Parent-process watchdog for CUDA autotuning commands.

Threads and signals cannot recover from a GPU kernel which never returns.  The
only reliable boundary is a fresh process and CUDA context, so this supervisor
owns the child process group and treats prolonged output *and process-tree CPU*
silence as a stall.  This distinction matters because NVVM can legitimately
compile a large candidate for several minutes without printing anything.
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


def _process_tree_cpu_ticks(root_pid: int) -> int | None:
    """Return aggregate Linux CPU ticks for ``root_pid`` and descendants.

    A missing ``/proc`` or a process racing with collection is harmless.  The
    watchdog then falls back to its original output-only behavior.
    """

    proc = "/proc"
    try:
        entries = os.listdir(proc)
    except OSError:
        return None
    parents: dict[int, int] = {}
    ticks: dict[int, int] = {}
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"{proc}/{entry}/stat", encoding="utf-8") as handle:
                stat = handle.read()
            tail = stat[stat.rfind(")") + 2 :].split()
            parents[pid] = int(tail[1])
            ticks[pid] = int(tail[11]) + int(tail[12])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    found = [ticks[pid] for pid in descendants if pid in ticks]
    return sum(found) if found else None


def supervise_command(
    command: Sequence[str],
    *,
    stall_timeout_s: float,
    active_stall_timeout_s: float | None = None,
    terminate_grace_s: float = 10.0,
    environment: Mapping[str, str] | None = None,
    output: TextIO | None = None,
) -> int:
    """Run ``command`` and kill its process group after prolonged silence.

    Output is streamed unchanged.  Output or increasing process-tree CPU time
    is a heartbeat; active-but-silent work still has a separate hard ceiling.
    The returned exit code is 75 after a watchdog kill so callers can resume
    from append-only residuals in a fresh process.
    """

    if active_stall_timeout_s is None:
        active_stall_timeout_s = max(5.0 * stall_timeout_s, 1200.0)
    if (
        stall_timeout_s <= 0
        or active_stall_timeout_s <= 0
        or terminate_grace_s <= 0
    ):
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
    last_output = time.monotonic()
    last_activity = last_output
    last_cpu_ticks = _process_tree_cpu_ticks(process.pid)
    last_activity_notice = last_output
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
                    last_output = time.monotonic()
                    last_activity = last_output
                    while b"\n" in pending:
                        line, _, pending = pending.partition(b"\n")
                        emit(line + b"\n")
                    continue
            now = time.monotonic()
            cpu_ticks = _process_tree_cpu_ticks(process.pid)
            cpu_active = (
                cpu_ticks is not None
                and last_cpu_ticks is not None
                and cpu_ticks > last_cpu_ticks
            )
            if cpu_ticks is not None:
                last_cpu_ticks = cpu_ticks
            if cpu_active:
                last_activity = now
                if now - last_activity_notice >= stall_timeout_s:
                    emit(
                        (
                            "WATCHDOG child is output-silent but CPU-active; "
                            f"pid={process.pid} output_silence={now - last_output:.1f}s\n"
                        ).encode()
                    )
                    last_activity_notice = now
            inactive_s = now - last_activity
            output_silent_s = now - last_output
            if (
                inactive_s < stall_timeout_s
                and output_silent_s < active_stall_timeout_s
            ):
                continue
            reason = (
                f"no child output or CPU activity for {inactive_s:.1f}s"
                if inactive_s >= stall_timeout_s
                else f"active output silence exceeded {active_stall_timeout_s:.1f}s"
            )
            emit(
                (
                    f"WATCHDOG {reason}; "
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
