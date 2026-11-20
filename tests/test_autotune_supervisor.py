from __future__ import annotations

import io
import sys
import unittest

from rtx.autotune.supervisor import (
    DEVICE_FAULT_EXIT_CODE, STALL_EXIT_CODE, TIMEOUT_EXIT_CODE, supervise_command,
)


class AutotuneSupervisorTests(unittest.TestCase):
    def test_fatal_warning_is_not_a_successful_child_exit(self):
        output = io.StringIO()
        status = supervise_command([sys.executable, '-c',
            "print('CUDA warning: an illegal memory access was encountered', flush=True)"],
            stall_timeout_s=1, output=output)
        self.assertEqual(status, DEVICE_FAULT_EXIT_CODE)
        self.assertIn('fatal CUDA device fault', output.getvalue())

    def test_split_fault_marker_aborts_before_the_child_continues(self):
        output = io.StringIO()
        status = supervise_command([sys.executable, '-c',
            "import sys,time;sys.stdout.write('cudaErrorIllegal');sys.stdout.flush();"
            "time.sleep(.05);print('Address',flush=True);time.sleep(10);print('BAD_CONTINUATION')"],
            stall_timeout_s=1, terminate_grace_s=.2, output=output)
        self.assertEqual(status, DEVICE_FAULT_EXIT_CODE)
        self.assertNotIn('BAD_CONTINUATION', output.getvalue())

    def test_hard_wall_limit_applies_even_with_continuous_output(self):
        status = supervise_command([sys.executable, '-c',
            "import time\nwhile True: print('active',flush=True);time.sleep(.01)"],
            stall_timeout_s=1, wall_timeout_s=.15, terminate_grace_s=.2, output=io.StringIO())
        self.assertEqual(status, TIMEOUT_EXIT_CODE)

    def test_cleans_up_nested_worker_even_when_parent_exits_zero(self):
        import os
        import time
        output = io.StringIO()
        started = time.monotonic()
        status = supervise_command([sys.executable, '-c',
            "import subprocess,sys;child=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)']);"
            "print('CHILD',child.pid,flush=True)"],
            stall_timeout_s=1, terminate_grace_s=.2, output=output)
        self.assertEqual(status, 0)
        self.assertLess(time.monotonic()-started, 4)
        pid = int(output.getvalue().split('CHILD ')[1].split()[0])
        path = f'/proc/{pid}/stat'
        # SIGKILL delivery is asynchronous; wait for scheduling/reaping rather
        # than assuming a different core has handled it before killpg returns.
        deadline=time.monotonic()+1
        while os.path.exists(path):
            with open(path) as source:
                if source.read().split(') ')[1][0]=='Z':break
            if time.monotonic()>=deadline:self.fail('Owned grandchild survived cleanup')
            time.sleep(.01)

    def test_streams_output_and_returns_child_status(self) -> None:
        output = io.StringIO()
        status = supervise_command(
            [sys.executable, "-c", "print('healthy', flush=True)"],
            stall_timeout_s=1.0,
            output=output,
        )
        self.assertEqual(status, 0)
        self.assertIn("healthy", output.getvalue())

    def test_kills_silent_process_group_with_restartable_status(self) -> None:
        output = io.StringIO()
        status = supervise_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stall_timeout_s=0.1,
            terminate_grace_s=0.5,
            output=output,
        )
        self.assertEqual(status, STALL_EXIT_CODE)
        self.assertIn("WATCHDOG", output.getvalue())

    def test_cpu_active_silent_compile_is_not_mistaken_for_a_stall(self) -> None:
        output = io.StringIO()
        status = supervise_command(
            [
                sys.executable,
                "-c",
                "import time\nend=time.monotonic()+0.35\nwhile time.monotonic()<end: pass",
            ],
            stall_timeout_s=0.05,
            active_stall_timeout_s=2.0,
            output=output,
        )
        self.assertEqual(status, 0)
        self.assertNotIn("terminating", output.getvalue())


if __name__ == "__main__":
    unittest.main()
