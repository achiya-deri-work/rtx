from __future__ import annotations

import io
import sys
import unittest

from rtx.autotune.supervisor import STALL_EXIT_CODE, supervise_command


class AutotuneSupervisorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
