import unittest
from tools.research.startup_trace import StartupTrace


class StartupTraceTests(unittest.TestCase):
    def test_nested_intervals_do_not_double_count(self):
        times=iter((0,1,2,4,6))
        trace=StartupTrace(lambda:next(times))
        with trace.span('parent'):
            with trace.span('child'):pass
        rows=trace.report()['spans']
        self.assertEqual(rows[0]['inclusive_s'],5)
        self.assertEqual(rows[0]['self_s'],3)
        self.assertEqual(rows[1]['self_s'],2)
        self.assertEqual(rows[1]['parent'],rows[0]['id'])

    def test_failure_is_closed_and_propagated(self):
        trace=StartupTrace()
        def fail():raise RuntimeError('original failure')
        with self.assertRaisesRegex(RuntimeError,'original failure'):
            trace.wrap('failure',fail)()
        self.assertTrue(trace.report()['spans'][0]['failed'])
        self.assertEqual(trace.local.stack,[])

    def test_open_span_report_rejected(self):
        trace=StartupTrace()
        with trace.span('open'):
            with self.assertRaises(ValueError):trace.report()

    def test_hook_restoration_on_exception(self):
        import rtx.autotune.effort as effort
        from tools.research.startup_trace import install
        original=effort.make_effort_autotuner
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with install(StartupTrace()):
                self.assertIsNot(effort.make_effort_autotuner,original)
                raise RuntimeError('stop')
        self.assertIs(effort.make_effort_autotuner,original)


if __name__=='__main__':unittest.main()
