"""Opt-in research-only wall-time spans; never imported by production launchers."""
from contextlib import contextmanager, ExitStack
from functools import wraps
import threading
import time
from unittest.mock import patch


class StartupTrace:
    def __init__(self,clock=time.monotonic):
        self.clock=clock
        self.origin=clock()
        self.spans=[]
        self.local=threading.local()
        self.lock=threading.Lock()

    @contextmanager
    def span(self,label):
        stack=getattr(self.local,'stack',[])
        self.local.stack=stack
        with self.lock:
            row=dict(id=len(self.spans),label=label,thread=threading.get_ident(),
                     parent=stack[-1] if stack else None,start_s=self.clock()-self.origin)
            self.spans.append(row)
        stack.append(row['id'])
        try:
            yield
        except BaseException:
            row['failed']=True
            raise
        finally:
            row['end_s']=self.clock()-self.origin
            stack.pop()

    def wrap(self,label,fn):
        @wraps(fn)
        def call(*args,**kwargs):
            with self.span(label):return fn(*args,**kwargs)
        return call

    def report(self):
        if any('end_s' not in row for row in self.spans):
            raise ValueError('trace still has open spans')
        rows=[]
        for row in self.spans:
            inclusive=row['end_s']-row['start_s']
            children=[child for child in self.spans if child['parent']==row['id']]
            exclusive=inclusive-sum(child['end_s']-child['start_s'] for child in children)
            if exclusive < -1e-9:raise ValueError('invalid nested timing')
            rows.append(dict(row,inclusive_s=inclusive,self_s=max(0.,exclusive)))
        return dict(schema=1,spans=rows,limitations=[
            'Wall time, not GPU event or NVVM-only timing.',
            'self_s subtracts immediate instrumented children on the same thread only.',
            'Do not sum inclusive spans or add overlapping spans across threads.',
            'Hook installation imports modules before first-call timing and changes cold-start scope.',
            'Uninstrumented work remains in its nearest measured parent.'])


@contextmanager
def install(trace):
    """Install hooks only inside one diagnostic probe; restore even on failure."""
    import rtx.autotune.effort as effort
    import rtx.autotune.predictors as predictors
    from rtx.autotune.orchestrator import AutotuneOrchestrator
    from rtx.nvfp4_inference_experiments import NVFP4DynamicBenchmarkHarness
    with ExitStack() as stack:
        for owner,name,label in (
            (effort,'make_effort_autotuner','tuner_build'),
            (predictors,'load_predictor','model_load'),
            (predictors.PortablePredictor,'predict','model_predict'),
            (predictors.PortablePredictor,'retrieval','portfolio_retrieval'),
            (AutotuneOrchestrator,'tune','tuner_run'),
            (NVFP4DynamicBenchmarkHarness,'__init__','nvfp4_harness_init'),
        ):
            stack.enter_context(patch.object(owner,name,trace.wrap(label,getattr(owner,name))))
        yield
