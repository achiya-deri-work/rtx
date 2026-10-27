"""Scoped research dispatch; use inside an isolated candidate worker only."""
from contextlib import ExitStack, contextmanager

from rtx.autotune.ms_eden_design import implementation_rejection


@contextmanager
def dispatch(design):
    rejection = implementation_rejection(design)
    if rejection:
        raise ValueError(rejection[1])
    records = []
    with ExitStack() as stack:
        if design.transpose_loader != "native":
            from benchmarks.experiment_ms_eden_bank_layout import patched
            stack.enter_context(patched(design.transpose_loader, records=records, strict=True))
        if design.finalizer != "separate":
            from benchmarks.experiment_ms_eden_dual_finalizer import patched
            stack.enter_context(patched("both",
                inline_max=design.finalizer == "dual_cta",
                inline_scope="cta" if design.finalizer == "dual_cta" else "warp",
                pair_values=design.finalizer == "dual_paired"))
        yield records
