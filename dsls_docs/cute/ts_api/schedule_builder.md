# task\_scheduling.schedule\_builder

cutlass.experimental.task\_scheduling.schedule\_builder.schedule( : *fn: Callable[[...], None]*, ) → Callable[[...], Schedule]
:   Decorator that traces a schedule function into a `Schedule`.

    Parameters:
    :   **fn** (*Callable*) – Function whose arguments are `MemoryResource` instances and whose body
        records resource calls through schedule-builder context managers.

    Returns:
    :   Wrapper that accepts concrete resources and returns the captured
        `Schedule`.

    Return type:
    :   Callable[…, Schedule]

    Notes

    The decorated function receives `ResourceProxy` wrappers for each
    `MemoryResource`. Method calls on the proxies record schedule
    entries and routing edges. `with work_tile_loop(wq):` and
    `with domain_loop(start, end, step):` mark the structural
    boundaries. Plain Python control flow inside the function executes at
    trace time.
