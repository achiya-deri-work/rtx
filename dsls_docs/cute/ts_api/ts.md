# task\_scheduling

Task Scheduling (TS) public API.

The top-level package re-exports the captured-schedule helpers and work
decorators most kernels need directly:

`schedule`
:   Decorator that traces a schedule function into a `Schedule`.

`domain_loop` and `work_tile_loop`
:   Context managers for the runtime domain loop and optional persistent
    work-tile loop.

`consumer_work` and `producer_work`
:   Decorators for resource methods that read from or write into a resource.

`WorkAttr`
:   Work callback attributes such as `WorkAttr.AUXILIARY`.

cutlass.experimental.task\_scheduling.consumer\_work(*method: ~collections.abc.Callable[[...], ~typing.Any] | None = None, \*, work\_attrs: ~cutlass.experimental.task\_scheduling.enums.WorkAttr = <WorkAttr.NONE: 0>, returns: str | ~dataclasses.Field | tuple[str | ~dataclasses.Field, ...] | list[str | ~dataclasses.Field] | None = None*) → Callable[[...], Any]
:   Register a method as a named consumer work function on a `MemoryResource`.

    Consumer work reads data out of the owning `MemoryResource` from the
    resource’s point of view. The decorator registers the method under its
    Python name, which is also the raw `schedule_list` label.

    Parameters:
    :   - **method** (*Callable**,* *optional*) – Method being decorated. Omitted when using decorator-factory form.
        - **work\_attrs** ([*cutlass.experimental.task\_scheduling.enums.WorkAttr*](enums.md#cutlass.experimental.task_scheduling.enums.WorkAttr "cutlass.experimental.task_scheduling.enums.WorkAttr")*,* *optional*) – Verification-visible attributes for the work callback. Use
          `WorkAttr.AUXILIARY` for helper work that carries no data payload.
        - **returns** (*str* *or* *dataclasses.Field* *or* *sequence**,* *optional*) – `TaskLocalVariable` output slot or slots updated by this consumer.
          Field references must point at fields declared with
          `TaskLocalVariable.uninitialized()`.

    Returns:
    :   Decorated method or decorator factory.

    Return type:
    :   Callable

    Notes

    Captured schedules pass returned values as data-flow tokens. A typical
    method declaration is:

    ```console
    item: TaskLocalVariable = TaskLocalVariable.uninitialized()

    @consumer_work(returns=item)
    @cute.jit
    def load(self, stage_info):
        return self.tensor[stage_info.loop_offset]
    ```

    When a raw schedule list is used and a resource has multiple named
    consumer methods, the label is the final tuple element, for example
    `(smem, ScheduleStage.ConsumerWork, "build_desc_a")`.

cutlass.experimental.task\_scheduling.domain\_loop( : *\*bounds: object | Callable[[...], object]*, : *unroll: int | None = None*, ) → Iterator[DomainLoopProxy]
:   Open a domain (for-) loop over the work tile’s iteration space.

    Bounds mirror Python’s `range`: `domain_loop(end)`,
    `domain_loop(start, end)`, or `domain_loop(start, end, step)` — an
    omitted `start` defaults to `0` and an omitted `step` to `1`.
    **Any** bound may be a callable, which makes that dimension dynamic: pass a
    `Task` subclass method accessed on the class (`MyTask.get_domain`, not
    on an instance) so its `self` stays an explicit parameter; the Task
    runtime calls it as `(self, work_tile_coord)` to get that bound for the
    current work tile.

    Parameters:
    :   - **\*bounds** (*int**,* *DSL value**, or* *callable*) – 1 to 3 range-style bounds (see above).
        - **unroll** (*int* *or* *None*) – Loop unroll hint. `None` lets the compiler decide.

cutlass.experimental.task\_scheduling.producer\_work(*method: ~collections.abc.Callable[[...], ~typing.Any] | None = None, \*, work\_attrs: ~cutlass.experimental.task\_scheduling.enums.WorkAttr = <WorkAttr.NONE: 0>*) → Callable[[...], Any]
:   Register a method as a named producer work function on a `MemoryResource`.

    Producer work writes data into the owning `MemoryResource` from the
    resource’s point of view. Captured schedules pass consumer tokens into
    producer keyword parameters by name.

    Parameters:
    :   - **method** (*Callable**,* *optional*) – Method being decorated. Omitted when using decorator-factory form.
        - **work\_attrs** ([*cutlass.experimental.task\_scheduling.enums.WorkAttr*](enums.md#cutlass.experimental.task_scheduling.enums.WorkAttr "cutlass.experimental.task_scheduling.enums.WorkAttr")*,* *optional*) – Verification-visible attributes for the work callback. Use
          `WorkAttr.AUXILIARY` for helper work that carries no data payload.

    Returns:
    :   Decorated method or decorator factory.

    Return type:
    :   Callable

    Notes

    A typical captured producer receives token values as keyword parameters:

    ```console
    @producer_work
    @cute.jit
    def store(self, stage_info, *, item):
        self.tensor[stage_info.loop_offset] = item
    ```

    When a raw schedule list is used and a resource has multiple named producer
    methods, the label is the final tuple element, for example
    `(smem, ScheduleStage.ProducerWork, "tma_load_a")`.

cutlass.experimental.task\_scheduling.schedule( : *fn: Callable[[...], None]*, ) → Callable[[...], Schedule]
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

*class* cutlass.experimental.task\_scheduling.WorkAttr(*value*)
:   Bases: `IntFlag`

    Verification-visible attributes attached to TS work callbacks.

    Work attributes describe semantic properties that the schedule verifier
    must account for. `AUXILIARY` marks callbacks that do not model memory
    access and should not participate in normal resource ordering checks.
    The type is an `IntFlag` so future verifier attributes can be composed
    without changing the decorator API.

    NONE *= 0*

    AUXILIARY *= 1*

    *classmethod* is\_valid\_combination( : *value: [WorkAttr](enums.md#cutlass.experimental.task_scheduling.enums.WorkAttr "cutlass.experimental.task_scheduling.enums.WorkAttr")*, ) → bool
    :   Return true for work-attribute sets supported by verification.

    *classmethod* validate( : *value: [WorkAttr](enums.md#cutlass.experimental.task_scheduling.enums.WorkAttr "cutlass.experimental.task_scheduling.enums.WorkAttr")*, : *field\_name: str*, ) → None
    :   Raise if `value` is not a supported work-attribute flag set.

    is\_auxiliary() → bool
    :   Return true when auxiliary-work semantics are requested.

cutlass.experimental.task\_scheduling.work\_tile\_loop( : *wq: object*, : *\**, : *skip\_if: Callable[[...], [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")] | None = None*, ) → Iterator[WorkTileLoopProxy]
:   Open the persistent work-tile (while) loop over `wq`.

    `skip_if` is an optional predicate `(work_queue, work_tile) -> Boolean`;
    when set, `skippable()` regions in the body are omitted for skipped tiles
    while the surrounding WorkQueue bookkeeping still runs. It may be a plain
    `(work_queue, work_tile)` callable or a method of `wq`.
