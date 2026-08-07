# task\_scheduling.task

Task for the Task Scheduling (TS) framework.

**Compile-time abstraction**: `Task.run()` is traced by the DSL into a
monolithic loop. The
captured schedule list, loop guards, and domain computation are all resolved
at trace time. A task whose resolved domain has zero loop iterations emits no
loop-body work. The generated PTX is identical to a hand-coded bare-metal
kernel with the same schedule.

This module provides the Task class and schedule normalisation utilities
that sit on top of the resource abstractions defined in `resources.py`.

*Tasks* are assigned to a contiguous range of warps and bind resource roles to
a captured `ScheduleResult`. A task reads from its `src_resources`
(consumer side) and writes to its `dst_resources` (producer side). The
captured schedule records ordered pipeline operations such as acquire, wait,
work, commit, and release. `TaskManager` executes every task: each warp
enters `task.run()`, but only the warps in
`[warp_idx, warp_idx + num_warps)` execute that task’s schedule body.

*class* cutlass.experimental.task\_scheduling.task.Task( : *src\_resources: List[[MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")]*, : *dst\_resources: List[[MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")]*, : *warp\_idx: int*, : *num\_warps: int*, : *\**, : *schedule: ScheduleResult*, : *num\_registers: int | None = None*, : *name: str = ''*, : *debug\_print: bool = False*, : *run\_only\_on\_cta\_id: int | None = None*, )
:   Bases: `TraversableLeafMixin`

    A unit of warp-specialised work in the TS framework.

    A Task maps a contiguous range of warps to a pair of resource lists
    (`src_resources` consumed by the task and `dst_resources` produced
    by the task). Its schedule is supplied only as a `ScheduleResult`
    captured by the `@schedule` decorator. The captured schedule defines
    the pipeline operations (e.g. wait, acquire, work, commit, release)
    executed each iteration of the main loop over the computational domain.

    The captured schedule list is normalized into three phases
    (head / loop / tail) at construction time.

    name
    :   Human-readable label used in debug prints and PTX comments.

        Type:
        :   str

    src\_resources
    :   Resources from which this task reads (consumer side).

        Type:
        :   List[[MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")]

    dst\_resources
    :   Resources to which this task writes (producer side).

        Type:
        :   List[[MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")]

    warp\_idx
    :   Index of the first warp assigned to this task.

        Type:
        :   [cutlass.Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")

    num\_warps
    :   Number of consecutive warps assigned to this task.

        Type:
        :   int

    num\_registers
    :   Per-task register budget (must be 8-256, divisible by 8).
        `setmaxregister_increase/decrease` is emitted accordingly.

        Type:
        :   int | None

    schedule
    :   Captured output of the `@schedule` decorator. It supplies the
        schedule list, loop bounds, loop step, unroll hint, routing table,
        and any persistent-work-tile metadata.

        Type:
        :   ScheduleResult

    domain\_getter
    :   Explicit `Task` subclass method captured from the callable `end`
        bound, e.g. `domain_loop(MyTask.get_domain)`. When present, `Task`
        validates that this instance inherits that exact `get_domain` override.

        Type:
        :   callable | None

    is\_persistent
    :   `True` when `src_resources` contains a `WorkQueue`.

        Type:
        :   bool

    run\_only\_on\_cta\_id
    :   Hoist CTA selection for the whole task body to one explicit CTA rank.
        When set, the CTA-rank check is hoisted into `is_selected()` so the
        entire task body, including all pipeline operations, runs only on that
        CTA. Requires that every pipelined resource touched by this task sets
        the matching signaling side to `SignalingThreads.CtaLeader`; a
        `ValueError` is raised at trace time otherwise.

        Type:
        :   int | None

    debug\_print
    :   Emit per-step debug prints to stdout (default: False).

        Type:
        :   bool

    head\_schedule\_list, loop\_schedule\_list, tail\_schedule\_list
    :   Normalised schedule phases, each a list of
        `(MemoryResource, ScheduleStage, call_id)` tuples.

    \_\_init\_\_( : *src\_resources: List[[MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")]*, : *dst\_resources: List[[MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")]*, : *warp\_idx: int*, : *num\_warps: int*, : *\**, : *schedule: ScheduleResult*, : *num\_registers: int | None = None*, : *name: str = ''*, : *debug\_print: bool = False*, : *run\_only\_on\_cta\_id: int | None = None*, ) → None
    :   Create a task from a captured schedule.

        `schedule` is the sole source of schedule metadata. It
        supplies the schedule list, slot routing, loop bounds, loop step,
        unroll hint, skip predicate, and persistent-work-tile placement
        metadata captured by the `@schedule` decorator.

        Parameters:
        :   - **src\_resources** (*List**[*[*MemoryResource*](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")*]*) – Resources read by this task on the consumer side.
            - **dst\_resources** (*List**[*[*MemoryResource*](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")*]*) – Resources written by this task on the producer side.
            - **warp\_idx** (*int*) – Index of the first warp assigned to this task.
            - **num\_warps** (*int*) – Number of consecutive warps assigned to this task.
            - **schedule** (*object*) – Captured result returned by a `@schedule` function. Must be
              provided; legacy constructor schedule metadata is not accepted by
              this API.
            - **num\_registers** (*int**,* *optional*) – Per-task register budget. When set, it must be in `[8, 256]` and
              divisible by 8.
            - **name** (*str**,* *optional*) – Human-readable task name used in diagnostics and PTX comments.
            - **debug\_print** (*bool**,* *optional*) – Emit per-step debug prints from generated code.
            - **run\_only\_on\_cta\_id** (*int**,* *optional*) – Hoist task selection so all task work runs only on the given CTA
              rank. Requires matching CTA-leader signaling on every pipelined
              resource touched by the task.

    *property* resources*: chain*
    :   Iterate over all resources (src then dst) without duplicates.

    get\_domain(*tile\_coord: object*) → Any
    :   Return the iteration domain for the given tile coordinate.

        Override in subclasses for tile-dependent domain logic.
        The default implementation returns `self.domain`.

        Parameters:
        :   **tile\_coord** (*object*) – `work_tile.tile_idx` – a 3-tuple `(bx, by, bz)` of the
            current tile’s block coordinates.

        Returns:
        :   The number of loop iterations for this tile.

        Return type:
        :   int or [cutlass.Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")

    is\_selected() → bool
    :   Return `True` if the current warp falls within this task’s range.

    init\_variables( : *context: [ResourceContext](memory.md#cutlass.experimental.task_scheduling.memory.ResourceContext "cutlass.experimental.task_scheduling.memory.ResourceContext") | None = None*, ) → None
    :   Create resource variables outside dynamic control flow.

        Must be called before any `scf.if` dispatch to keep the DSL
        IR structure stable. `TaskManager.run()` calls this for
        every task before entering the dispatch chain.

        Parameters:
        :   **context** ([*ResourceContext*](memory.md#cutlass.experimental.task_scheduling.memory.ResourceContext "cutlass.experimental.task_scheduling.memory.ResourceContext") *or* *None*) – Carries `smem_base` and/or `tmem_ptr_i32` when
            allocators are in use.

    make\_task\_cache() → object
    :   Return an optional task-defined payload captured into `StageInfo`.

        Override in task subclasses when hot resource paths need a few cached
        task-local values without widening `StageInfo` field-by-field.

    run\_body( : *context: [ResourceContext](memory.md#cutlass.experimental.task_scheduling.memory.ResourceContext "cutlass.experimental.task_scheduling.memory.ResourceContext") | None = None*, ) → None
    :   Gate on warp selection and execute the task body.

        Call only after `init_variables()` has already been invoked
        for *all* tasks. `context` is forwarded explicitly into
        `StageInfo` so traced task structure does not change at runtime.

    run() → None
    :   Top-level entry point executed by `TaskManager.run()`.

        1. Initialises function-level and role-level variables on
           *every* resource (outside dynamic control flow to keep the
           DSL IR structure stable).
        2. Gates on `is_selected()` so only the assigned warps proceed.
        3. Sets the register budget via `_set_max_register`.
        4. Dispatches to persistent or non-persistent body.
