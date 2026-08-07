# task\_scheduling.resources

Resource abstractions for the Task Scheduling (TS) framework.

**Compile-time abstraction**: Pipeline operations (acquire, release, commit,
wait) defined here are traced by the DSL at compile time and produce the
same mbarrier PTX instructions as hand-coded bare-metal kernels. There is
no additional runtime overhead from using these abstractions — the framework
is a code generator, not a runtime scheduler.

This module defines the data-flow building blocks that a warp-specialised
kernel assembles into a pipeline:

## Classes

PipelineConfig
:   Immutable descriptor that selects a pipeline type (TMA, UMMA, CLC, etc.)
    and captures its parameters (stage count, byte count, cooperative groups,
    signalling policy). Static factory methods hide the per-type details.

TileSchedulerConfig
:   Pairs a `TileSchedulerType` with the scheduler-specific parameters
    (static persistent or CLC dynamic persistent) and, for CLC, the
    SMEM response pointer.

StageInfo
:   Read-only snapshot passed into every `producer_work` / `consumer_work`
    call. Carries the current loop offset, pipeline stage index, mbarrier
    pointer, per-call index, and the active `WorkTileInfo`.

MemoryResource
:   Base class for every resource in the dataflow graph. A resource owns:
    :   - an optional `PipelineConfig` -> materialised pipeline + barriers,
        - per-role variable dicts (`consumer_vars`, `producer_vars`),
        - pipeline state objects (`consumer_state`, `producer_state`).

    Subclasses declare public data-flow state as `TaskLocalVariable` fields and
    implement consumer / producer work methods to define the resource’s
    behaviour at each schedule stage.

WorkQueue (MemoryResource)
:   Specialised resource that wraps a tile scheduler
    (`StaticPersistentTileScheduler` or
    `ClcDynamicPersistentTileScheduler`). It drives the persistent
    work loop by producing/consuming `WorkTileInfo` tiles.

## Typical lifecycle (driven by `TaskManager`)

> - `resource.create()` allocates SMEM barriers and materialises the
>   pipeline object. Called once per resource from
>   `TaskManager.setup_resources_and_tasks()`.
> - `resource.initialize_runtime_state_internal()` initialises pipeline
>   states, status flags, and task-local storage defaults. Executed once per
>   kernel invocation, outside any dynamic control flow.
> - `resource.create_consumer_variables_internal()` /
>   `resource.create_producer_variables_internal()` populates
>   `consumer_vars` / `producer_vars` dicts with the user-defined
>   variables that flow between resources.
> - `consumer_work(stage_info)` / `producer_work(stage_info)` runs
>   user-defined per-stage logic invoked by the `Task` schedule.
> - `copy_consumer_vars_to(dst_resource)` propagates matching consumer
>   variables into a downstream resource’s producer variables. This is
>   automatic and called by `Task`.

*class* cutlass.experimental.task\_scheduling.resources.PipelineConfig( : *num\_stages: int*, : *num\_bytes: int*, : *producer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *consumer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *pipeline\_type: ~cutlass.experimental.task\_scheduling.enums.PipelineType*, : *barrier\_ptr: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | tuple | None = None*, : *producer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_wait\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads | None = None*, : *umma\_consumer\_producer\_op: ~cutlass.pipeline.helpers.PipelineOp = PipelineOp.AsyncThread*, : *advance\_on\_wait: bool = False*, : *advance\_on\_acquire: bool = False*, : *num\_bytes\_per\_warp\_per\_cta: int | None = None*, : *mcast\_mode\_mn: tuple[int*, : *int] = (1*, : *1)*, : *interleave\_stride: int | tuple[int*, : *int*, : *int*, : *int] = 1*, : *async\_producer\_op: ~cutlass.pipeline.helpers.PipelineOp = PipelineOp.AsyncThread*, )
:   Bases: `object`

    Immutable descriptor for a single pipeline instance.

    Captures everything `MemoryResource.create_pipeline()` needs to
    materialise a concrete pipeline object (barrier storage, stage count,
    transaction bytes, cooperative groups, CTA layout, and signalling
    policy).

    Users should not construct `PipelineConfig` directly. Instead, use
    one of the static factory methods which fill in the correct
    `PipelineType` and sensible defaults:

    - `create_async_async_pipeline_cfg` - generic async-producer + async-consumer pipeline.
    - `create_tma_async_pipeline_cfg` - TMA-producer + async-consumer pipeline.
    - `create_tma_umma_pipeline_cfg` - TMA-producer + UMMA-consumer pipeline.
    - `create_umma_async_pipeline_cfg` - UMMA-producer + async-consumer pipeline.
    - `create_async_umma_pipeline_cfg` - async-producer + UMMA-consumer pipeline.
    - `create_umma_umma_pipeline_cfg` - UMMA-producer + UMMA-consumer pipeline.
    - `create_clc_fetch_async_pipeline_cfg`- CLC tile-fetch + async-consumer pipeline.

    num\_stages
    :   Number of buffering stages (pipeline depth).

        Type:
        :   int

    num\_bytes
    :   Expected transaction byte count per stage (0 when not applicable).

        Type:
        :   int

    producer\_group, consumer\_group
    :   Cooperative groups that define the producer / consumer agents.

        Type:
        :   [pipeline.CooperativeGroup](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")

    pipeline\_type
    :   Selects the concrete pipeline implementation.

        Type:
        :   [PipelineType](enums.md#cutlass.experimental.task_scheduling.enums.PipelineType "cutlass.experimental.task_scheduling.enums.PipelineType")

    barrier\_ptr
    :   Pre-allocated SMEM barrier storage (Int64, 2 \* num\_stages).
        When *None*, `MemoryResource.create_pipeline()` allocates it.

        Type:
        :   cute.Pointer, optional

    cta\_layout\_vmnk
    :   Cluster decomposition layout; required for UMMA / CLC pipelines.

        Type:
        :   cute.Layout, optional

    producer\_signaling\_threads
    :   Which threads execute producer-side barrier operations
        (acquire, commit). `CtaLeader` restricts signalling to CTA 0.

        Type:
        :   [SignalingThreads](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")

    consumer\_signaling\_threads
    :   Which threads execute consumer-side barrier operations
        (wait, release). Used for ConsumerRelease (and ConsumerWait
        when `consumer_wait_signaling_threads` is *None*).

        Type:
        :   [SignalingThreads](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")

    consumer\_wait\_signaling\_threads
    :   Override for ConsumerTryWait / ConsumerWait signaling threads.
        When *None* (default), falls back to `consumer_signaling_threads`.
        Set this when ConsumerWait and ConsumerRelease need different
        CTA signaling (e.g. split-consumer pattern where one task waits
        on all CTAs and another releases on leader CTA only).

        Type:
        :   [SignalingThreads](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads") or None

    async\_producer\_op
    :   Producer-side barrier operation for `AsyncAsync`. Defaults to
        `AsyncThread`. `AsyncLoad` selects cp.async-style producer
        commits that use `cp.async.mbarrier.arrive` on local per-CTA
        full barriers.

        Type:
        :   [pipeline.PipelineOp](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.PipelineOp")

    umma\_consumer\_producer\_op
    :   Producer-side barrier operation for `AsyncUmma`.
        Defaults to `AsyncThread`. `AsyncLoad` selects
        cp.async-style producer commits and local per-CTA arrivals.

        Type:
        :   [pipeline.PipelineOp](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.PipelineOp")

    num\_bytes\_per\_warp\_per\_cta
    :   Declares that each producer CTA routes its TMA completion to the
        leader-visible full barrier. The value is the per-producer-warp, per-CTA
        transaction byte count; validation checks that `num_bytes` covers
        every producer warp across the full cluster.

        Type:
        :   int, optional

    mcast\_mode\_mn
    :   Multicast mode passed to CUTLASS TMA pipeline creation for cluster
        arrival masks.

        Type:
        :   tuple[int, int]

    advance\_on\_wait
    :   Controls when the pipeline stage index is advanced. When *False*
        (default), the stage is advanced on `release`; when *True*, the
        stage is advanced on the `wait` call and a separate pipeline state
        is used to release the consumer.
        Disabled by default pending performance measurements.

        Type:
        :   bool

    advance\_on\_acquire
    :   Producer-side analogue of `advance_on_wait`. When *True*,
        `ProducerAcquire` advances `producer_state` immediately.
        `ProducerWork` either derives the just-acquired state from
        `producer_state` or follows the lagging `producer_commit_state`.
        `ProducerCommit` always uses `producer_commit_state`.

        Type:
        :   bool

    interleave\_stride
    :   Stride for interleaved pipeline advancement. When the stride is > 1,
        N lanes share one pipeline with `num_stages` total barriers; each
        lane starts at its lane index and advances by the stride, which must
        evenly divide `num_stages`. A lane may be selected by a task-local
        warp or by a task’s domain start. A single
        integer applies the same stride to every role. A 4-tuple assigns
        role-specific strides interpreted as `(producer_acquire,
        producer_commit, consumer_wait, consumer_release)`. Splitting
        `producer_acquire` from `producer_commit` requires
        `advance_on_acquire`; splitting `consumer_wait` from
        `consumer_release` requires `advance_on_wait`.

        Type:
        :   int or tuple[int, int, int, int]

    num\_stages*: int*

    num\_bytes*: int*

    producer\_group*: [CooperativeGroup](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*

    consumer\_group*: [CooperativeGroup](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*

    pipeline\_type*: [PipelineType](enums.md#cutlass.experimental.task_scheduling.enums.PipelineType "cutlass.experimental.task_scheduling.enums.PipelineType")*

    barrier\_ptr*: cutlass.cute.typing.Pointer | None* *= None*

    cta\_layout\_vmnk*: cutlass.cute.typing.Layout | tuple | None* *= None*

    producer\_signaling\_threads*: [SignalingThreads](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")* *= 1*

    consumer\_signaling\_threads*: [SignalingThreads](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")* *= 1*

    consumer\_wait\_signaling\_threads*: [SignalingThreads](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads") | None* *= None*

    umma\_consumer\_producer\_op*: [PipelineOp](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp")* *= 1*

    advance\_on\_wait*: bool* *= False*

    advance\_on\_acquire*: bool* *= False*

    num\_bytes\_per\_warp\_per\_cta*: int | None* *= None*

    mcast\_mode\_mn*: tuple[int, int]* *= (1, 1)*

    interleave\_stride*: int | tuple[int, int, int, int]* *= 1*

    async\_producer\_op*: [PipelineOp](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp")* *= 1*

    *property* interleave\_strides*: tuple[int, int, int, int]*
    :   Return `(producer_acquire, producer_commit, consumer_wait,
        consumer_release)` strides.

    *property* producer\_acquire\_interleave\_stride*: int*
    :   Stride used when advancing the producer acquire cursor.

    *property* producer\_commit\_interleave\_stride*: int*
    :   Stride used when advancing the producer commit cursor.

    *property* consumer\_wait\_interleave\_stride*: int*
    :   Stride used when `advance_on_wait` advances consumer wait state.

    *property* consumer\_release\_interleave\_stride*: int*
    :   Stride used when advancing consumer release state.

    *property* max\_interleave\_stride*: int*
    :   Largest role-specific interleave stride.

    *property* has\_interleaved\_stride*: bool*
    :   Whether any role advances through the pipeline with stride > 1.

    *static* create\_async\_async\_pipeline\_cfg( : *num\_stages: int*, : *producer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *consumer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *producer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *barrier\_ptr: cutlass.cute.typing.Pointer | None = None*, : *producer\_op: ~cutlass.pipeline.helpers.PipelineOp = PipelineOp.AsyncThread*, : *advance\_on\_acquire: bool = False*, : *interleave\_stride: int | tuple[int*, : *int*, : *int*, : *int] = 1*, ) → [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")
    :   Create a config for a generic async pipeline.

        `producer_op=AsyncLoad` matches async global-to-shared producers that
        signal full barriers with `cp.async.mbarrier.arrive` while their
        consumers still release through ordinary async-thread mbarriers.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffering stages.
            - **producer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **consumer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **cta\_layout\_vmnk** (*cute.Layout*) – CTA cluster layout.
            - **producer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **consumer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **barrier\_ptr** (*cute.Pointer**,* *optional*) – Pre-allocated barrier storage. `None` lets the allocator place it.
            - **producer\_op** ([*pipeline.PipelineOp*](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.PipelineOp")*,* *optional*) – Async producer operation, normally `AsyncThread` or `AsyncLoad`.
            - **advance\_on\_acquire** (*bool**,* *optional*) – Advance producer state at acquire time instead of commit time.
            - **interleave\_stride** (*int* *or* *tuple**[**int**,* *int**,* *int**,* *int**]**,* *optional*) – Role-specific pipeline-state stride. Mismatched opening and
              closing roles require the corresponding advance flag.

        Returns:
        :   Configuration for an `AsyncAsync` pipeline.

        Return type:
        :   [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")

    *static* create\_tma\_async\_pipeline\_cfg( : *num\_stages: int*, : *num\_bytes: int*, : *producer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *consumer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *producer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *barrier\_ptr: cutlass.cute.typing.Pointer | None = None*, : *interleave\_stride: int | tuple[int*, : *int*, : *int*, : *int] = 1*, : *num\_bytes\_per\_warp\_per\_cta: int | None = None*, ) → [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")
    :   Create a config for a TMA-producer async pipeline.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffering stages.
            - **num\_bytes** (*int*) – TMA transaction byte count for each stage.
            - **producer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **consumer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) – CTA cluster layout.
            - **producer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **consumer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **barrier\_ptr** (*cute.Pointer**,* *optional*) – Pre-allocated barrier storage. `None` lets the allocator place it.
            - **interleave\_stride** (*int* *or* *tuple**[**int**,* *int**,* *int**,* *int**]**,* *optional*) – Role-specific pipeline-state stride. Mismatched opening and
              closing roles require the corresponding advance flag.
            - **num\_bytes\_per\_warp\_per\_cta** (*int**,* *optional*) – Per-producer-warp, per-CTA transaction byte count for leader-routed
              clustered TMA completion validation.

        Returns:
        :   Configuration for a `TmaAsync` pipeline.

        Return type:
        :   [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")

    *static* create\_tma\_umma\_pipeline\_cfg( : *num\_stages: int*, : *num\_bytes: int*, : *producer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *consumer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *producer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *barrier\_ptr: cutlass.cute.typing.Pointer | None = None*, : *advance\_on\_wait: bool = False*, : *interleave\_stride: int | tuple[int*, : *int*, : *int*, : *int] = 1*, : *mcast\_mode\_mn: tuple[int*, : *int] = (1*, : *1)*, : *num\_bytes\_per\_warp\_per\_cta: int | None = None*, ) → [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")
    :   Create a config for TMA producer + UMMA consumer async pipeline.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffering stages.
            - **num\_bytes** (*int*) – TMA transaction byte count for each stage.
            - **producer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **consumer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **cta\_layout\_vmnk** (*cute.Layout*) – CTA cluster layout.
            - **producer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **consumer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **barrier\_ptr** (*cute.Pointer**,* *optional*) – Pre-allocated barrier storage. `None` lets the allocator place it.
            - **advance\_on\_wait** (*bool**,* *optional*) – Advance consumer state at wait time instead of release time.
            - **interleave\_stride** (*int* *or* *tuple**[**int**,* *int**,* *int**,* *int**]**,* *optional*) – Role-specific pipeline-state stride. Mismatched opening and
              closing roles require the corresponding advance flag.
            - **mcast\_mode\_mn** (*tuple**[**int**,* *int**]**,* *optional*) – TMA multicast mode in M and N.
            - **num\_bytes\_per\_warp\_per\_cta** (*int**,* *optional*) – Per-producer-warp, per-CTA transaction byte count for validation.

        Returns:
        :   Configuration for a `TmaUmma` pipeline.

        Return type:
        :   [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")

    *static* create\_umma\_async\_pipeline\_cfg( : *num\_stages: int*, : *producer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *consumer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *producer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *barrier\_ptr: cutlass.cute.typing.Pointer | None = None*, : *interleave\_stride: int | tuple[int*, : *int*, : *int*, : *int] = 1*, ) → [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")
    :   Create a config for a UMMA-producer async pipeline.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffering stages.
            - **producer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **consumer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **cta\_layout\_vmnk** (*cute.Layout*) – CTA cluster layout.
            - **producer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **consumer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **barrier\_ptr** (*cute.Pointer**,* *optional*) – Pre-allocated barrier storage. `None` lets the allocator place it.
            - **interleave\_stride** (*int* *or* *tuple**[**int**,* *int**,* *int**,* *int**]**,* *optional*) – Role-specific pipeline-state stride. Mismatched opening and
              closing roles require the corresponding advance flag.

        Returns:
        :   Configuration for an `UmmaAsync` pipeline.

        Return type:
        :   [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")

    *static* create\_async\_umma\_pipeline\_cfg( : *num\_stages: int*, : *producer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *consumer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *producer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_wait\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads | None = None*, : *producer\_op: ~cutlass.pipeline.helpers.PipelineOp = PipelineOp.AsyncThread*, : *barrier\_ptr: cutlass.cute.typing.Pointer | None = None*, : *advance\_on\_wait: bool = False*, : *advance\_on\_acquire: bool = False*, : *interleave\_stride: int | tuple[int*, : *int*, : *int*, : *int] = 1*, ) → [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")
    :   Create a config for an async-producer + UMMA-consumer pipeline.

        Used when async producer threads create data and the UMMA warp
        consumes it. `producer_op=AsyncLoad` matches async global-to-shared
        producer commits that use `cp.async.mbarrier.arrive` on local
        per-CTA full barriers.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffering stages.
            - **producer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **consumer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **cta\_layout\_vmnk** (*cute.Layout*) – CTA cluster layout.
            - **producer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **consumer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **consumer\_wait\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Override for wait-side signaling when wait and release use different
              participants.
            - **producer\_op** ([*pipeline.PipelineOp*](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.PipelineOp")*,* *optional*) – Async producer operation, normally `AsyncThread` or `AsyncLoad`.
            - **barrier\_ptr** (*cute.Pointer**,* *optional*) – Pre-allocated barrier storage. `None` lets the allocator place it.
            - **advance\_on\_wait** (*bool**,* *optional*) – Advance consumer state at wait time instead of release time.
            - **advance\_on\_acquire** (*bool**,* *optional*) – Advance producer state at acquire time instead of commit time.
            - **interleave\_stride** (*int* *or* *tuple**[**int**,* *int**,* *int**,* *int**]**,* *optional*) – Role-specific pipeline-state stride. Mismatched opening and
              closing roles require the corresponding advance flag.

        Returns:
        :   Configuration for an `AsyncUmma` pipeline.

        Return type:
        :   [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")

    *static* create\_umma\_umma\_pipeline\_cfg( : *num\_stages: int*, : *producer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *consumer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *producer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *barrier\_ptr: cutlass.cute.typing.Pointer | None = None*, ) → [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")
    :   Create a config for a UMMA producer + UMMA consumer pipeline.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffering stages.
            - **producer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **consumer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **cta\_layout\_vmnk** (*cute.Layout*) – CTA cluster layout.
            - **producer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **consumer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **barrier\_ptr** (*cute.Pointer**,* *optional*) – Pre-allocated barrier storage. `None` lets the allocator place it.

        Returns:
        :   Configuration for an `UmmaUmma` pipeline.

        Return type:
        :   [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")

    *static* create\_clc\_fetch\_async\_pipeline\_cfg( : *num\_stages: int*, : *num\_bytes: int*, : *producer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *consumer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *producer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.CtaLeader: 2>*, : *consumer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *barrier\_ptr: cutlass.cute.typing.Pointer | None = None*, ) → [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")
    :   Create a config for a CLC tile-fetch async pipeline.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffering stages.
            - **num\_bytes** (*int*) – Work-tile fetch transaction byte count for each stage.
            - **producer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **consumer\_group** ([*pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups on the producer and consumer sides.
            - **cta\_layout\_vmnk** (*cute.Layout*) – CTA cluster layout.
            - **producer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **consumer\_signaling\_threads** ([*SignalingThreads*](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*,* *optional*) – Threads that execute producer- and consumer-side barrier operations.
            - **barrier\_ptr** (*cute.Pointer**,* *optional*) – Pre-allocated barrier storage. `None` lets the allocator place it.

        Returns:
        :   Configuration for a `ClcFetchAsync` pipeline.

        Return type:
        :   [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")

    \_\_init\_\_( : *num\_stages: int*, : *num\_bytes: int*, : *producer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *consumer\_group: ~cutlass.pipeline.helpers.CooperativeGroup*, : *pipeline\_type: ~cutlass.experimental.task\_scheduling.enums.PipelineType*, : *barrier\_ptr: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | tuple | None = None*, : *producer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads = <SignalingThreads.All: 1>*, : *consumer\_wait\_signaling\_threads: ~cutlass.experimental.task\_scheduling.enums.SignalingThreads | None = None*, : *umma\_consumer\_producer\_op: ~cutlass.pipeline.helpers.PipelineOp = PipelineOp.AsyncThread*, : *advance\_on\_wait: bool = False*, : *advance\_on\_acquire: bool = False*, : *num\_bytes\_per\_warp\_per\_cta: int | None = None*, : *mcast\_mode\_mn: tuple[int*, : *int] = (1*, : *1)*, : *interleave\_stride: int | tuple[int*, : *int*, : *int*, : *int] = 1*, : *async\_producer\_op: ~cutlass.pipeline.helpers.PipelineOp = PipelineOp.AsyncThread*, ) → None

*class* cutlass.experimental.task\_scheduling.resources.TileSchedulerConfig( : *tile\_scheduler\_type: [TileSchedulerType](enums.md#cutlass.experimental.task_scheduling.enums.TileSchedulerType "cutlass.experimental.task_scheduling.enums.TileSchedulerType")*, : *tile\_scheduler\_params: [PersistentTileSchedulerParams](../cute_dsl_api/utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams") | [ClcDynamicPersistentTileSchedulerParams](../cute_dsl_api/utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")*, : *response\_ptr: cutlass.cute.typing.Pointer | None = None*, )
:   Bases: `object`

    Immutable descriptor that pairs a tile-scheduler type with its params.

    tile\_scheduler\_type
    :   `StaticPersistent` or `ClcDynamicPersistent`.

        Type:
        :   [TileSchedulerType](enums.md#cutlass.experimental.task_scheduling.enums.TileSchedulerType "cutlass.experimental.task_scheduling.enums.TileSchedulerType")

    tile\_scheduler\_params
    :   Scheduler-specific parameters (grid dims, cluster shape, etc.).

        Type:
        :   [PersistentTileSchedulerParams](../cute_dsl_api/utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")

    response\_ptr
    :   SMEM response buffer pointer; required only for CLC dynamic mode.

        Type:
        :   cute.Pointer, optional

    tile\_scheduler\_type*: [TileSchedulerType](enums.md#cutlass.experimental.task_scheduling.enums.TileSchedulerType "cutlass.experimental.task_scheduling.enums.TileSchedulerType")*

    tile\_scheduler\_params*: [PersistentTileSchedulerParams](../cute_dsl_api/utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams") | [ClcDynamicPersistentTileSchedulerParams](../cute_dsl_api/utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")*

    response\_ptr*: cutlass.cute.typing.Pointer | None* *= None*

    *static* create\_static\_persistent\_tile\_scheduler\_params( : *tile\_scheduler\_params: [PersistentTileSchedulerParams](../cute_dsl_api/utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, ) → [TileSchedulerConfig](resources.md#cutlass.experimental.task_scheduling.resources.TileSchedulerConfig "cutlass.experimental.task_scheduling.resources.TileSchedulerConfig")
    :   Create a config for the static persistent tile scheduler.

        Parameters:
        :   **tile\_scheduler\_params** ([*PersistentTileSchedulerParams*](../cute_dsl_api/utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – CUTLASS static persistent scheduler parameters.

        Returns:
        :   Configuration selecting `TileSchedulerType.StaticPersistent`.

        Return type:
        :   [TileSchedulerConfig](resources.md#cutlass.experimental.task_scheduling.resources.TileSchedulerConfig "cutlass.experimental.task_scheduling.resources.TileSchedulerConfig")

    *static* create\_clc\_dynamic\_persistent\_tile\_scheduler\_params( : *tile\_scheduler\_params: [ClcDynamicPersistentTileSchedulerParams](../cute_dsl_api/utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")*, : *response\_ptr: cutlass.cute.typing.Pointer*, ) → [TileSchedulerConfig](resources.md#cutlass.experimental.task_scheduling.resources.TileSchedulerConfig "cutlass.experimental.task_scheduling.resources.TileSchedulerConfig")
    :   Create a config for the CLC dynamic persistent tile scheduler.

        Parameters:
        :   - **tile\_scheduler\_params** ([*ClcDynamicPersistentTileSchedulerParams*](../cute_dsl_api/utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) – CUTLASS CLC dynamic scheduler parameters.
            - **response\_ptr** (*cute.Pointer*) – SMEM response buffer pointer used by the CLC fetch pipeline.

        Returns:
        :   Configuration selecting `TileSchedulerType.ClcDynamicPersistent`.

        Return type:
        :   [TileSchedulerConfig](resources.md#cutlass.experimental.task_scheduling.resources.TileSchedulerConfig "cutlass.experimental.task_scheduling.resources.TileSchedulerConfig")

    \_\_init\_\_( : *tile\_scheduler\_type: [TileSchedulerType](enums.md#cutlass.experimental.task_scheduling.enums.TileSchedulerType "cutlass.experimental.task_scheduling.enums.TileSchedulerType")*, : *tile\_scheduler\_params: [PersistentTileSchedulerParams](../cute_dsl_api/utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams") | [ClcDynamicPersistentTileSchedulerParams](../cute_dsl_api/utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")*, : *response\_ptr: cutlass.cute.typing.Pointer | None = None*, ) → None

*class* cutlass.experimental.task\_scheduling.resources.StageInfo( : *loop\_offset: int*, : *loop\_start: int*, : *loop\_end: int*, : *loop\_step: int*, : *stage\_idx: int | None*, : *label: object*, : *barrier: Array | None*, : *work\_tile: [WorkTileInfo](../cute_dsl_api/utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo") | None*, : *num\_active\_stages: \_MockObject = 0*, : *context: [ResourceContext](memory.md#cutlass.experimental.task_scheduling.memory.ResourceContext "cutlass.experimental.task_scheduling.memory.ResourceContext") | None = None*, : *task\_cache: object | None = None*, )
:   Bases: `object`

    Read-only context passed to `producer_work` / `consumer_work`.

    loop\_offset
    :   Current iteration index within the K-tile loop over the computational domain.

        Type:
        :   int

    loop\_start, loop\_end, loop\_step
    :   Bounds and stride of the K-tile loop
        (`range(loop_start, loop_end, loop_step)`).

        Type:
        :   int

    stage\_idx
    :   Pipeline stage index for the current operation (`None` when the
        resource has no pipeline).

        Type:
        :   int or None

    label
    :   User-defined compile-time work label from the schedule entry.
        Selects a named work hook:

        ```console
        if cutlass.const_expr(stage_info.label == WorkLabel.K_DESC):
            ...  # K descriptor logic
        elif cutlass.const_expr(stage_info.label == WorkLabel.V_DESC):
            ...  # V descriptor logic
        ```

        `None` when no label is specified (backward-compatible default).

        Type:
        :   object or None

    barrier
    :   Mbarrier pointer for the current pipeline stage (`None` when the
        resource has no pipeline).

        Type:
        :   cutlass.Array or None

    work\_tile
    :   Tile coordinates and validity flag from the tile scheduler.

        Type:
        :   [WorkTileInfo](../cute_dsl_api/utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.WorkTileInfo") or None

    num\_active\_stages
    :   Number of producer stages currently in flight for delayed-commit
        producer schedules. This is 0 for the default immediate-commit
        schedule.

        Type:
        :   int

    context
    :   Unified context carrying `smem_base`, `tmem_ptr_i32`, and
        any future framework-level state. `None` when no allocator
        is in use.

        Type:
        :   [ResourceContext](memory.md#cutlass.experimental.task_scheduling.memory.ResourceContext "cutlass.experimental.task_scheduling.memory.ResourceContext") or None

    task\_cache
    :   Optional task-defined payload returned by `Task.make_task_cache()`.
        This is intended for kernel-specific cached values that hot resource
        paths want to read without widening `StageInfo` field-by-field.
        The payload shape is task-defined and should remain fixed per task
        class.

        Type:
        :   object or None

    loop\_offset*: int*

    loop\_start*: int*

    loop\_end*: int*

    loop\_step*: int*

    stage\_idx*: int | None*

    label*: object*

    barrier*: Array | None*

    work\_tile*: [WorkTileInfo](../cute_dsl_api/utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo") | None*

    num\_active\_stages*: \_MockObject* *= 0*

    context*: [ResourceContext](memory.md#cutlass.experimental.task_scheduling.memory.ResourceContext "cutlass.experimental.task_scheduling.memory.ResourceContext") | None* *= None*

    task\_cache*: object* *= None*

    \_\_init\_\_( : *loop\_offset: int*, : *loop\_start: int*, : *loop\_end: int*, : *loop\_step: int*, : *stage\_idx: int | None*, : *label: object*, : *barrier: Array | None*, : *work\_tile: [WorkTileInfo](../cute_dsl_api/utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo") | None*, : *num\_active\_stages: \_MockObject = 0*, : *context: [ResourceContext](memory.md#cutlass.experimental.task_scheduling.memory.ResourceContext "cutlass.experimental.task_scheduling.memory.ResourceContext") | None = None*, : *task\_cache: object | None = None*, ) → None

*class* cutlass.experimental.task\_scheduling.resources.TaskLocalVariable( : *dtype: object = <object object>*, : *default: object = <object object>*, : *default\_factory: ~collections.abc.Callable[[]*, : *object] | None = None*, : *docs: str | None = None*, : *runtime\_slot\_name: str | None = None*, )
:   Bases: `object`

    Metadata for one resource-owned variable in captured TS schedules.

    A `TaskLocalVariable` is the public identity of a logical variable owned by
    a `MemoryResource`. In generated code it materializes as task-local
    register state, with consumer work calls producing new
    versions of that register value. The current implementation still lowers
    through string-named `consumer_vars` / `producer_vars` slots, but
    schedule code should carry `TaskLocalVariable` identities rather than
    naming those internal dictionaries directly.

    dtype
    :   DSL type of the value stored in the task-local register slot.

        Type:
        :   object

    default
    :   Sink-safe initial value used before any producer writes and at SSA joins.

        Type:
        :   object

    default\_factory
    :   Factory for the initial value. Mutually exclusive with `default`.

        Type:
        :   Callable[[], object], optional

    docs
    :   Short user-facing description rendered by API documentation.

        Type:
        :   str, optional

    runtime\_slot\_name
    :   Override for the internal slot name. Most resources use the dataclass
        field name.

        Type:
        :   str, optional

    dtype*: object* *= <object object>*

    default*: object* *= <object object>*

    default\_factory*: Callable[[], object] | None* *= None*

    docs*: str | None* *= None*

    runtime\_slot\_name*: str | None* *= None*

    *static* uninitialized() → Field
    :   Declare a dataclass field that backs a `TaskLocalVariable` slot.

        The slot must be assigned a `TaskLocalVariable` instance during
        `__init__` or `__post_init__` of the owning class:

        ```console
        item: TaskLocalVariable = TaskLocalVariable.uninitialized()

        def __post_init__(self) -> None:
            self.item = TaskLocalVariable(dtype=..., default=...)
        ```

        Forgetting the assignment surfaces as a clear `ValueError` the
        first time TS walks the resource’s task-local variables (see
        `bind_task_local_variables`).

        The returned `Field` carries framework metadata under the
        `"ts"` namespace so that `@consumer_work(returns=...)`
        can verify that a field reference points at an actual task-local
        slot rather than an arbitrary `dataclasses.Field`.

        Returns:
        :   Field placeholder to assign a concrete `TaskLocalVariable` in the
            owning resource constructor or `__post_init__`.

        Return type:
        :   dataclasses.Field

    *property* owner*: object | None*
    :   Resource instance that owns this variable, once bound.

    *property* field\_name*: str | None*
    :   Stable resource field name used for legacy slot lowering.

    *property* slot\_name*: str | None*
    :   Current internal slot name for this variable.

    \_\_init\_\_( : *dtype: object = <object object>*, : *default: object = <object object>*, : *default\_factory: ~collections.abc.Callable[[]*, : *object] | None = None*, : *docs: str | None = None*, : *runtime\_slot\_name: str | None = None*, ) → None

cutlass.experimental.task\_scheduling.resources.consumer\_work(*method: ~collections.abc.Callable[[...], ~typing.Any] | None = None, \*, work\_attrs: ~cutlass.experimental.task\_scheduling.enums.WorkAttr = <WorkAttr.NONE: 0>, returns: str | ~dataclasses.Field | tuple[str | ~dataclasses.Field, ...] | list[str | ~dataclasses.Field] | None = None*) → Callable[[...], Any]
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

cutlass.experimental.task\_scheduling.resources.producer\_work(*method: ~collections.abc.Callable[[...], ~typing.Any] | None = None, \*, work\_attrs: ~cutlass.experimental.task\_scheduling.enums.WorkAttr = <WorkAttr.NONE: 0>*) → Callable[[...], Any]
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

*class* cutlass.experimental.task\_scheduling.resources.MemoryResource( : *\**, : *name: ~sphinx.ext.autodoc.mock.\_MockObject = ''*, : *is\_barrier: ~sphinx.ext.autodoc.mock.\_MockObject = False*, : *pipeline\_config: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *producer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *pipeline: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_wait\_signaling\_threads: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, )
:   Bases: `object`

    Base class for every resource in the TS dataflow graph.

    A `MemoryResource` represents a named piece of memory (GMEM, SMEM,
    TMEM, etc.) together with the pipeline that guards access to it.
    Subclasses override the hook methods below to define resource-specific
    behaviour; the `Task` scheduler calls them at the appropriate points.

    ## Named work functions

    Instead of (or in addition to) overriding the monolithic
    `consumer_work` / `producer_work`, subclasses can use the
    `@consumer_work` / `@producer_work` decorators to
    register multiple named work methods. The schedule-list label
    selects which method to call at trace time:

    ```console
    @consumer_work
    @cute.jit
    def k_desc(self, stage_info):
        ...

    @consumer_work
    @cute.jit
    def v_desc(self, stage_info):
        ...
    ```

    In the schedule list, each `ConsumerWork` entry must carry a
    label when two or more named methods are registered:

    ```console
    (smem_kv, ScheduleStage.ConsumerWork, "k_desc")
    ```

    Label validation:

    - Typos raise `ValueError` with a did-you-mean suggestion.
    - Missing labels with ≥2 named methods raise `ValueError`.
    - A single named method auto-dispatches without a label.
    - No named methods → monolithic `consumer_work()` is called.

    ## Hook methods (override in subclasses)

    - `get_smem_requirements() -> list[SmemAllocation]` returns
      `SmemAllocation` objects for data SMEM this resource needs. Default
      returns `[]`.
    - `get_tmem_requirements() -> list[TmemAllocation]` returns
      `TmemAllocation` objects for TMEM columns this resource needs. Default
      returns `[]`.
    - `create_consumer_variables() -> dict` returns a `{name: default}`
      dict of consumer variables that the consumer side produces and forwards
      to downstream resources via `copy_consumer_vars_to`.
    - `create_producer_variables() -> dict` returns a `{name: default}`
      dict of variables consumed by the producer side and populated from an
      upstream resource’s consumer vars.
    - `consumer_aux_work(stage_info)` is helper-variable logic executed
      during `ScheduleStage.ConsumerAuxWork`.
    - `consumer_work(stage_info)` is user logic executed during
      `ScheduleStage.ConsumerWork`. Overridden by `@consumer_work` methods
      when labels are used.
    - `producer_aux_work(stage_info)` is helper-variable logic executed
      during `ScheduleStage.ProducerAuxWork`.
    - `producer_work(stage_info)` is user logic executed during
      `ScheduleStage.ProducerWork`. Overridden by `@producer_work` methods
      when labels are used.

    name
    :   Human-readable label (used in debug prints and PTX comments).

        Type:
        :   str

    pipeline\_config
    :   Pipeline descriptor; `None` for resources that need no pipeline
        (e.g. GMEM source/sink).

        Type:
        :   [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig") or None

    consumer\_vars, producer\_vars
    :   Variable dicts populated by `create_consumer/producer_variables`.

        Type:
        :   dict

    pipeline
    :   Materialised pipeline object, set by `create()`.

        Type:
        :   object or None

    consumer\_state, producer\_state
    :   Pipeline state objects (stage index + phase bit). With
        `advance_on_acquire`, `producer_state` is the acquire cursor and
        `producer_commit_state` is the lagging commit cursor. Producer work
        derives the just-acquired state from `producer_state` or follows the
        commit cursor, depending on the work slot.

    consumer\_status, producer\_status
    :   Boolean flags used by try\_wait / try\_acquire.

    is\_barrier
    :   `True` for pure signaling resources that carry no data (e.g.
        sequence barriers, done notifications). Barrier resources get
        relaxed dependency-backing rules in `_verify_resource_deps`.

        Type:
        :   bool

    dummy
    :   Keep DSL state live across dynamic control-flow boundaries.

        Type:
        :   bool

    consumer\_var\_names*: ClassVar[Tuple[str, ...] | None]* *= None*

    name*: \_MockObject* *= ''*

    is\_barrier*: \_MockObject* *= False*

    pipeline\_config*: \_MockObject* *= None*

    consumer\_vars*: \_MockObject*

    producer\_vars*: \_MockObject*

    pipeline*: \_MockObject* *= None*

    consumer\_state*: Any* *= None*

    producer\_state*: Any* *= None*

    producer\_commit\_state*: Any* *= None*

    consumer\_status*: Any* *= None*

    producer\_status*: Any* *= None*

    consumer\_release\_state*: Any* *= None*

    consumer\_work\_stage*: Any* *= None*

    consumer\_wait\_signaling\_threads*: \_MockObject* *= None*

    pipeline\_group*: \_MockObject* *= None*

    dummy*: \_MockObject* *= False*

    *property* state\_src*: [MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")*
    :   The object that owns this resource’s *consumer* pipeline state.

        Returns `self` by default. For a member of a *Merge*
        `PipelineGroup` this returns the group itself, so all
        members share the group’s canonical consumer state (one
        consumer drives pipeline progression). Fork members and
        non-group resources use their own state (returns `self`).

        **Producer state is NOT redirected.** Each producer task
        independently tracks its own stage via `resource.producer_state`.
        Code on the producer side must use `resource.producer_state`
        directly, not `resource.state_src.producer_state`.

        Pre-resolving the state source as a plain Python attribute
        access (instead of a `_resolve_dispatch` function call
        inside `@cute.jit` regions) avoids DSL-tracer side
        effects that would otherwise emit IR dominance errors.
        The `_state_src_owner` field is set by
        `PipelineGroup.__post_init__` for Merge members only.

    create\_pipeline( : *pipeline\_config: [PipelineConfig](resources.md#cutlass.experimental.task_scheduling.resources.PipelineConfig "cutlass.experimental.task_scheduling.resources.PipelineConfig")*, ) → object
    :   Materialise a pipeline object from the given config.

        Allocates SMEM barrier storage (if not pre-supplied in
        `pipeline_config.barrier_ptr`), then dispatches to the correct
        pipeline constructor based on `pipeline_config.pipeline_type`.

        All pipelines are created with `defer_sync=True` so that
        barrier initialisation fencing is left to the caller
        (`TaskManager` / kernel).

    get\_producer\_acquire\_state() → object
    :   Return the state used for producer-side empty-barrier waits.

    get\_producer\_commit\_state() → object
    :   Return the state used for producer-side full-barrier commits.

    get\_producer\_work\_state( : *follows\_acquire: bool = True*, ) → object
    :   Return the state used for producer-side data movement.

        With `advance_on_acquire`, work does not have an independent
        pipeline cursor. Work that follows acquire derives its stage from the
        acquire cursor in `Task._create_stage_info`; work that does not
        follow acquire uses the lagging commit cursor.

    create() → None
    :   Materialise the pipeline (if configured).

        Called once per resource from
        `TaskManager.setup_resources_and_tasks()`. Subclasses that
        need additional one-time setup (e.g. `WorkQueue` creating its
        tile scheduler) should call `super().create()` first.

        Members of a `PipelineGroup` skip pipeline creation — their
        barrier ops are routed through the group’s shared pipeline.

    create\_consumer\_variables\_internal( : *captured\_schedule: bool = False*, ) → None
    :   Populate `consumer_vars` by calling the user hook (legacy only).

    create\_producer\_variables\_internal( : *captured\_schedule: bool = False*, ) → None
    :   Populate `producer_vars` by calling the user hook (legacy only).

        In captured-schedule mode, `producer_vars` is auto-allocated
        by `Task._allocate_slots_from_routing` from upstream
        `consumer_vars` based on captured schedule data-flow tokens,
        and overriding `create_producer_variables` is forbidden. This
        method becomes a no-op (and raises on illegal overrides) — the
        actual slot allocation runs from `Task.init_variables`.

    create\_consumer\_variables() → dict
    :   Return initial consumer variable dict. Override in subclasses.

    create\_producer\_variables() → dict
    :   Return initial producer variable dict. Override in subclasses.

    get\_smem\_requirements() → List[[SmemAllocation](memory.md#cutlass.experimental.task_scheduling.memory.SmemAllocation "cutlass.experimental.task_scheduling.memory.SmemAllocation")]
    :   Return SMEM allocations required by this resource.

        Override in subclasses that use data SMEM (not barrier SMEM,
        which is managed by `create_pipeline`). The returned
        `SmemAllocation` objects should be stored as instance
        attributes so the resource can read their `.offset` later.

        Default returns an empty list (no data SMEM needed).

    get\_tmem\_requirements() → List[[TmemAllocation](memory.md#cutlass.experimental.task_scheduling.memory.TmemAllocation "cutlass.experimental.task_scheduling.memory.TmemAllocation")]
    :   Return TMEM column allocations required by this resource.

        Override in subclasses that use TMEM. The returned
        `TmemAllocation` objects should be stored as instance
        attributes so the resource can read their `.offset` later.

        Default returns an empty list (no TMEM needed).

    get\_producer\_requirements() → list | None
    :   Return allocations accessed during `ProducerWork`.

        Override in subclasses where the producer only accesses a subset
        of the resource’s allocations. Return a list containing any mix
        of `SmemAllocation` and `TmemAllocation` objects.

        The exhaustive checker uses this to build a producer-specific
        alias map so that `prod_work` only conflicts with aliases that
        overlap these ranges.

        Default returns `None` (all allocations from
        `get_smem_requirements` + `get_tmem_requirements` are
        considered producer-accessible).

    get\_consumer\_requirements() → list | None
    :   Return allocations accessed during `ConsumerWork`.

        Symmetric counterpart of `get_producer_requirements`. Override
        in subclasses where the consumer only accesses a subset of the
        resource’s allocations.

        Default returns `None` (all allocations are consumer-accessible).

    initialize\_runtime\_state\_internal( : *context: [ResourceContext](memory.md#cutlass.experimental.task_scheduling.memory.ResourceContext "cutlass.experimental.task_scheduling.memory.ResourceContext") | None = None*, : *captured\_schedule: bool = False*, ) → None
    :   Initialise pipeline state/status and task-local storage defaults.

        Always creates `consumer_status` / `producer_status` (Int32)
        and `consumer_state` / `producer_state` (pipeline state or
        dummy Int32) so that the DSL tree shape is consistent regardless
        of whether a pipeline is attached.

        For CLC pipelines the producer
        state uses `ProducerConsumer` mode to support both roles.

    set\_consumer\_var( : *name: \_MockObject*, : *value: object*, ) → None
    :   Save a named consumer variable (called from `consumer_work`).

    get\_consumer\_var(*name: str*) → object
    :   Read a named consumer variable (called from consumer work).

    get\_producer\_var( : *name: \_MockObject*, ) → object
    :   Read a named producer variable (called from `producer_work`).

    copy\_consumer\_vars\_to( : *dst\_resource: [MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")*, : *var\_names: list[str | tuple[str, str]] | None = None*, ) → None
    :   Forward matching consumer vars into *dst\_resource*’s producer vars.

        When *var\_names* is `None` (broadcast mode), every key present in
        both `self.consumer_vars` and `dst_resource.producer_vars` is
        copied. When *var\_names* is given, only those specific routes are
        copied. A route may be `"name"` for same-name copy or
        `("src_name", "dst_name")` for explicit remapping.

        Called automatically by `Task._consumer_work` after
        `consumer_work(stage_info)` returns, for destinations whose
        `dst_stage` in the resolved slot routing is a `Producer*Work`
        stage.

    copy\_consumer\_vars\_to\_consumer\_of( : *dst\_resource: [MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")*, : *var\_names: list[str | tuple[str, str]] | None = None*, ) → None
    :   Forward matching consumer vars into *dst\_resource*’s consumer vars.

        Counterpart to [`copy_consumer_vars_to()`](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource.copy_consumer_vars_to "cutlass.experimental.task_scheduling.resources.MemoryResource.copy_consumer_vars_to") for
        consumer-to-consumer variable flow: the source resource’s
        `consumer_vars` are copied into the destination resource’s
        `consumer_vars` so that a later `consumer_work*` call on
        `dst_resource` can read them with `get_consumer_var`.

        When *var\_names* is `None` (broadcast mode), every key present in
        both `self.consumer_vars` and `dst_resource.consumer_vars` is
        copied. When *var\_names* is given, only those specific routes are
        copied. A route may be `"name"` for same-name copy or
        `("src_name", "dst_name")` for explicit remapping.

        Called automatically by `Task._consumer_work*` when the
        resolved slot routing for this slot targets a `ConsumerWork*`
        stage on `dst_resource` (which must therefore be in
        `src_resources`, since consumer vars belong to the consumer
        side of a resource).

    store\_consumer\_status(*value: [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*) → None
    :   Store consumer\_status via self so the IR pass instruments the write.

        Converts the Boolean (i1) from consumer\_try\_wait to Int32 to
        avoid IR boolean-ref edge cases (see initialize\_runtime\_state\_internal).

    load\_consumer\_status() → [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")
    :   Load consumer\_status via self so the IR pass instruments the read.

        Returns Int32; the pipeline’s consumer\_wait already compares
        with 0 via arith.cmpi so no conversion back to Boolean is needed.

    store\_producer\_status(*value: [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*) → None
    :   Store producer\_status via self so the IR pass instruments the write.

        Converts the Boolean (i1) from producer\_try\_acquire to Int32 to
        avoid IR boolean-ref edge cases (see initialize\_runtime\_state\_internal).

    load\_producer\_status() → [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")
    :   Load producer\_status via self so the IR pass instruments the read.

        Returns Int32; the pipeline’s producer\_acquire already compares
        with 0 via arith.cmpi so no conversion back to Boolean is needed.

    consumer\_work( : *stage\_info: [StageInfo](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")*, ) → None
    :   Consumer-side user logic.

        Parameters:
        :   - **stage\_info** ([*StageInfo*](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")) – Current loop, stage, barrier, work-tile, and resource context.
            - **of** (*Override in subclasses that use the monolithic work hook instead*)
            - **methods.** (*named @consumer\_work*)

    consumer\_aux\_work( : *stage\_info: [StageInfo](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")*, ) → None
    :   Helper-variable consumer work that does not model memory access.

        Parameters:
        :   **stage\_info** ([*StageInfo*](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")) – Current loop, stage, barrier, work-tile, and resource context.

    producer\_aux\_work( : *stage\_info: [StageInfo](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")*, ) → None
    :   Helper-variable producer work that does not model memory access.

        Parameters:
        :   **stage\_info** ([*StageInfo*](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")) – Current loop, stage, barrier, work-tile, and resource context.

    producer\_work( : *stage\_info: [StageInfo](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")*, ) → None
    :   Producer-side user logic.

        Parameters:
        :   - **stage\_info** ([*StageInfo*](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")) – Current loop, stage, barrier, work-tile, and resource context.
            - **of** (*Override in subclasses that use the monolithic work hook instead*)
            - **methods.** (*named @producer\_work*)

    physical\_ranges() → list[tuple[str, int, int]]
    :   Declare physical memory regions this resource occupies.

        Returns a list of `(memory_space, start_col, end_col)` tuples
        describing the physical address ranges. Used by the cross-tile
        aliasing verifier to detect potential data races between resources
        that share the same physical memory (e.g. TMEM column overlaps).

        Override in subclasses whose physical storage overlaps with other
        resources. Default: empty (no declared ranges, no aliasing checks).

    \_\_init\_\_( : *\**, : *name: ~sphinx.ext.autodoc.mock.\_MockObject = ''*, : *is\_barrier: ~sphinx.ext.autodoc.mock.\_MockObject = False*, : *pipeline\_config: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *producer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *pipeline: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_wait\_signaling\_threads: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, ) → None

*class* cutlass.experimental.task\_scheduling.resources.PdlWaitBarrier( : *\**, : *name: ~sphinx.ext.autodoc.mock.\_MockObject = ''*, : *is\_barrier: ~sphinx.ext.autodoc.mock.\_MockObject = True*, : *pipeline\_config: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *producer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *pipeline: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_wait\_signaling\_threads: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, )
:   Bases: [`MemoryResource`](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")

    Wait side of CUDA Programmatic Dependent Launch (PDL).

    A barrier-only resource (`is_barrier=True`, no `PipelineConfig`)
    whose single user-facing method, [`wait_griddep()`](resources.md#cutlass.experimental.task_scheduling.resources.PdlWaitBarrier.wait_griddep "cutlass.experimental.task_scheduling.resources.PdlWaitBarrier.wait_griddep"), emits a
    `griddepcontrol.wait` PTX instruction. The instruction blocks the
    issuing thread until the direct predecessor grid dependency has completed
    and made its global-memory results visible.

    ## Wiring contract

    Any TS resource whose data is sourced from the predecessor grid
    (e.g. SMEM-A loaded via TMA, GMEM-A read directly, DSMEM-A copied from
    a peer CTA) declares the dependency by listing `pdl_wait` as one of
    that resource’s upstreams in `TaskManager.resource_dependency_graph`,
    e.g. `resource_dependency_graph[smem_a] = [pdl_wait, ...]`. Because
    `PdlWaitBarrier` has `is_barrier=True`, the verifier interprets the
    edge as *ordering-only*: no consumer/producer variable copy plan is set
    up, but it does require the schedule entry that emits the wait to precede
    the producer entries of any task that produces the dependent resource.

    The `wait_griddep` entry may sit in any phase (Head, Loop with
    `LoopFirstIter` / `LoopLastIter` guards, Tail, post-WTL).

    ## Encouraged pattern

    The encouraged pattern is **inline wait**: every task that produces a
    PDL-dependent resource issues its own `pdl_wait.wait_griddep()` call.

    See also

    [`PdlLaunchBarrier`](resources.md#cutlass.experimental.task_scheduling.resources.PdlLaunchBarrier "cutlass.experimental.task_scheduling.resources.PdlLaunchBarrier"), `side.`

    is\_barrier*: \_MockObject* *= True*

    wait\_griddep( : *stage\_info: [StageInfo](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")*, ) → None

    \_\_init\_\_( : *\**, : *name: ~sphinx.ext.autodoc.mock.\_MockObject = ''*, : *is\_barrier: ~sphinx.ext.autodoc.mock.\_MockObject = True*, : *pipeline\_config: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *producer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *pipeline: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_wait\_signaling\_threads: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, ) → None

*class* cutlass.experimental.task\_scheduling.resources.PdlLaunchBarrier( : *\**, : *name: ~sphinx.ext.autodoc.mock.\_MockObject = ''*, : *is\_barrier: ~sphinx.ext.autodoc.mock.\_MockObject = True*, : *pipeline\_config: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *producer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *pipeline: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_wait\_signaling\_threads: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, )
:   Bases: [`MemoryResource`](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")

    Launch-dependents side of CUDA Programmatic Dependent Launch (PDL).

    A barrier-only resource (`is_barrier=True`, no `PipelineConfig`)
    whose single user-facing method, [`launch_griddep()`](resources.md#cutlass.experimental.task_scheduling.resources.PdlLaunchBarrier.launch_griddep "cutlass.experimental.task_scheduling.resources.PdlLaunchBarrier.launch_griddep"), emits a
    `griddepcontrol.launch_dependents` PTX instruction. The instruction
    notifies the successor grid that it may begin launching CTAs.

    ## Wiring contract

    A `PdlLaunchBarrier` carries no data dependency: it does **not**
    appear on the consumer side of any data flow. Therefore it is not
    expected as a destination in `resource_dependency_graph`. The
    verifier treats `launch_griddep` entries as schedule-only emissions
    that may sit in any task and any phase. The exhaustive interleaving
    checker still requires every executable launch interleaving to have
    already executed at least one `PdlWaitBarrier.wait_griddep`.

    No global “at least one launch” rule is enforced — a kernel may
    legitimately omit launch (e.g. when it is the last grid in a pipeline
    chain or launch is gated by a non-PDL host policy).

    See also

    [`PdlWaitBarrier`](resources.md#cutlass.experimental.task_scheduling.resources.PdlWaitBarrier "cutlass.experimental.task_scheduling.resources.PdlWaitBarrier")

    is\_barrier*: \_MockObject* *= True*

    launch\_griddep( : *stage\_info: [StageInfo](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")*, ) → None

    \_\_init\_\_( : *\**, : *name: ~sphinx.ext.autodoc.mock.\_MockObject = ''*, : *is\_barrier: ~sphinx.ext.autodoc.mock.\_MockObject = True*, : *pipeline\_config: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *producer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *pipeline: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_wait\_signaling\_threads: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, ) → None

cutlass.experimental.task\_scheduling.resources.PDL\_BARRIER\_TYPES*: tuple* *= (<class 'cutlass.experimental.task\_scheduling.resources.PdlWaitBarrier'>, <class 'cutlass.experimental.task\_scheduling.resources.PdlLaunchBarrier'>)*
:   Tuple of all PDL barrier classes for `isinstance` checks.

    Use this in framework code that needs to recognise any PDL-style barrier
    regardless of whether it is the wait side (`PdlWaitBarrier`) or the
    launch side (`PdlLaunchBarrier`).

*class* cutlass.experimental.task\_scheduling.resources.WorkQueue( : *tile\_scheduler\_config: [TileSchedulerConfig](resources.md#cutlass.experimental.task_scheduling.resources.TileSchedulerConfig "cutlass.experimental.task_scheduling.resources.TileSchedulerConfig")*, : *\*\*kwargs: Any*, )
:   Bases: [`MemoryResource`](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")

    Resource that wraps a persistent tile scheduler.

    It participates in the schedule of every task:

    - **Static persistent mode** - Number of launched CTAs is exactly to fill 1
      wave of SMs. Work-tile indices are assigned statically to each CTA. It
      needs no dedicated scheduler warp.
      `get_and_advance_work_tile` calls `advance_to_next_work` directly.
    - **CLC dynamic persistent mode** - a dedicated scheduler warp acts
      as the producer, issuing work-tile fetch requests in `fetch_work_tile`.
      Consumer tasks simply wait on the pipeline.

    In both modes, the consumer-side variable `work_tile`
    (`WorkTileInfo`) carries the tile coordinates and validity flag
    that other resources read.

    tile\_scheduler
    :   Materialised scheduler, set by `create()`.

        Type:
        :   [StaticPersistentTileScheduler](../cute_dsl_api/utils.md#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.StaticPersistentTileScheduler") or [ClcDynamicPersistentTileScheduler](../cute_dsl_api/utils.md#cutlass.utils.ClcDynamicPersistentTileScheduler "cutlass.utils.ClcDynamicPersistentTileScheduler")

    tile\_scheduler\_config
    :   Descriptor selecting the scheduler type and parameters.

        Type:
        :   [TileSchedulerConfig](resources.md#cutlass.experimental.task_scheduling.resources.TileSchedulerConfig "cutlass.experimental.task_scheduling.resources.TileSchedulerConfig")

    tile\_scheduler*: [StaticPersistentTileScheduler](../cute_dsl_api/utils.md#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.static_persistent_tile_scheduler.StaticPersistentTileScheduler") | [ClcDynamicPersistentTileScheduler](../cute_dsl_api/utils.md#cutlass.utils.ClcDynamicPersistentTileScheduler "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileScheduler") | None* *= None*

    \_\_init\_\_( : *tile\_scheduler\_config: [TileSchedulerConfig](resources.md#cutlass.experimental.task_scheduling.resources.TileSchedulerConfig "cutlass.experimental.task_scheduling.resources.TileSchedulerConfig")*, : *\*\*kwargs: Any*, ) → None

    tile\_scheduler\_config*: \_MockObject* *= None*

    work\_tile*: \_MockObject*

    skip\_work\_tile*: \_MockObject*

    create\_tile\_scheduler() → [StaticPersistentTileScheduler](../cute_dsl_api/utils.md#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.static_persistent_tile_scheduler.StaticPersistentTileScheduler") | [ClcDynamicPersistentTileScheduler](../cute_dsl_api/utils.md#cutlass.utils.ClcDynamicPersistentTileScheduler "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileScheduler")
    :   Instantiate the concrete tile scheduler from `tile_scheduler_config`.

    create() → None
    :   Create the pipeline (from base class) and the tile scheduler.

    initial\_work\_tile\_info() → [WorkTileInfo](../cute_dsl_api/utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")
    :   Return the initial work tile from the underlying scheduler.

    skip\_work\_tile\_if( : *work\_tile: [WorkTileInfo](../cute_dsl_api/utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")*, ) → [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
    :   Default skipped-tile predicate used by captured TS schedules.

    init\_work\_tile( : *stage\_info: [StageInfo](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")*, ) → tuple[[WorkTileInfo](../cute_dsl_api/utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo"), [cutlass.Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")]
    :   Seed the persistent loop state before the first work tile.

    get\_and\_advance\_work\_tile( : *stage\_info: [StageInfo](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")*, ) → [WorkTileInfo](../cute_dsl_api/utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")
    :   Typed-schedule work-tile advance callback.

    fetch\_work\_tile( : *stage\_info: [StageInfo](resources.md#cutlass.experimental.task_scheduling.resources.StageInfo "cutlass.experimental.task_scheduling.resources.StageInfo")*, ) → None
    :   Typed-schedule work-tile fetch callback.

        Takes no routed input: the current work tile is supplied by the
        persistent-loop machinery through `stage_info`, so schedules call
        `wq.fetch_work_tile()` directly.

    producer\_tail() → None
    :   Drain in-flight pipeline stages after the persistent loop exits.
