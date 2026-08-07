# task\_scheduling.enums

*class* cutlass.experimental.task\_scheduling.enums.PipelineType(*value*)
:   Bases: `Enum`

    Type of the concrete pipeline implementation in `PipelineConfig`.

    Pipeline names follow the `<Producer><Consumer>` convention.

    Public members:

    - `AsyncAsync` - Generic async (thread-based) producer + async-thread
      consumer pipeline.
    - `TmaAsync` - TMA-producer + async-thread consumer pipeline
      (cp.async.bulk).
    - `TmaUmma` - TMA-producer + UMMA-consumer async pipeline.
    - `UmmaAsync` - UMMA-producer + AsyncThread-consumer async pipeline.
      `UmmaAsync` matches the CUTLASS pipeline name.
    - `AsyncUmma` - AsyncThread-producer + UMMA-consumer async pipeline.
      `AsyncUmma` matches the CUTLASS pipeline name.
    - `UmmaUmma` - UMMA-producer + UMMA-consumer async pipeline. Used when
      tensor-core producer and consumer stages are both UMMA-backed.
    - `ClcFetchAsync` - CLC tile-fetch producer + async-thread consumer pipeline.

    AsyncAsync *= 'AsyncAsync'*

    TmaAsync *= 'TmaAsync'*

    TmaUmma *= 'TmaUmma'*

    UmmaAsync *= 'UmmaAsync'*

    AsyncUmma *= 'AsyncUmma'*

    UmmaUmma *= 'UmmaUmma'*

    ClcFetchAsync *= 'ClcFetchAsync'*

*class* cutlass.experimental.task\_scheduling.enums.TileSchedulerType(*value*)
:   Bases: `Enum`

    Selects the persistent tile-scheduling strategy.

    Members:

    - `StaticPersistent` - CTAs are launched to fill one SM wave; work-tile
      indices are assigned statically per CTA.
    - `ClcDynamicPersistent` - A dedicated scheduler warp fetches work tiles
      dynamically at runtime.

    StaticPersistent *= 'StaticPersistent'*

    ClcDynamicPersistent *= 'ClcDynamicPersistent'*

*class* cutlass.experimental.task\_scheduling.enums.PipelineGroupMode(*value*)
:   Bases: `Enum`

    Dataflow topology of a `PipelineGroup`.

    A group collapses the barrier on whichever side is driven by a single
    task. Group construction derives a merged config from the member configs:
    `num_bytes` is combined across members as required by the group mode,
    while the collapsed side’s cooperative-group size must match across
    members.

    Members:

    - `Merge` - N-to-1: each member has a separate producer task; all
      members share one consumer task. The consumer side is collapsed: member
      pipeline types must have compatible consumer kinds, and
      `consumer_group.size` must match.
    - `Fork` - 1-to-N: all members share one producer task; each member
      has a separate consumer task. The producer side is collapsed: member
      pipeline types must have compatible producer kinds, and
      `producer_group.size` must match.

    Merge *= 'Merge'*

    Fork *= 'Fork'*

*class* cutlass.experimental.task\_scheduling.enums.SignalingThreads(*value*)
:   Bases: `IntFlag`

    Controls which threads execute pipeline barrier operations.

    Members:

    - `All` - Every thread in the cooperative group participates.
    - `CtaLeader` - Only CTA 0 in the cluster signals (used when a single
      CTA drives a 2-CTA pipeline, e.g. MMA producer).
    - `TaskWarpLeader` - Only the first warp assigned to the task arms the
      producer-side transaction barrier. This is orthogonal to CTA ownership:
      it may be combined with `CtaLeader`.

    `CtaLeader` and `TaskWarpLeader` are composable. `All` is
    exclusive: combining it with another flag is invalid because it
    contradicts a narrower signaling predicate.

    All *= 1*

    CtaLeader *= 2*

    TaskWarpLeader *= 4*

    *classmethod* is\_valid\_combination( : *value: [SignalingThreads](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*, ) → bool
    :   Return true for signaling modes supported by TS codegen.

    *classmethod* validate( : *value: [SignalingThreads](enums.md#cutlass.experimental.task_scheduling.enums.SignalingThreads "cutlass.experimental.task_scheduling.enums.SignalingThreads")*, : *field\_name: str*, ) → None
    :   Raise if `value` is not a supported signaling mode.

    has\_cta\_leader() → bool
    :   Return true when CTA-leader gating is part of this mode.

    has\_task\_warp\_leader() → bool
    :   Return true when task-warp-leader gating is part of this mode.

*class* cutlass.experimental.task\_scheduling.enums.WorkAttr(*value*)
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

*class* cutlass.experimental.task\_scheduling.enums.ScheduleStage(*value*)
:   Bases: `Enum`

    Individual pipeline operations that a `Task` schedule can reference.

    Each member maps to a method on `Task` that delegates to the
    corresponding pipeline / resource call:

    Consumer side (src in the Task):
    :   - `ConsumerAuxWork` - execute side-effect-free helper-variable work.
        - `ConsumerTryWait` - non-blocking check for data availability.
        - `ConsumerWait` - blocking wait for data availability.
        - `ConsumerWork` - execute the resource’s `consumer_work()`.
        - `ConsumerRelease` - signal the producer that the buffer is free.

    Producer side (dst in the Task):
    :   - `ProducerAuxWork` - execute side-effect-free helper-variable work.
        - `ProducerTryAcquire` - non-blocking check for a free buffer.
        - `ProducerAcquire` - blocking wait for a free buffer.
        - `ProducerWork` - execute the resource’s `producer_work()`.
        - `ProducerCommit` - signal the consumer that data is ready.

    The `__str__` method returns compact labels used in schedule printouts.

    ConsumerAuxWork *= 'ConsumerAuxWork'*

    ConsumerRelease *= 'ConsumerRelease'*

    ConsumerTryWait *= 'ConsumerTryWait'*

    ConsumerWait *= 'ConsumerWait'*

    ConsumerWork *= 'ConsumerWork'*

    ProducerAuxWork *= 'ProducerAuxWork'*

    ProducerTryAcquire *= 'ProducerTryAcquire'*

    ProducerAcquire *= 'ProducerAcquire'*

    ProducerCommit *= 'ProducerCommit'*

    ProducerWork *= 'ProducerWork'*
