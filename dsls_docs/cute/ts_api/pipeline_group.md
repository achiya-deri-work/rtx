# task\_scheduling.pipeline\_group

PipelineGroup — shared pipeline barriers for fork/merge dataflow patterns.

`PipelineGroup` is a `MemoryResource` subclass (`is_barrier=True`)
that groups multiple data resources (members) under a shared set of
pipeline barriers. It enables:

- **Merge** (N producers → 1 consumer): multiple producers each fill
  their own member; one consumer waits for all of them.
- **Fork** (1 producer → N consumers): one producer fills all members;
  multiple consumers each read their own member.

## Barrier Layout (Heterogeneous)

For N members with S pipeline stages:

```console
[full_0 × S] [full_1 × S] … [full_{N-1} × S] [shared_empty × S]
Total entries: (N + 1) × S
```

Each member’s pipeline object receives a pointer to its dedicated full
barrier section and the shared empty barrier.

## Allowed Pipeline Types

All six public `PipelineType` values are supported:

| PipelineType | Producer | Consumer | Pipeline class |
| --- | --- | --- | --- |
| AsyncAsync | async | async | `PipelineAsync` |
| TmaAsync | tma | async | `PipelineTmaAsync` |
| TmaUmma | tma | umma | `PipelineTmaUmma` |
| UmmaAsync | umma | async | `PipelineUmmaAsync` |
| AsyncUmma | async | umma | `TSPipelineAsyncUmma` |
| UmmaUmma | umma | umma | `TSPipelineUmmaUmma` |

Valid heterogeneous merge pairs (consumer side homogeneous):

- Consumer = `async`: AsyncAsync + TmaAsync, AsyncAsync + UmmaAsync,
  TmaAsync + UmmaAsync
- Consumer = `umma`: TmaUmma + AsyncUmma, TmaUmma + UmmaUmma,
  AsyncUmma + UmmaUmma

Valid heterogeneous fork pairs (producer side homogeneous):

- Producer = `tma`: TmaAsync + TmaUmma
- Producer = `async`: AsyncAsync + AsyncUmma
- Producer = `umma`: UmmaAsync + UmmaUmma

## Producer / Consumer Decomposition

The merge/fork topology is auto-derived from the kind decomposition:

- Producer kinds differ → **Merge**
- Consumer kinds differ → **Fork**
- Both homogeneous → uses explicit `mode` parameter
- Both differ → **Error**

*class* cutlass.experimental.task\_scheduling.pipeline\_group.PipelineGroup( : *\**, : *name: ~sphinx.ext.autodoc.mock.\_MockObject = ''*, : *is\_barrier: ~sphinx.ext.autodoc.mock.\_MockObject = False*, : *pipeline\_config: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *producer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *pipeline: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_wait\_signaling\_threads: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *members: ~typing.List[~cutlass.experimental.task\_scheduling.resources.MemoryResource] = <factory>*, : *mode: ~sphinx.ext.autodoc.mock.\_MockObject = PipelineGroupMode.Merge*, )
:   Bases: [`MemoryResource`](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")

    Shared pipeline barriers derived by merging member pipeline configs.

    Each member declares its own `pipeline_config` exactly as it would for a
    standalone pipeline. The group validates those member configs and creates a
    shared barrier layout for the side driven by a single task:

    - `Merge` collapses the consumer side. Members commit individually, and
      the shared consumer task calls `group.release()` once.
    - `Fork` collapses the producer side. Members release individually, and
      the shared producer task calls `group.commit()` once.

    `PipelineGroup`:

    1. Validates compatibility (same `num_stages` and compatible
       `pipeline_type` on the collapsed side).
    2. Derives a merged config from the member configs. The collapsed side’s
       cooperative-group size must match across members; the many side keeps
       per-member barriers.
    3. Creates one shared pipeline with `(N + 1) * num_stages` barriers.
    4. Re-points each member at the shared barrier allocation.

    Raises `ValueError` if member configs are not mergeable.

    When an explicit `pipeline_config` is supplied by the caller it is
    validated against the members using the same mode-specific compatibility
    rules. When `pipeline_config` is `None`, it is derived automatically.

    members
    :   Data resources whose pipelines are merged into this group.

        Type:
        :   List[[MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")]

    mode
    :   `Merge` - N producers, 1 consumer; collapse the consumer side.
        `Fork` - 1 producer, N consumers; collapse the producer side.

        Type:
        :   [PipelineGroupMode](enums.md#cutlass.experimental.task_scheduling.enums.PipelineGroupMode "cutlass.experimental.task_scheduling.enums.PipelineGroupMode")

    members*: List[[MemoryResource](resources.md#cutlass.experimental.task_scheduling.resources.MemoryResource "cutlass.experimental.task_scheduling.resources.MemoryResource")]*

    mode*: \_MockObject* *= 'Merge'*

    *property* group\_pipeline*: \_MockObject*
    :   Access the first member’s pipeline for group-level ops.

        Group-level barrier operations (ConsumerRelease for Merge,
        ProducerCommit for Fork) need a pipeline object to call
        `consumer_release` / `producer_commit`. Instead of aliasing
        `self.pipeline` (which breaks under `inject_leaves`), this
        property reads the first member’s pipeline on demand.

    *property* is\_heterogeneous*: bool*
    :   True when members have different pipeline types.

        Heterogeneous groups maintain per-member full barriers and a
        shared empty barrier, whereas homogeneous groups share a single
        merged pipeline across all members.

    create() → None
    :   Create pipeline(s) for this group and assign to members.

        Always uses per-member barriers to avoid double-arming issues
        when multiple producers/consumers call acquire on the same
        shared pipeline. The layout depends on the mode:

        - *Merge*: per-member full barriers + 1 shared empty barrier.
        - *Fork*: 1 shared full barrier + per-member empty barriers.

        A per-member pipeline object of the correct type is constructed
        and assigned to `m.pipeline`. The group’s `self.pipeline`
        is also set for group-level operations (e.g. `producer_tail`).

    \_\_init\_\_( : *\**, : *name: ~sphinx.ext.autodoc.mock.\_MockObject = ''*, : *is\_barrier: ~sphinx.ext.autodoc.mock.\_MockObject = False*, : *pipeline\_config: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *producer\_vars: ~sphinx.ext.autodoc.mock.\_MockObject = <factory>*, : *pipeline: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *consumer\_wait\_signaling\_threads: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *members: ~typing.List[~cutlass.experimental.task\_scheduling.resources.MemoryResource] = <factory>*, : *mode: ~sphinx.ext.autodoc.mock.\_MockObject = PipelineGroupMode.Merge*, ) → None
