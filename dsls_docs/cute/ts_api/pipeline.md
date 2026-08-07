# task\_scheduling.pipeline

TS pipeline overloads.

CuTeDSL TMA pipelines arm transaction barriers from every warp that calls
`producer_acquire` because `MbarrierArray.arrive_and_expect_tx` is guarded
only by per-warp `elect_one`. TS sometimes intentionally maps several TMA
producer warps in one task to the same full barrier and wants one software
arrival carrying the aggregate transaction byte count. `TaskWarpLeader`
therefore keeps every producer warp in the empty-barrier acquire path, but
guards the full-barrier transaction arrive to the first warp assigned to the
task.

cutlass.experimental.task\_scheduling.pipeline.dsl\_user\_op(*fn=None*, *\*args*, *\*\*kwargs*)

*class* cutlass.experimental.task\_scheduling.pipeline.TSPipelineTmaAsync( : *sync\_object\_full: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_signaling\_thread: \_MockObject*, : *is\_task\_warp\_leader: \_MockObject*, )
:   Bases: [`PipelineTmaAsync`](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineTmaAsync "cutlass.pipeline.sm90.PipelineTmaAsync")

    TMA async pipeline with TS task-warp-leader barrier arming.

    is\_task\_warp\_leader
    :   Predicate selecting the first warp assigned to the task. Only this
        warp arms the transaction barrier when TS uses task-warp-leader
        signaling.

        Type:
        :   [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")

    is\_task\_warp\_leader*: \_MockObject*

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer*, : *is\_task\_warp\_leader: \_MockObject*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, : *defer\_sync: bool = False*, ) → [TSPipelineTmaAsync](pipeline.md#cutlass.experimental.task_scheduling.pipeline.TSPipelineTmaAsync "cutlass.experimental.task_scheduling.pipeline.TSPipelineTmaAsync")
    :   Create an TS TMA async pipeline.

        Parameters:
        :   - **num\_stages** (*int*) – Number of pipeline stages.
            - **producer\_group** ([*cutlass.pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups for producer and consumer sides.
            - **consumer\_group** ([*cutlass.pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups for producer and consumer sides.
            - **tx\_count** (*int*) – Transaction byte count expected by the full barrier.
            - **barrier\_storage** (*cute.Pointer*) – SMEM mbarrier storage.
            - **is\_task\_warp\_leader** ([*Boolean*](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")) – Predicate controlling task-warp-leader transaction arming.
            - **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) – Cluster layout.
            - **mcast\_mode\_mn** (*tuple**[**int**,* *int**]**,* *optional*) – TMA multicast mode.
            - **defer\_sync** (*bool**,* *optional*) – Defer pipeline initialization synchronization.

        Returns:
        :   Pipeline instance with TS producer acquire semantics.

        Return type:
        :   [TSPipelineTmaAsync](pipeline.md#cutlass.experimental.task_scheduling.pipeline.TSPipelineTmaAsync "cutlass.experimental.task_scheduling.pipeline.TSPipelineTmaAsync")

    producer\_acquire( : *state: [PipelineState](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: \_MockObject | None = None*, ) → None
    :   Wait for empty on all producer warps; arm full only on task leader.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_signaling\_thread: \_MockObject*, : *is\_task\_warp\_leader: \_MockObject*, ) → None

*class* cutlass.experimental.task\_scheduling.pipeline.TSPipelineUmmaUmma( : *sync\_object\_full: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *cta\_group: [CtaGroup](../cute_dsl_api/cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )
:   Bases: [`PipelineAsync`](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    Pipeline for a UMMA producer feeding a UMMA consumer.

    cta\_group
    :   TCGen05 CTA group used by producer commit and consumer release.

        Type:
        :   [cute.nvgpu.tcgen05.CtaGroup](../cute_dsl_api/cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.CtaGroup")

    cta\_group*: [CtaGroup](../cute_dsl_api/cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, ) → [TSPipelineUmmaUmma](pipeline.md#cutlass.experimental.task_scheduling.pipeline.TSPipelineUmmaUmma "cutlass.experimental.task_scheduling.pipeline.TSPipelineUmmaUmma")
    :   Create an async pipeline with TCGen05 arrivals on both sides.

        Parameters:
        :   - **num\_stages** (*int*) – Number of pipeline stages.
            - **producer\_group** ([*cutlass.pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups for producer and consumer sides.
            - **consumer\_group** ([*cutlass.pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups for producer and consumer sides.
            - **barrier\_storage** (*cute.Pointer*) – SMEM mbarrier storage.
            - **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) – Cluster layout used to select CTA group and peer masks.
            - **defer\_sync** (*bool**,* *optional*) – Defer pipeline initialization synchronization.

        Returns:
        :   Pipeline with UMMA-style full and empty barrier arrivals.

        Return type:
        :   [TSPipelineUmmaUmma](pipeline.md#cutlass.experimental.task_scheduling.pipeline.TSPipelineUmmaUmma "cutlass.experimental.task_scheduling.pipeline.TSPipelineUmmaUmma")

    producer\_commit( : *state: [PipelineState](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   UMMA producer commit of the full barrier.

    consumer\_release( : *state: [PipelineState](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   UMMA consumer release of the empty barrier.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *cta\_group: [CtaGroup](../cute_dsl_api/cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None

*class* cutlass.experimental.task\_scheduling.pipeline.TSPipelineAsyncUmma( : *sync\_object\_full: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *cta\_group: [CtaGroup](../cute_dsl_api/cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *producer\_op: [PipelineOp](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp") = PipelineOp.AsyncThread*, )
:   Bases: [`PipelineAsyncUmma`](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineAsyncUmma "cutlass.pipeline.sm100.PipelineAsyncUmma")

    TS async-producer pipeline feeding a UMMA consumer.

    CUTLASS `PipelineUmmaConsumerAsync` does not use the
    generic async-thread destination-rank arrive for 2SM producer commits.
    It clears SM100’s peer bit on the full-barrier address and emits one
    cluster-shared mbarrier arrive per producer thread, targeting SM0 of the
    collaborating pair.

    producer\_op*: [PipelineOp](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp")* *= 1*

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *producer\_op: [PipelineOp](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp") = PipelineOp.AsyncThread*, : *defer\_sync: bool = False*, ) → [TSPipelineAsyncUmma](pipeline.md#cutlass.experimental.task_scheduling.pipeline.TSPipelineAsyncUmma "cutlass.experimental.task_scheduling.pipeline.TSPipelineAsyncUmma")
    :   Create an TS async-producer to UMMA-consumer pipeline.

        Parameters:
        :   - **num\_stages** (*int*) – Number of pipeline stages.
            - **producer\_group** ([*cutlass.pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups for producer and consumer sides.
            - **consumer\_group** ([*cutlass.pipeline.CooperativeGroup*](../cute_dsl_api/pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – Cooperative groups for producer and consumer sides.
            - **barrier\_storage** (*cute.Pointer*) – SMEM mbarrier storage.
            - **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) – Cluster layout.
            - **producer\_op** ([*cutlass.pipeline.PipelineOp*](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.PipelineOp")*,* *optional*) – Producer-side operation, either `AsyncThread` or `AsyncLoad`.
            - **defer\_sync** (*bool**,* *optional*) – Defer pipeline initialization synchronization.

        Returns:
        :   Pipeline instance with TS producer commit semantics.

        Return type:
        :   [TSPipelineAsyncUmma](pipeline.md#cutlass.experimental.task_scheduling.pipeline.TSPipelineAsyncUmma "cutlass.experimental.task_scheduling.pipeline.TSPipelineAsyncUmma")

    producer\_commit( : *state: [PipelineState](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   Publish full barrier using CUTLASS 2SM UMMA-consumer semantics.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](../cute_dsl_api/pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *cta\_group: [CtaGroup](../cute_dsl_api/cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *producer\_op: [PipelineOp](../cute_dsl_api/pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp") = PipelineOp.AsyncThread*, ) → None
