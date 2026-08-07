# cutlass.pipeline

*class* cutlass.pipeline.Agent(*value*)
:   Bases: `Enum`

    Agent indicates what is participating in the pipeline synchronization.

    Thread *= 1*

    Warp *= 2*

    ThreadBlock *= 3*

    ThreadBlockCluster *= 4*

*class* cutlass.pipeline.CooperativeGroup( : *agent: [Agent](pipeline.md#cutlass.pipeline.Agent "cutlass.pipeline.helpers.Agent")*, : *size: int | \_MockObject = 1*, : *alignment: int | None = None*, )
:   Bases: `object`

    CooperativeGroup contains size restrictions for an Agent.

    \_\_init\_\_( : *agent: [Agent](pipeline.md#cutlass.pipeline.Agent "cutlass.pipeline.helpers.Agent")*, : *size: int | \_MockObject = 1*, : *alignment: int | None = None*, )

*class* cutlass.pipeline.PipelineOp(*value*)
:   Bases: `Enum`

    PipelineOp assigns an operation to an agent corresponding to a specific hardware feature.

    AsyncThread *= 1*

    TCGen05Mma *= 2*

    TmaLoad *= 3*

    ClcLoad *= 4*

    TmaStore *= 5*

    Composite *= 6*

    AsyncLoad *= 7*

*class* cutlass.pipeline.SyncObject
:   Bases: `ABC`

    Abstract base class for hardware synchronization primitives.

    This class defines the interface for different types of hardware synchronization
    mechanisms including shared memory barriers, named barriers, and fences.

    *abstract* arrive(*\*args: Any*, *\*\*kwargs: Any*) → None

    *abstract* wait(*\*args: Any*, *\*\*kwargs: Any*) → None

    *abstract* arrive\_and\_wait() → None

    *abstract* arrive\_and\_drop() → None

    *abstract* get\_barrier() → cutlass.cute.typing.Pointer | int | None

    *abstract* max() → int | None

    \_abc\_impl *= <\_abc.\_abc\_data object>*

*class* cutlass.pipeline.MbarrierLayout(*value*)
:   Bases: `Enum`

    Layout of mbarrier used for synchronization.

    V0 *= 1*

*class* cutlass.pipeline.MbarrierArray
:   Bases: [`SyncObject`](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

    MbarrierArray implements an abstraction for an array of smem barriers.

    \_\_init\_\_() → None

    recast\_to\_new\_op\_type( : *new\_op\_type: [PipelineOp](pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp")*, ) → [MbarrierArray](pipeline.md#cutlass.pipeline.MbarrierArray "cutlass.pipeline.helpers.MbarrierArray")
    :   Creates a copy of MbarrierArray with a different op\_type without re-initializing barriers

    \_mbar\_scope(*op: str*) → Any
    :   Return a Scope context manager for barrier identification.

        Format: `name:op` (e.g. `smem_kv:wait`, `tmem_sp0:arrive`).
        Profiling tools group by the `name` prefix and classify by the `op` suffix.

        Usage:

        ```console
        with self._mbar_scope("wait"):
            cute.arch.mbarrier_wait(...)
        ```

    mbarrier\_init() → None
    :   Initializes an array of mbarriers using warp 0.

    arrive( : *index: int*, : *dst: int*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup") | None = None*, ) → None
    :   Select the arrive corresponding to this MbarrierArray’s PipelineOp.

        Parameters:
        :   - **index** (*int*) – Index of the mbarrier in the array to arrive on
            - **dst** (*int* *|* *None*) – Destination parameter for selective arrival, which can be either a mask or destination cta rank.
              When None, both `TCGen05Mma` and `AsyncThread` will arrive on their local mbarrier.
              - For `TCGen05Mma`, `dst` serves as a multicast mask (e.g., 0b1011 allows arrive signal to be multicast to CTAs
              in the cluster with rank = 0, 1, and 3).
              - For `AsyncThread`, `dst` serves as a destination cta rank (e.g., 3 means threads will arrive on
              the mbarrier with rank = 3 in the cluster).
            - **cta\_group** (`cute.nvgpu.tcgen05.CtaGroup`, optional) – CTA group for `TCGen05Mma`, defaults to None for other op types

    arrive\_mbarrier( : *index: int*, : *dst\_rank: int | None = None*, ) → None

    arrive\_cp\_async\_mbarrier(*index: int*) → None

    arrive\_tcgen05mma( : *index: int*, : *mask: int | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None

    arrive\_and\_expect\_tx(*index: int*, *tx\_count: int*) → None

    arrive\_and\_expect\_tx\_with\_dst( : *index: int*, : *tx\_count: int*, : *dst: int | None = None*, ) → None

    try\_wait( : *index: int*, : *phase: int*, ) → \_MockObject

    test\_wait( : *index: int*, : *phase: int*, ) → \_MockObject

    wait(*index: int*, *phase: int*) → None
    :   Wait on mbarrier.

        Parameters:
        :   - **index** – Index of the mbarrier in the array
            - **phase** – Phase/parity to wait for (0 or 1)

    arrive\_and\_wait( : *index: int*, : *phase: int*, : *dst: int*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup") | None = None*, ) → None

    arrive\_and\_drop() → None

    get\_barrier(*index: int*) → cutlass.cute.typing.Pointer

    max() → int

    \_abc\_impl *= <\_abc.\_abc\_data object>*

*class* cutlass.pipeline.NamedBarrier( : *barrier\_id: int | \_MockObject*, : *num\_threads: int | \_MockObject*, )
:   Bases: [`SyncObject`](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

    NamedBarrier is an abstraction for named barriers managed by hardware.
    There are 16 named barriers available, with barrier\_ids 0-15.

    See the [PTX documentation](https://https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar).

    barrier\_id*: int | \_MockObject*

    num\_threads*: int | \_MockObject*

    arrive() → None
    :   The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.

    arrive\_unaligned() → None
    :   The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

    wait() → None
    :   NamedBarriers do not have a standalone wait like mbarriers, only an arrive\_and\_wait.
        If synchronizing two warps in a producer/consumer pairing, the arrive count would be
        32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
        or consumer are counted for mbarriers, while all threads participating in the sync
        are counted for NamedBarriers.

    wait\_unaligned() → None

    arrive\_and\_wait() → None

    arrive\_and\_drop() → None

    sync() → None

    get\_barrier() → int | \_MockObject

    max() → int

    \_\_init\_\_( : *barrier\_id: int | \_MockObject*, : *num\_threads: int | \_MockObject*, ) → None

    \_abc\_impl *= <\_abc.\_abc\_data object>*

*class* cutlass.pipeline.PipelineOrder( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *depth: int*, : *length: int*, : *group\_id: int*, : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )
:   Bases: `object`

    PipelineOrder is used for managing ordered pipeline execution with multiple groups.

    This class implements a pipeline ordering mechanism where work is divided into groups
    and stages, allowing for controlled progression through pipeline stages with proper
    synchronization between different groups.

    The pipeline ordering works as follows:
    - The pipeline is divided into ‘length’ number of groups
    - Each group has ‘depth’ number of stages
    - Groups execute in a specific order with synchronization barriers
    - Each group waits for the previous group to complete before proceeding

    **Example:**

    ```python
    # Create pipeline order with 3 groups, each with 2 stages
    pipeline_order = PipelineOrder.create(
        barrier_storage=smem_ptr,      # shared memory pointer for barriers
        depth=2,                       # 2 stages per group
        length=3,                      # 3 groups total
        group_id=0,                    # current group ID (0, 1, or 2)
        producer_group=producer_warp   # cooperative group for producers
    )

    # In the pipeline loop
    for stage in range(num_stages):
        pipeline_order.wait()          # Wait for previous group to complete
        # Process current stage
        pipeline_order.arrive()        # Signal completion to next group
    ```

    sync\_object\_full*: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    depth*: int*

    length*: int*

    group\_id*: int*

    state*: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*

    *static* create( : *\**, : *depth: int*, : *length: int*, : *group\_id: int*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *defer\_sync: bool = False*, : *name: str = ''*, ) → [PipelineOrder](pipeline.md#cutlass.pipeline.PipelineOrder "cutlass.pipeline.sm90.PipelineOrder")

    get\_barrier\_for\_current\_stage\_idx( : *group\_id: int*, : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState") | None = None*, ) → \_MockObject

    arrive( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState") | None = None*, ) → [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState") | None

    wait( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState") | None = None*, ) → None

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *depth: int*, : *length: int*, : *group\_id: int*, : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None

*class* cutlass.pipeline.TmaStoreFence(*num\_stages: int = 0*)
:   Bases: [`SyncObject`](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

    TmaStoreFence is used for a multi-stage epilogue buffer.

    \_\_init\_\_(*num\_stages: int = 0*) → None

    arrive() → None

    wait() → None

    arrive\_and\_wait() → None

    arrive\_and\_drop() → None

    get\_barrier() → None

    max() → None

    tail() → None

    \_abc\_impl *= <\_abc.\_abc\_data object>*

*class* cutlass.pipeline.PipelineUserType(*value*)
:   Bases: `Enum`

    An enumeration.

    Producer *= 1*

    Consumer *= 2*

    ProducerConsumer *= 3*

*class* cutlass.pipeline.PipelineState( : *stages: int*, : *count: \_MockObject*, : *index: \_MockObject*, : *phase: \_MockObject*, )
:   Bases: `object`

    Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.

    \_\_init\_\_( : *stages: int*, : *count: \_MockObject*, : *index: \_MockObject*, : *phase: \_MockObject*, )

    clone() → [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")

    *property* index*: \_MockObject*

    *property* count*: \_MockObject*

    *property* stages*: int*

    *property* phase*: \_MockObject*

    reset\_count() → None

    advance() → None

    reverse() → None

*class* cutlass.pipeline.PipelineAsync( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, )
:   Bases: `object`

    PipelineAsync is a generic pipeline class where both the producer and consumer are
    AsyncThreads. It also serves as a base class for specialized pipeline classes.

    This class implements a producer-consumer pipeline pattern where both sides operate
    asynchronously. The pipeline maintains synchronization state using barrier objects
    to coordinate between producer and consumer threads.

    The pipeline state transitions of one pipeline entry(mbarrier) can be represented as:

    Table 5 Pipeline State Transitions

    | Barrier | State | p.acquire | p.commit | c.wait | c.release |
    | --- | --- | --- | --- | --- | --- |
    | empty\_bar | empty | <Return> | n/a | n/a |  |
    | empty\_bar | wait | <Block> | n/a | n/a | -> empty |
    | full\_bar | wait | n/a | -> full | <Block > | n/a |
    | full\_bar | full | n/a |  | <Return> | n/a |

    Where:

    - p: producer
    - c: consumer
    - <Block>: This action is blocked until transition to a state allow it to proceed by other side
      - e.g. `p.acquire()` is blocked until `empty_bar` transition to `empty` state by `c.release()`

    ```text
    Array of mbarriers as circular buffer:

         Advance Direction
       <-------------------

        Producer   Consumer
            |         ^
            V         |
       +-----------------+
     --|X|X|W|D|D|D|D|R|X|<-.
    /  +-----------------+   \
    |                        |
    `------------------------'
    ```

    Where:

    - X: Empty buffer (initial state)
    - W: Producer writing (producer is waiting for buffer to be empty)
    - D: Data ready (producer has written data to buffer)
    - R: Consumer reading (consumer is consuming data from buffer)

    **Example:**

    ```python
    # Create pipeline with 5 stages
    pipeline = PipelineAsync.create(
        num_stages=5,                   # number of pipeline stages
        producer_group=producer_warp,
        consumer_group=consumer_warp
        barrier_storage=smem_ptr,       # smem pointer for array of mbarriers in shared memory
    )

    producer, consumer = pipeline.make_participants()
    # Producer side
    for i in range(num_iterations):
        handle = producer.acquire_and_advance()  # Wait for buffer to be empty & Move index to next stage
        # Write data to pipeline buffer
        handle.commit()   # Signal buffer is full

    # Consumer side
    for i in range(num_iterations):
        handle = consumer.wait_and_advance()     # Wait for buffer to be full & Move index to next stage
        # Read data from pipeline buffer
        handle.release()  # Signal buffer is empty
    ```

    sync\_object\_full*: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    sync\_object\_empty*: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    num\_stages*: int*

    producer\_mask*: \_MockObject | None*

    consumer\_mask*: \_MockObject | None*

    *static* \_make\_sync\_object( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: int*, : *agent: tuple[[PipelineOp](pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp"), [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")]*, : *tx\_count: int = 0*, : *name: str = ''*, : *phase: Literal['', 'full', 'empty'] = ''*, ) → [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")
    :   Returns a SyncObject corresponding to an agent’s PipelineOp.

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *producer\_mask: \_MockObject | None = None*, : *consumer\_mask: \_MockObject | None = None*, : *defer\_sync: bool = False*, : *name: str = ''*, ) → [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")
    :   Creates and initializes a new PipelineAsync instance.

        This helper function computes necessary attributes and returns an instance of PipelineAsync
        with the specified configuration for producer and consumer synchronization.

        Parameters:
        :   - **barrier\_storage** (*cute.Pointer*) – Pointer to the shared memory address for this pipeline’s mbarriers
            - **num\_stages** (*int*) – Number of buffer stages for this pipeline
            - **producer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent
            - **consumer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the consumer agent
            - **producer\_mask** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")*,* *optional*) – Mask for signaling arrives for the producer agent
            - **consumer\_mask** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")*,* *optional*) – Mask for signaling arrives for the consumer agent

        Raises:
        :   **ValueError** – If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   A new `PipelineAsync` instance

        Return type:
        :   [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")

    producer\_acquire( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: \_MockObject | None = None*, ) → None

    producer\_try\_acquire( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → \_MockObject

    producer\_commit( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None

    consumer\_wait( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_wait\_token: \_MockObject | None = None*, ) → None

    consumer\_try\_wait( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → \_MockObject

    consumer\_release( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None

    consumer\_get\_barrier( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → cutlass.cute.typing.Pointer

    producer\_get\_barrier( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → cutlass.cute.typing.Pointer

    producer\_tail( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   Make sure the last used buffer empty signal is visible to producer.
        Producer tail is usually executed by producer before exit, to avoid dangling
        mbarrier arrive signals after kernel exit.

        Parameters:
        :   **state** ([*PipelineState*](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) – The pipeline state that points to next useful buffer

    make\_producer() → [PipelineProducer](pipeline.md#cutlass.pipeline.PipelineProducer "cutlass.pipeline.sm90.PipelineProducer")

    make\_consumer() → [PipelineConsumer](pipeline.md#cutlass.pipeline.PipelineConsumer "cutlass.pipeline.sm90.PipelineConsumer")

    make\_participants() → tuple[[PipelineProducer](pipeline.md#cutlass.pipeline.PipelineProducer "cutlass.pipeline.sm90.PipelineProducer"), [PipelineConsumer](pipeline.md#cutlass.pipeline.PipelineConsumer "cutlass.pipeline.sm90.PipelineConsumer")]

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, ) → None

*class* cutlass.pipeline.PipelineCpAsync( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, )
:   Bases: [`PipelineAsync`](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineCpAsync is used for CpAsync producers and AsyncThread consumers (e.g. Hopper load mainloops).

    *static* create( : *\**, : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: \_MockObject*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *producer\_mask: \_MockObject | None = None*, : *consumer\_mask: \_MockObject | None = None*, : *defer\_sync: bool = False*, : *name: str = ''*, ) → [PipelineCpAsync](pipeline.md#cutlass.pipeline.PipelineCpAsync "cutlass.pipeline.sm90.PipelineCpAsync")
    :   Helper function that computes necessary attributes and returns a `PipelineCpAsync` instance.

        Parameters:
        :   - **barrier\_storage** (*cute.Pointer*) – Pointer to the shared memory address for this pipeline’s mbarriers
            - **num\_stages** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Number of buffer stages for this pipeline
            - **producer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent
            - **consumer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the consumer agent
            - **producer\_mask** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")*,* *optional*) – Mask for signaling arrives for the producer agent, defaults to None
            - **consumer\_mask** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")*,* *optional*) – Mask for signaling arrives for the consumer agent, defaults to None

        Returns:
        :   A new `PipelineCpAsync` instance configured with the provided parameters

        Return type:
        :   [PipelineCpAsync](pipeline.md#cutlass.pipeline.PipelineCpAsync "cutlass.pipeline.PipelineCpAsync")

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, ) → None

*class* cutlass.pipeline.PipelineTmaAsync( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_signaling\_thread: \_MockObject*, )
:   Bases: [`PipelineAsync`](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaAsync is used for TMA producers and AsyncThread consumers (e.g. Hopper mainloops).

    is\_signaling\_thread*: \_MockObject*

    *static* init\_empty\_barrier\_arrive\_signal( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *tidx: \_MockObject*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, ) → tuple[\_MockObject, \_MockObject]
    :   Initialize the empty barrier arrive signal.

        This function determines which threads should signal empty barrier arrives based on the cluster layout
        and multicast modes. It returns the destination CTA rank and whether the current thread should signal.

        Parameters:
        :   - **cta\_layout\_vmnk** (*cute.Layout*) – Layout describing the cluster shape and CTA arrangement
            - **tidx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Thread index within the warp
            - **mcast\_mode\_mn** (*tuple**[**int**,* *int**]*) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)

        Raises:
        :   **AssertionError** – If both multicast modes are disabled (0,0)

        Returns:
        :   Tuple containing destination CTA rank and boolean indicating if current thread signals

        Return type:
        :   tuple[[Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32"), [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")]

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *tidx: \_MockObject | None = None*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, : *enable\_multicast\_signaling: bool = False*, : *defer\_sync: bool = False*, : *name: str = ''*, ) → [PipelineTmaAsync](pipeline.md#cutlass.pipeline.PipelineTmaAsync "cutlass.pipeline.sm90.PipelineTmaAsync")
    :   Create a new `PipelineTmaAsync` instance.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffer stages for this pipeline
            - **producer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent
            - **consumer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the consumer agent
            - **tx\_count** (*int*) – Number of bytes expected to be written to the transaction barrier for one stage
            - **barrier\_storage** (*cute.Pointer**,* *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers, defaults to None
            - **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) – Layout of the cluster shape, defaults to None
            - **tidx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")*,* *optional*) – Thread index to consumer async threads, defaults to None
            - **mcast\_mode\_mn** (*tuple**[**int**,* *int**]**,* *optional*) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)
            - **enable\_multicast\_signaling** (*bool**,* *optional*) – When `True`, the CooperativeGroup is expected
              to represent the number of threads in a CTA calling
              consumer\_wait/consumer\_release, and the actual arrive count is recomputed
              internally. Multicast is handled automatically based on cta\_layout\_vmnk and
              mcast\_mode\_mn. Defaults to `False`, which skips this logic and uses the
              consumer arrive count specified by the user.
            - **defer\_sync** (*bool**,* *optional*) – Bool specifying whether or not to skip the built-in mbarrier fence and sync for performance, defaults to False

        Raises:
        :   **ValueError** – If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   New `PipelineTmaAsync` instance

        Return type:
        :   [PipelineTmaAsync](pipeline.md#cutlass.pipeline.PipelineTmaAsync "cutlass.pipeline.PipelineTmaAsync")

    producer\_acquire( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: \_MockObject | None = None*, ) → None
    :   TMA producer commit conditionally waits on buffer empty and sets the transaction barrier.

    producer\_commit( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   TMA producer commit is a noop since TMA instruction itself updates the transaction count.

    consumer\_release( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   TMA consumer release conditionally signals the empty buffer to the producer.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_signaling\_thread: \_MockObject*, ) → None

*class* cutlass.pipeline.PipelineTmaUmma( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_leader\_cta: bool*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )
:   Bases: [`PipelineAsync`](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).

    is\_leader\_cta*: bool*

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*

    *static* \_make\_sync\_object( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: int*, : *agent: tuple[[PipelineOp](pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp"), [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")]*, : *tx\_count: int = 0*, : *mbarrier\_layout: [MbarrierLayout](pipeline.md#cutlass.pipeline.MbarrierLayout "cutlass.pipeline.helpers.MbarrierLayout") = MbarrierLayout.V0*, : *name: str = ''*, : *phase: Literal['', 'full', 'empty'] = ''*, ) → [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")
    :   Returns a SyncObject corresponding to an agent’s PipelineOp.

    *static* \_compute\_mcast\_arrival\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *mcast\_mode\_mn: tuple[int, int]*, ) → \_MockObject
    :   Computes a mask for signaling arrivals to multicasting threadblocks.

    *static* \_compute\_is\_leader\_cta( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, ) → \_MockObject
    :   Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, : *enable\_multicast\_signaling: bool = False*, : *defer\_sync: bool = False*, : *name: str = ''*, ) → [PipelineTmaUmma](pipeline.md#cutlass.pipeline.PipelineTmaUmma "cutlass.pipeline.sm100.PipelineTmaUmma")
    :   Creates and initializes a new PipelineTmaUmma instance.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffer stages for this pipeline
            - **producer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – CooperativeGroup for the producer agent
            - **consumer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – CooperativeGroup for the consumer agent
            - **tx\_count** (*int*) – Number of bytes expected to be written to the transaction barrier for one stage
            - **barrier\_storage** (*cute.Pointer**,* *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers
            - **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) – Layout of the cluster shape
            - **mcast\_mode\_mn** (*tuple**[**int**,* *int**]**,* *optional*) – Tuple specifying multicast modes for m and n dimensions (each 0 or 1)
            - **enable\_multicast\_signaling** (*bool**,* *optional*) – See docstring in PipelineTmaAsync.create() for details
            - **defer\_sync** (*bool**,* *optional*) – Bool specifying whether or not to skip the built-in mbarrier fence and sync for performance, defaults to False

        Raises:
        :   **ValueError** – If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   A new PipelineTmaUmma instance configured with the provided parameters

        Return type:
        :   [PipelineTmaUmma](pipeline.md#cutlass.pipeline.PipelineTmaUmma "cutlass.pipeline.PipelineTmaUmma")

    consumer\_release( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   UMMA consumer release buffer empty, cta\_group needs to be provided.

    producer\_acquire( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: \_MockObject | None = None*, : *\**, : *expected\_tx: \_MockObject | None = None*, ) → None
    :   TMA producer conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.

        Parameters:
        :   **expected\_tx** – Override the expected transaction byte count for this
            acquire. When `None` (default), uses the `tx_count` from barrier init.
            Pass a dynamic value for workloads where the byte count varies per
            iteration (e.g. sparse GEMM with conditional metadata loading).

    producer\_commit( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   TMA producer commit is a noop since TMA instruction itself updates the transaction count.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_leader\_cta: bool*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None

*class* cutlass.pipeline.PipelineAsyncUmma( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )
:   Bases: [`PipelineAsync`](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineAsyncUmma is used for AsyncThread producers and UMMA consumers (e.g. Blackwell input fusion pipelines).

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*

    *static* \_compute\_leading\_cta\_rank( : *cta\_v\_size: int*, ) → \_MockObject
    :   Computes the leading CTA rank.

    *static* \_compute\_is\_leader\_cta( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, ) → \_MockObject
    :   Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

    *static* \_compute\_peer\_cta\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, ) → \_MockObject
    :   Computes a mask for signaling arrivals to multicasting threadblocks.

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, : *name: str = ''*, ) → [PipelineAsyncUmma](pipeline.md#cutlass.pipeline.PipelineAsyncUmma "cutlass.pipeline.sm100.PipelineAsyncUmma")
    :   Creates and initializes a new PipelineAsyncUmma instance.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffer stages for this pipeline
            - **producer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – CooperativeGroup for the producer agent
            - **consumer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – CooperativeGroup for the consumer agent
            - **barrier\_storage** (*cute.Pointer**,* *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers
            - **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) – Layout of the cluster shape

        Raises:
        :   **ValueError** – If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   A new PipelineAsyncUmma instance configured with the provided parameters

        Return type:
        :   [PipelineAsyncUmma](pipeline.md#cutlass.pipeline.PipelineAsyncUmma "cutlass.pipeline.PipelineAsyncUmma")

    consumer\_release( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   UMMA consumer release buffer empty, cta\_group needs to be provided.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None

*class* cutlass.pipeline.PipelineUmmaAsync( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )
:   Bases: [`PipelineAsync`](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineUmmaAsync is used for UMMA producers and AsyncThread consumers (e.g. Blackwell accumulator pipelines).

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*

    *static* \_compute\_tmem\_sync\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, ) → \_MockObject
    :   Computes a mask to signal completion of tmem buffers for 2CTA kernels.

    *static* \_compute\_peer\_cta\_rank() → \_MockObject
    :   Computes a mask to signal release of tmem buffers for 2CTA kernels.

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, : *name: str = ''*, ) → [PipelineUmmaAsync](pipeline.md#cutlass.pipeline.PipelineUmmaAsync "cutlass.pipeline.sm100.PipelineUmmaAsync")
    :   Creates an instance of PipelineUmmaAsync with computed attributes.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffer stages for this pipeline
            - **producer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent
            - **consumer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the consumer agent
            - **barrier\_storage** (*cute.Pointer**,* *optional*) – Pointer to the shared memory address for this pipeline’s mbarriers
            - **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) – Layout of the cluster shape

        Raises:
        :   **ValueError** – If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   New instance of `PipelineUmmaAsync`

        Return type:
        :   [PipelineUmmaAsync](pipeline.md#cutlass.pipeline.PipelineUmmaAsync "cutlass.pipeline.PipelineUmmaAsync")

    producer\_commit( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   UMMA producer commit buffer full, cta\_group needs to be provided.

    producer\_tail( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   Make sure the last used buffer empty signal is visible to producer.
        Producer tail is usually executed by producer before exit, to avoid dangling
        mbarrier arrive signals after kernel exit.

        Parameters:
        :   **state** ([*PipelineState*](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) – The pipeline state that points to next useful buffer

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None

*class* cutlass.pipeline.PipelineClcFetchAsync( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_signaling\_thread: \_MockObject*, )
:   Bases: [`PipelineAsync`](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineClcFetchAsync implements a producer-consumer pipeline for Cluster Launch
    Control based dynamic scheduling. Both producer and consumer operate asynchronously
    using barrier synchronization to coordinate across pipeline stages and cluster CTAs.

    - Producer: waits for empty buffer, signals full barrier with transection bytes
      across all CTAs in cluster, hardware autosignals each CTA’s mbarrier when
      transaction bytes are written, then the satte advance to next buffer slot.
    - Consumer: waits for full barrier, then load respinse from local SMEM, then
      sigals CTA 0’s empty barrier to allow buffer reuse.

    sync\_object\_full*: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    sync\_object\_empty*: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    num\_stages*: int*

    producer\_mask*: \_MockObject | None*

    consumer\_mask*: \_MockObject | None*

    is\_signaling\_thread*: \_MockObject*

    *static* \_init\_full\_barrier\_arrive\_signal( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *tidx: \_MockObject*, ) → tuple[\_MockObject, \_MockObject]
    :   Computes producer barrier signaling parameters, returns destination CTA rank
        (0 to cluster\_size-1) based on thread ID, and a boolean flag indicating if
        this thread participates in signaling.

        Parameters:
        :   - **cta\_layout\_vmnk** – Cluster layout defining CTA count
            - **tidx** – Thread ID within the CTA

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *producer\_mask: \_MockObject | None = None*, : *consumer\_mask: \_MockObject | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, : *name: str = ''*, ) → [PipelineClcFetchAsync](pipeline.md#cutlass.pipeline.PipelineClcFetchAsync "cutlass.pipeline.sm100.PipelineClcFetchAsync")
    :   This helper function computes any necessary attributes and returns an instance of PipelineClcFetchAsync.
        :param barrier\_storage: Pointer to the shared memory address for this pipeline’s mbarriers
        :type barrier\_storage: cute.Pointer
        :param num\_stages: Number of buffer stages for this pipeline
        :type num\_stages: int
        :param producer\_group: CooperativeGroup for the producer agent
        :type producer\_group: CooperativeGroup
        :param consumer\_group: CooperativeGroup for the consumer agent
        :type consumer\_group: CooperativeGroup
        :param tx\_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx\_count: int
        :param producer\_mask: Mask for signaling arrives for the producer agent, defaults to `None`
        :type producer\_mask: Int32, optional
        :param consumer\_mask: Mask for signaling arrives for the consumer agent, defaults to `None`
        :type consumer\_mask: Int32, optional

    producer\_acquire( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: \_MockObject | None = None*, ) → None
    :   Producer acquire waits for empty buffer and sets transaction expectation on full barrier.

        Parameters:
        :   - **state** – Pipeline state pointing to the current buffer stage
            - **try\_acquire\_token** – Optional token to skip the empty barrier wait

    consumer\_wait( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_wait\_token: \_MockObject | None = None*, ) → None
    :   Consumer waits for full barrier to be signaled by hardware multicast.

        Parameters:
        :   - **state** – Pipeline state pointing to the current buffer stage
            - **try\_wait\_token** – Optional token to skip the full barrier wait

    consumer\_release( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None

    producer\_get\_barrier( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → cutlass.cute.typing.Pointer

    consumer\_get\_barrier( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → cutlass.cute.typing.Pointer

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_signaling\_thread: \_MockObject*, ) → None

*class* cutlass.pipeline.PipelineTmaMultiConsumersAsync( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_leader\_cta: bool*, : *sync\_object\_empty\_umma: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty\_async: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *consumer\_dst\_rank\_async: \_MockObject | None = None*, : *is\_signaling\_thread: \_MockObject = True*, )
:   Bases: [`PipelineAsync`](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaMultiConsumersAsync is used for TMA producers and UMMA+Async consumers.

    is\_leader\_cta*: bool*

    sync\_object\_empty\_umma*: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    sync\_object\_empty\_async*: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*

    consumer\_dst\_rank\_async*: \_MockObject | None* *= None*

    is\_signaling\_thread*: \_MockObject* *= True*

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group\_umma: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group\_async: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, : *tidx: \_MockObject | None = None*, : *enable\_multicast\_signaling: bool = False*, : *defer\_sync: bool = False*, : *force\_deprecated\_per\_lane\_signaling: bool | None = None*, : *name: str = ''*, ) → [PipelineTmaMultiConsumersAsync](pipeline.md#cutlass.pipeline.PipelineTmaMultiConsumersAsync "cutlass.pipeline.sm100.PipelineTmaMultiConsumersAsync")
    :   This helper function computes any necessary attributes and returns an instance of PipelineTmaMultiConsumersAsync.
        :param barrier\_storage: Pointer to the smem address for this pipeline’s mbarriers
        :type barrier\_storage: cute.Pointer
        :param num\_stages: Number of buffer stages for this pipeline
        :type num\_stages: Int32
        :param producer\_group: CooperativeGroup for the producer agent
        :type producer\_group: CooperativeGroup
        :param consumer\_group\_umma: CooperativeGroup for the UMMA consumer agent
        :type consumer\_group\_umma: CooperativeGroup
        :param consumer\_group\_async: CooperativeGroup for the AsyncThread consumer agent
        :type consumer\_group\_async: CooperativeGroup
        :param tx\_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx\_count: int
        :param cta\_layout\_vmnk: Layout of the cluster shape
        :type cta\_layout\_vmnk: cute.Layout | None
        :param mcast\_mode\_mn: Tuple specifying multicast modes for m and n dimensions (each 0 or 1)
        :type mcast\_mode\_mn: tuple[int, int]
        :param tidx: Thread index for computing AsyncThread consumer signaling, defaults to thread\_idx()[0]
        :type tidx: Int32 | None
        :param enable\_multicast\_signaling: See docstring in PipelineTmaAsync.create() for details
        :type enable\_multicast\_signaling: bool, optional
        :param force\_deprecated\_per\_lane\_signaling: **Deprecated.** Set `False` if your arrive count is a multiple of `WARP_SIZE` and you do not want the legacy fallback. Leave unset otherwise.
        :type force\_deprecated\_per\_lane\_signaling: bool | None

    *static* \_init\_empty\_barrier\_arrive\_signal\_2sm( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *tidx: \_MockObject*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, ) → tuple[\_MockObject, \_MockObject]
    :   Identical to sm90.py PipelineTmaAsync.init\_empty\_barrier\_arrive\_signal except
        that CTAs in the multicast will also signal CTAs with a different V-coordinate (i.e. leader/follower CTA pairs).

    producer\_acquire( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: \_MockObject | None = None*, ) → None
    :   TMA producer acquire waits on buffer empty and sets the transaction barrier for leader threadblocks.

    producer\_commit( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None
    :   TMA producer commit is a noop since TMA instruction itself updates the transaction count.

    consumer\_wait( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_wait\_token: \_MockObject | None = None*, ) → None
    :   Consumer waits for full barrier to be signaled.

    consumer\_try\_wait( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → \_MockObject
    :   Non-blocking check if data is ready.

    consumer\_release( : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *op\_type: [PipelineOp](pipeline.md#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp") = PipelineOp.TCGen05Mma*, ) → None

    make\_participants() → tuple[[PipelineProducer](pipeline.md#cutlass.pipeline.PipelineProducer "cutlass.pipeline.PipelineProducer"), [PipelineConsumer](pipeline.md#cutlass.pipeline.PipelineConsumer "cutlass.pipeline.PipelineConsumer"), None]
    :   Returns (producer, umma\_consumer, None).

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, : *is\_leader\_cta: bool*, : *sync\_object\_empty\_umma: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty\_async: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *consumer\_dst\_rank\_async: \_MockObject | None = None*, : *is\_signaling\_thread: \_MockObject = True*, ) → None

*class* cutlass.pipeline.PipelineTmaStore( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, )
:   Bases: [`PipelineAsync`](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaStore is used for synchronizing TMA stores in the epilogue. It does not use mbarriers.

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, ) → [PipelineTmaStore](pipeline.md#cutlass.pipeline.PipelineTmaStore "cutlass.pipeline.sm90.PipelineTmaStore")
    :   This helper function computes any necessary attributes and returns an instance of `PipelineTmaStore`.

        Parameters:
        :   - **num\_stages** (*int*) – Number of buffer stages for this pipeline
            - **producer\_group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – `CooperativeGroup` for the producer agent

        Returns:
        :   A new `PipelineTmaStore` instance

        Return type:
        :   [PipelineTmaStore](pipeline.md#cutlass.pipeline.PipelineTmaStore "cutlass.pipeline.PipelineTmaStore")

    producer\_acquire() → None

    producer\_commit() → None

    consumer\_wait() → None

    consumer\_release() → None

    producer\_tail() → None
    :   Make sure the last used buffer empty signal is visible to producer.
        Producer tail is usually executed by producer before exit, to avoid dangling
        mbarrier arrive signals after kernel exit.

        Parameters:
        :   **state** ([*PipelineState*](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) – The pipeline state that points to next useful buffer

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](pipeline.md#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: \_MockObject | None*, : *consumer\_mask: \_MockObject | None*, ) → None

*class* cutlass.pipeline.PipelineProducer( : *pipeline: [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )
:   Bases: `object`

    A class representing a producer in an asynchronous pipeline.

    This class manages the producer side of an asynchronous pipeline, handling
    synchronization and state management for producing data. It provides methods for
    acquiring, committing, and advancing through pipeline stages.

    Variables:
    :   - **\_\_pipeline** – The asynchronous pipeline this producer belongs to
        - **\_\_state** – The current state of the producer in the pipeline
        - **\_\_group** – The cooperative group this producer operates in

    **Examples:**

    ```python
    pipeline = PipelineAsync.create(...)
    producer, consumer = pipeline.make_participants()
    for i in range(iterations):
        # Try to acquire the current buffer without blocking
        try_acquire_token = producer.try_acquire()

        # Do something else independently
        ...

        # Wait for current buffer to be empty & Move index to next stage
        # If try_acquire_token is True, return immediately
        # If try_acquire_token is False, block until buffer is empty
        handle = producer.acquire_and_advance(try_acquire_token)

        # Produce data
        handle.commit()
    ```

    *class* ImmutableResourceHandle( : *\_ImmutableResourceHandle\_\_origin: [cutlass.pipeline.sm90.PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [cutlass.pipeline.helpers.PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )
    :   Bases: `ImmutableResourceHandle`

        *property* barrier*: cutlass.cute.typing.Pointer*
        :   Get the barrier pointer for the current pipeline stage.

            Returns:
            :   Pointer to the barrier for the current stage

            Return type:
            :   cute.Pointer

        commit() → None
        :   Signal that data production is complete for the current stage.

            This allows consumers to start processing the data.

        \_\_init\_\_( : *\_ImmutableResourceHandle\_\_origin: [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None

    \_\_init\_\_( : *pipeline: [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, ) → None
    :   Initialize a new Producer instance.

        Parameters:
        :   - **pipeline** ([*PipelineAsync*](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")) – The pipeline this producer belongs to
            - **state** ([*PipelineState*](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) – Initial pipeline state
            - **group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – The cooperative group for synchronization

    \_\_pipeline*: [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*

    \_\_state*: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*

    \_\_group*: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*

    clone() → [PipelineProducer](pipeline.md#cutlass.pipeline.PipelineProducer "cutlass.pipeline.sm90.PipelineProducer")
    :   Create a new Producer instance with the same state.

    reset() → None
    :   Reset the count of how many handles this producer has committed.

    current\_handle() → [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle")
    :   Get the current handle for the producer.

    acquire( : *try\_acquire\_token: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *\**, : *\*\*kwargs: ~typing.Any*, ) → [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle")
    :   Wait for the current buffer to be empty before producing data.
        This is a blocking operation.

        Parameters:
        :   **try\_acquire\_token** (*Optional**[*[*Boolean*](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*]*) – Optional token to try to acquire the buffer

        Returns:
        :   A handle to the producer for committing the data

        Return type:
        :   [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")

    advance() → None
    :   Move to the next pipeline stage.

    acquire\_and\_advance( : *try\_acquire\_token: ~sphinx.ext.autodoc.mock.\_MockObject | None = None*, : *\**, : *\*\*kwargs: ~typing.Any*, ) → [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle")
    :   Acquire the current buffer and advance to the next pipeline stage.

        This method combines the acquire() and advance() operations into a single call.
        It first waits for the current buffer to be empty before producing data,
        then advances the pipeline to the next stage.

        Parameters:
        :   **try\_acquire\_token** (*Optional**[*[*Boolean*](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*]*) – Token indicating whether to try non-blocking acquire.
            If True, returns immediately without waiting. If False or None, blocks
            until buffer is empty.

        Returns:
        :   A handle to the producer that can be used to commit data to the
            acquired buffer stage

        Return type:
        :   [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")

    try\_acquire() → \_MockObject
    :   Attempt to acquire the current buffer without blocking.

        This method tries to acquire the current buffer stage for producing data
        without waiting. It can be used to check buffer availability before
        committing to a blocking acquire operation.

        Returns:
        :   A boolean token indicating whether the buffer was successfully acquired

        Return type:
        :   [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")

    commit( : *handle: [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle") | None = None*, ) → None
    :   Signal that data production is complete for the current stage.

        This allows consumers to start processing the data.

        Parameters:
        :   **handle** (*Optional**[*[*ImmutableResourceHandle*](pipeline.md#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")*]*) – Optional handle to commit, defaults to None

        Raises:
        :   **AssertionError** – If provided handle does not belong to this producer

    tail() → None
    :   Ensure all used buffers are properly synchronized before producer exit.

        This should be called before the producer finishes to avoid dangling signals.

*class* cutlass.pipeline.PipelineConsumer( : *pipeline: [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )
:   Bases: `object`

    A class representing a consumer in an asynchronous pipeline.

    The Consumer class manages the consumer side of an asynchronous pipeline, handling
    synchronization and state management for consuming data. It provides methods for
    waiting, releasing, and advancing through pipeline stages.

    Variables:
    :   - **\_\_pipeline** – The asynchronous pipeline this consumer belongs to
        - **\_\_state** – The current state of the consumer in the pipeline
        - **\_\_group** – The cooperative group this consumer operates in

    **Examples:**

    ```python
    pipeline = PipelineAsync.create(...)
    producer, consumer = pipeline.make_participants()
    for i in range(iterations):
        # Try to wait for buffer to be full
        try_wait_token = consumer.try_wait()

        # Do something else independently
        ...

        # Wait for buffer to be full & Move index to next stage
        # If try_wait_token is True, return immediately
        # If try_wait_token is False, block until buffer is full
        handle = consumer.wait_and_advance(try_wait_token)

        # Consume data
        handle.release(  )  # Signal buffer is empty

        # Alternative way to do this is:
        # handle.release()  # Signal buffer is empty
    ```

    *class* ImmutableResourceHandle( : *\_ImmutableResourceHandle\_\_origin: [cutlass.pipeline.sm90.PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [cutlass.pipeline.helpers.PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )
    :   Bases: `ImmutableResourceHandle`

        *property* barrier*: cutlass.cute.typing.Pointer*
        :   Get the barrier pointer for the current pipeline stage.

            Returns:
            :   Pointer to the barrier for the current stage

            Return type:
            :   cute.Pointer

        release() → None
        :   Signal that data production is complete for the current stage.
            This allows consumers to start processing the data.

        \_\_init\_\_( : *\_ImmutableResourceHandle\_\_origin: [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None

    \_\_init\_\_( : *pipeline: [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *state: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *group: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, ) → None
    :   Initialize a new Consumer instance.

        Parameters:
        :   - **pipeline** ([*PipelineAsync*](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")) – The pipeline this consumer belongs to
            - **state** ([*PipelineState*](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) – Initial pipeline state
            - **group** ([*CooperativeGroup*](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) – The cooperative group for synchronization

    \_\_pipeline*: [PipelineAsync](pipeline.md#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*

    \_\_group*: [CooperativeGroup](pipeline.md#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*

    \_\_state*: [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*

    clone() → [PipelineConsumer](pipeline.md#cutlass.pipeline.PipelineConsumer "cutlass.pipeline.sm90.PipelineConsumer")
    :   Create a new Consumer instance with the same state.

    reset() → None
    :   Reset the count of how many handles this consumer has consumed.

    current\_handle() → [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle")
    :   Get the current handle for the consumer.

    wait( : *try\_wait\_token: \_MockObject | None = None*, ) → [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle")
    :   Wait for data to be ready in the current buffer. This is a blocking operation
        that will not return until data is available.

        Parameters:
        :   **try\_wait\_token** (*Optional**[*[*Boolean*](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*]*) – Token used to attempt a non-blocking wait for the buffer.
            If provided and True, returns immediately if buffer is not ready.

        Returns:
        :   An immutable handle to the consumer that can be used to release the buffer
            once data consumption is complete

        Return type:
        :   [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle")

    advance() → None
    :   Advance the consumer to the next pipeline stage.

        This updates the internal state to point to the next buffer in the pipeline.
        Should be called after consuming data from the current buffer.

    wait\_and\_advance( : *try\_wait\_token: \_MockObject | None = None*, ) → [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle")
    :   Atomically wait for data and advance to next pipeline stage.

        This is a convenience method that combines wait() and advance() into a single
        atomic operation. It will block until data is available in the current buffer,
        then automatically advance to the next stage.

        Parameters:
        :   **try\_wait\_token** (*Optional**[*[*Boolean*](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*]*) – Token used to attempt a non-blocking wait for the buffer.
            If provided and True, returns immediately if buffer is not ready.

        Returns:
        :   An immutable handle to the consumer that can be used to release the buffer
            once data consumption is complete

        Return type:
        :   [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle")

    try\_wait() → \_MockObject
    :   Non-blocking check if data is ready in the current buffer.

        This method provides a way to test if data is available without blocking.
        Unlike wait(), this will return immediately regardless of buffer state.

        Returns:
        :   True if data is ready to be consumed, False if the buffer is not yet ready

        Return type:
        :   [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")

    release( : *handle: [ImmutableResourceHandle](pipeline.md#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle") | None = None*, ) → None
    :   Signal that data consumption is complete for the current stage.
        This allows producers to start producing new data.

cutlass.pipeline.make\_pipeline\_state( : *type: [PipelineUserType](pipeline.md#cutlass.pipeline.PipelineUserType "cutlass.pipeline.helpers.PipelineUserType")*, : *stages: int*, ) → [PipelineState](pipeline.md#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")
:   Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.

cutlass.pipeline.pipeline\_init\_arrive( : *cluster\_shape\_mn: cutlass.cute.typing.Layout | None = None*, : *is\_relaxed: bool = False*, ) → None
:   Fences the mbarrier\_init and sends an arrive if using clusters.

cutlass.pipeline.pipeline\_init\_wait( : *cluster\_shape\_mn: cutlass.cute.typing.Layout | None = None*, ) → None
:   Syncs the threadblock or cluster

cutlass.pipeline.agent\_sync( : *group: [Agent](pipeline.md#cutlass.pipeline.Agent "cutlass.pipeline.helpers.Agent")*, : *is\_relaxed: bool = False*, ) → None
:   Syncs all threads within an agent.

cutlass.pipeline.arrive(*barrier\_id: int*, *num\_threads: int*) → None
:   The aligned flavor of arrive is used when all threads in the CTA will execute the
    same instruction. See PTX documentation.

cutlass.pipeline.arrive\_unaligned(*barrier\_id: int*, *num\_threads: int*) → None
:   The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

cutlass.pipeline.wait() → None
:   NamedBarriers do not have a standalone wait like mbarriers, only an arrive\_and\_wait.
    If synchronizing two warps in a producer/consumer pairing, the arrive count would be
    32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
    or consumer are counted for mbarriers, while all threads participating in the sync
    are counted for NamedBarriers.

cutlass.pipeline.wait\_unaligned(*barrier\_id: int*, *num\_threads: int*) → None

cutlass.pipeline.arrive\_and\_wait(*barrier\_id: int*, *num\_threads: int*) → None

cutlass.pipeline.sync(*barrier\_id: int = 0*) → None
