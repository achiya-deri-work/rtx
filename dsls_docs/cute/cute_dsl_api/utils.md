# Utilities

The `cutlass.utils` package contains utilities for developing kernels with
CuTe DSL.

cutlass.utils.get\_smem\_capacity\_in\_bytes( : *compute\_capability: str | None = None*, ) → int
:   Get the shared memory capacity in bytes for a given compute capability.

    Returns the maximum shared memory capacity in bytes available for the specified
    GPU compute capability.

    Parameters:
    :   **compute\_capability** (*Optional**[**str**]*) – The compute capability string (e.g. “70”, “75”, “80”)

    Returns:
    :   The shared memory capacity in bytes

    Return type:
    :   int

    Raises:
    :   **ValueError** – If the compute capability is not supported

cutlass.utils.get\_kernel\_smem\_size(*kernel: Callable*) → int
:   Get the total static shared memory allocation in bytes for a kernel.

    Uses `cute.kernel_smem_size` to query the total smem bytes that will be
    allocated by a kernel. The result is lowered to a compile-time constant by
    `InferKernelSmemUsagePass`.

    Must be called from within a `@cute.jit` body after the kernel’s
    `.launch()` has been called, which triggers tracing and registers the
    kernel’s MLIR symbol.

    Parameters:
    :   **kernel** (*Callable*) – A `@cute.kernel`-decorated function. The MLIR symbol is
        retrieved automatically from state stored by the DSL after `.launch()`.

    Returns:
    :   Total shared memory allocated by the kernel, in bytes.

    Return type:
    :   int (i64 MLIR value during tracing)

*class* cutlass.utils.SmemAllocator
:   Bases: `object`

    A helper class for managing shared memory allocation on GPU.

    This class manages shared memory and provides APIs for allocation of raw bytes,
    numeric types, arrays, and tensors with specified layouts and alignments.

    Note

    - SmemAllocator will automatically calculate the usage upon kernel launch.
    - There is no need to explicitly specify shared memory size in kernel launch.
    - Currently only supports static layouts. Dynamic layouts are not supported.

    **Examples**:

    ```python
    smem = SmemAllocator()

    # Allocate raw bytes
    buf_ptr = smem.allocate(100)  # 100 bytes

    # Allocate numeric type
    int8_ptr = smem.allocate(Int8)  # 1 byte

    # Define a struct
    @cute.struct
    class SharedStorage:
        alpha: cutlass.Float32
        x: cutlass.Int32

    # Allocate struct
    struct_ptr = smem.allocate(SharedStorage)  # 8 bytes

    # use of struct members
    struct_ptr.alpha = 1.0
    struct_ptr.x = 2
    x_ptr = struct_ptr.x.ptr

    # Allocate array
    int8_array = smem.allocate_array(Int8, 10)  # 10 bytes

    # Allocate tensor
    layout = cute.make_layout((16, 16))
    tensor = smem.allocate_tensor(Int8, layout)  # 256 bytes
    ```

    *static* capacity\_in\_bytes( : *compute\_capability: str | None = None*, ) → int
    :   Get the shared memory capacity in bytes for a given compute capability.

        Returns the maximum shared memory capacity in bytes available for the specified
        GPU compute capability.

        Parameters:
        :   **compute\_capability** (*Optional**[**str**]*) – The compute capability string (e.g. “70”, “75”, “80”)

        Returns:
        :   The shared memory capacity in bytes

        Return type:
        :   int

        Raises:
        :   **ValueError** – If the compute capability is not supported

    \_\_init\_\_()
    :   Initialize a new SmemAllocator instance.

        Parameters:
        :   - **loc** (*Optional**[**ir.Location**]*) – Source location information for debugging, defaults to None
            - **ip** (*Optional**[**ir.InsertionPoint**]*) – Insertion point for MLIR operations, defaults to None

    calculate\_partition\_size( : *partition: SmemPartition*, : *\**, : *cumulative: bool = False*, ) → \_MockObject
    :   Get the size of shared memory allocation at given smem partition.

        Parameters:
        :   - **partition** (*SmemPartition*) – The smem partition to query
            - **cumulative** (*bool**,* *optional*) – Whether to return the cumulative size of all partitions up to and including the given partition
            - **loc** (*Optional**[**ir.Location**]*) – Source location information for debugging, defaults to None
            - **ip** (*Optional**[**ir.InsertionPoint**]*) – Insertion point for MLIR operations, defaults to None

    calculate\_total\_usage() → \_MockObject
    :   Get total kernel smem usage calculated by allocator.

        Parameters:
        :   - **loc** (*Optional**[**ir.Location**]*) – Source location information for debugging, defaults to None
            - **ip** (*Optional**[**ir.InsertionPoint**]*) – Insertion point for MLIR operations, defaults to None

    *property* \_allocated\_bytes*: \_MockObject*

    \_smem\_alloca( : *layout: cutlass.cute.typing.Layout*, : *dtype: \_MockObject*, : *byte\_alignment: int*, : *swizzle: [Swizzle](cute.md#cutlass.cute.Swizzle "cutlass.cute.core.Swizzle") | None = None*, : *struct\_fields: list[tuple[str, int, int]] | None = None*, : *\**, : *partition: SmemPartition = SmemPartition.USER*, ) → cutlass.cute.typing.Pointer
    :   Allocate shared memory using cute.memref.alloca with given layout, data type, and alignment.

        Returns:
        :   An iterator (pointer) to the allocated shared memory.

        Return type:
        :   cute.Pointer

    allocate( : *size\_or\_type: int*, : *byte\_alignment: int = 1*, : *\**, : *partition: SmemPartition = SmemPartition.USER*, ) → cutlass.cute.typing.Pointer

    allocate( : *size\_or\_type: Type[\_MockObject]*, : *byte\_alignment: int = 1*, : *\**, : *partition: SmemPartition = SmemPartition.USER*, ) → cutlass.cute.typing.Pointer

    allocate( : *size\_or\_type: [struct](cute.md#cutlass.cute.struct "cutlass.cute.core.struct")*, : *byte\_alignment: int = 1*, : *\**, : *partition: SmemPartition = SmemPartition.USER*, ) → cutlass.cute.typing.Pointer
    :   Allocate a block of memory with specified size and alignment.

        This method allocates a block of shared memory with the specified size and alignment requirements.
        It supports allocating raw bytes, numeric types(as scalar value), and struct types.

        Parameters:
        :   - **size\_or\_type** (*Union**[**int**,* *Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]**,* [*cute.struct*](cute.md#cutlass.cute.struct "cutlass.cute.struct")*]*) – The allocation specification, which can be:
              - An integer specifying the number of bytes to allocate
              - A Numeric type (e.g., Int8, Float32) to allocate space for one element
              - A struct type to allocate space for the entire struct
            - **byte\_alignment** (*int**,* *optional*) – The minimum byte alignment requirement for the allocation, defaults to 1
            - **loc** (*Optional**[**ir.Location**]*) – Source location information for debugging, defaults to None
            - **ip** (*Optional**[**ir.InsertionPoint**]*) – Insertion point for MLIR operations, defaults to None

        Returns:
        :   For raw bytes and numeric types, returns a pointer to the allocated memory.
            For struct types, returns an initialized struct instance at the allocated location.

        Return type:
        :   cute.Pointer

        Raises:
        :   - **ValueError** – If size is negative or alignment is less than 1
            - **TypeError** – If size\_or\_type is not an integer, Numeric type, or struct
            - **RuntimeError** – If allocation would exceed available shared memory

    allocate\_array( : *element\_type: Type[\_MockObject]*, : *num\_elems: int = 1*, : *\**, : *byte\_alignment: int = 1*, : *partition: SmemPartition = SmemPartition.USER*, ) → cutlass.cute.typing.Pointer
    :   Allocate an array of elements in shared memory.

        Parameters:
        :   - **element\_type** (*Union**[**Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]**]*) – The type of elements to allocate
            - **num\_elems** (*int**,* *optional*) – Number of elements to allocate, defaults to 1

        Returns:
        :   Pointer to the start of the allocated array

        Return type:
        :   cute.Pointer

        Raises:
        :   - **ValueError** – If num\_elems is less than 1
            - **TypeError** – If element\_type is not a Numeric type

    allocate\_tensor( : *element\_type: Type[\_MockObject]*, : *layout: int | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *byte\_alignment: int = 1*, : *swizzle: [Swizzle](cute.md#cutlass.cute.Swizzle "cutlass.cute.core.Swizzle") | None = None*, : *\**, : *partition: SmemPartition = SmemPartition.USER*, ) → cutlass.cute.typing.Tensor
    :   Allocate a tensor in shared memory.

        Note: Currently only supports static layouts. Dynamic layouts are not supported.

        Parameters:
        :   - **element\_type** (*Union**[**Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]**]*) – The type of elements in the tensor
            - **layout** (*Union**[**int**,* *cute.Layout**,* *cute.ComposedLayout**]*) – The layout specification for the tensor. Must be a static layout.
            - **byte\_alignment** (*int**,* *optional*) – The byte alignment requirement, defaults to 1
            - **swizzle** ([*cute.Swizzle*](cute.md#cutlass.cute.Swizzle "cutlass.cute.Swizzle")*,* *optional*) – Swizzle for position-dependent swizzling, defaults to None

        Returns:
        :   The allocated tensor with specified properties

        Return type:
        :   cute.Tensor

        Raises:
        :   - **TypeError** – If element\_type is not a Numeric type, or if swizzle conflicts with layout
            - **ValueError** – If allocation is not byte-aligned
            - **NotImplementedError** – If dynamic layout is specified

*class* cutlass.utils.TmemAllocator( : *alloc\_result\_dst\_smem\_ptr: cutlass.cute.typing.Pointer | None*, : *barrier\_for\_retrieve: [NamedBarrier](pipeline.md#cutlass.pipeline.NamedBarrier "cutlass.pipeline.helpers.NamedBarrier")*, : *allocator\_warp\_id: int = 0*, : *is\_two\_cta: bool = False*, : *num\_allocated\_columns: int = 0*, : *two\_cta\_tmem\_dealloc\_mbar\_ptr: cutlass.cute.typing.Pointer | None = None*, )
:   Bases: `object`

    A class for managing tensor memory allocation on GPUs.

    This class manages allocation/deallocation of tensor memory, including the mbarrier
    synchronization for two cta use case.

    Variables:
    :   - **\_alloc\_result\_dst\_smem\_ptr** – The smem pointer that holds the base address of allocated tensor memory.
        - **\_barrier\_for\_retrieve** – The barrier for retrieving tensor memory ptr.
        - **\_allocator\_warp\_id** – The warp id of the allocator warp.
        - **\_is\_two\_cta** – Whether the allocator is for two cta.
        - **\_num\_allocated\_columns** – The number of columns allocated in the tensor memory.
        - **\_two\_cta\_tmem\_dealloc\_mbar\_ptr** – The mbarrier pointer required when deallocating tensor memory for two cta.
        - **\_arch** – The architecture of the GPU.

    \_init\_dealloc\_mbarrier() → None

    \_\_init\_\_( : *alloc\_result\_dst\_smem\_ptr: cutlass.cute.typing.Pointer | None*, : *barrier\_for\_retrieve: [NamedBarrier](pipeline.md#cutlass.pipeline.NamedBarrier "cutlass.pipeline.helpers.NamedBarrier")*, : *allocator\_warp\_id: int = 0*, : *is\_two\_cta: bool = False*, : *num\_allocated\_columns: int = 0*, : *two\_cta\_tmem\_dealloc\_mbar\_ptr: cutlass.cute.typing.Pointer | None = None*, )
    :   Initialize a TmemAllocator instance for managing tensor memory on Blackwell GPUs.

        This initializer sets up the allocator’s state, including the shared memory (smem) pointer
        holding the base address of the allocated tensor memory, barrier synchronization for
        retrieving the tensor memory pointer, allocator warp ID, whether the allocator is being used
        for a 2-SM configuration, number of allocated columns in tensor
        memory, and the optional mbarrier pointer for deallocation in the 2-SM case.

        If is\_two\_cta is set to True, this will initialize the mbarrier pointer required for tensor
        memory deallocation across two CTAs.

        Auto-allocated smem pointers: If alloc\_result\_dst and two\_cta\_tmem\_dealloc ptrs are omitted,
        the allocator creates them to low address of the smem region,
        so they can be treated as reserved partition allocations and survive kernel smem resize if needed.

        Parameters:
        :   - **alloc\_result\_dst\_smem\_ptr** (*Optional**[**cute.Pointer**]*) – Shared memory pointer holding the base address of allocated tensor memory. If None, the allocator auto-allocates it in the reserved address.
            - **barrier\_for\_retrieve** ([*pipeline.NamedBarrier*](pipeline.md#cutlass.pipeline.NamedBarrier "cutlass.pipeline.NamedBarrier")) – The named barrier for retrieving the tensor memory pointer.
            - **allocator\_warp\_id** (*int**,* *optional*) – The warp ID of the allocator warp, defaults to 0.
            - **is\_two\_cta** (*bool**,* *optional*) – Whether the allocator should coordinate two CTAs, defaults to False.
            - **num\_allocated\_columns** (*int**,* *optional*) – The number of columns allocated in tensor memory, defaults to 0.
            - **two\_cta\_tmem\_dealloc\_mbar\_ptr** (*Optional**[**cute.Pointer**]*) – Mbarrier pointer for two-CTA tensor memory deallocation. If None and is\_two\_cta, the allocator auto-allocates it in the reserved address.
            - **initialize\_mbarrier** (*bool**,* *optional*) – Whether to initialize the mbarrier for two cta, defaults to True.
            - **loc** (*Any**,* *optional*) – Optional codegen location for debugging and error reporting.
            - **ip** (*Any**,* *optional*) – Optional insertion point for codegen.

        Raises:
        :   **ValueError** – If only provided one of required smem ptr in two cta.

    check\_valid\_num\_columns(*num\_columns: int*) → bool
    :   Check if the number of columns is valid.

        This method checks if the number of columns is valid.
        It checks if the number of columns is larger than 0, smaller than max capacity, a multiple of 32, and a power of two.

    allocate(*num\_columns: int*) → None
    :   Allocate a block of tensor memory.

        This method allocates a block of tensor memory from allocator warp and returns a handle to retrieve
        the allocated tensor memory address.

    wait\_for\_alloc() → None
    :   Wait for the allocator warp to finish allocation.

        This method is used to synchronize the allocator warp with the other warps before retrieving tmem ptr.

    retrieve\_ptr( : *dtype: ~typing.Type[~sphinx.ext.autodoc.mock.\_MockObject] = <class 'sphinx.ext.autodoc.mock.\_MockObject'>*, ) → cutlass.cute.typing.Pointer
    :   Retrieve the pointer to the allocated tensor memory.

        This method can be called by all warps after allocation has been performed
        by the allocator warp.

    reserve( : *num\_columns: int*, ) → [TmemBufferPool](utils.md#cutlass.utils.TmemBufferPool "cutlass.utils.tmem_allocator.TmemBufferPool")
    :   Reserve a block of tensor memory and return a pool for sub-allocation.

        This method allocates a block of tensor memory, waits for the allocation
        to complete, and returns a TmemBufferPool that can be used to sub-allocate
        regions within that block without manual offset calculations.

        Example usage:

        ```console
        tmem_pool = tmem_allocator.reserve(tmem_total_size)

        # Allocate and create tensors in one call
        tCtAcc = tmem_pool.allocate_tensor(tCtAcc_layout, cutlass.Float32)
        tCtSFA = tmem_pool.allocate_tensor(tCtSFA_layout, sf_dtype)

        # Or allocate pointer only, then create tensor manually
        sfb_ptr = tmem_pool.allocate(tCtSFB_layout, sf_dtype)
        tCtSFB = cute.make_tensor(sfb_ptr, tCtSFB_layout)
        ```

        Parameters:
        :   **num\_columns** (*int*) – The total number of columns to reserve.

        Returns:
        :   A TmemBufferPool for sub-allocating within the reserved region.

        Return type:
        :   [TmemBufferPool](utils.md#cutlass.utils.TmemBufferPool "cutlass.utils.TmemBufferPool")

    relinquish\_alloc\_permit() → None
    :   Relinquish the tensor memory allocation permit.

        This method relinquishes the tensor memory allocation permit for the allocator warp, promising
        the allocator warp will not allocate any more tensor memory.

    free( : *tmem\_ptr: cutlass.cute.typing.Pointer*, : *num\_columns: int = 0*, ) → None
    :   Deallocate the tensor memory.

        This method sync on mbarrier (for two cta use case) and deallocates the tensor memory from the allocator warp.
        User can optionally specify the number of columns to deallocate. If not specified, all allocated columns will be deallocated.

*class* cutlass.utils.TmemBufferPool(*base\_ptr: cutlass.cute.typing.Pointer*, *total\_cols: int*)
:   Bases: `object`

    A pool for sub-allocating from a reserved chunk of tensor memory.

    This class enables sub-allocation from a pre-reserved TMEM region,
    eliminating the need for manual offset calculations when allocating
    multiple tensors in TMEM.

    Example usage:

    ```console
    tmem_pool = tmem_allocator.reserve(tmem_total_size)

    # Allocate and create tensors in one call
    tCtAcc = tmem_pool.allocate_tensor(tCtAcc_layout, cutlass.Float32)
    tCtSFA = tmem_pool.allocate_tensor(tCtSFA_layout, sf_dtype)

    # Or allocate pointer only, then create tensor manually
    sfb_ptr = tmem_pool.allocate(tCtSFB_layout, sf_dtype)
    tCtSFB = cute.make_tensor(sfb_ptr, tCtSFB_layout)
    ```

    Variables:
    :   - **\_base\_ptr** – The base pointer to the reserved TMEM region.
        - **\_total\_cols** – The total number of columns in the pool.
        - **\_current\_offset** – The current offset within the pool (in columns).

    \_\_init\_\_( : *base\_ptr: cutlass.cute.typing.Pointer*, : *total\_cols: int*, )
    :   Initialize a TmemBufferPool instance.

        Parameters:
        :   - **base\_ptr** (*cute.Pointer*) – The base pointer to the reserved TMEM region.
            - **total\_cols** (*int*) – The total number of columns in the pool.

    *property* base\_ptr*: cutlass.cute.typing.Pointer*
    :   Return the base pointer of the pool.

    *property* total\_cols*: int*
    :   Return the total number of columns in the pool.

    *property* current\_offset*: int*
    :   Return the current offset within the pool.

    *property* remaining\_cols*: int*
    :   Return the number of remaining columns available for allocation.

    allocate( : *size: int | cutlass.cute.typing.Layout*, : *dtype: Type[\_MockObject]*, ) → cutlass.cute.typing.Pointer
    :   Allocate a sub-region from the pool and return a pointer.

        This method allocates a contiguous region of TMEM columns from the pool
        and returns a pointer to the start of that region.

        Parameters:
        :   - **size** (*Union**[**int**,* *cute.Layout**]*) – The allocation size, which can be:
              - int: explicit number of columns to allocate
              - cute.Layout: a TMEM layout that, combined with dtype, determines the size
            - **dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The data type for the returned pointer and for computing
              layout size (when size is a Layout).

        Returns:
        :   A pointer to the allocated region with the specified dtype.

        Return type:
        :   cute.Pointer

        Raises:
        :   **AssertionError** – If there are not enough columns remaining in the pool.

        Example usage:

        ```console
        # Allocate with explicit column count
        acc_ptr = pool.allocate(64, cutlass.Float32)

        # Allocate based on layout and dtype
        sfa_ptr = pool.allocate(tCtSFA_layout, sf_dtype)
        ```

    allocate\_tensor( : *layout: cutlass.cute.typing.Layout*, : *dtype: Type[\_MockObject]*, ) → cutlass.cute.typing.Tensor
    :   Allocate a sub-region from the pool and return a tensor.

        This is a convenience method that combines allocate() and cute.make\_tensor()
        into a single call.

        Parameters:
        :   - **layout** (*cute.Layout*) – The TMEM layout for the tensor.
            - **dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The data type for the tensor elements.

        Returns:
        :   A tensor backed by the allocated TMEM region.

        Return type:
        :   cute.Tensor

        Raises:
        :   **AssertionError** – If there are not enough columns remaining in the pool.

        Example usage:

        ```console
        tCtAcc = pool.allocate_tensor(tCtAcc_layout, cutlass.Float32)
        tCtSFA = pool.allocate_tensor(tCtSFA_layout, sf_dtype)
        ```

cutlass.utils.get\_num\_tmem\_alloc\_cols( : *tmem\_tensors: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor]*, : *rounding: bool = True*, : *\**, : *arch: str = 'sm\_100'*, ) → int
:   Get the total number of TMEM allocation columns for the given TMEM tensors.

    Parameters:
    :   - **tmem\_tensors** (*Union**[**cute.Tensor**,* *List**[**cute.Tensor**]**]*) – The TMEM tensors to get the number of allocation columns for.
        - **rounding** (*bool*) – Whether to round up the number of allocation columns to the nearest power of 2.
        - **arch** (*str*) – The architecture of the GPU.

    Returns:
    :   The total number of TMEM allocation columns.

    Return type:
    :   int

    Raises:
    :   **ValueError** – If the number of TMEM allocation columns exceeds the maximum capacity or is less than 32.

cutlass.utils.compute\_tmem\_cols\_from\_layout( : *layout: cutlass.cute.typing.Layout*, : *dtype: Type[\_MockObject]*, ) → int
:   Compute the number of TMEM columns required for a layout with a given dtype.

    This function calculates the column offset by recasting the layout to Int32
    and computing its cosize, similar to how find\_tmem\_tensor\_col\_offset works
    but without requiring a tensor.

    Parameters:
    :   - **layout** (*cute.Layout*) – The TMEM layout to compute columns for.
        - **dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The data type of the elements in the layout.

    Returns:
    :   The number of TMEM columns (always a Python int).

    Return type:
    :   int

    Raises:
    :   **ValueError** – If the layout size cannot be determined at compile time.

*class* cutlass.utils.LayoutEnum(*value*)
:   Bases: `Enum`

    An enumeration.

    ROW\_MAJOR *= 'row\_major'*

    COL\_MAJOR *= 'col\_major'*

    mma\_major\_mode() → [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode")

    sm90\_mma\_major\_mode() → [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode")

    is\_k\_major\_a() → bool

    is\_m\_major\_a() → bool

    is\_n\_major\_b() → bool

    is\_k\_major\_b() → bool

    is\_n\_major\_c() → bool

    is\_m\_major\_c() → bool

    *static* from\_tensor( : *tensor: cutlass.cute.typing.Tensor*, ) → [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")

*class* cutlass.utils.WorkTileInfo( : *tile\_idx: cutlass.cute.typing.Coord*, : *is\_valid\_tile: \_MockObject*, )
:   Bases: `object`

    A class to represent information about a work tile.

    Variables:
    :   - **tile\_idx** – The index of the tile.
        - **is\_valid\_tile** – Whether the tile is valid.

    \_\_init\_\_( : *tile\_idx: cutlass.cute.typing.Coord*, : *is\_valid\_tile: \_MockObject*, )

    *property* is\_valid\_tile*: \_MockObject*
    :   Check latest tile returned by the scheduler is valid or not. Any scheduling
        requests after all tasks completed will return an invalid tile.

        Returns:
        :   The validity of the tile.

        Return type:
        :   [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")

    *property* tile\_idx*: cutlass.cute.typing.Coord*
    :   Get the index of the tile.

        Returns:
        :   The index of the tile.

        Return type:
        :   cute.Coord

*class* cutlass.utils.PersistentTileSchedulerParams(*\*\*kwargs*)
:   Bases: `MixedClusterParamsMixin`

    A class to represent parameters for a persistent tile scheduler.

    This class is designed to manage and compute the layout of clusters and tiles
    in a batched gemm problem.

    Variables:
    :   - **cluster\_shape\_mn** – Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
        - **problem\_layout\_ncluster\_mnl** – Layout of the problem in terms of
          number of clusters in (m, n, l) dimensions.

    \_\_init\_\_() → None
    :   Initializes the PersistentTileSchedulerParams with the given parameters.

        Parameters:
        :   - **problem\_shape\_ntile\_mnl** (*cute.Shape*) – The shape of the problem in terms of
              number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
            - **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster in (m, n) dimensions.
            - **swizzle\_size** (*int*) – Swizzling size in the unit of cluster. 1 means no swizzle
            - **raster\_along\_m** (*bool*) – Rasterization order of clusters. Only used when swizzle\_size > 1.
              True means along M, false means along N.
            - **fallback\_cluster\_shape\_mnk** (*Optional**[**cute.Shape**]*) – Optional. When provided and
              different from cluster\_shape\_mnk, the kernel runs in mixed-cluster mode.

        Raises:
        :   **ValueError** – If cluster\_shape\_k is not 1.

    \_extract\_primary\_mlir\_values() → list[ir.Value]

    *property* \_primary\_values\_count*: int*

    \_new\_primary\_from\_mlir\_values( : *values: list[ir.Value]*, ) → [PersistentTileSchedulerParams](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")

    get\_grid\_shape( : *max\_active\_clusters: \_MockObject*, ) → Tuple[\_MockObject, \_MockObject, \_MockObject]
    :   Computes the grid shape based on the maximum active clusters allowed.

        Parameters:
        :   **max\_active\_clusters** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The maximum number of active clusters that
            can run in one wave.

        Returns:
        :   A tuple containing the grid shape in (m, n, persistent\_clusters).
            - m: self.cluster\_shape\_m.
            - n: self.cluster\_shape\_n.
            - persistent\_clusters: Number of persistent clusters that can run.

*class* cutlass.utils.StaticPersistentTileScheduler( : *params: [PersistentTileSchedulerParams](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *num\_persistent\_clusters: \_MockObject*, : *current\_work\_linear\_idx: \_MockObject*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: \_MockObject*, )
:   Bases: `object`

    A scheduler for static persistent tile execution in CUTLASS/CuTe kernels.

    Variables:
    :   - **params** – Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl
        - **num\_persistent\_clusters** – Number of persistent clusters that can be launched
        - **cta\_id\_in\_cluster** – ID of the CTA within its cluster
        - **\_num\_tiles\_executed** – Counter for executed tiles
        - **\_current\_work\_linear\_idx** – Current cluster index

    \_\_init\_\_( : *params: [PersistentTileSchedulerParams](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *num\_persistent\_clusters: \_MockObject*, : *current\_work\_linear\_idx: \_MockObject*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: \_MockObject*, )
    :   Initializes the StaticPersistentTileScheduler with the given parameters.

        Parameters:
        :   - **params** ([*PersistentTileSchedulerParams*](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl.
            - **num\_persistent\_clusters** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Number of persistent clusters that can be launched.
            - **current\_work\_linear\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Current cluster index.
            - **cta\_id\_in\_cluster** (*cute.Coord*) – ID of the CTA within its cluster.
            - **num\_tiles\_executed** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Counter for executed tiles.

    *static* create( : *params: [PersistentTileSchedulerParams](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *block\_idx: Tuple[\_MockObject, \_MockObject, \_MockObject]*, : *grid\_dim: Tuple[\_MockObject, \_MockObject, \_MockObject]*, ) → [StaticPersistentTileScheduler](utils.md#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.static_persistent_tile_scheduler.StaticPersistentTileScheduler")
    :   Initialize the static persistent tile scheduler.

        Parameters:
        :   - **params** ([*PersistentTileSchedulerParams*](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Parameters for the persistent
              tile scheduler.
            - **block\_idx** (*Tuple**[*[*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*]*) – The 3d block index in the format (bidx, bidy, bidz).
            - **grid\_dim** (*Tuple**[*[*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*]*) – The 3d grid dimensions for kernel launch.

        Returns:
        :   A StaticPersistentTileScheduler object.

        Return type:
        :   [StaticPersistentTileScheduler](utils.md#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.StaticPersistentTileScheduler")

    *static* get\_grid\_shape( : *params: [PersistentTileSchedulerParams](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *max\_active\_clusters: \_MockObject*, ) → Tuple[\_MockObject, \_MockObject, \_MockObject]
    :   Calculates the grid shape to be launched on GPU using problem shape,
        threadblock shape, and active cluster size.

        Parameters:
        :   - **params** ([*PersistentTileSchedulerParams*](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Parameters for grid shape calculation.
            - **max\_active\_clusters** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Maximum active clusters allowed.

        Returns:
        :   The calculated 3d grid shape.

        Return type:
        :   Tuple[[Integer](../basic_data_types.md#cutlass.Integer "cutlass.Integer"), [Integer](../basic_data_types.md#cutlass.Integer "cutlass.Integer"), [Integer](../basic_data_types.md#cutlass.Integer "cutlass.Integer")]

    \_get\_current\_work\_for\_linear\_idx( : *current\_work\_linear\_idx: \_MockObject*, ) → [WorkTileInfo](utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")
    :   Compute current tile coord given current\_work\_linear\_idx and cta\_id\_in\_cluster.

        Parameters:
        :   **current\_work\_linear\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The linear index of the current work.

        Returns:
        :   An object containing information about the current tile coordinates
            and validity status.

        Return type:
        :   [WorkTileInfo](utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.WorkTileInfo")

    \_get\_cluster\_work\_idx\_with\_fastdivmod( : *current\_work\_linear\_idx: \_MockObject*, ) → Tuple[\_MockObject, \_MockObject, \_MockObject]
    :   FastDivmod optimized CLUSTER coordinate calculation.

        CRITICAL: This should mimic problem\_layout\_ncluster\_mnl.get\_hier\_coord()
        which returns CLUSTER coordinates, not tile coordinates!

        Parameters:
        :   **current\_work\_linear\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Linear index in the work space

        Returns:
        :   Cluster coordinates (m, n, l) or None if FastDivmod not available

        Return type:
        :   Tuple[[Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32"), [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32"), [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")] or None

    get\_current\_work() → [WorkTileInfo](utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")

    initial\_work\_tile\_info() → [WorkTileInfo](utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")

    advance\_to\_next\_work( : *\**, : *advance\_count: int = 1*, ) → None

    *property* num\_tiles\_executed*: \_MockObject*

*class* cutlass.utils.StaticPersistentRuntimeTileScheduler(*\*\*kwargs*)
:   Bases: [`StaticPersistentTileScheduler`](utils.md#cutlass.utils.StaticPersistentTileScheduler "cutlass.utils.static_persistent_tile_scheduler.StaticPersistentTileScheduler")

    A scheduler for static persistent runtime tile execution in CUTLASS/CuTe kernels.
    This scheduler will always launch all the SMs and the scheduler will generate the real tile info for each SM.

    Variables:
    :   - **params** – Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl
        - **num\_persistent\_clusters** – Number of persistent clusters that can be launched
        - **cta\_id\_in\_cluster** – ID of the CTA within its cluster
        - **\_num\_tiles\_executed** – Counter for executed tiles
        - **\_current\_work\_linear\_idx** – Current cluster index

    \_\_init\_\_( : *params: [PersistentTileSchedulerParams](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *num\_persistent\_clusters: \_MockObject*, : *current\_work\_linear\_idx: \_MockObject*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: \_MockObject*, : *inner\_mode: int = 1*, )
    :   Initializes the StaticPersistentTileScheduler with the given parameters.

        Parameters:
        :   - **params** ([*PersistentTileSchedulerParams*](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Tile schedule related params, including cluster shape and problem\_layout\_ncluster\_mnl.
            - **num\_persistent\_clusters** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Number of persistent clusters that can be launched.
            - **current\_work\_linear\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Current cluster index.
            - **cta\_id\_in\_cluster** (*cute.Coord*) – ID of the CTA within its cluster.
            - **num\_tiles\_executed** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Counter for executed tiles.
            - **inner\_mode** (*int*) – The inner mode along which the linear index will be decomposed first.

    *static* create( : *params: [PersistentTileSchedulerParams](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *block\_idx: Tuple[\_MockObject, \_MockObject, \_MockObject]*, : *grid\_dim: Tuple[\_MockObject, \_MockObject, \_MockObject]*, : *inner\_mode: int = 1*, ) → [StaticPersistentRuntimeTileScheduler](utils.md#cutlass.utils.StaticPersistentRuntimeTileScheduler "cutlass.utils.static_persistent_tile_scheduler.StaticPersistentRuntimeTileScheduler")
    :   Initialize the static persistent tile scheduler.

        Parameters:
        :   - **params** ([*PersistentTileSchedulerParams*](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Parameters for the persistent
              tile scheduler.
            - **block\_idx** (*Tuple**[*[*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*]*) – The 3d block index in the format (bidx, bidy, bidz).
            - **grid\_dim** (*Tuple**[*[*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*]*) – The 3d grid dimensions for kernel launch.
            - **inner\_mode** (*int*) – The inner mode along which the linear index will be decomposed first.

        Returns:
        :   A StaticPersistentRuntimeTileScheduler object.

        Return type:
        :   [StaticPersistentRuntimeTileScheduler](utils.md#cutlass.utils.StaticPersistentRuntimeTileScheduler "cutlass.utils.StaticPersistentRuntimeTileScheduler")

    \_get\_current\_work\_for\_linear\_idx( : *current\_work\_linear\_idx: \_MockObject*, ) → [WorkTileInfo](utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")
    :   Compute current tile coord given current\_work\_linear\_idx and cta\_id\_in\_cluster.

        Parameters:
        :   **current\_work\_linear\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The linear index of the current work.

        Returns:
        :   An object containing information about the current tile coordinates
            and validity status.

        Return type:
        :   [WorkTileInfo](utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.WorkTileInfo")

*class* cutlass.utils.TensorMapUpdateMode(*value*)
:   Bases: `Enum`

    Enum class defining tensor map update modes.

    Modes:
    GMEM: Update tensormap in global memory
    SMEM: Load tensormap from global memory to shared memory,
    update it in shared memory, then store back to global memory

    GMEM *= 1*

    SMEM *= 2*

*class* cutlass.utils.TensorMapManager( : *tensormap\_update\_mode: [TensorMapUpdateMode](utils.md#cutlass.utils.TensorMapUpdateMode "cutlass.utils.tensormap_manager.TensorMapUpdateMode")*, : *bytes\_per\_tensormap: int*, )
:   Bases: `object`

    Manages TensorMap operations including initialization and updates.
    Provides utilities to convert tensormap pointer to across different memory spaces.

    tensormap\_update\_mode*: [TensorMapUpdateMode](utils.md#cutlass.utils.TensorMapUpdateMode "cutlass.utils.tensormap_manager.TensorMapUpdateMode")*

    bytes\_per\_tensormap*: int*

    get\_tensormap\_ptr( : *ptr: cutlass.cute.typing.Pointer*, : *address\_space: [AddressSpace](cute.md#cutlass.cute.AddressSpace "cutlass.base_dsl.enums.AddressSpace") = AddressSpace.gmem*, ) → cutlass.cute.typing.Pointer

    init\_tensormap\_from\_atom( : *copy\_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *dst\_ptr: cutlass.cute.typing.Pointer*, : *warp\_id: int*, ) → None

    fence\_tensormap\_initialization() → None

    fence\_tensormap\_update( : *tensormap\_ptr: cutlass.cute.typing.Pointer*, ) → None

    update\_tensormap( : *tensor\_gmem: Tuple[cutlass.cute.typing.Tensor, ...]*, : *tma\_copy\_atom: Tuple[[CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), ...]*, : *tensormap\_gmem\_ptr: Tuple[cutlass.cute.typing.Pointer, ...]*, : *warp\_id: int*, : *tensormap\_smem\_ptr: Tuple[cutlass.cute.typing.Pointer, ...]*, ) → None

    \_\_init\_\_( : *tensormap\_update\_mode: [TensorMapUpdateMode](utils.md#cutlass.utils.TensorMapUpdateMode "cutlass.utils.tensormap_manager.TensorMapUpdateMode")*, : *bytes\_per\_tensormap: int*, ) → None

*class* cutlass.utils.GroupSearchResult(*\*\*kwargs*)
:   Bases: `object`

    The result of the group search for grouped gemm.

    Parameters:
    :   - **group\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The result group index
        - **cta\_tile\_idx\_m** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – CTA tile index along M dimension after rasterization
        - **cta\_tile\_idx\_n** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – CTA tile index along N dimension after rasterization
        - **problem\_shape\_m** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The M dimension of the gemm problem
        - **problem\_shape\_n** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The N dimension of the gemm problem
        - **problem\_shape\_k** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The K dimension of the gemm problem
        - **cta\_tile\_count\_k** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Number of tiles along K dimension

    \_\_init\_\_( : *group\_idx: \_MockObject*, : *cta\_tile\_idx\_m: \_MockObject*, : *cta\_tile\_idx\_n: \_MockObject*, : *problem\_shape\_m: \_MockObject*, : *problem\_shape\_n: \_MockObject*, : *problem\_shape\_k: \_MockObject*, : *cta\_tile\_count\_k: \_MockObject*, ) → None

*class* cutlass.utils.GroupedGemmGroupSearchState(*\*\*kwargs*)
:   Bases: `object`

    The state of group index search for grouped gemm.

    The state will be initialized once and updated in every round of group index search.

    Parameters:
    :   - **start\_group\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The group idx to start the search with
        - **tile\_count\_prev\_group** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Number of tiles before the matched group
        - **tile\_count\_searched** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Number of tiles we have searched. When the matched group
          is found, it records the number of tiles including the
          matched group

    \_\_init\_\_( : *start\_group\_idx: \_MockObject*, : *tile\_count\_prev\_group: \_MockObject*, : *tile\_count\_searched: \_MockObject*, : *found: \_MockObject*, ) → None

cutlass.utils.create\_initial\_search\_state() → [GroupedGemmGroupSearchState](utils.md#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.grouped_gemm_persistent_tile_scheduler.GroupedGemmGroupSearchState")
:   Create an initial search state for grouped gemm.

    Returns:
    :   A new search state with initial values

    Return type:
    :   [GroupedGemmGroupSearchState](utils.md#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.GroupedGemmGroupSearchState")

*class* cutlass.utils.GroupedGemmTileSchedulerHelper(*\*\*kwargs*)
:   Bases: `object`

    A helper to translate the raw block index (x, y, z) from tile scheduler to real CTA
    tile index for grouped gemm.

    Parameters:
    :   - **group\_count** (*int*) – Number of groups in current grouped gemm problem
        - **tile\_sched\_params** ([*PersistentTileSchedulerParams*](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.PersistentTileSchedulerParams")) – Parameter used to create the tile scheduler this helper
          works with
        - **cluster\_tile\_shape\_mnk** (*tuple**[**int**,* *int**,* *int**]*) – The shape of cluster tile as (m, n, k)
        - **search\_state** ([*GroupedGemmGroupSearchState*](utils.md#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.GroupedGemmGroupSearchState")) – The initial search state

    \_\_init\_\_( : *group\_count: int*, : *tile\_sched\_params: [PersistentTileSchedulerParams](utils.md#cutlass.utils.PersistentTileSchedulerParams "cutlass.utils.static_persistent_tile_scheduler.PersistentTileSchedulerParams")*, : *cluster\_tile\_shape\_mnk: tuple[int, int, int]*, : *search\_state: [GroupedGemmGroupSearchState](utils.md#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.grouped_gemm_persistent_tile_scheduler.GroupedGemmGroupSearchState")*, ) → None

    delinearize\_z( : *cta\_tile\_coord: tuple*, : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, ) → [GroupSearchResult](utils.md#cutlass.utils.GroupSearchResult "cutlass.utils.grouped_gemm_persistent_tile_scheduler.GroupSearchResult")
    :   Delinearize the linear z index and return GroupSearchResult.

        This function should be used by warps that need to know the CTA tile index on M
        and N dimensions.

        Parameters:
        :   - **cta\_tile\_coord** (*tuple* *of* [*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The raw CTA coordinate from tile scheduler
            - **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor containing gemm problem size (M, N, K, L) for
              each group

        Returns:
        :   The search result containing group index and tile coordinates

        Return type:
        :   [GroupSearchResult](utils.md#cutlass.utils.GroupSearchResult "cutlass.utils.GroupSearchResult")

    search\_cluster\_tile\_count\_k( : *cta\_tile\_coord: tuple*, : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, ) → Tuple[\_MockObject, \_MockObject]
    :   Search the matched group for given linear index and compute the number of tiles
        along K dimension for the matched group.

        This function should be used by warps that are only interested in the number of
        tiles along K dimension.

        Parameters:
        :   - **cta\_tile\_coord** (*tuple* *of* [*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The raw CTA coordinate from tile scheduler
            - **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor containing gemm problem size (M, N, K, L) for
              all groups

        Returns:
        :   A tuple containing cluster count along K dimension and the group index

        Return type:
        :   Tuple[[Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32"), [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")]

    \_prefix\_sum( : *value\_per\_thread: \_MockObject*, ) → \_MockObject
    :   Perform prefix sum within a full warp.

        Parameters:
        :   **value\_per\_thread** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The value for this thread to contribute to the prefix
            sum

        Returns:
        :   The prefix sum result for this thread

        Return type:
        :   [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")

    \_get\_problem\_for\_group( : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, : *group\_idx: \_MockObject*, ) → cutlass.cute.typing.Tensor
    :   Load gemm problem (m,n,k,l) for the specified group from global memory to
        register.

        Parameters:
        :   - **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor in global memory with layout
              (group\_count, 4):(4, 1)
            - **group\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The index of the group to load

        Returns:
        :   The problem shape tensor for the specified group

        Return type:
        :   cute.Tensor

    \_get\_cluster\_tile\_count\_mn( : *problem\_shape: cutlass.cute.typing.Tensor*, ) → \_MockObject
    :   Compute total cluster count.

        Parameters:
        :   **problem\_shape** (*cute.Tensor*) – Tensor containing problem shape (m, n, k, l)

        Returns:
        :   The total cluster tile count for M and N dimensions

        Return type:
        :   [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")

    \_compute\_cta\_tile\_coord( : *cluster\_tile\_idx: \_MockObject*, : *cta\_tile\_coord\_in\_cluster: tuple*, : *cluster\_tile\_count\_m: \_MockObject*, : *cluster\_tile\_count\_n: \_MockObject*, ) → tuple
    :   Compute CTA tile indices along M and N dimensions based on the linear index
        within a group.

        It uses the AlongM mode to decompose the linear index onto M and N dimensions.

        Parameters:
        :   - **cluster\_tile\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The linear index within a group
            - **cta\_tile\_coord\_in\_cluster** (*tuple* *of* [*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – CTA indices along M and N dimensions within a
              cluster
            - **cluster\_tile\_count\_m** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The number of clusters along M dimension of the
              matched group
            - **cluster\_tile\_count\_n** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The number of clusters along N dimension of the
              matched group

        Returns:
        :   A tuple containing CTA tile indices along M and N dimensions

        Return type:
        :   tuple of ([Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32"), [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32"))

    \_group\_search( : *linear\_idx: \_MockObject*, : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, : *init\_group\_idx: \_MockObject*, : *init\_tile\_count\_searched: \_MockObject*, ) → [GroupedGemmGroupSearchState](utils.md#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.grouped_gemm_persistent_tile_scheduler.GroupedGemmGroupSearchState")
    :   Search which group the linear index belongs to.

        Parameters:
        :   - **linear\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The linear index to be decomposed
            - **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor containing gemm problem size (M, N, K, L) for
              all groups
            - **init\_group\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The group idx to start the search with
            - **init\_tile\_count\_searched** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The number of tiles we have searched

        Returns:
        :   The updated search state

        Return type:
        :   [GroupedGemmGroupSearchState](utils.md#cutlass.utils.GroupedGemmGroupSearchState "cutlass.utils.GroupedGemmGroupSearchState")

    \_group\_search\_and\_load\_problem\_shape( : *linear\_idx: \_MockObject*, : *problem\_shape\_mnkl: cutlass.cute.typing.Tensor*, : *start\_group\_idx: \_MockObject*, : *tile\_count\_searched: \_MockObject*, ) → Tuple[\_MockObject, cutlass.cute.typing.Tensor]
    :   Perform group search and load problem shape for the matched group.

        Parameters:
        :   - **linear\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The linear index to be decomposed
            - **problem\_shape\_mnkl** (*cute.Tensor*) – Tensor containing gemm problem size (M, N, K, L) for
              all groups
            - **start\_group\_idx** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The group idx to start the search with
            - **tile\_count\_searched** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – The number of tiles we have searched

        Returns:
        :   A tuple containing the final group index and the problem shape tensor

        Return type:
        :   Tuple[[Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32"), cute.Tensor]

*class* cutlass.utils.HardwareInfo(*device\_id: int = 0*)
:   Bases: `object`

    device\_id: CUDA device ID to get the hardware info.

    \_\_init\_\_(*device\_id: int = 0*)

    get\_max\_active\_clusters( : *cluster\_size: int*, : *stream: cuda.bindings.driver.CUstream | None = None*, ) → int
    :   Get the maximum number of active clusters for a given cluster size.

        When a stream from a green context is provided, the occupancy calculation
        will reflect the reduced SM partition of the green context.

        Parameters:
        :   - **cluster\_size** (*int*) – Number of blocks per cluster (must be between 1 and 32)
            - **stream** (*driver.CUstream**,* *optional*) – Optional CUDA stream handle. If provided (especially from a green context),
              the occupancy calculation reflects the stream’s SM partition.

        Returns:
        :   Maximum number of active clusters

        Return type:
        :   int

    get\_l2\_cache\_size\_in\_bytes() → int

    get\_device\_multiprocessor\_count() → int

    \_checkCudaErrors(*result: Any*) → Any

    \_cudaGetErrorEnum(*error: Any*) → str

    \_cuda\_driver\_version\_ge(*major: int*, *minor: int*) → bool

    \_cuda\_driver\_version\_lt(*major: int*, *minor: int*) → bool

    \_empty\_kernel() → None

    \_host\_function() → None

    \_get\_device\_function() → cuda.bindings.driver.CUfunction
    :   Get a device function by compiling a dummy kernel using cuteDSL pipeline.

*class* cutlass.utils.TransformMode(*value*)
:   Bases: `Enum`

    An enumeration for the possible transform modes of a mixed-input GEMM.

    ConvertOnly *= 1*

    ConvertScale *= 2*

cutlass.utils.scale\_tma\_partition( : *tCsS: cutlass.cute.typing.Tensor*, : *tCgS: cutlass.cute.typing.Tensor*, : *tma\_atom\_s: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *block\_in\_cluster\_coord\_vmnk: cutlass.cute.typing.Coord*, : *scale\_cta\_layout: cutlass.cute.typing.Layout*, ) → tuple[cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor]
:   Perform TMA partition for scale tensor.
    This method partitions the global memory and shared memory buffer for the scale tensor for TMA load.
    :param tCsS: Input scale shared memory tensor
    :type tCsS: cute.Tensor
    :param tCgS: Input scale global memory tensor
    :type tCgS: cute.Tensor
    :param tma\_atom\_s: TMA copy atom for scale tensor
    :type tma\_atom\_s: cute.CopyAtom
    :param block\_in\_cluster\_coord\_vmnk: CTA coord in the cluster
    :type block\_in\_cluster\_coord\_vmnk: cute.Coord
    :param scale\_cta\_layout: Layout of CTA from the view of the scale tensor
    :type scale\_cta\_layout: cute.Layout
    :return: A tuple containing (tSsS, tSgS) where:

    > - tSsS: Partitioned scale tensor in shared memory
    > - tSgS: Partitioned scale tensor in global memory

    Return type:
    :   tuple[cute.Tensor, cute.Tensor]

cutlass.utils.transform\_partition( : *transform\_a\_source: [OperandSource](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *scale\_mode: [TransformMode](utils.md#cutlass.utils.TransformMode "cutlass.utils.mixed_input_helpers.TransformMode")*, : *copy\_atom\_a\_input: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *copy\_atom\_a\_transform: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *sA\_input: cutlass.cute.typing.Tensor*, : *A\_transform: cutlass.cute.typing.Tensor*, : *transform\_local\_tidx: [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, ) → tuple[[TiledCopy](cute.md#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy") | None, [TiledCopy](cute.md#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy") | None, cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor]
:   Partition tensors for transform input and output.
    This method sets up the copy atoms and partitions the shared/tensor memory
    for the transformation of tensor A.
    :param transform\_a\_source: Where the transformed tensor A is stored (TMEM or SMEM)
    :type transform\_a\_source: tcgen05.OperandSource
    :param scale\_mode: The transform mode (ConvertOnly or ConvertScale)
    :type scale\_mode: TransformMode
    :param copy\_atom\_a\_input: Copy atom for loading A from shared memory
    :type copy\_atom\_a\_input: cute.CopyAtom
    :param copy\_atom\_a\_transform: Copy atom for storing transformed A
    :type copy\_atom\_a\_transform: cute.CopyAtom
    :param sA\_input: Input tensor A in shared memory
    :type sA\_input: cute.Tensor
    :param A\_transform: Transformed tensor A in tensor or shared memory
    :type A\_transform: cute.Tensor
    :param transform\_local\_tidx: Local thread index for transformation warps
    :type transform\_local\_tidx: cutlass.Int32
    :return: A tuple containing (src\_copy\_a, dst\_copy\_a, tAsA\_input, tA\_transform) where:

    > - src\_copy\_a: Tiled copy for source tensor
    > - dst\_copy\_a: Tiled copy for destination tensor
    > - tAsA\_input: Partitioned input tensor A
    > - tA\_transform: Partitioned transformed tensor A

    Return type:
    :   tuple[Optional[[cute.TiledCopy](cute.md#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")], Optional[[cute.TiledCopy](cute.md#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")], cute.Tensor, cute.Tensor]

cutlass.utils.scale\_partition( : *src\_copy\_a: [TiledCopy](cute.md#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")*, : *tCsS: cutlass.cute.typing.Tensor*, : *transform\_local\_tidx: [Int32](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, : *mma\_dtype: type[[Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, ) → tuple[[TiledCopy](cute.md#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy"), cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor]
:   Partition the scale tensor for transformation.
    This method prepares the copy atom and partitions the shared memory for the scale tensor.
    :param src\_copy\_a: Tiled copy for the source tensor
    :type src\_copy\_a: cute.TiledCopy
    :param tCsS: Scale tensor in shared memory
    :type tCsS: cute.Tensor
    :param transform\_local\_tidx: Local thread index for transformation warps
    :type transform\_local\_tidx: cutlass.Int32
    :param mma\_dtype: Data type for the MMA operation
    :type mma\_dtype: type[cutlass.Numeric]
    :return: A tuple containing (smem\_thr\_copy\_S, tSsS\_trans, tSrS\_copy, tSrS) where:

    > - smem\_thr\_copy\_S: Tiled copy for the scale tensor
    > - tSsS\_trans: Partitioned scale tensor for transformation
    > - tSrS\_copy: Register fragment for the scale tensor
    > - tSrS: View of scale tensor used for transformation computation

    Return type:
    :   tuple[[cute.TiledCopy](cute.md#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy"), cute.Tensor, cute.Tensor, cute.Tensor]

cutlass.utils.get\_gmem\_layout\_scale( : *scale\_shape\_mkl: tuple[int, int, int]*, : *scale\_granularity\_m: int*, : *scale\_granularity\_k: int*, : *scale\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode")*, ) → cutlass.cute.typing.Layout
:   Get the layout of the scale tensor in global memory.
    :param scale\_shape\_mkl: The shape of the scale tensor (M, K, L).
    :type scale\_shape\_mkl: tuple[int, int, int]
    :return: The layout of the scale tensor in global memory.
    :rtype: cute.Layout

cutlass.utils.get\_smem\_layout\_scale( : *mma\_tiler: tuple[int, int, int]*, : *use\_2cta\_instrs: bool*, : *scale\_granularity\_m: int*, : *scale\_granularity\_k: int*, : *scale\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode")*, : *a\_scale\_dtype: type[[Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, : *num\_scale\_load2trans\_stage: int*, ) → tuple[tuple[int, int], cutlass.cute.typing.ComposedLayout, cutlass.cute.typing.ComposedLayout]
:   Get the layout of the scale tensor in shared memory.
    :return: A tuple containing (scale\_tile\_shape, smem\_layout\_scale\_per\_stage, smem\_layout\_scale) where:

    > - scale\_tile\_shape: The tile shape
    > - smem\_layout\_scale\_per\_stage: Shared memory layout for scale tensor per stage
    > - smem\_layout\_scale: Shared memory layout for scale tensor

    Return type:
    :   tuple[tuple[int, int], cute.ComposedLayout, cute.ComposedLayout]

cutlass.utils.compute\_smem\_layout( : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: tuple[int, int, int]*, : *a\_dtype: type[[Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, : *b\_dtype: type[[Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, : *load2trans\_stage\_count: int*, : *trans2mma\_stage\_count: int*, ) → tuple[cutlass.cute.typing.ComposedLayout, cutlass.cute.typing.ComposedLayout, cutlass.cute.typing.ComposedLayout]
:   Compute shared memory layouts for tensor A, transformed A and tensor B.
    :param tiled\_mma: The tiled MMA object defining the core computation.
    :type tiled\_mma: cute.TiledMma
    :param mma\_tiler\_mnk: The shape (M, N, K) of the MMA tiler.
    :type mma\_tiler\_mnk: tuple[int, int, int]
    :param a\_dtype: Data type of operand A.
    :type a\_dtype: type[cutlass.Numeric]
    :param b\_dtype: Data type of operand B.
    :type b\_dtype: type[cutlass.Numeric]
    :param load2trans\_stage\_count: Number of stages for load-to-transform pipeline.
    :type load2trans\_stage\_count: int
    :param trans2mma\_stage\_count: Number of stages for transform-to-MMA pipeline.
    :type trans2mma\_stage\_count: int
    :return: A tuple containing (smem\_layout\_a, smem\_layout\_a\_transform, smem\_layout\_b) where:

    > - smem\_layout\_a: Shared memory layout for tensor A
    > - smem\_layout\_a\_transform: Shared memory layout for transformed tensor A
    > - smem\_layout\_b: Shared memory layout for tensor B

    Return type:
    :   tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]

cutlass.utils.get\_transform\_a\_source( : *a\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode")*, ) → [OperandSource](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")
:   Determine the operand source for transformed A tensor based on the operand major mode.

cutlass.utils.get\_tma\_atom\_kind( : *mcast: [Boolean](../basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*, : *use\_2cta\_instrs: bool*, : *is\_b: bool*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
:   Get the TMA atom kind based on 1) whether it’s a multicast operation,
    2) whether 2CTA tcgen05.mma instruction is enabled, and
    3) whether it’s a B tensor

cutlass.utils.get\_copy\_atom\_a\_transform( : *mma\_dtype: type[[Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, : *use\_2cta\_instrs: bool*, : *transform\_a\_source: [OperandSource](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_smem\_shape: cutlass.cute.typing.Shape*, : *a\_dtype: type[[Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, ) → [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")
:   Determine the copy atom for transformed A tensor based on the operand source and tile size.

cutlass.utils.is\_valid\_scale\_granularity( : *scale\_granularity\_m: int*, : *scale\_granularity\_k: int*, : *a\_dtype: type[[Numeric](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, : *k: int*, : *mma\_tiler\_k: int*, ) → bool
:   Check if the scale granularity settings are valid for the given data type and problem size.

cutlass.utils.get\_divisibility( : *contiguous\_dim\_size: int*, : *upper\_bound: int = 128*, ) → int
:   Calculate the largest power of 2 divisibility factor for memory alignment.

cutlass.utils.cluster\_shape\_to\_tma\_atom\_A( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
:   Select the appropriate TMA copy atom for A based on the number of SMs and the multicast flag.

    Parameters:
    :   - **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster
        - **atom\_thr\_id** (*cute.Layout*) – The thread ID of the atom

    Returns:
    :   The appropriate TMA copy atom kind

    Return type:
    :   [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

    Raises:
    :   - **ValueError** – If the atom\_sm\_cnt is invalid
        - **ValueError** – If the cluster shape is not divisible by the atom SM count

cutlass.utils.cluster\_shape\_to\_tma\_atom\_B( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
:   Select the appropriate TMA copy atom for Bbased on the number of SMs and the multicast flag.

    Parameters:
    :   - **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster
        - **atom\_thr\_id** (*cute.Layout*) – The thread ID of the atom

    Returns:
    :   The appropriate TMA copy atom kind

    Return type:
    :   [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

    Raises:
    :   - **ValueError** – If the atom\_sm\_cnt is invalid
        - **ValueError** – If the cluster shape is not divisible by the atom SM count

cutlass.utils.cluster\_shape\_to\_tma\_atom\_SFB( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
:   Select the appropriate TMA copy atom for SFB based on the number of SMs and the multicast flag.

    Parameters:
    :   - **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster
        - **atom\_thr\_id** (*cute.Layout*) – The thread ID of the atom

    Returns:
    :   The appropriate TMA copy atom kind

    Return type:
    :   [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

    Raises:
    :   - **ValueError** – If the atom\_sm\_cnt is invalid
        - **ValueError** – If the cluster shape is not divisible by the atom SM count

cutlass.utils.compute\_epilogue\_tile\_shape( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *use\_2cta\_instrs: bool*, : *layout\_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[\_MockObject]*, : *\**, : *layout\_c: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum") | None = None*, : *elem\_ty\_c: Type[\_MockObject] | None = None*, : *tmem\_warp\_shape\_mn: Tuple[int, int] | None = None*, ) → cutlass.cute.typing.Tile
:   Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

    Parameters:
    :   - **cta\_tile\_shape** (*cute.Shape*) – A tuple or list representing the dimensions of the CTA tile, where
          cta\_tile\_shape[0] corresponds to the height (M) and cta\_tile\_shape[1]
          corresponds to the width (N) of the tile.
        - **use\_2cta\_instrs** (*bool*) – A flag indicating whether the configuration is for a 2SM setup.
        - **layout\_d** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum of the output tensor D.
        - **elem\_ty\_d** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type of output tensor D.
        - **layout\_c** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")*,* *optional*) – The layout enum of the input tensor C. Defaults to None.
        - **elem\_ty\_c** (*Union**[**Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]**,* *None**]**,* *optional*) – The element type for input tensor C. Defaults to None.
        - **tmem\_warp\_shape\_mn** (*Tuple**[**int**,* *int**]**,* *optional*) – Optional (warp\_m, warp\_n) override for the tmem
          subpartition layout. When omitted, the layout is derived from
          `cta_tile_shape` and `use_2cta_instrs`.

    Returns:
    :   Returns epilog tiler, which is used in subsequent epilog partitions.

    Return type:
    :   cute.Tile

    Raises:
    :   **ValueError** – If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.

cutlass.utils.get\_permutation\_mnk( : *tile\_shape\_mnk: cutlass.cute.typing.Shape*, : *sf\_vec\_size: int*, : *use\_mxf8f6f4: bool*, ) → Tuple[int, int, int]
:   Get the permutation of M, N, K for the tiled MMA.

    Parameters:
    :   - **tile\_shape\_mnk** (*cute.Shape*) – The shape of the tile
        - **sf\_vec\_size** (*int*) – The vector size of the Scale Factor.
        - **use\_mxf8f6f4** (*bool*) – Whether to use MXF8F6F4 or MXF4NVF4.

    Returns:
    :   The permutation of M, N, K

    Return type:
    :   Tuple[int, int, int]

    Raises:
    :   **ValueError** – If the tile shape is not divisible by the sf\_vec\_size

cutlass.utils.get\_smem\_layout\_atom\_ab( : *major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode")*, : *element\_type: Type[\_MockObject]*, : *smem\_shape\_mn\_k: cutlass.cute.typing.Tile*, ) → [SmemLayoutAtomKind](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "cutlass.cute.nvgpu.tcgen05.mma.SmemLayoutAtomKind")
:   Simple heuristics to select the optimal SMEM layout atom based on the
    majorness, the data type, and the major mode size.

    Parameters:
    :   - **major\_mode** ([*cutlass.cute.nvgpu.OperandMajorMode*](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.OperandMajorMode")) – The major mode for the SMEM tensor is K major.
        - **element\_type** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for the SMEM tensor.
        - **smem\_shape\_mn\_k** (*cute.Tile*) – The shape of the SMEM tensor.

    Returns:
    :   The SMEM layout atom kind

    Return type:
    :   [cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind")

cutlass.utils.get\_smem\_layout\_atom\_epi( : *layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *element\_type: Type[\_MockObject]*, : *epi\_tile: cutlass.cute.typing.Tile*, ) → [SmemLayoutAtomKind](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "cutlass.cute.nvgpu.tcgen05.mma.SmemLayoutAtomKind")
:   Simple heuristics to select the optimal SMEM layout atom for epilog tensors.

    Parameters:
    :   - **layout** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum for the SMEM tensor.
        - **element\_type** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for the SMEM tensor.
        - **epi\_tile** (*cute.Tile*) – The epilogue tile shape.

    Returns:
    :   The SMEM layout atom kind

    Return type:
    :   [cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind")

cutlass.utils.get\_smem\_store\_op( : *layout\_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[\_MockObject]*, : *elem\_ty\_acc: Type[\_MockObject]*, : *tiled\_tmem\_load: [TiledCopy](cute.md#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")*, ) → [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")
:   Selects the largest vectorized smem store atom available subject to
    constraint of gmem layout and chosen TMEM\_LOAD’s thread-value ownership.

    Parameters:
    :   - **layout\_d** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum of the output tensor D.
        - **elem\_ty\_d** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for output tensor D.
        - **elem\_ty\_acc** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for accumulator.
        - **tiled\_tmem\_load** ([*cute.TiledCopy*](cute.md#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) – An instance of TiledCopy that represents the tmem load operation.

    Returns:
    :   Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

    Return type:
    :   [cute.CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

cutlass.utils.get\_tmem\_load\_op( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *layout\_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[\_MockObject]*, : *elem\_ty\_acc: Type[\_MockObject]*, : *epi\_tile: cutlass.cute.typing.Tile*, : *use\_2cta\_instrs: bool*, : *\**, : *tmem\_warp\_shape\_mn: Tuple[int, int] | None = None*, ) → [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")
:   Finds a performant TMEM\_LOAD copy op for the selected epilogue
    tile (epi\_tile), element types, and tcgen05.mma instruction used.

    Parameters:
    :   - **cta\_tile\_shape** (*cute.Shape*) – A tuple or list representing the dimensions of the CTA tile.
        - **layout\_d** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum of the output tensor D.
        - **elem\_ty\_d** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for output tensor D.
        - **elem\_ty\_acc** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for accumulation.
        - **epi\_tile** (*cute.Tile*) – The epilogue tile configuration.
        - **use\_2cta\_instrs** (*bool*) – A flag indicating whether the configuration is for 2 SMs.

    Returns:
    :   An instance of Sm100TmemLoad with the computed configuration.

    Return type:
    :   [cute.CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

    Raises:
    :   **ValueError** – If the function cannot handle the given combination of accumulation
        and dimension types, or if it cannot determine the appropriate configuration based on
        the input parameters.

cutlass.utils.make\_smem\_layout( : *leading\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode")*, : *smem\_tile\_shape: cutlass.cute.typing.Tile*, : *a\_dtype: Type[\_MockObject]*, : *num\_stages: int*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
:   Construct a staged SMEM layout for an operand given its major mode and tile shape.

    This helper:

    1. Selects a SMEM layout atom using simple heuristics based on the operand’s major mode,
       element type, and the size of the major dimension in `smem_tile_shape`.
    2. Tiles the atom to `smem_tile_shape` and appends a staging dimension of length `num_stages`.
    3. Orders the `(M, N, stage)` axes so the major dimension is contiguous, then coalesces.

    Parameters:
    :   - **leading\_mode** ([*cutlass.cute.nvgpu.OperandMajorMode*](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.OperandMajorMode")) – Operand major mode (`MN` or `K`) of the staged operand.
        - **smem\_tile\_shape** (*cute.Tile*) – 2D SMEM tile shape to stage (before the staging dimension is appended).
        - **a\_dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Element type of the staged operand.
        - **num\_stages** (*int*) – Number of pipeline stages (depth of the staging dimension).

    Returns:
    :   Staged SMEM layout for the operand.

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.make\_smem\_layout\_a( : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *a\_dtype: Type[\_MockObject]*, : *num\_stages: int*, : *\**, : *is\_k\_major: bool | None = None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
:   This function helps with:

    1. Get the partitioned shape of the A tensor based on the tiled\_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   - **tiled\_mma** ([*cute.TiledMma*](cute.md#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) – The tiled MMA used to partition tensor A
        - **mma\_tiler\_mnk** (*cute.cute.Tile*) – The MMA tile shape
        - **a\_dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for tensor A
        - **num\_stages** (*int*) – The number of pipeline stages for tensor A

    Returns:
    :   SMEM layout for tensor A

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.make\_smem\_layout\_b( : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *b\_dtype: Type[\_MockObject]*, : *num\_stages: int*, : *\**, : *is\_k\_major: bool | None = None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
:   This function helps:

    1. Get the partitioned shape of the B tensor based on the tiled\_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   - **tiled\_mma** ([*cute.TiledMma*](cute.md#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) – The tiled MMA which is used to partition the B tensor.
        - **mma\_tiler\_mnk** (*cute.cute.Tile*) – The MMA tile shape.
        - **b\_dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for the B tensor.
        - **num\_stages** (*int*) – The stage of the B tensor.

    Returns:
    :   SMEM layout for the B tensor.

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.make\_smem\_layout\_epi( : *epi\_dtype: Type[\_MockObject]*, : *epi\_layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *epi\_tile: cutlass.cute.typing.Tile*, : *epi\_stage: int*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
:   This function helps:

    1. Select the heuristic SMEM layout atom based on the epilog tile shape,
       the epilog tensor’s majorness, and the element type.
    2. cute.Tile the SMEM layout atom to the epilog tile shape.
    3. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   - **epi\_dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for the epilog tensor.
        - **epi\_layout** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum for the epilog tensor.
        - **epi\_tile** (*cute.cute.Tile*) – The epilogue tile shape.
        - **epi\_stage** (*int*) – The stage of the epilog tensor.

    Returns:
    :   SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.make\_trivial\_tiled\_mma( : *\*args: Any*, : *\*\*kwargs: Any*, ) → [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")
:   Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    Supports two calling conventions:

    **New (recommended):** separate `a_dtype` and `b_dtype`:

    ```console
    make_trivial_tiled_mma(
        a_dtype, b_dtype, a_leading_mode, b_leading_mode,
        acc_dtype, cta_group, mma_tiler_mn, [a_source])
    ```

    **Legacy (deprecated):** single `ab_dtype`:

    ```console
    make_trivial_tiled_mma(
        ab_dtype, a_leading_mode, b_leading_mode,
        acc_dtype, cta_group, mma_tiler_mn, [a_source])
    ```

cutlass.utils.make\_blockscaled\_trivial\_tiled\_mma( : *\*args: Any*, : *\*\*kwargs: Any*, ) → [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")
:   Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    Supports two calling conventions:

    **New (recommended):** separate `a_dtype` and `b_dtype`:

    ```console
    make_blockscaled_trivial_tiled_mma(
        a_dtype, b_dtype, a_leading_mode, b_leading_mode,
        sf_dtype, sf_vec_size, cta_group, mma_tiler_mn, [a_source])
    ```

    **Legacy (deprecated):** single `ab_dtype`:

    ```console
    make_blockscaled_trivial_tiled_mma(
        ab_dtype, a_leading_mode, b_leading_mode,
        sf_dtype, sf_vec_size, cta_group, mma_tiler_mn, [a_source])
    ```

cutlass.utils.sm90\_get\_smem\_layout\_atom( : *layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *element\_type: Type[\_MockObject]*, : *major\_mode\_size: int*, ) → Any
:   Select the optimal shared memory layout atom based on parameters.

    Parameters:
    :   - **layout** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – Layout enum of the tensor
        - **element\_type** (*type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Data type of the elements
        - **major\_mode\_size** (*int*) – Size of the major mode dimension

    Returns:
    :   Selected shared memory layout atom kind

    Return type:
    :   [cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind](cute_nvgpu_warpgroup.md#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind")

cutlass.utils.sm90\_make\_trivial\_tiled\_mma(*a\_dtype: ~typing.Type[~sphinx.ext.autodoc.mock.\_MockObject], b\_dtype: ~typing.Type[~sphinx.ext.autodoc.mock.\_MockObject], a\_leading\_mode: ~cutlass.cute.nvgpu.common.OperandMajorMode, b\_leading\_mode: ~cutlass.cute.nvgpu.common.OperandMajorMode, acc\_dtype: ~typing.Type[~sphinx.ext.autodoc.mock.\_MockObject], atom\_layout\_mnk: ~typing.Tuple[int, int, int], tiler\_mn: ~typing.Tuple[int, int], a\_source: ~cutlass.cute.nvgpu.warpgroup.mma.OperandSource = <OperandSource.RMEM>*) → [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")
:   Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    Parameters:
    :   - **a\_dtype** (*type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Data type of operand A.
        - **b\_dtype** (*type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Data type of operand B.
        - **a\_leading\_mode** ([*cutlass.cute.nvgpu.OperandMajorMode*](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.OperandMajorMode")) – Leading dimension of operand A (1 for K, 0 for M/N).
        - **b\_leading\_mode** ([*cutlass.cute.nvgpu.OperandMajorMode*](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.OperandMajorMode")) – Leading dimension of operand B (1 for K, 0 for M/N).
        - **acc\_dtype** (*type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Data type of the accumulator.
        - **atom\_layout\_mnk** (*Tuple**[**int**,* *int**,* *int**]*) – A integer tuple describing the tiling of Atom across threads.
        - **tiler\_mn** (*Tuple**[**int**,* *int**]*) – The shape (M, N) of the cta tiler.

    Returns:
    :   A tiled MMA atom.

    Return type:
    :   [cute.TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

    Raises:
    :   **TypeError** – If the data type is not supported.

cutlass.utils.block\_copy( : *tiled\_copy: ~cutlass.cute.atom.TiledCopy*, : *src: cutlass.cute.typing.Tensor*, : *dst: cutlass.cute.typing.Tensor*, : *\**, : *\*\*kwargs: ~typing.Any*, ) → None
:   Performs a block-level copy operation.

    This function adds an abstraction layer over the cute.copy usage model by
    allowing operands with layouts shaped like tiles to be passed directly. This
    removes the need to manually partition. The API is designed to support multiple
    copy kinds; currently TMA-based copies and S2T (SMEM to TMEM) copies are supported.

    **TMA copy requirements**:

    When using TMA-based tiled copies, the `src` and `dst` tensors must have
    their first mode representing the TMATile, i.e. tensors shaped as `(TMATile, Rest...)`.
    For a rank-2 tensor with logical layout (e.g., `(TILE_M, TILE_N)`), call
    `group_modes(tensor, 0, 2)` before passing it to this function.

    **TMA multicast support**:

    For TMA-based copies that enable compiler-driven multicast in a 2D cluster, pass the
    `tma_multicast` argument as a dict with the following keys:

    > - `cluster_shape`: a tuple of 2 integers `(cluster_m, cluster_n)`
    >   representing the **2D cluster shape**.
    > - `multicast_dim`: either `"M"` or `"N"` indicating which
    >   cluster dimension the multicast happens along.
    > - `use_2cta_mma_inst` (optional): a `bool` indicating whether to
    >   use 2CTA MMA instructions when the loaded data is consumed by MMA.
    >   Defaults to `False` when omitted.

    **S2T (SMEM to TMEM) copy**:

    When using S2T copy operations (e.g., `tcgen05.Cp4x32x128bOp`), the function
    automatically handles the filtering, partitioning, and SMEM descriptor creation.
    Pass a copy atom created with `cute.make_copy_atom(tcgen05.Cp*Op(...), dtype)`
    along with source (SMEM) and destination (TMEM) tensors.

    Examples:

    ```python
    # 1) TMA load without compiler-driven multicast
    #    Note: group_modes is called to make the first mode TMATile
    block_copy(tma_atom_a, group_modes(tCgA_, 0, 2), group_modes(tCsA_, 0, 2),
               tma_bar_ptr=tma_bar_ptr)

    # 2) TMA load with compiler-driven multicast along M in a (4,2) cluster
    block_copy(
        tma_atom_a,
        group_modes(tCgA_, 0, 2),
        group_modes(tCsA_, 0, 2),
        tma_multicast={
            "cluster_shape": (4, 2),
            "multicast_dim": "M",
            "use_2cta_mma_inst": True,
        },
        tma_bar_ptr=tma_bar_ptr,
    )

    # 3) TMA store
    #    Note that `tma_bar_ptr` and CTA params (`cta_coord` and `cta_layout`)
    #    are not needed for TMA store
    block_copy(tma_atom_c, group_modes(tCsC_, 0, 2), group_modes(tCgC_, 0, 2))

    # 4) S2T copy (SMEM to TMEM)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype
    )
    block_copy(copy_atom_s2t, tCsSF, tCtSF)
    ```

    Parameters:
    :   - **tiled\_copy** ([*TiledCopy*](cute.md#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) – The tiled\_copy or copy\_atom of the current copy operation.
        - **src** (*Tensor*) – The source tensor.
        - **dst** (*Tensor*) – The destination tensor.
        - **tma\_multicast** (*dict**,* *optional*) – Optional dict for TMA multicast configuration with keys
          `cluster_shape`, `multicast_dim`, and optionally
          `use_2cta_mma_inst`.

*class* cutlass.utils.ClcDynamicPersistentTileSchedulerParams(*\*\*kwargs*)
:   Bases: `MixedClusterParamsMixin`

    A class to represent parameters for a dynamic persistent tile scheduler.

    This class is designed to manage and compute the layout of clusters and tiles
    in a batched gemm problem.

    Variables:
    :   **cluster\_shape\_mn** – Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).

    \_\_init\_\_() → None
    :   Initializes the ClcDynamicPersistentTileSchedulerParams with the given parameters.

        Parameters:
        :   - **problem\_shape\_ntile\_mnl** (*cute.Shape*) – The shape of the problem in terms of
              number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
            - **cluster\_shape\_mnk** (*cute.Shape*) – The shape of the cluster in (m, n) dimensions.
            - **swizzle\_size** (*int*) – Swizzling size in the unit of cluster. 1 means no swizzle
            - **raster\_along\_m** (*bool*) – Rasterization order of clusters. Only used when swizzle\_size > 1.
              True means along M, false means along N.
            - **fallback\_cluster\_shape\_mnk** (*Optional**[**cute.Shape**]*) – Optional. When provided and
              different from cluster\_shape\_mnk, the kernel runs in mixed-cluster mode.

        Raises:
        :   **ValueError** – If cluster\_shape\_k is not 1.

    \_extract\_primary\_mlir\_values() → list[ir.Value]

    *property* \_primary\_values\_count*: int*

    \_new\_primary\_from\_mlir\_values( : *values: list[ir.Value]*, ) → [ClcDynamicPersistentTileSchedulerParams](utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")

    get\_grid\_shape() → Tuple[\_MockObject, \_MockObject, \_MockObject]
    :   Computes the grid shape based on the problem shape and cluster shape.

        Returns:
        :   the grid is the CTA numbers that has aligned with cluster shape.

*class* cutlass.utils.ClcDynamicPersistentTileScheduler(*\*\*kwargs*)
:   Bases: `object`

    A scheduler for dynamic persistent tile execution in CUTLASS/CuTe kernels.

    Variables:
    :   - **params** – Tile schedule related params, including cluster shape.
        - **cta\_id\_in\_cluster** – ID of the CTA within its cluster
        - **\_num\_tiles\_executed** – Counter for executed tiles

    \_\_init\_\_( : *params: [ClcDynamicPersistentTileSchedulerParams](utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")*, : *cta\_id\_in\_cluster: cutlass.cute.typing.Coord*, : *num\_tiles\_executed: \_MockObject*, : *clc\_response\_ptr: cutlass.cute.typing.Pointer*, : *block\_idx: Tuple[\_MockObject, \_MockObject, \_MockObject]*, : *insert\_fence: bool = True*, )
    :   Initializes the ClcDynamicPersistentTileScheduler with the given parameters.

        Parameters:
        :   - **params** ([*ClcDynamicPersistentTileSchedulerParams*](utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) – Tile schedule related params, including cluster shape.
            - **cta\_id\_in\_cluster** (*cute.Coord*) – ID of the CTA within its cluster.
            - **num\_tiles\_executed** ([*Int32*](../basic_data_types.md#cutlass.Int32 "cutlass.Int32")) – Counter for executed tiles.
            - **clc\_response\_ptr** (*cute.Pointer*) – Pointer of the clc rsponse.
            - **block\_idx** (*Tuple**[*[*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*]*) – The block index.
            - **insert\_fence** (*bool*) – Whether to insert a fence to ensure generic-async proxy order.
              CLC issue is in async proxy while loading the response from shared memory is in
              generic proxy. A cross-proxy fence is needed to ensure producer’s next issue
              won’t race with consumer’s current loading. Therefore the scheduler inserts a
              fence by default after loading the response.
              Developers may insert the fence in pipeline acquire/release functions. In that case,
              the fence here can be omitted.

    *static* create( : *params: [ClcDynamicPersistentTileSchedulerParams](utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileSchedulerParams")*, : *block\_idx: Tuple[\_MockObject, \_MockObject, \_MockObject]*, : *grid\_dim: Tuple[\_MockObject, \_MockObject, \_MockObject]*, : *clc\_response\_ptr: cutlass.cute.typing.Pointer*, : *insert\_fence: bool = True*, ) → [ClcDynamicPersistentTileScheduler](utils.md#cutlass.utils.ClcDynamicPersistentTileScheduler "cutlass.utils.dynamic_persistent_tile_scheduler.ClcDynamicPersistentTileScheduler")
    :   Initialize the dynamic persistent tile scheduler.

        Parameters:
        :   - **params** ([*ClcDynamicPersistentTileSchedulerParams*](utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) – Parameters for the persistent
              tile scheduler.
            - **block\_idx** (*Tuple**[*[*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*]*) – The 3d block index in the format (bidx, bidy, bidz).
            - **grid\_dim** (*Tuple**[*[*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*,* [*Integer*](../basic_data_types.md#cutlass.Integer "cutlass.Integer")*]*) – The 3d grid dimensions for kernel launch.

        Returns:
        :   A ClcDynamicPersistentTileScheduler object.

        Return type:
        :   [ClcDynamicPersistentTileScheduler](utils.md#cutlass.utils.ClcDynamicPersistentTileScheduler "cutlass.utils.ClcDynamicPersistentTileScheduler")

    get\_grid\_shape() → Tuple[\_MockObject, \_MockObject, \_MockObject]
    :   Calculates the grid shape to be launched on GPU using problem shape,
        threadblock shape, and active cluster size.

        Parameters:
        :   **params** ([*ClcDynamicPersistentTileSchedulerParams*](utils.md#cutlass.utils.ClcDynamicPersistentTileSchedulerParams "cutlass.utils.ClcDynamicPersistentTileSchedulerParams")) – Parameters for grid shape calculation.

        Returns:
        :   The calculated 3d grid shape.

        Return type:
        :   Tuple[[Integer](../basic_data_types.md#cutlass.Integer "cutlass.Integer"), [Integer](../basic_data_types.md#cutlass.Integer "cutlass.Integer"), [Integer](../basic_data_types.md#cutlass.Integer "cutlass.Integer")]

    \_swizzle\_and\_rasterize( : *x\_idx: \_MockObject*, : *y\_idx: \_MockObject*, : *z\_idx: \_MockObject*, ) → Tuple[\_MockObject, \_MockObject, \_MockObject]
    :   Swizzle and rasterize the given coordinates for leader CTA of the cluster.
        x\_idx, y\_idx, and z\_idx must be divisible by cluster shape x, y, and z respectively. They should not be offset
        by the ID of the CTA in the cluster.

    work\_tile\_info\_from\_clc\_response( : *result\_addr: cutlass.cute.typing.Pointer*, ) → [WorkTileInfo](utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")
    :   Simulates parsing CLC response data in Python.
        result\_addr: 16-byte response data (simulating shared memory access)

    get\_current\_work() → [WorkTileInfo](utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")

    initial\_work\_tile\_info() → [WorkTileInfo](utils.md#cutlass.utils.WorkTileInfo "cutlass.utils.static_persistent_tile_scheduler.WorkTileInfo")

    advance\_to\_next\_work( : *mbarrier\_addr: cutlass.cute.typing.Pointer*, ) → None

    *property* num\_tiles\_executed*: \_MockObject*

cutlass.utils.print\_latex(*x: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, \*, color: ~typing.Callable = <function tikz\_color\_bwx8>, render\_func: ~typing.Callable[[str], None] | None = None*) → None
:   Prints a layout.

    Parameters:
    :   - **x** (*Union**[**Layout**,* *ComposedLayout**]*) – A layout
        - **color** (*Callable*) – A function that returns TiKZ colors
        - **render\_func** (*Callable**[**[**str**]**,* *None**]* *|* *None*) – Optional callback fed the `{tikzpicture}` body
          (without the standalone-document wrapper) in a single call,
          instead of printing a full LaTeX document to stdout.

cutlass.utils.print\_latex\_tv(*layout\_tv: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout, tile\_mn: cutlass.cute.typing.IntTuple | cutlass.cute.typing.Layout, \*, color: ~typing.Callable = <function tikz\_color\_tv>, palette: str | ~typing.Callable | None = None, title: str | None = None, axis\_labels: bool = False, render\_func: ~typing.Callable[[str], None] | None = None*) → None
:   Prints a tv layout for a tile M N. Everything must be static.

    Parameters:
    :   - **layout\_tv** (*Union**[**Layout**,* *ComposedLayout**]*) – A static thread value layout
        - **tile\_mn** (*Union**[**IntTuple**,* *Layout**]*) – A static M N tile
        - **color** (*Callable*) – A function `color(tid, vid) -> str` returning a TikZ
          fill color for the cell owned by thread `tid` value `vid`.
          Used when `palette` is `None`; ignored otherwise.
        - **palette** (*Optional**[**Union**[**str**,* *Callable**]**]*) – Optional richer cell coloring that supersedes
          `color`. Either the name of a built-in palette (a key of
          `PALETTES`: the color `"pastel"`, `"rainbow"`,
          `"rainbow_dual"` or the monochrome `"white"`, `"bw"`,
          `"bw_dual"`) or a factory `palette(num_tid, num_vid) -> cell`
          where
          `cell(tid, vid)` returns either a TikZ fill string (one fill
          spanning the cell) or a list of ``` Band``s (stacked horizontal
          fills).  The factory form lets a palette capture the thread /
          value counts -- needed to spread hues evenly over the wheel --
          without widening the per-cell ``(tid, vid) ``` contract.
        - **title** (*Optional**[**str**]*) – Optional title drawn above the figure, one line per
          `\n`-separated segment (e.g. the operator / function /
          tensor that produced this layout). LaTeX specials are escaped.
        - **axis\_labels** (*bool*) – When `True`, annotate the M (row, downward)
          and N (column, rightward) axis directions. The picture’s
          coordinate basis runs M down and N right; these labels make
          that orientation explicit so a reader can tell whether a
          thread’s values are contiguous along the memory-major axis.
        - **render\_func** (*Callable**[**[**str**]**,* *None**]* *|* *None*) – Optional callback fed the `{tikzpicture}` body
          (without the standalone-document wrapper) in a single call,
          instead of printing a full LaTeX document to stdout.

*class* cutlass.utils.Band(*lo: float*, *hi: float*, *color: str*)
:   Bases: `NamedTuple`

    One horizontal fill band of a TV cell. `lo` and `hi` are
    fractions in `[0, 1]` along the screen-vertical M axis (0 = the
    cell’s top edge, 1 = its bottom edge); `color` is a TikZ fill spec.
    A single `Band(0.0, 1.0, ...)` is an ordinary one-fill cell; the
    `"rainbow_dual"` palette stacks two half-height bands.

    lo*: float*
    :   Alias for field number 0

    hi*: float*
    :   Alias for field number 1

    color*: str*
    :   Alias for field number 2

    \_asdict()
    :   Return a new dict which maps field names to their values.

    \_field\_defaults *= {}*

    \_fields *= ('lo', 'hi', 'color')*

    *classmethod* \_make(*iterable*)
    :   Make a new Band object from a sequence or iterable

    \_replace(*\*\*kwds*)
    :   Return a new Band object replacing specified fields with new values

cutlass.utils.is\_fp8\_dtype(*dtype: Type[cutlass.cute.typing.Numeric]*) → bool
:   Check if dtype is a float8 type that doesn’t support dlpack.
    params dtype: The cutlass numeric type to check
    type dtype: Type[cutlass.Numeric]
    return: True if the dtype is Float8E5M2 or Float8E4M3FN, False otherwise

cutlass.utils.create\_cute\_tensor\_for\_fp8( : *storage\_tensor: Any*, : *dtype: Type[cutlass.cute.typing.Numeric]*, : *leading\_dim: int*, : *source\_f32\_tensor: Any | None = None*, : *assumed\_align: int = 16*, : *mark\_dynamic\_layout: bool = True*, ) → cutlass.cute.typing.Tensor
:   Create cute tensor, handling float8 types that don’t support dlpack.

    For float8 types, the storage\_tensor should use byte storage (for DLPack compatibility).
    The source\_f32\_tensor provides the actual float32 values to convert to fp8.

    params storage\_tensor: Tensor for DLPack (byte storage for fp8, otherwise the actual dtype)
    params dtype: Target cutlass dtype
    params leading\_dim: Leading dimension for dynamic layout
    paramas source\_f32\_tensor: Float32 source data for fp8 conversion (required for fp8)
    params assumed\_align: Assumed alignment for the DLPack tensor
    params mark\_dynamic\_layout: Whether to mark the resulting tensor layout dynamic
    return: A cute tensor with the appropriate dtype and layout
