# nvgpu.common

*class* cutlass.cute.nvgpu.OperandMajorMode(*value*)
:   Bases: `Enum`

    An enumeration for the majorness of the input operands of the MMA.

    MN *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    K *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

*class* cutlass.cute.nvgpu.OutputMajorMode(*value*)
:   Bases: `Enum`

    Major mode for the output operand D(M, N).

    M = M-major (column-major): stride=(1, M), contiguous along M.
    N = N-major (row-major): stride=(N, 1), contiguous along N.

    M *= 'm'*

    N *= 'n'*

cutlass.cute.nvgpu.normalize\_field\_to\_ir\_name( : *field: Any*, : *admissible\_fields: Any*, ) → str
:   Normalize a field specifier to its IR logical field name.

    Accepted inputs:

    - Enum value present in admissible\_fields (must expose \_to\_ir\_field\_name()).
    - Exact string IR name (e.g., “accum\_c”, “neg\_a”, “sf\_a”).

    Any other form is rejected.

*class* cutlass.cute.nvgpu.MmaUniversalOp(*abacc\_dtype: Type[cutlass.cute.typing.Numeric]*)
:   Bases: `MmaOp`

    The universal MMA Operation.

    This Operation currently expects the A/B operands as well as the accumulator to share the same
    data types.

    **Supported architectures:** all (universal FMA)

    Parameters:
    :   **abacc\_dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The data type for the A/B operands and the accumulator

    abacc\_dtype*: Type[cutlass.cute.typing.Numeric]*

*class* cutlass.cute.nvgpu.MmaUniversalTrait(*value: ir.Value*)
:   Bases: `Trait`

*class* cutlass.cute.nvgpu.CopyUniversalOp
:   Bases: `CopyOp`

    The universal Copy Operation.

    This operation is equivalent to the `a = b` assignment without any extra
    memory attributes. For advanced memory features (memory order, memory scope,
    cache eviction priority, invariant loads, etc.) please use the specialized copy
    operations instead:

    - [`CopyG2ROp`](cute_nvgpu_common.md#cutlass.cute.nvgpu.CopyG2ROp "cutlass.cute.nvgpu.CopyG2ROp") – global memory to register
    - [`CopyR2GOp`](cute_nvgpu_common.md#cutlass.cute.nvgpu.CopyR2GOp "cutlass.cute.nvgpu.CopyR2GOp") – register to global memory
    - [`CopyS2ROp`](cute_nvgpu_common.md#cutlass.cute.nvgpu.CopyS2ROp "cutlass.cute.nvgpu.CopyS2ROp") – shared memory to register
    - [`CopyR2SOp`](cute_nvgpu_common.md#cutlass.cute.nvgpu.CopyR2SOp "cutlass.cute.nvgpu.CopyR2SOp") – register to shared memory

    When creating a Copy Atom out of this operation, the expected usage pattern is

    ```python
    op = cute.nvgpu.CopyUniversalOp()
    atom = cute.make_copy_atom(op, tensor_dtype, num_bits_per_copy=64)
    ```

    - `tensor_dtype` is the data type used to build the reference TV Layout (either the source or the destination TV Layout) in unit of tensor elements and is used for partitioning by `TiledCopy` for example
    - `num_bits_per_copy` is a kw argument specifying the number of bits to copy per Atom execution. This can be larger than the width of the above data type. When not provided, the compiler will do a best effort at auto-vectorizing.

*class* cutlass.cute.nvgpu.CopyUniversalTrait(*value: ir.Value*)
:   Bases: `Trait`

*class* cutlass.cute.nvgpu.CopyG2ROp
:   Bases: `CopyOp`

    The G2R copy operation.

    When creating a Copy Atom out of this operation, the expected usage pattern is

    ```python
    op = cute.nvgpu.CopyG2ROp()
    atom = cute.make_copy_atom(
        op,
        tensor_dtype,
        num_bits_per_copy=64,
        memory_order=cute.nvgpu.MemoryOrder.VOLATILE,
        memory_scope=cute.nvgpu.MemoryScope.SYS,
        l2_prefetch_size=cute.nvgpu.L2PrefetchSize.NONE,
        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL,
        load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        shared_space=cute.nvgpu.SharedSpace.CTA,
        invariant=False,
    )
    ```

*class* cutlass.cute.nvgpu.CopyG2RTrait(*value: ir.Value*)
:   Bases: `Trait`

    unpack( : *\**, : *cache\_policy: cutlass.cute.typing.Int64 | None = None*, : *\*\*kwargs: Any*, ) → ir.Value

*class* cutlass.cute.nvgpu.CopyR2GOp
:   Bases: `CopyOp`

    The R2G copy operation.

    When creating a Copy Atom out of this operation, the expected usage pattern is

    ```python
    op = cute.nvgpu.CopyR2GOp()
    atom = cute.make_copy_atom(
        op,
        tensor_dtype,
        num_bits_per_copy=64,
        memory_order=cute.nvgpu.MemoryOrder.RELEASE,
        memory_scope=cute.nvgpu.MemoryScope.CLUSTER,
        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL,
        shared_space=cute.nvgpu.SharedSpace.CTA,
    )
    ```

*class* cutlass.cute.nvgpu.CopyR2GTrait(*value: ir.Value*)
:   Bases: `Trait`

    unpack( : *\**, : *cache\_policy: cutlass.cute.typing.Int64 | None = None*, : *\*\*kwargs: Any*, ) → ir.Value

*class* cutlass.cute.nvgpu.CopyS2ROp
:   Bases: `CopyOp`

    The S2R copy operation.

    When creating a Copy Atom out of this operation, the expected usage pattern is

    ```python
    op = cute.nvgpu.CopyS2ROp()
    atom = cute.make_copy_atom(
        op,
        tensor_dtype,
        num_bits_per_copy=64,
        memory_order=cute.nvgpu.MemoryOrder.WEAK,
        memory_scope=cute.nvgpu.MemoryScope.CTA,
        shared_space=cute.nvgpu.SharedSpace.CTA,
    )
    ```

*class* cutlass.cute.nvgpu.CopyS2RTrait(*value: ir.Value*)
:   Bases: `Trait`

*class* cutlass.cute.nvgpu.CopyR2SOp
:   Bases: `CopyOp`

    The R2S copy operation.

    When creating a Copy Atom out of this operation, the expected usage pattern is

    ```python
    op = cute.nvgpu.CopyR2SOp()
    atom = cute.make_copy_atom(
        op,
        tensor_dtype,
        num_bits_per_copy=64,
        memory_order=cute.nvgpu.MemoryOrder.WEAK,
        memory_scope=cute.nvgpu.MemoryScope.CTA,
        shared_space=cute.nvgpu.SharedSpace.CTA,
    )
    ```

*class* cutlass.cute.nvgpu.CopyR2STrait(*value: ir.Value*)
:   Bases: `Trait`

*class* cutlass.cute.nvgpu.MemoryOrder(*value*)
:   Bases: `Enum`

    An enumeration.

    WEAK *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    RELAXED *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    ACQUIRE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    RELEASE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    ACQ\_REL *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    SC *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    MMIO *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    CONSTANT *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    VOLATILE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

*class* cutlass.cute.nvgpu.MemoryScope(*value*)
:   Bases: `Enum`

    An enumeration.

    CTA *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    CLUSTER *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    GPU *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    SYS *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

*class* cutlass.cute.nvgpu.L2PrefetchSize(*value*)
:   Bases: `Enum`

    An enumeration.

    NONE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    RESERVED *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    SIZE\_64B *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    SIZE\_128B *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    SIZE\_256B *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

*class* cutlass.cute.nvgpu.CacheEvictionPriority(*value*)
:   Bases: `Enum`

    An enumeration.

    EVICT\_NORMAL *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    EVICT\_FIRST *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    EVICT\_LAST *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    EVICT\_UNCHANGED *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    NO\_ALLOCATE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

*class* cutlass.cute.nvgpu.LoadCacheMode(*value*)
:   Bases: `Enum`

    An enumeration.

    ALWAYS *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    GLOBAL *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    STREAMING *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    LAST\_USE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    NONE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

*class* cutlass.cute.nvgpu.StoreCacheMode(*value*)
:   Bases: `Enum`

    An enumeration.

    WRITE\_BACK *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    GLOBAL *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    STREAMING *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    WRITE\_THROUGH *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    NONE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

*class* cutlass.cute.nvgpu.SharedSpace(*value*)
:   Bases: `Enum`

    An enumeration.

    CTA *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    CLUSTER *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

cutlass.cute.nvgpu.make\_tiled\_tma\_atom\_A( : *op: [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp") | [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *mma\_tiler\_mnk: cutlass.cute.typing.Shape*, : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *cluster\_shape\_vmnk: cutlass.cute.typing.Shape | None = None*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, ) → [TmaInfo](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo "cutlass.cute.nvgpu.cpasync.helpers.TmaInfo")
:   Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
    accounting for the MK projections of the TiledMMA for A tensor loads.

    Given

    - a GMEM tensor
    - a SMEM layout
    - a MMA Tiler
    - a TiledMma
    - a Cluster-level shape

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    “TMA vector length” to copy tiles of the GMEM tensor to an SMEM buffer with the provided
    layout and consistent with the provided Tiler & tiled\_mma (considering the M-mode & K-mode).
    The Cluster-level shape is used to determine the multicast factor across the N-mode for A tensor loads.

    This function returns two results:

    1. the Copy Atom
    2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
       similarly to any other CuTe tensors using the algebra.

    Parameters:
    :   - **op** (*Union**[*[*CopyBulkTensorTileG2SOp*](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")*,* [*CopyBulkTensorTileG2SMulticastOp*](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp")*]*) – The Copy Operation to construct an Atom for
        - **gmem\_tensor** (*Tensor*) – The GMEM tensor to be loaded by this copy atom
        - **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) – Shared memory layout to load the tensor into (PDSL)
        - **mma\_tiler\_mnk** (*Shape*) – The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
        - **tiled\_mma** ([*atom.TiledMma*](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")) – The TiledMMA that will consume the load as operands
        - **cluster\_shape\_vmnk** (*Shape*) – The Cluster-level shape in VMNK dimensions
        - **internal\_type** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Optional element-format override used when the
          tensor element type does not match the copy type

    Returns:
    :   A TmaInfo containing the Copy Atom, TMA tensor, and SMEM layout

    Return type:
    :   [TmaInfo](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo "cutlass.cute.nvgpu.cpasync.TmaInfo")

cutlass.cute.nvgpu.make\_tiled\_tma\_atom\_B( : *op: [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp") | [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *mma\_tiler\_mnk: cutlass.cute.typing.Shape*, : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *cluster\_shape\_vmnk: cutlass.cute.typing.Shape | None = None*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, ) → [TmaInfo](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo "cutlass.cute.nvgpu.cpasync.helpers.TmaInfo")
:   Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
    accounting for the NK projections of the TiledMMA for B tensor loads.

    Given

    - a GMEM tensor
    - a SMEM layout
    - a MMA Tiler
    - a TiledMma
    - a Cluster-level shape

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    “TMA vector length” to copy tiles of the GMEM tensor to an SMEM buffer with the provided
    layout and consistent with the provided Tiler & tiled\_mma (considering the N-mode & K-mode).
    The Cluster-level shape is used to determine the multicast factor across the M-mode for B tensor loads.

    This function returns two results:

    1. the Copy Atom
    2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
       similarly to any other CuTe tensors using the algebra.

    Parameters:
    :   - **op** (*Union**[*[*CopyBulkTensorTileG2SOp*](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")*,* [*CopyBulkTensorTileG2SMulticastOp*](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp")*]*) – The Copy Operation to construct an Atom for
        - **gmem\_tensor** (*Tensor*) – The GMEM tensor to be loaded by this copy atom
        - **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) – Shared memory layout to load the tensor into (PDSL)
        - **mma\_tiler\_mnk** (*Shape*) – The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
        - **tiled\_mma** (*core.TiledMma*) – The TiledMMA that will consume the load as operands
        - **cluster\_shape\_vmnk** (*Shape*) – The Cluster-level shape in VMNK dimensions
        - **internal\_type** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Optional element-format override used when the
          tensor element type does not match the copy type

    Returns:
    :   A TmaInfo containing the Copy Atom, TMA tensor, and SMEM layout

    Return type:
    :   [TmaInfo](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo "cutlass.cute.nvgpu.cpasync.TmaInfo")

cutlass.cute.nvgpu.make\_im2col\_tma\_atom\_A( : *op: CopyBulkTensorIm2ColG2SOp | CopyBulkTensorIm2ColG2SMulticastOp*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *mma\_tiler\_mnk: cutlass.cute.typing.Shape*, : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *filter\_trs: Tuple[int, int, int]*, : *upper\_padding\_dhw: Tuple[int, int, int]*, : *lower\_padding\_dhw: Tuple[int, int, int]*, : *stride\_dhw: Tuple[int, int, int]*, : *dilation\_dhw: Tuple[int, int, int]*, : *cluster\_shape\_vmnk: cutlass.cute.typing.Shape | None = None*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, ) → [TmaInfo](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo "cutlass.cute.nvgpu.cpasync.helpers.TmaInfo")
:   Makes a TMA Copy atom mapping to `.im2col` mode for `cp.async.bulk.tensor` PTX operation accounting for the MK projections of the TiledMMA for A tensor loads.

    Given

    - a GMEM tensor
    - a SMEM layout
    - a MMA Tiler
    - a TiledMma
    - a filter shape
    - a padding shape
    - a stride shape
    - a dilation shape
    - a Cluster-level shape

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    “TMA vector length” to copy tiles of the GMEM tensor to/from an SMEM buffer with the provided
    layout while maintaining consistency with the provided Tiler.

    This function returns two results:

    1. the Copy Atom
    2. the TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
       similarly to any other CuTe tensors using the algebra.

    Parameters:
    :   - **op** (*Union**[**CopyBulkTensorIm2ColG2SOp**,* *CopyBulkTensorIm2ColG2SMulticastOp**]*) – The Copy Operation to construct an Atom for
        - **gmem\_tensor** (*Tensor*) – The GMEM tensor to be loaded by this copy atom
        - **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) – Shared memory layout to load the tensor into (PDSL)
        - **mma\_tiler\_mnk** (*Shape*) – The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
        - **tiled\_mma** ([*atom.TiledMma*](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")) – The TiledMMA that will consume the load as operands
        - **filter\_trs** (*Tuple**[**int**,* *int**,* *int**]*) – The filter shape (T, R, S) in TRS dimensions
        - **upper\_padding\_dhw** (*Tuple**[**int**,* *int**,* *int**]*) – The upper padding shape (D, H, W) in DHW dimensions
        - **lower\_padding\_dhw** (*Tuple**[**int**,* *int**,* *int**]*) – The lower padding shape (D, H, W) in DHW dimensions
        - **stride\_dhw** (*Tuple**[**int**,* *int**,* *int**]*) – The stride shape (D, H, W) in DHW dimensions
        - **dilation\_dhw** (*Tuple**[**int**,* *int**,* *int**]*) – The dilation shape (D, H, W) in DHW dimensions
        - **cluster\_shape\_vmnk** (*Shape*) – The Cluster-level shape in VMNK dimensions
        - **internal\_type** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Optional element-format override used when the
          tensor element type does not match the copy type

    Returns:
    :   A TmaInfo containing the Copy Atom, TMA tensor, and SMEM layout

    Return type:
    :   [TmaInfo](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo "cutlass.cute.nvgpu.cpasync.TmaInfo")
