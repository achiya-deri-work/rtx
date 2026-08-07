# nvgpu.cpasync

*class* cutlass.cute.nvgpu.cpasync.LoadCacheMode(*value*)
:   Bases: `Enum`

    An enumeration for the possible cache modes of a non-bulk `cp.async` instruction.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators).

    ALWAYS *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    GLOBAL *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    STREAMING *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    LAST\_USE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    NONE *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

*class* cutlass.cute.nvgpu.cpasync.CopyG2SOp( : *cache\_mode: ~cutlass.cute.nvgpu.common.LoadCacheMode | ~cutlass.cute.nvgpu.cpasync.copy.LoadCacheMode = <LoadCacheMode.ALWAYS>*, )
:   Bases: `CopyOp`

    Non-bulk asynchronous GMEM to SMEM Copy Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy).

    cache\_mode*: [LoadCacheMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.LoadCacheMode "cutlass.cute.nvgpu.common.LoadCacheMode")* *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    \_\_init\_\_( : *cache\_mode: ~cutlass.cute.nvgpu.common.LoadCacheMode | ~cutlass.cute.nvgpu.cpasync.copy.LoadCacheMode = <LoadCacheMode.ALWAYS>*, )

*class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, )
:   Bases: `CopyG2STileBaseOp`

    Bulk tensor asynchronous GMEM to SMEM Copy Operation using the TMA unit.

    TMA copy operations are issued by a single thread within a warp, but the DSL **automatically handles this** by
    implicitly adding `elect_one()` around the copy operation.

    ```python
    # CORRECT: TMA copy without elect_one
    cute.copy(
        tma_atom,
        gmem_tensor,  # TMA partition ensures single-thread automatically
        smem_tensor,
        tma_bar_ptr=barrier_ptr
    )

    # WRONG: Do NOT wrap in elect_one (can cause deadlock)
    with cute.arch.elect_one():  # INCORRECT
        cute.copy(tma_atom, gmem_tensor, smem_tensor, tma_bar_ptr=barrier_ptr)
    ```

    While the TMA copy itself does not need `elect_one()`, barrier initialization and transaction byte setup **must** use `elect_one()`:

    ```python
    # Barrier setup requires elect_one
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(barrier_ptr, arrival_count)
        cute.arch.mbarrier_expect_tx(barrier_ptr, num_tma_bytes)

    # TMA copy does NOT need elect_one
    cute.copy(tma_atom, gmem_tensor, smem_tensor, tma_bar_ptr=barrier_ptr)
    ```

    **PTX Programming Model**: In PTX, TMA operations (`cp.async.bulk.tensor`) must be issued
    by a single thread. The DSL automatically handles this.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
    This Operation uses TMA in the `.tile` mode.

    See also

    - [`cute.arch.elect_one`](cute_arch.md#cutlass.cute.arch.elect_one "cutlass.cute.arch.elect_one") - **NOT** needed for TMA copy, but needed for barrier setup
    - [`cute.arch.mbarrier_init`](cute_arch.md#cutlass.cute.arch.mbarrier_init "cutlass.cute.arch.mbarrier_init") - Requires elect\_one
    - [`cute.arch.mbarrier_expect_tx`](cute_arch.md#cutlass.cute.arch.mbarrier_expect_tx "cutlass.cute.arch.mbarrier_expect_tx") - Requires elect\_one
    - Tutorial example: `examples/blackwell/tutorial_tma/tma_v0.py`

    \_\_init\_\_( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, ) → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, )
:   Bases: `CopyG2STileBaseOp`

    Bulk tensor asynchronous multicast GMEM to SMEM Copy Operation using the TMA unit.

    TMA multicast operations are issued by a single thread within a warp, but the DSL **automatically handles this** by
    implicitly adding `elect_one()` around the copy operation.

    ```python
    # CORRECT: TMA multicast without elect_one
    cute.copy(
        tma_atom.with_(mcast_mask=cluster_mask),
        gmem_tensor,
        smem_tensor,
        tma_bar_ptr=barrier_ptr
    )

    # WRONG: Do NOT wrap in elect_one (can cause deadlock)
    with cute.arch.elect_one():  # INCORRECT
        cute.copy(tma_atom.with_(mcast_mask=mask), gmem_tensor, smem_tensor)
    ```

    **PTX Programming Model**: In PTX, TMA multicast operations (`cp.async.bulk.tensor.multicast`)
    must be issued by a single thread.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
    This Operation uses TMA in the `.tile` mode.

    See also

    - [`cute.arch.elect_one`](cute_arch.md#cutlass.cute.arch.elect_one "cutlass.cute.arch.elect_one") - **NOT** needed for TMA copy
    - [`CopyBulkTensorTileG2SOp`](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp") - Non-multicast TMA load

    \_\_init\_\_( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, ) → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp
:   Bases: [`TmaCopyOp`](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaCopyOp "cutlass.cute.nvgpu.cpasync.copy.TmaCopyOp")

    Bulk tensor asynchronous SMEM to GMEM Copy Operation using the TMA unit.

    TMA store operations are issued by a single thread within a warp, but the DSL **automatically handles this** by
    implicitly adding `elect_one()` around the copy operation.

    ```python
    # CORRECT: TMA store without elect_one
    cute.copy(
        tma_atom,
        smem_tensor,  # Source: shared memory
        gmem_tensor,  # Destination: global memory
    )

    # WRONG: Do NOT wrap in elect_one (causes deadlock)
    with cute.arch.elect_one():  # INCORRECT
        cute.copy(tma_atom, smem_tensor, gmem_tensor)
    ```

    **PTX Programming Model**: In PTX, TMA store operations must be issued by a single thread.
    The DSL automatically handles this.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
    This Operation uses TMA in the `.tile` mode.

    See also

    - [`cute.arch.elect_one`](cute_arch.md#cutlass.cute.arch.elect_one "cutlass.cute.arch.elect_one") - **NOT** needed for TMA store
    - [`CopyBulkTensorTileG2SOp`](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp") - TMA load operation
    - Tutorial example: `examples/blackwell/tutorial_tma/tma_v0.py`

    \_\_init\_\_() → None

*class* cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp( : *reduction\_kind: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject = <class 'sphinx.ext.autodoc.mock.\_MockObject'>*, )
:   Bases: [`TmaCopyOp`](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaCopyOp "cutlass.cute.nvgpu.cpasync.copy.TmaCopyOp")

    Bulk tensor asynchronous SMEM to GMEM Reduction Operation using the TMA unit.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk).
    This Operation uses TMA in the `.tile` mode.

    reduction\_kind
    :   alias of `_MockObject`

    \_\_init\_\_( : *reduction\_kind: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject = <class 'sphinx.ext.autodoc.mock.\_MockObject'>*, ) → None

*class* cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp
:   Bases: `CopyOp`

    Asynchronous Store operation to DSMEM with explicit synchronization.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async).

    \_\_init\_\_() → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkG2SOp
:   Bases: `CopyOp`

    Bulk copy asynchronous GMEM to SMEM Copy Operation.

    Invoke `cute.copy()` collectively from a converged warp. The compiler
    elects one issuing lane for this operation; do not wrap the call in
    [`cute.arch.elect_one`](cute_arch.md#cutlass.cute.arch.elect_one "cutlass.cute.arch.elect_one"). An outer election would leave only one lane
    able to reach the compiler-generated full-warp election, creating an
    invalid synchronization that can deadlock. With NVVM diagnostics enabled,
    the compiler rejects this pattern.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk).

    \_\_init\_\_() → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkG2SMulticastOp
:   Bases: `CopyOp`

    Bulk multicast copy asynchronous GMEM to SMEM Copy Operation.

    Invoke `cute.copy()` collectively from a converged warp. The compiler
    elects one issuing lane for this operation; do not wrap the call in
    [`cute.arch.elect_one`](cute_arch.md#cutlass.cute.arch.elect_one "cutlass.cute.arch.elect_one"). An outer election would leave only one lane
    able to reach the compiler-generated full-warp election, creating an
    invalid synchronization that can deadlock. With NVVM diagnostics enabled,
    the compiler rejects this pattern.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk).

    \_\_init\_\_() → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkS2GOp
:   Bases: `CopyOp`

    Bulk copy asynchronous SMEM to GMEM Copy Operation.

    Invoke `cute.copy()` collectively from a converged warp. The compiler
    elects one issuing lane for this operation; do not wrap the call in
    [`cute.arch.elect_one`](cute_arch.md#cutlass.cute.arch.elect_one "cutlass.cute.arch.elect_one"). An outer election would leave only one lane
    able to reach the compiler-generated full-warp election, creating an
    invalid synchronization that can deadlock. With NVVM diagnostics enabled,
    the compiler rejects this pattern.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk).

    \_\_init\_\_() → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkS2GByteMaskOp
:   Bases: `CopyOp`

    Bulk copy asynchronous SMEM to GMEM Copy Operation with mask.
    The i-th bit in the 16-bit wide byteMask operand specifies whether
    the i-th byte of each 16-byte wide chunk of source data is copied to the destination.

    Invoke `cute.copy()` collectively from a converged warp. The compiler
    elects one issuing lane for this operation; do not wrap the call in
    [`cute.arch.elect_one`](cute_arch.md#cutlass.cute.arch.elect_one "cutlass.cute.arch.elect_one"). An outer election would leave only one lane
    able to reach the compiler-generated full-warp election, creating an
    invalid synchronization that can deadlock. With NVVM diagnostics enabled,
    the compiler rejects this pattern.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk).

    \_\_init\_\_() → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkS2SOp
:   Bases: `CopyOp`

    Bulk copy asynchronous SMEM CTA to Cluster Copy Operation.

    Invoke `cute.copy()` collectively from a converged warp. The compiler
    elects one issuing lane for this operation; do not wrap the call in
    [`cute.arch.elect_one`](cute_arch.md#cutlass.cute.arch.elect_one "cutlass.cute.arch.elect_one"). An outer election would leave only one lane
    able to reach the compiler-generated full-warp election, creating an
    invalid synchronization that can deadlock. With NVVM diagnostics enabled,
    the compiler rejects this pattern.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk).

    \_\_init\_\_() → None

*class* cutlass.cute.nvgpu.cpasync.TmaCopyOp
:   Bases: `CopyOp`

    Base class for all TMA copy operations.

*class* cutlass.cute.nvgpu.cpasync.TmaInfo( : *copy\_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *tma\_tensor: Any*, : *smem\_layout: Any | None = None*, )
:   Bases: `object`

    Container for TMA Copy Atom and related data.

    This class uses software composition to bundle a CopyAtom with the SMEM
    layout and TMA tensor.

    Supports tuple unpacking for backward compatibility:

    ```console
    atom, tma_tensor = make_tiled_tma_atom(...)
    ```

    Access smem\_layout via the container:

    ```console
    tma_info = make_tiled_tma_atom(...)
    layout = tma_info.smem_layout
    ```

    Parameters:
    :   - **atom** ([*CopyAtom*](cute.md#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – The TMA Copy Atom
        - **tma\_tensor** – The TMA tensor for coordinate mapping
        - **smem\_layout** – The SMEM layout used to construct the TMA descriptor

    \_\_init\_\_( : *copy\_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *tma\_tensor: Any*, : *smem\_layout: Any | None = None*, ) → None

    *property* atom*: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*
    :   The TMA Copy Atom.

    *property* tma\_tensor*: Any*
    :   The TMA tensor for coordinate mapping.

    *property* smem\_layout*: Any*
    :   The SMEM layout used to construct the TMA descriptor.

cutlass.cute.nvgpu.cpasync.make\_tiled\_tma\_atom( : *op: [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp") | [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileS2GOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileS2GOp") | CopyBulkTensorIm2ColG2SOp | CopyBulkTensorIm2ColS2GOp | [CopyReduceBulkTensorTileS2GOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp "cutlass.cute.nvgpu.cpasync.copy.CopyReduceBulkTensorTileS2GOp")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout\_: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *cta\_tiler: cutlass.cute.typing.Tiler*, : *num\_multicast: int = 1*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, ) → [TmaInfo](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo "cutlass.cute.nvgpu.cpasync.helpers.TmaInfo")
:   Makes a TMA Copy Atom to copy tiles of a GMEM tensor to/from SMEM buffer with the given Layout.

    Supports `.tile` mode (default), `.tile::gather4` mode, and
    `.tile::scatter4` mode. Gather4 and scatter4 require `gmem_coord_tensor`
    so their index tensor layout can drive the returned TMA tensor shape. For
    layout conventions, examples, and restrictions, see [`tma_partition()`](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.tma_partition "cutlass.cute.nvgpu.cpasync.tma_partition").

    Given

    - a GMEM tensor
    - a SMEM layout
    - a CTA-level Tiler
    - (optional) a GMEM index tensor for gather4/scatter4 mode

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    “TMA vector length” to copy tiles of the GMEM tensor to/from an SMEM buffer with the provided
    layout while maintaining consistency with the provided Tiler.

    This function returns two results:

    1. the Copy Atom
    2. a TMA tensor that maps logical coordinates of the GMEM tensor to coordinates consumed by the TMA unit. TMA tensors contain basis stride elements that enable their associated layout to compute coordinates. Like other CuTe tensors, TMA tensors can be partitioned.

    Parameters:
    :   - **op** (*TMAOp*) – The TMA Copy Operation to construct an Atom
        - **gmem\_tensor** (*Tensor*) – The GMEM tensor involved in the Copy
        - **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) – The SMEM layout to construct the Copy Atom, either w/ or w/o the stage mode
        - **cta\_tiler** (*Tiler*) – The CTA Tiler to use
        - **num\_multicast** (*int*) – The multicast factor
        - **internal\_type** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Optional internal data type to use when the tensor data type is not supported by the TMA unit

    Returns:
    :   A TmaInfo containing the Copy Atom, TMA tensor, and SMEM layout

    Return type:
    :   [TmaInfo](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.TmaInfo "cutlass.cute.nvgpu.cpasync.TmaInfo")

cutlass.cute.nvgpu.cpasync.tma\_partition( : *atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *cta\_coord: cutlass.cute.typing.Coord*, : *cta\_layout: cutlass.cute.typing.Layout*, : *smem\_tensor: cutlass.cute.typing.Tensor*, : *gmem\_tensor: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, ...]*, ) → Tuple[cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor] | Tuple[cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor]
:   Tiles the GMEM and SMEM tensors for the provided TMA Copy Atom.

    For standard TMA modes (tiled, im2col, etc.), pass a single GMEM tensor:

    > tAsA, tAgA = tma\_partition(atom, cta\_coord, cta\_layout, sA, gA)

    For gather4 mode, pass a list of `[data_coord_tensor, index_tensor]`:

    > tAsA, tAgA, tAgI = tma\_partition(atom, cta\_coord, cta\_layout, sA, [gA, gI])

    For scatter4 mode, pass the destination coordinate tensor and index tensor
    as `[dst_coord_tensor, index_tensor]` and then pass the returned pair as
    the destination of `cute.copy()`. The same index tensor must be passed
    to [`make_tiled_tma_atom()`](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom "cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom") as `gmem_coord_tensor` so the returned TMA
    tensor shape follows the index tensor shape rather than the destination
    data tensor shape:

    > atom, gB = make\_tiled\_tma\_atom(op, dst, smem\_layout, cta\_tiler, gmem\_coord\_tensor=gI)
    > tBsB, tBgB, tBgI = tma\_partition(atom, cta\_coord, cta\_layout, sB, [gB, gI])
    > cute.copy(atom, tBsB, [tBgB[crd], tBgI[crd]])

    The 2D gather4 TMA atom (`CopyBulkTensor2DGather4G2SOp`) issues
    `cp.async.bulk.tensor.2d.tile::gather4`, which loads four rows from a
    GMEM tile by gathering at four indirectly-addressed indices. Each PTX
    instruction consumes 5 coordinates `{crd0, crd1_i0, crd1_i1, crd1_i2,
    crd1_i3}`: one *contiguous* coordinate (`crd0`) and four *gather*
    coordinates pulled from `gmem_coord_tensor`.

    The `gmem_coord_tensor` is a rank-2 `Int32` tensor whose layout selects
    which mode is the gather mode. One mode is broadcast (stride 0) and the
    other supplies the per-row gather indices:

    For TMA override, pass `(gmem_tensor, residue_tensor)` where the
    gmem\_tensor tracks the gmem address and stride, residue\_tensor is a coordinate
    tensor with negated strides to track the remaining shape to copy.

    - **Column-major data, gather along rows:** `stride=(0, 1)` — mode 0 is
      the broadcast (contiguous) mode, mode 1 carries the gather indices.

      ```python
      # GMEM data is (M, N) col-major; gather along the M dimension.
      gI = cute.make_tensor(
          idx_ptr, cute.make_layout((M, N), stride=(0, 1))
      )
      atom, gA = cpasync.make_tiled_tma_atom(
          cpasync.CopyBulkTensor2DGather4G2SOp(),
          gA, smem_layout, cta_tiler,
          gmem_coord_tensor=gI,
      )
      tAsA, tAgA, tAgI = cpasync.tma_partition(
          atom, 0, cute.make_layout(1), sA, [gA, gI],
      )
      cute.copy(atom, [tAgA[crd], tAgI[crd]], tAsA, tma_bar_ptr=mbar)
      ```
    - **Row-major data, gather along cols:** `stride=(1, 0)` — mode 1 is
      the broadcast (contiguous) mode, mode 0 carries the gather indices.

      ```python
      gI = cute.make_tensor(
          idx_ptr, cute.make_layout((M, N), stride=(1, 0))
      )
      ```

    **Restrictions** (enforced by Python and MLIR verifiers):

    - `gmem_coord_tensor` layout must be 2D.
    - The gather mode size must be `>= 4` (4 row indices per instruction).
    - Exactly one mode of `gmem_coord_tensor` must be a broadcast (stride 0);
      that broadcast mode is *not* the gather dimension.

    Parameters:
    :   - **atom** – The TMA Copy Atom
        - **cta\_coord** – CTA coordinate within the cluster
        - **cta\_layout** – Layout of CTAs in the cluster
        - **smem\_tensor** – The SMEM tensor to partition
        - **gmem\_tensor** – A single GMEM tensor, `[data_coord_tensor, index_tensor]`
          for gather4, or `[dst_coord_tensor, index_tensor]`
          for scatter4

    Returns:
    :   `(smem_tensor, gmem_tensor)` for standard TMA, or
        `(smem_tensor, coord_tensor, index_tensor)` for gather4/scatter4

cutlass.cute.nvgpu.cpasync.create\_tma\_multicast\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *cta\_coord\_vmnk: cutlass.cute.typing.Coord*, : *mcast\_mode: int*, ) → cutlass.cute.typing.Int16
:   Computes a multicast mask for a TMA load Copy.

    Parameters:
    :   - **cta\_layout\_vmnk** (*Layout*) – The VMNK layout of the cluster
        - **cta\_coord\_vmnk** (*Coord*) – The VMNK coordinate of the current CTA
        - **mcast\_mode** (*int*) – The tensor mode in which to multicast

    Returns:
    :   The resulting mask

    Return type:
    :   [Int16](../basic_data_types.md#cutlass.Int16 "cutlass.Int16")

cutlass.cute.nvgpu.cpasync.prefetch\_descriptor(*tma\_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*) → None
:   Prefetches the TMA descriptor associated with the TMA Atom.

cutlass.cute.nvgpu.cpasync.copy\_tensormap( : *tma\_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *tensormap\_ptr: cutlass.cute.typing.Pointer*, ) → None
:   Copies the tensormap held by a TMA Copy Atom to the memory location pointed to by the provided
    pointer.

    Parameters:
    :   - **tma\_atom** ([*CopyAtom*](cute.md#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – The TMA Copy Atom
        - **tensormap\_ptr** ([*Pointer*](../basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – The pointer to the memory location to copy the tensormap to

cutlass.cute.nvgpu.cpasync.update\_tma\_descriptor( : *tma\_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *tma\_desc\_ptr: cutlass.cute.typing.Pointer*, ) → None
:   Updates the TMA descriptor in the memory location pointed to by the provided pointer using
    information from a TMA Copy Atom and the provided GMEM tensor.

    Specifically, the following fields of the TMA descriptor will be updated:

    1. the GMEM tensor base address
    2. the GMEM tensor shape
    3. the GMEM tensor stride

    Other fields of the TMA descriptor are left unchanged.

    Parameters:
    :   - **tma\_atom** ([*CopyAtom*](cute.md#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – The TMA Copy Atom
        - **gmem\_tensor** (*Tensor*) – The GMEM tensor
        - **tensormap\_ptr** ([*Pointer*](../basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – The pointer to the memory location of the descriptor to udpate

cutlass.cute.nvgpu.cpasync.fence\_tma\_desc\_acquire( : *tma\_desc\_ptr: cutlass.cute.typing.Pointer*, ) → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.nvgpu.cpasync.cp\_fence\_tma\_desc\_release( : *tma\_desc\_global\_ptr: cutlass.cute.typing.Pointer*, : *tma\_desc\_shared\_ptr: cutlass.cute.typing.Pointer*, ) → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy).

cutlass.cute.nvgpu.cpasync.fence\_tma\_desc\_release() → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.nvgpu.cpasync.group\_bulk\_copy\_modes( : *src: cutlass.cute.typing.Tensor*, : *dst: cutlass.cute.typing.Tensor*, ) → Tuple[cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor]
:   Copy async bulk need group mode 0, acquiring whole tensor for bulk copy
