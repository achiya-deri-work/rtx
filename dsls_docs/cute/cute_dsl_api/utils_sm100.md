# Blackwell (SM100)

cutlass.utils.sm100.compute\_epilogue\_tile\_size( : *cta\_tile\_m: int*, : *cta\_tile\_n: int*, : *use\_2cta: bool*, : *elem\_width\_d: int*, : *elem\_width\_c: int | None = None*, : *d\_is\_m\_major: bool = True*, : *c\_is\_m\_major: bool = True*, : *tmem\_warp\_shape\_mn: Tuple[int, int] | None = None*, ) → tuple[int, int]
:   Compute epilogue subtile dimensions `(tile_m, tile_n)` (pure Python, no MLIR).

    Used by [`compute_epilogue_tile_shape()`](utils_sm100.md#cutlass.utils.sm100.compute_epilogue_tile_shape "cutlass.utils.sm100.compute_epilogue_tile_shape") and at kernel-discovery time
    for SMEM capacity estimation.
    Background – SM100 epilogue flow
    ———————————
    After the MMA, the accumulator lives in **TMEM** (Tensor Memory, 128
    datapaths x N columns). The epilogue:

    1. Loads a subtile from TMEM into registers (`tcgen05.tmem_load`).
    2. Optionally loads a source tile C from GMEM -> SMEM -> registers.
    3. Applies the fusion (bias, activation, …) in registers.
    4. Stores the result D to SMEM, then to GMEM via TMA.

    The full CTA output (`cta_tile_m x cta_tile_n`) is processed in
    multiple **epilogue iterations**, each covering one subtile of shape
    `(tile_m, tile_n)`. Each subtile is worked on by 4 warps arranged
    in a `(warp_m, warp_n)` grid.

    This function picks `(tile_m, tile_n)` to balance three constraints:

    - **TMEM load** only supports 16 or 32 datapaths per warp, which
      caps `tile_m`.
    - **TMA store alignment** requires a minimum contiguous transaction
      size, which sets `n_min` floors.
    - **SMEM budget** – the epilogue subtile lives in SMEM alongside the
      mainloop’s A/B pipeline buffers. A larger subtile means fewer
      epilogue iterations (good) but steals SMEM from the mainloop,
      reducing pipeline depth (bad). `n_perf` targets the sweet spot
      found by benchmarking.

    ## Algorithm

    1. **Warp grid** `(warp_m, warp_n)`:

       - `(2, 2)` when `use_2cta and cta_tile_m == 64` – each of
         the 2 M-warps gets 32 datapaths.
       - `(4, 1)` otherwise – 4 warps split M, each gets
         `tile_m / 4` datapaths (16 or 32).
    2. **tile\_m** `= min(cta_tile_m, 32 * warp_m)`

       `32` is `dp_full`, the number of TMEM datapaths in one
       subpartition (hardware constant). Capping here ensures each
       warp owns at most 32 datapaths, which is the widest mode the
       `tcgen05.tmem_load` instruction supports.
    3. **n\_perf** – performance target for N:

       - **Without source C** (elementwise-only epilogue): SMEM pressure
         is low because there is no C tile to stage. Target a fixed
         element count per iteration so that each iteration does enough
         work to amortise the epilogue overhead. The constants are:

         ```console
         4096 elements  (general)  -> e.g. tile_m=128, n=32
         8192 elements  (4-bit)    -> e.g. tile_m=128, n=64
         ```

         4-bit elements are half a byte each, so doubling the count
         keeps roughly the same SMEM footprint while cutting epilogue
         iterations in half (experimentally best for 4-bit types).
       - **With source C** (residual-load epilogue): the source tile C
         also occupies SMEM, so the epilogue tile must be smaller to
         leave room for mainloop A/B pipeline stages. Targets are
         chosen by element width and CTA shape to balance SMEM
         partitioning:

         ```console
         32-bit elements: n=16 when M>64 and N<=128, else n=32
         16-bit elements: n=32 when N<=128, else n=64
         <=8-bit elements: n=64
         ```

         Wider elements consume more bytes per element, so N is
         reduced to stay within the SMEM budget. When CTA N is
         large (>128), N is increased because the mainloop tile is
         also large and SMEM is more abundant.

       After choosing, `n_perf` is halved until it evenly divides
       `cta_tile_n` (ensures the CTA output tiles evenly into
       subtiles with no ragged remainder).
    4. **n\_min\_d, n\_min\_c** – hard minimums from TMA store alignment:

       - **M-major** (contiguous dim is M): `8 * warp_n`. N is the
         strided dimension so the minimum is small (8 elements per
         warp is enough for the store to issue).
       - **N-major** (contiguous dim is N): each TMA store transaction
         is 128 bits wide, so the minimum contiguous N per warp is
         `128 / elem_width` elements, times `warp_n` warps.
       - **FP6 special case**: TMA store only supports the SW128B
         swizzle mode for 6-bit types, requiring 128 contiguous
         elements per warp, i.e. `128 * warp_n`.
    5. **tile\_n** `= min(cta_tile_n, max(n_perf, n_min_c, n_min_d))`.

       If the chosen N doesn’t evenly divide `cta_tile_n`, fall back
       to `cta_tile_n` (process the full N in one iteration).

    param cta\_tile\_m:
    :   Per-CTA tile size in M.

    param cta\_tile\_n:
    :   Per-CTA tile size in N.

    param use\_2cta:
    :   True when 2-CTA (2-SM) MMA instructions are used.

    param elem\_width\_d:
    :   Bit-width of output element type D (e.g. 16).

    param elem\_width\_c:
    :   Bit-width of source element type C, or `None`
        if the epilogue has no source (elementwise-only).

    param d\_is\_m\_major:
    :   `True` if D is column-major (M-contiguous).

    param c\_is\_m\_major:
    :   `True` if C is column-major (M-contiguous).

    return:
    :   `(tile_m, tile_n)` – epilogue subtile dimensions.

cutlass.utils.sm100.compute\_acc\_tmem\_cols\_per\_stage( : *cta\_tile\_m: int*, : *cta\_tile\_n: int*, : *use\_2cta: bool*, : *mma\_n: int*, : *transform\_a\_source\_is\_tmem: bool*, ) → int
:   Compute the accumulator TMEM column footprint for one pipeline stage.

    Returns the **raw** layout footprint — the caller must enforce hardware
    allocation constraints (min 32 columns, power-of-2 total) at the final
    `alloc_tmem` call site. See `TmemAllocator.check_valid_num_columns`
    in `cutlass/utils/tmem_allocator.py`.

    **How TMEM packing works**

    TMEM has 128 datapaths (rows). M maps to datapaths, N maps to
    columns. When fewer than 128 DPs are needed, multiple N-values can
    share a column by occupying different DP rows:

    ```console
    NonInterleaved (each N-tile owns its columns):

        columns 0..N-1       columns N..2N-1
        ┌────────────────┐  ┌────────────────┐
    DP  │ ████ tile 0    │  │ ████ tile 1    │  ← 64 DPs used
    0-  │                │  │                │
    127 │ ···· unused    │  │ ···· unused    │  ← 64 DPs wasted
        └────────────────┘  └────────────────┘
        Total: 2N columns

    Interleaved (pairs of N-tiles share columns):

        columns 0..N-1
        ┌────────────────┐
    DP  │ ████ tile 0    │  ← DPs 0-15, 32-47, 64-79, 96-111
    0-  │ ▓▓▓▓ tile 1    │  ← DPs 16-31, 48-63, 80-95, 112-127
    127 │                │
        └────────────────┘
        Total: N columns  (halved)
    ```

    For **1CTA**, the accumulator uses Interleaved when A is from SMEM
    *and* `cta_tile_m` == 64. NonInterleaved is forced when
    `cta_tile_m` == 128 (all datapaths already occupied) or when A is
    from TMEM (each datapath can only access its own row, so A and C
    must share the same M-to-datapath mapping).

    For **2CTA**, each SM has its own 128-DP TMEM and the fragment layout
    is computed for the per-CTA shape (M = `cta_tile_m`, which the
    caller must set to the per-CTA value). Because `cta_tile_m` only
    occupies part of the 128 DPs, the remaining rows can hold additional
    N-values in the same column. The number of N-values that share a
    column is 128 / `cta_tile_m`, so the columns needed are
    `cta_tile_n` / (128 / `cta_tile_m`):

    - `cta_tile_m` = 32 → 128/32 = 4 per column → `cta_tile_n` / 4
    - `cta_tile_m` = 64 → 128/64 = 2 per column → `cta_tile_n` / 2
    - `cta_tile_m` = 128 → 128/128 = 1 (no sharing) → `cta_tile_n`

    Parameters:
    :   - **cta\_tile\_m** – Per-CTA tile size in M dimension (for 2CTA the
          caller divides the full tile M by 2).
        - **cta\_tile\_n** – CTA tile size in N dimension.
        - **use\_2cta** – Whether 2CTA MMA instructions are used.
        - **mma\_n** – MMA atom size in N dimension.
        - **transform\_a\_source\_is\_tmem** – Whether operand A is sourced from
          TMEM (forces NonInterleaved allocation).

    Returns:
    :   TMEM columns per accumulator stage (before HW constraints).

cutlass.utils.sm100.compute\_epilogue\_tile\_shape( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *use\_2cta\_instrs: bool*, : *layout\_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[\_MockObject]*, : *\**, : *layout\_c: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum") | None = None*, : *elem\_ty\_c: Type[\_MockObject] | None = None*, : *tmem\_warp\_shape\_mn: Tuple[int, int] | None = None*, ) → cutlass.cute.typing.Tile
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

cutlass.utils.sm100.get\_smem\_store\_op( : *layout\_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[\_MockObject]*, : *elem\_ty\_acc: Type[\_MockObject]*, : *tiled\_tmem\_load: [TiledCopy](cute.md#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")*, ) → [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")
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

cutlass.utils.sm100.get\_tmem\_load\_op( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *layout\_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[\_MockObject]*, : *elem\_ty\_acc: Type[\_MockObject]*, : *epi\_tile: cutlass.cute.typing.Tile*, : *use\_2cta\_instrs: bool*, : *\**, : *tmem\_warp\_shape\_mn: Tuple[int, int] | None = None*, ) → [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")
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

cutlass.utils.sm100.make\_smem\_layout\_a( : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *a\_dtype: Type[\_MockObject]*, : *num\_stages: int*, : *\**, : *is\_k\_major: bool | None = None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
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

cutlass.utils.sm100.make\_smem\_layout\_b( : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *b\_dtype: Type[\_MockObject]*, : *num\_stages: int*, : *\**, : *is\_k\_major: bool | None = None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
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

cutlass.utils.sm100.make\_smem\_layout\_epi( : *epi\_dtype: Type[\_MockObject]*, : *epi\_layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *epi\_tile: cutlass.cute.typing.Tile*, : *epi\_stage: int*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
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

cutlass.utils.sm100.make\_trivial\_tiled\_mma( : *\*args: Any*, : *\*\*kwargs: Any*, ) → [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")
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

cutlass.utils.sm100.make\_blockscaled\_trivial\_tiled\_mma( : *\*args: Any*, : *\*\*kwargs: Any*, ) → [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")
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

cutlass.utils.sm100.cluster\_shape\_to\_tma\_atom\_A( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
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

cutlass.utils.sm100.cluster\_shape\_to\_tma\_atom\_B( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
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

cutlass.utils.sm100.cluster\_shape\_to\_tma\_atom\_SFB( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
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

cutlass.utils.sm100.get\_permutation\_mnk( : *tile\_shape\_mnk: cutlass.cute.typing.Shape*, : *sf\_vec\_size: int*, : *use\_mxf8f6f4: bool*, ) → Tuple[int, int, int]
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

cutlass.utils.sm100.get\_num\_tmem\_alloc\_cols( : *tmem\_tensors: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor]*, : *rounding: bool = True*, ) → int

cutlass.utils.sm100.thrfrg\_SFA( : *sfa\_tensor: cutlass.cute.typing.Tensor*, : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, ) → cutlass.cute.typing.Tensor
:   Thread-fragment scale factor A tensor for SM120 block-scaled MMA.

    Implements the ThrFrg partitioning for scale factor A according to the
    corresponding C++ code in cutlass/include/cute/atom/mma\_traits\_sm120.hpp:
    SFALayout for SM120 MXF4 16x8x64 uses K=64, SM120 MXF8F6F4 16x8x32 uses
    K=32; the stride pattern `((_8,_0,_1), _16)` is shared.

cutlass.utils.sm100.thrfrg\_SFB( : *sfb\_tensor: cutlass.cute.typing.Tensor*, : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, ) → cutlass.cute.typing.Tensor
:   Thread-fragment scale factor B tensor for SM120 block-scaled MMA.

    Implements the ThrFrg partitioning for scale factor B according to the
    corresponding C++ code in cutlass/include/cute/atom/mma\_traits\_sm120.hpp:
    SFBLayout for SM120 MXF4 16x8x64 uses K=64, SM120 MXF8F6F4 16x8x32 uses
    K=32; the stride pattern `((_0,_1), _8)` is shared.

cutlass.utils.sm100.partition\_fragment\_SFA( : *sfa\_tensor: cutlass.cute.typing.Tensor*, : *thr\_mma: [ThrMma](cute.md#cutlass.cute.ThrMma "cutlass.cute.atom.ThrMma")*, : *tidx: int*, ) → cutlass.cute.typing.Tensor
:   Partition and create a register fragment for scale factor A.

cutlass.utils.sm100.partition\_fragment\_SFB( : *sfb\_tensor: cutlass.cute.typing.Tensor*, : *thr\_mma: [ThrMma](cute.md#cutlass.cute.ThrMma "cutlass.cute.atom.ThrMma")*, : *tidx: int*, ) → cutlass.cute.typing.Tensor
:   Partition and create a register fragment for scale factor B.

cutlass.utils.sm100.get\_layoutSFA\_TV( : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, ) → cutlass.cute.typing.Layout
:   Get the Thread-Value layout for scale factor A.

cutlass.utils.sm100.get\_layoutSFB\_TV( : *tiled\_mma: [TiledMma](cute.md#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, ) → cutlass.cute.typing.Layout
:   Get the Thread-Value layout for scale factor B.
