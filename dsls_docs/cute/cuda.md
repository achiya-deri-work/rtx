# CUDA (Jittable)

`cutlass.experimental.cuda` provides CUDA descriptor helpers that can be used
from CUTLASS Python JIT code. The current public surface focuses on TensorMap
creation and metadata for TMA-based kernels.

## TensorMap

*class* cutlass.experimental.cuda.TensorMap( : *value: ir.Value*, : *swizzle: Constexpr[[TensorMapSwizzle](cuda.md#cutlass.experimental.cuda.TensorMapSwizzle "cutlass.experimental.cuda.tensor_map.TensorMapSwizzle")] = TensorMapSwizzle.none*, : *dtype: Constexpr | None = None*, : *tma\_format: Constexpr[[TensorMapDataFormat](cuda.md#cutlass.experimental.cuda.TensorMapDataFormat "cutlass.experimental.cuda.tensor_map.TensorMapDataFormat")] = TensorMapDataFormat.DEFAULT*, : *box\_dims: Constexpr[Tuple[[Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | int, ...]] = ()*, )
:   Abstract TMA descriptor handle reflecting a CUDA tensor map.

    Wraps an MLIR `!cuda.tensor_map` value. Call [`get_ptr()`](cuda.md#cutlass.experimental.cuda.TensorMap.get_ptr "cutlass.experimental.cuda.TensorMap.get_ptr") inside a
    `@cute.kernel` to obtain a pointer suitable for TMA intrinsics. The
    host-side wrapper also retains static metadata chosen at descriptor
    construction time: the logical CUTLASS element `dtype` when known, the
    consumer-facing [`TensorMapDataFormat`](cuda.md#cutlass.experimental.cuda.TensorMapDataFormat "cutlass.experimental.cuda.TensorMapDataFormat"), the TMA-order `box_dims`,
    and the [`TensorMapSwizzle`](cuda.md#cutlass.experimental.cuda.TensorMapSwizzle "cutlass.experimental.cuda.TensorMapSwizzle") mode. For descriptors built from a
    `cute.Tensor`, `box_dims` are stored after the builder’s TMA-order
    reordering; use the stored metadata inside kernels rather than passing a
    second copy of the box or byte count.

    The shared-memory storage dtype, shared storage byte count, and TMA global
    transaction byte count for one TMA box are derived from that metadata.
    Derive downstream kernel settings from the same source of truth:
    `box_volume` for logical element count, `shared_storage_bytes` for SMEM
    staging size, and [`global_tx_bytes()`](cuda.md#cutlass.experimental.cuda.TensorMap.global_tx_bytes "cutlass.experimental.cuda.TensorMap.global_tx_bytes") for
    `mbarrier_arrive_expect_tx`. This avoids drift between the TensorMap
    descriptor and separately passed tile constants while the raw CUDA encoding
    stays internal to descriptor construction.

    Use `GridConstant[TensorMap]` for kernel parameters that carry TMA
    descriptors. This marks the argument as `__grid_constant__` so that the
    descriptor lives in constant memory:

    ```python
    @cute.kernel
    def kernel(desc: GridConstant[TensorMap], smem: cute.Tensor, ...):
        mbar_ptr = ...
        nvvm.cp_async_bulk_tensor_shared_cta_global(
            smem, desc.get_ptr(), (coord_k, coord_m), mbar_ptr
        )

    @cute.jit
    def host(a: cute.Tensor, ...):
        desc = create_tensor_map_tiled_from_view(
            a, box_dims=(128, 64), swizzle=TensorMapSwizzle.s128b
        )
        kernel(desc, ...).launch(...)
    ```

    *property* element\_type*: Type[[Numeric](basic_data_types.md#cutlass.Numeric "cutlass.Numeric")] | None*
    :   Alias for `dtype`, matching `cute.Tensor.element_type`.

    *property* box\_volume*: [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | int*
    :   Number of logical TMA elements in one descriptor box.

        The product is computed from the descriptor’s stored TMA-order
        `box_dims`. Use this instead of passing duplicate physical box
        dimensions when sizing per-box loops or logical staging arrays.

    *property* shared\_storage\_dtype*: Type[[Numeric](basic_data_types.md#cutlass.Numeric "cutlass.Numeric")] | None*
    :   Dtype that sizes the shared-memory side of one TMA box.

    *property* shared\_storage\_bytes*: [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | int*
    :   Shared-memory bytes needed for one descriptor box.

        Use this value when allocating byte-addressed SMEM staging for a TMA
        box. It can differ from [`global_tx_bytes()`](cuda.md#cutlass.experimental.cuda.TensorMap.global_tx_bytes "cutlass.experimental.cuda.TensorMap.global_tx_bytes") for formats whose SMEM
        representation expands or pads the global representation.

        Raises:
        :   **ValueError** – If the TensorMap does not carry enough metadata to
            derive a shared-memory storage dtype.

    global\_tx\_bytes() → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | int
    :   Global-memory bytes completed by one TMA copy of this descriptor box.

        This value is the per-copy byte count to pass to
        `mbarrier_arrive_expect_tx` for one TMA completion. It intentionally
        does not account for multicast fanout or multiple TMA producers sharing
        one mbarrier; callers should sum those completions explicitly.

        Raises:
        :   **ValueError** – If the TensorMap does not carry enough metadata to
            derive a global transaction dtype.

    get\_ptr() → [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")
    :   Return a pointer to the tensor map.

        Returns:
        :   Pointer to the tensor map.

        Return type:
        :   [`cutlass.Pointer`](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")

    \_\_init\_\_( : *value: ir.Value*, : *swizzle: Constexpr[[TensorMapSwizzle](cuda.md#cutlass.experimental.cuda.TensorMapSwizzle "cutlass.experimental.cuda.tensor_map.TensorMapSwizzle")] = TensorMapSwizzle.none*, : *dtype: Constexpr | None = None*, : *tma\_format: Constexpr[[TensorMapDataFormat](cuda.md#cutlass.experimental.cuda.TensorMapDataFormat "cutlass.experimental.cuda.tensor_map.TensorMapDataFormat")] = TensorMapDataFormat.DEFAULT*, : *box\_dims: Constexpr[Tuple[[Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | int, ...]] = ()*, ) → None

## Descriptor Builders

cutlass.experimental.cuda.create\_tensor\_map\_tiled( : *global\_address: [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | int*, : *dtype: Type[[Numeric](basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, : *global\_dims: Sequence[[Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | int]*, : *global\_strides: Sequence[[Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | int]*, : *box\_dims: Sequence[[Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | int]*, : *\**, : *traversal\_strides: Sequence[[Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | int] | None = None*, : *interleave: [TensorMapInterleave](cuda.md#cutlass.experimental.cuda.TensorMapInterleave "cutlass.experimental.cuda.tensor_map.TensorMapInterleave") | None = None*, : *swizzle: [TensorMapSwizzle](cuda.md#cutlass.experimental.cuda.TensorMapSwizzle "cutlass.experimental.cuda.tensor_map.TensorMapSwizzle") | None = None*, : *l2\_promotion: [TensorMapL2Promotion](cuda.md#cutlass.experimental.cuda.TensorMapL2Promotion "cutlass.experimental.cuda.tensor_map.TensorMapL2Promotion") | None = None*, : *oob\_fill: [TensorMapFloatOOBFill](cuda.md#cutlass.experimental.cuda.TensorMapFloatOOBFill "cutlass.experimental.cuda.tensor_map.TensorMapFloatOOBFill") | None = None*, : *tma\_format: [TensorMapDataFormat](cuda.md#cutlass.experimental.cuda.TensorMapDataFormat "cutlass.experimental.cuda.tensor_map.TensorMapDataFormat") | TensorMapDataType | None = None*, ) → [TensorMap](cuda.md#cutlass.experimental.cuda.TensorMap "cutlass.experimental.cuda.tensor_map.TensorMap")
:   Build a tiled TMA descriptor from explicit parameters.

    TMA uses **column-major** dimension ordering: `global_dims[0]` is the
    fastest-varying (innermost, contiguous) axis. For a C-order (row-major)
    `(M, K)` fp16 array whose rows are contiguous, pass `global_dims=[K, M]`
    and `global_strides=[K * 2 // 16]` (row stride in 16-byte units).

    Pass the logical CUTLASS `Numeric` type explicitly via *dtype*.
    The descriptor builder maps that logical dtype to both a consumer-facing
    [`TensorMapDataFormat`](cuda.md#cutlass.experimental.cuda.TensorMapDataFormat "cutlass.experimental.cuda.TensorMapDataFormat") and the required CUDA-driver encoding
    internally when the mapping is unambiguous.

    Use `TensorMapDataFormat` when you need to describe the *transfer
    container layout* in the same terms a TMA consumer sees, or when a tensor
    reduce path needs an FTZ-specific data-type variant:

    - `BYTE` → byte-addressed payload (the consumer interprets the bytes)
    - `F32_FTZ` → `Float32` tensor-map format with FTZ semantics for
      tensor reduce operations
    - `TF32_FTZ` → `TFloat32` tensor-map format with FTZ semantics for
      tensor reduce operations
    - `B4X16` → PTX `.b4x16` / CUDA packed-FP4 8-byte layout
    - `B4X16_P64` → PTX `.b4x16_p64` / CUDA packed-FP4 16-byte layout
    - `B6X16_P32` → PTX `.b6x16_p32` for loads and `.b6p2x16` for
      stores, both backed by CUDA’s packed-FP6 descriptor encoding

    Example — 2-D fp16 `(M=1024, K=64)` row-major tensor, 128×64 tile,
    128-byte swizzle:

    ```python
    desc = create_tensor_map_tiled(
        global_address=a.iterator.toint(),   # Int64 device pointer
        dtype=cutlass.Float16,
        global_dims=[64, 1024],              # [K, M] — K is innermost
        global_strides=[64 * 2 // 16],       # row stride in 16-B units: 8
        box_dims=[64, 128],                  # box: 64 in K, 128 in M
        swizzle=TensorMapSwizzle.s128b,
    )
    ```

    Example — FP8 tile. No explicit `TensorMapDataFormat.BYTE` is needed;
    the logical FP8 dtype is retained while TMA uses byte storage:

    ```python
    desc = create_tensor_map_tiled(
        global_address=a_fp8.iterator.toint(),
        dtype=cutlass.Float8E4M3FN,
        global_dims=[128, 128],
        global_strides=[128 * 1 // 16],
        box_dims=[128, 128],
        swizzle=TensorMapSwizzle.s128b,
    )
    ```

    Example — packed FP4 tile. `cutlass.Float4E2M1FNx2` names the packed
    global-memory storage type, while `B4X16_P64` selects the logical scalar
    FP4 lane format consumed by TMA. Dimensions, strides, and boxes are still
    expressed in logical scalar FP4 lanes. Per PTX, `B4X16_P64` requires
    `Box-Size[0] == 64B` (128 FP4 lanes); larger K tiles must be issued as
    multiple TMA copies with different coordinates.

    ```python
    desc = create_tensor_map_tiled(
        global_address=b_packed.iterator.toint(),
        dtype=cutlass.Float4E2M1FNx2,
        tma_format=TensorMapDataFormat.B4X16_P64,
        global_dims=[k, n],
        global_strides=[k * 4 // 128],
        box_dims=[128, n_tile],
        swizzle=TensorMapSwizzle.s128b,
    )
    ```

    Parameters:
    :   - **global\_address** ([*Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* *int*) – Device pointer (as `Int64` or integer) to the
          first element of the global tensor.
        - **dtype** (*CUTLASS dtype*) – Logical CUTLASS element dtype such as `cutlass.Float16` or
          `cutlass.Float8E4M3FN`. The descriptor retains this logical dtype even
          when the backing TMA encoding uses a different storage container.
        - **global\_dims** (*Sequence**[*[*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* *int**]*) –

          Shape of the global tensor in TMA (column-major) order:
          `global_dims[0]` is the contiguous dimension.
          Rank must be between 1 and 5.

          Sub-byte / packed dtypes — this raw-args helper takes
          `global_dims` verbatim in the units expected by the selected
          tensor-map format. For `BYTE` over `Float4E2M1FNx2`, one element is
          one packed fp4x2 storage byte. For `B4X16` / `B4X16_P64`, one
          element is one scalar FP4 lane.
        - **global\_strides** (*Sequence**[*[*Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* *int**]*) –

          Inter-dimension strides in **16-byte units**,
          length `rank - 1` (the innermost stride is implicit/unused).
          For a C-order tensor with element size `E` bytes and inner-dim size
          `D`, the stride value is `D * E // 16`.

          Sub-byte / packed dtypes — this raw-args helper takes the value
          verbatim. For `BYTE` over `Float4E2M1FNx2` the stride is in fp4x2
          storage-byte units, so an FP4 row of `K` scalar values uses
          `(K // 2) * 8 // 128` (= `K // 32`). For `B4X16` /
          `B4X16_P64` the same logical row uses scalar-lane units:
          `K * 4 // 128`.
        - **box\_dims** (*Sequence**[*[*Int8*](basic_data_types.md#cutlass.Int8 "cutlass.Int8") *or* *int**]*) – Tile (box) dimensions for TMA, one per dimension,
          in the same column-major order as *global\_dims*.
        - **traversal\_strides** (*Sequence**[*[*Int8*](basic_data_types.md#cutlass.Int8 "cutlass.Int8") *or* *int**]**,* *optional*) – Element strides within each box dimension,
          defaults to None (all-ones).
        - **interleave** ([*TensorMapInterleave*](cuda.md#cutlass.experimental.cuda.TensorMapInterleave "cutlass.experimental.cuda.TensorMapInterleave")*,* *optional*) – Interleave mode, defaults to None (`none`).
        - **swizzle** ([*TensorMapSwizzle*](cuda.md#cutlass.experimental.cuda.TensorMapSwizzle "cutlass.experimental.cuda.TensorMapSwizzle") *or* *compatible swizzle descriptor**,* *optional*) – Shared-memory swizzle mode, defaults to None (`none`).
          Accepts [`TensorMapSwizzle`](cuda.md#cutlass.experimental.cuda.TensorMapSwizzle "cutlass.experimental.cuda.TensorMapSwizzle") or a canonical swizzle descriptor that
          can be converted to a tensor-map encoding.
          For ordinary element formats, `s128b` requires
          `box_dims[0] * sizeof(elem) == 128`. Padded sub-byte formats have
          stricter PTX requirements validated at construction: `B4X16_P64`
          requires `Box-Size[0] == 64B` and `B6X16_P32` requires
          `Box-Size[0] == 96B`.
        - **l2\_promotion** ([*TensorMapL2Promotion*](cuda.md#cutlass.experimental.cuda.TensorMapL2Promotion "cutlass.experimental.cuda.TensorMapL2Promotion")*,* *optional*) – L2 promotion hint, defaults to None (`none`).
        - **oob\_fill** ([*TensorMapFloatOOBFill*](cuda.md#cutlass.experimental.cuda.TensorMapFloatOOBFill "cutlass.experimental.cuda.TensorMapFloatOOBFill")*,* *optional*) – Out-of-bounds fill mode, defaults to None (`none`).
        - **tma\_format** ([*TensorMapDataFormat*](cuda.md#cutlass.experimental.cuda.TensorMapDataFormat "cutlass.experimental.cuda.TensorMapDataFormat")*,* *optional*) – Optional consumer-facing transfer-layout override or
          FTZ variant. Leave as None to derive the default layout from *dtype*.
          Legacy `TensorMapDataType` values are accepted for compatibility and
          are converted to the corresponding public `TensorMapDataFormat`.

    Raises:
    :   **ValueError** – If tensor rank is not between 1 and 5, or if
        *box\_dims*, *global\_strides*, or *traversal\_strides* have
        inconsistent lengths.

    Returns:
    :   A [`TensorMap`](cuda.md#cutlass.experimental.cuda.TensorMap "cutlass.experimental.cuda.TensorMap") wrapping the created `!cuda.tensor_map` and
        retaining the selected static *dtype*, *tma\_format*, *box\_dims*, and
        *swizzle* metadata on the Python object. The shared-memory storage dtype
        and per-box byte count are derived from those fields.

    Return type:
    :   [TensorMap](cuda.md#cutlass.experimental.cuda.TensorMap "cutlass.experimental.cuda.TensorMap")

cutlass.experimental.cuda.create\_tensor\_map\_tiled\_from\_view( : *tensor: cutlass.cute.typing.Tensor*, : *box\_dims: Tuple[[Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | int, ...]*, : *\**, : *stride\_order: Tuple[int, ...] | None = None*, : *interleave: [TensorMapInterleave](cuda.md#cutlass.experimental.cuda.TensorMapInterleave "cutlass.experimental.cuda.tensor_map.TensorMapInterleave") | None = None*, : *swizzle: [TensorMapSwizzle](cuda.md#cutlass.experimental.cuda.TensorMapSwizzle "cutlass.experimental.cuda.tensor_map.TensorMapSwizzle") | None = None*, : *l2\_promotion: [TensorMapL2Promotion](cuda.md#cutlass.experimental.cuda.TensorMapL2Promotion "cutlass.experimental.cuda.tensor_map.TensorMapL2Promotion") | None = None*, : *oob\_fill: [TensorMapFloatOOBFill](cuda.md#cutlass.experimental.cuda.TensorMapFloatOOBFill "cutlass.experimental.cuda.tensor_map.TensorMapFloatOOBFill") | None = None*, : *dtype: Type[[Numeric](basic_data_types.md#cutlass.Numeric "cutlass.Numeric")] | None = None*, : *tma\_format: [TensorMapDataFormat](cuda.md#cutlass.experimental.cuda.TensorMapDataFormat "cutlass.experimental.cuda.tensor_map.TensorMapDataFormat") | TensorMapDataType | None = None*, ) → [TensorMap](cuda.md#cutlass.experimental.cuda.TensorMap "cutlass.experimental.cuda.tensor_map.TensorMap")
:   Build a tiled TMA descriptor from a `cute.Tensor` or `cutlass.Array` view.

    Convenience wrapper around [`create_tensor_map_tiled()`](cuda.md#cutlass.experimental.cuda.create_tensor_map_tiled "cutlass.experimental.cuda.create_tensor_map_tiled") that
    auto-detects the global address, data type, dimensions, and strides
    from *tensor*, which may be either a layout-carrying `cute.Tensor` or a
    bare-metal `cutlass.Array` view over the same memory.

    Prefer leaving *dtype* and *tma\_format* as `None` so the descriptor uses
    the tensor’s element type and its default transfer layout. If a logical
    dtype override is required, pass a CUTLASS `Numeric` type via *dtype*.
    Use `TensorMapDataFormat` when the descriptor must express a
    consumer-facing packed layout, for example `B4X16_P64` to request PTX
    `.b4x16_p64` for FP4 TMA, or when a tensor reduce path requires
    `F32_FTZ` / `TF32_FTZ`.

    *box\_dims* must be given in the **tensor’s original mode order**, not
    in TMA column-major order. For a row-major `(M, K)` tensor where K
    is contiguous, pass `box_dims=(M_tile, K_tile)`.

    When *stride\_order* is provided, the tensor map dimensions are builtin from
    the tensor in the order specified by *stride\_order*, otherwise the function
    will compute the dimensions in the **stride ascending order** from the
    tensor’s strides. TMA coordinates later used in the kernel must follow the
    same order. For example, without *stride\_order*, for a row-major `(B, S, H, D)`
    tensor, the tensor map dimensions used are `(D, H, S, B)` and the TMA
    coordinates should be `(d_off, head, seq, batch)`.

    Example — fp16 `A` matrix `(M, K)` row-major, 128-row × 64-col tile,
    128-byte swizzle (requires `K_tile * 2 == 128`, i.e. `K_tile = 64`):

    ```python
    # a.shape = (M, K), a is row-major (K is the contiguous axis)
    desc_a = create_tensor_map_tiled_from_view(
        a,
        box_dims=(128, 64),              # (M_tile, K_tile) — tensor order
        swizzle=TensorMapSwizzle.s128b,  # needs box_dims[K] * 2 == 128
    )
    ```

    Example — fp16 `B` matrix `(N, K)` row-major, 128×64 tile:

    ```python
    desc_b = create_tensor_map_tiled_from_view(
        b,
        box_dims=(128, 64),              # (N_tile, K_tile)
        swizzle=TensorMapSwizzle.s128b,
    )
    ```

    Swizzle constraints (fp16, 2 bytes/elem):

    - `s128b`: contiguous box dimension must be exactly 64 elements (128 B)
    - `s64b`: contiguous box dimension must be exactly 32 elements (64 B)
    - `s32b`: contiguous box dimension must be exactly 16 elements (32 B)
    - `none`: no constraint; layout in SMEM is purely linear

    **Sub-byte dtype shape / stride convention** — the input tensor’s shape,
    *box\_dims*, and TMA coordinates are in logical element units. For default
    packed storage formats such as `Float4E2M1FNx2` with `BYTE` TMA format,
    one logical element is one packed fp4x2 storage byte. When explicitly
    selecting scalar-lane formats such as `B4X16` or `B4X16_P64` over a
    packed tensor, one logical element is one scalar FP4 lane; the helper keeps
    the packed pointer dtype but converts tensor strides using the scalar lane
    bit width. `B4X16_P64` descriptors must still satisfy the PTX padded
    sub-byte restrictions, including `Box-Size[0] == 64B`.

    Parameters:
    :   - **tensor** (*cute.Tensor* *or* *cutlass.Array*) – A `cute.Tensor` with a flattened (depth-1) layout, or a
          `cutlass.Array` view over the same memory. Both expose the shape /
          stride / element-type / base-pointer facts the TMA builder needs.
        - **box\_dims** (*tuple**[*[*Int8*](basic_data_types.md#cutlass.Int8 "cutlass.Int8") *or* *int**,* *...**]*) – Tile dimensions, one per tensor mode, in **tensor mode
          order** (not TMA column-major order).
        - **stride\_order** (*tuple**[**int**,* *...**]**,* *optional*) – Explicit dimension order from innermost to
          outermost. When provided the automatic stride sort is skipped.
          E.g. `(0, 1, 2, 3)` means mode 0 is innermost.
        - **interleave** ([*TensorMapInterleave*](cuda.md#cutlass.experimental.cuda.TensorMapInterleave "cutlass.experimental.cuda.TensorMapInterleave")*,* *optional*) – Interleave mode, defaults to None (`none`).
        - **swizzle** ([*TensorMapSwizzle*](cuda.md#cutlass.experimental.cuda.TensorMapSwizzle "cutlass.experimental.cuda.TensorMapSwizzle")*,* *optional*) – Shared-memory swizzle mode, defaults to None (`none`).
        - **l2\_promotion** ([*TensorMapL2Promotion*](cuda.md#cutlass.experimental.cuda.TensorMapL2Promotion "cutlass.experimental.cuda.TensorMapL2Promotion")*,* *optional*) – L2 promotion hint, defaults to None (`none`).
        - **oob\_fill** ([*TensorMapFloatOOBFill*](cuda.md#cutlass.experimental.cuda.TensorMapFloatOOBFill "cutlass.experimental.cuda.TensorMapFloatOOBFill")*,* *optional*) – Out-of-bounds fill mode, defaults to None (`none`).
        - **dtype** (*CUTLASS dtype**,* *optional*) – Optional logical element dtype override; defaults to None
          (inferred from the tensor’s element type). Use this when a runtime
          tensor surfaces as byte storage but should be encoded as a narrower
          logical format, for example `Float4E2M1FNx2` plus `B4X16_P64`.
        - **tma\_format** ([*TensorMapDataFormat*](cuda.md#cutlass.experimental.cuda.TensorMapDataFormat "cutlass.experimental.cuda.TensorMapDataFormat")*,* *optional*) – Optional consumer-facing packed-layout or FTZ override.
          Scalar FP4 inference defaults to
          `TensorMapDataFormat.B4X16_P64`. Packed `Float4E2M1FNx2`
          inference defaults to `TensorMapDataFormat.BYTE` because one
          tensor element is one packed storage byte. Pass
          `TensorMapDataFormat.B4X16` or
          `TensorMapDataFormat.B4X16_P64` explicitly for unpacking-style
          FP4 tensor maps over packed inputs.

    Raises:
    :   **ValueError** – If the tensor layout is not flattened (depth > 1)
        or has no leading (stride-1) dimension.

    Returns:
    :   A [`TensorMap`](cuda.md#cutlass.experimental.cuda.TensorMap "cutlass.experimental.cuda.TensorMap") wrapping the created `!cuda.tensor_map` and
        retaining the selected static *dtype*, *tma\_format*, reordered
        *box\_dims*, and *swizzle* metadata on the Python object. The
        shared-memory storage dtype and per-box byte count are derived from
        those fields.

    Return type:
    :   [TensorMap](cuda.md#cutlass.experimental.cuda.TensorMap "cutlass.experimental.cuda.TensorMap")

## TensorMap Enums

*class* cutlass.experimental.cuda.TensorMapDataFormat(*value*)
:   Consumer-facing tensor-map data format.

    Most entries describe the transfer/container layout that downstream
    consumers such as `ldmatrix` or `tcgen05.cp` care about. The FTZ
    entries select tensor-map data-type variants used by tensor reduce
    operations. Exact CUDA driver encodings are derived internally from the
    logical dtype plus this format when needed.

    `DEFAULT` is the ordinary resolved format for dtypes that do not require
    a special transfer layout.

*class* cutlass.experimental.cuda.TensorMapFloatOOBFill(*value*)
:   Enumerated type describing tensor map out-of-bound fill modes

*class* cutlass.experimental.cuda.TensorMapInterleave(*value*)
:   Enumerated type describing tensor map interleave modes

*class* cutlass.experimental.cuda.TensorMapL2Promotion(*value*)
:   Enumerated type describing tensor map L2 promotion sizes

*class* cutlass.experimental.cuda.TensorMapSwizzle(*value*)
:   Swizzle pattern applied by the TMA hardware during `cp.async.bulk` transfers.

    Passed as the `swizzle` argument to the TensorMap creation helpers. The hardware
    automatically applies the XOR permutation to the SMEM physical addresses as it
    writes each element — no manual address calculation is needed on the TMA path.

    | Member | Value | Effect |
    | --- | --- | --- |
    | `none` | 0 | No swizzle — linear SMEM layout |
    | `s32b` | 1 | 32-byte XOR swizzle |
    | `s64b` | 2 | 64-byte XOR swizzle |
    | `s128b` | 3 | 128-byte XOR swizzle — **required** for `tcgen05.mma kind::f16` on SM100 |
    | `s128b_atom_32b` | 4 | 128B XOR with 32B atomic sub-partition |
    | `s128b_atom_32b_flip_8b` | 5 | 128B XOR + 32B atom + 8B flip |
    | `s128b_atom_64b` | 6 | 128B XOR with 64B atomic sub-partition |

    Generic swizzle descriptor classes with a `from_name()` constructor can
    be used as conversion targets for the common `none` / `s32b` /
    `s64b` / `s128b` presets.

    to(*target\_type: type[object]*) → object
    :   Convert this TensorMap swizzle to another swizzle representation.

        Atom-specific variants (`s128b_atom_32b` etc.) have no
        `Swizzle` counterpart, so converting them via the
        `Swizzle.from_name` round-trip would raise. `s128b_atom_32b`
        has a direct `Tcgen05SmemSwizzle` mapping
        (`SWIZZLE_128B_ATOM_32B`) which we hand off here without going
        through `Swizzle`.
