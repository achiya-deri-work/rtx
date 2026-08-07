# nvgpu.warpgroup

*class* cutlass.cute.nvgpu.warpgroup.OperandSource(*value*)
:   Bases: `Enum`

    An enumeration for the source memory location of the A input operand of the MMA.

    RMEM *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

    SMEM *= <class 'sphinx.ext.autodoc.mock.\_MockObject'>*

*class* cutlass.cute.nvgpu.warpgroup.Field(*value*)
:   Bases: `Enum`

    An enumeration for the fields of the MMA Atom that can be modified at runtime.

    ACCUMULATE *= 'accum\_c'*

*class* cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](cute_nvgpu_warpgroup.md#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode") | OperandMajorMode*, : *b\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode") | OperandMajorMode*, )
:   Bases: `MmaOp`

    F16/BF16 warpgroup MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
    This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

    **Supported data type combinations:**

    | A Data Type | B Data Type | Acc Type | Mma-K |
    | --- | --- | --- | --- |
    | F16 | F16 | F16, F32 | 16 |
    | BF16 | BF16 | F32 | 16 |

    **Supported architectures:** sm\_90a

    **Constraints:**

    - Mma-M = 64
    - 8 <= Mma-N <= 256, step 8
    - A and B support both K-major and MN-major (transpose) when A is in shared memory (descriptor).
      When A is in registers, only B can be transposed.

    **Execution Model:**

    - WGMMA is asynchronous and collective at warpgroup scope (4 contiguous warps).
      In user code, `cute.gemm(...)` should be issued warpgroup-uniformly.
    - Before issuing `cute.gemm(...)`, call `cute.nvgpu.warpgroup.fence()` to order
      prior register writes to accumulator/A fragments with subsequent WGMMA reads.
    - After issuing `cute.gemm(...)`, call `cute.nvgpu.warpgroup.commit_group()`.
      Use `cute.nvgpu.warpgroup.wait_group(N)` before consuming or reusing accumulator
      values from pending WGMMA groups.

    ```python
    cute.nvgpu.warpgroup.fence()
    cute.gemm(tiled_mma, acc, tCrA[tile_crd], tCrB[tile_crd], acc)
    cute.nvgpu.warpgroup.commit_group()
    cute.nvgpu.warpgroup.wait_group(1)
    # ... pipeline continues ...
    cute.nvgpu.warpgroup.wait_group(0)
    ```

    descriptive\_name*: ClassVar[str]* *= 'warpgroup F16/BF16 MMA Operation'*

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](cute_nvgpu_warpgroup.md#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode") | OperandMajorMode*, : *b\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode") | OperandMajorMode*, ) → None

*class* cutlass.cute.nvgpu.warpgroup.MmaF8Op( : *a\_dtype: Type[cutlass.cute.typing.Numeric]*, : *b\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](cute_nvgpu_warpgroup.md#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode") | OperandMajorMode*, : *b\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode") | OperandMajorMode*, )
:   Bases: `MmaOp`

    F8 warpgroup MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
    This Operation covers the instructions using the `.e4m3` or `.e5m2` qualifiers for the input operands.

    **Supported data type combinations:**

    | A Data Type | B Data Type | Acc Type | Mma-K |
    | --- | --- | --- | --- |
    | E4M3, E5M2 | E4M3, E5M2 | F16, F32 | 32 |

    **Supported architectures:** sm\_90a

    **Constraints:**

    - Mma-M = 64
    - 8 <= Mma-N <= 256, step 8
    - A and B data types are independent (mixed FP8 allowed)
    - Transpose (MN-major) is not supported for A or B. Both operands must be K-major.

    **Execution Model:**

    - WGMMA is asynchronous and collective at warpgroup scope (4 contiguous warps).
      In user code, `cute.gemm(...)` should be issued warpgroup-uniformly.
    - Before issuing `cute.gemm(...)`, call `cute.nvgpu.warpgroup.fence()` to order
      prior register writes to accumulator/A fragments with subsequent WGMMA reads.
    - After issuing `cute.gemm(...)`, call `cute.nvgpu.warpgroup.commit_group()`.
      Use `cute.nvgpu.warpgroup.wait_group(N)` before consuming or reusing accumulator
      values from pending WGMMA groups.

    ```python
    cute.nvgpu.warpgroup.fence()
    cute.gemm(tiled_mma, acc, tCrA[tile_crd], tCrB[tile_crd], acc)
    cute.nvgpu.warpgroup.commit_group()
    cute.nvgpu.warpgroup.wait_group(1)
    # ... pipeline continues ...
    cute.nvgpu.warpgroup.wait_group(0)
    ```

    descriptive\_name*: ClassVar[str]* *= 'warpgroup F8 MMA Operation'*

    \_\_init\_\_( : *a\_dtype: Type[cutlass.cute.typing.Numeric]*, : *b\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](cute_nvgpu_warpgroup.md#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode") | OperandMajorMode*, : *b\_major\_mode: [OperandMajorMode](cute_nvgpu_common.md#cutlass.cute.nvgpu.OperandMajorMode "cutlass.cute.nvgpu.common.OperandMajorMode") | OperandMajorMode*, ) → None

*class* cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind(*value*)
:   Bases: `Enum`

    Enum class for the kinds of SMEM layout atoms for SM90.

    Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can
    be used to construct an SMEM layout using blocked product for operand A or B such that the
    resulting layout is legal for both TMA and UMMA.

    Note that there are other ways of creating legal layouts for operand A and B.

    MN\_INTER *= 1*

    MN\_SW32 *= 2*

    MN\_SW64 *= 3*

    MN\_SW128 *= 4*

    K\_INTER *= 5*

    K\_SW32 *= 6*

    K\_SW64 *= 7*

    K\_SW128 *= 8*

cutlass.cute.nvgpu.warpgroup.make\_smem\_layout\_atom( : *kind: [SmemLayoutAtomKind](cute_nvgpu_warpgroup.md#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "cutlass.cute.nvgpu.warpgroup.mma.SmemLayoutAtomKind")*, : *element\_type: Type[cutlass.cute.typing.Numeric]*, ) → cutlass.cute.typing.ComposedLayout
:   Makes a SMEM layout Atom.

    This function creates a composed layout in unit of elements consistent with the requested layout
    Atom kind and element data type.

    Parameters:
    :   - **kind** ([*SmemLayoutAtomKind*](cute_nvgpu_warpgroup.md#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind")) – The kind of layout Atom
        - **element\_type** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element data type to construct the layout for

    Returns:
    :   The SMEM layout atom

    Return type:
    :   ComposedLayout

cutlass.cute.nvgpu.warpgroup.fence() → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-fence).

cutlass.cute.nvgpu.warpgroup.commit\_group() → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group).

cutlass.cute.nvgpu.warpgroup.wait\_group(*group: Any*) → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group).
