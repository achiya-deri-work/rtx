# Hopper (SM90)

cutlass.utils.sm90.get\_smem\_store\_op( : *layout\_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[\_MockObject]*, : *elem\_ty\_acc: Type[\_MockObject]*, ) → [CopyAtom](cute.md#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")
:   Selects the largest vectorized smem store atom available subject to constraint of gmem layout.

    ## Parameters:

    layout\_dLayoutEnum
    :   The layout enum of the output tensor D.

    elem\_ty\_dType[Numeric]
    :   The element type for output tensor D.

    elem\_ty\_accType[Numeric]
    :   The element type for accumulator.

    ## Returns:

    Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

cutlass.utils.sm90.make\_smem\_layout\_a( : *a\_layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *a\_dtype: Type[\_MockObject]*, : *num\_stages: int*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
:   This function helps with:

    1. Get the partitioned shape of the A tensor based on the MMA tiler.
    2. Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   - **a\_layout** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum for tensor A
        - **mma\_tiler\_mnk** (*cute.cute.Tile*) – The MMA tile shape
        - **a\_dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for tensor A
        - **num\_stages** (*int*) – The number of pipeline stages for tensor A

    Returns:
    :   SMEM layout for tensor A

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm90.make\_smem\_layout\_b( : *b\_layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *b\_dtype: Type[\_MockObject]*, : *num\_stages: int*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
:   This function helps with:

    1. Get the partitioned shape of the B tensor based on the MMA tiler.
    2. Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   - **b\_layout** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) – The layout enum for tensor B
        - **mma\_tiler\_mnk** (*cute.cute.Tile*) – The MMA tile shape
        - **b\_dtype** (*Type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – The element type for tensor B
        - **num\_stages** (*int*) – The number of pipeline stages for tensor B

    Returns:
    :   SMEM layout for tensor B

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm90.make\_smem\_layout\_epi( : *epi\_dtype: Type[\_MockObject]*, : *epi\_layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *epi\_tile: cutlass.cute.typing.Tile*, : *epi\_stage: int*, : *smem\_trg\_shape: cutlass.cute.typing.Layout | None = None*, : *smem\_order: tuple | None = None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
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
        - **smem\_trg\_shape** (*cute.Layout* *|* *None*) – Target shape for SMEM layout (optional).
        - **smem\_order** (*tuple* *|* *None*) – Order for SMEM layout (optional).

    Returns:
    :   SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm90.compute\_tile\_shape\_or\_override( : *tile\_shape\_mnk: tuple[int, int, int]*, : *element\_type: type[\_MockObject]*, : *is\_cooperative: bool = False*, : *epi\_tile\_override: tuple[int, int] | None = None*, ) → tuple[int, int]
:   Compute the epilogue tile shape or use override if provided.

    Parameters:
    :   - **tile\_shape\_mnk** (*Tuple**[**int**,* *int**,* *int**]*) – CTA tile shape (M,N,K)
        - **element\_type** (*type**[*[*Numeric*](../basic_data_types.md#cutlass.Numeric "cutlass.Numeric")*]*) – Data type of elements
        - **is\_cooperative** (*bool*) – Whether to use cooperative approach
        - **epi\_tile\_override** (*Tuple**[**int**,* *int**] or* *None*) – Optional override for epilogue tile shape

    Returns:
    :   Computed epilogue tile shape

    Return type:
    :   Tuple[int, int]
