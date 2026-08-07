# Primitives

NVVM wrapper namespace — hand-maintained wrappers over raw MLIR NVVM dialect ops.

## Purpose

This module gives users a single `nvvm.*` namespace for all
NVVM-level GPU operations (barriers, TMA copies, tcgen05 tensor-core
ops, PTX special-register reads, etc.) without requiring them to deal
with raw MLIR ceremony.

## How ops are exposed

There are three categories of entries in this module:

1. **Wrapped ops** (`@dsl_user_op` functions):
   A wrapper exists when it adds genuine value over the raw MLIR op:

   - *Return-type hiding* — the raw MLIR op requires the caller to
     pass the result type as the first positional argument (e.g.
     `T.i32()`). The wrapper inserts it automatically and wraps the
     return value in a typed wrapper (`Int32`, `Boolean`, …).
     Examples: `shfl_sync`.
   - *Python-to-DSL type coercion* — the raw MLIR op expects every
     operand to be an `ir.Value`. The wrapper accepts plain Python
     `int` / `bool` and converts them to the correct DSL type
     (`Int32`, `Int64`, `Boolean`, …) so the proxy can turn them
     into `ir.Value`. The proxy alone cannot do this because it does
     not know *which* MLIR type a Python literal should become.
     Because MLIR integers are signless, coerced parameters also accept
     the unsigned counterpart (e.g. `count: int | Int32 | Uint32`).
     The coercion always uses the signed type internally — `Int32`
     and `Uint32` both produce `i32`.
     Examples: `mbarrier_init`, `tcgen05_alloc`, `tcgen05_mma`.

   Every wrapper maps 1:1 to a single MLIR NVVM dialect op with the
   same argument order. Higher-level convenience ops (e.g. computing
   derived parameters or specialising generic ops) belong in the
   higher-level namespace, not here.
2. **Direct aliases** (plain attribute assignments):
   When a raw NVVM op needs no coercion, no return-type hiding, and
   no default parameters, it is re-exported as-is so that it still
   appears in the `nvvm.*` namespace. `fence_mbarrier_init` has a
   thin wrapper below for documentation; other fence aliases are bare.
3. **Auto-converting proxy** (`nvvm.dialect`):
   For any NVVM op that is *not* listed in this module at all, the
   proxy can be used directly. It auto-converts any argument that has
   an `.ir_value()` method before forwarding to the raw dialect:

   ```console
   nvvm.dialect.some_unlisted_nvvm_op(T.i32(), my_int32_value, ...)
   ```

   The proxy does **not** handle `int -> Int32` coercion (it cannot
   guess the target MLIR type), so callers must wrap Python literals
   themselves when using it.

cutlass.experimental.primitives.nvvm\_wrapper.add\_packed\_f32x2( : *src\_a: tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *src\_b: tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *\**, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *ftz: bool | None = None*, ) → tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   Wrapper over `nvvm.add_packed_f32x2`.

    Accepts a 2-tuple of f32 scalars or a `Vector` for each operand and
    returns a tuple when called with tuples, else a `Vector`

cutlass.experimental.primitives.nvvm\_wrapper.atomicrmw( : *op: ~cutlass.experimental.primitives.nvvm\_wrapper.AtomicOp*, : *ptr: ~cutlass.Array | ~cutlass.Pointer*, : *a: int | float | ~cutlass.Int32 | ~cutlass.Uint32 | ~cutlass.Int64 | ~cutlass.Uint64 | ~cutlass.Float32 | ~cutlass.Float64*, : *\**, : *b: int | float | ~cutlass.Int32 | ~cutlass.Uint32 | ~cutlass.Int64 | ~cutlass.Uint64 | ~cutlass.Float32 | ~cutlass.Float64 | None = None*, : *mem\_order: ~cutlass.experimental.primitives.nvvm\_wrapper.MemOrder | None = None*, : *syncscope: ~cutlass.base\_dsl.array.MemScope | None = None*, : *space: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject | None = None*, : *results: list | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32") | [Float64](basic_data_types.md#cutlass.Float64 "cutlass.Float64")
:   Atomic read-modify-write on a memory location.

    Emits `atom.{op}.{mem_order}.{scope}`. The
    op is performed atomically on `*ptr`: the prior value is
    returned, and the new value (a function of the old value, `a`,
    and optionally `b`) is written back. No data race is observable
    by other threads in `syncscope`.

    Parameters:
    :   - **op** – One of [`AtomicOp`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.AtomicOp "cutlass.experimental.primitives.nvvm_wrapper.AtomicOp"). For `"add"` / `"min"` /
          `"max"` the wrapper picks the dialect-level FADD / UMIN / UMAX
          variant from the operand dtype: float operands pick FADD;
          unsigned-integer operands pick UMIN / UMAX; signed-integer
          operands pick the signed variant.
        - **ptr** – Pointer/Array to the target memory cell.
        - **a** – First operand — for `"cas"` this is the *expected* old
          value; for everything else the value combined with `*ptr`.
        - **b** – Second operand — only used for `"cas"` (the *new* value).
        - **mem\_order** – Memory ordering — `"relaxed"` / `"acquire"` /
          `"release"` / `"acq_rel"`.
        - **syncscope** – Scope across which `mem_order` is enforced —
          `"cta"` / `"cluster"` / `"gpu"` / `"sys"`.
        - **results** – Optional preallocated result list (advanced).

    Returns:
    :   The old value at `*ptr` before the op was applied.

    Raises:
    :   **ValueError** – if *op* is not a valid [`AtomicOp`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.AtomicOp "cutlass.experimental.primitives.nvvm_wrapper.AtomicOp"), or if
        `op="cas"` is used without the second operand *b* (the new value).

cutlass.experimental.primitives.nvvm\_wrapper.bar\_warp\_sync(*mask: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*) → None
:   Rendezvous all lanes named in *mask* at a warp-level barrier.

    Maps to PTX `bar.warp.sync membermask`. Every lane whose bit is set in
    *mask* must execute this call before any of them may proceed. The barrier
    also acts as an acquire-release memory fence: stores to shared or global
    memory issued by any lane in *mask* before the call are visible to all other
    lanes in *mask* after it, and loads issued after the call observe those
    stores. Use `cute.arch.FULL_MASK` (`0xFFFFFFFF`) to rendezvous all 32 lanes;
    pass a narrower mask to synchronize only a known active subset.

    This is the warp-level equivalent of `__syncwarp(mask)` in CUDA C++.

    **Constraints:**

    - Every lane named in *mask* must reach `bar_warp_sync` with the
      **same** *mask* value. If any named lane diverges (e.g. is inside a
      branch only some lanes take), the remaining lanes stall indefinitely.
    - Do not call inside a branch unless **all** lanes in *mask* are guaranteed
      to enter that branch.

    ```python
    # All 32 lanes rendezvous in a uniform region (SMEM read-after-write)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    # Warp-specialization: all lanes converge before diverging by role.
    # bar_warp_sync guarantees every lane sees any prior register/SMEM
    # writes (e.g. from nvvm.setmaxregister) before the if-branch.
    warp = cute.arch.warp_idx()
    is_tma_warp = warp == cutlass.Int32(0)
    nvvm.setmaxregister(40, "decrease")
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)   # all lanes rendezvous here
    if is_tma_warp:
        ...  # TMA producer path
    else:
        ...  # compute consumer path

    # Partial mask — only lanes 0–15 synchronize (must all be active)
    nvvm.bar_warp_sync(0x0000FFFF)
    ```

    Parameters:
    :   **mask** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – 32-bit member mask; bit *i* set means lane *i* participates.
        All participating lanes must execute this call with the **same** *mask*
        value or the remaining lanes stall indefinitely.
        Pass `cute.arch.FULL_MASK` (`0xFFFFFFFF`) for all 32 lanes.

    Raises:
    :   **ValueError** – if a static `int` *mask* does not fit in 32 bits
        (outside `[0, 0xFFFFFFFF]`). Runtime `Int32` / `Uint32` values
        pass through unchecked.

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cluster\_arrive() → None
:   Register an acquire-release arrival on the cluster-wide barrier.

    Emits `barrier.cluster.arrive` (non-aligned, the ordered `.release`
    form): each CTA registers its arrival, and the arrival also orders the
    issuing CTA’s prior memory writes so they become visible to every other
    CTA once it returns from the paired wait. This is the ordered
    counterpart of the relaxed [`barrier_cluster_arrive_relaxed()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_relaxed "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_relaxed") (which
    registers arrival without any memory ordering); prefer the relaxed form
    plus an explicit fence when you do not need the built-in acquire-release
    ordering. Pair every arrive with [`barrier_cluster_wait()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_wait "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_wait") from every
    CTA.

    The aligned variant [`barrier_cluster_arrive_aligned()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_aligned "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_aligned") additionally
    asserts that every thread in the issuing warp executes the instruction
    convergently.

    ```python
    # Cluster rendezvous with built-in acquire-release ordering.
    if nvvm.elect_sync():
        nvvm.barrier_cluster_arrive()
        nvvm.barrier_cluster_wait()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cluster\_arrive\_aligned() → None
:   Aligned variant of [`barrier_cluster_arrive()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive").

    Emits `barrier.cluster.arrive.aligned`. Same acquire-release cluster
    arrival, but the `.aligned` qualifier additionally asserts that every
    thread in the issuing warp executes this instruction convergently
    (behaviour is undefined if any lane in the warp does not reach it). Do
    not combine with single-thread election (e.g. inside `elect_sync`): the
    other lanes would never reach the instruction.

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cluster\_arrive\_relaxed() → None
:   Register a relaxed arrival on the cluster-wide barrier.

    Emits `barrier.cluster.arrive.relaxed` (non-aligned): each CTA in the
    cluster registers its arrival, but no memory ordering is implied. Pair
    every arrive with a [`barrier_cluster_wait()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_wait "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_wait") from every CTA to block
    until all have arrived. The relaxed variant is cheaper than the
    acquire-release [`barrier_cluster_arrive()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive"); prefer relaxed + an
    explicit fence where ordering is genuinely needed.

    The aligned variant [`barrier_cluster_arrive_relaxed_aligned()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_relaxed_aligned "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_relaxed_aligned")
    additionally asserts that every thread in the issuing warp executes the
    instruction convergently.

    ```python
    # CTA_2 GEMM cleanup — both CTAs rendezvous before dealloc.
    if nvvm.elect_sync():
        nvvm.barrier_cluster_arrive_relaxed()
        nvvm.barrier_cluster_wait()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cluster\_arrive\_relaxed\_aligned() → None
:   Aligned variant of [`barrier_cluster_arrive_relaxed()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_relaxed "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_relaxed").

    Emits `barrier.cluster.arrive.relaxed.aligned`. Same relaxed cluster
    arrival (no memory ordering), but the `.aligned` qualifier additionally
    asserts that every thread in the issuing warp executes this instruction
    convergently (behaviour is undefined if any lane in the warp does not
    reach it). Do not combine with single-thread election (e.g. inside
    `elect_sync`): the other lanes would never reach the instruction.

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cluster\_wait() → None
:   Block until every CTA in the cluster has called `barrier.cluster.arrive*`.

    Emits `barrier.cluster.wait` (non-aligned): the issuing thread stalls
    until the cluster-wide arrival counter reaches the cluster’s CTA count.
    The counter is set by the paired arrive call from every CTA
    (see [`barrier_cluster_arrive_relaxed()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_relaxed "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive_relaxed") for the relaxed form,
    or [`barrier_cluster_arrive()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_arrive") when acquire-release ordering is
    needed).

    The aligned variant [`barrier_cluster_wait_aligned()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_wait_aligned "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_wait_aligned") additionally
    asserts that every thread in the issuing warp executes the instruction
    convergently.

    ```python
    # See barrier_cluster_arrive_relaxed for the paired usage.
    if nvvm.elect_sync():
        nvvm.barrier_cluster_arrive_relaxed()
        nvvm.barrier_cluster_wait()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cluster\_wait\_aligned() → None
:   Aligned variant of [`barrier_cluster_wait()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_wait "cutlass.experimental.primitives.nvvm_wrapper.barrier_cluster_wait").

    Emits `barrier.cluster.wait.aligned`. Same cluster-scope wait, but the
    `.aligned` qualifier additionally asserts that every thread in the
    issuing warp executes this instruction convergently (behaviour is
    undefined if any lane in the warp does not reach it). Do not combine
    with single-thread election (e.g. inside `elect_sync`): the other lanes
    would never reach the instruction.

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cta\_arrive( : *barrier\_id: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *thread\_count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, ) → None
:   Signal arrival at a named CTA barrier without waiting.

    Emits `barrier.cta.arrive a, b;` (non-aligned). Works in any
    control flow, including divergent on sm\_70+. Producer/consumer
    pairs use the arrive/sync split: the producer arrives and runs ahead
    while the consumer [`barrier_cta_sync()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync") blocks until the count
    is reached. Each CTA has 16 named barrier slots (`barrier_id` in
    0..15).

    For the aligned variant (`barrier.cta.arrive.aligned`, equivalent
    to the legacy `bar.cta.arrive`) use [`barrier_cta_arrive_aligned()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_arrive_aligned "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_arrive_aligned")
    — that form promises that every CTA thread executes the barrier
    and is undefined behavior under divergent control flow.

    Parameters:
    :   - **barrier\_id** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Barrier slot ID in 0..15. Must match the consumer’s id.
        - **thread\_count** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Number of participating threads. Required by PTX
          for `barrier.cta.arrive`; must be a non-zero multiple of the warp
          size (32) and consistent across all arrive/sync calls on this slot.

    Raises:
    :   **ValueError** – if a static `barrier_id` is outside `[0, 15]` or a
        static `thread_count` is not a positive multiple of 32. Runtime
        `Int32` / `Uint32` values pass through unchecked.

    ```python
    # Producer warps signal arrival, then continue work
    if warp < N_PRODUCERS:
        nvvm.barrier_cta_arrive(0, (N_PRODUCERS + N_CONSUMERS) * 32)
        # ... continue producing ...
    else:
        # Consumer warps wait at the same id
        nvvm.barrier_cta_sync(0, thread_count=(N_PRODUCERS + N_CONSUMERS) * 32)
        # ... safe to read producer outputs ...
    ```

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cta\_arrive\_aligned( : *barrier\_id: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *thread\_count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, ) → None
:   Aligned variant of [`barrier_cta_arrive()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_arrive "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_arrive").

    Emits `barrier.cta.arrive.aligned a, b;` — equivalent to the legacy
    `bar.cta.arrive`. Promises that every CTA thread executes this
    barrier in convergence; undefined behavior on sm\_70+ when a strict
    subset of CTA threads reaches the instruction. Use this only when
    the call site is provably all-CTA-converged; otherwise fall back to
    the non-aligned [`barrier_cta_arrive()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_arrive "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_arrive").

    Parameters:
    :   - **barrier\_id** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Barrier slot ID in 0..15. Must match the consumer’s id.
        - **thread\_count** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Number of participating threads. Non-zero
          multiple of the warp size (32), consistent across arrive/sync
          calls on this slot.

    Raises:
    :   **ValueError** – if a static `barrier_id` is outside `[0, 15]` or a
        static `thread_count` is not a positive multiple of 32. Runtime
        `Int32` / `Uint32` values pass through unchecked.

    ```python
    # Every CTA thread reaches this barrier, no divergent guards above
    nvvm.barrier_cta_arrive_aligned(0, threads_per_cta)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cta\_red( : *pred: int | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*, : *barrier\_id: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *kind: [BarrierRedux](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.BarrierRedux "cutlass.experimental.primitives.nvvm_wrapper.BarrierRedux")*, : *\**, : *thread\_count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Synchronize a CTA barrier and reduce a predicate.

    Emits the `barrier.cta.red.{popc,and,or}` family (non-aligned).
    The call marks the issuing thread’s arrival at a named CTA barrier,
    waits until the barrier’s participant count is reached, then
    broadcasts the predicate reduction result to every waiting thread.

    For the aligned variant (`.aligned` modifier, equivalent to the
    legacy `bar.cta.red`) use [`barrier_cta_red_aligned()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_red_aligned "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_red_aligned") — that
    form promises all CTA threads execute the barrier and is undefined
    behavior under divergent control flow.

    Reduction kinds (selected by `kind`):

    - `"and"` → `.and.pred` — returns `Boolean`,
      `True` iff every participant contributed `pred=True`.
    - `"or"` → `.or.pred` — returns `Boolean`,
      `True` iff any participant contributed `pred=True`.
    - `"popc"` → `.popc.u32` — returns `Int32`,
      the count of participants whose `pred` was `True`.

    Do not mix `barrier_cta_red` with the non-reducing variants
    ([`barrier_cta_sync()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync"), [`barrier_cta_arrive()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_arrive "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_arrive")) on the same
    active barrier generation. PTX marks that use as unpredictable; use
    a different `barrier_id` or wait for the barrier to complete and
    reinitialize before reusing it.

    Parameters:
    :   - **pred** (*int* *or* [*Boolean*](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")) – Per-thread predicate contributed to the reduction.
        - **barrier\_id** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – CTA barrier slot ID in 0..15.
        - **kind** ([*BarrierRedux*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.BarrierRedux "cutlass.experimental.primitives.nvvm_wrapper.BarrierRedux")) – Reduction kind — `"and"` / `"or"` yield `Boolean`,
          `"popc"` yields `Int32`.
        - **thread\_count** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*,* *optional*) – Number of participating threads. Omit for all CTA
          threads; otherwise pass a non-zero multiple of the warp size and
          keep it consistent across participants on this slot.

    Returns:
    :   For AND/OR: the reduced `Boolean` broadcast to all
        participants. For POPC: the count `Int32`.

    Raises:
    :   **ValueError** – if `kind` is not one of `"and"` / `"or"` /
        `"popc"`, if a static `barrier_id` is outside `[0, 15]`, or a
        static `thread_count` is not a positive multiple of 32. Runtime
        `Int32` / `Uint32` values pass through unchecked.

    ```python
    tx, _, _ = cute.arch.thread_idx()

    any_lane_zero = nvvm.barrier_cta_red(
        tx == 0,
        barrier_id=0,
        kind="or",
        thread_count=64,
    )
    all_in_range = nvvm.barrier_cta_red(
        tx < 64,
        barrier_id=1,
        kind="and",
        thread_count=64,
    )
    n_true = nvvm.barrier_cta_red(   # returns Int32
        tx % 2 == 0,
        barrier_id=2,
        kind="popc",
        thread_count=64,
    )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cta\_red\_aligned( : *pred: int | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*, : *barrier\_id: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *kind: [BarrierRedux](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.BarrierRedux "cutlass.experimental.primitives.nvvm_wrapper.BarrierRedux")*, : *\**, : *thread\_count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Aligned variant of [`barrier_cta_red()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_red "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_red").

    Emits `barrier.cta.red.{popc,and,or}.aligned` — equivalent to the
    legacy `bar.cta.red`. Promises that every CTA thread executes this
    barrier in convergence; undefined behavior on sm\_70+ when a strict
    subset of CTA threads reaches the instruction. Use this only when
    the call site is provably all-CTA-converged; otherwise fall back to
    the non-aligned [`barrier_cta_red()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_red "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_red").

    Reduction kinds and return-type rules match [`barrier_cta_red()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_red "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_red").

    Parameters:
    :   - **pred** (*int* *or* [*Boolean*](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")) – Per-thread predicate contributed to the reduction.
        - **barrier\_id** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – CTA barrier slot ID in 0..15.
        - **kind** ([*BarrierRedux*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.BarrierRedux "cutlass.experimental.primitives.nvvm_wrapper.BarrierRedux")) – Reduction kind — `"and"` / `"or"` yield `Boolean`,
          `"popc"` yields `Int32`.
        - **thread\_count** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*,* *optional*) – Number of participating threads. Omit for all
          CTA threads; otherwise a non-zero multiple of the warp size and
          consistent across the barrier slot.

    Returns:
    :   For AND/OR: the reduced `Boolean`. For POPC: the count `Int32`.

    Raises:
    :   **ValueError** – if `kind` is not one of `"and"` / `"or"` /
        `"popc"`, if a static `barrier_id` is outside `[0, 15]`, or a
        static `thread_count` is not a positive multiple of 32. Runtime
        `Int32` / `Uint32` values pass through unchecked.

    ```python
    # Every CTA thread reaches this reduction barrier
    all_true = nvvm.barrier_cta_red_aligned(
        tx < threads_per_cta,
        barrier_id=0,
        kind="and",
    )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cta\_sync( : *barrier\_id: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") = 0*, : *\**, : *thread\_count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, ) → None
:   Synchronize threads at a named CTA barrier.

    Emits `barrier.cta.sync a{, b};` (non-aligned). Works in any
    control flow, including divergent on sm\_70+. All participants at
    slot `barrier_id` wait until `thread_count` of them have arrived,
    then proceed together. Omit `thread_count` for the “all CTA
    threads” rendezvous. `nvvm.barrier_cta_sync()` with no arguments
    is the `__syncthreads()` equivalent (slot 0, all CTA threads).

    For the aligned variant (`barrier.cta.sync.aligned`, equivalent to
    the legacy `bar.cta.sync`) use [`barrier_cta_sync_aligned()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync_aligned "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync_aligned") —
    that form promises every CTA thread executes the barrier and is
    undefined behavior under divergent control flow.

    Parameters:
    :   - **barrier\_id** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Barrier slot ID in 0..15.
        - **thread\_count** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*,* *optional*) – Number of participating threads. Omit for all CTA
          threads; otherwise pass a non-zero multiple of the warp size and
          keep it consistent across all uses of this barrier slot.

    Raises:
    :   **ValueError** – if a static `barrier_id` is outside `[0, 15]` or a
        static `thread_count` is not a positive multiple of 32. Runtime
        `Int32` / `Uint32` values pass through unchecked.

    ```python
    # All CTA threads sync (equivalent to __syncthreads at slot 0)
    nvvm.barrier_cta_sync(0)

    # Only warps 2–3 (64 threads) sync at slot 1
    nvvm.barrier_cta_sync(1, thread_count=64)
    ```

    Note

    In warp-specialized kernels, a slot-0 all-CTA sync stalls *every*
    warp, including idle producer warps. Use a named barrier scoped
    to the relevant warps, or use per-consumer `mbarrier` signals
    instead.

cutlass.experimental.primitives.nvvm\_wrapper.barrier\_cta\_sync\_aligned( : *barrier\_id: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") = 0*, : *\**, : *thread\_count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, ) → None
:   Aligned variant of [`barrier_cta_sync()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync").

    Emits `barrier.cta.sync.aligned a{, b};` — equivalent to the legacy
    `bar.cta.sync`. Promises that every CTA thread executes this
    barrier in convergence; undefined behavior on sm\_70+ when a strict
    subset of CTA threads reaches the instruction. Use this only when
    the call site is provably all-CTA-converged; otherwise fall back to
    the non-aligned [`barrier_cta_sync()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync "cutlass.experimental.primitives.nvvm_wrapper.barrier_cta_sync").

    Parameters:
    :   - **barrier\_id** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Barrier slot ID in 0..15.
        - **thread\_count** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*,* *optional*) – Number of participating threads. Omit for all
          CTA threads; otherwise a non-zero multiple of the warp size and
          consistent across the barrier slot.

    Raises:
    :   **ValueError** – if a static `barrier_id` is outside `[0, 15]` or a
        static `thread_count` is not a positive multiple of 32. Runtime
        `Int32` / `Uint32` values pass through unchecked.

    ```python
    # Every CTA thread reaches this barrier (no divergent guards above)
    nvvm.barrier_cta_sync_aligned(0)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.breakpoint() → None
:   Suspend the executing thread for an attached debugger.

    Emits `brkpt`. Suspends the issuing thread so a debugger can inspect
    state; it is effectively a no-op when no debugger is attached.

    ```python
    nvvm.breakpoint()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.cluster\_ctarank() → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   Read `%cluster_ctarank` — this CTA’s linear rank within its cluster.

    ```python
    rank = prims.cluster_ctarank()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.clusterlaunchcontrol\_query\_cancel( : *query\_type: [ClusterLaunchControlQueryType](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.ClusterLaunchControlQueryType "cutlass.experimental.primitives.nvvm_wrapper.ClusterLaunchControlQueryType")*, : *try\_cancel\_response: int | [Int128](basic_data_types.md#cutlass.Int128 "cutlass.Int128") | [Uint128](basic_data_types.md#cutlass.Uint128 "cutlass.Uint128")*, ) → [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   1:1 wrapper over `nvvm.clusterlaunchcontrol_query_cancel`.

    Returns `Boolean` for `IS_CANCELED`, `Int32` for
    `GET_FIRST_CTA_ID_{X,Y,Z}`.

cutlass.experimental.primitives.nvvm\_wrapper.convert( : *src: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *dst\_dtype: object*, : *\**, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *sat: [SaturationMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.SaturationMode "cutlass.experimental.primitives.nvvm_wrapper.SaturationMode") | None = None*, : *relu: bool | None = None*, : *scale\_factor: int | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16") | None = None*, : *scale\_factor\_kind: [ConvertScale](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.ConvertScale "cutlass.experimental.primitives.nvvm_wrapper.ConvertScale") | None = None*, : *random\_bits: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, : *result\_type: object | None = None*, ) → [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   Convert a packed float vector to `dst_dtype`, dispatching on types.

    A single entry point over the typed `convert_*` ops: it reads the element
    type and lane count of `src` plus the requested `dst_dtype` and routes
    to the matching NVVM convert, verifying both the combination and the
    arguments. Covers the float<->float packed conversions:

    - `f32x2` -> `f16` / `bf16` / `f8` / `f6` / `f4`
    - `f32x4` -> `f8` / `f6` / `f4` (requires `random_bits`)
    - `f16x2` / `bf16x2` -> `f8` / `f6` / `f4`
    - `f8x2` / `f6x2` / `f4x2` -> `f16`

    Exotic / non-float-float conversions (`s2f6`, scaled `f8 -> bf16`,
    float<->integer, `tf32`) are not routed here; call the explicit
    `nvvm.convert_*` wrapper for those.

    Parameters:
    :   - **src** ([*Vector*](basic_data_types.md#cutlass.Vector "cutlass.Vector")) – Packed source vector (e.g. `Vector[Float32, 2]`,
          `Vector[Float8E4M3FN, 2]`); its element type and lane count drive
          dispatch.
        - **dst\_dtype** – Target element type (e.g. `Float16`, `Float8E4M3FN`).
        - **rnd** – Rounding mode, where the matched convert accepts one.
        - **sat** – Saturation mode (f8 / f16 / bf16 narrowing only).
        - **relu** – Clamp negatives to zero, where supported.
        - **scale\_factor** – Block scale, for the scaled narrowing converts.
        - **scale\_factor\_kind** – Scale-factor kind paired with `scale_factor`.
        - **random\_bits** – Stochastic-rounding bits; required for `f32x4`
          narrowing and optional for `f32x2 -> f16 / bf16`.
        - **result\_type** – Packed return-shape override (`Int16` /
          `Vector[Int8, 2]`) for the narrowing converts that support it.

    Raises:
    :   **ValueError** – the (source dtype, lane count, destination dtype)
        triple is not a supported float<->float conversion, an argument is not
        accepted by the matched convert, or `random_bits` is missing for an
        `f32x4` narrowing.

    Note

    Narrowing returns a **packed integer carrier** (`Int16` /
    `Vector[Int8, N]`), whereas widening expects a **typed** narrow
    vector (e.g. `Vector[Float8E4M3FN, 2]`, such as one loaded from an
    FP8 tensor) and bitcasts it to the byte carrier internally. A packed
    narrowing result is therefore not directly re-widenable: reinterpret it
    as the typed narrow vector first (`carrier.bitcast(<narrow dtype>)`).

    ```python
    # Narrowing: f32 pair -> packed FP8x2 (an Int16 carrier).
    packed = nvvm.convert(f32x2, cutlass.Float8E4M3FN, rnd="rn", sat="satfinite")

    # Widening: a *typed* FP8x2 vector (e.g. loaded from memory) -> f16x2.
    f16x2 = nvvm.convert(fp8_vec, cutlass.Float16)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.convert\_and\_pack\_integer( : *src\_a: int | ~cutlass.Int32 | ~cutlass.Uint32*, : *src\_b: int | ~cutlass.Int32 | ~cutlass.Uint32*, : *convert\_type: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocMlirType*, : *\**, : *src\_c: int | ~cutlass.Int32 | ~cutlass.Uint32 | None = None*, : *is\_signed: bool | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   Wrapper over `nvvm.convert_and_pack_integer`.

cutlass.experimental.primitives.nvvm\_wrapper.convert\_bf16x2\_to\_s2f6x2( : *\*args: Any*, : *\*\*kwargs: Any*, ) → Any
:   Gated 1:1 wrapper over `nvvm.convert_bf16x2_to_s2f6x2` (PTX ISA 9.1).

    The `.s2f6x2` cvt instruction type was introduced in PTX ISA 9.1 and is
    unavailable on CTK 12.9 (PTX ISA 8.8).

cutlass.experimental.primitives.nvvm\_wrapper.convert\_f32x2\_to\_s2f6x2( : *a: float | ~cutlass.Float32*, : *b: float | ~cutlass.Float32*, : *\**, : *result\_type: type[~cutlass.Int16] | type[~cutlass.Vector] = <class 'cutlass.Int16'>*, : *scale\_factor: int | ~cutlass.Int16 | ~cutlass.Uint16 | None = None*, : *relu: bool | None = None*, ) → [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   1:1 wrapper over `nvvm.convert_f32x2_to_s2f6x2`.

    *result\_type* selects the packed return shape: `Int16` (default,
    one i16 with the two converted values packed into the high and low
    bytes) or `Vector[Int8, 2]` (each lane holds one converted value).
    Both shapes hold equivalent bits.

cutlass.experimental.primitives.nvvm\_wrapper.convert\_f8x2\_to\_bf16x2( : *\*args: Any*, : *\*\*kwargs: Any*, ) → Any
:   Gated 1:1 wrapper over `nvvm.convert_f8x2_to_bf16x2` (PTX ISA 9.2).

    The `.bf16x2` destination from an `.e4m3x2` / `.e5m2x2` source was
    introduced in PTX ISA 9.2 and is unavailable on CTK 12.9 (PTX ISA 8.8).

cutlass.experimental.primitives.nvvm\_wrapper.convert\_float\_to\_integer( : *src: float | ~cutlass.Float32*, : *\**, : *result\_type: type[~cutlass.Int8] | type[~cutlass.Int32] = <class 'cutlass.Int32'>*, : *rnd: ~cutlass.experimental.primitives.nvvm\_wrapper.IntRoundingMode | None = None*, : *sat: bool | None = None*, : *ftz: bool | None = None*, : *is\_signed: bool | None = None*, ) → [Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   1:1 wrapper over `nvvm.convert_float_to_integer`.

cutlass.experimental.primitives.nvvm\_wrapper.convert\_float\_to\_tf32( : *src: float | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")*, : *\**, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *sat: [SaturationMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.SaturationMode "cutlass.experimental.primitives.nvvm_wrapper.SaturationMode") | None = None*, : *relu: bool | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   Wrapper over `nvvm.convert_float_to_tf32`.

cutlass.experimental.primitives.nvvm\_wrapper.convert\_s2f6x2\_to\_bf16x2( : *src: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *\**, : *scale\_factor: int | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16") | None = None*, : *sat: [SaturationMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.SaturationMode "cutlass.experimental.primitives.nvvm_wrapper.SaturationMode") | None = None*, : *relu: bool | None = None*, ) → [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   1:1 wrapper over `nvvm.convert_s2f6x2_to_bf16x2`.

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_commit\_group() → None
:   Commits all prior initiated but uncommitted cp.async.bulk instructions.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group).

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_global\_shared\_cta( : *dst\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *src\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *size: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *l2\_cache\_hint: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, : *byte\_mask: int | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16") | None = None*, ) → None
:   Async bulk-copy a byte range from CTA shared memory to global memory.

    Emits the `.shared::cta -> .global` form of `cp.async.bulk` with
    `.bulk_group` completion: `cp.async.bulk.global.shared::cta.bulk_group
    [dst], [src], size;`. Copies a byte range from SMEM to GMEM without a
    tensor-map descriptor; useful for 1-D / flat buffers where TMA setup is
    not justified. `size` must be a positive multiple of 16 and both
    `dst_mem` and `src_mem` must be 16-byte aligned (the PTX ISA leaves
    non-conforming values undefined). The wrapper exposes the full option set this
    direction has in PTX (unchanged across ISA 8.8 and 9.3): the
    `.L2::cache_hint` cache policy (`l2_cache_hint`) and the `.cp_mask`
    byte mask (`byte_mask`).

    Parameters:
    :   - **dst\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – GMEM destination pointer/array (the `[dst]` operand);
          must be 16-byte aligned.
        - **src\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – CTA-scope SMEM source pointer/array (the `[src]` operand);
          must be 16-byte aligned.
        - **size** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Number of bytes to copy; must be a positive multiple of 16.
        - **l2\_cache\_hint** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional 64-bit L2 cache-eviction policy descriptor
          (emits `.L2::cache_hint` with the policy operand). Defaults to None.
        - **byte\_mask** (*int* *or* [*cutlass.Int16*](basic_data_types.md#cutlass.Int16 "cutlass.Int16") *or* [*cutlass.Uint16*](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16")*,* *optional*) – Optional 16-bit `.cp_mask` selecting which bytes of
          each 16-byte source chunk are written: bit *i* set copies byte *i* of
          every 16-byte chunk, bit *i* clear skips it. Defaults to None (all
          bytes copied).

    Raises:
    :   - **TypeError** – if `src_mem` exposes an address space that is not
          shared memory.
        - **ValueError** – if a statically known `size` is not a positive
          multiple of 16.

    ```python
    # Drain-on-completion SMEM -> GMEM bulk store of `nbytes` bytes.
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_global_shared_cta(gmem_dst, smem_src, nbytes)
    nvvm.cp_async_bulk_commit_group()
    nvvm.cp_async_bulk_wait_group(0)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_prefetch( : *src\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *size: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *l2\_cache\_hint: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, ) → None
:   Prefetch a byte range from global memory into L2.

    Emits `cp.async.bulk.prefetch.L2.global  [src], size;`. Pulls
    `size` bytes starting at `src_mem` into L2 without writing any
    destination; the GMEM access is asynchronous and best-effort.

    Parameters:
    :   - **src\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Global-memory source pointer/array.
        - **size** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Number of bytes to prefetch; must be a positive
          multiple of 16.
        - **l2\_cache\_hint** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional 64-bit L2 cache-eviction policy
          descriptor.

    Raises:
    :   **ValueError** – if a statically known `size` is not a
        positive multiple of 16.

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_shared\_cluster\_global( : *dst\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *src\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *mbar: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *size: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *multicast\_mask: [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | None = None*, : *l2\_cache\_hint: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, ) → None
:   Async bulk-copy a byte range from global memory into cluster shared memory.

    Emits `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [dst], [src], size, [mbar];`. Copies a byte range (`size` a positive
    multiple of 16) from GMEM to SMEM without a tensor map; hardware fires the
    mbarrier’s `complete_tx` automatically when the transfer finishes. For
    per-CTA self-delivery, omit `multicast_mask` (or pass `None`); passing
    `1 << cta_rank` still emits the `.multicast::cluster` PTX modifier and
    pays the multicast-routing overhead even though every byte only lands in the
    issuing CTA. The tensor-descriptor variant
    [`cp_async_bulk_tensor_shared_cluster_global()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_tensor_shared_cluster_global "cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_tensor_shared_cluster_global") follows the same rule.

    Parameters:
    :   - **dst\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Cluster-scope SMEM destination pointer/array; must be
          16-byte aligned. A `shared::cta` pointer is auto-cast to
          `shared::cluster` (the cluster bulk-copy intrinsic requires it).
        - **src\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – GMEM source pointer/array; must be 16-byte aligned.
        - **mbar** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer/Array to the SMEM mbarrier signalled on completion.
        - **size** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Number of bytes to copy (the same value the consumer arms via
          `arrive_expect_tx`); must be a positive multiple of 16.
        - **multicast\_mask** ([*cutlass.Int16*](basic_data_types.md#cutlass.Int16 "cutlass.Int16") *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*,* *optional*) – Optional per-bit mask over CTA ranks. Defaults to
          None: omit it for per-CTA self-delivery (each issuing CTA delivers to
          itself only). A non-None value emits the `.multicast::cluster`
          modifier, gated on mask *presence* not value, so `1 << cta_rank` is a
          footgun (same delivery as omitted, but pays the multicast-routing
          overhead). Set it only for genuine cluster broadcast: e.g. `3`
          (`0b11`) on a 2-CTA cluster delivers identical bytes to both CTAs
          from one issuer.
        - **l2\_cache\_hint** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional 64-bit L2 cache-eviction policy descriptor.

    Raises:
    :   - **TypeError** – if `dst_mem` is not shared or cluster-shared memory,
          or `mbar` is not shared memory.
        - **ValueError** – if a statically known `size` is not a positive
          multiple of 16, or a statically known `multicast_mask` does not fit
          in 32 bits.

    ```python
    # GMEM -> cluster-SMEM bulk load of `nbytes`, signalled on `mbar`.
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_shared_cluster_global(smem_dst, gmem_src, mbar, nbytes)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_shared\_cluster\_shared\_cta( : *dst\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *src\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *mbar: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *size: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, ) → None
:   Async bulk-copy a byte range between two CTAs’ shared memory.

    Emits `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
    [dst], [src], size, [mbar];`. Copies `size` bytes from the
    issuing CTA’s SMEM (`src_mem`) to a peer CTA’s SMEM within the
    same cluster (`dst_mem`, addressed via `shared::cluster`);
    hardware fires the destination `mbarrier`’s `complete_tx` when
    the transfer finishes.

    Parameters:
    :   - **dst\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Cluster-scope SMEM destination pointer/array on
          the peer CTA; must be 16-byte aligned.
        - **src\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – CTA-scope SMEM source pointer/array on the issuing
          CTA; must be 16-byte aligned.
        - **mbar** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer to the destination CTA’s SMEM mbarrier; signalled
          on completion.
        - **size** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Number of bytes to copy (must match the consumer’s
          `arrive_expect_tx` count); positive multiple of 16.

    Raises:
    :   **ValueError** – if a statically known `size` is not a
        positive multiple of 16.

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_tensor\_global\_shared\_cta( : *tma\_descriptor: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *src\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *coordinates: list[int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")]*, : *\**, : *l2\_cache\_hint: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, : *mode: [TMAStoreMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMAStoreMode "cutlass.experimental.primitives.nvvm_wrapper.TMAStoreMode") | None = None*, ) → None
:   Async TMA-store a tile from CTA shared memory to global memory.

    Emits `cp.async.bulk.tensor.<N>d.global.shared::cta.bulk_group
    [tensor_map, {coords}], [src];` where `<N>` is the number of
    `coordinates`. Stores a CTA-shared tile to the global tensor selected by
    the TMA descriptor at `coordinates`. Completion uses the bulk-group
    mechanism (no mbarrier signal), so the store must be drained with
    [`cp_async_bulk_commit_group()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_commit_group "cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_commit_group") and [`cp_async_bulk_wait_group()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_wait_group "cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_wait_group").

    Parameters:
    :   - **tma\_descriptor** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMA tensor-map descriptor for the destination
          global tensor.
        - **src\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – CTA-scope SMEM source pointer/array.
        - **coordinates** (*list* *of* *(**int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*)*) – 1-5D tile coordinate into the descriptor’s tensor.
        - **l2\_cache\_hint** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional 64-bit L2 cache-eviction policy descriptor.
        - **mode** ([*TMAStoreMode*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMAStoreMode "cutlass.experimental.primitives.nvvm_wrapper.TMAStoreMode")*,* *optional*) – Optional TMA store mode (`tile` default, `im2col`, or
          `tile_scatter4`).

    Raises:
    :   - **TypeError** – if `src_mem` does not reside in shared memory.
        - **ValueError** – if the `coordinates` count is invalid for `mode`
          (1-5 for tile, 3-5 for im2col, exactly 2 for scatter4).

    Reverse of [`cp_async_bulk_tensor_shared_cta_global()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_tensor_shared_cta_global "cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_tensor_shared_cta_global") (load).
    TMA hardware reads SMEM and writes the tile to the global tensor
    described by `tma_descriptor` at the given `coordinates`.
    Argument order is `(desc, smem, coords)` — no mbarrier; TMA stores
    use commit/wait groups instead of mbarriers.

    > **Proxy fence required before issue**: thread SMEM writes go
    > through the “generic” proxy; the TMA engine reads through the
    > “async” proxy. Issue
    > `nvvm.fence_proxy("async_shared", space=SharedSpace.shared_cta)`
    > before this call so the TMA engine sees the latest SMEM data.
    > Without it, TMA may read stale SMEM.

    ```python
    # TMA store of an SMEM tile to GMEM at (x, y), then drain.
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_global_shared_cta(tma_desc, smem_src, [x, y])
    nvvm.cp_async_bulk_commit_group()
    nvvm.cp_async_bulk_wait_group(0)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_tensor\_prefetch( : *tma\_descriptor: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *coordinates: list[int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")]*, : *im2col\_offsets: list[int | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16")]*, : *\**, : *mode: [TMALoadMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode "cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode") | None = None*, : *l2\_cache\_hint: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, ) → None
:   Prefetch a TMA-tensor tile from global memory into L2.

    Emits `cp.async.bulk.prefetch.tensor.<N>d.L2.global  [tma_desc,
    {coords}];`. Pulls the tile selected by `coordinates` (and
    `im2col_offsets` for im2col mode) into L2 without any
    destination; the access is asynchronous and best-effort.

    Parameters:
    :   - **tma\_descriptor** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMA tensor-map descriptor for the global tensor.
        - **coordinates** (*list* *of* *(**int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*)*) – 1-5D tile coordinate into the descriptor’s tensor.
        - **im2col\_offsets** (*list* *of* *(**int* *or* [*cutlass.Int16*](basic_data_types.md#cutlass.Int16 "cutlass.Int16") *or* [*cutlass.Uint16*](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16")*)*) – Im2col offsets; empty list for tile mode.
        - **mode** ([*TMALoadMode*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode "cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode")*,* *optional*) – Optional TMA load mode (`tile` default, `im2col`).
        - **l2\_cache\_hint** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional 64-bit L2 cache-eviction policy descriptor.

    Raises:
    :   **ValueError** – if the `coordinates` count is invalid for `mode`.

    The descriptor-override path is exposed separately as
    `cp_async_bulk_tensor_prefetch_override()`.

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_tensor\_reduce( : *tma\_descriptor: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *src\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *red\_kind: [TMARedux](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMARedux "cutlass.experimental.primitives.nvvm_wrapper.TMARedux")*, : *coordinates: list[int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")]*, : *\**, : *mode: [TMAStoreMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMAStoreMode "cutlass.experimental.primitives.nvvm_wrapper.TMAStoreMode") | None = None*, : *l2\_cache\_hint: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, ) → None
:   Issue a TMA async tensor reduction from shared memory to global memory.

    Lowers to PTX `cp.reduce.async.bulk.tensor`. The source tile in
    shared::cta memory is reduced into the destination tensor described by
    `tma_descriptor` at `coordinates` using `red_kind`. The tensor-map
    element type determines which reduction kinds are valid; PTX supports
    `ADD` for integer/fp elements, `MIN`/`MAX` for integer and fp16/bf16
    element types, `INC`/`DEC` for `u32`, and bitwise reductions for
    `b32`/`b64`.

    The operation is non-blocking and uses bulk async-group completion. It does
    not take an mbarrier operand; issue the operation, commit the bulk group,
    then wait on the group before consuming completion.

    Parameters:
    :   - **tma\_descriptor** (*Array* *or* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Tensor-map descriptor for the global destination.
        - **src\_mem** (*Array* *or* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Shared-memory source tile.
        - **red\_kind** ([*TMARedux*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMARedux "cutlass.experimental.primitives.nvvm_wrapper.TMARedux")) – TMA reduction operation, such as `ADD`, `MIN`,
          `MAX`, `INC`, `DEC`, `AND`, `OR`, or `XOR`.
        - **coordinates** (*list**[**int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*]*) – One to five `s32` tensor coordinates matching the
          descriptor rank.
        - **mode** ([*TMAStoreMode*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMAStoreMode "cutlass.experimental.primitives.nvvm_wrapper.TMAStoreMode")*,* *optional*) – Optional TMA store mode. Omit for tile mode; im2col modes
          require descriptor-compatible ranks and coordinates.
        - **l2\_cache\_hint** (*int* *or* [*Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional 64-bit L2 cache policy.

    Raises:
    :   - **TypeError** – if `src_mem` does not reside in shared memory.
        - **ValueError** – if `red_kind` is not a valid `TMARedux`, if `mode`
          is `tile_scatter4` (unsupported by reduce), or if the `coordinates`
          count is invalid for `mode` (1-5 for tile, 3-5 for im2col).

    ```python
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_reduce(
            desc,
            smem_tile,
            "add",
            [row, col],
        )
        nvvm.cp_async_bulk_commit_group()
        nvvm.cp_async_bulk_wait_group(0)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_tensor\_shared\_cluster\_global( : *dst\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *tma\_descriptor: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *coordinates: list[int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")]*, : *mbar: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *im2col\_offsets: list[int | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16")]*, : *\**, : *multicast\_mask: [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | None = None*, : *l2\_cache\_hint: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, : *mode: [TMALoadMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode "cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode") | None = None*, : *group: [CTAGroup](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup") | None = None*, ) → None
:   Issue a TMA async load into shared memory with optional cluster multicast.

    Like `cp_async_bulk_tensor_shared_cta_global` but supports multicast
    across all CTAs in a cluster (`group="cta_2"`).

    **Arg order**: `(dst_smem, tma_desc, coords, mbar, im2col_offsets, ...)`
    — coords come **before** mbar. This is the opposite of
    `cp_async_bulk_tensor_shared_cta_global` (mbar before coords) and is
    the #1 footgun when porting CTA\_1 code to CTA\_2.

    **CTA\_2 mbar routing — bit 24 maps to 2-SM-group leader, NOT
    cluster leader**. When `group="cta_2"`, this wrapper
    masks the mbar pointer with `& 0xFEFFFFFF` (clears bit 24 only).
    Bit 24 holds the LSB of `%cluster_ctarank` — clearing it routes
    the address from a peer CTA back to its **2-SM-group leader** (the
    even-rank CTA in the same pair).

    - In a 2-CTA cluster (`cluster_shape=(2,1,1)`), the only group
      leader is cluster rank 0, so this is equivalent to “route to
      cluster leader” and is what `2cta_mma_basic.py` relies on.
    - In multi-group clusters (`cluster_shape=(2, n, 1)` with n > 1),
      complete\_tx still routes only to the **issuer’s** group leader
      (the even-rank CTA in the issuer’s pair). It does NOT route
      across groups: a TMA issued in group 0 cannot deliver
      `complete_tx` to group 1’s mbar through this wrapper. Cross-
      group multicast TMA (e.g. broadcast B to all groups) requires
      manually constructing a cluster-shared mbar pointer via
      `mapa.shared::cluster` so all issuers target the same
      cluster-leader mbar; this wrapper does not expose that path.

    **Shared-tile multicast vs per-CTA unicast** — two distinct topologies,
    commonly confused. The right choice comes from the pseudo-code:

    - *Shared-tile multicast* (`multicast @cta` in the pseudo-code on a
      tile declared at the cluster scale, e.g. `tile B_smem: f16:K×N @
      SMEM` with `N = N_TILE` covering the full cluster width). Each
      CTA gets an identical copy of one full tile. One descriptor with
      `box_dims` spanning the whole tile; `multicast_mask` includes
      **every CTA** in the cluster (`0b11` for 2-CTA, `0xff` for 8-CTA).
      **Mbar protocol depends on cluster\_size:**

      - *cluster\_size == 2 (with downstream MMA reuse)*: optionally pass
        `group="cta_2"` to enable bit-24 mbar collapse — only
        the leader inits + does `arrive_expect_tx` with the **full-tile**
        byte count, and only the leader waits. All CTAs’ `complete_tx`
        is redirected to the leader’s mbar. This is the right shape when
        the same mbar is then reused as the MMA’s input-ready bar.
      - *cluster\_size ≥ 2 (general, including > 2)*: **omit** `group=`
        and use **per-CTA local mbars** — every CTA inits its own mbar
        (at the same SMEM offset), every CTA does `arrive_expect_tx`
        with the **per-CTA** tile byte count, only the leader issues the
        TMA, every CTA waits on its own mbar. Per the PTX ISA,
        the hardware multicasts `complete_tx` to that same SMEM offset
        in every destination CTA’s local SMEM. Works for cluster\_size
        ∈ {2, 4, 8, 16}.

      Trying `group="cta_2"` with cluster\_size > 2 hangs:
      the bit-24 mask redirects to the *issuer’s* 2-SM-group leader only
      (cluster ranks 0+1), so cross-group receivers’ mbars never arrive.
    - *Per-CTA unicast* (used when each CTA consumes a different slice of
      the tensor, e.g. each CTA gets a different half of the M-dimension).
      Descriptor `box_dims` covers only the per-CTA slice, **omit
      ``multicast\_mask`` entirely** (or pass `None`), coordinates shift
      by `cta_rank * slice` per CTA, and each CTA calls
      `mbarrier_arrive_expect_tx` independently. This is functionally
      the CTA\_1 pattern wrapped in `group=CTA_2` for the routing
      bookkeeping; it is **not** a multicast.

    **Multicast modifier rule** — the `multicast::cluster` PTX modifier
    is gated by *whether ``multicast\_mask`` is present*, not by the value
    the mask carries. A “selfcast” mask such as `1 << cta_rank`
    therefore still emits the modifier and pays the multicast-routing
    overhead even though the bytes only land in the issuing CTA. For
    per-CTA unicast topology, omit `multicast_mask`. Set
    `multicast_mask` only when the topology is genuinely
    *shared-tile multicast* (one issuer, mask covers every receiving
    CTA). Applies to both `group=CTA_1` and `group=CTA_2`.

    Do not split a shared-tile multicast into per-CTA unicasts to “save
    bytes” — the multicast is a single HBM read fanned out over the
    cluster interconnect, so per-CTA unicast **increases** HBM pressure
    instead of reducing it, in addition to producing a different SMEM
    layout than the spec.

    ```python
    # Shared-tile multicast — both CTAs see the same 128×128 tile.
    # arrive_expect_tx is called by the leader only, with the FULL tile's
    # byte count (not per-CTA).
    if is_leader:
        nvvm.mbarrier_arrive_expect_tx(mbar + s, A_full_bytes + B_full_bytes)
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_shared_cluster_global(
            smem_B + s * tile_b,
            tma_b_desc,
            (k, n),                    # full tile origin
            mbar + s,
            [],                        # ← coords BEFORE mbar
            multicast_mask=Int16(0b11),   # every CTA in the 2-CTA cluster
            group="cta_2",
        )
    ```

    Parameters:
    :   - **dst\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Cluster-scope SMEM destination tile (shared or
          cluster-shared); must be 16-byte aligned.
        - **tma\_descriptor** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMA tensor-map descriptor for the source tensor.
        - **coordinates** (*list* *of* *(**int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*)*) – 1-5D tile coordinate into the descriptor’s tensor
          (note: coords come BEFORE `mbar` in the arg order).
        - **mbar** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Shared-memory mbarrier signalled on completion.
        - **im2col\_offsets** (*list* *of* *(**int* *or* [*cutlass.Int16*](basic_data_types.md#cutlass.Int16 "cutlass.Int16") *or* [*cutlass.Uint16*](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16")*)*) – im2col offsets (empty list for tile mode).
        - **multicast\_mask** ([*cutlass.Int16*](basic_data_types.md#cutlass.Int16 "cutlass.Int16") *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*,* *optional*) – Optional per-bit CTA-rank mask for cluster
          multicast (omit for per-CTA unicast).
        - **l2\_cache\_hint** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional 64-bit L2 cache-eviction policy descriptor.
        - **mode** ([*TMALoadMode*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode "cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode")*,* *optional*) – Optional TMA load mode (tile default, or im2col).
        - **group** ([*CTAGroup*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup")*,* *optional*) – CTA group selector (`cta_1` default, or `cta_2`).

    Raises:
    :   - **TypeError** – if `dst_mem` is not shared/cluster-shared memory or
          `mbar` is not shared memory.
        - **ValueError** – if the `coordinates` count is invalid for `mode`,
          or a statically known `multicast_mask` does not fit in 32 bits.

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_tensor\_shared\_cta\_global( : *dst\_mem: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *tma\_descriptor: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *coordinates: list[int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")]*, : *mbar: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *im2col\_offsets: list[int | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16")] | None = None*, : *\**, : *l2\_cache\_hint: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, : *mode: [TMALoadMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode "cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode") | None = None*, ) → None
:   Issue a TMA async load from global memory into this CTA’s shared memory.

    TMA hardware performs the DMA without stalling the issuing warp.
    Completion is signaled by an mbarrier `complete_tx` decrement, which
    fires the barrier once all bytes have arrived.

    **Arg order**: `(dst_smem, tma_desc, coords, mbar)` — coords **before**
    mbar, matching [`cp_async_bulk_tensor_shared_cluster_global()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_tensor_shared_cluster_global "cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_tensor_shared_cluster_global") so porting
    CTA\_1 code to CTA\_2 only changes the function name, not the argument order.

    **Calling convention**:

    - Call from exactly **one thread** (e.g. `if nvvm.elect_sync()`).
    - Call `arrive_expect_tx(mbar, nbytes)` **before** this function so the
      transaction counter is set before TMA can decrement it.
    - `nbytes` = rows × cols × sizeof(dtype) for the tile being loaded.
    - Coordinates are tensor-space element indices, not byte offsets. For a
      2-D descriptor created via `create_tensor_map_tiled_from_view` on a
      row-major (M, K) tensor, the TMA coord order is column-major:
      `(k_offset, m_offset)` — K (innermost) first.

    ```python
    # Separate elect_sync for arrive vs each TMA load (performance)
    if nvvm.elect_sync():
        nvvm.mbarrier_arrive_expect_tx(mbar + s, A_bytes + B_bytes)
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_shared_cta_global(
            smem_A + s * tile_a, tma_a_desc, (k, m), mbar + s)
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_shared_cta_global(
            smem_B + s * tile_b, tma_b_desc, (k, n), mbar + s)
    ```

    For multicast / cluster TMA (CTA\_2) use
    `cp_async_bulk_tensor_shared_cluster_global` instead.

    `mode` selects the TMA access pattern (`TILE` default; also the
    `IM2COL` family and `TILE_GATHER4`). For im2col modes pass the per-dim
    `im2col_offsets`; tile and gather4 modes leave it empty.

    Parameters:
    :   - **dst\_mem** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – This CTA’s SMEM destination tile; must be 16-byte aligned.
        - **tma\_descriptor** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMA tensor-map descriptor for the source tensor.
        - **coordinates** (*list* *of* *(**int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*)*) – 1-5D tile coordinate into the descriptor’s tensor
          (coords come before `mbar` in the arg order).
        - **mbar** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Shared-memory mbarrier signalled on completion.
        - **im2col\_offsets** (*list* *of* *(**int* *or* [*cutlass.Int16*](basic_data_types.md#cutlass.Int16 "cutlass.Int16") *or* [*cutlass.Uint16*](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16")*)**,* *optional*) – im2col offsets (empty / omitted for tile mode).
        - **l2\_cache\_hint** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional 64-bit L2 cache-eviction policy descriptor.
        - **mode** ([*TMALoadMode*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode "cutlass.experimental.primitives.nvvm_wrapper.TMALoadMode")*,* *optional*) – Optional TMA load mode (`TILE` default, im2col, or gather4).

    Raises:
    :   - **TypeError** – if `dst_mem` or `mbar` is not shared memory.
        - **ValueError** – if the `coordinates` count is invalid for `mode`.

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_bulk\_wait\_group( : *group: cutlass.cute.typing.Int*, : *\**, : *read: bool | None = None*, ) → None
:   Waits till only a specified numbers of cp.async.bulk groups are pending.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_commit\_group() → None
:   Commits all prior initiated but uncommitted cp.async instructions.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group).

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_mbarrier\_arrive( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *noinc: bool | None = None*, ) → None
:   Tie outstanding `cp.async` operations to an mbarrier.

    Emits `cp.async.mbarrier.arrive[.noinc][.shared{::cta}]`. Causes an asynchronous arrive-on operation to fire
    on the mbarrier at `addr` once all prior `cp.async` ops issued
    by the executing thread have completed. This lets a consumer
    block on the mbarrier and only wake when the cp.async data is
    ready, without polling.

    Two semantic flavours via `noinc`:

    - `noinc=False` (default) — pending count is incremented by 1
      before the asynchronous arrive, giving a net-zero effect on the
      pending count for the current phase. `mbarrier.init` only
      needs to account for `mbarrier.arrive` arrivals.
    - `noinc=True` — no pre-increment. The asynchronous arrive
      decrement must be pre-accounted for in `mbarrier.init`’s
      thread-count. Use when issuing many `cp.async` operations
      and aggregating them into a single arrival.

    In a `cp_async_shared_global` pipeline, this is the producer’s completion signal:
    consumers wait the mbarrier, not `cp_async_wait_group`. Count the
    lane-level async-arrives that will be delivered to the mbarrier. For
    example, a hybrid TMA + per-thread cp.async producer commonly initializes the full
    barrier with `1 + 32`: one elected-thread TMA arrival plus one
    `cp_async_mbarrier_arrive(noinc=True)` from each lane in a cp.async
    warp. The TMA transaction byte count should not include the cp.async
    bytes because these copies complete through the async arrive path.

    `cp_async_mbarrier_arrive` is CTA-local. In CTA\_2 kernels where a
    leader CTA’s barrier gates collective MMA, have each CTA’s cp.async lanes
    arrive on a local mbarrier, then use a completion-forwarder warp to wait
    the local mbarrier and cross-CTA arrive on the leader barrier with the
    `shared::cluster` pointer returned by `mapa`.

    ```python
    # One TMA elected-thread arrival plus 32 cp.async lane arrivals.
    if nvvm.elect_sync():
        nvvm.mbarrier_init(full_mbar + stage, 1 + 32)

    # Each cp.async lane issues its copies, then contributes one
    # asynchronous arrive after its prior cp.async operations retire.
    nvvm.cp_async_shared_global(
        sfa_smem,
        sfa_gmem,
        8,
        "ca",
    )
    nvvm.cp_async_mbarrier_arrive(full_mbar + stage, noinc=True)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_shared\_global( : *dst: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *src: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *size: int*, : *modifier: [LoadCacheModifier](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.LoadCacheModifier "cutlass.base_dsl.array.LoadCacheModifier")*, : *\**, : *cp\_size: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, ) → None
:   Issue a per-thread async copy from global to shared memory (SM80+).

    Each thread independently copies `size` bytes from `src` (GMEM) to
    `dst` (SMEM) without stalling. Unlike TMA, no descriptor is required
    and every participating thread issues its own copy.

    Parameters:
    :   - **dst** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Destination in shared memory (addr-space 3).
        - **src** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Source pointer in global memory.
        - **size** (*int*) – Bytes per thread. Must be `4`, `8`, or `16`.
          Use `16` (128-bit) for maximum throughput (one `float4`).
        - **modifier** ([*LoadCacheModifier*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.LoadCacheModifier "cutlass.experimental.primitives.nvvm_wrapper.LoadCacheModifier")) – Cache policy — `"ca"` (cache L1+L2, 4/8/16 B) or
          `"cg"` (bypass L1, L2 only, 16 B only). Prefer `"cg"` for
          streaming loads that won’t be reused.
        - **cp\_size** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*,* *optional*) – Source byte count for a **masked (zero-fill) copy**.
          When `cp_size < size`, the bytes `[cp_size, size)` in `dst` are
          zeroed rather than left undefined. Pass `cp_size=0` to write all
          zeros (useful for out-of-bounds boundary tiles). Leave `None` for
          full-size copies.

    Raises:
    :   - **TypeError** – if `dst` is not shared memory.
        - **ValueError** – if `size` is not 4/8/16, if `modifier` is `cg`
          with `size` != 16, or if `cp_size` falls outside `[0, size]`.

    **Synchronization:** copies are asynchronous. Use
    `nvvm.cp_async_commit_group()` to mark a batch and
    `nvvm.cp_async_wait_group(n)` to drain until ≤ `n` groups remain.

    **Swizzle requirement for tcgen05.mma:** when the SMEM tile will be read
    by `tcgen05.mma`, use the SMEM layout expected by the corresponding
    matrix descriptor. For the common 128B XOR layout, produce the same layout
    that `Pointer.store_swizzled` or a 128B-swizzled tensor map would create.

    ```python
    # 16-byte streaming load per thread, bypass L1
    nvvm.cp_async_shared_global(smem_dst, gmem_src, 16, "cg")
    nvvm.cp_async_commit_group()
    # ... later ...
    nvvm.cp_async_wait_group(0)   # wait for all groups
    ```

cutlass.experimental.primitives.nvvm\_wrapper.cp\_async\_wait\_group(*n: cutlass.cute.typing.Int*) → None
:   Waits till only a specified numbers of cp.async groups are pending.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all).

cutlass.experimental.primitives.nvvm\_wrapper.cp\_reduce\_async\_bulk\_global\_shared\_cta( : *dst: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *src: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *size: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *op: [CpReduceOp](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CpReduceOp "cutlass.experimental.primitives.nvvm_wrapper.CpReduceOp") = CpReduceOp.ADD*, : *type: [CpReduceType](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CpReduceType "cutlass.experimental.primitives.nvvm_wrapper.CpReduceType") = CpReduceType.BF16*, : *noftz: bool = False*, : *l2\_cache\_hint: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, ) → None
:   `cp.reduce.async.bulk.global.shared::cta` — non-TMA bulk reduction.

    Asynchronously reduces *size* bytes from *src* (shared::cta) into *dst*
    (global) using *op* / *type*. Unlike
    [`cp_async_bulk_tensor_reduce()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_tensor_reduce "cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_tensor_reduce"), this operates on raw pointers and a
    byte count (irregular / scatter access, e.g. MoE finalize scatter-reduce).
    Uses `bulk_group` completion — bracket with
    [`cp_async_bulk_commit_group()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_commit_group "cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_commit_group") / [`cp_async_bulk_wait_group()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_wait_group "cutlass.experimental.primitives.nvvm_wrapper.cp_async_bulk_wait_group").

    There is no NVVM dialect op for this instruction, so it emits inline PTX.

    Parameters:
    :   - **size** – byte count, must be a multiple of 16.
        - **noftz** – disable flush-to-zero; only valid with `op=ADD` and
          `type` in `{F16, BF16}`.
        - **l2\_cache\_hint** – optional 64-bit L2 eviction policy.

cutlass.experimental.primitives.nvvm\_wrapper.cvta\_to( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*, : *space: [CvtaSpace](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CvtaSpace "cutlass.experimental.primitives.nvvm_wrapper.CvtaSpace")*, : *\**, : *size: [CvtaSize](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CvtaSize "cutlass.experimental.primitives.nvvm_wrapper.CvtaSize") = CvtaSize.U64*, ) → Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")
:   `cvta.to.{space}` — convert a generic address to a space-specific one.

    For `Array` / `Pointer` inputs this is an `llvm.addrspacecast` to the
    target space (same wrapper type returned); integer inputs round-trip through
    `inttoptr` / `addrspacecast` / `ptrtoint`. `.param` spaces have no
    addrspacecast and use inline PTX.

cutlass.experimental.primitives.nvvm\_wrapper.cvt\_f32x2\_to\_f4x2( : *a: float | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")*, : *b: float | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")*, : *dst\_type: object*, : *\**, : *is\_pzo: bool | None = None*, : *scale\_factor: int | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16") | None = None*, : *scale\_factor\_kind: [ConvertScale](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.ConvertScale "cutlass.experimental.primitives.nvvm_wrapper.ConvertScale") | None = None*, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *relu: bool | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   `cvt.{rnd}.{f4x2}.f32` — convert an `f32` pair to packed `f4x2`.

    Returns the packed byte in the low 8 bits of an `Int32` (callers typically `& 0xFF` and
    shift it into a 32-bit word).

cutlass.experimental.primitives.nvvm\_wrapper.cvt\_f32x2\_to\_f8x2( : *a: float | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")*, : *b: float | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")*, : *dst\_ty: object*, : *\**, : *is\_pzo: bool | None = None*, : *scale\_factor: int | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16") | None = None*, : *scale\_factor\_kind: [ConvertScale](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.ConvertScale "cutlass.experimental.primitives.nvvm_wrapper.ConvertScale") | None = None*, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *sat: [SaturationMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.SaturationMode "cutlass.experimental.primitives.nvvm_wrapper.SaturationMode") | None = None*, : *relu: bool | None = None*, ) → [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16")
:   `cvt.{rnd}.{f8x2}.f32` — convert an `f32` pair to packed `f8x2`.

    Returns the two f8 lanes packed into an `Int16`.

cutlass.experimental.primitives.nvvm\_wrapper.cvt\_packfloat( : *src\_a: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *src\_c: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *from\_: [CVTPackFloat](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CVTPackFloat "cutlass.experimental.primitives.nvvm_wrapper.CVTPackFloat")*, : *to: [CVTPackFloat](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CVTPackFloat "cutlass.experimental.primitives.nvvm_wrapper.CVTPackFloat")*, : *\**, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *sat: [SaturationModeKind](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.SaturationModeKind "cutlass.experimental.primitives.nvvm_wrapper.SaturationModeKind") | None = None*, : *relu: bool | None = None*, : *extract\_hi: bool | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   Wrapper over `nvvm.cvt_packfloat`.

cutlass.experimental.primitives.nvvm\_wrapper.cvt\_packfloat\_f32( : *src\_a: float | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")*, : *src\_b: float | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")*, : *src\_c: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *to: [CVTPackFloat](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CVTPackFloat "cutlass.experimental.primitives.nvvm_wrapper.CVTPackFloat")*, : *\**, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *sat: [SaturationModeKind](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.SaturationModeKind "cutlass.experimental.primitives.nvvm_wrapper.SaturationModeKind") | None = None*, : *relu: bool | None = None*, : *extract\_hi: bool | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   Wrapper over `nvvm.cvt_packfloat_f32`.

cutlass.experimental.primitives.nvvm\_wrapper.dot\_accumulate\_2way( : *a: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *a\_type: [DotAccumulateType](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.DotAccumulateType "cutlass.experimental.primitives.nvvm_wrapper.DotAccumulateType")*, : *b: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *b\_type: [DotAccumulateType](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.DotAccumulateType "cutlass.experimental.primitives.nvvm_wrapper.DotAccumulateType")*, : *c: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *b\_hi: bool*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   Wrapper over `nvvm.dot_accumulate_2way`.

cutlass.experimental.primitives.nvvm\_wrapper.dot\_accumulate\_4way( : *a: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *a\_type: [DotAccumulateType](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.DotAccumulateType "cutlass.experimental.primitives.nvvm_wrapper.DotAccumulateType")*, : *b: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *b\_type: [DotAccumulateType](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.DotAccumulateType "cutlass.experimental.primitives.nvvm_wrapper.DotAccumulateType")*, : *c: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   Wrapper over `nvvm.dot_accumulate_4way`.

cutlass.experimental.primitives.nvvm\_wrapper.elect\_sync( : *\**, : *membermask: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") = 4294967295*, ) → [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Elect one lane from a warp-convergent group.

    Exactly one predicated active lane in `membermask` is elected. That
    lane receives `True`; every other participating lane receives
    `False`. PTX guarantees deterministic election for a fixed member
    mask, but does not specify which lane ID wins. Use this as the
    compiler-visible gate for non-idempotent single-issuer operations such
    as one TMA issuer per warp, one `mbarrier_arrive_expect_tx` call, or
    one `tcgen05_commit` call.

    Convergence requirement: every executing lane must be named in
    `membermask`, and all lanes named by `membermask` must actively
    execute the instruction. Lanes outside the mask should branch around
    the call.

    Parameters:
    :   **membermask** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – 32-bit warp participation mask; every set bit
        identifies a lane that must be executing this instruction.
        Defaults to `FULL_MASK` (`0xFFFFFFFF`, all 32 lanes).

    Returns:
    :   `True` in the elected lane; `False` in every other
        participating lane.

    Return type:
    :   [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")

    Raises:
    :   **ValueError** – if `membermask` is a Python `int` outside
        `[0, 0xFFFFFFFF]`.

    ```python
    # Elect one lane per warp to perform a non-idempotent op.
    if nvvm.elect_sync():           # uses FULL_MASK by default
        nvvm.mbarrier_arrive_expect_tx(full_bar + s, tile_bytes)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.exit() → None
:   Terminate the issuing thread’s execution of the kernel.

    Emits PTX `exit`: the issuing thread stops running immediately.
    Other threads in the warp / CTA continue. Rarely needed: a natural
    return from `@cute.kernel` is almost always preferable, because
    `exit` does not cooperate with mbarrier-based control-flow
    staging. CTA barriers (`bar.sync` / `barrier.cta`) exclusively
    waiting on arrivals from exited threads are released automatically by
    hardware (a PTX guarantee), but `mbarrier` arrivals are explicit and
    are NOT auto-completed: a thread that exits before its
    `mbarrier.arrive` leaves any consumer of that arrival hung
    indefinitely.

    ```python
    # Rarely needed — prefer an ``if`` guard around the work.
    if tid >= num_valid:
        nvvm.exit()
    # ... remaining threads continue ...
    ```

cutlass.experimental.primitives.nvvm\_wrapper.fence\_acq\_rel(*scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope")*) → None
:   `fence.acq_rel.{scope}` — acquire-release memory fence (PTX §9.7.13.4).

    There is no NVVM dialect op for the generic acquire-release fence, so this
    emits inline PTX.

    Parameters:
    :   **scope** – memory scope — `MemScope.{CTA,CLUSTER,GPU,SYS}`.

    ```python
    prims.fence_acq_rel(prims.MemScope.CTA)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.fence\_mbarrier\_init() → None
:   Make mbarrier init writes visible to all threads before first use.

    PTX address-space semantics require an explicit fence between writing an
    mbarrier object (via [`mbarrier_init()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.mbarrier_init "cutlass.experimental.primitives.nvvm_wrapper.mbarrier_init")) and the first arrive or wait
    on that barrier. Without this fence the init write may not be visible
    to threads in a different warp or address space. The fence itself is a
    per-thread no-op; visibility comes from the CTA-wide sync that follows.

    ```python
    # One elected thread initialises all stages.
    if warp_idx == 0:
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(NUM_STAGES):
                nvvm.mbarrier_init(full_bar + i, 1)
                nvvm.mbarrier_init(empty_bar + i, 1)
    # Make init visible before any warp calls arrive/wait.
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.fence\_proxy( : *kind: ~cutlass.experimental.primitives.nvvm\_wrapper.Proxy*, : *\**, : *space: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject | None = None*, ) → None
:   Order writes across memory proxy domains.

    Emits `fence.proxy.{kind}`. The most common use case is
    `fence_proxy("async_shared", space=SharedSpace.shared_cta)` before a
    TMA store (`cp_async_bulk_tensor_global_shared_cta`), which
    makes thread SMEM writes visible to the TMA async-copy engine.

    Without this fence, the TMA store reads stale SMEM data because regular
    thread stores go through a different memory proxy domain than TMA
    async copies.

    Parameters:
    :   - **kind** ([*Proxy*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy")) – Proxy kind. Use `"async_shared"` for TMA store
          fencing (SMEM to global via TMA).
        - **space** ([*SharedSpace*](cute_dsl_api/cute_nvgpu_common.md#cutlass.cute.nvgpu.SharedSpace "cutlass.cute.nvgpu.SharedSpace") *|* *None*) – Shared-memory space qualifier. Only valid with the
          `"async"` / `"async_shared"` proxy kinds. Use
          `SharedSpace.shared_cta` for CTA-local SMEM.

    Raises:
    :   - **ValueError** – `space` is supplied together with a proxy kind
          that does not accept a space qualifier (anything other than
          `"async"` / `"async_shared"`).
        - **TypeError** – `kind` is a raw NVVM `ProxyKind` dialect enum; pass
          a [`Proxy`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") member or its string alias instead.

    ```python
    # Thread writes to SMEM, then TMA stores SMEM -> global:
    nvvm.barrier_cta_sync()                       # all threads done writing SMEM
    nvvm.fence_proxy(
        "async_shared",
        space=nvvm.SharedSpace.shared_cta,
    )
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_global_shared_cta(
            tma_descriptor=tma_desc,
            src_mem=smem,
            coordinates=dst_coords,
        )
        nvvm.cp_async_bulk_commit_group()
        nvvm.cp_async_bulk_wait_group(0)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.fence\_proxy\_acquire( : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope")*, : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *size: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *from\_proxy: [Proxy](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") | None = None*, : *to\_proxy: [Proxy](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") | None = None*, ) → None
:   Acquire a memory region modified in another proxy.

    Emits the uni-directional acquire proxy fence
    `fence.proxy.{to_proxy}::{from_proxy}.acquire.{scope} [addr], size`.
    The canonical use is acquiring a tensormap (`to_proxy="tensormap"`,
    `from_proxy="generic"`) that was edited in the generic proxy (e.g. via
    `tensormap.replace`) before reading it through the tensormap proxy with
    a TMA copy. Unlike most proxies, the tensormap proxy is **not** acquired
    from the generic proxy at kernel start, so this explicit fence is
    required whenever a tensormap is modified at runtime.

    Parameters:
    :   - **scope** ([*MemScope*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.experimental.primitives.nvvm_wrapper.MemScope")) – Scope at which the prior writes become visible
          (`"cta"` / `"cluster"` / `"gpu"` / `"sys"`).
        - **addr** (*Array* *|* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Base address of the region being acquired (e.g. the
          tensormap object). Generic-addressed; the runtime address must
          fall within the `.global` window (not enforced at trace time).
        - **size** (*int* *|* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *|* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Size of that region in bytes. The only value the
          instruction supports is `128` (the tensormap size), and it must
          be an immediate.
        - **from\_proxy** ([*Proxy*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") *|* *None*) – Source proxy the writes were performed in. Only
          `"generic"` is valid (the default); leave as `None` to use it.
        - **to\_proxy** ([*Proxy*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") *|* *None*) – Target proxy the subsequent reads use. Only
          `"tensormap"` is valid (the default); leave as `None` to use it.

    Raises:
    :   **ValueError** – a static `int` `size` other than `128`; an
        explicit `from_proxy` other than `"generic"`; or an explicit
        `to_proxy` other than `"tensormap"`.

    ```python
    # Acquire a runtime-edited tensormap before using it in a TMA copy:
    nvvm.fence_proxy_acquire(
        "gpu",
        tma_desc,
        128,
        from_proxy="generic",
        to_proxy="tensormap",
    )
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_shared_cta_global(
            smem, tma_desc, src_coords, mbar
        )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.fence\_proxy\_async\_acquire\_sync\_restrict() → None
:   `fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster`.

    Lowers to the `nvvm.fence.proxy.sync_restrict` op with `acquire` order;
    per its definition, `acquire` restricts the ordering to `shared::cluster`
    between the generic and async proxies.

cutlass.experimental.primitives.nvvm\_wrapper.fence\_proxy\_async\_release\_sync\_restrict() → None
:   `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster`.

    Lowers to the `nvvm.fence.proxy.sync_restrict` op with `release` order;
    per its definition, `release` restricts the ordering to `shared::cta`
    with cluster scope between the generic and async proxies.

cutlass.experimental.primitives.nvvm\_wrapper.fence\_proxy\_release( : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope")*, : *\**, : *from\_proxy: [Proxy](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") | None = None*, : *to\_proxy: [Proxy](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") | None = None*, ) → None
:   Release a memory region to another proxy.

    Emits the uni-directional release proxy fence
    `fence.proxy.{to_proxy}::{from_proxy}.release.{scope}`. It is the
    release counterpart of [`fence_proxy_acquire()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.fence_proxy_acquire "cutlass.experimental.primitives.nvvm_wrapper.fence_proxy_acquire"): a
    `fence.proxy.release` forms a release sequence that synchronises with
    an acquire sequence containing a matching `fence.proxy.acquire`. The
    canonical use is publishing a tensormap (`to_proxy="tensormap"`,
    `from_proxy="generic"`) edited in the generic proxy before another
    agent acquires and reads it through the tensormap proxy. Unlike
    [`fence_proxy_acquire()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.fence_proxy_acquire "cutlass.experimental.primitives.nvvm_wrapper.fence_proxy_acquire"), the release form takes no address window.

    Parameters:
    :   - **scope** ([*MemScope*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.experimental.primitives.nvvm_wrapper.MemScope")) – Scope at which the prior writes are released
          (`"cta"` / `"cluster"` / `"gpu"` / `"sys"`).
        - **from\_proxy** ([*Proxy*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") *|* *None*) – Source proxy the writes were performed in. Only
          `"generic"` is valid (the default); leave as `None` to use it.
        - **to\_proxy** ([*Proxy*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") *|* *None*) – Target proxy the subsequent reads use. Only
          `"tensormap"` is valid (the default); leave as `None` to use it.

    Raises:
    :   **ValueError** – an explicit `from_proxy` other than `"generic"`
        or an explicit `to_proxy` other than `"tensormap"`.

    ```python
    # Publish a runtime-edited tensormap to the tensormap proxy:
    nvvm.fence_proxy_release(
        "gpu",
        from_proxy="generic",
        to_proxy="tensormap",
    )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.fence\_proxy\_sync\_restrict( : *order: [MemOrder](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder")*, : *\**, : *from\_proxy: [Proxy](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") | None = None*, : *to\_proxy: [Proxy](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") | None = None*, ) → None
:   Order memory between the async and generic proxies, sync-restricted.

    Emits `fence.proxy.async::generic.{order}.sync_restrict...`. The
    `sync_restrict` qualifier narrows the ordering so that `acquire`
    applies to `shared::cluster` and `release` to `shared::cta`, both
    at cluster scope. Ordering is supported only between the async and
    generic proxies.

    Parameters:
    :   - **order** ([*MemOrder*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder")) – `"acquire"` or `"release"`.
        - **from\_proxy** ([*Proxy*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") *|* *None*) – Source proxy. Only `"generic"` is valid (the
          default); leave as `None` to use it.
        - **to\_proxy** ([*Proxy*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Proxy "cutlass.experimental.primitives.nvvm_wrapper.Proxy") *|* *None*) – Target proxy. Only `"async"` is valid (the
          default); leave as `None` to use it.

    Raises:
    :   **ValueError** – `order` is not `"acquire"` / `"release"`; an
        explicit `from_proxy` other than `"generic"`; or an explicit
        `to_proxy` other than `"async"`.

    ```python
    nvvm.fence_proxy_sync_restrict("acquire")
    ```

cutlass.experimental.primitives.nvvm\_wrapper.fence\_sc\_cluster() → None
:   Sequentially-consistent memory fence at cluster scope.

    Emits `fence.sc.cluster`. Contributes this thread’s prior memory
    accesses to a single total order over the sequentially-consistent
    operations observed by all threads in the cluster, ordering them before
    the thread’s subsequent accesses with respect to the whole cluster.

    ```python
    nvvm.fence_sc_cluster()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.fence\_sync\_restrict( : *order: [MemOrder](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder")*, ) → None
:   Thread fence restricted to a memory class at cluster scope.

    Emits `fence.{order}.sync_restrict::shared::{cluster|cta}.cluster`.
    The `sync_restrict` qualifier restricts `acquire` ordering to
    `shared::cluster` and `release` to `shared::cta`, both at cluster
    scope.

    Parameters:
    :   **order** ([*MemOrder*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder")) – `"acquire"` or `"release"`.

    Raises:
    :   **ValueError** – `order` is not `"acquire"` / `"release"`.

    ```python
    nvvm.fence_sync_restrict("release")
    ```

cutlass.experimental.primitives.nvvm\_wrapper.fma\_packed\_f32x2( : *src\_a: tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *src\_b: tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *src\_c: tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *\**, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *ftz: bool | None = None*, ) → tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   Wrapper over `nvvm.fma_packed_f32x2`.

    Accepts a 2-tuple of f32 scalars or a `Vector` for each operand and
    returns a tuple when called with tuples, else a `Vector`.

cutlass.experimental.primitives.nvvm\_wrapper.griddepcontrol( : *kind: [GridDepAction](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.GridDepAction "cutlass.experimental.primitives.nvvm_wrapper.GridDepAction")*, ) → None
:   Coordinate execution between consecutive dependent grids.

    Emits `griddepcontrol.{launch_dependents|wait}`. Used between back-to-back kernels that the runtime
    has wired with a producer/consumer dependency: the producer kernel
    issues `launch_dependents` to let the dependent kernel start as
    soon as scheduling permits, and the dependent kernel issues
    `wait` to ensure all prerequisite-grid memory operations have
    drained before reading.

    Parameters:
    :   **kind** ([*GridDepAction*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.GridDepAction "cutlass.experimental.primitives.nvvm_wrapper.GridDepAction")) – `"launch_dependents"` (producer side, hint that
        dependents may start) or `"wait"` (consumer side, block
        until prerequisites done).

    Raises:
    :   **ValueError** – if `kind` is not a [`GridDepAction`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.GridDepAction "cutlass.experimental.primitives.nvvm_wrapper.GridDepAction")
        member or its string value (raw NVVM dialect enums are rejected).

    ```python
    # Producer kernel — at the end, allow the consumer kernel to start
    if nvvm.elect_sync():
        nvvm.griddepcontrol("launch_dependents")

    # Consumer kernel — at the start, wait for producer's writes
    if nvvm.elect_sync():
        nvvm.griddepcontrol("wait")
    ```

cutlass.experimental.primitives.nvvm\_wrapper.inline\_ptx\_hl( : *ptx\_code: str*, : *\**, : *write\_only\_types: list | None = None*, : *read\_only\_args: list | None = None*, : *read\_write\_args: list | None = None*, : *pred: [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean") | None = None*, ) → object
:   Public high-level inline-PTX builder (`{$r0}` / `{$w0}` named refs, DSL
    `write_only_types`, built-in `@p` predication via `pred=`). Distinct
    from `inline_ptx`, which is the *raw* `nvvm.inline_ptx` op
    (`write_only_args` / positional `ptx_code`).

cutlass.experimental.primitives.nvvm\_wrapper.ldmatrix( : *ptr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *num: int*, : *layout: [MMALayout](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMALayout "cutlass.experimental.primitives.nvvm_wrapper.MMALayout")*, : *\**, : *shape: [LoadShape](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.LoadShape "cutlass.experimental.primitives.nvvm_wrapper.LoadShape") | None = None*, : *src\_format: [LoadSrcFormat](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.LoadSrcFormat "cutlass.experimental.primitives.nvvm_wrapper.LoadSrcFormat") | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   Warp-cooperative load of one to four 8x8 matrix tiles from shared memory.

    Emits `ldmatrix.sync.aligned.{shape}.{num}{.trans}{.ss}.{type} d, [a]`.
    All 32 lanes of the issuing warp collectively load `num` 8x8 tiles whose
    row starts each lane holds in `ptr`; the result is the per-thread
    fragment carried in 32-bit register words as required by subsequent
    `mma.sync` / `stmatrix` instructions.

    Lane addressing convention (per `num`):

    | num | lanes 0..7 | lanes 8..15 | lanes 16..31 |
    | --- | --- | --- | --- |
    | 1 2 4 | row starts of tile 0 rows of tile 0 rows of tile 0 | rows of tile 1 rows of tile 1 | rows of tiles 2..3 |

    Parameters:
    :   - **ptr** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer/Array into shared memory; per the PTX ISA the address
          space must be `.shared{::cta}`.
        - **num** (*int*) – Number of 8x8 tiles per warp. Must be one of `1`, `2`,
          `4` (the PTX `.x1` / `.x2` / `.x4` qualifiers).
        - **layout** ([*MMALayout*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMALayout "cutlass.experimental.primitives.nvvm_wrapper.MMALayout")) – `MMALayout.ROW` for the default load. `MMALayout.COL`
          selects `.trans`, which transposes the loaded tile inside the lane
          registers without reading from a transposed memory layout.
        - **shape** ([*LoadShape*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.LoadShape "cutlass.experimental.primitives.nvvm_wrapper.LoadShape") *or* *None*) – Tile shape selector. Defaults to `m8n8` (the historical
          SM75 form). `m8n16` and `m16n16` unpack the narrow-float
          `.dst_fmt.src_fmt` forms and require `src_format` to be set.
          `m16n16` additionally requires `layout=MMALayout.COL` (`.trans`).
        - **src\_format** ([*LoadSrcFormat*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.LoadSrcFormat "cutlass.experimental.primitives.nvvm_wrapper.LoadSrcFormat") *or* *None*) – Source packing for `m8n16` / `m16n16`.
          `b6x16_p32` / `b4x16_p64` unpack `e3m2` / `e2m1` matrices into
          the `.b8x16` destination format; `b8` is the byte form.
          Must be paired with one of `m8n16` / `m16n16`.

    Returns:
    :   `Int32` when `num=1`; `Vector[num x Int32]` when `num=2`
        or `num=4`. Each element is one 32-bit register word.

    Raises:
    :   - **ValueError** – `num` is not one of `1` / `2` / `4`.
        - **ValueError** – `src_format` is given without a matching
          `shape in {m8n16, m16n16}`, or `shape in {m8n16, m16n16}` is given
          without a matching `src_format`.
        - **ValueError** – `shape=m16n16` without `layout=MMALayout.COL`
          (the PTX ISA requires `.trans` for the `m16n16` form).

    ```python
    # 4 x 8x8 b16 tiles, non-transposed, into a vector<4xi32> fragment.
    smem = cutlass.Array(cutlass.Int16, 4 * 8 * 8, space=cutlass.AddressSpace.smem)
    regs = nvvm.ldmatrix(smem, num=4, layout=nvvm.MMALayout.ROW)
    # regs is Vector[4 x Int32], ready as a multiplicand for mma_sync.
    ```

cutlass.experimental.primitives.nvvm\_wrapper.load\_ext( : *addr: ~cutlass.Array | ~cutlass.Pointer*, : *\**, : *dtype: type | None = None*, : *count: int | None = None*, : *l2\_cache\_hint: int | ~cutlass.Int64 | ~cutlass.Uint64 | None = None*, : *order: ~cutlass.experimental.primitives.nvvm\_wrapper.MemOrder | None = None*, : *scope: ~cutlass.base\_dsl.array.MemScope | None = None*, : *prefetch: ~cutlass.base\_dsl.array.L2PrefetchSize | None = None*, : *evict: ~cutlass.base\_dsl.array.L1EvictKind | None = None*, : *cache\_modifier: ~cutlass.base\_dsl.array.LoadCacheModifier | None = None*, : *shared\_space: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject | None = None*, : *unified: bool | None = None*, ) → [Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | [Uint8](basic_data_types.md#cutlass.Uint8 "cutlass.Uint8") | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | [Int128](basic_data_types.md#cutlass.Int128 "cutlass.Int128") | [Uint128](basic_data_types.md#cutlass.Uint128 "cutlass.Uint128") | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32") | [Float64](basic_data_types.md#cutlass.Float64 "cutlass.Float64") | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   Load a scalar (or a `count`-element vector) from generic, global, or
    shared memory with explicit cache, eviction, and memory-ordering qualifiers
    (`nvvm.load.ext` / PTX `ld` / `ld.<vec>`).

    The element type is inferred from the pointer’s `dtype`; pass an explicit
    `dtype` to override (e.g. for untyped raw pointers). Pass `count` to
    load a vector (PTX `.v2`/`.v4`/`.v8`) and get a `Vector` back.
    The underlying op supports only `b8/b16/b32/b64/b128` integer widths and
    `f32`/`f64` floats: load a 16-bit float as `Int16` and bitcast it
    yourself.

    Parameters:
    :   - **addr** (*Array* *|* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Address to load from (generic, global, or shared pointer).
        - **dtype** (*type* *|* *None*) – DSL type of the loaded value; inferred from `addr.dtype`
          when omitted. Must be an 8/16/32/64/128-bit integer (signed or
          unsigned) or `Float32`/`Float64`; a 16-bit float must be loaded as
          `Int16` and bitcast.
        - **count** (*int* *|* *None*) – If set, load `count` elements as a `Vector` (PTX
          `.v2`/`.v4`/`.v8`); if omitted, load a scalar.
        - **l2\_cache\_hint** (*int* *|* [*Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *|* [*Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") *|* *None*) – 64-bit L2 cache-eviction policy handle (generic /
          global space only).
        - **order** ([*MemOrder*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder") *|* *None*) – Memory ordering (`weak` default, `relaxed`, `acquire`,
          `volatile`, `mmio`). `relaxed` / `acquire` require `scope`.
        - **scope** ([*MemScope*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.experimental.primitives.nvvm_wrapper.MemScope") *|* *None*) – Memory scope (`cta`, `cluster`, `gpu`, `sys`) for an
          ordered load.
        - **prefetch** ([*L2PrefetchSize*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.L2PrefetchSize "cutlass.experimental.primitives.nvvm_wrapper.L2PrefetchSize") *|* *None*) – L2 prefetch size hint (generic / global space only).
        - **evict** ([*L1EvictKind*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.L1EvictKind "cutlass.experimental.primitives.nvvm_wrapper.L1EvictKind") *|* *None*) – L1 eviction-priority hint; mutually exclusive with
          `cache_modifier`.
        - **cache\_modifier** ([*LoadCacheModifier*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.LoadCacheModifier "cutlass.experimental.primitives.nvvm_wrapper.LoadCacheModifier") *|* *None*) – Cache operator (`ca`/`cg`/`cs`/`lu`/`cv`);
          only valid on the default `weak` ordering.
        - **shared\_space** ([*SharedSpace*](cute_dsl_api/cute_nvgpu_common.md#cutlass.cute.nvgpu.SharedSpace "cutlass.cute.nvgpu.SharedSpace") *|* *None*) – Shared sub-space (`cta` default, `cluster` for
          distributed shared memory); for shared-space pointers only.
        - **unified** (*bool* *|* *None*) – Set the `.unified` qualifier (generic / global space
          only).

    Returns:
    :   The loaded scalar as the requested DSL type, or a `Vector`
        of `count` elements when `count` is set.

    Return type:
    :   [Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | [Uint8](basic_data_types.md#cutlass.Uint8 "cutlass.Uint8") | [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Uint16](basic_data_types.md#cutlass.Uint16 "cutlass.Uint16") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | [Int128](basic_data_types.md#cutlass.Int128 "cutlass.Int128") | [Uint128](basic_data_types.md#cutlass.Uint128 "cutlass.Uint128") | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32") | [Float64](basic_data_types.md#cutlass.Float64 "cutlass.Float64") | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")

    Raises:
    :   - **TypeError** – if `dtype` is omitted and `addr` carries no
          `dtype` to infer from.
        - **ValueError** – if the qualifier combination is illegal, e.g.
          `cache_modifier` with `evict` or with non-`weak` ordering;
          `relaxed`/`acquire` without `scope`; `volatile` with a cache
          op/hint or `unified` (`prefetch` is allowed); `mmio` without
          `scope=sys`; or `shared_space` combined with
          `l2_cache_hint`/`prefetch`/`unified`/`mmio`.

    ```python
    ptr = arr.data_ptr() + tx
    # Stream a global value through L2 only (bypass L1).
    v = nvvm.load_ext(ptr, dtype=cutlass.Int32,
                      cache_modifier=LoadCacheModifier.CG)
    # Vectorized load: 4 x f32 in one ld.global.v4.b32, with cache control.
    v4 = nvvm.load_ext(ptr, dtype=cutlass.Float32, count=4,
                       cache_modifier=LoadCacheModifier.CG)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.mapa( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *cta\_rank: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *addrspace: int = 7*, ) → Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")
:   Translate a local SMEM address to a peer CTA’s distributed-SMEM address.

    Emits `mapa.shared::cluster` — given a
    pointer to local shared memory and a peer CTA’s cluster rank, returns
    a pointer to the same SMEM offset in that peer’s local SMEM, valid
    in the cluster (`shared::cluster`) state space. The translation
    is a single-cycle hardware operation that exposes another CTA’s
    SMEM through the cluster interconnect. Passing `addrspace=0`
    selects the generic-addressing form (PTX `mapa` without `.space`),
    where both the source and result are generic addresses pointing to
    shared memory.

    Used to construct cluster-shared mbarrier pointers (so all
    participating CTAs can signal/wait on the same physical mbar) and
    for direct peer-CTA SMEM reads/writes that bypass GMEM.

    Parameters:
    :   - **addr** – Local-SMEM `Array`/`Pointer`. The same offset
          will be translated in the peer CTA’s SMEM.
        - **cta\_rank** – Cluster rank of the target peer CTA. Must be a
          valid rank within the launched cluster (`< cluster_size`);
          out-of-range values produce undefined results.
        - **addrspace** – Address space of the returned pointer. Default
          `7` is the distributed-shared (`shared::cluster`) space; pass
          `0` for the generic-addressing form. No other address space is
          representable. (A generic-addressed pointer to shared memory may be
          used with either form; the NVVM verifier enforces the exact rule.)

    Returns:
    :   Pointer/Array (matching input type) addressing the peer
        CTA’s SMEM at the same offset, in the requested address space.

    Raises:
    :   **ValueError** – if `addrspace` is neither `7`
        (shared::cluster) nor `0` (generic), or if a statically known
        `cta_rank` is negative.

cutlass.experimental.primitives.nvvm\_wrapper.match\_sync( : *thread\_mask: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *val: [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, : *kind: [MatchSync](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MatchSync "cutlass.experimental.primitives.nvvm_wrapper.MatchSync")*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | tuple[[Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32"), [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")]
:   1:1 wrapper over `nvvm.match_sync`.

    Returns `Int32` for `any`, `(Int32, Boolean)` for `all`.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_arrive( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope") | None = None*, : *relaxed: bool | None = None*, ) → [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | None
:   Decrement an mbarrier’s pending arrival count by `count` (default 1).

    When the total arrivals satisfy the barrier’s `count` threshold, the
    barrier fires: its internal parity flips and all waiters unblock.

    Parameters:
    :   - **addr** – Pointer/Array to the 64-bit SMEM mbarrier.
        - **count** –

          How many arrival credits to consume in one call. Defaults
          to 1 (the usual case). Pass `count=N` to let a single thread
          batch-arrive on behalf of N threads — useful for pre-signaling slots:

          ```python
          # Prologue: pre-signal all empty_bar slots so the producer's
          # first wait(parity=0) passes immediately.
          if nvvm.elect_sync():
              for i in cutlass.range_constexpr(NUM_STAGES):
                  nvvm.mbarrier_arrive(empty_bar + i)
          ```
        - **scope** – Memory-ordering scope for the arrive (default CTA).
        - **relaxed** – When `True` use `.relaxed` ordering instead of the
          default `.release`.

    Raises:
    :   - **TypeError** – `addr` is not in shared, shared::cluster, or generic
          memory.
        - **ValueError** – `count` is a Python `int` outside `[1, 2**20 - 1]`.

    **`mbarrier\_arrive` vs `mbarrier\_arrive\_expect\_tx`**: both count as a
    software arrive. Use `arrive_expect_tx` instead when a TMA load is
    involved — it additionally registers the byte count that TMA hardware must
    deliver via `complete_tx` before the barrier fires. For pure software
    producer-consumer pipelines (non-TMA), use `mbarrier_arrive`.

    Returns:
    :   Opaque 64-bit state token for shared/generic pointers; `None`
        for cluster-space pointers.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_arrive\_drop( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope") | None = None*, : *relaxed: bool | None = None*, ) → [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | None
:   1:1 wrapper over `nvvm.mbarrier_arrive_drop`.

    Returns an opaque 64-bit state token for shared/generic pointers.
    Returns None for shared::cluster pointers.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_arrive\_drop\_expect\_tx( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *txcount: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope") | None = None*, : *relaxed: bool | None = None*, ) → [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | None
:   1:1 wrapper over `nvvm.mbarrier_arrive_drop_expect_tx`.

    Returns an opaque 64-bit state token for shared/generic pointers.
    Returns None for shared::cluster pointers.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_arrive\_drop\_nocomplete( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, ) → [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")
:   Wrapper over `nvvm.mbarrier_arrive_drop_nocomplete`.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_arrive\_expect\_tx( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *txcount: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope") | None = None*, : *relaxed: bool | None = None*, ) → [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | None
:   Signal an mbarrier’s TMA transaction count and count as the software arrive.

    Used in TMA pipelines where `mbarrier_init(bar, count=1)` was used:
    this call serves as the single software arrive *and* registers `txcount`
    bytes that TMA hardware must deliver via `complete_tx` before the barrier
    fires. No separate `mbarrier_arrive` call is needed.

    Parameters:
    :   - **addr** – Pointer/Array to the 64-bit SMEM mbarrier.
        - **txcount** –

          Bytes that TMA `complete_tx` signals must deliver before
          the barrier fires. The interpretation depends on how many producers
          share this barrier:

          - **Single producer** (`mbarrier_init(bar, count=1)`): one call,
            pass the *total* bytes for all TMA loads sharing this barrier:
            `txcount = A_bytes + B_bytes`. The barrier fires once TMA
            delivers all bytes.
          - **Two producers** (`mbarrier_init(bar, count=2)`): each producer
            calls `arrive_expect_tx` independently with *its own share only*
            (producer-A passes `tx_A`, producer-B passes `tx_B`). The
            barrier fires after both arrives **and** `tx_A + tx_B` bytes
            have been delivered.

          Formula per load: `num_rows * num_cols * sizeof(dtype)`.

    Raises:
    :   - **ValueError** – `txcount` is a negative Python `int`.
        - **TypeError** – `addr` is not in shared, shared::cluster, or generic
          memory.

    **Must be called before** the corresponding `cp_async_bulk_tensor_*`
    call(s), so the transaction counter is set before TMA can decrement it.

    **Call from one thread only** — either `tx == 0` or a single elected
    thread (`nvvm.elect_sync()`). For CTA\_2 multicast: leader only;
    multiply `txcount` by the number of CTAs in the cluster.

    Returns:
    :   Opaque 64-bit state token for shared/generic pointers; `None`
        for cluster-space pointers.

    ```python
    # Single producer: separate elect_sync for arrive vs TMA loads (performance)
    if nvvm.elect_sync():
        nvvm.mbarrier_arrive_expect_tx(full_bar + s, A_bytes + B_bytes)
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_shared_cta_global(sA, tma_a, full_bar + s, coord_a)
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_shared_cta_global(sB, tma_b, full_bar + s, coord_b)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_arrive\_nocomplete( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, ) → [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")
:   Wrapper over `nvvm.mbarrier_arrive_nocomplete`.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_complete\_tx( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *txcount: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope") | None = None*, ) → None
:   Manually report completed async-copy bytes to an mbarrier (SM90+).

    Decrements the barrier’s expected transaction count by `txcount` bytes
    without issuing a software arrive. Use this when the hardware does not
    deliver the `complete_tx` signal automatically — for example, after a
    `cp_async_shared_global` pipeline that does *not* use TMA.

    For TMA-based pipelines (`cp_async_bulk_tensor_*`), the TMA hardware
    delivers `complete_tx` automatically when the copy finishes — do **not**
    call `mbarrier_complete_tx` in that case (double-counting corrupts the
    barrier state).

    Parameters:
    :   - **addr** – Pointer/Array to the 64-bit SMEM mbarrier.
        - **txcount** – Number of bytes to report as completed. Must equal the
          total bytes delivered by the corresponding async copies.
        - **scope** – Memory scope (default: `cta`).

    Raises:
    :   - **ValueError** – `txcount` is a negative Python `int`.
        - **TypeError** – `addr` is not in shared, shared::cluster, or generic
          memory.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_expect\_tx( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *txcount: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope") | None = None*, ) → None
:   Wrapper over `nvvm.mbarrier_expect_tx`.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_init( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *count: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, ) → None
:   Initialize a 64-bit mbarrier object in shared memory.

    Sets the expected-arrival count and resets the barrier’s phase to 0.
    Lowers to `mbarrier.init{.shared{::cta}}.b64 [addr], count`.

    Parameters:
    :   - **addr** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer or Array addressing the 64-bit mbarrier object.
          Must resolve to shared memory at runtime. `cutlass.AddressSpace.smem`
          is preferred (NVVM emits the `.shared` variant with a 32-bit
          address operand); `cutlass.AddressSpace.generic` is also accepted and
          emits the generic-form `mbarrier.init.b64` instruction, which
          requires the runtime address to fall within the `.shared::cta`
          window (behavior is undefined otherwise).
          Typically allocated via
          `cutlass.Array(cutlass.Int64, N, space=cutlass.AddressSpace.smem)`.
        - **count** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Expected arrival count: how many `mbarrier_arrive` (or
          `mbarrier_arrive_expect_tx`) calls must be satisfied before the
          barrier fires and flips its phase. Valid range is
          `[1, 2**20 - 1]`. Constexpr `int` values are checked at trace
          time; dynamic values get a `--enable-assertions`-gated runtime
          check. For TMA pipelines use `1`: one elected thread calls
          `arrive_expect_tx` and TMA hardware delivers `complete_tx`.

    Raises:
    :   - **TypeError** – `addr` is in an address space other than
          `SHARED` or `GENERIC` (e.g. global, local, tensor memory, or
          `SHARED_CLUSTER`: the PTX ISA does not define `mbarrier.init`
          for these).
        - **ValueError** – `count` is a Python `int` outside `[1, 2**20 - 1]`.

    ```python
    # Preferred: one warp initializes one disjoint mbarrier group.
    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane_idx = tidx & 31

    if warp_idx == 0:
        if lane_idx < NUM_AB_STAGES:
            nvvm.mbarrier_init(ab_full_bar + lane_idx, 1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    elif warp_idx == 1:
        if lane_idx < NUM_AB_STAGES:
            nvvm.mbarrier_init(ab_empty_bar + lane_idx, ab_empty_count)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    elif warp_idx == 2:
        if lane_idx < NUM_SF_STAGES:
            nvvm.mbarrier_init(sf_full_bar + lane_idx, 1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    elif warp_idx == 3:
        if lane_idx < NUM_SF_STAGES:
            nvvm.mbarrier_init(sf_empty_bar + lane_idx, sf_empty_count)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    elif warp_idx == 4:
        if lane_idx < 3:
            count = 1 if lane_idx == 0 else 8 if lane_idx == 1 else 32
            nvvm.mbarrier_init(aux_bar + lane_idx, count)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    ```

    ```python
    # Fallback: simple elected-thread init for a few barriers.
    if warp_idx == 0:
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(NUM_STAGES):
                nvvm.mbarrier_init(full_bar + i, 1)
                nvvm.mbarrier_init(empty_bar + i, 1)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_inval(*addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*) → None
:   Invalidate an mbarrier object so its storage can be reused.

    Emits `mbarrier.inval`. Marks the mbarrier at `addr` invalid; the
    underlying shared-memory bytes may then be repurposed. Pair with
    [`mbarrier_init()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.mbarrier_init "cutlass.experimental.primitives.nvvm_wrapper.mbarrier_init") to recreate a barrier in the same storage.

    Parameters:
    :   **addr** (*Array* *|* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer to the mbarrier. Must reside in shared memory
        (generic addressing into `.shared` is also accepted).

    Raises:
    :   **TypeError** – `addr` is a typed operand in an address space other
        than shared or generic.

    ```python
    if nvvm.elect_sync():
        nvvm.mbarrier_inval(mbar)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_test\_wait( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *state\_or\_phase: [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, : *\**, : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope") | None = None*, : *relaxed: bool | None = None*, ) → [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Test whether an mbarrier phase has completed.

    Returns the PTX `waitComplete` predicate for `mbarrier.test_wait`.
    The PTX ISA 9.3 primary-phase form can also produce `reportPredicate` and
    `reportValue` operands; this wrapper intentionally exposes only the
    completion boolean.

    Parameters:
    :   - **addr** – Pointer/Array to the 64-bit mbarrier object.
        - **state\_or\_phase** – State token returned by `mbarrier_arrive` or a
          parity value for parity-style waits.
        - **scope** – Optional memory scope when using explicit acquire/relaxed
          semantics.
        - **relaxed** – Emit relaxed ordering when `True`; omit for default
          acquire semantics.

    Returns:
    :   `True` when the requested phase has completed.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_try\_wait( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *state\_or\_phase: [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, : *\**, : *ticks: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope") | None = None*, : *relaxed: bool | None = None*, ) → [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Try to wait for an mbarrier phase, optionally with a time hint.

    Returns the PTX `waitComplete` predicate for `mbarrier.try_wait`. If the
    phase has not completed, the executing thread may suspend for up to
    `ticks` nanoseconds, or for an implementation-defined time when `ticks`
    is omitted. PTX ISA 9.3 primary-phase report operands are not exposed by
    this wrapper.

    Parameters:
    :   - **addr** – Pointer/Array to the 64-bit mbarrier object.
        - **state\_or\_phase** – State token returned by `mbarrier_arrive` or a
          parity value for parity-style waits.
        - **ticks** – Optional time hint in nanoseconds.
        - **scope** – Optional memory scope when using explicit acquire/relaxed
          semantics.
        - **relaxed** – Emit relaxed ordering when `True`; omit for default
          acquire semantics.

    Returns:
    :   `True` when the requested phase has completed.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_try\_wait\_parity( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *phase: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *time\_limit: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") = 10000000*, : *scope: [MBarrierScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MBarrierScope "cutlass.experimental.primitives.nvvm_wrapper.MBarrierScope") | None = None*, : *order: [MemOrder](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder") | None = None*, ) → [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Attempt a parity try-wait on an mbarrier phase.

    This issues one `mbarrier.try_wait.parity` attempt. If the phase has not
    completed, the executing thread may be hardware-suspended for up to
    `time_limit` nanoseconds before returning `False`. The caller is
    responsible for retrying when a blocking wait is needed.

    Parameters:
    :   - **addr** – Pointer/Array to the 64-bit SMEM mbarrier object.
        - **phase** –

          Parity value to wait **against**. Returns `True` once the
          barrier’s internal parity differs from `phase` (i.e. the barrier
          fired and advanced its phase). Returns `False` on timeout.

          - Fresh barrier starts at parity **0**.
          - Pass `phase=0` to block until the first arrival: waits because
            current parity (0) equals `phase` (0).
          - Pass `phase=1` on a fresh barrier to pass immediately: current
            parity (0) ≠ `phase` (1), so no waiting needed.
        - **time\_limit** – Hardware suspend timeout in nanoseconds. Defaults to
          `10_000_000` (10 ms). Omit it unless you need a different suspend
          window. The warp may be hardware-suspended for up to this many
          nanoseconds on each call.

    Returns:
    :   `True` when the barrier phase has advanced past `phase`;
        `False` if `time_limit` expired without completion.

    Raises:
    :   - **ValueError** – `phase` is a Python `int` other than 0 or 1.
        - **ValueError** – `time_limit` is a negative Python `int`.
        - **TypeError** – `addr` is not in shared or generic memory.

    **Always wrap in a while loop** — the retry is the caller’s responsibility:

    ```python
    while not nvvm.mbarrier_try_wait_parity(bar + s, parity):
        pass
    ```

    This lowers to `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`
    with explicit `.acquire.cta` ordering (ensures TMA writes are visible
    after the wait). The same intrinsic without the `.acquire.cta`
    qualifier has weaker ordering and benchmarks ~5% slower, so avoid it.

    **Phase formula for a circular N-stage pipeline at iteration k:**

    ```python
    cons_parity = (k // cutlass.Int32(NUM_STAGES)) & cutlass.Int32(1)
    while not nvvm.mbarrier_try_wait_parity(full_bar + s, cons_parity):
        pass
    ```

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_try\_wait\_timelimit( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *state: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*, : *time\_limit: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *scope: [MBarrierScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MBarrierScope "cutlass.experimental.primitives.nvvm_wrapper.MBarrierScope") | None = None*, : *order: [MemOrder](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder") | None = None*, ) → [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Try to wait for a state-token mbarrier phase with an explicit time limit.

    This is the non-parity time-limited `mbarrier.try_wait` form. It returns
    only the PTX `waitComplete` predicate; PTX ISA 9.3
    `reportPredicate`/`reportValue` operands are not exposed here.

    Parameters:
    :   - **addr** – Pointer/Array to the 64-bit mbarrier object.
        - **state** – State token returned by a previous mbarrier arrive operation.
        - **time\_limit** – Time hint in nanoseconds before the suspended thread may
          resume and return `False`.
        - **scope** – Optional memory scope for the wait.
        - **order** – Optional memory-order qualifier.

    Returns:
    :   `True` when the state-token phase has completed; `False` when
        the time limit expires first.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_wait( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *state: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*, : *kind: [MBarrierWait](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MBarrierWait "cutlass.experimental.primitives.nvvm_wrapper.MBarrierWait")*, : *\**, : *scope: [MBarrierScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MBarrierScope "cutlass.experimental.primitives.nvvm_wrapper.MBarrierScope") | None = None*, : *order: [MemOrder](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder") | None = None*, ) → [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Wrapper over `nvvm.mbarrier_wait`.

cutlass.experimental.primitives.nvvm\_wrapper.mbarrier\_wait\_parity( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *phase: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *kind: [MBarrierWait](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MBarrierWait "cutlass.experimental.primitives.nvvm_wrapper.MBarrierWait")*, : *\**, : *scope: [MBarrierScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MBarrierScope "cutlass.experimental.primitives.nvvm_wrapper.MBarrierScope") | None = None*, : *order: [MemOrder](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder") | None = None*, ) → [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Test- or try-wait on an mbarrier’s parity (no time-limit form).

    Returns the `waitComplete` predicate: `True` once the barrier’s phase
    has advanced past `phase` (its internal parity differs), else `False`.

    Parameters:
    :   - **addr** – Pointer/Array to the 64-bit mbarrier, in shared or generic
          memory.
        - **phase** – Parity to wait against (`0` or `1`). Completes when the
          barrier’s internal parity differs from `phase`; a fresh barrier
          starts at parity 0.
        - **kind** – `MBarrierWait.TEST` selects `mbarrier.test_wait.parity`
          (a non-blocking check that never suspends); `MBarrierWait.TRY`
          selects `mbarrier.try_wait.parity` (may hardware-suspend the thread
          until the phase completes or an implementation-defined limit). Both
          return the `waitComplete` predicate, so a `TRY` wait is wrapped in
          a retry loop.
        - **scope** – Memory-ordering scope (default CTA).
        - **order** – Memory ordering; the underlying op supports only `acquire`
          (the effective default) or `relaxed`.

    Raises:
    :   **ValueError** – `phase` is a Python `int` other than 0 or 1.

    Returns:
    :   `True` when the barrier phase has advanced past `phase`;
        `False` otherwise.

    ```python
    # Blocking consumer wait (TRY): retry until the phase flips.
    while not nvvm.mbarrier_wait_parity(bar, parity, nvvm.MBarrierWait.TRY):
        pass

    # Non-blocking probe (TEST): single check, never suspends.
    done = nvvm.mbarrier_wait_parity(bar, parity, nvvm.MBarrierWait.TEST)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.memory\_barrier(*scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope")*) → None
:   Order this thread’s memory accesses at the given scope.

    Emits `membar.{scope}`. Guarantees that the issuing thread’s prior
    memory accesses are performed at `scope` before any of its subsequent
    accesses. This is the legacy `membar` ordering primitive; prefer
    [`fence_proxy()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.fence_proxy "cutlass.experimental.primitives.nvvm_wrapper.fence_proxy") / the acquire-release atomics for finer control.

    Parameters:
    :   **scope** ([*MemScope*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.experimental.primitives.nvvm_wrapper.MemScope")) – Scope at which the ordering is observed
        (`"cta"` / `"cluster"` / `"gpu"` / `"sys"`).

    ```python
    nvvm.memory_barrier("gpu")
    ```

cutlass.experimental.primitives.nvvm\_wrapper.mma\_block\_scale( : *res: Any*, : *shape: Any*, : *scale\_vec\_size: Any*, : *block\_scale\_format: Any*, : *kind: Any*, : *\*args: Any*, : *\*\*kwargs: Any*, ) → Any
:   Gated 1:1 wrapper over `nvvm.mma.block_scale`.

    The `.scale_vec::4X` + `.ue8m0` scale type + `.kind::mxf4nvf4`
    combination was introduced in PTX ISA 9.1 and is unavailable on CTK 12.9
    (PTX ISA 8.8); every other combination predates it, so only that one is
    gated.

cutlass.experimental.primitives.nvvm\_wrapper.mma\_smem\_desc( : *pointer: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *ldm: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *stride: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *base\_offset: int | [Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | [Uint8](basic_data_types.md#cutlass.Uint8 "cutlass.Uint8")*, : *swizzle: int | [Int8](basic_data_types.md#cutlass.Int8 "cutlass.Int8") | [Uint8](basic_data_types.md#cutlass.Uint8 "cutlass.Uint8")*, : *\**, : *mma\_desc\_version: int | None = None*, ) → [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")
:   Wrapper over `nvvm.mma_smem_desc`.

cutlass.experimental.primitives.nvvm\_wrapper.mma\_sp\_block\_scale( : *res: Any*, : *shape: Any*, : *scale\_vec\_size: Any*, : *block\_scale\_format: Any*, : *kind: Any*, : *\*args: Any*, : *\*\*kwargs: Any*, ) → Any
:   Gated 1:1 wrapper over `nvvm.mma.sp.block_scale`.

    The `.scale_vec::4X` + `.ue8m0` scale type + `.kind::mxf4nvf4`
    combination was introduced in PTX ISA 9.1 and is unavailable on CTK 12.9
    (PTX ISA 8.8); every other combination predates it, so only that one is
    gated.

cutlass.experimental.primitives.nvvm\_wrapper.mma\_sp\_sync( : *res: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocMlirType*, : *shape: ~sphinx.ext.autodoc.mock.\_MockObject*, : *operand\_a: ir.Value*, : *operand\_b: ir.Value*, : *operand\_c: ir.Value*, : *sparse\_metadata: int | ~cutlass.Int32 | ~cutlass.Uint32*, : *sparsity\_selector: int | ~cutlass.Int32 | ~cutlass.Uint32*, : *\**, : *int\_overflow\_behavior: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAIntOverflow | None = None*, : *multiplicand\_a\_ptx\_type: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAType | None = None*, : *multiplicand\_b\_ptx\_type: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAType | None = None*, : *ordered\_metadata: bool | None = None*, : *kind: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAKind | None = None*, ) → ir.Value
:   Wrapper over `nvvm.mma_sp_sync`.

    Returns an LLVM struct. Caller provides the raw MLIR result type
    as *res*.

cutlass.experimental.primitives.nvvm\_wrapper.mma\_sync(*res: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocMlirType, shape: ~sphinx.ext.autodoc.mock.\_MockObject | tuple[int, int, int] | dict, layout\_a: ~cutlass.experimental.primitives.nvvm\_wrapper.MMALayout, layout\_b: ~cutlass.experimental.primitives.nvvm\_wrapper.MMALayout, operand\_a: list[ir.Value], operand\_b: list[ir.Value], operand\_c: list[ir.Value], \*, b1\_op: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAB1Op | None = None, int\_overflow\_behavior: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAIntOverflow | None = None, multiplicand\_a\_ptx\_type: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAType | None = None, multiplicand\_b\_ptx\_type: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAType | None = None*) → Any
:   Cooperative warp-wide matrix multiply-accumulate (`D = A*B + C`).

    Emits `mma.sync.aligned.{shape}.{alayout}.{blayout}{.kind}.{dtype}.{atype}.{btype}.{ctype}`.
    All 32 lanes of the issuing warp collectively compute one MMA on the
    fragments distributed across their registers; the four matrices are sliced
    across the warp per the PTX ISA’s per-shape fragment layout.

    The wrapper is a 1:1 mapping over the NVVM dialect `nvvm.mma.sync` op.
    The dialect verifier validates the shape/type/operand-count combination
    (e.g. `m16n8k16.f16` requires 4 `f16x2` `a`, 2 `f16x2` `b`, and
    2 `f16x2` or 4 `f32` `c`/`d` registers). This wrapper adds string/StrEnum
    coercion for the qualifier attributes and trace-time guards for the
    qualifier-vs-multiplicand-type coupling that the dialect cannot infer.

    Parameters:
    :   - **res** (*ir.Type*) – MLIR result type of the fragment `D` — typically an
          `llvm.struct<...>` whose element layout matches the per-thread
          fragment for the given `shape` x type combination.
        - **shape** (*ir.Attribute* *or* *tuple**[**int**,* *int**,* *int**] or* *dict*) – MMA shape attribute. Either a pre-built `ir.Attribute`
          (`#nvvm.shape<m = ..., n = ..., k = ...>`), a 3-tuple
          `(m, n, k)`, or a `{"m": ..., "n": ..., "k": ...}` dict.
        - **layout\_a** ([*MMALayout*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMALayout "cutlass.experimental.primitives.nvvm_wrapper.MMALayout")) – Layout of multiplicand A. Usually `MMALayout.ROW`.
          `MMALayout.COL` is only legal for `mma.m8n8k4.f16`.
        - **layout\_b** ([*MMALayout*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMALayout "cutlass.experimental.primitives.nvvm_wrapper.MMALayout")) – Layout of multiplicand B. Usually `MMALayout.COL`.
          `MMALayout.ROW` is only legal for `mma.m8n8k4.f16`.
        - **operand\_a** – Per-thread fragment of A as a sequence of `ir.Value`.
        - **operand\_b** – Per-thread fragment of B as a sequence of `ir.Value`.
        - **operand\_c** – Per-thread fragment of accumulator C as a sequence of
          `ir.Value`.
        - **b1\_op** ([*MMAB1Op*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMAB1Op "cutlass.experimental.primitives.nvvm_wrapper.MMAB1Op") *or* *None*) – Bit-op selector for single-bit (`.b1`) multiplicands;
          `MMAB1Op.XOR_POPC` for `mma.xor.popc` (default for `.b1`),
          `MMAB1Op.AND_POPC` for `mma.and.popc`. Only valid when both
          multiplicand types are `.b1`.
        - **int\_overflow\_behavior** ([*MMAIntOverflow*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMAIntOverflow "cutlass.experimental.primitives.nvvm_wrapper.MMAIntOverflow") *or* *None*) – Accumulator overflow handling for integer
          multiplicands (`.u8` / `.s8` / `.u4` / `.s4`).
          `MMAIntOverflow.SATFINITE` clamps to the `s32` range (PTX
          `.satfinite` modifier); `MMAIntOverflow.WRAPPED` wraps modulo
          `2**32`.
        - **multiplicand\_a\_ptx\_type** ([*MMAType*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMAType "cutlass.experimental.primitives.nvvm_wrapper.MMAType") *or* *None*) – Element type of multiplicand A as a
          `MMAType`. Defaults to the type the dialect infers from
          `operand_a`’s MLIR type; pass it explicitly when the operand
          carrier type does not uniquely determine the PTX element type
          (e.g. `i32` carriers for packed `s8` / `u8` / `f8` data).
        - **multiplicand\_b\_ptx\_type** ([*MMAType*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMAType "cutlass.experimental.primitives.nvvm_wrapper.MMAType") *or* *None*) – Element type of multiplicand B as a
          `MMAType`. Same inference rules as `multiplicand_a_ptx_type`.

    Raises:
    :   - **ValueError** – `b1_op` is given but `multiplicand_{a,b}_ptx_type`
          is set to a non-`.b1` type.
        - **ValueError** – `int_overflow_behavior` is given but
          `multiplicand_{a,b}_ptx_type` is set to a non-integer type.
        - **ValueError** – `shape` is a tuple/list of the wrong arity or with
          non-`int` entries, or a dict missing the required `m`/`n`/`k`
          keys.

    ```python
    # m16n8k16 f16 = f16 * f16 + f16
    d = nvvm.mma_sync(
        T.struct([T.vector(2, T.f16())] * 2),
        shape=(16, 8, 16),
        layout_a=nvvm.MMALayout.ROW,
        layout_b=nvvm.MMALayout.COL,
        operand_a=[a0, a1, a2, a3],   # 4 x f16x2
        operand_b=[b0, b1],           # 2 x f16x2
        operand_c=[c0, c1],           # 2 x f16x2
    )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.mov\_b32( : *a: int | float | ~cutlass.Int32 | ~cutlass.Uint32 | ~cutlass.Float32*, : *\**, : *target\_type: type = <class 'cutlass.Int32'>*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")
:   `mov.b32` — reinterpret a 32-bit value’s bits as *target\_type*.

    Emits an `arith.bitcast` (no value conversion), e.g. float bits → int
    for NaN-safe integer compares.

cutlass.experimental.primitives.nvvm\_wrapper.mul( : *a: [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, : *b: [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, : *mode: [MulMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MulMode "cutlass.experimental.primitives.nvvm_wrapper.MulMode")*, : *\**, : *is\_signed: bool | None = None*, ) → [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")
:   Wrapper over `nvvm.mul`.

cutlass.experimental.primitives.nvvm\_wrapper.mul\_bf16x2(*a: [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, *b: [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   `mul.bf16x2` — packed bf16x2 multiply (two bf16 lanes packed in i32).

cutlass.experimental.primitives.nvvm\_wrapper.mul\_packed\_f32x2( : *src\_a: tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *src\_b: tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *\**, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *ftz: bool | None = None*, ) → tuple | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   Wrapper over `nvvm.mul_packed_f32x2`.

    Accepts a 2-tuple of f32 scalars or a `Vector` for each operand and
    returns a tuple when called with tuples, else a `Vector`.

cutlass.experimental.primitives.nvvm\_wrapper.nanosleep(*duration: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*) → None
:   Wrapper over `nvvm.nanosleep`.

cutlass.experimental.primitives.nvvm\_wrapper.pmevent( : *event\_id: int | None = None*, : *\**, : *mask: int | None = None*, ) → None
:   Trigger one or more performance-monitor events.

    Emits `pmevent` (single event) or `pmevent.mask` (a set of events).
    Exactly one of `event_id` / `mask` must be given.

    Parameters:
    :   - **event\_id** (*int* *|* *None*) – Single event index in `[0, 15]` (`pmevent`).
        - **mask** (*int* *|* *None*) – 16-bit mask selecting a set of events (`pmevent.mask`);
          bit `i` triggers event `i`.

    Raises:
    :   **ValueError** – neither or both of `event_id` / `mask` are given,
        `event_id` is outside `[0, 15]`, or `mask` is outside
        `[0, 0xFFFF]`.

    ```python
    nvvm.pmevent(event_id=3)
    nvvm.pmevent(mask=0b1010)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.prefetch\_l1(*addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*) → None
:   Bring a cache line into L1.

    Emits `prefetch.L1 [addr]` (or `prefetch.global.L1` /
    `prefetch.local.L1` when the address space is statically
    visible). Per-thread, non-collective; the cache line is warm
    but no register is loaded — subsequent reads of `addr` still
    have to issue an `ld`.

    For TMA descriptor warm-up use [`prefetch_tensormap()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.prefetch_tensormap "cutlass.experimental.primitives.nvvm_wrapper.prefetch_tensormap"); for
    the uniform-cache hint use [`prefetchu()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.prefetchu "cutlass.experimental.primitives.nvvm_wrapper.prefetchu"); for an L2 warm-up
    (with optional eviction-priority hint) use [`prefetch_l2()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.prefetch_l2 "cutlass.experimental.primitives.nvvm_wrapper.prefetch_l2").

    For conditional execution wrap the call: `if pred:
    nvvm.prefetch_l1(addr)` — the dialect does support a PTX
    `@p prefetch` guard, but that lowering is undocumented and
    equivalent to the explicit `if` for every observable effect.

    Parameters:
    :   **addr** (*Array* *or* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer/Array in generic, global, or local space.
        Prefetch on SMEM is a no-op per the PTX ISA.

    ```python
    # Warm L1 ahead of a global load
    nvvm.prefetch_l1(gmem_ptr)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.prefetch\_l2( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *evict\_priority: [EvictPriority](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.EvictPriority "cutlass.base_dsl.array.EvictPriority") | None = None*, ) → None
:   Bring a cache line into L2, optionally with eviction-priority hint.

    Emits `prefetch.L2 [addr]` (or
    `prefetch.global.L2::<priority> [addr]` when
    `evict_priority` is set). Use to warm L2 before a TMA
    descriptor read or to bias L2 replacement against re-fetched
    data with `"last"`.

    For conditional execution wrap the call in `if`; the same
    note on [`prefetch_l1()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.prefetch_l1 "cutlass.experimental.primitives.nvvm_wrapper.prefetch_l1") applies.

    Parameters:
    :   - **addr** (*Array* *or* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer/Array in generic, global, or local space.
          Prefetch on SMEM is a no-op per the PTX ISA.
        - **evict\_priority** ([*EvictPriority*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.EvictPriority "cutlass.experimental.primitives.nvvm_wrapper.EvictPriority")*,* *optional*) – `"normal"` or `"last"` (the only two policies
          the `prefetch` instruction supports), or `None` (default) to leave
          the default policy. Maps to the PTX `.L2::evict_normal` /
          `.L2::evict_last` modifier with the `evict_` prefix dropped, to
          match the rest of the memory-model API. Other `EvictPriority`
          members are valid on `ld` / `st` / `cp` but rejected here.

    Raises:
    :   **ValueError** – if `evict_priority` is neither `"normal"` nor
        `"last"`.

    ```python
    # Warm L2 with an eviction hint
    nvvm.prefetch_l2(gmem_ptr, evict_priority="last")
    ```

cutlass.experimental.primitives.nvvm\_wrapper.prefetch\_tensormap( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *space: [TensormapSpace](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapSpace "cutlass.experimental.primitives.nvvm_wrapper.TensormapSpace") = 'const'*, ) → None
:   Warm the TMA tensormap descriptor cache.

    Emits `prefetch.tensormap [addr]` (default `space="const"`)
    or `prefetch.param.tensormap [addr]` (`space="param"`).
    Issue once, on a single thread, before the first
    `cp.async.bulk.tensor` that consumes the descriptor — cuts the
    first TMA’s launch latency by hiding the descriptor fetch behind
    independent work.

    The canonical pattern is `if nvvm.elect_sync(): prefetch_tensormap(...)`
    — the explicit `if` is the recommended way to make the prefetch
    conditional; the dialect’s undocumented `@p` guard is equivalent.

    Typical pattern (one lane warms each descriptor):

    ```python
    if nvvm.elect_sync():
        nvvm.prefetch_tensormap(tma_desc_a.get_ptr())
        nvvm.prefetch_tensormap(tma_desc_b.get_ptr())
    ```

    Parameters:
    :   - **addr** (*Array* *or* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer to the tensormap descriptor.
        - **space** ([*TensormapSpace*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapSpace "cutlass.experimental.primitives.nvvm_wrapper.TensormapSpace")) – `"const"` (default) or `"param"` — the state
          space the descriptor lives in. `"param"` is the
          kernel-argument case; `"const"` covers the typical
          `__constant__` / `cutlass.GridConstant` case.

cutlass.experimental.primitives.nvvm\_wrapper.prefetchu(*addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*) → None
:   Prefetch into the uniform L1 cache.

    Emits `prefetchu.L1 [addr]`. The uniform cache backs
    addresses that all lanes in a warp agree on (e.g. constants,
    kernel parameters) and is separate from the per-thread L1
    data cache. Use when a uniformly-addressed value is about
    to be read by many warps.

    For conditional execution wrap the call in `if`; the same
    note on [`prefetch_l1()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.prefetch_l1 "cutlass.experimental.primitives.nvvm_wrapper.prefetch_l1") applies.

    Parameters:
    :   **addr** (*Array* *or* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer/Array to a uniformly-addressed location.

    ```python
    nvvm.prefetchu(param_ptr)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.prmt( : *lo: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *selector: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *mode: [PermuteMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.PermuteMode "cutlass.experimental.primitives.nvvm_wrapper.PermuteMode")*, : *\**, : *hi: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   Wrapper over `nvvm.prmt`.

cutlass.experimental.primitives.nvvm\_wrapper.read\_sreg\_hw( : *num: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")
:   Wrapper over `nvvm.read_sreg_hw`.

cutlass.experimental.primitives.nvvm\_wrapper.red( : *op: ~cutlass.experimental.primitives.nvvm\_wrapper.ReductionOp*, : *type\_: ~cutlass.experimental.primitives.nvvm\_wrapper.ReductionType*, : *a: ~cutlass.Array | ~cutlass.Pointer*, : *b: ~cutlass.Int32 | ~cutlass.Int64 | ~cutlass.Float64 | ~cutlass.BFloat16 | ~cutlass.Float16 | ~cutlass.Float32 | ~cutlass.Vector*, : *\**, : *mem\_order: ~cutlass.experimental.primitives.nvvm\_wrapper.MemOrder | None = None*, : *mem\_scope: ~cutlass.base\_dsl.array.MemScope | None = None*, : *shared\_space: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject | None = None*, : *cache\_hint: int | ~cutlass.Int64 | ~cutlass.Uint64 | None = None*, ) → None
:   Apply a non-returning atomic reduction to a global or shared memory cell.

    Emits the PTX `red` instruction family. The value in memory at `a` is
    combined with operand `b` using `op` and the result is written back to
    `a`. Unlike [`atomicrmw()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.atomicrmw "cutlass.experimental.primitives.nvvm_wrapper.atomicrmw"), this operation does not return the old
    memory value.

    Scalar reductions may target global or shared memory. Vector reductions
    are global-memory only; the hardware guarantees atomicity independently for
    each scalar element, not for the whole vector as one transaction. When
    `mem_order` is omitted PTX assumes `.relaxed`; when `mem_scope` is
    omitted PTX assumes `.gpu`.

    Note

    `shared_space` selects the explicit shared-memory PTX spelling. When
    omitted for a `mapa`-produced shared-cluster pointer (addrspace 7),
    the wrapper selects `SharedSpace.shared_cluster` automatically.

    Parameters:
    :   - **op** ([*ReductionOp*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.ReductionOp "cutlass.experimental.primitives.nvvm_wrapper.ReductionOp")) – Reduction operation: `AND`, `OR`, `XOR`, `ADD`,
          `INC`, `DEC`, `MIN`, or `MAX`.
        - **type** ([*ReductionType*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.ReductionType "cutlass.experimental.primitives.nvvm_wrapper.ReductionType")) – PTX reduction type such as `S32`, `U32`, `F32`,
          `F64`, `F16`, `F16X2`, `BF16`, or `BF16X2`.
        - **a** (*Array* *or* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer or Array naming the destination memory cell.
        - **b** ([*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*,* [*Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*,* [*Float16*](basic_data_types.md#cutlass.Float16 "cutlass.Float16")*,* [*BFloat16*](basic_data_types.md#cutlass.BFloat16 "cutlass.BFloat16")*,* [*Float32*](basic_data_types.md#cutlass.Float32 "cutlass.Float32")*,* [*Float64*](basic_data_types.md#cutlass.Float64 "cutlass.Float64")*, or* [*Vector*](basic_data_types.md#cutlass.Vector "cutlass.Vector")) – Value contributed to the reduction. For vector reductions, pass
          a vector matching the PTX vector/type combination.
        - **mem\_order** ([*MemOrder*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder")*,* *optional*) – Optional memory ordering qualifier. `RELAXED` and
          `RELEASE` are the PTX `red` semantics.
        - **mem\_scope** ([*MemScope*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.experimental.primitives.nvvm_wrapper.MemScope")*,* *optional*) – Optional memory scope: `CTA`, `CLUSTER`, `GPU`,
          or `SYS`.
        - **cache\_hint** (*int* *or* [*Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional 64-bit L2 cache policy. PTX permits this only
          for global memory reductions with `.L2::cache_hint`.

    Raises:
    :   **ValueError** – if *op* is not a valid [`ReductionOp`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.ReductionOp "cutlass.experimental.primitives.nvvm_wrapper.ReductionOp") or *type\_*
        is not a valid [`ReductionType`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.ReductionType "cutlass.experimental.primitives.nvvm_wrapper.ReductionType").

    ```python
    # Every valid thread contributes one Int32 to a global sum.
    ptr = sum_out.iterator.raw_ptr()
    nvvm.red(
        "add",
        "s32",
        ptr,
        contribution,
        mem_order="relaxed",
        mem_scope="gpu",
    )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.redux\_sync( : *val: ~cutlass.Int32 | ~cutlass.Float32*, : *kind: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject*, : *mask\_and\_clamp: int | ~cutlass.Int32 | ~cutlass.Uint32*, : *\**, : *abs: bool | None = None*, : *nan: bool | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")
:   Reduce `val` across warp lanes selected by `mask_and_clamp` (sm\_80+).

    Low-level NVVM dialect wrapper for `redux.sync`. All participating lanes
    receive the same result (implicit broadcast). Prefer this over a 5-step
    butterfly shuffle loop for simple reductions.

    Parameters:
    :   - **val** ([*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Float32*](basic_data_types.md#cutlass.Float32 "cutlass.Float32")) – Each lane’s input value.
        - **kind** (*ReductionKind*) – Reduction operation (ADD, MIN, MAX, AND, OR, XOR, UMIN,
          UMAX, FMIN, FMAX).
        - **mask\_and\_clamp** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – 32-bit member mask (`0xFFFFFFFF` for all lanes).
        - **abs** (*bool**,* *optional*) – Apply `|val|` before reducing; `FMIN`/`FMAX` only,
          sm\_100+, defaults to None (disabled).
        - **nan** (*bool**,* *optional*) – Propagate NaN to result; `FMIN`/`FMAX` only, sm\_100+,
          defaults to None (NaN inputs are ignored).

    Returns:
    :   Warp-reduced result broadcast to all participating lanes.

    Return type:
    :   Float32 for FMIN/FMAX; Int32 for all other kinds.

    Raises:
    :   **ValueError** – if a static *mask\_and\_clamp* does not fit in 32 bits, or
        if *abs* / *nan* is set for a non-FMIN/FMAX *kind*. A runtime
        (non-`int`) *mask\_and\_clamp* is not checked at trace time.

    ```python
    # Per-block abs-max for MXFP8 quantization (sm_100+):
    amax = nvvm.redux_sync(gv, ReductionKind.FMAX, 0xFFFFFFFF, abs=True)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.setmaxregister( : *reg\_count: int*, : *action: [SetMaxRegisterAction](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.SetMaxRegisterAction "cutlass.experimental.primitives.nvvm_wrapper.SetMaxRegisterAction")*, ) → None
:   Adjust the per-thread register budget for the issuing warp.

    Emits `setmaxnreg.inc` / `setmaxnreg.dec`. The instruction
    provides a hint that changes the maximum number of per-thread
    registers owned by the executing warp, claiming registers from or
    releasing registers to the CTA register pool. PTX requires every
    warp in a warpgroup to execute the same `setmaxnreg` instruction;
    branch on a warpgroup-uniform role, not on an individual warp role.

    Parameters:
    :   - **reg\_count** (*int*) – Target per-thread register count. Must be a
          multiple of 8, within `[24, 256]`.
        - **action** ([*SetMaxRegisterAction*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.SetMaxRegisterAction "cutlass.experimental.primitives.nvvm_wrapper.SetMaxRegisterAction")) – `"increase"` (claim more from the pool) or
          `"decrease"` (release to the pool).

    Raises:
    :   - **TypeError** – if `reg_count` is not an `int` literal.
        - **ValueError** – if `reg_count` is outside `[24, 256]` or not
          a multiple of 8.

    ```python
    warpgroup = cute.arch.warp_idx() // 4
    if warpgroup == PROD_WARPGROUP:
        nvvm.setmaxregister(40, "decrease")
        # ... TMA issue + mbarrier arrive ...
    else:
        nvvm.setmaxregister(232, "increase")
        # ... MMA + epilogue ...
    nvvm.barrier_cta_sync()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.shfl\_sync( : *thread\_mask: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *val: [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32")*, : *offset: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *mask\_and\_clamp: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *kind: [Shfl](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Shfl "cutlass.experimental.primitives.nvvm_wrapper.Shfl")*, : *\**, : *return\_value\_and\_is\_valid: bool | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32") | tuple[[Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32"), [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")]
:   Synchronise participating lanes and shuffle a 32-bit value within a warp.

    Emits `shfl.sync.{idx|up|down|bfly}.b32` — the PTX warp-shuffle family.
    All lanes named in *thread\_mask* must execute the same instruction before
    any lane receives a shuffled value from another lane. This synchronizes the
    register exchange itself, but it does **not** provide the memory-ordering
    guarantee of `bar.sync` / [`bar_warp_sync()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.bar_warp_sync "cutlass.experimental.primitives.nvvm_wrapper.bar_warp_sync").

    **ShflKind variants:**

    - `"idx"` — each lane reads from absolute source lane *offset*;
      `result[lane] = val[offset]`. Used for broadcast (`offset=0`).
    - `"up"` — each lane reads from lane `max(lane - offset, lower_bound)`;
      result is the value *offset* lanes earlier in the warp.
    - `"down"` — each lane reads from lane `min(lane + offset, upper_bound)`;
      result is the value *offset* lanes ahead. Used in butterfly reductions.
    - `"bfly"` — each lane reads from lane `lane XOR offset`; enables
      butterfly reduction trees without out-of-range clamping.

    **mask\_and\_clamp encoding:**

    `mask_and_clamp` is a packed 32-bit integer that controls sub-warp
    segmentation and the out-of-range clamp boundary:

    - Bits `[12:8]` — *segmask*: `(WARP_SIZE - 1) XOR (width - 1)`. Lanes
      that differ only in the low `log2(width)` bits form one shuffle segment.
      Use `(31 << 8) | clamp` for a full 32-lane warp.
    - Bits `[4:0]` — *clamp*: upper boundary (`width - 1`) for
      `idx` / `down` / `bfly`; lower boundary (`0`) for `up`.
      When a source lane would fall outside the segment, the clamped boundary
      lane’s value is returned instead.

    For a full 32-lane warp these precomputed values are correct:

    - `"idx"` / `down` / `bfly`: `mask_and_clamp = 0x1F`
      (segmask = 0, clamp = 31)
    - `"up"`: `mask_and_clamp = 0x00`
      (segmask = 0, clamp = 0)

    The following higher-level helpers compute `mask_and_clamp` automatically
    from a `width` argument; prefer them unless you need explicit control
    over the packed field:

    - `shuffle_sync`
    - `shuffle_sync_up`
    - `shuffle_sync_down`
    - `shuffle_sync_xor`

    Parameters:
    :   - **thread\_mask** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – 32-bit participation mask; bit *i* = 1 means lane *i*
          takes part. All participating lanes must execute the instruction
          together. Pass `0xFFFFFFFF` for a full-warp shuffle.
        - **val** ([*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Float32*](basic_data_types.md#cutlass.Float32 "cutlass.Float32")) – The 32-bit value this lane contributes to the shuffle.
        - **offset** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Interpretation depends on *kind*:
          `idx` → absolute source lane ID `[0, 31]`;
          `up` / `down` → relative lane delta `[0, 31]`;
          `bfly` → XOR lane mask `[0, 31]`.
        - **mask\_and\_clamp** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Packed sub-warp segmentation and clamp boundary.
          See encoding description above. For full-warp shuffles use
          `0x1F` (idx/down/bfly) or `0x00` (up).
        - **kind** ([*Shfl*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Shfl "cutlass.experimental.primitives.nvvm_wrapper.Shfl")) – Shuffle direction. One of `"idx"`,
          `"up"`, `"down"`, `"bfly"`.
        - **return\_value\_and\_is\_valid** (*bool**,* *optional*) – If `True`, return a
          `(value, is_valid)` tuple where *is\_valid* is `True` when the
          source lane was within the active segment (i.e. the result is not the
          clamped fallback). Defaults to `None` (return value only).

    Returns:
    :   Shuffled value from the source lane, or a
        `(value, is_valid)` tuple when *return\_value\_and\_is\_valid* is
        `True`.

    Return type:
    :   [cutlass.Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") or [cutlass.Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32"), or
        tuple[[cutlass.Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") or [cutlass.Float32](basic_data_types.md#cutlass.Float32 "cutlass.Float32"), [cutlass.Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")]

    Raises:
    :   **ValueError** – if a static *thread\_mask* or *mask\_and\_clamp* does not
        fit in 32 bits, if a static *offset* is outside `[0, 31]`, or if
        *kind* is not one of `"idx"`, `"up"`, `"down"`, `"bfly"`.
        Runtime (non-`int`) values are not checked at trace time.

    **Constraints:**

    - All lanes in *thread\_mask* must reach the instruction; a subset that
      diverges before `shfl.sync` causes undefined behaviour.
    - *offset* and the low bits of *mask\_and\_clamp* must be in `[0, 31]`.
    - Available on SM30+ (Kepler); the synchronisation guarantee requires
      SM70+ (Volta) for correctness in independently-scheduled warps.
    - Prefer the higher-level helpers (`shuffle_sync`,
      `shuffle_sync_down`, `shuffle_sync_xor`) for common
      patterns; use `nvvm.shfl_sync` directly only when you need
      fine-grained control over `mask_and_clamp` or `kind`.

    ```python
    # Broadcast lane 0's value to all lanes (full warp)
    val = cutlass.Float32(cute.arch.lane_idx)
    broadcast = nvvm.shfl_sync(0xFFFFFFFF, val, 0, 0x1F, "idx")

    # Butterfly reduction: warp sum
    acc = cutlass.Int32(cute.arch.lane_idx)
    for delta in [16, 8, 4, 2, 1]:
        other = nvvm.shfl_sync(0xFFFFFFFF, acc, delta, 0x1F, "bfly")
        acc = acc + other
    # acc now holds the warp sum on all lanes
    ```

cutlass.experimental.primitives.nvvm\_wrapper.st\_bulk( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *size: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*, : *\**, : *init\_val: int | None = None*, ) → None
:   Bulk-initialize a shared-memory byte range to a constant value.

    Emits `st.bulk.shared::cta  [addr], size, init_val;`. Writes a
    contiguous run of `size` bytes at `addr` (SMEM) to
    `init_val` (currently the only legal value is `0`). Useful
    for zero-initializing tiles without a per-thread loop.

    Parameters:
    :   - **addr** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – SMEM destination pointer/array; must be 16-byte
          aligned.
        - **size** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")) – Number of bytes to write; positive multiple of 16.
        - **init\_val** (*int**,* *optional*) – Constant byte pattern; PTX currently mandates 0.

    Raises:
    :   **ValueError** – if a statically known `size` is not a
        positive multiple of 16, or `init_val` is not `0` / `None`.

cutlass.experimental.primitives.nvvm\_wrapper.stmatrix( : *ptr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *sources: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector") | list | tuple*, : *layout: [MMALayout](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMALayout "cutlass.experimental.primitives.nvvm_wrapper.MMALayout")*, : *\**, : *shape: [StoreShape](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.StoreShape "cutlass.experimental.primitives.nvvm_wrapper.StoreShape") | None = None*, ) → None
:   Warp-cooperative store of one to four 8x8 (or 16x8) matrix tiles to SMEM.

    Emits `stmatrix.sync.aligned.{shape}.{num}{.trans}{.ss}.{type} [a], d`.
    All 32 lanes of the issuing warp collectively store `num` tiles whose
    fragment registers they hold; the per-lane row-start address goes through
    `ptr` exactly as for [`ldmatrix()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.ldmatrix "cutlass.experimental.primitives.nvvm_wrapper.ldmatrix").

    `sources` accepts three shapes:

    - a single scalar (`int` / `Int32` / `Uint32`) – stores `num=1`,
    - a `Vector[N x Int32]` – stores `num=N` (`N` in `{1, 2, 4}`);
      decomposed via `vector.extract` before forwarding to the dialect,
    - a Python `list` / `tuple` of scalars – stores
      `num=len(sources)`; each element is coerced to `Int32`.

    Parameters:
    :   - **ptr** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer/Array into shared memory; per the PTX ISA the address
          space must be `.shared{::cta}`.
        - **sources** (*int* *or* [*Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32") *or* [*Vector*](basic_data_types.md#cutlass.Vector "cutlass.Vector") *or* *list* *or* *tuple*) – Per-lane source fragment. Length (or `Vector` shape)
          must be 1, 2, or 4 – the PTX `.x1` / `.x2` / `.x4` qualifiers.
        - **layout** ([*MMALayout*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMALayout "cutlass.experimental.primitives.nvvm_wrapper.MMALayout")) – `MMALayout.ROW` for the default store.
          `MMALayout.COL` selects `.trans` (in-register transpose before
          committing to SMEM).
        - **shape** ([*StoreShape*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.StoreShape "cutlass.experimental.primitives.nvvm_wrapper.StoreShape") *or* *None*) – Tile shape selector. Defaults to `m8n8`; `m16n8` is
          the new PTX 9.3 `.b8` store variant.

    Returns:
    :   `None` – `stmatrix` writes to SMEM and has no SSA result.

    Raises:
    :   **ValueError** – `sources` count is statically known and is not in
        `{1, 2, 4}`.

    ```python
    # Store a Vector[4 x Int32] accumulator fragment to a 4 x 8x8 SMEM tile.
    smem = cutlass.Array(cutlass.Int16, 4 * 8 * 8, space=cutlass.AddressSpace.smem)
    nvvm.stmatrix(smem, frag, nvvm.MMALayout.ROW)  # frag : Vector[4 x Int32]
    ```

cutlass.experimental.primitives.nvvm\_wrapper.store\_ext( : *value: ir.Value*, : *addr: ~cutlass.Array | ~cutlass.Pointer*, : *\**, : *l2\_cache\_hint: int | ~cutlass.Int64 | ~cutlass.Uint64 | None = None*, : *order: ~cutlass.experimental.primitives.nvvm\_wrapper.MemOrder | None = None*, : *scope: ~cutlass.base\_dsl.array.MemScope | None = None*, : *evict: ~cutlass.base\_dsl.array.L1EvictKind | None = None*, : *cache\_modifier: ~cutlass.base\_dsl.array.StoreCacheModifier | None = None*, : *shared\_space: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject | None = None*, ) → None
:   Store a scalar to generic, global, or shared memory with explicit cache,
    eviction, and memory-ordering qualifiers (`nvvm.store.ext` / PTX `st`).

    The underlying op supports only `b8/b16/b32/b64/b128` integer widths and
    `f32`/`f64` floats: store a 16-bit float by bitcasting it to `Int16`
    first.

    Parameters:
    :   - **value** (*ir.Value* *|* [*Vector*](basic_data_types.md#cutlass.Vector "cutlass.Vector")) – Register value to store; its type selects the store width.
          May be a scalar or a `Vector` (PTX `.v2`/`.v4`/`.v8`).
        - **addr** (*Array* *|* [*Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Destination address (generic, global, or shared pointer).
        - **l2\_cache\_hint** (*int* *|* [*Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *|* [*Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") *|* *None*) – 64-bit L2 cache-eviction policy handle (generic /
          global space only).
        - **order** ([*MemOrder*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemOrder "cutlass.experimental.primitives.nvvm_wrapper.MemOrder") *|* *None*) – Memory ordering (`weak` default, `relaxed`, `release`,
          `volatile`, `mmio`). `relaxed` / `release` require `scope`.
        - **scope** ([*MemScope*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.experimental.primitives.nvvm_wrapper.MemScope") *|* *None*) – Memory scope (`cta`, `cluster`, `gpu`, `sys`) for an
          ordered store.
        - **evict** ([*L1EvictKind*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.L1EvictKind "cutlass.experimental.primitives.nvvm_wrapper.L1EvictKind") *|* *None*) – L1 eviction-priority hint; mutually exclusive with
          `cache_modifier`.
        - **cache\_modifier** ([*StoreCacheModifier*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.StoreCacheModifier "cutlass.experimental.primitives.nvvm_wrapper.StoreCacheModifier") *|* *None*) – Cache operator (`wb`/`cg`/`cs`/`wt`); only
          valid on the default `weak` ordering.
        - **shared\_space** ([*SharedSpace*](cute_dsl_api/cute_nvgpu_common.md#cutlass.cute.nvgpu.SharedSpace "cutlass.cute.nvgpu.SharedSpace") *|* *None*) – Shared sub-space (`cta` default, `cluster` for
          distributed shared memory); for shared-space pointers only.

    Raises:
    :   - **ValueError** – if the qualifier combination is illegal, e.g.
          `cache_modifier` with `evict` or with non-`weak` ordering;
          `relaxed`/`release` without `scope`; `volatile` with any cache
          op/hint; `mmio` without `scope=sys`; or `shared_space` combined
          with `l2_cache_hint`/`mmio`.
        - **TypeError** – if `value` is a 16-bit float (`Float16`/`BFloat16`);
          bitcast it to `Int16` before calling `store_ext`.

    ```python
    ptr = arr.data_ptr() + tx
    # Streaming store (likely written once): bypass L1 reuse tracking.
    nvvm.store_ext(val.ir_value(), ptr,
                   cache_modifier=StoreCacheModifier.CS)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.sub\_packed\_f32x2( : *src\_a: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *src\_b: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *\**, : *rnd: [FPRoundingMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode "cutlass.experimental.primitives.nvvm_wrapper.FPRoundingMode") | None = None*, : *ftz: bool | None = None*, ) → [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   Wrapper over `nvvm.sub_packed_f32x2`.

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_alloc( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *n\_cols: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *group: [CTAGroup](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup") | None = None*, ) → None
:   Allocate TMEM columns for tcgen05 operations.

    Parameters:
    :   - **addr** –

          Pointer/Array to a 32-bit SMEM cell that receives the TMEM
          token (start column address). Pass `tmem_ptr` (an `Array` or
          `Pointer` of dtype `Int32`) and read it back after alloc.

          Validated at trace time: must reside in shared memory when the
          passed object exposes `.space`; opaque pointers are deferred to
          the MLIR verifier.
        - **n\_cols** –

          Number of TMEM columns to allocate. The allocation unit
          is 32 columns and all lanes per column. Statically known `int`
          values must be a power of 2 in `[32, 512]` (validated at trace
          time); dynamic IR values are forwarded as-is and may fault at
          runtime if out of range. PTX also requires the number of columns
          allocated not to increase between any two allocations in CTA
          execution order. Standard values:

          - **CTA\_1**: `n_cols = (N_TILE // 8) * 32` where `N_TILE` is the
            per-CTA accumulator N. `N_TILE=128` → `512` columns (fills
            entire TMEM); `N_TILE=64` → `256` columns (two accumulators
            fit: 256 + 256 = 512).
          - **CTA\_2**: each CTA in the 2-SM group still allocates from its own
            512-column TMEM bank, but the accumulator is split M-wise (Layout
            A: leader holds top M-half × full pair-N, peer holds bot M-half
            × full pair-N). The CTA\_1 formula applied to the **per-CTA**
            half (`N_TILE = N_PER_GROUP / 2`) gives the minimum, but the
            simpler safe default is to **always allocate 512** for any CTA\_2
            GEMM (over-allocation is harmless). Applying the CTA\_1 formula
            with `N_TILE = N_PER_GROUP` (collective N) instead of per-CTA
            N over-allocates beyond 512 and faults with
            `cudaErrorIllegalInstruction`.
        - **group** – `'CTA_1'` (default) or `'CTA_2'`. See [`CTAGroup`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup").
          All `tcgen05` instructions within a kernel must use the same group.

    ```python
    # Correct — both alloc and relinquish are warp-collective, neither
    # is inside elect_sync.  mbarrier_init IS elect-safe so it stays
    # inside the elect_sync block.
    if warp == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(S):
                nvvm.mbarrier_init(full_bar + s, 1)
        nvvm.tcgen05_alloc(tmem_ptr, num_cols, group="cta_1")
        nvvm.tcgen05_relinquish_alloc_permit(group="cta_1")
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_commit( : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *multicast\_mask: [Int16](basic_data_types.md#cutlass.Int16 "cutlass.Int16") | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | None = None*, : *smem\_a\_read: bool | None = None*, : *group: [CTAGroup](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup") | None = None*, ) → None
:   Track prior async tcgen05 operations with an mbarrier.

    Used in pipelined kernels to release the SMEM staging buffer back to the
    producer after prior async `tcgen05` operations have completed. Emits
    `tcgen05.commit.cta_group::N.mbarrier::arrive::one` and makes *addr*
    track all prior async `tcgen05` operations of the same CTA group that
    were initiated by the executing thread.

    Parameters:
    :   - **addr** – Pointer/Array to the 64-bit mbarrier to signal (typically
          `empty_bar + stage`).
        - **smem\_a\_read** – When `True`, tells the hardware that this commit
          releases the A-operand SMEM buffer (default `None` = both A and B).
        - **multicast\_mask** –

          CTA participation mask for CTA\_2 multicast —
          a per-bit mask **over cluster ranks** (not pair-internal). Bit
          `i` set means the arrive lands on the mbar copy at the same SMEM
          offset in cluster rank `i`. Two regimes:

          - **Single 2-CTA cluster** (`cluster_shape=(2,1,1)`) — only
            one 2-SM group, leader at rank 0. Canonical value is
            `3` (= `0b11` covering ranks 0 and 1 = both pair members);
            this is the value you see in `2cta_mma_basic.py` and is
            right whenever there is exactly one 2-SM group in the
            cluster.
          - **Multi-group clusters** (`cluster_shape=(2, n_groups, 1)`
            with `n_groups > 1`) — each 2-SM group `G` has its
            leader at cluster rank `2*G`. Use
            `multicast_mask = 3 << cluster_rank` so the issuing group
            leader signals ranks `2G` and `2G+1` (its own pair),
            NOT ranks 0,1. Hard-coding `mask=3` is a frequent
            deadlock cause: groups 1, 2, … never receive the arrive
            and stall at the next `try_wait_parity`.

          `multicast_mask=1 << cta_rank` (commit only to the issuer)
          is also valid when the follower CTA never waits on this
          mbar; otherwise the follower deadlocks.

          Note that the *value semantics differ* from cluster TMA
          loads, where commits count arrives on named mbar copies
          but TMA counts bytes delivered per CTA. Do not cross-apply
          multicast masks between TMA loads and `tcgen05.commit` without
          rechecking which mbarrier copies are signaled. Default `None`
          selects the CTA\_1 path with no multicast. Static `int` values are
          validated at trace time to fit in 16 bits (the dialect mask is
          `i16`); larger literals would be silently truncated.
        - **group** – `"cta_1"` (default) or `"cta_2"`.

    ```python
    # Per-k-tile: signal empty_bar when prior async tcgen05 ops complete.
    if warp == MMA_WARP:
        if nvvm.elect_sync():
            nvvm.tcgen05_commit(empty_bar + s, group="cta_1")

    # After K-loop: signal acc_mbar to release the TMEM accumulator
    if warp == MMA_WARP:
        if nvvm.elect_sync():
            nvvm.tcgen05_commit(acc_mbar, group="cta_1")
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_cp( : *shape: [Tcgen05CpShape](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05CpShape "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05CpShape")*, : *taddr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *smem\_desc: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*, : *\**, : *group: [CTAGroup](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup") | None = None*, : *multicast: [Tcgen05CpMulticast](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05CpMulticast "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05CpMulticast") | None = None*, : *src\_format: [Tcgen05CpSrcFormat](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05CpSrcFormat "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05CpSrcFormat") | None = None*, ) → None
:   Asynchronous SMEM → TMEM copy with optional decompression / multicast.

    Emits `tcgen05.cp` — staged into the TC issue queue alongside
    `tcgen05.mma` and `tcgen05.shift` (the PTX ISA lists the legal
    ordering pairs). Used to seed an A-from-TMEM operand, to gather
    narrow-format data (FP6/FP4) into TMEM with on-the-fly widening, or
    to place block-scaled SFA/SFB metadata in TMEM before the matching
    [`tcgen05_mma_block_scale()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale")-style wrapper consumes `scale_a` /
    `scale_b`.

    Parameters:
    :   - **shape** – `Tcgen05CpShape` selecting the data dimensions.
          Per the PTX ISA these come with `.multicast` constraints:
          `.64x128b` requires `warpx2::02_13` or `warpx2::01_23`;
          `.32x128b` requires `warpx4`; the wider shapes
          (`.128x256b` / `.4x256b` / `.128x128b`) take no multicast.
          The `.64x128b.warpx2::01_23` form is the direct SFB scale
          metadata copy used by the 2x2 CTA\_2 block-scaled N=256 example.
          The shape<->multicast coupling is validated at trace time.
        - **taddr** – TMEM destination address (`Array`/`Pointer`).
          Validated at trace time: must reside in tensor memory when the passed
          object exposes `.space` (dialect operand type `LLVM_PointerTensor`).
        - **smem\_desc** – 64-bit SMEM matrix descriptor (built via
          `cutlass.experimental.primitives.Tcgen05SmemDesc.build()`).
        - **group** – `"cta_1"` (default) — destination is the
          issuing CTA’s TMEM. `"cta_2"` — also writes the
          peer CTA’s TMEM (cluster-collective; both CTAs’ issuing warp
          must reach this op cooperatively). All `tcgen05.*` ops in a
          kernel must agree on `group`.
        - **multicast** – Warp-pair / quad-warp multicast policy for the
          narrow shapes; see `shape` constraints above.
        - **src\_format** – `B6x16_P32` / `B4x16_P64` to enable on-the-fly
          decompression to `b8x16` in TMEM. `None` means same-format
          copy.

    ```python
    # Stage A from SMEM to TMEM ahead of an A-from-TMEM MMA
    if warp == TMA_WARP:
        if nvvm.elect_sync():
            nvvm.tcgen05_cp(
                "shape_128x256b",
                a_tmem_addr,
                smem_desc_a,
                group="cta_1",
            )
            nvvm.tcgen05_commit(empty_bar)

    # Stage 2x2 SFB scale metadata before the block-scaled MMA reads scale_b.
    sfb_shape, sfb_multicast = ...  # S2T copy mode, e.g. 64x128b WARPX2_01_23
    if warp == MMA_WARP:
        if nvvm.elect_sync():
            nvvm.tcgen05_cp(
                sfb_shape,
                sfb_tmem_addr,
                sfb_smem_desc,
                group=nvvm.CTAGroup.CTA_2,
                multicast=sfb_multicast,
            )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_dealloc( : *taddr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *n\_cols: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *\**, : *group: [CTAGroup](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup") | None = None*, ) → None
:   Free TMEM columns previously allocated by [`tcgen05_alloc()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc").

    Parameters:
    :   - **taddr** –

          TMEM base pointer (addrspace 6) — typically the value read
          back from the SMEM slot [`tcgen05_alloc()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc") wrote to, converted with
          an addrspace-6 (TMEM) pointer.

          Validated at trace time: must reside in tensor memory when the passed
          object exposes `.space`; opaque pointers are deferred to the MLIR
          verifier (the dialect operand type is `LLVM_PointerTensor`).
        - **n\_cols** – Number of TMEM columns to free; must equal the value passed
          to the paired [`tcgen05_alloc()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc"). Statically known `int` values
          are validated against the same whitelist as [`tcgen05_alloc()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc")
          (`{32, 64, 128, 256, 512}`) at trace time.
        - **group** – `'CTA_1'` (default) or `'CTA_2'` — must match the group
          used at alloc time. See [`CTAGroup`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup").

    ```python
    # After the epilogue's TMEM reads and their sync:
    nvvm.barrier_cta_sync()
    if warp_idx == 0:
        nvvm.tcgen05_dealloc(tmem_ptr, NUM_TMEM_COLS, group="cta_1")
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_fence( : *kind: [Tcgen05Fence](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05Fence "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05Fence")*, ) → None
:   Order async tcgen05 operations around thread synchronization.

    Emits `tcgen05.fence.{before_thread_sync|after_thread_sync}`.

    Parameters:
    :   **kind** ([*Tcgen05Fence*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05Fence "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05Fence")) – `"before_thread_sync"` — place **before**
        an execution-ordering operation such as `nvvm.barrier_cta_sync()` or a flag
        store to order all prior async `tcgen05` operations before that
        synchronization point. `"after_thread_sync"` — place **after** such
        a synchronization point and before subsequent
        async `tcgen05` operations.

    ```python
    # After reading TMEM, fence before syncing:
    c_vec = nvvm.tcgen05_ld(shape, tmem_addr, num=n)
    nvvm.tcgen05_wait("load")
    nvvm.tcgen05_fence("before_thread_sync")
    nvvm.barrier_cta_sync()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_ld( : *shape: [Tcgen05LdStShape](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05LdStShape "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05LdStShape")*, : *tmem\_addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *num: int = 1*, : *pack: bool | None = None*, : *offset: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")
:   Load data from TMEM into registers.

    Emits `tcgen05.ld.sync.aligned.{shape}{.xN}{.pack}.b32`. The issuing
    warp collectively loads from TMEM; each lane receives its own register
    slice from the warp’s accessible TMEM sub-partition. Pair with
    [`tcgen05_wait()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_wait "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_wait") (`LOAD`) before reading the result.

    **TMEM access restriction.** TMEM is divided into
    4 lane-chunks of 32 lanes each. A given lane is accessible by **exactly
    one warp** in each warpgroup, determined by the warp’s position within
    its warpgroup (i.e. `warp_idx % 4`):

    | Warp position in warpgroup | Accessible lanes |
    | --- | --- |
    | 0 | 0–31 |
    | 1 | 32–63 |
    | 2 | 64–95 |
    | 3 | 96–127 |

    All warps see all 512 columns; the lane (row) restriction is the only
    cross-warp partitioning. A warp cannot read another warp’s lanes by
    changing the address — the row field in `tmem_addr[31:16]` is a
    **local** index within the warp’s own chunk (5 bits; values ≥ 32 wrap
    as `row mod 32`).

    **Implication for ``M\_TILE=128`` GEMM epilogues.** Covering all 128
    accumulator rows requires 4 issuing warps whose warpgroup positions cover
    0..3. For a 4-warp CTA those are warps 0..3. A shifted TMEM-load range
    such as warps 2..5 is also valid because `warp_idx % 4` covers
    2, 3, 0, 1. In that case, derive the TMEM row and output row from the
    physical SP position, not from the logical TMEM-load rank:
    `tmem_sp = warp_idx % 4` and `row = tmem_sp * 32 + lane`. Using
    `warp_idx - tmem_ld_warp_start` for the row offset makes warp 4 try
    to read SP2 rows while the hardware routes it to SP0, causing wrong data
    or an illegal instruction. Valid organizations include:

    - **Time-multiplex** — 4-warp CTA; warp 0 does TMA + MMA in the
      K-loop, then warps 0..3 do epilogue.
    - **Split with shifted epilogue** — producer / MMA use earlier warps,
      and four epilogue warps such as 2..5 drain rows via `warp_idx % 4`.

    Also note: `tcgen05.mma` only fills rows for the `m_dim` it was given.
    If `m_dim < 8` (i.e. `M_TILE < 128`), rows beyond `m_dim * 4` per
    chunk contain stale data. Use `M_TILE=128` (`m_dim=8`) to populate
    all rows.

    Parameters:
    :   - **shape** ([*Tcgen05LdStShape*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05LdStShape "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05LdStShape")) – Load shape (PTX `.shape1` / `.shape2`). One of
          `"16x32bx2"`, `"16x64b"`, `"32x32b"`, `"16x128b"`,
          `"16x256b"`. Most common is `"32x32b"` (one 32-bit register per
          thread, reads 32 rows × 32 bits = 128 bytes per warp).
          `num > 1` stacks multiple shapes into a contiguous `Vector`.
        - **tmem\_addr** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer/Array to the TMEM location. The address encodes
          `(row << 16) | col` where *row* is the local row within the
          sub-partition and *col* is the TMEM column from `tcgen05_alloc`.
          Validated at trace time: must reside in tensor memory when the passed
          object exposes `.space` (the dialect operand type is
          `LLVM_PointerTensor`); opaque pointers are deferred to the verifier.
        - **num** (*int*) – Number of shape repetitions; result has `regs_per_elem * num`
          registers. `regs_per_elem` is determined by the `shape`:
          `"16x32bx2"` / `"16x64b"` / `"32x32b"` → 1, `"16x128b"` → 2,
          `"16x256b"` → 4. Must be a power of 2 in [1, 128]; total
          registers (`regs_per_elem * num`) must not exceed 128.
        - **pack** (*bool**,* *optional*) – Enable element packing (reduces register count for sub-32-bit
          dtypes).
        - **offset** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Required (and only valid) for `"16x32bx2"` — column
          offset added to `tmem_addr` at runtime. Must be `None` for all
          other shapes.

    Returns:
    :   A `Vector` of the loaded registers (always a vector, even for a
        single register). If `tmem_addr.dtype` is not `Int32`, the result
        is bitcast to that dtype automatically (e.g. `Float32`).

    Return type:
    :   [cutlass.Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")

    Raises:
    :   - **ValueError** – `shape` is not a recognized literal; `num` is
          not a power of 2 in [1, 128]; total registers exceed 128; `offset`
          is set for a shape other than `"16x32bx2"` (or missing for
          `"16x32bx2"`).
        - **TypeError** – `tmem_addr` exposes a `.space` that is not tensor
          memory (TMEM).

    ```python
    # Each issuing warp loads its own 32 lanes from the accumulator tile.
    # Use the physical TMEM/SP owner, not the logical TMEM-load rank.
    # Build the encoded (row << 16) | col integer address, then convert
    # to a TMEM pointer in one shot -- pointer arithmetic on an already-
    # constructed TMEM pointer does not interpret the row/col layout the
    # way callers expect.
    tmem_sp = warp_idx % 4
    base_row = tmem_sp * 32
    tmem_addr = (base_row << 16) | base_col
    tmem_ptr = ...  # addrspace-6 TMEM pointer built from tmem_addr
    result = nvvm.tcgen05_ld("32x32b", tmem_ptr)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_mma( : *mma\_kind: [Tcgen05MMAKind](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind")*, : *cta\_group: [CTAGroup](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup")*, : *d: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *a: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, : *b: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*, : *idesc: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *enable\_input\_d: int | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*, : *\**, : *collector\_op: [Tcgen05MMACollectorOp](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp") | None = None*, : *a\_shift: bool | None = None*, : *scale\_input\_d: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, : *write\_disable\_mask: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector") | None = None*, ) → None
:   Issue a 5th-generation Blackwell tensor-core multiply-accumulate (`tcgen05.mma`).

    Emits `tcgen05.mma.cta_group::{1|2}.kind::<kind>` with optional
    `.collector::a::*`, `.ashift`, and `scale-input-d` modifiers.
    The accumulator `D = A * B [+ D]` lives in Tensor Memory (TMEM); A is
    read from SMEM (default) or TMEM; B is always an SMEM descriptor.
    Requires `sm_100a` or a supported family target.

    Parameters:
    :   - **mma\_kind** ([*Tcgen05MMAKind*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind")) –

          Data-type/kind selector. Dense kinds are issued by this
          wrapper; block-scaled kinds are listed for descriptor compatibility and
          should be issued with [`tcgen05_mma_block_scale()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale").

          - `"f16"` — f16/bf16 operands, f32 accumulator. K=16.
          - `"tf32"` — tf32 operands, f32 accumulator. K=8.
          - `"int8"` — signed/unsigned 8-bit, i32 accumulator.
            K=32.
          - `"f8f6f4"` — mixed {E4M3, E5M2, E2M3, E3M2, E2M1}
            inputs, f32 accumulator. K=32.
          - `"mxf8f6f4"` — block-scaled F8F6F4 (use
            [`tcgen05_mma_block_scale()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale")). K=32.
          - `"mxf4"` — block-scaled E2M1. K=64.
          - `"mxf4nvf4"` — block-scaled E2M1+NVFP4. K=64.
        - **cta\_group** ([*CTAGroup*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup")) – `"cta_1"` for single-CTA scope
          (M ∈ {32, 64, 128}); `"cta_2"` for 2-CTA cooperative
          scope (collective M ∈ {128, 256} across peer CTAs; N effectively
          doubled via peer SMEM). Every `tcgen05.*` op in a kernel must
          use the same group.
        - **d** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMEM accumulator pointer returned by
          [`tcgen05_alloc()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc") (address-space 6). Rows fill according to
          `m_dim`; hardware routes writes to sub-partitions by row.
          Validated at trace time: must reside in tensor memory when the passed
          object exposes `.space` (dialect operand type `LLVM_PointerTensor`).
        - **a** ([*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*,* *cutlass.Array**, or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) –

          Either a 64-bit SMEM descriptor from
          `cutlass.experimental.primitives.Tcgen05SmemDesc.build()` (`A-from-SMEM` path, common) **or** a
          TMEM pointer (`A-from-TMEM` path — used in BMM2 of FMHA where the
          previous MMA’s output is reused as the next A).

          For the TMEM path:

          - Build an addrspace-6 pointer from `tmem_addr`
            where `tmem_addr = (row << 16) | col` is the packed 32-bit
            TMEM address. Pointer arithmetic on TMEM pointers applies raw
            packed-token offsets; it does not interpret row/col fields.
          - A’s columns advance by the K-granule per K-step (e.g. 64 TMEM
            columns per BF16 K-step, not per K element).
          - Load A into TMEM beforehand via [`tcgen05_cp()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_cp "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_cp") (SMEM→TMEM
            copy) or [`tcgen05_st()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_st "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_st") (register→TMEM store), or produce it
            as the output of a prior [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma") accumulator.
        - **b** (*int**,* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")) – 64-bit SMEM descriptor for B from
          `cutlass.experimental.primitives.Tcgen05SmemDesc.build()`.
        - **idesc** (*int**,* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) –

          Packed 32-bit instruction descriptor encoding
          `c_format` / `a_format` / `b_format` / `a_major` /
          `b_major` / `m_dim` / `n_dim`. Build with
          `cutlass.experimental.primitives.Tcgen05InstrDesc.build()`:

          ```python
          idesc = cutlass.experimental.primitives.Tcgen05InstrDesc.build(
              c_dtype=cutlass.Float32,        # 1 = f32 accumulator
              a_dtype=cutlass.Float16,        # MUST match A dtype — see table below
              b_dtype=cutlass.Float16,        # MUST match B dtype — see table below
              n_dim=N_TILE,      # logical N (multiple of 8); 3 LSBs not encoded
              m_dim=M_TILE,      # logical M (multiple of 16); 4 LSBs not encoded
          )
          ```

          Common `a_format` / `b_format` values (must match operand dtype):

          | Value | Dtype | Applicable `mma_kind` |
          | --- | --- | --- |
          | 0 | FP16 | `F16` |
          | 1 | BF16 | `F16` |
          | 2 | TF32 | `TF32` |
          | 0..4 | E4M3..E2M1 | `F8F6F4` / `MX*` variants |
          | 0 | U8 | `INT8` |
          | 1 | S8 | `INT8` |

          See `cutlass.experimental.primitives.Tcgen05InstrDesc` for the full bit layout. For
          `CTA_2` the `m_dim` / `n_dim` describe the **collective**
          tile across both CTAs (e.g. M=256 → `m_dim=16`).

          Warning

          `a_format` / `b_format` **must match the operand dtype**.
          A mismatch (e.g. BF16 operands with `a_format=0`) produces
          **silently wrong results** (~40–60 max\_err on random data); no
          compile or runtime error is raised.
        - **enable\_input\_d** (*int* *or* [*cutlass.Boolean*](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")) – Controls first-tile behavior. `True` / non-zero
          computes `D = A*B + D` (accumulate into existing TMEM content);
          `False` / zero computes `D = A*B` (overwrite). Typically passed
          as `k > 0` in a k-tile loop so the first k-tile clears the
          accumulator and subsequent tiles accumulate.
        - **collector\_op** ([*Tcgen05MMACollectorOp*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp")*,* *optional*) –

          Collector cache usage for operand A. The collector
          is a small per-MMA-issuer cache that lets consecutive MMAs reuse the
          same A operand without re-reading SMEM. Values:

          - `"fill"` — load A into the collector and use
            it (seeds the cache on the first MMA of a chain).
          - `"use"` — read A from the collector (later
            MMAs in a chain that reuse the same A).
          - `"lastuse"` — read from the collector, then
            invalidate the entry (final MMA of the chain).
          - `"discard"` (default) — do not cache A.

          Reuse is opportunistic; hardware may reload despite the permission.
          Treat the collector strictly as a performance hint. The source memory
          for A must not be modified while any MMA using that matrix has not
          completed, regardless of collector state.
        - **a\_shift** (*bool**,* *optional*) –

          When `True`, emits the `.ashift` modifier. In
          the `.ashift` MMA pipeline, the shift is a **post-MMA**
          operation: the current MMA reads unshifted A and produces
          `A @ B`, then the A TMEM region is shifted for subsequent
          reads. To observe the shift you need a follow-up MMA (or
          [`tcgen05_ld()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld")) that targets the same A TMEM region.

          **A MUST be in TMEM (the ``[a-tmem]`` form).** The PTX ISA
          defines `.ashift` only for
          `tcgen05.mma.kind.ashift [d-tmem], [a-tmem], b-desc, ...`;
          there is no `.ashift` form that takes an SMEM descriptor for
          A. Passing `a_shift=True` with an SMEM-descriptor `a` is
          silently ignored by HW.

          **Shift semantics (per-SP, not global):** PTX wording “shifts
          rows down by 1 except the last row” refers to the last row of
          each TMEM sub-partition, not the global last row. For M=128
          (4 SPs × 32 rows) global M-rows {31, 63, 95, 127} retain their
          original values:

          ```text
          shifted[r] = A[r + 1]   when r % 32 != 31
          shifted[r] = A[r]       when r % 32 == 31
          ```

          Other constraints: `M ∈ {128, 256}` only; mutually exclusive
          with `collector_op in {FILL, USE}` and with `.ws` (warp-
          specialized) MMA variants. `idesc.max_shift` does NOT control
          `.ashift` — that field is for the `.ws` variant (values
          0/1/2/3 → max shifts 0/8/16/32 rows). For plain `.ashift` the
          shift is always exactly 1 row per MMA regardless of
          `max_shift`; leave it at 0.
        - **scale\_input\_d** (*int**,* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Immediate in `[0, 15]`. When set, scales the
          input accumulator as `D *= 2**(-scale_input_d)` before the MAC.
          Valid only for `"f16"` and `"tf32"` (PTX ISA 9.3). The kind
          restriction and the static `[0, 15]` range are validated at trace
          time.
        - **write\_disable\_mask** ([*cutlass.Vector*](basic_data_types.md#cutlass.Vector "cutlass.Vector")*,* *optional*) –

          Per-row TMEM write-disable vector. 4-element
          `Vector[Int32]` for `CTA_1` (128 mask bits / M-rows), 8-element
          for `CTA_2` (256 mask bits across the collective M=256 tile).

          **Bit mapping (Layout D, CTA\_1)**: bit `i`
          of element `v` masks output M-row `v * 32 + i`. A set bit
          suppresses the TMEM write for that row, leaving it at the
          TMEM-initial value (zero after [`tcgen05_alloc()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_alloc")).

          Typical use: row-wise partial-tile handling — suppress writes to
          M-rows beyond the live fraction of the tile so they don’t clobber
          accumulator rows still needed by the adjacent tile. The element count
          (4 for `CTA_1`, 8 for `CTA_2`) is validated at trace time.

    **Constraints:**

    - **Single-thread issue:** call inside `if nvvm.elect_sync():`. All
      32 warp threads issuing the intrinsic would emit 32 duplicate MMAs
      (undefined behavior / observed hangs). For `CTA_2`, only the
      leader CTA’s elected thread issues.
    - **A/B SMEM visibility:** producer writes (TMA, `cp.async`) must be
      ordered before the MMA via an mbarrier wait; `tcgen05.mma` itself
      is async with respect to generic memory but synchronous with respect
      to an arrive-on-completion mbarrier.
    - **Completion tracking:** follow the last MMA of a chain with
      [`tcgen05_commit()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_commit "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_commit") to release the A/B SMEM buffers, and with a
      commit on an accumulator mbarrier (or `tcgen05_wait`) before the
      epilogue reads TMEM.
    - **TMEM read ordering:** after the accumulator mbarrier fires, use
      [`tcgen05_fence()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_fence "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_fence") (`BEFORE_THREAD_SYNC`) between
      [`tcgen05_ld()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld") and the following [`barrier()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.barrier "cutlass.experimental.primitives.nvvm_wrapper.barrier") / next MMA.
    - **Predication:** not supported. `tcgen05.mma` has variable-length
      operand lists (`write_disable_mask`, collector ops,
      `scale_input_d`) that require NVVM-level register allocation —
      gate the intrinsic with a surrounding `if` instead.

    ```python
    # Per k-tile of a pipelined GEMM (warp 0 issues MMA):
    if warp_idx == 0:
        if nvvm.elect_sync():
            nvvm.tcgen05_mma(
                nvvm.Tcgen05MMAKind.F16,
                nvvm.CTAGroup.CTA_1,
                tmem_ptr,          # D: accumulator in TMEM
                desc_a,            # A: SMEM descriptor
                desc_b,            # B: SMEM descriptor
                idesc,             # Packed instr descriptor
                k > 0,             # enable_input_d: k==0 clears, k>0 accumulates
            )

    # Multi-MMA-warp shared TMEM: the first MMA that starts a fresh
    # accumulator region must clear (enable_input_d=False); every later
    # MMA targeting the same accumulator region accumulates.

    # A-reuse across 2 back-to-back MMAs (same A, different B):
    if nvvm.elect_sync():
        nvvm.tcgen05_mma(..., collector_op=nvvm.Tcgen05MMACollectorOp.FILL)
        nvvm.tcgen05_mma(..., collector_op=nvvm.Tcgen05MMACollectorOp.LASTUSE)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_mma\_block\_scale( : *mma\_kind: ~cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMAKind*, : *cta\_group: ~cutlass.experimental.primitives.nvvm\_wrapper.CTAGroup*, : *d: ~cutlass.Array | ~cutlass.Pointer*, : *a: ~cutlass.Array | ~cutlass.Pointer | ~cutlass.Int64*, : *b: int | ~cutlass.Int64 | ~cutlass.Uint64*, : *idesc: int | ~cutlass.Int32 | ~cutlass.Uint32*, : *enable\_input\_d: int | ~cutlass.Boolean*, : *scale\_a: ~cutlass.Array | ~cutlass.Pointer*, : *scale\_b: ~cutlass.Array | ~cutlass.Pointer*, : *\**, : *scale\_vec\_size: ~cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMAScaleVecSize | \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject | None = None*, : *collector\_op: ~cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMACollectorOp | None = None*, : *a\_shift: bool | None = None*, ) → None
:   MMA with per-block scale factors (MXFP / NVFP block scaling).

    Emits `tcgen05.mma.cta_group::N.{kind}.block_scale[.scale_vectorsize]`.
    Like [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma"), but each MMA additionally multiplies blocks
    of A and B by per-block scale factors before accumulating, enabling
    MXFP8 / MXFP6 / MXFP4 / NVFP4 block-scaled formats whose dynamic range
    is otherwise too narrow for direct GEMM. See the PTX ISA
    “Block Scaling for tcgen05.mma” section for the full scale-factor layout
    spec — the layout depends on `scale_vec_size` and the K-dim, and
    is dense (different from per-row or per-channel scaling).

    Parameters:
    :   - **mma\_kind** – Top-level block-scale kind (for example `MXF8F6F4` for
          MXFP8/6/4 narrow formats; selects which block-scaling variants are
          legal). Validated at trace time: must be a block-scaled kind
          (`mxf8f6f4` / `mxf4` / `mxf4nvf4`); use [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma") for
          non-block-scaled kinds.
        - **cta\_group** – `CTA_1` or `CTA_2` (cluster shape (2,1,1)).
        - **d** – TMEM accumulator destination. Validated at trace time to reside
          in tensor memory (dialect operand type `LLVM_PointerTensor`).
        - **a** – A operand — SMEM descriptor (Int64) or TMEM address.
        - **b** – B operand SMEM descriptor (Int64).
        - **idesc** – Packed 32-bit instruction descriptor. Build FP8/FP6/MX
          descriptors via `build()`, and FP4/NVFP4
          descriptors via `build()`.
        - **enable\_input\_d** – Boolean — when False, ignores prior D
          contents (D = A·B·scale instead of D += A·B·scale).
        - **scale\_b** (*scale\_a**,*) – TMEM addresses of the scale-factor tiles
          for A and B respectively. Layout depends on
          `scale_vec_size` — see PTX ISA. Both are validated at trace time to
          reside in tensor memory (dialect operand type `LLVM_PointerTensor`).
        - **scale\_vec\_size** – `Tcgen05MMAScaleVecSize` — selects 1X /
          2X / 4X scale-vector packing within each block. Different
          sizes have different K-dim compatibility tables.
        - **collector\_op** – A operand reuse policy (see [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma")).
        - **a\_shift** – Same semantics as [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma").

    ```python
    if warp == MMA_WARP:
        if nvvm.elect_sync():
            nvvm.tcgen05_mma_block_scale(
                nvvm.Tcgen05MMAKind.MXF8F6F4,
                nvvm.CTAGroup.CTA_1,
                d_tmem, a_smem_desc, b_smem_desc, idesc,
                enable_input_d=k > 0,
                scale_a=scale_a_tmem, scale_b=scale_b_tmem,
                scale_vec_size=nvvm.Tcgen05MMAScaleVecSize.X2,
            )
            nvvm.tcgen05_commit(empty_bar)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_mma\_sp( : *mma\_kind: [Tcgen05MMAKind](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind")*, : *cta\_group: [CTAGroup](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup")*, : *d: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *a: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, : *b: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*, : *idesc: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *enable\_input\_d: int | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*, : *sparse\_metadata: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *collector\_op: [Tcgen05MMACollectorOp](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp") | None = None*, : *a\_shift: bool | None = None*, : *scale\_input\_d: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, : *write\_disable\_mask: [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector") | None = None*, ) → None
:   Issue a structured-sparse 5th-gen tensor-core MMA (`tcgen05.mma.sp`).

    Like [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma"), but operand A is a structured-sparse
    `M x (K/2)` matrix and `sparse_metadata` (in TMEM) maps the compressed
    columns back to the logical K dimension. `D = A * B [+ D]` accumulates
    into TMEM. See [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma") for the shared single-thread-issue /
    elect-safe / commit semantics, A/B descriptor construction, and the
    `collector_op` / `a_shift` / `scale_input_d` / `write_disable_mask`
    parameters.

    Parameters:
    :   - **mma\_kind** ([*Tcgen05MMAKind*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind")) – Data-type/kind selector (see [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma")).
        - **cta\_group** ([*CTAGroup*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup")) – `'CTA_1'` or `'CTA_2'` (see [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma")).
        - **d** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMEM accumulator pointer. Validated at trace time to reside in
          tensor memory (dialect operand type `LLVM_PointerTensor`).
        - **a** (*cutlass.Array**,* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")) – A operand – SMEM descriptor (Int64) or TMEM pointer.
        - **b** (*int**,* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")) – 64-bit SMEM descriptor for B.
        - **idesc** (*int**,* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Packed 32-bit instruction descriptor (sparse bit set).
        - **enable\_input\_d** (*int* *or* [*cutlass.Boolean*](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")) – `D = A*B + D` when true, else `D = A*B`.
        - **sparse\_metadata** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMEM pointer to the sparsity metadata mapping the
          K/2 packed columns to the logical K dimension. Validated at trace
          time to reside in tensor memory.
        - **collector\_op** – A-operand collector policy (see [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma")).
        - **scale\_input\_d** – `[0, 15]`; valid only for `"f16"` / `"tf32"`
          (validated at trace time). See [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma").
        - **write\_disable\_mask** – 4-element (`CTA_1`) / 8-element (`CTA_2`)
          per-row TMEM write-disable vector (validated at trace time).

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_mma\_sp\_block\_scale( : *mma\_kind: ~cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMAKind*, : *cta\_group: ~cutlass.experimental.primitives.nvvm\_wrapper.CTAGroup*, : *d: ~cutlass.Array | ~cutlass.Pointer*, : *a: ~cutlass.Array | ~cutlass.Pointer | ~cutlass.Int64*, : *b: int | ~cutlass.Int64 | ~cutlass.Uint64*, : *idesc: int | ~cutlass.Int32 | ~cutlass.Uint32*, : *enable\_input\_d: int | ~cutlass.Boolean*, : *sparse\_metadata: ~cutlass.Array | ~cutlass.Pointer*, : *scale\_a: ~cutlass.Array | ~cutlass.Pointer*, : *scale\_b: ~cutlass.Array | ~cutlass.Pointer*, : *\**, : *scale\_vec\_size: ~cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMAScaleVecSize | \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocDialectObject | None = None*, : *collector\_op: ~cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMACollectorOp | None = None*, : *a\_shift: bool | None = None*, ) → None
:   Structured-sparse MMA with per-block scale factors (`tcgen05.mma.sp.block_scale`).

    Combines the structured-sparse A path of [`tcgen05_mma_sp()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_sp "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_sp") with the
    per-block scaling of [`tcgen05_mma_block_scale()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale"): A is sparse with
    `sparse_metadata` in TMEM, and blocks of A/B are scaled by the
    `scale_a` / `scale_b` factor tiles (also in TMEM) before accumulation.
    See [`tcgen05_mma_block_scale()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale") for the block-scale descriptor /
    scale-vector details and [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma") for the shared issue/commit
    semantics.

    Parameters:
    :   - **mma\_kind** ([*Tcgen05MMAKind*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind")) – Block-scale kind; validated at trace time to be a
          block-scaled kind (`mxf8f6f4` / `mxf4` / `mxf4nvf4`).
        - **cta\_group** ([*CTAGroup*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup")) – `'CTA_1'` or `'CTA_2'`.
        - **d** – TMEM accumulator pointer; validated to reside in tensor memory.
        - **a** – A operand – SMEM descriptor or TMEM pointer.
        - **b** – 64-bit SMEM descriptor for B.
        - **idesc** – Packed 32-bit instruction descriptor.
        - **enable\_input\_d** – `D = A*B*scale + D` when true, else `D = A*B*scale`.
        - **sparse\_metadata** – TMEM pointer to the sparsity metadata; validated to
          reside in tensor memory.
        - **scale\_b** (*scale\_a**,*) – TMEM addresses of the A/B scale-factor tiles;
          both validated at trace time to reside in tensor memory.
        - **scale\_vec\_size** – 1X / 2X / 4X scale-vector packing
          (see [`tcgen05_mma_block_scale()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale")).
        - **collector\_op** – A-operand collector policy (see [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma")).

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_mma\_ws( : *mma\_kind: [Tcgen05MMAKind](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind")*, : *d: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *a: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, : *b: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*, : *idesc: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *enable\_input\_d: int | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*, : *\**, : *collector\_b\_buffer: [Tcgen05MMACollectorBBuffer](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorBBuffer "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorBBuffer") | None = None*, : *collector\_op: [Tcgen05MMACollectorOp](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp") | None = None*, : *col\_b\_zero\_mask: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, ) → None
:   Issue a weight-stationary tcgen05 MMA.

    Emits `tcgen05.mma.ws.cta_group::1`. The instruction initiates
    `D = A*B+D` for a dense A matrix and uses a B-matrix collector buffer for
    weight-stationary convolution-style reuse. A may be an SMEM descriptor or
    a TMEM pointer; B is an SMEM descriptor. When `enable_input_d` is false,
    the operation computes `D = A*B`.

    Parameters:
    :   - **mma\_kind** ([*Tcgen05MMAKind*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind")) – Data-type/kind selector. Public PTX ISA 9.3 forms support
          `F16`, `TF32`, `F8F6F4`, and `INT8` for `tcgen05.mma.ws`.
          (The dialect’s non-block-scale kind attribute rejects block-scaled
          kinds; use [`tcgen05_mma_block_scale()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma_block_scale") for those.)
        - **d** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMEM accumulator destination. Validated at trace time to reside
          in tensor memory (dialect operand type `LLVM_PointerTensor`).
        - **a** (*cutlass.Array**,* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")) – A operand as an SMEM descriptor or TMEM pointer.
        - **b** (*int**,* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")) – 64-bit SMEM descriptor for B.
        - **idesc** (*int**,* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Packed 32-bit instruction descriptor.
        - **enable\_input\_d** (*int* *or* [*cutlass.Boolean*](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")) – When true, accumulate into existing D; when false,
          compute `D = A*B`.
        - **collector\_b\_buffer** ([*Tcgen05MMACollectorBBuffer*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorBBuffer "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorBBuffer")*,* *optional*) – Optional B collector buffer selector
          (`B0` through `B3`). PTX defaults to `B0` with `DISCARD` when
          no collector usage is specified.
        - **collector\_op** ([*Tcgen05MMACollectorOp*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp")*,* *optional*) – Optional B collector operation: `FILL`, `USE`,
          `LASTUSE`, or `DISCARD`.
        - **col\_b\_zero\_mask** (*int**,* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional zero-column-mask descriptor for B columns
          that should be treated as zero regardless of SMEM contents.

    ```python
    if nvvm.elect_sync():
        nvvm.tcgen05_mma_ws(
            nvvm.Tcgen05MMAKind.INT8,
            d_tmem,
            a_tmem,
            b_desc,
            idesc,
            k > 0,
            collector_b_buffer=nvvm.Tcgen05MMACollectorBBuffer.B2,
            collector_op=nvvm.Tcgen05MMACollectorOp.USE,
        )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_mma\_ws\_sp( : *mma\_kind: [Tcgen05MMAKind](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind")*, : *d: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *a: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, : *b: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*, : *idesc: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *enable\_input\_d: int | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*, : *sparse\_metadata: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *collector\_b\_buffer: [Tcgen05MMACollectorBBuffer](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorBBuffer "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorBBuffer") | None = None*, : *collector\_op: [Tcgen05MMACollectorOp](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp") | None = None*, : *col\_b\_zero\_mask: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, ) → None
:   Issue a sparse weight-stationary tcgen05 MMA.

    Emits `tcgen05.mma.ws.sp.cta_group::1`. The instruction initiates
    `D = A*B+D` where A is a structured sparse matrix packed as
    `M x (K/2)` and accompanied by sparse metadata in TMEM. A may be an SMEM
    descriptor or a TMEM pointer; B is an SMEM descriptor. When
    `enable_input_d` is false, the operation computes `D = A*B`.

    Parameters:
    :   - **mma\_kind** ([*Tcgen05MMAKind*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMAKind")) – Data-type/kind selector. Public PTX ISA 9.3 forms support
          `F16`, `TF32`, `F8F6F4`, and `INT8` for `tcgen05.mma.ws.sp`.
        - **d** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMEM accumulator destination.
        - **a** (*cutlass.Array**,* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")) – A operand as an SMEM descriptor or TMEM pointer.
        - **b** (*int**,* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")) – 64-bit SMEM descriptor for B.
        - **idesc** (*int**,* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Packed 32-bit instruction descriptor.
        - **enable\_input\_d** (*int* *or* [*cutlass.Boolean*](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")) – When true, accumulate into existing D; when false,
          compute `D = A*B`.
        - **sparse\_metadata** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMEM address or pointer for sparse-A metadata.
        - **collector\_b\_buffer** ([*Tcgen05MMACollectorBBuffer*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorBBuffer "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorBBuffer")*,* *optional*) – Optional B collector buffer selector
          (`B0` through `B3`). PTX defaults to `B0` with `DISCARD` when
          no collector usage is specified.
        - **collector\_op** ([*Tcgen05MMACollectorOp*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05MMACollectorOp")*,* *optional*) – Optional B collector operation: `FILL`, `USE`,
          `LASTUSE`, or `DISCARD`.
        - **col\_b\_zero\_mask** (*int**,* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64")*, or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Optional zero-column-mask descriptor for B columns
          that should be treated as zero regardless of SMEM contents.

    ```python
    if nvvm.elect_sync():
        nvvm.tcgen05_mma_ws_sp(
            nvvm.Tcgen05MMAKind.TF32,
            d_tmem,
            a_tmem,
            b_desc,
            idesc,
            k > 0,
            sparse_metadata_tmem,
            collector_b_buffer=nvvm.Tcgen05MMACollectorBBuffer.B1,
            collector_op=nvvm.Tcgen05MMACollectorOp.FILL,
        )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_relinquish\_alloc\_permit( : *\**, : *group: [CTAGroup](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup") | None = None*, ) → None
:   Release the SM-level TMEM allocation permit after `tcgen05_alloc`.

    Emits `tcgen05.relinquish_alloc_permit.sync.aligned`. The `.sync.aligned`
    qualifier means this is a **warp-collective** instruction: all 32 threads
    of the warp must execute it simultaneously.

    **CTA\_2 placement**: call from a warp that runs on both CTAs at a
    convergence point (e.g. warp 0 right after `barrier_cluster_wait` and before any
    warp-role branch that may diverge between CTAs).

    Parameters:
    :   **group** – `'CTA_1'` (default) or `'CTA_2'`. See [`CTAGroup`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup").

    ```python
    # CTA_1: inside the MMA warp, outside elect_sync (both alloc and
    # relinquish are .sync.aligned — all 32 threads must participate).
    if warp == MMA_WARP:
        nvvm.tcgen05_alloc(tmem_ptr, num_cols, group="cta_1")
        nvvm.tcgen05_relinquish_alloc_permit(group="cta_1")

    # CTA_2: from warp 0 BEFORE warp-role branches — ensures both CTAs converge
    if warp == 0:
        nvvm.tcgen05_alloc(tmem_ptr, num_cols, group="cta_2")
    nvvm.barrier_cluster_wait()
    nvvm.barrier_cta_sync()
    # Both CTAs' warp 0 reach here simultaneously → safe collective call:
    if warp == 0:
        nvvm.tcgen05_relinquish_alloc_permit(group="cta_2")
    # Only NOW diverge into warp-specialized roles (TMA / MMA / epilogue)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_shift( : *taddr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *group: [CTAGroup](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup") | None = None*, ) → None
:   Shift TMEM rows down by one within each sub-partition (`tcgen05.shift`).

    Asynchronous instruction that shifts the 32-byte elements of the matrix at
    `taddr` downwards by one row across all rows except the last of each
    sub-partition. Used to advance a sliding-window operand in TMEM (e.g. the
    A operand of a chained [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma")). Staged into the tensor-core
    issue queue alongside `tcgen05.mma` / `tcgen05.cp`; pair with
    [`tcgen05_commit()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_commit "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_commit") (or a downstream consumer) to observe completion.

    Parameters:
    :   - **taddr** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMEM base pointer of the matrix whose rows are shifted.
          Validated at trace time: must reside in tensor memory when the passed
          object exposes `.space` (dialect operand type `LLVM_PointerTensor`).
        - **group** ([*CTAGroup*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup")*,* *optional*) – `'CTA_1'` (default) or `'CTA_2'`. Selects the
          single-CTA vs 2-CTA shift; must match the group of the other
          `tcgen05` ops in the kernel. See [`CTAGroup`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.CTAGroup "cutlass.experimental.primitives.nvvm_wrapper.CTAGroup").

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_st( : *shape: [Tcgen05LdStShape](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05LdStShape "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05LdStShape")*, : *tmem\_addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *val: [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Vector](basic_data_types.md#cutlass.Vector "cutlass.Vector")*, : *\**, : *unpack: bool | None = None*, : *offset: int | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | [Uint64](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64") | None = None*, ) → None
:   Store register data into TMEM.

    Emits `tcgen05.st.sync.aligned.{shape}{.xN}{.unpack}.b32`. The issuing
    warp collectively stores into TMEM; each lane supplies its own register
    slice for the warp’s accessible TMEM sub-partition. Mirror of
    [`tcgen05_ld()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld"): same shape literals,
    same address encoding, same TMEM access restriction (see
    [`tcgen05_ld()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld") for the canonical
    warp-position-to-lane-range table and the `M_TILE=128` epilogue /
    writer implications). Read both docstrings together — [`tcgen05_ld()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld")
    is the source of truth for the access table; this docstring covers
    only the store-direction differences.

    Accepts any `Vector` or scalar for *val*; non-`Int32` values are
    bitcast to `Int32` internally (the hardware requires i32 register
    words) — no manual `.bitcast(Int32)` wrapping needed at the call site.

    Parameters:
    :   - **shape** ([*Tcgen05LdStShape*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05LdStShape "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05LdStShape")) – Store shape (PTX `.shape1` / `.shape2`). One of
          `"16x32bx2"`, `"16x64b"`, `"32x32b"`, `"16x128b"`,
          `"16x256b"`. Same access restriction as [`tcgen05_ld()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld").
        - **tmem\_addr** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – Pointer/Array to the TMEM location;
          `(row << 16) | col` encoding (see [`tcgen05_ld()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld") for full
          encoding details).
          Validated at trace time: must reside in tensor memory when the passed
          object exposes `.space` (the dialect operand type is
          `LLVM_PointerTensor`); opaque pointers are deferred to the verifier.
        - **val** ([*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Vector*](basic_data_types.md#cutlass.Vector "cutlass.Vector")) – Value to store. `Int32` scalar or `Vector[any dtype]`;
          non-Int32 dtypes are auto-bitcast to `Int32` internally.
        - **unpack** (*bool**,* *optional*) – Enable element unpacking — mirror of
          `pack` on [`tcgen05_ld()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld").
        - **offset** (*int* *or* [*cutlass.Int64*](basic_data_types.md#cutlass.Int64 "cutlass.Int64") *or* [*cutlass.Uint64*](basic_data_types.md#cutlass.Uint64 "cutlass.Uint64")*,* *optional*) – Required (and only valid) for `"16x32bx2"` — column
          offset added to `tmem_addr` at runtime. Must be `None` for all
          other shapes.

    Raises:
    :   - **ValueError** – `shape` is not a recognized literal; `offset`
          is set for a shape other than `"16x32bx2"` (or missing for
          `"16x32bx2"`).
        - **TypeError** – `tmem_addr` exposes a `.space` that is not tensor
          memory (TMEM).

    ```python
    # Store a vector of accumulator data back into TMEM.
    nvvm.tcgen05_st("32x32b", tmem_ptr, data_vec)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tcgen05\_wait( : *kind: [Tcgen05Wait](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.Tcgen05Wait "cutlass.experimental.primitives.nvvm_wrapper.Tcgen05Wait")*, ) → None
:   Wait for pending TMEM load or store operations to complete.

    Emits `tcgen05.wait::{ld|st}.sync.aligned`. Place after
    [`tcgen05_ld()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_ld")
    (`LOAD`) to ensure the TMEM→register transfer has finished
    before any thread reads the returned register values, or after
    [`tcgen05_st()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_st "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_st") (`STORE`) to ensure register→TMEM writes
    are visible before a downstream [`tcgen05_mma()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma "cutlass.experimental.primitives.nvvm_wrapper.tcgen05_mma") reads them.

    Parameters:
    :   **kind** – `"load"` or `"store"`.

    ```python
    # Epilogue: read accumulator from TMEM into registers
    c_vec = nvvm.tcgen05_ld(shape, tmem_addr, num=num_cols)
    nvvm.tcgen05_wait("load")
    # c_vec is now safe to use
    ```

cutlass.experimental.primitives.nvvm\_wrapper.tensormap\_cp\_fenceproxy( : *dst: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *src: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *size: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *scope: [MemScope](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.base_dsl.array.MemScope")*, ) → None
:   Copy a TMA descriptor with a proxy fence at the requested scope.

    Emits `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.<scope>.sync.aligned
    [dst], [src], size;`. Copies `size` bytes of TMA descriptor
    data from `src` to `dst` and acts as a release fence between
    the generic and tensormap proxies, so subsequent TMA ops see the
    new descriptor at the chosen visibility scope (`cta` / `cluster`
    / `gpu` / `sys`).

    Parameters:
    :   - **dst** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMA-descriptor destination pointer / array (typically
          a constant-banked descriptor slot).
        - **src** (*cutlass.Array* *or* [*cutlass.Pointer*](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")) – TMA-descriptor source pointer / array.
        - **size** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – Number of bytes to copy; the descriptor size (128 for
          standard TMA descriptors).
        - **scope** ([*MemScope*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MemScope "cutlass.experimental.primitives.nvvm_wrapper.MemScope")) – Visibility scope of the release fence.

    Raises:
    :   **ValueError** – if a statically known `size` is not 128.

cutlass.experimental.primitives.nvvm\_wrapper.tensormap\_replace( : *field: [TensormapField](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapField "cutlass.experimental.primitives.nvvm_wrapper.TensormapField")*, : *addr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *\**, : *new\_value: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | None = None*, : *ord: int | None = None*, : *new\_value\_attr: [TensormapElemtype](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapElemtype "cutlass.experimental.primitives.nvvm_wrapper.TensormapElemtype") | [TensormapInterleaveLayout](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapInterleaveLayout "cutlass.experimental.primitives.nvvm_wrapper.TensormapInterleaveLayout") | [TensormapSwizzleMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapSwizzleMode "cutlass.experimental.primitives.nvvm_wrapper.TensormapSwizzleMode") | [TensormapSwizzleAtomicity](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapSwizzleAtomicity "cutlass.experimental.primitives.nvvm_wrapper.TensormapSwizzleAtomicity") | [TensormapFillMode](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapFillMode "cutlass.experimental.primitives.nvvm_wrapper.TensormapFillMode") | None = None*, ) → None
:   Replace one field of an in-memory TMA tensor-map descriptor.

    Emits `tensormap.replace` against the descriptor at `addr` (global or
    shared memory). Used to patch a TMA descriptor at runtime, e.g. swap the
    `global_address` or resize a `global_dim` between launches without
    rebuilding the whole descriptor on the host.

    Fields split into two groups:

    - **integer-valued** (pass `new_value`): `global_address`, `rank`,
      `box_dim`, `global_dim`, `global_stride`, `element_stride`.
      `global_address` / `global_stride` are 64-bit; the rest are 32-bit
      (coerced for you). `rank` takes one less than the desired tensor rank
      (zero-based).
    - **enum-valued** (pass `new_value_attr`): `elemtype`,
      `interleave_layout`, `swizzle_mode`, `swizzle_atomicity`,
      `fill_mode`, each taking the matching enum.

    The `ord` (dimension ordinal) is required for `box_dim`,
    `global_dim`, `global_stride`, and `element_stride`, and rejected for
    all other fields.

    Parameters:
    :   - **field** ([*TensormapField*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapField "cutlass.experimental.primitives.nvvm_wrapper.TensormapField")) – Which descriptor field to replace.
        - **addr** – Pointer/Array to the tensor-map descriptor (global or shared).
        - **new\_value** – New value for an integer-valued field (coerced to
          `Int64` or `Int32` by field width). Mutually exclusive with
          `new_value_attr`.
        - **ord** – Dimension ordinal of the field across the tensor; required for
          `box_dim` / `global_dim` / `global_stride` / `element_stride`
          and rejected otherwise. The valid range is enforced by the dialect
          verifier for the target build.
        - **new\_value\_attr** – New value for an enum-valued field, as the matching
          enum (or its string). Mutually exclusive with `new_value`.

    Raises:
    :   - **ValueError** – if `field` is not a [`TensormapField`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.TensormapField "cutlass.experimental.primitives.nvvm_wrapper.TensormapField") (raw NVVM
          dialect enums are rejected); if the wrong one of
          `new_value` / `new_value_attr` is supplied for the field; if
          `ord` is supplied for a field that forbids it or omitted for one that
          requires it, or is outside the dialect’s valid range; or if
          `new_value_attr` is not
          the enum type the field expects.
        - **TypeError** – if `ord` is supplied but is not a Python `int`.

    ```python
    # Patch the source pointer of a copied TMA descriptor at runtime.
    if nvvm.elect_sync():
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_ADDRESS, desc_ptr,
            new_value=cutlass.Int64(new_base),
        )
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_DIM, desc_ptr,
            new_value=cutlass.Int32(new_dim), ord=0,
        )
    ```

cutlass.experimental.primitives.nvvm\_wrapper.trace\_mark( : *event\_type: int*, : *domain: str*, : *event: str*, : *\**, : *payload: [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Int64](basic_data_types.md#cutlass.Int64 "cutlass.Int64") | None = None*, : *payload\_descriptor: str | None = None*, ) → None
:   Wrapper over `nvvm.trace_mark`.

cutlass.experimental.primitives.nvvm\_wrapper.vote\_sync( : *mask: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, : *pred: int | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")*, : *kind: [VoteSync](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.VoteSync "cutlass.experimental.primitives.nvvm_wrapper.VoteSync")*, ) → [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")
:   Perform a collective warp vote across masked lanes.

    Maps to `vote.sync.{all|any|uni|ballot}` — each participating lane
    contributes a boolean predicate and all receive a shared result. All
    lanes named in *mask* must be actively executing the instruction
    (convergence requirement).

    **VoteSyncKind variants:**

    - `"all"` → `Boolean`: `True` iff *every* masked lane
      set `pred = True`.
    - `"any"` → `Boolean`: `True` iff *at least one* masked
      lane set `pred = True`.
    - `"uni"` → `Boolean`: `True` iff all masked lanes cast
      the *same* vote (all-True **or** all-False). Use to detect uniform
      control flow without requiring all lanes to be true.
    - `"ballot"` → `Int32`: 32-bit bitmask where bit *i* is
      set when lane *i* set `pred = True`. Bit *i* is 0 for lanes not
      in *mask*.

    Parameters:
    :   - **mask** (*int* *or* [*cutlass.Int32*](basic_data_types.md#cutlass.Int32 "cutlass.Int32") *or* [*cutlass.Uint32*](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")) – 32-bit member mask; bit *i* set means lane *i* participates.
          All named lanes must execute the instruction. Pass `0xFFFFFFFF`
          for a full-warp vote.
        - **pred** (*int* *or* [*cutlass.Boolean*](basic_data_types.md#cutlass.Boolean "cutlass.Boolean")) – Per-lane boolean vote input — what each lane contributes
          to the collective result. Pass `~pred` to vote on the negation.
        - **kind** ([*VoteSync*](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.VoteSync "cutlass.experimental.primitives.nvvm_wrapper.VoteSync")) – Vote mode — determines the semantics and return type.

    Returns:
    :   `Boolean` for `all`/`any`/`uni`; `Int32` bitmask for
        `ballot`.

    Return type:
    :   [cutlass.Boolean](basic_data_types.md#cutlass.Boolean "cutlass.Boolean") or [cutlass.Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")

    Raises:
    :   **ValueError** – if a static *mask* does not fit in 32 bits, or if *kind*
        is not one of `"all"`, `"any"`, `"uni"`, `"ballot"`. A runtime
        (non-`int`) *mask* is not checked at trace time.

    **Constraints:**

    - All lanes named in *mask* must reach the instruction; a divergent lane
      causes the remaining lanes to stall indefinitely.
    - Available on SM30+; `ballot` requires SM35+.
    - This wrapper has no `exec_pred` argument.

    ```python
    tx, _, _ = cute.arch.thread_idx()
    is_even = (tx % cutlass.Int32(2)) == cutlass.Int32(0)

    # all: True only when every lane voted True
    all_even = nvvm.vote_sync(0xFFFFFFFF, is_even, "all")

    # any: True when at least one lane voted True
    any_even = nvvm.vote_sync(0xFFFFFFFF, is_even, "any")

    # uni: True when all lanes agree (all-True OR all-False)
    uniform  = nvvm.vote_sync(0xFFFFFFFF, is_even, "uni")

    # ballot: bitmask of lanes that voted True (even lanes → 0x55555555)
    even_mask = nvvm.vote_sync(0xFFFFFFFF, is_even, "ballot")
    ```

cutlass.experimental.primitives.nvvm\_wrapper.wgmma\_commit\_group\_sync\_aligned() → None
:   Commit outstanding async warpgroup MMAs into a group.

    Emits `wgmma.commit_group.sync.aligned`. Bundles all `wgmma.mma_async`
    operations issued by the warpgroup since the last commit into a new group,
    which [`wgmma_wait_group_sync_aligned()`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.wgmma_wait_group_sync_aligned "cutlass.experimental.primitives.nvvm_wrapper.wgmma_wait_group_sync_aligned") can later wait on.

    ```python
    nvvm.wgmma_commit_group_sync_aligned()
    nvvm.wgmma_wait_group_sync_aligned(0)
    ```

cutlass.experimental.primitives.nvvm\_wrapper.wgmma\_fence\_aligned() → None
:   Fence register accesses around async warpgroup MMA.

    Emits `wgmma.fence.sync.aligned`. Orders the executing warpgroup’s
    accesses to the registers/shared memory that feed `wgmma.mma_async`,
    so the async MMA observes a consistent view. Issue once before the
    first `wgmma.mma_async` of a sequence (and after writing its inputs).

    ```python
    nvvm.wgmma_fence_aligned()
    # ... wgmma.mma_async issues ...
    nvvm.wgmma_commit_group_sync_aligned()
    ```

cutlass.experimental.primitives.nvvm\_wrapper.wgmma\_mma\_async( : *results\_: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocMlirType*, : *inouts: ir.Value*, : *descriptor\_a: int | ~cutlass.Int64 | ~cutlass.Uint64*, : *descriptor\_b: int | ~cutlass.Int64 | ~cutlass.Uint64*, : *shape: ~sphinx.ext.autodoc.mock.\_MockObject*, : *type\_a: ~cutlass.experimental.primitives.nvvm\_wrapper.WGMMAType*, : *type\_b: ~cutlass.experimental.primitives.nvvm\_wrapper.WGMMAType*, : *type\_d: ~cutlass.experimental.primitives.nvvm\_wrapper.WGMMAType*, : *scale\_d: ~cutlass.experimental.primitives.nvvm\_wrapper.WGMMAScaleOut*, : *scale\_a: ~cutlass.experimental.primitives.nvvm\_wrapper.WGMMAScaleIn*, : *scale\_b: ~cutlass.experimental.primitives.nvvm\_wrapper.WGMMAScaleIn*, : *layout\_a: ~cutlass.experimental.primitives.nvvm\_wrapper.MMALayout*, : *layout\_b: ~cutlass.experimental.primitives.nvvm\_wrapper.MMALayout*, : *\**, : *satfinite: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAIntOverflow | None = None*, ) → ir.Value
:   Wrapper over `nvvm.wgmma_mma_async`.

    Returns an LLVM struct. Caller provides the raw MLIR result type
    as *results\_*.

cutlass.experimental.primitives.nvvm\_wrapper.wgmma\_wait\_group\_sync\_aligned(*group: int*) → None
:   Wait until at most `group` async warpgroup-MMA groups are pending.

    Emits `wgmma.wait_group.sync.aligned`. Blocks the warpgroup until no
    more than `group` previously-committed MMA groups remain in flight;
    `group=0` waits for all of them. Registers written by completed MMAs
    are safe to read afterwards.

    Parameters:
    :   **group** (*int*) – Maximum number of committed groups allowed to remain
        pending. Must be a non-negative `int`.

    Raises:
    :   **ValueError** – `group` is a negative `int`.

    ```python
    nvvm.wgmma_wait_group_sync_aligned(0)  # drain all pending MMAs
    ```

cutlass.experimental.primitives.nvvm\_wrapper.wmma\_load( : *res: \_install\_cutlass\_mlir\_autodoc\_stub.<locals>.\_DocMlirType*, : *ptr: ~cutlass.Array | ~cutlass.Pointer*, : *stride: int | ~cutlass.Int32 | ~cutlass.Uint32*, : *m: int*, : *n: int*, : *k: int*, : *layout: ~cutlass.experimental.primitives.nvvm\_wrapper.MMALayout*, : *eltype: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAType*, : *frag: ~cutlass.experimental.primitives.nvvm\_wrapper.MMAFrag*, ) → ir.Value
:   Wrapper over `nvvm.wmma_load`.

    Returns an LLVM struct. Caller provides the raw MLIR result type
    as *res*.

cutlass.experimental.primitives.nvvm\_wrapper.wmma\_store( : *ptr: Array | [Pointer](basic_data_types.md#cutlass.Pointer "cutlass.Pointer")*, : *m: int*, : *n: int*, : *k: int*, : *layout: [MMALayout](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMALayout "cutlass.experimental.primitives.nvvm_wrapper.MMALayout")*, : *eltype: [MMAType](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.MMAType "cutlass.experimental.primitives.nvvm_wrapper.MMAType")*, : *args: ir.Value*, : *stride: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32") | [Uint32](basic_data_types.md#cutlass.Uint32 "cutlass.Uint32")*, ) → None
:   Wrapper over `nvvm.wmma_store`.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05LdStShape(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMAScaleVecSize
:   Bases: `object`

    Compatibility namespace for tcgen05 block-scale selectors.

    Older call sites spell the selector as `Tcgen05MMAScaleVecSize` while
    newer NVVM bindings split 1X/2X/4X and block16/block32 into separate
    enums. Keep both spellings available at the wrapper boundary.

    X1
    :   alias of `_MockObject`

    X2
    :   alias of `_MockObject`

    X4
    :   alias of `_MockObject`

    DEFAULT
    :   alias of `_MockObject`

    Default
    :   alias of `_MockObject`

    BLOCK16
    :   alias of `_MockObject`

    BLOCK32
    :   alias of `_MockObject`

*class* cutlass.experimental.primitives.nvvm\_wrapper.TensormapElemtype(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.TensormapField(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.TensormapFillMode(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.TensormapInterleaveLayout(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.TensormapSwizzleAtomicity(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.TensormapSwizzleMode(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.AtomicOp(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.BarrierRedux(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.CTAGroup(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.CVTPackFloat(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.CacheLevel(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.ClusterLaunchControlQueryType(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.CmpOp(*value*)
:   Bases: `StrEnum`

    PTX comparison operator for `setp` (PTX ISA §9.7.6).

*class* cutlass.experimental.primitives.nvvm\_wrapper.ConvertScale(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.CpReduceOp(*value*)
:   Bases: `StrEnum`

    Reduction op for `cp.reduce.async.bulk` (non-TMA, PTX §9.7.9.25.4.2).

*class* cutlass.experimental.primitives.nvvm\_wrapper.CpReduceType(*value*)
:   Bases: `StrEnum`

    Element type for `cp.reduce.async.bulk` (non-TMA, PTX §9.7.9.25.4.2).

*class* cutlass.experimental.primitives.nvvm\_wrapper.CvtaSize(*value*)
:   Bases: `StrEnum`

    Address-width qualifier for `cvta`: `.u32` or `.u64`.

*class* cutlass.experimental.primitives.nvvm\_wrapper.CvtaSpace(*value*)
:   Bases: `StrEnum`

    Target address space for `cvta` (PTX ISA §9.7.9.20).

*class* cutlass.experimental.primitives.nvvm\_wrapper.DotAccumulateType(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.EvictPriority(*value*)
:   Bases: `StrEnum`

    Eviction priority hint for cache lines.

    Controls the priority of cache line eviction.

    Reference: PTX ISA - Cache Eviction Priority

*class* cutlass.experimental.primitives.nvvm\_wrapper.FPRoundingMode(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.GridDepAction(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.IntRoundingMode(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.L1EvictKind(*value*)
:   Bases: `StrEnum`

    L1 cache eviction priority hint.

    Controls the eviction policy for L1 cache lines. Members match the
    5-member subset of [`EvictPriority`](primitives.md#cutlass.experimental.primitives.nvvm_wrapper.EvictPriority "cutlass.experimental.primitives.nvvm_wrapper.EvictPriority") that the L1 path supports
    (the L2 path adds the sm\_90+ `*_demote` / `*_near` variants).

    Reference: PTX ISA - Cache Eviction Priority
    <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-eviction-priority>

*class* cutlass.experimental.primitives.nvvm\_wrapper.L2PrefetchSize(*value*)
:   Bases: `StrEnum`

    L2 cache prefetch size hint.

    Specifies the prefetch granularity for L2 cache operations.

    Reference: PTX ISA - Data Movement Instructions
    <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu>

*class* cutlass.experimental.primitives.nvvm\_wrapper.LoadCacheModifier(*value*)
:   Bases: `StrEnum`

    Cache operation modifier for load instructions.

    Controls L1/L2 cache behavior for memory loads.

    Reference: PTX ISA - Cache Operators
    <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>

    PTX cache operators for ld:
    :   .ca - Cache at all levels (L1 and L2), default
        .cg - Cache at global level (L2 only, bypass L1)
        .cs - Cache streaming (likely accessed once, evict first)
        .lu - Last use (hint data won’t be needed again)
        .cv - Cache volatile (don’t cache, always fetch from memory)

*class* cutlass.experimental.primitives.nvvm\_wrapper.LoadShape(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.LoadSrcFormat(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MBarrierScope(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MBarrierWait(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MMAB1Op(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MMAFrag(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MMAIntOverflow(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MMAKind(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MMALayout(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MMAType(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MatchSync(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MemOrder(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.MemScope(*value*)
:   Bases: `StrEnum`

    Memory scope for memory operations.

    Controls which threads observe the memory operation effects.
    Used for loads, stores, atomics, and memory barriers.

    Reference: PTX ISA - Scope
    <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scope>

    PTX scope qualifiers:
    :   .cta - Threads within the same CTA (thread block)
        .cluster - Threads within the same cluster
        .gpu - All threads on the same GPU device
        .sys - All threads in the system (including host CPU)

*class* cutlass.experimental.primitives.nvvm\_wrapper.MulMode(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.PermuteMode(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Proxy(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.ReductionOp(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.ReductionType(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.SaturationMode(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.SaturationModeKind(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.SetMaxRegisterAction(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.SetpType(*value*)
:   Bases: `StrEnum`

    Source type suffix for `setp` / `selp` (PTX ISA §9.7.6).

*class* cutlass.experimental.primitives.nvvm\_wrapper.Shfl(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.StoreCacheModifier(*value*)
:   Bases: `StrEnum`

    Cache operation modifier for store instructions.

    Controls L1/L2 cache behavior for memory stores.

    Reference: PTX ISA - Cache Operators
    <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>

    PTX cache operators for st:
    :   .wb - Write-back (cache at all levels), default
        .cg - Cache at global level (L2 only, bypass L1)
        .cs - Cache streaming (likely accessed once)
        .wt - Write-through (write to memory immediately)

*class* cutlass.experimental.primitives.nvvm\_wrapper.StoreShape(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.TMALoadMode(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.TMARedux(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.TMAStoreMode(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05CpMulticast(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05CpShape(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05CpSrcFormat(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05Fence(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMACollectorBBuffer(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMACollectorOp(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05MMAKind(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.Tcgen05Wait(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.TensormapSpace(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.VoteSync(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.WGMMAScaleIn(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.WGMMAScaleOut(*value*)
:   Bases: `StrEnum`

    An enumeration.

*class* cutlass.experimental.primitives.nvvm\_wrapper.WGMMAType(*value*)
:   Bases: `StrEnum`

    An enumeration.

cutlass.experimental.primitives.nvvm\_wrapper.barrier( : *\**, : *barrier\_id: cutlass.cute.typing.Int | None = None*, : *number\_of\_threads: cutlass.cute.typing.Int | None = None*, ) → None
:   Creates a barrier, optionally named.

*class* cutlass.experimental.primitives.nvvm\_wrapper.S2TCopyMode
:   Bases: `object`

    S2T (SMEM->TMEM) copy mode enumeration for `prims.tcgen05_cp`.

    Combines shape and multicast into valid configurations for SMEM-to-TMEM copy.
    Each mode specifies both the data shape and the required warp broadcast pattern.

    Available modes:
    - S2T\_128x256b: 128 rows x 256 bits, no multicast
    - S2T\_128x128b: 128 rows x 128 bits, no multicast
    - S2T\_4x256b: 4 rows x 256 bits, no multicast
    - S2T\_32x128b\_WARPX4: 32 rows x 128 bits, broadcast to all 4 warps
    - S2T\_64x128b\_WARPX2\_01\_23: 64 rows x 128 bits, broadcast to warp pairs (0,1)(2,3)
    - S2T\_64x128b\_WARPX2\_02\_13: 64 rows x 128 bits, broadcast to warp pairs (0,2)(1,3)

cutlass.experimental.primitives.nvvm\_wrapper.make\_tmem\_ptr( : *tmem\_addr: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, : *dtype: type[[Numeric](basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, ) → Array
:   Convert a TMEM address to a typed TMEM `Array` view (address space 6).

    Parameters:
    :   - **tmem\_addr** – The TMEM address value (`int` or a DSL integer).
        - **dtype** – The element type for the returned view.

    Returns:
    :   An `Array` over TMEM (address space 6).

cutlass.experimental.primitives.nvvm\_wrapper.make\_tmem\_ptr\_from\_warp\_row\_col( : *tmem\_base: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, : *warp: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, : *base\_col: int | [Int32](basic_data_types.md#cutlass.Int32 "cutlass.Int32")*, : *dtype: type[[Numeric](basic_data_types.md#cutlass.Numeric "cutlass.Numeric")]*, ) → Array
:   Build a typed TMEM `Array` view for TMEM/SP `warp` at row `warp*32`.

    Each warp in a tcgen05 MMA group owns one sub-partition of the TMEM
    accumulator. The canonical epilogue formula

    ```python
    tmem_sp = warp_idx % 4
    tmem_addr = (tmem_base_row + tmem_sp * 32) << 16 | base_col
    ```

    is bundled here so callers don’t reassemble the bitfield by hand. For shifted
    epilogue ranges such as warps 2..5, pass `warp_idx % 4` rather than the
    logical epilogue rank.

    Parameters:
    :   - **tmem\_base** – TMEM address of the accumulator base (row 0, col 0);
          bits [0:16) hold the starting column, [16:32) hold the starting row.
        - **warp** – TMEM/SP index (0..3), usually `warp_idx % 4`.
        - **base\_col** – Column offset within the accumulator.
        - **dtype** – Element type of the returned TMEM view.

    Returns:
    :   An `Array` over TMEM at `(base_row + warp*32, base_col)`.
