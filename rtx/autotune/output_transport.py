"""Legal compound BF16 row-store proposals, not a device dispatch policy.

Direct-to-TMA transitions can require both a smaller output tile and an extra
operand buffer. Coordinate descent cannot cross an illegal intermediate. Keep
these moves together, preserve arithmetic/register policies, and let whole-op
validation and timing decide whether the extra shared memory is worthwhile.
"""
from dataclasses import replace


def row_store_features(gemm, problem):
    """Static store ownership, not measured traffic or throughput."""
    m,n=gemm.tile_m,gemm.tile_n
    total=((problem.m+m-1)//m)*((problem.n+n-1)//n)
    interior=(problem.m//m)*(problem.n//n)
    bulk=interior if gemm.epilogue=='hybrid' else total if gemm.epilogue=='tma' else 0
    return dict(output_tiles=total,output_bulk_tiles=bulk,
                output_direct_tiles=total-bulk,
                output_bulk_tile_fraction=bulk/max(1,total))


def row_store_candidates(gemm, problem):
    """Yield a small ordered set from existing 64/128/256 output tile axes.

    Only BF16 row outputs belong here: never apply this to FP32 split-dW
    workspaces. Callers must also check their complete adapter's legality.
    There is no padding and no assumption that logical and physical M match.
    """
    geometries=[(gemm.tile_m,gemm.tile_n,gemm.atom_layout_m,gemm.atom_layout_n)]
    choices=[]
    for m in (64,128,256):
        for n in (64,128,256):
            am=2 if m==64 else 8 if m==256 else gemm.atom_layout_m
            an=2 if n==64 else n//64
            choices.append((m,n,am,an))
    choices.sort(key=lambda v:(abs(v[0]-gemm.tile_m)+abs(v[1]-gemm.tile_n),v))
    geometries.extend(choices)
    seen=set()
    # Preserve full-tile recipes before introducing the per-CTA tail branch.
    # Ragged M/N no longer forces every output tile onto scalar direct stores.
    for epilogue in ('tma','hybrid'):
        for m,n,am,an in geometries:
            full=problem.m%m==0 and problem.n%n==0
            if (epilogue=='tma')!=full:continue
            # Inherited buffering first; lower/higher legal buffering remains
            # available when output SMEM makes the inherited depth infeasible.
            for stages in dict.fromkeys((gemm.stages,2,1,3,4)):
                for vector in (4,1):
                    candidate=replace(gemm,tile_m=m,tile_n=n,atom_layout_m=am,
                        atom_layout_n=an,stages=stages,epilogue=epilogue,
                        store_vec=vector,epilogue_stages=1)
                    if candidate not in seen and candidate.rejection(problem) is None:
                        seen.add(candidate)
                        yield candidate
