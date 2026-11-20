"""Paired CUDA-event timing of four original-layout MoE preparation kernels.

Use a frozen benchmark_moe_varlen expert fixture. Neither the active scan,
global X/weight quantization nor any consumer is timed. Whole expert/layer
latency belongs in the composition benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter


def run(args):
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest, _use_source, write

    _use_source(args.source)
    import torch
    from benchmarks.benchmark_moe_joint_recompute import _capture
    from benchmarks.profile_moe import hardware_attributes, statistics_ms
    from rtx._moe_execution import Builder, layout
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.kernels.mxfp8_moe_active import compile_active_prefix
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_quant, compile_packed_input_gather
    from rtx.kernels.mxfp8_moe_dy import compile_routed_dy_quant
    from rtx.kernels.mxfp8_moe_warp_quant import compile_indexed_expert_transpose
    from rtx.moe import MXFP8MoEConfig, _config_key

    started = perf_counter()
    source = args.source.resolve()
    digest = _python_source_digest(source)
    fixture = torch.load(args.fixture, map_location="cuda", weights_only=True)
    cfg_dict = json.loads(args.config.read_text())
    if "configs" in cfg_dict:
        cfg_dict = cfg_dict["configs"]["active_both"]
    cfg = MXFP8MoEConfig.from_dict(cfg_dict)
    p = cfg.prepared
    if not (p.active_prefix and p.input_preparation == "indexed_transpose"
            and p.dy_preparation == "fused" and p.weight_preparation == "dual_orientation"
            and p.quant.scale_layout == "row_major"):
        raise ValueError("A frozen active indexed/separate-transpose expert config is required")
    x, dy, ids, weights, inverse = (fixture[name] for name in ("x", "dy", "ids", "weights", "inverse"))
    n, d = x.shape
    e, c = ids.shape
    k, h = inverse.shape[1], fixture["down"].shape[-1]
    cs = p.physical_capacity(c)
    width = (d+31)//32*32
    shape = (n, d, e, c, k, h)
    alignment = Builder(shape, cfg, "backward").recompute_alignment()

    def make(rows, cols, dtype):
        return torch.empty(rows, cols, device="cuda", dtype=dtype)

    def pair(rows, cols):
        return make(rows, cols, torch.float8_e4m3fn), make(rows, cols//32, torch.uint8)

    gx = pair(n, width)
    fwd, bwd, dyr = (pair(e*cs, width) for _ in range(3))
    dyt, xt = (pair(e*d, cs) for _ in range(2))
    mapping = make(e*cs, 1, torch.int32)
    prefix = torch.empty((1,), device="cuda", dtype=torch.int32)
    compile_expert_quant(e, ((n,d,"row",False,1),), p.quant, p.transpose_mode, p.quant_grid,
                         p.transpose_warp_tokens, p.transpose_warp_raster)(x, (x,), gx[:1], gx[1:])
    compile_active_prefix(n,e,c,k)(inverse,prefix)
    active = int(prefix[0])

    def extent(rows, tile):
        return min(cs, (rows+tile-1)//tile*tile)

    fwd_rows = extent(extent(active,p.down.tile_m),p.gate_up.tile_m)
    bwd_rows = extent(active,alignment)
    dy_rows = extent(bwd_rows,p.down_dx.tile_m)
    dyt_cols, xt_cols = extent(active,p.down_dw.tile_k), extent(active,p.gate_dw.tile_k)
    map_rows = max(bwd_rows,dyt_cols,xt_cols)
    gather = (n,d,e,c,k,cs,cfg.movement_threads,p.quant.scale_layout)
    common = (p.quant,p.transpose_warp_tokens,p.transpose_warp_raster,p.quant_grid)
    fwd_args = (*gx,x,ids,inverse,*fwd,x)
    bwd_args = (*gx,x,ids,inverse,*bwd,mapping)
    dy_args = (dy,mapping,weights,*dyr,*dyt)
    xt_args = (x,mapping,*xt)

    def bind(compiler, params, runtime):
        call = compiler(*params)
        return lambda: call(*runtime)

    full = (
        bind(compile_packed_input_gather,gather,fwd_args),
        bind(compile_packed_input_gather,(*gather,False,True),bwd_args),
        bind(compile_routed_dy_quant,(n,d,e,c,cs,*common),dy_args),
        bind(compile_indexed_expert_transpose,(n,d,e,cs,*common),xt_args),
    )
    cases = {"full": full}
    if not args.full_only:
        bounded = (
            bind(compile_packed_input_gather,(*gather,False,False,p.down.tile_m,p.gate_up.tile_m),(*fwd_args,prefix)),
            bind(compile_packed_input_gather,(*gather,False,True,alignment,p.recompute.tile_m,
                          (p.down_dw.tile_k,p.gate_dw.tile_k)),(*bwd_args,prefix)),
            bind(compile_routed_dy_quant,(n,d,e,c,cs,*common,alignment,p.down_dw.tile_k,p.down_dx.tile_m),(*dy_args,prefix)),
            bind(compile_indexed_expert_transpose,(n,d,e,cs,*common,p.gate_dw.tile_k),(*xt_args,prefix)),
        )
        gather_with_full_map = bind(compile_packed_input_gather,
            (*gather,False,True,alignment,p.recompute.tile_m,(p.down_dw.tile_k,p.gate_dw.tile_k),True),
            (*bwd_args,prefix))
        cases.update(gather=(bounded[0],gather_with_full_map,full[2],full[3]),
                     dy=(full[0],full[1],bounded[2],full[3]), xt=(*full[:3],bounded[3]), all=bounded)

    def sequence(kernels):
        def step():
            for launch in kernels:
                launch()
        return step

    def read_views(name):
        bounded_gather = name in ("all","gather")
        bounded_dy = name in ("all","dy")
        bounded_xt = name in ("all","xt")
        # When another producer stays full it needs the full initialized map.
        r_f,r_x,r_d,r_t,r_xt,r_m = (
            fwd_rows if bounded_gather else cs, bwd_rows if bounded_gather else cs,
            dy_rows if bounded_dy else cs, dyt_cols if bounded_dy else cs, xt_cols if bounded_xt else cs,
            map_rows if name == "all" else cs,
        )
        buffers = [value.view(e,cs,-1).view(torch.uint8)[:, :rows]
                   for (q,s),rows in ((fwd,r_f),(bwd,r_x),(dyr,r_d)) for value in (q,s)]
        buffers += [value.view(e,d,-1).view(torch.uint8)[..., :cols//div]
                    for (q,s),cols in ((dyt,r_t),(xt,r_xt)) for div,value in ((1,q),(32,s))]
        return (*buffers,mapping.view(e,cs)[:,:r_m])

    sequence(full)()
    reference = tuple(v.clone() for v in read_views("full"))
    graphs = {}
    stream = torch.cuda.Stream()
    for name,kernels in cases.items():
        graphs[name],_ = _capture(sequence(kernels),stream)
        graphs[name].replay()
        for actual,expected in zip(read_views(name),reference,strict=True):
            if actual.ndim == 2:
                sliced = expected[:,:actual.shape[1]]
            else:
                sliced = expected[:,:actual.shape[1],:actual.shape[2]]
            assert torch.equal(actual,sliced), name
        print("VALIDATED",name,flush=True)
    del reference
    torch.cuda.synchronize()
    setup_s = perf_counter()-started

    rounds = {name:[] for name in cases}
    for visit in range(4):
        for name in (cases if visit%2 == 0 else reversed(cases)):
            _warm_measurement_graph(graphs[name])
            samples,meta = measure_cuda_batches(graphs[name].replay, samples=7, calls_per_sample=1,
                                                 min_batch_ms=5., device=x.device)
            stats = statistics_ms(samples)
            rounds[name].append(dict(visit=visit,measurement=meta,**stats))
            print("TIME",visit,name,stats["median_ms"],flush=True)
    for graph in graphs.values():
        graph.reset()

    def modeled(rows_f,rows_x,rows_y,cols_y,cols_x,rows_map):
        tiles = (max(rows_y,cols_y)+p.quant.transposed_tile_k-1)//p.quant.transposed_tile_k
        feature_tiles=(d+p.quant.transposed_tile_rows-1)//p.quant.transposed_tile_rows
        warp_width=64//p.transpose_warp_tokens
        return dict(
            rows_m={"forward_x":rows_f,"backward_x":rows_x,"weighted_dy":rows_y,"index_map":rows_map},
            rows_k={"weighted_dy_transpose":cols_y,"indexed_x_transpose":cols_x},
            fp8_scale_payload_write_bytes=e*((rows_f+rows_x+rows_y)*(width+width//32)
                                             +d*(cols_y+cols_y//32+cols_x+cols_x//32)),
            index_write_bytes=e*rows_map*4,
            gather_warp_rows=e*(rows_f+max(rows_x,rows_map)),
            routed_dy_tile_visits=e*tiles*feature_tiles,
            indexed_x_warp_tasks=e*((d+warp_width-1)//warp_width)*(cols_x//32),
        )

    assert digest == _python_source_digest(source)
    plans = [layout(shape,_config_key(cfg),phase)[1:3] for phase in ("forward","backward")]
    arena_bytes = [sum(dtype.itemsize*size for dtype,size in zip(dtypes,sizes,strict=True))
                   for dtypes,sizes in plans]
    write(args.output, dict(
        source=str(source), runtime_digest=digest, script_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=str(torch.__version__), cuda_device=str(torch.cuda.get_device_properties(x.device)),
        hardware=hardware_attributes(), fixture=str(args.fixture), fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        tensor_hashes={name:_tensor_digest(fixture[name]) for name in ("x","dy","gu","down","ids","weights","inverse")},
        shape=shape, storage_capacity=cs, active_group_rows=active, config=cfg_dict,
        setup_s=setup_s, fixed_static_arena_bytes=dict(zip(("forward","backward"),arena_bytes)),
        write_work_model=dict(full=modeled(cs,cs,cs,cs,cs,cs),
                              bounded=modeled(fwd_rows,bwd_rows,dy_rows,dyt_cols,xt_cols,map_rows)),
        rounds=rounds, median_ms={name:statistics.median(v["median_ms"] for v in visits)
                                 for name,visits in rounds.items()},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--full-only", action="store_true", help="Run the frozen v25 full ABI")
    arguments=parser.parse_args()
    for field in ("source","fixture","config","output"):
        setattr(arguments,field,getattr(arguments,field).resolve())
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    run(arguments)
