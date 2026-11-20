from benchmarks.benchmark_moe_geometry_frontier import candidates,pipeline_candidates,schedule_candidates
from rtx.autotune.moe import make_moe_adapter


def test_frontier_uses_legal_distinct_adapter_templates_without_touching_registers():
    adapter=make_moe_adapter((2000,384,32,256,4,128),None)
    base=adapter.initial_config.prepared
    pool=candidates(adapter)
    assert pool and len({adapter.config_id(c) for c in pool})==len(pool)
    assert any(c.prepared.down!=base.down for c in pool)
    assert any(c.prepared.gate_dx!=base.gate_dx for c in pool)
    for c in pool:
        assert adapter.rejection(c) is None
        assert c.prepared.quant==base.quant
        for stage in ('down','gate_dx'):
            a=getattr(base,stage);b=getattr(c.prepared,stage)
            # Cooperative transport has its existing fixed role convention;
            # TMA register settings are never manually retuned by this study.
            if c.prepared.transports[1 if stage=='down' else 5]=='tma':
                for field in ('producer_registers','consumer_registers','maxrregcount'):
                    assert getattr(a,field)==getattr(b,field)


def test_pipeline_study_preserves_geometry_and_all_numerical_fields():
    adapter=make_moe_adapter((2000,384,32,256,4,128),None)
    base=adapter.initial_config.prepared
    pool=pipeline_candidates(adapter,['down'])
    assert pool
    for c in pool:
        assert c.prepared.gate_dx==base.gate_dx and c.prepared.quant==base.quant
        a=base.down;b=c.prepared.down
        for field in ('tile_m','tile_n','tile_k','atom_layout_m','atom_layout_n',
                      'producer_registers','consumer_registers','maxrregcount','scale_layout'):
            assert getattr(a,field)==getattr(b,field)


def test_schedule_panel_couples_tall_tiles_safe_buffers_and_transport():
    adapter=make_moe_adapter((16384,384,32,2048,4,320),None)
    pool=schedule_candidates(adapter,['down'])
    assert pool and len({adapter.config_id(c) for c in pool})==len(pool)
    assert {c.prepared.transports[1] for c in pool}=={'tma','cooperative'}
    assert {c.prepared.down.tile_m for c in pool}=={64,128,256}
    assert any(c.prepared.down.stages>=3 for c in pool)
    assert any(c.prepared.down.epilogue=='hybrid' for c in pool)
    for c in pool:
        assert adapter.rejection(c) is None
        assert c.prepared.quant==adapter.initial_config.prepared.quant
        assert c.prepared.gate_dx==adapter.initial_config.prepared.gate_dx
