from dataclasses import asdict
import pytest
from rtx.autotune.posthoc import PosthocPreparationSchedule,make_posthoc_preparation_adapter


def test_structural_features_match_launch_caps_and_staging():
    from dataclasses import replace
    from rtx.autotune.posthoc import preparation_schedule_features
    base=PosthocPreparationSchedule()
    f=preparation_schedule_features((128,128,128),"dx",base,sm_count=170,l2_bytes=1024)
    assert f["a_producer_ctas"] == 32
    assert f["a_producer_partial_count"] == 128
    assert f["a_finalizer_ctas"] == 8
    assert f["amax_logical_load_elements"] == 8192
    assert f["preparation_scratch_bytes"] == 5120
    assert f["packed_payload_bytes"] == 18432
    assert f["payload_l2_ratio"] == 18
    assert f["a_producer_waves"] == 32/170
    assert f["preparation_launches"] == 5
    staged=preparation_schedule_features((128,128,128),"dx",replace(base,amax_reduction="staged"))
    assert staged["amax_logical_load_elements"] == 2304
    assert staged["preparation_scratch_bytes"] == 5128
    assert staged["preparation_launches"] == 7
    assert staged["sm_count_known"] == staged["l2_known"] == 0
    assert staged["payload_l2_ratio"] == staged["a_producer_waves"] == 0
    coop=preparation_schedule_features((128,128,128),"dx",replace(base,producer_load="coalesced"))
    assert coop["a_producer_smem_bytes"] == 0
    assert coop["b_producer_smem_bytes"] == 1278
    assert coop["a_source_column_group"] == 1
    assert coop["b_source_column_group"] == 4
    packed=preparation_schedule_features((128,128,128),"dw",replace(base,reader_mode="staged"),packed_reader=True)
    assert packed["decoded_workspace_bytes"] == 32768
    assert packed["preparation_scratch_bytes"] == 37888
    assert packed["preparation_launches"] == 6
    adapter=make_posthoc_preparation_adapter((128,128,128),"dx","mma128",lambda c:None,
        device={"multiprocessor_count":170,"l2_cache_size":1024})
    assert adapter.features(base)["derived.amax_logical_load_elements"] == 8192
    assert adapter.features(base)["derived.payload_l2_ratio"] == 18


def test_device_grid_proposals_are_bounded_and_policy_preserving():
    from rtx.autotune.posthoc import preparation_grid_candidates
    from dataclasses import replace
    base = PosthocPreparationSchedule(reader_mode="staged", decoder_blocks=512)
    proposals = preparation_grid_candidates((32768,768,1536), "dw", 170, base)
    assert proposals[0] == base
    assert len(proposals) == 25
    assert replace(base, producer_blocks=1376) in proposals
    assert replace(base, finalizer_blocks=1376) in proposals
    for c in proposals:
        assert replace(c, producer_blocks=128, finalizer_blocks=128) == base
    tiny = preparation_grid_candidates((128,128,128), "dw", 170)
    assert len(tiny) == 4
    assert max(c.producer_blocks for c in tiny[1:]) <= 128
    assert any(c.producer_blocks == 32 and c.finalizer_blocks == 8 for c in tiny)
    assert proposals != preparation_grid_candidates((32768,768,1536), "dw", 48, base)
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            preparation_grid_candidates((128,128,128), "dw", bad)


def test_resource_seed_prior_keeps_baseline_and_coupled_alternatives():
    from dataclasses import replace
    from rtx.autotune.posthoc import preparation_resource_seeds
    base=PosthocPreparationSchedule(reader_mode="staged",decoder_blocks=512)
    seeds=preparation_resource_seeds((32768,768,1536),"dw",170,base)
    assert len(seeds)==4 and seeds[0]==base
    assert (seeds[1].producer_threads,seeds[1].producer_blocks,seeds[1].finalizer_blocks)==(128,352,1376)
    assert (seeds[2].producer_threads,seeds[2].producer_blocks)==(256,704)
    assert seeds[1].producer_order=="reduction_major"
    assert seeds[3].producer_load=="direct"
    assert all(c.reader_mode==base.reader_mode and c.decoder_blocks==512 and c.gemm==base.gemm for c in seeds)
    assert all(c.producer_order=="row_major" for c in preparation_resource_seeds((2048,768,1536),"dx",48))


def test_schedule_axes_and_policy_contexts():
    def build(phase="dx",layout="mma128"):
        return make_posthoc_preparation_adapter((2048,768,1536),phase,layout,lambda c:None)
    adapter=build()
    assert set(asdict(adapter.initial_config))-{"gemm"} <= set(adapter.axes)
    assert "gemm:gemm_geometry" in adapter.axes
    assert adapter.context.identifier != build("dw").context.identifier
    assert adapter.context.identifier != build(layout="row_major").context.identifier
    for name,values in adapter.axes.items():
        for value in values:
            candidate=adapter.update_fn(adapter.initial_config,name,value)
            if name.startswith("gemm:"):
                assert all(getattr(candidate.gemm,key)==item for key,item in value.items())
            else:
                assert getattr(candidate,name)==value
            assert adapter.deserialize(adapter.serialize(candidate))==candidate
            if not name.startswith("gemm:"):
                assert adapter.rejection(candidate) is None


def test_invalid_shapes_and_geometry_rejected():
    for kwargs in ({"producer_threads":32},{"producer_blocks":0},{"finalizer_blocks":True},
                   {"producer_order":"unknown"},{"amax_reduction":"unknown"},{"producer_load":"unknown"}):
        with pytest.raises(ValueError):
            PosthocPreparationSchedule(**kwargs)
    with pytest.raises(ValueError):
        make_posthoc_preparation_adapter((129,128,128),"dx","mma128",lambda c:None)


def test_packed_source_policy_has_separate_cache_identity():
    args=((128,256,128),"dw","mma128",lambda c:None)
    original=make_posthoc_preparation_adapter(*args)
    decoded=make_posthoc_preparation_adapter(*args,source_policy="forward_current_roundtrip_bf16")
    packed=make_posthoc_preparation_adapter(*args,source_policy="forward_current_roundtrip_bf16",packed_reader=True)
    assert len({a.context.identifier for a in (original,decoded,packed)}) == 3
    with pytest.raises(ValueError,match="explicit"):
        make_posthoc_preparation_adapter(*args,packed_reader=True)
    from dataclasses import replace
    base=packed.initial_config
    assert packed.config_id(base) == packed.config_id(replace(base,decoder_blocks=1024))
    staged=replace(base,reader_mode="staged")
    assert packed.config_id(staged) != packed.config_id(replace(staged,decoder_blocks=1024))
    assert packed.rejection(staged) is None
    assert original.rejection(staged)[0] == "policy_rejected"
