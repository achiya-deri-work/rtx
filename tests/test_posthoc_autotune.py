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
    wide=preparation_schedule_features((128,128,128),"dx",replace(base,producer_load="coalesced_wide"))
    assert wide["a_source_column_group"] == 1
    assert wide["b_source_column_group"] == 8
    assert wide["b_producer_smem_bytes"] == 2302
    # Wider staging retains the producer's existing conservative CTA cap;
    # idle cooperative CTAs still publish a zero maximum.
    assert wide["b_producer_partial_count"] == coop["b_producer_partial_count"]
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
    assert len(seeds)==8 and seeds[0]==base
    assert (seeds[1].producer_threads,seeds[1].producer_blocks,seeds[1].finalizer_blocks)==(128,352,1376)
    assert (seeds[2].producer_threads,seeds[2].producer_blocks)==(256,704)
    assert seeds[1].producer_order=="reduction_major"
    assert seeds[3].producer_load=="direct"
    assert seeds[4].producer_load=="coalesced_wide"
    assert seeds[5].producer_blocks>seeds[4].producer_blocks
    assert {c.producer_load for c in seeds[5:]}=={"coalesced_wide","coalesced_x4","coalesced_x8"}
    assert all(c.reader_mode==base.reader_mode and c.decoder_blocks==512 and c.gemm==base.gemm for c in seeds)
    assert all(c.producer_order=="row_major" for c in preparation_resource_seeds((2048,768,1536),"dx",48))


def test_schedule_axes_and_policy_contexts():
    def build(phase="dx",layout="mma128"):
        return make_posthoc_preparation_adapter((2048,768,1536),phase,layout,lambda c:None)
    adapter=build()
    assert "coalesced_wide" in adapter.axes["producer_load"]
    assert set(asdict(adapter.initial_config))-{"gemm"} <= set(adapter.axes)
    assert "gemm:gemm_geometry" in adapter.axes
    assert adapter.context.identifier != build("dw").context.identifier
    assert adapter.context.identifier != build(layout="row_major").context.identifier
    for name,values in adapter.axes.items():
        for value in values:
            candidate=adapter.update_fn(adapter.initial_config,name,value)
            if name.startswith("gemm:"):
                assert all(getattr(candidate.gemm,key)==item for key,item in value.items())
            elif name.endswith("_geometry") and value is not None:
                assert asdict(getattr(candidate,name))==value
            else:
                assert getattr(candidate,name)==value
            assert adapter.deserialize(adapter.serialize(candidate))==candidate
            if not name.startswith("gemm:"):
                assert adapter.rejection(candidate) is None


def test_invalid_shapes_and_geometry_rejected():
    for kwargs in ({"producer_threads":32},{"producer_blocks":0},{"finalizer_blocks":True},
                   {"producer_order":"unknown"},{"amax_reduction":"unknown"},{"producer_load":"unknown"},
                   {"producer_signs":"unknown"},{"rng_mode":"unknown"},{"philox_packet_size":2},
                   {"philox_packet_size":True}):
        with pytest.raises(ValueError):
            PosthocPreparationSchedule(**kwargs)
    with pytest.raises(ValueError):
        make_posthoc_preparation_adapter((129,128,128),"dx","mma128",lambda c:None)


def test_fused_rng_is_not_falsely_tuned_by_fixed_uniform_adapter():
    from dataclasses import replace
    adapter=make_posthoc_preparation_adapter((128,256,128),"dw","mma128",lambda c:None)
    assert adapter.axes["rng_mode"]==("explicit",)
    assert adapter.rejection(replace(adapter.initial_config,rng_mode="philox"))[0]=="policy_rejected"
    cached=replace(adapter.initial_config,producer_signs="register")
    assert adapter.config_id(cached)!=adapter.config_id(adapter.initial_config)
    assert adapter.deserialize(adapter.serialize(cached))==cached
    public=make_posthoc_preparation_adapter((128,256,128),"dw","mma128",lambda c:None,randomness="public")
    assert public.context.identifier!=adapter.context.identifier
    assert public.axes["rng_mode"]==("explicit","philox")
    assert public.rejection(replace(public.initial_config,rng_mode="philox")) is None
    from rtx.autotune.posthoc import preparation_schedule_features
    baseline=preparation_schedule_features((128,256,128),"dw",adapter.initial_config)
    fused=preparation_schedule_features((128,256,128),"dw",replace(adapter.initial_config,rng_mode="philox"))
    assert baseline["rng_uniform_bytes"]==(128*256+128*128)//4
    assert baseline["rng_seed_bytes"]==0
    assert fused["rng_uniform_bytes"]==0 and fused["rng_seed_bytes"]==16
    assert fused["rng_sign_bytes"]==baseline["rng_sign_bytes"]==512
    assert adapter.config_id(adapter.initial_config)==adapter.config_id(replace(adapter.initial_config,philox_packet_size=1))
    packet4=replace(public.initial_config,rng_mode="philox")
    assert public.config_id(packet4)!=public.config_id(replace(packet4,philox_packet_size=1))


@pytest.mark.parametrize("mode,rows",[("coalesced_wide",8),("coalesced_x4",16),("coalesced_x8",32)])
def test_wider_tiles_have_distinct_identity_and_resource_features(mode,rows):
    from dataclasses import replace
    from rtx.autotune.posthoc import preparation_schedule_features
    base=PosthocPreparationSchedule(producer_threads=128)
    config=replace(base,producer_load=mode)
    features=preparation_schedule_features((4096,2048,512),"dw",config)
    assert features["a_source_column_group"]==rows
    assert features["a_producer_smem_bytes"]==2*(127*(rows+1)+rows)
    adapter=make_posthoc_preparation_adapter((4096,2048,512),"dw","mma128",lambda c:None)
    assert mode in adapter.axes["producer_load"]
    assert adapter.config_id(config)!=adapter.config_id(base)
    assert adapter.deserialize(adapter.serialize(config))==config
    # Tile extent changes only the logical-transpose path.
    dx=preparation_schedule_features((4096,2048,512),"dx",config)
    assert dx["a_source_column_group"]==1
    assert dx["a_producer_smem_bytes"]==0


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


def test_public_measurement_contract_is_not_mislabeled_as_explicit_benchmark():
    args=((128,256,128),"dw","mma128",lambda _:None)
    fixed=make_posthoc_preparation_adapter(*args)
    unspecified=make_posthoc_preparation_adapter(*args,randomness="public")
    public=make_posthoc_preparation_adapter(*args,randomness="public",measurement_contract="public-test-v1")
    assert fixed.context.tags["measurement"]=="graph8x10_samples15_warm200ms_revision2"
    assert unspecified.context.tags["measurement"]=="public_rng_evaluator_unspecified"
    assert public.context.tags["measurement"]=="public-test-v1"
    assert len({a.context.identifier for a in (fixed,unspecified,public)})==3


def test_philox_packets_and_sign_reuse_match_actual_launch_geometry():
    from dataclasses import replace
    from rtx.autotune.posthoc import preparation_schedule_features
    base=PosthocPreparationSchedule()
    shape=(128,128,128)
    scalar=preparation_schedule_features(shape,"dw",replace(base,rng_mode="philox",philox_packet_size=1))
    packet=preparation_schedule_features(shape,"dw",replace(base,rng_mode="philox",philox_packet_size=4))
    assert scalar["a_finalizer_ctas"]==8 and packet["a_finalizer_ctas"]==2
    assert packet["a_finalizer_values_per_thread"]==4
    assert packet["amax_logical_load_elements"]*4==scalar["amax_logical_load_elements"]
    register=preparation_schedule_features((4096,2048,512),"dw",
        replace(base,producer_blocks=256,producer_signs="register"))
    assert register["a_logical_sign_reads"]==256*4*128
    assert register["schedule_features_revision"]==22


def test_buffered_pipeline_proposals_and_physical_storage():
    from dataclasses import replace
    from rtx.autotune.posthoc import preparation_schedule_features
    adapter=make_posthoc_preparation_adapter((4096,2048,512),"dx","mma128",lambda c:None)
    baseline=adapter.initial_config
    candidate=adapter.update_fn(baseline,"producer_stages",3)
    assert candidate.producer_async_copy and candidate.producer_load=="coalesced"
    assert adapter.rejection(candidate) is None
    features=preparation_schedule_features((4096,2048,512),"dx",candidate)
    assert features["a_producer_smem_bytes"]==0
    assert features["a_producer_stages"]==1
    assert features["b_producer_stages"]==3
    assert features["b_producer_smem_bytes"]==2*(3*128*(4+2)-2)
    sync=adapter.update_fn(candidate,"producer_async_copy",False)
    assert sync.producer_stages==1 and not sync.producer_async_copy
    assert adapter.rejection(replace(candidate,producer_async_copy=False))[0]=="implementation_rejected"
    for changes in ({"producer_stages":0},{"producer_stages":True},{"producer_async_copy":1}):
        with pytest.raises(ValueError): PosthocPreparationSchedule(**changes)
