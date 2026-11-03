import json

from benchmarks.compare_ms_eden_amortization import options,schedule_pair


def test_baseline_is_qualified_256_cta_not_old_seed():
    pair=schedule_pair()
    for phase,encoded in pair.items():
        schedule=json.loads(encoded)
        assert schedule['producer_threads']==128
        assert schedule['producer_blocks']==256
        assert schedule['finalizer_blocks']==224
        assert schedule['producer_load']=='coalesced_wide'
        assert schedule['producer_order']==('row_major' if phase=='dx' else 'reduction_major')


def test_per_phase_overrides_and_compiled_public_options():
    pair=schedule_pair(dict(producer_blocks=512,producer_load='coalesced_x8'))
    args=options((4096,2048,512),pair)
    assert (args.m,args.n,args.k)==(4096,2048,512)
    assert args.policy=='fp4_fp4'
    assert args.dx_schedule_json==pair['dx']
    assert args.dw_schedule_json==pair['dw']
    assert all(json.loads(encoded)['producer_blocks']==512 for encoded in pair.values())


def test_single_phase_cli_preserves_schedule_and_quality(monkeypatch,tmp_path):
    from benchmarks import compare_ms_eden_amortization as benchmark
    seen=[]
    result=dict(timings=dict(median_ms=.1,samples_ms=[.1]*9),
                quality=dict(sampling=dict(collection=dict(stationary=True))))
    def fake(shape,phase,schedule,**kwargs):
        seen.append((shape,phase,schedule,kwargs))
        return result
    monkeypatch.setattr(benchmark,'evaluate_public_phase',fake)
    output=tmp_path/'result.json'
    monkeypatch.setattr('sys.argv',['benchmark','--shape','128,256,128','--evaluate-phase','dw',
        '--schedule-json','{"rng_mode":"philox","producer_blocks":256}',
        '--replays','12','--output',str(output)])
    benchmark.main()
    assert seen==[((128,256,128),'dw',{'rng_mode':'philox','producer_blocks':256},{'replays':12})]
    assert json.loads(output.read_text())==result
    default_output=tmp_path/'default.json'
    monkeypatch.setattr('sys.argv',['benchmark','--shape','128,256,128','--evaluate-phase','dx',
        '--output',str(default_output)])
    benchmark.main()
    assert seen[-1]==((128,256,128),'dx',{},dict(replays=16))
    assert benchmark.PUBLIC_PHASE_MEASUREMENT_CONTRACT.startswith('public_rng_graph8_')
    assert benchmark.PUBLIC_PHASE_MEASUREMENT_CONTRACT.endswith('_v2')
