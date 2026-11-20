import json

import pytest

from benchmarks.consolidate_moe_profiles import consolidate,grouped


def test_grouped_keeps_waves_layouts_and_main_matrix_separate():
    def row(wave,layout,b,t,value):
        return dict(wave=wave,layout=layout,samples=b,total_tokens=t,
            **{key:value for key in ('speedup_checkpoint','speedup_compact_checkpoint',
                                    'speedup_saved','speedup_frozen_experts','speedup_shared')})
    rows=[row('legacy','packed',16,16384,1.1),row('legacy','packed',16,16384,1.3),
          row('legacy','packed',8,8192,4.),row('fanout','flat',16,16384,.9)]
    values={(r['wave'],r['layout'],r['scope']):r for r in grouped(rows)}
    main=values['legacy','packed','main_b16_l1024']
    assert main['count']==2 and main['speedup_compact_checkpoint']['median']==pytest.approx(1.2)
    assert values['legacy','packed','all']['count']==3
    assert values['fanout','flat','main_b16_l1024']['speedup_saved']['median']==.9
    assert ('fanout','packed','all') not in values


def test_consolidation_rejects_incomplete_or_mixed_runtime_without_rewriting_inputs(tmp_path):
    def source(name,runtime,incomplete):
        folder=tmp_path/name;folder.mkdir()
        identity={'runtime':runtime,'drivers':{}}
        (folder/'manifest.json').write_text(json.dumps(dict(identity=identity,contexts=[],settings={})))
        (folder/'summary.json').write_text(json.dumps(dict(identity=identity,rows=[],incomplete=incomplete)))
        return folder
    a=source('a','runtime-a',[]);b=source('b','runtime-b',[]);bad=source('bad','runtime-a',['missing'])
    before={p:p.read_bytes() for d in (a,b,bad) for p in d.iterdir()}
    with pytest.raises(ValueError,match='Incomplete'):consolidate([bad],tmp_path/'output')
    with pytest.raises(ValueError,match='Different production'):consolidate([a,b],tmp_path/'output')
    assert all(p.read_bytes()==value for p,value in before.items())
    assert not (tmp_path/'output').exists()
