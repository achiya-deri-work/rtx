import ast
import importlib.util
import json
import hashlib
import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('boundary_harness',Path(__file__).parents[1]/'tools/run_afrl_boundary_qualification.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_coverage_assertion_is_replaced():
    source="""
assert finite, 'numerics'
assert len(SEEN)==(6 if mode=='candidate' else 4), ('missing executed contexts',SEEN)
assert retries==0, 'capacity'
"""
    tree=module.fixed_harness_tree(source)
    assert sum(isinstance(n,ast.Assert) for n in ast.walk(tree))==2
    observed=[]
    values=dict(finite=True,SEEN={1,2,3,4},mode='candidate',retries=0,
                verify_fixed_execution=lambda seen:observed.append(seen))
    exec(compile(tree,'test','exec'),values)
    assert observed==[{1,2,3,4}]
    for name,value in [('finite',False),('retries',1)]:
        with pytest.raises(AssertionError):
            exec(compile(tree,'test','exec'),values|{name:value})


@pytest.mark.parametrize('source',[
    'assert len(SEEN)==6',
    "assert len(SEEN)==(6 if mode=='candidate' else 4)\n"*2,
])
def test_unknown_or_ambiguous_harness_refused(source):
    with pytest.raises(RuntimeError,match='refusing'):
        module.fixed_harness_tree(source)


def test_profile_callback_preserves_owner_assertions_and_order():
    source="""
events.append('synchronized')
state['update'] += 1
assert finite, 'numerics'
assert retries == 0, 'capacity'
"""
    tree=module.profiling_harness_tree(ast.parse(source))
    events=[]
    values=dict(events=events,state={'update':2},finite=True,retries=0,
                profile_update=lambda n:events.append(n))
    exec(compile(tree,'test','exec'),values)
    assert events==['synchronized',3]
    assert sum(isinstance(n,ast.Assert) for n in ast.walk(tree))==2
    for name,value in [('finite',False),('retries',1)]:
        with pytest.raises(AssertionError):
            exec(compile(tree,'test','exec'),values|{name:value})


@pytest.mark.parametrize('source',['pass',"state['update'] += 1\n"*2])
def test_profile_unknown_boundary_refused(source):
    with pytest.raises(RuntimeError,match='refusing'):
        module.profiling_harness_tree(ast.parse(source))


def test_profile_lifecycle_no_retained_shapes_and_runtime_only_labels(tmp_path):
    calls=[]
    class Profile:
        def start(self): calls.append('start')
        def stop(self): calls.append('stop')
        def export_chrome_trace(self,path): calls.append(Path(path).name)
        def key_averages(self):
            return [SimpleNamespace(key='kernel',count=2,self_cpu_time_total=1,
                cpu_time_total=2,self_device_time_total=3,device_time_total=4)]
    def profile(**kwargs):
        assert kwargs==dict(activities=['cpu','cuda'],record_shapes=False,
                           profile_memory=False,with_stack=False)
        return Profile()
    @contextmanager
    def record_function(name):
        calls.append(name)
        yield
    torch=SimpleNamespace(profiler=SimpleNamespace(profile=profile,
        ProfilerActivity=SimpleNamespace(CPU='cpu',CUDA='cuda'),record_function=record_function))
    capture=module.WarmUpdateProfiler(torch,tmp_path)
    x=SimpleNamespace(shape=(20480,256),numel=lambda:20480*256)
    gu=SimpleNamespace(shape=(1536,256))
    output=object()
    def forward(x,gu): return output
    wrapped=capture.wrap(forward,'forward')
    assert wrapped(x,gu) is output and calls==[]
    capture.update(1)
    capture.update(2)
    assert wrapped(x,gu) is output
    capture.update(3)
    assert capture.completed and not capture.active
    capture.close()
    assert calls==['start','rtx_mlp/tokenizer/forward/m20480','stop','steady-update.trace.json.gz']
    result=json.loads((tmp_path/'steady-update.profile.json').read_text())
    assert result['diagnostic_only'] and result['aggregate_events_omitted']
    assert capture.profiler is None
    assert wrapped(x,gu) is output
    with pytest.raises(RuntimeError,match='twice'):
        capture.update(2)


@pytest.fixture
def backbone_bundle(tmp_path,monkeypatch):
    import rtx
    from rtx.autotune import replacement_benchmark
    root=tmp_path/'rtx'; root.mkdir()
    (root/'rtx').mkdir()
    (root/'rtx/core.py').write_text('current RTX')
    monkeypatch.setattr(module,'RTX',root)
    sources={'core.py':hashlib.sha256((root/'rtx/core.py').read_bytes()).hexdigest()}
    monkeypatch.setattr(replacement_benchmark,'source_manifest',lambda:sources)
    afrl=tmp_path/'afrl'; afrl.mkdir()
    (afrl/'train.py').write_text('unchanged AFRL')
    owner=afrl/'reports/analysis/owner'; owner.mkdir(parents=True)
    config=rtx.MXFP8MLPConfig().to_dict()
    rows=[]; checks={}; candidates={}
    for m in (9216,11296):
        for phase in ('train','forward'):
            name=f'backbone-{m}-{phase}'
            request=root/'autotune_reports'/name
            rows.append(dict(name=name,shape=[m,256,1536,256],input_shape=[m,256],
                             phase=phase,config=config,request=str(request)))
            checks[str(request)]=dict(status='preflight_passed',current_sources_match=True,
                requires_requalification=False,selected_plan={'config':config})
            candidates[str(request)]=SimpleNamespace(shape=(m,256,1536,256),phase=phase,
                activation='silu',inspect=lambda path:checks[str(path)])
    monkeypatch.setattr(rtx.MLPCandidate,'load',lambda path:candidates[str(Path(path).parent)])
    manifest=dict(afrl={'train.py':hashlib.sha256((afrl/'train.py').read_bytes()).hexdigest()},
                  current={'rtx/core.py':'obsolete RTX hash'})
    for name,data in (('manifest.json',manifest),('request-index.json',rows),
                      ('application-gates.json',{'passed':True})):
        (owner/name).write_text(json.dumps(data))
    (owner/'application.py').write_text('assert True')
    bundle=dict(schema=1,scope='backbone_only_bf16_tokenizer',rtx_sources=sources,
        requests=rows,owner_files={name:hashlib.sha256((owner/name).read_bytes()).hexdigest()
            for name in ('manifest.json','request-index.json','application.py','application-gates.json')})
    return owner,bundle,checks


def test_backbone_bundle_fresh_qualification_not_historical_relabel(backbone_bundle,tmp_path):
    owner,bundle,_=backbone_bundle
    assert len(module.qualified_backbone_index(owner,bundle))==4
    path=module.RTX/'autotune_reports/plans.json'
    written=module.write_backbone_plan_bundle(owner,{r['name']:r['request'] for r in bundle['requests']},path)
    assert not written['application_qualified']
    manifest,afrl=module.verify(owner,path)
    assert manifest['current']=={'rtx/'+name:digest for name,digest in bundle['rtx_sources'].items()}
    with pytest.raises(RuntimeError,match='source differs'):
        module.verify(owner)
    (afrl/'train.py').write_text('changed AFRL')
    with pytest.raises(RuntimeError,match='afrl source differs'):
        module.verify(owner,path)
    with pytest.raises(FileExistsError):
        module.write_backbone_plan_bundle(owner,{r['name']:r['request'] for r in bundle['requests']},path)


@pytest.mark.parametrize('mutation', ['sources','missing','duplicate','shape','phase','config','outside','harness'])
def test_backbone_bundle_rejects_drift(backbone_bundle,mutation):
    owner,original,_=backbone_bundle
    bundle=copy.deepcopy(original)
    if mutation=='sources': bundle['rtx_sources']={}
    elif mutation=='missing': bundle['requests'].pop()
    elif mutation=='duplicate': bundle['requests'][1]=bundle['requests'][0]
    elif mutation=='shape': bundle['requests'][0]['shape'][0]+=1
    elif mutation=='phase': bundle['requests'][0]['phase']='forward'
    elif mutation=='config': bundle['requests'][0]['config']['backward_overlap']=True
    elif mutation=='outside': bundle['requests'][0]['request']=str(owner)
    elif mutation=='harness': (owner/'application.py').write_text('assert False')
    with pytest.raises((ValueError,RuntimeError)):
        module.qualified_backbone_index(owner,bundle)


@pytest.mark.parametrize('field,value',[('status','failed'),('current_sources_match',False),
                                      ('requires_requalification',True)])
def test_backbone_bundle_requires_current_completed_requests(backbone_bundle,field,value):
    owner,bundle,checks=backbone_bundle
    checks[bundle['requests'][0]['request']][field]=value
    with pytest.raises(RuntimeError,match='not currently qualified'):
        module.qualified_backbone_index(owner,bundle)
