"""MoE launch-code LRU cleanup and observability, without a CUDA device."""
from collections import OrderedDict
import gc
import weakref

import pytest

from rtx import _moe_execution as execution
from rtx import _moe_selection as selection
from rtx.autotune import moe as tuning
from rtx.runtime import BoundedCache,retain_runtime_resources,runtime_cache_stats


@pytest.fixture
def isolated_caches(monkeypatch):
    monkeypatch.setattr(execution,'_COMPILED',BoundedCache(128))
    monkeypatch.setattr(tuning,'_choices',BoundedCache(256))
    monkeypatch.setattr(selection,'_CELLS',{})
    monkeypatch.setattr(selection,'_RECENCY',OrderedDict())
    monkeypatch.setattr(selection,'_ATTRS',{})
    monkeypatch.setattr(selection,'_SOURCES',selection.SelectionSources())


def test_moe_launch_cache_clear_respects_preserve_compiled(isolated_caches):
    launch=object()
    execution._COMPILED[('plan',0)]=launch
    tuning._choices['shape']='selected'
    selection.bind_selected((32,128),'cuda:0','policy',True,'selected')
    assert tuning._clear_runtime_caches(preserve_compiled=True)=={}
    assert execution._COMPILED[('plan',0)] is launch
    assert tuning._choices['shape']=='selected'
    assert selection.bound_config((32,128),'cuda:0','policy',True)=='selected'
    assert tuning._clear_runtime_caches()=={
        'moe_selections':1,'moe_selection_cells':1,'moe_compiled_plans':1}
    assert len(execution._COMPILED)==len(tuning._choices)==len(selection._CELLS)==0


def test_moe_launch_cache_is_visible_without_claiming_byte_accounting(isolated_caches):
    before=runtime_cache_stats()['_moe_execution._COMPILED']
    assert before['entries']==0 and before['max_entries']==128
    execution._COMPILED[('plan',0)]=object()
    assert execution._COMPILED.get(('plan',0)) is not None
    row=runtime_cache_stats()['_moe_execution._COMPILED']
    assert row['entries']==1 and row['insertions']==1 and row['hits']==1
    # Code resources are not a measured allocation inventory. A zero counter
    # is not a blanket claim about graph pins, allocator pools or GPU memory.
    assert row['bytes_accounted'] is False
    tuning._clear_runtime_caches()
    assert runtime_cache_stats()['_moe_execution._COMPILED']['entries']==0


def test_explicit_refinement_survives_hot_cell_eviction(isolated_caches):
    import torch
    from dataclasses import replace
    from rtx.moe import DEFAULT_MOE_CONFIG
    operands=(torch.empty(2,4),torch.empty(3,8,4),torch.empty(3,4,4),
              torch.empty(3,2),torch.empty(3,2),torch.empty(2,2))
    initial=DEFAULT_MOE_CONFIG
    refined=replace(initial,movement_threads=256)
    tuning.remember_config(operands,'policy',True,initial)
    assert tuning.resolve_config(operands,'policy',True) is initial
    tuning.remember_config(operands,'policy',True,refined)
    selection.clear_selection_cells()
    # Cold selection after metadata eviction must not resurrect the old
    # process-local config instead of the explicitly refined winner.
    assert tuning.resolve_config(operands,'policy',True) is refined


def test_clearing_moe_code_lru_keeps_explicit_graph_resource_ownership(isolated_caches):
    class Resource:pass
    resource=Resource();reference=weakref.ref(resource)
    def launch():return 'still callable'
    launch.__rtx_resources__=(resource,)
    with retain_runtime_resources() as owners:
        execution._COMPILED[('plan',0)]=launch
    del resource,launch
    tuning._clear_runtime_caches()
    gc.collect()
    assert reference() is not None
    assert len(owners)==1
    assert next(iter(owners.values()))()=='still callable'
    owners.clear();gc.collect()
    assert reference() is None
