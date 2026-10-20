"""Warm selection does not reconstruct immutable JSON metadata."""
import json
from unittest.mock import patch

import torch

import rtx._mlp_autotune as selection


def test_fixed_metadata_parsed_once_and_returns_no_mutable_state():
    selection._spec_mode_and_fixed_key.cache_clear()
    spec=json.dumps(dict(mode='fixed',initial={'nested':{'value':3}}))
    with patch.object(selection.json,'loads',wraps=json.loads) as loads:
        a=selection._spec_mode_and_fixed_key(spec)
        b=selection._spec_mode_and_fixed_key(spec)
        assert loads.call_count==1
    assert a==b==('fixed',json.dumps({'nested':{'value':3}},sort_keys=True))


def test_warm_dynamic_selection_does_not_parse_spec(monkeypatch):
    selection.SELECTIONS.clear()
    spec=json.dumps(dict(mode='balanced',policy={}))
    selection._spec_mode_and_fixed_key(spec)
    x=torch.empty(2,3)
    gu=torch.empty(8,3)
    down=torch.empty(3,4)
    identity=(tuple(x.shape),tuple(gu.shape),tuple(down.shape),str(x.device),
              'silu',spec,True,False)
    selection.SELECTIONS[identity]='chosen'
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda,'device',lambda *a:nullcontext())
    with patch.object(selection.json,'loads',side_effect=AssertionError('warm JSON parse')):
        assert selection._select(x,gu,down,'silu',spec,True)=='chosen'
    selection.SELECTIONS.clear()


def test_spec_metadata_bound_and_cleanup():
    from rtx.mlp import _clear_runtime_caches
    selection._spec_mode_and_fixed_key.cache_clear()
    for i in range(100):
        selection._spec_mode_and_fixed_key(json.dumps(dict(mode='fixed',initial={'i':i})))
    assert selection._spec_mode_and_fixed_key.cache_info().currsize==64
    _clear_runtime_caches(preserve_compiled=True)
    assert selection._spec_mode_and_fixed_key.cache_info().currsize==64
    _clear_runtime_caches()
    assert selection._spec_mode_and_fixed_key.cache_info().currsize==0
