from types import SimpleNamespace

import pytest

from rtx.autotune.deployment_qualification import qualify_pair


def outcome(value=1., status='ok'):
    return SimpleNamespace(median_ms=value, status=status, error=None)


@pytest.mark.parametrize('candidate,expected', [(0.8,'new'), (1.,'old'), (1.2,'old'), (.995,'old')])
def test_alternating_qualification(candidate, expected):
    order=[]
    def evaluate(config):
        order.append(config)
        return outcome(1. if config=='old' else candidate)
    winner, decision=qualify_pair('old','new',evaluate)
    assert winner==expected
    assert order==['old','new','new','old']*3
    assert decision['promoted']==(expected=='new')


def test_challenger_failure_preserves_incumbent():
    winner, decision=qualify_pair('old','new',lambda c: outcome(.1,'compile_error') if c=='new' else outcome())
    assert winner=='old'
    assert len(decision['failures'])==6


def test_unverified_incumbent_cannot_be_returned():
    with pytest.raises(RuntimeError,match='incumbent'):
        qualify_pair('old','new',lambda c: outcome(float('nan')) if c=='old' else outcome())


def test_order_sensitive_gain_is_not_promoted():
    values=iter([.8,1.2,.8,1.2,.8,1.2])
    winner,_=qualify_pair('old','new',lambda c: outcome(next(values)) if c=='new' else outcome())
    assert winner=='old'


def test_paired_gain_with_worse_aggregate_median_is_not_promoted():
    # Five favorable pairs cannot hide contradictory aggregate ordering.
    values={'old':iter([.4725,.5746,.5757,.5012,.4879,.5839]),
            'new':iter([.5633,.5650,.5644,.4663,.4800,.5463])}
    winner,decision=qualify_pair('old','new',lambda c:outcome(next(values[c])))
    assert sorted(decision['paired_speedups'])[1]>.01
    assert decision['aggregate_speedup']<0
    assert winner=='old' and not decision['promoted']
    assert decision['decision_revision']==2


@pytest.mark.parametrize('rounds',[1,2,3,5,6.0,True])
def test_bad_rounds(rounds):
    with pytest.raises(ValueError):
        qualify_pair('old','new',lambda c:outcome(),rounds=rounds)


@pytest.mark.parametrize('margin',[-1,1,float('nan'),float('inf')])
def test_bad_margin(margin):
    with pytest.raises(ValueError):
        qualify_pair('old','new',lambda c:outcome(),margin=margin)


def test_fixed_candidate_bypasses_cache_and_cuda():
    import json
    from rtx._mlp_autotune import _select
    from rtx.mlp import DEFAULT_MLP_CONFIG
    tensor=SimpleNamespace(shape=(32,256),device='cpu')
    spec=json.dumps({'mode':'fixed','initial':DEFAULT_MLP_CONFIG.to_dict()})
    assert json.loads(_select(tensor,tensor,tensor,'silu',spec,True))==DEFAULT_MLP_CONFIG.to_dict()
