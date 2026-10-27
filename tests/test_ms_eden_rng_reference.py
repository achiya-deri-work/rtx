import math
import statistics

import pytest

from benchmarks.reference_ms_eden_rng import MASK, philox4x32, uniform_at, uniforms


@pytest.mark.parametrize("counter,key,expected", [
    ((0,0,0,0),(0,0),(0x6627E8D5,0xE169C58D,0xBC57AC4C,0x9B00DBD8)),
    ((MASK,)*4,(MASK,)*2,(0x408F276D,0x41C83B0E,0xA20BC7C6,0x6D5451FD)),
    ((0x243F6A88,0x85A308D3,0x13198A2E,0x03707344),(0xA4093822,0x299F31D0),
     (0xD16CFE09,0x94FDCCEB,0x5001E420,0x24126EA1)),
])
def test_random123_known_answers(counter,key,expected):
    assert philox4x32(counter,key)==expected


@pytest.mark.parametrize("seed",[(0,0),(MASK,MASK),(941,1234567)])
@pytest.mark.parametrize("stream",[0,1])
def test_mapping_tails_range_and_exact_float32_resolution(seed,stream):
    values=uniforms(seed,131,stream)
    assert values==[uniform_at(seed,i,stream) for i in range(131)]
    assert all(0<=v<1 and v*(1<<24)==int(v*(1<<24)) for v in values)
    assert uniforms(seed,0,stream)==[]
    assert values==uniforms(seed,131,stream)
    assert values!=uniforms(seed,131,1-stream)


def test_substreams_have_bounded_distribution_smoke():
    # Deterministic sanity checks, not a proof of randomness or independence.
    a,b=(uniforms((941,826),16384,s) for s in (0,1))
    for values in (a,b):
        assert abs(statistics.mean(values)-.5)<.012
        assert abs(statistics.pvariance(values)-1/12)<.004
        for threshold in (.125,.25,.5,.75,.875):
            observed=sum(v<threshold for v in values)/len(values)
            assert abs(observed-threshold)<.015
    ma,mb=statistics.mean(a),statistics.mean(b)
    covariance=statistics.mean((x-ma)*(y-mb) for x,y in zip(a,b))
    assert abs(covariance/math.sqrt(statistics.pvariance(a)*statistics.pvariance(b)))<.03


@pytest.mark.parametrize("seed,index,stream", [((-1,0),0,0),((0,1<<32),0,0),
    ((0,),0,0),((0,0),-1,0),((0,0),1<<34,0),((0,0),0,2),((0,0),True,0)])
def test_invalid_reference_contract(seed,index,stream):
    with pytest.raises(ValueError): uniform_at(seed,index,stream)
