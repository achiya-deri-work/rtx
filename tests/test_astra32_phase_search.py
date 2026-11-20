"""Research proposal ownership must not change anchors or numerical policy."""
from dataclasses import dataclass
import random
from types import SimpleNamespace

from benchmarks.astra32_phase_search import PhaseBalancedSearch,phase_group


@dataclass
class Adapter:
    axes:dict
    context:object


class Ordered:
    name='coordinate_local'
    def propose(self,adapter,history,rng,limit):return list(adapter.axes)[:limit]
    def observe(self,row):return row


def test_opening_unchanged_then_unexplored_phase_preferred():
    a=Adapter(dict([('gate_up.tile_m',(64,128)),('gate_up.stages',(1,2)),
                   ('down_gemm.stages',(1,2)),('gate_backward.dx.stages',(1,2))]),
        SimpleNamespace(family='mxfp8_recompute_mlp'))
    strategy=PhaseBalancedSearch(Ordered())
    h=SimpleNamespace(current=[SimpleNamespace(coordinate='gate_up.tile_m')]*7)
    assert strategy.propose(a,h,random.Random(0),1)==['gate_up.tile_m']
    h.current.append(SimpleNamespace(coordinate='gate_up.stages'))
    assert strategy.propose(a,h,random.Random(0),1)==['down_gemm.stages']
    # Axis order of the original adapter is never mutated.
    assert next(iter(a.axes))=='gate_up.tile_m'
    h.current.append(SimpleNamespace(coordinate='down_gemm.stages'))
    assert strategy.propose(a,h,random.Random(0),1)==['gate_backward.dx.stages']


def test_backward_legs_and_preparation_are_distinct():
    for leg in ('gate_backward.dx','gate_backward.dw','down_backward.dx','down_backward.dw'):
        for suffix in ('.backend','_backend','_layout_transport','_a_vector'):
            assert phase_group('mxfp8_recompute_mlp',leg+suffix)==leg
    assert phase_group('mxfp8_balanced_experts','prepared.gate_dw_splits')=='gate_dw'
    assert phase_group('mxfp8_balanced_experts','prepared.gate_dx.geometry')=='gate_dx'
    assert phase_group('mxfp8_balanced_experts','prepared.quant.num_warps')=='preparation'
