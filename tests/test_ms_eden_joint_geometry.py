from collections import Counter
from dataclasses import asdict,replace
from itertools import combinations
import pytest
from benchmarks.experiment_ms_eden_joint_geometry import balanced_indices,portfolio


def test_all_coordinate_pairs_cover_all_level_pairs_once():
    rows=balanced_indices()
    assert len(rows)==len(set(rows))==16
    for i,j in combinations(range(4),2):
        counts=Counter((row[i],row[j]) for row in rows)
        assert len(counts)==16 and set(counts.values())=={1}


def test_persistence_portfolio_changes_only_gemm_grid_waves():
    from benchmarks.experiment_ms_eden_joint_geometry import tma_persistence_portfolio
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
    initial=MSEdenPipelineSchedule(backward_pipeline='operand_paired')
    choices=tma_persistence_portfolio(initial)
    assert len(choices)==9
    assert {(c.dx.gemm.persistent_waves,c.dw.gemm.persistent_waves) for c in choices}=={
        (a,b) for a in (0,1,2) for b in (0,1,2)}
    for c in choices:
        for phase in ('dx','dw'):
            actual,base=getattr(c,phase),getattr(initial,phase)
            assert replace(actual,gemm=replace(actual.gemm,persistent_waves=base.gemm.persistent_waves))==base


def test_portfolio_changes_only_producer_blocks():
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
    initial=MSEdenPipelineSchedule(backward_pipeline='operand_paired')
    choices=portfolio(initial)
    assert choices[0]==initial and len(choices)==17
    for candidate in choices[1:]:
        assert candidate.backward_pipeline==initial.backward_pipeline
        for phase in ('dx','dw'):
            base=getattr(initial,phase);actual=getattr(candidate,phase)
            assert replace(actual,a_geometry=base.a_geometry,b_geometry=base.b_geometry)==base
            for operand in ('a','b'):
                b=asdict(base.operand_schedule(operand));c=asdict(actual.operand_schedule(operand))
                assert c.pop('producer_blocks') in (64,128,256,512)
                b.pop('producer_blocks');assert b==c


def test_compact_gemm_portfolio_covers_both_legs_without_preparation_changes():
    from benchmarks.experiment_ms_eden_joint_geometry import compact_gemm_portfolio
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
    initial = MSEdenPipelineSchedule(backward_pipeline='operand_paired')
    choices = compact_gemm_portfolio(initial)
    assert len(choices) == len(set(choices)) == 12
    assert choices[0] == initial
    assert {(c.dx.gemm.atom_layout_m, c.dw.gemm.atom_layout_m) for c in choices} == {
        (a, b) for a in (2, 4, 8) for b in (2, 4, 8)}
    for c in choices:
        assert c.backward_pipeline == initial.backward_pipeline
        for phase in ('dx', 'dw'):
            actual, base = getattr(c, phase), getattr(initial, phase)
            assert replace(actual, gemm=base.gemm) == base
            assert replace(actual.gemm, atom_layout_m=base.gemm.atom_layout_m,
                           stages=base.gemm.stages) == base.gemm


@pytest.mark.parametrize('space', ['producer_grids', 'compact_gemms', 'compact64_grids','tma_finalizers','tma_cta_finalizers','tma_producer_grids','tma_gemms','tma_loops','h16_grouped_grids','tma_persistence','tma_paired_finalizers'])
def test_real_tuner_accepts_already_validated_worker_outcome(monkeypatch,tmp_path,space):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from benchmarks import experiment_ms_eden_joint_geometry as study
    from benchmarks import tune_ms_eden_pipeline as tuner
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
    from test_ms_eden_pipeline_autotune import evidence
    initial=MSEdenPipelineSchedule(backward_pipeline='operand_paired')
    initial=replace(initial,**{phase:replace(getattr(initial,phase),
        producer_load='coalesced_x8',producer_transpose_values_per_lane=16,
        amax_reduction='staged') for phase in ('dx','dw')})
    source=tmp_path/'input.json';source.write_text(json.dumps({'candidate':asdict(initial)}))
    monkeypatch.setattr(tuner,'runtime_environment',lambda:({'name':'synthetic-cpu'}, {'source_sha256':{}}))
    def worker(command,**kwargs):
        config=MSEdenPipelineSchedule(**json.loads(command[command.index('--config-json')+1]))
        assert config.dx.producer_grouped_store_bits == (64 if space in ('compact64_grids','tma_finalizers','tma_cta_finalizers','tma_producer_grids','tma_gemms','tma_loops','h16_grouped_grids','tma_persistence','tma_paired_finalizers') else 32)
        assert config.dw.producer_grouped_store_bits == config.dx.producer_grouped_store_bits
        output=Path(command[command.index('--output')+1])
        result=evidence(config)
        import copy
        timing=result['timings']['backward']
        timing['order_quality']={order:copy.deepcopy(timing['quality']) for order in ('AB','BA')}
        output.write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess,'run',worker)
    root=tmp_path/'study'
    study.main(['--selection',str(source),'--output',str(root),'--shape','512,512,512',
        '--trials','2','--seconds','10','--space',space])
    observations=[json.loads(line) for line in (root/'residuals'/'observations.jsonl').read_text().splitlines()]
    assert len(observations)==2
    assert all(row['outcome']['status']=='ok' for row in observations)
    assert json.loads((root/'selection.json').read_text())['automatic_promotion'] is False


def test_tma_finalizer_portfolio_preserves_math_producers_and_equal_leg_threads():
    from benchmarks.experiment_ms_eden_joint_geometry import tma_finalizer_portfolio
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
    initial=MSEdenPipelineSchedule(backward_pipeline='operand_paired')
    choices=tma_finalizer_portfolio(initial)
    assert len(choices)==len(set(choices))==20 and choices[0]==initial
    for candidate in choices:
        for phase in ('dx','dw'):
            base=getattr(initial,phase);actual=getattr(candidate,phase)
            assert replace(actual,a_geometry=base.a_geometry,b_geometry=base.b_geometry)==base
            assert actual.operand_schedule('a').finalizer_threads==actual.operand_schedule('b').finalizer_threads
            for operand in ('a','b'):
                b=asdict(base.operand_schedule(operand));c=asdict(actual.operand_schedule(operand))
                for name in ('finalizer_threads','finalizer_blocks'):
                    b.pop(name);c.pop(name)
                assert b==c


def test_tma_loops_preserve_preparation_and_offer_one_leg_and_joint_changes():
    from benchmarks.experiment_ms_eden_joint_geometry import tma_loop_portfolio
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
    initial=MSEdenPipelineSchedule(backward_pipeline='operand_paired')
    choices=tma_loop_portfolio(initial)
    assert len(choices)==len(set(choices))==28 and choices[0]==initial
    for offset in range(1,28,3):
        dx,dw,both=choices[offset:offset+3]
        assert dx.dw==initial.dw and dw.dx==initial.dx
        assert both.dx==dx.dx and both.dw==dw.dw
    allowed={'k_tile_unroll','mma_schedule','a_ldmatrix_matrices','b_ldmatrix_matrices'}
    for candidate in choices:
        for phase in ('dx','dw'):
            actual,base=getattr(candidate,phase),getattr(initial,phase)
            assert replace(actual,gemm=base.gemm)==base
            assert actual.gemm.tile_m>=128
            assert {k:v for k,v in asdict(actual.gemm).items() if k not in allowed}=={
                k:v for k,v in asdict(base.gemm).items() if k not in allowed}
