"""CPU-only diagnostic harness contracts, not GPU performance qualification."""
import math

import pytest
import torch
from benchmarks import compare_ms_eden_tf32_hadamard as experiment


def test_defaults_and_configurable_shape_geometry():
    args=experiment.parser().parse_args(["--output","unused"])
    assert args.shape==(4096,1536)
    assert (args.threads,args.blocks,args.rounds,args.warmup_seconds,args.min_batch_ms)==(128,128,24,15.,2.)
    assert args.graph_batch==8 and args.mma_kind=="tf32"
    args=experiment.parser().parse_args(["--output","unused","--shape","7x384","--threads","256","--blocks","52",
        "--graph-batch","1","--mma-kind","bf16_padded"])
    assert (args.shape,args.threads,args.blocks)==((7,384),256,52)
    assert args.graph_batch==1 and args.mma_kind=="bf16_padded"
    for value in ("1,127","0,128","1,128,2","invalid"):
        with pytest.raises(SystemExit):experiment.parser().parse_args(["--output","unused","--shape",value])
    for value in ("0","-1","1.5"):
        with pytest.raises(SystemExit):experiment.parser().parse_args(["--output","unused","--graph-batch",value])


@pytest.mark.parametrize("graph_batch",[1,8,32])
def test_event_window_normalization_counts_each_transform_once(graph_batch):
    result=experiment.window_times(4.,20,graph_batch)
    assert result["elapsed_window_ms"]==4.
    assert result["per_graph_replay_ms"]==.2
    assert result["per_transform_ms"]==pytest.approx(.2/graph_batch)
    assert result["graph_replays"]==20 and result["transforms"]==20*graph_batch
    with pytest.raises(ValueError):experiment.window_times(4.,20,0)


def test_mma_config_forwarding_contract_available_without_gpu():
    from rtx.kernels.ms_eden_tf32_hadamard import TF32H8Config
    for kind in ("tf32","bf16_padded"):
        for scalar in (False,True):
            config=TF32H8Config(128,52,force_scalar=scalar,mma_kind=kind)
            assert config.mma_kind==kind and config.force_scalar==scalar


def test_dense_fp64_oracle_matches_independent_bit_parity_basis():
    source=torch.zeros(4,128,dtype=torch.bfloat16)
    signs=torch.ones(128);signs[7]=-1
    indices=(0,1,7,127)
    for row,index in enumerate(indices):source[row,index]=2
    oracle=experiment.dense_fp64_oracle(source,signs).reshape(4,128)
    assert oracle.dtype==torch.float64
    for row,index in enumerate(indices):
        expected=torch.tensor([(-1 if (index&column).bit_count()%2 else 1)*2*float(signs[index])/math.sqrt(128)
                               for column in range(128)],dtype=torch.float64)
        torch.testing.assert_close(oracle[row],expected,rtol=0,atol=0)
    assert experiment.quality(oracle.flatten().float(),oracle.flatten())["nrms"]<1e-7
    with pytest.raises(RuntimeError,match="nonfinite"):
        experiment.quality(torch.full((4,),float("inf")),torch.ones(4,dtype=torch.float64))
    with pytest.raises(RuntimeError,match="quality gate failed"):
        experiment.quality(torch.zeros(4),torch.ones(4,dtype=torch.float64))


def test_clustered_pair_summary_reports_ratio_stationarity_and_order_splits():
    rows=[]
    for index in range(24):
        rows.append(dict(round=index,reference_ms=.04,candidate_ms=.03,windows=[
            dict(order_kind=kind,reference_ms=.04,candidate_ms=.03)for kind in ("AB","BA")]))
    result=experiment.timing_summary(rows,959)
    assert result["paired"]["geomean_reference_over_candidate"]==pytest.approx(4/3)
    assert result["paired"]["rounds"]==24 and result["stationary"]
    assert result["median_ms"]==dict(baseline=.04,candidate=.03)
    assert set(result["by_order"])=={"AB","BA"}
    assert all(item["rounds"]==24 for item in result["by_order"].values())
