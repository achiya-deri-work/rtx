"""Complete training proposals must survive real legality and recipe gates."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('m', [11296, 11712])
def test_complete_prequant_training_seeds(m):
    from rtx.autotune.mlp import make_mlp_adapter, mlp_structural_seeds
    torch.manual_seed(317)
    values=tuple(torch.randn(*s,device='cuda',dtype=torch.bfloat16)*.02
                 for s in ((m,256),(1536,256),(256,768)))
    adapter=make_mlp_adapter(*values,phase='train',measurement_mode='compiled_direct',
                             warmup=1,samples=3,calls_per_sample=1)
    seeds=mlp_structural_seeds(adapter,'mxfp8','train')
    assert [(c.gate_backward.dw.split_reduction,c.backward_overlap)
            for c in seeds[:2]]==[(4,False),(8,False)]
    assert all(c.gate_backend==c.down_backend==c.recompute_backend=='prequant'
               for c in seeds[:2])
    candidates=[c for c in seeds
        if c.gate_backend=='prequant' and c.down_backend=='prequant'
        and c.recompute_backend=='prequant'
        and c.recompute.gemm.tile_m==128 and c.recompute.gemm.tile_n==64
        and c.gate_backward.dw.split_reduction in (4,8)]
    assert {c.gate_backward.dw.split_reduction for c in candidates}=={4,8}
    assert {(c.gate_backward.dw.split_reduction,c.backward_overlap)
            for c in candidates}=={(4,False),(8,False)}
    for c in candidates:
        assert c.down_backward.dw.split_reduction==c.gate_backward.dw.split_reduction
        assert c.gate_up.glu_silu_math==adapter.initial_config.gate_up.glu_silu_math
        assert adapter.rejection(c) is None
        result=adapter.evaluate(c)
        assert result.status=='ok',result.error
        assert set(result.metadata['numerics'])=={'output','dx','dgate_up','ddown'}
        for metric in result.metadata['numerics'].values():
            assert metric['finite'] and metric['relative_l2']<.025
