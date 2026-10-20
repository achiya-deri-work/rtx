"""Regression for generic shared reads racing asynchronous TMA slot reuse."""
from dataclasses import replace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('changing_inputs', [False, True])
def test_split_gemm_different_concurrent_kernel(changing_inputs):
    from rtx.configs.mxfp8 import MXFP8GemmConfig, MXFP8QuantConfig
    from rtx.fp8_bwd import _build_matmul_runner
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_bwd import MXFP8BwdMatmulConfig

    torch.manual_seed(731)
    quant = MXFP8QuantConfig(load_bits=32, quant_store_bits=32, quant_math='fp32',
        quant_amax='fp32', persistent_waves=6, maxrregcount=96)
    gemm = MXFP8GemmConfig(atom_layout_m=4, b_swizzle='128b', epilogue='direct',
                          store_vec=1, producer_registers=64, consumer_registers=232)
    cfg = MXFP8BwdMatmulConfig(a_orientation='transpose', b_orientation='transpose',
        backend='decomposed', quant_a=quant,
        quant_b=replace(quant, transposed_load_engine='cp_async', transposed_smem_padding=0),
        gemm=replace(gemm, sfb_s2r_bits=0, scale_l2_prefetch='64b', scale_l1_evict='first'),
        reduction='split_fp32_workspace', split_reduction=8, reduction_tile=2560,
        workspace_epilogue='tree')
    other_cfg = replace(cfg, quant_a=replace(quant, transposed_tile_rows=64),
                        quant_b=None, gemm=gemm)
    runner = _build_matmul_runner(problem=MXFP8Problem(1536,256,20480), config=cfg,
                                  device=torch.device('cuda:0'))
    competitor = _build_matmul_runner(problem=MXFP8Problem(256,768,20480), config=other_cfg,
                                      device=torch.device('cuda:0'))
    sources = tuple(torch.randn(20480, rows, device='cuda', dtype=torch.bfloat16)*.2
                    for rows in (1536,256))
    other_sources = tuple(torch.randn(20480, rows, device='cuda', dtype=torch.bfloat16)*.2
                          for rows in (256,768))
    other_out = torch.empty(256,768,device='cuda',dtype=torch.bfloat16)
    competitor.quantize(*other_sources)
    caller, side = torch.cuda.current_stream(), torch.cuda.Stream()
    for batch in range(4):
        if changing_inputs and batch:
            for source in sources:
                source.add_(.003)
        runner.quantize(*sources)
        a = runner.quantized_a.float()*runner.scales_a.float().repeat_interleave(32,dim=1)
        b = runner.quantized_b.float()*runner.scales_b.float().repeat_interleave(32,dim=1)
        expected = torch.stack([a[:,i:i+2560] @ b[:,i:i+2560].T
                                for i in range(0,20480,2560)]).flatten()
        del a,b
        first = None
        for _ in range(24):
            side.wait_stream(caller)
            with torch.cuda.stream(side):
                for _ in range(3):
                    competitor.matmul(other_out)
            runner.quantize(*sources)
            runner.gemm(runner.quantized_a,runner.quantized_b,
                        runner.scales_a,runner.scales_b,runner.workspace)
            caller.wait_stream(side)
            caller.synchronize()
            actual = runner.workspace
            assert bool(actual.isfinite().all())
            assert float((actual-expected).norm()/expected.norm()) < 1e-5
            if first is None:
                first = actual.clone()
            assert torch.equal(actual,first), 'identical-input FP32 partials changed'
