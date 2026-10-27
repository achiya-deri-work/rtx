"""Address ownership and exact-output gates for the research XOR layout."""
import os
import pytest


def test_layout_bijection_pairs_and_banks():
    from rtx.kernels.ms_eden_bank_layout import bank_layout_offset as offset
    assert sorted(offset(r, c) for r in range(128) for c in range(32)) == list(range(4096))
    for r in range(128):
        for c in range(0, 32, 2):
            assert offset(r, c) % 2 == 0
            assert offset(r, c + 1) == offset(r, c) + 1
        for c in range(32):
            base = r // 16 * 16
            parity_base = base + r % 2
            assert offset(r, c) == offset(parity_base, c) + ((r % 16) // 2) * 64
    for warp in range(4):
        for base in (0, 4):
            for j in range(16):
                banks = [(offset((lane % 8) * 16 + j, warp + (base + lane // 8) * 4) // 2) % 32 for lane in range(32)]
                assert len(set(banks)) == 32


@pytest.mark.skipif(os.getenv('RTX_RUN_BANK_LAYOUT_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128, 128), (512, 768), (4096, 1536)])
@pytest.mark.parametrize('stages', [1, 2, 3])
@pytest.mark.parametrize('hoist,unroll', [(False, True), (True, True), (True, False)])
def test_exact_outputs_and_current_buffer_graph(shape, stages, hoist, unroll, min_blocks_hint=0, butterfly_fma=False, wide=False, regrouped_warp_partials=False):
    import torch
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation, PosthocScaleConfig
    from rtx.kernels.ms_eden_bank_layout import compile_bank_layout
    torch.manual_seed(176)
    n = shape[0] * shape[1]
    x = torch.randn(*shape, device='cuda', dtype=torch.bfloat16)
    signs = (torch.randint(0, 2, (128,), device='cuda') * 2 - 1).float()
    kwargs = dict(load_mode='coalesced_x8', signs_mode='register', stages=stages,
                  async_copy=True, correction_decode='native', producer_values_per_lane=16,
                  producer_transpose_values_per_lane=16, producer_grouped_store_bits=32)
    runners, outputs = [], []
    for compiler in (compile_hadamard_preparation, compile_bank_layout):
        extra = {'address_hoist': hoist, 'producer_unroll_rows': unroll,
                 'min_blocks_hint': min_blocks_hint, 'butterfly_fma': butterfly_fma} if compiler is compile_bank_layout else {}
        run, count = compiler(*shape, PosthocScaleConfig(128, 256), 4, True, **kwargs, **extra)
        out = (torch.empty(n // 2, device='cuda', dtype=torch.uint8),
               torch.empty(n // 16, device='cuda', dtype=torch.bfloat16),
               torch.empty(count, device='cuda'))
        runners.append(run)
        outputs.append(out)
        run(x.flatten(), signs, *out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runners[1](x.flatten(), signs, *outputs[1])
    distributions = ('gaussian', 'zero', 'outliers')
    if butterfly_fma or wide:
        distributions += ('wide', 'arbitrary_signs')
    for distribution in distributions:
        signs.random_(0, 2).mul_(2).sub_(1)
        if distribution == 'zero':
            x.zero_()
        elif distribution == 'outliers':
            x.normal_().mul_(0.01)
            x.flatten()[::17] = 128
        elif distribution == 'wide':
            exponents = torch.randint(-100, 101, x.shape, device=x.device)
            x.copy_(torch.ldexp(torch.randn_like(x, dtype=torch.float32), exponents))
        elif distribution == 'arbitrary_signs':
            x.normal_().mul_(.25)
            signs.fill_(1.001)
        runners[0](x.flatten(), signs, *outputs[0])
        graph.replay()
        torch.cuda.synchronize()
        for index, (expected, actual) in enumerate(zip(*outputs)):
            if regrouped_warp_partials and index == 2:
                # Column reassignment changes warp ownership, but not the CTA
                # work set or the exact maximum consumed by finalization.
                expected = expected.reshape(-1,4).amax(dim=1)
                actual = actual.reshape(-1,4).amax(dim=1)
            torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
        assert torch.isfinite(outputs[1][1]).all() and torch.isfinite(outputs[1][2]).all()


@pytest.mark.skipif(os.getenv('RTX_RUN_BANK_LAYOUT_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('hint', [7, 8])
def test_compact_occupancy_hint_exact(hint):
    test_exact_outputs_and_current_buffer_graph((4096, 1536), 1, True, False, hint)


@pytest.mark.skipif(os.getenv('RTX_RUN_BANK_LAYOUT_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128, 128), (4096, 1536)])
@pytest.mark.parametrize('stages', [1, 2])
def test_compact_fma_butterfly_exact(shape, stages):
    test_exact_outputs_and_current_buffer_graph(shape, stages, True, False, butterfly_fma=True)


@pytest.mark.skipif(os.getenv('RTX_RUN_BANK_LAYOUT_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128,128), (4096,1536)])
def test_compact_store64_exact(monkeypatch, shape):
    from rtx.kernels import ms_eden_bank_layout as bank
    original = bank.compile_bank_layout
    def store64(*args, **kwargs):
        kwargs['producer_grouped_store_bits'] = 64
        return original(*args, **kwargs)
    monkeypatch.setattr(bank, 'compile_bank_layout', store64)
    test_exact_outputs_and_current_buffer_graph(shape, 1, True, False, wide=True)
