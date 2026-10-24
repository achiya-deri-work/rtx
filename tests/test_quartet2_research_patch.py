"""CPU checks for the optional external-source research transformer."""
from benchmarks.patch_quartet2_research import transform


def test_stream_patch_handles_nested_launch_arguments():
    source = '''k<<<blocks, dim3(32, 8)>>>(x);
k<<<blocks, dim3(32, 8), smem>>>(x);
config.stream = nullptr;
cudaMemset(max_scale, 0, sizeof(unsigned));'''
    result = transform(source, 'test.cu', False)
    assert '<<<blocks, dim3(32, 8), 0, rtx_research_stream()>>>' in result
    assert '<<<blocks, dim3(32, 8), smem, rtx_research_stream()>>>' in result
    assert 'config.stream = rtx_research_stream();' in result
    assert 'cudaMemsetAsync(max_scale, 0, sizeof(unsigned), rtx_research_stream())' in result


def test_rng_patch_keeps_seed_value_arithmetic_on_device():
    source = 'void kernel(long seed) { philox<10>(seed, 1, 2); philox<10>(seed + 1, 1, 2); }'
    result = transform(source, 'test.cu', True)
    assert 'const long* seed' in result
    assert 'philox<10>(*seed,' in result
    assert 'philox<10>(*seed + 1,' in result


def test_rng_patch_moves_auto_seed_to_input_device():
    source = '''seed = torch.randint(0, 2**32, (), dtype=torch.int64)
    _quartet2.rht128_eden(o, s, t, x, seed)
'''
    result = transform(source, 'quant.py', True)
    assert 'dtype=torch.int64, device=x.device' in result
    assert 'seed = seed.to(device=x.device)' in result
    assert 'device=x.device' not in transform(source, 'quant.py', False)


def test_strided_gradient_adapter_is_explicit():
    source = 'flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])'
    assert transform(source, 'linear.py', True) == source
    assert transform(source, 'linear.py', True, True) == source + '.contiguous()'
