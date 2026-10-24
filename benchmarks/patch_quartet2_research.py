"""Create an isolated stream/RNG experiment from pinned Quartet-II kernels.

Never operates on RTX runtime or an installed package. The output directory
must not exist. Upstream copyright notices remain intact. This adapter is for
single-device experiments, not a supported public backend.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def transform(text, name, gpu_seed, contiguous_backward=False):
    if name.endswith('.cu'):
        text = '#include "rtx_research_stream.h"\n' + text
        def launch(match):
            args = match[1]
            # Commas in dim3(...) are not launch-argument separators.
            depth = 0
            count = 1
            for c in args:
                depth += (c == '(') - (c == ')')
                count += c == ',' and depth == 0
            assert count in (2, 3), (name, args)
            return '<<<' + args + (', 0' if count == 2 else '') + ', rtx_research_stream()>>>'
        text = re.sub(r'<<<(.*?)>>>', launch, text, flags=re.S)
        text = text.replace('cudaMemset(max_scale, 0, sizeof(unsigned))',
                            'cudaMemsetAsync(max_scale, 0, sizeof(unsigned), rtx_research_stream())')
        text = text.replace('cudaLaunchConfig_t config = {};',
                            'cudaLaunchConfig_t config = {};\n    config.stream = rtx_research_stream();')
        text = text.replace('config.stream = nullptr;', 'config.stream = rtx_research_stream();')
    if name == 'binding.cpp':
        text = '#include "rtx_research_stream.h"\n' + text
        text += '''
// Research bridge only: nanobind entrypoints retain the GIL.
cudaStream_t rtx_research_stream() {
    auto stream = nb::module_::import_("torch").attr("cuda").attr("current_stream")();
    return reinterpret_cast<cudaStream_t>(nb::cast<std::uintptr_t>(stream.attr("cuda_stream")));
}
'''
    if gpu_seed and (name.endswith('.cu') or name == 'binding.cpp'):
        text = text.replace('long seed', 'const long* seed')
        text = text.replace('std::uint64_t seed', 'const long* seed')
        text = text.replace('philox<10>(seed,', 'philox<10>(*seed,')
        text = text.replace('philox<10>(seed + 1,', 'philox<10>(*seed + 1,')
        if name == 'binding.cpp':
            text = text.replace('nb::device::cpu', 'nb::device::cuda')
            text = text.replace('*seed.data()', 'seed.data()')
    if gpu_seed and name == 'quant.py':
        text = text.replace('torch.randint(0, 2**32, (), dtype=torch.int64)',
                            'torch.randint(0, 2**32, (), dtype=torch.int64, device=x.device)')
        # Preserve explicit CPU-seed convenience outside capture; automatic
        # seeds are generated directly on GPU and advance on graph replay.
        text = re.sub(r'(    _quartet2\.(?:eden_fp4|group_transform_128_eden|rht128_eden|dequant_tp_had_quant|rht128_requant)\()',
                      r'    seed = seed.to(device=x.device)\n\1', text)
    if contiguous_backward and name == 'linear.py':
        old = 'flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])'
        assert text.count(old) == 1
        text = text.replace(old, old + '.contiguous()')
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--gpu-seed', action='store_true')
    p.add_argument('--contiguous-backward', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        p.error('output already exists; choose a new experiment directory')
    shutil.copytree(a.source, a.output)
    manifest = {}
    for path in list((a.output/'csrc').glob('*.cu')) + [a.output/'csrc/binding.cpp', a.output/'python/quartet2/quant.py', a.output/'python/quartet2/linear.py']:
        before = path.read_text()
        after = transform(before, path.name, a.gpu_seed, a.contiguous_backward)
        path.write_text(after)
        manifest[str(path.relative_to(a.output))] = {
            'before': hashlib.sha256(before.encode()).hexdigest(),
            'after': hashlib.sha256(after.encode()).hexdigest()}
    (a.output/'csrc/rtx_research_stream.h').write_text(
        '#pragma once\n#include <cuda_runtime.h>\ncudaStream_t rtx_research_stream();\n')
    (a.output/'rtx-research-patch.json').write_text(json.dumps(
        {'gpu_seed': a.gpu_seed, 'contiguous_backward': a.contiguous_backward,
         'files': manifest}, indent=2)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
