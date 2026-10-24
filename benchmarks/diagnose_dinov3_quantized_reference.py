"""Separate model quantization drift from RTX implementation error.

The deliberately slow oracle uses FP32 or FP64 PyTorch GEMMs and independent
format conversion, not RTX kernels. Optional split candidates do use native
RTX kernels and are reported separately. This is not a performance benchmark.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import torch
from torch import nn
from benchmarks.dinov3_regression import (
    DinoBiasAdapter, load_dinov3_vits16, deterministic_images,
    convert_dinov3_variant, tensor_metrics,
)
from tests.test_mxfp8_fwd import _reference_to_mx
from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, f4_unpacked_to_f32
from benchmarks.diagnose_mxfp8_rounding import round_fp64_to_bf16


def audit_disagreements(x, weight, actual, expected):
    """Audit only disagreeing cells; exact rational sums verify FP64 here.

    This does not establish correctness of cells where both kernels agree.
    Inputs must already be independently dequantized operands.
    """
    actual = actual.detach().reshape(-1, weight.shape[0]).cpu()
    expected = expected.detach().reshape_as(actual).cpu()
    indices = (actual != expected).nonzero()
    counts = dict(cells=len(indices), native_exact=0, oracle_exact=0,
                  neither_exact=0, fp64_matches_rational=0, examples=[])
    if not len(indices):
        return counts
    x = x.detach().reshape(-1, weight.shape[1]).cpu().double()
    weight = weight.detach().cpu().double()
    for row, col in indices.tolist():
        left, right = x[row], weight[col]
        rational = sum((Fraction(float(a)) * Fraction(float(b))
                        for a, b in zip(left, right)), Fraction())
        dot = (left * right).sum()
        if Fraction(float(dot)) != rational:
            raise AssertionError("FP64 sum was not exact; extend rational rounding oracle")
        rounded = round_fp64_to_bf16(dot)
        native_ok = bool(actual[row, col] == rounded)
        oracle_ok = bool(expected[row, col] == rounded)
        counts['fp64_matches_rational'] += 1
        counts['native_exact'] += int(native_ok)
        counts['oracle_exact'] += int(oracle_ok)
        counts['neither_exact'] += int(not native_ok and not oracle_ok)
        if len(counts['examples']) < 4:
            counts['examples'].append(dict(index=[row, col], exact=float(dot),
                nearest_bf16=float(rounded), native=float(actual[row, col]),
                oracle=float(expected[row, col])))
    return counts


def dequantized_reference(value, mode):
    if mode == 'mxfp8':
        q, s = _reference_to_mx(value)
        return q.float()*s.float().repeat_interleave(32, -1)
    blocks = value.float().reshape(*value.shape[:-1], -1, 16)
    scales = (blocks.abs().amax(-1, keepdim=True)/6).clamp(2**-6, 448)
    scales = scales.to(torch.float8_e4m3fn).float()
    # Match the documented reciprocal multiplication quantization contract.
    codes = f32_to_f4_unpacked(blocks*(1/scales))
    return (f4_unpacked_to_f32(codes)*scales).reshape_as(value)


class ReferenceLinear(nn.Module):
    def __init__(self, source, mode, accumulation='fp32'):
        super().__init__()
        self.mode = mode
        self.accumulation = accumulation
        self.register_buffer('weight', dequantized_reference(source.weight.detach(), mode))

    def forward(self, x):
        value = dequantized_reference(x, self.mode)
        if self.accumulation == 'fp64':
            return round_fp64_to_bf16(torch.nn.functional.linear(value.double(), self.weight.double()))
        return torch.nn.functional.linear(value, self.weight).bfloat16()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit-rounding', action='store_true',
                   help='Verify every differing cell with exact rational sums')
    p.add_argument('--oracle-accumulation', choices=('fp32', 'fp64'), default='fp32')
    p.add_argument('--native-reduction-tiles', type=int, default=0,
                   help='Use the research native split-accumulation path for MXFP8')
    p.add_argument('--compare-split-tiles', type=int, default=0,
                   help='Compare a split candidate on exactly the native path inputs')
    args = p.parse_args()
    if args.native_reduction_tiles < 0 or args.compare_split_tiles < 0:
        p.error('native reduction tiles must be nonnegative')
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for mode, variant, seed in [('mxfp8', 'mxfp8_ptq', 4403), ('block', 'nvfp4_ptq_block', 4410)]:
        images = deterministic_images(2, 224, device='cuda', seed=seed)
        bf16 = convert_dinov3_variant(load_dinov3_vits16(device='cuda'), 'bf16_bias_adapter')
        baseline = bf16.forward_features(images)
        del bf16
        native = load_dinov3_vits16(device='cuda')
        references = {n: ReferenceLinear(m, mode, args.oracle_accumulation) for n, m in native.named_modules()
                      if n.startswith('blocks.') and isinstance(m, nn.Linear)}
        candidates = {}
        if mode == 'mxfp8' and args.compare_split_tiles:
            from benchmarks.study_dinov3_quality_policies import SplitAccumulationLinear, SplitAccumulationStudy
            candidates = {n: SplitAccumulationLinear(m, SplitAccumulationStudy(args.compare_split_tiles))
                          for n, m in native.named_modules()
                          if n.startswith('blocks.') and isinstance(m, nn.Linear)}
        if mode == 'mxfp8' and args.native_reduction_tiles:
            from benchmarks.study_dinov3_quality_policies import convert, SplitAccumulationStudy
            native = convert(native, SplitAccumulationStudy(args.native_reduction_tiles))
        else:
            native = convert_dinov3_variant(native, variant)
        local, handles = {}, []
        for name, reference in references.items():
            def compare(_module, inputs, output, name=name, reference=reference):
                expected = reference(inputs[0])
                local[name] = dict(tensor_metrics(output, expected),
                                   unequal_values=int((output != expected).sum()),
                                   values=output.numel())
                if args.audit_rounding:
                    local[name]['rounding_audit'] = audit_disagreements(
                        dequantized_reference(inputs[0], mode), reference.weight,
                        output, expected)
                if name in candidates:
                    candidate = candidates[name](inputs[0])
                    local[name]['split_candidate'] = dict(
                        tensor_metrics(candidate, expected),
                        unequal_values=int((candidate != expected).sum()))
            handles.append(native.get_submodule(name).core.register_forward_hook(compare))
        actual = native.forward_features(images)
        for handle in handles:
            handle.remove()
        del references, candidates
        del native
        oracle = load_dinov3_vits16(device='cuda')
        targets = [(n, m) for n, m in oracle.named_modules() if n.startswith('blocks.') and isinstance(m, nn.Linear)]
        for name, source in targets:
            parent, child = name.rsplit('.', 1)
            oracle.get_submodule(parent)._modules[child] = DinoBiasAdapter(
                ReferenceLinear(source, mode, args.oracle_accumulation), source)
        expected = oracle.forward_features(images)
        del oracle
        result = {'mode': mode, 'seed': seed, 'oracle_accumulation': args.oracle_accumulation,
                  'native_reduction_tiles': args.native_reduction_tiles if mode == 'mxfp8' else 0,
                  'compare_split_tiles': args.compare_split_tiles if mode == 'mxfp8' else 0,
                  'same_input_linears': local, 'outputs': {}}
        for key in ('x_norm_clstoken', 'x_norm_patchtokens', 'x_storage_tokens'):
            result['outputs'][key] = {
                'native_vs_bf16': tensor_metrics(actual[key], baseline[key]),
                'oracle_vs_bf16': tensor_metrics(expected[key], baseline[key]),
                'native_vs_oracle': tensor_metrics(actual[key], expected[key]),
            }
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'scope': 'independent quantized-model oracle, not task accuracy or timing', 'results': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
