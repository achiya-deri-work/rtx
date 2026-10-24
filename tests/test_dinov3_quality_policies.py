"""Keep the diagnostic study controlled without requiring a checkpoint/GPU."""
from dataclasses import replace

import pytest
import torch
from torch import nn

import rtx
from benchmarks.dinov3_regression import (
    DEFAULT_CHECKPOINT, DinoBiasAdapter, _biasless_source,
    convert_dinov3_variant, deterministic_images, load_dinov3_vits16,
    tensor_metrics,
)

from benchmarks.study_dinov3_quality_policies import (
    SplitAccumulationLinear, SplitAccumulationStudy, convert, policies,
)
from benchmarks.diagnose_dinov3_quantized_reference import audit_disagreements
from benchmarks.diagnose_dinov3_quantized_reference import dequantized_reference
from rtx.configs import NVFP4DynamicConfig, NVFP4QuantConfig
from rtx.fp8 import DEFAULT_MXFP8_INFERENCE_CONFIG


@pytest.mark.parametrize("native,oracle,counts", [
    (1.0078125, 1., (1, 0, 0)),
    (1., 1.0078125, (0, 1, 0)),
    (1.015625, 1., (0, 0, 1)),
])
def test_rounding_audit_uses_exact_sum_not_fp32_double_rounding(native, oracle, counts):
    x = torch.tensor([[1., 2**-8, 2**-25]])
    weight = torch.ones(1, 3)
    result = audit_disagreements(x, weight, torch.tensor([[native]]).bfloat16(),
                                torch.tensor([[oracle]]).bfloat16())
    assert result['cells'] == result['fp64_matches_rational'] == 1
    assert tuple(result[k] for k in ('native_exact', 'oracle_exact', 'neither_exact')) == counts
    assert result['examples'][0]['nearest_bf16'] == 1.0078125


def test_rounding_audit_does_not_claim_agreeing_cells_are_exact():
    result = audit_disagreements(torch.ones(1, 3), torch.ones(1, 3),
                                torch.zeros(1, 1), torch.zeros(1, 1))
    assert result['cells'] == result['native_exact'] == result['oracle_exact'] == 0


@pytest.mark.skipif(not torch.cuda.is_available()
                   or torch.cuda.get_device_capability()[0] != 12, reason="SM12 required")
@pytest.mark.parametrize("k,tiles", [(384, 1), (384, 4), (1536, 2)])
@torch.inference_mode()
def test_split_study_wrapper_matches_independent_quantized_dot(k, tiles):
    torch.manual_seed(4620)
    source = nn.Linear(k, 129, bias=False, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 17, k, device="cuda", dtype=torch.bfloat16) * .1
    candidate = SplitAccumulationLinear(source, SplitAccumulationStudy(tiles))
    actual = candidate(x)
    expected = torch.nn.functional.linear(dequantized_reference(x, 'mxfp8').double(),
                                         dequantized_reference(source.weight, 'mxfp8').double())
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
    assert bool(actual.isfinite().all())
    relative = (actual.double() - expected).norm() / expected.norm()
    assert float(relative) < .004


def test_mx_policy_sweep_only_changes_scale_rounding():
    configs = policies()
    base = DEFAULT_MXFP8_INFERENCE_CONFIG
    for x in ("floor", "infinity"):
        for w in ("floor", "infinity"):
            policy = configs[f"mx_x_{x}_w_{w}"]
            config = policy.dynamic_config
            assert policy.autotune == "off"
            assert config.quant == replace(base.quant, scale_rounding=x)
            assert config.resolved_weight_quant() == replace(base.resolved_weight_quant(), scale_rounding=w)
            assert replace(config, quant=base.quant, weight_quant=None) == base


def test_nv_policy_sweep_preserves_explicit_scaling_and_both_quantizers():
    configs = policies()
    assert len(configs) == 12
    for scaling in ("block", "current"):
        for selection in ("amax", "four_over_six", "mse5", "lsq1"):
            policy = configs[f"nv_{scaling}_{selection}"]
            assert policy.scaling == scaling
            assert policy.autotune == "off"
            assert policy.quant_config.block_scale_selection == selection
            assert policy.weight_prequant_config.quant_x == policy.quant_config
            assert policy.quant_config.rejection(384, 384) is None


@pytest.mark.skipif(
    not DEFAULT_CHECKPOINT.is_file() or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 12,
    reason="requires local DINO checkpoint and SM12 GPU",
)
@torch.inference_mode()
@pytest.mark.dinov3_research
def test_explicit_four_over_six_quality_and_dynamic_packed_parity():
    """An opt-in recipe; does not replace the original default-policy gates."""
    seeds = (4403, 4410, 4511, 4512, 4600, 4615)
    inputs = {s: deterministic_images(2, 224, device="cuda", seed=s) for s in seeds}
    baseline = convert_dinov3_variant(load_dinov3_vits16(device="cuda"), "bf16_bias_adapter")
    expected = {s: baseline.forward_features(x) for s, x in inputs.items()}
    del baseline
    packed = convert(load_dinov3_vits16(device="cuda"), policies()["nv_block_four_over_six"])
    packed_outputs = {s: packed.forward_features(x) for s, x in inputs.items()}
    del packed
    dynamic = load_dinov3_vits16(device="cuda")
    config = rtx.NVFP4TrainingConfig(
        scaling="block", autotune="off", backend="materialized",
        dynamic_config=NVFP4DynamicConfig(
            quant=NVFP4QuantConfig(block_scale_selection="four_over_six")),
    )
    for name, source in list(dynamic.named_modules()):
        if name.startswith("blocks.") and isinstance(source, nn.Linear):
            core = rtx.convert_to_nvfp4_training(_biasless_source(source), config=config)
            parent, child = name.rsplit(".", 1)
            dynamic.get_submodule(parent)._modules[child] = DinoBiasAdapter(core, source)
    for seed, x in inputs.items():
        actual = dynamic.forward_features(x)
        for key, cosine in (("x_norm_patchtokens", .92), ("x_storage_tokens", .95)):
            metrics = tensor_metrics(actual[key], expected[seed][key])
            assert metrics["finite"], (seed, key, metrics)
            assert metrics["minimum_cosine"] > cosine, (seed, key, metrics)
        for key in ("x_norm_clstoken", "x_norm_patchtokens", "x_storage_tokens"):
            torch.testing.assert_close(actual[key], packed_outputs[seed][key], rtol=0, atol=0)
